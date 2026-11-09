"""CPU diagnostic for the thin-sheet trace-basis design (docs/plans/fdtd-3d-thin-sheet-trace-basis.md).

Works on complexes produced by the real compiler and on chord operators
assembled from them as specified in the design; no solver runs, no GPU.

a   spectrum of the current trace-space matrix S = I + R W C^-1 W^T R
b   PEC-limit interior modes of a rotated square prism (quasi-2D): current PEC
    path, current copper basis restricted exactly to W E = 0, chord basis with
    the FIT removal rule (and with chord-aware stiffness floors)
c   explicit leapfrog bound of the chord basis for mass rules and floors
d   sliver energy bias of the chord masses (additive vs removal)
e   iterative floors on the top leapfrog mode: bound vs PEC-limit accuracy
f   heavy additive chords with chord-aware floors
g   locally implicit chords (Newmark 1/4): explicit bound and the block-Jacobi
    condition of the implicit chord system (g: removal masses; g2: light
    additive chords; g4: no sheet-face floor; g5: geometric sheet-face floor)
g3  implicit set = chords + split segments

Run: CUDA_VISIBLE_DEVICES="" python scripts/dev/fdtd/thin_sheet_trace_basis_diagnostic.py [a b c d e f g g2 g3 g4 g5]
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import witwin.maxwell as mw
from witwin.maxwell.compiler.thin_sheet import _FRAME, compile_thin_sheet, owns_thin_sheet
from witwin.maxwell.scene import prepare_scene
from tests.materials.thin_sheet import harness as hn
from tests.materials.thin_sheet import test_thin_sheet_runtime as rt

EPS0, MU0 = 8.8541878128e-12, 1.25663706212e-6


def compile_pair(build):
    """(prepared scene, PEC complex, copper complex) for the same geometry."""
    out = []
    for kind in ("pec", "cu"):
        scene = prepare_scene(build(kind))
        structure = next(s for s in scene.structures if owns_thin_sheet(s))
        out.append((scene, compile_thin_sheet(scene, structure)))
    (scene, pec), (_, cu) = out
    assert np.array_equal(pec.e_volume, cu.e_volume) and np.array_equal(pec.h_area, cu.h_area)
    return scene, pec, cu


def k_matrix(c, extra_cols=0, extra_rows=0):
    return sp.csr_matrix((c.k_vals, (c.k_rows, c.k_cols)), shape=(len(c.h_area) + extra_rows, len(c.e_volume) + extra_cols))


def chord_geometry(scene, c):
    """Per trace (active cut face, compile order): rows, path pieces, sub-face centroid distances."""
    surface = c.surface
    nodes = [np.asarray(getattr(scene, f"{a}_nodes64")) for a in "xyz"]
    h = max(float(np.max(np.diff(n))) for n in nodes)
    zero_tol, region_tol = 1e-9 * h, 1e-7 * h
    plus = np.flatnonzero(c.h_extra >= 0)
    plus = plus[np.argsort(c.h_extra[plus])]
    chords = []
    for t, row in enumerate(plus):
        a = int(c.h_component[row]); g = np.array(c.h_index[row]); _, b, cc = _FRAME[a]
        eb, ec = np.eye(3, dtype=int)[b], np.eye(3, dtype=int)[cc]
        idx = [g, g + eb, g + eb + ec, g + ec]
        pts = np.array([[nodes[ax][i[ax]] for ax in range(3)] for i in idx])
        phi, _ = surface.classify(pts)
        sg = np.where(np.abs(phi) <= zero_tol, 0, np.sign(phi)).astype(int)
        cycle = []
        for m in range(4):
            n_ = (m + 1) % 4
            cycle.append(("z" if sg[m] == 0 else sg[m], pts[m]))
            if sg[m] * sg[n_] < 0:
                if surface.planar:
                    tt = phi[m] / (phi[m] - phi[n_])
                    cycle.append(("z", pts[m] + tt * (pts[n_] - pts[m])))
                else:
                    tt = surface.pierce(pts[m], pts[n_], zero_tol)
                    cycle.append(("z", pts[m] + tt * (pts[n_] - pts[m])))
        zeros = [p for lab, p in cycle if lab == "z"]
        assert len(zeros) == 2
        path = [zeros[0], zeros[1]]
        if not surface.planar:
            first, second = surface.members(np.stack(zeros), region_tol)
            if not (first & second):
                for fa in sorted(first):
                    for fb in sorted(second):
                        p = surface.fold_point(fa, fb, a, nodes[a][g[a]])
                        if p is not None and all(nodes[ax][g[ax]] - region_tol <= p[ax] <= nodes[ax][g[ax] + 1] + region_tol
                                                 for ax in (b, cc)):
                            path = [zeros[0], p, zeros[1]]
                            break
                    if len(path) == 3:
                        break
        pieces = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lengths = np.array([np.linalg.norm(e - s) for s, e in pieces])
        tensor = sum((l / lengths.sum()) * np.outer((e - s) / l, (e - s) / l) for (s, e), l in zip(pieces, lengths))
        centroid_distance = {}
        for side in (-1, 1):
            poly = [p for lab, p in cycle if lab in ("z", side)]
            if len(path) == 3:
                # insert the fold point between the two consecutive zeros
                k = next(i for i in range(len(poly)) if any(np.allclose(poly[i], z) for z in zeros)
                         and any(np.allclose(poly[(i + 1) % len(poly)], z) for z in zeros))
                poly = poly[:k + 1] + [path[1]] + poly[k + 1:]
            q = np.array([[p[b], p[cc]] for p in poly])
            x, y = q[:, 0], q[:, 1]
            xs, ys = np.roll(x, -1), np.roll(y, -1)
            cross = x * ys - xs * y
            area = 0.5 * cross.sum()
            cen2 = np.array([((x + xs) * cross).sum(), ((y + ys) * cross).sum()]) / (6 * area)
            cen = np.zeros(3); cen[a] = nodes[a][g[a]]; cen[b], cen[cc] = cen2
            centroid_distance[side] = abs(float(surface.classify(cen[None])[0][0]))
        # Strip width w = dual_a sqrt(1 - n_a^2), length-weighted over fold pieces (as the compiler).
        dual = 0.5 * (nodes[a][min(g[a] + 1, len(nodes[a]) - 1)] - nodes[a][max(g[a] - 1, 0)])
        facets = surface.classify(np.array([0.5 * (s_ + e_) for s_, e_ in pieces]))[1]
        widths = [dual * math.sqrt(max(0.0, 1 - surface.facets[int(f)].normal[a] ** 2)) for f in facets]
        width = float(np.dot(widths, lengths) / lengths.sum())
        chords.append(dict(rows={-1: row - 1, 1: row}, face=(a, tuple(g)), path=path, length=float(lengths.sum()),
                           tensor=tensor, delta=centroid_distance, width=width))
    return chords, h


def chord_volumes(chords, h, delta_min):
    return {(t, s): ch["length"] * ch["width"] * max(ch["delta"][s], delta_min * h)
            for t, ch in enumerate(chords) for s in (-1, 1)}


def removal(c, K_pec, chords, volumes, keep=0.25):
    """Rule "remove": take V_c t_a^2 from the side's own Cartesian edges of component a of the sub-face."""
    vol = c.e_volume.copy()
    floor = keep * c.e_volume
    removed = np.zeros((len(chords), 2, 3))
    short = 0.0
    for t, ch in enumerate(chords):
        for si, s in enumerate((-1, 1)):
            row = K_pec.getrow(ch["rows"][s])
            cols = row.indices[~c.e_fixed[row.indices]]
            for a in range(3):
                want = volumes[(t, s)] * ch["tensor"][a, a]
                if want <= 0:
                    continue
                own = cols[c.e_component[cols] == a]
                avail = np.maximum(vol[own] - floor[own], 0.0)
                if not len(own) or avail.sum() <= 0:
                    short += want
                    continue
                take = min(want, avail.sum())
                vol[own] -= take * avail / avail.sum()
                removed[t, si, a] = take
                short += want - take
    return vol, removed, short


def sliver_bias(chords, volumes, removed, normal, h):
    """Energy bias tensor per unit sheet area and side, in cells: sum_c (V_c t t^T - diag(removed)) / (A h)."""
    area = 0.5 * sum(ch["length"] * ch["width"] for ch in chords)
    result = {}
    for si, s in enumerate((-1, 1)):
        T = np.zeros((3, 3))
        for t, ch in enumerate(chords):
            T += volumes[(t, s)] * ch["tensor"]
            if removed is not None:
                T -= np.diag(removed[t, si])
        T /= area * h
        n = normal / np.linalg.norm(normal)
        basis = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2]
        tang = np.linalg.eigvalsh(basis.T @ T @ basis)
        result[s] = dict(normal=float(n @ T @ n), tangential=(float(tang[0]), float(tang[1])))
    return result


# ---------------------------------------------------------------------------
# A. Current trace-space spectrum
# ---------------------------------------------------------------------------

def part_a():
    from witwin.maxwell.compiler.copper_foil import compile_copper_response
    foil = mw.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 4e9))
    scene = prepare_scene(rt._cavity(mw.Structure(geometry=hn.rectangle_mesh(rt.CENTER, rt.NORMAL, 0.2, 0.2), material=foil),
                                     source_position=tuple(rt.CENTER - 0.03 * rt.NORMAL)))
    c = compile_thin_sheet(scene, next(s for s in scene.structures if owns_thin_sheet(s)))
    h = 0.005
    dt = 0.985 / (2.998e8 * math.sqrt(3.0) / h)
    foil._frequencies((1e9,))
    resp = compile_copper_response(foil)
    rates, B, D = resp.rates.double(), resp.state_space.B.double(), resp.state_space.D.double()
    drive = dt / 2 * B / (1 + dt * rates / 2)[:, None]
    Y = (D + B.T @ drive).numpy(); Y = 0.5 * (Y + Y.T)
    T = len(c.trace_rows)
    dyn = ~c.e_fixed
    rows, cols, vals = [], [], []
    for t, r in enumerate(c.trace_rows):
        for si, s in enumerate((-1, 1)):
            for dof, w in r[s].items():
                if dyn[dof]:
                    rows.append(2 * t + si); cols.append(dof); vals.append(w)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(2 * T, len(c.e_volume)))
    Cinv = sp.diags(np.where(dyn, dt / (EPS0 * c.e_volume), 0.0))
    Z = (W @ Cinv @ W.T).toarray()
    w_, v_ = np.linalg.eigh(0.5 * Y)
    root = v_ @ np.diag(np.sqrt(np.clip(w_, 0, None))) @ v_.T
    R = np.kron(np.diag(np.sqrt(c.trace_width / c.trace_length)), root)
    S = np.eye(2 * T) + R @ Z @ R
    ev = np.linalg.eigvalsh(S)
    dj = 1 / np.sqrt(np.diag(Z))
    zj = np.linalg.eigvalsh(dj[:, None] * Z * dj[None])
    print("A. current basis, cavity-spanning 35 um Cu, traces", T)
    print(f"   S eigenvalues [{ev[0]:.3g}, {ev[-1]:.3g}], quantiles 10/50/90 %: "
          + ", ".join(f"{q:.3g}" for q in np.quantile(ev, (0.1, 0.5, 0.9))))
    print(f"   Jacobi-scaled Z: fraction < 1e-2: {np.mean(zj < 1e-2):.2f}, < 1e-3: {np.mean(zj < 1e-3):.2f}")
    print("   chord basis: S is block diagonal (one 2x2 block per trace), applied exactly")


# ---------------------------------------------------------------------------
# B. Rotated square prism: interior cavity eigenfrequencies in the PEC limit
# ---------------------------------------------------------------------------

def prism_scene(kind, dl, side_cells, phase=0.5, center=(0.37, 0.21)):
    side = side_cells * dl
    radius = side / math.sqrt(2)
    extent = math.ceil((radius + 5 * dl) / dl) * dl
    pts = hn.polygon_points(radius, 4, center=(center[0] * dl, center[1] * dl), phase=phase)
    kw = {} if kind == "cu" else dict(sheet_model="conformal_two_sided")
    mesh = hn.extruded_polyline_mesh(pts, (-2 * dl, 4 * dl), closed=True, **kw)
    material = mw.CopperFoil(thickness=35e-6, frequency_range=(0.3e9, 2e9)) if kind == "cu" else hn.pec_material()
    return mw.Scene(domain=mw.Domain(bounds=((-extent, extent), (-extent, extent), (0.0, 2 * dl))),
                    grid=mw.GridSpec.uniform(dl), boundary=mw.BoundarySpec.faces(default="pec", z="pmc"),
                    device="cpu", structures=[mw.Structure(geometry=mesh, material=material)])


def interior_modes(c, K, volume, side_length, constraint=None):
    """Interior (side -1) cavity modes, dense; ``constraint`` rows are enforced exactly (null space)."""
    import scipy.linalg as la
    keep = np.flatnonzero(~c.e_fixed & (c.e_side == -1))
    Kd = K[:, keep]
    lh = c.h_dual / (MU0 * c.h_area)
    Ms = 1 / np.sqrt(EPS0 * volume[keep])
    A = (Kd.T @ sp.diags(lh) @ Kd).toarray() * Ms[:, None] * Ms[None, :]
    basis = None
    if constraint is not None:
        Wt = constraint[:, keep].toarray() * Ms[None, :]
        Wt = Wt[np.abs(Wt).sum(1) > 0]
        basis = la.null_space(Wt, rcond=1e-10)
        A = basis.T @ A @ basis
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    k1 = math.pi / side_length
    target = k1 ** 2 / (EPS0 * MU0)
    zcomp = c.e_component[keep] == 2
    found = []
    for lam, v in zip(vals, vecs.T):
        if lam <= 1e-8 * target or lam > 4.5 * target:
            continue
        v = basis @ v if basis is not None else v
        energy = v ** 2
        family = "Ez" if energy[zcomp].sum() > 0.5 * energy.sum() else "Hz"
        participation = energy.sum() ** 2 / np.sum(energy ** 2) / len(energy)
        found.append((math.sqrt(lam * EPS0 * MU0) / k1, family, participation))
    return found


def floored_areas(c, K_pec, chords, volumes, e_volume, h, stiffness):
    """Sub-face magnetic areas with the stiffness floor counting the chord (l_c^2 / V_c) as an own piece."""
    area = c.h_area.copy()
    full, parent = h * h, 4.0 / h
    for t, ch in enumerate(chords):
        for s in (-1, 1):
            row = ch["rows"][s]
            r = K_pec.getrow(row)
            own = sum(v * v / e_volume[j] for j, v in zip(r.indices, r.data) if not c.e_fixed[j])
            own += ch["length"] ** 2 / volumes[(t, s)]
            area[row] = max(c.h_true_area[row], full * own / (stiffness * parent), 0.02 * full)
    return area


class _Areas:
    """A complex view with replaced magnetic areas."""
    def __init__(self, c, area):
        self.__dict__.update(c.__dict__)
        self.h_area = area


def part_b():
    print("B. rotated square PEC prism, interior modes, omega / (pi c / L); analytic Hz: 1, 1, 1.414, 2, 2; Ez: 1.414")
    print("   'low' = number of modes with 0 < omega < 0.5 (spurious), PR = participation ratio of the mode")
    dl = 0.005
    for side_cells in (10, 20, 30):
        L = side_cells * dl
        for center in ((0.37, 0.21), (0.13, 0.41), (0.29, 0.07)):
            try:
                scene, pec, cu = compile_pair(lambda kind: prism_scene(kind, dl, side_cells, center=center))
            except NotImplementedError:
                continue
            if side_cells == 20 and center != (0.13, 0.41):
                continue
            K_pec = k_matrix(pec)
            chords, h = chord_geometry(scene, pec)
            volumes = chord_volumes(chords, h, 0.0)
            vol_remove, removed, short = removal(pec, K_pec, chords, volumes)
            rows, cols, vals = [], [], []
            for t, r in enumerate(cu.trace_rows):
                for si, s in enumerate((-1, 1)):
                    for dof, w in r[s].items():
                        rows.append(2 * t + si); cols.append(dof); vals.append(w / cu.trace_length[t])
            W = sp.csr_matrix((vals, (rows, cols)), shape=(2 * len(cu.trace_rows), len(cu.e_volume)))
            results = {
                "pec": interior_modes(pec, K_pec, pec.e_volume, L),
                "cu_lim": interior_modes(cu, K_pec, cu.e_volume, L, constraint=W),
                "chord_remove": interior_modes(pec, K_pec, vol_remove, L),
            }
            volumes_f = chord_volumes(chords, h, 0.25)
            vol_f = removal(pec, K_pec, chords, volumes_f)[0]
            for stiffness in (1.0, 0.5):
                area = floored_areas(pec, K_pec, chords, volumes_f, vol_f, h, stiffness)
                floored = float(np.mean([area[ch["rows"][s]] > pec.h_true_area[ch["rows"][s]] * (1 + 1e-12)
                                         for ch in chords for s in (-1, 1)]))
                results[f"chord s{stiffness:g} ({floored:.2f})"] = interior_modes(_Areas(pec, area), K_pec, vol_f, L)
            print(f"   L = {side_cells} h, centre {center}, traces {len(chords)}, removal shortfall "
                  f"{short / max(sum(volumes.values()), 1e-300):.3f}")
            for name, modes in results.items():
                low = sum(1 for w, f, _ in modes if w < 0.5)
                local = sum(1 for w, f, pr in modes if 0.5 <= w <= 2.05 and pr < 0.05)
                band = [(w, f) for w, f, pr in modes if w >= 0.5 and pr >= 0.15][:5]
                print(f"   {name:22s} spurious <0.5: {low:4d}, localized in band: {local:3d}, extended: "
                      + " ".join(f"{w:.4f} {f}" for w, f in band))


# ---------------------------------------------------------------------------
# C. Leapfrog bound with dynamic chords and polygons; D. sliver bias
# ---------------------------------------------------------------------------

def plate(kind, normal=rt.NORMAL, half=(0.03, 0.025)):
    if kind == "cu":
        material = mw.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 4e9)); kw = {}
    else:
        material = hn.pec_material(); kw = dict(sheet_model="conformal_two_sided")
    return rt._cavity(mw.Structure(geometry=hn.rectangle_mesh(rt.CENTER, normal, *half, **kw), material=material))


def lambda_max(K, lh, me, chord_columns=None):
    scaled = sp.diags(np.sqrt(lh)) @ K @ sp.diags(1 / np.sqrt(me))
    op = (scaled.T @ scaled).tocsr()
    vals, vecs = eigsh(op, k=1, which="LA", tol=1e-6, v0=np.random.default_rng(0).standard_normal(op.shape[0]))
    share = float(np.sum(vecs[chord_columns, 0] ** 2)) if chord_columns is not None else 0.0
    return float(vals[0]), share


def chord_operator(scene, pec, cu, chords, h, volumes, e_volume, floors=True, stiffness=2.0, return_areas=False):
    nE, nH, T = len(pec.e_volume), len(pec.h_area), len(chords)
    surface = pec.surface
    K_pec = k_matrix(pec).tolil()
    K_cu = k_matrix(cu)
    # Orientation of each chord in its sub-face rows (sign of the reconstruction entries).
    orient = {}
    for t, ch in enumerate(chords):
        for s in (-1, 1):
            row = ch["rows"][s]
            diff = (K_cu.getrow(row) - sp.csr_matrix(K_pec.getrow(row))).toarray()[0]
            ref = np.zeros(nE)
            for dof, w in cu.trace_rows[t][s].items():
                ref[dof] += w
            orient[(t, s)] = 1.0 if diff @ ref > 0 else -1.0
    # Polygons: cut cells whose chords close a loop.
    cells = {}
    for t, ch in enumerate(chords):
        a, g = ch["face"]
        for cell in (g, tuple(np.array(g) - np.eye(3, dtype=int)[a])):
            cells.setdefault(cell, []).append(t)
    polygons = []
    for cell, ts in cells.items():
        ends = {}
        for t in ts:
            for p in (chords[t]["path"][0], chords[t]["path"][-1]):
                key = tuple(np.round(p / h, 7))
                ends[key] = ends.get(key, 0) + 1
        if len(ts) >= 3 and all(v == 2 for v in ends.values()):
            centre = np.mean([chords[t]["path"][0] for t in ts] + [chords[t]["path"][-1] for t in ts], 0)
            normal = surface.facets[int(surface.classify(centre[None])[1][0])].normal
            entries, area = [], 0.0
            for t in ts:
                p0, p1 = chords[t]["path"][0], chords[t]["path"][-1]
                mid = 0.5 * (p0 + p1)
                o = 1.0 if (p1 - p0) @ np.cross(normal, mid - centre) > 0 else -1.0
                area += 0.5 * o * float(normal @ np.cross(p0 - centre, p1 - centre))
                entries.append((t, o))
            polygons.append((entries, area))
    P = len(polygons)
    K = sp.lil_matrix((nH + P, nE + 2 * T))
    K[:nH, :nE] = K_pec
    vol = np.concatenate([e_volume, np.zeros(2 * T)])
    fixed = np.concatenate([pec.e_fixed, np.zeros(2 * T, dtype=bool)])
    for t, ch in enumerate(chords):
        for si, s in enumerate((-1, 1)):
            col = nE + 2 * t + si
            vol[col] = volumes[(t, s)]
            K[ch["rows"][s], col] = orient[(t, s)] * ch["length"]
    h_area = np.concatenate([pec.h_true_area.copy(), np.zeros(P)])
    h_dual = np.concatenate([pec.h_dual, np.zeros(P)])
    for p, (entries, area) in enumerate(polygons):
        ltilde = 0.0
        for s in (-1, 1):
            ltilde += np.mean([max(chords[t]["delta"][s], 1e-300) for t, _ in entries])
        for t, o in entries:
            v = (volumes[(t, -1)], volumes[(t, 1)])
            for si in range(2):
                K[nH + p, nE + 2 * t + si] = o * chords[t]["length"] * v[si] / (v[0] + v[1])
        h_area[nH + p], h_dual[nH + p] = area, ltilde
    K = K.tocsr()
    # Stiffness-matched magnetic masses (sub-faces as in the compiler, now counting the chord; polygons alike).
    if isinstance(floors, float):
        # Geometric sheet-face floor: A_eff = max(A_P, floors * h^2); sub-faces keep the PEC areas.
        h_area[:nH] = pec.h_area
        h_area[nH:] = np.maximum(h_area[nH:], floors * h * h)
    elif floors:
        full, parent = h * h, 4.0 / h
        sub_rows = [ch["rows"][s] for ch in chords for s in (-1, 1)]
        h_area[sub_rows] = pec.h_true_area[sub_rows]
        if floors in ("sheet_faces", "pec_only"):
            h_area[:nH] = pec.h_area
        for row in (sub_rows if floors is True else []) + (list(range(nH, nH + P)) if floors != "pec_only" else []):
            r = K.getrow(row)
            own = sum(v * v / vol[j] for j, v in zip(r.indices, r.data) if not fixed[j])
            limit_area = (h_dual[row] / h) * full * own / (stiffness * parent)
            h_area[row] = max(h_area[row], limit_area, 0.02 * full * (h_dual[row] / h if row >= nH else 1.0))
    else:
        h_area[:nH] = pec.h_area
    dyn = ~fixed
    if return_areas:
        return K[:, dyn], h_dual / (MU0 * h_area), EPS0 * vol[dyn], P, h_area, dyn
    return K[:, dyn], h_dual / (MU0 * h_area), EPS0 * vol[dyn], P


def part_c():
    print("C. leapfrog bound on the tilted cavity plate (lambda_max / lambda_Yee, lambda_Yee = 12 c^2 / h^2)")
    scene, pec, cu = compile_pair(lambda kind: plate(kind))
    h = 0.005
    yee = 12.0 / (EPS0 * MU0 * h * h)
    for name, c in (("pec (current)", pec), ("cu current", cu)):
        dyn = ~c.e_fixed
        K = k_matrix(c)[:, dyn]
        print(f"   {name:44s} {lambda_max(K, c.h_dual / (MU0 * c.h_area), EPS0 * c.e_volume[dyn])[0] / yee:.3f}")
    chords, _ = chord_geometry(scene, pec)
    K_pec = k_matrix(pec)
    nE = int(np.sum(~pec.e_fixed))
    for stiffness in (2.0, 1.0, 0.5):
        for delta_min in (0.25, 0.5):
            volumes = chord_volumes(chords, h, delta_min)
            for rule in ("additive", "remove"):
                e_volume = pec.e_volume if rule == "additive" else removal(pec, K_pec, chords, volumes)[0]
                K, lh, me, P = chord_operator(scene, pec, cu, chords, h, volumes, e_volume, stiffness=stiffness)
                chord_cols = np.arange(nE, nE + 2 * len(chords))
                lam, share = lambda_max(K, lh, me, chord_cols)
                print(f"   chord {rule:8s} stiffness {stiffness:3.1f} delta_min {delta_min:4.2f} h: {lam / yee:.3f} "
                      f"(chord share of top mode {share:.2f}, polygons {P})")


def part_d():
    print("D. sliver energy bias per side (cells of thickness per unit area); "
          "normal < 0 = missing normal energy, tangential > 0 = extra tangential capacitance")
    normals = {"(1,.6,.35)": rt.NORMAL, "30 deg about z": np.array((math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)),
               "board (.04,.025,1)": np.array((0.04, 0.025, 1.0)), "axis z, off-node": np.array((0.0, 0.0, 1.0))}
    for label, normal in normals.items():
        def build(kind, normal=normal):
            return plate(kind, normal=normal, half=(0.02, 0.02))
        scene, pec, cu = compile_pair(build)
        chords, h = chord_geometry(scene, pec)
        volumes = chord_volumes(chords, h, 0.0)
        K_pec = k_matrix(pec)
        _, removed, short = removal(pec, K_pec, chords, volumes)
        for rule, rem in (("additive", None), ("remove", removed)):
            bias = sliver_bias(chords, volumes, rem, normal, h)
            text = "  ".join(f"side {s:+d}: n {b['normal']:+.3f} t {b['tangential'][0]:+.3f}/{b['tangential'][1]:+.3f}"
                             for s, b in bias.items())
            print(f"   {label:20s} {rule:8s} {text}")


def targeted_floors(K, lh, me, target, nH, max_iterations=400, factor=1.3, share=0.2):
    """Raise the magnetic mass of the H rows carrying the top leapfrog mode until lambda_max <= target."""
    lh = lh.copy()
    scale = np.ones(len(lh))
    Ms = sp.diags(1 / np.sqrt(me))
    for iteration in range(max_iterations):
        scaled = sp.diags(np.sqrt(lh)) @ K @ Ms
        op = (scaled.T @ scaled).tocsr()
        vals, vecs = eigsh(op, k=1, which="LA", tol=1e-6, v0=np.random.default_rng(iteration).standard_normal(op.shape[0]))
        if vals[0] <= target:
            break
        y = (scaled @ vecs[:, 0]) ** 2
        rows = np.flatnonzero(y >= share * y.max())
        lh[rows] /= factor
        scale[rows] *= factor
    floored = scale > 1
    return lh, float(vals[0]), iteration, scale, floored


def part_e():
    print("E. chord basis CFL vs PEC-limit accuracy. Sub-faces keep today's PEC floors; sheet faces floored "
          "(stiffness 2); then targeted floors on the rows of the top leapfrog mode until lambda_max <= current copper")
    dl = 0.005
    cases = [(f"prism L={n}h", (lambda n: lambda kind: prism_scene(kind, dl, n))(n), n * dl) for n in (10, 30)]
    cases.append(("plate", lambda kind: plate(kind), None))
    for label, build, L in cases:
        scene, pec, cu = compile_pair(build)
        h = 0.005
        yee = 12.0 / (EPS0 * MU0 * h * h)
        dyn = ~cu.e_fixed
        target = lambda_max(k_matrix(cu)[:, dyn], cu.h_dual / (MU0 * cu.h_area), EPS0 * cu.e_volume[dyn])[0]
        chords, _ = chord_geometry(scene, pec)
        K_pec = k_matrix(pec)
        nH = len(pec.h_area)
        sub_rows = np.array([ch["rows"][s] for ch in chords for s in (-1, 1)])
        for delta_min in (0.0, 0.25):
            volumes = chord_volumes(chords, h, delta_min)
            vol_r = removal(pec, K_pec, chords, volumes)[0]
            K, lh, me, P, areas, _ = chord_operator(scene, pec, cu, chords, h, volumes, vol_r, floors="sheet_faces",
                                                    return_areas=True)
            start = lambda_max(K, lh, me)[0]
            lh2, final, iterations, scale, floored = targeted_floors(K, lh, me, target, nH)
            print(f"   {label:12s} delta_min {delta_min:4.2f} h: current copper {target / yee:.3f}, chord {start / yee:.3f} "
                  f"-> {final / yee:.3f} after {iterations} targeted steps; floored {int(floored[sub_rows].sum())} of "
                  f"{len(sub_rows)} sub-faces, {int(floored[:nH].sum()) - int(floored[sub_rows].sum())} other faces, "
                  f"{int(floored[nH:].sum())} of {P} sheet faces, max factor {scale.max():.2f}")
            if L is not None:
                for name, area in (("no targeted", areas[:nH].copy()), ("targeted", areas[:nH] * scale[:nH])):
                    modes = interior_modes(_Areas(pec, area), K_pec, vol_r, L)
                    band = [w for w, f, pr in modes if w >= 0.5 and pr >= 0.15][:3]
                    local = sum(1 for w, f, pr in modes if 0.5 <= w <= 2.05 and pr < 0.05)
                    print(f"      PEC limit, {name:12s} localized {local:3d}, extended: " + " ".join(f"{w:.4f}" for w in band))


def part_f():
    print("F. heavy additive chords (delta = max(d, delta_min), Cartesian volumes untouched), chord-aware floors: "
          "lambda_max / lambda_Yee and PEC-limit modes (PEC limit uses the same floored areas)")
    dl = 0.005
    cases = [(f"prism L={n}h", (lambda n: lambda kind: prism_scene(kind, dl, n))(n), n * dl) for n in (10, 30)]
    cases.append(("plate", lambda kind: plate(kind), None))
    for label, build, L in cases:
        scene, pec, cu = compile_pair(build)
        h = 0.005
        yee = 12.0 / (EPS0 * MU0 * h * h)
        dyn = ~cu.e_fixed
        target = lambda_max(k_matrix(cu)[:, dyn], cu.h_dual / (MU0 * cu.h_area), EPS0 * cu.e_volume[dyn])[0]
        dpec = ~pec.e_fixed
        lam_pec = lambda_max(k_matrix(pec)[:, dpec], pec.h_dual / (MU0 * pec.h_area), EPS0 * pec.e_volume[dpec])[0]
        chords, _ = chord_geometry(scene, pec)
        K_pec = k_matrix(pec)
        nH = len(pec.h_area)
        print(f"   {label}: current copper {target / yee:.3f}, current PEC {lam_pec / yee:.3f}")
        if L is not None:
            modes = interior_modes(pec, K_pec, pec.e_volume, L)
            print("      PEC path modes: " + " ".join([f"{w:.4f}" for w, f, pr in modes if w >= 0.5 and pr >= 0.15][:3]))
        for delta_min in (0.5, 0.75):
            volumes = chord_volumes(chords, h, delta_min)
            for stiffness in (2.0, 1.0):
                K, lh, me, P, areas, _ = chord_operator(scene, pec, cu, chords, h, volumes, pec.e_volume,
                                                        stiffness=stiffness, return_areas=True)
                lam = lambda_max(K, lh, me)[0]
                sub_rows = np.array([ch["rows"][s] for ch in chords for s in (-1, 1)])
                floored = int(np.sum(areas[sub_rows] > pec.h_true_area[sub_rows] * (1 + 1e-12)))
                text = ""
                if L is not None:
                    modes = interior_modes(_Areas(pec, areas[:nH].copy()), K_pec, pec.e_volume, L)
                    text = " modes " + " ".join([f"{w:.4f}" for w, f, pr in modes if w >= 0.5 and pr >= 0.15][:3])
                print(f"      delta_min {delta_min:4.2f} h, stiffness {stiffness:3.1f}: {lam / yee:.3f}, "
                      f"floored sub-faces {floored} of {len(sub_rows)};{text}")


def effective_admittance(material, dt):
    from witwin.maxwell.compiler.copper_foil import compile_copper_response
    material._frequencies((1e9,))
    resp = compile_copper_response(material)
    rates, B, D = resp.rates.double(), resp.state_space.B.double(), resp.state_space.D.double()
    drive = dt / 2 * B / (1 + dt * rates / 2)[:, None]
    Y = (D + B.T @ drive).numpy()
    return 0.5 * (Y + Y.T)


def part_g(rule="remove", kappas=(None,), floor_mode="sheet_faces", case_names=None):
    print("G. locally implicit chords (Newmark 1/4 on chord rows and columns, A_e = Cartesian block): explicit bound "
          "and the implicit chord system S_c = M_c/dt^2 + G/(2 dt) + A_cc/4 - (1/16) A_cx (M_x/dt^2)^-1 A_xc")
    dl = 0.005
    h = 0.005
    yee = 12.0 / (EPS0 * MU0 * h * h)
    materials = {"35um Cu": mw.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 4e9)),
                 "g=4": hn.resistive_foil(4 / hn.ETA0), "g=0.5": hn.resistive_foil(0.5 / hn.ETA0), "no sheet (Y=0)": None}
    cases = [("plate", lambda kind: plate(kind)), ("prism L=10h", lambda kind: prism_scene(kind, dl, 10)),
             ("prism L=30h", lambda kind: prism_scene(kind, dl, 30))]
    if case_names is not None:
        cases = [case for case in cases if case[0] in case_names]
    for label, build in cases:
        scene, pec, cu = compile_pair(build)
        dyn = ~cu.e_fixed
        lam_cu = lambda_max(k_matrix(cu)[:, dyn], cu.h_dual / (MU0 * cu.h_area), EPS0 * cu.e_volume[dyn])[0]
        chords, _ = chord_geometry(scene, pec)
        K_pec = k_matrix(pec)
        for kappa in kappas:
          if kappa is None:
            volumes = chord_volumes(chords, h, 0.25)
            vol_r = removal(pec, K_pec, chords, volumes)[0]
          else:
            volumes = {key: kappa * v for key, v in chord_volumes(chords, h, 0.0).items()}
            vol_r = pec.e_volume
          K, lh, me, P = chord_operator(scene, pec, cu, chords, h, volumes, vol_r, floors=floor_mode)
          T = len(chords)
          nx = K.shape[1] - 2 * T
          A = (K.T @ sp.diags(lh) @ K).tocsr()
          Axx, Axc, Acc = A[:nx, :nx], A[:nx, nx:], A[nx:, nx:]
          lam_e = lambda_max(K[:, :nx], lh, me[:nx])[0]
          dt = 0.985 * 2 / math.sqrt(lam_e)
          widths = np.array([ch["width"] for ch in chords]); lengths = np.array([ch["length"] for ch in chords])
          base = (sp.diags(me[nx:] / dt ** 2) + Acc / 4 - (Axc.T @ sp.diags(dt ** 2 / me[:nx]) @ Axc) / 16).tocsr()
          nnz = base.nnz / base.shape[0]
          print(f"   {label} chord mass {kappa}: current copper {lam_cu / yee:.3f}, explicit part (PEC limit, new masses) {lam_e / yee:.3f}; "
                f"S_c non-zeros per row {nnz:.1f}")
          for name, material in materials.items():
              Sc = base
              if material is not None:
                  Y = effective_admittance(material, dt)
                  G = sp.kron(sp.diags(lengths * widths), sp.csr_matrix(Y))
                  Sc = (base + G / (2 * dt)).tocsr()
              dense = Sc.toarray()
              # block Jacobi (one 2x2 per chord pair)
              blocks = np.zeros_like(dense)
              for t in range(T):
                  blocks[2 * t:2 * t + 2, 2 * t:2 * t + 2] = dense[2 * t:2 * t + 2, 2 * t:2 * t + 2]
              w, v = np.linalg.eigh(blocks)
              inv_sqrt = v @ np.diag(1 / np.sqrt(w)) @ v.T
              ev = np.linalg.eigvalsh(inv_sqrt @ dense @ inv_sqrt)
              print(f"      {name:16s} block-Jacobi spectrum [{ev[0]:.3f}, {ev[-1]:.3f}], condition {ev[-1] / ev[0]:.2f}")


def part_g3(kappa=0.25):
    print("G3. implicit set = chords + split segments (light additive chords, Cartesian masses unchanged): explicit "
          "bound and block-Jacobi condition of the implicit Schur system")
    dl = 0.005
    h = 0.005
    yee = 12.0 / (EPS0 * MU0 * h * h)
    materials = {"35um Cu": mw.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 4e9)),
                 "g=0.5": hn.resistive_foil(0.5 / hn.ETA0), "no sheet (Y=0)": None}
    cases = [("plate", lambda kind: plate(kind)), ("prism L=10h", lambda kind: prism_scene(kind, dl, 10)),
             ("prism L=30h", lambda kind: prism_scene(kind, dl, 30)),
             ("board normal", lambda kind: plate(kind, normal=np.array((0.04, 0.025, 1.0)), half=(0.03, 0.025)))]
    for label, build in cases:
        scene, pec, cu = compile_pair(build)
        dyn = ~cu.e_fixed
        lam_cu = lambda_max(k_matrix(cu)[:, dyn], cu.h_dual / (MU0 * cu.h_area), EPS0 * cu.e_volume[dyn])[0]
        dpec = ~pec.e_fixed
        lam_pec = lambda_max(k_matrix(pec)[:, dpec], pec.h_dual / (MU0 * pec.h_area), EPS0 * pec.e_volume[dpec])[0]
        chords, _ = chord_geometry(scene, pec)
        volumes = {key: kappa * v for key, v in chord_volumes(chords, h, 0.0).items()}
        K, lh, me, P = chord_operator(scene, pec, cu, chords, h, volumes, pec.e_volume, floors="sheet_faces")
        T = len(chords)
        nx = K.shape[1] - 2 * T
        split = np.concatenate([pec.e_split[dpec], np.ones(2 * T, dtype=bool)])
        I, R = np.flatnonzero(split), np.flatnonzero(~split)
        A = (K.T @ sp.diags(lh) @ K).tocsr()
        lam_e = lambda_max(K[:, R], lh, me[R])[0]
        dt = 0.985 * 2 / math.sqrt(lam_e)
        AII, ARI = A[I][:, I], A[R][:, I]
        base = (sp.diags(me[I] / dt ** 2) + AII / 4 - (ARI.T @ sp.diags(dt ** 2 / me[R]) @ ARI) / 16).tocsr()
        print(f"   {label}: current copper {lam_cu / yee:.3f}, current PEC {lam_pec / yee:.3f}, explicit part {lam_e / yee:.3f}; "
              f"implicit unknowns {len(I)} ({2 * T} chord), non-zeros per row {base.nnz / base.shape[0]:.1f}")
        widths = np.array([ch["width"] for ch in chords]); lengths = np.array([ch["length"] for ch in chords])
        nI_x = len(I) - 2 * T
        for name, material in materials.items():
            Sc = base
            if material is not None:
                Y = effective_admittance(material, dt)
                G = sp.block_diag([sp.csr_matrix((nI_x, nI_x)), sp.kron(sp.diags(lengths * widths), sp.csr_matrix(Y))])
                Sc = (base + G / (2 * dt)).tocsr()
            dense = Sc.toarray()
            d = np.diag(dense).copy()
            blocks = np.diag(d)
            for t in range(T):
                j = nI_x + 2 * t
                blocks[j:j + 2, j:j + 2] = dense[j:j + 2, j:j + 2]
            w, v = np.linalg.eigh(blocks)
            inv_sqrt = v @ np.diag(1 / np.sqrt(w)) @ v.T
            ev = np.linalg.eigvalsh(inv_sqrt @ dense @ inv_sqrt)
            print(f"      {name:16s} block-Jacobi spectrum [{ev[0]:.3f}, {ev[-1]:.3f}], condition {ev[-1] / ev[0]:.2f}")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["a", "b", "c", "d"]
    for part in parts:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d, "e": part_e, "f": part_f, "g": part_g, "g2": lambda: part_g("additive", (1.0, 0.25, 0.1)), "g3": part_g3, "g4": lambda: part_g("additive", (0.1, 1e-2, 1e-6), "pec_only", ("plate", "prism L=10h")), "g5": lambda: [part_g("additive", (0.1,), a, None) for a in (0.1, 0.25)]}[part]()

"""Independent accuracy check of the printed CPW / coupled-pair line results.

Subcommands (CPU only):

* ``rows``      export the fixed 2.5 mm public V/I functional (sample points,
                components, weights) and the FDTD transverse grid. This is the
                only step that imports the FDTD package, for geometry only.
* ``reference`` run the independent cross-section reference
                (benchmark/line_cross_section_reference.py): refinement,
                Richardson extrapolation, box, thickness and Cu-loss studies.
* ``compare``   compare the recorded FDTD two-length results against it under
                the tolerances declared in the accuracy README.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/assessments/printed-lines-2026-10-29-accuracy"
BRIDGE = ROOT / "docs/assessments/printed-lines-2026-10-28/public-line-bridge-remaining11-actual"
V4 = ROOT / "docs/assessments/printed-lines-2026-10-28/v4-all12-material-controls-actual/actual-fixed11-controller/original-all-material-reduction.json"
FREQUENCIES = (0.8e9, 0.9e9, 1.0e9, 1.1e9, 1.2e9)
SIGMA = 0.01
# Actual FDTD steps (recorded clocks) and longitudinal Yee step.
FDTD_DT = {"cpw": 2.7797007933179227e-12, "differential_pair": 2.5017306581323802e-12, "stripline": 4.8145830038661275e-12}
FDTD_DX_LONG = 0.0025
# Proposed FDTD convergence grids (see printed_line_accuracy_queue.sh).
CONVERGENCE_GRIDS = {"T0_dx2.5": dict(dx=.0025), "T1_dx1.25_current": {}, "T2_dx0.625": dict(dx=.000625),
                     "T3_dx0.3125": dict(dx=.0003125), "P2_padding0.16": dict(padding=.16)}
# Declared before comparison; see README "Acceptance tolerances".
TOLERANCE = dict(beta=0.01, alpha_total=0.05, alpha_dielectric=0.05, alpha_conductor=0.10,
                 z_same_functional=0.02, z_definition=0.05)


def encode(value):
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"real", "imag"}:
            return complex(value["real"], value["imag"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


# ---------------------------------------------------------------- rows ----
def export_rows(output):
    from scripts.dev.fdtd.printed_line_public_execution import column_case, _scene, build_public_column
    from witwin.maxwell.scene import prepare_scene
    record = {}
    for kind in ("cpw", "differential_pair"):
        case = column_case(0)
        plan = build_public_column(kind, case)["plan"]
        scene, _ = _scene(kind, case)
        prepared = prepare_scene(scene)
        nodes = [np.asarray(getattr(prepared, a + "_nodes64")) for a in "xyz"]

        def where(name, component):
            if name in plan.queries:
                return [float(v) for v in plan.queries[name].points_m[0]]
            native, index = plan.expected_points[name]
            if native != component:
                raise ValueError("Functional row component changed.")
            axis = "xyz".index(component[-1].lower())
            out = []
            for a in range(3):
                half = (a == axis) if component.startswith("E") else (a != axis)
                out.append(float((nodes[a][index[a]] + nodes[a][index[a] + 1]) / 2 if half else nodes[a][index[a]]))
            return out
        stations = [[[[c, where(n, c), float(w)] for n, c, w in rows] for rows in station] for station in plan.rows]
        record[kind] = dict(stations=stations, y_nodes=nodes[1].tolist(), z_nodes=nodes[2].tolist(),
                            station_x=[s[0][0][1][0] for s in stations])
    # Transverse nodes of every proposed convergence grid (geometry only).
    from benchmark.scenes.rf.printed_lines import printed_line_scene
    record["grids"] = {}
    for label, overrides in CONVERGENCE_GRIDS.items():
        case = column_case(0)
        dx, dl, padding = overrides.get("dx", case.dx), overrides.get("longitudinal_dl", case.longitudinal_dl), overrides.get("padding", case.padding)
        scene = printed_line_scene("cpw", dx=dx, longitudinal_dl=dl, axial_half_length=case.axial_half_length,
                                   port_separation=case.separation, transverse_padding=padding,
                                   transverse_refinement_radius=.02, device="cpu")
        record["grids"][label] = dict(overrides=overrides, y_nodes=list(map(float, scene.grid.y_coords)),
                                      z_nodes=list(map(float, scene.grid.z_coords)))
    record["scope"] = ("Geometry only: sample coordinates/weights of the fixed 2.5 mm public V/I functional "
                       "(length index 0), the executed FDTD transverse nodes and the proposed convergence grids.")
    Path(output).write_text(json.dumps(record, indent=1) + "\n")


# ------------------------------------------------------------ helpers ----
def richardson(values, ratio=2.0):
    """Extrapolate the last three refinement levels; uncertainty is the larger of
    the extrapolation correction and its change from the previous triple."""
    v = np.asarray(values, dtype=np.float64)
    out = dict(levels=v.tolist())
    if len(v) < 3:
        out.update(value=float(v[-1]), uncertainty=float(abs(v[-1] - v[-2])) if len(v) > 1 else float("nan"), order=None)
        return out

    def triple(a, b, c):
        d1, d2 = b - a, c - b
        if d2 == 0:
            return c, float("inf"), 0.0
        r = d1 / d2
        p = math.log(abs(r), ratio) if r > 0 else float("nan")
        p_used = p if (math.isfinite(p) and 0.5 <= p <= 4) else 1.0
        extrap = c + d2 / (ratio ** p_used - 1)
        return extrap, p, p_used
    e, p, used = triple(*v[-3:])
    unc = abs(e - v[-1])
    if len(v) >= 4:
        e_prev, _, _ = triple(*v[-4:-1])
        unc = max(unc, abs(e - e_prev))
    out.update(value=float(e), uncertainty=float(unc), order=float(p), order_used=float(used),
               relative_uncertainty=float(unc / abs(e)) if e else float("inf"))
    return out


def analog_frequency(frequency, dt):
    """Yee time discretization replaces omega by (2/dt) sin(omega dt / 2)."""
    x = math.pi * frequency * dt
    return frequency * math.sin(x) / x


def analog_beta(beta_2d, dl=FDTD_DX_LONG):
    """Yee longitudinal step: (2/dl) sin(beta dl/2) equals the 2D eigenvalue root."""
    return (2 / dl) * math.asin(beta_2d * dl / 2)


def sample_functional(fw, mode, rows, station_x):
    total = 0j
    for component, (x, y, z), weight in rows:
        phase = np.exp(-1j * mode["kx"] * (x - station_x))
        total += weight * fw.sample(mode, component, y, z) * phase
    return total


def classify_pair(fw, mode):
    from benchmark.line_cross_section_reference import PAIR_OFFSET, HEIGHT
    vp = fw.line_voltage(mode, (PAIR_OFFSET, 0.0), (PAIR_OFFSET, -HEIGHT))
    vn = fw.line_voltage(mode, (-PAIR_OFFSET, 0.0), (-PAIR_OFFSET, -HEIGHT))
    ip, i_n = fw.conductor_current(mode, 1), fw.conductor_current(mode, 2)
    return ("even" if abs(vp + vn) > abs(vp - vn) else "odd"), np.array([vp, vn]), np.array([ip, i_n])


def mixed(V, I):
    """Columns = modes; rows = (positive, negative) conductors -> mixed basis."""
    A = np.array([[.5, .5], [1., -1.]])
    return A @ V, np.linalg.inv(A).T @ I


def line_modes(fw, kind, guess, count=4):
    modes = fw.solve(guess, count=count)
    if kind != "differential_pair":
        return {"line": modes[0]}
    out = {}
    for mode in modes[:2]:
        label, V, I = classify_pair(fw, mode)
        mode["V"], mode["I"] = V, I
        out[label] = mode
    if set(out) != {"even", "odd"}:
        raise RuntimeError("Pair even/odd modes were not both found.")
    return out


def mode_impedances(fw, kind, modes):
    """Continuum V/I (path voltage, Ampere current), P-V and P-I, exp(-i w t)."""
    from benchmark.line_cross_section_reference import WIDTH, GAP
    rec = {}
    if kind == "cpw":
        m = modes["line"]
        V = fw.line_voltage(m, (0.0, 0.0), (WIDTH / 2 + GAP, 0.0))
        I = fw.conductor_current(m, 1)
        P = 0.5 * np.sum(fw.Me * m["e"] * np.conj(m["hhat"]))
        rec["line"] = dict(VI=np.conj(V / I), PV=np.conj(abs(V) ** 2 / (2 * np.conj(P))), PI=np.conj(2 * P / abs(I) ** 2))
        return rec
    order = ("odd", "even")
    V = np.stack([modes[k]["V"] for k in order], axis=1)
    I = np.stack([modes[k]["I"] for k in order], axis=1)
    Vm, Im = mixed(V, I)
    rec["mixed_VI_matrix"] = np.conj(Vm @ np.linalg.inv(Im))
    for k, index in (("odd", 1), ("even", 0)):
        m = modes[k]
        P = 0.5 * np.sum(fw.Me * m["e"] * np.conj(m["hhat"]))
        col = order.index(k)
        v, i = Vm[index, col], Im[index, col]
        rec[k] = dict(VI_per_line=np.conj(m["V"][0] / m["I"][0]), mixed_VI=np.conj(v / i),
                      mixed_PV=np.conj(abs(v) ** 2 / (2 * np.conj(P))), mixed_PI=np.conj(2 * P / abs(i) ** 2))
    return rec


def functional_impedance(fw, kind, modes, rows):
    data = rows[kind]
    if kind == "cpw":
        station = len(data["stations"]) // 2
        Vrows, Irows = data["stations"][station]
        x0 = data["station_x"][station]
        m = modes["line"]
        V = sample_functional(fw, m, Vrows, x0)
        I = sample_functional(fw, m, Irows, x0)
        return dict(Z=np.conj(V / I), V=V, I=I)
    # pair: stations are flattened (common, differential) per position
    index = 2 * (len(data["stations"]) // 4)
    (Vc_rows, Ic_rows), (Vd_rows, Ip_rows) = data["stations"][index], data["stations"][index + 1]
    x0 = data["station_x"][index]
    order = ("odd", "even")
    Vs, Is = [], []
    for k in order:
        m = modes[k]
        v1, v2 = sample_functional(fw, m, Vc_rows, x0), sample_functional(fw, m, Vd_rows, x0)
        i1, i2 = sample_functional(fw, m, Ic_rows, x0), sample_functional(fw, m, Ip_rows, x0)
        Vs.append([v1, v1 - v2])
        Is.append([i2, i1 - i2])
    Vm, Im = mixed(np.array(Vs).T, np.array(Is).T)
    return dict(mixed_matrix=np.conj(Vm @ np.linalg.inv(Im)))


# --------------------------------------------------------- reference ----
def reference(output, quick=False):
    import benchmark.line_cross_section_reference as R
    rows = json.loads((DOCS / "functional-rows.json").read_text())
    report = dict(schema="printed-line-accuracy-reference/1", frequencies_hz=FREQUENCIES, sigma_S_per_m=SIGMA,
                  copper=dict(thickness_m=R.COPPER_THICKNESS, conductivity_S_per_m=R.COPPER_CONDUCTIVITY,
                              per_frequency=[R.foil_surface_impedance(f) for f in FREQUENCIES]))
    timing = {}
    base = R.MeshSpec(2e-4, 4e-3, 0.3)       # zero-thickness geometry
    thick = R.MeshSpec(4e-6, 4e-3, 0.3)      # resolves 35 um foil corners
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    def fw_modes(section, spec_or_nodes, f, kind, guess_index):
        mesh = R.Mesh(section, spec_or_nodes) if isinstance(spec_or_nodes, R.MeshSpec) else R.Mesh(section, nodes=spec_or_nodes)
        fw = R.FullWave(mesh, f)
        k0 = 2 * math.pi * f / R.C0
        return fw, line_modes(fw, kind, k0 * guess_index), mesh

    def analog_nodes(kind, section):
        data = rows[kind]
        y, z = np.asarray(data["y_nodes"]), np.asarray(data["z_nodes"])
        y0, y1, z0, z1 = section.box
        return (y[(y >= y0 - 1e-12) & (y <= y1 + 1e-12)], z[(z >= z0 - 1e-12) & (z <= z1 + 1e-12)])

    levels_fw = (1, 2, 3) if not quick else (1, 2)
    for kind in ("cpw", "differential_pair"):
        rk = report[kind] = {}
        guess = 1.5 if kind == "cpw" else 1.45
        names = ("line",) if kind == "cpw" else ("odd", "even")
        sec0 = R.printed_line_section(kind, thickness=0.0, sigma=SIGMA)
        sec35 = R.printed_line_section(kind, thickness=R.COPPER_THICKNESS, sigma=SIGMA)
        # 1. propagation, model geometry (t = 0), every frequency
        prop = {n: dict(beta=[], alpha_d=[]) for n in names}
        zfw, zfun = [], []
        for f in FREQUENCIES:
            per = {n: dict(beta=[], alpha=[]) for n in names}
            zlev, flev = [], []
            for L in levels_fw:
                fw, modes, mesh = fw_modes(sec0, base.refined(L), f, kind, guess)
                for n in names:
                    per[n]["beta"].append(modes[n]["beta_rad_per_m"])
                    per[n]["alpha"].append(modes[n]["alpha_np_per_m"])
                zlev.append(mode_impedances(fw, kind, modes))
                flev.append(functional_impedance(fw, kind, modes, rows))
                log(kind, "fw t=0", f, L, mesh.ny, mesh.nz, {n: (round(per[n]["beta"][-1], 6), round(per[n]["alpha"][-1], 6)) for n in names})
            for n in names:
                prop[n]["beta"].append(richardson(per[n]["beta"]))
                prop[n]["alpha_d"].append(richardson(per[n]["alpha"]))
            zfw.append(zlev)
            zfun.append(flev)
        if kind == "differential_pair" and not quick:
            fw, modes, mesh = fw_modes(sec0, base.refined(4), 1e9, kind, guess)
            rk["fw_level4_1GHz"] = {n: dict(beta=modes[n]["beta_rad_per_m"], alpha=modes[n]["alpha_np_per_m"]) for n in names}
            log(kind, "fw L4", rk["fw_level4_1GHz"])
        if kind == "cpw":
            exact = [R.homogeneous_gamma(f, 2.25, SIGMA) for f in FREQUENCIES]
            rk["exact_homogeneous"] = dict(alpha_d=[a for a, _ in exact], beta=[b for _, b in exact])
        rk["propagation_t0"] = prop
        rk["impedance_fw_levels"] = zfw
        rk["impedance_functional_levels"] = zfun
        # 2. box (open-boundary) study at 1 GHz, coarse level; delta vs default box
        box = {}
        Lb = 1
        for walls in ("pec", "pmc"):
            for scale in (1, 2, 4):
                sec = R.printed_line_section(kind, thickness=0.0, sigma=SIGMA, half_y=0.14 * scale, half_z=0.13 * scale, walls=walls)
                fw, modes, mesh = fw_modes(sec, base.refined(Lb), 1e9, kind, guess)
                imp = mode_impedances(fw, kind, modes)
                box[f"{walls}_x{scale}"] = {n: dict(beta=modes[n]["beta_rad_per_m"], alpha=modes[n]["alpha_np_per_m"]) for n in names}
                box[f"{walls}_x{scale}"]["impedance"] = imp
                log(kind, "box", walls, scale, {n: box[f"{walls}_x{scale}"][n] for n in names})
        rk["box_level1_1GHz"] = box
        # quasi-static Z (t = 0) refinement, default box, plus open conformal for CPW
        qsz = {n: [] for n in names}
        for L in range(0, 5 if kind == "cpw" else 4):
            mesh = R.Mesh(sec0, base.refined(L))
            qs = R.QuasiStatic(mesh)
            for n, m in zip(names, qs.modes(1e9)):
                qsz[n].append(abs(m["voltage"][0] / m["current"][0]))
            log(kind, "qs Z", L, {n: qsz[n][-1] for n in names})
        rk["qs_Z_per_line_t0"] = {n: richardson(v) for n, v in qsz.items()}
        if kind == "cpw":
            rk["conformal_open_Z_ohm"] = R.cpw_conformal()["z0_ohm"]
        # 3. thickness and Cu loss: full-wave (dispersion-inclusive) and quasi-static
        fwc = {n: dict(beta=[], alpha_d=[], alpha_c=[]) for n in names}
        for f in FREQUENCIES:
            rs = R.foil_surface_impedance(f)["surface_resistance_ohm"]
            per = {n: dict(beta=[], alpha=[], ac=[]) for n in names}
            for L in (0, 1, 2):
                fw, modes, mesh = fw_modes(sec35, thick.refined(L), f, kind, guess)
                for n in names:
                    per[n]["beta"].append(modes[n]["beta_rad_per_m"])
                    per[n]["alpha"].append(modes[n]["alpha_np_per_m"])
                    per[n]["ac"].append(fw.conductor_alpha(modes[n], rs))
                log(kind, "fw t=35um", f, L, {n: (round(per[n]["beta"][-1], 6), round(per[n]["alpha"][-1], 6), round(per[n]["ac"][-1], 7)) for n in names})
            for n in names:
                fwc[n]["beta"].append(richardson(per[n]["beta"]))
                fwc[n]["alpha_d"].append(richardson(per[n]["alpha"]))
                fwc[n]["alpha_c"].append(richardson(per[n]["ac"]))
        rk["full_wave_t35um"] = fwc
        qsc = {n: dict(alpha_c=[], Z=[], eps_eff=[]) for n in names}
        rs1 = R.foil_surface_impedance(1e9)["surface_resistance_ohm"]
        for L in (0, 1, 2, 3):
            qs = R.QuasiStatic(R.Mesh(sec35, thick.refined(L)))
            for n, m in zip(names, qs.modes(1e9, surface_resistance=rs1)):
                qsc[n]["alpha_c"].append(m["alpha_c_np_per_m"])
                qsc[n]["Z"].append(abs(m["voltage"][0] / m["current"][0]))
                qsc[n]["eps_eff"].append(m["eps_eff"])
            log(kind, "qs t=35um", L, {n: qsc[n]["alpha_c"][-1] for n in names})
        rk["quasi_static_t35um_1GHz"] = {n: {k: richardson(v) for k, v in d.items()} for n, d in qsc.items()}
        # 4. FDTD-grid analog: same transverse nodes, PEC at the inner PML face
        analog = {n: dict(beta_2d=[], beta_fdtd_equivalent=[], alpha_d=[]) for n in names}
        for f in FREQUENCIES:
            fa = analog_frequency(f, FDTD_DT[kind])
            fw, modes, mesh = fw_modes(sec0, analog_nodes(kind, sec0), fa, kind, guess)
            for n in names:
                analog[n]["beta_2d"].append(modes[n]["beta_rad_per_m"])
                analog[n]["beta_fdtd_equivalent"].append(analog_beta(modes[n]["beta_rad_per_m"]))
                analog[n]["alpha_d"].append(modes[n]["alpha_np_per_m"])
            if f == 1e9:
                rs = R.foil_surface_impedance(f)["surface_resistance_ohm"]
                analog["yee_sheet_alpha_c_1GHz"] = {n: fw.conductor_alpha(modes[n], rs) for n in names}
                analog["impedance_1GHz"] = mode_impedances(fw, kind, modes)
                analog["functional_1GHz"] = functional_impedance(fw, kind, modes, rows)
                qs = R.QuasiStatic(mesh)
                analog["qs_Z_per_line"] = {n: abs(m["voltage"][0] / m["current"][0]) for n, m in zip(names, qs.modes(1e9))}
            log(kind, "analog", f, {n: analog[n]["beta_fdtd_equivalent"][-1] for n in names})
        analog["grid"] = dict(ny=mesh.ny, nz=mesh.nz, box=list(sec0.box))
        rk["fdtd_grid_analog"] = analog
        # 5. zero-thickness sheet loss on uniform grids (regularization diagnostic)
        uni = {}
        for h in (2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4):
            qs = R.QuasiStatic(R.Mesh(sec0, R.MeshSpec(h, h, 0.0)))
            uni[str(h)] = {n: m["alpha_c_np_per_m"] for n, m in zip(names, qs.modes(1e9, surface_resistance=rs1))}
            log(kind, "uniform sheet", h, uni[str(h)])
        rk["uniform_zero_thickness_sheet_alpha_c_1GHz"] = uni
        timing[kind] = time.time() - t0
        Path(output).write_text(json.dumps(encode(report), indent=1) + "\n")

    # Stripline: the V4 material-separated Cu/dielectric controls.
    rs_ = report["stripline"] = {}
    secs = R.printed_line_section("stripline", thickness=R.COPPER_THICKNESS)
    ac, zt = [], []
    for L in (0, 1, 2, 3, 4):
        qs = R.QuasiStatic(R.Mesh(secs, thick.refined(L)))
        m = qs.modes(1e9, surface_resistance=R.foil_surface_impedance(1e9)["surface_resistance_ohm"])[0]
        ac.append(m["alpha_c_np_per_m"])
        zt.append(abs(m["voltage"][0] / m["current"][0]))
        log("stripline qs t=35um", L, ac[-1], zt[-1])
    rs_["alpha_c_1GHz_t35um"] = richardson(ac)
    rs_["Z_t35um"] = richardson(zt)
    rs_["cohn_Z_t0_ohm"] = R.stripline_cohn()
    rs_["exact_homogeneous"] = {str(s): [R.homogeneous_gamma(f, 2.25, s) for f in FREQUENCIES] for s in (0.0, SIGMA)}
    # V4 grid: uniform 2.5 mm over the physical domain (PML outside), +-0.14 m
    secv = R.printed_line_section("stripline", thickness=0.0)
    meshv = R.Mesh(secv, R.MeshSpec(2.5e-3, 2.5e-3, 0.0))
    fa = analog_frequency(1e9, FDTD_DT["stripline"])
    fw = R.FullWave(meshv, fa)
    m = fw.solve(2 * math.pi * fa / R.C0 * 1.5, count=2)[0]
    qs = R.QuasiStatic(meshv)
    q = qs.modes(1e9, surface_resistance=R.foil_surface_impedance(1e9)["surface_resistance_ohm"])[0]
    rs_["v4_grid_analog"] = dict(beta_fdtd_equivalent=analog_beta(m["beta_rad_per_m"]),
                                 yee_sheet_alpha_c=fw.conductor_alpha(m, R.foil_surface_impedance(1e9)["surface_resistance_ohm"]),
                                 qs_sheet_alpha_c=q["alpha_c_np_per_m"], qs_Z=abs(q["voltage"][0] / q["current"][0]))
    report["sanity_closed_forms"] = dict(
        cpw_conformal=R.cpw_conformal(), cpw_owyang_wu_alpha_c=[R.cpw_conductor_loss_owyang_wu(f) for f in FREQUENCIES],
        stripline_cohn_Z=R.stripline_cohn(), stripline_wheeler_alpha_c=[R.stripline_conductor_loss_wheeler(f) for f in FREQUENCIES])
    report["timing_s"] = dict(timing, total=time.time() - t0)
    Path(output).write_text(json.dumps(encode(report), indent=1) + "\n")
    log("done")


def predict(output):
    """Yee-grid analog of each proposed FDTD grid: same transverse nodes, PEC at
    the inner PML face, Yee time/longitudinal dispersion applied analytically."""
    import benchmark.line_cross_section_reference as R
    rows = json.loads((DOCS / "functional-rows.json").read_text())
    out = dict(schema="printed-line-accuracy-grid-prediction/1", frequency_hz=1e9, grids={})
    f = 1e9
    rs = R.foil_surface_impedance(f)["surface_resistance_ohm"]
    for label, grid in rows["grids"].items():
        out["grids"][label] = record = dict(overrides=grid["overrides"])
        padding = grid["overrides"].get("padding", .08)
        for kind in ("cpw", "differential_pair"):
            names = ("line",) if kind == "cpw" else ("odd", "even")
            hy, hz = .06 + padding, .05 + padding  # PML lies outside the declared domain
            sec = R.printed_line_section(kind, thickness=0.0, sigma=SIGMA, half_y=hy, half_z=hz)
            y, z = np.asarray(grid["y_nodes"]), np.asarray(grid["z_nodes"])
            y0, y1, z0, z1 = sec.box
            nodes = (y[(y >= y0 - 1e-12) & (y <= y1 + 1e-12)], z[(z >= z0 - 1e-12) & (z <= z1 + 1e-12)])
            mesh = R.Mesh(sec, nodes=nodes)
            # Same CFL rule scaling as the executed columns: dt ~ 1/sqrt(sum 1/d^2).
            d = [grid["overrides"].get("longitudinal_dl", .0025), np.min(np.diff(nodes[0])), np.min(np.diff(nodes[1]))]
            dt = FDTD_DT[kind] * math.sqrt(sum(1 / v ** 2 for v in (.0025, .00125, .00125))) / math.sqrt(sum(1 / v ** 2 for v in d))
            fw = R.FullWave(mesh, analog_frequency(f, dt))
            modes = line_modes(fw, kind, 2 * math.pi * f / R.C0 * (1.5 if kind == "cpw" else 1.45))
            qs = R.QuasiStatic(mesh)
            record[kind] = dict(
                nodes=[mesh.ny, mesh.nz], dt_s=dt,
                beta_fdtd_equivalent={n: analog_beta(modes[n]["beta_rad_per_m"], d[0]) for n in names},
                alpha_d={n: modes[n]["alpha_np_per_m"] for n in names},
                yee_sheet_alpha_c={n: fw.conductor_alpha(modes[n], rs) for n in names},
                qs_Z_per_line={n: abs(m["voltage"][0] / m["current"][0]) for n, m in zip(names, qs.modes(f))},
                impedance=mode_impedances(fw, kind, modes),
                functional=functional_impedance(fw, kind, modes, rows))
            print(label, kind, record[kind]["beta_fdtd_equivalent"], record[kind]["yee_sheet_alpha_c"], flush=True)
    Path(output).write_text(json.dumps(encode(out), indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rows", "reference", "predict", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name == "reference":
            p.add_argument("--quick", action="store_true")
        if name == "compare":
            p.add_argument("--fdtd-dir", type=Path, default=BRIDGE)
    args = parser.parse_args()
    if args.command == "rows":
        export_rows(args.output)
    elif args.command == "reference":
        reference(args.output, quick=args.quick)
    elif args.command == "predict":
        predict(args.output)
    else:
        from scripts.dev.fdtd.printed_line_accuracy_compare import compare
        compare(args.output, fdtd_dir=args.fdtd_dir)


if __name__ == "__main__":
    main()

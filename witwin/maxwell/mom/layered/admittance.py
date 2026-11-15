"""Surface-admittance lossy conductors: differential surface admittance (DSA).

Each lossy body V (one conductivity, closed boundary dV = exposed faces S plus
contact faces C shared with PEC) is replaced by the surrounding background
medium carrying the equivalent electric current J_D = n x (H_c - H_b) on dV.
H_c is the conductor interior field and H_b the background interior field with
the same tangential electric field e on dV (e = 0 on C). The layered exterior
MPIE is reused unchanged; only J_D on S couples to e through

    (Z_ext + W) x = b,   W = R_x^T G_S Y_S^-1 R_x,

where x are the exterior junction-RWG coefficients and R_x restricts them to
the RT space on S (S-interior RWGs plus S-side half functions at contact
seams). e is expanded in that same RT space (coefficients eps), so the
exterior coupling <f, e> = G_S eps uses the SPD Gram; the rotated Gram
<f, n x g> (an RT/rotated-RT pairing with a large near-null space) never
enters the coupling. Y_S = R_S Y_D maps eps to the S restriction of J_D.

Interior identity (region medium m, outward n, J = n x H, M = E x n):
E_rad,m[J]_t - K_m[M]_t + (1/2) n x M = 0. With M = e x n, n x M = e, and
Galerkin RWG testing on dV: T_m j + K'_m eps + G eps / 2 = 0, where
T_m = i w mu0 (V_m - P_m / k_m^2), K'_m = <f, curl S_m[n x g]>, G = <f, g>.
With dk2 = kc^2 - kb^2 and the divided-difference kernel F = (G_c - G_b)/dk2
(bounded at R = 0):

    j_b = T_b^-1 (-K'_b - G/2) eps,   J_D = T_c^-1 [-dk2 K'hat eps - dT j_b],
    dT  = i w mu0 dk2 Vhat - (i/(w eps_c)) P_c + (i/(w eps_b)) P_b.

The scalar operators enter only tree rows through exact face charges, so no
loop row ever sees the O(1/w^2) background scalar potential; the loop rows of
the bracket are O(w) quantities formed from the bounded kernel F. At DC the
tree part of J_D carries the exact P0 normal-current divergence, so a linear
potential (straight bar with full contacts) reproduces R = l/(sigma A) up to
quadrature accuracy of the static operators (closed-form inner integrals).

Two regimes. The DSA loop rows lose internal inductance at low frequency
(near-cancellation once delta >> t), so below a crossover in t/delta each body
uses the uniform-current local conduction J = e / z on its S faces,

    z = P / (sigma A) - i w dL'_int P,   A_low = <e, e / z>,

with (A, P) the area and perimeter of the face's owner-box cross-section
normal to its longest dimension (the assumed current direction). For one z
per body W = B A_low^-1 B^T = z <g, g>: a uniform field E along a prismatic
bar gives K = E sigma A / P on the perimeter, I = sigma A E and R = l/(sigma A)
exactly. dL'_int is the per-length inductance of the uniform area current
minus that of the uniform perimeter current (the exterior MPIE already holds
the latter): dL'_int = mu0/(2 pi) (ln GMD_perimeter - ln GMD_area), closed form.
Inside the CROSSOVER window both interiors act on the same e space and their
e -> J maps are blended, A = (1 - s) A_low + s A_DSA, W = B A^-1 B^T, with s
the C1 smoothstep of log(t/delta); t is the largest smallest-box-dimension of
the body. Blending A (currents driven by one e) keeps the network between the
two regimes; blending W (one current through both impedances) overshoots R.
s = 1 folds exactly the DSA block (unchanged), s = 0 never evaluates the DSA.
"""

from dataclasses import dataclass
import math

import torch

from ...layers import relative_permittivity
from ...errors import UnsupportedFeatureError

MU0 = 4e-7 * math.pi
C0 = 299792458.0
EPS0 = 1 / (MU0 * C0 * C0)
CONTACT = 3


def admittance_geometry(rectangles, materials, vias, via_materials, geometry, stack):
    """Solid boxes, material groups (0 = PEC) and PEC block port bridges.

    Each horizontal delta-gap bridge becomes a PEC box spanning the terminal
    metal thickness, so both terminals touch it through contact faces.
    """
    from .network import _closed_geometry
    records = [*materials, *via_materials]
    if any(m[2] is not None for m in records):
        raise UnsupportedFeatureError("surface_admittance models smooth conductors; surface roughness is not supported",
                                      feature="guard:0e2b0056", method="mom.layered")
    boxes, lifted, shortened = _closed_geometry(rectangles, materials, vias, geometry, stack)
    sigmas = sorted({m[0] for m in records if m[0] is not None})
    groups = [0 if m[0] is None else 1 + sigmas.index(m[0]) for m in records]
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[:i]):
            if groups[i] and groups[j] and groups[i] != groups[j] and all(
                    max(a[2*d], b[2*d]) <= min(a[2*d+1], b[2*d+1]) for d in range(3)):
                raise UnsupportedFeatureError("surface_admittance bodies of different conductivity must not touch",
                                              feature="guard:3032b104", method="mom.layered")
    interfaces = stack.interfaces()
    media = [stack.bottom, *stack.layers, stack.top]
    # One homogeneous background for every body: all lossy boxes in one dielectric region.
    regions = {sum(1 for z in interfaces if z <= box[4]) for box, group in zip(boxes, groups) if group}
    if (any(group and any(box[4] < z < box[5] for z in interfaces) for box, group in zip(boxes, groups))
            or len(regions) > 1 or any(media[r] == "pec" for r in regions)):
        raise UnsupportedFeatureError("Every surface_admittance body must lie inside one common dielectric layer",
                                      feature="guard:74059a6a", method="mom.layered")
    blocks = []
    for record in lifted:
        if len(record) == 5:
            # Vertical PEC block port (network._vertical_block).
            blocks.append(record)
            continue
        bridge, top, axis, cut, centre, direction = record
        # Terminals share a mid-plane and (checked above) the top height, so one bottom.
        x, y = (bridge[0], centre) if axis == 0 else (centre, bridge[2])
        bottom = next(b[4] for b in boxes if b[5] == top and b[0] <= x <= b[1] and b[2] <= y <= b[3])
        blocks.append(((*bridge[:4], bottom, top), axis, cut, centre, direction))
    return boxes, groups, sigmas, blocks, shortened


def background_permittivity(stack, z0, z1, frequency):
    """Complex permittivity eps0*(eps_r(f) + i sigma/(w eps0)) of the medium containing [z0, z1].

    eps_r(f) is the constant or the dispersive layer model at ``frequency`` (layers.relative_permittivity).
    """
    interfaces = stack.interfaces()
    medium = [stack.bottom, *stack.layers, stack.top][sum(1 for z in interfaces if z <= z0)]
    return EPS0 * relative_permittivity(medium, frequency) + 1j * medium.sigma / (2 * math.pi * frequency)


@dataclass
class AdmittanceBody:
    """Frequency-independent topology of one lossy body (all tensors on device).

    basis: (faces[N,2], slots[N,2], coefficients[N,2]) RWG on dV (local face ids).
    surface: the RT space on S (S-interior RWGs and S-side seam halves), the
    restriction of exterior currents; trial: the e space (the same RT space).
    exterior: R_x [Nsurface, E] for exterior ids.
    exterior, transform and charges are stored sparse (COO of the exact dense
    entries) and densified per use, one body at a time.
    """
    owner: int
    group: int
    sigma: float
    z_range: tuple
    faces: torch.Tensor
    vertices: torch.Tensor
    local_faces: torch.Tensor
    basis: tuple
    surface: tuple
    trial: tuple
    exterior_ids: torch.Tensor
    exterior: torch.Tensor
    transform: torch.Tensor
    loops: int
    charges: torch.Tensor
    # Low regime: body thickness for t/delta and per-face (P/(sigma A), dL'_int P) [F, 2].
    thickness: float
    low_faces: torch.Tensor


# Regime window in t/delta (body thickness over skin depth); see the module docstring.
CROSSOVER = (0.3, 0.6)


def _segment_self(length):
    """Double integral of ln|x - x'| over one segment."""
    return length * length * (math.log(length) - 1.5)


def _segment_parallel(length, gap):
    """Double integral of ln distance between two aligned parallel segments."""
    l2 = length * length + gap * gap
    return (length * (length * math.log(l2) - 2 * length + 2 * gap * math.atan(length / gap))
            - .5 * (l2 * math.log(l2) - length * length - gap * gap * math.log(gap * gap)))


def _segment_corner(a, b):
    """Double integral of ln distance between perpendicular segments sharing an end point."""
    return .5 * (b * (a * math.log(a * a + b * b) - 2 * a + 2 * b * math.atan(a / b)) - 2 * b * a
                 + a * a * math.atan(b / a) + b * (a - b * math.atan(a / b)))


def internal_inductance(a, b, device):
    """Per-length inductance [H/m] of a uniform a x b area current minus a uniform perimeter current.

    mu0/(2 pi) (ln GMD_perimeter - ln GMD_area). The perimeter term is closed form;
    the area term (4 int_0^a int_0^b (a-u)(b-v) ln r) is a Duffy split of the
    corner singularity with the radial integral exact and a smooth 40-point rule.
    """
    from ..quadrature import gauss_rule
    perimeter = 2 * (a + b)
    ln_perimeter = (2 * _segment_self(a) + 2 * _segment_self(b) + 2 * _segment_parallel(a, b)
                    + 2 * _segment_parallel(b, a) + 8 * _segment_corner(a, b)) / perimeter ** 2
    tau, w = gauss_rule(40, device=device)   # nodes and weights on [0, 1]
    ln_area = 0.
    for p, q in ((a, b), (b, a)):
        log_r = .5 * torch.log(p * p + q * q * tau * tau)
        ln_area += 4 * float((w * ((-5 / 36 + 7 * tau / 144) + (1 / 6 - tau / 12) * log_r)).sum())
    return MU0 / (2 * math.pi) * (ln_perimeter - ln_area)


def box_section(box):
    """(width, height) of the cross-section normal to the longest box dimension."""
    dims = sorted((box[1] - box[0], box[3] - box[2], box[5] - box[4]))
    return dims[1], dims[0]


def regime_weight(body, frequency):
    """DSA weight s in [0, 1]: C1 smoothstep of log(t/delta) over CROSSOVER."""
    ratio = body.thickness * math.sqrt(math.pi * frequency * MU0 * body.sigma)
    low, high = CROSSOVER
    u = min(max(math.log(ratio / low) / math.log(high / low), 0.), 1.)
    return u * u * (3 - 2 * u)


def _components(faces, face_edges):
    """Connected face components through shared edges (deterministic labels)."""
    label = torch.arange(len(faces), device=faces.device)
    edges = face_edges.reshape(-1)
    owner = torch.arange(len(faces), device=faces.device).repeat_interleave(3)
    while True:
        smallest = torch.full((int(edges.max()) + 1,), len(faces), device=faces.device, dtype=torch.int64)
        smallest = smallest.scatter_reduce(0, edges, label[owner], "amin")
        updated = torch.minimum(label, smallest[face_edges].amin(-1))
        if torch.equal(updated, label):
            return label
        label = updated


def _opposite(faces, edges):
    """Local slot of the vertex of each face not on the given edge."""
    on = (faces[..., None] == edges[..., None, :]).any(-1)
    return (~on).to(torch.int64).argmax(-1)


def prepare_bodies(layout, boxes, groups, sigmas, *, length_scale):
    from .build import build_extension
    build_extension()
    op = torch.ops.witwin_maxwell_mom_layered
    mesh, basis = layout.mesh, layout.basis
    device = mesh.vertices.device
    topology = mesh.topology
    group_of = torch.tensor(groups + [0] * (int(layout.face_owners.max()) + 1 - len(groups)),
                            device=device, dtype=torch.int64)
    face_group = group_of[layout.face_owners.clamp_min(0)] * (layout.face_owners >= 0)
    lossy = (face_group > 0).nonzero().flatten()
    if not len(lossy):
        return []
    label = _components(lossy, topology.face_edges[lossy])
    areas = mesh.areas
    # Exterior junction-RWG face keys: (topology edge, face) for both supports.
    edge_keys = topology.edges[:, 0] * len(mesh.vertices) + topology.edges[:, 1]
    edge_order = torch.argsort(edge_keys)
    ext = basis.edges.sort(-1).values
    ext_edge = edge_order[torch.searchsorted(edge_keys[edge_order], ext[:, 0] * len(mesh.vertices) + ext[:, 1])]
    bodies = []
    for value in torch.unique(label).tolist():
        faces = lossy[label == value]
        local = torch.full((len(mesh.faces),), -1, device=device, dtype=torch.int64)
        local[faces] = torch.arange(len(faces), device=device)
        fe = topology.face_edges[faces]
        counts = torch.bincount(fe.reshape(-1), minlength=len(topology.edges))
        used = (counts > 0).nonzero().flatten()
        if bool((counts[used] != 2).any()):
            raise UnsupportedFeatureError("surface_admittance bodies need manifold boundaries (no edge-touching boxes)",
                                          feature="guard:b5a0c4ad", method="mom.layered")
        # Two incident body faces per edge, ascending global face id.
        entries = torch.argsort(fe.reshape(-1) * len(faces) + torch.arange(len(faces), device=device).repeat_interleave(3))
        incident = (entries // 3).reshape(-1, 2)
        edge_ids = fe.reshape(-1)[entries].reshape(-1, 2)[:, 0]
        plus, minus = faces[incident[:, 0]], faces[incident[:, 1]]
        endpoints = topology.edges[edge_ids]
        lengths = torch.linalg.vector_norm(mesh.vertices[endpoints[:, 0]] - mesh.vertices[endpoints[:, 1]], dim=-1)
        ps, ms = _opposite(mesh.faces[plus], endpoints), _opposite(mesh.faces[minus], endpoints)
        coefficient = torch.stack((lengths / (2 * areas[plus]), -lengths / (2 * areas[minus])), -1)
        vbasis = (torch.stack((local[plus], local[minus]), -1), torch.stack((ps, ms), -1), coefficient)
        contact = layout.face_kinds[faces] == CONTACT
        cp, cm = contact[incident[:, 0]], contact[incident[:, 1]]
        interior = ~cp & ~cm
        seam = cp ^ cm
        n = len(edge_ids)
        # mu space: S-interior RWGs (same functions), then S-side seam halves.
        sint = interior.nonzero().flatten()
        sseam = seam.nonzero().flatten()
        s_is_plus = ~cp[sseam]
        sface = torch.where(s_is_plus, plus[sseam], minus[sseam])
        sslot = torch.where(s_is_plus, ps[sseam], ms[sseam])
        half = lengths[sseam] / (2 * areas[sface])
        surface = (torch.cat((vbasis[0][sint], torch.stack((local[sface], torch.full_like(sface, -1)), -1))),
                   torch.cat((vbasis[1][sint], torch.stack((sslot, torch.zeros_like(sslot)), -1))),
                   torch.cat((coefficient[sint], torch.stack((half, torch.zeros_like(half)), -1))))
        nmu = len(sint) + len(sseam)
        # e is expanded in the same RT space on S (Gram coupling, see module docstring).
        trial = surface
        # Exterior map: S-face outflow of every exterior function at mu edges.
        mu_edge = torch.cat((edge_ids[sint], edge_ids[sseam]))
        mu_face = torch.cat((plus[sint], sface))
        key = mu_edge * len(mesh.faces) + mu_face
        rows, cols, signs = [], [], []
        for side, sign in ((basis.plus_faces, 1.), (basis.minus_faces, -1.)):
            ext_key = ext_edge * len(mesh.faces) + side
            order = torch.argsort(key)
            position = torch.searchsorted(key[order], ext_key).clamp_max(nmu - 1)
            hit = key[order][position] == ext_key
            rows.append(order[position][hit])
            cols.append(hit.nonzero().flatten())
            signs.append(torch.full((int(hit.sum()),), sign, device=device, dtype=torch.float64))
        rows, cols, signs = torch.cat(rows), torch.cat(cols), torch.cat(signs)
        exterior_ids = torch.unique(cols)
        exterior = torch.zeros((nmu, len(exterior_ids)), device=device, dtype=torch.float64)
        exterior[rows, torch.searchsorted(exterior_ids, cols)] += signs
        # Loop/tree of dV with the shared deterministic forest/cycle kernels.
        flat = incident.reshape(-1)
        by_face = torch.argsort(flat, stable=True)
        incidence = torch.arange(n, device=device).repeat_interleave(2)[by_face].reshape(-1, 3).contiguous()
        isigns = torch.tensor([1, -1], device=device).repeat(n)[by_face].reshape(-1, 3).contiguous()
        order, parents, counts_ = op.forest(local[plus], local[minus], incidence, isigns)
        components, loops = counts_.tolist()
        transform = op.cycles(local[plus], local[minus], lengths.contiguous(), order, parents, loops)
        transform = transform / torch.linalg.vector_norm(transform, dim=0)
        charges = torch.zeros((len(faces), n - loops), device=device, dtype=torch.float64)
        tree_rows = transform[:, loops:].abs().argmax(0)
        tvalue = transform[tree_rows, torch.arange(loops, n, device=device)]
        tcols = torch.arange(n - loops, device=device)
        # Face charge D q: integrated outflow +l on plus, -l on minus.
        charges[local[plus[tree_rows]], tcols] = lengths[tree_rows] * tvalue
        charges[local[minus[tree_rows]], tcols] = -lengths[tree_rows] * tvalue
        owner_box = int(layout.face_owners[faces[0]])
        bz = boxes[owner_box]
        sigma = sigmas[groups[owner_box] - 1]
        used, inverse = torch.unique(mesh.faces[faces], return_inverse=True)
        # Low regime: owner-box cross-section of every face (assumed current along the longest dimension).
        # Contact faces (never in S) take an S owner so PEC boxes never set the thickness.
        owners = layout.face_owners[faces]
        owners = torch.where(contact, owners[~contact][0], owners)
        members = torch.unique(owners).tolist()
        sections = {b: box_section(boxes[b]) for b in members}
        rows = {b: (2 * (w + h) / (sigma * w * h), 2 * (w + h) * internal_inductance(w, h, device))
                for b, (w, h) in sections.items()}
        table = torch.tensor([rows[b] for b in members], device=device, dtype=torch.float64)
        per_face = table[torch.searchsorted(torch.tensor(members, device=device), owners)]
        vertices = mesh.vertices[used]
        bodies.append(AdmittanceBody(owner_box, groups[owner_box], sigma,
                                     (bz[4], bz[5]), faces, vertices, inverse,
                                     vbasis, surface, trial, exterior_ids, exterior.to_sparse(),
                                     transform.to_sparse(), loops, charges.to_sparse(),
                                     max(h for _, h in sections.values()), per_face))
    return bodies


RULES = (6, 4, 8, 8, 12, 3)   # touching order, radial panels, near outer, angular, radial, far base order
GAUSS_ORDERS = 24
# Pairs per far launch (one warp each) and near/touching pairs (blocks) per near launch: bounded
# launch durations (<= ~0.7 s on F-B 80/20) on time-sliced shared devices. Outputs do not depend
# on them (one owner per pair).
LAUNCH = (1 << 15, 1 << 10)


def _gauss_table(device):
    from ..quadrature import gauss_rule
    return torch.cat([torch.stack(gauss_rule(n, device=device), -1).reshape(-1)
                      for n in range(1, GAUSS_ORDERS + 1)]).contiguous()


def interior_operators(body, kc, kb, rules=RULES, launch=LAUNCH):
    """Native Galerkin interior blocks of one body: V_c, V_b, Vhat, K_b, Khat, P_c, P_b."""
    from .build import build_extension
    build_extension()
    vertices = (body.vertices - body.vertices.mean(0)).contiguous()
    face_bytes = 47 * 16 * len(body.local_faces)
    tile = max(1, min(len(body.local_faces), (1 << 30) // face_bytes))
    names = ("vc", "vb", "vhat", "kb", "khat", "pc", "pb")
    values = torch.ops.witwin_maxwell_mom_layered.admittance_operators(
        vertices, body.local_faces.contiguous(), _gauss_table(vertices.device),
        *(t.contiguous() for t in body.basis), *(t.contiguous() for t in body.trial),
        kc.real, kc.imag, kb.real, kb.imag, [*rules, GAUSS_ORDERS, *launch], tile)
    return dict(zip(names, values))


def _local(basis, count):
    faces, slots, coefficient = basis
    entries = (faces >= 0).nonzero()
    face = faces[entries[:, 0], entries[:, 1]]
    return face, entries[:, 0], slots[entries[:, 0], entries[:, 1]], coefficient[entries[:, 0], entries[:, 1]]


def local_grams(body):
    """Grams G=<f, e> (dV RWG x e space) and B=<g, e> (surface RT x e space); exact degree-4 rule."""
    from ..quadrature import triangle_rule
    t = body.vertices[body.local_faces]
    points, w = triangle_rule(t, 4)
    f = points[:, :, None] - t[:, None]
    local = torch.einsum("tqic,tqjc,tq->tij", f, f, w)
    out = []
    for test in (body.basis, body.surface):
        fa, ma, sa, ca = _local(test, len(t))
        fb, nb, sb, cb = _local(body.trial, len(t))
        i, j = (fa[:, None] == fb[None]).nonzero().unbind(-1)
        dense = torch.zeros((len(test[0]), len(body.trial[0])), device=t.device, dtype=torch.float64)
        dense.index_put_((ma[i], nb[j]), ca[i] * cb[j] * local[fa[i], sa[i], sb[j]], accumulate=True)
        out.append(dense)
    return out


def surface_gram(vertices, local_faces, surface, weight):
    """Face-weighted Gram <g, w g> on the S space for a (complex) per-face weight [F]; exact degree-4 rule."""
    from ..quadrature import triangle_rule
    t = vertices[local_faces]
    points, w = triangle_rule(t, 4)
    f = points[:, :, None] - t[:, None]
    local = torch.einsum("tqic,tqjc,tq->tij", f, f, w)
    fa, ma, sa, ca = _local(surface, len(t))
    i, j = (fa[:, None] == fa[None]).nonzero().unbind(-1)
    dense = torch.zeros((len(surface[0]), len(surface[0])), device=t.device, dtype=weight.dtype)
    dense.index_put_((ma[i], ma[j]), weight[fa[i]] * ca[i] * ca[j] * local[fa[i], sa[i], sa[j]],
                     accumulate=True)
    return dense


def low_admittance(body, frequency):
    """A_low = <e, e / z>: the e space driving the local conduction current e / z."""
    z = torch.complex(body.low_faces[:, 0], (-2 * math.pi * frequency) * body.low_faces[:, 1])
    return surface_gram(body.vertices, body.local_faces, body.trial, 1 / z)


def body_impedance(body, frequency, eps_b, operators=None, grams=None):
    """W_S = B A^-1 B^T [Nsurface, Nsurface] and the power form A for one body and frequency.

    All loop/tree algebra uses the real cycle transform Q (loops first). Scalar
    operators enter only the tree-tree block, through exact face charges.
    Large operands are released as soon as they are consumed.
    """
    omega = 2 * math.pi * frequency
    eps_c = eps_b + 1j * body.sigma / omega
    kc = omega * (MU0 * eps_c) ** .5
    kb = omega * (MU0 * eps_b) ** .5
    dk2 = omega * omega * MU0 * (eps_c - eps_b)
    ops = interior_operators(body, kc, kb) if operators is None else dict(operators)
    gram, coupling = local_grams(body) if grams is None else grams
    q, loops = body.transform.to_dense(), body.loops
    t = body.vertices[body.local_faces]
    area = torch.linalg.vector_norm(torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), dim=-1) / 2
    density = body.charges.to_dense() / area[:, None]

    def congruence(name, factor):
        # factor * Q^T v Q of operator ``name``, released before the products (it is popped here).
        v = ops.pop(name)
        real, imag = v.real.contiguous(), v.imag.contiguous()
        del v
        real = q.mT @ real @ q
        imag = q.mT @ imag @ q
        out = torch.complex(real, imag)
        del real, imag
        out *= factor
        return out

    def left(v):
        return torch.complex(q.mT @ v.real, q.mT @ v.imag)

    def tree(p):
        return torch.complex(density.mT @ p.real @ density, density.mT @ p.imag @ density)

    pc, pb = tree(ops.pop("pc")), tree(ops.pop("pb"))
    tb = congruence("vb", 1j * omega * MU0)
    tb[loops:, loops:] -= (1j / (omega * eps_b)) * pb
    # K' = <f, curl S[g x n]> = -<f, curl S[n x g]> (native K uses n x g).
    kterm = ops.pop("kb")
    kterm.neg_()
    kterm -= gram / 2
    rhs = left(kterm)
    del kterm
    jb = _solve(tb, rhs)
    del tb, rhs
    dt = congruence("vhat", 1j * omega * MU0 * dk2)
    dt[loops:, loops:] += (1j / (omega * eps_b)) * pb - (1j / (omega * eps_c)) * pc
    bracket = left(ops.pop("khat"))
    bracket *= -dk2
    bracket -= dt @ jb
    del dt, jb
    tc = congruence("vc", 1j * omega * MU0)
    tc[loops:, loops:] -= (1j / (omega * eps_c)) * pc
    solution = _solve(tc, bracket)
    del tc, bracket
    jd = torch.complex(q @ solution.real, q @ solution.imag)
    # Weak DSA relation tested with the e space: B^T x = A eps, A = <e, J_D(e)>.
    admittance = gram.mT.to(torch.complex128) @ jd
    b = coupling.to(torch.complex128)
    return b @ torch.linalg.solve(admittance, b.mT), admittance


def _solve(a, b):
    """Diagonally equilibrated dense solve; ``a`` is equilibrated in place (callers release it)."""
    scale = a.diagonal().abs().rsqrt()
    torch.mul(scale[:, None], a, out=a)
    torch.mul(a, scale[None], out=a)
    return scale[:, None] * torch.linalg.solve(a, scale[:, None] * b)


class SurfaceAdmittance:
    """Per-body DSA blocks folded into dense layered systems.

    Blocks stay addressable by body owner (first box id of the body): the
    exterior ids, R_x and the per-frequency W_S of every body are separate.
    """

    def __init__(self, bodies, stack, frequencies):
        self.bodies, self.stack, self.frequencies = bodies, stack, frequencies

    @classmethod
    def prepare(cls, layout, boxes, groups, sigmas, stack, frequencies):
        bodies = prepare_bodies(layout, boxes, groups, sigmas, length_scale=None)
        return cls(bodies, stack, frequencies)

    def with_frequencies(self, frequencies):
        return SurfaceAdmittance(self.bodies, self.stack, frequencies)

    def metadata(self):
        return dict(conductor_impedance="differential_surface_admittance",
                    interior_field="lossy_helmholtz_per_body",
                    low_frequency_interior="uniform_current_local_impedance",
                    regime_crossover_t_over_delta=CROSSOVER, regime_blend="c1_smoothstep_log_t_over_delta",
                    admittance_bodies=tuple(dict(owner=b.owner, sigma=b.sigma, faces=len(b.faces),
                                                 rwg=len(b.basis[0]), trial=len(b.trial[0]), loops=b.loops,
                                                 exterior=len(b.exterior_ids), thickness=b.thickness)
                                            for b in self.bodies))

    def body_block(self, index, frequency):
        """(exterior ids, W_ext) of body ``index`` for one frequency (exterior RWG units).

        s = 1 is exactly the DSA block; s = 0 never evaluates the DSA operators.
        The Grams and the dense interior work set exist only during the call.
        """
        body = self.bodies[index]
        s = regime_weight(body, frequency)
        if s > 0:
            eps_b = background_permittivity(self.stack, *body.z_range, frequency)
            # The Grams are formed inside, after the interior operators (not alongside their work set).
            ws, admittance = body_impedance(body, frequency, eps_b)
        if s < 1:
            # Same e space for both interiors: blend the e -> J maps, W = B A^-1 B^T.
            low = low_admittance(body, frequency)
            admittance = low if s == 0 else (1 - s) * low + s * admittance
            b = local_grams(body)[1].to(torch.complex128)
            ws = b @ torch.linalg.solve(admittance, b.mT)
        del admittance
        rx = body.exterior.to_dense().to(torch.complex128)
        return body.exterior_ids, rx.mT @ ws @ rx

    def body_blocks(self, frequency):
        """{owner: (exterior ids, W_ext)} of every body for one frequency."""
        return {body.owner: self.body_block(index, frequency) for index, body in enumerate(self.bodies)}

    def block_bytes(self):
        """Bytes of the folded blocks of one frequency."""
        return sum(16 * len(body.exterior_ids) ** 2 for body in self.bodies)

    def blocks(self, frequencies):
        """Per-frequency body blocks, computed before any dense chunk is allocated."""
        return [self.body_blocks(frequency) for frequency in frequencies]

    def fold(self, system, order, blocks):
        for k, frequency in enumerate(system.frequencies.tolist()):
            scale = 1j / (MU0 * 2 * math.pi * frequency)
            for ids, block in blocks[k].values():
                index = order[ids]
                system.matrix[k][index[:, None], index[None, :]] += block * scale

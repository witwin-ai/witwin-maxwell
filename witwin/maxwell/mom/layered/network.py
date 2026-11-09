"""Delta-gap ports, mixed-precision dense solve and N-port networks for layered MPIE.

Sign chain (exp(-i*omega*t), NetworkData convention). A port is an ideal voltage
source in the gap of a bridge strip from ``negative`` to ``positive``. Its
impressed field E_inc = V * delta(cut) * d points from the negative to the
positive terminal (d = unit bridge direction), so the terminal voltage is
V = phi(+) - phi(-) = int_-^+ E_inc . dl. Inside the source the current also
flows from - to +, i.e. it leaves the + terminal into the structure, which is
NetworkData's port current. With s_n = sign(RWG direction . d) on cut edges:

    b_m = <f_m, E_inc> = V * l_m * s_m,      I = sum_n s_n * l_n * x_n.

b and I share the same vector, so Y = I/V inherits the symmetry of Z. A
passive inductive loop then has Im(Z11) = -omega*L < 0.

The same sign vector applies to a z-directed bridge. A PEC-attached strip has
one-face RWGs at its grounded edge: the image cancels their boundary line charge,
so no unbalanced boundary delta enters the scalar potential. The loop/tree
forest completes these half functions through an uncharged PEC reservoir.
"""

from dataclasses import dataclass
import math

import torch

from ...network import NetworkData
from .lowfreq import MIN_FREQUENCY, LoopTree, build_loop_tree, scaled_system
from .green import LayeredGreenTable
from .mesh import LayeredMesh, mesh_conductors, validate_vias
from .loss import SheetLoss, prepare_sheet_loss, sheet_material_payload


# Scaled and standalone solves use the same normwise backward-error gate.
_BACKWARD = 64 * 2.0**-53
_ITERATIONS = 10


def _port_geometry(port, rectangles, bridge_width):
    """Return (bridge rectangle, plane z, bridge axis, cut coordinate, direction sign)."""
    # LumpedPort already guarantees distinct terminals on an axis-aligned voltage_path.
    if port.voltage_path is None or port.voltage_path.axis == "z":
        raise ValueError(f"Port {port.name!r} needs a horizontal voltage_path between point terminals")
    axis = "xy".index(port.voltage_path.axis)
    positive, negative = (tuple(float(v) for v in point) for point in (port.positive, port.negative))
    z = positive[2]

    def on_metal(point):
        return any(r[4] == z and r[0] <= point[0] <= r[1] and r[2] <= point[1] <= r[3] for r in rectangles)

    if not (on_metal(positive) and on_metal(negative)):
        raise ValueError(f"Port {port.name!r} terminals must lie on metal of their plane")
    low, high = sorted((positive[axis], negative[axis]))
    centre = positive[1 - axis]
    cut = 0.5 * (low + high)
    midpoint = (cut, centre) if axis == 0 else (centre, cut)
    if any(r[4] == z and r[0] < midpoint[0] < r[1] and r[2] < midpoint[1] < r[3] for r in rectangles):
        raise ValueError(f"Port {port.name!r} must bridge a gap: its midpoint lies on metal")
    half = 0.5 * bridge_width
    bridge = ((low, high, centre - half, centre + half, z) if axis == 0 else
              (centre - half, centre + half, low, high, z))
    direction = 1.0 if positive[axis] > negative[axis] else -1.0
    return bridge, z, axis, cut, centre, direction


def _vertical_port_geometry(port, rectangles, stack, width):
    """A vertical strip may bend onto a sheet boundary or terminate at PEC.

    Interior sheet attachments would create three-face edges; reject them until
    a junction space is requested. The strip's transverse axis is chosen from
    the common sheet boundary, with x-normal preferred.
    """
    positive, negative = (tuple(float(v) for v in p) for p in (port.positive, port.negative))
    x, y = positive[:2]
    za, zb = sorted((positive[2], negative[2]))
    planes = sorted({r[4] for r in rectangles})
    grounds = ([stack.interfaces()[0]] if stack.bottom == "pec" else []) + (
        [stack.interfaces()[-1]] if stack.top == "pec" else [])
    for point in (positive, negative):
        if point[2] not in grounds and not any(
                r[4] == point[2] and r[0] <= x <= r[1] and r[2] <= y <= r[3] for r in rectangles):
            raise ValueError("Vertical port terminals need metal or PEC ground; horizontal ports need one metal plane")
    if any(za < z < zb for z in planes):
        raise NotImplementedError("Vertical ports must join consecutive metal planes or a PEC ground")
    candidates = []
    for normal in (0, 1):
        centre = (x, y)[1-normal]
        lo, hi = centre-width/2, centre+width/2
        valid = True
        for z in (za, zb):
            if z in grounds:
                continue
            attached = [r for r in rectangles if r[4] == z and
                        r[2*normal] <= (x,y)[normal] <= r[2*normal+1] and
                        r[2*(1-normal)] < hi and lo < r[2*(1-normal)+1]]
            bounds = sorted({lo,hi,*(v for r in attached for v in r[2*(1-normal):2*(1-normal)+2] if lo<v<hi)})
            for a, c in zip(bounds[:-1],bounds[1:]):
                owners = [r for r in attached if r[2*(1-normal)] <= (a+c)/2 <= r[2*(1-normal)+1]]
                sides = {side for r in owners for side in (0,1) if (x,y)[normal] == r[2*normal+side]}
                if not owners or len(sides)!=1 or any(
                        r[2*normal] < (x,y)[normal] < r[2*normal+1] for r in owners):
                    valid = False
        if valid:
            candidates.append((normal,(x,y)[normal],lo,hi,za,zb,(za+zb)/2,centre,
                               1.0 if positive[2]>negative[2] else -1.0))
    if not candidates:
        raise NotImplementedError("Vertical bridge attachment requires a sheet boundary; interior attachments need junction RWGs")
    return candidates[0]


def _cut_vectors(layout, table, ports, bridge_width):
    """Signed edge-length vectors [N, P] of every port cut (device tensors)."""
    mesh, basis = layout.mesh, layout.basis
    vertices = mesh.vertices
    a, b = vertices[basis.edges[:, 0]], vertices[basis.edges[:, 1]]
    flow = mesh.centroids[basis.minus_faces] - mesh.centroids[basis.plus_faces]
    columns = []
    for geometry in ports:
        if len(geometry) == 9:
            normal, fixed, lo, hi, za, zb, cut, centre, direction = geometry
            lateral = 1-normal
            on_cut = ((a[:, 2] == cut) & (b[:, 2] == cut)
                      & (a[:, normal] == fixed) & (b[:, normal] == fixed)
                      & ((.5*(a[:, lateral]+b[:, lateral])-centre).abs() < .5*bridge_width))
            if not bool(on_cut.any()):
                raise RuntimeError("A vertical port cut carries no RWG edge")
            columns.append(torch.where(on_cut, torch.sign(flow[:, 2])*direction*basis.edge_lengths, 0.0))
            continue
        bridge, z, axis, cut, centre, direction = geometry
        lateral = 1 - axis
        half = 0.5 * bridge_width
        # Every edge on the cut segment must be driven: an undriven one shorts
        # the gap. Edge midpoints lie strictly inside the bridge, so no
        # endpoint round-off at the bridge sides can drop an edge.
        on_cut = ((a[:, axis] == cut) & (b[:, axis] == cut) & (a[:, 2] == z) & (b[:, 2] == z)
                  & ((0.5 * (a[:, lateral] + b[:, lateral]) - centre).abs() < half))
        if not bool(on_cut.any()):
            raise RuntimeError("A port cut carries no RWG edge")
        sign = torch.sign(flow[:, axis]) * direction
        columns.append(torch.where(on_cut, sign * basis.edge_lengths, 0.0))
    return torch.stack(columns, 1).to(torch.complex128)


def solve_dense(z, b):
    """Solve Z X = B: complex64 LU with complex128 refinement, escalating to complex128 LU.

    When cond(Z) * eps(float32) is not small the float32 refinement stalls
    and the complex128 factorization is used. Returns X and (factor dtype,
    refinement steps, max backward error).
    """
    scale = torch.linalg.matrix_norm(z)
    norm = torch.linalg.vector_norm(b, dim=0)
    for dtype in (torch.complex64, torch.complex128):
        factor = torch.linalg.lu_factor(z.to(dtype))
        x = torch.linalg.lu_solve(*factor, b.to(dtype)).to(torch.complex128)
        for step in range(_ITERATIONS + 1):
            residual = b - z @ x
            error = (torch.linalg.vector_norm(residual, dim=0) /
                     (scale * torch.linalg.vector_norm(x, dim=0) + norm)).max().item()
            if error <= _BACKWARD:
                return x, (str(dtype).removeprefix("torch."), step, error)
            if step < _ITERATIONS:
                x = x + torch.linalg.lu_solve(*factor, residual.to(dtype)).to(torch.complex128)
        del factor
    raise RuntimeError(f"Dense refinement did not reach backward error {_BACKWARD:.3g}: {error:.3g}")


@dataclass
class PreparedLayeredNetwork:
    """Meshed layout, Green table and port excitation, ready for per-frequency solves."""

    table: LayeredGreenTable
    layout: LayeredMesh
    excitation: torch.Tensor
    ports: tuple
    loss: SheetLoss | None
    loop_tree: LoopTree
    metadata: dict


def prepare_layered_network(stack, rectangles, ports, frequencies, *, bridge_width, max_edge,
                            edge_cell=None, growth_ratio=1.5, sheet_materials=None, vias=(), via_materials=None, device="cuda", prepared_geometry=None):
    """Mesh a layered layout and build its Green table and port excitation.

    ``rectangles`` are (x0,x1,y0,y1,z) sheets; ``vias`` are
    (x0,x1,y0,y1,z0,z1) four-wall tubes between consecutive metal planes.
    ``sheet_materials`` and ``via_materials`` parallel those records as
    (sigma, thickness, roughness), with sigma=None for PEC. Via thickness is
    unused: walls use the single-face bulk Leontovich impedance (1-i)/(sigma*delta).
    Horizontal ports bridge a sheet gap; vertical ports bend onto a sheet
    boundary or attach to the stack PEC ground. Bridges are ideal/lossless.
    All coordinates must exactly share the declared planes and breakpoints.
    prepared_geometry reuses fixed mesh/basis/loop-tree/port tensors for adaptive
    frequency probes of the identical compiled layout.
    """
    if bridge_width <= 0:
        raise ValueError("bridge_width must be positive")
    if not ports:
        raise ValueError("A layered network needs at least one LumpedPort")
    rectangles = [tuple(float(v) for v in r) for r in rectangles]
    vias = [tuple(float(v) for v in via) for via in vias]
    validate_vias(rectangles, vias)
    geometry = [_vertical_port_geometry(port, rectangles, stack, bridge_width)
                if port.voltage_path is not None and port.voltage_path.axis == "z" else
                _port_geometry(port, rectangles, bridge_width) for port in ports]
    bridges = [g for g in geometry if len(g) == 9]
    sheets = rectangles + [g[0] for g in geometry if len(g) == 6]
    planes = tuple(sorted({r[4] for r in sheets} | {v for g in bridges for v in g[4:6]}))
    xs = [v for r in sheets for v in r[:2]]
    ys = [v for r in sheets for v in r[2:4]]
    rho_max = 1.01 * math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    frequencies = tuple(float(f) for f in frequencies)
    if any(not math.isfinite(f) or f < MIN_FREQUENCY for f in frequencies):
        raise ValueError(f"Layered networks require frequencies >= {MIN_FREQUENCY:g} Hz; DC is a limit")
    # Images farther than the assembly near distance (near_ratio=2 times the
    # largest cell diagonal sqrt(2)*max_edge) are never near for any face pair;
    # they stay in the tabulated remainder. The table needs >= the thinnest layer.
    thinnest = min((layer.thickness for layer in stack.layers), default=0.0)
    extract = max(2 * math.sqrt(2) * max_edge, thinnest)
    spans = ((min(planes), max(planes)),) if vias or bridges else ()
    table = LayeredGreenTable.build(stack, planes, frequencies, rho_max, device=device,
                                    via_spans=spans, z_nodes=12 if bridges and (stack.bottom == "pec" or stack.top == "pec") else 8,
                                    extract_distance=extract)
    cuts = [([], []) for _ in planes]
    for g in geometry:
        if len(g) == 6:
            _, z, axis, cut, _, _ = g
            cuts[planes.index(z)][axis].append(cut)
    grounds = tuple(z for z, condition in ((stack.interfaces()[0], stack.bottom == "pec"),
                    (stack.interfaces()[-1], stack.top == "pec")) if condition and
                    any(z in g[4:6] for g in bridges))
    layout = (mesh_conductors(sheets, table.planes, max_edge, cut_lines=cuts, vias=vias, bridges=bridges,
                              h_edge=edge_cell, growth_ratio=growth_ratio, grounds=grounds,
                              interfaces=stack.interfaces()) if prepared_geometry is None else prepared_geometry.layout)
    materials = [(None, 0.0, None)] * len(rectangles) if sheet_materials is None else sheet_materials
    via_materials = [(None, 0.0, None)] * len(vias) if via_materials is None else via_materials
    loss = prepare_sheet_loss(layout, rectangles, materials, table.frequencies, vias=vias,
                              via_materials=via_materials)
    unknowns = layout.basis.edges.shape[0]
    # Admission depends on ONE frequency, never the sweep length. The runtime
    # chooses bounded chunks again after persistent loop/tree setup is resident.
    frequency_chunk_capacity(unknowns, len(layout.mesh.faces), len(frequencies),
                             available_work_memory(layout.mesh.device))
    metadata = {"solver": "layered_mpie_dense", "bridge_width": bridge_width, "max_edge": max_edge,
                "edge_cell": edge_cell, "unknowns": unknowns, "faces": layout.mesh.faces.shape[0],
                "rho_max": rho_max, "extract_distance": extract,
                "mesh_quality": layout.quality,
                "sheets": tuple(sheet_material_payload(m) for m in materials), "port_bridges": "ideal_lossless"}
    if vias or bridges:
        metadata.update(vias=tuple(vias), via_impedance="single_face_Leontovich_bulk",
                        vertical_ports=len(bridges), ground_half_rwg=bool(grounds))
    loop_tree = (build_loop_tree(layout, length_scale=max_edge) if prepared_geometry is None
                 else prepared_geometry.loop_tree)
    metadata.update(low_frequency_method="loop_tree_scaled", loop_unknowns=loop_tree.loops,
                    tree_unknowns=loop_tree.trees, components=loop_tree.components,
                    minimum_frequency_Hz=MIN_FREQUENCY, dc="positive_frequency_limit")
    excitation = (_cut_vectors(layout, table, geometry, bridge_width) if prepared_geometry is None
                  else prepared_geometry.excitation)
    return PreparedLayeredNetwork(table, layout, excitation, tuple(ports), loss, loop_tree, metadata)


def available_work_memory(device):
    """Allocator-reusable plus driver-free memory, respecting the process cap."""
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    process_free = int(total * torch.cuda.get_per_process_memory_fraction(device)) - allocated
    return min(free + reusable, process_free)


def frequency_chunk_capacity(unknowns, faces, count, available_bytes):
    """Reserve 20% for vendor workspace; refuse only if one frequency cannot fit."""
    per_frequency = 16 * (12 * unknowns * unknowns + faces * faces)
    capacity = int(.8 * available_bytes) // per_frequency
    if capacity < 1:
        raise RuntimeError(f"One dense layered frequency needs ~{per_frequency / 2**30:.2f} GiB "
                           f"plus workspace; {available_bytes / 2**30:.2f} GiB available")
    return min(count, capacity)


def _refined_factor_solve(matrix, rhs, dtype):
    """Fixed device refinement budget, followed by one chunk convergence transfer."""
    op = torch.ops.witwin_maxwell_mom_layered
    lu, pivots, factor_info = op.lu_factor(matrix.to(dtype))
    solution = op.lu_solve(lu, pivots, rhs.to(dtype)).to(torch.complex128).contiguous()
    steps = torch.full((matrix.shape[0],), -1, device=matrix.device, dtype=torch.int64)
    matrix_norm = op.frequency_norm(matrix, True)
    rhs_norm = op.frequency_norm(rhs, False)
    # All convergence decisions and first-passing-step latches stay on device.
    # Freeze each converged frequency so chunk composition cannot change X.
    for step in range(_ITERATIONS + 1):
        residual = op.frequency_residual(matrix, solution, rhs)
        error, active = op.refinement_check(
            op.frequency_norm(residual, False), matrix_norm,
            op.frequency_norm(solution, False), rhs_norm,
            factor_info, steps, step, _BACKWARD)
        if step < _ITERATIONS:
            correction = op.lu_solve(lu, pivots, residual.to(dtype)).to(torch.complex128).contiguous()
            op.refinement_update(solution, correction, active)
    status = torch.stack((steps.to(torch.float64), error), -1).tolist()
    return solution, status


def solve_scaled_chunk(matrix, rhs):
    """Chunk-batched complex64 LU + complex128 refinement, escalating failures only."""
    solution, status = _refined_factor_solve(matrix, rhs, torch.complex64)
    factor_dtype = ["complex64"] * len(status)
    failed = [i for i, (step, _) in enumerate(status) if step < 0]
    if failed:
        indices = torch.tensor(failed, device=matrix.device, dtype=torch.int64)
        corrected, final = _refined_factor_solve(
            matrix.index_select(0, indices), rhs.index_select(0, indices), torch.complex128)
        if any(step < 0 for step, _ in final):
            raise RuntimeError(f"Scaled dense refinement did not reach backward error {_BACKWARD:.3g}")
        solution.index_copy_(0, indices, corrected)
        for index, value in zip(failed, final):
            factor_dtype[index], status[index] = "complex128", value
    return solution, (tuple(factor_dtype), tuple(int(s[0]) for s in status), tuple(s[1] for s in status))


def recover_scaled_chunk(prepared, solution, rhs, recovery, *, frequency_start=0):
    """Native current/admittance recovery for one solved frequency chunk."""
    count = solution.shape[0]
    frequencies = prepared.table.frequencies[frequency_start:frequency_start + count]
    coefficients = torch.ops.witwin_maxwell_mom_layered.unscale(solution.contiguous(), recovery)
    currents = torch.ops.witwin_maxwell_mom_layered.frequency_mm(
        prepared.loop_tree.q.unsqueeze(0), coefficients)
    # Form Y in the scaled space, retaining the original charge-port accuracy.
    y = torch.ops.witwin_maxwell_mom_layered.frequency_mm(rhs.transpose(-1, -2), solution)
    admittance = torch.ops.witwin_maxwell_mom_layered.port_admittance(y, frequencies)
    return admittance, currents


def solve_prepared_network(prepared, *, frequency_chunk_size=None):
    """Memory-sized frequency chunks; return NetworkData and [frequency,edge,port] currents.

    An explicit chunk size is a cap for reproducible workflow/benchmark control.
    Admission refuses only when a single-frequency work set does not fit.
    """
    total = len(prepared.table.frequencies)
    capacity = frequency_chunk_capacity(
        prepared.layout.basis.n_edges, len(prepared.layout.mesh.faces), total,
        available_work_memory(prepared.excitation.device))
    if frequency_chunk_size is not None:
        if type(frequency_chunk_size) is not int or frequency_chunk_size < 1:
            raise ValueError("frequency_chunk_size must be a positive integer")
        capacity = min(capacity, frequency_chunk_size)
    ports = prepared.excitation.shape[1]
    currents = torch.empty((total, prepared.layout.basis.n_edges, ports),
                           device=prepared.excitation.device, dtype=torch.complex128)
    admittance = torch.empty((total, ports, ports), device=currents.device, dtype=currents.dtype)
    factors, steps, errors = [], [], []
    for start in range(0, total, capacity):
        stop = min(start + capacity, total)
        matrix, rhs, recovery = scaled_system(prepared, frequency_start=start, frequency_stop=stop)
        solution, info = solve_scaled_chunk(matrix, rhs)
        y, x = recover_scaled_chunk(prepared, solution, rhs, recovery, frequency_start=start)
        admittance[start:stop], currents[start:stop] = y, x
        factors.extend(info[0])
        steps.extend(info[1])
        errors.extend(info[2])
        # Release dense chunk temporaries before the next chunk is assembled.
        del matrix, rhs, recovery, solution, y, x
    metadata = {**prepared.metadata, "factor_dtype": tuple(factors),
                "refinement_steps": tuple(steps), "backward_error": tuple(errors),
                "frequency_chunk_size": capacity}
    network = NetworkData.from_y(frequencies=prepared.table.frequencies, y=admittance,
                                 z0=[port.reference_impedance for port in prepared.ports],
                                 port_names=[port.name for port in prepared.ports], metadata=metadata)
    return network, currents


def solve_layered_network(stack, rectangles, ports, frequencies, **options):
    """Prepare and solve in one call; ``options`` are prepare_layered_network keywords."""
    return solve_prepared_network(prepare_layered_network(stack, rectangles, ports, frequencies, **options))[0]

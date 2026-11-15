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
from .operator import BACKWARD, ITERATIONS, dense_chunk_bytes, dense_frequency_bytes
from .green import LayeredGreenTable
from .vertical_refine import grading_metadata, mesh_interfaces
from .mesh import LayeredMesh, mesh_conductors, mesh_closed_conductors, validate_vias
from .loss import SheetLoss, prepare_sheet_loss, prepare_closed_loss, sheet_material_payload
from .port_system import horizontal_span, port_records, vertical_span
from ...errors import UnsupportedFeatureError



def _port_geometry(port, rectangles, bridge_width):
    """Return ((bridge rectangle, plane z, bridge axis, cut, centre, direction sign), half width).

    The bridge width is bridge_width, or the voltage_aperture width of an edge port (port_system.py).
    """
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
    lo, hi, centre, half = horizontal_span(port, rectangles, axis, z, bridge_width)
    bridge = ((low, high, lo, hi, z) if axis == 0 else (lo, hi, low, high, z))
    direction = 1.0 if positive[axis] > negative[axis] else -1.0
    return (bridge, z, axis, cut, centre, direction), half


def _vertical_port_geometry(port, rectangles, stack, width):
    """A vertical strip may bend onto a sheet boundary or terminate at PEC.

    Interior sheet attachments would create three-face edges; reject them until
    a junction space is requested. The strip's transverse axis is chosen from
    the common sheet boundary, with x-normal preferred, or set by the
    voltage_aperture of an edge port, which also sets the width.
    Returns (bridge record, half width).
    """
    span = vertical_span(port, rectangles)
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
        raise UnsupportedFeatureError("Vertical ports must join consecutive metal planes or a PEC ground",
                                      feature="guard:684ed00c", method="mom.layered")
    candidates = []
    for normal in ((0, 1) if span is None else span[:1]):
        centre = (x, y)[1-normal]
        lo, hi = (centre-width/2, centre+width/2) if span is None else span[1:]
        centre = centre if span is None else (lo+hi)/2
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
        raise UnsupportedFeatureError("Vertical bridge attachment requires a sheet boundary; interior attachments need junction RWGs",
                                      feature="guard:10f67271", method="mom.layered")
    return candidates[0], (candidates[0][3]-candidates[0][2])/2 if span is not None else width/2


def _cut_vectors(layout, table, ports, halves):
    """Signed edge-length vectors [N, P] of every port cut (device tensors); halves are bridge half widths."""
    mesh, basis = layout.mesh, layout.basis
    vertices = mesh.vertices
    a, b = vertices[basis.edges[:, 0]], vertices[basis.edges[:, 1]]
    flow = mesh.centroids[basis.minus_faces] - mesh.centroids[basis.plus_faces]
    columns = []
    for geometry, half in zip(ports, halves):
        if len(geometry) == 5:
            # PEC block bridge: every surface edge on the mid cross-section ring.
            # Horizontal (axis 0/1) or vertical (axis 2, R6-4) block; the ring lies inside the other two extents.
            block, axis, cut, centre, direction = geometry
            on_cut = (a[:, axis] == cut) & (b[:, axis] == cut)
            for other in (d for d in range(3) if d != axis):
                on_cut = (on_cut & (torch.minimum(a[:, other], b[:, other]) >= block[2*other])
                          & (torch.maximum(a[:, other], b[:, other]) <= block[2*other+1]))
            if not bool(on_cut.any()):
                raise RuntimeError("A block port cut carries no RWG edge")
            columns.append(torch.where(on_cut, torch.sign(flow[:, axis]) * direction * basis.edge_lengths, 0.0))
            continue
        if len(geometry) == 9:
            normal, fixed, lo, hi, za, zb, cut, centre, direction = geometry
            lateral = 1-normal
            on_cut = ((a[:, 2] == cut) & (b[:, 2] == cut)
                      & (a[:, normal] == fixed) & (b[:, normal] == fixed)
                      & ((.5*(a[:, lateral]+b[:, lateral])-centre).abs() < half))
            if not bool(on_cut.any()):
                raise RuntimeError("A vertical port cut carries no RWG edge")
            columns.append(torch.where(on_cut, torch.sign(flow[:, 2])*direction*basis.edge_lengths, 0.0))
            continue
        bridge, z, axis, cut, centre, direction = geometry
        lateral = 1 - axis
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
        for step in range(ITERATIONS + 1):
            residual = b - z @ x
            error = (torch.linalg.vector_norm(residual, dim=0) /
                     (scale * torch.linalg.vector_norm(x, dim=0) + norm)).max().item()
            if error <= BACKWARD:
                return x, (str(dtype).removeprefix("torch."), step, error)
            if step < ITERATIONS:
                x = x + torch.linalg.lu_solve(*factor, residual.to(dtype)).to(torch.complex128)
        del factor
    raise RuntimeError(f"Dense refinement did not reach backward error {BACKWARD:.3g}: {error:.3g}")


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
    # conductor_model="surface_admittance": per-body dense admittance blocks.
    admittance: object = None


def _closed_geometry(rectangles, materials, vias, geometry, stack):
    """Extrude the metal union with centered thickness and shorten each via.

    Analytically coincident dielectric/metal heights are canonicalized only
    within eight fp64 ULPs. Exact interface heights use the upper-region limit
    in the horizontal Green table; tangential electric fields are continuous.
    """
    if any(m[1] <= 0 for m in materials):
        raise UnsupportedFeatureError("Closed conductors require positive physical_thickness on every PlanarPEC",
                                      feature="geometry.planar_pec", method="mom.layered")
    for i, (a, material) in enumerate(zip(rectangles, materials)):
        for b, other in zip(rectangles[:i], materials[:i]):
            if (a[4] == b[4] and max(a[0], b[0]) < min(a[1], b[1])
                    and max(a[2], b[2]) < min(a[3], b[3]) and material != other):
                raise ValueError("Overlapping rectangles on one plane require identical materials and thicknesses")
    def height(z):
        return next((edge for edge in stack.interfaces()
                     if abs(edge-z) <= 8*max(math.ulp(edge), math.ulp(z))), z)
    boxes = [(*r[:4], height(r[4]-m[1]/2), height(r[4]+m[1]/2))
             for r,m in zip(rectangles, materials)]
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[:i]):
            if (rectangles[i][4] != rectangles[j][4] and max(a[0],b[0]) < min(a[1],b[1])
                    and max(a[2],b[2]) < min(a[3],b[3]) and max(a[4],b[4]) <= min(a[5],b[5])):
                raise UnsupportedFeatureError("Closed bodies on distinct metal planes must have a positive face gap",
                                              feature="guard:9324cec6", method="mom.layered")
    shortened = []
    for x0,x1,y0,y1,za,zb in vias:
        lower = {box[5] for r,box in zip(rectangles,boxes) if r[4]==za and
                 max(x0,r[0]) < min(x1,r[1]) and max(y0,r[2]) < min(y1,r[3])}
        upper = {box[4] for r,box in zip(rectangles,boxes) if r[4]==zb and
                 max(x0,r[0]) < min(x1,r[1]) and max(y0,r[2]) < min(y1,r[3])}
        if len(lower) != 1 or len(upper) != 1 or min(upper) <= max(lower):
            raise UnsupportedFeatureError("Closed via rims require uniform terminal thickness and a positive face gap",
                                          feature="guard:1a3cff5c", method="mom.layered")
        shortened.append((x0,x1,y0,y1,next(iter(lower)),next(iter(upper))))
    lifted = []
    for record in geometry:
        if len(record) == 9:
            lifted.append(_vertical_block(record, rectangles, boxes))
            continue
        bridge,z,axis,cut,centre,direction = record
        ends = ((bridge[0],centre),(bridge[1],centre)) if axis==0 else ((centre,bridge[2]),(centre,bridge[3]))
        heights = [{box[5] for r,box in zip(rectangles,boxes) if r[4]==z and
                    r[0]<=x<=r[1] and r[2]<=y<=r[3]} for x,y in ends]
        if any(len(h)!=1 for h in heights) or heights[0] != heights[1]:
            raise UnsupportedFeatureError("Closed bridge terminals require identical top-face heights",
                                          feature="guard:9c9442f8", method="mom.layered")
        top = next(iter(heights[0]))
        lifted.append(((*bridge[:4],top),top,axis,cut,centre,direction))
    return boxes + shortened, lifted, shortened


def _covered(rectangles, x0, x1, y0, y1):
    """True when the union of (x0,x1,y0,y1,...) rectangles covers the box footprint."""
    xs = sorted({x0, x1, *(v for r in rectangles for v in r[:2] if x0 < v < x1)})
    ys = sorted({y0, y1, *(v for r in rectangles for v in r[2:4] if y0 < v < y1)})
    return all(any(r[0] <= (a+b)/2 <= r[1] and r[2] <= (c+d)/2 <= r[3] for r in rectangles)
               for a, b in zip(xs, xs[1:]) for c, d in zip(ys, ys[1:]))


def _vertical_block(record, rectangles, boxes):
    """PEC block record (block, 2, cut, centre, direction) of a vertical port on closed bodies (R6-4).

    A vertical thick-terminal port: an ideal PEC block from the lower terminal's
    top face to the upper terminal's bottom face, spanning the aperture (bridge)
    width along the edge and the thinner terminal thickness into the metal from
    the shared edge, so its outer wall is coplanar with both terminal end faces.
    Its top and bottom weld to the bodies; the delta gap is its mid-height ring.
    Both terminals must be bodies whose union covers the footprint with one face
    height; a stack PEC ground has no body (grounded closed RWGs do not exist).
    """
    normal, fixed, lo, hi, za, zb, _, centre, direction = record
    lateral = 1 - normal
    pairs = list(zip(rectangles, boxes))
    edge = [[(r, b) for r, b in pairs if r[4] == z and r[2*lateral] < hi and lo < r[2*lateral+1]
             and fixed in (r[2*normal], r[2*normal+1])] for z in (za, zb)]
    sides = {1 if r[2*normal] == fixed else -1 for terminal in edge for r, _ in terminal}
    block = None
    if all(edge) and len(sides) == 1:
        depth = min(b[5] - b[4] for terminal in edge for _, b in terminal)
        side = next(iter(sides))
        span = sorted((fixed, fixed + side * depth))
        footprint = (*span, lo, hi) if normal == 0 else (lo, hi, *span)
        faces = []
        for z, face in ((za, 5), (zb, 4)):
            plane = [(r, b) for r, b in pairs if r[4] == z]
            inside = {b[face] for r, b in plane if max(r[0], footprint[0]) < min(r[1], footprint[1])
                      and max(r[2], footprint[2]) < min(r[3], footprint[3])}
            faces.append(inside if _covered([r for r, _ in plane], *footprint) else set())
        if all(len(f) == 1 for f in faces):
            block = (*footprint, next(iter(faces[0])), next(iter(faces[1])))
    if block is None:
        raise UnsupportedFeatureError("Closed vertical ports need terminal bodies on both metal planes covering the PEC "
                                      "block footprint at one shared edge; a stack PEC ground terminal needs grounded "
                                      "closed RWGs", feature="guard:5cec8f9a", method="mom.layered")
    return block, 2, 0.5 * (block[4] + block[5]), centre, direction


def _block_boxes(geometry):
    """Mesh boxes of the PEC port blocks; a vertical block is split at its cut so the ring is a mesh line."""
    boxes = []
    for record in geometry:
        if len(record) != 5:
            continue
        block, axis, cut = record[:3]
        boxes.extend([block] if axis < 2 else [(*block[:5], cut), (*block[:4], cut, block[5])])
    return boxes


def _thick_terminal_blocks(boxes, lifted, edge):
    """Replace each lifted top-face bridge flagged in ``edge`` by its PEC block record (R4-3).

    A closed horizontal edge port is a thick-terminal port: an ideal PEC block fills
    the gap over the aperture width and the full terminal thickness, the union mesher
    welds it to both terminal end faces, and the delta gap is its mid cross-section
    ring (the surface_admittance block port). A top-face delta-gap bridge carries the
    port current on one sheet instead of the full cross-section: +1.1% L on the
    10x20 um strip loop (G3); the edge port is its de-embedded form.

    ``lifted`` are _closed_geometry bridge records (bridge, top, axis, cut, centre,
    direction); both terminals share the top height (checked there) and the plane,
    hence one bottom. Returns (geometry, block boxes).
    """
    geometry, blocks = [], []
    for record, is_edge in zip(lifted, edge):
        if len(record) == 5:
            # Vertical PEC block (_vertical_block).
            geometry.append(record)
            blocks.append(record[0])
            continue
        if not is_edge:
            geometry.append(record)
            continue
        bridge, top, axis, cut, centre, direction = record
        x, y = (bridge[0], centre) if axis == 0 else (centre, bridge[2])
        bottom = next(b[4] for b in boxes if b[5] == top and b[0] <= x <= b[1] and b[2] <= y <= b[3])
        block = (*bridge[:4], bottom, top)
        blocks.append(block)
        geometry.append((block, axis, cut, centre, direction))
    return geometry, blocks


def prepare_layered_network(stack, rectangles, ports, frequencies, *, bridge_width, max_edge,
                            edge_cell=None, growth_ratio=1.5, sheet_materials=None, vias=(), via_materials=None, device="cuda", prepared_geometry=None,
                            conductor_model="sheet", conductor_ids=None, operator_backend="dense",
                            rectangle_names=None):
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
    conductor_model="closed" selects centered solid extrusion, single-face SIBC
    and top-face horizontal bridges; the default "sheet" retains its DC limit.
    conductor_ids groups compiled rectangle pieces by declared PlanarPEC for
    face-count metadata. Overlapping union cells belong to the first declaration.
    operator_backend="dense" admits one dense frequency here; "matrix_free"
    stores only O(N) near-field data (matrix_free.py) and admits its sparse
    factorization when the solve plans it.
    rectangle_names (declaring Structure name per rectangle) labels the
    conductors of metadata["ports"] (port_system.port_records).
    Several frequencies that need a graded vertical table return a
    frequency_groups.StreamedLayeredNetwork instead (R4-2).
    """
    # R4-2 hook: the call arguments rebuild single-frequency calls when streamed.
    arguments = dict(locals())
    stream = len(frequencies) > 1 and prepared_geometry is None
    from .planar import is_planar_record, prepare_planar_network
    from .frequency_groups import StreamedLayeredNetwork
    if any(map(is_planar_record, (*rectangles, *vias))):
        # B1 hook: polygon sheets and polygonal/multi-plane vias use the planar mesher.
        prepared = prepare_planar_network(
            stack, rectangles, ports, frequencies, bridge_width=bridge_width, max_edge=max_edge,
            edge_cell=edge_cell, growth_ratio=growth_ratio, sheet_materials=sheet_materials, vias=vias,
            via_materials=via_materials, device=device, prepared_geometry=prepared_geometry,
            conductor_model=conductor_model, conductor_ids=conductor_ids, operator_backend=operator_backend,
            rectangle_names=rectangle_names, stream_graded=stream)
        return StreamedLayeredNetwork(arguments) if prepared is None else prepared
    if bridge_width <= 0:
        raise ValueError("bridge_width must be positive")
    if not ports:
        raise ValueError("A layered network needs at least one LumpedPort")
    rectangles = [tuple(float(v) for v in r) for r in rectangles]
    vias = [tuple(float(v) for v in via) for via in vias]
    validate_vias(rectangles, vias)
    geometry, halves = zip(*(_vertical_port_geometry(port, rectangles, stack, bridge_width)
                             if port.voltage_path is not None and port.voltage_path.axis == "z" else
                             _port_geometry(port, rectangles, bridge_width) for port in ports))
    geometry = list(geometry)
    records = port_records(ports, geometry, halves, rectangles,
                           [None] * len(rectangles) if rectangle_names is None else rectangle_names, stack)
    materials = [(None, 0.0, None)] * len(rectangles) if sheet_materials is None else sheet_materials
    via_materials = [(None, 0.0, None)] * len(vias) if via_materials is None else via_materials
    port_blocks = []
    if conductor_model == "closed":
        boxes, geometry, shortened_vias = _closed_geometry(rectangles, materials, vias, geometry, stack)
        # Edge ports: thick-terminal PEC blocks; delta-gap ports keep top-face bridges.
        edge = [port.voltage_aperture is not None or len(g) == 5 for port, g in zip(ports, geometry)]
        geometry, port_blocks = _thick_terminal_blocks(boxes, geometry, edge)
        records = tuple({**record, "bridge": "ideal_pec_block"} if e else record for record, e in zip(records, edge))
    if conductor_model == "surface_admittance":
        records = tuple({**record, "bridge": "ideal_pec_block"} for record in records)
        from .admittance import admittance_geometry
        boxes, groups, sigmas, geometry, shortened_vias = admittance_geometry(
            rectangles, materials, vias, via_materials, geometry, stack)
        bodies = len(boxes)
        boxes = boxes + _block_boxes(geometry)
        groups = groups + [0] * (len(boxes) - bodies)
    if conductor_model in ("closed", "surface_admittance"):
        # Vertical PEC blocks drive their face mid-height, not the metal plane midpoint.
        records = tuple({**record, "cut_m": g[2]} if len(g) == 5 and g[1] == 2 else record
                        for record, g in zip(records, geometry))
    bridges = [g for g in geometry if len(g) == 9]
    sheets = rectangles + [g[0] for g in geometry if len(g) == 6]
    planes = tuple(sorted({r[4] for r in sheets} | {v for g in bridges for v in g[4:6]}))
    if conductor_model == "closed":
        planes = tuple(sorted({v for b in boxes for v in b[4:6]} | {g[1] for g in geometry if len(g) == 6}))
    if conductor_model == "surface_admittance":
        planes = tuple(sorted({v for b in boxes[:bodies] for v in b[4:6]}))
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
    spans = ((min(planes), max(planes)),) if vias or bridges or conductor_model != "sheet" else ()
    table = LayeredGreenTable.build(stack, planes, frequencies, rho_max, device=device,
                                    via_spans=spans, z_nodes=12 if bridges and (stack.bottom == "pec" or stack.top == "pec") else 8,
                                    extract_distance=extract, stream_graded=stream)
    if table is None:
        return StreamedLayeredNetwork(arguments)
    cuts = [([], []) for _ in planes]
    for g in geometry:
        if len(g) == 6:
            _, z, axis, cut, _, _ = g
            cuts[planes.index(z)][axis].append(cut)
        if len(g) == 5 and g[1] < 2:
            cuts[planes.index(g[0][5])][g[1]].append(g[2])
    grounds = tuple(z for z, condition in ((stack.interfaces()[0], stack.bottom == "pec"),
                    (stack.interfaces()[-1], stack.top == "pec")) if condition and
                    any(z in g[4:6] for g in bridges))
    if prepared_geometry is not None:
        layout = prepared_geometry.layout
    elif conductor_model == "closed":
        layout = mesh_closed_conductors(
            boxes + _block_boxes(geometry), [g[0] for g in geometry if len(g) == 6], table.planes, max_edge, cuts=cuts,
            h_edge=edge_cell, growth_ratio=growth_ratio, interfaces=mesh_interfaces(stack, table))
    elif conductor_model == "surface_admittance":
        layout = mesh_closed_conductors(
            boxes, [], table.planes, max_edge, cuts=cuts, h_edge=edge_cell, growth_ratio=growth_ratio,
            interfaces=mesh_interfaces(stack, table), groups=groups)
    else:
        layout = mesh_conductors(
            sheets, table.planes, max_edge, cut_lines=cuts, vias=vias, bridges=bridges,
            h_edge=edge_cell, growth_ratio=growth_ratio, grounds=grounds, interfaces=mesh_interfaces(stack, table))
    admittance = None
    if conductor_model == "surface_admittance":
        from .admittance import SurfaceAdmittance
        loss = None
        admittance = (prepared_geometry.admittance.with_frequencies(table.frequencies)
                      if prepared_geometry is not None else
                      SurfaceAdmittance.prepare(layout, boxes, groups, sigmas, stack, table.frequencies))
    else:
        loss = (prepare_closed_loss(layout, [*materials, *via_materials, *[(None, 0.0, None)] * len(_block_boxes(geometry))],
                                    table.frequencies)
                if conductor_model == "closed" else
                prepare_sheet_loss(layout, rectangles, materials, table.frequencies, vias=vias,
                                   via_materials=via_materials))
    unknowns = layout.basis.edges.shape[0]
    # Admission depends on ONE frequency, never the sweep length. The runtime
    # chooses bounded chunks again after persistent loop/tree setup is resident.
    if operator_backend == "dense":
        frequency_chunk_capacity(unknowns, len(layout.mesh.faces), len(ports), len(frequencies),
                                 available_work_memory(layout.mesh.device), wall_nodes=_wall_nodes(table),
                                 extra_per_frequency=0 if admittance is None else admittance.block_bytes())
    metadata = {"solver": f"layered_mpie_{operator_backend}", "operator_backend": operator_backend,
                "bridge_width": bridge_width, "max_edge": max_edge,
                "edge_cell": edge_cell, "unknowns": unknowns, "faces": layout.mesh.faces.shape[0],
                "rho_max": rho_max, "extract_distance": extract,
                "mesh_quality": layout.quality,
                "sheets": tuple(sheet_material_payload(m) for m in materials), "port_bridges": "ideal_lossless", "conductor_model": conductor_model,
                "ports": records,
                # "sheet" never meshes thickness: it enters lossy metal only through Zs, PEC is infinitely thin.
                "thickness_modelled_geometrically": conductor_model != "sheet"}
    metadata.update(grading_metadata(table))
    if conductor_model == "surface_admittance":
        metadata.update(admittance.metadata(), port_attachment="pec_block_bridge_contact_faces",
                        port_bridges="ideal_pec_block", thickness_convention="centered",
                        via_face_gaps=tuple(shortened_vias), port_blocks=tuple(g[0] for g in geometry),
                        port_voltage="block mid cross-section ring delta gap")
    if conductor_model == "closed":
        owners, kinds = layout.face_owners, layout.face_kinds
        ids = list(range(len(rectangles))) if conductor_ids is None else conductor_ids
        identifiers = sorted(set(ids))
        index = {identifier: i for i, identifier in enumerate(identifiers)}
        # Exact integer metadata reduction; transfer all face counts once.
        body_map = torch.tensor([index[i] for i in ids] +
                                [len(identifiers)+i for i in range(len(vias))],
                                dtype=torch.int64, device=owners.device)
        # Port blocks own the faces after the rectangle and via records; they are no conductor.
        physical = (owners >= 0) & (owners < len(body_map))
        numbers = torch.bincount(body_map[owners[physical]]*3+kinds[physical],
                                 minlength=3*(len(identifiers)+len(vias))).reshape(-1, 3).tolist()
        counts = [dict(conductor=identifier, bottom=numbers[i][0], top=numbers[i][1],
                       sidewall=numbers[i][2]) for i, identifier in enumerate(identifiers)]
        metadata.update(conductor_faces=tuple(counts), port_attachment="top_face",
                        conductor_impedance="single_face_Leontovich",
                        interior_field="zero", thickness_convention="centered",
                        dielectric_interface_limit="upper_region_tangential_continuity",
                        junction_edges=layout.mesh.topology.junction_edges,
                        via_face_gaps=tuple(shortened_vias),
                        via_faces=tuple(numbers[len(identifiers):]))
        if port_blocks:
            metadata.update(port_blocks=tuple(port_blocks), port_voltage="block mid cross-section ring delta gap",
                            port_attachment=("pec_block_end_faces" if len(port_blocks) == len(ports)
                                             else "top_face_and_pec_block_end_faces"))
    if vias or bridges:
        metadata.update(vias=tuple(vias), via_impedance="single_face_Leontovich_bulk",
                        vertical_ports=len(bridges), ground_half_rwg=bool(grounds))
    loop_tree = (build_loop_tree(layout, length_scale=max_edge, dense_setup=operator_backend == "dense")
                 if prepared_geometry is None
                 else prepared_geometry.loop_tree)
    metadata.update(low_frequency_method="loop_tree_scaled", loop_unknowns=loop_tree.loops,
                    tree_unknowns=loop_tree.trees, components=loop_tree.components,
                    minimum_frequency_Hz=MIN_FREQUENCY, dc="positive_frequency_limit")
    excitation = (_cut_vectors(layout, table, geometry, halves) if prepared_geometry is None
                  else prepared_geometry.excitation)
    return PreparedLayeredNetwork(table, layout, excitation, tuple(ports), loss, loop_tree, metadata, admittance)


def available_work_memory(device):
    """Allocator-reusable plus driver-free memory, respecting the process cap."""
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    process_free = int(total * torch.cuda.get_per_process_memory_fraction(device)) - allocated
    return min(free + reusable, process_free)


def _wall_nodes(table):
    """Vertical wall z nodes: they size the cached wall quadrature weights."""
    return 0 if table.vertical is None or table.exact_mode else table.vertical.z_nodes


def frequency_chunk_capacity(unknowns, faces, ports, count, available_bytes, *, wall_nodes=0,
                             extra_per_frequency=0):
    """Reserve 20% for vendor workspace; refuse only if one frequency cannot fit.

    One frequency needs operator.dense_frequency_bytes (16 N^2 complex128
    system plus the 8 max(N^2, F(F+1)) complex64 P/LU workspace, O(F) tile and
    O(N P) vectors); each chunk adds the O(F) assembly scratch once.
    """
    # extra_per_frequency: surface-admittance blocks held through the chunk.
    per_frequency = dense_frequency_bytes(unknowns, faces, ports) + extra_per_frequency
    usable = int(.8 * available_bytes) - dense_chunk_bytes(faces, wall_nodes)
    capacity = max(usable, 0) // per_frequency
    if capacity < 1:
        needed = per_frequency + dense_chunk_bytes(faces, wall_nodes)
        raise RuntimeError(f"One dense layered frequency needs ~{needed / 2**30:.2f} GiB "
                           f"plus workspace; {available_bytes / 2**30:.2f} GiB available")
    return min(count, capacity)


def solve_scaled_chunk(system):
    """In-place complex64 LU + complex128 refinement, escalating failures only."""
    system.factor()
    return system.solve()


def recover_scaled_chunk(prepared, solution, rhs, recovery, *, frequency_start=0):
    """Native current/admittance recovery for one solved frequency chunk."""
    op = torch.ops.witwin_maxwell_mom_layered
    count = solution.shape[0]
    frequencies = prepared.table.frequencies[frequency_start:frequency_start + count]
    coefficients = op.unscale(solution.contiguous(), recovery)
    # Edge currents Q c through the sparse rows of Q: O(nnz P), no dense Q.
    currents = op.sparse_product(coefficients, prepared.loop_tree.sparse_q_rows, True)
    # Form Y in the scaled space, retaining the original charge-port accuracy.
    y = op.frequency_mm(rhs.transpose(-1, -2), solution)
    admittance = op.port_admittance(y, frequencies)
    return admittance, currents


def solve_prepared_network(prepared, *, frequency_chunk_size=None):
    """Memory-sized frequency chunks; return NetworkData and [frequency,edge,port] currents.

    An explicit chunk size is a cap for reproducible workflow/benchmark control.
    Admission refuses only when a single-frequency work set does not fit.
    A StreamedLayeredNetwork returns grouped currents (frequency_groups.py).
    """
    from .frequency_groups import StreamedLayeredNetwork, solve_streamed_network
    if isinstance(prepared, StreamedLayeredNetwork):
        return solve_streamed_network(prepared, frequency_chunk_size=frequency_chunk_size)
    if prepared.metadata["operator_backend"] in ("matrix_free", "hmatrix"):
        from .matrix_free import solve_matrix_free
        from .hmatrix import solve_hmatrix
        solve = solve_hmatrix if prepared.metadata["operator_backend"] == "hmatrix" else solve_matrix_free
        admittance, currents, info = solve(prepared)
        frequencies = info["frequencies"]
        metadata = {**prepared.metadata, "near_field": info["structure"],
                    "block_gmres_steps": tuple(log["block_steps"] for log in frequencies),
                    "operator_actions": tuple(log["actions"] for log in frequencies),
                    "admittance_bound": tuple(log["admittance_bound"] for log in frequencies),
                    "preconditioner_factor_nnz": tuple(log["factor_nnz"] for log in frequencies),
                    "preconditioner_factor_bytes_estimate": frequencies[0]["factor_peak_bytes_estimate"],
                    "frequency_seconds": tuple(log["seconds"] for log in frequencies),
                    "near_setup_seconds": tuple(log["setup_seconds"] for log in frequencies),
                    "action_seconds": tuple(log["action_seconds"] for log in frequencies)}
        if prepared.admittance is not None:
            # Compressed surface-admittance body blocks (body_compress.py), per frequency.
            metadata["body_compression"] = tuple(log["body_compression"] for log in frequencies)
        if prepared.metadata["operator_backend"] == "hmatrix":
            metadata.update(compression_bound_worst_case=tuple(log["compression_bound_worst_case"] for log in frequencies),
                            compression_estimate_quadrature=tuple(log["compression_estimate_quadrature"]
                                                                  for log in frequencies),
                            compression_exact_check=tuple(log["compression_exact_check"] for log in frequencies),
                            hmatrix=tuple({k: v for k, v in log.items() if k.startswith(("vector_", "scalar_", "hmatrix", "leaf_", "sample"))}
                                          for log in frequencies))
        network = NetworkData.from_y(frequencies=prepared.table.frequencies, y=admittance,
                                     z0=[port.reference_impedance for port in prepared.ports],
                                     port_names=[port.name for port in prepared.ports], metadata=metadata)
        return network, currents
    total = len(prepared.table.frequencies)
    ports = prepared.excitation.shape[1]
    currents = torch.empty((total, prepared.layout.basis.n_edges, ports),
                           device=prepared.excitation.device, dtype=torch.complex128)
    admittance = torch.empty((total, ports, ports), device=currents.device, dtype=currents.dtype)
    capacity = frequency_chunk_capacity(
        prepared.layout.basis.n_edges, len(prepared.layout.mesh.faces), ports, total,
        available_work_memory(prepared.excitation.device), wall_nodes=_wall_nodes(prepared.table),
        extra_per_frequency=0 if prepared.admittance is None else prepared.admittance.block_bytes())
    if frequency_chunk_size is not None:
        if type(frequency_chunk_size) is not int or frequency_chunk_size < 1:
            raise ValueError("frequency_chunk_size must be a positive integer")
        capacity = min(capacity, frequency_chunk_size)
    factors, steps, errors = [], [], []
    for start in range(0, total, capacity):
        stop = min(start + capacity, total)
        system = scaled_system(prepared, frequency_start=start, frequency_stop=stop)
        solution, info = solve_scaled_chunk(system)
        y, x = recover_scaled_chunk(prepared, solution, system.rhs, system.recovery, frequency_start=start)
        admittance[start:stop], currents[start:stop] = y, x
        factors.extend(info[0])
        steps.extend(info[1])
        errors.extend(info[2])
        # Release dense chunk storage before the next chunk is assembled.
        del system, solution, y, x
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

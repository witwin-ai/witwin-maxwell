"""Public Scene -> Simulation -> Result runtime for layered sheets, vias and ports."""

import copy
import math
from dataclasses import fields
import time

import torch

from ...layers import PERMITTIVITY_MODELS, LayerStack
from ...media import LossyMetalMedium
from ...ports import LumpedPort
from .network import prepare_layered_network, solve_prepared_network
from .loss import sheet_material_payload
from .port_system import aperture_record, map_aperture, structure_names
from ...errors import UnsupportedFeatureError
from .admittance import CROSSOVER, MU0


CURRENT_KIND = "layered_rwg_port_currents_v1"


def _manhattan_rectangles(geometry, coordinate_maps=None):
    """Split a z-normal Manhattan PlanarPEC (holes allowed) into covering rectangles.

    Cells of the polygon's own vertex-coordinate grid are kept when their centre
    is inside under the even-odd rule over every loop. Host metadata only.
    compile_layered_scene routes non-Manhattan polygons to the planar path first.
    """
    if geometry.normal_axis != "z":
        raise UnsupportedFeatureError("Layered MoM supports PlanarPEC sheets with normal_axis='z' only",
                                      feature="geometry.planar_pec", method="mom.layered")
    loops = [loop.detach().cpu().tolist() for loop in (geometry.vertices, *geometry.holes)]
    if coordinate_maps is not None:
        loops = [[[coordinate_maps[a][p[a]] for a in (0, 1)] for p in loop] for loop in loops]
    xs = sorted({p[0] for loop in loops for p in loop})
    ys = sorted({p[1] for loop in loops for p in loop})

    def inside(x, y):
        crossings = 0
        for loop in loops:
            for a, b in zip(loop, loop[1:] + loop[:1]):
                if a[0] == b[0] and a[0] > x and min(a[1], b[1]) < y < max(a[1], b[1]):
                    crossings += 1
        return crossings % 2 == 1

    z = geometry.coordinate
    return [(x0, x1, y0, y1, z) for x0, x1 in zip(xs, xs[1:]) for y0, y1 in zip(ys, ys[1:])
            if inside((x0 + x1) / 2, (y0 + y1) / 2)]


def _snap_via(bounds, position, size, rectangles):
    """Snap Box bounds to declared sheet coordinates within the Box's own rounding scale.

    position +- size/2 of a float32 Box misses a plane such as 300.5e-6 by ~1e-11 m,
    while the mesher needs exact plane identity. A bound within 1e-6*(|position|+size)
    of a sheet coordinate on its axis (sub-nanometre at RFIC scale, several float32
    roundings) is replaced by that coordinate; other bounds stay and validate_vias
    rejects them.
    """
    candidates = ({v for r in rectangles for v in r[:2]}, {v for r in rectangles for v in r[2:4]},
                  {r[4] for r in rectangles})
    snapped = []
    for i, value in enumerate(bounds):
        axis = i // 2
        tolerance = 1e-6 * (abs(position[axis]) + size[axis])
        near = [c for c in candidates[axis] if abs(c - value) <= tolerance]
        snapped.append(near[0] if len(near) == 1 else value)
    return tuple(snapped)


def _coordinate_maps(scene, tolerance):
    from .mesh import clustered_coordinates
    from witwin.core import Box
    coordinates = [[], []]
    for structure in scene.structures:
        if not getattr(structure, "enabled", True):
            continue
        geometry = structure.geometry
        if getattr(geometry, "kind", None) == "planar_pec":
            for loop in (geometry.vertices, *geometry.holes):
                for p in loop.detach().cpu().tolist():
                    for a in (0, 1):
                        coordinates[a].append(p[a])
        elif isinstance(geometry, Box):
            position, size = (v.detach().to(torch.float64).cpu().tolist()
                              for v in (geometry.position, geometry.size))
            for a in (0, 1):
                coordinates[a].extend((position[a]-size[a]/2, position[a]+size[a]/2))
    for port in scene.ports:
        if type(port) is LumpedPort and port.positive is not None:
            for p in (port.positive, port.negative):
                for a in (0, 1):
                    coordinates[a].append(float(p[a]))
            if port.voltage_aperture is not None:
                bounds = aperture_record(port)
                for a in (0, 1):
                    coordinates[a].extend(bounds[2*a:2*a+2])
    return [clustered_coordinates(c, tolerance) for c in coordinates]


# Scene collections the layered runtime never reads, with the supported alternative.
_UNREAD_COLLECTIONS = (
    ("sources", "the network is driven by its LumpedPorts only"),
    ("monitors", "results are Result.network and mw.mom.surface_currents(result)"),
    ("thin_wires", "conductors are PlanarPEC sheets and Box vias"),
    ("networks", "connect networks to Result.network with NetworkData algebra"),
    ("lumped_elements", "load ports with Result.network.terminate(port, impedance=load)"),
    ("circuits", "connect circuits to Result.network with NetworkData algebra"),
    ("material_regions", "dielectrics are LayerStack layers"),
    ("electrostatic_terminals", "use Simulation.electrostatic(...) or Simulation.capacitance(...)"),
    ("charge_densities", "use Simulation.electrostatic(...)"),
    ("internal_absorbers", "the layered Green function is laterally unbounded"),
)

# LumpedPort fields the layered delta-gap port does not apply.
_UNREAD_PORT_FIELDS = (
    ("termination", "Result.network is the open N-port; close it with Result.network.terminate(port, "
                    "impedance=load) (exp(-i omega t): an inductor load is -1j*omega*L)"),
    ("reference_plane", "the port reference is the delta gap; de-embed Result.network instead"),
    ("current_surface", "the port current is the delta-gap MoM current; omit current_surface"),
    ("wire_binding", "layered ports need positive/negative point terminals on PlanarPEC metal"),
)


def _kind(value):
    name = getattr(value, "name", None)
    return type(value).__name__ if name is None else f"{type(value).__name__} {name!r}"


def unsupported_layered_inputs(scene):
    """Name every declared Scene input the layered network solve would not apply."""
    items = []
    for field, alternative in _UNREAD_COLLECTIONS:
        values = tuple(getattr(scene, field))
        if values:
            items.append(f"Scene.{field} = [{', '.join(_kind(v) for v in values)}]: {alternative}")
    if any(axis is not None for axis in scene.symmetry):
        items.append(f"Scene.symmetry = {scene.symmetry}: the layered solve has no symmetry reduction")
    kinds = sorted({kind for pair in scene.boundary.face_kinds for kind in pair} - {"none"})
    if kinds:
        items.append(f"Scene.boundary face kinds {kinds}: the layered Green function is laterally unbounded; "
                     "use BoundarySpec.none()")
    if scene.coordinates is not None:
        items.append(f"Scene.coordinates = {_kind(scene.coordinates)}: layered layouts are Cartesian")
    for index, port in enumerate(scene.ports):
        if type(port) is not LumpedPort:
            items.append(f"Scene.ports[{index}] = {_kind(port)}: layered ports are LumpedPort delta gaps")
            continue
        for field, alternative in _UNREAD_PORT_FIELDS:
            value = getattr(port, field)
            if value is not None:
                items.append(f"LumpedPort {port.name!r}.{field} = {type(value).__name__}: {alternative}")
    return items


def refuse_unsupported_layered_inputs(scene):
    """Raise one error that lists every unsupported layered MoM input."""
    items = unsupported_layered_inputs(scene)
    if items:
        raise UnsupportedFeatureError(
            "Layered MoM does not apply these declared inputs; remove them or use the stated alternative:\n"
            + "\n".join(f"- {item}" for item in items),
            feature="guard:65ab7282", method="mom.layered")


def _needs_planar(scene, coordinate_maps):
    """B1/B2: non-Manhattan sheets, Prism/ExtrudedPolygon vias or vias skipping a metal plane need the planar mesher."""
    from witwin.core import Box, ExtrudedPolygon, Prism
    from .planar import is_manhattan, sheet_record
    structures = [s for s in scene.structures if getattr(s, "enabled", True)]
    sheets = [sheet_record(s.geometry, coordinate_maps) for s in structures
              if getattr(s.geometry, "kind", None) == "planar_pec" and s.geometry.normal_axis == "z"]
    if any(not is_manhattan([list(loop) for loop in sheet.loops]) for sheet in sheets):
        return True
    planes = sorted({sheet.z for sheet in sheets})
    for structure in structures:
        geometry = structure.geometry
        if isinstance(geometry, (Prism, ExtrudedPolygon)):
            return True
        if isinstance(geometry, Box):
            z, height = (float(v[2]) for v in (geometry.position.detach().to(torch.float64),
                                               geometry.size.detach().to(torch.float64)))
            margin = 1e-6 * (abs(z) + height)
            if any(z - height / 2 + margin < plane < z + height / 2 - margin for plane in planes):
                return True
    return False


def compile_layered_scene(scene, *, breakpoint_tolerance_m=0.0, conductor_ids=None, sources=None, planar=False):
    """Compile PEC/lossy PlanarPEC sheets, Box vias and LumpedPorts to material records.

    A LumpedPort voltage_aperture makes a full-width edge port (port_system.py).

    Every declared input the layered solve would not apply is refused first, in
    one error. Records hold detached float values. ``sources`` (GR1 hook) is an
    optional pair of lists receiving the declaring Structure of each rectangle
    and via record. Manhattan layouts compile to rectangles and Box bounds.
    Non-Manhattan sheets, z-axis Prism or ExtrudedPolygon vias and vias that pass a metal plane
    (or ``planar=True``) compile to one PolygonSheet per PlanarPEC and one
    PolygonVia per via for the planar mesher (B1).
    """
    from .adjoint import fixed_record
    if not isinstance(scene.background, LayerStack):
        raise TypeError("Layered MoM requires Scene(background=LayerStack(...))")
    refuse_unsupported_layered_inputs(scene)
    coordinate_maps = _coordinate_maps(scene, breakpoint_tolerance_m) if breakpoint_tolerance_m else None
    planar = planar or _needs_planar(scene, coordinate_maps)
    rectangles, materials, sheets, vias, via_materials = [], [], [], [], []
    for structure in scene.structures:
        if not getattr(structure, "enabled", True):
            continue
        from witwin.core import Box, ExtrudedPolygon, Prism
        if planar and isinstance(structure.geometry, (Box, Prism, ExtrudedPolygon)):
            material = structure.material
            if not (isinstance(material, LossyMetalMedium) or material.is_pec):
                raise ValueError("Layered via requires Material.pec() or LossyMetalMedium")
            if sources is not None:
                sources[1].append(structure)
            vias.append(structure.geometry)
            via_materials.append(fixed_record(material, 0.0))
            continue
        if isinstance(structure.geometry, Box):
            geometry, material = structure.geometry, structure.material
            rotation = geometry.rotation.detach().cpu().tolist()
            if rotation not in ([1.,0.,0.,0.], [0.,0.,0.]):
                raise UnsupportedFeatureError("Layered vias require an axis-aligned Box",
                                              feature="guard:ee0fdce6", method="mom.layered")
            position, size = (v.detach().to(torch.float64).cpu().tolist()
                              for v in (geometry.position, geometry.size))
            bounds = tuple(v for p, length in zip(position, size) for v in (p-length/2, p+length/2))
            if not (isinstance(material, LossyMetalMedium) or material.is_pec):
                raise ValueError("Layered via requires Material.pec() or LossyMetalMedium")
            record = fixed_record(material, 0.0)
            if sources is not None:
                sources[1].append(structure)
            vias.append((bounds, position, size))
            via_materials.append(record)
            continue
        if getattr(structure.geometry, "kind", None) != "planar_pec":
            raise UnsupportedFeatureError("Layered MoM supports PlanarPEC sheets and Box vias only",
                                          feature="geometry.planar_pec", method="mom.layered")
        material = structure.material
        if not (isinstance(material, LossyMetalMedium) or material.is_pec):
            raise ValueError("Layered PlanarPEC requires Material.pec() or LossyMetalMedium")
        record = fixed_record(material, structure.geometry.physical_thickness)
        if isinstance(material, LossyMetalMedium) and record[1] <= 0:
            raise ValueError("LossyMetalMedium PlanarPEC requires physical_thickness > 0")
        if planar:
            from .planar import sheet_record
            pieces = [sheet_record(structure.geometry, coordinate_maps)]
        else:
            pieces = _manhattan_rectangles(structure.geometry, coordinate_maps)
        if sources is not None:
            sources[0].extend([structure] * len(pieces))
        if conductor_ids is not None:
            conductor_ids.extend([len(sheets)] * len(pieces))
        rectangles.extend(pieces)
        materials.extend([record] * len(pieces))
        sheets.append(sheet_material_payload(record))
    if not rectangles:
        raise ValueError("Layered MoM needs at least one PlanarPEC sheet")
    if planar:
        from .planar import via_record
        planes = sorted({sheet.z for sheet in rectangles})
        vias = [via_record(geometry, planes, coordinate_maps) for geometry in vias]
    elif coordinate_maps is not None:
        vias = [(tuple(coordinate_maps[i//2][v] if i < 4 else v for i,v in enumerate(bounds)),
                 position, size) for bounds,position,size in vias]
    if not planar:
        vias = [_snap_via(bounds, position, size, rectangles) for bounds, position, size in vias]
    if not scene.ports:
        raise ValueError("Layered MoM needs at least one LumpedPort")
    ports = tuple(scene.ports)
    if coordinate_maps is not None:
        ports = tuple(copy.copy(port) for port in ports)
        for port in ports:
            for name in ("positive", "negative"):
                point = getattr(port, name)
                object.__setattr__(port, name, (*(coordinate_maps[a][float(point[a])] for a in (0,1)), point[2]))
            if port.voltage_aperture is not None:
                object.__setattr__(port, "voltage_aperture", map_aperture(port, coordinate_maps))
    return scene.background, rectangles, ports, materials, tuple(sheets), vias, via_materials


def layered_scene_signature(compiled, device):
    """Deterministic plain-Python signature of compile_layered_scene output.

    Tensor values become Python numbers, so an in-place update of a Scene tensor
    changes the signature.
    """
    from .planar import is_planar_record
    stack, rectangles, ports, materials, _, vias, via_materials = compiled

    def medium(value):
        return value if isinstance(value, str) else (type(value).__name__, *(
            v if isinstance(v, str) else v.key() if isinstance(v, PERMITTIVITY_MODELS) else complex(v)
            for v in (getattr(value, f.name) for f in fields(value))))

    def record(value):
        sigma, thickness, roughness = value
        return (None if sigma is None else float(sigma), float(thickness), None if roughness is None else (
            type(roughness).__name__, tuple((name, float(v)) for name, v in sorted(vars(roughness).items()))))

    return (str(device), tuple(medium(m) for m in (stack.bottom, *stack.layers, stack.top)),
            tuple(rectangles), tuple(record(m) for m in materials),
            tuple(via if is_planar_record(via) else tuple(float(v) for v in via) for via in vias),
            tuple(record(m) for m in via_materials),
            tuple((port.name, tuple(map(float, port.positive)), tuple(map(float, port.negative)),
                   port.voltage_path.axis, complex(port.reference_impedance), aperture_record(port))
                  for port in ports))


class LayeredSurfaceData:
    """Port-driven RWG currents [frequency, edge, port] on the layered sheet mesh."""

    def __init__(self, currents, layout, frequencies, port_names):
        self.currents = currents
        self.layout = layout
        self.frequencies = frequencies
        self.port_names = port_names
        self.current_kind = CURRENT_KIND

    @property
    def mesh_quality(self):
        return self.layout.quality

    def snapshot(self):
        raise UnsupportedFeatureError("Layered MoM results do not support snapshot persistence yet: no payload schema "
                                  "binds the sheet mesh and RWG layout; save Result.network instead",
                                      feature="guard:0666abfe", method="mom.layered")



# conductor_model="auto" (default, R6-1 / plan A2).
# Rule (plan D-A1, A2). Let t be a lossy PlanarPEC's physical_thickness, sigma its
# conductivity and delta = sqrt(2 / (omega mu0 sigma)) its skin depth at the
# highest frequency of the request (AdaptiveSweep.f_max when a sweep is
# configured). When max t/delta reaches THICK_T_OVER_DELTA (the lower edge of the
# surface_admittance crossover window, admittance.CROSSOVER), the layout leaves
# the uniform-current regime of the sheet model and "surface_admittance" is
# selected; otherwise "sheet" (thin layers, PEC-only layouts and lossy metal far
# below one skin depth keep the sheet model, including its exact DC limit).
#
# The rule is physics only; there is no fallback. When it selects
# surface_admittance for a layout with a declaration surface_admittance refuses
# (polygon layout, zero-thickness PlanarPEC, surface roughness, vertical port,
# gradient run), auto raises and names the feature and t/delta: declare
# conductor_model="sheet" to accept the sheet approximation, or remove the
# feature. Geometry-level refusals of the closed extrusion (bodies across a
# dielectric interface, touching bodies of different conductivity, via rims and
# face gaps) raise from surface_admittance itself, as for an explicit request.
# The selection is recorded in network and Result metadata as
# ``conductor_model_selection``.
THICK_T_OVER_DELTA = CROSSOVER[0]


def max_t_over_delta(materials, frequency):
    """Largest t/delta over lossy sheet records (sigma, thickness, roughness); 0 if none."""
    omega = 2 * math.pi * frequency
    return max((t * math.sqrt(omega * MU0 * sigma / 2) for sigma, t, _ in materials
                if sigma is not None and t > 0), default=0.0)


def select_conductor_model(requested, compiled, frequencies, *, sweep, gradients, planar):
    """Resolved model name and selection record (None unless requested is "auto")."""
    if requested != "auto":
        return requested, None
    from .planar import is_planar_record
    _, rectangles, ports, materials, _, vias, via_materials = compiled
    f_max = sweep.f_max if sweep is not None else max(frequencies)
    ratio = max_t_over_delta(materials, f_max)
    blockers = [name for name, present in (
        ("polygon_layout", planar or any(map(is_planar_record, (*rectangles, *vias)))),
        ("zero_thickness_sheet", any(m[1] <= 0 for m in materials)),
        ("surface_roughness", any(m[2] is not None for m in (*materials, *via_materials))),
        ("vertical_port", any(p.voltage_path is not None and p.voltage_path.axis == "z" for p in ports)),
        ("gradients", gradients)) if present]
    model = "surface_admittance" if ratio >= THICK_T_OVER_DELTA else "sheet"
    if model == "surface_admittance" and blockers:
        raise UnsupportedFeatureError(
            f"conductor_model='auto' selects 'surface_admittance' (lossy metal t/delta = {ratio:.3g} >= "
            f"{THICK_T_OVER_DELTA:g} at {f_max:.6g} Hz), which does not support: {', '.join(blockers)}. "
            "Set conductor_model='sheet' explicitly to accept the sheet approximation, or remove the feature",
            feature="guard:5a2c81e7", method="mom.layered")
    return model, {"requested": "auto", "selected": model, "rule": "max t/delta at f_max >= threshold",
                   "threshold_t_over_delta": THICK_T_OVER_DELTA, "f_max_Hz": f_max, "max_t_over_delta": ratio}


class LayeredNetworkRuntime:
    """Prepared layered MPIE network solve: meshing, Green table and port excitation."""

    def __init__(self, simulation):
        options = simulation.config.layered
        self.conductor_ids = []
        sources = ([], [])
        stack, rectangles, ports, materials, sheets, vias, via_materials = compile_layered_scene(
            simulation.scene, breakpoint_tolerance_m=options.breakpoint_tolerance_m,
            conductor_ids=self.conductor_ids, sources=sources, planar=options.planar_mesher == "triangle")
        # GR1 hook: admitted trainable inputs, or None (the unchanged forward-only path).
        from .adjoint import layered_gradients
        self.gradients = layered_gradients(simulation, sources)
        # A2 hook: conductor_model="auto" resolves once per prepared Simulation.
        self.conductor_model, self.conductor_selection = select_conductor_model(
            options.conductor_model.value, (stack, rectangles, ports, materials, sheets, vias, via_materials),
            simulation.frequencies, sweep=options.sweep, gradients=self.gradients is not None,
            planar=options.planar_mesher == "triangle")
        names = structure_names(simulation.scene.structures)
        self.rectangle_names = [names[id(structure)] for structure in sources[0]]
        self.simulation = simulation
        self.compiled = (stack, rectangles, ports, materials, sheets, vias, via_materials)
        self.prepared = prepare_layered_network(
            stack, rectangles, ports, simulation.frequencies, bridge_width=options.bridge_width_m,
            max_edge=options.max_edge_length_m, edge_cell=options.edge_cell_m,
            growth_ratio=options.growth_ratio,
            sheet_materials=materials, vias=vias, via_materials=via_materials,
            conductor_model=self.conductor_model, conductor_ids=self.conductor_ids,
            operator_backend=options.operator_backend.value, device=simulation.config.device,
            rectangle_names=self.rectangle_names)
        self.prepared.metadata.update(sheets=sheets, breakpoint_tolerance_m=options.breakpoint_tolerance_m,
                                      hmatrix_exact_check=options.hmatrix_exact_check,
                                      **self._selection_metadata())
        self.prepared_config, self.prepared_frequencies = simulation.config, simulation.frequencies
        from .frequency_groups import StreamedLayeredNetwork
        # R4-2 hook: streamed mm-wave networks have one wall mesh per frequency group.
        self.streamed = isinstance(self.prepared, StreamedLayeredNetwork)
        if self.streamed and self.gradients is not None:
            raise UnsupportedFeatureError("Gradients need one prepared geometry for every frequency; these "
                                          "frequencies need graded mm-wave Green tables with per-frequency "
                                          "meshes (streamed); request the frequencies in separate calls",
                                          feature="guard:5e7a41c3", method="mom.layered")
        self.scene_signature = layered_scene_signature(self.compiled, simulation.config.device)

    def _selection_metadata(self):
        return {} if self.conductor_selection is None else {"conductor_model_selection": self.conductor_selection}

    def _solve_frequencies(self, frequencies):
        if self.streamed:
            # R6-3 hook: a streamed sweep sample is its own single-frequency prepare+solve.
            from .frequency_groups import resolve_streamed
            return resolve_streamed(self.prepared, frequencies)
        options = self.simulation.config.layered
        stack, rectangles, ports, materials, sheets, vias, via_materials = self.compiled
        prepared = prepare_layered_network(
            stack, rectangles, ports, frequencies, bridge_width=options.bridge_width_m,
            max_edge=options.max_edge_length_m, edge_cell=options.edge_cell_m,
            growth_ratio=options.growth_ratio,
            sheet_materials=materials, vias=vias, via_materials=via_materials,
            conductor_model=self.conductor_model, conductor_ids=self.conductor_ids,
            operator_backend=options.operator_backend.value, device=self.simulation.config.device,
            prepared_geometry=self.prepared, rectangle_names=self.rectangle_names)
        prepared.metadata.update(sheets=sheets, breakpoint_tolerance_m=options.breakpoint_tolerance_m,
                                 hmatrix_exact_check=options.hmatrix_exact_check, **self._selection_metadata())
        network, currents = solve_prepared_network(prepared)
        return network, currents

    def run(self):
        from ...result import Result
        sim = self.simulation
        if sim.config != self.prepared_config or sim.frequencies != self.prepared_frequencies:
            raise ValueError("Numerical configuration or frequencies changed after layered MoM preparation")
        compiled = compile_layered_scene(sim.scene, breakpoint_tolerance_m=sim.config.layered.breakpoint_tolerance_m,
                                         planar=sim.config.layered.planar_mesher == "triangle")
        if layered_scene_signature(compiled, sim.config.device) != self.scene_signature:
            raise ValueError("The Scene changed after layered MoM preparation (LayerStack, sheets, vias, materials, "
                             "ports or device); call Simulation.prepare() again")
        device = self.prepared.device if self.streamed else self.prepared.excitation.device
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        network, currents = solve_prepared_network(self.prepared)
        sweep_result = None
        if sim.config.layered.sweep is not None:
            from .sweep import adaptive_network
            from .frequency_groups import merge_streamed_sweep
            network, currents, sweep_result = adaptive_network(
                sim.config.layered.sweep, network, currents, self._solve_frequencies,
                merge=merge_streamed_sweep if self.streamed else None)
        if self.gradients is not None:
            # GR1 hook: same forward values; Y and z0 now carry autograd graphs.
            network = self.gradients.attach(self.prepared, network, currents)
        torch.cuda.synchronize(device)
        if self.streamed:
            from .frequency_groups import GroupedLayeredSurfaceData
            data = GroupedLayeredSurfaceData(currents, network.frequencies, network.port_names)
        else:
            data = LayeredSurfaceData(currents, self.prepared.layout, network.frequencies, network.port_names)
        data.sweep = sweep_result
        stats = {"solve_seconds": time.perf_counter() - started, **copy.deepcopy(dict(network.metadata))}
        return Result(method="mom", scene=sim.scene, frequencies=tuple(network.frequencies.tolist()), solver=self,
                      surface_data=data, network=network, solver_stats=stats,
                      metadata={**sim.metadata, "native_data": CURRENT_KIND,
                                "layered": sim.config.layered.payload(),
                                "sheets": network.metadata["sheets"], "port_bridges": "ideal_lossless",
                                "ports": network.metadata["ports"],
                                "conductor_model": network.metadata["conductor_model"],
                                **self._selection_metadata(),
                                "thickness_modelled_geometrically":
                                    network.metadata["thickness_modelled_geometrically"],
                                "conductor_faces": network.metadata.get("conductor_faces", ())})

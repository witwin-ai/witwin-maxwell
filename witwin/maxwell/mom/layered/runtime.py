"""Public Scene -> Simulation -> Result runtime for layered sheets, vias and ports."""

import copy
import time

import torch

from ...layers import LayerStack
from ...media import LossyMetalMedium
from ...ports import LumpedPort
from .network import prepare_layered_network, solve_prepared_network
from .loss import sheet_material_payload


CURRENT_KIND = "layered_rwg_port_currents_v1"


def _manhattan_rectangles(geometry, coordinate_maps=None):
    """Split a z-normal Manhattan PlanarPEC (holes allowed) into covering rectangles.

    Cells of the polygon's own vertex-coordinate grid are kept when their centre
    is inside under the even-odd rule over every loop. Host metadata only.
    """
    if geometry.normal_axis != "z":
        raise NotImplementedError("Layered MoM supports PlanarPEC sheets with normal_axis='z' only")
    loops = [loop.detach().cpu().tolist() for loop in (geometry.vertices, *geometry.holes)]
    if coordinate_maps is not None:
        loops = [[[coordinate_maps[a][p[a]] for a in (0, 1)] for p in loop] for loop in loops]
    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if a[0] != b[0] and a[1] != b[1]:
                raise NotImplementedError("Layered MoM requires Manhattan PlanarPEC polygons (axis-aligned edges)")
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
    return [clustered_coordinates(c, tolerance) for c in coordinates]


def compile_layered_scene(scene, *, breakpoint_tolerance_m=0.0):
    """Compile PEC/lossy PlanarPEC sheets, Box vias and LumpedPorts to material records."""
    if not isinstance(scene.background, LayerStack):
        raise TypeError("Layered MoM requires Scene(background=LayerStack(...))")
    if scene.sources or scene.monitors:
        raise NotImplementedError("Layered MoM v1 is driven by LumpedPorts only; sources and monitors are not supported")
    coordinate_maps = _coordinate_maps(scene, breakpoint_tolerance_m) if breakpoint_tolerance_m else None
    rectangles, materials, sheets, vias, via_materials = [], [], [], [], []
    for structure in scene.structures:
        if not getattr(structure, "enabled", True):
            continue
        from witwin.core import Box
        if isinstance(structure.geometry, Box):
            geometry, material = structure.geometry, structure.material
            rotation = geometry.rotation.detach().cpu().tolist()
            if rotation not in ([1.,0.,0.,0.], [0.,0.,0.]):
                raise NotImplementedError("Layered vias require an axis-aligned Box")
            position, size = (v.detach().to(torch.float64).cpu().tolist()
                              for v in (geometry.position, geometry.size))
            bounds = tuple(v for p, length in zip(position, size) for v in (p-length/2, p+length/2))
            if isinstance(material, LossyMetalMedium):
                record = (material.conductivity, 0.0, material.roughness)
            elif material.is_pec:
                record = (None, 0.0, None)
            else:
                raise ValueError("Layered via requires Material.pec() or LossyMetalMedium")
            vias.append((bounds, position, size))
            via_materials.append(record)
            continue
        if getattr(structure.geometry, "kind", None) != "planar_pec":
            raise NotImplementedError("Layered MoM supports PlanarPEC sheets and Box vias only")
        thickness = structure.geometry.physical_thickness
        material = structure.material
        if isinstance(material, LossyMetalMedium):
            if thickness <= 0:
                raise ValueError("LossyMetalMedium PlanarPEC requires physical_thickness > 0")
            record = (material.conductivity, thickness, material.roughness)
        elif material.is_pec:
            record = (None, thickness, None)
        else:
            raise ValueError("Layered PlanarPEC requires Material.pec() or LossyMetalMedium")
        pieces = _manhattan_rectangles(structure.geometry, coordinate_maps)
        rectangles.extend(pieces)
        materials.extend([record] * len(pieces))
        sheets.append(sheet_material_payload(record))
    if not rectangles:
        raise ValueError("Layered MoM needs at least one PlanarPEC sheet")
    if coordinate_maps is not None:
        vias = [(tuple(coordinate_maps[i//2][v] if i < 4 else v for i,v in enumerate(bounds)),
                 position, size) for bounds,position,size in vias]
    vias = [_snap_via(bounds, position, size, rectangles) for bounds, position, size in vias]
    if not scene.ports or any(type(port) is not LumpedPort for port in scene.ports):
        raise NotImplementedError("Layered MoM needs LumpedPort terminals only")
    ports = tuple(scene.ports)
    if coordinate_maps is not None:
        ports = tuple(copy.copy(port) for port in ports)
        for port in ports:
            for name in ("positive", "negative"):
                point = getattr(port, name)
                object.__setattr__(port, name, (*(coordinate_maps[a][float(point[a])] for a in (0,1)), point[2]))
    return scene.background, rectangles, ports, materials, tuple(sheets), vias, via_materials


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
        raise NotImplementedError("Layered MoM results do not support snapshot persistence yet: no payload schema "
                                  "binds the sheet mesh and RWG layout; save Result.network instead")


class LayeredNetworkRuntime:
    """Prepared layered MPIE network solve: meshing, Green table and port excitation."""

    def __init__(self, simulation):
        if simulation.has_trainable_parameters:
            raise NotImplementedError("Layered MoM geometry/material derivatives are not supported")
        options = simulation.config.layered
        stack, rectangles, ports, materials, sheets, vias, via_materials = compile_layered_scene(
            simulation.scene, breakpoint_tolerance_m=options.breakpoint_tolerance_m)
        self.simulation = simulation
        self.compiled = (stack, rectangles, ports, materials, sheets, vias, via_materials)
        self.prepared = prepare_layered_network(
            stack, rectangles, ports, simulation.frequencies, bridge_width=options.bridge_width_m,
            max_edge=options.max_edge_length_m, edge_cell=options.edge_cell_m,
            growth_ratio=options.growth_ratio,
            sheet_materials=materials, vias=vias, via_materials=via_materials,
            device=simulation.scene.device)
        self.prepared.metadata.update(sheets=sheets, breakpoint_tolerance_m=options.breakpoint_tolerance_m)
        self.prepared_config, self.prepared_frequencies = simulation.config, simulation.frequencies

    def _solve_frequencies(self, frequencies):
        options = self.simulation.config.layered
        stack, rectangles, ports, materials, sheets, vias, via_materials = self.compiled
        prepared = prepare_layered_network(
            stack, rectangles, ports, frequencies, bridge_width=options.bridge_width_m,
            max_edge=options.max_edge_length_m, edge_cell=options.edge_cell_m,
            growth_ratio=options.growth_ratio,
            sheet_materials=materials, vias=vias, via_materials=via_materials,
            device=self.simulation.scene.device, prepared_geometry=self.prepared)
        prepared.metadata.update(sheets=sheets, breakpoint_tolerance_m=options.breakpoint_tolerance_m)
        network, currents = solve_prepared_network(prepared)
        return network, currents

    def run(self):
        from ...result import Result
        sim = self.simulation
        if sim.config != self.prepared_config or sim.frequencies != self.prepared_frequencies:
            raise ValueError("Numerical configuration or frequencies changed after layered MoM preparation")
        device = self.prepared.excitation.device
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        network, currents = solve_prepared_network(self.prepared)
        sweep_result = None
        if sim.config.layered.sweep is not None:
            from .sweep import adaptive_network
            network, currents, sweep_result = adaptive_network(
                sim.config.layered.sweep, network, currents, self._solve_frequencies)
        torch.cuda.synchronize(device)
        data = LayeredSurfaceData(currents, self.prepared.layout, network.frequencies, network.port_names)
        data.sweep = sweep_result
        stats = {"solve_seconds": time.perf_counter() - started, **copy.deepcopy(dict(network.metadata))}
        return Result(method="mom", scene=sim.scene, frequencies=tuple(network.frequencies.tolist()), solver=self,
                      surface_data=data, network=network, solver_stats=stats,
                      metadata={**sim.metadata, "native_data": CURRENT_KIND,
                                "layered": sim.config.layered.payload(),
                                "sheets": network.metadata["sheets"], "port_bridges": "ideal_lossless"})

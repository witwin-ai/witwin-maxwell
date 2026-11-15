"""Declarative SI process maps and GDS import into the standard Scene model."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from ...layers import LayerStack
from ...media import LossyMetalMedium
from ...errors import UnsupportedFeatureError
from ..._stability import experimental


@experimental("mom_layered")
@dataclass(frozen=True)
class LayerMap:
    """One GDS metal (layer, datatype), at exact SI z with physical thickness."""

    name: str
    layer: int
    z: float
    physical_thickness: float
    material: object
    datatype: int = 0


@experimental("mom_layered")
@dataclass(frozen=True)
class ViaMap:
    """Separate via tubes between two named metal planes.

    Rectangles import as Box, regular polygons as z-axis Prism and any other
    simple outline as a z-axis ExtrudedPolygon with the exact grid vertices.
    A via that passes intermediate planes must clear their metal or land fully
    inside it.
    """

    layer: int
    between: tuple[str, str]
    material: object
    datatype: int = 0


@experimental("mom_layered")
@dataclass(frozen=True)
class ProcessStack:
    """GDS process in SI; import yields Scene(background=dielectrics).

    Labels on pin_layer=(layer, texttype) use port:+:metal and port:-:metal.
    The label origin is the terminal. A ring/ground terminal is explicitly
    the minus label on that metal, never an inferred global net. Explicit
    LumpedPorts may be supplied instead (or with distinct names).
    GDS user coordinates multiply library.unit; library.precision is the
    physical database grid and is also the Boolean precision. Flattened xy
    coordinates round to that grid, including quarter-turn references and
    fractional-grid magnifications. No unit override. Metal polygons may be
    non-Manhattan (45-degree routing, discretized arcs); those layouts use the
    planar triangle mesher.

    breakpoint_tolerance_m is opt-in layout repair of near-coincident vertices:
    complete-span coordinate clusters snap to its SI lattice. Choose <= the
    minimum feature/10 and >= the database grid when repairing grid jitter.
    z plane identities are never clustered.
    """

    dielectrics: LayerStack
    metals: tuple[LayerMap, ...]
    vias: tuple[ViaMap, ...] = ()
    pin_layer: tuple[int, int] | None = None
    breakpoint_tolerance_m: float = 0.0

    def __post_init__(self):
        if not isinstance(self.dielectrics, LayerStack):
            raise TypeError("ProcessStack.dielectrics must be a LayerStack")
        names = [m.name for m in self.metals]
        keys = [(m.layer, m.datatype) for m in (*self.metals, *self.vias)]
        if len(set(names)) != len(names) or len(set(keys)) != len(keys):
            raise ValueError("Process metal names and GDS mappings must be unique")
        if not math.isfinite(self.breakpoint_tolerance_m) or self.breakpoint_tolerance_m < 0:
            raise ValueError("breakpoint_tolerance_m must be finite and nonnegative")
        for m in self.metals:
            if not math.isfinite(m.z) or not math.isfinite(m.physical_thickness) or m.physical_thickness < 0:
                raise ValueError("Metal z and physical_thickness must be finite; thickness must be nonnegative")
            if not (isinstance(m.material, LossyMetalMedium) or m.material.is_pec):
                raise ValueError("Process metals require Material.pec() or LossyMetalMedium")
            if isinstance(m.material, LossyMetalMedium) and m.physical_thickness <= 0:
                raise ValueError("Lossy process metal requires positive physical_thickness")
        for v in self.vias:
            if len(v.between) != 2 or any(n not in names for n in v.between):
                raise ValueError("ViaMap.between must name two declared metal planes")
            heights = sorted(next(m.z for m in self.metals if m.name == n) for n in v.between)
            if heights[0] == heights[1]:
                raise ValueError("ViaMap.between must name two distinct metal planes")
            if not (isinstance(v.material, LossyMetalMedium) or v.material.is_pec):
                raise ValueError("Process vias require Material.pec() or LossyMetalMedium")

    def scene_from_gds(self, path, *, cell=None, ports=(), strict=True):
        """Flatten references/arrays, union each plane and preserve polygon holes."""
        from .planar_mesh import optional_module
        gdstk = optional_module("gdstk", "ProcessStack.scene_from_gds")
        from ...adapters.gds import _resolve_cell
        from ...geometry import PlanarPEC
        from ...ports import AxisPath, LumpedPort
        from ...scene import Scene
        from ... import Structure, Box
        from .mesh import clustered_coordinates

        library = gdstk.read_gds(str(path))
        selected = _resolve_cell(library, cell)
        polygons = selected.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
        labels = selected.get_labels(apply_repetitions=True, depth=None)
        known = {(m.layer, m.datatype) for m in (*self.metals, *self.vias)}
        if strict:
            unknown = {(p.layer, p.datatype) for p in polygons} - known
            unknown |= {(l.layer, l.texttype) for l in labels} - (
                {self.pin_layer} if self.pin_layer is not None else set())
            if unknown:
                raise ValueError(f"Unknown GDS (layer, datatype/texttype) in strict mode: {sorted(unknown)}")
        polygons = [p for p in polygons if (p.layer, p.datatype) in known]
        labels = [l for l in labels if (l.layer, l.texttype) == self.pin_layer]
        scale = library.unit
        grid = library.precision / scale
        # Flattened references can carry sin/cos roundoff at quarter turns.
        # Canonicalize to the declared Boolean/database grid; this also
        # defines sub-database magnification.
        polygons = [gdstk.Polygon(np.rint(p.points / grid) * grid,
                                 layer=p.layer, datatype=p.datatype) for p in polygons]
        arrays = [np.asarray(p.points) * scale for p in polygons]
        origins = [np.rint(np.asarray(l.origin) / grid) * grid * scale for l in labels]
        maps = [clustered_coordinates([float(x[a]) for pts in arrays for x in pts] +
                                     [float(x[a]) for x in origins], self.breakpoint_tolerance_m)
                for a in (0, 1)]
        repaired = []
        for polygon, points in zip(polygons, arrays):
            points = np.array([[maps[a][float(p[a])] for a in (0, 1)] for p in points])
            repaired.append(gdstk.Polygon(points / scale, layer=polygon.layer, datatype=polygon.datatype)
                            if self.breakpoint_tolerance_m else polygon)
        polygons = repaired
        structures, scene_ports = [], []
        planes = {}
        for m in self.metals:
            if m.z in planes:
                previous = planes[m.z][0]
                def physical_record(mapping):
                    material = mapping.material
                    return ((material.conductivity, material.roughness) if isinstance(material, LossyMetalMedium)
                            else (None, None), mapping.physical_thickness)
                if physical_record(m) != physical_record(previous):
                    raise ValueError("Metal mappings on one plane must share material and thickness")
                planes[m.z][1].append(m)
            else:
                planes[m.z] = (m, [m])
        for z, (metal, mappings) in sorted(planes.items()):
            keys = {(m.layer, m.datatype) for m in mappings}
            selected_polygons = [p for p in polygons if (p.layer, p.datatype) in keys]
            merged = gdstk.boolean(selected_polygons, [], "or", precision=library.precision / scale)
            for polygon in merged:
                outer, holes = _boundary_loops(polygon.points * scale)
                structures.append(Structure(PlanarPEC(outer, holes=holes, normal_axis="z",
                    coordinate=z, physical_thickness=metal.physical_thickness), metal.material))
        for via in self.vias:
            za, zb = sorted(next(m.z for m in self.metals if m.name == n) for n in via.between)
            for p in polygons:
                if (p.layer, p.datatype) != (via.layer, via.datatype):
                    continue
                points = p.points * scale
                low, high = points.min(0), points.max(0)
                loop = [tuple(point) for point in points]
                loop = [point for i,point in enumerate(loop) if point != loop[i-1]]
                # Rectangle admission uses exact turns, including tiny notches.
                corners = [b for a,b,c in zip(loop[-1:]+loop[:-1],loop,loop[1:]+loop[:1])
                           if not (a[0] == b[0] == c[0] or a[1] == b[1] == c[1])]
                expected = {(low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1])}
                if len(corners) != 4 or set(corners) != expected:
                    structures.append(Structure(_polygon_via(loop, za, zb, library.precision), via.material))
                    continue
                # Form Box centre/size in user units before SI conversion.
                # This retains the same arithmetic as a declarative Box built
                # from layout coordinates, avoiding an extra SI round trip.
                user_low, user_high = p.points.min(0), p.points.max(0)
                position = torch.tensor((*((user_low + user_high) / 2 * scale), (za + zb) / 2),
                                        dtype=torch.float64)
                size = torch.tensor((*(user_high - user_low) * scale, zb - za), dtype=torch.float64)
                structures.append(Structure(Box(position=position, size=size), via.material))
        terminals = {}
        for label, origin in zip(labels, origins):
            if (label.layer, label.texttype) != self.pin_layer:
                continue
            parts = label.text.split(":")
            if len(parts) != 3 or parts[1] not in ("+", "-") or not parts[0]:
                raise ValueError("Pin label must have form name:+:metal or name:-:metal")
            name, sign, metal = parts
            if metal not in {m.name for m in self.metals}:
                raise ValueError(f"Pin label names an unknown metal plane: {metal}")
            pair = terminals.setdefault(name, {})
            if sign in pair:
                raise ValueError(f"Duplicate pin terminal {name}:{sign}")
            pair[sign] = (*(maps[a][float(origin[a])] for a in (0, 1)),
                          next(m.z for m in self.metals if m.name == metal))
        for name, pair in sorted(terminals.items()):
            if set(pair) != {"+", "-"}:
                raise ValueError(f"Pin label {name!r} has no partner terminal")
            positive, negative = pair["+"], pair["-"]
            axes = [a for a in range(3) if positive[a] != negative[a]]
            if len(axes) != 1:
                raise ValueError("Pin terminals must define a distinct axis-aligned path")
            scene_ports.append(LumpedPort(name, positive=positive, negative=negative,
                voltage_path=AxisPath("xyz"[axes[0]]), reference_impedance=50.0))
        return Scene(background=self.dielectrics, structures=structures, ports=(*scene_ports, *ports))


def _polygon_via(loop, za, zb, grid):
    """z-axis via of a non-rectangular outline in SI: a Prism when the outline is a regular polygon
    within the database grid, otherwise an ExtrudedPolygon with the exact outline vertices."""
    from witwin.core import ExtrudedPolygon, Prism
    points = np.array(loop, dtype=np.float64)
    before, after = points - np.roll(points, 1, 0), np.roll(points, -1, 0) - points
    turns = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    corners = points[turns != 0]
    centre = corners.mean(0)
    radii = np.hypot(*(corners - centre).T)
    count = len(corners)
    angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    steps = (np.roll(angles, -1) - angles) % (2 * np.pi)
    if np.sum(turns) < 0:
        steps = 2 * np.pi - steps
    regular = (count >= 3 and np.ptp(radii) <= 2 * grid and
               np.abs(steps - 2 * np.pi / count).max() * radii.mean() <= 2 * grid)
    if not regular:
        return ExtrudedPolygon(torch.tensor(points, dtype=torch.float64), height=torch.tensor(zb - za, dtype=torch.float64),
                               position=torch.tensor((0.0, 0.0, (za + zb) / 2), dtype=torch.float64), axis="z")
    theta = angles[0]
    return Prism(position=torch.tensor((*centre, (za + zb) / 2), dtype=torch.float64),
                 radius=torch.tensor(radii.mean(), dtype=torch.float64),
                 height=torch.tensor(zb - za, dtype=torch.float64), num_sides=count, axis="z",
                 rotation=torch.tensor((math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)), dtype=torch.float64))


def _boundary_loops(points):
    """Undo GDS keyhole bridges into one outer loop and its strictly inner holes."""
    edges = {}
    vertices = [tuple(p) for p in points]
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if (b, a) in edges:
            del edges[(b, a)]
        else:
            edges[(a, b)] = None
    loops = []
    while edges:
        a, b = next(iter(edges))
        loop, first = [a], a
        del edges[(a, b)]
        while b != first:
            loop.append(b)
            a, b = next(edge for edge in edges if edge[0] == b)
            del edges[(a, b)]
        loop = [p for i, p in enumerate(loop)
                if not (loop[i-1][0] == p[0] == loop[(i+1)%len(loop)][0] or
                        loop[i-1][1] == p[1] == loop[(i+1)%len(loop)][1])]
        loops.append(loop)
    def area(loop):
        return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(loop,loop[1:]+loop[:1])))
    loops.sort(key=area, reverse=True)
    return loops[0], loops[1:]

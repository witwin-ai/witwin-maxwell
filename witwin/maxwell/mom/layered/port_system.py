"""Layered MoM port types: delta-gap, full-width edge, vertical and mixed-mode pairs.

Every layered port is a shared ``LumpedPort`` realized as an ideal, lossless
bridge strip between its two terminals with a delta-gap voltage source on the
strip's midpoint cut (sign chain in network.py, ``exp(-i*omega*t)``):

- Voltage: ``V = phi(positive) - phi(negative)``, the impressed delta-gap
  voltage across the cut, integrated from the negative to the positive
  terminal. It is the same on every parallel path across the bridge width.
- Current: ``I`` is the total RWG surface current crossing the cut, counted
  positive when it leaves the positive terminal into the structure (the
  NetworkData port current). Power is ``0.5*Re(V*conj(I))``.

The bridge width selects the port type:

- *delta-gap* (no ``voltage_aperture``): width ``LayeredSurfaceConfig.bridge_width_m``,
  centred on the terminal path.
- *edge* (``voltage_aperture`` Box): the bridge spans the aperture's
  transverse extent, normally the full width of a conductor edge, so the
  source current is distributed over the whole edge. This is the layered
  meaning of the shared ``LumpedPort.voltage_aperture`` (voltage distributed
  over parallel paths with one port current), which FDTD already uses; FDTD
  behaviour is unchanged. Both terminal ends must be covered by metal across
  the full width and the gap interior must be free of metal.

The negative terminal names the ground reference explicitly by where it lies:

- a conductor on the same metal plane (horizontal bridge across a gap, e.g.
  a coplanar ground or the other conductor of a floating differential port);
- a stack PEC ground (``LayerStack(bottom="pec")`` / ``top="pec"``): a vertical
  bridge from a sheet boundary to the plane;
- a conductor on the adjacent metal plane: a vertical bridge between the two
  sheet boundaries (vertical inter-layer port; the sheet model needs both
  ends on a sheet boundary, interior landings need junction RWGs and raise).

``edge_port`` builds such ports from named Structures. ``port_records``
recovers the type, conductors, width and the V/I definitions of compiled
ports for result metadata. Differential/common-mode pairs of ground-referenced
ports are a lossless basis change of the solved network (``PortPair``,
``mixed_mode``); a single port between two signal conductors is a floating
differential port.
"""

from dataclasses import dataclass

import torch

from ...errors import UnsupportedFeatureError

VOLTAGE_DEFINITION = ("V = phi(positive) - phi(negative): impressed delta-gap voltage across the bridge midpoint cut, "
                      "integrated from the negative to the positive terminal, equal on every parallel path")
CURRENT_DEFINITION = ("I = total RWG current crossing the bridge cut over the full bridge width, positive leaving the "
                      "positive terminal into the structure; P = 0.5*Re(V*conj(I))")
PEC_REFERENCES = ("pec_bottom", "pec_top")


def _bounds(box):
    """((lo, hi) per axis) of an axis-aligned Box in float64 plus its float rounding scale."""
    rotation = box.rotation.detach().cpu().tolist()
    if rotation not in ([1., 0., 0., 0.], [0., 0., 0.]):
        raise UnsupportedFeatureError("A layered voltage_aperture must be an axis-aligned Box",
                                      feature="guard:35c875fb", method="mom.layered")
    position, size = (v.detach().to(torch.float64).cpu().tolist() for v in (box.position, box.size))
    scale = [1e-6 * (abs(p) + s) for p, s in zip(position, size)]
    return [(p - s / 2, p + s / 2) for p, s in zip(position, size)], scale


def _snap(value, candidates, tolerance):
    """Replace a float32-rounded aperture bound by the unique layout coordinate within tolerance."""
    near = sorted({c for c in candidates if abs(c - value) <= tolerance})
    return near[0] if len(near) == 1 else value


def _close(a, b, tolerance):
    return abs(a - b) <= tolerance


def horizontal_span(port, rectangles, axis, z, bridge_width):
    """Lateral (lo, hi, centre, half) of a horizontal bridge along ``axis`` on plane ``z``.

    Without an aperture this is the configured bridge width centred on the
    terminal path. With one, the aperture's lateral extent, after validating
    that the aperture is the terminal span on that plane and that the bridge
    ends are fully covered by metal while its interior is metal-free.
    """
    lateral = 1 - axis
    centre = float(port.positive[lateral])
    if port.voltage_aperture is None:
        half = 0.5 * bridge_width
        return centre - half, centre + half, centre, half
    name = port.name
    bounds, scale = _bounds(port.voltage_aperture)
    plane = [r for r in rectangles if r[4] == z]
    lo, hi = (_snap(v, [c for r in plane for c in r[2 * lateral:2 * lateral + 2]], scale[lateral])
              for v in bounds[lateral])
    low, high = sorted((float(port.positive[axis]), float(port.negative[axis])))
    if not (_close(bounds[2][0], z, scale[2]) and _close(bounds[2][1], z, scale[2])):
        raise ValueError(f"Port {name!r} voltage_aperture of a horizontal port must have zero z extent on its plane")
    if not (_close(bounds[axis][0], low, scale[axis]) and _close(bounds[axis][1], high, scale[axis])):
        raise ValueError(f"Port {name!r} voltage_aperture must span exactly the terminal path along "
                         f"{'xy'[axis]} ({low:g}..{high:g})")
    if not lo < centre < hi:
        raise ValueError(f"Port {name!r} voltage_aperture must contain the terminal path in its width")
    for end, outside in ((low, -1), (high, 1)):
        # Metal adjoining the bridge end from outside the gap, over the whole width.
        touching = [r for r in plane if (r[2 * axis] < end <= r[2 * axis + 1] if outside < 0 else
                                         r[2 * axis] <= end < r[2 * axis + 1])]
        if not _covers([r[2 * lateral:2 * lateral + 2] for r in touching], lo, hi):
            raise ValueError(f"Port {name!r} edge bridge end at {'xy'[axis]}={end:g} is not covered by metal "
                             "over the full voltage_aperture width")
    if any(r[2 * axis] < high and low < r[2 * axis + 1] and r[2 * lateral] < hi and lo < r[2 * lateral + 1]
           for r in plane):
        raise ValueError(f"Port {name!r} edge bridge interior overlaps metal; the aperture must span an open gap")
    return lo, hi, 0.5 * (lo + hi), 0.5 * (hi - lo)


def vertical_span(port, rectangles):
    """None, or (normal axis, lo, hi) of a vertical aperture's strip plane and lateral width."""
    if port.voltage_aperture is None:
        return None
    name = port.name
    bounds, scale = _bounds(port.voltage_aperture)
    point = [float(v) for v in port.positive]
    za, zb = sorted((float(port.positive[2]), float(port.negative[2])))
    if not (_close(bounds[2][0], za, scale[2]) and _close(bounds[2][1], zb, scale[2])):
        raise ValueError(f"Port {name!r} voltage_aperture must span exactly the terminal path along z ({za:g}..{zb:g})")
    flat = [a for a in (0, 1) if _close(bounds[a][0], bounds[a][1], scale[a])]
    if len(flat) != 1:
        raise ValueError(f"Port {name!r} vertical voltage_aperture needs zero extent along exactly one of x, y "
                         "(the strip plane normal) and a positive width along the other")
    normal = flat[0]
    lateral = 1 - normal
    if not _close(0.5 * (bounds[normal][0] + bounds[normal][1]), point[normal], scale[normal]):
        raise ValueError(f"Port {name!r} vertical voltage_aperture must contain the terminal path")
    lo, hi = (_snap(v, [c for r in rectangles for c in r[2 * lateral:2 * lateral + 2]], scale[lateral])
              for v in bounds[lateral])
    if not lo < point[lateral] < hi:
        raise ValueError(f"Port {name!r} voltage_aperture must contain the terminal path in its width")
    return normal, lo, hi


def _covers(intervals, lo, hi):
    reach = lo
    for a, b in sorted(intervals):
        if a > reach:
            break
        reach = max(reach, b)
    return reach >= hi


def _sheet_covers(record, point):
    """True when the sheet record (rectangle tuple or planar PolygonSheet) covers the point on its plane."""
    if hasattr(record, "loops"):
        from .planar_mesh import optional_module
        shapely = optional_module("shapely", "Layered polygon port records")
        return record.z == point[2] and shapely.Polygon(record.loops[0], record.loops[1:]).covers(
            shapely.Point(point[0], point[1]))
    return record[4] == point[2] and record[0] <= point[0] <= record[1] and record[2] <= point[1] <= record[3]


def _owner(point, rectangles, owners):
    names = {owners[i] for i, r in enumerate(rectangles) if _sheet_covers(r, point)}
    return min(names) if names else None


def port_records(ports, geometry, halves, rectangles, owners, stack):
    """Per-port metadata: type, conductors, width, length, reference impedance and the V/I definitions.

    ``owners`` names the declaring Structure of every rectangle (or planar
    PolygonSheet); ``geometry`` and ``halves`` are the compiled bridge records
    of network.py or planar.py.
    """
    interfaces = stack.interfaces()
    planes = {interfaces[0]: "pec_bottom"} if stack.bottom == "pec" else {}
    if stack.top == "pec":
        planes[interfaces[-1]] = "pec_top"
    records = []
    for port, record, half in zip(ports, geometry, halves):
        positive, negative = (tuple(float(v) for v in p) for p in (port.positive, port.negative))
        vertical = len(record) == 9
        path = 2 if vertical else record[2]
        reference = _owner(negative, rectangles, owners)
        if reference is None and vertical:
            reference = planes[negative[2]]
        records.append(dict(
            name=port.name, kind="edge" if port.voltage_aperture is not None else "delta_gap",
            orientation="vertical" if vertical else "horizontal",
            positive_conductor=_owner(positive, rectangles, owners) if not (vertical and positive[2] in planes)
            else planes[positive[2]],
            reference=reference,
            reference_kind="pec_plane" if reference in PEC_REFERENCES else "conductor",
            width_m=2 * half, length_m=abs(positive[path] - negative[path]),
            cut_m=record[6] if vertical else record[3],
            # (real, imag) in ohm: the records stay JSON-serializable like the rest of the network metadata.
            reference_impedance_ohm=(complex(port.reference_impedance).real, complex(port.reference_impedance).imag),
            voltage_definition=VOLTAGE_DEFINITION, current_definition=CURRENT_DEFINITION,
            bridge="ideal_lossless"))
    return tuple(records)


def structure_names(structures):
    """Stable names of declared Structures: ``name`` or ``structure[i]`` by declaration order."""
    return {id(s): s.name if s.name else f"structure[{i}]" for i, s in enumerate(structures)}


def _loops(structure):
    """(outer loop, every loop including holes, plane z) of a z-normal PlanarPEC."""
    geometry = structure.geometry
    if getattr(geometry, "kind", None) != "planar_pec" or geometry.normal_axis != "z":
        raise ValueError(f"edge_port conductors must be z-normal PlanarPEC structures; {structure.name!r} is not")
    loops = [loop.detach().to(torch.float64).cpu().tolist() for loop in (geometry.vertices, *geometry.holes)]
    return loops[0], loops, float(geometry.coordinate)


def _named(scene, name):
    matches = [s for s in scene.structures if s.name == name and getattr(s, "enabled", True)]
    if len(matches) != 1:
        raise ValueError(f"edge_port needs exactly one enabled Structure named {name!r}; found {len(matches)}")
    return matches[0]


def _edge(structure, side):
    """(axis, coordinate, lo, hi, z) of the conductor's single outermost boundary segment on ``side``."""
    if side not in ("-x", "+x", "-y", "+y"):
        raise ValueError("edge_port side must be one of '-x', '+x', '-y', '+y'")
    axis = "xy".index(side[1])
    vertices, _, z = _loops(structure)
    values = [p[axis] for p in vertices]
    coordinate = min(values) if side[0] == "-" else max(values)
    segments = [sorted((a[1 - axis], b[1 - axis])) for a, b in zip(vertices, vertices[1:] + vertices[:1])
                if a[axis] == coordinate == b[axis]]
    segments.sort()
    if not segments or any(b[0] != a[1] for a, b in zip(segments, segments[1:])):
        raise ValueError(f"Conductor {structure.name!r} has no single contiguous edge on side {side}; "
                         "declare a LumpedPort with voltage_aperture directly")
    return axis, coordinate, segments[0][0], segments[-1][1], z


def edge_port(scene, name, *, signal, side, reference, reference_impedance=50.0):
    """Full-width edge port on the ``side`` edge of the PlanarPEC named ``signal``.

    ``reference`` is the explicit ground reference: ``"pec_bottom"`` or
    ``"pec_top"`` (the stack PEC plane; vertical port), or the name of a
    PlanarPEC conductor. A conductor on the same plane is reached across the
    open gap outward from the edge; its facing edge must cover the full width
    (horizontal edge port). A conductor on another plane must have a boundary
    on the same edge line (vertical inter-layer edge port). The positive
    terminal is on ``signal``. Returns a ``LumpedPort`` with a
    ``voltage_aperture`` spanning the edge; pass it in the Scene ``ports``.
    """
    from witwin.core import Box
    from ...ports import AxisPath, LumpedPort
    axis, coordinate, lo, hi, z = _edge(_named(scene, signal), side)
    lateral = 1 - axis
    centre = 0.5 * (lo + hi)

    def point(a, b, height):
        values = [0., 0., height]
        values[axis], values[lateral] = a, b
        return tuple(values)

    def box(along, span, low_z, high_z):
        position, size = [0., 0., .5 * (low_z + high_z)], [0., 0., high_z - low_z]
        position[axis], size[axis] = .5 * (along[0] + along[1]), along[1] - along[0]
        position[lateral], size[lateral] = .5 * (span[0] + span[1]), span[1] - span[0]
        return Box(position=torch.tensor(position, dtype=torch.float64), size=torch.tensor(size, dtype=torch.float64))

    stack = scene.background
    if reference in PEC_REFERENCES:
        interfaces = stack.interfaces()
        if getattr(stack, reference[4:]) != "pec":
            raise ValueError(f"edge_port reference {reference!r} needs LayerStack({reference[4:]}='pec')")
        ground = interfaces[0] if reference == "pec_bottom" else interfaces[-1]
        negative = point(coordinate, centre, ground)
        return LumpedPort(name, positive=point(coordinate, centre, z), negative=negative,
                          voltage_path=AxisPath("z"), reference_impedance=reference_impedance,
                          voltage_aperture=box((coordinate, coordinate), (lo, hi), *sorted((z, ground))))
    _, loops, ground = _loops(_named(scene, reference))
    if ground != z:
        return LumpedPort(name, positive=point(coordinate, centre, z), negative=point(coordinate, centre, ground),
                          voltage_path=AxisPath("z"), reference_impedance=reference_impedance,
                          voltage_aperture=box((coordinate, coordinate), (lo, hi), *sorted((z, ground))))
    outward = 1 if side[0] == "+" else -1
    facing = [a[axis] for loop in loops for a, b in zip(loop, loop[1:] + loop[:1])
              if a[axis] == b[axis] and outward * (a[axis] - coordinate) > 0
              and min(a[lateral], b[lateral]) <= lo and max(a[lateral], b[lateral]) >= hi]
    if not facing:
        raise ValueError(f"Reference {reference!r} has no edge facing the {side} edge of {signal!r} over its full width")
    target = min(facing, key=lambda v: outward * (v - coordinate))
    return LumpedPort(name, positive=point(coordinate, centre, z), negative=point(target, centre, z),
                      voltage_path=AxisPath("xy"[axis]), reference_impedance=reference_impedance,
                      voltage_aperture=box(tuple(sorted((coordinate, target))), (lo, hi), z, z))


@dataclass(frozen=True)
class PortPair:
    """A differential/common-mode pair of two ground-referenced ports.

    ``Vd = Vp - Vn``, ``Id = (Ip - In)/2``; ``Vc = (Vp + Vn)/2``, ``Ic = Ip + In``
    (power-preserving); default references ``Zd = 2 z0`` and ``Zc = z0/2``.
    """

    name: str
    positive: str
    negative: str

    def __post_init__(self):
        if not self.name or self.positive == self.negative:
            raise ValueError("PortPair needs a name and two distinct port names")


def mixed_mode(network, pairs, z0=None):
    """Differential then common mode of each pair, then unpaired ports (NetworkData.to_mixed_mode).

    Mode ports are named ``"<pair>:d"`` and ``"<pair>:c"``; the definitions are
    recorded in ``metadata["mixed_mode_pairs"]``.
    """
    pairs = tuple(pairs)
    if not all(isinstance(p, PortPair) for p in pairs):
        raise TypeError("mixed_mode pairs must be PortPair instances")
    if len({p.name for p in pairs}) != len(pairs):
        raise ValueError("mixed_mode pair names must be unique")
    converted = network.to_mixed_mode([(p.positive, p.negative) for p in pairs], z0=z0)
    names = [n for p in pairs for n in (f"{p.name}:d", f"{p.name}:c")]
    names += list(converted.port_names[2 * len(pairs):])
    metadata = dict(converted.metadata)
    metadata["mixed_mode_pairs"] = tuple(dict(
        name=p.name, positive=p.positive, negative=p.negative,
        differential="Vd = Vp - Vn, Id = (Ip - In)/2, default Zd = 2*z0",
        common="Vc = (Vp + Vn)/2, Ic = Ip + In, default Zc = z0/2") for p in pairs)
    return type(converted)(frequencies=converted.frequencies, s=converted.s, z0=converted.z0,
                           port_names=tuple(names), valid_columns=converted.valid_columns, metadata=metadata,
                           phasor_convention=converted.phasor_convention,
                           power_wave_convention=converted.power_wave_convention)


def aperture_record(port):
    """Signature/metadata tuple of a port's aperture bounds (None without one)."""
    if port.voltage_aperture is None:
        return None
    bounds, _ = _bounds(port.voltage_aperture)
    return tuple(v for pair in bounds for v in pair)


def map_aperture(port, coordinate_maps):
    """Copy of the aperture Box with its x/y bounds moved by the breakpoint clustering maps."""
    from witwin.core import Box
    bounds, _ = _bounds(port.voltage_aperture)
    mapped = [tuple(coordinate_maps[a].get(v, v) for v in bounds[a]) if a < 2 else bounds[a] for a in range(3)]
    return Box(position=torch.tensor([.5 * (a + b) for a, b in mapped], dtype=torch.float64),
               size=torch.tensor([b - a for a, b in mapped], dtype=torch.float64))

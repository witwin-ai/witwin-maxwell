"""B1/B2 polygon fixtures through Scene -> Simulation -> Result, and their closed-form references.

Coordinates are micrometres unless stated. The spiral pair shares the F-A
one-port arrangement (3.5 turns, outer side 200 um, width/spacing 10/5 um,
underpass at z=302 um through two 4x4 um vias, leads to x=125 um and a y gap
port at y=85..90 um). The octagonal spiral chamfers every corner at 45 degrees
so each diagonal lies at the same distance from the centre as the sides around
it: a regular octagonal spiral with the square spiral's turn pitch.
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from tests.support.rfic_fixtures import SPIRAL_XY

shapely = pytest.importorskip("shapely")


UM = 1e-6
MU0 = 4e-7 * math.pi


def vacuum():
    return mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())


def polygon_structure(polygon, z, material, thickness=0.0):
    loops = [list(polygon.exterior.coords)[:-1], *[list(r.coords)[:-1] for r in polygon.interiors]]
    loops = [torch.tensor(loop, dtype=torch.float64) * UM for loop in loops]
    return mw.Structure(mw.geometry.PlanarPEC(loops[0], holes=loops[1:], normal_axis="z", coordinate=z * UM,
                                     physical_thickness=thickness * UM), material)


def gap_port(name, positive, negative, axis):
    return mw.LumpedPort(name, positive=tuple(v * UM for v in positive), negative=tuple(v * UM for v in negative),
                         voltage_path=mw.AxisPath(axis), reference_impedance=50.0)


def ring_loop(radius=250.0, width=20.0, sides=96, gap=10.0, z=500.0, bridge=16.0):
    """PEC regular-polygon annulus with a flat bottom edge, a centred x gap and its port.

    Returns (scene, bridge width in um). The bridge is narrower than the flat
    strip so its sides stay inside the metal.
    """
    phase = -math.pi / 2 + math.pi / sides

    def arc(r):
        return [(r * math.cos(phase + 2 * math.pi * k / sides), r * math.sin(phase + 2 * math.pi * k / sides))
                for k in range(sides)]

    outer, inner = arc(radius + width / 2), arc(radius - width / 2)
    y_out, y_in = outer[-1][1], inner[-1][1]
    loop = [(gap / 2, y_out), *outer[:-1], (-gap / 2, y_out), (-gap / 2, y_in), *inner[:-1][::-1], (gap / 2, y_in)]
    centre = (y_out + y_in) / 2
    scene = mw.Scene(background=vacuum(), structures=(
                polygon_structure(shapely.Polygon(loop), z, mw.Material.pec()),
            ), ports=(
                gap_port("loop", (gap / 2, centre, z), (-gap / 2, centre, z), "x"),
            ))
    return scene, bridge


def circular_loop_inductance(radius, strip_width):
    """Grover/Kirchhoff loop of round wire with surface current, mu0 R (ln(8R/a) - 2), a = w/4.

    A thin PEC strip of width w has the equivalent (logarithmic-capacity)
    radius w/4. Corrections are O((a/R)^2 ln) (1e-4 here) and are omitted.
    """
    return MU0 * radius * (math.log(8 * radius / (strip_width / 4)) - 2)


def spiral_centreline(octagonal):
    """F-A spiral centreline from the outer lead end; corners chamfered at 45 degrees when octagonal.

    A corner (x, y) between sides at distances |x| and |y| from the centre is
    replaced by two points at leg t = (|x|+|y|)(1 - 1/sqrt 2) along its sides;
    the chamfer then lies at the mean side distance from the centre. The
    octagonal spiral ends at the midpoint of its last chamfer (the angular
    position of the square spiral's inner corner (-50, -50)), which keeps the
    inner turn one pitch away from the previous one.
    """
    points = [(125, 95), *SPIRAL_XY]
    if not octagonal:
        return points
    result = [points[0], points[1]]
    for previous, corner, following in zip(points[1:-1], points[2:], points[3:] + [(-50, 0)]):
        leg = (abs(corner[0]) + abs(corner[1])) * (1 - 1 / math.sqrt(2))
        if corner == points[-1]:
            following = (corner[0] + 1, corner[1])
        for neighbour in (previous, following):
            length = math.dist(neighbour, corner)
            result.append(tuple(c + (n - c) * leg / length for c, n in zip(corner, neighbour)))
    first, second = result[-2:]
    result[-1] = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
    return result


def spiral_layout(octagonal):
    """(top-plane metal, underpass, inner via centre, outer via centre) of the one-port spiral."""
    centreline = spiral_centreline(octagonal)
    end = centreline[-1]
    style = dict(cap_style="square", join_style="mitre", mitre_limit=10.0)
    top = shapely.union_all([shapely.LineString(centreline).buffer(5, **style),
                             shapely.LineString(((125, end[1]), (125, 80))).buffer(5, **style)])
    under = shapely.LineString((end, (125, end[1]))).buffer(5, **style)
    return top, under, end, (125, end[1])


def spiral_scene(octagonal, *, lossy=False):
    """One-port spiral on the F-A silicon/oxide stack; PEC unless lossy (copper, F-A thickness)."""
    stack = mw.mom.LayerStack((mw.mom.Layer("silicon", 300 * UM, 11.9, 10.), mw.mom.Layer("oxide", 6 * UM, 4.1)),
                          mw.mom.HalfSpace(11.9, 10.), mw.mom.HalfSpace())
    scene = mw.Scene(background=stack)
    material = mw.media.LossyMetalMedium(conductivity=5.8e7) if lossy else mw.Material.pec()
    top, under, inner, outer = spiral_layout(octagonal)
    for polygon, z, thickness in ((top, 305, 2.), (under, 302, 1.)):
        scene = scene.updated(structures=(
            *scene.structures,
            *(polygon_structure(part, z, material, thickness) for part in shapely.get_parts(polygon)),
        ))
    scene = scene.updated(structures=(
        *scene.structures,
        *(mw.Structure(mw.Box(position=torch.tensor((x, y, 303.5), dtype=torch.float64) * UM,
                              size=torch.tensor((4, 4, 3), dtype=torch.float64) * UM), material) for x, y in (inner, outer)),
    ))
    scene = scene.updated(ports=(*scene.ports, gap_port("loop", (125, 90, 305), (125, 85, 305), "y")))
    return scene


def modified_wheeler(outer, inner, turns, octagonal):
    """Spiral-only modified Wheeler inductance (K1, K2 = 2.34, 2.75 square; 2.33, 3.82 octagon), SI."""
    k1, k2 = (2.33, 3.82) if octagonal else (2.34, 2.75)
    average, fill = (outer + inner) / 2, (outer - inner) / (outer + inner)
    return k1 * MU0 * turns ** 2 * average / (1 + k2 * fill)


def regular_polygon_via_radius(circumradius, sides):
    """Logarithmic capacity of a regular N-gon (equivalent surface-current wire radius).

    The exterior map f'(z) = C (1 - z^-N)^(2/N) sends the unit circle to the
    polygon; one side has length C int_0^(2 pi/N) (2 sin(N t/2))^(2/N) dt
    = C (4/N) 2^(2/N) sqrt(pi) Gamma(1/N + 1/2) / Gamma(1/N + 1) / 2. N = 4
    reproduces the square value 0.5901702995 * side.
    """
    p = 2 / sides
    integral = (2 / sides) * 2 ** p * math.sqrt(math.pi) * math.gamma((p + 1) / 2) / math.gamma(p / 2 + 1)
    side = 2 * circumradius * math.sin(math.pi / sides)
    return side / integral

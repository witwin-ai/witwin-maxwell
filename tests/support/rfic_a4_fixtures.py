"""A4 certification fixtures F-D (MIM capacitor), F-E (microstrip / CPW lines) and F-F (octagonal differential inductor).

Public Scene -> Simulation -> Result path; lengths in metres. References are independent of the solver
(scripts/dev/rfic/quasistatic_sda_r7.py, closed forms, the FastHenry generator scripts/dev/rfic/octagon_reference_r7.py).
"""
import math

import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.port_system import edge_port

UM = 1e-6


def rectangle(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def sheet(name, vertices, z, holes=()):
    return mw.Structure(mw.geometry.PlanarPEC(vertices, holes=holes, normal_axis="z", coordinate=z),
                        mw.Material.pec(), name=name)


def config(**options):
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(**options))


# F-D: MIM capacitor. Two equal rectangular PEC plates (length along x, width along y) on the two faces of a nitride film
# inside oxide; one vertical delta-gap port at the middle of the +x edge. The film is laterally infinite (the layered
# model; local dielectric blocks are G9). Cases: (width, length); equal widths at two lengths give C' by differencing.
MIM_D, MIM_EPS, OX_EPS, MIM_BELOW, MIM_ABOVE = 1 * UM, 7.0, 4.1, 4 * UM, 2 * UM
MIM_Z = (MIM_BELOW, MIM_BELOW + MIM_D)
MIM_CASES = ((20 * UM, 60 * UM), (20 * UM, 120 * UM), (40 * UM, 60 * UM), (40 * UM, 120 * UM),
             (20 * UM, 20 * UM), (40 * UM, 40 * UM))
MIM_PORT_WIDTH = 2 * UM


def mim_stack():
    return mw.mom.LayerStack((mw.mom.Layer("oxide_lo", MIM_BELOW, OX_EPS), mw.mom.Layer("nitride", MIM_D, MIM_EPS),
                              mw.mom.Layer("oxide_hi", MIM_ABOVE, OX_EPS)), mw.mom.HalfSpace(OX_EPS), mw.mom.HalfSpace())


def mim(width, length):
    scene = mw.Scene(background=mim_stack())
    scene = scene.updated(structures=(
        *scene.structures,
        *(sheet(name, rectangle(-length / 2, length / 2, -width / 2, width / 2), z) for name, z in (("bottom", MIM_Z[0]), ("top", MIM_Z[1]))),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        mw.LumpedPort("p", positive=(length / 2, 0., MIM_Z[1]), negative=(length / 2, 0., MIM_Z[0]),
                      voltage_path=mw.AxisPath("z"), reference_impedance=50.),
    ))
    return scene


def mim_options(edge):
    return dict(bridge_width_m=MIM_PORT_WIDTH, max_edge_length_m=4 * UM, edge_cell_m=edge, growth_ratio=1.5)


# F-E (a): microstrip on a PEC ground (the C1 geometry: h = w = 254 um, eps 9.8), full-width vertical edge ports.
MS_H, MS_W, MS_EPS = 254 * UM, 254 * UM, 9.8
MS_LENGTHS = (2e-3, 4e-3)


def microstrip(length, eps=MS_EPS):
    """``eps=1`` is the air-filled diagnostic (exact eps_eff = 1, L' = 1/(c^2 C0'))."""
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", MS_H, eps),), "pec", mw.mom.HalfSpace())
    scene = mw.Scene(background=stack, structures=(
                sheet("sig", rectangle(0., length, -MS_W / 2, MS_W / 2), MS_H),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal="sig", side=side, reference="pec_bottom") for name, side in (("p1", "-x"), ("p2", "+x"))),
    ))
    return scene


def microstrip_options(edge):
    return dict(bridge_width_m=MS_W, max_edge_length_m=MS_H / 2, edge_cell_m=edge)


# F-E (b): finite-ground CPW at the interface of a silicon half-space (the C1 geometry), horizontal edge ports to the
# ground frame. S = slot = 10 um, grounds 40 um.
CPW_EPS, CPW_Z, CPW_S, CPW_SLOT, CPW_G, CPW_END = 11.9, 100 * UM, 10 * UM, 10 * UM, 40 * UM, 10 * UM
CPW_LENGTHS = (.6e-3, 1.2e-3)


def cpw(length):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", CPW_Z, CPW_EPS),), mw.mom.HalfSpace(CPW_EPS),
                              mw.mom.HalfSpace())
    outer, inner = CPW_S / 2 + CPW_SLOT + CPW_G, CPW_S / 2 + CPW_SLOT
    scene = mw.Scene(background=stack, structures=(
                sheet("gnd", rectangle(-CPW_G, length + CPW_G, -outer, outer), CPW_Z,
              holes=(rectangle(0., length, -inner, inner),)),
                sheet("sig", rectangle(CPW_END, length - CPW_END, -CPW_S / 2, CPW_S / 2), CPW_Z),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(edge_port(scene, name, signal="sig", side=side, reference="gnd") for name, side in (("p1", "-x"), ("p2", "+x"))),
    ))
    return scene


def cpw_options(edge):
    return dict(bridge_width_m=2 * UM, max_edge_length_m=20 * UM, edge_cell_m=edge)


def simulate(scene, frequencies, options):
    return mw.Simulation.mom(scene, frequencies=frequencies, config=config(**options)).run()


# F-F: two-turn symmetric octagonal differential inductor (regular octagons, flat sides on the axes). Apothems 70 / 58 um,
# width 8, spacing 4. Path (all clockwise from +z): port gap at the bottom of the outer ring (x = -6..6, the port bridge)
# -> outer-left half -> top-plane diagonal (-X, R1) -> (X, R2) -> inner ring (closed at the bottom) -> via at (-X-3, R2)
# -> underpass diagonal on the lower plane -> via at (X+3, R1) -> outer-right half -> port gap. PEC sheets in a lossless
# oxide (quasi-static L does not depend on eps); vias 4x4 um Box between z = 6.5 and 8 um.
FF_R, FF_W, FF_X, FF_GAP, FF_VIA = (70., 58.), 8., 12., 12., 4.
FF_Z = (6.5, 8.)


def octagon(apothem):
    radius = apothem / math.cos(math.pi / 8)
    return [(radius * math.cos(math.pi / 8 + k * math.pi / 4), radius * math.sin(math.pi / 8 + k * math.pi / 4))
            for k in range(8)]


def ff_layout():
    """(top-plane polygons, underpass polygon, via centres) in micrometres."""
    import shapely
    w, x = FF_W, FF_X

    def ring(apothem):
        return shapely.Polygon(octagon(apothem + w / 2)).difference(shapely.Polygon(octagon(apothem - w / 2)))

    big = 4 * FF_R[0]
    top_cut = shapely.box(-x, 0., x, big)
    outer = ring(FF_R[0]).difference(top_cut).difference(shapely.box(-FF_GAP / 2, -big, FF_GAP / 2, 0.))
    inner = ring(FF_R[1]).difference(top_cut)
    r1, r2 = FF_R

    def diagonal(xa, ya, xb, yb):
        return shapely.Polygon(((xa, ya - w / 2), (xb, yb - w / 2), (xb, yb + w / 2), (xa, ya + w / 2)))

    top = shapely.union_all([outer, inner, diagonal(-x, r1, x, r2)])
    under = shapely.union_all([shapely.box(-x - 6, r2 - w / 2, -x, r2 + w / 2), diagonal(-x, r2, x, r1),
                               shapely.box(x, r1 - w / 2, x + 6, r1 + w / 2)])
    # Snap to a 1e-6 um grid: removes the 1e-14 um slivers the boolean operations leave at shared vertices.
    return shapely.set_precision(top, 1e-6), shapely.set_precision(under, 1e-6), ((-x - 3, r2), (x + 3, r1))


def octagonal_inductor():
    import shapely
    stack = mw.mom.LayerStack((mw.mom.Layer("oxide", 10 * UM, 4.1),), mw.mom.HalfSpace(4.1), mw.mom.HalfSpace())
    scene = mw.Scene(background=stack)
    top, under, vias = ff_layout()
    for polygon, z in ((top, FF_Z[1]), (under, FF_Z[0])):
        for k, part in enumerate(shapely.get_parts(polygon)):
            loops = [list(part.exterior.coords)[:-1], *[list(r.coords)[:-1] for r in part.interiors]]
            loops = [torch.tensor(loop, dtype=torch.float64) * UM for loop in loops]
            scene = scene.updated(structures=(
                *scene.structures,
                mw.Structure(mw.geometry.PlanarPEC(loops[0], holes=loops[1:], normal_axis="z",
                                                   coordinate=z * UM), mw.Material.pec(),
                             name=f"metal_{z:g}_{k}"),
            ))
    scene = scene.updated(structures=(
        *scene.structures,
        *(mw.Structure(mw.Box(
              position=torch.tensor((vx, vy, sum(FF_Z) / 2), dtype=torch.float64) * UM,
              size=torch.tensor((FF_VIA, FF_VIA, FF_Z[1] - FF_Z[0]), dtype=torch.float64) * UM), mw.Material.pec()) for vx, vy in vias),
    ))
    y = -FF_R[0] * UM
    scene = scene.updated(ports=(
        *scene.ports,
        mw.LumpedPort("diff", positive=(-FF_GAP / 2 * UM, y, FF_Z[1] * UM),
                      negative=(FF_GAP / 2 * UM, y, FF_Z[1] * UM), voltage_path=mw.AxisPath("x"),
                      reference_impedance=50.),
    ))
    return scene


def ff_options(edge, maximum=4 * UM):
    return dict(bridge_width_m=FF_W * UM, max_edge_length_m=maximum, edge_cell_m=edge, growth_ratio=1.5)


def richardson(values, ratio):
    """Three-level Richardson estimate of a mesh sequence (coarse to fine, refinement ``ratio``).

    In the asymptotic range ((q1-q2)/(q2-q3) > 1) the observed order p gives the extrapolated value and the
    uncertainty 1.25 |q_ext - q3| (the grid-convergence safety factor); otherwise no extrapolation is made and the
    uncertainty is the larger of the two steps.
    """
    q1, q2, q3 = values
    d1, d2 = q1 - q2, q2 - q3
    if d2 != 0 and d1 / d2 > 1:
        order = math.log(d1 / d2) / math.log(ratio)
        value = q3 - d2 / (ratio ** order - 1)
        return dict(value=value, order=order, uncertainty=1.25 * abs(value - q3))
    return dict(value=q3, order=None, uncertainty=max(abs(d1), abs(d2)))


def richardson_h(sizes, values):
    """Three-level Richardson estimate for arbitrary decreasing sizes h1 > h2 > h3 (q = q_inf + a h^p).

    The order p solves (q1-q2)/(q2-q3) = (h1^p-h2^p)/(h2^p-h3^p) by bisection on [0.1, 8]; outside the asymptotic
    range (no root) the estimate falls back to richardson()'s no-extrapolation rule.
    """
    (h1, h2, h3), (q1, q2, q3) = sizes, values
    d1, d2 = q1 - q2, q2 - q3
    if d2 == 0 or d1 / d2 <= 0:
        return dict(value=q3, order=None, uncertainty=max(abs(d1), abs(d2)))
    target = d1 / d2

    def ratio(p):
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p)

    lo, hi = .1, 8.
    if not (ratio(lo) - target) * (ratio(hi) - target) < 0:
        return dict(value=q3, order=None, uncertainty=max(abs(d1), abs(d2)))
    for _ in range(200):
        mid = (lo + hi) / 2
        if (ratio(lo) - target) * (ratio(mid) - target) <= 0:
            hi = mid
        else:
            lo = mid
    p = (lo + hi) / 2
    value = q3 - d2 * h3 ** p / (h2 ** p - h3 ** p)
    return dict(value=value, order=p, uncertainty=1.25 * abs(value - q3))

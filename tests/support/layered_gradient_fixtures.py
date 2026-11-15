"""Small two-plane, two-via, two-port loop for layered MoM gradient gates.

Binary-exact coordinates (u = 2**-20 m) keep plane/Box endpoint identity. The
lower plane z=64u carries strips A (x 0..32) and B (x 40..72) with port p1
across their gap; the upper plane z=128u carries C and D with port p2. Vias at
x 12..20 and 52..60 close one series loop p1 -> B -> via -> D -> p2 -> C -> via
-> A, so sheet and via loss both carry the full loop current.
"""

import torch

import witwin.maxwell as mw

U = 2.0**-20
ZA, ZB = 64 * U, 128 * U
SIGMA = 5.8e7
FREQUENCIES = (1e9, 1e10)


def vacuum():
    return mw.mom.LayerStack((mw.mom.Layer("air", 1e-3, 1.0),), mw.mom.HalfSpace(), mw.mom.HalfSpace())


def loop_scene(*, sheet_sigma=SIGMA, thickness=2 * U, rms_height=None, via_sigma=SIGMA,
               sphere_radius=None, z0=(50.0, 50.0)):
    """Every argument may be a trainable tensor; None roughness is a smooth conductor."""
    sheet_roughness = None if rms_height is None else mw.media.HammerstadRoughness(rms_height=rms_height)
    via_roughness = None if sphere_radius is None else mw.media.HurayRoughness(
        sphere_radius=sphere_radius, sphere_density=1e11)
    sheet = mw.media.LossyMetalMedium(conductivity=sheet_sigma, roughness=sheet_roughness)
    via = mw.media.LossyMetalMedium(conductivity=via_sigma, roughness=via_roughness)
    scene = mw.Scene(background=vacuum())
    for x0, x1 in ((0, 32), (40, 72)):
        scene = scene.updated(structures=(
            *scene.structures,
            *(mw.Structure(mw.geometry.PlanarPEC(
                  [(x0 * U, -8 * U), (x1 * U, -8 * U), (x1 * U, 8 * U), (x0 * U, 8 * U)],
                  normal_axis="z", coordinate=z, physical_thickness=thickness), sheet) for z in (ZA, ZB)),
        ))
    scene = scene.updated(structures=(
        *scene.structures,
        *(mw.Structure(mw.Box(
              position=torch.tensor((x * U, 0., (ZA + ZB) / 2), dtype=torch.float64),
              size=torch.tensor((8 * U, 8 * U, ZB - ZA), dtype=torch.float64)), via) for x in (16, 56)),
    ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(mw.LumpedPort(name, positive=(40 * U, 0., z), negative=(32 * U, 0., z),
                        voltage_path=mw.AxisPath("x"), reference_impedance=reference) for name, z, reference in (("p1", ZA, z0[0]), ("p2", ZB, z0[1]))),
    ))
    return scene


def loop_config(model="sheet", max_edge=16 * U):
    return mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(
        bridge_width_m=8 * U, max_edge_length_m=max_edge, conductor_model=model))

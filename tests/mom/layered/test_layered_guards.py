"""Negative controls for the layered MoM capability boundaries (guard census evidence)."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.mesh import mesh_conductors
from witwin.maxwell.mom.layered.network import prepare_layered_network
from witwin.maxwell.mom.layered.runtime import LayeredSurfaceData
from tests.mom.layered.test_layered_network import gap_port

Z = 1e-4


def vacuum():
    return mw.LayerStack((mw.Layer("air", 1e-3),), mw.HalfSpace(), mw.HalfSpace())


def config():
    return mw.MoMConfig(layered=mw.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5))


def sheet(z=Z, normal_axis="z"):
    return mw.Structure(mw.PlanarPEC([(0., 0.), (1e-4, 0.), (1e-4, 2e-5), (0., 2e-5)],
                                     normal_axis=normal_axis, coordinate=z), mw.Material.pec())


def layered_scene(*structures, port=True):
    scene = mw.Scene(background=vacuum(), structures=list(structures))
    if port:
        scene.add_port(gap_port("p", (6e-5, 1e-5, Z), (4e-5, 1e-5, Z), "x", 1e-5))
    return scene


def test_public_layered_scene_capability_guards():
    with pytest.raises(NotImplementedError, match="normal_axis='z' only"):
        mw.Simulation.mom(layered_scene(sheet(normal_axis="x")), frequency=1e9, config=config()).prepare()
    rotated = mw.Structure(mw.Box(position=(5e-5, 1e-5, Z), size=(4e-6, 4e-6, 4e-6), rotation=(0., 0., 0.3)),
                           mw.Material.pec())
    with pytest.raises(NotImplementedError, match="axis-aligned Box"):
        mw.Simulation.mom(layered_scene(sheet(), rotated), frequency=1e9, config=config()).prepare()
    sphere = mw.Structure(mw.Sphere(position=(5e-5, 1e-5, Z), radius=2e-6), mw.Material.pec())
    with pytest.raises(NotImplementedError, match="PlanarPEC sheets and Box vias only"):
        mw.Simulation.mom(layered_scene(sheet(), sphere), frequency=1e9, config=config()).prepare()
    with pytest.raises(NotImplementedError, match="LumpedPort terminals only"):
        mw.Simulation.mom(layered_scene(sheet(), port=False), frequency=1e9, config=config()).prepare()
    trainable = mw.Structure(mw.Box(position=torch.tensor((5e-5, 1e-5, 1.5e-4), requires_grad=True),
                                    size=(4e-6, 4e-6, 1e-4)), mw.Material.pec())
    with pytest.raises(NotImplementedError, match="Layered MoM geometry/material derivatives"):
        mw.Simulation.mom(layered_scene(sheet(), sheet(2e-4), trainable), frequency=1e9,
                          config=config()).prepare()
    with pytest.raises(NotImplementedError, match="snapshot persistence"):
        LayeredSurfaceData(None, None, (1e9,), ("p",)).snapshot()


def test_planar_pec_requires_fdtd_or_layered_mom():
    scene = mw.Scene(structures=[sheet()])
    with pytest.raises(NotImplementedError, match="PlanarPEC requires the FDTD sheet runtime or MoM"):
        mw.Simulation.fdfd(scene, frequency=1e9).prepare()


@pytest.mark.cuda
def test_junction_and_vertical_port_attachment_guards():
    z0, z1, z2 = 1e-4, 2e-4, 3e-4
    with pytest.raises(NotImplementedError, match="three or more faces"):
        mesh_conductors([(0., 1e-4, 0., 1e-4, z0)], torch.tensor((z0, z1), device="cuda", dtype=torch.float64), 2e-5,
                        bridges=[(0, 5e-5, 0., 1e-4, z0, z1, (z0+z1)/2, 5e-5, 1.)])
    stack = mw.LayerStack((mw.Layer("air", 1e-3),), mw.HalfSpace(), mw.HalfSpace())
    plates = [(0., 1e-4, 0., 2e-5, z) for z in (z0, z1, z2)]
    skipped = gap_port("p", (0., 1e-5, z2), (0., 1e-5, z0), "z", 1e-5)
    with pytest.raises(NotImplementedError, match="consecutive metal planes or a PEC ground"):
        prepare_layered_network(stack, plates, [skipped], (1e9,), bridge_width=1e-5, max_edge=2e-5)
    interior = gap_port("p", (5e-5, 1e-5, z1), (5e-5, 1e-5, z0), "z", 1e-5)
    with pytest.raises(NotImplementedError, match="requires a sheet boundary"):
        prepare_layered_network(stack, plates[:2], [interior], (1e9,), bridge_width=1e-5, max_edge=2e-5)

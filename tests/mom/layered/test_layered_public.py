"""Public Scene -> Simulation -> Result path for PlanarPEC sheets in a LayerStack background."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.network import solve_layered_network
from tests.mom.layered.test_layered_network import gap_port, square_loop


pytestmark = pytest.mark.cuda
SIDE, WIDTH, Z = 1e-3, 20e-6, .5e-3


def vacuum():
    return mw.LayerStack((mw.Layer("air", 1e-3, 1.0),), mw.HalfSpace(), mw.HalfSpace())


def c_shaped_loop():
    """The square loop of test_layered_network as one Manhattan polygon with a bottom gap."""
    edge, inner, gap = SIDE / 2 + WIDTH / 2, SIDE / 2 - WIDTH / 2, WIDTH / 2
    vertices = [(gap, -edge), (edge, -edge), (edge, edge), (-edge, edge), (-edge, -edge), (-gap, -edge),
                (-gap, -inner), (-inner, -inner), (-inner, inner), (inner, inner), (inner, -inner), (gap, -inner)]
    return mw.Structure(mw.PlanarPEC(vertices, normal_axis="z", coordinate=Z), mw.Material.pec())


def loop_port():
    return gap_port("loop", (WIDTH / 2, -SIDE / 2, Z), (-WIDTH / 2, -SIDE / 2, Z), "x", WIDTH)


def config():
    # Coarse: this test checks equality with the internal solver, not accuracy.
    return mw.MoMConfig(layered=mw.LayeredSurfaceConfig(bridge_width_m=WIDTH, max_edge_length_m=40e-6,
                                                         edge_cell_m=10e-6))


def test_public_layered_network_matches_the_internal_solver():
    scene = mw.Scene(background=vacuum(), structures=[c_shaped_loop()])
    scene.add_port(loop_port())
    frequencies = (1e9, 2e9)
    result = mw.Simulation.mom(scene, frequencies=frequencies, config=config()).run()
    reference = solve_layered_network(vacuum(), square_loop(SIDE, WIDTH, WIDTH, Z), [loop_port()], frequencies,
                                      bridge_width=WIDTH, max_edge=40e-6, edge_cell=10e-6)
    assert result.method == "mom" and result.frequencies == frequencies
    assert torch.equal(result.network.s, reference.s)
    assert result.network.port_names == ("loop",)
    currents = result.surface_currents.currents
    assert currents.shape == (2, reference.metadata["unknowns"], 1)
    # exp(-i*omega*t): the loop is inductive, Im(Z11) < 0 at both frequencies.
    assert (result.network.to_z()[:, 0, 0].imag < 0).all()


def test_public_layered_guards():
    scene = mw.Scene(background=vacuum(), structures=[c_shaped_loop()])
    scene.add_port(loop_port())
    with pytest.raises(ValueError, match="LayerStack background requires MoMConfig"):
        mw.Simulation.mom(scene, frequencies=(1e9,))
    with pytest.raises(ValueError, match="LayerStack background requires MoMConfig"):
        mw.Simulation.mom(mw.Scene(), frequency=1e9, config=config())
    with pytest.raises(NotImplementedError, match="LayerStack background requires Simulation.mom"):
        mw.Simulation.fdtd(scene, frequencies=(1e9,))
    with pytest.raises(ValueError, match="distinct"):
        mw.Simulation.mom(scene, frequencies=(1e9, 1e9), config=config())
    with pytest.raises(TypeError, match="LayerStack"):
        mw.Scene(background=(mw.Layer("air", 1e-3),))

    slanted = mw.Scene(background=vacuum(), structures=[mw.Structure(
        mw.PlanarPEC([(0., 0.), (1e-4, 0.), (0., 1e-4)], normal_axis="z", coordinate=Z), mw.Material.pec())])
    slanted.add_port(loop_port())
    with pytest.raises(NotImplementedError, match="Manhattan"):
        mw.Simulation.mom(slanted, frequency=1e9, config=config()).prepare()
    driven = mw.Scene(background=vacuum(), structures=[c_shaped_loop()])
    driven.add_port(loop_port())
    driven.add_source(mw.PointDipole(position=(0., 0., Z), polarization=(1., 0., 0.)))
    with pytest.raises(NotImplementedError, match="LumpedPorts only"):
        mw.Simulation.mom(driven, frequency=1e9, config=config()).prepare()
    with pytest.raises(ValueError, match="bridge_width_m"):
        mw.LayeredSurfaceConfig(bridge_width_m=0., max_edge_length_m=1e-6)

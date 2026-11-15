"""R4-3 thick-terminal ports: closed-conductor edge ports as PEC blocks over the full terminal cross-section.

A closed delta-gap port bridges its gap with a zero-thickness top-face strip; on the G3 strip loop
(10 um wide, 20 um thick, 10 um gap in the return) that bridge adds +1.10% L against the true-corner
lattice reference, whose port is a cut-plane delta gap in the uncut ring. A closed edge port
(``voltage_aperture``) fills the gap with an ideal PEC block of the terminal thickness, welded to
both end faces by the union mesher, and drives its mid cross-section ring: the reference geometry.
"""
import gc
import json
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.network import prepare_layered_network, solve_prepared_network
from tests.support.rfic_closed_references import SIGMA, UM, network_metrics, strip_fixture

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]

# G3 true-geometry lattice (cell-centred PEEC, exact corners, cut-plane delta gap), PEC, 1 GHz:
# t = 20 um multipole order 6 (direct vs multipole differed by 0.13% at t = 2), t = 2 um direct.
LATTICE_L = {20.: 0.265744e-9, 2.: 0.383802e-9}


@pytest.fixture(autouse=True)
def budget(record_property):
    """Memory cap per test; wall time is recorded only."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    record_property("seconds", time.perf_counter() - start)
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())
    assert torch.cuda.max_memory_allocated() < 6 * 2**30


def edge_strip(**options):
    """strip_fixture with its delta-gap port declared as the full-width edge port over the return gap."""
    scene, config = strip_fixture(**options)
    edge = mw.Scene(background=scene.background)
    edge = edge.updated(structures=(*edge.structures, *scene.structures))
    port = scene.ports[0]
    edge = edge.updated(ports=(
        *edge.ports,
        mw.LumpedPort("p", positive=port.positive, negative=port.negative, voltage_path=mw.AxisPath("x"),
                      reference_impedance=50., voltage_aperture=mw.Box(
                          position=(100 * UM, -110 * UM, 100 * UM), size=(10 * UM, 10 * UM, 0.))),
    ))
    return edge, config


@pytest.mark.parametrize("thickness", [20., 2.])
def test_closed_thick_port_true_corner_reference(thickness, record_property):
    """Closed PEC strip loop, edge port, default 20/2 um mesh: L within 1% of the lattice reference."""
    scene, config = edge_strip(model="closed", thickness_um=thickness, lossy=False)
    simulation = mw.Simulation.mom(scene, frequencies=(1e9,), config=config).prepare()
    result = simulation.run()
    metadata = result.network.metadata
    assert metadata["port_attachment"] == "pec_block_end_faces"
    assert metadata["ports"][0]["bridge"] == "ideal_pec_block"
    (block,) = metadata["port_blocks"]
    # Gap x 95..105, return width y -115..-105, full terminal thickness about the 100 um mid-plane.
    expected = (95., 105., -115., -105., 100. - thickness / 2, 100. + thickness / 2)
    assert all(abs(v / UM - e) < 1e-9 for v, e in zip(block, expected))
    assert result.solver.prepared.loss is None
    assert abs(result.network.s.abs().item() - 1) < 1e-8
    inductance = network_metrics(result)["L_H"][0]
    error = inductance / LATTICE_L[thickness] - 1
    record_property("L_H", inductance)
    record_property("relative_to_lattice", error)
    assert abs(error) < 1e-2
    again, currents = solve_prepared_network(simulation.solver.prepared)
    assert torch.equal(again.s, result.network.s)
    assert torch.equal(currents, mw.mom.surface_currents(result).currents)


def test_closed_thick_port_removes_top_face_bridge_excess(record_property):
    """Same 20 um loop and mesh: the top-face delta-gap bridge is >= 1% above the lattice, the block is not."""
    values = {}
    for kind, (scene, config) in (("delta", strip_fixture(model="closed", thickness_um=20., lossy=False)),
                                  ("edge", edge_strip(model="closed", thickness_um=20., lossy=False))):
        result = mw.Simulation.mom(scene, frequencies=(1e9,), config=config).run()
        values[kind] = network_metrics(result)["L_H"][0] / LATTICE_L[20.] - 1
        del result
        gc.collect()
        torch.cuda.empty_cache()
    record_property("relative_to_lattice", json.dumps(values))
    assert values["delta"] > 1e-2 > abs(values["edge"])


def test_surface_admittance_thick_edge_port_dc_resistance(record_property):
    """surface_admittance 20 um copper strip: edge port R = l/(sigma A) at DC and torch.equal to its block delta gap."""
    frequencies = (1.0, 1e9)
    scene, config = edge_strip(model="surface_admittance", thickness_um=20., maximum_um=40., edge_um=4.)
    result = mw.Simulation.mom(scene, frequencies=frequencies, config=config).run()
    assert result.network.metadata["ports"][0]["bridge"] == "ideal_pec_block"
    metrics = network_metrics(result)
    expected = 200e-6 / (SIGMA * 10e-6 * 20e-6)
    record_property("metrics", json.dumps(metrics))
    record_property("dc_relative", metrics["R_ohm"][0] / expected - 1)
    assert abs(metrics["R_ohm"][0] / expected - 1) < 1e-6
    assert torch.linalg.svdvals(result.network.s).amax().item() <= 1 + 1e-12
    scene, config = strip_fixture(model="surface_admittance", thickness_um=20., maximum_um=40., edge_um=4.)
    delta = mw.Simulation.mom(scene, frequencies=frequencies, config=config).run()
    assert torch.equal(result.network.s, delta.network.s)


def test_closed_mixed_delta_gap_and_thick_edge_ports(record_property):
    """Copper coplanar strips: a top-face delta gap at one end, a thick edge block at the other; reciprocal, passive."""
    stack = mw.mom.LayerStack((mw.mom.Layer("air", 1e-3),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    z = 100 * UM
    rectangles = [(0., 100 * UM, 5 * UM, 15 * UM, z), (0., 100 * UM, -15 * UM, -5 * UM, z)]
    ports = [mw.LumpedPort("p1", positive=(1 * UM, 5 * UM, z), negative=(1 * UM, -5 * UM, z),
                           voltage_path=mw.AxisPath("y")),
             mw.LumpedPort("p2", positive=(95 * UM, 5 * UM, z), negative=(95 * UM, -5 * UM, z),
                           voltage_path=mw.AxisPath("y"),
                           voltage_aperture=mw.Box(position=(95 * UM, 0., z), size=(10 * UM, 10 * UM, 0.)))]
    first = None
    for _ in range(2):
        prepared = prepare_layered_network(stack, rectangles, ports, (1e9, 10e9), bridge_width=2 * UM,
                                           max_edge=25 * UM, sheet_materials=[(SIGMA, 2 * UM, None)] * 2,
                                           conductor_model="closed")
        assert prepared.metadata["port_attachment"] == "top_face_and_pec_block_end_faces"
        assert [r["bridge"] for r in prepared.metadata["ports"]] == ["ideal_lossless", "ideal_pec_block"]
        assert len(prepared.metadata["conductor_faces"]) == 2
        network, current = solve_prepared_network(prepared)
        y = network.to_y()
        reciprocity = ((y - y.transpose(-1, -2)).abs().max() / y.abs().max()).item()
        record_property("reciprocity", reciprocity)
        assert reciprocity < 1e-10
        assert torch.linalg.matrix_norm(network.s, ord=2).max() <= 1 + 1e-9
        assert torch.linalg.eigvalsh(y.real).min() > 0
        if first is None:
            first = (network.s.clone(), current.clone())
        else:
            assert torch.equal(first[0], network.s) and torch.equal(first[1], current)
        del prepared, network, current, y
        gc.collect()
        torch.cuda.empty_cache()


def test_closed_thick_port_gradient_impedance_map():
    """Trainable copper conductivity: port-block faces are ideal in the adjoint group map and in the forward loss."""
    scene, config = strip_fixture(model="closed", thickness_um=20., maximum_um=40., edge_um=4.)
    sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
    trainable = mw.Scene(background=scene.background)
    for structure in scene.structures:
        material = structure.material
        if isinstance(material, mw.media.LossyMetalMedium):
            material = mw.media.LossyMetalMedium(conductivity=sigma)
        trainable = trainable.updated(structures=(*trainable.structures, mw.Structure(structure.geometry, material)))
    edge, _ = edge_strip(model="closed", thickness_um=20., maximum_um=40., edge_um=4.)
    trainable = trainable.updated(ports=(*trainable.ports, edge.ports[0]))
    runtime = mw.Simulation.mom(trainable, frequencies=(1e9, 1e10), config=config).prepare().solver
    prepared = runtime.prepared
    _, impedance = runtime.gradients.loss_adjoint(prepared)
    groups = runtime.gradients.face_groups(prepared)
    block = prepared.layout.face_owners >= 5
    assert bool(block.any()) and bool((groups[block] == -1).all())
    padded = torch.cat((impedance.new_zeros(2, 1), impedance.detach()), 1)
    faces = padded[:, groups + 1]
    reference = prepared.loss.impedance
    assert torch.equal(faces == 0, reference == 0)
    assert ((faces - reference).abs().max() <= 1e-14 * reference.abs().max()).item()

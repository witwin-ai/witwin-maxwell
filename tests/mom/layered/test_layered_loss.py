"""R4 GPU gates for finite-thickness conductor loss on layered sheets."""

import math
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.loss import MU0, prepare_sheet_loss, sheet_impedance
from witwin.maxwell.mom.layered.mesh import mesh_rectangles
from witwin.maxwell.mom.layered.assembly import assemble_layered
from witwin.maxwell.mom.layered.network import (
    prepare_layered_network, solve_dense, solve_layered_network, solve_prepared_network)
from tests.mom.layered.test_layered_network import coplanar_strips, rfic_stack, square_loop
from tests.support.layered_network_references import square_loop_inductance_uniform
from tests.mom.layered.test_layered_public import SIDE, WIDTH, Z, c_shaped_loop, config, loop_port, vacuum


pytestmark = pytest.mark.cuda
SIGMA = 5.8e7


@pytest.fixture
def metrics(request):
    return lambda name, value: request.node.user_properties.append((name, value))


@pytest.fixture(autouse=True)
def gpu_budget(metrics):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_reserved() / 2**30
    metrics("gpu_seconds", elapsed)
    metrics("peak_reserved_GiB", peak)
    assert elapsed < 120
    assert peak < 12


@pytest.mark.parametrize("roughness", [None, mw.HammerstadRoughness(rms_height=.4e-6)])
def test_sheet_impedance_limits(roughness):
    thickness = .2e-6
    delta = torch.tensor([1e3 * thickness, thickness / 50, thickness / 1000],
                         device="cuda", dtype=torch.float64)
    frequencies = 2 / (MU0 * SIGMA * delta.square()) / (2 * math.pi)
    smooth = sheet_impedance(frequencies, SIGMA, thickness)
    # At delta=1000*t the real DC error is O((t/delta)^4); the nonzero
    # O((t/delta)^2) imaginary part is the physical internal inductance.
    dc = 1 / (SIGMA * thickness)
    assert abs(smooth[0].real / dc - 1) < 1e-9
    assert abs(smooth[0].imag / dc + 1 / (6e6)) < 1e-15
    impedance = sheet_impedance(frequencies, SIGMA, thickness, roughness)
    material = mw.LossyMetalMedium(conductivity=SIGMA, roughness=roughness)
    expected = torch.tensor([material.surface_impedance(2 * math.pi * f) / 2
                             for f in frequencies[1:].tolist()], device="cuda", dtype=torch.complex128)
    assert ((impedance[1:] / expected - 1).abs() < 1e-9).all()
    if roughness is not None:
        assert torch.allclose(impedance, smooth * roughness.impedance_factor(delta), rtol=2e-15, atol=0.)


def test_low_frequency_loop_resistance(metrics):
    frequency, thickness = 100e6, .2e-6
    rectangles = square_loop(SIDE, WIDTH, WIDTH, Z)
    prepared = prepare_layered_network(vacuum(), rectangles, [loop_port()], (frequency,),
                                        bridge_width=WIDTH, max_edge=20e-6, edge_cell=5e-6,
                                        sheet_materials=[(SIGMA, thickness, None)] * len(rectangles))
    # The same Green operator serves both materials: assemble it once to keep
    # the two solves inside the per-test time budget, even with a cold build.
    operator = assemble_layered(prepared.layout, prepared.table)
    reference_currents, _ = solve_dense(operator, prepared.excitation)
    lossless = (1 / (prepared.excitation.T @ reference_currents))[0, 0].item()
    prepared.loss.add_to(operator, 0)
    currents, _ = solve_dense(operator, prepared.excitation)
    impedance = (1 / (prepared.excitation.T @ currents))[0, 0].item()
    # Each side runs between adjacent strip centreline intersections: four
    # lengths SIDE, minus the ideal bridge gap WIDTH. Count each corner once;
    # the small 2-D corner-crowding correction is part of the 3% gate.
    path_length = 4 * SIDE - WIDTH
    dc = path_length / (SIGMA * WIDTH * thickness)
    inductance = -impedance.imag / (2 * math.pi * frequency)
    reference = -lossless.imag / (2 * math.pi * frequency)
    metrics("resistance_ohm", impedance.real)
    metrics("dc_resistance_ohm", dc)
    metrics("inductance_H", inductance)
    metrics("lossless_inductance_H", reference)
    local = (currents[:, 0] / (prepared.excitation.T @ currents)[0, 0])[prepared.layout.basis.face_edges]
    norm = torch.einsum("ti,tij,tj->t", local.conj(),
                        prepared.layout.basis.mass_blocks().to(torch.complex128), local).real
    internal = -(norm * prepared.loss.impedance[0].imag).sum().item() / (2 * math.pi * frequency)
    metrics("sheet_internal_inductance_H", internal)
    assert abs(impedance.real / dc - 1) < .03
    # R = 17 ohm >> omega*L = 2.3 ohm spreads the current uniformly across the strip, so the
    # inductance moves from the PEC edge-current value toward the uniform-current (GMD) value.
    uniform = square_loop_inductance_uniform(SIDE, WIDTH, thickness)
    metrics("uniform_current_inductance_H", uniform)
    assert reference < inductance
    assert abs(inductance / uniform - 1) < .01


def test_loop_skin_regime(metrics):
    # The 4 mm loop perimeter stays below lambda/9 here; at 32 GHz it nears anti-resonance
    # and the port resistance no longer measures the conductor loss.
    frequencies = (2e9, 8e9)
    rectangles = square_loop(SIDE, WIDTH, WIDTH, Z)
    prepared = prepare_layered_network(vacuum(), rectangles, [loop_port()], frequencies,
                                        bridge_width=WIDTH, max_edge=40e-6, edge_cell=5e-6,
                                        sheet_materials=[(SIGMA, 3e-6, None)] * len(rectangles))
    network, currents = solve_prepared_network(prepared)
    impedance = network.to_z()[:, 0, 0]
    gram = prepared.layout.basis.mass_blocks().to(torch.complex128)
    for index, frequency in enumerate(frequencies):
        # Normalize to unit port current and split Joule loss from radiation.
        port_current = (prepared.excitation.T @ currents[index])[0, 0]
        local = (currents[index, :, 0] / port_current)[prepared.layout.basis.face_edges]
        norm = torch.einsum("ti,tij,tj->t", local.conj(), gram, local).real
        conductor = (norm * prepared.loss.impedance[index].real).sum().item()
        metrics(f"R_conductor_{frequency / 1e9:g}GHz_ohm", conductor)
        metrics(f"R_radiation_{frequency / 1e9:g}GHz_ohm", impedance[index].real.item() - conductor)
    resistance = impedance.real
    quality = -impedance.imag / resistance
    ratio = (resistance[1] / resistance[0]).item()
    for f, z, q in zip(frequencies, impedance.tolist(), quality.tolist()):
        metrics(f"Z_{f / 1e9:g}GHz_ohm", str(z))
        metrics(f"Q_{f / 1e9:g}GHz", q)
    metrics("R8_over_R2", ratio)
    assert (resistance.diff() > 0).all()
    assert abs(ratio / 2 - 1) < .15


@pytest.fixture(scope="module")
def lossy_coupled_network():
    stack = rfic_stack()
    rectangles, ports = coplanar_strips(100e-6, 5e-6, 5e-6, 2e-6, stack.interfaces()[5])
    def run():
        return solve_layered_network(stack, rectangles, ports, (1e9, 50e9), bridge_width=2e-6,
                                     max_edge=5e-6, edge_cell=1.25e-6,
                                     sheet_materials=[(SIGMA, .5e-6, None)] * len(rectangles))
    return run(), run


def test_lossy_network_physics(lossy_coupled_network, metrics):
    network, _ = lossy_coupled_network
    y = network.to_y()
    symmetry = (torch.linalg.matrix_norm(y - y.transpose(-1, -2)) / torch.linalg.matrix_norm(y)).max().item()
    conductance = torch.linalg.eigvalsh(y.real).min().item()
    singular = torch.linalg.matrix_norm(network.s, ord=2).max().item()
    metrics("relative_Y_asymmetry", symmetry)
    metrics("min_ReY_eigenvalue_S", conductance)
    metrics("max_S_singular_value", singular)
    assert symmetry < 1e-10
    assert conductance > 0
    assert singular <= 1


def test_lossy_determinism(lossy_coupled_network):
    network, run = lossy_coupled_network
    for _ in range(2):
        again = run()
        assert torch.equal(network.s, again.s)
        assert torch.equal(network.to_y(), again.to_y())


def lossy_loop(thickness, material):
    geometry = c_shaped_loop().geometry
    return mw.Structure(mw.PlanarPEC(geometry.vertices, normal_axis="z", coordinate=Z,
                                     physical_thickness=thickness), material)


def test_public_lossy_loop_matches_internal():
    roughness = mw.HammerstadRoughness(rms_height=.4e-6)
    thickness = .2e-6
    material = mw.LossyMetalMedium(conductivity=SIGMA, roughness=roughness)
    scene = mw.Scene(background=vacuum(), structures=[lossy_loop(thickness, material)])
    scene.add_port(loop_port())
    frequencies = (1e9, 2e9)
    result = mw.Simulation.mom(scene, frequencies=frequencies, config=config()).run()
    rectangles = square_loop(SIDE, WIDTH, WIDTH, Z)
    reference = solve_layered_network(vacuum(), rectangles, [loop_port()], frequencies,
                                      bridge_width=WIDTH, max_edge=40e-6, edge_cell=10e-6,
                                      sheet_materials=[(SIGMA, thickness, roughness)] * len(rectangles))
    assert torch.equal(result.network.s, reference.s)
    assert torch.equal(result.network.to_y(), reference.to_y())
    sheets = result.metadata["sheets"]
    assert len(sheets) == 1
    assert sheets[0]["sigma"] == SIGMA and sheets[0]["thickness"] == thickness
    assert sheets[0]["roughness"] == {"model": "HammerstadRoughness", "rms_height": .4e-6,
                                     "roughness_factor": 2.0}
    assert result.metadata["port_bridges"] == "ideal_lossless"


@pytest.mark.parametrize("thickness,material,message", [
    (0., mw.LossyMetalMedium(conductivity=SIGMA), "physical_thickness > 0"),
    (.2e-6, mw.Material(eps_r=2.), r"Material.pec\(\) or LossyMetalMedium"),
])
def test_public_lossy_material_guards(thickness, material, message):
    scene = mw.Scene(background=vacuum(), structures=[lossy_loop(thickness, material)])
    scene.add_port(loop_port())
    with pytest.raises(ValueError, match=message):
        mw.Simulation.mom(scene, frequency=1e9, config=config()).prepare()


def test_gram_mixed_materials_and_ideal_bridge():
    # Two physical regions and a middle ideal bridge. Independent degree-two
    # barycentric integration checks every global entry, including shared edges.
    rectangles = [(0., 1., 0., 1., 0.), (2., 3., 0., 1., 0.)]
    layout = mesh_rectangles(rectangles + [(1., 2., 0., 1., 0.)],
                             torch.tensor([0.], device="cuda", dtype=torch.float64), .5)
    frequencies = torch.tensor([2e9], device="cuda", dtype=torch.float64)
    roughness = mw.HammerstadRoughness(rms_height=.4e-6)
    materials = [(SIGMA, .2e-6, None), (SIGMA / 2, 3e-6, roughness)]
    loss = prepare_sheet_loss(layout, rectangles, materials, frequencies)
    centres = layout.mesh.centroids
    assert (loss.impedance[:, (centres[:, 0] > 1) & (centres[:, 0] < 2)] == 0).all()
    basis, mesh = layout.basis, layout.mesh
    bary = mesh.vertices.new_tensor([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    points = torch.einsum("qv,tvd->tqd", bary, mesh.triangles)
    values = basis.local_values(points)
    expected = torch.zeros((basis.n_edges, basis.n_edges), device="cuda", dtype=torch.complex128)
    for face in range(len(mesh.faces)):
        for i in range(3):
            for j in range(3):
                expected[basis.face_edges[face, i], basis.face_edges[face, j]] += (
                    (values[face, :, i] * values[face, :, j]).sum() * mesh.areas[face] / 3
                    * loss.impedance[0, face])
    actual = torch.zeros_like(expected)
    loss.add_to(actual, 0)
    assert torch.allclose(actual, expected, rtol=2e-14, atol=1e-17)
    for _ in range(2):
        again = torch.zeros_like(expected)
        loss.add_to(again, 0)
        assert torch.equal(actual, again)


@pytest.mark.parametrize("other", [(SIGMA / 2, .2e-6, None), (SIGMA, .3e-6, None),
                                    (None, 0., None),
                                    (SIGMA, .2e-6, mw.HammerstadRoughness(rms_height=.4e-6))])
def test_overlapping_materials_raise(other):
    rectangles = [(0., 1., 0., 1., 0.), (.5, 1.5, 0., 1., 0.)]
    layout = mesh_rectangles(rectangles, torch.tensor([0.], device="cuda", dtype=torch.float64), .5)
    with pytest.raises(ValueError, match="identical materials and thicknesses"):
        prepare_sheet_loss(layout, rectangles, [(SIGMA, .2e-6, None), other],
                           torch.tensor([1e9], device="cuda", dtype=torch.float64))

"""E1 gates: causal frequency-dependent layer permittivity in the layered MoM.

Model gates run on the host. CUDA gates compare a multi-frequency dispersive table and
network with per-frequency tables/networks of the frozen constant permittivity eps(f):
a kernel that read the wrong permittivity row would show an O(delta eps) difference.
"""

import cmath
import dataclasses
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.green import LayeredGreenTable
from witwin.maxwell.mom.layered.network import prepare_layered_network, solve_prepared_network

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
FR4 = dict(eps_r=4.0, tan_delta=0.02, frequency=1e9)


def wideband_reference(eps_inf, delta_eps, f1, f2, f):
    """Djordjevic-Sarkar in its published exp(+j w t) log10 form, conjugated to exp(-i w t)."""
    w, w1, w2 = (2 * math.pi * v for v in (f, f1, f2))
    value = eps_inf + delta_eps / (math.log10(w2) - math.log10(w1)) * cmath.log10((w2 + 1j * w) / (w1 + 1j * w))
    return value.conjugate()


def kramers_kronig_real(model, f0, decades=6, samples=400001):
    """eps'(f0) - eps_inf from eps'' alone (subtracted principal-value form, log grid)."""
    u = np.linspace(math.log(model.f1) - decades * math.log(10), math.log(model.f2) + decades * math.log(10), samples)
    f = np.exp(u)
    loss = np.array([model.relative_permittivity(x).imag for x in f])
    loss0 = model.relative_permittivity(f0).imag
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (f * loss - f0 * loss0) / (f * f - f0 * f0) * f
    singular = ~np.isfinite(g)
    g[singular] = np.interp(u[singular], u[~singular], g[~singular])
    return 2 / math.pi * np.trapezoid(g, u)


def test_wideband_debye_closed_form_and_kramers_kronig(record_property):
    model = mw.media.WidebandDebye.from_loss_tangent(**FR4)
    match = model.relative_permittivity(1e9)
    assert abs(match.real - 4.0) <= 1e-14 and abs(match.imag / match.real - 0.02) <= 1e-14
    worst_closed, worst_kk = 0.0, 0.0
    for f in np.logspace(7, 11.5, 10):
        value = model.relative_permittivity(f)
        assert value.imag > 0
        reference = wideband_reference(model.eps_inf, model.delta_eps, model.f1, model.f2, f)
        worst_closed = max(worst_closed, abs(value - reference) / abs(reference))
        recovered = kramers_kronig_real(model, f)
        worst_kk = max(worst_kk, abs(recovered - (value.real - model.eps_inf)) / model.delta_eps)
    record_property("closed_form_relative", float(worst_closed))
    record_property("kramers_kronig_relative", float(worst_kk))
    assert worst_closed <= 1e-14
    assert worst_kk <= 1e-3
    assert model.relative_permittivity(1e-3).real == pytest.approx(model.eps_inf + model.delta_eps, rel=1e-9)


def test_pole_permittivity_matches_material_poles():
    poles = (mw.DebyePole(delta_eps=0.5, tau=1e-11), mw.LorentzPole(delta_eps=0.3, resonance_frequency=2e11, gamma=1e10))
    material = mw.Material(eps_r=3.0, debye_poles=poles[:1], lorentz_poles=poles[1:])
    model = mw.media.PolePermittivity.from_material(material)
    for f in (1e8, 1e10, 3e11):
        assert model.relative_permittivity(f) == material.relative_permittivity(f)
        assert model.relative_permittivity(f).imag > 0


def test_nonphysical_permittivity_models_are_refused():
    with pytest.raises(ValueError, match="gain"):
        mw.media.WidebandDebye(eps_inf=3., delta_eps=-.1, f1=1e3, f2=1e12)
    with pytest.raises(ValueError, match="eps_inf"):
        mw.media.WidebandDebye(eps_inf=.5, delta_eps=.1, f1=1e3, f2=1e12)
    with pytest.raises(ValueError, match="0 < f1 < f2"):
        mw.media.WidebandDebye(eps_inf=3., delta_eps=.1, f1=1e12, f2=1e3)
    with pytest.raises(ValueError, match="finite"):
        mw.media.WidebandDebye(eps_inf=3., delta_eps=math.inf, f1=1e3, f2=1e12)
    with pytest.raises(ValueError, match="eps_inf"):
        mw.media.WidebandDebye.from_loss_tangent(1.2, 0.5, 1e9)
    with pytest.raises(NotImplementedError, match="Trainable"):
        mw.media.WidebandDebye(eps_inf=torch.tensor(3., requires_grad=True), delta_eps=.1, f1=1e3, f2=1e12)
    with pytest.warns(UserWarning, match="gain"):
        gain = mw.LorentzPole(delta_eps=-.1, resonance_frequency=1e11, gamma=1e9, allow_gain=True)
    with pytest.raises(ValueError, match="gain"):
        mw.media.PolePermittivity(3., (gain,))
    with pytest.raises(NotImplementedError, match="Trainable"):
        mw.media.PolePermittivity(3., (mw.DebyePole(delta_eps=torch.tensor(.1, requires_grad=True), tau=1e-11),))
    custom = mw.media.CustomDebyePole(delta_eps=torch.ones(2, 2, 2), tau=1e-11)
    with pytest.raises(TypeError, match="uniform"):
        mw.media.PolePermittivity(3., (custom,))
    with pytest.raises(ValueError, match="sigma_e"):
        mw.media.PolePermittivity.from_material(mw.Material(eps_r=3., sigma_e=1.))
    with pytest.raises(ValueError, match="magnetic poles"):
        mw.media.PolePermittivity.from_material(mw.Material(eps_r=3., mu_debye_poles=(mw.DebyePole(.1, 1e-11),)))


def test_dispersive_scene_signature_tracks_the_model():
    from witwin.maxwell.mom.layered.runtime import layered_scene_signature
    first = LayerStack((Layer("sub", 1e-4, mw.media.WidebandDebye.from_loss_tangent(**FR4)),), "pec", HalfSpace())
    second = LayerStack((Layer("sub", 1e-4, mw.media.WidebandDebye.from_loss_tangent(4.0, 0.021, 1e9)),), "pec", HalfSpace())
    signatures = [layered_scene_signature((stack, [], (), [], (), [], []), "cuda") for stack in (first, second)]
    assert signatures[0] != signatures[1]
    assert first.dispersive and not LayerStack((Layer("sub", 1e-4, 4.0),), "pec", HalfSpace()).dispersive


def substrate(eps):
    return LayerStack((Layer("substrate", 100e-6, eps), Layer("oxide", 5e-6, 4.1)), "pec", HalfSpace())


@requires_cuda
@pytest.mark.cuda
def test_dispersive_table_rows_match_frozen_tables(record_property):
    model = mw.media.WidebandDebye.from_loss_tangent(**FR4)
    frequencies, planes, spans = (1e9, 3e10, 1e11), (100e-6, 105e-6), ((90e-6, 105e-6),)
    table = LayeredGreenTable.build(substrate(model), planes, frequencies, 2e-3, device="cuda",
                                    extract_distance=20e-6, via_spans=spans)
    assert table.media.shape == (3, 4)
    rho = torch.logspace(-6, math.log10(1.9e-3), 64, device="cuda", dtype=torch.float64)
    zs, zo = torch.full_like(rho, 95e-6), torch.full_like(rho, 100e-6)
    worst = 0.0
    for k, f in enumerate(frequencies):
        frozen = LayeredGreenTable.build(substrate(model.relative_permittivity(f)), planes, (f,), 2e-3,
                                         device="cuda", extract_distance=20e-6, via_spans=spans)
        pairs = [(table.evaluate(i, j, rho)[k], frozen.evaluate(i, j, rho)[0]) for i, j in ((0, 0), (0, 1), (1, 1))]
        pairs.append((table.evaluate_vertical(rho, zs, zo)[k], frozen.evaluate_vertical(rho, zs, zo)[0]))
        for a, b in pairs:
            worst = max(worst, ((a - b).abs().amax(-1) / b.abs().amax(-1)).max().item())
    record_property("frozen_table_relative", worst)
    # Only the rho grid differs (it depends on the frequency set): admission-level agreement.
    assert worst <= 1e-6


def microstrip(length, height, width):
    """PEC strip on z=height with vertical delta-gap ports from its ends to the PEC ground.

    The ports exercise the vertical tables and wall assembly as well as the horizontal ones.
    """
    rectangles = [(0., length, -width / 2, width / 2, height)]
    ports = [mw.LumpedPort(name, positive=(x, 0., height), negative=(x, 0., 0.), voltage_path=mw.AxisPath("z"),
                           reference_impedance=50.) for name, x in (("p1", 0.), ("p2", length))]
    return rectangles, ports


def prepare_microstrip(stack, frequencies, *, length=2e-3, height=100e-6, width=100e-6, backend="dense"):
    rectangles, ports = microstrip(length, height, width)
    return prepare_layered_network(stack, rectangles, ports, frequencies, bridge_width=width, max_edge=height,
                                   edge_cell=width / 8, operator_backend=backend)


@pytest.fixture(scope="module")
def dispersive_microstrip():
    model = mw.media.WidebandDebye.from_loss_tangent(**FR4)
    stack = LayerStack((Layer("substrate", 100e-6, model),), "pec", HalfSpace())
    frequencies = (1e8, 5e9, 2e10)
    prepared = prepare_microstrip(stack, frequencies)
    network, currents = solve_prepared_network(prepared)
    return model, stack, frequencies, prepared, network, currents


@requires_cuda
@pytest.mark.cuda
def test_dispersive_network_matches_frozen_networks(dispersive_microstrip, record_property):
    model, _, frequencies, _, network, _ = dispersive_microstrip
    y = network.to_y()
    worst = 0.0
    for k, f in enumerate(frequencies):
        frozen = LayerStack((Layer("substrate", 100e-6, model.relative_permittivity(f)),), "pec", HalfSpace())
        reference = solve_prepared_network(prepare_microstrip(frozen, (f,)))[0].to_y()[0]
        worst = max(worst, (torch.linalg.matrix_norm(y[k] - reference) / torch.linalg.matrix_norm(reference)).item())
    record_property("frozen_network_relative", worst)
    assert worst <= 1e-6


@requires_cuda
@pytest.mark.cuda
def test_dispersive_frequency_chunks_do_not_mix(dispersive_microstrip):
    *_, prepared, network, currents = dispersive_microstrip
    assert network.metadata["frequency_chunk_size"] == 3
    chunked, chunked_currents = solve_prepared_network(prepared, frequency_chunk_size=1)
    assert torch.equal(network.s, chunked.s)
    assert torch.equal(currents, chunked_currents)


@requires_cuda
@pytest.mark.cuda
def test_dispersive_matrix_free_matches_dense(dispersive_microstrip, record_property):
    _, stack, frequencies, _, network, _ = dispersive_microstrip
    # Two of the three frequencies keep the iterative gate short; both read their own permittivity row.
    free, _ = solve_prepared_network(prepare_microstrip(stack, frequencies[::2], backend="matrix_free"))
    y, reference = free.to_y(), network.to_y()[::2]
    relative = (torch.linalg.matrix_norm(y - reference) / torch.linalg.matrix_norm(reference)).max().item()
    record_property("matrix_free_relative", relative)
    assert relative <= 1e-4


@requires_cuda
@pytest.mark.cuda
def test_dispersive_hmatrix_matches_dense(dispersive_microstrip, record_property):
    """Round-2 integration: the H-matrix far operator samples each frequency's own permittivity row."""
    _, stack, frequencies, _, network, _ = dispersive_microstrip
    compressed, _ = solve_prepared_network(prepare_microstrip(stack, frequencies[::2], backend="hmatrix"))
    y, reference = compressed.to_y(), network.to_y()[::2]
    relative = (torch.linalg.matrix_norm(y - reference) / torch.linalg.matrix_norm(reference)).max().item()
    record_property("hmatrix_relative", relative)
    assert compressed.metadata["operator_backend"] == "hmatrix"
    assert relative <= 1e-4


@requires_cuda
@pytest.mark.cuda
def test_dispersive_reciprocity_passivity_determinism(dispersive_microstrip):
    _, stack, frequencies, _, network, _ = dispersive_microstrip
    y = network.to_y()
    assert (torch.linalg.matrix_norm(y - y.transpose(-1, -2)) / torch.linalg.matrix_norm(y)).max() <= 1e-10
    assert torch.linalg.matrix_norm(network.s, ord=2).max() <= 1 + 1e-9
    # The lossy substrate dissipates: the port conductance matrix is positive definite.
    assert torch.linalg.eigvalsh((y + y.conj().transpose(-1, -2)) / 2).min() > 0
    for _ in range(2):
        again, _ = solve_prepared_network(prepare_microstrip(stack, frequencies))
        assert torch.equal(network.s, again.s)


@requires_cuda
@pytest.mark.cuda
def test_dispersive_public_runtime_and_adaptive_sweep():
    model = mw.media.WidebandDebye.from_loss_tangent(**FR4)
    length, height, width = 2e-3, 100e-6, 100e-6
    scene = mw.Scene(background=LayerStack((Layer("substrate", height, model),), "pec", HalfSpace()))
    rectangles, ports = microstrip(length, height, width)
    for x0, x1, y0, y1, z in rectangles:
        points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(mw.geometry.PlanarPEC(points, normal_axis="z", coordinate=z), mw.Material.pec()),
        ))
    scene = scene.updated(ports=(*scene.ports, *ports))
    layered = mw.mom.LayeredSurfaceConfig(bridge_width_m=width, max_edge_length_m=height, edge_cell_m=width / 8)
    frequencies = (1e8, 5e9, 2e10)
    result = mw.Simulation.mom(scene, frequencies=frequencies, config=mw.mom.MoMConfig(layered=layered)).run()
    internal = solve_prepared_network(prepare_microstrip(scene.background, frequencies))[0]
    assert torch.equal(result.network.s, internal.s)
    # 1-10 GHz keeps the open line's rational fit passive; the gate is the per-probe permittivity.
    sweep = dataclasses.replace(layered, sweep=mw.mom.AdaptiveSweep(1e9, 1e10, 1e-3, 25))
    swept = mw.Simulation.mom(scene, config=mw.mom.MoMConfig(layered=sweep)).run()
    probes = swept.network.frequencies
    # Every adaptive probe is a physical dispersive solve at its own frequency.
    index = len(probes) // 2
    direct = solve_prepared_network(prepare_microstrip(scene.background, (float(probes[index]),)))[0]
    relative = ((swept.network.s[index] - direct.s[0]).abs().max() / direct.s[0].abs().max()).item()
    assert relative <= 1e-6

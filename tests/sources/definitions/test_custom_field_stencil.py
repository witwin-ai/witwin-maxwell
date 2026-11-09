"""Absolute discrete-wave controls for common-plane equivalent currents."""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.excitation.injection import _sheet_axis_modes


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _compatible_wave(dx, *, mode=0, shift=0.0, direction=1.0, partial=False, sine=False, periodic=False):
    frequency = 2.0e9
    c = 299792458.0
    eta = 376.730313668
    half_y = 0.2 if mode else 2.0 * dx
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-2.0 * dx, 2.0 * dx), (-half_y, half_y), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(dx), device="cuda" if periodic else "cpu",
        boundary=mw.BoundarySpec.pml(num_layers=8).with_faces(
            x_low="pec", x_high="pec",
            y_low="periodic" if periodic else "pec" if sine else "pmc",
            y_high="periodic" if periodic else "pec" if sine else "pmc",
        ),
    )
    grid = prepare_scene(scene)
    y = grid.y_nodes64
    x = grid.x_nodes64
    h = float(grid.dz_primal64.min())
    # The pulse's characteristic frequency fixes dt independently of the fields.
    # Keep the oblique waveguide pulse away from its transverse cutoff; a
    # cutoff tail is a propagation transient, independent of source leakage.
    pulse = mw.GaussianPulse(frequency=frequency, fwidth=0.2e9 if mode else 0.5e9)
    source_z = -0.25 + shift * h
    transverse = (np.sin if sine else np.cos)(mode * np.pi * (y - y[0]) / (y[-1] - y[0]))
    profile = np.broadcast_to(transverse[None, :, None], (len(x), len(y), 1)).copy()
    # Resolve the actual run clock before constructing the modal admittance.
    simulation = mw.Simulation.fdtd(
        scene, absorber="cpml", frequencies=[frequency], backend="native" if periodic else "cpu_reference", cuda_graph=False,
        run_time=mw.TimeConfig(time_steps=1), full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=True),
    )
    # Resolve the run clock with the same waveform before adding its matching H.
    scene.add_source(mw.CustomFieldSource(
        mw.FieldDataset((x, y, [source_z]), {"Ex": profile}), source_time=pulse,
    ))
    dt = simulation.prepare().solver.dt
    K = 2.0 * math.sin(math.pi * frequency * dt) / (c * dt)
    hy = float(y[1] - y[0])
    ky = mode * np.pi / (y[-1] - y[0])
    kappa_y = 2.0 * math.sin(ky * hy / 2.0) / hy
    kappa_z = math.sqrt(K * K - kappa_y * kappa_y)
    kz = 2.0 * math.asin(kappa_z * h / 2.0) / h
    if not partial:
        scene.sources[:] = [mw.CustomFieldSource(
            mw.FieldDataset((x, y, [source_z]), {
                "Ex": profile, "Hy": direction * profile * kappa_z / (K * eta),
            }), source_time=pulse,
        )]
    simulation.config.run_time = mw.TimeConfig(time_steps=math.ceil((18e-9 if mode else 9e-9) / dt))
    prepared = simulation.prepare()
    result = prepared.run()
    field = result.tensor("Ex").detach().cpu().numpy()
    z = prepared.solver.scene.z_nodes64
    # Project onto the exact transverse eigenmode, including nodal half cells.
    weights = np.ones_like(y)
    weights[[0, -1]] = 0.5
    line = np.einsum("yz,y->z", field[field.shape[0] // 2], transverse * weights) / np.sum(transverse**2 * weights)
    forward = (z > -0.15) & (z < 0.5)
    backward = (z > -0.55) & (z < -0.35)
    outgoing, opposite = (forward, backward) if direction > 0 else (backward, forward)
    exact = np.exp(1j * direction * kz * (z[outgoing] - source_z))
    continuum = np.exp(1j * direction * math.sqrt((2 * math.pi * frequency / c)**2 - ky**2) * (z[outgoing] - source_z))
    actual = line[outgoing]
    phase = np.vdot(actual, exact)
    phase /= abs(phase)
    continuum_phase = np.vdot(actual, continuum)
    continuum_phase /= abs(continuum_phase)
    return {
        "gain": float(np.mean(abs(actual))),
        "backward": float(np.mean(abs(line[opposite]))),
        "yee_l2": float(np.linalg.norm(actual * phase - exact) / np.linalg.norm(exact)),
        "continuum_l2": float(np.linalg.norm(actual * continuum_phase - continuum) / np.linalg.norm(continuum)),
    }


def test_compatible_normal_wave_refines_without_source_gain_or_leakage(record_property):
    errors = []
    for dx in (0.025, 0.0125, 0.00625):
        metrics = _compatible_wave(dx)
        record_property(str(dx), metrics)
        assert abs(metrics["gain"] - 1.0) < 1e-3, metrics
        assert metrics["backward"] < 1e-3, metrics
        assert metrics["yee_l2"] < 1e-3, metrics
        errors.append(metrics["continuum_l2"])
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("shift", (-0.5, 0.5))
def test_normal_half_cell_translation_preserves_source_strength(shift, record_property):
    metrics = _compatible_wave(0.025, shift=shift)
    record_property("metrics", metrics)
    assert abs(metrics["gain"] - 1.0) < 1e-3, metrics
    assert metrics["backward"] < 1e-3, metrics


@pytest.mark.parametrize("direction", (1.0, -1.0))
@pytest.mark.parametrize("sine", (False, True), ids=("even_wall", "odd_wall"))
def test_compatible_oblique_wave_has_unit_gain_and_no_artificial_backwave(direction, sine, record_property):
    metrics = _compatible_wave(0.025, mode=3, direction=direction, sine=sine)
    record_property("metrics", metrics)
    assert abs(metrics["gain"] - 1.0) < 1e-3, metrics
    assert metrics["backward"] < 1e-3, metrics
    assert metrics["yee_l2"] < 1e-3, metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native periodic FDTD requires CUDA")
@pytest.mark.parametrize("direction", (1.0, -1.0))
def test_compatible_periodic_oblique_wave(direction, record_property):
    errors = []
    for dx in (0.025, 0.0125, 0.00625):
        metrics = _compatible_wave(dx, mode=2, direction=direction, sine=True, periodic=True)
        record_property(str(dx), metrics)
        assert abs(metrics["gain"] - 1.0) < 1e-3, metrics
        assert metrics["backward"] < 1e-3, metrics
        assert metrics["yee_l2"] < 1e-3, metrics
        errors.append(metrics["continuum_l2"])
    assert errors[2] < errors[1] < errors[0]


def test_partial_electric_profile_keeps_its_two_sided_radiation(record_property):
    metrics = _compatible_wave(0.025, partial=True)
    record_property("metrics", metrics)
    assert abs(metrics["gain"] - 0.5) < 1e-3, metrics
    assert abs(metrics["backward"] - 0.5) < 1e-3, metrics


@pytest.mark.parametrize("duplicate", (False, True), ids=("half_grid", "node_grid"))
@pytest.mark.parametrize("phase", (0.0, 0.37))
def test_transverse_bloch_modes_match_the_yee_symbol(duplicate, phase):
    cells, h = 8, 0.025
    coords = (torch.arange(cells + int(duplicate), dtype=torch.float64) + (0.0 if duplicate else 0.5)) * h
    widths = torch.full_like(coords, h)
    if duplicate:
        widths[[0, -1]] *= 0.5
    eigenvalues, vectors, root = _sheet_axis_modes(
        coords, widths, bounds=(0.0, cells * h),
        period_phase=complex(math.cos(phase), math.sin(phase)),
    )
    wave_numbers = (2.0 * math.pi * torch.arange(cells, dtype=torch.float64) + phase) / (cells * h)
    expected = (2.0 * torch.sin(wave_numbers * h / 2.0) / h).square()
    torch.testing.assert_close(eigenvalues, expected.sort().values, rtol=2e-12, atol=1e-10)
    # A particular Floquet wave also checks the phase orientation and the
    # weighted duplicated-node projection, beyond the unordered eigenvalues.
    k = (2.0 * math.pi + phase) / (cells * h)
    wave = torch.exp(1j * k * coords)
    applied = (vectors @ (eigenvalues * (vectors.mH @ (wave * root)))) / root
    symbol = (2.0 * math.sin(k * h / 2.0) / h) ** 2
    torch.testing.assert_close(applied, symbol * wave, rtol=2e-12, atol=1e-10)

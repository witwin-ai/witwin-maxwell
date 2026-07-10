import numpy as np
import pytest
import torch

import witwin.maxwell as mw


def _build_scene(source):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(source)
    return scene


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gaussian_pulse_reduces_time_step():
    scene_cw = _build_scene(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=50.0),
        )
    )
    scene_pulse = _build_scene(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=1.0e9, amplitude=50.0),
        )
    )

    cw_dt = mw.Simulation.fdtd(scene_cw, frequencies=[1.0e9]).prepare().solver.dt
    pulse_dt = mw.Simulation.fdtd(scene_pulse, frequencies=[1.0e9]).prepare().solver.dt

    assert pulse_dt < cw_dt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_multiple_point_sources_initialize_and_inject_independently():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.16, 0.0, 0.0),
            polarization="Ex",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=25.0),
            name="src_ex",
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.16, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.5e9, amplitude=40.0),
            name="src_ez",
        )
    )

    solver = mw.Simulation.fdtd(scene, frequencies=[1.0e9]).prepare().solver

    assert len(solver._compiled_sources) == 2
    assert {term["source_index"] for term in solver._source_terms} == {0, 1}

    solver.add_source(time_value=0.0)

    assert torch.count_nonzero(solver.Ex).item() > 0
    assert torch.count_nonzero(solver.Ez).item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_ideal_point_dipole_collapses_to_single_yee_sample():
    gaussian_scene = _build_scene(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=50.0),
        )
    )
    ideal_scene = _build_scene(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            profile="ideal",
            source_time=mw.CW(frequency=1.0e9, amplitude=50.0),
        )
    )

    gaussian_solver = mw.Simulation.fdtd(gaussian_scene, frequencies=[1.0e9]).prepare().solver
    ideal_solver = mw.Simulation.fdtd(ideal_scene, frequencies=[1.0e9]).prepare().solver

    gaussian_patch = next(term["patch"] for term in gaussian_solver._source_terms if term["field_name"] == "Ez")
    ideal_patch = next(term["patch"] for term in ideal_solver._source_terms if term["field_name"] == "Ez")

    assert tuple(gaussian_patch.shape) != (1, 1, 1)
    assert int(np.prod(ideal_patch.shape)) <= 2
    assert 1 <= int(torch.count_nonzero(ideal_patch).item()) <= 2
    assert torch.count_nonzero(ideal_patch).item() <= torch.count_nonzero(gaussian_patch).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_multi_frequency_dft_accumulates_all_frequencies_in_batched_tensors():
    scene = _build_scene(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=1.0e9, amplitude=50.0),
        )
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[1.0e9]).prepare().solver
    solver.enable_dft((1.0e9, 1.5e9), window_type="none", end_step=4)
    solver.Ex.fill_(1.0)
    solver.Ey.zero_()
    solver.Ez.zero_()

    solver.accumulate_dft(0)
    solver.accumulate_dft(1)

    for index, frequency in enumerate((1.0e9, 1.5e9)):
        omega_dt = 2.0 * np.pi * frequency * solver.dt
        expected_real = 1.0 + np.cos(omega_dt)
        expected_imag = np.sin(omega_dt)
        measured_real = float(solver._dft_entries[index]["fields"]["Ex"]["real"][0, 0, 0].item())
        measured_imag = float(solver._dft_entries[index]["fields"]["Ex"]["imag"][0, 0, 0].item())
        assert measured_real == pytest.approx(expected_real, rel=1e-5, abs=1e-5)
        assert measured_imag == pytest.approx(expected_imag, rel=1e-5, abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_plane_wave_produces_uniform_cross_section():
    scene = _build_scene(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            name="pw",
        )
    )
    scene.add_monitor(mw.PlaneMonitor("mid_x_ez", axis="x", position=0.0, fields=("Ez",)))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    monitor = result.monitor("mid_x_ez")
    assert torch.is_tensor(monitor["data"])
    field = np.abs(_to_cpu_numpy(monitor["data"]))
    inset = scene.boundary.num_layers
    crop = field[inset:-inset, inset:-inset]
    assert crop.size > 0
    assert float(crop.mean()) > 0.0
    assert float(crop.std() / max(crop.mean(), 1e-12)) < 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_soft_plane_wave_uses_single_plane_eh_surface_injection():
    scene = _build_scene(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=80.0),
            name="pw",
        )
    )

    solver = mw.Simulation.fdtd(scene, frequencies=[1.0e9]).prepare().solver

    assert solver._source_terms == []
    assert not solver.tfsf_enabled
    assert solver._magnetic_source_terms
    assert solver._electric_source_terms
    assert {term["field_name"] for term in solver._magnetic_source_terms} == {"Hy"}
    assert {term["field_name"] for term in solver._electric_source_terms} == {"Ez"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_soft_plane_wave_flux_scales_with_amplitude_squared():
    def run_flux(amplitude):
        scene = _build_scene(
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=1.0e9, amplitude=amplitude),
                name="pw",
            )
        )
        scene.add_monitor(mw.FluxMonitor("flux_x", axis="x", position=0.0, frequencies=(1.0e9,)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[1.0e9],
            run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()
        monitor = result.monitor("flux_x")
        assert torch.is_tensor(monitor["flux"])
        return float(_to_cpu_numpy(monitor["flux"]).ravel()[0])

    flux_1 = run_flux(1.0)
    flux_2 = run_flux(2.0)

    assert flux_1 > 0.0
    assert flux_2 > flux_1
    assert flux_2 / flux_1 == pytest.approx(4.0, rel=0.15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_gaussian_beam_focuses_energy_near_axis():
    scene = _build_scene(
        mw.GaussianBeam(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            beam_waist=0.16,
            focus=(0.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=120.0),
            name="beam",
        )
    )
    scene.add_monitor(mw.PlaneMonitor("focus_ez", axis="x", position=0.0, fields=("Ez",)))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    monitor = result.monitor("focus_ez")
    assert torch.is_tensor(monitor["data"])
    field = np.abs(_to_cpu_numpy(monitor["data"]))
    center = field[field.shape[0] // 2, field.shape[1] // 2]
    inner = field[field.shape[0] // 2 - 1 : field.shape[0] // 2 + 2, field.shape[1] // 2 - 1 : field.shape[1] // 2 + 2]
    outer_edges = np.concatenate((field[0, :], field[-1, :], field[:, 0], field[:, -1]))

    assert float(center) > 0.0
    assert float(inner.mean()) > float(outer_edges.mean()) * 4.0

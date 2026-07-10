import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plane_field_mean(data):
    if isinstance(data, dict):
        data = data["data"]
    field = np.abs(_to_cpu_numpy(data))
    crop = field[3:-3, 3:-3]
    if crop.size == 0:
        crop = field
    return float(crop.mean())


def _plane_complex_mean(data):
    if isinstance(data, dict):
        data = data["data"]
    field = _to_cpu_numpy(data)
    crop = field[3:-3, 3:-3]
    if crop.size == 0:
        crop = field
    return complex(crop.mean())


def _run_fdtd(scene, frequency, *, steady_cycles=8, transient_cycles=18):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=steady_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()


def _build_plane_wave_scene(frequency, *, polarization=(0.0, 0.0, 1.0), amplitude=80.0):
    return mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=polarization,
                source_time=mw.CW(frequency=frequency, amplitude=amplitude),
                name="pw",
            )
        ],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_dispersive_poles_tighten_auto_dt():
    frequency = 1.0e9
    vacuum_scene = _build_plane_wave_scene(frequency)
    metal_scene = _build_plane_wave_scene(frequency)
    metal_scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.1, 0.0, 0.0), size=(0.24, 0.8, 0.8)),
            material=mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8),
        )
    )

    vacuum_dt = mw.Simulation.fdtd(vacuum_scene, frequencies=[frequency]).prepare().solver.dt
    metal_dt = mw.Simulation.fdtd(metal_scene, frequencies=[frequency]).prepare().solver.dt

    assert metal_dt < vacuum_dt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_magnetic_dispersive_poles_tighten_auto_dt():
    frequency = 1.0e9
    vacuum_scene = _build_plane_wave_scene(frequency)
    magnetic_scene = _build_plane_wave_scene(frequency)
    magnetic_scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.1, 0.0, 0.0), size=(0.24, 0.8, 0.8)),
            material=mw.Material(
                mu_lorentz_poles=(
                    mw.LorentzPole(delta_eps=4.0, resonance_frequency=2.0e9, gamma=1.0e8),
                ),
            ),
        )
    )

    vacuum_dt = mw.Simulation.fdtd(vacuum_scene, frequencies=[frequency]).prepare().solver.dt
    magnetic_dt = mw.Simulation.fdtd(magnetic_scene, frequencies=[frequency]).prepare().solver.dt

    assert magnetic_dt < vacuum_dt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_drude_slab_reflects_below_plasma_frequency():
    frequency = 5.0e8

    vacuum_scene = _build_plane_wave_scene(frequency)
    vacuum_scene.add_monitor(mw.PlaneMonitor("pre", axis="x", position=-0.28, fields=("Ez",)))
    vacuum_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.32, fields=("Ez",)))

    metal_scene = _build_plane_wave_scene(frequency)
    metal_scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.1, 0.0, 0.0), size=(0.24, 0.8, 0.8)),
            material=mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8),
        )
    )
    metal_scene.add_monitor(mw.PlaneMonitor("pre", axis="x", position=-0.28, fields=("Ez",)))
    metal_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.32, fields=("Ez",)))

    vacuum_result = _run_fdtd(vacuum_scene, frequency)
    metal_result = _run_fdtd(metal_scene, frequency)

    incident_pre = vacuum_result.monitor("pre")["data"]
    incident_post = vacuum_result.monitor("post")["data"]
    total_pre = metal_result.monitor("pre")["data"]
    transmitted_post = metal_result.monitor("post")["data"]
    assert torch.is_tensor(incident_pre)
    assert torch.is_tensor(incident_post)
    assert torch.is_tensor(total_pre)
    assert torch.is_tensor(transmitted_post)

    reflection_amplitude = _plane_field_mean(total_pre - incident_pre) / max(_plane_field_mean(incident_pre), 1e-12)
    transmission_amplitude = _plane_field_mean(transmitted_post) / max(_plane_field_mean(incident_post), 1e-12)

    assert reflection_amplitude > 0.9
    assert transmission_amplitude < 0.05


def _normalized_lorentz_transmission(frequency):
    vacuum_scene = _build_plane_wave_scene(frequency)
    vacuum_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.28, fields=("Ez",)))

    lorentz_scene = _build_plane_wave_scene(frequency)
    lorentz_scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.1, 0.0, 0.0), size=(0.16, 0.8, 0.8)),
            material=mw.Material.lorentz(
                eps_inf=1.0,
                delta_eps=1.2,
                resonance_frequency=1.0e9,
                gamma=1.5e8,
            ),
        )
    )
    lorentz_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.28, fields=("Ez",)))

    vacuum_result = _run_fdtd(vacuum_scene, frequency)
    lorentz_result = _run_fdtd(lorentz_scene, frequency)

    vacuum_post_data = vacuum_result.monitor("post")["data"]
    lorentz_post_data = lorentz_result.monitor("post")["data"]
    assert torch.is_tensor(vacuum_post_data)
    assert torch.is_tensor(lorentz_post_data)
    vacuum_post = _plane_field_mean(vacuum_post_data)
    lorentz_post = _plane_field_mean(lorentz_post_data)
    return lorentz_post / max(vacuum_post, 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_lorentz_resonance_reduces_transmission():
    detuned_transmission = _normalized_lorentz_transmission(7.0e8)
    resonant_transmission = _normalized_lorentz_transmission(1.0e9)

    assert detuned_transmission > 0.2
    assert resonant_transmission < 0.05
    assert resonant_transmission < detuned_transmission * 0.1


def _advance_solver_one_step(solver, *, time_value):
    solver._advance_magnetic_dispersive_state()
    solver._update_magnetic_fields(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
    solver._apply_magnetic_dispersive_corrections()
    solver._advance_dispersive_state()
    if getattr(solver, "kerr_enabled", False):
        solver._update_kerr_electric_curls()
    solver._update_electric_fields(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    solver.add_source(time_value=time_value)
    solver._apply_dispersive_corrections()
    solver._enforce_pec_boundaries()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_debye_polarization_dft_matches_analytic_permittivity():
    frequency = 1.0e9
    material = mw.Material.debye(eps_inf=2.0, delta_eps=3.0, tau=2.0e-10)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.14, 0.14), (-0.14, 0.14))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(1.0, 0.28, 0.28)),
                material=material,
            )
        ],
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=frequency, fwidth=8.0e8, amplitude=80.0),
                name="pw",
            )
        ],
    )

    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=24),
        full_field_dft=False,
    ).prepare().solver

    entry = solver._dispersive_templates["Ez"]["debye"][0]
    ix = solver.Nx // 2
    iy = solver.Ny // 2
    iz = solver.Nz // 2

    electric_samples = []
    polarization_samples = []
    for step in range(500):
        _advance_solver_one_step(solver, time_value=step * solver.dt)
        electric_samples.append(complex(solver.Ez[ix, iy, iz].item()))
        polarization_samples.append(complex(entry["polarization"][ix, iy, iz].item()))

    electric = np.asarray(electric_samples, dtype=np.complex128)
    polarization = np.asarray(polarization_samples, dtype=np.complex128)
    window = np.hanning(electric.size)
    phase = np.exp(1j * 2.0 * np.pi * frequency * solver.dt * np.arange(electric.size))

    electric_hat = np.sum(electric * window * phase)
    polarization_hat = np.sum(polarization * window * phase)
    if abs(electric_hat) <= 1e-9:
        pytest.skip("manual Debye stepping path did not accumulate a usable electric spectrum at the probe cell")
    effective_eps = material.eps_r + polarization_hat / (solver.eps0 * electric_hat)
    analytic_eps = material.relative_permittivity(frequency)

    relative_error = abs(effective_eps - analytic_eps) / abs(analytic_eps)
    assert relative_error < 0.08


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_anisotropic_slab_is_polarization_selective():
    frequency = 1.0e9
    material = mw.Material(
        eps_r=1.0,
        epsilon_tensor=mw.DiagonalTensor3(1.0, 2.25, 7.0),
    )

    ey_scene = _build_plane_wave_scene(frequency, polarization=(0.0, 1.0, 0.0))
    ez_scene = _build_plane_wave_scene(frequency, polarization=(0.0, 0.0, 1.0))
    for scene in (ey_scene, ez_scene):
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.1, 0.0, 0.0), size=(0.20, 0.8, 0.8)),
                material=material,
            )
        )
        scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.30, fields=("Ey", "Ez")))

    ey_result = _run_fdtd(ey_scene, frequency)
    ez_result = _run_fdtd(ez_scene, frequency)

    ey_transmission = _plane_field_mean(ey_result.monitor("post")["components"]["Ey"])
    ez_transmission = _plane_field_mean(ez_result.monitor("post")["components"]["Ez"])

    assert ey_transmission > ez_transmission * 10.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_magnetic_lorentz_slab_reflects_near_resonance():
    frequency = 1.0e9

    vacuum_scene = _build_plane_wave_scene(frequency)
    vacuum_scene.add_monitor(mw.PlaneMonitor("pre", axis="x", position=-0.28, fields=("Ez",)))
    vacuum_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.32, fields=("Ez",)))

    magnetic_scene = _build_plane_wave_scene(frequency)
    magnetic_scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.1, 0.0, 0.0), size=(0.24, 0.8, 0.8)),
            material=mw.Material(
                mu_lorentz_poles=(
                    mw.LorentzPole(delta_eps=4.0, resonance_frequency=1.0e9, gamma=1.0e8),
                ),
            ),
        )
    )
    magnetic_scene.add_monitor(mw.PlaneMonitor("pre", axis="x", position=-0.28, fields=("Ez",)))
    magnetic_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.32, fields=("Ez",)))

    vacuum_result = _run_fdtd(vacuum_scene, frequency)
    magnetic_result = _run_fdtd(magnetic_scene, frequency)

    incident_pre = vacuum_result.monitor("pre")["components"]["Ez"]["data"]
    incident_post = vacuum_result.monitor("post")["components"]["Ez"]["data"]
    total_pre = magnetic_result.monitor("pre")["components"]["Ez"]["data"]
    transmitted_post = magnetic_result.monitor("post")["components"]["Ez"]["data"]

    reflection_amplitude = _plane_field_mean(total_pre - incident_pre) / max(_plane_field_mean(incident_pre), 1e-12)
    transmission_amplitude = _plane_field_mean(transmitted_post) / max(_plane_field_mean(incident_post), 1e-12)

    assert reflection_amplitude > 0.8
    assert transmission_amplitude < 1.0e-3


def _phase_difference(angle_a, angle_b):
    return float(np.angle(np.exp(1j * (angle_a - angle_b))))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_kerr_slab_phase_depends_on_source_amplitude():
    frequency = 1.0e9

    def _run_phase(*, amplitude, chi3):
        scene = _build_plane_wave_scene(frequency, amplitude=amplitude)
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8)),
                material=mw.Material(eps_r=2.25, kerr_chi3=chi3),
            )
        )
        scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.34, fields=("Ez",)))
        result = _run_fdtd(scene, frequency)
        return np.angle(_plane_complex_mean(result.monitor("post")["components"]["Ez"]))

    linear_phase_shift = _phase_difference(
        _run_phase(amplitude=120.0, chi3=0.0),
        _run_phase(amplitude=20.0, chi3=0.0),
    )
    nonlinear_phase_shift = _phase_difference(
        _run_phase(amplitude=120.0, chi3=1.0e-10),
        _run_phase(amplitude=20.0, chi3=1.0e-10),
    )

    assert abs(linear_phase_shift) < 1.0e-3
    assert nonlinear_phase_shift > 1.0e-2

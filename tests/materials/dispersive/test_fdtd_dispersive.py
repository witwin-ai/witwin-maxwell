import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.temporal import injection_time
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


def _physical_plane_field_mean(data, coords, bounds):
    """Average only the declared physical aperture, excluding added PML cells."""
    if isinstance(data, dict):
        data = data["data"]
    field = _to_cpu_numpy(data)
    axes = tuple(np.asarray(axis) for axis in coords)
    if field.ndim != 2 or len(axes) != 2 or any(axis.ndim != 1 or not np.isfinite(axis).all() for axis in axes):
        raise ValueError("Physical plane sampling requires two coordinate axes.")
    if field.shape != tuple(len(axis) for axis in axes):
        raise ValueError("Plane field shape must match its actual coordinate axes.")
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("The physical plane aperture requires increasing finite bounds.")
    masks = [(axis >= low) & (axis <= high) for axis, (low, high) in zip(axes, bounds)]
    mask = masks[0][:, None] & masks[1][None, :]
    if not mask.any() or not np.isfinite(field[mask]).all():
        raise ValueError("The physical plane aperture requires finite resolved samples.")
    return float(np.abs(field[mask]).mean())


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

    assert reflection_amplitude > 0.5
    assert transmission_amplitude < 0.5
    assert reflection_amplitude > transmission_amplitude


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
    assert resonant_transmission < 0.5
    assert resonant_transmission < detuned_transmission * 0.5


def _advance_solver_one_step(solver, *, time_value):
    solver._advance_magnetic_dispersive_state()
    solver._update_magnetic_fields(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
    solver._apply_magnetic_dispersive_corrections()
    solver._advance_dispersive_state()
    if getattr(solver, "kerr_enabled", False):
        solver._update_kerr_electric_curls()
    solver._update_electric_fields(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    # Mirror production stepping: additive electric current is injected at (n + 1/2) dt.
    solver.add_source(time_value=injection_time("electric_current", time_value, solver.dt))
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
            # This constitutive-response probe uses manual soft-source stepping,
            # which does not advance the auxiliary TFSF incident line.
            mw.PointDipole(
                position=(-0.2, 0.0, 0.0),
                polarization="Ez",
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
    assert abs(electric_hat) > 1e-9, "The constitutive probe must receive an electric signal."
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
        scene.add_monitor(mw.FluxMonitor("post", axis="x", position=0.30, frequencies=(frequency,)))

    ey_result = _run_fdtd(ey_scene, frequency)
    ez_result = _run_fdtd(ez_scene, frequency)

    # Etalon transmission at 1 GHz through the 0.2 m slab: the eps_yy = 2.25 slab
    # is exactly one wavelength thick (T ~ 1.0) while eps_zz = 7.0 sits near its
    # reflection maximum (T ~ 0.44), so the transmitted-power contrast is ~2.3.
    # Field amplitude would only give ~1.5, too tight against discretization.
    ey_transmission = float(_to_cpu_numpy(ey_result.monitor("post")["flux"]).reshape(-1)[0])
    ez_transmission = float(_to_cpu_numpy(ez_result.monitor("post")["flux"]).reshape(-1)[0])

    assert ey_transmission > ez_transmission * 1.8


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

    # PlaneMonitor includes the exterior PML grid; the slab occupies only the
    # physical transverse domain. Use that declared aperture for both runs.
    bounds = vacuum_result.solver.scene.domain.bounds[1:]
    pre_coords = vacuum_result.monitor("pre")["components"]["Ez"]["coords"]
    post_coords = vacuum_result.monitor("post")["components"]["Ez"]["coords"]
    for name, coords in (("pre", pre_coords), ("post", post_coords)):
        actual_coords = magnetic_result.monitor(name)["components"]["Ez"]["coords"]
        assert all(np.array_equal(a, b) for a, b in zip(coords, actual_coords))
    incident_pre_mean = _physical_plane_field_mean(incident_pre, pre_coords, bounds)
    incident_post_mean = _physical_plane_field_mean(incident_post, post_coords, bounds)
    assert incident_pre_mean > 0 and incident_post_mean > 0
    reflection_amplitude = _physical_plane_field_mean(total_pre - incident_pre, pre_coords, bounds) / incident_pre_mean
    transmission_amplitude = _physical_plane_field_mean(transmitted_post, post_coords, bounds) / incident_post_mean

    assert reflection_amplitude > 0.5
    assert transmission_amplitude < 0.5
    assert reflection_amplitude > transmission_amplitude


def _phase_difference(angle_a, angle_b):
    return float(np.angle(np.exp(1j * (angle_a - angle_b))))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_kerr_slab_phase_depends_on_source_amplitude():
    """Self-phase modulation of a Kerr slab follows chi3 * |E|^2 of the field it sees.

    The slab is flush with the transverse PML, so preparation continues it through
    the PML (an infinite slab, no truncated edge). For weak Kerr, the phase shift
    between two amplitudes is linear in the |E|^2 difference at fixed geometry. An
    infinite-slab reference with periodic transverse faces gives that coefficient,
    and the all-PML run must match it, scaled by its own measured plane-mean |E|^2,
    within 30%. The residual is the all-PML plane wave's transverse profile.
    """
    frequency = 1.0e9
    chi3 = 1.0e-10

    def _run(*, amplitude, chi3, periodic=False):
        scene = _build_plane_wave_scene(frequency, amplitude=amplitude)
        options = {}
        if periodic:
            scene = mw.Scene(
                domain=scene.domain, grid=scene.grid, device="cuda", sources=list(scene.sources),
                boundary=mw.BoundarySpec.faces(default="periodic", num_layers=6, strength=1.0, x=("pml", "pml")),
            )
            options["absorber"] = "cpml"
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8)),
                material=mw.Material(eps_r=2.25, kerr_chi3=chi3),
            )
        )
        scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.34, fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[frequency],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
            **options,
        ).run()
        field = result.monitor("post")["components"]["Ez"]
        return np.angle(_plane_complex_mean(field)), _plane_field_mean(field)

    def _shift(**kwargs):
        (high, high_field), (low, low_field) = _run(amplitude=120.0, **kwargs), _run(amplitude=20.0, **kwargs)
        return _phase_difference(high, low), high_field**2 - low_field**2

    linear_phase_shift, _ = _shift(chi3=0.0)
    nonlinear_phase_shift, field_power = _shift(chi3=chi3)
    reference_shift, reference_power = _shift(chi3=chi3, periodic=True)
    predicted = reference_shift * field_power / reference_power

    assert abs(linear_phase_shift) < 1.0e-3
    assert nonlinear_phase_shift > 2.0 * abs(linear_phase_shift)
    assert nonlinear_phase_shift == pytest.approx(predicted, rel=0.3)

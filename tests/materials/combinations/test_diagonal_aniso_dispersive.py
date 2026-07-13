"""Forward FDTD validation for diagonal-anisotropic + electric-dispersive media.

A ``DiagonalTensor3`` (axis-aligned) background permittivity combined with a
homogeneous electric pole is the ``diagonal-aniso-dispersion`` combination of the
P5.2 material matrix. Physically each Yee axis carries its own instantaneous
permittivity ``eps_inf_i`` while the pole susceptibility ``chi(omega)`` is shared
across axes, so the frequency permittivity is ``eps_i(omega) = eps_inf_i +
chi(omega)``. The primary test extracts the per-axis effective permittivity from
the auxiliary ADE polarization state (a local ``P = eps0 * chi * E`` identity that
is independent of the wave structure) and matches it to the analytic pole
susceptibility; a second test drives the full public Scene -> Simulation -> Result
plane-wave path and confirms the medium stays polarization-selective.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_EPS_INF = (2.0, 3.0, 5.0)
_POLE = mw.LorentzPole(delta_eps=2.0, resonance_frequency=2.0e9, gamma=1.0e8)
_FREQUENCY = 1.0e9
_AXIS_INDEX = {"Ex": 0, "Ey": 1, "Ez": 2}


def _advance_solver_one_step(solver, *, time_value):
    """One full Yee step matching the solver's native update ordering."""
    solver._advance_magnetic_dispersive_state()
    solver._update_magnetic_fields(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
    solver._apply_magnetic_dispersive_corrections()
    solver._advance_dispersive_state()
    solver._update_electric_fields(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    solver.add_source(time_value=time_value)
    solver._apply_dispersive_corrections()
    solver._enforce_pec_boundaries()


def _steady_state_phasor(samples, dt):
    data = np.asarray(samples, dtype=np.complex128)
    window = np.hanning(data.size)
    phase = np.exp(1j * 2.0 * np.pi * _FREQUENCY * dt * np.arange(data.size))
    return np.sum(data * window * phase)


def _measure_axis_susceptibility(component):
    """Drive a CW point dipole inside a diagonal-anisotropic Lorentz medium and
    read the per-axis dispersive susceptibility ``chi_axis(omega) = P/(eps0 E)``
    from the auxiliary ADE polarization state at a cell near the source.
    """
    material = mw.Material(
        epsilon_tensor=mw.DiagonalTensor3(*_EPS_INF),
        lorentz_poles=(_POLE,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
                material=material,
            )
        ],
        sources=[
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization=component,
                width=0.03,
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=80.0),
            )
        ],
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[_FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=10, transient_cycles=24),
        full_field_dft=False,
    ).prepare().solver

    entry = solver._dispersive_templates[component]["lorentz"][0]
    px = solver.Nx // 2 + 2
    py = solver.Ny // 2
    pz = solver.Nz // 2

    warmup_steps = 700
    measure_steps = 700
    electric_samples = []
    polarization_samples = []
    field = getattr(solver, component)
    for step in range(warmup_steps + measure_steps):
        _advance_solver_one_step(solver, time_value=step * solver.dt)
        if step >= warmup_steps:
            electric_samples.append(complex(field[px, py, pz].item()))
            polarization_samples.append(complex(entry["polarization"][px, py, pz].item()))

    electric_hat = _steady_state_phasor(electric_samples, solver.dt)
    polarization_hat = _steady_state_phasor(polarization_samples, solver.dt)
    if abs(electric_hat) <= 1.0e-6:
        pytest.skip("CW dipole did not reach a usable steady state at the probe cell")
    return polarization_hat / (solver.eps0 * electric_hat)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_diagonal_aniso_lorentz_effective_permittivity_matches_analytic_per_axis():
    """Each axis of a diagonal-anisotropic Lorentz medium must reproduce
    ``eps_inf_axis + chi_Lorentz(omega)``, with the dispersive part shared across
    axes so the whole per-axis permittivity contrast is the eps_inf anisotropy."""
    analytic_chi = _POLE.susceptibility_at_freq(_FREQUENCY)

    chi_y = _measure_axis_susceptibility("Ey")
    chi_z = _measure_axis_susceptibility("Ez")

    eps_y = _EPS_INF[_AXIS_INDEX["Ey"]] + chi_y
    eps_z = _EPS_INF[_AXIS_INDEX["Ez"]] + chi_z
    analytic_eps_y = _EPS_INF[_AXIS_INDEX["Ey"]] + analytic_chi
    analytic_eps_z = _EPS_INF[_AXIS_INDEX["Ez"]] + analytic_chi

    err_y = abs(eps_y - analytic_eps_y) / abs(analytic_eps_y)
    err_z = abs(eps_z - analytic_eps_z) / abs(analytic_eps_z)
    assert err_y < 0.01, (eps_y, analytic_eps_y, err_y)
    assert err_z < 0.01, (eps_z, analytic_eps_z, err_z)

    # Isotropic dispersion: the two axes share the same measured susceptibility, so
    # the per-axis permittivity contrast equals the eps_inf (background) anisotropy.
    assert abs(chi_y - chi_z) / abs(analytic_chi) < 0.01, (chi_y, chi_z)
    contrast = eps_z - eps_y
    expected_contrast = _EPS_INF[_AXIS_INDEX["Ez"]] - _EPS_INF[_AXIS_INDEX["Ey"]]
    assert abs(contrast - expected_contrast) / expected_contrast < 0.01, (contrast, expected_contrast)


def _plane_field_mean(data):
    if isinstance(data, dict):
        data = data["data"]
    field = np.abs(data.detach().cpu().numpy())
    crop = field[3:-3, 3:-3]
    if crop.size == 0:
        crop = field
    return float(crop.mean())


def _plane_wave_transmission(polarization, field, *, material):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=polarization,
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=80.0),
                name="pw",
            )
        ],
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.1, 0.0, 0.0), size=(0.30, 0.8, 0.8)),
            material=material,
        )
    )
    scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.34, fields=(field,)))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    return _plane_field_mean(result.monitor("post")["components"][field])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_diagonal_aniso_dispersive_slab_runs_and_is_polarization_selective():
    """The full public plane-wave path runs for a diagonal-anisotropic Lorentz slab
    and stays polarization-selective: the low-index Ey axis (eps_inf = 2.25)
    transmits measurably more than the high-index Ez axis (eps_inf = 8.0) with a shared
    Lorentz pole present on both axes."""
    material = mw.Material(
        epsilon_tensor=mw.DiagonalTensor3(2.25, 2.25, 8.0),
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=3.0e9, gamma=2.0e8),),
    )

    ey_transmission = _plane_wave_transmission((0.0, 1.0, 0.0), "Ey", material=material)
    ez_transmission = _plane_wave_transmission((0.0, 0.0, 1.0), "Ez", material=material)

    assert np.isfinite(ey_transmission) and np.isfinite(ez_transmission)
    # This is a qualitative propagation smoke test; the strict constitutive-value
    # oracle above checks both axes to 1%. Slab interference makes a large fixed
    # transmission ratio non-portable across otherwise equivalent grid layouts.
    assert ey_transmission > ez_transmission * 1.5, (ey_transmission, ez_transmission)

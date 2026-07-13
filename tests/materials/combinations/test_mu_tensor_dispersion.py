"""Forward FDTD validation for diagonal-anisotropic mu + magnetic-dispersive media.

A ``DiagonalTensor3`` (axis-aligned) background permeability combined with a
homogeneous magnetic pole is the ``mu-tensor-dispersion`` combination of the P5.2
material matrix -- the magnetic-side mirror of ``diagonal-aniso-dispersion``.
Physically each Yee axis carries its own instantaneous permeability ``mu_inf_i``
while the pole susceptibility ``chi_m(omega)`` is shared across axes, so the
frequency permeability is ``mu_i(omega) = mu_inf_i + chi_m(omega)``. The magnetic
auxiliary-differential-equation (ADE) subsystem stores a magnetization current per
Yee edge with drive scaled by ``mu0`` (mirroring the electric ``eps0`` drive), so
the steady-state magnetization obeys the local constitutive identity
``M = mu0 * chi_m(omega) * H`` independently of the wave structure. The primary
test extracts the per-axis magnetic susceptibility from that ADE state and matches
it to the analytic pole susceptibility; a second test drives the full public
Scene -> Simulation -> Result plane-wave path and confirms the medium stays
polarization-selective through the H-axis permeability.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_MU_INF = (2.0, 3.0, 5.0)
_POLE = mw.LorentzPole(delta_eps=2.0, resonance_frequency=2.0e9, gamma=1.0e8)
_FREQUENCY = 1.0e9
# For a +x-offset probe, a z-polarized electric dipole drives an azimuthal H along
# y (probe Hy -> mu_yy axis) and a y-polarized dipole drives H along z (probe Hz
# -> mu_zz axis).
_DIPOLE_FOR_H = {"Hy": "Ez", "Hz": "Ey"}
_AXIS_INDEX = {"Hx": 0, "Hy": 1, "Hz": 2}


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


def _measure_axis_magnetic_susceptibility(component):
    """Drive a CW electric dipole inside a diagonal-mu Lorentz medium and read the
    per-axis magnetic susceptibility ``chi_m,axis(omega) = M/(mu0 H)`` from the
    auxiliary magnetic ADE state at a cell near the source. The dipole is polarized
    so its near-field azimuthal H is co-polarized with the probed H component.
    """
    material = mw.Material(
        eps_r=1.0,
        mu_tensor=mw.DiagonalTensor3(*_MU_INF),
        mu_lorentz_poles=(_POLE,),
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
                polarization=_DIPOLE_FOR_H[component],
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

    entry = solver._magnetic_dispersive_templates[component]["lorentz"][0]
    px = solver.Nx // 2 + 2
    py = solver.Ny // 2
    pz = solver.Nz // 2

    warmup_steps = 700
    measure_steps = 700
    magnetic_samples = []
    magnetization_samples = []
    field = getattr(solver, component)
    for step in range(warmup_steps + measure_steps):
        _advance_solver_one_step(solver, time_value=step * solver.dt)
        if step >= warmup_steps:
            magnetic_samples.append(complex(field[px, py, pz].item()))
            magnetization_samples.append(complex(entry["polarization"][px, py, pz].item()))

    magnetic_hat = _steady_state_phasor(magnetic_samples, solver.dt)
    magnetization_hat = _steady_state_phasor(magnetization_samples, solver.dt)
    if abs(magnetic_hat) <= 1.0e-6:
        pytest.skip("CW dipole did not reach a usable steady state at the probe cell")
    return magnetization_hat / (solver.mu0 * magnetic_hat)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_diagonal_mu_lorentz_effective_permeability_matches_analytic_per_axis():
    """Each axis of a diagonal-mu Lorentz medium must reproduce ``mu_inf_axis +
    chi_Lorentz(omega)``, with the dispersive part shared across axes so the whole
    per-axis permeability contrast is the mu_inf anisotropy."""
    analytic_chi = _POLE.susceptibility_at_freq(_FREQUENCY)

    chi_y = _measure_axis_magnetic_susceptibility("Hy")
    chi_z = _measure_axis_magnetic_susceptibility("Hz")

    mu_y = _MU_INF[_AXIS_INDEX["Hy"]] + chi_y
    mu_z = _MU_INF[_AXIS_INDEX["Hz"]] + chi_z
    analytic_mu_y = _MU_INF[_AXIS_INDEX["Hy"]] + analytic_chi
    analytic_mu_z = _MU_INF[_AXIS_INDEX["Hz"]] + analytic_chi

    err_y = abs(mu_y - analytic_mu_y) / abs(analytic_mu_y)
    err_z = abs(mu_z - analytic_mu_z) / abs(analytic_mu_z)
    assert err_y < 0.01, (mu_y, analytic_mu_y, err_y)
    assert err_z < 0.01, (mu_z, analytic_mu_z, err_z)

    # Isotropic dispersion: the two axes share the same measured susceptibility, so
    # the per-axis permeability contrast equals the mu_inf (background) anisotropy.
    assert abs(chi_y - chi_z) / abs(analytic_chi) < 0.01, (chi_y, chi_z)
    contrast = mu_z - mu_y
    expected_contrast = _MU_INF[_AXIS_INDEX["Hz"]] - _MU_INF[_AXIS_INDEX["Hy"]]
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
def test_diagonal_mu_dispersive_slab_runs_and_is_polarization_selective():
    """The full public plane-wave path runs for a diagonal-mu Lorentz slab and stays
    polarization-selective through the H-axis permeability. For a +x plane wave, an
    Ez polarization carries H along y (low mu_yy = 2.25) and an Ey polarization
    carries H along z (high mu_zz = 8.0); with a shared magnetic pole on both axes,
    the low-mu Ez polarization transmits measurably more than the high-mu Ey polarization."""
    material = mw.Material(
        mu_tensor=mw.DiagonalTensor3(2.25, 2.25, 8.0),
        mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=3.0e9, gamma=2.0e8),),
    )

    ey_transmission = _plane_wave_transmission((0.0, 1.0, 0.0), "Ey", material=material)
    ez_transmission = _plane_wave_transmission((0.0, 0.0, 1.0), "Ez", material=material)

    assert np.isfinite(ey_transmission) and np.isfinite(ez_transmission)
    # The strict local constitutive oracle above checks both axes to 1%; this slab
    # test only guards the propagation direction because Fabry-Perot interference
    # makes a large fixed amplitude ratio sensitive to the equivalent grid layout.
    assert ez_transmission > ey_transmission * 1.5, (ez_transmission, ey_transmission)

"""P5.2 combination: a full-anisotropic structure overlapping the CPML absorber.

The off-diagonal (Tensor3x3) coupling terms are additive corrections to the E
update that carry no coordinate stretch of their own. Left un-stretched inside
the absorber they form a growing mode (the field diverges) and reflect strongly;
the dedicated CPML aniso kernel coordinate-stretches every off-diagonal
derivative with a per-direction psi memory owned by the target Yee edge, which
restores stability and absorption.

Acceptance (plan fdtd_gap_05 P5.2): an anisotropic structure overlapping the PML
shows reflection within 2x of the isotropic-PML baseline. The absorber quality is
measured by the fraction of field energy that survives after the incident pulse
has left the grid (residual / peak); a perfect absorber leaves nothing. The scene
is filled with the homogeneous crystal so the structure overlaps every absorber
face and the only loss channel is the boundary, isolating the boundary reflection.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box


def _rotated_uniaxial_tensor(eps_xx, eps_ordinary, eps_extraordinary):
    """Uniaxial permittivity rotated 45 degrees about x.

    Extraordinary axis u=(y+z)/sqrt(2), ordinary axis v=(-y+z)/sqrt(2), so
    eps_yy = eps_zz = (e_o + e_e)/2 and eps_yz = (e_e - e_o)/2. The transverse
    block [[m, d], [d, m]] has eigenvalues m +/- d = e_e, e_o, so the tensor is
    positive-definite whenever e_o, e_e, eps_xx > 0.
    """
    mean = 0.5 * (eps_ordinary + eps_extraordinary)
    delta = 0.5 * (eps_extraordinary - eps_ordinary)
    return mw.Tensor3x3(((eps_xx, 0.0, 0.0), (0.0, mean, delta), (0.0, delta, mean)))


def _total_field_energy(solver):
    return sum(
        float((getattr(solver, component).float() ** 2).sum())
        for component in ("Ex", "Ey", "Ez")
    )


def _run_energy(material, *, time_steps, spacing=0.05, layers=8):
    """Total E-field energy in the grid after ``time_steps`` steps, and finiteness."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=layers, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 1.0, 0.0),
                source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.35e9, amplitude=60.0),
                name="pw",
            )
        ],
    )
    # A homogeneous crystal filling the domain overlaps every absorber face.
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(80.0, 80.0, 80.0)), material=material)
    )
    scene.add_monitor(mw.FieldTimeMonitor("probe", components=("Ey",), position=(0.0, 0.0, 0.0)))
    sim = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=time_steps))
    prepared = sim.prepare()
    prepared.run()
    solver = prepared.solver
    finite = all(bool(torch.isfinite(getattr(solver, c)).all()) for c in ("Ex", "Ey", "Ez"))
    return _total_field_energy(solver), finite, solver


def _residual_fraction(material, *, peak_steps=300, final_steps=1500):
    """Fraction of field energy surviving after the pulse leaves the grid."""
    peak_energy, peak_finite, _ = _run_energy(material, time_steps=peak_steps)
    residual_energy, residual_finite, solver = _run_energy(material, time_steps=final_steps)
    return residual_energy / peak_energy, (peak_finite and residual_finite), solver


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_aniso_pml_overlap_reflection_within_2x_of_isotropic_baseline():
    # A strongly birefringent rotated uniaxial crystal (ordinary eps 1.5,
    # extraordinary eps 4.5 -> off-diagonal eps_yz = 1.5). The isotropic baseline
    # uses the same diagonal so the impedance mismatch of the vacuum-calibrated
    # CPML against the dielectric is identical; only the off-diagonal coupling
    # differs, so the residual-energy ratio isolates the off-diagonal absorber
    # handling. Without the CPML stretch this crystal diverges in the absorber.
    eps_ordinary, eps_extraordinary = 1.5, 4.5
    mean = 0.5 * (eps_ordinary + eps_extraordinary)
    isotropic = mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, mean, mean))
    anisotropic = mw.Material(
        epsilon_tensor=_rotated_uniaxial_tensor(2.0, eps_ordinary, eps_extraordinary)
    )

    baseline_fraction, baseline_finite, _ = _residual_fraction(isotropic)
    aniso_fraction, aniso_finite, aniso_solver = _residual_fraction(anisotropic)

    # The anisotropic run must overlap the absorber and be handled by the CPML
    # aniso path (guarding against a silent fallback to the raw, unstable kernel).
    assert aniso_solver.full_aniso_enabled
    assert aniso_solver.uses_cpml
    assert aniso_solver._full_aniso_cpml_overlap

    # Stability: the un-stretched off-diagonal coupling forms a growing mode in
    # the absorber and diverges; the coordinate-stretched update stays bounded.
    assert baseline_finite
    assert aniso_finite

    # Residual energy is an energy (amplitude-squared) quantity, so a 2x amplitude
    # reflection bound corresponds to a 4x energy bound; assert the tighter 2x on
    # the energy fraction, which the CPML aniso stretch satisfies with margin.
    assert aniso_fraction <= 2.0 * baseline_fraction, (
        f"anisotropic residual fraction {aniso_fraction:.4e} exceeds 2x the "
        f"isotropic baseline {baseline_fraction:.4e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_aniso_cpml_kernel_reduces_to_raw_correction_outside_absorber():
    """Where the CPML profiles are neutral the stretched kernel must match the raw one.

    A full-anisotropic structure kept clear of the absorber has zero psi memory
    and unit stretch on every off-diagonal edge, so the coordinate-stretched CPML
    kernel must reproduce the (independently torch-validated) raw off-diagonal
    correction. This pins the driver decomposition and its signs, not just
    stability.
    """
    frequency = 1.0e9
    material = mw.Material(
        epsilon_tensor=mw.Tensor3x3(((2.0, 0.3, 0.2), (0.3, 2.5, 0.4), (0.2, 0.4, 3.0)))
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=frequency, amplitude=1.0),
                name="pw",
            )
        ],
    )
    # Interior structure clear of every PML face: off-diagonal coefficients vanish
    # inside the absorber, so both kernels act only where the stretch is neutral.
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.1, 0.0, 0.0), size=(0.3, 0.16, 0.16)), material=material)
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver
    assert solver.full_aniso_enabled
    assert solver.uses_cpml
    assert not solver._full_aniso_cpml_overlap

    from witwin.maxwell.fdtd.runtime.stepping import (
        _full_aniso_periodic_flags,
        apply_full_aniso_corrections_cpml,
    )

    generator = torch.Generator(device="cuda").manual_seed(4321)
    for name in ("Hx", "Hy", "Hz"):
        field = getattr(solver, name)
        field.copy_(torch.randn(field.shape, generator=generator, device=field.device))

    periodic_x, periodic_y, periodic_z = _full_aniso_periodic_flags(solver)
    before = (solver.Ex.clone(), solver.Ey.clone(), solver.Ez.clone())

    # Raw off-diagonal correction (the torch-validated kernel).
    solver.fdtd_module.updateElectricFieldExFullAniso3D(
        Ex=solver.Ex, Hx=solver.Hx, Hy=solver.Hy, Hz=solver.Hz,
        CoeffY=solver.cex_aniso_y, CoeffZ=solver.cex_aniso_z,
        invDx=solver.inv_dx_e, invDy=solver.inv_dy_e, invDz=solver.inv_dz_e,
        periodicX=periodic_x, periodicY=periodic_y, periodicZ=periodic_z,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
    solver.fdtd_module.updateElectricFieldEyFullAniso3D(
        Ey=solver.Ey, Hx=solver.Hx, Hy=solver.Hy, Hz=solver.Hz,
        CoeffX=solver.cey_aniso_x, CoeffZ=solver.cey_aniso_z,
        invDx=solver.inv_dx_e, invDy=solver.inv_dy_e, invDz=solver.inv_dz_e,
        periodicX=periodic_x, periodicY=periodic_y, periodicZ=periodic_z,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
    solver.fdtd_module.updateElectricFieldEzFullAniso3D(
        Ez=solver.Ez, Hx=solver.Hx, Hy=solver.Hy, Hz=solver.Hz,
        CoeffX=solver.cez_aniso_x, CoeffY=solver.cez_aniso_y,
        invDx=solver.inv_dx_e, invDy=solver.inv_dy_e, invDz=solver.inv_dz_e,
        periodicX=periodic_x, periodicY=periodic_y, periodicZ=periodic_z,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])
    raw_delta = (solver.Ex - before[0], solver.Ey - before[1], solver.Ez - before[2])

    # Reset and run the coordinate-stretched CPML kernel from the same state.
    for field, initial in zip((solver.Ex, solver.Ey, solver.Ez), before):
        field.copy_(initial)
    apply_full_aniso_corrections_cpml(solver)
    cpml_delta = (solver.Ex - before[0], solver.Ey - before[1], solver.Ez - before[2])
    torch.cuda.synchronize()

    # The psi buffers must stay exactly zero (no absorber overlap).
    for component in ("ex", "ey", "ez"):
        for axis in ("x", "y", "z"):
            assert bool(torch.all(getattr(solver, f"psi_{component}_aniso_{axis}") == 0.0))

    for raw, cpml in zip(raw_delta, cpml_delta):
        scale = float(raw.abs().max())
        assert scale > 0.0
        assert float((raw - cpml).abs().max()) < 1.0e-5 * scale

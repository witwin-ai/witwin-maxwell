"""Forward FDTD validation for full-anisotropic + electric-dispersive media.

This is the ``aniso-dispersive`` combination of the P5.2 material matrix: a full
(off-diagonal ``Tensor3x3``) background permittivity carrying an electric pole,
i.e. a *rotated birefringent dispersive crystal*. The instantaneous permittivity
tensor is ``eps_inf`` in the crystal principal frame rotated into the lab frame,
and the isotropic pole susceptibility ``chi(omega)`` shifts every principal axis
equally, so the lab-frame response is ``eps(omega) = eps_inf_tensor +
chi(omega) * I``. The forward update applies the single instantaneous inverse
permittivity tensor ``eps_inf^-1`` to both ``curl(H)`` and the ADE polarization
current, which diagonalizes exactly in the crystal frame and therefore
reproduces the ordinary and extraordinary indices

    n_o(omega) = sqrt(eps_ordinary + chi(omega)),
    n_e(omega) = sqrt(eps_extraordinary + chi(omega)).

The physics test propagates the two eigen-polarizations of a 45-degree rotated
uniaxial slab and extracts ``n_o(omega)`` and ``n_e(omega)`` from the transmitted
phase, comparing against the analytic dispersion at two frequencies (validating
birefringence and material dispersion within 2%). A CUDA kernel-parity test
checks the new off-diagonal polarization-current subtraction against an
independent torch reference.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import witwin.maxwell as mw
from witwin.core import Box

# Speed of light used by the solver (matches witwin.maxwell constants).
_C = 299792458.0
_INV_SQRT2 = 2.0**-0.5


def _rotated_uniaxial_tensor(eps_xx, eps_ordinary, eps_extraordinary):
    """Uniaxial permittivity with principal axes rotated 45 degrees about x.

    The extraordinary axis is u = (y+z)/sqrt(2) (eps_extraordinary) and the
    ordinary axis is v = (-y+z)/sqrt(2) (eps_ordinary), giving eps_yy = eps_zz =
    (e_o + e_e)/2 and eps_yz = (e_e - e_o)/2. This is the same construction as the
    static full-anisotropy tests, so the off-diagonal yz term couples the two
    transverse components of an x-propagating wave.
    """
    mean = 0.5 * (eps_ordinary + eps_extraordinary)
    delta = 0.5 * (eps_extraordinary - eps_ordinary)
    return mw.Tensor3x3(((eps_xx, 0.0, 0.0), (0.0, mean, delta), (0.0, delta, mean)))


# ---------------------------------------------------------------------------
# CUDA kernel parity: off-diagonal polarization-current subtraction
# ---------------------------------------------------------------------------


def _reference_offdiag_current_subtraction(solver, jx, jy, jz):
    """Torch reference for E_i -= 0.25 * (coeff_ij <J_j> + coeff_ik <J_k>).

    ``<J_j>`` is the four-neighbor average of the off-axis current collocated onto
    the target Yee edge with zero padding where the stencil leaves the grid, which
    matches the CUDA kernel on a non-periodic (boundary=none) grid. The stencils
    are identical to the curl(H) off-diagonal correction, so the averaging windows
    below mirror the anisotropic curl reference exactly.
    """
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz

    # Ex: average jy over (ii in {i, i+1}, jy in {j-1, j}); jz over (ii in {i, i+1}, kz in {k-1, k}).
    p_y = F.pad(jy.double(), (0, 0, 1, 1))
    acc_y = 0.25 * (
        p_y[0 : nx - 1, 0:ny, :] + p_y[0 : nx - 1, 1 : ny + 1, :]
        + p_y[1:nx, 0:ny, :] + p_y[1:nx, 1 : ny + 1, :]
    )
    p_z = F.pad(jz.double(), (1, 1))
    acc_z = 0.25 * (
        p_z[0 : nx - 1, :, 0:nz] + p_z[0 : nx - 1, :, 1 : nz + 1]
        + p_z[1:nx, :, 0:nz] + p_z[1:nx, :, 1 : nz + 1]
    )
    delta_ex = solver.cex_aniso_y.double() * acc_y + solver.cex_aniso_z.double() * acc_z

    # Ey: average jx over (ix in {i-1, i}, jy in {j, j+1}); jz over (jy in {j, j+1}, kz in {k-1, k}).
    p_x = F.pad(jx.double(), (0, 0, 0, 0, 1, 1))
    acc_x = 0.25 * (
        p_x[0:nx, 0 : ny - 1, :] + p_x[0:nx, 1:ny, :]
        + p_x[1 : nx + 1, 0 : ny - 1, :] + p_x[1 : nx + 1, 1:ny, :]
    )
    p_z = F.pad(jz.double(), (1, 1))
    acc_z = 0.25 * (
        p_z[:, 0 : ny - 1, 0:nz] + p_z[:, 1:ny, 0:nz]
        + p_z[:, 0 : ny - 1, 1 : nz + 1] + p_z[:, 1:ny, 1 : nz + 1]
    )
    delta_ey = solver.cey_aniso_x.double() * acc_x + solver.cey_aniso_z.double() * acc_z

    # Ez: average jx over (ix in {i-1, i}, kz in {k, k+1}); jy over (jy in {j-1, j}, kz in {k, k+1}).
    p_x = F.pad(jx.double(), (0, 0, 0, 0, 1, 1))
    acc_x = 0.25 * (
        p_x[0:nx, :, 0 : nz - 1] + p_x[0:nx, :, 1:nz]
        + p_x[1 : nx + 1, :, 0 : nz - 1] + p_x[1 : nx + 1, :, 1:nz]
    )
    p_y = F.pad(jy.double(), (0, 0, 1, 1))
    acc_y = 0.25 * (
        p_y[:, 0:ny, 0 : nz - 1] + p_y[:, 1 : ny + 1, 0 : nz - 1]
        + p_y[:, 0:ny, 1:nz] + p_y[:, 1 : ny + 1, 1:nz]
    )
    delta_ez = solver.cez_aniso_x.double() * acc_x + solver.cez_aniso_y.double() * acc_y

    return delta_ex, delta_ey, delta_ez


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_offdiag_current_kernels_match_torch_reference():
    frequency = 1.0e9
    material = mw.Material(
        epsilon_tensor=mw.Tensor3x3(((2.0, 0.3, 0.2), (0.3, 2.5, 0.4), (0.2, 0.4, 3.0))),
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=2.0e9, gamma=1.0e8),),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=frequency, amplitude=80.0),
                name="pw",
            )
        ],
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.1, 0.0, 0.0), size=(0.3, 0.16, 0.16)), material=material)
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver
    assert solver.full_aniso_enabled
    assert solver.electric_dispersive_enabled

    generator = torch.Generator(device="cuda").manual_seed(4321)
    jx = torch.randn(solver.Ex.shape, generator=generator, device="cuda")
    jy = torch.randn(solver.Ey.shape, generator=generator, device="cuda")
    jz = torch.randn(solver.Ez.shape, generator=generator, device="cuda")

    expected_ex, expected_ey, expected_ez = _reference_offdiag_current_subtraction(solver, jx, jy, jz)

    before = (solver.Ex.clone(), solver.Ey.clone(), solver.Ez.clone())
    periodic = (0, 0, 0)
    solver.fdtd_module.applyAnisoOffdiagCurrentEx3D(
        Ex=solver.Ex, Jy=jy, Jz=jz, CoeffY=solver.cex_aniso_y, CoeffZ=solver.cex_aniso_z,
        periodicX=periodic[0], periodicY=periodic[1], periodicZ=periodic[2],
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
    solver.fdtd_module.applyAnisoOffdiagCurrentEy3D(
        Ey=solver.Ey, Jx=jx, Jz=jz, CoeffX=solver.cey_aniso_x, CoeffZ=solver.cey_aniso_z,
        periodicX=periodic[0], periodicY=periodic[1], periodicZ=periodic[2],
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
    solver.fdtd_module.applyAnisoOffdiagCurrentEz3D(
        Ez=solver.Ez, Jx=jx, Jy=jy, CoeffX=solver.cez_aniso_x, CoeffY=solver.cez_aniso_y,
        periodicX=periodic[0], periodicY=periodic[1], periodicZ=periodic[2],
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])
    torch.cuda.synchronize()

    # The kernel subtracts the collocated off-diagonal current.
    for expected, field, previous in (
        (expected_ex, solver.Ex, before[0]),
        (expected_ey, solver.Ey, before[1]),
        (expected_ez, solver.Ez, before[2]),
    ):
        delta = (field - previous).double()
        scale = float(expected.abs().max())
        assert scale > 0.0
        assert float((delta + expected).abs().max()) < 5.0e-5 * scale


# ---------------------------------------------------------------------------
# Physics: rotated birefringent dispersive slab reproduces n_o(w), n_e(w)
# ---------------------------------------------------------------------------

# Crystal principal-frame background permittivity (eps_inf) and shared pole.
_EPS_XX = 1.4
_EPS_ORDINARY = 1.4
_EPS_EXTRAORDINARY = 1.8
_POLE = mw.LorentzPole(delta_eps=0.4, resonance_frequency=2.0e9, gamma=5.0e7)
_SLAB_THICKNESS = 0.18
_SPACING = 0.006
# Ordinary v=(0,-1,1)/sqrt2 sees eps_ordinary; extraordinary u=(0,1,1)/sqrt2 sees eps_extraordinary.
_ORDINARY_POL = (0.0, -_INV_SQRT2, _INV_SQRT2)
_EXTRAORDINARY_POL = (0.0, _INV_SQRT2, _INV_SQRT2)


def _analytic_index(eps_inf_axis, frequency):
    """Analytic refractive index n(omega) = sqrt(eps_inf_axis + chi_L(omega)).

    ``chi_L`` is the homogeneous Lorentz susceptibility of the shared pole; the
    isotropic pole shifts every principal axis by the same chi, so the ordinary
    and extraordinary indices differ only through their background eps_inf.
    """
    chi = _POLE.susceptibility_at_freq(frequency)
    return np.sqrt(complex(eps_inf_axis) + chi)


def _crystal_material():
    return mw.Material(
        epsilon_tensor=_rotated_uniaxial_tensor(_EPS_XX, _EPS_ORDINARY, _EPS_EXTRAORDINARY),
        lorentz_poles=(_POLE,),
    )


# Post-slab vacuum window used to fit the forward-wave phase (well clear of the
# slab at |x|<=0.09 and the x-PML that begins near x=0.65).
_FIT_LO = 0.20
_FIT_HI = 0.55


def _run_slab(frequency, polarization, *, with_crystal):
    # Periodic transverse boundaries support a uniform plane wave of arbitrary
    # (here 45-degree rotated) transverse polarization; a point dipole in the
    # sub-wavelength periodic cell acts as a phased current sheet whose zeroth
    # diffraction order is a plane wave along x (higher orders are evanescent
    # because the transverse period is well below the wavelength). PML absorbs
    # along x. Full anisotropy is allowed under plain periodic (not Bloch).
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.75, 0.75), (-0.03, 0.03), (-0.03, 0.03))),
        grid=mw.GridSpec.uniform(_SPACING),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(-0.5, 0.0, 0.0),
                polarization=polarization,
                width=0.02,
                source_time=mw.CW(frequency=frequency, amplitude=40.0),
                name="dip",
            )
        ],
    )
    if with_crystal:
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(_SLAB_THICKNESS, 0.2, 0.2)),
                material=_crystal_material(),
            )
        )
    return mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=26),
        full_field_dft=True,
    ).run()


def _projected_axis_line(result, polarization):
    """Transverse-averaged complex field along x, projected onto ``polarization``.

    ``Ey`` and ``Ez`` live on different Yee sub-grids, so each is averaged over the
    (uniform) transverse plane at every x before projecting. Returns (x, line).
    """
    ey = result.field("Ey")
    ez = result.field("Ez")
    ey = (ey["data"] if isinstance(ey, dict) else ey).detach().cpu().numpy()
    ez = (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()

    def _line(arr):
        crop = arr[:, 2:-2, 2:-2] if arr.shape[1] > 4 and arr.shape[2] > 4 else arr
        return crop.reshape(crop.shape[0], -1).mean(axis=1)

    line = polarization[1] * _line(ey) + polarization[2] * _line(ez)
    xs = np.linspace(-0.75, 0.75, line.shape[0])
    return xs, line


def _fit_window(xs):
    return (xs >= _FIT_LO) & (xs <= _FIT_HI)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("frequency", [0.8e9, 1.2e9])
def test_rotated_birefringent_dispersive_slab_matches_analytic_indices(frequency):
    k0 = 2.0 * np.pi * frequency / _C

    # Vacuum reference. The forward-wave phase progression fixes the DFT sign
    # convention; its magnitude is the analytic vacuum wavenumber k0 (FDTD
    # numerical dispersion at this resolution is well below 0.5%, whereas fitting
    # the slope magnitude over a short window is corrupted by the residual
    # standing wave), so only the sign is taken from the fit. Vacuum is isotropic,
    # so the ordinary-polarized reference also serves the extraordinary run.
    vacuum = _run_slab(frequency, _ORDINARY_POL, with_crystal=False)
    xs, vac_line = _projected_axis_line(vacuum, _ORDINARY_POL)
    window = _fit_window(xs)
    slope = np.polyfit(xs[window], np.unwrap(np.angle(vac_line[window])), 1)[0]
    assert abs(abs(slope) / k0 - 1.0) < 0.1  # forward plane wave travels near c
    signed_k = np.sign(slope) * k0

    results = {}
    for label, polarization, eps_axis in (
        ("ordinary", _ORDINARY_POL, _EPS_ORDINARY),
        ("extraordinary", _EXTRAORDINARY_POL, _EPS_EXTRAORDINARY),
    ):
        crystal = _run_slab(frequency, polarization, with_crystal=True)
        _, crystal_line = _projected_axis_line(crystal, polarization)
        # Per-x transmitted/incident phase in the post-slab window is ~constant
        # and equals signed_k * (n - 1) * L (kept below pi by the slab thickness);
        # averaging the ratio phase over the window suppresses the standing wave.
        extra_phase = float(np.mean(np.angle(crystal_line[window] / vac_line[window])))
        n_measured = 1.0 + extra_phase / (signed_k * _SLAB_THICKNESS)

        n_analytic = _analytic_index(eps_axis, frequency).real
        rel_error = abs(n_measured - n_analytic) / n_analytic
        results[label] = (n_measured, n_analytic, rel_error)

        # Polarization purity: an eigen-polarization stays linearly polarized, so
        # the orthogonal transverse projection is small. A wrong off-diagonal
        # coupling would rotate the polarization and break this.
        orthogonal = (0.0, -polarization[2], polarization[1])
        _, cross_line = _projected_axis_line(crystal, orthogonal)
        cross_ratio = np.abs(cross_line[window]).mean() / np.abs(crystal_line[window]).mean()
        assert cross_ratio < 0.1

        assert rel_error < 0.02, (
            f"{label} n at {frequency:.2e} Hz: measured {n_measured:.4f} vs analytic "
            f"{n_analytic:.4f} ({rel_error * 100:.2f}%)"
        )

    # Birefringence (n_e - n_o) reproduces the analytic value within 2%.
    n_o_measured, n_o_analytic, _ = results["ordinary"]
    n_e_measured, n_e_analytic, _ = results["extraordinary"]
    delta_measured = n_e_measured - n_o_measured
    delta_analytic = n_e_analytic - n_o_analytic
    assert abs(delta_measured - delta_analytic) / delta_analytic < 0.02

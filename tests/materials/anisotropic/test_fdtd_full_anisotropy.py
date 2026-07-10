import pytest
import torch
import torch.nn.functional as F

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene


def _rotated_uniaxial_tensor(eps_xx, eps_ordinary, eps_extraordinary):
    """Uniaxial permittivity with principal axes rotated 45 degrees about x.

    The extraordinary axis is u = (y+z)/sqrt(2) and the ordinary axis is
    v = (-y+z)/sqrt(2), giving eps_yy = eps_zz = (e_o + e_e)/2 and
    eps_yz = (e_e - e_o)/2.
    """
    mean = 0.5 * (eps_ordinary + eps_extraordinary)
    delta = 0.5 * (eps_extraordinary - eps_ordinary)
    return mw.Tensor3x3(((eps_xx, 0.0, 0.0), (0.0, mean, delta), (0.0, delta, mean)))


# ---------------------------------------------------------------------------
# Construction / compile-level behavior (CPU)
# ---------------------------------------------------------------------------


def test_tensor3x3_epsilon_requires_symmetry():
    with pytest.raises(ValueError, match="symmetric"):
        mw.Material(epsilon_tensor=mw.Tensor3x3(((2.0, 0.9, 0.0), (0.1, 2.0, 0.0), (0.0, 0.0, 2.0))))


def test_tensor3x3_epsilon_requires_positive_definite():
    with pytest.raises(ValueError, match="positive-definite"):
        mw.Material(epsilon_tensor=mw.Tensor3x3(((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0))))


def test_tensor3x3_epsilon_rejects_conductivity():
    tensor = _rotated_uniaxial_tensor(2.0, 2.0, 3.0)
    with pytest.raises(NotImplementedError, match="conductivity"):
        mw.Material(epsilon_tensor=tensor, sigma_e=0.5)


def test_tensor3x3_mu_and_sigma_tensors_stay_rejected():
    diagonal = mw.Tensor3x3(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)))
    with pytest.raises(NotImplementedError, match="mu_tensor"):
        mw.Material(mu_tensor=diagonal)
    with pytest.raises(NotImplementedError, match="sigma_e_tensor"):
        mw.Material(sigma_e_tensor=diagonal)


def _build_cpu_scene(material, **scene_kwargs):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
        **scene_kwargs,
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)), material=material)
    )
    return scene


def test_compiled_model_carries_offdiagonal_permittivity():
    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0))
    model = prepare_scene(_build_cpu_scene(material)).compile_materials()

    from witwin.maxwell.compiler.materials import material_model_has_full_anisotropy

    offdiag = model["eps_offdiag_components"]
    assert sorted(offdiag.keys()) == ["xy", "xz", "yz"]
    assert float(offdiag["yz"].max()) == pytest.approx(0.5)
    assert float(offdiag["xy"].abs().max()) == 0.0
    assert float(offdiag["xz"].abs().max()) == 0.0
    assert material_model_has_full_anisotropy(model)
    assert float(model["eps_components"]["y"].max()) == pytest.approx(2.5)


def test_diagonal_tensor3x3_compiles_identically_to_diagonal_tensor3():
    full = mw.Material(
        epsilon_tensor=mw.Tensor3x3(((1.0, 0.0, 0.0), (0.0, 2.25, 0.0), (0.0, 0.0, 7.0)))
    )
    diagonal = mw.Material(epsilon_tensor=mw.DiagonalTensor3(1.0, 2.25, 7.0))

    from witwin.maxwell.compiler.materials import material_model_has_full_anisotropy

    full_model = prepare_scene(_build_cpu_scene(full)).compile_materials()
    diagonal_model = prepare_scene(_build_cpu_scene(diagonal)).compile_materials()

    assert not material_model_has_full_anisotropy(full_model)
    for axis in ("x", "y", "z"):
        assert torch.equal(full_model["eps_components"][axis], diagonal_model["eps_components"][axis])
    for pair in ("xy", "xz", "yz"):
        assert float(full_model["eps_offdiag_components"][pair].abs().max()) == 0.0


def test_polarized_subpixel_rejects_full_tensor():
    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0))
    scene = _build_cpu_scene(
        material, subpixel_samples=mw.SubpixelSpec(samples=(2, 2, 2), averaging="polarized")
    )
    with pytest.raises(NotImplementedError, match="Polarized"):
        prepare_scene(scene).compile_materials()


def test_fdfd_rejects_full_anisotropy():
    from witwin.maxwell.fdfd.solver import _validate_supported_fdfd_materials

    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0))
    scene = _build_cpu_scene(material)
    with pytest.raises(NotImplementedError, match="off-diagonal"):
        _validate_supported_fdfd_materials(scene)


def test_adjoint_bridge_accepts_full_anisotropy():
    """Full off-diagonal epsilon tensors are adjoint-supported since P5.1."""
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0))
    scene = _build_cpu_scene(material)
    assert _unsupported_adjoint_medium(scene) is None


# ---------------------------------------------------------------------------
# Solver-level behavior (CUDA)
# ---------------------------------------------------------------------------


def _build_plane_wave_scene(
    frequency,
    *,
    spacing,
    polarization=(0.0, 0.0, 1.0),
    boundary=None,
):
    return mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=boundary or mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=polarization,
                source_time=mw.CW(frequency=frequency, amplitude=80.0),
                name="pw",
            )
        ],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_diagonal_tensor3x3_solver_matches_diagonal_path():
    frequency = 1.0e9
    solvers = []
    for material in (
        mw.Material(epsilon_tensor=mw.Tensor3x3(((1.0, 0.0, 0.0), (0.0, 2.25, 0.0), (0.0, 0.0, 7.0)))),
        mw.Material(epsilon_tensor=mw.DiagonalTensor3(1.0, 2.25, 7.0)),
    ):
        scene = _build_plane_wave_scene(frequency, spacing=0.04)
        scene.add_structure(
            mw.Structure(geometry=Box(position=(0.1, 0.0, 0.0), size=(0.2, 0.4, 0.4)), material=material)
        )
        solvers.append(mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver)

    full, diagonal = solvers
    assert not full.full_aniso_enabled
    assert torch.equal(full.cex_curl, diagonal.cex_curl)
    assert torch.equal(full.cey_curl, diagonal.cey_curl)
    assert torch.equal(full.cez_curl, diagonal.cez_curl)
    assert torch.equal(full.cex_decay, diagonal.cex_decay)


def _pad(tensor, pad):
    return F.pad(tensor, pad)


def _reference_curls(solver):
    """curl(H) collocated at Ey/Ez/Ex edges, zero where the stencil leaves the grid."""
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    hx = solver.Hx.double()
    hy = solver.Hy.double()
    hz = solver.Hz.double()
    inv_dx = solver.inv_dx_e.double()
    inv_dy = solver.inv_dy_e.double()
    inv_dz = solver.inv_dz_e.double()

    curl_x = torch.zeros((nx - 1, ny, nz), dtype=torch.float64, device=hx.device)
    curl_x[:, 1 : ny - 1, 1 : nz - 1] = (
        hz[:, 1 : ny - 1, 1 : nz - 1] - hz[:, 0 : ny - 2, 1 : nz - 1]
    ) * inv_dy[1 : ny - 1].view(1, -1, 1) - (
        hy[:, 1 : ny - 1, 1 : nz - 1] - hy[:, 1 : ny - 1, 0 : nz - 2]
    ) * inv_dz[1 : nz - 1].view(1, 1, -1)

    curl_y = torch.zeros((nx, ny - 1, nz), dtype=torch.float64, device=hx.device)
    curl_y[1 : nx - 1, :, 1 : nz - 1] = (
        hx[1 : nx - 1, :, 1 : nz - 1] - hx[1 : nx - 1, :, 0 : nz - 2]
    ) * inv_dz[1 : nz - 1].view(1, 1, -1) - (
        hz[1 : nx - 1, :, 1 : nz - 1] - hz[0 : nx - 2, :, 1 : nz - 1]
    ) * inv_dx[1 : nx - 1].view(-1, 1, 1)

    curl_z = torch.zeros((nx, ny, nz - 1), dtype=torch.float64, device=hx.device)
    curl_z[1 : nx - 1, 1 : ny - 1, :] = (
        hy[1 : nx - 1, 1 : ny - 1, :] - hy[0 : nx - 2, 1 : ny - 1, :]
    ) * inv_dx[1 : nx - 1].view(-1, 1, 1) - (
        hx[1 : nx - 1, 1 : ny - 1, :] - hx[1 : nx - 1, 0 : ny - 2, :]
    ) * inv_dy[1 : ny - 1].view(1, -1, 1)

    return curl_x, curl_y, curl_z


def _reference_corrections(solver):
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    curl_x, curl_y, curl_z = _reference_curls(solver)

    # Ex: average curl_y over (ii in {i, i+1}, jy in {j-1, j}) and curl_z over
    # (ii in {i, i+1}, kz in {k-1, k}).
    p_y = _pad(curl_y, (0, 0, 1, 1))
    acc_y = 0.25 * (
        p_y[0 : nx - 1, 0:ny, :] + p_y[0 : nx - 1, 1 : ny + 1, :]
        + p_y[1:nx, 0:ny, :] + p_y[1:nx, 1 : ny + 1, :]
    )
    p_z = _pad(curl_z, (1, 1))
    acc_z = 0.25 * (
        p_z[0 : nx - 1, :, 0:nz] + p_z[0 : nx - 1, :, 1 : nz + 1]
        + p_z[1:nx, :, 0:nz] + p_z[1:nx, :, 1 : nz + 1]
    )
    delta_ex = solver.cex_aniso_y.double() * acc_y + solver.cex_aniso_z.double() * acc_z

    # Ey: average curl_x over (ix in {i-1, i}, jy in {j, j+1}) and curl_z over
    # (jy in {j, j+1}, kz in {k-1, k}).
    p_x = _pad(curl_x, (0, 0, 0, 0, 1, 1))
    acc_x = 0.25 * (
        p_x[0:nx, 0 : ny - 1, :] + p_x[0:nx, 1:ny, :]
        + p_x[1 : nx + 1, 0 : ny - 1, :] + p_x[1 : nx + 1, 1:ny, :]
    )
    p_z = _pad(curl_z, (1, 1))
    acc_z = 0.25 * (
        p_z[:, 0 : ny - 1, 0:nz] + p_z[:, 1:ny, 0:nz]
        + p_z[:, 0 : ny - 1, 1 : nz + 1] + p_z[:, 1:ny, 1 : nz + 1]
    )
    delta_ey = solver.cey_aniso_x.double() * acc_x + solver.cey_aniso_z.double() * acc_z

    # Ez: average curl_x over (ix in {i-1, i}, kz in {k, k+1}) and curl_y over
    # (jy in {j-1, j}, kz in {k, k+1}).
    p_x = _pad(curl_x, (0, 0, 0, 0, 1, 1))
    acc_x = 0.25 * (
        p_x[0:nx, :, 0 : nz - 1] + p_x[0:nx, :, 1:nz]
        + p_x[1 : nx + 1, :, 0 : nz - 1] + p_x[1 : nx + 1, :, 1:nz]
    )
    p_y = _pad(curl_y, (0, 0, 1, 1))
    acc_y = 0.25 * (
        p_y[:, 0:ny, 0 : nz - 1] + p_y[:, 1 : ny + 1, 0 : nz - 1]
        + p_y[:, 0:ny, 1:nz] + p_y[:, 1 : ny + 1, 1:nz]
    )
    delta_ez = solver.cez_aniso_x.double() * acc_x + solver.cez_aniso_y.double() * acc_y

    return delta_ex, delta_ey, delta_ez


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_full_aniso_correction_kernels_match_torch_reference():
    frequency = 1.0e9
    material = mw.Material(
        epsilon_tensor=mw.Tensor3x3(((2.0, 0.3, 0.2), (0.3, 2.5, 0.4), (0.2, 0.4, 3.0)))
    )
    scene = _build_plane_wave_scene(
        frequency,
        spacing=0.04,
        boundary=mw.BoundarySpec.none(),
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.1, 0.0, 0.0), size=(0.3, 0.16, 0.16)), material=material)
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver
    assert solver.full_aniso_enabled

    generator = torch.Generator(device="cuda").manual_seed(1234)
    for name in ("Hx", "Hy", "Hz"):
        field = getattr(solver, name)
        field.copy_(torch.randn(field.shape, generator=generator, device=field.device))

    expected_ex, expected_ey, expected_ez = _reference_corrections(solver)

    from witwin.maxwell.fdtd.runtime.stepping import apply_full_aniso_corrections

    before = (solver.Ex.clone(), solver.Ey.clone(), solver.Ez.clone())
    apply_full_aniso_corrections(solver)
    torch.cuda.synchronize()

    for expected, field, previous in (
        (expected_ex, solver.Ex, before[0]),
        (expected_ey, solver.Ey, before[1]),
        (expected_ez, solver.Ez, before[2]),
    ):
        delta = (field - previous).double()
        scale = float(expected.abs().max())
        assert scale > 0.0
        assert float((delta - expected).abs().max()) < 5.0e-5 * scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_rotated_uniaxial_update_matches_principal_axis_superposition():
    eps_ordinary = 2.0
    eps_extraordinary = 3.0
    inv_sqrt_two = 2.0**-0.5

    def build_uniform_solver(material):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.3, 0.3), (-0.3, 0.3))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.none(),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)),
                material=material,
            )
        )
        return mw.Simulation.fdtd(scene, frequency=1.0e9).prepare().solver

    full = build_uniform_solver(
        mw.Material(
            epsilon_tensor=_rotated_uniaxial_tensor(
                eps_ordinary,
                eps_ordinary,
                eps_extraordinary,
            )
        )
    )
    extraordinary = build_uniform_solver(mw.Material(eps_r=eps_extraordinary))
    ordinary = build_uniform_solver(mw.Material(eps_r=eps_ordinary))

    # A z-polarized curl is the equal superposition of the rotated principal
    # axes u=(y+z)/sqrt(2) and v=(-y+z)/sqrt(2).  Encode those curls through
    # x-varying Hy/Hz fields, then compare one complete electric-field update.
    profile = torch.sin(
        torch.linspace(0.0, 3.0, full.Hy.shape[0], device="cuda", dtype=full.Hy.dtype)
    )
    full.Hy.copy_(profile[:, None, None])
    extraordinary.Hy.copy_(inv_sqrt_two * profile[:, None, None])
    extraordinary.Hz.copy_(-inv_sqrt_two * profile[:, None, None])
    ordinary.Hy.copy_(inv_sqrt_two * profile[:, None, None])
    ordinary.Hz.copy_(inv_sqrt_two * profile[:, None, None])

    from witwin.maxwell.fdtd.runtime.stepping import (
        apply_full_aniso_corrections,
        update_electric_fields,
    )

    for solver in (full, extraordinary, ordinary):
        update_electric_fields(solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
        if solver.full_aniso_enabled:
            apply_full_aniso_corrections(solver)
    torch.cuda.synchronize()

    assert full.full_aniso_enabled
    for component in ("Ey", "Ez"):
        actual = getattr(full, component)[2:-2, 2:-2, 2:-2]
        expected = inv_sqrt_two * (
            getattr(extraordinary, component)[2:-2, 2:-2, 2:-2]
            + getattr(ordinary, component)[2:-2, 2:-2, 2:-2]
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=1.0e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_full_aniso_structure_inside_pml_is_rejected():
    frequency = 1.0e9
    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0))
    scene = _build_plane_wave_scene(frequency, spacing=0.04)
    # Full-cross-section slab reaches into the transverse PML layers.
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.1, 0.0, 0.0), size=(0.2, 0.8, 0.8)), material=material)
    )
    with pytest.raises(NotImplementedError, match="PML"):
        mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_full_aniso_rejects_bloch_boundaries():
    frequency = 1.0e9
    material = mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.bloch((0.1, 0.0, 0.0)),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.CW(frequency=frequency),
            )
        ],
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3)), material=material)
    )
    with pytest.raises(NotImplementedError, match="Bloch"):
        mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_full_aniso_composes_with_dispersive_structures():
    """Full anisotropy now coexists with dispersion in the same Scene.

    A separate isotropic dispersive structure and a full-anisotropic structure
    compile and prepare together: the dispersive region gets the diagonal ADE
    subtraction and the anisotropic region gets the coupled tensor inverse, and
    they do not overlap here so the off-diagonal current correction is a no-op.
    """
    frequency = 1.0e9
    scene = _build_plane_wave_scene(frequency, spacing=0.04)
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(-0.4, 0.0, 0.0), size=(0.2, 0.4, 0.4)),
            material=mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1.0e-10),
        )
    )
    # The anisotropic structure stays clear of the transverse PML (full
    # off-diagonal media may not overlap the absorber); the isotropic dispersive
    # slab above may touch it.
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.3, 0.0, 0.0), size=(0.2, 0.16, 0.16)),
            material=mw.Material(epsilon_tensor=_rotated_uniaxial_tensor(2.0, 2.0, 3.0)),
        )
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver
    assert solver.full_aniso_enabled
    assert solver.electric_dispersive_enabled
    # The diagonal ADE subtraction divides by the effective permittivity that the
    # tensor curl coefficient uses, so it is precomputed for the full-aniso path.
    assert solver._aniso_disp_inv_eps is not None
    assert solver._aniso_disp_current is not None

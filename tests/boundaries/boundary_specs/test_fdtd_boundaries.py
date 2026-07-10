import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary import (
    BOUNDARY_BLOCH,
    BOUNDARY_PEC,
    BOUNDARY_PERIODIC,
    BOUNDARY_PMC,
    BOUNDARY_PML,
    DEFAULT_CPML_CONFIG,
    combine_complex_spectral_components,
    initialize_boundary_state,
)
from witwin.maxwell.scene import prepare_scene


class DummyBoundarySolver:
    def __init__(self, boundary, absorber_type="cpml", symmetry=(None, None, None)):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))),
            grid=mw.GridSpec.uniform(1.0),
            boundary=boundary,
            symmetry=symmetry,
            device="cpu",
        )
        self.scene = prepare_scene(scene)
        self.Nx = self.scene.Nx
        self.Ny = self.scene.Ny
        self.Nz = self.scene.Nz
        self.dx = self.scene.dx
        self.dy = self.scene.dy
        self.dz = self.scene.dz
        self.dt = 1.0
        self.device = self.scene.device
        self.absorber_type = absorber_type
        self.cpml_config = dict(DEFAULT_CPML_CONFIG)
        self.eps0 = 1.0
        self.mu0 = 1.0

        self.Ex = torch.zeros((self.Nx - 1, self.Ny, self.Nz), dtype=torch.float32)
        self.Ey = torch.zeros((self.Nx, self.Ny - 1, self.Nz), dtype=torch.float32)
        self.Ez = torch.zeros((self.Nx, self.Ny, self.Nz - 1), dtype=torch.float32)
        self.Hx = torch.zeros((self.Nx, self.Ny - 1, self.Nz - 1), dtype=torch.float32)
        self.Hy = torch.zeros((self.Nx - 1, self.Ny, self.Nz - 1), dtype=torch.float32)
        self.Hz = torch.zeros((self.Nx - 1, self.Ny - 1, self.Nz), dtype=torch.float32)

        initialize_boundary_state(self)


def test_periodic_boundary_initializes_native_mode_code():
    solver = DummyBoundarySolver(mw.BoundarySpec.periodic())
    assert solver.boundary_code == BOUNDARY_PERIODIC
    assert solver.active_absorber_type == "none"
    assert solver.uses_cpml is False
    assert solver.psi_ex_y is None


def test_bloch_boundary_initializes_complex_fields_and_phase():
    solver = DummyBoundarySolver(mw.BoundarySpec.bloch((np.pi / 8.0, 0.0, 0.0)))
    assert solver.boundary_code == BOUNDARY_BLOCH
    assert solver.complex_fields_enabled is True
    assert solver.Ex_imag.shape == solver.Ex.shape
    expected_phase = np.exp(1j * (np.pi / 8.0) * 4.0)
    assert np.isclose(solver.boundary_phase_cos[0], expected_phase.real, atol=1e-6)
    assert np.isclose(solver.boundary_phase_sin[0], expected_phase.imag, atol=1e-6)


def test_pec_and_pmc_boundaries_map_to_distinct_native_codes():
    pec_solver = DummyBoundarySolver(mw.BoundarySpec.pec())
    pmc_solver = DummyBoundarySolver(mw.BoundarySpec.pmc())
    assert pec_solver.boundary_code == BOUNDARY_PEC
    assert pmc_solver.boundary_code == BOUNDARY_PMC


def test_pml_boundary_preserves_absorber_mode_code():
    solver = DummyBoundarySolver(mw.BoundarySpec.pml(num_layers=4, strength=1.0), absorber_type="cpml")
    assert solver.boundary_code == BOUNDARY_PML
    assert solver.active_absorber_type == "cpml"
    assert solver.uses_cpml is True
    assert solver.psi_ex_y is not None


def test_mixed_boundary_initializes_per_face_codes_and_cpml_state():
    solver = DummyBoundarySolver(
        mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            y="periodic",
            z=("pec", "pmc"),
        ),
        absorber_type="cpml",
    )
    assert solver.boundary_kind == "mixed"
    assert solver.boundary_code is None
    assert solver.boundary_x_low_code == BOUNDARY_PML
    assert solver.boundary_x_high_code == BOUNDARY_PML
    assert solver.boundary_y_low_code == BOUNDARY_PERIODIC
    assert solver.boundary_y_high_code == BOUNDARY_PERIODIC
    assert solver.boundary_z_low_code == BOUNDARY_PEC
    assert solver.boundary_z_high_code == BOUNDARY_PMC
    assert solver.active_absorber_type == "cpml"
    assert solver.uses_cpml is True
    assert solver.has_pml_faces is True
    assert solver.has_pec_faces is True


def test_mixed_bloch_boundaries_initialize_complex_fields_and_cpml_state():
    solver = DummyBoundarySolver(
        mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            bloch_wavevector=(np.pi / 8.0, 0.0, 0.0),
        )
    )
    assert solver.boundary_kind == "mixed"
    assert solver.boundary_code is None
    assert solver.boundary_x_low_code == BOUNDARY_BLOCH
    assert solver.boundary_x_high_code == BOUNDARY_BLOCH
    assert solver.boundary_y_low_code == BOUNDARY_PML
    assert solver.boundary_y_high_code == BOUNDARY_PML
    assert solver.complex_fields_enabled is True
    assert solver.Ex_imag.shape == solver.Ex.shape
    assert solver.active_absorber_type == "cpml"
    assert solver.uses_cpml is True
    assert solver.psi_ex_y is not None
    expected_phase = np.exp(1j * (np.pi / 8.0) * 4.0)
    assert np.isclose(solver.boundary_phase_cos[0], expected_phase.real, atol=1e-6)
    assert np.isclose(solver.boundary_phase_sin[0], expected_phase.imag, atol=1e-6)


def test_symmetry_overrides_only_low_faces():
    solver = DummyBoundarySolver(
        mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        absorber_type="cpml",
        symmetry=("PMC", None, "PEC"),
    )
    assert solver.boundary_code == BOUNDARY_PML
    assert solver.boundary_x_low_code == BOUNDARY_PMC
    assert solver.boundary_x_high_code == BOUNDARY_PML
    assert solver.boundary_z_low_code == BOUNDARY_PEC
    assert solver.boundary_z_high_code == BOUNDARY_PML
    assert solver.has_pec_faces is True


def test_complex_spectral_components_are_combined_correctly():
    combined = combine_complex_spectral_components(1.0, 2.0, 3.0, 4.0)
    assert combined == complex(-3.0, 5.0)


@pytest.fixture(scope="module")
def cuda_fdtd_solver():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for FDTD")

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.3),
        boundary=mw.BoundarySpec.periodic(),
        device="cuda",
    )
    return mw.Simulation.fdtd(scene, frequencies=[1e9]).prepare().solver


def _apply_expected_periodic_projection(field, axis):
    if axis == 0:
        average = 0.5 * (field[0, :, :] + field[-1, :, :])
        field[0, :, :] = average
        field[-1, :, :] = average
        return

    if axis == 1:
        average = 0.5 * (field[:, 0, :] + field[:, -1, :])
        field[:, 0, :] = average
        field[:, -1, :] = average
        return

    average = 0.5 * (field[:, :, 0] + field[:, :, -1])
    field[:, :, 0] = average
    field[:, :, -1] = average


def _apply_expected_bloch_projection(real_field, imag_field, axis, phase):
    field = real_field + 1j * imag_field
    if axis == 0:
        projected_low = 0.5 * (field[0, :, :] + np.conj(phase) * field[-1, :, :])
        projected_high = phase * projected_low
        field[0, :, :] = projected_low
        field[-1, :, :] = projected_high
    elif axis == 1:
        projected_low = 0.5 * (field[:, 0, :] + np.conj(phase) * field[:, -1, :])
        projected_high = phase * projected_low
        field[:, 0, :] = projected_low
        field[:, -1, :] = projected_high
    else:
        projected_low = 0.5 * (field[:, :, 0] + np.conj(phase) * field[:, :, -1])
        projected_high = phase * projected_low
        field[:, :, 0] = projected_low
        field[:, :, -1] = projected_high
    real_field[...] = field.real
    imag_field[...] = field.imag


def _apply_expected_pec_clamp(field, axis_a, axis_b):
    for axis in (axis_a, axis_b):
        if axis == 0:
            field[0, :, :] = 0.0
            field[-1, :, :] = 0.0
        elif axis == 1:
            field[:, 0, :] = 0.0
            field[:, -1, :] = 0.0
        else:
            field[:, :, 0] = 0.0
            field[:, :, -1] = 0.0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_generic_periodic_projection_kernel_matches_expected_faces(cuda_fdtd_solver, axis):
    field = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    expected = field.detach().cpu().numpy().copy()
    _apply_expected_periodic_projection(expected, axis)

    cuda_fdtd_solver.fdtd_module.projectPeriodicBoundary3D(
        field=field,
        axis=axis,
    ).launchRaw(
        blockSize=cuda_fdtd_solver.kernel_block_size,
        gridSize=cuda_fdtd_solver._compute_face_launch_shape(field, axis),
    )
    torch.cuda.synchronize()

    assert np.allclose(field.detach().cpu().numpy(), expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_generic_bloch_projection_kernel_matches_expected_phase(cuda_fdtd_solver, axis):
    real_field = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    imag_field = torch.arange(24, 48, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    expected_real = real_field.detach().cpu().numpy().copy()
    expected_imag = imag_field.detach().cpu().numpy().copy()
    phase = np.exp(1j * np.pi / 3.0)
    _apply_expected_bloch_projection(expected_real, expected_imag, axis, phase)

    cuda_fdtd_solver.fdtd_module.projectBlochBoundary3D(
        fieldReal=real_field,
        fieldImag=imag_field,
        axis=axis,
        phaseCos=float(phase.real),
        phaseSin=float(phase.imag),
    ).launchRaw(
        blockSize=cuda_fdtd_solver.kernel_block_size,
        gridSize=cuda_fdtd_solver._compute_face_launch_shape(real_field, axis),
    )
    torch.cuda.synchronize()

    assert np.allclose(real_field.detach().cpu().numpy(), expected_real)
    assert np.allclose(imag_field.detach().cpu().numpy(), expected_imag)


@pytest.mark.parametrize(
    ("axis_a", "axis_b"),
    [(1, 2), (0, 2), (0, 1)],
)
def test_generic_pec_clamp_kernel_zeroes_selected_boundary_axes(cuda_fdtd_solver, axis_a, axis_b):
    field = torch.arange(60, device="cuda", dtype=torch.float32).reshape(3, 4, 5)
    expected = field.detach().cpu().numpy().copy()
    _apply_expected_pec_clamp(expected, axis_a, axis_b)

    cuda_fdtd_solver.fdtd_module.clampPecBoundary3D(
        field=field,
        axisA=axis_a,
        axisB=axis_b,
    ).launchRaw(
        blockSize=cuda_fdtd_solver.kernel_block_size,
        gridSize=cuda_fdtd_solver._compute_linear_launch_shape(field.numel()),
    )
    torch.cuda.synchronize()

    assert np.allclose(field.detach().cpu().numpy(), expected)

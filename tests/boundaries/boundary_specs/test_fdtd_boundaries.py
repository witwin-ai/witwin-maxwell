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
        )
        self.scene = prepare_scene(scene, device="cpu")
        self.Nx = self.scene.Nx
        self.Ny = self.scene.Ny
        self.Nz = self.scene.Nz
        self.dx = self.scene.dx
        self.dy = self.scene.dy
        self.dz = self.scene.dz
        self.dt = 1.0
        self.device = self.scene.device
        self.absorber_type = absorber_type
        self.cpml = mw.fdtd.CPMLConfig()
        self.eps0 = 1.0
        self.mu0 = 1.0
        self.c = 1.0
        # Band floor for the complex fixed-k CFS-CPML shift; set by Simulation.prepare().
        self._requested_port_frequencies = (0.1,)

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
    )
    return mw.Simulation.fdtd(scene, frequencies=[1e9]).prepare().solver


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
    ).launchRaw()
    torch.cuda.synchronize()

    assert np.allclose(field.detach().cpu().numpy(), expected)

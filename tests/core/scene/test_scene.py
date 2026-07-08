"""
Unit tests for the Scene class with correctness verification and PNG output.

Each geometry test verifies:
  - permittivity values inside the object == eps_r
  - permittivity values outside the object == 1.0 (background)
  - geometry record stored correctly
  - saves a Z-mid cross-section PNG for visual inspection

Usage:
    pytest tests/core/scene/test_scene.py -v
    python tests/core/scene/test_scene.py # standalone with all PNGs
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

TESTS_ROOT = Path(__file__).resolve()
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if TESTS_ROOT.name != "tests":
    raise RuntimeError("Unable to locate tests root directory.")

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene

OUTPUT_DIR = str(TESTS_ROOT / "test_output" / "scene")


def make_scene(
    *,
    domain_half=1.0,
    resolution=0.05,
    device='cpu',
    grid_spacing=None,
    boundary=None,
    lazy_meshgrid=True,
    symmetry=None,
):
    spacing = resolution if grid_spacing is None else grid_spacing
    if isinstance(spacing, (tuple, list)):
        grid = mw.GridSpec.anisotropic(*spacing)
    else:
        grid = mw.GridSpec.uniform(spacing)
    return mw.Scene(
        domain=mw.Domain(bounds=((-domain_half, domain_half), (-domain_half, domain_half), (-domain_half, domain_half))),
        grid=grid,
        boundary=boundary or mw.BoundarySpec.none(),
        device=device,
        lazy_meshgrid=lazy_meshgrid,
        symmetry=symmetry,
    )


def add_box(scene, *, center, size, eps_r, mu_r=1.0, rotation=None):
    scene.add_structure(
        mw.Structure(
            name=f"box_{len(scene.structures)}",
            geometry=mw.Box(position=center, size=size, rotation=rotation),
            material=mw.Material(eps_r=eps_r, mu_r=mu_r),
        )
    )


def add_sphere(scene, *, center, radius, eps_r, mu_r=1.0):
    scene.add_structure(
        mw.Structure(
            name=f"sphere_{len(scene.structures)}",
            geometry=mw.Sphere(position=center, radius=radius),
            material=mw.Material(eps_r=eps_r, mu_r=mu_r),
        )
    )


def add_cylinder(scene, *, center, radius, height, eps_r, mu_r=1.0, axis='z', rotation=None):
    scene.add_structure(
        mw.Structure(
            name=f"cylinder_{len(scene.structures)}",
            geometry=mw.Cylinder(position=center, radius=radius, height=height, axis=axis, rotation=rotation),
            material=mw.Material(eps_r=eps_r, mu_r=mu_r),
        )
    )


def set_point_source(scene, *, position, width=0.02, amplitude=1.0, polarization=(0, 0, 1), name="src", frequency=1e9):
    scene.sources = [
        mw.PointDipole(
            position=position,
            polarization=polarization,
            width=width,
            source_time=mw.CW(frequency=frequency, amplitude=amplitude),
            name=name,
        )
    ]


def add_vector_source(
    scene,
    *,
    position,
    width=0.02,
    polarization_vector=(1, 0, 0),
    amplitude=1.0,
    name=None,
    frequency=1e9,
):
    scene.add_source(
        mw.PointDipole(
            position=position,
            polarization=polarization_vector,
            width=width,
            source_time=mw.CW(frequency=frequency, amplitude=amplitude),
            name=name or f"src_{len(scene.sources)}",
        )
    )


def save_permittivity_slices(scene, name, axes=('x', 'y', 'z')):
    """Save permittivity cross-section PNGs through the domain center."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prepared_scene = prepare_scene(scene)
    eps = prepared_scene.permittivity.cpu().numpy()
    x = prepared_scene.x.cpu().numpy()
    y = prepared_scene.y.cpu().numpy()
    z = prepared_scene.z.cpu().numpy()

    mid = {'x': len(x) // 2, 'y': len(y) // 2, 'z': len(z) // 2}

    for axis in axes:
        fig, ax = plt.subplots(figsize=(6, 5))
        if axis == 'x':
            sl = eps[mid['x'], :, :]
            extent = [y[0], y[-1], z[0], z[-1]]
            ax.set_xlabel('Y [m]'); ax.set_ylabel('Z [m]')
        elif axis == 'y':
            sl = eps[:, mid['y'], :]
            extent = [x[0], x[-1], z[0], z[-1]]
            ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]')
        else:
            sl = eps[:, :, mid['z']]
            extent = [x[0], x[-1], y[0], y[-1]]
            ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')

        im = ax.imshow(sl.T, origin='lower', extent=extent, cmap='viridis',
                       vmin=1.0, vmax=max(eps.max(), 1.01))
        ax.set_title(f"{name} - eps_r @ {axis}=0")
        plt.colorbar(im, ax=ax, label='eps_r')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'{name}_{axis}.png')
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  saved: {path}")


def sample_eps_at(scene, point):
    """Sample permittivity at a physical coordinate by finding nearest grid cell."""
    prepared_scene = prepare_scene(scene)
    x = prepared_scene.x.cpu().numpy()
    y = prepared_scene.y.cpu().numpy()
    z = prepared_scene.z.cpu().numpy()
    ix = np.argmin(np.abs(x - point[0]))
    iy = np.argmin(np.abs(y - point[1]))
    iz = np.argmin(np.abs(z - point[2]))
    return prepared_scene.permittivity[ix, iy, iz].item()


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

class TestGridConstruction:

    def test_grid_dimensions(self):
        scene = make_scene(resolution=0.1, device='cpu')
        prepared_scene = prepare_scene(scene)
        assert prepared_scene.Nx == len(prepared_scene.x)
        assert prepared_scene.Ny == len(prepared_scene.y)
        assert prepared_scene.Nz == len(prepared_scene.z)
        assert prepared_scene.N_total == prepared_scene.Nx * prepared_scene.Ny * prepared_scene.Nz

    def test_yee_component_sizes(self):
        scene = make_scene(resolution=0.1, device='cpu')
        prepared_scene = prepare_scene(scene)
        assert prepared_scene.Nx_ex == prepared_scene.Nx - 1
        assert prepared_scene.Ny_ey == prepared_scene.Ny - 1
        assert prepared_scene.Nz_ez == prepared_scene.Nz - 1
        assert prepared_scene.N_vector_total == prepared_scene.N_ex + prepared_scene.N_ey + prepared_scene.N_ez

    def test_resolution_alias(self):
        s1 = make_scene(grid_spacing=0.05, device='cpu')
        s2 = make_scene(resolution=0.05, device='cpu')
        assert prepare_scene(s1).Nx == prepare_scene(s2).Nx and s1.dx == s2.dx

    def test_lazy_meshgrid(self):
        scene = make_scene(resolution=0.1, device='cpu', lazy_meshgrid=True)
        prepared_scene = prepare_scene(scene)
        assert prepared_scene._xx is None
        _ = prepared_scene.xx
        assert prepared_scene._xx is not None
        assert prepared_scene._xx.shape == (prepared_scene.Nx, prepared_scene.Ny, prepared_scene.Nz)

    def test_anisotropic_spacing(self):
        scene = make_scene(grid_spacing=(0.1, 0.2, 0.05), device='cpu')
        assert abs(scene.dx - 0.1) < 1e-9
        assert abs(scene.dy - 0.2) < 1e-9
        assert abs(scene.dz - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Geometry primitives - correctness + PNG
# ---------------------------------------------------------------------------

class TestBox:

    def test_box_values(self):
        """Box interior should be eps_r=4, exterior should be 1."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_box(scene, center=[0, 0, 0], size=[0.5, 0.5, 0.5], eps_r=4.0)

        # Interior point
        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(4.0, abs=1e-2)
        # Still interior (near edge)
        assert sample_eps_at(scene, [0.2, 0.2, 0.2]) == pytest.approx(4.0, abs=1e-2)
        # Exterior
        assert sample_eps_at(scene, [0.5, 0.5, 0.5]) == pytest.approx(1.0, abs=1e-2)
        assert sample_eps_at(scene, [0.8, 0, 0]) == pytest.approx(1.0, abs=1e-2)
        # Geometry record
        assert len(scene.structures) == 1
        assert scene.structures[0].geometry.kind == 'box'
        assert scene.structures[0].material.eps_r == 4.0

        save_permittivity_slices(scene, 'box')

    def test_box_offset(self):
        """Box at offset position - verify center and far corner."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_box(scene, center=[0.5, 0.5, 0], size=[0.3, 0.3, 0.3], eps_r=3.0)

        assert sample_eps_at(scene, [0.5, 0.5, 0]) == pytest.approx(3.0, abs=1e-2)
        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(1.0, abs=1e-2)

        save_permittivity_slices(scene, 'box_offset')

    def test_box_volume_fraction(self):
        """Check that occupancy integrates to the expected box volume."""
        res = 0.05
        scene = make_scene(resolution=res, device='cpu')
        add_box(scene, center=[0, 0, 0], size=[0.5, 0.5, 0.5], eps_r=2.0)
        occupancy = prepare_scene(scene).permittivity - 1.0
        estimated_volume = occupancy.sum().item() * res**3
        expected_volume = 0.5**3
        assert abs(estimated_volume - expected_volume) / expected_volume < 0.1, \
            f"Volume mismatch: {estimated_volume} vs expected ~{expected_volume}"


class TestSphere:

    def test_sphere_values(self):
        """Sphere interior = eps_r, exterior = 1."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_sphere(scene, center=[0, 0, 0], radius=0.3, eps_r=6.0)

        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(6.0, abs=1e-2)
        assert sample_eps_at(scene, [0.2, 0, 0]) == pytest.approx(6.0, abs=1e-2)
        assert sample_eps_at(scene, [0.5, 0, 0]) == pytest.approx(1.0, abs=1e-2)
        assert len(scene.structures) == 1
        assert scene.structures[0].geometry.kind == 'sphere'

        save_permittivity_slices(scene, 'sphere')

    def test_sphere_volume_fraction(self):
        """Check occupancy integrates to the expected sphere volume."""
        res = 0.05
        r = 0.3
        scene = make_scene(resolution=res, device='cpu')
        add_sphere(scene, center=[0, 0, 0], radius=r, eps_r=5.0)
        occupancy = (prepare_scene(scene).permittivity - 1.0) / 4.0
        estimated_volume = occupancy.sum().item() * res**3
        expected_volume = (4 / 3) * np.pi * r**3
        assert abs(estimated_volume - expected_volume) / expected_volume < 0.15, \
            f"Volume mismatch: {estimated_volume} vs expected ~{expected_volume:.4f}"


class TestCylinder:

    def test_cylinder_z_axis(self):
        """Z-axis cylinder: center inside, top/bottom outside."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_cylinder(scene, center=[0, 0, 0], radius=0.2, height=0.6, eps_r=5.0, axis='z')

        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(5.0, abs=1e-2)
        assert sample_eps_at(scene, [0.15, 0, 0]) == pytest.approx(5.0, abs=1e-2)
        # Outside radius
        assert sample_eps_at(scene, [0.3, 0, 0]) == pytest.approx(1.0, abs=1e-2)
        # Outside height
        assert sample_eps_at(scene, [0, 0, 0.5]) == pytest.approx(1.0, abs=1e-2)

        save_permittivity_slices(scene, 'cylinder_z')

    def test_cylinder_x_axis(self):
        """X-axis cylinder."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_cylinder(scene, center=[0, 0, 0], radius=0.2, height=0.8, eps_r=4.0, axis='x')

        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(4.0, abs=1e-2)
        # Along x-axis (inside height)
        assert sample_eps_at(scene, [0.3, 0, 0]) == pytest.approx(4.0, abs=1e-2)
        # Outside radius in y
        assert sample_eps_at(scene, [0, 0.3, 0]) == pytest.approx(1.0, abs=1e-2)

        save_permittivity_slices(scene, 'cylinder_x')


class TestMultipleObjects:

    def test_two_boxes(self):
        """Two separate boxes with different eps_r."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_box(scene, center=[-0.5, 0, 0], size=[0.3, 0.3, 0.3], eps_r=2.0)
        add_box(scene, center=[0.5, 0, 0], size=[0.3, 0.3, 0.3], eps_r=4.0)

        assert sample_eps_at(scene, [-0.5, 0, 0]) == pytest.approx(2.0, abs=1e-2)
        assert sample_eps_at(scene, [0.5, 0, 0]) == pytest.approx(4.0, abs=1e-2)
        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(1.0, abs=1e-2)
        assert len(scene.structures) == 2

        save_permittivity_slices(scene, 'two_boxes')

    def test_box_and_sphere(self):
        """Box + sphere, sphere overwrites overlapping region."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_box(scene, center=[0, 0, 0], size=[0.6, 0.6, 0.6], eps_r=3.0)
        add_sphere(scene, center=[0, 0, 0], radius=0.2, eps_r=8.0)

        # Center is inside sphere (last write wins)
        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(8.0, abs=1e-2)
        # Inside box but outside sphere
        assert sample_eps_at(scene, [0.25, 0.25, 0]) == pytest.approx(3.0, abs=1e-2)
        # Outside both
        assert sample_eps_at(scene, [0.5, 0.5, 0.5]) == pytest.approx(1.0, abs=1e-2)

        save_permittivity_slices(scene, 'box_and_sphere')

    def test_complex_scene(self):
        """Box + sphere + cylinder - realistic test scene."""
        scene = make_scene(resolution=0.05, device='cpu')
        add_box(scene, center=[0.3, 0.3, 0], size=[0.4, 0.4, 0.4], eps_r=4.0)
        add_sphere(scene, center=[-0.3, -0.3, 0], radius=0.25, eps_r=6.0)
        add_cylinder(scene, center=[0, 0, 0], radius=0.1, height=1.5, eps_r=2.5, axis='z')

        assert sample_eps_at(scene, [0.3, 0.3, 0]) == pytest.approx(4.0, abs=1e-2)
        assert sample_eps_at(scene, [-0.3, -0.3, 0]) == pytest.approx(6.0, abs=1e-2)
        # Cylinder at origin (overwrites box if overlapping)
        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(2.5, abs=1e-2)
        assert len(scene.structures) == 3

        save_permittivity_slices(scene, 'complex_scene')


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class TestSources:

    def test_set_source(self):
        scene = make_scene(resolution=0.1, device='cpu')
        set_point_source(scene, position=[0, 0, 0], width=0.02, amplitude=1.0, polarization=[0, 0, 1])
        assert scene.sources[0].position == (0.0, 0.0, 0.0)
        assert scene.sources[0].polarization == (0.0, 0.0, 1.0)
        assert scene.sources[0].source_time == mw.CW(frequency=1e9, amplitude=1.0)

    def test_add_vector_source(self):
        scene = make_scene(resolution=0.1, device='cpu')
        add_vector_source(scene, position=[0.1, 0.2, 0.3], width=0.03, polarization_vector=(1, 0, 0))
        assert len(scene.sources) == 1
        assert scene.sources[0].position == (0.1, 0.2, 0.3)
        assert scene.sources[0].width == 0.03

    def test_multiple_sources(self):
        scene = make_scene(resolution=0.1, device='cpu')
        add_vector_source(scene, position=[0, 0, 0], polarization_vector=(1, 0, 0))
        add_vector_source(scene, position=[0.5, 0, 0], polarization_vector=(0, 1, 0))
        assert len(scene.sources) == 2

    def test_clear_sources(self):
        scene = make_scene(resolution=0.1, device='cpu')
        add_vector_source(scene, position=[0, 0, 0], polarization_vector=(1, 0, 0))
        scene.sources = []
        assert len(scene.sources) == 0


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

class TestBoundary:

    def test_pml_setup(self):
        scene = make_scene(resolution=0.1, device='cpu', boundary=mw.BoundarySpec.pml(num_layers=10, strength=1e6))
        assert scene.boundary_type == 'PML'
        assert scene.pml_thickness == 10
        assert scene.pml_strength == 1e6

    def test_additional_boundary_factories_are_exposed(self):
        assert mw.BoundarySpec.periodic().kind == "periodic"
        assert mw.BoundarySpec.pec().kind == "pec"
        assert mw.BoundarySpec.pmc().kind == "pmc"

    def test_per_face_boundary_setup_with_global_default_and_local_overrides(self):
        scene = make_scene(
            resolution=0.1,
            device='cpu',
            boundary=mw.BoundarySpec(
                kind="pml",
                num_layers=6,
                strength=2.5,
                y="periodic",
                z_high="pmc",
            ),
        )
        assert scene.boundary_type == 'Mixed'
        assert scene.boundary_face_kind("x", "low") == "pml"
        assert scene.boundary_face_kind("x", "high") == "pml"
        assert scene.boundary_face_kind("y", "low") == "periodic"
        assert scene.boundary_face_kind("y", "high") == "periodic"
        assert scene.boundary_face_kind("z", "low") == "pml"
        assert scene.boundary_face_kind("z", "high") == "pmc"
        assert scene.pml_thickness == 6
        assert scene.pml_thickness_for_face("x", "low") == 6
        assert scene.pml_thickness_for_face("y", "low") == 0
        assert scene.pml_thickness_for_face("z", "high") == 0

    def test_faces_factory_supports_axis_pairs_and_single_face_overrides(self):
        boundary = mw.BoundarySpec.faces(
            default="pml",
            num_layers=5,
            strength=1.0,
            y="periodic",
            z=("pec", "pmc"),
        )
        scene = make_scene(resolution=0.1, device='cpu', boundary=boundary)
        assert scene.boundary_type == 'Mixed'
        assert boundary.axis_kind("x") == "pml"
        assert boundary.axis_kind("y") == "periodic"
        assert boundary.axis_kind("z") == "mixed"
        assert boundary.face_kind("z", "low") == "pec"
        assert boundary.face_kind("z", "high") == "pmc"
        assert boundary.uses_kind("pml") is True
        assert boundary.uses_kind("periodic") is True
        assert boundary.uses_kind("bloch") is False

    def test_periodic_boundary_setup(self):
        scene = make_scene(resolution=0.1, device='cpu', boundary=mw.BoundarySpec.periodic())
        assert scene.boundary_type == 'Periodic'
        assert scene.pml_thickness == 0
        assert scene.pml_strength == 0.0

    def test_bloch_boundary_tracks_wavevector_and_phase(self):
        scene = make_scene(
            resolution=0.1,
            device='cpu',
            boundary=mw.BoundarySpec.bloch((np.pi / 2.0, 0.0, 0.0)),
        )
        assert scene.boundary_type == 'Bloch'
        assert scene.bloch_wavevector == (np.pi / 2.0, 0.0, 0.0)
        phase_x, phase_y, phase_z = scene.bloch_phase_factors
        assert np.isclose(phase_x.real, -1.0)
        assert np.isclose(phase_x.imag, 0.0, atol=1e-6)
        assert phase_y == 1.0 + 0.0j
        assert phase_z == 1.0 + 0.0j

    def test_mixed_boundary_tracks_bloch_phase_only_on_bloch_axes(self):
        scene = make_scene(
            resolution=0.1,
            device='cpu',
            boundary=mw.BoundarySpec.faces(
                default="pml",
                num_layers=4,
                strength=1.0,
                x="bloch",
                bloch_wavevector=(np.pi / 2.0, 0.0, 0.0),
            ),
        )
        phase_x, phase_y, phase_z = scene.bloch_phase_factors
        assert scene.boundary_type == 'Mixed'
        assert scene.bloch_wavevector == (np.pi / 2.0, 0.0, 0.0)
        assert np.isclose(phase_x.real, -1.0)
        assert np.isclose(phase_x.imag, 0.0, atol=1e-6)
        assert phase_y == 1.0 + 0.0j
        assert phase_z == 1.0 + 0.0j

    def test_mixed_bloch_boundary_accepts_auto_wavevector_marker(self):
        boundary = mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            y="bloch",
            bloch_wavevector="auto",
        )
        assert boundary.kind == "mixed"
        assert boundary.bloch_wavevector == "auto"

    def test_auto_bloch_phase_requires_solver_resolution(self):
        scene = make_scene(
            resolution=0.1,
            device="cpu",
            boundary=mw.BoundarySpec.faces(
                default="pml",
                num_layers=4,
                strength=1.0,
                x="bloch",
                bloch_wavevector="auto",
            ),
        )
        with pytest.raises(ValueError, match="Simulation.prepare"):
            _ = scene.bloch_phase_factors

    def test_auto_bloch_wavevector_accessor_requires_solver_resolution(self):
        scene = make_scene(
            resolution=0.1,
            device="cpu",
            boundary=mw.BoundarySpec.faces(
                default="pml",
                num_layers=4,
                strength=1.0,
                x="bloch",
                bloch_wavevector="auto",
            ),
        )
        with pytest.raises(ValueError, match="Simulation.prepare"):
            _ = scene.bloch_wavevector

    def test_with_faces_updates_existing_boundary_configuration(self):
        boundary = mw.BoundarySpec.pml(num_layers=4, strength=1.0).with_faces(
            y="periodic",
            z_high="pmc",
        )
        assert boundary.kind == "mixed"
        assert boundary.face_kind("x", "low") == "pml"
        assert boundary.face_kind("y", "low") == "periodic"
        assert boundary.face_kind("y", "high") == "periodic"
        assert boundary.face_kind("z", "low") == "pml"
        assert boundary.face_kind("z", "high") == "pmc"

    def test_low_face_symmetry_is_tracked_on_scene(self):
        scene = make_scene(
            resolution=0.1,
            device='cpu',
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            symmetry=("PMC", None, "PEC"),
        )
        # Bare mode strings normalize to (mode, face) pairs defaulting to "low".
        assert scene.symmetry == (("PMC", "low"), None, ("PEC", "low"))
        assert scene.has_symmetry is True


# ---------------------------------------------------------------------------
# Background default
# ---------------------------------------------------------------------------

class TestBackground:

    def test_default_is_vacuum(self):
        scene = make_scene(resolution=0.1, device='cpu')
        prepared_scene = prepare_scene(scene)
        assert torch.allclose(prepared_scene.permittivity,
                              torch.ones_like(prepared_scene.permittivity))
        assert torch.allclose(prepared_scene.permeability,
                              torch.ones_like(prepared_scene.permeability))


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

class TestDevice:

    def test_default_scene_device_is_cuda_when_available(self):
        if not torch.cuda.is_available():
            return
        scene = make_scene(resolution=0.2, device=None)
        assert prepare_scene(scene).permittivity.device.type == 'cuda'

    def test_cpu(self):
        scene = make_scene(resolution=0.2, device='cpu')
        assert prepare_scene(scene).permittivity.device.type == 'cpu'

    def test_gpu(self):
        if not torch.cuda.is_available():
            return
        scene = make_scene(resolution=0.2, device='cuda')
        assert prepare_scene(scene).permittivity.device.type == 'cuda'
        # Geometry should still work on GPU
        add_box(scene, center=[0, 0, 0], size=[0.5, 0.5, 0.5], eps_r=3.0)
        assert sample_eps_at(scene, [0, 0, 0]) == pytest.approx(3.0, abs=1e-2)

    def test_default_scene_device_requires_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="Scene requires CUDA by default"):
            make_scene(resolution=0.2, device=None)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print()

    suite_classes = [
        TestGridConstruction, TestBox, TestSphere, TestCylinder,
        TestMultipleObjects, TestSources, TestBoundary, TestBackground, TestDevice,
    ]
    passed, failed = 0, 0
    t0 = time.time()
    for cls in suite_classes:
        inst = cls()
        tests = [m for m in dir(inst) if m.startswith('test_')]
        for name in sorted(tests):
            try:
                getattr(inst, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}: {e}")
                failed += 1

    elapsed = time.time() - t0
    print(f"\n{passed} passed, {failed} failed in {elapsed:.2f}s")
    print(f"PNGs saved to: {os.path.abspath(OUTPUT_DIR)}")

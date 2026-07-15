from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import centered_cell_coords, component_coords
from witwin.maxwell.scene import prepare_scene

GRADED_NODES = np.array([0.0, 0.1, 0.25, 0.45, 0.7, 1.0], dtype=np.float64)
UNIFORM_NODES = np.linspace(0.0, 1.0, 11)


def _custom_scene(nodes=GRADED_NODES, boundary=None, device="cpu"):
    lo, hi = float(nodes[0]), float(nodes[-1])
    return mw.Scene(
        domain=mw.Domain(bounds=((lo, hi), (lo, hi), (lo, hi))),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=boundary or mw.BoundarySpec.none(),
        device=device,
    )


def test_custom_accepts_list_numpy_and_torch_inputs():
    as_list = [0.0, 0.1, 0.25, 0.45]
    as_numpy = np.asarray(as_list, dtype=np.float32)
    as_torch = torch.tensor(as_list, dtype=torch.float64)

    grid = mw.GridSpec.custom(as_list, as_numpy, as_torch)

    for coords in (grid.x_coords, grid.y_coords, grid.z_coords):
        assert coords.dtype == np.float64
        assert coords.flags.writeable is False
        np.testing.assert_allclose(coords, as_list, atol=1e-7)

    # Stored coordinates are copies of the caller's data.
    source = np.asarray(as_list, dtype=np.float64)
    grid = mw.GridSpec.custom(source, source, source)
    source[0] = -100.0
    assert grid.x_coords[0] == 0.0


@pytest.mark.parametrize(
    "bad_coords",
    [
        np.zeros((2, 2)),
        [0.0],
        [0.0, 0.1, 0.1],
        [0.0, 0.2, 0.1],
        [0.0, np.nan, 1.0],
        [0.0, np.inf],
    ],
)
def test_custom_rejects_invalid_coords(bad_coords):
    good = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="y_coords"):
        mw.GridSpec.custom(good, bad_coords, good)


def test_custom_grid_properties():
    grid = mw.GridSpec.custom(GRADED_NODES, GRADED_NODES, GRADED_NODES)

    assert grid.is_custom is True
    assert grid.dx is None and grid.dy is None and grid.dz is None
    np.testing.assert_allclose(grid.min_spacing, (0.1, 0.1, 0.1))
    np.testing.assert_allclose(grid.axis_coords("x"), GRADED_NODES)
    with pytest.raises(ValueError, match="no scalar spacing"):
        _ = grid.spacing
    with pytest.raises(ValueError, match="no scalar spacing"):
        _ = grid.is_uniform

    uniform = mw.GridSpec.uniform(0.25)
    assert uniform.is_custom is False
    assert uniform.min_spacing == (0.25, 0.25, 0.25)
    assert uniform.axis_coords("z") is None


def test_scene_scalar_spacing_raises_on_custom_grid():
    scene = _custom_scene()
    with pytest.raises(ValueError, match="scalar spacing undefined"):
        _ = scene.dx
    with pytest.raises(ValueError, match="no scalar spacing"):
        _ = scene.grid_spacing


def test_domain_extent_mismatch_raises():
    nodes = GRADED_NODES
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 2.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        device="cpu",
    )
    with pytest.raises(ValueError, match="span the domain exactly"):
        prepare_scene(scene)


def test_prepared_scene_uniform_masters_fill_domain_with_tidy3d_spacing_semantics():
    dl = 0.01
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(dl),
        device="cpu",
    )
    prepared = prepare_scene(scene)

    cell_count = int(np.ceil(1.28 / dl))
    assert prepared.Nx == cell_count + 1
    assert len(prepared.x_nodes64) == cell_count + 1

    expected_nodes = np.linspace(-0.64, 0.64, cell_count + 1, endpoint=True, dtype=np.float64)
    effective_dl = 1.28 / cell_count
    np.testing.assert_array_equal(prepared.x_nodes64, expected_nodes)
    np.testing.assert_allclose(prepared.dx_primal64, effective_dl, rtol=1e-12)
    np.testing.assert_allclose(prepared.dx_dual64, effective_dl, rtol=1e-12)
    np.testing.assert_array_equal(
        prepared.x_half64, prepared.x_nodes64[:-1] + 0.5 * prepared.dx_primal64
    )

    assert prepared.x.dtype == torch.float32
    np.testing.assert_allclose(prepared.x.cpu().numpy(), expected_nodes, atol=1e-7)
    np.testing.assert_allclose(prepared.x_half.cpu().numpy(), prepared.x_half64, atol=1e-7)


def test_uniform_grid_treats_requested_dl_as_maximum_step():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=mw.GridSpec.uniform(0.3),
        device="cpu",
    )

    prepared = prepare_scene(scene)

    assert prepared.Nx == 5
    np.testing.assert_allclose(prepared.x_nodes64, (0.0, 0.25, 0.5, 0.75, 1.0))
    np.testing.assert_allclose(prepared.dx_primal64, 0.25)


def test_prepared_scene_appends_uniform_pml_outside_domain():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    )

    prepared = prepare_scene(scene)

    assert prepared.Nx == 15
    np.testing.assert_allclose(prepared.x_nodes64[:3], (-0.7, -0.6, -0.5))
    np.testing.assert_allclose(prepared.x_nodes64[-3:], (0.5, 0.6, 0.7))
    assert prepared.physical_domain_range == pytest.approx((-0.5, 0.5) * 3)
    assert prepared.domain_range == pytest.approx((-0.7, 0.7) * 3)


def test_prepared_scene_appends_custom_pml_with_edge_steps():
    scene = _custom_scene(boundary=mw.BoundarySpec.pml(num_layers=1))

    prepared = prepare_scene(scene)

    np.testing.assert_allclose(
        prepared.x_nodes64,
        np.concatenate(([-0.1], GRADED_NODES, [1.3])),
    )
    assert prepared.domain.bounds[0] == (0.0, 1.0)


def test_prepared_scene_custom_masters_on_graded_axis():
    prepared = prepare_scene(_custom_scene(boundary=mw.BoundarySpec.pec()))

    nodes = GRADED_NODES
    primal = np.diff(nodes)
    np.testing.assert_array_equal(prepared.x_nodes64, nodes)
    np.testing.assert_array_equal(prepared.dx_primal64, primal)
    np.testing.assert_array_equal(prepared.x_half64, nodes[:-1] + 0.5 * primal)
    assert prepared.Nx == len(nodes)

    dual = prepared.dx_dual64
    assert dual.shape == nodes.shape
    np.testing.assert_array_equal(dual[1:-1], 0.5 * (primal[:-1] + primal[1:]))
    # Non-periodic boundary: mirror distance across the boundary node.
    assert dual[0] == primal[0]
    assert dual[-1] == primal[-1]


def test_prepared_scene_dual_boundary_entries_periodic():
    prepared = prepare_scene(_custom_scene(boundary=mw.BoundarySpec.periodic()))
    primal = np.diff(GRADED_NODES)
    wrap = 0.5 * (primal[0] + primal[-1])
    assert prepared.dy_dual64[0] == wrap
    assert prepared.dy_dual64[-1] == wrap


def test_component_coords_use_per_cell_half_steps_on_custom_grid():
    scene = _custom_scene()
    nodes = GRADED_NODES
    half = nodes[:-1] + 0.5 * np.diff(nodes)

    ex_x, ex_y, ex_z = component_coords(scene, "Ex")
    np.testing.assert_array_equal(ex_x, half)
    np.testing.assert_array_equal(ex_y, nodes)
    np.testing.assert_array_equal(ex_z, nodes)

    hx_x, hx_y, hx_z = component_coords(scene, "Hx")
    np.testing.assert_array_equal(hx_x, nodes)
    np.testing.assert_array_equal(hx_y, half)
    np.testing.assert_array_equal(hx_z, half)

    cx, cy, cz = centered_cell_coords(scene)
    np.testing.assert_array_equal(cx, half)
    np.testing.assert_array_equal(cy, half)
    np.testing.assert_array_equal(cz, half)


def test_fdfd_rejects_custom_grid():
    simulation = mw.Simulation.fdfd(_custom_scene(), frequency=1e9)
    with pytest.raises(NotImplementedError, match="FDFD does not support nonuniform"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_fdtd_custom_grid_with_pml_runs_with_physical_depth_profiles():
    scene = _custom_scene(boundary=mw.BoundarySpec.pml(num_layers=2), device="cuda")
    scene.add_source(
        mw.PointDipole(
            position=(0.5, 0.5, 0.5),
            polarization="Ez",
            width=0.1,
            source_time=mw.CW(frequency=1e9, amplitude=1.0),
            name="src",
        )
    )
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver

    assert solver.cpml_kappa_e_x.shape == (solver.Nx,)
    assert solver.cpml_kappa_h_x.shape == (solver.Nx - 1,)
    # The default propagating-wave profile is impedance matched (kappa=1) and
    # absorbs through its non-zero convolutional conductivity coefficients.
    assert torch.all(solver.cpml_kappa_e_x == 1.0)
    assert float(solver.cpml_c_e_x[0]) < 0.0
    assert float(solver.cpml_c_e_x[-1]) < 0.0
    # Nodes at or inside the layer interfaces carry no conductivity grading.
    assert float(solver.cpml_kappa_e_x[2]) == 1.0
    assert float(solver.cpml_kappa_e_x[3]) == 1.0
    assert float(solver.cpml_c_e_x[2]) == 0.0
    assert float(solver.cpml_c_e_x[3]) == 0.0

    solver.solve(time_steps=20, dft_frequency=None, dft_window="none", full_field_dft=False)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.isfinite(getattr(solver, name)).all(), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_tfsf_rejects_nonuniform_injection_region():
    # Uniform 0.05 axis with a graded patch inside the TFSF box footprint.
    nodes = np.linspace(-0.5, 0.5, 21)
    graded = nodes.copy()
    graded[10] += 0.02
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.custom(graded, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.GaussianPulse(frequency=3e9, fwidth=1e9, amplitude=1.0),
            injection=mw.TFSF(bounds=((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))),
            name="pw",
        )
    )
    simulation = mw.Simulation.fdtd(scene, frequency=3e9)
    with pytest.raises(ValueError, match="locally uniform grid spacing along axis 'x'"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_mode_source_rejects_nonuniform_mode_plane():
    # Uniform 0.05 axes with a graded patch on the y (transverse) axis inside
    # the mode-plane aperture window.
    nodes = np.linspace(-0.5, 0.5, 21)
    graded = nodes.copy()
    graded[10] += 0.02
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.custom(nodes, graded, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.2, 0.0, 0.0),
            size=(0.0, 0.4, 0.4),
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="mode0",
        )
    )
    simulation = mw.Simulation.fdtd(scene, frequency=1e9)
    with pytest.raises(ValueError, match="locally uniform grid spacing along axis 'y'"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_solver_spacing_tensors_on_uniform_grid():
    dl = 0.05
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(dl),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        device="cuda",
    )
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver

    assert solver.min_dx == pytest.approx(dl, rel=1e-12)
    for name in ("inv_dx_e", "inv_dy_e", "inv_dz_e"):
        tensor = getattr(solver, name)
        assert tensor.dtype == torch.float32 and tensor.is_cuda
        assert tensor.numel() == solver.Nx
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, 1.0 / dl), rtol=1e-6, atol=0.0
        )
    for name in ("inv_dx_h", "inv_dy_h", "inv_dz_h"):
        tensor = getattr(solver, name)
        assert tensor.dtype == torch.float32 and tensor.is_cuda
        assert tensor.numel() == solver.Nx - 1
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, 1.0 / dl), rtol=1e-6, atol=0.0
        )

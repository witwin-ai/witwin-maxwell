"""Smoothed PEC node occupancy on nonuniform (custom / auto) Yee grids.

The compiled PEC node occupancy smooths the metal over roughly one cell. The
smoothing width was the single global ``0.5*min(min_spacing)``: on a graded grid a
fine feature anywhere shrinks that minimum, so a PEC wall sitting in a locally
coarse region gets a beta far narrower than its own cell and the occupancy collapses
to a hard step there.

The width is now a per-node field, half the local Yee dual-cell width (min over
axes), so it tracks the local cell wherever the wall is. It reduces bit-exactly to
``0.5*spacing`` on a uniform grid, which the regression guard locks in.

This node field is what the mode solver, the modal ports, the terminal-contact
checks and the material summaries read. The FDTD *conformal* edge fill is a separate
artifact computed directly on each Yee edge from the signed distance (see
``tests/materials/compiler/test_pec_conformal_alignment.py``); it involves no
smoothing width and is not exercised here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _pec_geometry_beta, _pec_occupancy
from witwin.maxwell.fdtd.runtime.materials import average_node_to_component
from witwin.maxwell.scene import prepare_scene


def _pec_scene(grid, bounds, wall_low_x, *, pec_mode="conformal"):
    """A PEC half-space whose low x face sits at ``wall_low_x``.

    The slab is a fixed, bounds-independent box: its low face is ``wall_low_x`` and
    it extends far past the domain in +x and transversely, so re-expressing the same
    node grid under different bounds keeps the identical geometry (needed for the
    bit-exact regression comparison).
    """
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=grid,
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
    )
    scene.add_structure(
        mw.Structure(
            name="wall",
            geometry=mw.Box(
                position=(wall_low_x + 5.0, 0.0, 0.0),
                size=(10.0, 8.0, 8.0),
            ),
            material=mw.Material.pec(),
        )
    )
    return prepare_scene(scene)


def _ex_edge_fill(occupancy, i0, jy, jz):
    """Node->edge PEC fill on the Ex edge with low node ``i0`` (solver stencil)."""
    fill = average_node_to_component(None, occupancy, "Ex")
    return float(fill[i0, jy, jz])


# --- Regression guard: uniform grid PEC occupancy stays bit-exact through the new path ---

def test_uniform_grid_custom_master_pec_occupancy_bit_close():
    # A uniform grid re-expressed as GridSpec.custom takes the new per-node beta
    # path; its PEC occupancy must match the scalar-spacing GridSpec.uniform run
    # bitwise. This is the P5.3 bit-close regression guard for conformal PEC.
    dl = 0.05
    bounds = ((-0.4, 0.35), (-0.4, 0.35), (-0.4, 0.35))
    uniform = _pec_scene(mw.GridSpec.uniform(dl), bounds, wall_low_x=0.06)
    uniform_occ = _pec_occupancy(uniform)

    custom_grid = mw.GridSpec.custom(uniform.x_nodes64, uniform.y_nodes64, uniform.z_nodes64)
    custom_bounds = tuple(
        (float(nodes[0]), float(nodes[-1]))
        for nodes in (uniform.x_nodes64, uniform.y_nodes64, uniform.z_nodes64)
    )
    custom = _pec_scene(custom_grid, custom_bounds, wall_low_x=0.06)
    custom_occ = _pec_occupancy(custom)

    assert uniform_occ is not None and custom_occ is not None
    # Guard is only meaningful if the occupancy actually has a fractional interface.
    assert float(uniform_occ.max()) > 0.9 and float(uniform_occ.min()) < 0.1
    fractional = (uniform_occ > 0.05) & (uniform_occ < 0.95)
    assert int(fractional.sum()) > 0
    assert torch.equal(uniform_occ, custom_occ)


# --- The smoothing width is a scalar on uniform grids, per node on graded ones ---

def test_pec_beta_scalar_on_uniform_tensor_on_custom():
    uniform = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.uniform(0.1),
            device="cpu",
        )
    )
    beta = _pec_geometry_beta(uniform)
    assert not torch.is_tensor(beta)
    assert beta == pytest.approx(0.5 * min(uniform.grid.min_spacing))

    graded = np.array([-0.5, -0.30, -0.15, -0.05, 0.0, 0.08, 0.20, 0.35, 0.5], dtype=np.float64)
    scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
            grid=mw.GridSpec.custom(graded, graded, graded),
            device="cpu",
        )
    )
    beta = _pec_geometry_beta(scene)
    assert torch.is_tensor(beta)
    assert beta.shape == (scene.Nx, scene.Ny, scene.Nz)
    dx = torch.as_tensor(scene.dx_dual64, dtype=torch.float32)
    dy = torch.as_tensor(scene.dy_dual64, dtype=torch.float32)
    dz = torch.as_tensor(scene.dz_dual64, dtype=torch.float32)
    expected = 0.5 * torch.minimum(
        torch.minimum(dx[:, None, None], dy[None, :, None]), dz[None, None, :]
    )
    assert torch.equal(beta, expected)
    # A uniformly spaced custom axis collapses the per-node field back to 0.5*spacing.
    uniform_custom = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            grid=mw.GridSpec.custom(*(np.linspace(0.0, 1.0, 11),) * 3),
            device="cpu",
        )
    )
    beta_uc = _pec_geometry_beta(uniform_custom)
    assert torch.is_tensor(beta_uc)
    assert torch.allclose(beta_uc, torch.full_like(beta_uc, 0.05), atol=0.0, rtol=0.0)


# --- Centerpiece: the node occupancy resolves a sub-cell wall in a coarse region ---

def _graded_nodes():
    # x is coarse (0.04) everywhere except a fine cluster near the low edge that
    # shrinks the *global* minimum spacing to 0.005 -- far from the PEC wall.
    fine = np.arange(-0.50, -0.40, 0.005)
    coarse = np.arange(-0.40, 0.50 + 1e-9, 0.04)
    xg = np.unique(np.concatenate([fine, coarse]).astype(np.float64))
    yg = np.arange(-0.50, 0.50 + 1e-9, 0.04)
    return xg, yg, yg.copy()


def test_conformal_pec_tracks_subcell_wall_in_coarse_region():
    xg, yg, zg = _graded_nodes()
    bounds = (
        (float(xg[0]), float(xg[-1])),
        (float(yg[0]), float(yg[-1])),
        (float(zg[0]), float(zg[-1])),
    )
    grid = mw.GridSpec.custom(xg, yg, zg)

    ref = prepare_scene(
        mw.Scene(domain=mw.Domain(bounds=bounds), grid=grid, device="cpu")
    )
    # Global-min beta would be 0.5*0.005 = 0.0025; the wall cell is 0.04 wide.
    global_beta = 0.5 * min(ref.grid.min_spacing)
    assert global_beta == pytest.approx(0.0025, abs=1e-4)

    # The Ex edge whose low node sits at x = 0.20 spans the coarse cell [0.20, 0.24].
    i0 = int(np.argmin(np.abs(ref.x_nodes64 - 0.20)))
    assert float(ref.x_nodes64[i0 + 1] - ref.x_nodes64[i0]) == pytest.approx(0.04, abs=1e-6)
    jy = (yg.size - 1) // 2
    jz = (zg.size - 1) // 2

    # The per-node beta at this wall cell follows the local (coarse) cell, ~0.02,
    # an order of magnitude wider than the global-min beta that caused staircasing.
    beta = _pec_geometry_beta(ref)
    assert float(beta[i0, jy, jz]) == pytest.approx(0.02, abs=2e-3)
    assert float(beta[i0, jy, jz]) > 4.0 * global_beta

    # Sweep the wall across the interior of the coarse cell. Conformal placement
    # makes the edge fill fall monotonically with the wall position; under the old
    # global-min beta these interior fills were all pinned at ~0.5 (staircase).
    walls = np.linspace(0.205, 0.235, 4)
    fills = np.array(
        [
            _ex_edge_fill(_pec_occupancy(_pec_scene(grid, bounds, float(w))), i0, jy, jz)
            for w in walls
        ]
    )
    assert np.all(np.diff(fills) < -0.02)  # strictly, materially decreasing
    assert float(fills.max() - fills.min()) > 0.15  # genuine sub-cell resolution
    assert bool((fills > 0.0).all()) and bool((fills < 1.0).all())


# --- AutoGrid + PEC compiles end to end with the per-node beta ---

def test_auto_grid_pec_compiles_with_per_node_beta():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.auto(min_steps_per_wavelength=10, wavelength=0.3),
        device="cpu",
        subpixel_samples=mw.SubpixelSpec(pec="conformal"),
    )
    scene.add_structure(
        mw.Structure(
            name="pec",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            material=mw.Material.pec(),
        )
    )
    prepared = prepare_scene(scene)
    assert prepared.grid.is_custom  # auto resolved to a nonuniform grid
    beta = _pec_geometry_beta(prepared)
    assert torch.is_tensor(beta)
    occ = _pec_occupancy(prepared)
    assert occ is not None
    assert torch.isfinite(occ).all()
    assert float(occ.min()) >= 0.0 and float(occ.max()) <= 1.0
    # The interface is genuinely smoothed: some node lies strictly inside (0, 1).
    assert int(((occ > 1e-3) & (occ < 1.0 - 1e-3)).sum()) > 0


# --- PEC occupancy stays differentiable in the wall geometry on a graded grid ---

def test_conformal_pec_nonuniform_geometry_gradient_flows():
    xg, yg, zg = _graded_nodes()
    bounds = (
        (float(xg[0]), float(xg[-1])),
        (float(yg[0]), float(yg[-1])),
        (float(zg[0]), float(zg[-1])),
    )
    position = torch.tensor([0.22, 0.0, 0.0], requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.custom(xg, yg, zg), device="cpu"
    )
    scene.add_structure(
        mw.Structure(
            name="wall",
            geometry=mw.Box(position=position, size=(0.3, 4.0, 4.0)),
            material=mw.Material.pec(),
        )
    )
    occ = _pec_occupancy(prepare_scene(scene))
    occ.sum().backward()
    assert position.grad is not None
    assert torch.isfinite(position.grad).all()
    assert float(position.grad[0].abs()) > 0.0  # wall normal is along x

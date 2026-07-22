"""Conformal PEC must collapse onto the exact staircase treatment on aligned faces.

The conformal PEC open fraction multiplies the electric update every step, so a
fill fraction ``f`` on an edge is an effective conductivity ``eps*f/dt`` there. The
fill must therefore have *compact support*: exactly zero on any edge the conductor
surface does not reach, and exactly one on any edge wholly inside it. Otherwise the
smoothed node occupancy's tails paint a lossy shell several cells thick around every
conductor -- which is what made a grid-aligned PEC slab (whose staircase
representation is exact) an order of magnitude worse under ``pec="conformal"``.

These gates assert on the compiled per-edge fill, which is sharper and far cheaper
than a field comparison; ``tests/validation/physics/test_pec_conformal.py`` carries
the matching field-level runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import (
    _pec_edge_fill,
    _pec_occupancy,
    _pec_signed_distance,
)
from witwin.maxwell.fdtd.runtime.materials import average_node_to_component
from witwin.maxwell.scene import prepare_scene


DX = 0.02
BOUNDS = ((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))
_EDGE_ENDPOINT_SLICES = {
    "Ex": ((slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
    "Ey": ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
    "Ez": ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
}


def _pec_scene(geometry, pec_mode="conformal"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(DX),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(geometry=geometry, material=mw.Material.pec(), name="metal")
    )
    return prepare_scene(scene)


def _staircase_edge_fill(prepared, component):
    """The staircase fill: the node->edge average of the occupancy, thresholded."""
    occupancy = _pec_occupancy(prepared)
    return (average_node_to_component(None, occupancy, component) >= 0.5).to(torch.float32)


def _node_aligned_box(half_extent=0.1):
    """A cube whose six faces land exactly on Yee node planes of the shared grid."""
    reference = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=BOUNDS),
            grid=mw.GridSpec.uniform(DX),
            boundary=mw.BoundarySpec.pml(num_layers=8),
            device="cpu",
        )
    )
    nodes = np.asarray(reference.x_nodes64)
    low = float(nodes[int(np.argmin(np.abs(nodes + half_extent)))])
    high = float(nodes[int(np.argmin(np.abs(nodes - half_extent)))])
    assert abs(low + half_extent) < 1e-9 and abs(high - half_extent) < 1e-9
    side = high - low
    return mw.Box(position=(0.0, 0.0, 0.0), size=(side, side, side))


def test_conformal_pec_edge_fill_equals_staircase_for_grid_aligned_faces():
    prepared = _pec_scene(_node_aligned_box())
    edge_fill = _pec_edge_fill(prepared)
    assert edge_fill is not None
    for component in ("Ex", "Ey", "Ez"):
        conformal = edge_fill[component]
        staircase = _staircase_edge_fill(prepared, component)
        # Bitwise: a face lying on a node plane cuts no edge, so nothing is fractional.
        assert torch.equal(conformal, staircase), component
        fractional = (conformal > 0.0) & (conformal < 1.0)
        assert int(fractional.sum()) == 0, component


def test_conformal_pec_fill_has_compact_support_around_an_aligned_conductor():
    """No fill on edges the conductor never reaches -- the vacuum-halo regression."""
    prepared = _pec_scene(_node_aligned_box())
    edge_fill = _pec_edge_fill(prepared)
    signed_distance = _pec_signed_distance(prepared)
    occupancy = _pec_occupancy(prepared)
    halo_totals = []
    for component in ("Ex", "Ey", "Ez"):
        low_slice, high_slice = _EDGE_ENDPOINT_SLICES[component]
        low = signed_distance[low_slice]
        high = signed_distance[high_slice]
        outside = (low > 0.0) & (high > 0.0)
        assert float(edge_fill[component][outside].abs().max()) == 0.0, component

        # The pre-fix node-average path put a nonzero (hence lossy) fill on
        # thousands of these strictly-outside vacuum edges; keep that visible so a
        # regression back to the smoothed node occupancy is unmistakable.
        smoothed = average_node_to_component(None, occupancy, component)
        halo_totals.append(int(((smoothed > 0.0) & outside).sum()))
    assert min(halo_totals) > 1000


def test_conformal_pec_fill_is_fractional_only_on_edges_the_surface_cuts():
    prepared = _pec_scene(mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.11))
    edge_fill = _pec_edge_fill(prepared)
    signed_distance = _pec_signed_distance(prepared)
    for component in ("Ex", "Ey", "Ez"):
        low_slice, high_slice = _EDGE_ENDPOINT_SLICES[component]
        low = signed_distance[low_slice]
        high = signed_distance[high_slice]
        cut = (torch.minimum(low, high) < 0.0) & (torch.maximum(low, high) > 0.0)
        fill = edge_fill[component]
        fractional = (fill > 0.0) & (fill < 1.0)
        assert int(fractional.sum()) > 100, component
        assert int((fractional & ~cut).sum()) == 0, component
        assert float(fill.min()) >= 0.0 and float(fill.max()) <= 1.0


def test_conformal_pec_fill_resolves_a_curved_surface_better_than_staircase():
    radius = 0.11
    prepared = _pec_scene(mw.Sphere(position=(0.0, 0.0, 0.0), radius=radius))
    edge_fill = _pec_edge_fill(prepared)
    analytic_volume = 4.0 / 3.0 * np.pi * radius**3
    cell_volume = DX**3
    for component in ("Ex", "Ey", "Ez"):
        conformal_volume = float(edge_fill[component].sum()) * cell_volume
        staircase_volume = float(_staircase_edge_fill(prepared, component).sum()) * cell_volume
        conformal_error = abs(conformal_volume - analytic_volume) / analytic_volume
        staircase_error = abs(staircase_volume - analytic_volume) / analytic_volume
        # Measured at dx=0.02, r=0.11: conformal 0.74%, staircase 1.85%.
        assert conformal_error < 0.5 * staircase_error, component
        assert conformal_error < 0.01, component


def test_conformal_pec_edge_fill_is_differentiable_in_the_conductor_geometry():
    radius = torch.tensor(0.11, requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(DX),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        subpixel_samples=mw.SubpixelSpec(pec="conformal"),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=radius),
            material=mw.Material.pec(),
            name="metal",
        )
    )
    edge_fill = _pec_edge_fill(prepare_scene(scene))
    total = sum(fill.sum() for fill in edge_fill.values())
    total.backward()
    assert radius.grad is not None
    assert torch.isfinite(radius.grad).all()
    # Growing the sphere covers more edge length.
    assert float(radius.grad) > 0.0


def test_staircase_mode_carries_no_conformal_edge_fill():
    model = _pec_scene(_node_aligned_box(), pec_mode="staircase").compile_materials()
    assert model["pec_mode"] == "staircase"
    assert model["pec_edge_fill"] is None
    conformal_model = _pec_scene(_node_aligned_box()).compile_materials()
    assert conformal_model["pec_mode"] == "conformal"
    assert set(conformal_model["pec_edge_fill"]) == {"Ex", "Ey", "Ez"}


def test_non_pec_scene_has_no_edge_fill():
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(DX),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        subpixel_samples=mw.SubpixelSpec(pec="conformal"),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
            material=mw.Material(eps_r=4.0),
            name="dielectric",
        )
    )
    model = prepare_scene(scene).compile_materials()
    assert model["pec_occupancy"] is None
    assert model["pec_edge_fill"] is None


@pytest.mark.parametrize("pec_mode", ["staircase", "conformal"])
def test_aligned_conductor_open_fractions_agree_across_modes(pec_mode):
    """The runtime-visible open fractions match between modes on an aligned box."""
    prepared = _pec_scene(_node_aligned_box(), pec_mode=pec_mode)
    model = prepared.compile_materials()
    reference = _pec_scene(_node_aligned_box(), pec_mode="staircase")
    for component in ("Ex", "Ey", "Ez"):
        fill = (
            model["pec_edge_fill"][component]
            if model["pec_edge_fill"] is not None
            else _staircase_edge_fill(prepared, component)
        )
        assert torch.equal(fill, _staircase_edge_fill(reference, component)), component

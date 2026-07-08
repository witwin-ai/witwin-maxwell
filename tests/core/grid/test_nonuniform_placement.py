from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.fdtd.excitation.injection import initialize_source_terms
from witwin.maxwell.fdtd.observers import (
    resolve_plane_observer,
    resolve_point_observer,
)

# Strongly graded, distinct per-axis node masters so an axis mixup or a
# uniform-spacing assumption anywhere in the placement path shifts an index.
X_NODES = np.array(
    [-0.40, -0.28, -0.19, -0.12, -0.07, -0.04, -0.02, 0.0, 0.02, 0.05, 0.10, 0.18, 0.30, 0.46],
    dtype=np.float64,
)
Y_NODES = np.array(
    [-0.50, -0.33, -0.21, -0.13, -0.08, -0.05, -0.03, -0.01, 0.02, 0.07, 0.15, 0.27, 0.44],
    dtype=np.float64,
)
Z_NODES = np.array(
    [-0.36, -0.24, -0.15, -0.09, -0.05, -0.02, 0.01, 0.05, 0.11, 0.20, 0.33],
    dtype=np.float64,
)

_HALF = {
    "x": X_NODES[:-1] + 0.5 * np.diff(X_NODES),
    "y": Y_NODES[:-1] + 0.5 * np.diff(Y_NODES),
    "z": Z_NODES[:-1] + 0.5 * np.diff(Z_NODES),
}
_NODES = {"x": X_NODES, "y": Y_NODES, "z": Z_NODES}

# Yee half-offset axes per component (the 1.1 table of the design doc).
_COMPONENT_HALF_AXES = {
    "Ex": ("x",),
    "Ey": ("y",),
    "Ez": ("z",),
    "Hx": ("y", "z"),
    "Hy": ("x", "z"),
    "Hz": ("x", "y"),
}


def _expected_coords(component):
    return tuple(
        _HALF[axis] if axis in _COMPONENT_HALF_AXES[component] else _NODES[axis]
        for axis in "xyz"
    )


def _graded_scene(device="cpu"):
    return mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(X_NODES[0]), float(X_NODES[-1])),
                (float(Y_NODES[0]), float(Y_NODES[-1])),
                (float(Z_NODES[0]), float(Z_NODES[-1])),
            )
        ),
        grid=mw.GridSpec.custom(X_NODES, Y_NODES, Z_NODES),
        boundary=mw.BoundarySpec.pec(),
        device=device,
    )


def test_component_coords_midpoints_match_numpy_recomputation():
    scene = _graded_scene()
    for component in _COMPONENT_HALF_AXES:
        actual = component_coords(scene, component)
        for axis, got, expected in zip("xyz", actual, _expected_coords(component)):
            np.testing.assert_array_equal(got, expected, err_msg=f"{component}/{axis}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_point_observer_resolves_to_nearest_component_coordinate():
    scene = _graded_scene(device="cuda")
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver

    # Off-node inside the fine region, clearly away from nearest-node ties.
    position = (0.030, 0.055, 0.024)
    for component in ("Ez", "Ex", "Hy"):
        record = {"name": "probe", "kind": "point", "position": position, "component": component}
        resolved = resolve_point_observer(solver, record)
        expected = tuple(
            int(np.argmin(np.abs(coords - p)))
            for coords, p in zip(_expected_coords(component), position)
        )
        assert resolved == expected, component


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_plane_observer_resolves_graded_axis_neighbors_and_weights():
    scene = _graded_scene(device="cuda")
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver

    # Ez plane samples live on the z half grid; 0.06 falls between the
    # half-points 0.03 and 0.08 (weights from the local, unequal spacing).
    record = {"name": "cut", "kind": "plane", "axis": "z", "position": 0.06, "component": "Ez"}
    samples, in_plane = resolve_plane_observer(solver, record)
    z_half = _HALF["z"]
    lower = int(np.searchsorted(z_half, 0.06, side="left")) - 1

    assert len(samples) == 2
    assert samples[0]["plane_index"] == lower
    assert samples[1]["plane_index"] == lower + 1
    assert samples[0]["plane_position"] == pytest.approx(z_half[lower])
    assert samples[1]["plane_position"] == pytest.approx(z_half[lower + 1])
    span = z_half[lower + 1] - z_half[lower]
    assert samples[1]["weight"] == pytest.approx((0.06 - z_half[lower]) / span, rel=1e-6)
    assert samples[0]["weight"] + samples[1]["weight"] == pytest.approx(1.0)
    np.testing.assert_array_equal(in_plane[0], _NODES["x"])
    np.testing.assert_array_equal(in_plane[1], _NODES["y"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_point_dipole_patch_peaks_at_nearest_component_cell():
    position = (0.030, 0.055, 0.024)
    scene = _graded_scene(device="cuda")
    scene.add_source(
        mw.PointDipole(
            position=position,
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=1.0),
            name="src",
        )
    )
    solver = mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver
    initialize_source_terms(solver)

    terms = [term for term in solver._source_terms if term["field_name"] == "Ez"]
    assert len(terms) == 1
    term = terms[0]
    patch = term["patch"].abs()
    local_peak = np.unravel_index(int(patch.argmax().item()), patch.shape)
    global_peak = tuple(int(o) + int(i) for o, i in zip(term["offsets"], local_peak))

    expected = tuple(
        int(np.argmin(np.abs(coords - p)))
        for coords, p in zip(_expected_coords("Ez"), position)
    )
    assert global_peak == expected

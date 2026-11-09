import copy

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import compile_port_geometry


def configuration(device="cpu", axis="y", reversed_polarity=False):
    base = [np.array([-.02, -.01, 0, .01, .03]),
            np.array([-.02, -.005, 0, .01, .03]), np.array([-.01, 0, .01])]
    permutation = {"y": [0, 1, 2], "x": [1, 0, 2], "z": [0, 2, 1]}[axis]
    axes = [base[i] for i in permutation]
    pos, neg = np.array([0, .01, 0]), np.array([0, -.005, 0])
    contour_position, contour_size = np.array([.0025, .005, 0]), np.array([.035, 0, .01])
    aperture_position, aperture_size = np.array([0, .0025, 0]), np.array([.02, .015, 0])
    port = mw.LumpedPort(
        "feed", positive=(neg if reversed_polarity else pos)[permutation],
        negative=(pos if reversed_polarity else neg)[permutation], voltage_path=mw.AxisPath(axis),
        current_surface=mw.Box(position=torch.tensor(contour_position[permutation]),
                               size=torch.tensor(contour_size[permutation])),
        voltage_aperture=mw.Box(position=torch.tensor(aperture_position[permutation]),
                                size=torch.tensor(aperture_size[permutation])),
    )
    scene = mw.Scene(domain=mw.Domain(bounds=[(a[0], a[-1]) for a in axes]),
                     grid=mw.GridSpec.custom(*axes), boundary=mw.BoundarySpec.none(),
                     ports=[port], device=device)
    return scene, port


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("reverse", [False, True])
def test_aperture_exact_integral_and_nonuniform_transverse_quadrature(axis, reverse):
    scene, port = configuration(axis=axis, reversed_polarity=reverse)
    geometry = compile_port_geometry(scene, port)
    assert geometry.voltage_indices.shape == (6, 3)
    expected = -.015 if reverse else .015
    assert geometry.voltage_weights.sum().item() == pytest.approx(expected)
    # A linear transverse E profile has its aperture average exactly at x=0.
    from witwin.maxwell.scene import prepare_scene
    prepared = prepare_scene(scene)
    axes = [np.asarray(getattr(prepared, f"{a}_nodes64")) for a in "xyz"]
    normal = "xyz".index(axis)
    transverse = next(d for d in range(3) if d != normal and len(axes[d]) == 5)
    field_shape = tuple(len(a) - (d == normal) for d, a in enumerate(axes))
    component = torch.empty(field_shape, dtype=torch.float64)
    vector_shape = [1, 1, 1]
    vector_shape[transverse] = len(axes[transverse])
    component[:] = 2 + 30 * torch.tensor(axes[transverse]).reshape(vector_shape)
    assert geometry.integrate_voltage({f"E{axis}": component}).item() == pytest.approx(2 * expected)
    restored = compile_port_geometry(scene, copy.deepcopy(port))
    assert torch.equal(restored.voltage_indices, geometry.voltage_indices)
    assert torch.equal(restored.voltage_weights, geometry.voltage_weights)


def test_aperture_rejects_wrong_gap_and_unenclosed_current_contour():
    scene, port = configuration()
    bad = mw.Box(position=torch.tensor([0, .0025, 0]), size=torch.tensor([.02, .005, 0]))
    with pytest.raises(ValueError, match="aperture"):
        changed = copy.deepcopy(port)
        object.__setattr__(changed, "voltage_aperture", bad)
        compile_port_geometry(scene, changed)
    bad = mw.Box(position=torch.tensor([0, .005, 0]), size=torch.tensor([.01, 0, .01]))
    with pytest.raises(ValueError, match="enclose"):
        changed = copy.deepcopy(port)
        object.__setattr__(changed, "current_surface", bad)
        compile_port_geometry(scene, changed)


def test_aperture_sampling_and_deposition_are_power_dual():
    scene, port = configuration()
    geometry = compile_port_geometry(scene, port)
    values = torch.linspace(-.2, .8, geometry.voltage_weights.numel(), dtype=torch.float64)
    volumes = torch.linspace(.3, 1.1, len(values), dtype=torch.float64)
    current = .4
    density = geometry.voltage_weights * current / volumes
    work = torch.sum(values * density * volumes)
    assert work == pytest.approx(float(torch.dot(values, geometry.voltage_weights) * current), abs=1e-16)

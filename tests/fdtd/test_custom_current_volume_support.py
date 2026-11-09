"""Recover finite-volume current densities from actual prepared source terms."""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.fdtd.excitation.injection import _prepare_custom_current_source
from witwin.maxwell.scene import prepare_scene


_COMPONENTS = dict(zip(("Jx", "Jy", "Jz", "Mx", "My", "Mz"),
                       ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _scene(grid):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64),) * 3), grid=grid, device="cpu",
        boundary=mw.BoundarySpec.pml(num_layers=2),
    )


def _prepare(scene):
    return mw.Simulation.fdtd(
        scene, frequency=2e9, backend="cpu_reference", cuda_graph=False,
        full_field_dft=False, run_time=mw.TimeConfig(time_steps=1),
    ).prepare().solver


def _rematerialize(solver, scene, dtype):
    # FP64 source preparation is checked independently of FP64 forward admission.
    solver.scene = prepare_scene(scene, material_dtype=dtype)
    for field in _COMPONENTS.values():
        setattr(solver, field, getattr(solver, field).to(dtype))
        attr = ("eps_" if field.startswith("E") else "mu_") + field
        setattr(solver, attr, getattr(solver, attr).to(dtype))
    solver._source_terms = []
    solver._magnetic_source_terms = []
    source = compile_fdtd_sources(scene, default_frequency=2e9)[0]
    _prepare_custom_current_source(solver, source, source_index=0)


def _density(solver, field):
    density = torch.zeros_like(getattr(solver, field))
    for term in solver._source_terms + solver._magnetic_source_terms:
        if term["field_name"] != field:
            continue
        region = tuple(slice(i, i + n) for i, n in zip(term["offsets"], term["patch"].shape))
        denom = getattr(solver, ("eps_" if field.startswith("E") else "mu_") + field)
        density[region] += -term["patch"] * denom[region] / solver.dt
    return density.double().numpy()


def _control_cells(coords):
    mid = (coords[1:] + coords[:-1]) / 2
    left = np.r_[coords[0] - (coords[1] - coords[0]) / 2, mid]
    right = np.r_[mid, coords[-1] + (coords[-1] - coords[-2]) / 2]
    return left, right


def _expected(solver, field, lo, hi, *, singleton=False):
    factors, widths = [], []
    for a, coords in enumerate(component_coords(solver.scene, field)):
        left, right = _control_cells(coords)
        if solver._COMPONENT_HALF_OFFSET_AXES[field][a]:
            nodes = getattr(solver.scene, 'xyz'[a] + '_nodes64')
            left[0], right[-1] = nodes[0], nodes[-1]
        widths.append(right - left)
        if singleton and lo[a] == hi[a]:
            factor = np.zeros_like(coords)
            factor[np.argmin(abs(coords - lo[a]))] = 1
        else:
            lower, upper = max(lo[a], -0.64), min(hi[a], 0.64)
            factor = np.maximum(0, np.minimum(right, upper) - np.maximum(left, lower)) / (right - left)
        factors.append(factor)
    profile = np.einsum("i,j,k->ijk", *factors)
    volume = np.einsum("i,j,k->ijk", *widths)
    return profile, volume


_NONUNIFORM = np.r_[-0.64, -0.53, -0.41, -0.26, -0.17, -0.06, 0.02, 0.13, 0.25, 0.40, 0.51, 0.64]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grid", [mw.GridSpec.uniform(1.28 / 52), mw.GridSpec.uniform(1.28 / 103),
                                  mw.GridSpec.custom(_NONUNIFORM, _NONUNIFORM, _NONUNIFORM)],
                         ids=["coarse", "fine", "nonuniform"])
@pytest.mark.parametrize("center", [(0., 0., 0.), (0.031, -0.047, -0.25), (0.62, -0.61, 0.57)])
def test_all_prepared_current_moments_and_extents(dtype, grid, center):
    lo, hi = np.array(center) - .08, np.array(center) + .08
    axes = tuple(np.linspace(l, h, 3) for l, h in zip(lo, hi))
    scene = _scene(grid)
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(axes, {
        key: np.ones((3, 3, 3)) for key in _COMPONENTS
    }), source_time=mw.CW(frequency=2e9)))
    solver = _prepare(scene)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    for amplitude in (1., -2.5, .375, 0.):
        for values in scene.sources[0].current_dataset.components.values():
            values.fill(amplitude)
        _rematerialize(solver, scene, dtype)
        for field in _COMPONENTS.values():
            actual = _density(solver, field)
            expected, volume = _expected(solver, field, lo, hi)
            np.testing.assert_allclose(actual, amplitude * expected, rtol=tolerance, atol=tolerance * abs(amplitude))
            declared = amplitude * np.prod(np.maximum(0, np.minimum(hi, .64) - np.maximum(lo, -.64)))
            assert np.sum(actual * volume) == pytest.approx(declared, rel=tolerance, abs=1e-30)
            assert np.count_nonzero(actual[expected == 0]) == 0


@pytest.mark.parametrize("field", ["Ex", "Ey", "Ez"])
def test_constant_custom_and_uniform_prepared_profiles_agree(field):
    scene = _scene(mw.GridSpec.uniform(1.28 / 52))
    center, size = (0.031, -0.047, -0.25), (.16, .12, .14)
    axes = tuple(np.linspace(c - s / 2, c + s / 2, 3) for c, s in zip(center, size))
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(axes, {"J" + field[-1]: np.ones((3, 3, 3))})))
    custom = _prepare(scene)
    uniform = _prepare(scene.clone(sources=[mw.UniformCurrentSource(size=size, center=center, polarization=field)]))
    # The unchanged uniform path computes geometry in the field precision.
    np.testing.assert_allclose(_density(custom, field), _density(uniform, field), rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize("singleton_axes", [(0,), (1,), (2,), (0, 1, 2)])
def test_singleton_axes_keep_nearest_component_planes(singleton_axes):
    lo, hi = np.array([.031, -.047, -.25]), np.array([.131, .073, -.11])
    hi[list(singleton_axes)] = lo[list(singleton_axes)]
    axes = tuple([l] if l == h else [l, h] for l, h in zip(lo, hi))
    shape = tuple(len(axis) for axis in axes)
    scene = _scene(mw.GridSpec.uniform(.08))
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(axes, {key: np.ones(shape) for key in _COMPONENTS})))
    solver = _prepare(scene)
    for field in _COMPONENTS.values():
        expected, _ = _expected(solver, field, lo, hi, singleton=True)
        np.testing.assert_allclose(_density(solver, field), expected, rtol=1e-6, atol=1e-6)


def test_partial_cells_sample_nonconstant_data_inside_span():
    axes = (np.array([-.079, .083]), np.array([-.061, .093]), np.array([-.33, -.17]))
    x, y, z = np.meshgrid(*axes, indexing="ij")
    scene = _scene(mw.GridSpec.uniform(1.28 / 52))
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(axes, {
        key: 2 + x + 2*y + 3*z for key in _COMPONENTS
    })))
    solver = _prepare(scene)
    lo, hi = [a[0] for a in axes], [a[-1] for a in axes]
    for field in _COMPONENTS.values():
        coverage, _ = _expected(solver, field, lo, hi)
        q = [np.clip(c, l, h) for c, l, h in zip(component_coords(solver.scene, field), lo, hi)]
        xq, yq, zq = np.meshgrid(*q, indexing="ij")
        np.testing.assert_allclose(_density(solver, field), coverage * (2 + xq + 2*yq + 3*zq), rtol=1e-6, atol=1e-6)


def test_entirely_uncovered_dataset_emits_no_terms():
    scene = _scene(mw.GridSpec.uniform(.1))
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(([.8, .9],) * 3, {
        key: np.ones((2, 2, 2)) for key in _COMPONENTS
    })))
    solver = _prepare(scene)
    assert not solver._source_terms and not solver._magnetic_source_terms


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nonuniform_terminal_cells_cover_the_physical_domain(dtype):
    scene = _scene(mw.GridSpec.custom(_NONUNIFORM, _NONUNIFORM, _NONUNIFORM)).clone(boundary=mw.BoundarySpec.none())
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(([-.7, .7],) * 3, {
        key: np.ones((2, 2, 2)) for key in _COMPONENTS
    })))
    solver = _prepare(scene)
    _rematerialize(solver, scene, dtype)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    for field in _COMPONENTS.values():
        expected, volume = _expected(solver, field, (-.7,) * 3, (.7,) * 3)
        np.testing.assert_allclose(_density(solver, field), expected, rtol=tolerance, atol=tolerance)
        assert np.sum(_density(solver, field) * volume) == pytest.approx(1.28**3, rel=tolerance)

"""Analytic scoring of the under-resolved sphere and slab rows (benchmark/analytic/benchmark_rows.py)."""
import math

import numpy as np
import pytest
import torch

from benchmark.analytic import benchmark_rows as rows
from benchmark import runner
from benchmark.analytic import berreman, mie
from benchmark.scenes import SCENARIOS, build_scene
from witwin.maxwell.scene import place_scene, prepare_scene


def _cells(scene):
    prepared = prepare_scene(place_scene(scene, "cpu"), device="cpu")
    return [len(np.asarray(getattr(prepared, axis + "_nodes64"))) - 1 for axis in "xyz"], prepared


@pytest.mark.parametrize("name", sorted(rows.ROWS))
def test_refined_scene_nests_the_benchmark_grid(name):
    base, base_prepared = _cells(build_scene(name))
    refined = rows.refined_scene(name, 3)
    fine, fine_prepared = _cells(refined)
    layers = 2 * refined.boundary.num_layers
    assert [n - layers for n in fine] == [3 * (n - layers) for n in base]
    # Every physical-domain Yee node of the row is a node of the refined grid (the absorber keeps
    # its layer count, so its cells shrink); nothing but the grid changes.
    for axis, (lo, hi) in zip("xyz", refined.domain.bounds):
        coarse = np.asarray(getattr(base_prepared, axis + "_nodes64"))
        coarse = coarse[(coarse >= lo - 1e-9) & (coarse <= hi + 1e-9)]
        assert coarse.size == base[0] - layers + 1
        nodes = np.asarray(getattr(fine_prepared, axis + "_nodes64"))
        assert np.abs(nodes[:, None] - coarse[None, :]).min(axis=0).max() < 1e-9
    original = build_scene(name)
    assert refined.grid.is_custom == original.grid.is_custom
    scenario = SCENARIOS[name]
    key = runner._benchmark_cache_key(original, scenario.frequencies, scenario.run_time_factor)
    assert runner._benchmark_cache_key(refined.updated(grid=original.grid), scenario.frequencies,
                                       scenario.run_time_factor) == key


def test_tfsf_composite_is_total_inside_and_scattered_outside():
    row = rows.ROWS["tfsf_dielectric_sphere"]
    k = 2 * math.pi * 2.0e9 / rows.C0
    points = np.array([[0.2, 0.0, 0.1], [0.0, 0.0, -0.27], [0.35, 0.0, 0.0], [0.0, 0.0, 0.5]])
    composite = rows.tfsf_composite_mie(row, k, points)
    total, _ = mie.near_fields(row.radius, k, row.eps, points, field_kind="total")
    scattered, _ = mie.near_fields(row.radius, k, row.eps, points, field_kind="scattered")
    np.testing.assert_array_equal(composite[:2], total[:2])
    np.testing.assert_array_equal(composite[2:], scattered[2:])


def test_shape_l2_removes_one_global_complex_scale():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=200) + 1j * rng.normal(size=200)
    error, scale = rows.shape_l2(reference * 0.7 * np.exp(-1.1j), reference)
    assert error < 1e-12 and scale == pytest.approx(np.exp(1.1j) / 0.7)
    noisy = reference + 0.05 * (rng.normal(size=200) + 1j * rng.normal(size=200))
    assert rows.shape_l2(noisy * 3j, reference)[0] == pytest.approx(rows.shape_l2(noisy, reference)[0])


def test_tfsf_mask_drops_the_face_band_and_soft_rows_drop_the_source_side():
    dx = 0.01
    x = z = np.arange(-0.635, 0.64, dx)
    analytic = np.ones((x.size, z.size), complex)
    tfsf = rows.sphere_mask(rows.ROWS["tfsf_dielectric_sphere"], x, z, analytic, half_span=0.64, dx=dx)
    X, Z = np.meshgrid(x, z, indexing="ij")
    face = (np.abs(np.abs(X) - 0.28) < 1.5 * dx) & (np.abs(Z) <= 0.28)
    assert not tfsf[face].any() and tfsf[(np.abs(X) < 0.2) & (np.abs(Z) < 0.2)].all()
    soft = rows.sphere_mask(rows.ROWS["dielectric_sphere"], x, z, analytic, half_span=0.64, dx=dx)
    assert not soft[Z <= -0.5].any() and soft[Z > -0.5].all()


def test_slab_oracle_is_the_airy_slab():
    row = rows.ROWS["custom_grid_slab"]
    frequency = SCENARIOS["custom_grid_slab"].frequencies[0]
    response = berreman.solve([berreman.Layer(row.thickness, row.eps * np.eye(3))], frequency)
    n, k0 = math.sqrt(row.eps), 2 * math.pi * frequency / berreman.C0
    r12, phase = (1 - n) / (1 + n), np.exp(2j * n * k0 * row.thickness)
    r = r12 * (1 - phase) / (1 - r12**2 * phase)
    assert response.reflectance[0, 0] == pytest.approx(abs(r) ** 2, rel=1e-12)
    assert response.transmittance[0, 0] == pytest.approx(1 - abs(r) ** 2, rel=1e-12)
    assert response.transmittance[0, 0] == pytest.approx(0.81367, abs=1e-5)


def test_limits_are_derived_from_the_recorded_measurement():
    assert set(rows.LIMITS) == set(rows.ROWS) == set(rows.MEASURED)
    for name, limits in rows.LIMITS.items():
        assert set(limits) == set(rows.MEASURED[name])
        for key, limit in limits.items():
            m = rows.MEASURED[name][key]
            assert limit == pytest.approx(math.ceil(max(1.25 * m, m + 0.02) * 1e4) / 1e4)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name", sorted(rows.ROWS))
def test_refined_row_matches_the_analytic_reference(name):
    report = rows.score(name)
    assert not rows.check(report), (report, rows.LIMITS[name])

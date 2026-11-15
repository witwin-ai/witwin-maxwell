"""CPU checks of ladder geometry, diagnostics, and unequal-level estimators."""
import math

import numpy as np
import pytest

from scripts.dev.fdtd.thin_wire_ladder.common import (
    DELTA0, LAMBDA0, RADIUS, Run, runs, wire_geometry, resonance,
)
from scripts.dev.fdtd.thin_wire_ladder.analyze import observed_order, group_diagnostics
from witwin.maxwell.compiler import compile_thin_wires
from witwin.maxwell.scene import prepare_scene


def test_manifest_has_full_distinct_matrix_and_six_controls():
    manifest = runs()
    assert len(manifest) == 72
    assert len({r.name for r in manifest}) == 72
    assert sum(r.control == "primary" for r in manifest) == 66
    assert sum(r.control != "primary" for r in manifest) == 6
    assert len(runs("D1_*_r3_*")) == 7
    for r in manifest:
        assert r.radius/r.delta <= .2
        if r.control == "primary":
            assert r.radius == RADIUS
    assert Run("D1", 3., "G-phys", "nodes").gap/Run("D1", 3., "G-phys", "nodes").delta == 3
    assert Run("D1", 1.5, "G-phys075", "nodes").gap/DELTA0 == pytest.approx(.75)


@pytest.mark.parametrize("run", runs(), ids=lambda r: r.name)
def test_geometry_has_exact_feed_gap_and_air_clearance(run):
    geometry, negative, positive = wire_geometry(run)
    pminus = np.array(geometry[negative[0]][negative[1]])
    pplus = np.array(geometry[positive[0]][positive[1]])
    assert np.linalg.norm(pplus-pminus) == pytest.approx(run.gap, rel=1e-13)
    for points in geometry.values():
        p = np.array(points)
        half = np.array(run.cells)*run.delta/2
        air = half-12*run.delta-np.max(np.abs(p), axis=0)
        assert np.all(air >= .5*LAMBDA0-1e-13)
    if run.antenna != "L1":
        all_points = np.concatenate(list(geometry.values()))
        assert np.ptp(all_points[:, 2]) == pytest.approx(run.length)
        # Circuit nodes follow source points, so explicitly refine these spans.
        lengths = []
        for name, points in geometry.items():
            for j, value in enumerate(np.linalg.norm(np.diff(points, axis=0), axis=1)):
                if run.binding == "gap" and (name, j) == negative:
                    continue
                lengths.append(value)
        assert max(lengths) <= run.delta*(1+1e-12)


def test_refinement_order_fit_uses_actual_unequal_levels():
    r = np.array([1., 1.5, 2., 3.])
    fit = observed_order(r, 9.-4*r**1.7)
    assert fit["p"] == pytest.approx(1.7, abs=1e-5)
    assert fit["residual_fraction"] < 1e-6
    convergent = observed_order(r, 9.-4*r**-2)
    assert convergent["p"] == pytest.approx(-2, abs=1e-5)
    assert observed_order(r, [1, 2, 1, 3]) is None


def test_resonance_interpolation_never_extrapolates():
    assert resonance([1, 2, 3], [1+2j, 2+1j, 3+.1j]) == (None, None)
    f, z = resonance([1, 2, 3], [1-1j, 2+1j, 3+2j], target=2)
    assert f == 1.5
    assert z == 1.5+0j


def test_series_capacitor_hypothesis_has_zero_residual_for_constructed_data():
    data = []
    frequencies = np.array([2.11e9, 2.4e9, 2.69e9])
    for r in (1., 1.5, 2.):
        cap = 1e-12/r
        z = 50-1j/(2*np.pi*frequencies*cap)
        data.append(dict(run=dict(level=r, name=f"r{r}"),
            z_f0_ohm=[z[1].real, z[1].imag],
            z_check_engineering_ohm=np.column_stack((z.real, z.imag)).tolist(),
            wire_static=dict(C_eff_F=cap), frequencies_Hz=frequencies.tolist(),
            z_engineering_ohm=np.column_stack((z.real, z.imag)).tolist()))
    diagnostic = group_diagnostics(data)
    for entry in diagnostic["capacitor"]:
        assert max(entry["residual_fraction_abs"]) < 1e-14
    assert diagnostic["order_X_f0"]["p"] == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("antenna,binding", [("D1", "nodes"), ("D1", "gap"), ("L1", "nodes"), ("L1", "gap")])
def test_public_scene_cpu_compiler_and_read_only_coefficients(antenna, binding):
    from types import SimpleNamespace
    from scripts.dev.fdtd.thin_wire_ladder.common import build_scene
    from scripts.dev.fdtd.thin_wire_ladder.run import static_diagnostics
    run = Run(antenna, 1., "G1", binding)
    scene = build_scene(run)
    network = compile_thin_wires(prepare_scene(scene, device="cpu"))
    diagnostics = static_diagnostics(SimpleNamespace(_wire_runtime=SimpleNamespace(network=network)), run)
    assert network.field_shapes[0] == (run.cells[0], run.cells[1]+1, run.cells[2]+1)
    assert len(diagnostics["feed_segments"]) == 2
    assert diagnostics["max_abs_lc_relative_error"] < 1e-6
    assert diagnostics["C_eff_F"] > 0
    for node in diagnostics["feed_nodes"]:
        assert node["C_node_F"] == pytest.approx(node["dual_length_m"]*node["adjacent_C_prime_F_m"][0])
    if binding == "gap":
        assert diagnostics["gap_segment"]["length_m"] == pytest.approx(run.gap)


def test_uniform_grid_rounding_does_not_add_unplanned_cells():
    for run in runs():
        step = math.nextafter(run.delta, math.inf)
        for count in run.cells:
            physical = count-24
            assert math.ceil((physical*run.delta)/step) == physical

from types import SimpleNamespace
import json
import math

import numpy as np
import pytest
import torch

from benchmark.label_qualification import random_convergence as rc


def test_f1_resolution_and_polarized_scene():
    specs = rc.draw_scenes(("F1",), 20, 1701)
    assert any(s["ppw"] == 20 for s in specs)
    assert any(s["ppw"] > 20 for s in specs)
    for spec in specs:
        assert spec["n_max"] == math.sqrt(spec["eps"])
        assert spec["ppw"] == max(20, 10 * spec["n_max"])
        assert spec["n"] == math.ceil(spec["ppw"] * spec["wavelengths"])
        for level in rc.LEVELS:
            assert len(rc.axes_for(spec, level)[0]) - 1 == int(spec["n"] * level)
    tasks, plans = rc.task_list(SimpleNamespace(
        family="F1", scenes=1, seed=1701, f1_averaging="polarized",
        long_cycles=120, points=32, plane_side=9, mesh_fill_tolerance=.1,
    ))
    plan = plans[tasks[0].task_id]
    assert json.loads(tasks[0].builder)["spec"] == plan["spec"]
    sample, _ = rc.build_sample(plan, device="cpu")
    assert sample.scene.subpixel.averaging == "polarized"
    assert sample.scene.subpixel.samples == (3, 3, 3)
    assert sample.fdtd_kwargs["reference_frequency"] == rc.F0


def test_f1_zero_dc_source():
    import witwin.maxwell as mw
    from benchmark.datagen import scenes as builders

    specs = rc.draw_scenes(("F1",), 20, 1701)
    tasks, plans = rc.task_list(SimpleNamespace(
        family="F1", scenes=20, seed=1701, f1_averaging="polarized",
        long_cycles=120, points=32, plane_side=9, mesh_fill_tolerance=.1,
    ))
    plan = plans[tasks[0].task_id]
    spec = specs[0]
    assert plan["spec"] == dict(spec, f1_averaging="polarized")
    expected = builders.dielectric(
        spec["scene_id"], n=spec["n"], wavelengths=spec["wavelengths"], bodies=spec["bodies"],
        kind=spec["kind"], eps=spec["eps"], lossy=spec["lossy"], device="cpu",
    ).scene.sources[0]
    sample, _ = rc.build_sample(plan, device="cpu")
    assert len(sample.scene.sources) == 1
    source = sample.scene.sources[0]
    assert isinstance(source, mw.PointDipole)
    for field in ("position", "polarization", "width", "profile", "name", "kind"):
        assert getattr(source, field) == getattr(expected, field)
    pulse = source.source_time
    assert isinstance(pulse, mw.GaussianPulse)
    assert pulse.phase == math.pi / 2
    assert pulse.frequency == rc.F0
    assert pulse.fwidth == .8 * rc.F0
    assert pulse.delay == mw.GaussianPulse(rc.F0, .8 * rc.F0).delay
    # The run starts at t = 0, so the 6-sigma delay drops an exp(-18) tail; the former
    # phase-0 builder pulse leaves a DC fraction many orders of magnitude larger.
    times = np.linspace(0., pulse.delay + 12 * pulse.sigma_t, 36001)
    def dc_fraction(waveform):
        values = np.array([waveform.evaluate(t) for t in times])
        return abs(np.trapezoid(values, times)) / np.trapezoid(abs(values), times)
    assert pulse == expected.source_time
    assert dc_fraction(pulse) <= 1e-8
    assert dc_fraction(mw.GaussianPulse(rc.F0, .8 * rc.F0)) >= 1e-2
    assert plan["f1_source"] == dict(kind="gaussian_pulse", frequency=rc.F0,
                                      fwidth=.8 * rc.F0, phase=math.pi / 2)
    assert json.loads(tasks[0].builder)["f1_source"] == plan["f1_source"]


def test_dielectric_catalog_sources_are_zero_dc_without_the_dc_warning():
    import warnings
    import witwin.maxwell as mw
    from benchmark.datagen.scenes import scene_builders
    from witwin.maxwell.fdtd.excitation.incident_line_spectrum import check_incident_line_spectrum

    builders = [b for b in scene_builders() if b.family == "F1"]
    assert len(builders) == 10
    for builder in builders:
        scene = builder.build(device="cpu").scene
        (source,) = scene.sources
        assert source.source_time == mw.GaussianPulse(rc.F0, .8 * rc.F0, phase=math.pi / 2)
        # Uniform cube: the 3D vacuum CFL step sets the sampling clock of the diagnostic.
        spacing = float(np.diff(np.asarray(scene.grid.x_coords)).min())
        dt = .99 * spacing / (rc.C0 * math.sqrt(3))
        steps = math.ceil((source.source_time.settling_time + 120 / rc.F0) / dt)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fractions = check_incident_line_spectrum(
                source.source_time, f_edge=None, dt=dt, ds=rc.C0 * dt, line_name=builder.name,
                record_steps=steps, sample_offset=.5)
        assert fractions["dc_fraction"] <= 1e-8, builder.name


def test_conductor_catalog_sources_are_zero_dc():
    import witwin.maxwell as mw
    from benchmark.datagen.scenes import scene_builders

    builders = [b for b in scene_builders() if b.family == "F2"]
    assert builders
    expected = mw.GaussianPulse(rc.F0, .8 * rc.F0, phase=math.pi / 2)
    pulse = expected
    times = np.linspace(0., pulse.delay + 12 * pulse.sigma_t, 36001)
    values = np.array([pulse.evaluate(t) for t in times])
    assert abs(np.trapezoid(values, times)) / np.trapezoid(abs(values), times) <= 1e-8
    for builder in builders:
        (source,) = builder.build(device="cpu").scene.sources
        assert source.source_time == expected, builder.name


def test_f2_box_faces_are_nodes_and_planar_axes_are_unchanged():
    for spec in rc.draw_scenes(("F2",), 20, 1701):
        span = math.ceil(20 * spec["wavelengths"]) * rc.C0 / rc.F0 / 20
        n = 4 * math.ceil(20 * rc.BAND[1] * span / rc.C0 / 4)
        h = (.5 * rc.C0 / rc.F0 * spec["scale"]) / 2
        for level in rc.LEVELS:
            for axis in rc.axes_for(spec, level):
                if spec["kind"] == "box":
                    assert -h in axis and h in axis and 0. in axis
                    dx = axis[len(axis) // 2 + 1]
                    assert np.ptp(np.diff(axis)) <= 1e-12 * dx
                    assert np.array_equal(axis, -axis[::-1])
                    assert abs(axis[-1] - span / 2) <= dx / 2
                    # Face cell counts keep the exact 1 : 1.5 : 2 ladder ratios.
                    cells = int(np.count_nonzero((axis > 0) & (axis <= h)))
                    base = int(np.count_nonzero((rc.axes_for(spec, 1.)[0] > 0)
                                                & (rc.axes_for(spec, 1.)[0] <= h)))
                    assert base % 2 == 0 and cells == round(base * level)
                else:
                    assert np.array_equal(axis, np.linspace(-span / 2, span / 2, int(n * level) + 1))


def test_f3_arm_anchors_strip_refinement_and_unchanged_z():
    from benchmark.datagen.scenes import _partition, _printed_axes

    for spec in rc.draw_scenes(("F3",), 20, 1701):
        height = spec["height"]
        for level, cells in zip(rc.LEVELS, (4, 6, 8)):
            x, y, z = rc.axes_for(spec, level)
            assert -spec["arm_length"] in x and spec["arm_length"] in x
            assert -.001 in y and .001 in y
            assert np.count_nonzero((y >= -.001) & (y <= .001)) - 1 == cells
            old_axes = list(_printed_axes(.002 / level, .004 / level, .0002 / level, height))
            old_axes[2] = _partition((-.025, -height, 0., .025), .004 / level,
                                     fine=((-height, 0., min(height / 7, .0002) / level),),
                                     half_planes=(-.0025, .0025))
            assert np.array_equal(z, old_axes[2])


def test_shards_balance_deterministically_with_family_overheads():
    costs = {"a": 9., "b": 8., "c": 7., "d": 6., "e": 5., "f": 5.}
    tasks = [rc.scheduler.Task(name, "") for name in reversed(costs)]
    plans = {name: {"estimated_seconds": cost} for name, cost in costs.items()}
    shards = rc.shard_tasks(tasks, plans, 2)
    assert [[t.task_id for t in shard] for shard in shards] == [["a", "d", "e"], ["b", "c", "f"]]
    assert rc.shard_tasks(list(reversed(tasks)), plans, 2) == shards
    for spec in rc.draw_scenes(rc.FAMILIES, 1, 1701):
        plan = rc.estimate(spec, 1., 8, 120)
        seconds = plan["estimated_cells"] * plan["estimated_steps"] / 2.7e9
        seconds += plan["estimated_steps"] * {"F3": .96e-3, "F4": .15e-3}.get(spec["family"], 0.)
        assert plan["estimated_seconds"] == pytest.approx(seconds)


def test_partial_shard_waits_for_complete_scene(tmp_path):
    records = []
    for level in rc.LEVELS:
        for pml in (8, 16):
            for stop in rc.STOPS:
                path = tmp_path / f"{level}_{pml}_{stop}.pt"
                torch.save(dict(labels={"points_E": torch.ones(2, 1, 3, dtype=torch.complex64)},
                                valid=torch.ones(2, dtype=torch.bool)), path)
                records.append(dict(
                    task_id=path.stem, path=str(path), status="ok",
                    plan=dict(spec=dict(scene_id="f1_000"), level=level, pml=pml, stopping=stop),
                    qc=dict(final_energy=0., peak_energy=1., finite=True, s11_passed=True,
                            shutoff_triggered=True, mesh_fill=[]),
                ))
    left, right = records[::2], records[1::2]
    assert all(not r["scene_complete"] for r in rc.enrich(left))
    assert all(not r["scene_complete"] for r in rc.enrich(right))
    combined = rc.enrich(left + right)
    assert all(r["scene_complete"] and r["reference_qc_passed"] for r in combined)
    assert all(r["stopping_vs_fixed"]["points_E"] == 0 for r in combined)


def test_f4_radius_ladder_stays_inside_the_thin_wire_band():
    from benchmark.datagen import scenes as datagen
    from witwin.maxwell.compiler import thin_wire

    assert rc.THIN_WIRE_MAX_RADIUS_TO_SPACING == thin_wire._MAX_RADIUS_TO_SPACING
    # The base builder radius, measured on the n=40 grid the F4 ladder starts from.
    sample = datagen.wire("f4_probe", n=40, kind="dipole", count=11, finite=False, device="cpu")
    span = 1.6 * rc.C0 / rc.F0
    assert all(abs(w.radius / (span / 40) - rc.F4_BASE_RADIUS_TO_SPACING) < 1e-12
               for w in sample.scene.thin_wires)
    specs = rc.draw_scenes(("F4",), 30, 1701)
    for spec in specs:
        for level in rc.LEVELS:
            ratio = rc.F4_BASE_RADIUS_TO_SPACING * spec["wire_radius_scale"] * level
            assert ratio <= rc.THIN_WIRE_MAX_RADIUS_TO_SPACING, (spec["scene_id"], level, ratio)
    scales = [spec["wire_radius_scale"] for spec in specs]
    assert min(scales) >= rc.F4_RADIUS_SCALE_RANGE[0] and max(scales) <= rc.F4_RADIUS_SCALE_RANGE[1]


def test_f4_radius_remap_leaves_other_families_and_fields_unchanged():
    families = ("F1", "F2", "F3")
    assert rc.draw_scenes(families, 30, 1701) == [s for s in rc.draw_scenes(rc.FAMILIES, 30, 1701)
                                                  if s["family"] in families]
    for spec in rc.draw_scenes(("F4",), 30, 1701):
        lo, hi = rc.F4_RADIUS_SCALE_RANGE
        draw = .5 + (spec["wire_radius_scale"] - lo) / (hi - lo)
        assert .5 <= draw <= 1.5


@pytest.mark.parametrize("kind", ["dipole", "loop", "helix"])
def test_f4_wire_ports_use_the_gap_binding_across_the_feed(kind):
    from benchmark.datagen import scenes as datagen
    from witwin.maxwell.compiler.thin_wire import _resolve_port_bindings, compile_thin_wires
    from witwin.maxwell.scene import prepare_scene

    n = 40
    dx = 1.6 * rc.C0 / rc.F0 / n
    scene = prepare_scene(datagen.wire("f4_probe", n=n, kind=kind, count=11, finite=False,
                                       device="cpu").scene)
    (wire,) = scene.thin_wires
    (record,) = _resolve_port_bindings(scene, scene.thin_wires)
    assert record["kind"] == "gap" and record["orientation"] == 1
    assert record["positive_point"] == record["negative_point"] + 1
    points = np.asarray(wire.points, dtype=np.float64)
    negative, positive = points[record["negative_point"]], points[record["positive_point"]]
    # The feed gap is the builder's physical dx, centred on the origin plane of the feed axis.
    axis = 0 if kind == "loop" else 2
    assert negative[axis] == pytest.approx(-dx / 2, abs=1e-15)
    assert positive[axis] == pytest.approx(dx / 2, abs=1e-15)
    assert np.linalg.norm(positive - negative) == pytest.approx(dx, rel=1e-12)
    assert bool((points[0] == points[-1]).all()) == (kind == "loop")
    compile_thin_wires(scene, device="cpu")


def test_f4_catalog_ports_use_the_gap_binding():
    from benchmark.datagen.scenes import scene_builders
    from witwin.maxwell.compiler.thin_wire import _resolve_port_bindings
    from witwin.maxwell.scene import prepare_scene

    builders = [b for b in scene_builders() if b.family == "F4"]
    assert len(builders) == 10
    for builder in builders:
        scene = prepare_scene(builder.build(device="cpu").scene)
        (record,) = _resolve_port_bindings(scene, scene.thin_wires)
        assert record["kind"] == "gap", builder.name

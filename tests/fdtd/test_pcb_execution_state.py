import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


def simulation(*, graph=True, thickness=35e-6, coordinate=0, cache=None, monitors=False, window="none", steps=128):

    sheet = mw.PlanarPEC.rectangle(bounds=((-0.025, .025),) * 2, normal_axis="x",
                                   coordinate=coordinate, physical_thickness=thickness)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.04, .04),) * 3), grid=mw.GridSpec.uniform(.01),
        boundary=mw.BoundarySpec.pml(num_layers=4), device="cuda",
        structures=[mw.Structure(sheet, mw.Material.pec(), name="screen")],
        sources=[mw.PointDipole(position=(-.02, 0, 0), polarization="Ez",
                                source_time=mw.CW(frequency=2.45e9))],
    )
    if monitors:
        scene.add_monitor(mw.PointMonitor("point", (-.01, 0, 0), fields=("Ez",)))
        scene.add_monitor(mw.PlaneMonitor("plane", axis="x", position=-.01, fields=("Ez", "Hy")))
        scene.add_monitor(mw.FieldTimeMonitor("trace", position=(-.01, 0, 0), components=("Ez",), interval=3))
        scene.add_monitor(mw.FluxTimeMonitor("flux", axis="x", position=-.01, interval=5))
    return mw.Simulation.fdtd(scene, frequency=2.45e9, cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=steps, transient_cycles=0, steady_cycles=1),
        spectral_sampler=mw.SpectralSampler(window=window, normalize_source=False),
        full_field_dft=True, material_cache=cache)


def compare(actual, expected):
    for name in expected.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=2e-6, atol=2e-8)


def test_planar_pec_graph_and_disk_resume_preserve_actual_fields(tmp_path):
    reference = simulation(graph=False).run()
    captured = simulation().run()
    assert captured.solver_stats["cuda_graph_active"]
    compare(captured, reference)
    checkpoint = simulation().prepare().run_until(64)
    assert any(float(v.abs().max()) > 0 for k, v in checkpoint.physics.tensors.items() if k in ("Ex", "Ey", "Ez"))
    checkpoint.save(tmp_path / "checkpoint.pt")
    restored = mw.FDTDResumeCheckpoint.load(tmp_path / "checkpoint.pt")
    resumed = simulation().prepare().run(resume_from=restored)
    compare(resumed, reference)
    resumed.save(tmp_path / "result.pt")
    reopened = mw.Result.load(tmp_path / "result.pt", scene=resumed.scene)
    for name in resumed.fields:
        torch.testing.assert_close(reopened.fields[name].cpu(), resumed.fields[name].cpu())


def test_planar_physical_identity_changes_reject_old_checkpoint():
    checkpoint = simulation().prepare().run_until(64)
    for changed in (simulation(coordinate=.01), simulation(thickness=18e-6)):
        with pytest.raises(ValueError, match="configuration|fingerprint|compatible|signature"):
            changed.prepare().run(resume_from=checkpoint)


def test_planar_material_cache_copies_masks_and_invalidates_geometry():
    cache = mw.FDTDMaterialCache(max_entries=2)
    uncached = simulation().run()
    first = simulation(cache=cache).prepare()
    second = simulation(cache=cache).prepare()
    assert cache.stats["hits"] == 1
    a, b = first.solver._compiled_material_model, second.solver._compiled_material_model
    assert a["pec_sheet_edges"]["Ey"].data_ptr() != b["pec_sheet_edges"]["Ey"].data_ptr()
    a["pec_sheet_edges"]["Ey"].zero_()
    third = simulation(cache=cache).prepare()
    assert third.solver._compiled_material_model["pec_sheet_edges"]["Ey"].sum() > 0
    compare(second.run(), uncached)
    simulation(cache=cache, thickness=18e-6).prepare()
    assert cache.stats["misses"] == 2


@pytest.mark.parametrize("window", ["none", "hanning"])
@pytest.mark.parametrize("split", [0, 253, 511])
def test_spectral_and_time_observations_survive_disk_resume(tmp_path, window, split):
    reference = simulation(monitors=True, window=window, steps=512).run()
    checkpoint = simulation(monitors=True, window=window, steps=512).prepare().run_until(split)
    checkpoint.save(tmp_path / "state.pt")
    checkpoint = mw.FDTDResumeCheckpoint.load(tmp_path / "state.pt")
    resumed = simulation(monitors=True, window=window, steps=512).prepare().run(resume_from=checkpoint)
    compare(resumed, reference)

    def check(a, b):
        if isinstance(b, dict):
            assert set(a) == set(b)
            for key in b:
                check(a[key], b[key])
        elif torch.is_tensor(b):
            torch.testing.assert_close(torch.as_tensor(a), b, rtol=2e-6, atol=2e-8)
        elif isinstance(b, __import__("numpy").ndarray):
            __import__("numpy").testing.assert_allclose(a, b, rtol=2e-6, atol=2e-8)
        elif isinstance(b, (int, float, complex)):
            assert a == pytest.approx(b, rel=2e-6, abs=2e-8)
    for name in ("point", "plane", "trace", "flux"):
        check(resumed.monitor(name), reference.monitor(name))


def test_corrupt_observation_state_is_rejected_before_field_restore():
    import copy
    from witwin.maxwell.fdtd.resume import restore_resume_checkpoint

    checkpoint = simulation().prepare().run_until(64)
    bad = copy.deepcopy(checkpoint)
    bad.auxiliary["observations"]["groups"][("dft", "Ez")]["real"] = torch.zeros(1)
    prepared = simulation().prepare()
    solver = prepared.solver
    solver.enable_dft(2.45e9, window_type="none", end_step=128)
    solver.build_dft_step_tables(128)
    solver._shutoff_peak = torch.zeros((), device=solver.device, dtype=solver.Ex.dtype)
    solver._normalize_source = False
    before = solver.Ex.clone()
    with pytest.raises(ValueError, match="shape"):
        restore_resume_checkpoint(solver, bad, total_steps=128)
    assert torch.equal(solver.Ex, before)


def test_planar_pec_native_batch_matches_independent_runs():
    configs = [simulation(coordinate=z) for z in (0, .01)]
    reference = [config.run() for config in configs]
    batch = mw.Simulation.batch([simulation(coordinate=z) for z in (0, .01)],
        outputs=mw.BatchOutput(fields=("Ex", "Ey", "Ez")),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=2,
                                    graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        results = prepared.run()
    for actual, expected in zip(results, reference):
        compare(actual, expected)


@pytest.mark.parametrize("initial_graph", [False, True])
def test_resume_can_change_observer_execution_mode(initial_graph):
    expected = simulation(monitors=True, steps=512).run()
    checkpoint = simulation(graph=initial_graph, monitors=True, steps=512).prepare().run_until(253)
    resumed = simulation(graph=not initial_graph, monitors=True, steps=512).prepare().run(resume_from=checkpoint)
    compare(resumed, expected)
    from tests.monitors.observers.test_grouped_ordinary_spectra import check
    check(resumed.monitor("plane"), expected.monitor("plane"))

"""Public Batch execution, result ownership, and lifecycle contracts on CUDA."""
from dataclasses import replace

import pytest
import torch
import witwin.maxwell as mw

from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Batch execution requires CUDA")


def batch(sims, *, size=2, outputs=None, **kwargs):
    if "parameters" not in kwargs:
        kwargs.setdefault("variation", mw.BatchVariation(sources=("drive",)))
    return mw.Simulation.batch(sims, execution=mw.BatchExecution(devices=("cuda:0",), batch_size=size),
                               outputs=outputs, **kwargs)


def probe(result):
    return result.raw_monitor("probe")["components"]["Ez"]


@pytest.mark.parametrize("size,count", [(1, 3), (2, 3), (3, 3), (4, 7)])
def test_public_batch_complete_results_tail_and_repeated_run(size, count):
    sims = [simulation(i) for i in range(count)]
    with batch(sims, size=size).prepare() as prepared:
        result = prepared.run()
        assert isinstance(result, mw.BatchResult)
        assert not result.failed
        saved = result.stack(probe).clone()
        ptrs = {name: value.data_ptr() for name, value in prepared._slot.buffers.items()}
        again = prepared.run()
        assert not again.failed
        assert {name: value.data_ptr() for name, value in prepared._slot.buffers.items()} == ptrs
        torch.testing.assert_close(again.stack(probe), saved, rtol=0, atol=0)
        assert all(r.solver is None and r.raw_output is None and r._prepared_scene is None for r in result)
    torch.testing.assert_close(result.stack(probe), saved, rtol=0, atol=0)
    for index, sim in enumerate(sims):
        torch.testing.assert_close(probe(result[index]), probe(sim.run()), rtol=1e-4, atol=1e-6)
        assert result[index]._metadata["variant_id"] == str(index)
    with pytest.raises(RuntimeError, match="closed"):
        prepared.run()


def test_template_source_updates_aba_reset_and_independent_results():
    sim = simulation()
    a = mw.BatchParameters(source_scales={"drive": torch.tensor([1., 2., 3.], device="cuda:0")})
    b = mw.BatchParameters(source_scales={"drive": torch.tensor([4., 0., -2.], device="cuda:0")})
    submission = batch(sim, parameters=a)
    with submission.prepare() as prepared:
        first = prepared.run()
        middle = prepared.run(parameters=b)
        final = prepared.run(parameters=a)
        assert not first.failed and not middle.failed and not final.failed
        torch.testing.assert_close(first.stack(probe), final.stack(probe), rtol=0, atol=0)
        torch.testing.assert_close(probe(middle[0]), probe(first[0]) * 4., rtol=1e-4, atol=1e-6)
        assert torch.count_nonzero(probe(middle[1])) == 0
        a.source_scales["drive"].mul_(2)
        updated = prepared.run()
        torch.testing.assert_close(updated.stack(probe), first.stack(probe) * 2, rtol=1e-4, atol=1e-6)


def test_monitor_only_31_frequencies_has_no_full_field_dft():
    frequencies = tuple(0.7e9 + i * 0.02e9 for i in range(31))
    sims = [simulation(i, frequencies=frequencies) for i in range(3)]
    with batch(sims).prepare() as prepared:
        result = prepared.run()
        assert not result.failed
        assert all(not s.dft_enabled and not s._dft_entries for s in prepared._slot.solvers)
        assert all(not r._fields for r in result)
    for sim, r in zip(sims, result):
        torch.testing.assert_close(probe(r), probe(sim.run()), rtol=1e-4, atol=1e-6)


def test_density_template_updates_match_independent_scenes():
    sim = simulation()
    sim.scene.add_material_region(mw.MaterialRegion(name="rho",
        geometry=mw.Box(size=(0.05, 0.05, 0.05), device="cuda:0"),
        density=torch.zeros(2, 2, 2, device="cuda:0"), eps_bounds=(1., 5.)))
    parameters = mw.BatchParameters(region_densities={"rho": torch.rand(3, 2, 2, 2, device="cuda:0")})
    submission = batch(sim, parameters=parameters)
    result = submission.run()
    assert not result.failed
    for i in range(3):
        reference = mw.Simulation.fdtd(submission.scene(i), frequencies=sim.frequencies,
            run_time=sim.config.run_time, spectral_sampler=sim.config.spectral_sampler, cuda_graph=False).run()
        torch.testing.assert_close(probe(result[i]), probe(reference), rtol=1e-4, atol=1e-6)


def test_iteration_close_cancel_nonreentrancy_and_output_ownership():
    submission = batch([simulation(i) for i in range(5)])
    with submission.prepare() as prepared:
        iterator = prepared.iter_results()
        with pytest.raises(RuntimeError, match="concurrent"):
            prepared.run()
        with pytest.raises(RuntimeError, match="active batch iterator"):
            prepared.close()
        first = next(iterator)
        saved = probe(first.result).clone()
        iterator.close()
        assert len(iterator.records) == 5
        assert iterator.records[0].completed
        assert all(r.status == "cancelled" for r in iterator.records[1:])
        prepared.run()
        assert all(r.status == "cancelled" for r in iterator.records[1:])
        torch.testing.assert_close(probe(first.result), saved, rtol=0, atol=0)
        never_started = prepared.iter_results()
        never_started.close()
        assert all(r.status == "cancelled" for r in never_started.records)
        with prepared.iter_results() as cancelled:
            prepared.request_cancel()
            assert all(item.failure.kind.value == "cancelled" for item in cancelled)
    top = submission.iter_results()
    next(top)
    top.close()
    assert top.prepared._closed


def test_result_persistence_cpu_export_and_explicit_field_selection(tmp_path):
    sims = [simulation(i) for i in range(2)]
    output = mw.BatchOutput(fields=("Ez",), result_device="cpu")
    result = batch(sims, outputs=output).run()
    assert not result.failed
    assert set(result[0].field("E")) == {"EZ"}
    assert result[0].tensor("Ez").device.type == "cpu"
    assert probe(result[0]).device.type == "cpu"
    result.save(tmp_path)
    loaded = mw.BatchResult.load(tmp_path, scenes=[s.scene for s in sims])
    torch.testing.assert_close(loaded.stack(probe), result.stack(probe), rtol=0, atol=0)
    torch.testing.assert_close(loaded[0].tensor("Ez"), result[0].tensor("Ez"), rtol=0, atol=0)
    assert loaded.variant_ids == result.variant_ids
    with pytest.raises(FileExistsError, match="empty"):
        result.save(tmp_path)


def test_input_order_across_compatible_groups():
    sims = [simulation(), simulation(frequencies=(2e9,)), simulation(2)]
    submission = mw.Simulation.batch(sims, execution=mw.BatchExecution(
        devices=("cuda:0",), batch_size=2, grouping="compatible"),
        variation=mw.BatchVariation(sources=("drive",)))
    with submission.iter_results(order="input") as iterator:
        assert [item.index for item in iterator] == [0, 1, 2]
    with submission.iter_results(order="completion") as iterator:
        assert [item.index for item in iterator] == [0, 2, 1]


def test_runtime_failure_records_and_cancelled_tail(monkeypatch):
    with batch([simulation(i) for i in range(5)]).prepare() as prepared:
        def fail():
            raise RuntimeError("injected field launch failure")
        monkeypatch.setattr(prepared._slot, "advance", fail)
        results = prepared.run()
        assert [r.status for r in results.records] == ["failed", "failed", "cancelled", "cancelled", "cancelled"]
        assert len(results.failures) == 5
        with pytest.raises(mw.DistributedFailure, match="injected"):
            results[0]


def test_result_lifetime_does_not_depend_on_cyclic_gc():
    import gc
    import weakref
    with batch([simulation(steps=8)], size=1).prepare() as prepared:
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            result = prepared.run()
            tensor = weakref.ref(probe(result[0]))
            view = weakref.ref(result[0])
            del result
            assert view() is None and tensor() is None
            baseline = torch.cuda.memory_allocated("cuda:0")
            for _ in range(25):
                result = prepared.run()
                assert not result.failed
                del result
            assert torch.cuda.memory_allocated("cuda:0") <= baseline + 1024**2
        finally:
            if was_enabled:
                gc.enable()


def test_trainable_scene_added_after_simulation_construction_is_detected():
    sim = simulation()
    sim.scene.add_material_region(mw.MaterialRegion(name="rho",
        geometry=mw.Box(size=(0.05, 0.05, 0.05), device="cuda:0"),
        density=torch.zeros(2, 2, 2, device="cuda:0", requires_grad=True), eps_bounds=(1., 5.)))
    results = batch([sim]).run()
    results.results()
    assert results[0].raw_monitor("probe")["components"]["Ez"].requires_grad

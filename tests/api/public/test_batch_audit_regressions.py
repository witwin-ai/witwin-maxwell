"""Regressions for independently reproduced batch cache and ownership failures."""
import threading
import weakref

import pytest
import torch
import witwin.maxwell as mw

from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Batch audit requires CUDA")


def submission(scales=None):
    scales = torch.ones(1, device="cuda:0") if scales is None else scales
    return mw.Simulation.batch(simulation(steps=8),
        parameters=mw.BatchParameters(source_scales={"drive": scales}),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1))


def probe(results):
    return results[0].raw_monitor("probe")["components"]["Ez"]


def test_temporary_source_parameters_are_anchored_until_cache_replaced():
    with submission().prepare() as prepared:
        reference = probe(prepared.run()).clone()
        previous = None
        for value in (2., 3., 5., 7., 11.):
            scales = torch.tensor([value], device="cuda:0")
            current = weakref.ref(scales)
            result = prepared.run(parameters=mw.BatchParameters(source_scales={"drive": scales}))
            del scales
            assert current() is not None, "Cache identity must stay alive after temporary inputs disappear."
            if previous is not None:
                assert previous() is None, "Replacing a cohort must release its previous input anchor."
            torch.testing.assert_close(probe(result), reference * value, rtol=1e-4, atol=0)
            previous = current
    assert previous() is None


def test_inference_tensors_refresh_values_without_version_counters():
    with torch.inference_mode():
        scales = torch.ones(1, device="cuda:0")
        with submission(scales).prepare() as prepared:
            reference = probe(prepared.run()).clone()
            scales.mul_(3)
            torch.testing.assert_close(probe(prepared.run()), reference * 3, rtol=1e-4, atol=0)


def test_concurrent_close_cannot_release_a_running_slot(monkeypatch):
    with submission().prepare() as prepared:
        iterator = prepared.iter_results()
        entered, release = threading.Event(), threading.Event()
        original = prepared._solve_cohort
        def held():
            entered.set()
            assert release.wait(10)
            return original()
        monkeypatch.setattr(prepared, "_solve_cohort", held)
        results, errors = [], []
        def consume():
            try:
                results.append(next(iterator))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert entered.wait(10)
            with pytest.raises(RuntimeError, match="next.*active"):
                iterator.close()
            with pytest.raises(RuntimeError, match="next.*active"):
                next(iterator)
            assert prepared._gate.locked()
            assert not iterator._closed
            with pytest.raises(RuntimeError, match="concurrent"):
                prepared.run()
            prepared.request_cancel()
        finally:
            release.set()
            worker.join(15)
        assert not worker.is_alive() and not errors
        assert results[0].failure.kind.value == "cancelled"
        iterator.close()
        assert not prepared._gate.locked()


def test_close_during_validation_cannot_revive_prepared_context(monkeypatch):
    prepared = submission().prepare()
    entered, release = threading.Event(), threading.Event()
    original = prepared._validate_inputs
    def held(parameters):
        original(parameters)
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(prepared, "_validate_inputs", held)
    results, errors = [], []
    def create():
        try:
            results.append(prepared.iter_results())
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=create)
    worker.start()
    try:
        assert entered.wait(10)
        prepared.close()
    finally:
        release.set()
        worker.join(15)
    assert not worker.is_alive() and not results
    assert len(errors) == 1 and "closed" in str(errors[0])
    assert not prepared._gate.locked() and prepared._slot is None


def test_structure_outside_its_envelope_fails_before_execution():
    sim = simulation(steps=8)
    batch = mw.Simulation.batch([sim], execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1),
        variation=mw.BatchVariation(structures={"design": mw.Box(
            position=(-.15, 0, 0), size=(.001, .001, .001), device="cuda:0")}))
    with pytest.raises(ValueError, match="influence envelope"):
        batch.prepare()


def test_parameter_cache_keeps_target_names_when_mapping_order_changes():
    sim = simulation(steps=35)
    sim.scene.add_source(mw.PointDipole(name="second", position=(0., -.05, 0.), polarization="Ez",
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9)))
    first = torch.tensor([.3], device="cuda:0")
    second = torch.tensor([1.2], device="cuda:0")
    batch = mw.Simulation.batch(sim,
        parameters=mw.BatchParameters(source_scales={"drive": first, "second": second}),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1, graph="required"))
    updated = mw.BatchParameters(source_scales={"second": first, "drive": second})
    with batch.prepare() as prepared:
        original = prepared.run()
        original.results()
        swapped = prepared.run(parameters=updated)
        swapped.results()
    fresh = mw.Simulation.batch(sim, parameters=updated, execution=batch.execution).run()
    fresh.results()
    actual = swapped[0].raw_monitor("probe")["components"]["Ez"]
    torch.testing.assert_close(actual, fresh[0].raw_monitor("probe")["components"]["Ez"], rtol=0, atol=0)
    assert not torch.equal(actual, original[0].raw_monitor("probe")["components"]["Ez"])

"""GPU observer batching checks independent of the ordinary stepping loop."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation
from witwin.maxwell.fdtd.cuda.backend import get_compiled_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("frequencies", [(1e9,), tuple(.7e9 + i*.02e9 for i in range(31))])
@pytest.mark.parametrize("count", [1, 3])
def test_frequency_and_lane_axes_use_one_group_launch(monkeypatch, frequencies, count):
    from witwin.maxwell.fdtd.observers import BatchObserverEngine
    native = get_compiled_extension()
    calls = []
    original = BatchObserverEngine.advance
    def measured(engine):
        calls.append(len(engine.groups))
        return original(engine)
    monkeypatch.setattr(BatchObserverEngine, "advance", measured)
    sims = [simulation(i, frequencies=frequencies, steps=24) for i in range(count)]
    result = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=4)).run()
    result.results()
    assert calls == [1] * 24
    for i, sim in enumerate(sims):
        torch.testing.assert_close(result[i].raw_monitor("probe")["components"]["Ez"],
            sim.run().raw_monitor("probe")["components"]["Ez"], rtol=1e-4, atol=1e-7)


def test_gather_mask_bounds_and_alias_guards():
    native = get_compiled_extension()
    field = torch.arange(16, device="cuda:0", dtype=torch.float32).view(2, 2, 2, 2)
    offsets = torch.tensor([[0, 7], [8, 15]], device="cuda:0")
    mapping = torch.tensor([[0, 1], [2, 3]], device="cuda:0")
    weights = torch.ones((2, 2, 4), device="cuda:0")
    active = torch.tensor([1, 0], dtype=torch.int32, device="cuda:0")
    real, imag = torch.zeros((2, 2, 2), device="cuda:0"), torch.zeros((2, 2, 2), device="cuda:0")
    native.batch_observer_gather(field, offsets, mapping, weights, active, real, imag, False)
    torch.testing.assert_close(real[0], field.flatten()[offsets[0]].expand(2, 2))
    assert real[1].count_nonzero() == 0
    with pytest.raises(RuntimeError, match="overlap"):
        native.batch_observer_gather(field, offsets, mapping, weights, active, real, real, False)
    backing = torch.empty(9, device="cuda:0")
    with pytest.raises(RuntimeError, match="overlap"):
        native.batch_observer_gather(field, offsets, mapping, weights, active,
                                     backing[:8].view(2, 2, 2), backing[1:].view(2, 2, 2), False)

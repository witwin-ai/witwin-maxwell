from __future__ import annotations

import gc

import pytest
import torch

from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA Graph tests.")


def test_cuda_graph_runner_replays_static_in_place_update():
    value = torch.zeros((8,), device="cuda", dtype=torch.float32)
    increment = torch.full_like(value, 0.25)
    runner = CudaGraphRunner(device=value.device, warmup_steps=0)

    replay = runner.capture(lambda: value.add_(increment))
    value.zero_()
    replay()
    replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(value, torch.full_like(value, 0.5), rtol=0.0, atol=0.0)
    assert runner.graph is not None


def test_cuda_graph_runner_disabled_returns_original_callable():
    value = torch.zeros((1,), device="cuda", dtype=torch.float32)
    runner = CudaGraphRunner(device=value.device, enabled=False)
    fn = lambda: value.add_(1.0)

    replay = runner.capture(fn)
    replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(value, torch.ones_like(value), rtol=0.0, atol=0.0)
    assert runner.graph is None


def test_capture_defers_cyclic_device_resource_finalizers():
    value = torch.zeros((1,), device="cuda")
    finalized = []

    class DeviceResource:
        def __del__(self):
            torch.cuda.synchronize()
            finalized.append(True)

    previous_threshold = gc.get_threshold()
    previous_enabled = gc.isenabled()
    gc.enable()
    try:
        gc.set_threshold(10, 10, 10)

        def step():
            resource = DeviceResource()
            resource.cycle = resource
            del resource
            # Allocation pressure makes the cyclic CUDA owner collectible
            # inside capture unless collection is deferred.
            cycles = []
            for _ in range(100):
                cycle = []
                cycle.append(cycle)
                cycles.append(cycle)
            value.add_(1.0)

        replay = CudaGraphRunner(device=value.device, warmup_steps=0).capture(step)
        assert gc.isenabled()
        replay()
        torch.testing.assert_close(value, torch.ones_like(value))
        gc.collect()
        assert finalized == [True]
    finally:
        gc.set_threshold(*previous_threshold)
        if not previous_enabled:
            gc.disable()

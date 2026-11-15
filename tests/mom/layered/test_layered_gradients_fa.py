"""GR1 timing gate on F-A: every Y gradient costs at most twice the forward solve.

The forward is one complete Scene -> Simulation -> Result run (Green table,
assembly, dense solve) with gradients attached. Backward is the complete
Jacobian dY/dsigma of all K*P*P complex entries: 2*K*P*P reverse passes, each
one native loss_adjoint contraction plus the PyTorch impedance map.
"""
import gc
import time

import pytest
import torch

import tests.support.rfic_fixtures as fixtures
from tests.mom.layered.test_layered_gradients import derivative

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


def test_fa_all_y_gradients_within_twice_forward(monkeypatch, record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    frequencies = fixtures.SWEEP[::6]
    with torch.no_grad():
        fixtures.fixture_fa().simulation(frequencies).run()  # warm the extension and allocator
    scale = torch.tensor(1., dtype=torch.float64, device="cuda", requires_grad=True)
    monkeypatch.setattr(fixtures, "SIGMA", fixtures.SIGMA * scale)
    simulation = fixtures.fixture_fa().simulation(frequencies)
    torch.cuda.synchronize()
    start = time.perf_counter()
    y = simulation.run().network.to_y()
    torch.cuda.synchronize()
    forward = time.perf_counter() - start
    start = time.perf_counter()
    single, = torch.autograd.grad(y.abs().square().sum(), scale, retain_graph=True)
    torch.cuda.synchronize()
    one_pass = time.perf_counter() - start
    start = time.perf_counter()
    jacobian = derivative(y, scale)
    torch.cuda.synchronize()
    complete = time.perf_counter() - start
    record_property("forward_seconds", forward)
    record_property("one_reverse_pass_seconds", one_pass)
    record_property("all_y_gradients_seconds", complete)
    record_property("reverse_passes", 2 * y.numel())
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())
    assert torch.isfinite(jacobian.abs()).all() and torch.isfinite(single)
    assert complete <= 2 * forward
    assert forward + complete < 120
    assert torch.cuda.max_memory_allocated() < 6 * 2**30

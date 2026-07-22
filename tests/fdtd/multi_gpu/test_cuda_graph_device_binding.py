"""CUDA-graph capture must bind to the solver's device, and must stand down
while a concurrent executor owns the process.

Both properties became load-bearing when CUDA-graph stepping turned into the
public default, because every run on a non-default GPU and every ensemble task
then went through capture.

1. ``torch.cuda.graph`` opens its capture stream on whatever device is current
   and caches that stream on its class. Capturing a closure whose tensors live
   on another device therefore records an *empty* graph: the warmup and capture
   launches still execute eagerly on the tensors' own stream, and every later
   ``replay()`` is a silent no-op. The solve does not crash -- it silently stops
   integrating after the warmup steps and reports plausible-looking but wrong
   fields. This is the configuration a ``cuda:0``-only test can never reach.

2. Capture is process-global: only one may be underway at a time, and PyTorch's
   default ``global`` capture mode makes an ordinary synchronizing call in *any*
   other thread fail with ``cudaErrorStreamCaptureUnsupported``. A concurrent
   executor must therefore suspend capture for the whole plan.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.cuda.runtime.graph import (
    CudaGraphRunner,
    capture_suspended,
    suspend_capture,
)

_FREQUENCY = 1.0e9


def _vacuum_scene(device):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.25, 0.25), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.none(),
        device=str(device),
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=1.0),
            name="src",
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ex", "Ey", "Ez")))
    return scene


def _run(device, *, cuda_graph: bool, time_steps: int = 40):
    return mw.Simulation.fdtd(
        _vacuum_scene(device),
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=cuda_graph,
    ).run()


def test_graph_stepping_on_non_default_device_matches_eager(
    cuda_p2p_devices, cuda_memory_cleanup
):
    """The excluded configuration: the solver device is not the current device.

    Run on ``cuda:1`` while the calling thread still has ``cuda:0`` current --
    exactly what an unbound capture gets wrong -- and require bit-identical
    fields against eager stepping. A silently empty graph shows up here as a
    field that stopped advancing after the capture warmup.
    """

    device = cuda_p2p_devices[1]
    torch.cuda.set_device(cuda_p2p_devices[0])
    assert torch.cuda.current_device() != device.index

    eager = _run(device, cuda_graph=False)
    graphed = _run(device, cuda_graph=True)

    assert set(eager.fields) == set(graphed.fields)
    for name, reference in eager.fields.items():
        assert str(graphed.fields[name].device) == str(device)
        assert torch.equal(graphed.fields[name], reference), (
            f"{name} diverges between graph and eager stepping on {device}"
        )
    # A no-op replay leaves the field at its warmup amplitude, so the peak is a
    # blunt second witness that the graph actually carried the whole run.
    assert float(torch.abs(graphed.fields["EZ"]).max()) == pytest.approx(
        float(torch.abs(eager.fields["EZ"]).max()), rel=0.0, abs=0.0
    )


def test_runner_rejects_a_non_cuda_device():
    with pytest.raises(ValueError, match="CUDA device"):
        CudaGraphRunner(device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_capture_stands_down_while_suspended():
    """A suspended capture must raise, not silently return the eager closure.

    The callers' fallback is ``except Exception: use the eager path``; returning
    ``fn`` instead would leave their ``*_graph_active`` flags set and drive the
    graph-completion bookkeeping for a graph that does not exist.
    """

    assert not capture_suspended()
    runner = CudaGraphRunner(device="cuda:0", warmup_steps=0)
    with suspend_capture():
        assert capture_suspended()
        with pytest.raises(RuntimeError, match="suspended"):
            runner.capture(lambda: None)
    assert not capture_suspended()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_suspend_capture_is_reference_counted():
    with suspend_capture():
        with suspend_capture():
            assert capture_suspended()
        assert capture_suspended()
    assert not capture_suspended()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_concurrent_plans_suspend_capture_and_serial_plans_do_not():
    """The executor is the layer that knows whether tasks overlap."""

    from witwin.maxwell.execution.executor import execute_plan
    from witwin.maxwell.execution.plan import ExecutionPlan, ExecutionTask

    observed: list[bool] = []

    def _probe(_device):
        observed.append(capture_suspended())
        return None

    def _plan(task_count, max_concurrency):
        return ExecutionPlan(
            tasks=tuple(
                ExecutionTask(index=index, run=_probe, estimated_bytes=None, label=str(index))
                for index in range(task_count)
            ),
            placement="round_robin",
            max_concurrency=max_concurrency,
            fail_fast=False,
        )

    from witwin.maxwell.execution.pool import DevicePool

    pool = DevicePool(("cuda:0",), per_device_concurrency=1, require_cuda=False)

    observed.clear()
    execute_plan(_plan(2, 1), pool)
    assert observed == [False, False]

    observed.clear()
    execute_plan(_plan(2, 2), pool)
    assert observed == [True, True]

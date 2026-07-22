"""CUDA-graph capture must bind to the solver's device.

This became load-bearing when CUDA-graph stepping turned into the public
default, because every run on a non-default GPU then went through capture.

``torch.cuda.graph`` opens its capture stream on whatever device is current
and caches that stream on its class. Capturing a closure whose tensors live on
another device therefore records an *empty* graph: the warmup and capture
launches still execute eagerly on the tensors' own stream, and every later
``replay()`` is a silent no-op. The solve does not crash -- it silently stops
integrating after the warmup steps and reports plausible-looking but wrong
fields. This is the configuration a ``cuda:0``-only test can never reach.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner

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

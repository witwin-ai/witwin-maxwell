"""Distributed timing configuration and detached recording contracts."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed.adjoint import (
    _DistributedFDTDGradientBridge, _NcclDistributedFDTDGradientBridge,
)


@pytest.mark.parametrize("bridge_type", [_DistributedFDTDGradientBridge, _NcclDistributedFDTDGradientBridge])
@pytest.mark.parametrize("explicit_dt", [None, 1e-12])
def test_distributed_bridges_bind_requested_timing_before_field_initialization(monkeypatch, bridge_type, explicit_dt):
    import witwin.maxwell.fdtd.distributed as distributed_module

    scene = mw.Scene(sources=(mw.PointDipole((0., 0., 0.), source_time=mw.CW(2e9)),))
    simulation = mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(run_time=mw.TimeConfig(dt=explicit_dt), device="cpu"))
    simulation._fdtd_optimization_dt = 2e-12
    seen = {}

    class InitializationReached(Exception):
        pass

    class ProbeSolver:
        def __init__(self, scene, **kwargs):
            self.scene = scene
            seen["frequency"] = kwargs["frequency"]

        def init_field(self):
            seen["dt"] = self._requested_dt
            seen["source_frequency"] = self._requested_source_frequency
            raise InitializationReached

    monkeypatch.setattr(distributed_module, "DistributedFDTD", ProbeSolver)
    bridge = object.__new__(bridge_type)
    bridge.simulation = simulation
    with pytest.raises(InitializationReached):
        bridge.forward(())
    assert seen == {"frequency": 1e9, "source_frequency": 2e9, "dt": explicit_dt or 2e-12}


def test_shard_local_adjoint_outputs_keep_time_records_outside_spectral_pack():
    from witwin.maxwell.fdtd.adjoint.core import _prepare_forward_pack

    trace = {"kind": "field_time", "differentiable": False, "t": torch.arange(3),
             "components": {"Ez": torch.arange(3.)}, "fields": ("Ez",)}
    solver = SimpleNamespace(dft_enabled=False, observers_enabled=False, time_observers_enabled=True,
                             get_time_observer_results=lambda: {"trace": trace})
    raw = _DistributedFDTDGradientBridge._shard_local_raw_output(solver)
    assert raw["time_observers"]["trace"] is trace
    assert _prepare_forward_pack(raw).output_tensors == ()


def _scene(density, *, time_only):
    scene = mw.Scene(subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"), domain=mw.Domain(bounds=((-0.4, .4),) * 3),
                grid=mw.GridSpec.uniform(.1), boundary=mw.BoundarySpec.none(),
                     material_regions=(
                         mw.MaterialRegion("design", geometry=mw.Box(position=(0., 0., 0.), size=(.4, .4, .4)),
                             density=density, eps_bounds=(1., 4.)),
                     ),
                     sources=(
                         mw.PointDipole((-.2, 0., 0.), polarization="Ez", profile="ideal",
                             source_time=mw.GaussianPulse(1e9, .5e9)),
                     ),
                     monitors=(
                         mw.monitors.FieldTimeMonitor("trace", components=("Ez", "Hy"), position=(.1, 0., 0.), interval=3),
                     ))
    if not time_only:
        scene = scene.updated(monitors=(*scene.monitors, mw.PointMonitor("spectrum", (.1, 0., 0.), fields=("Ez",))))
    return scene


@pytest.mark.parametrize("time_only", [False, True])
def test_distributed_trainable_time_records_and_explicit_dt_match_single_gpu(cuda_p2p_devices, cuda_memory_cleanup, time_only):
    results = []
    for distributed, grad in ((False, False), (True, False), (True, True)):
        density = torch.full((2, 2, 2), .5, device="cuda:0", requires_grad=grad)
        parallel = mw.fdtd.FDTDParallelConfig(devices=cuda_p2p_devices, gather_fields=False, overlap=False,
            result_device=cuda_p2p_devices[0]) if distributed else None
        result = mw.Simulation.fdtd(_scene(density, time_only=time_only), frequencies=1e9,
            config=mw.FDTDConfig(
                run_time=mw.TimeConfig(time_steps=100, dt=5e-11),
                spectral_sampler=mw.fdtd.SpectralSampler(window="none"), cuda_graph=False, parallel=parallel,
            )).run()
        results.append(result)
        assert result.solver.dt == 5e-11
        trace = result.raw_monitor("trace")
        assert trace["differentiable"] is False
        # A trainable run guards its detached traces: same values, and an objective built
        # from one fails at backward naming the monitor instead of a silent zero gradient.
        assert all(value.requires_grad == grad for value in trace["components"].values())
        if grad:
            with pytest.raises(RuntimeError, match=r"Monitor 'trace' \(\w+\) records detached time-domain traces"):
                trace["components"]["Ez"].sum().backward()
        if grad and not time_only:
            result.monitor("spectrum").sel(component="Ez").values.abs().square().sum().backward()
            assert density.grad is not None and torch.isfinite(density.grad).all()
    for result in results[1:]:
        expected, actual = results[0].raw_monitor("trace"), result.raw_monitor("trace")
        torch.testing.assert_close(actual["t"], expected["t"], rtol=0, atol=0)
        for component in expected["fields"]:
            torch.testing.assert_close(actual["components"][component], expected["components"][component], rtol=1e-5, atol=1e-5)

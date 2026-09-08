"""Independent source-order, sampling-clock, and raw-recording regressions."""
from types import SimpleNamespace
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.sources import compile_source_time
from witwin.maxwell.fdtd.runtime.timeplan import compile_fdtd_time_plan


def _plan(waveforms, config, **kwargs):
    solver = SimpleNamespace(dt=1e-11, c=299792458., _compiled_sources=[
        {"source_time": compile_source_time(source, default_frequency=1e9)} for source in waveforms])
    scene = SimpleNamespace(domain=SimpleNamespace(bounds=((-0.1, 0.1),) * 3), circuits=())
    return compile_fdtd_time_plan(solver, scene, config, [1e9], **kwargs)


def test_late_pulse_and_cw_plan_is_order_independent():
    sources = [mw.CW(1e9), mw.GaussianPulse(1e9, 0.5e9, delay=1e-6)]
    a = _plan(sources, mw.TimeConfig.auto())
    b = _plan(sources[::-1], mw.TimeConfig.auto())
    assert a == b
    assert a["total_steps"] > 100000
    assert a["spectral_start_step"] == 0
    assert a["window"] == "none"
    assert not a["shutoff_allowed"]


def test_all_cw_frequencies_control_measurement_period():
    plan = _plan([mw.CW(1e9), mw.CW(0.5e9)], mw.TimeConfig.auto(transient_cycles=100, steady_cycles=20))
    assert plan["spectral_start_step"] == math.ceil(200e-9 / 1e-11)
    assert plan["total_steps"] == math.ceil(240e-9 / 1e-11)


def test_explicit_short_run_and_unknown_support_are_visible():
    late = mw.GaussianPulse(1e9, 0.5e9, delay=1e-6)
    plan = _plan([late], mw.TimeConfig(time_steps=40))
    assert plan["total_steps"] == 40 and plan["truncated"] and not plan["sources_complete"]
    assert plan["shutoff_min_step"] > 100000
    custom = mw.CustomSourceTime(lambda t: math.sin(t * 1e9), characteristic_frequency=1e9)
    with pytest.raises(ValueError, match="explicit TimeConfig"):
        _plan([custom], mw.TimeConfig.auto())
    assert not _plan([custom], mw.TimeConfig(time_steps=40))["shutoff_allowed"]


def test_source_normalization_uses_common_spectrum():
    with pytest.raises(NotImplementedError, match="single incident"):
        _plan([mw.CW(1e9), mw.CW(2e9)], mw.TimeConfig(time_steps=10), normalize_source=True)
    _plan([mw.CW(1e9), mw.CW(1e9)], mw.TimeConfig(time_steps=10), normalize_source=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multisource_forward_is_invariant_under_source_permutation():
    outputs = []
    for reverse in (False, True):
        scene, _ = _scene(cw=True)
        sources = list(scene.sources) + [mw.PointDipole(position=(0.2, 0., 0.), polarization="Ez", width=0.1,
            source_time=mw.GaussianPulse(0.8e9, 0.3e9, delay=50e-9))]
        result = mw.Simulation.fdtd(scene.clone(sources=sources[::-1] if reverse else sources), frequency=1e9,
            run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=3),
            shutoff=0.1, shutoff_check_interval=20, cuda_graph=True).run()
        outputs.append(result)
        assert not result.stats()["shutoff_triggered"]
        assert result.stats()["time_plan"]["spectral_start_step"] == 0
    assert outputs[0].stats()["dt"] == outputs[1].stats()["dt"]
    assert outputs[0].stats()["time_plan"] == outputs[1].stats()["time_plan"]
    first, second = (output.monitor("point")["components"]["Ez"] for output in outputs)
    # Source addition order changes float32 rounding. Scale by the trace norm,
    # so zero crossings do not turn a few field ULPs into a relative failure.
    assert torch.linalg.vector_norm(first - second) / torch.linalg.vector_norm(first) < 1e-6


def _scene(grad=False, time_only=False, cw=False):
    density = torch.full((1, 1, 1), 0.5, device="cuda", requires_grad=grad)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.4, 0.4),) * 3), grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.mur(), device="cuda")
    scene.add_material_region(mw.MaterialRegion(name="density", geometry=mw.Box(position=(0., 0., 0.), size=(0.2,) * 3),
        density=density, eps_bounds=(1., 4.)))
    scene.add_source(mw.PointDipole(position=(-0.2, 0., 0.), polarization="Ez", width=0.1,
        source_time=mw.CW(1e9) if cw else mw.GaussianPulse(1e9, 0.5e9)))
    scene.add_monitor(mw.FieldTimeMonitor("point", components=("Ez", "Hy")))
    if not time_only:
        scene.add_monitor(mw.PointMonitor("spectrum", (0., 0., 0.), fields=("Ez", "Hy")))
    return scene, density


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("time_only", [False, True])
def test_adjoint_records_point_plane_and_flux_time_monitors(time_only):
    outputs = []
    for grad in (False, True):
        scene, density = _scene(grad, time_only)
        scene.add_monitor(mw.FieldTimeMonitor("plane", components=("Ez",), size=(0.4, 0.4, 0.), start=7, stop=138, interval=13))
        scene.add_monitor(mw.FluxTimeMonitor("flux", axis="x", position=0.1, start=5, stop=140, interval=17))
        simulation = mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=140),
            spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=False)
        simulation.config.adjoint_checkpoint_stride = 17
        result = simulation.run()
        outputs.append(result)
        if grad and not time_only:
            result.monitor("spectrum")["components"]["Ez"].abs().square().sum().backward()
            assert density.grad is not None and torch.isfinite(density.grad).all()
    for name in ("point", "plane", "flux"):
        expected, actual = (result.monitor(name) for result in outputs)
        torch.testing.assert_close(actual["t"], expected["t"], rtol=0, atol=0)
        assert actual["differentiable"] is False
        if name == "flux":
            torch.testing.assert_close(actual["flux"], expected["flux"], rtol=1e-5, atol=1e-7)
            assert not actual["flux"].requires_grad
        else:
            for component in actual["fields"]:
                torch.testing.assert_close(actual["components"][component], expected["components"][component], rtol=1e-5, atol=1e-7)
                assert not actual["components"][component].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph", [False, True])
def test_configured_transient_matches_independent_time_dft(graph):
    scene, _ = _scene(cw=True)
    result = mw.Simulation.fdtd(scene, frequency=1e9,
        run_time=mw.TimeConfig.auto(transient_cycles=100, steady_cycles=20), cuda_graph=graph).run()
    plan = result.stats()["time_plan"]
    trace = result.monitor("point")
    t = trace["t"]
    start = math.ceil(100e-9 / result.solver.dt)
    assert plan["spectral_start_step"] == start
    n = torch.arange(start, len(t), device=t.device, dtype=torch.float64)
    weights = 0.5 * (1 - torch.cos(2 * math.pi * (n - start) / len(n)))
    for component in ("Ez", "Hy"):
        times = t[start:] - (0.5 * result.solver.dt if component.startswith("H") else 0)
        phasor = 2 * (trace["components"][component][start:] * weights * torch.exp(2j * math.pi * 1e9 * times)).sum() / weights.sum()
        torch.testing.assert_close(result.monitor("spectrum")["components"][component].to(torch.complex128).squeeze(),
            phasor, rtol=3e-4, atol=1e-9)

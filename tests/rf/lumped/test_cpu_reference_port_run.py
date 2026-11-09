"""A CPU-reference port run must not touch CUDA timing events."""
import torch

import witwin.maxwell as mw


def test_cpu_reference_port_run_times_the_step_loop_on_the_host(monkeypatch):
    def no_cuda_event(*args, **kwargs):
        raise AssertionError("CUDA timing events are not available to CPU runs")

    monkeypatch.setattr(torch.cuda, "Event", no_cuda_event)
    port = mw.LumpedPort(
        name="feed", positive=(0.0, 0.0, 0.005), negative=(0.0, 0.0, -0.005), voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)), reference_impedance=50.0)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, 0.02),) * 3), grid=mw.GridSpec.uniform(0.005),
                     boundary=mw.BoundarySpec.none(), ports=(port,), device="cpu")
    result = mw.Simulation.fdtd(
        scene, backend="cpu_reference", cuda_graph=False, frequencies=(2.5e9,),
        excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(frequency=2.5e9, fwidth=1e9)),
        run_time=mw.TimeConfig(time_steps=16), spectral_sampler=mw.SpectralSampler(window="none")).run()
    assert torch.all(torch.isfinite(result.port("feed").voltage))
    assert result.solver.last_step_loop_steps == 16
    assert result.solver.last_step_loop_elapsed_s > 0.0

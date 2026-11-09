"""Long-duration RF carriers retain physical time independently of field dtype."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.ports import _advance_port_sample_time, _evaluate_drive


def test_100000_step_physical_clock_and_carrier_match_independent_analytic_time():
    dt, frequency, steps = 5e-14, 3e9, 100_003
    runtime = SimpleNamespace(electric_time=torch.tensor(dt, dtype=torch.float64),
        magnetic_time=torch.tensor(.5*dt, dtype=torch.float64), source_kind="cw",
        source_frequency=frequency, source_phase=.37, source_amplitude=torch.tensor(1.2+.4j),
        drive_buffer=torch.zeros((), dtype=torch.float32))
    solver = SimpleNamespace(dt=dt)
    bad = torch.tensor(.5*dt, dtype=torch.float32)
    for _ in range(steps):
        _advance_port_sample_time(solver, runtime)
        bad.add_(dt)
    analytic_time = (steps+.5)*dt
    assert abs(float(runtime.magnetic_time)-analytic_time) < 1e-19
    assert abs(float(runtime.electric_time)-(steps+1)*dt) < 1e-19
    phase = 2*np.pi*frequency*analytic_time+.37
    expected = 1.2*np.cos(phase)+.4*np.sin(phase)
    assert abs(float(_evaluate_drive(runtime))-expected) < 1e-7
    # Establish that the original field-precision clock fails this physical case.
    bad_phase = 2*np.pi*frequency*float(bad)+.37
    assert abs(1.2*np.cos(bad_phase)+.4*np.sin(bad_phase)-expected) > 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native cooperative RF tables require CUDA.")
def test_100000_step_gpu_drive_table_and_port_clocks_match_independent_carrier():
    from tests.materials.surface_impedance.test_copper_foil_port_sweep import _ports
    steps, dt, frequency = 100_003, 5e-12, 1e9
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device="cuda:0", ports=_ports())
    simulation = mw.Simulation.fdtd(scene, frequency=frequency,
        excitations=mw.PortExcitation("left", amplitude=1.2+.4j, source_time=mw.CW(frequency=frequency, phase=.37)),
        run_time=mw.TimeConfig(time_steps=steps, dt=dt), spectral_sampler=mw.SpectralSampler(window="none"))
    batch = mw.Simulation.batch([simulation], execution=mw.BatchExecution(devices=("cuda:0",), graph="off"))
    with batch.prepare() as prepared:
        lifecycle = prepared._cohort_steps()
        try:
            next(lifecycle)
            runtime = prepared._slot.solvers[0]._port_runtimes[0]
            assert runtime.electric_time.dtype == runtime.magnetic_time.dtype == torch.float64
            assert runtime.drive_buffer.dtype == runtime._drive_table.dtype == torch.float32
            indices = np.array([0, 1, 9999, 50_001, steps-1])
            phase = 2*np.pi*frequency*((indices+.5)*dt)+.37
            # Public amplitude is complex64, independently account for its
            # declared quantization while checking double-precision time.
            expected = float(np.float32(1.2))*np.cos(phase)+float(np.float32(.4))*np.sin(phase)
            actual = runtime._drive_table[torch.tensor(indices, device="cuda:0")].cpu().numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-7)
            for index, expected_value in zip(indices, expected):
                runtime.magnetic_time.fill_((int(index)+.5)*dt)
                assert abs(float(_evaluate_drive(runtime))-expected_value) < 1e-7
        finally:
            lifecycle.close()

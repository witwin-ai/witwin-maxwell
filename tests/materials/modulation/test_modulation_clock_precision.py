"""Device-clock and material-phase gates independent of field stability."""

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.stepping import update_electric_fields


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _simulation(*, graph=False, modulated=True):
    material = mw.Material(
        eps_r=2.0,
        modulation=mw.ModulationSpec(frequency=0.35e9, amplitude=0.2, phase=0.4)
        if modulated else None,
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.08, 0.08),) * 3),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.periodic(),
        device="cuda",
        structures=[mw.Structure(geometry=mw.Box(position=(0.0,) * 3, size=(0.4,) * 3), material=material)],
        sources=[mw.PointDipole(position=(0.0,) * 3, polarization="Ez",
                                source_time=mw.CW(frequency=1e9))],
        monitors=[mw.FieldTimeMonitor("probe", components=("Ez",),
                                      position=(0.0,) * 3, interval=1)],
    )
    return mw.Simulation.fdtd(
        scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=64),
        full_field_dft=False, cuda_graph=graph,
    )


def _advance(solver, dt):
    solver.fdtd_module.advanceModulationTime3D(
        ModulationTime=solver._modulation_time,
        ModulationStep=solver._modulation_step,
        dt=dt,
    ).launchRaw()


@pytest.mark.parametrize("refinement", [1, 2, 4])
def test_full_horizon_clock_and_multifrequency_phase(refinement, record_property):
    solver = _simulation().prepare().solver
    dt = (1.0 / 60e9) / refinement
    final_step = math.ceil(7.3991783675e-6 / dt) - 1
    samples = np.unique(np.r_[0, 1, 2, np.linspace(0, final_step, 257).astype(np.int64)])
    frequencies = np.array([0.2e9, 0.35e9, 0.2e9, 0.35e9])
    omega = torch.tensor(2 * np.pi * frequencies, device="cuda", dtype=torch.float32).reshape(1, 1, 4)
    current = torch.ones_like(omega)
    electric = torch.zeros_like(omega)
    zero = torch.zeros_like(omega)
    depth = torch.full_like(omega, 0.25)
    max_error = 0.0
    for step in samples:
        solver._modulation_step.fill_(int(step) - 1)
        _advance(solver, dt)
        expected = torch.tensor([(int(step) - 1) * dt, int(step) * dt], dtype=torch.float64)
        assert torch.equal(solver._modulation_time.cpu(), expected)
        quadratures = []
        for mc, ms in ((depth, zero), (zero, depth)):
            electric.zero_()
            solver.fdtd_module.applyPolarizationCurrentModulated3D(
                ElectricField=electric, PolarizationCurrent=current,
                InvPermittivity=current, ModCos=mc, ModSin=ms, ModOmega=omega,
                ModulationTime=solver._modulation_time, dt=1.0,
            ).launchRaw()
            quadratures.append((-1 / electric.double() - 1).cpu().numpy().reshape(-1) / 0.25)
        measured = np.arctan2(-quadratures[1], quadratures[0])
        # Independent host reference uses the requested frequencies, including
        # the existing FP32 frequency-map quantization in the error budget.
        reference = 2 * np.pi * frequencies * (int(step) * dt)
        error = np.abs(np.angle(np.exp(1j * (measured - reference))))
        max_error = max(max_error, float(error.max()))
    assert max_error < 1e-3
    record_property("sample_count", len(samples))
    record_property("max_phase_error_rad", max_error)


def test_first_two_electric_updates_preserve_endpoint_labels():
    solver = _simulation().prepare().solver
    dt = solver.dt
    for name in ("Hx", "Hy", "Hz"):
        getattr(solver, name).zero_()
    for step in (0, 1):
        _advance(solver, dt)
        assert torch.equal(solver._modulation_time.cpu(),
                           torch.tensor([(step - 1) * dt, step * dt], dtype=torch.float64))
        expected = {}
        for component in ("Ex", "Ey", "Ez"):
            field = getattr(solver, component)
            field.fill_(1.0)
            omega = getattr(solver, "mod_omega_" + component).double().cpu().numpy()
            mc = getattr(solver, "mod_cos_" + component).cpu().numpy()
            ms = getattr(solver, "mod_sin_" + component).cpu().numpy()
            def factor(time):
                return 1 + mc * np.cos(omega * time) - ms * np.sin(omega * time)
            expected[component] = factor((step - 1) * dt) / factor(step * dt)
        update_electric_fields(solver, solver.Ex, solver.Ey, solver.Ez,
                               solver.Hx, solver.Hy, solver.Hz, time_value=step * dt)
        for component in expected:
            np.testing.assert_allclose(getattr(solver, component).cpu().numpy(),
                                       expected[component], rtol=3e-7, atol=0)


def test_graph_eager_and_disk_resume_are_bit_exact(tmp_path):
    direct = _simulation(graph=False).run().solver
    graph = _simulation(graph=True).run().solver
    assert graph._cuda_graph_active
    assert torch.count_nonzero(direct.Ez) > 0
    execution = _simulation(graph=True).prepare()
    checkpoint = execution.run_until(31)
    assert checkpoint.auxiliary["modulation_step"].item() == 30
    path = tmp_path / "modulation.pt"
    checkpoint.save(path)
    resumed = _simulation(graph=True).prepare().run(
        resume_from=mw.FDTDResumeCheckpoint.load(path)
    ).solver
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "_modulation_time", "_modulation_step"):
        assert torch.equal(getattr(direct, name), getattr(graph, name)), name
        assert torch.equal(getattr(direct, name), getattr(resumed, name)), name


@pytest.mark.parametrize("fail_capture", [False, True])
def test_capture_restores_nonzero_clock_and_counter(monkeypatch, fail_capture):
    from witwin.maxwell.fdtd.checkpoint import graph_state_tensors
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner
    from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner

    solver = _simulation().prepare().solver
    solver.Ex.fill_(1.0)
    solver._modulation_step.fill_(100000)
    _advance(solver, solver.dt)
    saved = {name: value.clone() for name, value in graph_state_tensors(solver).items()}
    if fail_capture:
        def fail_after_warmup(self, fn):
            fn()
            fn()
            raise RuntimeError("injected capture failure")
        monkeypatch.setattr(CudaGraphRunner, "capture", fail_after_warmup)
    _make_field_update_runner(solver, True)
    assert solver._cuda_graph_active is not fail_capture
    for name, value in graph_state_tensors(solver).items():
        assert torch.equal(value, saved[name]), name


@pytest.mark.parametrize("step", [0, 1, 31])
def test_old_clock_is_reconstructed_from_completed_steps(tmp_path, step):
    checkpoint = _simulation().prepare().run_until(step)
    path = tmp_path / "old-clock.pt"
    checkpoint.save(path)
    payload = torch.load(path, weights_only=True)
    payload["auxiliary"].pop("modulation_step")
    # A deliberately drifted saved clock must never be rounded into an index.
    payload["auxiliary"]["modulation_time"] = torch.tensor([7.4e-6, 7.5e-6])
    torch.save(payload, path)
    loaded = mw.FDTDResumeCheckpoint.load(path)
    assert loaded.auxiliary["modulation_step"].item() == step - 1
    expected = [((step - 2) if step else -1) * loaded.dt, (step - 1) * loaded.dt]
    assert torch.equal(loaded.auxiliary["modulation_time"], torch.tensor(expected, dtype=torch.float64))
    actual = _simulation().prepare().run(resume_from=loaded).solver
    reference = _simulation().prepare().run(resume_from=checkpoint).solver
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "_modulation_time", "_modulation_step"):
        assert torch.equal(getattr(actual, name), getattr(reference, name)), name

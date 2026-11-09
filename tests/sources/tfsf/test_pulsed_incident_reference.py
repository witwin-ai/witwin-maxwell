"""Physical pulsed reference, discrete boundary continuation, and resume identity."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import resume
from witwin.maxwell.fdtd.distributed import resume as distributed_resume
from witwin.maxwell.fdtd.runtime.representation import normalization_sample_time
from witwin.maxwell.sources import evaluate_source_time


def _execution(layers=8, *, device="cpu", graph=False, direction=(0., 0., 1.), waveform=None):
    polarization = "Ez" if abs(direction[0]) == 1 else "Ex" if direction[0] == 0 else "Ey"
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(.05),
        boundary=mw.BoundarySpec.pml(num_layers=layers), device=device,
        sources=[mw.PlaneWave(direction=direction, polarization=polarization,
            source_time=waveform or mw.GaussianPulse(frequency=1e9, fwidth=3e8, amplitude=1.7, phase=.37),
            injection=mw.TFSF(bounds=((-0.2, 0.2),) * 3))],
        monitors=[mw.PointMonitor("reference", (0., 0., -.2), fields=("Ex",))],
    )
    return mw.Simulation.fdtd(scene, frequency=1e9,
        backend="cpu_reference" if device == "cpu" else "native", cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=640), full_field_dft=False,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False)).prepare()


@pytest.mark.parametrize("direction", [
    (1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.),
    (0., 0., 1.), (0., 0., -1.), (.6, 0., .8),
])
def test_pulsed_incident_phasors_are_padding_independent_and_reference_is_waveform(direction):
    records = []
    for layers in (8, 10, 12):
        solver = _execution(layers, direction=direction).solver
        solver._normalize_source = False
        aux = solver._tfsf_state["auxiliary_grid"]
        samples, reference, declared = [], [], []
        for step in range(640):
            declared.append(evaluate_source_time(solver._source_time, normalization_sample_time(solver, step)))
            aux.advance()
            samples.append(aux.electric[aux.source_index + 2].item())
            reference.append(aux.electric[aux.source_index].item())
        np.testing.assert_allclose(reference, declared, rtol=1e-7, atol=1e-30)
        phase = np.exp(2j * np.pi * np.array((.8e9, 1e9, 1.2e9))[:, None]
                       * (np.arange(640)[None, :] + 1) * solver.dt)
        # The reference's complex spectrum preserves the declared amplitude,
        # carrier phase, and pulse origin on the physical post-step E clock.
        np.testing.assert_allclose(phase @ reference, phase @ declared, rtol=2e-7, atol=1e-7)
        records.append(phase @ samples)
    np.testing.assert_array_equal(records, np.broadcast_to(records[0], (3, 3)))


def test_entry_boundary_satisfies_discrete_yee_update_and_transpose():
    aux = _execution().solver._tfsf_state["auxiliary_grid"]
    generator = torch.Generator().manual_seed(9)
    electric = torch.randn(aux.electric.shape, generator=generator, dtype=torch.float64, requires_grad=True)
    magnetic = torch.randn(aux.magnetic.shape, generator=generator, dtype=torch.float64, requires_grad=True)
    values = aux.incident_boundary(electric, magnetic, .7, -.2)
    weights = (.3, -.5, .8)
    loss = sum(w * v for w, v in zip(weights, values))
    expected = torch.autograd.grad(loss, (electric, magnetic))
    adj_e, adj_h = torch.zeros_like(electric), torch.zeros_like(magnetic)
    adj_e[aux.source_index - 1] = weights[0]
    aux.transpose_incident_boundary(adj_e, adj_h, torch.tensor(weights[1]), torch.tensor(weights[2]))
    torch.testing.assert_close(adj_e, expected[0], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(adj_h, expected[1], rtol=2e-6, atol=1e-6)
    j = aux.source_index
    right_h = magnetic[j] - aux.magnetic_curl[j] * (electric[j + 1] - electric[j])
    entry_e = electric[j] - aux.electric_curl[j] * (right_h - values[1])
    torch.testing.assert_close(entry_e, entry_e.new_tensor(.7), rtol=0, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TFSF kernels require CUDA"))])
@torch.no_grad()
def test_incident_replay_and_core_source_rows_match_eager(device):
    from witwin.maxwell.fdtd.adjoint import core
    from witwin.maxwell.fdtd.runtime.core_tables import compile_source_tables
    from benchmark.core_path.gates import source_table_reference
    solver = _execution(device=device).solver
    aux = solver._tfsf_state["auxiliary_grid"]
    extension = SimpleNamespace(CoreStepLaunch=lambda *args: None)
    rows = compile_source_tables(solver, extension,
        torch.zeros(1, dtype=torch.int64, device=solver.device), 64).rows
    expected_rows = source_table_reference(solver, 64)
    assert rows.keys() == expected_rows.keys()
    for name in rows:
        assert torch.equal(rows[name], expected_rows[name]), name
    state = {"electric": aux.electric.clone(), "magnetic": aux.magnetic.clone()}
    for step in range(64):
        state, boundary_h = core._prepare_tfsf_auxiliary_boundary(solver, state, step * aux.dt)
        state = core._advance_tfsf_auxiliary_magnetic_state(solver, state, boundary_h=boundary_h)
        state = core._advance_tfsf_auxiliary_electric_state(solver, state, time_value=step * aux.dt)
        aux.advance()
        assert torch.equal(state["electric"], aux.electric)
        assert torch.equal(state["magnetic"], aux.magnetic)


@pytest.mark.parametrize("upstream_cells", [2, 6])
def test_native_auxiliary_transpose_includes_physical_boundary(monkeypatch, upstream_cells):
    from tests.fdtd.cuda import torch_reference
    from witwin.maxwell.fdtd.adjoint import core, native
    from witwin.maxwell.fdtd.cuda import backend
    def reverse_electric(**kwargs):
        torch_reference._reverse_tfsf_auxiliary_electric(**kwargs)
        # The frozen oracle's advanced-index add_ writes to a copy. Supply the
        # native kernel's actual E scatter for this orchestration-only test.
        indices = torch.arange(1, kwargs["AdjElectricPost"].numel() - 1)
        indices = indices[indices != kwargs["sourceIndex"]]
        kwargs["AdjElectricPrev"].index_add_(0, indices,
            kwargs["ElectricDecay"][indices] * kwargs["AdjElectricPost"][indices])
    monkeypatch.setattr(backend, "_reverse_tfsf_auxiliary_electric", reverse_electric)
    monkeypatch.setattr(backend, "_reverse_tfsf_auxiliary_magnetic", torch_reference._reverse_tfsf_auxiliary_magnetic)
    solver = _execution().solver
    solver._tfsf_state.update(electric_terms=[], magnetic_terms=[])
    aux = solver._tfsf_state["auxiliary_grid"]
    if upstream_cells > 2:
        from witwin.maxwell.fdtd.excitation.spatial import AuxiliaryGrid1D
        aux = AuxiliaryGrid1D(s_min=-upstream_cells * .05, s_max=2., ds=.05,
            dt=.5 * .05 / solver.c, wave_speed=solver.c, impedance=aux.impedance,
            source_time=aux.source_time, device="cpu", dtype=torch.float64, incident_reference=0.)
        solver._tfsf_state["auxiliary_grid"] = aux
    generator = torch.Generator().manual_seed(31)
    electric = torch.randn(aux.electric.shape, generator=generator, dtype=torch.float64, requires_grad=True)
    magnetic = torch.randn(aux.magnetic.shape, generator=generator, dtype=torch.float64, requires_grad=True)
    state = {"electric": electric, "magnetic": magnetic}
    advanced, boundary_h = core._prepare_tfsf_auxiliary_boundary(solver, state, 0.)
    advanced = core._advance_tfsf_auxiliary_magnetic_state(solver, advanced, boundary_h=boundary_h)
    advanced = core._advance_tfsf_auxiliary_electric_state(solver, advanced, time_value=0.)
    grad_e = torch.randn(electric.shape, generator=generator, dtype=electric.dtype)
    grad_h = torch.randn(magnetic.shape, generator=generator, dtype=magnetic.dtype)
    expected = torch.autograd.grad((advanced["electric"] * grad_e).sum()
                                   + (advanced["magnetic"] * grad_h).sum(), (electric, magnetic))
    actual = native._reverse_tfsf_auxiliary_state_native(solver,
        {"tfsf_aux_electric": electric, "tfsf_aux_magnetic": magnetic},
        {"tfsf_aux_electric": grad_e, "tfsf_aux_magnetic": grad_h}, magnetic_output_adjoint={})
    torch.testing.assert_close(actual["tfsf_aux_electric"], expected[0], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(actual["tfsf_aux_magnetic"], expected[1], rtol=2e-6, atol=1e-6)


def test_anisotropic_oblique_ghost_extent_and_padding_independence():
    records = []
    for layers in (8, 10, 12):
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, .5), (-.01, .01), (-.01, .01))),
            grid=mw.GridSpec.custom(np.linspace(-.5, .5, 11), np.linspace(-.01, .01, 21),
                                    np.linspace(-.01, .01, 21)),
            boundary=mw.BoundarySpec.pml(num_layers=layers), device="cpu",
            sources=[mw.PlaneWave(direction=(.1, 0., np.sqrt(.99)), polarization="Ey",
                source_time=mw.GaussianPulse(frequency=1e8, fwidth=2e7),
                injection=mw.TFSF(bounds=((-0.2, .2), (-.005, .005), (-.005, .005))))])
        solver = mw.Simulation.fdtd(scene, frequency=1e8, backend="cpu_reference",
            run_time=mw.TimeConfig(time_steps=64), full_field_dft=False).prepare().solver
        aux = solver._tfsf_state["auxiliary_grid"]
        assert aux.source_index > 2
        first_h = min(float(term["sample_positions"].min()) for term in
                      solver._tfsf_state["electric_terms"])
        assert aux.s_min + .5 * aux.ds <= first_h
        record = []
        for step in range(64):
            aux.advance()
            record.append(aux.electric[aux.source_index + 2].item())
        records.append(record)
    np.testing.assert_array_equal(records, np.broadcast_to(records[0], (3, 64)))


def test_full_scene_reference_with_frozen_torch_face_kernels(monkeypatch):
    # The production CPU profile has no TFSF kernels. Exercise the full Yee
    # recurrence with the repository's independent frozen face implementation.
    from tests.fdtd.cuda import torch_reference
    monkeypatch.setattr(torch_reference, "_require_cuda_tensors", lambda *values: None)
    execution = _execution()
    execution.solver.fdtd_module = torch_reference.NativeFDTDModule()
    result = execution.run()
    value = result.monitors["reference"]["Ex"]
    solver = result.solver
    # Post-step E and its imposed entry waveform share physical time (n+1)dt.
    times = (np.arange(640) + 1) * solver.dt
    declared = np.array([evaluate_source_time(solver._source_time, t) for t in times])
    expected = (declared * np.exp(2j * np.pi * 1e9 * times)).sum() * solver.dt * 1e9
    np.testing.assert_allclose(value.cpu().numpy(), expected, rtol=2e-6, atol=1e-7)


def test_pulsed_reference_disk_resume_keeps_auxiliary_clock(tmp_path, monkeypatch):
    from tests.fdtd.cuda import torch_reference
    monkeypatch.setattr(torch_reference, "_require_cuda_tensors", lambda *values: None)

    def execution():
        prepared = _execution()
        prepared.solver.fdtd_module = torch_reference.NativeFDTDModule()
        return prepared

    expected = execution().run()
    checkpoint = execution().run_until(100)
    checkpoint.save(tmp_path / "pulse.pt")
    actual = execution().run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path / "pulse.pt"))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(actual.solver, name), getattr(expected.solver, name)), name
    expected_aux = expected.solver._tfsf_state["auxiliary_grid"]
    actual_aux = actual.solver._tfsf_state["auxiliary_grid"]
    assert actual_aux.time_step == expected_aux.time_step == 640
    assert torch.equal(actual_aux.electric, expected_aux.electric)
    assert torch.equal(actual_aux.magnetic, expected_aux.magnetic)


@pytest.mark.parametrize("joint", [False, True])
@torch.no_grad()
def test_pulsed_reference_adjoint_forward_and_segment_replay_match(monkeypatch, joint):
    from tests.fdtd.cuda import torch_reference
    from witwin.maxwell.fdtd.adjoint import _FDTDGradientBridge, _replay_segment_states
    from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
    monkeypatch.setattr(torch_reference, "_require_cuda_tensors", lambda *values: None)
    execution = _execution()
    solver = execution.solver
    solver.fdtd_module = torch_reference.NativeFDTDModule()
    solver._joint_material_adjoint = joint
    # Exercise the production forward loop with frozen CPU kernels, without
    # introducing a CPU differentiable solver entrypoint.
    bridge = object.__new__(_FDTDGradientBridge)
    bridge.simulation = execution.simulation
    bridge.simulation.config = replace(bridge.simulation.config, adjoint_checkpoint_stride=12)
    _, checkpoints = bridge._run_forward_with_checkpoints(solver, time_steps=48,
        dft_frequency=1e9, dft_window="none", full_field_dft=False, normalize_source=False)

    ordinary = _execution()
    ordinary.solver.fdtd_module = torch_reference.NativeFDTDModule()
    expected = ordinary.run_until(48).physics
    actual = capture_checkpoint_state(solver, 48)
    for name, tensor in expected.tensors.items():
        assert torch.equal(tensor, actual.tensors[name]), name

    states = _replay_segment_states(solver, checkpoints[0], 0, 48)
    for checkpoint in checkpoints:
        segment = _replay_segment_states(solver, checkpoint, checkpoint.step, 48)
        for name in ("tfsf_aux_electric", "tfsf_aux_magnetic"):
            assert torch.equal(checkpoint.tensors[name], states[checkpoint.step][name]), name
            for offset, state in enumerate(segment):
                assert torch.equal(state[name], states[checkpoint.step + offset][name]), name
    aux = solver._tfsf_state["auxiliary_grid"]
    assert torch.equal(states[-1]["tfsf_aux_electric"], aux.electric)
    assert torch.equal(states[-1]["tfsf_aux_magnetic"], aux.magnetic)


def test_pulsed_incident_is_causal_at_the_entry_face():
    waveform = mw.CustomSourceTime((0., 1e-9, 2e-9, 3e-9), (0., 0., 1., 0.), characteristic_frequency=1e9)
    aux = _execution(waveform=waveform).solver._tfsf_state["auxiliary_grid"]
    for step in range(640):
        aux.advance()
        # The entry seed after this update is W((step+1)dt).
        if (step + 1) * aux.dt <= 1e-9:
            assert torch.count_nonzero(aux.electric[aux.source_index:]) == 0
            assert torch.count_nonzero(aux.magnetic[aux.source_index:]) == 0


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("convention", [None, "padded_domain_minimum"])
def test_old_pulsed_tfsf_checkpoint_is_refused_before_fingerprint_comparison(distributed, convention):
    solver = _execution().solver
    if distributed:
        solver = SimpleNamespace(shard_layouts=(), logical_scene=solver.scene,
            shards=(SimpleNamespace(rank=0, solver=solver),), _nccl=False, dt=solver.dt,
            _time_plan=solver._time_plan)
    assert resume._pulsed_tfsf_signature(solver) == {"pulsed_tfsf_reference": "physical_entry_plane"}
    # A source-free checkpoint supplies a valid disk/state container. The old
    # convention must be diagnosed before its deliberately different fingerprint.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, .5),) * 3), grid=mw.GridSpec.uniform(.05),
                     boundary=mw.BoundarySpec.pml(num_layers=8), device="cpu")
    checkpoint = mw.Simulation.fdtd(scene, frequency=1e9, backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=640), full_field_dft=False).prepare().run_until(2)
    signature = {} if convention is None else {"pulsed_tfsf_reference": convention}
    checkpoint = replace(checkpoint, dt=solver.dt, signature=signature)
    with pytest.raises(NotImplementedError, match="checkpoint predates the pulsed TFSF physical entry-plane reference"):
        resume.preflight_resume_checkpoint(solver, checkpoint, total_steps=640)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native TFSF kernels require CUDA")
@pytest.mark.parametrize("graph", [False, True])
def test_native_pulsed_reference_pml_causality_and_resume(tmp_path, graph):
    phasors = []
    for layers in (8, 10, 12):
        expected = _execution(layers, device="cuda", graph=graph).run()
        phasors.append(expected.monitors["reference"]["Ex"])
        solver = expected.solver
        times = (np.arange(640) + 1) * solver.dt
        declared = np.array([evaluate_source_time(solver._source_time, t) for t in times])
        # alpha=beta=1: entry E and reference both sample physical post-step time.
        reference = (declared * np.exp(2j * np.pi * 1e9 * times)).sum() * solver.dt * 1e9
        np.testing.assert_allclose(phasors[-1].cpu().numpy(), reference, rtol=2e-5, atol=2e-6)
        execution = _execution(layers, device="cuda", graph=graph)
        checkpoint = execution.run_until(100)
        checkpoint.save(tmp_path / "pulse.pt")
        actual = _execution(layers, device="cuda", graph=graph).run(
            resume_from=mw.FDTDResumeCheckpoint.load(tmp_path / "pulse.pt"))
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert torch.equal(getattr(actual.solver, name), getattr(expected.solver, name))
    torch.testing.assert_close(phasors[1], phasors[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(phasors[2], phasors[0], rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native TFSF kernels require CUDA")
def test_native_pulsed_reference_has_no_field_before_entry_arrival():
    waveform = mw.CustomSourceTime((0., 1e-9, 2e-9, 3e-9), (0., 0., 1., 0.), characteristic_frequency=1e9)
    execution = _execution(device="cuda", waveform=waveform)
    stop = int(1e-9 / execution.solver.dt) - 1
    execution.run_until(stop)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.count_nonzero(getattr(execution.solver, name)) == 0

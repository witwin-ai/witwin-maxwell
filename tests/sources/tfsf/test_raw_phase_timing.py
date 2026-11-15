"""Physical incident clocks, reference rematerialization, and timing resume guard."""
from dataclasses import replace
from types import SimpleNamespace
import cmath
import math

import pytest
import torch

import witwin.maxwell as mw
from tests.fdtd.test_observer_origin_providers import CASES, execution
from witwin.maxwell.fdtd import resume
from witwin.maxwell.fdtd.distributed import resume as distributed_resume
from witwin.maxwell.fdtd.runtime.representation import normalization_sample_time
from witwin.maxwell.fdtd.runtime.spectral import source_normalization_spectrum
from witwin.maxwell.sources import evaluate_source_time, evaluate_source_time_normalization


AUX_CASES = ("axis_pulse", "axis_cw", "special_pulse", "special_cw", "oblique_pulse", "slab_pulse", "slab_cw")
CW_CASES = ("oblique_cw", "beam_cw", "astigmatic_cw", "grating_cw")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _solver(case, device="cuda", steps=64):
    return execution(case, device=device, steps=steps)[0].solver


@pytest.mark.parametrize("case", AUX_CASES)
def test_seeded_entry_e_is_declared_post_step_waveform(case):
    device = "cuda"
    solver = _solver(case, device)
    aux = solver._tfsf_state["auxiliary_grid"]
    # CW retains its transported spatial seed; pulse uses the physical entry W.
    assert torch.equal(aux.electric[aux.source_index], aux.electric.new_tensor(
        evaluate_source_time(aux.source_time, 0.)))
    generator = torch.Generator(device=device).manual_seed(20261103)
    aux.electric.copy_(torch.randn(aux.electric.shape, device=device, generator=generator))
    aux.magnetic.copy_(torch.randn(aux.magnetic.shape, device=device, generator=generator))
    for n in range(64):
        aux.advance()
        expected = aux.electric.new_tensor(evaluate_source_time(aux.source_time, (n + 1) * aux.dt))
        assert torch.equal(aux.electric[aux.source_index], expected), (case, n)


@pytest.mark.parametrize("case", AUX_CASES)
def test_auxiliary_reference_uses_physical_post_step_samples(case):
    solver = _solver(case)
    assert solver._normalization_time_offset == solver._normalization_label_offset == 1.
    frequency = solver.source_frequency
    steps = torch.arange(64, dtype=torch.float64, device=solver.device)
    times = (steps + 1) * solver.dt
    for normalized in (False, True):
        solver._normalize_source = normalized
        assert torch.equal(normalization_sample_time(solver, steps), times)
    signal = evaluate_source_time_normalization(solver._normalization_source_time, times)
    scale = solver.dt * frequency if not case.endswith("cw") else 2. / 64
    expected = (signal * torch.exp(2j * torch.pi * frequency * times)).sum() * scale
    actual = source_normalization_spectrum(solver, (frequency,), steps_run=64, window_type="none")[0]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("case", [case for case in AUX_CASES if case.endswith("pulse")])
def test_normalized_entry_spectrum_matches_a_within_four_fp32_epsilon(case):
    solver = _solver(case, steps=512)
    solver._normalize_source = True
    aux = solver._tfsf_state["auxiliary_grid"]
    raw = []
    for _ in range(512):
        aux.advance()
        raw.append(aux.electric[aux.source_index].clone())
    n = torch.arange(512, dtype=torch.float64, device=solver.device)
    phase = torch.exp(2j * torch.pi * solver.source_frequency * (n + 1) * aux.dt)
    scale = solver.dt * solver.source_frequency
    new_reference = source_normalization_spectrum(solver, (solver.source_frequency,),
        steps_run=512, window_type="none")[0]
    new_transfer = (torch.stack(raw).double() * phase).sum() * scale / new_reference
    # (A)'s imposed entry samples and reference both used W(n dt), beta=1.
    old_raw = aux.electric.new_tensor([
        evaluate_source_time(aux.source_time, step * aux.dt) for step in range(512)])
    old_unit = evaluate_source_time_normalization(solver._normalization_source_time, n * aux.dt)
    old_reference = (old_unit * phase).sum() * scale
    old_transfer = (old_raw.double() * phase).sum() * scale / old_reference
    torch.testing.assert_close(new_transfer, old_transfer,
        rtol=4 * torch.finfo(torch.float32).eps, atol=0.)


@pytest.mark.parametrize("case", AUX_CASES)
@pytest.mark.parametrize("start_step", [0, 19])
@torch.no_grad()
def test_resident_tables_and_functional_replay_share_eager_seed_clock(case, start_step):
    from witwin.maxwell.fdtd.adjoint import core
    from witwin.maxwell.fdtd.runtime.core_tables import compile_source_tables
    device = "cuda"
    solver = _solver(case, device)
    aux = solver._tfsf_state["auxiliary_grid"]
    for _ in range(start_step):
        aux.advance()
    rows = compile_source_tables(solver, SimpleNamespace(CoreStepLaunch=lambda *args: None),
        torch.zeros(1, dtype=torch.int64, device=device), 32).rows
    expected = aux.electric.new_tensor([
        evaluate_source_time(aux.source_time, (start_step + n + 1) * aux.dt) for n in range(32)])
    assert torch.equal(rows["auxiliary/signals"], expected)
    if aux.incident_reference is not None:
        expected_boundary = aux.electric.new_tensor([[
            evaluate_source_time(aux.source_time, (start_step + n + k + 1) * aux.dt)
            for k in range(aux.source_index)] for n in range(32)])
        assert torch.equal(rows["auxiliary/boundary_signals"], expected_boundary)
    state = dict(electric=aux.electric.clone(), magnetic=aux.magnetic.clone())
    for n in range(start_step, start_step + 32):
        state, boundary = core._prepare_tfsf_auxiliary_boundary(solver, state, n * aux.dt)
        state = core._advance_tfsf_auxiliary_magnetic_state(solver, state, boundary_h=boundary)
        state = core._advance_tfsf_auxiliary_electric_state(solver, state, time_value=n * aux.dt)
        aux.advance()
        assert torch.equal(state["electric"], aux.electric)
        assert torch.equal(state["magnetic"], aux.magnetic)


@pytest.mark.parametrize("case", (*AUX_CASES, *CW_CASES))
def test_new_timing_disk_resume_is_bitwise(case, tmp_path):
    def prepared():
        return execution(case, device="cuda", steps=64)[0]
    expected = prepared().run()
    checkpoint = prepared().run_until(19)
    assert checkpoint.signature["tfsf_incident_timing"] == "physical_yee_v2"
    checkpoint.save(tmp_path / "physical.pt")
    actual = prepared().run(resume_from=mw.fdtd.FDTDResumeCheckpoint.load(tmp_path / "physical.pt"))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(expected.solver, name), getattr(actual.solver, name)), name
        if expected.solver.complex_fields_enabled:
            assert torch.equal(getattr(expected.solver, name + "_imag"), getattr(actual.solver, name + "_imag")), name
    for field, value in expected.raw_monitor("point")["components"].items():
        assert torch.equal(value, actual.raw_monitor("point")["components"][field]), field


@pytest.mark.parametrize("case", (*CW_CASES, "axis_cw", "special_cw", "slab_cw"))
def test_cw_face_carrier_has_declared_phase_and_steady_normalization_matches_a(case):
    solver = _solver(case)
    theta = solver.source_omega * solver.dt
    auxiliary = solver._tfsf_state.get("auxiliary_grid")
    assert solver._normalization_label_offset == (1. if auxiliary else 0.)
    assert solver._normalization_time_offset == (1. if auxiliary else 0.)
    terms = (*solver._tfsf_state["electric_terms"], *solver._tfsf_state["magnetic_terms"])
    assert terms
    for term in terms:
        # Spatial phase remains in the patches; the temporal phase is W's own.
        if auxiliary is None:
            assert term["source_time"]["phase"] == solver._source_time["phase"]
        phase = float((auxiliary.source_time if auxiliary else term["source_time"])["phase"])
        physical = cmath.exp(-1j * phase)
        old_raw = cmath.exp(-1j * (phase - theta))
        old_reference = cmath.exp(1j * theta)  # (A): alpha=0, beta=1.
        torch.testing.assert_close(torch.tensor(physical, dtype=torch.complex128),
            torch.tensor(old_raw / old_reference, dtype=torch.complex128),
            rtol=4 * torch.finfo(torch.float64).eps, atol=0.)


@pytest.mark.parametrize("case,provider,beta", CASES)
@pytest.mark.parametrize("distributed", [False, True])
def test_old_affected_checkpoint_is_refused_before_fingerprint(case, provider, beta, distributed, tmp_path):
    if case.startswith("grating"):
        if case == "grating_pulse":
            pytest.skip("public fixed-k pulse refused")
    solver = _solver(case)
    affected = case in AUX_CASES or case in CW_CASES
    expected = {"tfsf_incident_timing": "physical_yee_v2"} if affected else {}
    assert resume._tfsf_timing_signature(solver) == expected
    if distributed:
        solver = SimpleNamespace(shard_layouts=(), logical_scene=solver.scene.scene,
            shards=(SimpleNamespace(rank=0, solver=solver),), _nccl=False, dt=solver.dt,
            _time_plan=solver._time_plan)
    assert resume._tfsf_timing_signature(solver) == expected
    if not affected:
        resume._preflight_tfsf_timing(solver, {})
        return
    # A valid disk container, with the old timing signature and a deliberately
    # wrong fingerprint: the source-clock diagnostic must precede all restore.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, .3),) * 3),
        grid=mw.GridSpec.uniform(.025), boundary=mw.BoundarySpec.pml(num_layers=4))
    checkpoint = mw.Simulation.fdtd(scene, frequencies=2e9,
        config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=64), full_field_dft=False)).prepare().run_until(1)
    signature = distributed_resume._signature(solver) if distributed else resume._solver_signature(solver)
    assert signature.pop("tfsf_incident_timing") == "physical_yee_v2"
    signature["logical_fingerprint" if distributed else "configuration_fingerprint"] = "old fingerprint"
    checkpoint = replace(checkpoint, dt=solver.dt, signature=signature)
    checkpoint.save(tmp_path / "old.pt")
    before = (solver.shards[0].solver if distributed else solver).Ex.clone()
    with pytest.raises(NotImplementedError, match="checkpoint predates the physical TFSF incident timing; re-run from the start"):
        resume.preflight_resume_checkpoint(solver, mw.fdtd.FDTDResumeCheckpoint.load(tmp_path / "old.pt"), total_steps=64)
    assert torch.equal(before, (solver.shards[0].solver if distributed else solver).Ex)

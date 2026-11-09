"""Auxiliary TFSF launch plans and exact native fusion qualification."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import core_path, core_tables


FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
DIRECTIONS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
              (0., -1., 0.), (0., 0., 1.), (0., 0., -1.))
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="TFSF fusion requires CUDA.")


@pytest.fixture(autouse=True)
def forbid_host_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: pytest.fail("Host test initialized CUDA"))


def _plan_inputs(reference=True, source_index=2, count=431, interpolated=False):
    aux = SimpleNamespace(electric=torch.zeros(count), magnetic=torch.zeros(count - 1),
        electric_decay=torch.linspace(1., .7, count), electric_curl=torch.linspace(.1, .07, count),
        magnetic_decay=torch.linspace(1., .7, count - 1), magnetic_curl=torch.linspace(.2, .14, count - 1),
        source_index=source_index, incident_reference=0. if reference else None,
        _incident_boundary_h=torch.zeros(source_index), s_min=-.25, ds=.125)
    from witwin.maxwell.fdtd.excitation.gather import build_source_gather
    batch = dict(coeff_data=torch.tensor([.5, -.3]),
                 source_gather=build_source_gather(torch.tensor([0, 0])))
    if interpolated:
        batch["sample_positions"] = torch.tensor([0., .125])
    else:
        batch["sample_indices_per_coeff"] = torch.tensor([2, 3], dtype=torch.int32)
    state = dict(provider="plane_wave_axis_aligned", auxiliary_grid=aux,
        magnetic_terms=[dict(sample_kind="electric")], electric_terms=[dict(sample_kind="magnetic")],
        magnetic_batch=batch, electric_batch=batch)
    solver = SimpleNamespace(tfsf_enabled=True, _tfsf_state=state,
                             **{name: torch.zeros(2, 2, 2) for name in FIELDS})
    sources = SimpleNamespace(auxiliary=object(), rows={"auxiliary/signals": torch.arange(4.)})
    if reference:
        sources.rows["auxiliary/boundary_signals"] = torch.arange(8.).reshape(4, 2)
    return solver, aux, batch, sources


@pytest.mark.host_validation
@pytest.mark.parametrize("interpolated", [False, True])
@pytest.mark.parametrize("minimum,clear", [(2, True), (1, False)])
def test_fold_gate_checks_both_correction_and_line_stencils(interpolated, minimum, clear):
    _, aux, batch, _ = _plan_inputs(interpolated=interpolated)
    if interpolated:
        batch["sample_positions"] = torch.tensor([aux.s_min + minimum * aux.ds])
    else:
        batch["sample_indices_per_coeff"] = torch.tensor([minimum], dtype=torch.int32)
    plan = core_tables.auxiliary_boundary_fold_plan(aux, batch)
    assert plan.sample_range == (minimum, minimum + int(interpolated))
    assert plan.boundary_limit == 2
    assert plan.correction_clear is clear
    assert not plan.line_clear
    assert not plan.folded


@pytest.mark.host_validation
def test_interpolated_gate_uses_reciprocal_multiply_and_both_clamped_fetches():
    _, aux, batch, _ = _plan_inputs(interpolated=True)
    aux.ds, aux.s_min = .0037, -.017
    positions = np.array([-.4, -.017, .031, 999.], dtype=np.float32)
    batch["sample_positions"] = torch.from_numpy(positions)
    plan = core_tables.auxiliary_boundary_fold_plan(aux, batch)
    assert plan.sample_range == (0, aux.electric.numel() - 1)
    # At an integer coordinate the upper endpoint is still fetched.
    batch["sample_positions"] = torch.tensor([aux.s_min])
    assert core_tables.auxiliary_boundary_fold_plan(aux, batch).sample_range == (0, 1)


@pytest.mark.host_validation
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("interpolated", [False, True])
@pytest.mark.parametrize("count", [29, 431])
def test_launch_plan_uses_full_prepared_loss_arrays(reference, interpolated, count):
    solver, aux, batch, sources = _plan_inputs(reference=reference, count=count, interpolated=interpolated)
    launches = []

    def launch(op, tensors, integers, scalars):
        value = SimpleNamespace(op=op, tensors=tensors, integers=integers, scalars=scalars)
        launches.append(value)
        return value

    step = torch.zeros(1, dtype=torch.int64)
    fusion = core_tables.compile_auxiliary_fusion(solver, SimpleNamespace(CoreStepLaunch=launch), step, sources)
    assert [item.op for item in launches] == ([4, 5, 6] if reference else [5, 6])
    assert solver._core_tfsf_aux_fused
    assert not solver._core_tfsf_aux_boundary_folded
    assert (fusion.boundary is not None) is reference
    for item, names, kind in ((fusion.magnetic, ("Hx", "Hy", "Hz"), "magnetic"),
                              (fusion.electric, ("Ex", "Ey", "Ez"), "electric")):
        assert item.tensors[0] is step
        assert item.tensors[1] is aux.electric
        assert item.tensors[2] is aux.magnetic
        assert item.tensors[3] is aux.electric_decay
        assert item.tensors[4] is aux.electric_curl
        assert item.tensors[5] is aux.magnetic_decay
        assert item.tensors[6] is aux.magnetic_curl
        assert item.tensors[7] is sources.rows["auxiliary/signals"]
        assert all(a is getattr(solver, b) for a, b in zip(item.tensors[10:13], names))
        assert item.tensors[13] is batch["coeff_data"]
        assert all(a is b for a, b in zip(item.tensors[15:], batch["source_gather"]))
        assert item.integers == [2, int(not interpolated), int(reference)]
        assert item.scalars == [aux.s_min + (0 if kind == "magnetic" else .5 * aux.ds), 1. / aux.ds]


@pytest.mark.host_validation
@pytest.mark.parametrize("reason", ["hook", "wide_stencil", "oblique", "missing_batch"])
def test_unqualified_routes_keep_historical_calls(reason):
    solver, aux, _, sources = _plan_inputs()
    if reason == "hook":
        solver._core_tfsf_aux_fusion_disabled = True
    elif reason == "wide_stencil":
        aux.source_index = 3
    elif reason == "oblique":
        solver._tfsf_state["provider"] = "plane_wave_aux"
    else:
        solver._tfsf_state["magnetic_batch"] = None
    extension = SimpleNamespace(CoreStepLaunch=lambda *a: pytest.fail("Unqualified fusion prepared a launch"))
    assert core_tables.compile_auxiliary_fusion(solver, extension, None, sources) is None
    assert not solver._core_tfsf_aux_fused
    assert not solver._core_tfsf_aux_boundary_folded


@pytest.mark.host_validation
@pytest.mark.parametrize("family,wrong", [("magnetic", "magnetic"), ("electric", "electric")])
def test_prepare_asserts_opposite_incident_field(family, wrong):
    solver, _, _, sources = _plan_inputs()
    solver._tfsf_state[family + "_terms"][0]["sample_kind"] = wrong
    with pytest.raises(AssertionError, match="opposite auxiliary field"):
        core_tables.compile_auxiliary_fusion(solver, None, None, sources)


@pytest.mark.host_validation
def test_core_abi_marker():
    from witwin.maxwell.fdtd.cuda.core.build import CORE_ABI_VERSION
    assert CORE_ABI_VERSION == "core_abi_v11"
    root = Path(core_path.__file__).parents[1] / "cuda" / "core"
    assert 'name="witwin_maxwell_fdtd_core_abi_v11"' in (root / "build.py").read_text()
    assert 'module.attr("abi") = "core_abi_v11"' in (root / "binding.cpp").read_text()


@pytest.mark.host_validation
def test_ordinary_closure_launches_boundary_h_and_e_once(monkeypatch):
    from witwin.maxwell.fdtd.runtime import stepping
    from witwin.maxwell.fdtd import cut_surface, observers
    events = []
    solver, _, _, sources = _plan_inputs()
    sources.groups = {name: [SimpleNamespace(launch=lambda n=name: events.append(n))]
                      for name in ("magnetic", "electric", "ordinary")}
    solver.scene = SimpleNamespace()
    solver.electric_dispersive_enabled = False
    solver.has_pec_faces = False
    for name in ("_advance_magnetic_dispersive_state", "_apply_magnetic_dispersive_corrections",
                 "_advance_dispersive_state", "_apply_dispersive_corrections"):
        setattr(solver, name, lambda n=name: events.append(n))
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.step, runtime.sources = torch.zeros(1, dtype=torch.int64), sources
    runtime.magnetic = lambda: events.append("yee_h")
    runtime.electric = lambda: events.append("yee_e")
    runtime.extension = SimpleNamespace(CoreStepLaunch=lambda op, *a:
        SimpleNamespace(launch=lambda: events.append(op)))
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda s: None)
    monkeypatch.setattr(observers, "uses_native_point_observers", lambda s: False)
    monkeypatch.setattr(stepping, "enforce_pec_boundaries", lambda s: events.append("clamp"))
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: events.append("mur"))
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: events.append("cut"))
    from witwin.maxwell.fdtd.excitation import tfsf_apply
    for name in ("apply_tfsf_h_correction", "apply_tfsf_e_correction", "advance_tfsf_auxiliary_magnetic"):
        monkeypatch.setattr(tfsf_apply, name, lambda *a: pytest.fail("Replaced per-step call survived"))
    step, _ = runtime.prepare_step(solver, use_gpu_dft=False, time_steps=4)
    step()
    assert events == ["_advance_magnetic_dispersive_state", "yee_h", 4, 5, "magnetic",
        "_apply_magnetic_dispersive_corrections", "_advance_dispersive_state", "yee_e", 6,
        "electric", "ordinary", "clamp", "_apply_dispersive_corrections", "mur", "cut", 3]


def _simulation(direction, waveform, *, core, device="cuda"):
    axis = next(i for i, value in enumerate(direction) if value)
    polarization = ("Ez", "Ex", "Ey")[axis]
    source_time = (mw.CW(frequency=1e9, amplitude=1.7, phase=.37) if waveform == "cw" else
                   mw.GaussianPulse(frequency=1e9, fwidth=3e8, amplitude=1.7, phase=.37, delay=.5e-9))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, .5),) * 3), grid=mw.GridSpec.uniform(.05),
        boundary=mw.BoundarySpec.pml(num_layers=4), device=device,
        sources=[mw.PlaneWave(direction=direction, polarization=polarization, source_time=source_time,
                             injection=mw.TFSF(bounds=((-0.2, .2),) * 3))],
        monitors=[mw.PointMonitor("point", (0., 0., 0.), fields=FIELDS),
                  mw.PlaneMonitor("plane", axis="z", position=0., fields=FIELDS)])
    return mw.Simulation.fdtd(scene, frequency=1e9, fdtd_core_path=core, full_field_dft=True,
        backend="cpu_reference" if device == "cpu" else "native",
        cuda_graph=True, run_time=mw.TimeConfig(time_steps=65), shutoff=0.,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))


@pytest.mark.host_validation
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("waveform", ["pulse", "cw"])
@pytest.mark.parametrize("reference", [None, False, True])
def test_axis_scenes_prepare_expected_reference_provider(direction, waveform, reference):
    solver = _prepare(direction, waveform, reference, core=False, device="cpu").solver
    state = solver._tfsf_state
    assert state["provider"] in {"plane_wave_ref_x_ez", "plane_wave_axis_aligned"}
    aux = state["auxiliary_grid"]
    expected_reference = waveform == "pulse" if reference is None else reference
    assert (aux.incident_reference is not None) == expected_reference
    assert aux.absorber_cells == 320
    assert aux.electric.numel() == aux.sampled_nodes + aux.absorber_cells
    plan = core_tables.auxiliary_boundary_fold_plan(aux, state["magnetic_batch"])
    assert not plan.folded
    launches = []
    extension = SimpleNamespace(CoreStepLaunch=lambda op, *a: launches.append(op) or SimpleNamespace())
    step = torch.zeros(1, dtype=torch.int64)
    tables = core_tables.compile_source_tables(solver, extension, step, 65)
    launches.clear()
    fusion = core_tables.compile_auxiliary_fusion(solver, extension, step, tables)
    assert fusion is not None
    assert launches == ([4, 5, 6] if expected_reference else [5, 6])


def _prepare(direction, waveform, reference, *, core, device="cuda"):
    execution = _simulation(direction, waveform, core=core, device=device).prepare()
    if reference is None:
        return execution
    solver = execution.solver
    aux = solver._tfsf_state["auxiliary_grid"]
    # Change only the internal stencil after normal public preparation: pulses
    # require physical reference metadata even when testing a boundary-free line.
    aux.incident_reference = aux.s_min + 2 * aux.ds if reference else None
    aux.source_index = 2 if reference else 0
    aux._incident_boundary_h = torch.zeros(aux.source_index, device=aux.device)
    aux._incident_values = torch.zeros_like(aux._incident_boundary_h)
    aux.electric.zero_()
    aux.electric[aux.source_index] = aux.source_value(-1)
    if core:
        runtime = solver._core_runtime
        runtime.sources = core_tables.compile_source_tables(solver, runtime.extension, runtime.step, 65)
    return execution


def _run(direction, waveform, reference, chunk, arm, minimum=None):
    from benchmark.core_path.gates import tensors
    execution = _prepare(direction, waveform, reference, core=arm != "existing")
    solver = execution.solver
    aux = solver._tfsf_state["auxiliary_grid"]
    if minimum is not None:
        batch = solver._tfsf_state["magnetic_batch"]
        indices = batch["sample_indices_per_coeff"]
        indices.clamp_(min=minimum)
        indices[0] = minimum
    if arm != "existing":
        solver._core_runtime.chunk_size = chunk
        solver._core_tfsf_aux_fusion_disabled = arm == "unfused"
    # Nonzero tails exercise every cell of the full prepared absorber arrays.
    generator = torch.Generator().manual_seed(542)
    for value in [*(getattr(solver, name) for name in FIELDS), aux.electric, aux.magnetic]:
        value.copy_(torch.randn(value.shape, generator=generator) * 2.**-20)
    result = execution.run()
    if arm != "existing":
        assert solver._core_step_graph_active
        assert solver._core_step_graph_k == chunk
        assert solver._core_tfsf_aux_fused is (arm == "fused")
        if arm == "fused":
            plan = solver._core_tfsf_aux_fold_plan
            assert not solver._core_tfsf_aux_boundary_folded
            if minimum is not None:
                assert plan.sample_range[0] == minimum
                assert plan.correction_clear is (minimum >= plan.boundary_limit)
                assert not plan.line_clear
    payload = tensors({name: getattr(solver, name) for name in FIELDS}, "field")
    payload.update(tensors(result.fields, "phasor"))
    payload.update(tensors(result.monitors, "monitor"))
    payload.update(tensors(solver._dft_batched_fields, "dft"))
    payload.update(tensors(dict(e=aux.electric, h=aux.magnetic, boundary_h=aux._incident_boundary_h,
                                time_step=aux.time_step), "aux"))
    assert any(key.startswith("monitor/") for key in payload)
    assert any(key.startswith("dft/") for key in payload)
    assert all(torch.isfinite(value).all() for value in payload.values())
    return payload


def _assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("waveform", ["pulse", "cw"])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("chunk", [1, 16, 32])
def test_g1_fields_aux_spectra_and_monitors_three_repeats(direction, waveform, reference, chunk):
    existing = _run(direction, waveform, reference, chunk, "existing")
    unfused = _run(direction, waveform, reference, chunk, "unfused")
    _assert_equal(unfused, existing)
    for _ in range(3):
        _assert_equal(_run(direction, waveform, reference, chunk, "fused"), existing)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("minimum", [2, 1])
@pytest.mark.parametrize("chunk", [1, 16, 32])
def test_g1_correction_samples_at_boundary_limit_and_just_below(minimum, chunk):
    args = ((0., 0., 1.), "pulse", True, chunk)
    existing = _run(*args, "existing", minimum=minimum)
    _assert_equal(_run(*args, "unfused", minimum=minimum), existing)
    for _ in range(3):
        _assert_equal(_run(*args, "fused", minimum=minimum), existing)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("interpolated", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_single_step_each_fused_side_matches_native_gather(interpolated, reference):
    from witwin.maxwell.fdtd.excitation.tfsf_apply import (
        _apply_batched_aux_terms, _apply_batched_reference_terms,
    )
    execution = _prepare((1., 0., 0.), "pulse", reference, core=True)
    solver = execution.solver
    runtime = solver._core_runtime
    aux = solver._tfsf_state["auxiliary_grid"]
    runtime.step.fill_(5)
    if interpolated:
        for key, origin in (("magnetic_batch", aux.s_min), ("electric_batch", aux.s_min + .5 * aux.ds)):
            batch = solver._tfsf_state[key]
            indices = batch.pop("sample_indices_per_coeff").float()
            # Fractional coordinates exercise the original interpolation body,
            # even though public axis-aligned providers normally gather integers.
            batch["sample_positions"] = (origin + (indices + .27) * aux.ds).contiguous()
    fusion = core_tables.compile_auxiliary_fusion(solver, runtime.extension, runtime.step, runtime.sources)
    state = [*(getattr(solver, name) for name in FIELDS), aux.electric, aux.magnetic, aux._incident_boundary_h]
    generator = torch.Generator().manual_seed(511)
    for value in state:
        value.copy_(torch.randn(value.shape, generator=generator) * 2.**-20)
    initial = [value.clone() for value in state]
    apply = _apply_batched_aux_terms if interpolated else _apply_batched_reference_terms
    if reference:
        aux.prepare_incident_boundary(runtime.sources.rows["auxiliary/boundary_signals"][5])
    apply(solver, solver._tfsf_state["magnetic_batch"], sample_kind="electric", field_names=FIELDS[3:])
    aux.advance_magnetic()
    expected_h = [value.clone() for value in state]
    apply(solver, solver._tfsf_state["electric_batch"], sample_kind="magnetic", field_names=FIELDS[:3])
    runtime.sources.auxiliary.launch()
    expected_e = [value.clone() for value in state]
    for _ in range(3):
        for value, saved in zip(state, initial):
            value.copy_(saved)
        if fusion.boundary is not None:
            fusion.boundary.launch()
        fusion.magnetic.launch()
        for index, (value, expected) in enumerate(zip(state, expected_h)):
            assert torch.equal(value, expected), ("H", index)
        fusion.electric.launch()
        for index, (value, expected) in enumerate(zip(state, expected_e)):
            assert torch.equal(value, expected), ("E", index)

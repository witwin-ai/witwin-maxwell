"""CPU contracts for ordered separate-physics capture and restoration."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.scenes import (
    CORE_SCENES, CORE_REFUSED_BENCHMARK_SCENES, W1_SCENES, W2_SCENES, DISPERSIVE_SCENES, build_core_case,
)
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.runtime import core_path, core_tables, stepping


pytestmark = pytest.mark.host_validation


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("W2 host checks must not initialize CUDA.")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


@pytest.mark.parametrize("name", tuple(dict.fromkeys(W1_SCENES + W2_SCENES + DISPERSIVE_SCENES)))
def test_scene_construction_and_admission(name):
    case = build_core_case(name)
    refused = name in CORE_REFUSED_BENCHMARK_SCENES
    assert (name in CORE_SCENES) is not refused
    assert str(case.scene.device) == "cpu"
    grid = prepare_scene(case.scene, lazy_meshgrid=True)
    assert grid.Nx > 0 and grid.Ny > 0 and grid.Nz > 0
    assert core_path.core_path_blockers(SimpleNamespace(scene=case.scene)) == ["non-CUDA execution"]
    # Project the runtime-only flags of these fixtures without preparing a CUDA solver.
    solver = SimpleNamespace(scene=case.scene,
                             active_absorber_type=case.absorber or "cpml",
                             dispersive_enabled=name in DISPERSIVE_SCENES,
                             electric_dispersive_enabled=name in DISPERSIVE_SCENES and name != "magnetic_lorentz_slab",
                             magnetic_dispersive_enabled=name == "magnetic_lorentz_slab",
                             _copper_foils=any(isinstance(s.material, mw.CopperFoil) for s in case.scene.structures),
                             _thin_sheets=name == "w1_thin_sheet_cavity",
                             _cut_surface=name == "w1_cut_surface",
                             surface_impedance_enabled=name == "w1_sibc")
    expected_blockers = (["closed-transverse CPML (passivity guard)"] if refused else []) + ["non-CUDA execution"]
    assert core_path.core_path_blockers(solver) == expected_blockers
    expected = {"w1_circuit": ["circuits"], "w1_cut_surface": ["cut-surface constraints"],
                "w1_sibc": ["surface impedance"]}.get(name, [])
    assert core_path.core_path_step_loop_blockers(solver) == expected


@pytest.mark.parametrize("attribute,value,reason", [
    ("_circuit_runtimes", (object(),), "circuits"),
    ("_port_runtimes", (SimpleNamespace(circuit_port=object()),), "circuit-bound ports"),
    ("tfsf_enabled", True, "TFSF with separate physics"),
    ("_magnetic_source_terms", (object(),), "magnetic sources with separate physics"),
    ("modulation_enabled", True, "modulation with separate physics"),
    ("surface_impedance_enabled", True, "surface impedance"),
    ("_cut_surface", object(), "cut-surface constraints"),
    ("_wire_runtime", SimpleNamespace(monitor_state=(object(),)), "host-driven wire monitors"),
])
def test_step_loop_exclusions(attribute, value, reason):
    solver = SimpleNamespace(scene=SimpleNamespace(ports=(object(),)), **{attribute: value})
    assert core_path.core_path_step_loop_blockers(solver) == [reason]
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    assert runtime.run_chunks(solver, time_steps=64, start_step=0, end_step=64,
                              use_cuda_graph=True, use_gpu_dft=True, cooperative=False,
                              cooperative_driver=False, cooperative_observer=None, resume_from=None,
                              shutoff=0., shutoff_check_interval=100, shutoff_min_step=0,
                              enable_plot=False, pbar=None) is None
    assert solver._core_step_graph_active is False
    assert solver._core_step_graph_chunks == []


@pytest.mark.parametrize("physics", ["network", "lossless_wire", "lossy_wire", "electric_ade", "magnetic_ade"])
def test_device_only_physics_admitted(physics):
    solver = SimpleNamespace(scene=SimpleNamespace())
    if physics == "network":
        solver._network_runtimes = (object(),)
    elif physics.endswith("_ade"):
        solver.scene = build_core_case("dispersive_lumped_port").scene
        solver.dispersive_enabled = True
        setattr(solver, physics.removesuffix("_ade") + "_dispersive_enabled", True)
        assert core_path.core_path_blockers(solver) == ["non-CUDA execution"]
        core_path._validate_separate_step(solver)
    else:
        solver._wire_runtime = SimpleNamespace(lossy_model=object() if physics == "lossy_wire" else None)
    assert core_path.core_path_has_separate_physics(solver)
    assert core_path.core_path_step_loop_blockers(solver) == []


@pytest.mark.parametrize("attribute", ["tfsf_enabled", "_magnetic_source_terms", "modulation_enabled"])
def test_source_only_scenes_keep_existing_closure(attribute):
    solver = SimpleNamespace(scene=SimpleNamespace(), **{attribute: True})
    assert core_path.core_path_step_loop_blockers(solver) == []


@pytest.mark.parametrize("attribute,value,message", [
    ("tfsf_enabled", True, "TFSF"),
    ("_magnetic_source_terms", (object(),), "magnetic surface"),
    ("modulation_enabled", True, "modulation"),
    ("scene", SimpleNamespace(ports=(object(),), boundary=mw.BoundarySpec.mur()), "Mur"),
    ("scene", SimpleNamespace(ports=(object(),), boundary=mw.BoundarySpec.faces(default="mode_abc")), "mode-ABC"),
])
def test_prepare_step_asserts_eligibility(attribute, value, message):
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    solver = SimpleNamespace(scene=SimpleNamespace(ports=(object(),), boundary=mw.BoundarySpec.none()))
    setattr(solver, attribute, value)
    with pytest.raises(AssertionError, match=message):
        runtime.prepare_step(solver, use_gpu_dft=True, time_steps=64)


@pytest.mark.parametrize("port_kind", ["ordinary", "none", "observed", "fed_sheet", "element_only"])
def test_separate_step_order(monkeypatch, port_kind):
    events = []
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.step = torch.zeros(1, dtype=torch.int64)
    runtime.magnetic = lambda: events.append("h")
    runtime.electric = lambda: events.append("e")
    runtime.sources = SimpleNamespace(groups={name: [SimpleNamespace(launch=lambda n=name: events.append(n))]
                                              for name in ("ordinary", "electric")})
    runtime.extension = SimpleNamespace(CoreStepLaunch=lambda *args:
                                        SimpleNamespace(launch=lambda: events.append("commit")))
    spectral = SimpleNamespace(launch=lambda: events.append("field_dft"))
    monkeypatch.setattr(core_tables, "compile_spectral_tables", lambda *args: spectral)
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda *args: None)
    from witwin.maxwell.fdtd import observers
    monkeypatch.setattr(observers, "uses_native_point_observers", lambda s: True)
    monkeypatch.setattr(core_tables, "NativePointObserverTables", lambda *args:
                        SimpleNamespace(launch=lambda: events.append("point_dft")))
    monkeypatch.setattr(stepping, "_field_update_block", lambda s, t: events.append(("fields", t)))
    monkeypatch.setattr(stepping, "enforce_pec_boundaries", lambda s: events.append("pec"))
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: events.append("mur"))
    from witwin.maxwell.fdtd import ports, cut_surface
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: events.append("cut"))
    monkeypatch.setattr(ports, "prepare_port_drive_tables", lambda s, n: events.append(("drive_tables", n)))
    monkeypatch.setattr(stepping, "apply_port_runtimes", lambda s, drive_step: events.append(("ports", drive_step)))
    monkeypatch.setattr(stepping, "apply_network_runtimes", lambda s, native_lu: events.append(("networks", native_lu)))
    monkeypatch.setattr(stepping, "accumulate_port_observers", lambda s, step: events.append(("port_dft", step)))
    observed = port_kind in ("observed", "fed_sheet")
    sheet = SimpleNamespace(feeds=port_kind == "fed_sheet", electric=lambda s: events.append("fed_sheet"))
    solver = SimpleNamespace(scene=SimpleNamespace(boundary=mw.BoundarySpec.none()), tfsf_enabled=False,
                             dispersive_enabled=True, electric_dispersive_enabled=True,
                             _advance_magnetic_dispersive_state=lambda: events.append("magnetic_ade_advance"),
                             _apply_magnetic_dispersive_corrections=lambda: events.append("magnetic_ade_correct"),
                             _advance_dispersive_state=lambda: events.append("electric_ade_advance"),
                             _apply_dispersive_corrections=lambda: events.append("electric_ade_correct"),
                             _copper_foils=() if port_kind == "ordinary" else (object(),),
                             _thin_sheets=() if port_kind == "ordinary" else (sheet,),
                             _port_runtimes=(SimpleNamespace(sample_index=7),) if observed else (),
                             _lumped_element_runtimes=(object(),) if port_kind == "element_only" else (),
                             _port_observer_step=torch.zeros(1, dtype=torch.int64))
    step, _ = runtime.prepare_step(solver, use_gpu_dft=True, time_steps=64)
    if observed:
        assert events == [("drive_tables", 64)]
        assert solver._port_observer_step.item() == 7
        assert solver._port_observer_graph_active
    events.clear()
    step()
    expected = [("fields", 0.0), "electric", "ordinary", "pec"]
    counter = solver._port_observer_step if observed else None
    if port_kind not in ("none", "ordinary"):
        expected.append(("ports", counter))
        if port_kind == "fed_sheet":
            expected.append("fed_sheet")
        expected.append(("networks", True))
    expected += ["electric_ade_correct", "pec", "mur", "cut", "field_dft"]
    if port_kind not in ("none", "ordinary"):
        expected.append(("port_dft", counter))
    if port_kind == "ordinary":
        expected = ["magnetic_ade_advance", "h", "magnetic_ade_correct", "electric_ade_advance", "e",
                    "electric", "ordinary", "pec", "electric_ade_correct", "pec", "mur", "cut", "field_dft"]
    assert events == expected + ["point_dft", "commit"]
    assert solver._port_observer_step.item() == (8 if observed else 0)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("dispersive", [False, True])
def test_warmup_and_capture_restore_physics_and_port_counts(monkeypatch, failure, dispersive):
    from witwin.maxwell.fdtd import checkpoint
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.sources, runtime.chunk_size, runtime.graphs = SimpleNamespace(steps=64), 16, {}
    runtime.step = torch.zeros(1, dtype=torch.int64)
    accumulator = SimpleNamespace(_sample_count=4, _voltage_sum=torch.tensor([3.]))
    port = SimpleNamespace(accumulator=accumulator, drive_accumulator=None, lumped=None,
                           wire_provider=None, substep_buffers=None, circuit_port=None,
                           electric_time=torch.tensor(1.), magnetic_time=torch.tensor(.5))
    sheet = SimpleNamespace(copper=None, feeds=(), h_old=torch.tensor([9.]))
    solver = SimpleNamespace(scene=SimpleNamespace(ports=(object(),)), device="cpu",
                             dispersive_enabled=dispersive,
                             _port_runtimes=(port,), _thin_sheets=(sheet,),
                             _port_observer_step=torch.zeros(1, dtype=torch.int64),
                             _port_observer_graph_active=False,
                             observers_enabled=False, dft_enabled=False, tfsf_enabled=False,
                             Ex=torch.tensor([2.]), ade_current=torch.tensor([5.]), ade_polarization=torch.tensor([6.]))
    material = {"current": solver.ade_current, "polarization": solver.ade_polarization} if dispersive else {}
    point_real, point_imag = torch.tensor([7.]), torch.tensor([-3.])
    completed = []
    runtime.point_observers = SimpleNamespace(state_tensors=(point_real, point_imag),
                                             finalize=lambda s, n: completed.append(n))
    state = [solver.Ex, *material.values(), runtime.step,
             *core_path._separate_graph_state_tensors(solver), point_real, point_imag]
    original = [tensor.clone() for tensor in state]
    monkeypatch.setattr(checkpoint, "graph_state_tensors", lambda s: {"Ex": s.Ex, **material})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    def step():
        for tensor in state:
            tensor.add_(1)
        accumulator._sample_count += 1

    def prepare(*args, **kwargs):
        solver._port_observer_graph_active = True
        return step, None

    def capture(s, closure, count):
        for tensor, value in zip(state, original):
            assert torch.equal(tensor, value)
        assert accumulator._sample_count == 4
        for _ in range(count):
            closure()
        if failure:
            raise RuntimeError("Failed after mutating driven state.")
        return lambda: [closure() for _ in range(count)]

    monkeypatch.setattr(runtime, "prepare_step", prepare)
    monkeypatch.setattr(runtime, "capture_chunk", capture)
    def run():
        return runtime.run_chunks(solver, time_steps=64, start_step=0, end_step=35,
                                  use_cuda_graph=True, use_gpu_dft=False, cooperative=False,
                                  cooperative_driver=False, cooperative_observer=None, resume_from=None,
                                  shutoff=0., shutoff_check_interval=100, shutoff_min_step=0,
                                  enable_plot=False, pbar=None)
    if failure and dispersive:
        with pytest.raises(RuntimeError, match="Failed after mutating driven state"):
            run()
        result = None
    else:
        result = run()
    for tensor, value in zip(state, original):
        assert torch.equal(tensor, value + (0 if failure else 35))
    assert completed == ([] if failure else [35])
    assert result == (None if failure else 35)
    assert solver._core_step_graph_active is (not failure)
    if failure:
        assert accumulator._sample_count == 4
        assert solver._port_observer_graph_active is False


@pytest.mark.parametrize("field_name", ["Ez", "Hx"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_point_rows_and_completed_metadata_match_eager(monkeypatch, field_name, dtype):
    from copy import copy
    from witwin.maxwell.fdtd import observers
    from witwin.maxwell.fdtd.cuda import backend
    from witwin.maxwell.fdtd.runtime import spectral

    solver = SimpleNamespace(device="cpu", dt=1.e-11, observers_enabled=True,
                             observer_window_type="hanning",
                             _source_time={"kind": "cw", "frequency": 1.e9},
                             _accumulate_source_spectrum=True, _normalize_source=False, _source_time_offset=True,
                             _time_plan={"response": "cw_steady_state", "spectral_start_step": 3})
    solver._compute_window_weight = lambda *args, **kwargs: spectral.compute_window_weight(solver, *args, **kwargs)
    solver._observer_spectral_entries = [spectral.build_observer_spectral_entry(
        solver, frequency, end_step=10, window_type="hanning") for frequency in (1.e9, 2.e9)]
    group = dict(field_name=field_name, point_i=torch.tensor([0], dtype=torch.int32),
                 point_j=torch.tensor([0], dtype=torch.int32), point_k=torch.tensor([0], dtype=torch.int32),
                 real=torch.zeros((2, 1), dtype=dtype), imag=torch.zeros((2, 1), dtype=dtype),
                 global_freq_indices=(0, 1), freq_local_lookup={0: 0, 1: 1})
    setattr(solver, field_name, torch.ones((1, 1, 1)))
    solver._point_observer_groups = {field_name.lower(): group}
    solver._plane_observer_groups = {}
    solver._observer_point_groups_by_frequency = [[(group, i)] for i in range(2)]
    solver._observer_plane_groups_by_frequency = [[], []]
    solver._sync_observer_primary_state = lambda: None
    calls = []
    solver.fdtd_module = SimpleNamespace(accumulatePointObservers3D=lambda **kw:
                                        SimpleNamespace(launchRaw=lambda: calls.append(kw)))
    monkeypatch.setattr(backend, "get_compiled_extension", lambda: object())
    initial = [entry.copy() for entry in solver._observer_spectral_entries]
    tables = core_tables.NativePointObserverTables(solver, torch.zeros(1, dtype=torch.int64), 12)
    assert solver._observer_spectral_entries == initial
    completed = []
    for n in range(12):
        calls.clear()
        observers.accumulate_observers(solver, n)
        for local, route in enumerate(tables.routes):
            row = route[-2][n]
            assert bool(row[0]) is (3 <= n < 10)
            if row[0]:
                expected = torch.tensor([calls[local]["weightedCos"], calls[local]["weightedSin"]], dtype=dtype)
                assert torch.equal(row[1:].view(torch.uint8), expected.view(torch.uint8))
        completed.append([entry.copy() for entry in solver._observer_spectral_entries])
    staged = copy(solver)
    staged._observer_spectral_entries = [entry.copy() for entry in initial]
    for steps in (0, 1, 5, 12):
        tables.finalize(staged, steps)
        assert staged._observer_spectral_entries == (initial if steps == 0 else completed[steps - 1])

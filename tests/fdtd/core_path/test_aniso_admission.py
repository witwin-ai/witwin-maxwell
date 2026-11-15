"""Full (off-diagonal) anisotropy on the core path: admission, closure order and clamp sites."""
from types import SimpleNamespace

import pytest
import torch

from benchmark.core_path.scenes import ANISO_SCENES, GATE_SCENES, build_core_case
from witwin.maxwell.fdtd import cut_surface
from witwin.maxwell.fdtd.runtime import core_path, core_tables, stepping
from witwin.maxwell.scene import prepare_scene


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **kw: pytest.fail("Host check initialized CUDA."))


@pytest.mark.host_validation
@pytest.mark.parametrize("name", ANISO_SCENES)
def test_aniso_gate_scenes_are_admitted(name):
    assert name in GATE_SCENES
    case = build_core_case(name)
    grid = prepare_scene(case.scene, device="cpu", lazy_meshgrid=True)
    assert grid.Nx > 0 and grid.Ny > 0 and grid.Nz > 0
    # Automatic absorber selection resolves the closed-transverse periodic crystal to the graded absorber.
    absorber = case.absorber or ("absorber" if name == "aniso_debye_periodic" else "cpml")
    solver = SimpleNamespace(scene=grid, active_absorber_type=absorber, full_aniso_enabled=True)
    assert core_path.core_path_step_loop_blockers(solver) == []


@pytest.mark.host_validation
@pytest.mark.parametrize("tfsf", [False, True])
def test_full_aniso_closure_order(monkeypatch, tfsf):
    events = []
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.step = torch.zeros(1, dtype=torch.int64)
    runtime.magnetic = lambda: events.append("yee_h")
    runtime.electric = lambda: events.append("yee_e")
    runtime.extension = SimpleNamespace(CoreStepLaunch=lambda *args:
        SimpleNamespace(launch=lambda: events.append("commit")))
    groups = ("magnetic", "electric", "ordinary", "tfsf_magnetic", "tfsf_electric")
    runtime.sources = SimpleNamespace(auxiliary=None, groups={
        name: [SimpleNamespace(launch=lambda n=name: events.append(n))] for name in groups})
    solver = SimpleNamespace(scene=SimpleNamespace(scene=SimpleNamespace()), tfsf_enabled=tfsf, observers_enabled=False,
        full_aniso_enabled=True,
        _advance_magnetic_dispersive_state=lambda: events.append("ade_h"),
        _apply_magnetic_dispersive_corrections=lambda: events.append("correct_h"),
        _advance_dispersive_state=lambda: events.append("ade_e"),
        _apply_dispersive_corrections=lambda: events.append("correct_e"))
    monkeypatch.setattr(core_tables, "compile_spectral_tables", lambda *a:
        (SimpleNamespace(launch=lambda: events.append("dft")),))
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda *a: None)
    monkeypatch.setattr(stepping, "enforce_pec_boundaries", lambda s: events.append("pec"))
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: events.append("mur"))
    monkeypatch.setattr(stepping, "capture_aniso_conduction_currents", lambda s: events.append("aniso_capture"))
    monkeypatch.setattr(stepping, "apply_full_aniso_corrections", lambda s: events.append("aniso_curl"))
    monkeypatch.setattr(stepping, "apply_full_aniso_conduction", lambda s: events.append("aniso_conduction"))
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: events.append("cut"))
    step, _ = runtime.prepare_step(solver, use_gpu_dft=True, time_steps=32)
    step()
    # The existing field block: capture sigma . E^n, base E, off-diagonal curl, conduction; TFSF after it.
    assert events == (["ade_h", "yee_h"] + (["tfsf_magnetic"] if tfsf else [])
        + ["magnetic", "correct_h", "ade_e", "aniso_capture", "yee_e", "aniso_curl", "aniso_conduction"]
        + (["tfsf_electric"] if tfsf else [])
        + ["electric", "ordinary", "pec", "correct_e"] + ([] if tfsf else ["pec"])
        + ["mur", "cut", "dft", "commit"])


@pytest.mark.host_validation
@pytest.mark.parametrize("ordinary,expected", [(True, (True, False)), (False, (False, True))])
def test_full_aniso_keeps_historical_pec_clamps(ordinary, expected):
    sources = SimpleNamespace(groups={"ordinary": [object()]} if ordinary else {}, auxiliary=None)

    def plan(full_aniso):
        solver = SimpleNamespace(scene=SimpleNamespace(scene=SimpleNamespace()), has_pec_faces=True, tfsf_enabled=False,
                                 electric_dispersive_enabled=False, full_aniso_enabled=full_aniso,
                                 _source_terms=[], _electric_source_terms=[])
        return core_path.pec_clamp_plan(solver, sources)

    # Without writers after the fused E stores both clamps are redundant; the
    # off-diagonal corrections write E after those stores, so the historical sites stay.
    assert plan(False) == (False, False)
    assert plan(True) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core execution requires CUDA.")
def test_conductive_crystal_runs_whole_step_core_deterministically():
    from benchmark.core_path.gates import make_simulation, result_tensors, tensors

    case = build_core_case("aniso_conductive_dipole")
    runs = []
    for _ in range(2):
        result = make_simulation(case, steps=40, full_field_dft=True).run()
        solver = result.solver
        assert solver._core_runtime is not None
        assert solver.full_aniso_enabled and solver._aniso_cond_current is not None
        assert solver._core_step_graph_active and solver._core_step_graph_chunks == [32, 8]
        payload = tensors({name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
        payload.update(result_tensors(result))
        runs.append(payload)
    assert runs[0] and runs[0].keys() == runs[1].keys()
    assert any(value.abs().max() > 0 for name, value in runs[0].items() if name.startswith("fields/E"))
    for name, value in runs[0].items():
        assert torch.equal(value, runs[1][name]), name

"""Gyromagnetic ferrites on the core path: admission, closure order and the CUDA magnetization kernels."""
from types import SimpleNamespace

import pytest
import torch

from benchmark.core_path.scenes import FERRITE_SCENES, GATE_SCENES, build_core_case
from tests.oracles import gyromagnetic_recurrence as oracle
from tests.support.numerical_equivalence import assert_numerically_equivalent
from witwin.maxwell.fdtd import cut_surface
from witwin.maxwell.fdtd.runtime import core_path, core_tables, gyromagnetic, stepping
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler import compile_gyromagnetic_materials

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="The gyromagnetic kernels require CUDA.")
STATE = ("m_u", "m_v", "dm_u", "dm_v")


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **kw: pytest.fail("Host check initialized CUDA."))


@pytest.mark.host_validation
@pytest.mark.parametrize("name", FERRITE_SCENES)
def test_ferrite_gate_scenes_are_admitted(name):
    assert name in GATE_SCENES
    case = build_core_case(name)
    grid = prepare_scene(case.scene, device="cpu", lazy_meshgrid=True)
    assert compile_gyromagnetic_materials(grid, dt=1e-12, device="cpu").enabled
    solver = SimpleNamespace(scene=SimpleNamespace(scene=case.scene), active_absorber_type="cpml", gyromagnetic_enabled=True)
    assert core_path.core_path_step_loop_blockers(solver) == []


@pytest.mark.host_validation
@pytest.mark.parametrize("tfsf", [False, True])
def test_gyromagnetic_closure_order(monkeypatch, tfsf):
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
        gyromagnetic_enabled=True,
        _advance_magnetic_dispersive_state=lambda: events.append("ade_h"),
        _snapshot_gyromagnetic_drive=lambda: events.append("gyro_snapshot"),
        _apply_magnetic_dispersive_corrections=lambda: events.append("correct_h"),
        _step_gyromagnetic_coupled=lambda: events.append("gyro_step"),
        _advance_dispersive_state=lambda: events.append("ade_e"),
        _apply_dispersive_corrections=lambda: events.append("correct_e"))
    monkeypatch.setattr(core_tables, "compile_spectral_tables", lambda *a:
        (SimpleNamespace(launch=lambda: events.append("dft")),))
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda *a: None)
    monkeypatch.setattr(stepping, "enforce_pec_boundaries", lambda s: events.append("pec"))
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: events.append("mur"))
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: events.append("cut"))
    step, _ = runtime.prepare_step(solver, use_gpu_dft=True, time_steps=32)
    step()
    # The existing field block: snapshot, magnetic update, TFSF H and magnetic sources,
    # magnetic dispersive corrections, then the coupled magnetization step.
    assert events == (["ade_h", "gyro_snapshot", "yee_h"] + (["tfsf_magnetic"] if tfsf else [])
        + ["magnetic", "correct_h", "gyro_step", "ade_e", "yee_e"]
        + (["tfsf_electric"] if tfsf else [])
        + ["electric", "ordinary", "pec", "correct_e"] + ([] if tfsf else ["pec"])
        + ["mur", "cut", "dft", "commit"])


def _prepared(name, steps=40):
    from benchmark.core_path.gates import make_simulation

    return make_simulation(build_core_case(name), steps=steps, full_field_dft=True)


def _twin(solver):
    """A Torch-oracle copy of the solver's H fields and gyromagnetic state."""
    state = {key: value.clone() if torch.is_tensor(value) else value
             for key, value in solver._gyromagnetic_state.items()}
    return SimpleNamespace(gyromagnetic_enabled=True, _gyromagnetic_state=state,
                           **{name: getattr(solver, name).clone() for name in ("Hx", "Hy", "Hz")})


def _fill(solvers, generator):
    for name in ("Hx", "Hy", "Hz"):
        values = torch.randn(getattr(solvers[0], name).shape, generator=generator, device="cuda")
        for solver in solvers:
            getattr(solver, name).copy_(values)


@cuda
@pytest.mark.parametrize("name", ["ferrite_axis_dipole", "ferrite_oblique_tfsf", "ferrite_mixed_bias"])
def test_kernel_step_matches_torch_oracle(name):
    """Rule 2 single-step state bound of the kernels against the former Torch hooks."""
    solver = _prepared(name).prepare().solver
    state = solver._gyromagnetic_state
    generator = torch.Generator(device="cuda").manual_seed(7)
    for _ in range(8):
        # The oracle restarts from the kernel's state each step: a pure single-step check.
        twin = _twin(solver)
        _fill((solver, twin), generator)
        gyromagnetic.snapshot_gyromagnetic_drive(solver)
        oracle.snapshot_gyromagnetic_drive(twin)
        _fill((solver, twin), generator)
        gyromagnetic.step_gyromagnetic_coupled(solver)
        oracle.step_gyromagnetic_coupled(twin)
        reference = {key: getattr(twin, key) for key in ("Hx", "Hy", "Hz")}
        reference.update({key: twin._gyromagnetic_state[key] for key in STATE})
        candidate = {key: getattr(solver, key) for key in ("Hx", "Hy", "Hz")}
        candidate.update({key: state[key] for key in STATE})
        assert_numerically_equivalent(reference, candidate, level="step")
    assert state["m_u"].abs().max() > 0 and state["m_v"].abs().max() > 0


@cuda
def test_general_kernel_reduces_to_fast_kernel_bitwise():
    solver = _prepared("ferrite_axis_dipole").prepare().solver
    general = SimpleNamespace(scene=solver.scene, dt=solver.dt, device=solver.device,
                              Nx=solver.Nx, Ny=solver.Ny, Nz=solver.Nz,
                              **{name: getattr(solver, name).clone() for name in ("Hx", "Hy", "Hz")})
    gyromagnetic.build_gyromagnetic(general, solver.scene, force_general=True)
    assert solver._gyromagnetic_state["general"] is False and general._gyromagnetic_state["general"] is True
    generator = torch.Generator(device="cuda").manual_seed(3)
    for _ in range(6):
        _fill((solver, general), generator)
        for target in (solver, general):
            gyromagnetic.snapshot_gyromagnetic_drive(target)
        _fill((solver, general), generator)
        for target in (solver, general):
            gyromagnetic.step_gyromagnetic_coupled(target)
        for name in ("Hx", "Hy", "Hz"):
            assert torch.equal(getattr(solver, name), getattr(general, name)), name
        for name in STATE:
            assert torch.equal(solver._gyromagnetic_state[name], general._gyromagnetic_state[name]), name


@cuda
@pytest.mark.parametrize("name", ["ferrite_mixed_bias", "ferrite_lumped_port"])
def test_ferrite_runs_whole_step_core_deterministically(name):
    from benchmark.core_path.gates import result_tensors, tensors

    runs = []
    for _ in range(3):
        result = _prepared(name).run()
        solver = result.solver
        assert solver._core_runtime is not None
        assert solver.gyromagnetic_enabled and solver._core_step_graph_active
        payload = tensors({key: getattr(solver, key) for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
        payload.update(tensors({key: solver._gyromagnetic_state[key] for key in STATE}, "gyromagnetic"))
        payload.update(result_tensors(result))
        runs.append(payload)
    assert runs[0]["gyromagnetic/m_u"].abs().max() > 0
    for run in runs[1:]:
        assert run.keys() == runs[0].keys()
        for key, value in runs[0].items():
            assert torch.equal(value, run[key]), key

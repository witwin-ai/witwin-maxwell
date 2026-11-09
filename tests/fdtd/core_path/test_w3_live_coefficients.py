"""Live coefficient storage, breakdown state and fused modulation resume contracts."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmark.core_path.scenes import CORE_SCENES, CORE_REFUSED_BENCHMARK_SCENES, W3_SCENES, build_core_case
from witwin.maxwell.breakdown_stress import BreakdownStressAccumulator
from witwin.maxwell.fdtd.boundary.runtime import closed_transverse_pml
from witwin.maxwell.fdtd.runtime import core_path, stepping
from witwin.maxwell.fdtd.runtime.breakdown import advance_breakdown_state


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation") is None:
        return
    def fail(*args, **kwargs):
        pytest.fail("W3 host checks must not initialize CUDA.")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


@pytest.mark.host_validation
@pytest.mark.parametrize("name", W3_SCENES)
def test_live_launch_owns_the_existing_coefficient_storage(name, monkeypatch):
    from witwin.maxwell.fdtd.cuda.core import build

    case = build_core_case(name)
    nonlinear = name.startswith(("kerr", "tpa")) or name.endswith(("_kerr", "_tpa"))
    general = name == "tpa_slab" or name.endswith("_tpa")
    # Also exercise the existing Kerr + modulation composition.
    modulation = name.startswith("modulated") or "modulation" in name or name == "kerr_slab_strong"
    solver = SimpleNamespace(scene=case.scene, device="cpu", Nx=4, Ny=5, Nz=6,
                             active_absorber_type=case.absorber or "cpml",
                             uses_cpml=False, nonlinear_enabled=nonlinear,
                             nonlinear_general_enabled=general, modulation_enabled=modulation,
                             breakdown_enabled=name.startswith("breakdown_"))
    for family in "eh":
        for axis in "xyz":
            setattr(solver, family.upper() + axis, torch.zeros(4, 5, 6))
            setattr(solver, f"inv_d{axis}_{family}", torch.ones(6))
            for kind in ("decay", "curl"):
                setattr(solver, f"c{family}{axis}_{kind}", torch.ones(4, 5, 6))
                setattr(solver, f"c{family}{axis}_{kind}_dynamic", torch.zeros(4, 5, 6))
    for axis in "xyz":
        for side in ("low", "high"):
            setattr(solver, f"boundary_{axis}_{side}_code", 0)
        for kind in ("cos", "sin", "omega"):
            setattr(solver, f"mod_{kind}_E{axis}", torch.ones(4, 5, 6))
    solver._modulation_time = torch.zeros(2, dtype=torch.float64)
    calls = []
    monkeypatch.setattr(build, "build_extension", lambda: SimpleNamespace(
        CoreLaunch=lambda *args: calls.append(args)))
    def compress(solver, family):
        assert family == "h"
        return torch.empty(0, dtype=torch.uint16), torch.empty(0), [1.] * 6, True
    monkeypatch.setattr(core_path, "_compress_coefficients", compress)
    core_path.CorePathRuntime(solver)
    refused = name in CORE_REFUSED_BENCHMARK_SCENES
    assert (name in CORE_SCENES) is not refused
    expected_blockers = (["closed-transverse CPML (passivity guard)"] if refused else []) + ["non-CUDA execution"]
    assert core_path.core_path_blockers(solver) == expected_blockers
    assert core_path.core_path_step_loop_blockers(solver) == []
    h, e = calls
    assert h[-2:] == (False, False)
    assert e[-2:] == (True, modulation)
    expected = [t for pair in zip(stepping._electric_decay_tensors(solver),
                                  stepping._electric_curl_tensors(solver)) for t in pair]
    assert all(actual is original for actual, original in zip(e[0][11:17], expected))
    expected[1].fill_(7.)
    assert torch.equal(e[0][12], expected[1])
    if modulation:
        assert e[0][-1] is solver._modulation_time
    # W6 admits electric ADE; composing it with live coefficients adds no blocker.
    solver.dispersive_enabled = True
    assert core_path.core_path_blockers(solver) == expected_blockers


@pytest.mark.host_validation
def test_breakdown_device_step_preserves_state_storage_and_scalar_arithmetic():
    indices = torch.arange(2)
    runtime = dict(state=torch.zeros(2, dtype=torch.int8), timer=torch.zeros(2),
                   trigger_step=torch.full((2,), -1, dtype=torch.int64),
                   field_before=torch.zeros(2), deposited_at_trigger=torch.zeros(2),
                   dissipated_energy=torch.zeros(2), total_dissipated=torch.zeros((), dtype=torch.float64),
                   dt=.25, critical=torch.ones(2), minimum_duration=torch.full((2,), .5),
                   ramp_time=torch.full((2,), .5), post_sigma=torch.full((2,), .1),
                   base_sigma=torch.zeros(2), eps_abs=torch.ones(2), node_volume=torch.ones(2),
                   colocation={f"{component}_{side}": indices for component in ("ex", "ey", "ez")
                               for side in ("lo", "hi")}, edge_sets={})
    for component in ("Ex", "Ey", "Ez"):
        runtime["edge_sets"][component] = dict(
            left_pad=indices, right_pad=indices, eps=torch.ones(2), sigma_base=torch.zeros(2),
            ext=torch.ones(2), base_decay=torch.ones(2), base_curl=torch.full((2,), .25),
            edge_flat=indices, volume=torch.ones(2), n_cap=torch.full((2,), 2.),
            left_nodes=indices, right_nodes=indices, left_sel=indices, right_sel=indices)
    def make_solver():
        solver = SimpleNamespace(device="cpu", breakdown_runtime=deepcopy(runtime))
        for axis in "xyz":
            setattr(solver, "E" + axis, torch.zeros(2))
            setattr(solver, f"ce{axis}_decay", torch.ones(2))
            setattr(solver, f"ce{axis}_curl", torch.full((2,), .25))
        return solver
    scalar, device = make_solver(), make_solver()
    pointers = {k: v.data_ptr() for k, v in device.breakdown_runtime.items() if isinstance(v, torch.Tensor)}
    coefficients = {f"ce{axis}_{kind}": getattr(device, f"ce{axis}_{kind}").data_ptr()
                    for axis in "xyz" for kind in ("decay", "curl")}
    step = torch.zeros(1, dtype=torch.int64)
    stress = BreakdownStressAccumulator.allocate(shape=(2,), critical_field=1., dt=.25)
    run_pointer = stress._current_run.data_ptr()
    for n, values in enumerate(((0., 2.), (2., 2.), (0., 2.), (2., 0.), (2., 0.), (2., 0.))):
        for solver, counter in ((scalar, n), (device, step)):
            solver.Ez.copy_(torch.tensor(values))
            advance_breakdown_state(solver, counter)
        stress.update(device.Ez.abs())
        step.add_(1)
        for key, value in scalar.breakdown_runtime.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, device.breakdown_runtime[key]), key
        for axis in "xyz":
            for kind in ("decay", "curl"):
                assert torch.equal(getattr(scalar, f"ce{axis}_{kind}"), getattr(device, f"ce{axis}_{kind}"))
    assert device.breakdown_runtime["trigger_step"].tolist() == [4, 1]
    assert device.breakdown_runtime["total_dissipated"] > 0
    assert pointers == {k: device.breakdown_runtime[k].data_ptr() for k in pointers}
    assert coefficients == {k: getattr(device, k).data_ptr() for k in coefficients}
    assert stress._current_run.data_ptr() == run_pointer


@pytest.mark.host_validation
def test_breakdown_trigger_is_exact_at_adjacent_field_values(monkeypatch):
    from witwin.maxwell.fdtd.runtime import breakdown

    critical = torch.ones(3)
    fields = torch.stack((torch.nextafter(critical[0], torch.tensor(0.)),
                          critical[1], torch.nextafter(critical[2], torch.tensor(2.))))
    indices = torch.arange(3)
    runtime = dict(
        state=torch.zeros(3, dtype=torch.int8), timer=torch.zeros(3), dt=.25,
        critical=critical, minimum_duration=torch.full((3,), .25),
        trigger_step=torch.full((3,), -1, dtype=torch.int64),
        field_before=torch.zeros(3), deposited_at_trigger=torch.zeros(3),
        eps_abs=torch.ones(3), node_volume=torch.ones(3),
        ramp_time=torch.full((3,), .25), post_sigma=torch.ones(3),
        base_sigma=torch.zeros(3),
        colocation={f"{component}_{side}": indices for component in ("ex", "ey", "ez")
                    for side in ("lo", "hi")},
    )
    monkeypatch.setattr(breakdown, "_scatter_edge_coefficients", lambda *args: None)
    solvers = [SimpleNamespace(Ex=torch.zeros(3), Ey=torch.zeros(3), Ez=fields.clone(),
                               breakdown_runtime=deepcopy(runtime)) for _ in range(2)]
    for solver, step in zip(solvers, (7, torch.tensor([7], dtype=torch.int64))):
        magnitude, _ = breakdown._node_field_magnitude(solver, solver.breakdown_runtime)
        assert torch.equal(magnitude, fields)
        advance_breakdown_state(solver, step)
        assert solver.breakdown_runtime["trigger_step"].tolist() == [-1, 7, 7]
        assert solver.breakdown_runtime["state"].tolist() == [
            breakdown.BREAKDOWN_STATE_INTACT,
            breakdown.BREAKDOWN_STATE_CONDUCTING,
            breakdown.BREAKDOWN_STATE_CONDUCTING,
        ]
        assert torch.equal(solver.breakdown_runtime["field_before"],
                           torch.tensor([0., fields[1], fields[2]]))
    for key, value in solvers[0].breakdown_runtime.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, solvers[1].breakdown_runtime[key]), key


@pytest.mark.host_validation
@pytest.mark.parametrize("name,closed", [
    ("w3_modulation_two_omega", False),
    ("modulated_slab_phase", True),
])
def test_modulated_phase_core_topology_admission(name, closed, monkeypatch):
    import witwin.maxwell as mw

    # Prepare the real solver on the host without loading or executing CUDA kernels.
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda *a, **kw: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._require_cuda_scene", lambda *a: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._get_fdtd_module", lambda: object())
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.initialization.require_cuda_scene", lambda *a: None)
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.initialization.get_fdtd_module", lambda: object())
    case = build_core_case(name)
    assert closed_transverse_pml(case.scene) is closed
    solver = mw.Simulation.fdtd(
        case.scene, frequencies=case.frequencies, run_time=mw.TimeConfig(time_steps=109),
        cpml_config={"memory_mode": case.cpml_memory, "kappa_max": case.cpml_kappa_max},
        **({"absorber": case.absorber} if case.absorber is not None else {}),
    ).prepare().solver
    assert solver.modulation_enabled
    assert solver.active_absorber_type == "cpml"
    expected = (["closed-transverse CPML (passivity guard)"] if closed else [])
    assert core_path.core_path_blockers(solver) == expected + ["non-CUDA execution"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")
def test_modulated_phase_checkpoint_resume_bitwise(tmp_path):
    import witwin.maxwell as mw
    from benchmark.core_path.gates import make_simulation, tensors

    total_steps, checkpoint_step, chunk_size = 109, 45, 32
    # Override the fixture's 256-step horizon to preserve this off-chunk resume case.
    case = replace(build_core_case("w3_modulation_two_omega"), steps=total_steps)
    assert not closed_transverse_pml(case.scene)

    def prepare():
        prepared = make_simulation(
            case, arm="candidate", steps=total_steps, full_field_dft=True,
        ).prepare()
        prepared.solver._core_runtime.chunk_size = chunk_size
        assert prepared.solver.modulation_enabled
        return prepared

    direct = prepare().run()
    partial = prepare()
    checkpoint = partial.run_until(checkpoint_step)
    for solver in (direct.solver, partial.solver):
        assert solver._core_step_graph_active
        assert solver._core_step_graph_k == chunk_size
        assert chunk_size in solver._core_step_graph_chunks
    assert checkpoint_step % chunk_size != 0
    assert "modulation_time" not in checkpoint.auxiliary
    assert checkpoint.auxiliary["modulation_step"].dtype == torch.int64
    assert checkpoint.auxiliary["modulation_step"].item() == checkpoint_step - 1
    path = tmp_path / "modulated-phase.pt"
    checkpoint.save(path)
    checkpoint = mw.FDTDResumeCheckpoint.load(path)
    assert "modulation_time" not in checkpoint.auxiliary
    resumed_prepared = prepare()
    assert resumed_prepared.solver is not partial.solver
    resumed = resumed_prepared.run(resume_from=checkpoint)

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "_modulation_step", "_modulation_time"):
        assert torch.equal(getattr(direct.solver, name), getattr(resumed.solver, name)), name
    # Graph capture is asserted on the fresh runs above only. Resumed runs never re-enter
    # whole-step capture (run_chunks refuses resume_from). The eager magnetic-surface
    # fallback of modulated_slab_phase is no longer covered here because the passivity
    # guard refuses closed-transverse CPML on the core path.
    # This open scene's soft plane wave also has a host-evaluated magnetic surface
    # source, so its resumed per-step fallback remains eager.
    assert direct.solver._modulation_step.item() == total_steps - 1
    assert direct.solver_stats["steps_run"] == resumed.solver_stats["steps_run"] == total_steps
    assert torch.count_nonzero(direct.solver.Ez)
    expected = tensors(direct.fields, "phasors") | tensors(direct.monitors, "monitors")
    actual = tensors(resumed.fields, "phasors") | tensors(resumed.monitors, "monitors")
    assert expected.keys() == actual.keys()
    for prefix in ("phasors/", "monitors/"):
        assert any(name.startswith(prefix) and value.is_complex() and torch.count_nonzero(value)
                   for name, value in expected.items())
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


def _field_family(path):
    for prefix, family in (("fields/E", "E"), ("fields/H", "H")):
        if path.startswith(prefix):
            return family
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")
@pytest.mark.parametrize("arm", ["chunked", "forced_fallback"])
def test_breakdown_core_arms_match_eager_existing_path(arm, monkeypatch):
    """Breakdown under a core runtime: the whole-step graph and its per-step fallback
    both match the eager existing path (Rule 2; discrete breakdown state exact), and
    the fallback never captures a per-step graph over the mutable coefficients."""
    from benchmark.core_path.gates import make_simulation, tensors
    from tests.support.numerical_equivalence import (
        assert_numerically_equivalent, assert_run_level_equivalent,
    )

    total_steps = 256
    case = build_core_case("breakdown_slab")
    graph_flags = ("_cuda_graph_active", "_port_cuda_graph_active",
                   "_network_cuda_graph_active", "_port_observer_graph_active")

    def state(solver):
        return (tensors({name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
                | tensors(dict(solver.breakdown_runtime), "breakdown"))

    def run(simulation_arm, steps=total_steps):
        # Breakdown refuses checkpoint execution (stop_step/resume), so the half
        # horizon is its own shorter run from the same seeded state.
        with monkeypatch.context() as patch:
            if simulation_arm == "candidate" and arm == "forced_fallback":
                patch.setattr(core_path.CorePathRuntime, "run_chunks", lambda self, solver, **kwargs: None)
            prepared = make_simulation(case, arm=simulation_arm, steps=steps, full_field_dft=True).prepare()
            solver = prepared.solver
            assert solver.breakdown_enabled
            assert (getattr(solver, "_core_runtime", None) is not None) is (simulation_arm == "candidate")
            # The G1 fixture's seeded fields, so breakdown triggers inside the horizon.
            for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
                field = getattr(solver, name)
                field.copy_(((torch.arange(field.numel(), device=field.device).reshape(field.shape) % 17) - 8)
                            * (2.0 ** (-20 - index)))
            return solver, prepared.run()

    eager_half, _ = run("baseline", total_steps // 2)
    eager_full, eager_result = run("baseline")
    core_half, _ = run("candidate", total_steps // 2)
    core_full, core_result = run("candidate")

    for solver in (eager_full, eager_half):
        assert not any(getattr(solver, flag, False) for flag in graph_flags)
    if arm == "chunked":
        assert core_full._core_step_graph_active
    else:
        assert not getattr(core_full, "_core_step_graph_active", False)
        assert not any(getattr(core_full, flag, False) for flag in graph_flags)
    assert (eager_full.breakdown_runtime["trigger_step"] >= 0).any().item()
    assert eager_full.breakdown_runtime["total_dissipated"].item() > 0

    assert_run_level_equivalent(state(eager_half), state(core_half), state(eager_full), state(core_full),
                                name=f"breakdown_slab/{arm}", family=_field_family)
    outputs = {"breakdown": eager_result.breakdown_data, "monitors": eager_result.monitors}
    core_outputs = {"breakdown": core_result.breakdown_data, "monitors": core_result.monitors}
    assert_numerically_equivalent(tensors(outputs, "outputs"), tensors(core_outputs, "outputs"),
                                  level="run", name=f"breakdown_slab/{arm}/outputs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")
def test_breakdown_refuses_lumped_coupling_on_core_path():
    """Coupled lumped objects never reach the port/network runners with breakdown:
    preparation refuses the composition on the core arm exactly as on the existing path."""
    import witwin.maxwell as mw
    from benchmark.core_path.gates import make_simulation
    from benchmark.core_path.scenes import _w1_case

    case = _w1_case("w1_lumped_port")
    case.scene.add_structure(mw.Structure(
        mw.Box(position=(-.01, 0, 0), size=(.01, .02, .02), device="cpu"),
        mw.Material(eps_r=2., breakdown=mw.DielectricBreakdown(
            critical_field=1e-8, minimum_duration=2e-10, post_breakdown_conductivity=0.1, ramp_time=1e-9))))
    for arm in ("baseline", "candidate"):
        with pytest.raises(NotImplementedError, match="dynamic dielectric breakdown"):
            make_simulation(case, arm=arm, steps=8, full_field_dft=False).prepare()

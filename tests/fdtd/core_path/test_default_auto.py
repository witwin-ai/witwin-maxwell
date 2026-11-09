"""Automatic admission, clean fallback and exact CUDA execution contracts."""
from dataclasses import asdict
from types import SimpleNamespace
import warnings

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell import simulation as simulation_module
from witwin.maxwell.compiler.accel import dispatch
from witwin.maxwell.fdtd.runtime import core_path, core_tables
from witwin.maxwell.scene import prepare_scene


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("cuda"):
        return

    def fail(*args, **kwargs):
        raise AssertionError("Automatic core-path CPU tests must never initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def _scene():
    return mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
                    grid=mw.GridSpec.uniform(.25), boundary=mw.BoundarySpec.pec(), device="cpu")


def _simulation(*, core="auto", setup="auto", cuda_label=False):
    scene = _scene()
    if cuda_label:
        scene.device = "cuda"
    return mw.Simulation.fdtd(scene, frequency=1e9, fdtd_core_path=core,
                              accelerated_setup=setup, run_time=mw.TimeConfig(time_steps=8),
                              spectral_sampler=mw.SpectralSampler(window="none"))


@pytest.mark.parametrize("name", ["fdtd_core_path", "accelerated_setup"])
@pytest.mark.parametrize("value", ["auto", True, False])
def test_config_accepts_tristate_without_coercion(name, value):
    config = mw.FDTDConfig(**{name: value})
    assert getattr(config, name) is value
    assert asdict(config)[name] is value
    assert config == mw.FDTDConfig(**{name: value})
    assert f"{name}={value!r}" in repr(config)


@pytest.mark.parametrize("name", ["fdtd_core_path", "accelerated_setup"])
@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", "AUTO", "", [], {}])
def test_config_rejects_other_values(name, value):
    with pytest.raises(ValueError, match=f"{name} must be 'auto', True or False"):
        mw.FDTDConfig(**{name: value})
    with pytest.raises(ValueError, match=name):
        mw.Simulation.fdtd(_scene(), frequency=1e9, **{name: value})


def test_defaults_are_auto():
    assert mw.FDTDConfig().fdtd_core_path == mw.FDTDConfig().accelerated_setup == "auto"
    simulation = mw.Simulation.fdtd(_scene(), frequency=1e9)
    assert simulation.config.fdtd_core_path == simulation.config.accelerated_setup == "auto"


@pytest.mark.parametrize("paths,reason", [
    (("core", "core"), None), (("core", "existing"), "PMC boundaries"),
])
def test_result_metadata_covers_adjoint_and_sweep_stats(paths, reason):
    scene = _scene()
    columns = [{"execution_path": path, "fallback_reason": reason if path == "existing" else None}
               for path in paths]
    for stats in (columns[-1], {"columns": columns}):
        result = mw.Result(method="fdtd", scene=scene, frequency=1e9, solver_stats=stats)
        assert result._metadata["execution_path"] == paths[-1]
        assert result._metadata["fallback_reason"] == reason
        assert result.stats()["execution_path"] == paths[-1]


def test_decision_uses_shared_blockers(monkeypatch):
    simulation = _simulation()
    assert core_path.core_path_blockers(SimpleNamespace(scene=simulation.scene)) == ["non-CUDA execution"]
    assert core_path.core_path_decision(simulation) == ("existing", "non-CUDA execution")
    seen = []

    def blockers(solver):
        seen.append(solver.scene)
        return ["first blocker", "second blocker"]

    monkeypatch.setattr(core_path, "core_path_blockers", blockers)
    assert core_path.core_path_decision(simulation) == ("existing", "first blocker")
    with pytest.raises(ValueError, match="first blocker; second blocker"):
        core_path.validate_core_path_request(simulation)
    assert seen == [simulation.scene, simulation.scene]


@pytest.mark.parametrize("control,reason", [
    ({"initialize": False}, "non-initializing prepare (initialize=False)"),
    ({"resume_from": object()}, "resume_from"),
    ({}, None),
])
def test_initializing_decision(control, reason):
    simulation = _simulation(cuda_label=True)
    assert core_path.core_path_blockers(SimpleNamespace(scene=simulation.scene)) == []
    assert core_path.core_path_decision(simulation, **control) == (
        "existing" if reason else "core", reason)


@pytest.mark.parametrize("symmetry,refused", [
    ((None, "PMC", None), True), ((None, ("PMC", "high"), None), True),
    ((None, "PEC", None), False), ((None, None, None), False),
])
def test_symmetry_pmc_plane_blocks_core(symmetry, refused):
    # A PMC symmetry plane lowers to a PMC face, which the core kernels do not implement.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(.05),
                     boundary=mw.BoundarySpec.pml(num_layers=2), symmetry=symmetry, device="cpu")
    blockers = core_path.core_path_blockers(SimpleNamespace(scene=prepare_scene(scene)))
    assert ("PMC boundaries" in blockers) is refused


@pytest.mark.parametrize("absorber,refused", [("cpml", True), ("stablepml", True), ("auto", False)])
def test_guard_blocker_decided_before_preparation(absorber, refused):
    # Closed-transverse CPML is refused before any solver build, so automatic
    # admission prepares the existing solver once instead of twice.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(.05),
                     boundary=mw.BoundarySpec.faces(x="periodic", y="pec", z="pml", num_layers=2), device="cpu")
    simulation = mw.Simulation.fdtd(scene, frequency=1e9, absorber=absorber)
    blockers = core_path.core_path_request_blockers(simulation)
    assert ("closed-transverse CPML (passivity guard)" in blockers) is refused


def test_material_cache_prevents_auto_core():
    simulation = _simulation(cuda_label=True)
    simulation.material_cache = object()
    assert core_path.core_path_decision(simulation) == ("existing", "material_cache override")


@pytest.mark.parametrize("config,reason", [
    ({"field_dtype": torch.float64}, "non-float32 fields"),
    ({"backend": "cpu_reference"}, "non-native backend"),
    ({"adjoint_checkpoint_stride": 4}, "adjoint/gradient-requiring execution"),
])
def test_config_blockers(config, reason):
    simulation = _simulation(cuda_label=True)
    for name, value in config.items():
        setattr(simulation.config, name, value)
    assert core_path.core_path_decision(simulation) == ("existing", reason)


@pytest.fixture
def cpu_backend(monkeypatch):
    """Exercise real preparation and Result plumbing with CPU-only computation."""
    builds = []

    def cpu_prepare(scene, **kwargs):
        device = scene.device
        try:
            scene.device = "cpu"
            prepared = prepare_scene(scene, **kwargs)
        finally:
            scene.device = device
        prepared.device = device
        return prepared

    class Backend:
        def __init__(self, scene, **kwargs):
            self.scene = scene
            self.cpml_config = dict(kwargs["cpml_config"])
            self.dt = 1e-12
            builds.append(self)

        def init_field(self):
            self.setup_enabled = dispatch.setup_enabled.get()
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                setattr(self, name, torch.zeros(3, 3, 3))
            self.spectrum = torch.zeros(2)

        def solve(self, **kwargs):
            if hasattr(self, "_core_runtime") and hasattr(self._core_runtime, "run_chunks"):
                self._core_runtime.run_chunks(self)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                getattr(self, name).add_(1)
            self.spectrum.add_(2)
            return {"Ex": self.Ex, "Ey": self.Ey, "Ez": self.Ez,
                    "observers": {"probe": {"data": self.spectrum}}}

    def time_steps(self, solver, scene=None):
        solver._time_plan = {"total_steps": 8}
        return 8

    monkeypatch.setattr(simulation_module, "prepare_scene", cpu_prepare)
    monkeypatch.setattr(simulation_module, "_require_cuda_scene", lambda *a, **kw: None)
    monkeypatch.setattr(simulation_module, "_resolve_fdtd_backend", lambda: (Backend, None))
    monkeypatch.setattr(simulation_module, "_to_tensor_fields", lambda fields, device: fields)
    monkeypatch.setattr(mw.Simulation, "_resolve_fdtd_time_steps", time_steps)
    monkeypatch.setattr(core_path, "CorePathRuntime", lambda solver: SimpleNamespace())
    monkeypatch.setattr(core_tables, "source_tables_supported", lambda solver: False)
    return builds


@pytest.mark.parametrize("core", ["auto", False])
def test_fallback_warning_once_per_simulation(cpu_backend, core):
    simulation = _simulation(core=core)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            simulation.prepare().run()
            result = simulation.run()
            assert result.stats()["execution_path"] == result._metadata["execution_path"] == "existing"
            reason = "non-CUDA execution" if core == "auto" else None
            assert result.stats()["fallback_reason"] == result._metadata["fallback_reason"] == reason
    assert len(caught) == (1 if core == "auto" else 0)
    if caught:
        assert caught[0].category is mw.FDTDCoreFallbackWarning


@pytest.mark.parametrize("entry", ["prepare", "run"])
def test_explicit_true_still_raises(entry):
    with pytest.raises(ValueError, match="^fdtd_core_path=True is blocked by: non-CUDA execution$"):
        getattr(_simulation(core=True), entry)()


def test_noninitializing_prepare(cpu_backend):
    simulation = _simulation(cuda_label=True)
    with pytest.warns(mw.FDTDCoreFallbackWarning, match="initialize=False"):
        solver = simulation._build_fdtd_solver(initialize=False)
    assert solver._fdtd_execution_path == "existing"
    assert not hasattr(solver, "Ex")
    with pytest.raises(ValueError, match="fdtd_core_path=True does not support resume / non-initializing prepare"):
        _simulation(core=True)._build_fdtd_solver(initialize=False)


@pytest.mark.parametrize("core,setup,enabled", [
    (False, False, False), (False, True, True), (True, False, True),
    ("auto", "auto", True), (False, "auto", True),
])
def test_setup_controls_are_independent(cpu_backend, core, setup, enabled):
    solver = _simulation(core=core, setup=setup, cuda_label=True).prepare().solver
    assert solver.setup_enabled is enabled
    assert not dispatch.setup_enabled.get()


def test_material_cache_disables_auto_setup_and_core(cpu_backend):
    simulation = _simulation(cuda_label=True)
    simulation.material_cache = SimpleNamespace(_bind=lambda *args, **kwargs: {})
    with pytest.warns(mw.FDTDCoreFallbackWarning, match="material_cache override"):
        solver = simulation.prepare().solver
    assert not solver.setup_enabled
    assert not hasattr(solver, "_core_runtime")
    assert solver._fdtd_execution_path == "existing"
    assert solver._fdtd_fallback_reason == "material_cache override"


@pytest.mark.parametrize("failure", ["decision", "activation", "run_chunks", "capture"])
def test_failure_rebuilds_clean_existing_solver(cpu_backend, monkeypatch, failure):
    simulation = _simulation(cuda_label=True)
    if failure == "decision":
        monkeypatch.setattr(core_path, "core_path_blockers",
                            lambda solver: (_ for _ in ()).throw(RuntimeError("forced admission failure")))
    elif failure == "activation":
        def activate(solver, simulation):
            solver.Ex.fill_(99)
            simulation.scene._mode_source_gauges = {"failed": object()}
            raise RuntimeError("forced activation failure")
        monkeypatch.setattr(core_path, "activate_core_path", activate)
    else:
        def run_chunks(solver):
            solver.Ex.fill_(99)
            solver.spectrum.fill_(99)
            if failure == "capture":
                raise core_path._FDTDCoreFallback("RuntimeError: forced capture failure")
            raise RuntimeError("forced run_chunks failure")
        monkeypatch.setattr(core_path, "CorePathRuntime", lambda solver: SimpleNamespace(run_chunks=run_chunks))
    with pytest.warns(mw.FDTDCoreFallbackWarning, match=f"RuntimeError: forced {failure if failure != 'decision' else 'admission'} failure"):
        prepared = simulation.prepare()
        result = prepared.run()
    baseline = _simulation(core=False, cuda_label=True).run()
    assert result.solver is prepared.solver
    assert not hasattr(result.solver, "_core_runtime")
    assert "failed" not in simulation.scene._mode_source_gauges
    assert result.stats()["execution_path"] == result._metadata["execution_path"] == "existing"
    assert result.stats()["fallback_reason"] == result._metadata["fallback_reason"]
    assert torch.equal(result.solver.Ex, baseline.solver.Ex)
    assert torch.equal(result.at().monitor("probe")["data"], baseline.at().monitor("probe")["data"])
    assert not dispatch.setup_enabled.get()


def test_runtime_admission_blocker_rebuilds(cpu_backend, monkeypatch):
    original = core_path.core_path_blockers
    monkeypatch.setattr(core_path, "core_path_blockers", lambda solver:
                        ["magnetic permeability"] if hasattr(solver, "Ex") else original(solver))
    with pytest.warns(mw.FDTDCoreFallbackWarning, match="magnetic permeability"):
        prepared = _simulation(cuda_label=True).prepare()
    assert len(cpu_backend) == 2
    assert not hasattr(prepared.solver, "_core_runtime")
    assert prepared.solver._fdtd_fallback_reason == "magnetic permeability"


def test_resume_rebuilds_before_preflight(cpu_backend, monkeypatch):
    from witwin.maxwell.fdtd import resume
    simulation = _simulation(cuda_label=True)
    prepared = simulation.prepare()
    original = prepared.solver
    checked = []

    def preflight(solver, checkpoint, **kwargs):
        assert not hasattr(solver, "_core_runtime")
        assert not solver.setup_enabled
        checked.append(solver)

    monkeypatch.setattr(resume, "preflight_resume_checkpoint", preflight)
    with pytest.warns(mw.FDTDCoreFallbackWarning, match="resume_from"):
        result = prepared.run(resume_from=object())
    assert result.solver is prepared.solver is checked[0]
    assert result.solver is not original
    assert result.stats()["fallback_reason"] == "resume_from"


def test_explicit_true_prepared_resume_retains_existing_contract(cpu_backend, monkeypatch):
    from witwin.maxwell.fdtd import resume
    prepared = _simulation(core=True, cuda_label=True).prepare()
    original = prepared.solver
    monkeypatch.setattr(resume, "preflight_resume_checkpoint", lambda *args, **kwargs: None)
    with warnings.catch_warnings(record=True) as caught:
        result = prepared.run(resume_from=object())
    assert result.solver is original
    assert hasattr(result.solver, "_core_runtime")
    assert not caught


@pytest.mark.parametrize("dispersive", [False, True])
def test_capture_exception_escapes_run_chunks_in_auto_mode(monkeypatch, dispersive):
    from witwin.maxwell.fdtd import checkpoint
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.sources, runtime.chunk_size, runtime.graphs = SimpleNamespace(steps=8), 16, {}
    runtime.step = torch.zeros(1, dtype=torch.int64)
    runtime.point_observers = None
    solver = SimpleNamespace(scene=SimpleNamespace(), device="cpu", _core_path_auto=True,
                             dispersive_enabled=dispersive, observers_enabled=False,
                             dft_enabled=False, Ex=torch.tensor([2.]))
    monkeypatch.setattr(checkpoint, "graph_state_tensors", lambda s: {"Ex": s.Ex})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    def step():
        solver.Ex.add_(1)

    def capture(solver, closure, count):
        closure()
        raise RuntimeError("forced capture failure after mutation")

    monkeypatch.setattr(runtime, "prepare_step", lambda *a, **kw: (step, None))
    monkeypatch.setattr(runtime, "capture_chunk", capture)
    with pytest.raises(core_path._FDTDCoreFallback, match="RuntimeError: forced capture failure after mutation"):
        runtime.run_chunks(solver, time_steps=8, start_step=0, end_step=8,
                           use_cuda_graph=True, use_gpu_dft=False, cooperative=False,
                           cooperative_driver=False, cooperative_observer=None, resume_from=None,
                           shutoff=0., shutoff_check_interval=100, shutoff_min_step=0,
                           enable_plot=False, pbar=None)
    assert torch.equal(solver.Ex, torch.tensor([2.]))
    assert runtime.graphs == {}


def test_auto_setup_failure_reprepares_without_acceleration(cpu_backend, monkeypatch):
    simulation = _simulation(cuda_label=True)
    prepared = simulation.prepare()
    backend = type(prepared.solver)
    init_field = backend.init_field

    def fail_setup(self):
        init_field(self)
        if self.setup_enabled:
            self.Ex.fill_(99)
            raise RuntimeError("forced setup failure")

    monkeypatch.setattr(backend, "init_field", fail_setup)
    with pytest.warns(mw.FDTDCoreFallbackWarning, match="RuntimeError: forced setup failure"):
        result = simulation.run()
    assert not result.solver.setup_enabled
    assert not hasattr(result.solver, "_core_runtime")
    assert torch.equal(result.solver.Ex, torch.ones_like(result.solver.Ex))
    assert not dispatch.setup_enabled.get()


def test_capture_retry_handles_auto_setup_failure(cpu_backend, monkeypatch):
    simulation = _simulation(cuda_label=True)
    prepared = simulation.prepare()
    backend = type(prepared.solver)
    init_field = backend.init_field

    def fail_setup(self):
        init_field(self)
        if self.setup_enabled:
            self.Ex.fill_(99)
            raise RuntimeError("failed setup during retry")

    def fail_capture(solver):
        solver.Ex.fill_(99)
        raise RuntimeError("forced capture failure")

    monkeypatch.setattr(backend, "init_field", fail_setup)
    prepared.solver._core_runtime.run_chunks = fail_capture
    with pytest.warns(mw.FDTDCoreFallbackWarning, match="RuntimeError: forced capture failure"):
        result = prepared.run()
    assert not result.solver.setup_enabled
    assert torch.equal(result.solver.Ex, torch.ones_like(result.solver.Ex))
    assert result.stats()["fallback_reason"] == "RuntimeError: forced capture failure"


def _gpu_simulation(name, core):
    from benchmark.core_path.scenes import build_core_case
    from benchmark.core_path.gates import make_simulation
    simulation = make_simulation(build_core_case(name), arm="baseline", steps=64, full_field_dft=True)
    simulation.config.fdtd_core_path = core
    simulation.config.accelerated_setup = "auto"
    return simulation


def _gpu_payload(result):
    from benchmark.core_path.gates import tensors
    payload = tensors(result.fields, "fields")
    payload.update(tensors(result.monitors, "monitors"))
    payload.update(tensors({name: getattr(result.solver, name)
                           for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "last_fields"))
    return payload


def _assert_payload_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert any(torch.count_nonzero(tensor) for tensor in expected.values())
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.cuda
@cuda
@pytest.mark.parametrize("kind", ["blocker", "resume", "capture", "dispersive_capture"])
def test_cuda_fallback_is_exact_existing(kind, monkeypatch):
    name = "pmc_cavity" if kind == "blocker" else "debye_slab" if kind == "dispersive_capture" else "cpml_dense"
    expected_simulation = _gpu_simulation(name, False)
    checkpoint = expected_simulation.prepare().run_until(17) if kind == "resume" else None
    expected = expected_simulation.prepare().run(resume_from=checkpoint)
    simulation = _gpu_simulation(name, "auto")
    prepared = simulation.prepare() if kind != "blocker" else None
    if "capture" in kind:
        def fail_capture(runtime, solver, step, count):
            step()
            raise RuntimeError("forced capture failure after mutation")
        monkeypatch.setattr(core_path.CorePathRuntime, "capture_chunk", fail_capture)
    with pytest.warns(mw.FDTDCoreFallbackWarning):
        result = (prepared or simulation.prepare()).run(resume_from=checkpoint)
    assert result.stats()["execution_path"] == result._metadata["execution_path"] == "existing"
    reason = result.stats()["fallback_reason"]
    assert reason == result._metadata["fallback_reason"]
    assert ("PMC boundaries" if kind == "blocker" else "resume_from" if kind == "resume"
            else "RuntimeError: forced capture failure after mutation") in reason
    _assert_payload_equal(_gpu_payload(result), _gpu_payload(expected))


@pytest.mark.cuda
@cuda
def test_cuda_eligible_and_three_repeats_are_exact():
    expected = _gpu_payload(_gpu_simulation("cpml_dense", True).run())
    simulation = _gpu_simulation("cpml_dense", "auto")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", mw.FDTDCoreFallbackWarning)
        for _ in range(3):
            result = simulation.run()
            assert result.stats()["execution_path"] == result._metadata["execution_path"] == "core"
            assert result.stats()["fallback_reason"] is result._metadata["fallback_reason"] is None
            _assert_payload_equal(_gpu_payload(result), expected)
    assert not [entry for entry in caught if entry.category is mw.FDTDCoreFallbackWarning]

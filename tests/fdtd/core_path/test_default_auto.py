"""Single-path admission, the failure policy and exact CUDA execution contracts."""
from dataclasses import replace
from types import SimpleNamespace
import warnings

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell import simulation as simulation_module
from witwin.maxwell.compiler.accel import dispatch
from witwin.maxwell.fdtd.runtime import core_path, core_tables
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.simulation import MODE_SOURCE_GAUGES


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("cuda"):
        return

    def fail(*args, **kwargs):
        raise AssertionError("Automatic core-path CPU tests must never initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def _scene():
    return mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
                    grid=mw.GridSpec.uniform(.25), boundary=mw.BoundarySpec.pec())


def _simulation(*, cuda_label=False):
    return mw.Simulation.fdtd(_scene(), frequencies=1e9, config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=8), spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
        device="cuda" if cuda_label else "cpu"))


def test_result_carries_no_execution_path_keys():
    # W7 row 16c (0.7.0, breaking): there is one FDTD path, so stats and metadata name none.
    scene = _scene()
    for stats in ({"execution_path": "core", "fallback_reason": None}, {"columns": [{"time_steps": 8}]}):
        result = mw.Result(method="fdtd", scene=scene, frequency=1e9, solver_stats=stats)
        assert "execution_path" not in result._metadata and "fallback_reason" not in result._metadata
    assert not hasattr(mw.fdtd, "FDTDCoreFallbackWarning")
    assert "FDTDCoreFallbackWarning" not in dir(mw.fdtd)


@pytest.mark.parametrize("config", [
    {}, {"field_dtype": torch.float64}, {"adjoint_checkpoint_stride": 4}, {"absorber": "stablepml"},
])
def test_every_scene_is_admitted(cpu_backend, config):
    # No admission blocker or path decision is left (W7 rows 16c/16d): every configuration
    # prepares on the core path.
    simulation = _simulation(cuda_label=True)
    simulation.config = replace(simulation.config, **config)
    if "field_dtype" not in config:
        assert simulation.prepare().solver._core_runtime is not None
    assert not hasattr(core_path, "core_path_decision")
    assert not hasattr(core_path, "core_path_blockers")
    assert not hasattr(core_path, "core_path_request_blockers")


def test_cpu_scene_is_refused_before_preparation():
    # CPU execution is refused by the CUDA requirement, not routed to another path.
    with pytest.raises(ValueError, match=r"requires config.device to be CUDA, got cpu"):
        _simulation().prepare()


def test_material_cache_refuses_gradient_execution():
    simulation = _simulation(cuda_label=True)
    simulation.config = replace(simulation.config, material_cache=mw.fdtd.FDTDMaterialCache())
    simulation.has_trainable_parameters = True
    with pytest.raises(NotImplementedError, match="single-device detached forward"):
        simulation._build_fdtd_solver_for_scene(simulation.scene, initialize=True)


@pytest.fixture
def cpu_backend(monkeypatch):
    """Exercise real preparation and Result plumbing with CPU-only computation."""
    builds = []

    def cpu_prepare(scene, **kwargs):
        # Prepare CPU tensors under the configured device label.
        label = kwargs.pop("device", None)
        prepared = prepare_scene(scene, device="cpu", **kwargs)
        prepared.device = "cuda" if label is None else label
        return prepared

    class Backend:
        def __init__(self, scene, **kwargs):
            self.scene = scene
            self.cpml = kwargs["cpml"]
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
                    "observers": {"probe": {"kind": "point", "fields": ("Ez",), "components": {"Ez": self.spectrum}}}}

    def time_steps(self, solver, scene=None):
        solver._time_plan = {"total_steps": 8}
        return 8

    monkeypatch.setattr(simulation_module, "prepare_scene", cpu_prepare)
    monkeypatch.setattr(simulation_module, "_require_cuda_device", lambda *a, **kw: None)
    monkeypatch.setattr(simulation_module, "_resolve_fdtd_backend", lambda: (Backend, None))
    monkeypatch.setattr(simulation_module, "_to_tensor_fields", lambda fields, device: fields)
    monkeypatch.setattr(mw.Simulation, "_resolve_fdtd_time_steps", time_steps)
    monkeypatch.setattr(core_path, "CorePathRuntime", lambda solver: SimpleNamespace())
    monkeypatch.setattr(core_tables, "source_tables_supported", lambda solver: False)
    return builds


def test_core_run_reports_no_path(cpu_backend):
    simulation = _simulation(cuda_label=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            result = simulation.prepare().run()
            assert result.solver._core_runtime is not None
            assert "execution_path" not in result.stats() and "fallback_reason" not in result.stats()
            assert "execution_path" not in result._metadata and "fallback_reason" not in result._metadata
    assert len(cpu_backend) == 2


def test_noninitializing_prepare(cpu_backend):
    # The initializer owns activation: the decision is recorded, nothing is allocated.
    simulation = _simulation(cuda_label=True)
    solver = simulation._build_fdtd_solver(initialize=False)
    assert not hasattr(solver, "_fdtd_fallback_reason")
    assert not hasattr(solver, "Ex")
    assert not hasattr(solver, "_core_runtime")


def test_core_prepares_with_accelerated_setup(cpu_backend):
    solver = _simulation(cuda_label=True).prepare().solver
    assert solver._core_runtime is not None
    assert solver.setup_enabled is True
    assert not dispatch.setup_enabled.get()


def test_material_cache_runs_core_with_torch_setup(cpu_backend, monkeypatch):
    # The cache binds a PyTorch-compiled material model; the core path steps the tables built from it.
    monkeypatch.setattr(mw.fdtd.FDTDMaterialCache, "_bind", lambda self, *args, **kwargs: {})
    simulation = _simulation(cuda_label=True)
    simulation.config = replace(simulation.config, material_cache=mw.fdtd.FDTDMaterialCache())
    solver = simulation.prepare().solver
    assert not solver.setup_enabled
    assert solver._core_runtime is not None


@pytest.mark.parametrize("failure", ["activation", "run_chunks", "capture"])
def test_core_failure_raises(cpu_backend, monkeypatch, failure):
    # User decision 2026-11-12 (W7 row 16c): an activation (core-load), solve or capture
    # failure raises its own error; nothing rebuilds another path.
    simulation = _simulation(cuda_label=True)
    if failure == "activation":
        def activate(solver):
            solver.Ex.fill_(99)
            MODE_SOURCE_GAUGES[simulation.scene] = {"failed": object()}
            raise RuntimeError("forced activation failure")
        monkeypatch.setattr(core_path, "activate_core_path", activate)
        with pytest.raises(RuntimeError, match="^forced activation failure$"):
            simulation.prepare()
        # A failed preparation never seeds the next one.
        assert "failed" not in MODE_SOURCE_GAUGES.get(simulation.scene, {})
    else:
        def run_chunks(solver):
            solver.Ex.fill_(99)
            raise RuntimeError(f"forced {failure} failure")
        monkeypatch.setattr(core_path, "CorePathRuntime", lambda solver: SimpleNamespace(run_chunks=run_chunks))
        prepared = simulation.prepare()
        with pytest.raises(RuntimeError, match=f"^forced {failure} failure$"):
            prepared.run()
        assert hasattr(prepared.solver, "_core_runtime")
    assert len(cpu_backend) == 1
    assert not dispatch.setup_enabled.get()


def test_resume_keeps_the_prepared_core_solver(cpu_backend, monkeypatch):
    from witwin.maxwell.fdtd import resume
    simulation = _simulation(cuda_label=True)
    prepared = simulation.prepare()
    original = prepared.solver
    checked = []

    def preflight(solver, checkpoint, **kwargs):
        assert hasattr(solver, "_core_runtime")
        assert solver.setup_enabled
        checked.append(solver)

    monkeypatch.setattr(resume, "preflight_resume_checkpoint", preflight)
    result = prepared.run(resume_from=object())
    assert result.solver is prepared.solver is checked[0] is original
    assert result.solver._core_runtime is not None


@pytest.mark.parametrize("dispersive", [False, True])
def test_capture_exception_escapes_run_chunks(monkeypatch, dispersive):
    from witwin.maxwell.fdtd import checkpoint
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.sources, runtime.chunk_size, runtime.graphs = SimpleNamespace(steps=8), 16, {}
    runtime.step = torch.zeros(1, dtype=torch.int64)
    runtime.point_observers = None
    solver = SimpleNamespace(scene=SimpleNamespace(scene=SimpleNamespace()), device="cpu",
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
    with pytest.raises(RuntimeError, match="^forced capture failure after mutation$"):
        runtime.run_chunks(solver, time_steps=8, start_step=0, end_step=8,
                           use_cuda_graph=True, use_gpu_dft=False, cooperative=False,
                           cooperative_driver=False, cooperative_observer=None,
                           shutoff=0., shutoff_check_interval=100, shutoff_min_step=0,
                           enable_plot=False, pbar=None)
    assert torch.equal(solver.Ex, torch.tensor([2.]))
    assert runtime.graphs == {}


def test_accelerated_setup_failure_retries_core_with_torch_setup(cpu_backend, monkeypatch):
    simulation = _simulation(cuda_label=True)
    prepared = simulation.prepare()
    backend = type(prepared.solver)
    init_field = backend.init_field

    def fail_setup(self):
        init_field(self)
        if self.setup_enabled:
            self.Ex.fill_(99)
            MODE_SOURCE_GAUGES[simulation.scene] = {"failed": object()}
            raise RuntimeError("forced setup failure")

    monkeypatch.setattr(backend, "init_field", fail_setup)
    with pytest.warns(mw.MaxwellWarning, match=r"Accelerated CUDA setup failed \(RuntimeError: forced setup failure\)"):
        result = simulation.run()
    assert not result.solver.setup_enabled
    assert result.solver._core_runtime is not None
    assert hasattr(result.solver, "_core_runtime")
    assert "failed" not in MODE_SOURCE_GAUGES.get(simulation.scene, {})
    assert torch.equal(result.solver.Ex, torch.ones_like(result.solver.Ex))
    assert not dispatch.setup_enabled.get()


def test_torch_setup_failure_raises(cpu_backend, monkeypatch):
    # The retry is the last attempt: its own failure raises.
    simulation = _simulation(cuda_label=True)
    backend = type(simulation.prepare().solver)
    init_field = backend.init_field

    def fail(self):
        init_field(self)
        raise RuntimeError("torch setup failure" if not self.setup_enabled else "accelerated setup failure")

    monkeypatch.setattr(backend, "init_field", fail)
    with pytest.raises(RuntimeError, match="^torch setup failure$"):
        simulation.prepare()
    assert not dispatch.setup_enabled.get()


def _gpu_simulation(name):
    from benchmark.core_path.scenes import build_core_case
    from benchmark.core_path.gates import make_simulation
    return make_simulation(build_core_case(name), steps=64, full_field_dft=True)


def _gpu_payload(result):
    from benchmark.core_path.gates import api_dialect, tensors
    payload = tensors(api_dialect.fields(result), "fields")
    payload.update(tensors(api_dialect.monitors(result), "monitors"))
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
def _cell(*monitors, bloch=True):
    boundary = (mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2, bloch_wavevector=(1., 0., 0.))
                if bloch else mw.BoundarySpec.pml(num_layers=2))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(.025), boundary=boundary,
                     sources=(mw.PointDipole(position=(0., 0., 0.), polarization="Ez",
                                             source_time=mw.GaussianPulse(frequency=1e9, fwidth=.5e9)),),
                     monitors=(mw.PointMonitor("point", position=(.05, 0., 0.), fields=("Ez",)), *monitors))
    return mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=64)))


@pytest.mark.cuda
@cuda
def test_cuda_bloch_cell_runs_whole_step_core():
    result = _cell().run()
    assert result.solver._core_runtime is not None and result.solver._core_step_graph_active
    assert any(torch.count_nonzero(tensor) for tensor in _gpu_payload(result).values())


@pytest.mark.cuda
@cuda
@pytest.mark.parametrize("kind", ["capture", "dispersive_capture", "resume_capture"])
def test_cuda_capture_failure_raises_and_restores(kind, monkeypatch):
    # A failed whole-step capture raises its own error after restoring the solver state
    # (W7 row 16c: no rebuild on another path). A fresh preparation of the same simulation
    # then reproduces the uninterrupted run.
    name = "debye_slab" if kind == "dispersive_capture" else "cpml_dense"
    expected = _gpu_simulation(name).run()
    simulation = _gpu_simulation(name)
    checkpoint = simulation.prepare().run_until(17) if kind == "resume_capture" else None
    prepared = simulation.prepare()
    assert hasattr(prepared.solver, "_core_runtime")
    with monkeypatch.context() as patch:
        def fail_capture(runtime, solver, step, count):
            step()
            raise RuntimeError("forced capture failure after mutation")
        patch.setattr(core_path.CorePathRuntime, "capture_chunk", fail_capture)
        if checkpoint is None:
            before = {name: getattr(prepared.solver, name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        with pytest.raises(RuntimeError, match="^forced capture failure after mutation$"):
            prepared.run(resume_from=checkpoint)
    assert not prepared.solver._core_step_graph_active
    if checkpoint is None:
        for field, value in before.items():
            assert torch.equal(getattr(prepared.solver, field), value), field
    result = simulation.prepare().run(resume_from=checkpoint)
    assert result.solver._core_step_graph_active
    _assert_payload_equal(_gpu_payload(result), _gpu_payload(expected))


@pytest.mark.cuda
@cuda
def test_cuda_accelerated_setup_failure_runs_core_with_torch_setup(monkeypatch):
    # A failed accelerated setup retries once on the core path with the torch setup; the run
    # equals the torch-setup oracle preparation of the same simulation on core.
    from tests.support.torch_setup_oracle import torch_setup_prepare
    expected = torch_setup_prepare(_gpu_simulation("debye_slab")).run()
    simulation = _gpu_simulation("debye_slab")
    backend = type(simulation.prepare().solver)
    init_field = backend.init_field

    def fail_accelerated(self):
        if dispatch.setup_enabled.get():
            raise RuntimeError("forced accelerated setup failure")
        init_field(self)

    monkeypatch.setattr(backend, "init_field", fail_accelerated)
    with pytest.warns(mw.MaxwellWarning, match="forced accelerated setup failure"):
        result = simulation.run()
    assert result.solver._core_runtime is not None and result.solver._core_step_graph_active
    _assert_payload_equal(_gpu_payload(result), _gpu_payload(expected))


@pytest.mark.cuda
@cuda
@pytest.mark.parametrize("name,stop", [("cpml_dense", 17), ("debye_slab", 32), ("debye_slab", 63)])
def test_cuda_resume_runs_core(name, stop):
    # run_until(stop) + resume on the core path reproduces the uninterrupted core run.
    expected = _gpu_simulation(name).run()
    simulation = _gpu_simulation(name)
    checkpoint = simulation.prepare().run_until(stop)
    result = simulation.prepare().run(resume_from=checkpoint)
    assert result.solver._core_runtime is not None
    assert result.solver._core_step_graph_active
    assert sum(result.solver._core_step_graph_chunks) == 64 - stop
    _assert_payload_equal(_gpu_payload(result), _gpu_payload(expected))


@pytest.mark.cuda
@cuda
@pytest.mark.parametrize("stop", [17, 32])
def test_cuda_resumed_embedded_network_ports_match_the_uninterrupted_run(stop):
    # The core step reads the port observer counter filled from the restored sample index,
    # so the resumed port and network V/I spectra equal the uninterrupted run (W7 row 7, T2).
    expected = _gpu_simulation("w1_network").run()
    simulation = _gpu_simulation("w1_network")
    checkpoint = simulation.prepare().run_until(stop)
    result = simulation.prepare().run(resume_from=checkpoint)
    assert result.solver._core_runtime is not None
    assert result.solver._core_step_graph_active
    _assert_payload_equal(_gpu_payload(result), _gpu_payload(expected))
    port, reference = result.port("feed"), expected.port("feed")
    assert torch.count_nonzero(reference.voltage)
    assert torch.equal(port.voltage, reference.voltage)
    assert torch.equal(port.current, reference.current)


@pytest.mark.cuda
@cuda
def test_cuda_eligible_and_three_repeats_are_exact():
    simulation = _gpu_simulation("cpml_dense")
    expected = _gpu_payload(simulation.run())
    for _ in range(3):
        result = simulation.run()
        assert result.solver._core_runtime is not None
        _assert_payload_equal(_gpu_payload(result), expected)

"""CPU admission walks and a separately marked full CUDA preparation check."""
from contextvars import Context
import gc
import weakref
from types import SimpleNamespace

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
        pytest.fail("Admission walk tests must not initialize CUDA.")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def _scene(count=0):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.25), boundary=mw.BoundarySpec.pec(), device="cpu",
        monitors=[mw.PointMonitor(f"point_{i}", position=(0., 0., 0.), fields=("Ez",))
                  for i in range(count)],
        metadata={"parameter": torch.tensor(1.)},
    )


@pytest.fixture
def cpu_backend(monkeypatch):
    """Preserve preparation/admission control flow, replacing CUDA computation only."""
    checks = []

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

        def init_field(self):
            assert dispatch._requires_grad_memo.get() is not None
            checks.append(dispatch.eligible(self.scene, 1., 1., self.scene.subpixel))
            checks.append(dispatch.eligible(
                self.scene, 1., 1., self.scene.subpixel, coefficients=True))

    monkeypatch.setattr(simulation_module, "prepare_scene", cpu_prepare)
    monkeypatch.setattr(simulation_module, "_require_cuda_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulation_module, "_resolve_fdtd_backend", lambda: (Backend, None))
    monkeypatch.setattr(mw.Simulation, "_resolve_fdtd_time_steps", lambda *args: 1)
    monkeypatch.setattr(core_path, "CorePathRuntime", lambda solver: object())
    monkeypatch.setattr(core_tables, "source_tables_supported", lambda solver: False)
    return checks


def _simulation(scene):
    # Device labels exercise admission; the backend fixture creates CPU tensors only.
    scene.device = "cuda"
    return mw.Simulation.fdtd(scene, frequency=1e9, fdtd_core_path=True, accelerated_setup=True)


def _count_walks(monkeypatch):
    roots, monitors = [], []
    walk = dispatch._walk_requires_grad

    def record(value, seen=None):
        if seen is None:
            roots.append(value)
        if isinstance(value, mw.PointMonitor):
            monitors.append(value)
        return walk(value, seen)

    monkeypatch.setattr(dispatch, "_walk_requires_grad", record)
    return roots, monitors


@pytest.mark.host_validation
@pytest.mark.parametrize("entry", ["prepare", "run"])
def test_one_scene_walk_per_preparation(monkeypatch, cpu_backend, entry):
    scene = _scene(16_384)
    simulation = _simulation(scene)
    roots, monitors = _count_walks(monkeypatch)

    def execute(self, solver):
        assert dispatch._requires_grad_memo.get() is None
        return solver

    monkeypatch.setattr(mw.Simulation, "_run_fdtd_from_solver", execute)
    getattr(simulation, entry)()
    assert [root for root in roots if isinstance(root, mw.Scene)] == [scene]
    assert len(monitors) == 16_384
    assert cpu_backend == [True, True]
    assert dispatch._requires_grad_memo.get() is None
    assert not dispatch.setup_enabled.get()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full preparation requires CUDA.")
def test_one_scene_walk_with_cuda_preparation(monkeypatch):
    scene = _scene(16_384).clone(device="cuda")
    simulation = _simulation(scene)
    roots, monitors = _count_walks(monkeypatch)
    prepared = simulation.prepare()
    assert hasattr(prepared.solver, "_core_runtime")
    assert [root for root in roots if isinstance(root, mw.Scene)] == [scene]
    assert len(monitors) == 16_384
    assert dispatch._requires_grad_memo.get() is None


@pytest.mark.host_validation
def test_flip_between_prepares(monkeypatch, cpu_backend):
    scene = _scene(32)
    scene.add_structure(mw.Structure(
        geometry=mw.Sphere(radius=.2, device="cpu"), material=mw.Material(eps_r=2.)))
    simulation = _simulation(scene)
    parameter = scene.structures[0].geometry.radius
    roots, _ = _count_walks(monkeypatch)
    for trainable in (False, True, False):
        parameter.requires_grad_(trainable)
        if trainable:
            with pytest.raises(ValueError, match=(
                    "^fdtd_core_path=True is blocked by: gradient-requiring scene tensors; "
                    "adjoint/gradient-requiring execution$")):
                simulation.prepare()
        else:
            assert hasattr(simulation.prepare().solver, "_core_runtime")
        assert dispatch._requires_grad_memo.get() is None
        token = dispatch.setup_enabled.set(True)
        try:
            with dispatch.requires_grad_scope():
                prepared = simulation_module.prepare_scene(scene)
                assert dispatch.eligible(prepared, 1., 1., prepared.subpixel) is (not trainable)
        finally:
            dispatch.setup_enabled.reset(token)
    assert sum(root is scene for root in roots) == 6


@pytest.mark.host_validation
@pytest.mark.parametrize("entry", ["prepare", "run"])
def test_scope_reset_after_backend_raises(monkeypatch, cpu_backend, entry):
    simulation = _simulation(_scene())

    def fail(*args):
        assert dispatch._requires_grad_memo.get() == {id(simulation.scene): (simulation.scene, False)}
        raise RuntimeError("forced preparation failure")

    monkeypatch.setattr(mw.Simulation, "_resolve_fdtd_time_steps", fail)
    with pytest.raises(RuntimeError, match="forced preparation failure"):
        getattr(simulation, entry)()
    assert dispatch._requires_grad_memo.get() is None
    assert not dispatch.setup_enabled.get()
    simulation.scene.metadata["parameter"].requires_grad_(True)
    assert dispatch.requires_grad(simulation.scene)


@pytest.mark.host_validation
def test_scope_identity_nesting_and_uncached_calls(monkeypatch):
    first, second = _scene(), _scene()
    roots, _ = _count_walks(monkeypatch)
    assert dispatch._requires_grad_memo.get() is None
    with dispatch.requires_grad_scope():
        assert not dispatch.requires_grad(first)
        assert not dispatch.requires_grad(first)
        assert not dispatch.requires_grad(second)
        outer = dispatch._requires_grad_memo.get()
        assert outer == {id(first): (first, False), id(second): (second, False)}
        first.metadata["parameter"].requires_grad_(True)
        with dispatch.requires_grad_scope():
            assert dispatch.requires_grad(first)
            assert dispatch._requires_grad_memo.get() == {id(first): (first, True)}
        assert dispatch._requires_grad_memo.get() is outer
        # A separate context cannot see the active scope.
        assert Context().run(dispatch._requires_grad_memo.get) is None
    assert dispatch._requires_grad_memo.get() is None
    assert dispatch.requires_grad(first)
    first.metadata["parameter"].requires_grad_(False)
    assert not dispatch.requires_grad(first)
    assert sum(root is first for root in roots) == 4
    assert sum(root is second for root in roots) == 1


@pytest.mark.host_validation
def test_memo_entry_pins_the_scene_for_the_scope():
    with dispatch.requires_grad_scope():
        scene = _scene()
        assert not dispatch.requires_grad(scene)
        alive = weakref.ref(scene)
        del scene
        gc.collect()
        # A dropped scene stays alive, so its id cannot be reused inside the scope.
        assert alive() is not None
    gc.collect()
    assert alive() is None


@pytest.mark.host_validation
def test_memo_identity_miss_walks_again():
    first, second = _scene(), _scene()
    second.metadata["parameter"].requires_grad_(True)
    with dispatch.requires_grad_scope():
        memo = dispatch._requires_grad_memo.get()
        # A stale entry under the same key for a different object is not reused.
        memo[id(second)] = (first, False)
        assert dispatch.requires_grad(second)
        assert memo[id(second)] == (second, True)


@pytest.mark.host_validation
def test_direct_solver_preparation_opens_its_own_scope(monkeypatch, cpu_backend):
    scene = _scene(32)
    simulation = _simulation(scene)
    roots, monitors = _count_walks(monkeypatch)
    assert hasattr(simulation._build_fdtd_solver(initialize=True), "_core_runtime")
    assert [root for root in roots if isinstance(root, mw.Scene)] == [scene]
    assert len(monitors) == 32
    assert dispatch._requires_grad_memo.get() is None


@pytest.mark.host_validation
@pytest.mark.parametrize("extra", ["eps", "mu", "subpixel", "prepared_state", "monitors"])
def test_wrapper_and_extra_arguments_keep_gradient_refusal(extra):
    scene = _scene(4)
    prepared = prepare_scene(scene)
    prepared.device = "cuda"
    parameter = torch.tensor(1., requires_grad=True)
    eps, mu, subpixel = 1., 1., prepared.subpixel
    with dispatch.requires_grad_scope():
        assert not dispatch.requires_grad(scene)
        # Check wrapper-only state even after an earlier eligibility check.
        token = dispatch.setup_enabled.set(True)
        try:
            assert dispatch.eligible(prepared, eps, mu, subpixel)
            if extra == "eps":
                eps = parameter
            elif extra == "mu":
                mu = parameter
            elif extra == "subpixel":
                subpixel = {"parameter": parameter}
            elif extra == "prepared_state":
                prepared._runtime_parameter = parameter
            else:
                prepared.monitors = [SimpleNamespace(parameter=parameter)]
            assert not dispatch.eligible(prepared, eps, mu, subpixel)
            if extra in ("prepared_state", "monitors"):
                assert "gradient-requiring scene tensors" in core_path.core_path_blockers(
                    SimpleNamespace(scene=prepared))
        finally:
            dispatch.setup_enabled.reset(token)


@pytest.mark.host_validation
def test_walker_cycles_shared_children_and_dict_keys():
    parameter = torch.tensor(1.)
    value = {"shared": [parameter, parameter]}
    value["cycle"] = value
    assert not dispatch.requires_grad(value)
    parameter.requires_grad_(True)
    assert dispatch.requires_grad(value)
    assert not dispatch.requires_grad({parameter: None})
    assert not dispatch.requires_grad(type("Declaration", (), {"parameter": parameter}))

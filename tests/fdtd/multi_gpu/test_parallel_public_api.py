from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.simulation as simulation_module
from witwin.maxwell.result import Result


def _scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec.none(),
        device="cuda:0",
    )


def _parallel(*, gather_fields: bool = False, result_device: str = "cuda:0"):
    return mw.FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"),
        gather_fields=gather_fields,
        result_device=result_device,
    )


def test_parallel_config_is_public_and_forwarded_by_fdtd_factory():
    parallel = _parallel()
    simulation = mw.Simulation.fdtd(_scene(), frequency=1.0e9, parallel=parallel)

    assert mw.FDTDParallelConfig is simulation_module.FDTDParallelConfig
    assert simulation.config.parallel is parallel


def test_parallel_none_keeps_single_gpu_backend_path(monkeypatch):
    scene = _scene()
    prepared_scene = object()
    calls = []

    class FakeFDTD:
        def __init__(self, prepared, **kwargs):
            calls.append((prepared, kwargs))
            self.initialized = False

        def init_field(self):
            self.initialized = True

    monkeypatch.setattr(simulation_module, "prepare_scene", lambda value: prepared_scene)
    monkeypatch.setattr(simulation_module, "_require_cuda_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulation_module, "_resolve_fdtd_backend", lambda: (FakeFDTD, object()))

    simulation = mw.Simulation.fdtd(scene, frequency=1.0e9)
    solver = simulation._build_fdtd_solver_for_scene(scene, initialize=True)

    assert solver.initialized is True
    assert calls == [
        (
            prepared_scene,
            {
                "frequency": 1.0e9,
                "absorber_type": mw.AbsorberKind.CPML,
                "cpml_config": {},
            },
        )
    ]


def test_parallel_builds_distributed_solver_and_initializes(monkeypatch):
    import witwin.maxwell.fdtd.distributed as distributed_module

    scene = _scene()
    parallel = _parallel()
    calls = []

    class FakeDistributedFDTD:
        def __init__(self, logical_scene, **kwargs):
            calls.append((logical_scene, kwargs))
            self.initialized = False

        def init_field(self):
            self.initialized = True

    monkeypatch.setattr(simulation_module, "prepare_scene", lambda value: value)
    monkeypatch.setattr(simulation_module, "_require_cuda_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed_module, "DistributedFDTD", FakeDistributedFDTD)

    simulation = mw.Simulation.fdtd(scene, frequency=2.0e9, parallel=parallel)
    solver = simulation._build_fdtd_solver_for_scene(scene, initialize=True)

    assert solver.initialized is True
    assert calls == [
        (
            scene,
            {
                "frequency": 2.0e9,
                "parallel": parallel,
                "absorber_type": mw.AbsorberKind.CPML,
                "cpml_config": {},
            },
        )
    ]


@pytest.mark.parametrize("entrypoint", ["prepare", "run"])
def test_trainable_parallel_is_rejected_before_solver_allocation(monkeypatch, entrypoint):
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequency=1.0e9,
        parallel=_parallel(),
    )
    simulation.has_trainable_parameters = True
    monkeypatch.setattr(simulation, "_refresh_scene", lambda: None)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("solver allocation must not be reached")

    monkeypatch.setattr(simulation, "_build_fdtd_solver", unexpected_allocation)

    with pytest.raises(ValueError, match="does not support trainable"):
        getattr(simulation, entrypoint)()


def _fake_solved_distributed_solver():
    return SimpleNamespace(
        scene=object(),
        dt=0.125,
        last_solve_elapsed_s=1.0,
        dft_sample_counts=(0,),
        observer_sample_counts=(0,),
        _shutoff_triggered=False,
        _shutoff_step=None,
        parallel_stats={"devices": ("cuda:0", "cuda:1"), "halo_bytes_total": 64},
    )


def test_gather_fields_false_does_not_fallback_to_shard_fields(monkeypatch):
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequency=1.0e9,
        run_time=mw.TimeConfig(time_steps=2),
        parallel=_parallel(gather_fields=False),
    )
    solver = _fake_solved_distributed_solver()
    monkeypatch.setattr(
        simulation,
        "_execute_fdtd_solve",
        lambda *args: (None, 2, False, simulation.config.spectral_sampler),
    )
    monkeypatch.setattr(
        simulation,
        "_fdtd_last_step_field_payload",
        lambda *args: (_ for _ in ()).throw(AssertionError("field fallback must not run")),
    )

    result = simulation._run_fdtd_from_solver(solver)

    assert result.fields == {}
    assert result.raw_output == {}
    assert result.solver_stats["parallel_stats"]["halo_bytes_total"] == 64


def test_gather_fields_true_requires_distributed_field_output(monkeypatch):
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequency=1.0e9,
        parallel=_parallel(gather_fields=True),
    )
    solver = _fake_solved_distributed_solver()
    monkeypatch.setattr(
        simulation,
        "_execute_fdtd_solve",
        lambda *args: (None, 1, False, simulation.config.spectral_sampler),
    )

    with pytest.raises(RuntimeError, match="did not return any output"):
        simulation._run_fdtd_from_solver(solver)


def test_gathered_fields_are_normalized_on_explicit_result_device(monkeypatch):
    parallel = _parallel(gather_fields=True, result_device="cuda:1")
    simulation = mw.Simulation.fdtd(_scene(), frequency=1.0e9, parallel=parallel)
    solver = _fake_solved_distributed_solver()
    raw_fields = {"Ex": torch.tensor([1.0]), "Ey": torch.tensor([2.0]), "Ez": torch.tensor([3.0])}
    requested_devices = []

    monkeypatch.setattr(
        simulation,
        "_execute_fdtd_solve",
        lambda *args: (raw_fields, 1, False, simulation.config.spectral_sampler),
    )

    def capture_device(fields, device):
        requested_devices.append(str(device))
        return {name.upper(): value for name, value in fields.items()}

    monkeypatch.setattr(simulation_module, "_to_tensor_fields", capture_device)

    result = simulation._run_fdtd_from_solver(solver)

    assert requested_devices == ["cuda:1"]
    assert set(result.fields) == {"EX", "EY", "EZ"}


def test_result_solver_stats_is_a_read_only_copy_property():
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequency=1.0e9,
        solver_stats={"steps_run": 3},
    )

    stats = result.solver_stats
    stats["steps_run"] = 99

    assert result.solver_stats == {"steps_run": 3}
    with pytest.raises(AttributeError):
        result.solver_stats = {}

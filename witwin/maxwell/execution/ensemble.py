from __future__ import annotations

import copy
from dataclasses import dataclass

from ..fdtd_parallel import _normalize_devices
from ..scene import Scene
from .capacity import estimate_simulation_footprint_bytes
from .executor import execute_plan
from .plan import ExecutionPlan, ExecutionTask
from .pool import DevicePool
from .records import ResultSequence


@dataclass(frozen=True)
class MultiGPUExecution:
    """Public execution descriptor for a horizontal ensemble of Simulations.

    This type performs structural validation only. CUDA availability and per
    device capacity are prepare/run-time checks so the descriptor stays usable in
    CPU-only API tests, mirroring :class:`FDTDParallelConfig`.
    """

    devices: tuple[str, ...]
    max_concurrency: int
    placement: str = "memory_aware"
    fail_fast: bool = False
    per_device_concurrency: int = 1

    def __post_init__(self) -> None:
        devices = _normalize_devices(self.devices, minimum=1)
        object.__setattr__(self, "devices", devices)
        if self.placement not in {"round_robin", "memory_aware"}:
            raise ValueError("placement must be 'round_robin' or 'memory_aware'.")
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int):
            raise TypeError("max_concurrency must be an integer.")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0.")
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a bool.")
        if isinstance(self.per_device_concurrency, bool) or not isinstance(
            self.per_device_concurrency, int
        ):
            raise TypeError("per_device_concurrency must be an integer.")
        if self.per_device_concurrency <= 0:
            raise ValueError("per_device_concurrency must be > 0.")

    @classmethod
    def ensemble(
        cls,
        *,
        devices,
        max_concurrency: int | None = None,
        placement: str = "memory_aware",
        fail_fast: bool = False,
        per_device_concurrency: int = 1,
    ) -> "MultiGPUExecution":
        normalized = _normalize_devices(devices, minimum=1)
        if max_concurrency is None:
            max_concurrency = len(normalized) * per_device_concurrency
        return cls(
            devices=normalized,
            max_concurrency=max_concurrency,
            placement=placement,
            fail_fast=fail_fast,
            per_device_concurrency=per_device_concurrency,
        )

    def build_pool(self, *, require_cuda: bool = True) -> DevicePool:
        if require_cuda:
            return DevicePool.discover(
                self.devices, per_device_concurrency=self.per_device_concurrency
            )
        return DevicePool(
            self.devices,
            per_device_concurrency=self.per_device_concurrency,
            require_cuda=False,
        )


def _run_simulation_on_device(simulation, device):
    """Bind a Simulation's declarative Scene to ``device`` and run it.

    The Scene is declarative and owns device placement, so prepare/run happen in
    the leased device context rather than materializing a full PreparedScene on a
    coordinator GPU. A per-run shallow copy of the Scene is pinned to the leased
    device so the returned ``Result.scene.device`` matches the device its tensors
    live on, while the caller's original Scene object is never mutated (device is
    a declarative attribute resolved at prepare time, so the copy is cheap).
    """

    original_input = simulation.scene_input
    column_scene = copy.copy(original_input)
    column_scene.device = str(device)
    simulation.scene_input = column_scene
    try:
        return simulation.run()
    finally:
        simulation.scene_input = original_input


def _reject_ensemble_incompatible(simulation, index: int) -> None:
    from ..simulation import Simulation

    if not isinstance(simulation, Simulation):
        raise TypeError(f"run_many item {index} must be a maxwell.Simulation.")
    if not isinstance(simulation.scene_input, Scene):
        raise ValueError(
            f"run_many item {index} uses a SceneModule input; ensemble execution "
            "requires a declarative Scene. Materialize it with to_scene() first."
        )
    simulation._refresh_scene()
    if simulation.has_trainable_parameters:
        raise ValueError(
            f"run_many item {index} has trainable parameters; ensemble execution does "
            "not run the adjoint through run_many. Call Simulation.run() per task for "
            "an independent backward, or omit the trainable parameters."
        )
    if getattr(simulation.config, "parallel", None) is not None:
        raise ValueError(
            f"run_many item {index} carries an FDTDParallelConfig (joint-solve); a single "
            "Simulation cannot both be ensemble-distributed and spatially decomposed. Use "
            "ensemble execution for independent Simulations only."
        )


def build_ensemble_plan(simulations, execution: MultiGPUExecution) -> ExecutionPlan:
    """Validate simulations and materialize an immutable ensemble ExecutionPlan."""

    if not isinstance(execution, MultiGPUExecution):
        raise TypeError("execution must be a maxwell.MultiGPUExecution.")
    resolved = list(simulations)
    seen_scene_ids: dict[int, int] = {}
    tasks = []
    for index, simulation in enumerate(resolved):
        _reject_ensemble_incompatible(simulation, index)
        scene_id = id(simulation.scene_input)
        if scene_id in seen_scene_ids:
            raise ValueError(
                f"run_many items {seen_scene_ids[scene_id]} and {index} share the same "
                "Scene object; ensemble tasks must own independent Scenes to avoid "
                "cross-task state bleed."
            )
        seen_scene_ids[scene_id] = index
        estimated = (
            estimate_simulation_footprint_bytes(simulation)
            if execution.placement == "memory_aware"
            else None
        )
        tasks.append(
            ExecutionTask(
                index=index,
                run=lambda device, sim=simulation: _run_simulation_on_device(sim, device),
                estimated_bytes=estimated,
                label=f"simulation[{index}]",
            )
        )
    return ExecutionPlan(
        tasks=tuple(tasks),
        placement=execution.placement,
        max_concurrency=execution.max_concurrency,
        fail_fast=execution.fail_fast,
    )


def run_many(simulations, *, execution: MultiGPUExecution) -> ResultSequence:
    """Run a horizontal ensemble of independent Simulations over multiple GPUs.

    Input and output are strictly ordered: ``results[i]`` is the outcome of
    ``simulations[i]``. Each Simulation still produces its own ``Result``; the
    executor only guarantees ordered execution, deterministic device leasing,
    structured failure aggregation and a submission-ordered ``ResultSequence``.
    No electromagnetic semantics live in the scheduler.
    """

    if not isinstance(execution, MultiGPUExecution):
        raise TypeError("execution must be a maxwell.MultiGPUExecution.")
    plan = build_ensemble_plan(simulations, execution)
    pool = execution.build_pool(require_cuda=True)
    return execute_plan(plan, pool)


__all__ = [
    "MultiGPUExecution",
    "build_ensemble_plan",
    "run_many",
]

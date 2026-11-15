"""Shared multi-GPU ensemble execution layer.

Public namespace ``mw.execution``: :class:`MultiGPUExecution`, :func:`run_many`, the
result/error containers (:class:`ResultSequence`, :class:`ExecutionRecord`,
:class:`DistributedFailure`) and the batch tier (:class:`BatchSimulation` and its inputs,
plan report and results). ``DevicePool``, ``ExecutionPlan``, ``ExecutionTask``
and ``execute_plan`` are internal building blocks that know nothing about Yee
fields, ports or PDE stencils; they only order tasks over leased devices.
"""

from .batch import BatchExecution, BatchOutput, BatchParameters, BatchSimulation, BatchVariation
from .batch_runtime import BatchResultIterator, PreparedBatch
from .capacity import (
    estimate_scene_footprint_bytes,
    estimate_simulation_footprint_bytes,
)
from .ensemble import (
    MultiGPUExecution,
    build_ensemble_plan,
    run_many,
)
from .executor import execute_plan
from .plan import BatchPlanReport, ExecutionPlan, ExecutionTask
from .pool import DeviceCapability, DeviceLease, DevicePool
from .records import (
    BatchItem,
    BatchResult,
    DistributedFailure,
    ExecutionRecord,
    FailureKind,
    ResultSequence,
)



def save_sharded(result, directory):
    """Persist the owned field shards of a distributed FDTD ``result`` without assembling global fields.

    Refuses an existing ``directory`` (no non-atomic overwrite). The Result metadata uses
    the same snapshot schema as ``Result.save``.
    """
    return result._save_sharded(directory)


def load_sharded(directory, *, scene, prepared_scene=None, gather_fields: bool = False, map_location="cpu"):
    """Load a ``save_sharded`` directory as a Result: metadata eagerly, owned fields only with
    ``gather_fields=True``."""
    from ..result import Result

    return Result._load_sharded(directory, scene=scene, prepared_scene=prepared_scene,
                                gather_fields=gather_fields, map_location=map_location)


def is_sharded(result) -> bool:
    """Whether ``result`` was loaded by :func:`load_sharded`."""
    return result._sharded_manifest is not None


def sharded_manifest(result):
    """The shard manifest of a :func:`load_sharded` ``result`` (``None`` otherwise)."""
    return result._sharded_manifest


def shard_paths(result):
    """The shard files of a :func:`load_sharded` ``result`` (empty otherwise)."""
    return result._shard_paths


__all__ = [
    "BatchExecution",
    "BatchItem",
    "BatchOutput",
    "BatchParameters",
    "BatchPlanReport",
    "BatchResult",
    "BatchResultIterator",
    "BatchSimulation",
    "BatchVariation",
    "DistributedFailure",
    "ExecutionRecord",
    "FailureKind",
    "MultiGPUExecution",
    "PreparedBatch",
    "ResultSequence",
    "is_sharded",
    "load_sharded",
    "run_many",
    "save_sharded",
    "shard_paths",
    "sharded_manifest",
]

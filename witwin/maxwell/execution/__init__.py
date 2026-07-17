"""Shared multi-GPU ensemble execution layer.

The public surface is :class:`MultiGPUExecution`, :func:`run_many`, and the
result/error containers (:class:`ResultSequence`, :class:`ExecutionRecord`,
:class:`DistributedFailure`). ``DevicePool``, ``ExecutionPlan``, ``ExecutionTask``
and ``execute_plan`` are internal building blocks that know nothing about Yee
fields, ports or PDE stencils; they only order tasks over leased devices.
"""

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
from .plan import ExecutionPlan, ExecutionTask
from .pool import DeviceCapability, DeviceLease, DevicePool
from .records import (
    DistributedFailure,
    ExecutionRecord,
    FailureKind,
    ResultSequence,
)

__all__ = [
    "DeviceCapability",
    "DeviceLease",
    "DevicePool",
    "DistributedFailure",
    "ExecutionPlan",
    "ExecutionRecord",
    "ExecutionTask",
    "FailureKind",
    "MultiGPUExecution",
    "ResultSequence",
    "build_ensemble_plan",
    "estimate_scene_footprint_bytes",
    "estimate_simulation_footprint_bytes",
    "execute_plan",
    "run_many",
]

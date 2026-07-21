"""Ensemble 2-GPU aggregation of the array scene-gradient VJP (plan 06 Phase 4).

The single-device aggregator :meth:`ArrayBasisData.scene_gradient_vjp` seeds each
embedded-pattern column with ``conj(w_n) * cot_E`` and sums the per-column
vector-Jacobian products onto the scene parameters. This module distributes that
work: the per-column forwards run as *independent* Simulations over the ensemble
device pool (the safe path -- the NCCL joint-solve adjoint stays out of scope),
their live embedded-pattern columns are gathered onto one reduction device, and
the per-column VJPs are summed in a fixed public-port reduction order so the
result is invariant to which GPU each column ran on.

Two coupled facts make the split legitimate:

- The seed for column ``n`` is ``conj(w_n) * cot_E`` where ``cot_E = dL/dE`` is a
  function of the combined field value ``E = sum_n w_n e_n`` only. Forming ``E``
  needs the column *values* (detached copies suffice); the combined-field
  cotangent computed from a detached ``E`` leaf is bit-identical to the one the
  single-device path takes from the live combine, because the objective node is
  the same and is evaluated at the same ``E``.
- The per-column VJP ``VJP_n(seed_n)`` is a self-contained backward through
  column ``n``'s own graph on its own device; cross-device it only needs the seed
  moved onto that device, and its gradient moved back to the reduction device.

The reduction is therefore deterministic: on homogeneous GPUs the per-column
gradients are bit-identical to their single-GPU values and a cross-device copy is
exact, so summing them in a fixed order reproduces the single-device gradient.
Zero-extra-FDTD-forward-steps is a forward-combine contract only (Phase 1);
gradients legitimately re-run the per-column forwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .array import ArrayBasisData, _single_beam_weights
from .execution import MultiGPUExecution
from .execution.plan import ExecutionPlan, ExecutionTask
from .execution.records import DistributedFailure


@dataclass(frozen=True)
class AggregatedSceneGradient:
    """Reduced scene-parameter gradient plus the aggregation provenance.

    ``gradient`` mirrors the per-column parameter structure (a bare tensor when
    each column differentiates a single leaf, otherwise a tuple) and lives on
    ``reduction_device``. The provenance fields make the deterministic reduction
    auditable: ``reduction_order`` is the fixed public-port summation order,
    ``column_devices[n]`` is the device column ``n`` ran on, and ``port_names`` is
    the basis port order the columns follow.
    """

    gradient: Any
    reduction_order: tuple[int, ...]
    reduction_device: str
    column_devices: tuple[str, ...]
    port_names: tuple[str, ...]


def _validate_column_parameters(
    parameters: Sequence[Any], *, port_count: int
) -> tuple[list[tuple[torch.Tensor, ...]], bool]:
    """Normalize per-column parameters to a list of leaf tuples.

    Every column must expose the same parameter structure (leaf count, shapes,
    dtypes) so the per-column gradients can be summed slot by slot. Devices may
    differ (that is the whole point of the 2-GPU replica case).
    """

    per_column = list(parameters)
    if len(per_column) != port_count:
        raise ValueError(
            f"parameters must contain one entry per port (N={port_count}); "
            f"got {len(per_column)}."
        )
    single_leaf = isinstance(per_column[0], torch.Tensor)
    normalized: list[tuple[torch.Tensor, ...]] = []
    template: tuple[tuple[torch.Size, torch.dtype], ...] | None = None
    for index, entry in enumerate(per_column):
        if isinstance(entry, torch.Tensor):
            leaves = (entry,)
            entry_single = True
        else:
            leaves = tuple(entry)
            entry_single = False
        if entry_single != single_leaf:
            raise ValueError(
                "parameters must use a consistent structure across columns: either a "
                "bare tensor per column or a tuple of tensors per column."
            )
        for leaf in leaves:
            if not isinstance(leaf, torch.Tensor):
                raise TypeError("parameters must be torch.Tensor instances.")
        signature = tuple((leaf.shape, leaf.dtype) for leaf in leaves)
        if template is None:
            template = signature
        elif signature != template:
            raise ValueError(
                f"parameters[{index}] has shape/dtype {signature} but the reduction "
                f"requires every column to share {template}."
            )
        normalized.append(leaves)
    return normalized, single_leaf


def _resolve_reduction_order(
    reduction_order: Sequence[int] | None, *, port_count: int
) -> tuple[int, ...]:
    if reduction_order is None:
        return tuple(range(port_count))
    order = tuple(int(index) for index in reduction_order)
    if sorted(order) != list(range(port_count)):
        raise ValueError("reduction_order must be a permutation of range(N).")
    return order


def aggregate_scene_gradient_vjp(
    basis: ArrayBasisData,
    *,
    columns: Sequence[tuple[torch.Tensor, torch.Tensor]],
    parameters: Sequence[Any],
    weights: Any,
    objective: Callable | None = None,
    field_cotangents: tuple[torch.Tensor, torch.Tensor] | None = None,
    reduction_order: Sequence[int] | None = None,
    reduction_device: torch.device | str | None = None,
) -> AggregatedSceneGradient:
    """Reduce per-column scene-gradient VJPs across (possibly) many devices.

    ``columns[n]`` are the live ``(e_theta_n, e_phi_n)`` embedded far-field columns
    from column ``n``'s re-run forward, on that column's device; ``parameters[n]``
    are the trainable leaf(s) column ``n``'s graph differentiates (the same object
    for every column in the single-device case, a per-device replica in the 2-GPU
    case). The combined field and its cotangent are formed on ``reduction_device``
    (default: the basis device); each column is seeded there and back-propagated on
    its own device, and the gradients are summed in ``reduction_order`` (default:
    public port order) on ``reduction_device``.
    """

    if (objective is None) == (field_cotangents is None):
        raise ValueError(
            "Provide exactly one of objective (a callable on the combined field) or "
            "field_cotangents (pre-computed combined-field cotangents)."
        )
    patterns = basis.embedded_patterns
    dtype = basis.dtype
    frequency_count, port_count, _ = basis.network.s.shape
    angular_shape = tuple(patterns.theta.shape)
    expected_shape = (frequency_count, *angular_shape)

    column_list = list(columns)
    if len(column_list) != port_count:
        raise ValueError(
            f"columns must contain one (e_theta, e_phi) pair per port (N={port_count}); "
            f"got {len(column_list)}."
        )
    reduction = torch.device(reduction_device) if reduction_device is not None else basis.device

    e_theta_cols: list[torch.Tensor] = []
    e_phi_cols: list[torch.Tensor] = []
    column_devices: list[str] = []
    for index, column in enumerate(column_list):
        if not isinstance(column, (tuple, list)) or len(column) != 2:
            raise TypeError(f"columns[{index}] must be an (e_theta, e_phi) pair.")
        e_theta_n, e_phi_n = column
        for name, tensor in (("e_theta", e_theta_n), ("e_phi", e_phi_n)):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_complex():
                raise TypeError(f"columns[{index}] {name} must be a complex torch.Tensor.")
            if tensor.dtype != dtype:
                raise TypeError(f"columns[{index}] {name} must use dtype {dtype}.")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"columns[{index}] {name} must have shape [F, T, P] = {expected_shape}."
                )
        if e_theta_n.device != e_phi_n.device:
            raise ValueError(f"columns[{index}] e_theta and e_phi must share a device.")
        if not (e_theta_n.requires_grad or e_phi_n.requires_grad):
            raise ValueError(
                f"columns[{index}] carries no autograd graph. The array basis stores "
                "detached patterns, so the per-column forwards must be re-run under "
                "autograd before aggregating."
            )
        e_theta_cols.append(e_theta_n)
        e_phi_cols.append(e_phi_n)
        column_devices.append(str(e_theta_n.device))

    parameter_columns, single_leaf = _validate_column_parameters(
        parameters, port_count=port_count
    )

    weight_matrix = _single_beam_weights(
        weights,
        frequency_count=frequency_count,
        port_count=port_count,
        device=reduction,
        dtype=dtype,
    )

    order = _resolve_reduction_order(reduction_order, port_count=port_count)

    # Combined-field cotangent on the reduction device. Detached column values are
    # sufficient to form E; the cotangent is a function of E's value only.
    if objective is not None:
        column_weight = weight_matrix[:, :, None, None]
        e_theta_combined = sum(
            column_weight[:, index] * e_theta_cols[index].detach().to(reduction)
            for index in range(port_count)
        )
        e_phi_combined = sum(
            column_weight[:, index] * e_phi_cols[index].detach().to(reduction)
            for index in range(port_count)
        )
        e_theta_combined = e_theta_combined.detach().requires_grad_(True)
        e_phi_combined = e_phi_combined.detach().requires_grad_(True)
        loss = objective(e_theta_combined, e_phi_combined)
        if not isinstance(loss, torch.Tensor) or loss.is_complex() or loss.ndim != 0:
            raise TypeError("objective must return a real scalar torch.Tensor.")
        if not loss.requires_grad:
            raise ValueError(
                "objective(combined field) does not depend on the supplied columns; "
                "re-run the per-column forwards under autograd before aggregating "
                "(the retained basis stores detached patterns)."
            )
        cot_theta, cot_phi = torch.autograd.grad(
            loss, (e_theta_combined, e_phi_combined), allow_unused=True
        )
        if cot_theta is None:
            cot_theta = torch.zeros_like(e_theta_combined)
        if cot_phi is None:
            cot_phi = torch.zeros_like(e_phi_combined)
    else:
        cot_theta, cot_phi = field_cotangents
        for name, tensor in (("cot_E_theta", cot_theta), ("cot_E_phi", cot_phi)):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_complex():
                raise TypeError(f"field_cotangents {name} must be a complex torch.Tensor.")
            if tensor.dtype != dtype:
                raise TypeError(f"field_cotangents {name} must use dtype {dtype}.")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"field_cotangents {name} must have shape [F, T, P] = {expected_shape}."
                )
        cot_theta = cot_theta.to(reduction)
        cot_phi = cot_phi.to(reduction)

    conjugate_weights = torch.conj(weight_matrix)  # [F, N] on the reduction device

    leaf_count = len(parameter_columns[0])
    totals = [
        torch.zeros_like(parameter_columns[0][slot], device=reduction)
        for slot in range(leaf_count)
    ]
    seen_contribution = False
    for index in order:
        device = e_theta_cols[index].device
        seed_scale = conjugate_weights[:, index, None, None].to(device)
        seed_theta = seed_scale * cot_theta.to(device)
        seed_phi = seed_scale * cot_phi.to(device)
        column_grads = torch.autograd.grad(
            (e_theta_cols[index], e_phi_cols[index]),
            parameter_columns[index],
            grad_outputs=(seed_theta, seed_phi),
            allow_unused=True,
        )
        for slot, gradient in enumerate(column_grads):
            if gradient is not None:
                totals[slot] = totals[slot] + gradient.to(reduction)
                seen_contribution = True

    if not seen_contribution:
        raise ValueError(
            "None of the embedded-pattern columns are differentiable w.r.t. the supplied "
            "parameters. The array basis stores detached patterns, so the per-column "
            "forwards must be re-run under autograd before aggregating."
        )

    gradient = totals[0] if single_leaf else tuple(totals)
    return AggregatedSceneGradient(
        gradient=gradient,
        reduction_order=order,
        reduction_device=str(reduction),
        column_devices=tuple(column_devices),
        port_names=basis.port_names,
    )


def ensemble_scene_gradient_vjp(
    basis: ArrayBasisData,
    *,
    column_forward: Callable[[int, torch.device], tuple[torch.Tensor, torch.Tensor, Any]],
    weights: Any,
    execution: MultiGPUExecution,
    objective: Callable | None = None,
    field_cotangents: tuple[torch.Tensor, torch.Tensor] | None = None,
    reduction_order: Sequence[int] | None = None,
    reduction_device: torch.device | str | None = None,
) -> AggregatedSceneGradient:
    """Distribute the per-column forwards over the pool, then aggregate the VJP.

    ``column_forward(index, device)`` runs column ``index``'s forward on ``device``
    under autograd and returns ``(e_theta_n, e_phi_n, parameters_n)`` -- the live
    embedded far-field column and the trainable leaf(s) its graph differentiates
    (a per-device replica of the shared design). The forwards run as independent
    ensemble tasks over ``execution``'s device pool; their live columns are then
    gathered and reduced by :func:`aggregate_scene_gradient_vjp` in a fixed public
    port order on ``reduction_device`` (default: the basis device).

    The per-column adjoint is intentionally not routed through ``run_many``: that
    path refuses trainable simulations because it does not run a backward. Here the
    pool only places and orders the forwards; the seeded backward and deterministic
    reduction happen on the caller thread after every device is synchronized.
    """

    if not isinstance(execution, MultiGPUExecution):
        raise TypeError("execution must be a maxwell.MultiGPUExecution.")
    _, port_count, _ = basis.network.s.shape

    tasks = tuple(
        ExecutionTask(
            index=index,
            run=lambda device, i=index: column_forward(i, device),
            estimated_bytes=None,
            label=f"array-column[{index}]",
        )
        for index in range(port_count)
    )
    plan = ExecutionPlan(
        tasks=tasks,
        placement="round_robin",
        max_concurrency=execution.max_concurrency,
        fail_fast=True,
    )
    pool = execution.build_pool(require_cuda=True)
    from .execution.executor import execute_plan

    sequence = execute_plan(plan, pool)
    if sequence.failed:
        failure = next(
            entry for entry in sequence.entries if isinstance(entry, DistributedFailure)
        )
        raise RuntimeError(
            f"array column forward {failure.index} failed during ensemble distribution: "
            f"{failure.detail}"
        ) from (failure.exception or failure)

    columns: list[tuple[torch.Tensor, torch.Tensor]] = []
    parameters: list[Any] = []
    for index in range(port_count):
        e_theta_n, e_phi_n, parameters_n = sequence[index]
        columns.append((e_theta_n, e_phi_n))
        parameters.append(parameters_n)
        # Ensure the worker-thread forward is complete before the caller-thread
        # backward reads its saved tensors.
        if e_theta_n.device.type == "cuda":
            torch.cuda.synchronize(e_theta_n.device)

    return aggregate_scene_gradient_vjp(
        basis,
        columns=columns,
        parameters=parameters,
        weights=weights,
        objective=objective,
        field_cotangents=field_cotangents,
        reduction_order=reduction_order,
        reduction_device=reduction_device,
    )


__all__ = [
    "AggregatedSceneGradient",
    "aggregate_scene_gradient_vjp",
    "ensemble_scene_gradient_vjp",
]

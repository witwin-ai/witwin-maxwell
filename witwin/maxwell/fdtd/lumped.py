from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LumpedRuntime:
    """Prepared single-device state for one lumped field-circuit coupling."""

    port_name: str
    topology: str
    field_shape: tuple[int, int, int]
    field_dtype: torch.dtype
    linear_indices: torch.Tensor
    voltage_weights: torch.Tensor
    injection: torch.Tensor
    dt: torch.Tensor
    resistance: torch.Tensor
    inductance: torch.Tensor
    capacitance: torch.Tensor
    inductance_active: bool
    capacitance_active: bool
    resistance_is_open: torch.Tensor
    inductance_enabled: torch.Tensor
    capacitance_enabled: torch.Tensor
    default_thevenin_voltage: torch.Tensor
    coupling_impedance: torch.Tensor
    discrete_port_impedance: torch.Tensor
    series_inductive_impedance: torch.Tensor
    series_capacitive_impedance: torch.Tensor
    parallel_resistive_admittance: torch.Tensor
    parallel_inductive_admittance: torch.Tensor
    parallel_capacitive_admittance: torch.Tensor
    parallel_total_admittance: torch.Tensor
    inverse_denominator: torch.Tensor
    edge_buffer: torch.Tensor
    correction_buffer: torch.Tensor
    inductor_current: torch.Tensor
    capacitor_voltage: torch.Tensor
    last_voltage_before: torch.Tensor
    last_voltage_midpoint: torch.Tensor
    last_voltage_after: torch.Tensor
    last_model_voltage_midpoint: torch.Tensor
    last_branch_current: torch.Tensor
    last_resistor_current: torch.Tensor
    last_capacitor_current: torch.Tensor
    last_inductor_current_midpoint: torch.Tensor
    last_dissipated_energy: torch.Tensor
    last_stored_energy_change: torch.Tensor
    last_source_work: torch.Tensor
    last_field_energy_change: torch.Tensor


@dataclass(frozen=True)
class LumpedStepTrace:
    """Scalar intermediates required by the exact local circuit pullback."""

    runtime: LumpedRuntime
    field_name: str
    kind: str
    index: int
    previous_voltage: torch.Tensor
    voltage_before: torch.Tensor
    branch_current: torch.Tensor
    voltage_midpoint: torch.Tensor
    drive: torch.Tensor
    old_inductor_current: torch.Tensor
    old_capacitor_voltage: torch.Tensor


@dataclass(frozen=True)
class LumpedStepPullback:
    field_adjoint: torch.Tensor
    previous_voltage_adjoint: torch.Tensor
    inductor_current_adjoint: torch.Tensor
    capacitor_voltage_adjoint: torch.Tensor
    grad_eps: torch.Tensor
    grad_resistance: torch.Tensor
    grad_inductance: torch.Tensor
    grad_capacitance: torch.Tensor
    grad_drive: torch.Tensor


def _prepared_scalar(
    value: torch.Tensor | float,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return tensor


def _linearize_indices(indices: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.shape[0] == 0:
        raise ValueError("voltage_indices must have non-empty shape [K, 3].")
    stride_x = shape[1] * shape[2]
    stride_y = shape[2]
    return indices[:, 0] * stride_x + indices[:, 1] * stride_y + indices[:, 2]


def _local_control_volume(
    yee_control_volume: torch.Tensor | float,
    *,
    shape: tuple[int, int, int],
    linear_indices: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    volume = torch.as_tensor(yee_control_volume, device=device, dtype=dtype)
    if volume.ndim == 0:
        return volume
    if tuple(volume.shape) != shape:
        raise ValueError("yee_control_volume must be scalar or match eps_edge shape.")
    return torch.index_select(volume.reshape(-1), 0, linear_indices)


def _termination_components(
    termination: Any | None,
    resistance: torch.Tensor | float | None,
) -> tuple[str, Any, Any, Any]:
    if termination is None:
        if resistance is None:
            raise ValueError("resistance or termination must be provided.")
        return "series", resistance, None, None
    if resistance is not None:
        raise ValueError("Specify either resistance or termination, not both.")

    kind = getattr(termination, "kind", None)
    if kind == "resistor":
        return "series", termination.resistance, None, None
    if kind == "capacitor":
        return "series", None, None, termination.capacitance
    if kind == "inductor":
        return "series", None, termination.inductance, None
    if kind == "series_rlc":
        return "series", termination.r, termination.l, termination.c
    if kind == "parallel_rlc":
        return "parallel", termination.r, termination.l, termination.c
    raise TypeError("termination must be a resistor, capacitor, inductor, SeriesRLC, or ParallelRLC.")


def _optional_component(
    value: torch.Tensor | float | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value is None:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, torch.zeros((), device=device, dtype=torch.bool)
    tensor = _prepared_scalar(value, name="RLC component", device=device, dtype=dtype)
    return tensor, tensor > 0.0


def prepare_lumped_runtime(
    geometry: Any,
    *,
    dt: torch.Tensor | float,
    eps_edge: torch.Tensor,
    yee_control_volume: torch.Tensor | float,
    resistance: torch.Tensor | float | None = None,
    termination: Any | None = None,
    thevenin_voltage: torch.Tensor | float = 0.0,
) -> LumpedRuntime:
    """Prepare ``q = dt * C_e^-1 w`` and all per-step work buffers.

    A direct ``resistance`` represents a passive termination or the series
    resistance of a Thevenin excitation. ``termination`` accepts the public
    R/C/L, SeriesRLC, or ParallelRLC descriptors. ``thevenin_voltage`` is the
    prepared default drive value; a device scalar may override it per update.
    """

    if not isinstance(eps_edge, torch.Tensor):
        raise TypeError("eps_edge must be a torch.Tensor.")
    if eps_edge.ndim != 3:
        raise ValueError("eps_edge must be a three-dimensional Yee-edge tensor.")
    if eps_edge.dtype not in (torch.float32, torch.float64):
        raise TypeError("eps_edge must use torch.float32 or torch.float64.")
    if not eps_edge.is_contiguous():
        raise ValueError("eps_edge must be contiguous.")

    shape = tuple(eps_edge.shape)
    device = eps_edge.device
    dtype = eps_edge.dtype
    indices = torch.as_tensor(
        geometry.voltage_indices,
        device=device,
        dtype=torch.int64,
    )
    linear_indices = _linearize_indices(indices, shape)
    weights = torch.as_tensor(
        geometry.voltage_weights,
        device=device,
        dtype=dtype,
    )
    if weights.ndim != 1 or weights.shape[0] != linear_indices.shape[0]:
        raise ValueError("voltage_weights must have shape [K] matching voltage_indices.")

    local_eps = torch.index_select(eps_edge.reshape(-1), 0, linear_indices)
    local_volume = _local_control_volume(
        yee_control_volume,
        shape=shape,
        linear_indices=linear_indices,
        device=device,
        dtype=dtype,
    )
    dt_tensor = _prepared_scalar(dt, name="dt", device=device, dtype=dtype)
    topology, resistance_value, inductance_value, capacitance_value = _termination_components(
        termination,
        resistance,
    )
    resistance_tensor, resistance_enabled = _optional_component(
        resistance_value,
        device=device,
        dtype=dtype,
    )
    inductance_tensor, inductance_enabled = _optional_component(
        inductance_value,
        device=device,
        dtype=dtype,
    )
    capacitance_tensor, capacitance_enabled = _optional_component(
        capacitance_value,
        device=device,
        dtype=dtype,
    )
    default_drive = _prepared_scalar(
        thevenin_voltage,
        name="thevenin_voltage",
        device=device,
        dtype=dtype,
    )

    edge_capacitance = local_eps * local_volume
    injection = dt_tensor * weights / edge_capacitance
    coupling_impedance = torch.dot(weights, injection)
    discrete_port_impedance = 0.5 * coupling_impedance
    scalar_zero = torch.zeros((), device=device, dtype=dtype)
    scalar_one = torch.ones((), device=device, dtype=dtype)
    safe_capacitance = torch.where(capacitance_enabled, capacitance_tensor, scalar_one)
    safe_inductance = torch.where(inductance_enabled, inductance_tensor, scalar_one)
    series_inductive_impedance = 2.0 * inductance_tensor / dt_tensor
    series_capacitive_impedance = torch.where(
        capacitance_enabled,
        dt_tensor / (2.0 * safe_capacitance),
        scalar_zero,
    )
    parallel_resistive_admittance = torch.where(
        resistance_enabled,
        torch.reciprocal(resistance_tensor),
        scalar_zero,
    )
    parallel_inductive_admittance = torch.where(
        inductance_enabled,
        dt_tensor / (2.0 * safe_inductance),
        scalar_zero,
    )
    parallel_capacitive_admittance = torch.where(
        capacitance_enabled,
        2.0 * capacitance_tensor / dt_tensor,
        scalar_zero,
    )
    parallel_total_admittance = (
        parallel_resistive_admittance
        + parallel_inductive_admittance
        + parallel_capacitive_admittance
    )
    if topology == "series":
        denominator = (
            resistance_tensor
            + series_inductive_impedance
            + series_capacitive_impedance
            + discrete_port_impedance
        )
    else:
        denominator = scalar_one + parallel_total_admittance * discrete_port_impedance
    inverse_denominator = torch.reciprocal(denominator)
    edge_buffer = torch.empty_like(weights)
    correction_buffer = torch.empty_like(weights)
    scalar_zeros = [torch.zeros((), device=device, dtype=dtype) for _ in range(14)]

    return LumpedRuntime(
        port_name=str(geometry.port_name),
        topology=topology,
        field_shape=shape,
        field_dtype=dtype,
        linear_indices=linear_indices,
        voltage_weights=weights,
        injection=injection,
        dt=dt_tensor,
        resistance=resistance_tensor,
        inductance=inductance_tensor,
        capacitance=capacitance_tensor,
        inductance_active=bool(inductance_enabled),
        capacitance_active=bool(capacitance_enabled),
        resistance_is_open=torch.isinf(resistance_tensor),
        inductance_enabled=inductance_enabled,
        capacitance_enabled=capacitance_enabled,
        default_thevenin_voltage=default_drive,
        coupling_impedance=coupling_impedance,
        discrete_port_impedance=discrete_port_impedance,
        series_inductive_impedance=series_inductive_impedance,
        series_capacitive_impedance=series_capacitive_impedance,
        parallel_resistive_admittance=parallel_resistive_admittance,
        parallel_inductive_admittance=parallel_inductive_admittance,
        parallel_capacitive_admittance=parallel_capacitive_admittance,
        parallel_total_admittance=parallel_total_admittance,
        inverse_denominator=inverse_denominator,
        edge_buffer=edge_buffer,
        correction_buffer=correction_buffer,
        inductor_current=scalar_zeros[0],
        capacitor_voltage=scalar_zeros[1],
        last_voltage_before=scalar_zeros[2],
        last_voltage_midpoint=scalar_zeros[3],
        last_voltage_after=scalar_zeros[4],
        last_model_voltage_midpoint=scalar_zeros[5],
        last_branch_current=scalar_zeros[6],
        last_resistor_current=scalar_zeros[7],
        last_capacitor_current=scalar_zeros[8],
        last_inductor_current_midpoint=scalar_zeros[9],
        last_dissipated_energy=scalar_zeros[10],
        last_stored_energy_change=scalar_zeros[11],
        last_source_work=scalar_zeros[12],
        last_field_energy_change=scalar_zeros[13],
    )


def replay_lumped_runtime(
    runtime: LumpedRuntime,
    electric_field: torch.Tensor,
    *,
    previous_voltage: torch.Tensor,
    inductor_current: torch.Tensor,
    capacitor_voltage: torch.Tensor,
    drive: torch.Tensor,
    field_name: str,
    kind: str,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LumpedStepTrace]:
    """Replay one series implicit branch without mutating prepared runtime state.

    ``previous_voltage`` is the scalar port voltage carried out of the previous
    step (the forward pass' ``last_voltage_after``). Replaying the incremental
    scalar recurrence -- rather than re-deriving the previous voltage from the
    field tensor -- keeps the reconstructed forward state bit-identical to the
    in-place forward update in :func:`apply_lumped_runtime`, which the
    checkpointed adjoint relies on. Each intermediate is evaluated with the same
    operations, operand order, and dtype as the forward path so no floating-point
    reassociation is introduced. The returned ``next_previous_voltage`` is the
    updated scalar to carry into the following step.
    """

    if runtime.topology != "series":
        raise NotImplementedError(
            "Differentiable lumped replay currently supports series R/L/C topology only."
        )
    flat = electric_field.reshape(-1)
    local_field = torch.index_select(flat, 0, runtime.linear_indices)
    voltage_before = torch.sum(local_field * runtime.voltage_weights, dim=0)
    averaged_voltage = (previous_voltage + voltage_before) * 0.5
    numerator = torch.addcmul(
        averaged_voltage - drive - capacitor_voltage,
        runtime.series_inductive_impedance,
        inductor_current,
    )
    branch_current = numerator * runtime.inverse_denominator
    corrected = electric_field.clone()
    corrected.reshape(-1).index_add_(
        0,
        runtime.linear_indices,
        -runtime.injection * branch_current,
    )
    voltage_midpoint = torch.addcmul(
        averaged_voltage,
        runtime.discrete_port_impedance,
        branch_current,
        value=-1.0,
    )
    next_previous_voltage = torch.addcmul(
        voltage_before,
        runtime.coupling_impedance,
        branch_current,
        value=-1.0,
    )

    next_inductor_current = 2.0 * branch_current - inductor_current
    if not runtime.inductance_active:
        next_inductor_current = torch.zeros_like(next_inductor_current)
    next_capacitor_voltage = (
        capacitor_voltage
        + 2.0 * runtime.series_capacitive_impedance * branch_current
    )
    if not runtime.capacitance_active:
        next_capacitor_voltage = torch.zeros_like(next_capacitor_voltage)

    trace = LumpedStepTrace(
        runtime=runtime,
        field_name=field_name,
        kind=kind,
        index=int(index),
        previous_voltage=previous_voltage,
        voltage_before=voltage_before,
        branch_current=branch_current,
        voltage_midpoint=voltage_midpoint,
        drive=drive,
        old_inductor_current=inductor_current,
        old_capacitor_voltage=capacitor_voltage,
    )
    return (
        corrected,
        next_inductor_current,
        next_capacitor_voltage,
        next_previous_voltage,
        trace,
    )


def pullback_lumped_runtime(
    trace: LumpedStepTrace,
    field_adjoint: torch.Tensor,
    *,
    inductor_current_adjoint: torch.Tensor,
    capacitor_voltage_adjoint: torch.Tensor,
    previous_voltage_adjoint: torch.Tensor,
    voltage_sample_adjoint: torch.Tensor,
    network_current_sample_adjoint: torch.Tensor,
    eps_edge: torch.Tensor,
) -> LumpedStepPullback:
    """Apply the exact VJP of one implicit series branch on the runtime device.

    ``previous_voltage_adjoint`` is the cotangent that flows in on this step's
    produced ``next_previous_voltage`` scalar; the returned
    ``previous_voltage_adjoint`` is the cotangent on the input ``previous_voltage``
    that continues to the preceding step.
    """

    runtime = trace.runtime
    if runtime.topology != "series":
        raise NotImplementedError(
            "Differentiable lumped pullback currently supports series R/L/C topology only."
        )
    flat_adjoint = field_adjoint.reshape(-1)
    local_field_adjoint = torch.index_select(
        flat_adjoint,
        0,
        runtime.linear_indices,
    )
    branch_current = trace.branch_current
    bar_current = -network_current_sample_adjoint
    bar_injection = -branch_current * local_field_adjoint
    bar_current = bar_current - torch.dot(runtime.injection, local_field_adjoint)

    bar_averaged_voltage = voltage_sample_adjoint
    bar_discrete_impedance = -branch_current * voltage_sample_adjoint
    bar_current = (
        bar_current
        - runtime.discrete_port_impedance * voltage_sample_adjoint
    )

    # next_previous_voltage = voltage_before - coupling_impedance * branch_current.
    # The scalar previous-voltage state carries the reverse cotangent instead of
    # the previous electric field tensor, mirroring the forward recurrence.
    bar_voltage_before = previous_voltage_adjoint
    bar_current = bar_current - runtime.coupling_impedance * previous_voltage_adjoint

    bar_old_inductor = torch.zeros_like(trace.old_inductor_current)
    bar_old_capacitor = torch.zeros_like(trace.old_capacitor_voltage)
    bar_inductive_impedance = torch.zeros_like(runtime.series_inductive_impedance)
    bar_capacitive_impedance = torch.zeros_like(runtime.series_capacitive_impedance)
    if runtime.inductance_active:
        bar_current = bar_current + 2.0 * inductor_current_adjoint
        bar_old_inductor = bar_old_inductor - inductor_current_adjoint
    if runtime.capacitance_active:
        bar_old_capacitor = bar_old_capacitor + capacitor_voltage_adjoint
        bar_capacitive_impedance = (
            bar_capacitive_impedance
            + 2.0 * branch_current * capacitor_voltage_adjoint
        )
        bar_current = (
            bar_current
            + 2.0 * runtime.series_capacitive_impedance
            * capacitor_voltage_adjoint
        )

    bar_numerator = bar_current * runtime.inverse_denominator
    bar_denominator = (
        -bar_current * branch_current * runtime.inverse_denominator
    )
    bar_averaged_voltage = bar_averaged_voltage + bar_numerator
    bar_drive = -bar_numerator
    bar_old_capacitor = bar_old_capacitor - bar_numerator
    bar_inductive_impedance = (
        bar_inductive_impedance
        + trace.old_inductor_current * bar_numerator
        + bar_denominator
    )
    bar_old_inductor = (
        bar_old_inductor
        + runtime.series_inductive_impedance * bar_numerator
    )
    bar_capacitive_impedance = bar_capacitive_impedance + bar_denominator
    bar_resistance = bar_denominator
    bar_discrete_impedance = bar_discrete_impedance + bar_denominator

    # averaged_voltage = 0.5 * (previous_voltage + voltage_before). The previous
    # half now flows to the scalar previous-voltage cotangent (returned to the
    # preceding step); the current half joins voltage_before's field cotangent.
    previous_voltage_grad = 0.5 * bar_averaged_voltage
    bar_voltage_before = bar_voltage_before + 0.5 * bar_averaged_voltage

    # discrete_port_impedance = 0.5 * coupling_impedance and the direct
    # coupling_impedance in next_previous_voltage both flow to injection via the
    # voltage weights (coupling_impedance = dot(voltage_weights, injection)).
    bar_injection = (
        bar_injection
        + 0.5 * runtime.voltage_weights * bar_discrete_impedance
    )
    bar_injection = (
        bar_injection
        + runtime.voltage_weights * (-branch_current * previous_voltage_adjoint)
    )
    free_field_adjoint = field_adjoint.clone()
    free_field_adjoint.reshape(-1).index_add_(
        0,
        runtime.linear_indices,
        runtime.voltage_weights * bar_voltage_before,
    )

    local_eps = torch.index_select(
        eps_edge.reshape(-1),
        0,
        runtime.linear_indices,
    )
    local_grad_eps = -bar_injection * runtime.injection / local_eps
    grad_eps = torch.zeros_like(eps_edge)
    grad_eps.reshape(-1).index_add_(
        0,
        runtime.linear_indices,
        local_grad_eps,
    )
    grad_inductance = (
        2.0 * bar_inductive_impedance / runtime.dt
        if runtime.inductance_active
        else torch.zeros_like(runtime.inductance)
    )
    grad_capacitance = (
        -bar_capacitive_impedance
        * runtime.series_capacitive_impedance
        / runtime.capacitance
        if runtime.capacitance_active
        else torch.zeros_like(runtime.capacitance)
    )
    return LumpedStepPullback(
        field_adjoint=free_field_adjoint,
        previous_voltage_adjoint=previous_voltage_grad,
        inductor_current_adjoint=bar_old_inductor,
        capacitor_voltage_adjoint=bar_old_capacitor,
        grad_eps=grad_eps,
        grad_resistance=bar_resistance,
        grad_inductance=grad_inductance,
        grad_capacitance=grad_capacitance,
        grad_drive=bar_drive,
    )


def apply_lumped_runtime(
    runtime: LumpedRuntime,
    electric_field: torch.Tensor,
    *,
    thevenin_voltage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one implicit midpoint circuit correction to ``electric_field`` in-place.

    Branch current is positive from the field into the termination. The update
    satisfies ``E_new = E_free - q I`` and uses the full Yee-step midpoint
    ``0.5 * (V_old + V_free)`` for the circuit constitutive relation.
    """

    if electric_field.device != runtime.linear_indices.device:
        raise ValueError("electric_field and lumped runtime must be on the same device.")
    if electric_field.dtype != runtime.field_dtype:
        raise TypeError("electric_field dtype must match the prepared runtime.")
    if tuple(electric_field.shape) != runtime.field_shape:
        raise ValueError("electric_field shape must match the prepared runtime.")
    if not electric_field.is_contiguous():
        raise ValueError("electric_field must be contiguous.")

    drive = runtime.default_thevenin_voltage
    if thevenin_voltage is not None:
        if thevenin_voltage.ndim != 0:
            raise ValueError("thevenin_voltage must be a scalar tensor.")
        if thevenin_voltage.device != electric_field.device:
            raise ValueError("thevenin_voltage must be on the runtime device.")
        if thevenin_voltage.dtype != runtime.field_dtype:
            raise TypeError("thevenin_voltage dtype must match the prepared runtime.")
        drive = thevenin_voltage

    flat_field = electric_field.view(-1)
    torch.index_select(
        flat_field,
        0,
        runtime.linear_indices,
        out=runtime.edge_buffer,
    )
    torch.mul(
        runtime.edge_buffer,
        runtime.voltage_weights,
        out=runtime.edge_buffer,
    )
    torch.sum(
        runtime.edge_buffer,
        dim=0,
        out=runtime.last_voltage_before,
    )

    runtime.last_branch_current.copy_(runtime.last_voltage_after)
    runtime.last_branch_current.add_(runtime.last_voltage_before)
    runtime.last_branch_current.mul_(0.5)
    if runtime.topology == "series":
        runtime.last_branch_current.sub_(drive)
        runtime.last_branch_current.sub_(runtime.capacitor_voltage)
        runtime.last_branch_current.addcmul_(
            runtime.series_inductive_impedance,
            runtime.inductor_current,
        )
    else:
        runtime.last_branch_current.sub_(drive)
        runtime.last_branch_current.mul_(runtime.parallel_total_admittance)
        runtime.last_branch_current.add_(runtime.inductor_current)
        runtime.last_branch_current.addcmul_(
            runtime.parallel_capacitive_admittance,
            runtime.capacitor_voltage,
            value=-1.0,
        )
    runtime.last_branch_current.mul_(runtime.inverse_denominator)
    torch.mul(
        runtime.injection,
        runtime.last_branch_current,
        out=runtime.correction_buffer,
    )
    flat_field.index_add_(
        0,
        runtime.linear_indices,
        runtime.correction_buffer,
        alpha=-1.0,
    )

    runtime.last_voltage_midpoint.copy_(runtime.last_voltage_after)
    runtime.last_voltage_midpoint.add_(runtime.last_voltage_before)
    runtime.last_voltage_midpoint.mul_(0.5)
    runtime.last_voltage_midpoint.addcmul_(
        runtime.discrete_port_impedance,
        runtime.last_branch_current,
        value=-1.0,
    )
    runtime.last_voltage_after.copy_(runtime.last_voltage_before)
    runtime.last_voltage_after.addcmul_(
        runtime.coupling_impedance,
        runtime.last_branch_current,
        value=-1.0,
    )
    runtime.last_model_voltage_midpoint.copy_(runtime.last_voltage_midpoint)
    runtime.last_model_voltage_midpoint.sub_(drive)

    if runtime.topology == "series":
        old_inductor_current = runtime.inductor_current.clone()
        old_capacitor_voltage = runtime.capacitor_voltage.clone()
        next_inductor_current = 2.0 * runtime.last_branch_current - old_inductor_current
        next_inductor_current = torch.where(
            runtime.inductance_enabled,
            next_inductor_current,
            torch.zeros_like(next_inductor_current),
        )
        next_inductor_current = torch.where(
            runtime.resistance_is_open,
            old_inductor_current,
            next_inductor_current,
        )
        next_capacitor_voltage = (
            old_capacitor_voltage
            + 2.0 * runtime.series_capacitive_impedance * runtime.last_branch_current
        )
        next_capacitor_voltage = torch.where(
            runtime.capacitance_enabled,
            next_capacitor_voltage,
            torch.zeros_like(next_capacitor_voltage),
        )
        next_capacitor_voltage = torch.where(
            runtime.resistance_is_open,
            old_capacitor_voltage,
            next_capacitor_voltage,
        )

        runtime.last_resistor_current.copy_(runtime.last_branch_current)
        runtime.last_capacitor_current.copy_(runtime.last_branch_current)
        runtime.last_capacitor_current.mul_(runtime.capacitance_enabled)
        runtime.last_inductor_current_midpoint.copy_(runtime.last_branch_current)
        runtime.last_inductor_current_midpoint.mul_(runtime.inductance_enabled)
        inductor_voltage = runtime.series_inductive_impedance * (
            runtime.last_branch_current - old_inductor_current
        )
        capacitor_voltage_midpoint = (
            old_capacitor_voltage
            + runtime.series_capacitive_impedance * runtime.last_branch_current
        )
        runtime.last_dissipated_energy.copy_(runtime.last_model_voltage_midpoint)
        runtime.last_dissipated_energy.sub_(inductor_voltage)
        runtime.last_dissipated_energy.sub_(capacitor_voltage_midpoint)
        runtime.last_dissipated_energy.mul_(runtime.last_branch_current)
        runtime.last_dissipated_energy.mul_(runtime.dt)
    else:
        old_inductor_current = runtime.inductor_current.clone()
        old_capacitor_voltage = runtime.capacitor_voltage.clone()
        runtime.last_resistor_current.copy_(runtime.last_model_voltage_midpoint)
        runtime.last_resistor_current.mul_(runtime.parallel_resistive_admittance)
        runtime.last_capacitor_current.copy_(runtime.last_model_voltage_midpoint)
        runtime.last_capacitor_current.sub_(old_capacitor_voltage)
        runtime.last_capacitor_current.mul_(runtime.parallel_capacitive_admittance)
        runtime.last_inductor_current_midpoint.copy_(runtime.last_model_voltage_midpoint)
        runtime.last_inductor_current_midpoint.mul_(runtime.parallel_inductive_admittance)
        runtime.last_inductor_current_midpoint.add_(old_inductor_current)

        next_inductor_current = 2.0 * runtime.last_inductor_current_midpoint - old_inductor_current
        next_inductor_current = torch.where(
            runtime.inductance_enabled,
            next_inductor_current,
            torch.zeros_like(next_inductor_current),
        )
        next_capacitor_voltage = 2.0 * runtime.last_model_voltage_midpoint - old_capacitor_voltage
        next_capacitor_voltage = torch.where(
            runtime.capacitance_enabled,
            next_capacitor_voltage,
            torch.zeros_like(next_capacitor_voltage),
        )
        runtime.last_dissipated_energy.copy_(runtime.last_model_voltage_midpoint)
        runtime.last_dissipated_energy.mul_(runtime.last_resistor_current)
        runtime.last_dissipated_energy.mul_(runtime.dt)

    old_stored_energy = (
        0.5 * runtime.inductance * old_inductor_current.square()
        + 0.5 * runtime.capacitance * old_capacitor_voltage.square()
    )
    next_stored_energy = (
        0.5 * runtime.inductance * next_inductor_current.square()
        + 0.5 * runtime.capacitance * next_capacitor_voltage.square()
    )
    runtime.last_stored_energy_change.copy_(next_stored_energy)
    runtime.last_stored_energy_change.sub_(old_stored_energy)
    runtime.inductor_current.copy_(next_inductor_current)
    runtime.capacitor_voltage.copy_(next_capacitor_voltage)

    runtime.last_source_work.copy_(drive)
    runtime.last_source_work.mul_(runtime.last_branch_current)
    runtime.last_source_work.mul_(runtime.dt)
    runtime.last_source_work.neg_()
    runtime.last_field_energy_change.copy_(runtime.last_source_work)
    runtime.last_field_energy_change.sub_(runtime.last_dissipated_energy)
    runtime.last_field_energy_change.sub_(runtime.last_stored_energy_change)
    return runtime.last_branch_current


__all__ = [
    "LumpedRuntime",
    "LumpedStepPullback",
    "LumpedStepTrace",
    "apply_lumped_runtime",
    "prepare_lumped_runtime",
    "pullback_lumped_runtime",
    "replay_lumped_runtime",
]

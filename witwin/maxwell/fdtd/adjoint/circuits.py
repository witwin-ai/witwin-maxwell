from __future__ import annotations

from dataclasses import dataclass

import torch

from ...compiler.mna import CircuitState, _source_value_at, _terminal_voltage
from ..checkpoint import circuit_state_name
from ..circuits import _stamp_norton


@dataclass(frozen=True)
class CircuitStepTrace:
    runtime: object
    circuit_index: int
    step_index: int
    field_inputs: dict[str, torch.Tensor]
    previous_state: CircuitState
    port_indices: tuple[int, ...]


def _checkpoint_state(runtime, circuit_index: int, state) -> CircuitState:
    def prefix(name):
        return circuit_state_name(circuit_index, name)

    return CircuitState(
        capacitor_voltage={
            device.name: state[prefix(f"capacitor_voltage_{device.name}")]
            for device in runtime.circuit.devices
            if device.kind == "capacitor"
        },
        capacitor_current={
            device.name: state[prefix(f"capacitor_current_{device.name}")]
            for device in runtime.circuit.devices
            if device.kind == "capacitor"
        },
        inductor_current=state[prefix("inductor_current")],
        inductor_voltage=state[prefix("inductor_voltage")],
    )


def _state_tensors(runtime, circuit_index: int, state: CircuitState):
    capacitors = tuple(
        device for device in runtime.circuit.devices if device.kind == "capacitor"
    )
    result = [
        (
            circuit_state_name(circuit_index, f"capacitor_voltage_{device.name}"),
            state.capacitor_voltage[device.name],
        )
        for device in capacitors
    ]
    result.extend(
        (
            circuit_state_name(circuit_index, f"capacitor_current_{device.name}"),
            state.capacitor_current[device.name],
        )
        for device in capacitors
    )
    result.extend(
        (
            (circuit_state_name(circuit_index, "inductor_current"), state.inductor_current),
            (circuit_state_name(circuit_index, "inductor_voltage"), state.inductor_voltage),
        )
    )
    return tuple(result)


def _port_terms(port, eps_edge: torch.Tensor | None):
    field = port.field
    if eps_edge is None:
        injection = field.injection
        coupling_impedance = field.coupling_impedance
    else:
        local_eps = torch.index_select(
            eps_edge.reshape(-1),
            0,
            field.linear_indices,
        )
        base_eps = torch.index_select(
            getattr(port.solver, f"eps_{port.field_name}").detach().reshape(-1),
            0,
            field.linear_indices,
        )
        injection = field.injection.detach() * base_eps / local_eps
        coupling_impedance = torch.dot(field.voltage_weights, injection)
    return injection, coupling_impedance


def _branch_current_row(runtime, currents, solution):
    physical_devices = runtime.physical_devices
    if not physical_devices:
        return solution.new_zeros((0,))
    return torch.stack(tuple(currents[device.name] for device in physical_devices))


def _matrix_condition(matrix: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(matrix)
    return singular_values[0] / singular_values[-1]


def _functional_dc_condition(runtime, eps_by_field) -> torch.Tensor:
    system = runtime.system
    matrix, rhs = system.assemble_dc()
    zero = rhs.new_zeros(())
    integration = runtime.circuit.config.integration
    for port in runtime.ports:
        _injection, coupling_impedance = _port_terms(
            port,
            eps_by_field[port.field_name],
        )
        conductance = torch.reciprocal(
            0.5 * coupling_impedance
            if integration == "trapezoidal"
            else coupling_impedance
        )
        _stamp_norton(
            matrix,
            rhs,
            system.graph,
            port.binding,
            conductance,
            zero,
        )
    constraint_system = system._initial_constraint_system()
    if constraint_system is not None:
        rows, _targets = constraint_system
        count = rows.shape[0]
        augmented = matrix.new_zeros(
            (system.unknown_count + count, system.unknown_count + count)
        )
        augmented[: system.unknown_count, : system.unknown_count] = matrix
        augmented[: system.unknown_count, system.unknown_count :] = rows.transpose(0, 1)
        augmented[system.unknown_count :, : system.unknown_count] = rows
        matrix = augmented
    return _matrix_condition(matrix)


def _functional_step(
    runtime,
    fields,
    previous_state,
    *,
    step_index: int,
    eps_by_field=None,
    differentiable: bool,
):
    integration = runtime.integration_keys[step_index]
    time = runtime.sample_times[step_index + 1]
    free_voltages = tuple(
        torch.dot(
            torch.index_select(fields[port.field_name].reshape(-1), 0, port.field.linear_indices),
            port.field.voltage_weights,
        )
        for port in runtime.ports
    )
    port_terms = tuple(
        _port_terms(
            port,
            None if eps_by_field is None else eps_by_field[port.field_name],
        )
        for port in runtime.ports
    )
    conductances = tuple(
        torch.reciprocal(
            0.5 * coupling_impedance
            if integration == "trapezoidal"
            else coupling_impedance
        )
        for _injection, coupling_impedance in port_terms
    )

    system = runtime.system
    if differentiable:
        system._refresh_parameter_cache()
        system._refresh_inductance_matrix()
        source_values = system._source_values(time.reshape(1))
        source_index = 0
        matrix, rhs = system.assemble_transient(
            previous_state,
            time=time,
            source_values=source_values,
            step_index=0,
            integration=integration,
            midpoint=integration == "trapezoidal",
        )
        for port, conductance, free_voltage in zip(
            runtime.ports,
            conductances,
            free_voltages,
        ):
            _stamp_norton(
                matrix,
                rhs,
                system.graph,
                port.binding,
                conductance,
                free_voltage,
            )
        condition = _matrix_condition(matrix)
        solution = torch.linalg.solve(matrix, rhs)
    else:
        source_values = runtime.source_values
        source_index = step_index
        _matrix, rhs = system.assemble_transient(
            previous_state,
            time=time,
            source_values=source_values,
            step_index=step_index,
            integration=integration,
            midpoint=integration == "trapezoidal",
            stamp_matrix=False,
        )
        for port, conductance, free_voltage in zip(
            runtime.ports,
            conductances,
            free_voltages,
        ):
            _stamp_norton(
                _matrix,
                rhs,
                system.graph,
                port.binding,
                conductance,
                free_voltage,
                stamp_matrix=False,
            )
        factors, pivots, condition = runtime.factor_cache[
            (integration, runtime.switch_keys[step_index])
        ]
        solution = torch.linalg.lu_solve(
            factors,
            pivots,
            rhs.unsqueeze(-1),
        ).squeeze(-1)

    currents = runtime._midpoint_currents(
        solution,
        previous_state,
        integration,
        source_index,
        time,
    )
    # Current sources are the only physical branches whose sampled current is
    # read directly from source_values rather than from the solved state.  The
    # runtime helper uses its forward cache, so replace those rows with the
    # source values reconstructed for this replay to retain waveform gradients.
    currents = dict(currents)
    for device in runtime.circuit.devices:
        if device.kind == "current_source":
            currents[device.name] = _source_value_at(
                source_values[device.name],
                source_index,
            )
    next_state = system._update_state(
        previous_state,
        solution,
        integration=integration,
        midpoint=integration == "trapezoidal",
    )
    corrected = dict(fields)
    samples = []
    port_powers = []
    field_energy_changes = []
    for port, free_voltage, (injection, coupling_impedance), conductance in zip(
        runtime.ports,
        free_voltages,
        port_terms,
        conductances,
    ):
        voltage = _terminal_voltage(
            solution,
            system.graph,
            port.binding.positive,
            port.binding.negative,
        )
        current = conductance * (free_voltage - voltage)
        field = corrected[port.field_name].clone()
        field.reshape(-1).index_add_(
            0,
            port.field.linear_indices,
            -injection * current,
        )
        corrected[port.field_name] = field
        samples.append((voltage, current))
        port_powers.append(voltage * current)
        field_energy_changes.append(
            -(free_voltage - 0.5 * coupling_impedance * current)
            * current
            * port.field.dt
        )
    node_row = torch.cat(
        (
            solution.new_zeros((1,)),
            solution[: len(system.graph.nodes) - 1],
        )
    )
    branch_row = _branch_current_row(runtime, currents, solution)
    device_powers = []
    nonreactive_powers = []
    for device in runtime.circuit.devices:
        if device.kind == "mutual_inductor":
            power = solution.new_zeros(())
        else:
            voltage = _terminal_voltage(
                solution,
                system.graph,
                device.positive,
                device.negative,
            )
            power = voltage * currents[device.name]
        device_powers.append(power)
        if device.kind not in {"capacitor", "inductor", "mutual_inductor"}:
            nonreactive_powers.append(power)
    device_power_row = (
        torch.stack(device_powers) if device_powers else solution.new_zeros((0,))
    )
    port_power_row = (
        torch.stack(port_powers) if port_powers else solution.new_zeros((0,))
    )
    field_energy_row = (
        torch.stack(field_energy_changes)
        if field_energy_changes
        else solution.new_zeros((0,))
    )
    nonreactive_power = (
        torch.stack(nonreactive_powers).sum()
        if nonreactive_powers
        else solution.new_zeros(())
    )
    balance_increment = (
        runtime._stored_energy(next_state)
        - runtime._stored_energy(previous_state)
        + system.dt * (nonreactive_power - port_power_row.sum())
    )
    return (
        corrected,
        next_state,
        tuple(samples),
        node_row,
        branch_row,
        device_power_row,
        balance_increment,
        port_power_row,
        field_energy_row,
        condition,
    )


def replay_circuit_runtimes(
    solver,
    electric_fields,
    state,
    *,
    step_index: int,
    capture=None,
):
    runtimes = tuple(getattr(solver, "_circuit_runtimes", ()))
    if not runtimes:
        return electric_fields, {}
    fields = dict(electric_fields)
    next_auxiliary = {}
    traces = []
    port_runtime_indices = {
        id(getattr(port_runtime, "circuit_port", None)): index
        for index, port_runtime in enumerate(getattr(solver, "_port_runtimes", ()))
        if getattr(port_runtime, "circuit_port", None) is not None
    }
    for circuit_index, runtime in enumerate(runtimes):
        previous_state = _checkpoint_state(runtime, circuit_index, state)
        field_names = tuple(dict.fromkeys(port.field_name for port in runtime.ports))
        field_inputs = {name: fields[name] for name in field_names}
        corrected, next_state, *_diagnostics = _functional_step(
            runtime,
            field_inputs,
            previous_state,
            step_index=step_index,
            differentiable=False,
        )
        fields.update(corrected)
        step_name = circuit_state_name(circuit_index, "step")
        next_auxiliary[step_name] = state[step_name] + 1
        next_auxiliary.update(dict(_state_tensors(runtime, circuit_index, next_state)))
        traces.append(
            CircuitStepTrace(
                runtime=runtime,
                circuit_index=circuit_index,
                step_index=step_index,
                field_inputs=field_inputs,
                previous_state=previous_state,
                port_indices=tuple(port_runtime_indices[id(port)] for port in runtime.ports),
            )
        )
    if capture is not None:
        capture.append(tuple(traces))
    return fields, next_auxiliary


def _parameter_items(circuit, differentiable_keys):
    items = []
    replacements = []
    leaves = {}
    for device in circuit.devices:
        for parameter_name, value in device.parameters.items():
            values = (
                ((parameter_name, device, parameter_name, value),)
                if isinstance(value, torch.Tensor)
                else tuple(
                    (f"{parameter_name}.{name}", value, name, tensor)
                    for name, tensor in vars(value).items()
                    if isinstance(tensor, torch.Tensor)
                )
                if value is not None and hasattr(value, "__dict__")
                else ()
            )
            for qualified_name, owner, attribute, tensor in values:
                key = ("circuit", circuit.name, device.name, qualified_name)
                if key not in differentiable_keys:
                    continue
                tensor_id = id(tensor)
                leaf = leaves.get(tensor_id)
                if leaf is None:
                    # Runtime circuit values may be non-leaf SceneModule
                    # expressions, or may have been dtype-cast while the custom
                    # Function forward ran under no-grad.  Rebuild every selected
                    # semantic value from an explicit local leaf, then route its
                    # cotangent through the fresh semantic scene graph in bridge.
                    leaf = tensor.detach().requires_grad_(True)
                    leaves[tensor_id] = leaf
                    items.append((key, leaf))
                replacements.append((owner, attribute, tensor, leaf))
    return tuple(items), tuple(replacements)


def pullback_circuit_runtimes(
    traces,
    adjoint_state,
    *,
    port_sample_adjoints,
    circuit_sample_adjoints,
    eps_by_field,
    differentiable_parameter_keys=(),
):
    updated = dict(adjoint_state)
    grad_eps = {name: torch.zeros_like(value) for name, value in eps_by_field.items()}
    semantic_grads = {}
    for trace in reversed(tuple(traces)):
        runtime = trace.runtime
        with torch.enable_grad():
            field_leaves = {
                name: value.detach().requires_grad_(True)
                for name, value in trace.field_inputs.items()
            }
            previous_state = CircuitState(
                capacitor_voltage={
                    name: value.detach().requires_grad_(True)
                    for name, value in trace.previous_state.capacitor_voltage.items()
                },
                capacitor_current={
                    name: value.detach().requires_grad_(True)
                    for name, value in trace.previous_state.capacitor_current.items()
                },
                inductor_current=trace.previous_state.inductor_current.detach().requires_grad_(True),
                inductor_voltage=trace.previous_state.inductor_voltage.detach().requires_grad_(True),
            )
            parameter_items, replacements = _parameter_items(
                runtime.circuit,
                differentiable_parameter_keys,
            )
            original_plan = runtime.system.plan
            for owner, attribute, _original, leaf in replacements:
                object.__setattr__(owner, attribute, leaf)
            try:
                (
                    corrected,
                    next_state,
                    samples,
                    node_row,
                    branch_row,
                    device_power_row,
                    balance_increment,
                    port_power_row,
                    field_energy_row,
                    condition,
                ) = _functional_step(
                    runtime,
                    field_leaves,
                    previous_state,
                    step_index=trace.step_index,
                    eps_by_field=eps_by_field,
                    differentiable=True,
                )
                dc_condition = (
                    _functional_dc_condition(runtime, eps_by_field)
                    if trace.step_index == 0
                    else None
                )
            finally:
                for owner, attribute, original, _leaf in reversed(replacements):
                    object.__setattr__(owner, attribute, original)
                runtime.system.plan = original_plan
                runtime.system._refresh_parameter_cache()

            output_pairs = [
                (corrected[name], updated[name])
                for name in field_leaves
            ]
            next_state_pairs = _state_tensors(runtime, trace.circuit_index, next_state)
            output_pairs.extend(
                (value, updated[name])
                for name, value in next_state_pairs
                if value.requires_grad
            )
            for port_index, (voltage, current) in zip(trace.port_indices, samples):
                voltage_seed, current_seed, _drive_seed = port_sample_adjoints.get(
                    port_index,
                    (torch.zeros_like(voltage), torch.zeros_like(current), torch.zeros_like(current)),
                )
                output_pairs.extend(
                    (
                        (voltage, voltage_seed.to(device=voltage.device, dtype=voltage.dtype)),
                        (current, -current_seed.to(device=current.device, dtype=current.dtype)),
                    )
                )
            (
                node_seed,
                branch_seed,
                device_power_seed,
                balance_increment_seed,
                port_power_seed,
                field_energy_seed,
                dc_condition_seed,
                last_condition_seed,
            ) = circuit_sample_adjoints.get(
                trace.circuit_index,
                (
                    torch.zeros_like(node_row),
                    torch.zeros_like(branch_row),
                    torch.zeros_like(device_power_row),
                    torch.zeros_like(balance_increment),
                    torch.zeros_like(port_power_row),
                    torch.zeros_like(field_energy_row),
                    torch.zeros_like(condition),
                    torch.zeros_like(condition),
                ),
            )
            output_pairs.extend(
                (
                    (node_row, node_seed.to(device=node_row.device, dtype=node_row.dtype)),
                    (
                        branch_row,
                        branch_seed.to(device=branch_row.device, dtype=branch_row.dtype),
                    ),
                    (
                        device_power_row,
                        device_power_seed.to(
                            device=device_power_row.device,
                            dtype=device_power_row.dtype,
                        ),
                    ),
                    (
                        balance_increment,
                        balance_increment_seed.to(
                            device=balance_increment.device,
                            dtype=balance_increment.dtype,
                        ),
                    ),
                    (
                        port_power_row,
                        port_power_seed.to(
                            device=port_power_row.device,
                            dtype=port_power_row.dtype,
                        ),
                    ),
                    (
                        field_energy_row,
                        field_energy_seed.to(
                            device=field_energy_row.device,
                            dtype=field_energy_row.dtype,
                        ),
                    ),
                    (
                        condition,
                        last_condition_seed.to(
                            device=condition.device,
                            dtype=condition.dtype,
                        ),
                    ),
                )
            )
            if dc_condition is not None:
                output_pairs.append(
                    (
                        dc_condition,
                        dc_condition_seed.to(
                            device=dc_condition.device,
                            dtype=dc_condition.dtype,
                        ),
                    )
                )

            state_inputs = _state_tensors(runtime, trace.circuit_index, previous_state)
            eps_items = tuple(
                (name, eps_by_field[name])
                for name in field_leaves
            )
            inputs = (
                tuple(field_leaves.items())
                + state_inputs
                + parameter_items
                + eps_items
            )
            outputs = tuple(value for value, _seed in output_pairs if value.requires_grad)
            output_grads = tuple(seed for value, seed in output_pairs if value.requires_grad)
            gradients = torch.autograd.grad(
                outputs,
                tuple(value for _name, value in inputs),
                grad_outputs=output_grads,
                allow_unused=True,
            )

        gradient_by_position = tuple(
            torch.zeros_like(value) if gradient is None else gradient
            for (_name, value), gradient in zip(inputs, gradients)
        )
        cursor = 0
        for name in field_leaves:
            updated[name] = gradient_by_position[cursor]
            cursor += 1
        for name, _value in state_inputs:
            updated[name] = gradient_by_position[cursor]
            cursor += 1
        for key, _value in parameter_items:
            contribution = gradient_by_position[cursor]
            semantic_grads[key] = semantic_grads.get(
                key,
                torch.zeros_like(contribution),
            ) + contribution
            cursor += 1
        for name, _value in eps_items:
            grad_eps[name] = grad_eps[name] + gradient_by_position[cursor]
            cursor += 1
        updated[circuit_state_name(trace.circuit_index, "step")] = torch.zeros_like(
            updated[circuit_state_name(trace.circuit_index, "step")]
        )
    return updated, grad_eps, semantic_grads


__all__ = [
    "CircuitStepTrace",
    "pullback_circuit_runtimes",
    "replay_circuit_runtimes",
]

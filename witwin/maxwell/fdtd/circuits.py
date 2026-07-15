from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..circuits import (
    CircuitData,
    MutualInductor,
    TimedSwitch,
)
from ..compiler.mna import (
    CircuitState,
    _node_unknown,
    _terminal_voltage,
    compile_coupled_mna_system,
)
from ..lumped import Capacitor, Inductor
from .lumped import FieldPortCoupling, prepare_field_port_coupling


def _add(matrix, row: int | None, column: int | None, value) -> None:
    if row is not None and column is not None:
        matrix[row, column] = matrix[row, column] + value


def _add_rhs(rhs, row: int | None, value) -> None:
    if row is not None:
        rhs[row] = rhs[row] + value


def _stamp_norton(
    matrix,
    rhs,
    graph,
    binding,
    conductance,
    free_voltage,
    *,
    stamp_matrix: bool = True,
) -> None:
    positive = _node_unknown(graph, binding.positive)
    negative = _node_unknown(graph, binding.negative)
    if stamp_matrix:
        _add(matrix, positive, positive, conductance)
        _add(matrix, negative, negative, conductance)
        _add(matrix, positive, negative, -conductance)
        _add(matrix, negative, positive, -conductance)
    _add_rhs(rhs, positive, conductance * free_voltage)
    _add_rhs(rhs, negative, -conductance * free_voltage)


@dataclass
class CircuitPortRuntime:
    circuit_name: str
    integration: str
    binding: object
    solver: object
    field_name: str
    field: FieldPortCoupling
    last_integration: str
    trapezoidal_conductance: torch.Tensor
    backward_euler_conductance: torch.Tensor

    def conductance(self, integration: str) -> torch.Tensor:
        return (
            self.trapezoidal_conductance
            if integration == "trapezoidal"
            else self.backward_euler_conductance
        )


@dataclass
class EMCircuitRuntime:
    circuit: object
    system: object
    ports: tuple[CircuitPortRuntime, ...]
    state: CircuitState
    initial_solution: torch.Tensor
    dc_condition: torch.Tensor
    step_tensor: torch.Tensor
    matrix_buffer: torch.Tensor
    rhs_buffer: torch.Tensor
    solution_column_buffer: torch.Tensor
    factor_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    sample_times: torch.Tensor | None = None
    source_values: dict[str, torch.Tensor] = field(default_factory=dict)
    switch_keys: tuple[tuple[bool, ...], ...] = ()
    integration_keys: tuple[str, ...] = ()
    node_samples: torch.Tensor | None = None
    branch_samples: torch.Tensor | None = None
    device_power_samples: dict[str, torch.Tensor] = field(default_factory=dict)
    energy_balance_samples: torch.Tensor | None = None
    port_power_samples: torch.Tensor | None = None
    field_energy_change_samples: torch.Tensor | None = None
    previous_stored_energy: torch.Tensor | None = None
    step_index: int = 0

    @property
    def physical_devices(self):
        return tuple(device for device in self.circuit.devices if not isinstance(device, MutualInductor))

    @property
    def physical_branch_names(self) -> tuple[str, ...]:
        return tuple(device.name for device in self.physical_devices)

    def _stored_energy(self, state: CircuitState) -> torch.Tensor:
        energy = self.initial_solution.new_zeros(())
        for device in self.circuit.devices:
            if isinstance(device, Capacitor):
                voltage = state.capacitor_voltage[device.name]
                energy = energy + 0.5 * self.system._value(
                    device.capacitance,
                    device.name,
                ) * voltage.square()
        if state.inductor_current.numel():
            energy = energy + 0.5 * torch.dot(
                state.inductor_current,
                self.system.plan.inductance_matrix @ state.inductor_current,
            )
        return energy

    def _sample_field_voltage(self, port: CircuitPortRuntime) -> torch.Tensor:
        field_tensor = getattr(port.solver, port.field_name)
        flat = field_tensor.view(-1)
        torch.index_select(
            flat,
            0,
            port.field.linear_indices,
            out=port.field.edge_buffer,
        )
        torch.mul(
            port.field.edge_buffer,
            port.field.voltage_weights,
            out=port.field.edge_buffer,
        )
        torch.sum(port.field.edge_buffer, dim=0, out=port.field.last_voltage_before)
        return port.field.last_voltage_before

    def _apply_field_current(
        self,
        port: CircuitPortRuntime,
        current: torch.Tensor,
        voltage: torch.Tensor,
    ) -> None:
        field_tensor = getattr(port.solver, port.field_name)
        torch.mul(port.field.injection, current, out=port.field.correction_buffer)
        field_tensor.view(-1).index_add_(
            0,
            port.field.linear_indices,
            port.field.correction_buffer,
            alpha=-1.0,
        )
        port.field.last_current.copy_(current)
        port.field.last_voltage.copy_(voltage)
        port.field.last_voltage_after.copy_(port.field.last_voltage_before)
        port.field.last_voltage_after.addcmul_(
            port.field.coupling_impedance,
            current,
            value=-1.0,
        )
        port.field.last_port_work.copy_(voltage)
        port.field.last_port_work.mul_(current)
        port.field.last_port_work.mul_(port.field.dt)
        port.field.last_field_energy_change.copy_(port.field.last_voltage_before)
        port.field.last_field_energy_change.addcmul_(
            port.field.coupling_impedance,
            current,
            value=-0.5,
        )
        port.field.last_field_energy_change.mul_(current)
        port.field.last_field_energy_change.mul_(port.field.dt)
        port.field.last_field_energy_change.neg_()

    def _midpoint_currents(
        self,
        solution: torch.Tensor,
        previous_state: CircuitState,
        integration: str,
        source_index: int,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if integration != "trapezoidal":
            next_state = self.system._update_state(previous_state, solution, integration=integration)
            return self.system._device_currents(
                solution,
                next_state,
                self.source_values,
                source_index,
                time,
            )
        capacitor_currents = {}
        for device in self.circuit.devices:
            if isinstance(device, Capacitor):
                voltage = _terminal_voltage(
                    solution,
                    self.system.graph,
                    device.positive,
                    device.negative,
                )
                capacitor_currents[device.name] = (
                    2.0
                    * self.system._value(device.capacitance, device.name)
                    / self.system.dt
                    * (voltage - previous_state.capacitor_voltage[device.name])
                )
        measurement_state = CircuitState(
            capacitor_voltage=previous_state.capacitor_voltage,
            capacitor_current=capacitor_currents,
            inductor_current=previous_state.inductor_current,
            inductor_voltage=previous_state.inductor_voltage,
        )
        return self.system._device_currents(
            solution,
            measurement_state,
            self.source_values,
            source_index,
            time,
        )

    def _record_initial(self) -> None:
        self.node_samples[0].zero_()
        self.node_samples[0, 1:].copy_(
            self.initial_solution[: len(self.system.graph.nodes) - 1]
        )
        dc_sources = {
            name: value.reshape(1)
            for name, value in self.system._dc_source_values().items()
        }
        currents = self.system._device_currents(
            self.initial_solution,
            self.state,
            dc_sources,
            0,
            self.sample_times[0],
        )
        for index, device in enumerate(self.physical_devices):
            self.branch_samples[0, index].copy_(currents[device.name])
        for device in self.circuit.devices:
            if isinstance(device, MutualInductor):
                power = self.initial_solution.new_zeros(())
            else:
                voltage = _terminal_voltage(
                    self.initial_solution,
                    self.system.graph,
                    device.positive,
                    device.negative,
                )
                power = voltage * currents[device.name]
            self.device_power_samples[device.name][0].copy_(power)
        self.previous_stored_energy = self._stored_energy(self.state)

    def prepare_time_series(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Circuit-coupled FDTD requires a positive integer step count.")
        dtype = self.system.dtype
        device = self.system.device
        default_integration = self.circuit.config.integration
        has_internal_time_event = any(
            isinstance(device, TimedSwitch)
            or getattr(device, "waveform", None) is not None
            for device in self.circuit.devices
        )
        backward_euler_steps = (
            set(self.system._local_backward_euler_steps(steps))
            if has_internal_time_event
            else set()
        )
        self.integration_keys = tuple(
            "backward_euler"
            if index + 1 in backward_euler_steps
            else default_integration
            for index in range(steps)
        )
        fractions = torch.as_tensor(
            tuple(
                0.5 if integration == "trapezoidal" else 1.0
                for integration in self.integration_keys
            ),
            device=device,
            dtype=dtype,
        )
        transient_times = (
            torch.arange(steps, device=device, dtype=dtype) + fractions
        ) * self.system.dt
        self.sample_times = torch.cat((transient_times.new_zeros((1,)), transient_times))
        self.source_values = self.system._source_values(transient_times)
        self.switch_keys = self.system._switch_state_keys(transient_times)
        self.node_samples = torch.empty(
            (steps + 1, len(self.system.graph.nodes)),
            device=device,
            dtype=dtype,
        )
        self.branch_samples = torch.empty(
            (steps + 1, len(self.physical_branch_names)),
            device=device,
            dtype=dtype,
        )
        self.device_power_samples = {
            device_item.name: torch.empty((steps + 1,), device=device, dtype=dtype)
            for device_item in self.circuit.devices
        }
        self.energy_balance_samples = torch.zeros((steps + 1,), device=device, dtype=dtype)
        self.port_power_samples = torch.zeros((steps + 1, len(self.ports)), device=device, dtype=dtype)
        self.field_energy_change_samples = torch.zeros(
            (steps + 1, len(self.ports)),
            device=device,
            dtype=dtype,
        )
        self.factor_cache.clear()
        self.step_index = 0
        self.step_tensor.zero_()
        self._record_initial()

        representatives = {}
        for index, (integration, switch_key) in enumerate(
            zip(self.integration_keys, self.switch_keys)
        ):
            representatives.setdefault((integration, switch_key), index)
        zero_voltage = self.initial_solution.new_zeros(())
        for factor_key, index in representatives.items():
            integration = factor_key[0]
            matrix, rhs = self.system.assemble_transient(
                self.state,
                time=transient_times[index],
                source_values=self.source_values,
                step_index=index,
                integration=integration,
                midpoint=integration == "trapezoidal",
                out=(self.matrix_buffer, self.rhs_buffer),
            )
            for port in self.ports:
                _stamp_norton(
                    matrix,
                    rhs,
                    self.system.graph,
                    port.binding,
                    port.conductance(integration),
                    zero_voltage,
                )
            self.factor_cache[factor_key] = self.system._factor(matrix)

    def apply(self) -> None:
        if self.sample_times is None or self.step_index >= self.sample_times.numel() - 1:
            raise RuntimeError("Circuit time-series buffers were not prepared for this FDTD step.")
        index = self.step_index
        integration = self.integration_keys[index]
        time = self.sample_times[index + 1]
        free_voltages = tuple(self._sample_field_voltage(port) for port in self.ports)
        previous_state = self.state
        matrix, rhs = self.system.assemble_transient(
            previous_state,
            time=time,
            source_values=self.source_values,
            step_index=index,
            integration=integration,
            midpoint=integration == "trapezoidal",
            out=(self.matrix_buffer, self.rhs_buffer),
            stamp_matrix=False,
        )
        for port, free_voltage in zip(self.ports, free_voltages):
            _stamp_norton(
                matrix,
                rhs,
                self.system.graph,
                port.binding,
                port.conductance(integration),
                free_voltage,
                stamp_matrix=False,
            )
        factor_key = (integration, self.switch_keys[index])
        factors, pivots, condition = self.factor_cache[factor_key]
        torch.linalg.lu_solve(
            factors,
            pivots,
            rhs.unsqueeze(-1),
            out=self.solution_column_buffer,
        )
        solution = self.solution_column_buffer.squeeze(-1)
        currents = self._midpoint_currents(
            solution,
            previous_state,
            integration,
            index,
            time,
        )
        self.state = self.system._update_state(
            previous_state,
            solution,
            integration=integration,
            midpoint=integration == "trapezoidal",
        )

        port_input_power = solution.new_zeros(())
        for port_index, (port, free_voltage) in enumerate(zip(self.ports, free_voltages)):
            port.last_integration = integration
            voltage = _terminal_voltage(
                solution,
                self.system.graph,
                port.binding.positive,
                port.binding.negative,
            )
            current = port.conductance(integration) * (free_voltage - voltage)
            self._apply_field_current(port, current, voltage)
            power = voltage * current
            self.port_power_samples[index + 1, port_index].copy_(power)
            self.field_energy_change_samples[index + 1, port_index].copy_(
                port.field.last_field_energy_change
            )
            port_input_power = port_input_power + power

        self.node_samples[index + 1].zero_()
        self.node_samples[index + 1, 1:].copy_(
            solution[: len(self.system.graph.nodes) - 1]
        )
        for branch_index, device in enumerate(self.physical_devices):
            self.branch_samples[index + 1, branch_index].copy_(currents[device.name])
        nonreactive_power = solution.new_zeros(())
        for device in self.circuit.devices:
            if isinstance(device, MutualInductor):
                power = solution.new_zeros(())
            else:
                voltage = _terminal_voltage(
                    solution,
                    self.system.graph,
                    device.positive,
                    device.negative,
                )
                power = voltage * currents[device.name]
            self.device_power_samples[device.name][index + 1].copy_(power)
            if not isinstance(device, (Capacitor, Inductor, MutualInductor)):
                nonreactive_power = nonreactive_power + power
        stored_energy = self._stored_energy(self.state)
        self.energy_balance_samples[index + 1].copy_(
            self.energy_balance_samples[index]
            + stored_energy
            - self.previous_stored_energy
            + self.system.dt * (nonreactive_power - port_input_power)
        )
        self.previous_stored_energy = stored_energy
        self.step_index += 1
        self.step_tensor.add_(1)
        self.last_condition = condition

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"step": self.step_tensor}
        for name, value in self.state.capacitor_voltage.items():
            tensors[f"capacitor_voltage_{name}"] = value
        for name, value in self.state.capacitor_current.items():
            tensors[f"capacitor_current_{name}"] = value
        tensors["inductor_current"] = self.state.inductor_current
        tensors["inductor_voltage"] = self.state.inductor_voltage
        return tensors

    def finalize(self) -> CircuitData:
        count = self.step_index + 1
        return CircuitData(
            circuit_name=self.circuit.name,
            times=self.sample_times[:count],
            node_names=tuple(node.name for node in self.system.graph.nodes),
            node_voltages=self.node_samples[:count],
            branch_names=self.physical_branch_names,
            branch_currents=self.branch_samples[:count],
            device_powers={
                name: values[:count] for name, values in self.device_power_samples.items()
            },
            energy_balance=self.energy_balance_samples[:count],
            diagnostics={
                "integration": self.circuit.config.integration,
                "dt": self.system.dt,
                "dc_condition": self.dc_condition,
                "last_condition": getattr(self, "last_condition", self.dc_condition),
                "solve_count": self.step_index,
                "factorization_count": len(self.factor_cache),
                "bound_ports": tuple(port.binding.port_name for port in self.ports),
                "port_powers": self.port_power_samples[:count],
                "field_energy_changes": self.field_energy_change_samples[:count],
                "field_energy_change_total": self.field_energy_change_samples[:count].sum(dim=1),
                "device": str(self.system.device),
                "dtype": str(self.system.dtype),
            },
        )


def _initial_coupled_solution(system, ports):
    matrix, rhs = system.assemble_dc()
    zero = rhs.new_zeros(())
    integration = system.circuit.config.integration
    for port in ports:
        _stamp_norton(
            matrix,
            rhs,
            system.graph,
            port.binding,
            port.conductance(integration),
            zero,
        )
    solution, condition = system._solve(matrix, rhs)
    constraint_system = system._initial_constraint_system()
    if constraint_system is None:
        return solution, condition
    rows, targets = constraint_system
    count = rows.shape[0]
    augmented = matrix.new_zeros((system.unknown_count + count, system.unknown_count + count))
    augmented[: system.unknown_count, : system.unknown_count] = matrix
    augmented[: system.unknown_count, system.unknown_count :] = rows.transpose(0, 1)
    augmented[system.unknown_count :, : system.unknown_count] = rows
    constrained, condition = system._solve(augmented, torch.cat((rhs, targets)))
    return constrained[: system.unknown_count], condition


def prepare_circuit_runtimes(solver, port_runtimes) -> tuple[EMCircuitRuntime, ...]:
    circuits = tuple(getattr(solver.scene, "circuits", ()))
    if not circuits:
        solver._circuit_runtimes = ()
        return ()
    runtime_by_name = {runtime.port.name: runtime for runtime in port_runtimes}
    results = []
    for circuit in circuits:
        system = compile_coupled_mna_system(
            circuit,
            dt=solver.dt,
            device=solver.device,
            dtype=solver.Ex.dtype,
        )
        ports = []
        for binding in circuit.bindings:
            port_runtime = runtime_by_name[binding.port_name]
            geometry = port_runtime.geometry
            if port_runtime.yee_control_volume is None:
                raise RuntimeError(
                    f"Circuit-bound port {binding.port_name!r} has no Yee control volume."
                )
            field = prepare_field_port_coupling(
                geometry,
                dt=solver.dt,
                eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
                yee_control_volume=port_runtime.yee_control_volume,
            )
            port = CircuitPortRuntime(
                circuit_name=circuit.name,
                integration=circuit.config.integration,
                binding=binding,
                solver=solver,
                field_name=port_runtime.field_name,
                field=field,
                last_integration=circuit.config.integration,
                trapezoidal_conductance=torch.reciprocal(
                    field.discrete_port_impedance
                ),
                backward_euler_conductance=torch.reciprocal(
                    field.coupling_impedance
                ),
            )
            port_runtime.circuit_port = port
            ports.append(port)
        initial_solution, condition = _initial_coupled_solution(system, tuple(ports))
        zero = initial_solution.new_zeros(())
        initial_port_currents = tuple(
            port.conductance(circuit.config.integration)
            * (
                zero
                - _terminal_voltage(
                    initial_solution,
                    system.graph,
                    port.binding.positive,
                    port.binding.negative,
                )
            )
            for port in ports
        )
        if any(
            not bool(torch.isclose(current, zero, rtol=1.0e-5, atol=1.0e-12))
            for current in initial_port_currents
        ):
            raise ValueError(
                f"Circuit {circuit.name!r} has a nonzero DC port current, which is "
                "inconsistent with the zero initial Yee field; ramp the source from zero "
                "or provide a zero-current coupled operating point."
            )
        runtime = EMCircuitRuntime(
            circuit=circuit,
            system=system,
            ports=tuple(ports),
            state=system._initial_state(initial_solution),
            initial_solution=initial_solution,
            dc_condition=condition,
            step_tensor=torch.zeros((), device=solver.device, dtype=torch.int64),
            matrix_buffer=torch.empty(
                (system.unknown_count, system.unknown_count),
                device=solver.device,
                dtype=solver.Ex.dtype,
            ),
            rhs_buffer=torch.empty(
                (system.unknown_count,),
                device=solver.device,
                dtype=solver.Ex.dtype,
            ),
            solution_column_buffer=torch.empty(
                (system.unknown_count, 1),
                device=solver.device,
                dtype=solver.Ex.dtype,
            ),
        )
        results.append(runtime)
    solver._circuit_runtimes = tuple(results)
    return solver._circuit_runtimes


def prepare_circuit_time_series(solver, steps: int) -> None:
    for runtime in getattr(solver, "_circuit_runtimes", ()):
        runtime.prepare_time_series(steps)


def apply_circuit_runtimes(solver) -> None:
    for runtime in getattr(solver, "_circuit_runtimes", ()):
        runtime.apply()


def finalize_circuit_data(solver) -> dict[str, CircuitData]:
    return {
        runtime.circuit.name: runtime.finalize()
        for runtime in getattr(solver, "_circuit_runtimes", ())
    }


__all__ = [
    "CircuitPortRuntime",
    "EMCircuitRuntime",
    "apply_circuit_runtimes",
    "finalize_circuit_data",
    "prepare_circuit_runtimes",
    "prepare_circuit_time_series",
]

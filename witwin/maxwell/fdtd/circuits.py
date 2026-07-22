from __future__ import annotations

from collections.abc import Mapping
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
from ..lumped import Capacitor, Inductor, Resistor
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
class _RCFastPlan:
    device_incidence: torch.Tensor
    capacitor_incidence: torch.Tensor
    capacitor_device_indices: torch.Tensor
    capacitances: torch.Tensor
    resistor_conductances: torch.Tensor
    resistor_mask: torch.Tensor
    capacitor_voltage: torch.Tensor
    capacitor_current: torch.Tensor


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
    device_power_sample_matrix: torch.Tensor | None = None
    energy_balance_samples: torch.Tensor | None = None
    port_power_samples: torch.Tensor | None = None
    field_energy_change_samples: torch.Tensor | None = None
    previous_stored_energy: torch.Tensor | None = None
    running_energy_balance: torch.Tensor | None = None
    step_index: int = 0
    graph_eligible: bool = False
    rc_fast_plan: _RCFastPlan | None = None

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

    def _coupling_voltage(
        self, port: CircuitPortRuntime, integration: str
    ) -> torch.Tensor:
        """Trapezoidal half-step port voltage fed into the field-port Norton source.

        Coordinator unification ruling (slice U1): the trapezoidal interface couples
        through 0.5*(V_after_prev + V_free), matching the native lumped runtime's
        averaged constitutive relation so cross-model agreement holds at the unified
        convention. The result is staged in a persistent scalar buffer so the
        CUDA-graph capture keeps a stable pointer. Backward-Euler steps keep the
        end-of-step V_free voltage, consistent with their full-step companion
        impedance. ``last_voltage_before`` stays the raw free voltage for the
        post-step recurrence and the field-side energy audit.
        """
        field = port.field
        if integration == "trapezoidal":
            field.coupling_voltage.copy_(field.last_voltage_after)
            field.coupling_voltage.add_(field.last_voltage_before)
            field.coupling_voltage.mul_(0.5)
        else:
            field.coupling_voltage.copy_(field.last_voltage_before)
        return field.coupling_voltage

    def _apply_field_current(
        self,
        port: CircuitPortRuntime,
        current: torch.Tensor,
        voltage: torch.Tensor,
        *,
        scatter_field: bool = True,
    ) -> None:
        if scatter_field:
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
        # Slice U1 (coordinator ruling): the field/port power audit uses the shared
        # exchange voltage -- the port terminal midpoint ``voltage`` = coupling_voltage
        # - Z_d I at which current I flows across the port -- so the field-energy
        # change equals minus the port work exactly under the unified trapezoidal
        # interface. (Pre-unification both sides shared V_free, so this reduced to the
        # V_free midpoint; the terminal voltage is the convention-independent statement.)
        port.field.last_field_energy_change.copy_(voltage)
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
        powers = []
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
            powers.append(power)
        power_row = torch.stack(powers) if powers else self.initial_solution.new_zeros((0,))
        self.device_power_sample_matrix[0].copy_(power_row)
        self.previous_stored_energy.copy_(self._stored_energy(self.state))
        self.running_energy_balance.zero_()

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
        self.device_power_sample_matrix = torch.empty(
            (steps + 1, len(self.circuit.devices)),
            device=device,
            dtype=dtype,
        )
        self.device_power_samples = {
            device_item.name: self.device_power_sample_matrix[:, index]
            for index, device_item in enumerate(self.circuit.devices)
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
        self.previous_stored_energy = self.initial_solution.new_zeros(())
        self.running_energy_balance = self.initial_solution.new_zeros(())
        # Built-in switch/source schedules are fully precomputed on the device.
        # A separate graph is captured for each integration/switch factor class.
        self.graph_eligible = not any(
            value.requires_grad for value in self.system._parameter_cache.values()
        )
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

    def _copy_state_(self, source: CircuitState) -> None:
        for name, value in source.capacitor_voltage.items():
            self.state.capacitor_voltage[name].copy_(value)
        for name, value in source.capacitor_current.items():
            self.state.capacitor_current[name].copy_(value)
        self.state.inductor_current.copy_(source.inductor_current)
        self.state.inductor_voltage.copy_(source.inductor_voltage)

    def _write_step_samples(
        self,
        index: int,
        *,
        device_indexed_samples: bool,
        node_row: torch.Tensor,
        branch_row: torch.Tensor,
        device_power_row: torch.Tensor,
        port_power_row: torch.Tensor,
        field_energy_row: torch.Tensor,
    ) -> None:
        if device_indexed_samples:
            sample_index = (self.step_tensor + 1).reshape(1)
            self.node_samples.index_copy_(0, sample_index, node_row.unsqueeze(0))
            self.branch_samples.index_copy_(0, sample_index, branch_row.unsqueeze(0))
            self.device_power_sample_matrix.index_copy_(
                0,
                sample_index,
                device_power_row.unsqueeze(0),
            )
            self.energy_balance_samples.index_copy_(
                0,
                sample_index,
                self.running_energy_balance.reshape(1),
            )
            self.port_power_samples.index_copy_(
                0,
                sample_index,
                port_power_row.unsqueeze(0),
            )
            self.field_energy_change_samples.index_copy_(
                0,
                sample_index,
                field_energy_row.unsqueeze(0),
            )
            return
        sample_index = index + 1
        self.node_samples[sample_index].copy_(node_row)
        self.branch_samples[sample_index].copy_(branch_row)
        self.device_power_sample_matrix[sample_index].copy_(device_power_row)
        self.energy_balance_samples[sample_index].copy_(self.running_energy_balance)
        self.port_power_samples[sample_index].copy_(port_power_row)
        self.field_energy_change_samples[sample_index].copy_(field_energy_row)

    def _apply_rc_step(
        self,
        index: int,
        *,
        integration: str,
        free_voltages: tuple[torch.Tensor, ...],
        device_indexed_samples: bool,
        scatter_field_currents: bool,
    ) -> tuple[torch.Tensor, ...]:
        plan = self.rc_fast_plan
        factor = 2.0 if integration == "trapezoidal" else 1.0
        capacitor_conductance = factor * plan.capacitances / self.system.dt
        torch.mv(
            plan.capacitor_incidence.transpose(0, 1),
            capacitor_conductance * plan.capacitor_voltage,
            out=self.rhs_buffer,
        )
        for port, free_voltage in zip(self.ports, free_voltages):
            _stamp_norton(
                self.matrix_buffer,
                self.rhs_buffer,
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
            self.rhs_buffer.unsqueeze(-1),
            out=self.solution_column_buffer,
        )
        solution = self.solution_column_buffer.squeeze(-1)
        device_voltages = torch.mv(plan.device_incidence, solution)
        capacitor_voltages = torch.index_select(
            device_voltages,
            0,
            plan.capacitor_device_indices,
        )
        capacitor_currents = capacitor_conductance * (
            capacitor_voltages - plan.capacitor_voltage
        )
        if integration == "trapezoidal":
            next_capacitor_voltage = 2.0 * capacitor_voltages - plan.capacitor_voltage
            next_capacitor_current = 2.0 * capacitor_currents - plan.capacitor_current
        else:
            next_capacitor_voltage = capacitor_voltages
            next_capacitor_current = capacitor_currents
        device_currents = device_voltages * plan.resistor_conductances
        device_currents.index_copy_(
            0,
            plan.capacitor_device_indices,
            capacitor_currents,
        )

        port_powers = []
        field_energy_changes = []
        for port, free_voltage in zip(self.ports, free_voltages):
            port.last_integration = integration
            voltage = _terminal_voltage(
                solution,
                self.system.graph,
                port.binding.positive,
                port.binding.negative,
            )
            current = port.conductance(integration) * (free_voltage - voltage)
            self._apply_field_current(
                port,
                current,
                voltage,
                scatter_field=scatter_field_currents,
            )
            port_powers.append(voltage * current)
            field_energy_changes.append(port.field.last_field_energy_change)

        port_power_row = torch.stack(port_powers)
        field_energy_row = torch.stack(field_energy_changes)
        device_power_row = device_voltages * device_currents
        nonreactive_power = torch.dot(device_power_row, plan.resistor_mask)
        stored_energy = 0.5 * torch.dot(
            plan.capacitances,
            next_capacitor_voltage.square(),
        )
        port_input_power = port_power_row.sum()
        balance_increment = (
            stored_energy
            - self.previous_stored_energy
            + self.system.dt * (nonreactive_power - port_input_power)
        )
        previous_balance = (
            torch.index_select(
                self.energy_balance_samples,
                0,
                self.step_tensor.reshape(1),
            ).squeeze(0)
            if device_indexed_samples
            else self.energy_balance_samples[index]
        )
        self.running_energy_balance.copy_(previous_balance + balance_increment)
        self.previous_stored_energy.copy_(stored_energy)
        plan.capacitor_voltage.copy_(next_capacitor_voltage)
        plan.capacitor_current.copy_(next_capacitor_current)
        node_row = torch.cat(
            (
                solution.new_zeros((1,)),
                solution[: len(self.system.graph.nodes) - 1],
            )
        )
        self._write_step_samples(
            index,
            device_indexed_samples=device_indexed_samples,
            node_row=node_row,
            branch_row=device_currents,
            device_power_row=device_power_row,
            port_power_row=port_power_row,
            field_energy_row=field_energy_row,
        )
        self.step_tensor.add_(1)
        self.last_condition = condition
        return tuple(port.field.last_current for port in self.ports)

    def _apply_step(
        self,
        index: int,
        *,
        device_indexed_samples: bool,
        free_voltages: tuple[torch.Tensor, ...] | None = None,
        scatter_field_currents: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        integration = self.integration_keys[index]
        time = self.sample_times[index + 1]
        source_index = self.step_tensor if device_indexed_samples else index
        if free_voltages is None:
            for port in self.ports:
                self._sample_field_voltage(port)
        else:
            for port, value in zip(self.ports, free_voltages):
                port.field.last_voltage_before.copy_(value)
        # Slice U1 (coordinator unification ruling): couple the field port through
        # the trapezoidal half-step averaged voltage 0.5*(V_after_prev + V_free),
        # the same interface the native lumped runtime uses, re-establishing exact
        # cross-model agreement at the unified (unconditionally stable) convention.
        # The raw ``last_voltage_before`` is preserved for the post-step recurrence
        # and the field-side energy audit.
        free_voltages = tuple(
            self._coupling_voltage(port, integration) for port in self.ports
        )
        if self.rc_fast_plan is not None:
            return self._apply_rc_step(
                index,
                integration=integration,
                free_voltages=free_voltages,
                device_indexed_samples=device_indexed_samples,
                scatter_field_currents=scatter_field_currents,
            )
        previous_state = self.state
        matrix, rhs = self.system.assemble_transient(
            previous_state,
            time=time,
            source_values=self.source_values,
            step_index=source_index,
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
            source_index,
            time,
        )
        next_state = self.system._update_state(
            previous_state,
            solution,
            integration=integration,
            midpoint=integration == "trapezoidal",
        )

        port_powers = []
        field_energy_changes = []
        for port, free_voltage in zip(self.ports, free_voltages):
            port.last_integration = integration
            voltage = _terminal_voltage(
                solution,
                self.system.graph,
                port.binding.positive,
                port.binding.negative,
            )
            current = port.conductance(integration) * (free_voltage - voltage)
            self._apply_field_current(
                port,
                current,
                voltage,
                scatter_field=scatter_field_currents,
            )
            port_powers.append(voltage * current)
            field_energy_changes.append(port.field.last_field_energy_change)

        node_row = torch.cat(
            (
                solution.new_zeros((1,)),
                solution[: len(self.system.graph.nodes) - 1],
            )
        )
        branch_row = (
            torch.stack(tuple(currents[device.name] for device in self.physical_devices))
            if self.physical_devices
            else solution.new_zeros((0,))
        )
        device_powers = []
        nonreactive_powers = []
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
            device_powers.append(power)
            if not isinstance(device, (Capacitor, Inductor, MutualInductor)):
                nonreactive_powers.append(power)
        port_power_row = torch.stack(port_powers)
        field_energy_row = torch.stack(field_energy_changes)
        device_power_row = (
            torch.stack(device_powers)
            if device_powers
            else solution.new_zeros((0,))
        )
        nonreactive_power = (
            torch.stack(nonreactive_powers).sum()
            if nonreactive_powers
            else solution.new_zeros(())
        )
        port_input_power = port_power_row.sum()
        stored_energy = self._stored_energy(next_state)
        balance_increment = (
            stored_energy
            - self.previous_stored_energy
            + self.system.dt * (nonreactive_power - port_input_power)
        )
        previous_balance = (
            torch.index_select(
                self.energy_balance_samples,
                0,
                self.step_tensor.reshape(1),
            ).squeeze(0)
            if device_indexed_samples
            else self.energy_balance_samples[index]
        )
        self.running_energy_balance.copy_(previous_balance + balance_increment)
        self.previous_stored_energy.copy_(stored_energy)
        self._copy_state_(next_state)

        self._write_step_samples(
            index,
            device_indexed_samples=device_indexed_samples,
            node_row=node_row,
            branch_row=branch_row,
            device_power_row=device_power_row,
            port_power_row=port_power_row,
            field_energy_row=field_energy_row,
        )
        self.step_tensor.add_(1)
        self.last_condition = condition
        return tuple(port.field.last_current for port in self.ports)

    def _validate_free_voltages(
        self,
        free_voltages: tuple[torch.Tensor, ...],
    ) -> None:
        if not isinstance(free_voltages, tuple) or len(free_voltages) != len(self.ports):
            raise ValueError("free_voltages must contain one scalar tensor per bound port.")
        for value in free_voltages:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError("free_voltages must contain one scalar tensor per bound port.")
            if value.device != self.system.device or value.dtype != self.system.dtype:
                raise ValueError(
                    "free_voltages must match the compiled circuit device and dtype."
                )

    def apply(
        self,
        *,
        free_voltages: tuple[torch.Tensor, ...] | None = None,
        device_indexed_samples: bool = False,
    ) -> None:
        if self.sample_times is None or self.step_index >= self.sample_times.numel() - 1:
            raise RuntimeError("Circuit time-series buffers were not prepared for this FDTD step.")
        if free_voltages is not None:
            self._validate_free_voltages(free_voltages)
        self._apply_step(
            self.step_index,
            device_indexed_samples=device_indexed_samples,
            free_voltages=free_voltages,
        )
        self.step_index += 1

    def apply_external(
        self,
        free_voltages: tuple[torch.Tensor, ...],
        *,
        apply_field_currents: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Advance from owner-gathered Norton voltages and optionally scatter locally."""

        if self.sample_times is None or self.step_index >= self.sample_times.numel() - 1:
            raise RuntimeError("Circuit time-series buffers were not prepared for this FDTD step.")
        self._validate_free_voltages(free_voltages)
        currents = self._apply_step(
            self.step_index,
            device_indexed_samples=False,
            free_voltages=free_voltages,
            scatter_field_currents=apply_field_currents,
        )
        self.step_index += 1
        return currents

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"step": self.step_tensor}
        for name, value in self.state.capacitor_voltage.items():
            tensors[f"capacitor_voltage_{name}"] = value
        for name, value in self.state.capacitor_current.items():
            tensors[f"capacitor_current_{name}"] = value
        tensors["inductor_current"] = self.state.inductor_current
        tensors["inductor_voltage"] = self.state.inductor_voltage
        # Slice U1: the trapezoidal field-port interface reads the previous step's
        # post-step port voltage, so it is carried scalar state that resume must
        # restore for a bit-exact continuation.
        for index, port in enumerate(self.ports):
            tensors[f"port_{index}_last_voltage_after"] = port.field.last_voltage_after
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


def _compile_rc_fast_plan(runtime: EMCircuitRuntime) -> _RCFastPlan | None:
    devices = tuple(runtime.circuit.devices)
    capacitors = tuple(device for device in devices if isinstance(device, Capacitor))
    if (
        not capacitors
        or any(not isinstance(device, (Resistor, Capacitor)) for device in devices)
        or any(value.requires_grad for value in runtime.system._parameter_cache.values())
    ):
        return None
    unknown_count = runtime.system.unknown_count

    def incidence_row(device) -> list[float]:
        row = [0.0] * unknown_count
        positive = _node_unknown(runtime.system.graph, device.positive)
        negative = _node_unknown(runtime.system.graph, device.negative)
        if positive is not None:
            row[positive] += 1.0
        if negative is not None:
            row[negative] -= 1.0
        return row

    device_incidence = runtime.initial_solution.new_tensor(
        tuple(incidence_row(device) for device in devices)
    )
    capacitor_indices = tuple(
        index for index, device in enumerate(devices) if isinstance(device, Capacitor)
    )
    capacitor_device_indices = torch.as_tensor(
        capacitor_indices,
        device=runtime.system.device,
        dtype=torch.int64,
    )
    capacitor_incidence = torch.index_select(
        device_incidence,
        0,
        capacitor_device_indices,
    )
    capacitances = torch.stack(
        tuple(runtime.system._value(device.capacitance, device.name) for device in capacitors)
    )
    zero = runtime.initial_solution.new_zeros(())
    resistor_conductances = torch.stack(
        tuple(
            torch.reciprocal(runtime.system._value(device.resistance, device.name))
            if isinstance(device, Resistor)
            else zero
            for device in devices
        )
    )
    resistor_mask = runtime.initial_solution.new_tensor(
        tuple(1.0 if isinstance(device, Resistor) else 0.0 for device in devices)
    )
    capacitor_voltage = torch.stack(
        tuple(runtime.state.capacitor_voltage[device.name] for device in capacitors)
    ).clone()
    capacitor_current = torch.stack(
        tuple(runtime.state.capacitor_current[device.name] for device in capacitors)
    ).clone()
    runtime.state.capacitor_voltage = {
        device.name: capacitor_voltage[index]
        for index, device in enumerate(capacitors)
    }
    runtime.state.capacitor_current = {
        device.name: capacitor_current[index]
        for index, device in enumerate(capacitors)
    }
    return _RCFastPlan(
        device_incidence=device_incidence,
        capacitor_incidence=capacitor_incidence,
        capacitor_device_indices=capacitor_device_indices,
        capacitances=capacitances,
        resistor_conductances=resistor_conductances,
        resistor_mask=resistor_mask,
        capacitor_voltage=capacitor_voltage,
        capacitor_current=capacitor_current,
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


def prepare_circuit_runtimes(
    solver,
    port_runtimes,
    *,
    circuits=None,
    field_couplings: Mapping[str, FieldPortCoupling] | None = None,
) -> tuple[EMCircuitRuntime, ...]:
    circuits = (
        tuple(getattr(solver.scene, "circuits", ()))
        if circuits is None
        else tuple(circuits)
    )
    field_couplings = {} if field_couplings is None else field_couplings
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
            field = field_couplings.get(binding.port_name)
            if field is None:
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
        runtime.rc_fast_plan = _compile_rc_fast_plan(runtime)
        results.append(runtime)
    solver._circuit_runtimes = tuple(results)
    return solver._circuit_runtimes


def prepare_circuit_time_series(solver, steps: int) -> None:
    for runtime in getattr(solver, "_circuit_runtimes", ()):
        runtime.prepare_time_series(steps)


def _graph_mutable_tensors(runtime: EMCircuitRuntime) -> tuple[torch.Tensor, ...]:
    tensors = [
        runtime.step_tensor,
        runtime.matrix_buffer,
        runtime.rhs_buffer,
        runtime.solution_column_buffer,
        runtime.node_samples,
        runtime.branch_samples,
        runtime.device_power_sample_matrix,
        runtime.energy_balance_samples,
        runtime.port_power_samples,
        runtime.field_energy_change_samples,
        runtime.previous_stored_energy,
        runtime.running_energy_balance,
        runtime.state.inductor_current,
        runtime.state.inductor_voltage,
        *runtime.state.capacitor_voltage.values(),
        *runtime.state.capacitor_current.values(),
    ]
    field_names = set()
    for port in runtime.ports:
        field_names.add(port.field_name)
        tensors.extend(
            (
                port.field.edge_buffer,
                port.field.correction_buffer,
                port.field.last_voltage_before,
                port.field.coupling_voltage,
                port.field.last_voltage,
                port.field.last_voltage_after,
                port.field.last_current,
                port.field.last_port_work,
                port.field.last_field_energy_change,
            )
        )
    tensors.extend(getattr(runtime.ports[0].solver, name) for name in sorted(field_names))
    unique = []
    pointers = set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        pointer = tensor.data_ptr()
        if pointer not in pointers:
            pointers.add(pointer)
            unique.append(tensor)
    return tuple(unique)


def prepare_circuit_graph_runners(solver, use_cuda_graph: bool) -> None:
    """Capture fixed-schedule circuit steps without changing the Scene contract."""

    runtimes = tuple(getattr(solver, "_circuit_runtimes", ()))
    solver._circuit_graph_runners = ()
    solver._circuit_graph_active = False
    solver._circuit_graph_error = None
    if not use_cuda_graph or not runtimes or not all(runtime.graph_eligible for runtime in runtimes):
        return

    from .cuda.runtime.graph import CudaGraphRunner

    replay_maps = []
    for runtime in runtimes:
        mutable = _graph_mutable_tensors(runtime)
        saved = tuple(tensor.clone() for tensor in mutable)
        saved_step_index = runtime.step_index

        def restore() -> None:
            for tensor, value in zip(mutable, saved):
                tensor.copy_(value)
            runtime.step_index = saved_step_index

        representatives = {}
        for index, factor_key in enumerate(zip(runtime.integration_keys, runtime.switch_keys)):
            representatives.setdefault(factor_key, index)
        runtime_replays = {}
        for factor_key, representative in representatives.items():
            try:
                # Warm the dense solve and allocator before capture. State and samples
                # are restored immediately, so the physical run still begins at step zero.
                runtime._apply_step(representative, device_indexed_samples=True)
                torch.cuda.synchronize(device=runtime.system.device)
                restore()
                runtime_replays[factor_key] = CudaGraphRunner(
                    device=runtime.system.device,
                    enabled=True,
                    warmup_steps=0,
                ).capture(
                    lambda runtime=runtime, representative=representative: runtime._apply_step(
                        representative,
                        device_indexed_samples=True,
                    )
                )
            except Exception as exc:
                restore()
                solver._circuit_graph_error = f"{type(exc).__name__}: {exc}"
                solver._circuit_graph_runners = ()
                return
            restore()
        replay_maps.append(runtime_replays)

    solver._circuit_graph_runners = tuple(replay_maps)
    solver._circuit_graph_active = bool(replay_maps)


def apply_circuit_runtimes(solver) -> None:
    runtimes = tuple(getattr(solver, "_circuit_runtimes", ()))
    replay_maps = tuple(getattr(solver, "_circuit_graph_runners", ()))
    if replay_maps:
        for runtime, replay_map in zip(runtimes, replay_maps):
            if runtime.sample_times is None or runtime.step_index >= runtime.sample_times.numel() - 1:
                raise RuntimeError("Circuit time-series buffers were not prepared for this FDTD step.")
            factor_key = (
                runtime.integration_keys[runtime.step_index],
                runtime.switch_keys[runtime.step_index],
            )
            replay_map[factor_key]()
            # A graph replay re-executes device work only; the host-side
            # assignment inside _apply_step does not run. Advance the observer's
            # integration tag here from the same precomputed schedule, or the
            # port DFT would keep the last captured factor class and sample
            # backward-Euler steps at the trapezoidal time.
            for port in runtime.ports:
                port.last_integration = factor_key[0]
            runtime.step_index += 1
        return
    for runtime in runtimes:
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
    "prepare_circuit_graph_runners",
    "prepare_circuit_runtimes",
    "prepare_circuit_time_series",
]

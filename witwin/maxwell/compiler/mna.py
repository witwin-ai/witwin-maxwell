from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..circuits import (
    Circuit,
    CircuitData,
    CurrentControlledCurrentSource,
    CurrentControlledVoltageSource,
    CurrentSource,
    MutualInductor,
    PiecewiseLinearWaveform,
    PulseWaveform,
    SineWaveform,
    TimedSwitch,
    VoltageControlledCurrentSource,
    VoltageControlledVoltageSource,
    VoltageSource,
)
from ..lumped import Capacitor, Inductor, Resistor
from .circuits import CircuitGraph, compile_circuit_graph


def _node_unknown(graph: CircuitGraph, node) -> int | None:
    index = graph.node_index[node.name]
    return None if index == 0 else index - 1


def _branch_unknown(graph: CircuitGraph, name: str) -> int:
    return len(graph.nodes) - 1 + graph.branch_index[name]


def _add(matrix, row: int | None, column: int | None, value) -> None:
    if row is not None and column is not None:
        matrix[row, column] = matrix[row, column] + value


def _add_rhs(rhs, row: int | None, value) -> None:
    if row is not None:
        rhs[row] = rhs[row] + value


def _stamp_conductance(matrix, graph: CircuitGraph, positive, negative, conductance) -> None:
    p = _node_unknown(graph, positive)
    n = _node_unknown(graph, negative)
    _add(matrix, p, p, conductance)
    _add(matrix, n, n, conductance)
    _add(matrix, p, n, -conductance)
    _add(matrix, n, p, -conductance)


def _stamp_branch(matrix, graph: CircuitGraph, device) -> int:
    p = _node_unknown(graph, device.positive)
    n = _node_unknown(graph, device.negative)
    branch = _branch_unknown(graph, device.name)
    _add(matrix, p, branch, 1.0)
    _add(matrix, n, branch, -1.0)
    _add(matrix, branch, p, 1.0)
    _add(matrix, branch, n, -1.0)
    return branch


def _terminal_voltage(vector: torch.Tensor, graph: CircuitGraph, positive, negative):
    zero = vector.new_zeros(())
    p = _node_unknown(graph, positive)
    n = _node_unknown(graph, negative)
    return (zero if p is None else vector[p]) - (zero if n is None else vector[n])


def _source_value_at(values: torch.Tensor, index: int | torch.Tensor) -> torch.Tensor:
    if isinstance(index, torch.Tensor):
        return torch.index_select(values, 0, index.reshape(1)).squeeze(0)
    return values[index]


def _device_tensors(circuit: Circuit):
    for device in circuit.devices:
        for value in device.parameters.values():
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (PulseWaveform, SineWaveform)):
                yield from vars(value).values()
            elif isinstance(value, PiecewiseLinearWaveform):
                yield value.times
                yield value.values


def _resolve_dtype(circuit: Circuit, dtype) -> torch.dtype:
    if dtype is not None:
        resolved = dtype
    else:
        resolved = torch.float32
        for tensor in _device_tensors(circuit):
            resolved = torch.promote_types(resolved, tensor.dtype)
    if resolved not in (torch.float32, torch.float64):
        raise ValueError("Linear MNA supports torch.float32 and torch.float64 only.")
    return resolved


def _parameter(value, *, device: torch.device, dtype: torch.dtype, name: str):
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=device, dtype=dtype)
    )
    if tensor.ndim != 0 or tensor.is_complex():
        raise ValueError(f"Circuit parameter {name!r} must be a real scalar tensor.")
    if tensor.requires_grad and tensor.device != device:
        raise ValueError(
            f"Trainable circuit parameter {name!r} is on {tensor.device}, expected {device}."
        )
    return tensor.to(device=device, dtype=dtype)


def evaluate_waveform(waveform, times: torch.Tensor) -> torch.Tensor:
    """Evaluate one supported source waveform with device-resident torch operations."""

    if isinstance(waveform, PulseWaveform):
        values = {
            name: getattr(waveform, name).to(device=times.device, dtype=times.dtype)
            for name in ("initial", "pulsed", "delay", "rise", "fall", "width", "period")
        }
        tau = times - values["delay"]
        period_safe = torch.where(values["period"] > 0.0, values["period"], torch.ones_like(values["period"]))
        local = torch.where(values["period"] > 0.0, torch.remainder(tau, period_safe), tau)
        rise_safe = torch.where(values["rise"] > 0.0, values["rise"], torch.ones_like(values["rise"]))
        fall_safe = torch.where(values["fall"] > 0.0, values["fall"], torch.ones_like(values["fall"]))
        rising = values["initial"] + (values["pulsed"] - values["initial"]) * torch.clamp(local / rise_safe, 0.0, 1.0)
        plateau_end = values["rise"] + values["width"]
        falling = values["pulsed"] + (values["initial"] - values["pulsed"]) * torch.clamp(
            (local - plateau_end) / fall_safe,
            0.0,
            1.0,
        )
        active = torch.where(local < values["rise"], rising, values["pulsed"])
        active = torch.where(local <= plateau_end, active, falling)
        active = torch.where(local <= plateau_end + values["fall"], active, values["initial"])
        return torch.where(tau >= 0.0, active, values["initial"])
    if isinstance(waveform, SineWaveform):
        offset = waveform.offset.to(device=times.device, dtype=times.dtype)
        amplitude = waveform.amplitude.to(device=times.device, dtype=times.dtype)
        frequency = waveform.frequency.to(device=times.device, dtype=times.dtype)
        delay = waveform.delay.to(device=times.device, dtype=times.dtype)
        damping = waveform.damping.to(device=times.device, dtype=times.dtype)
        phase = waveform.phase_degrees.to(device=times.device, dtype=times.dtype) * (math.pi / 180.0)
        tau = torch.clamp_min(times - delay, 0.0)
        active = offset + amplitude * torch.exp(-damping * tau) * torch.sin(
            2.0 * math.pi * frequency * tau + phase
        )
        return torch.where(times >= delay, active, offset)
    if isinstance(waveform, PiecewiseLinearWaveform):
        sample_times = waveform.times.to(device=times.device, dtype=times.dtype)
        sample_values = waveform.values.to(device=times.device, dtype=times.dtype)
        upper = torch.searchsorted(sample_times, times).clamp(1, sample_times.numel() - 1)
        lower = upper - 1
        fraction = (times - sample_times[lower]) / (sample_times[upper] - sample_times[lower])
        interpolated = sample_values[lower] + fraction * (sample_values[upper] - sample_values[lower])
        interpolated = torch.where(times <= sample_times[0], sample_values[0], interpolated)
        return torch.where(times >= sample_times[-1], sample_values[-1], interpolated)
    raise TypeError(f"Unsupported source waveform {type(waveform).__name__}.")


@dataclass
class CircuitState:
    capacitor_voltage: dict[str, torch.Tensor]
    capacitor_current: dict[str, torch.Tensor]
    inductor_current: torch.Tensor
    inductor_voltage: torch.Tensor


@dataclass(frozen=True)
class BatchedMNAFactors:
    """Cached GPU LU factors for fixed-shape task-level circuit batches."""

    factors: torch.Tensor
    pivots: torch.Tensor
    condition: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.factors.shape[0]

    @property
    def unknown_count(self) -> int:
        return self.factors.shape[1]

    def solve(self, rhs: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        expected = (self.batch_size, self.unknown_count)
        if rhs.shape != expected:
            raise ValueError(f"Batched MNA right-hand side must have shape {expected}.")
        if rhs.device != self.factors.device or rhs.dtype != self.factors.dtype:
            raise ValueError("Batched MNA right-hand side must match factor device and dtype.")
        if out is None:
            return torch.linalg.lu_solve(
                self.factors,
                self.pivots,
                rhs.unsqueeze(-1),
            ).squeeze(-1)
        if out.shape != expected or out.device != rhs.device or out.dtype != rhs.dtype:
            raise ValueError("Batched MNA output must match right-hand-side shape, device, and dtype.")
        torch.linalg.lu_solve(
            self.factors,
            self.pivots,
            rhs.unsqueeze(-1),
            out=out.unsqueeze(-1),
        )
        return out


@dataclass(frozen=True)
class CompiledStampPlan:
    graph: CircuitGraph
    device: torch.device
    dtype: torch.dtype
    dt: torch.Tensor
    inductor_names: tuple[str, ...]
    inductance_matrix: torch.Tensor


class LinearMNASystem:
    """Device-resident dense linear MNA/DAE runtime for one compiled circuit."""

    def __init__(self, circuit: Circuit, plan: CompiledStampPlan):
        self.circuit = circuit
        self.plan = plan
        self.graph = plan.graph
        self.device = plan.device
        self.dtype = plan.dtype
        self.dt = plan.dt
        self._inductor_index = {name: index for index, name in enumerate(plan.inductor_names)}
        self._parameter_cache: dict[int, torch.Tensor] = {}
        self._literal_cache: dict[tuple[type, float], torch.Tensor] = {}
        self._refresh_parameter_cache()

    @property
    def unknown_count(self) -> int:
        return self.graph.unknown_count

    def _zeros(self, out=None):
        if out is not None:
            matrix, rhs = out
            expected_matrix = (self.unknown_count, self.unknown_count)
            if matrix.shape != expected_matrix or rhs.shape != (self.unknown_count,):
                raise ValueError("MNA assembly buffers do not match the compiled unknown count.")
            if matrix.device != self.device or rhs.device != self.device:
                raise ValueError("MNA assembly buffers must stay on the compiled device.")
            if matrix.dtype != self.dtype or rhs.dtype != self.dtype:
                raise ValueError("MNA assembly buffers must use the compiled dtype.")
            matrix.zero_()
            rhs.zero_()
            return matrix, rhs
        return (
            torch.zeros((self.unknown_count, self.unknown_count), device=self.device, dtype=self.dtype),
            torch.zeros((self.unknown_count,), device=self.device, dtype=self.dtype),
        )

    def _value(self, value, name):
        if isinstance(value, torch.Tensor):
            cached = self._parameter_cache.get(id(value))
            if cached is None:
                raise RuntimeError(f"Circuit parameter {name!r} was not compiled into the CUDA stamp plan.")
            if cached.ndim != 0 or cached.is_complex():
                raise ValueError(f"Circuit parameter {name!r} must be a real scalar tensor.")
            return cached
        key = (type(value), float(value))
        cached = self._literal_cache.get(key)
        if cached is None:
            cached = _parameter(value, device=self.device, dtype=self.dtype, name=name)
            self._literal_cache[key] = cached
        return cached

    def _refresh_parameter_cache(self) -> None:
        self._parameter_cache = {
            id(tensor): tensor.to(device=self.device, dtype=self.dtype)
            for tensor in _device_tensors(self.circuit)
        }
        self._parameter_cache.update(
            {
                id(value): value.to(device=self.device, dtype=self.dtype)
                for value, _constraint in self.circuit.initial_conditions.values()
            }
        )

    def _refresh_inductance_matrix(self) -> None:
        names, matrix = _compile_inductance_matrix(
            self.circuit,
            self.graph,
            device=self.device,
            dtype=self.dtype,
        )
        self.plan = CompiledStampPlan(
            graph=self.graph,
            device=self.device,
            dtype=self.dtype,
            dt=self.dt,
            inductor_names=names,
            inductance_matrix=matrix,
        )

    def _source_values(self, times: torch.Tensor):
        values = {}
        for device in self.circuit.devices:
            if isinstance(device, (VoltageSource, CurrentSource)):
                constant = device.voltage if isinstance(device, VoltageSource) else device.current
                constant = self._value(constant, device.name)
                values[device.name] = (
                    constant.expand_as(times)
                    if device.waveform is None
                    else evaluate_waveform(device.waveform, times)
                )
        return values

    def _dc_source_values(self):
        values = {}
        for device in self.circuit.devices:
            if isinstance(device, (VoltageSource, CurrentSource)):
                constant = device.voltage if isinstance(device, VoltageSource) else device.current
                values[device.name] = self._value(constant, device.name)
        return values

    def _stamp_static_device(self, matrix, rhs, device, source_value, *, dc: bool):
        graph = self.graph
        if isinstance(device, Resistor):
            _stamp_conductance(
                matrix,
                graph,
                device.positive,
                device.negative,
                1.0 / self._value(device.resistance, device.name),
            )
        elif isinstance(device, VoltageSource):
            branch = _stamp_branch(matrix, graph, device)
            _add_rhs(rhs, branch, source_value)
        elif isinstance(device, CurrentSource):
            p = _node_unknown(graph, device.positive)
            n = _node_unknown(graph, device.negative)
            _add_rhs(rhs, p, -source_value)
            _add_rhs(rhs, n, source_value)
        elif isinstance(device, VoltageControlledVoltageSource):
            branch = _stamp_branch(matrix, graph, device)
            gain = self._value(device.gain, device.name)
            _add(matrix, branch, _node_unknown(graph, device.control_positive), -gain)
            _add(matrix, branch, _node_unknown(graph, device.control_negative), gain)
        elif isinstance(device, VoltageControlledCurrentSource):
            gain = self._value(device.transconductance, device.name)
            p = _node_unknown(graph, device.positive)
            n = _node_unknown(graph, device.negative)
            cp = _node_unknown(graph, device.control_positive)
            cn = _node_unknown(graph, device.control_negative)
            _add(matrix, p, cp, gain)
            _add(matrix, p, cn, -gain)
            _add(matrix, n, cp, -gain)
            _add(matrix, n, cn, gain)
        elif isinstance(device, CurrentControlledVoltageSource):
            branch = _stamp_branch(matrix, graph, device)
            control = _branch_unknown(graph, self.graph.source_dependencies[device.name][0])
            _add(matrix, branch, control, -self._value(device.transresistance, device.name))
        elif isinstance(device, CurrentControlledCurrentSource):
            control = _branch_unknown(graph, self.graph.source_dependencies[device.name][0])
            gain = self._value(device.gain, device.name)
            _add(matrix, _node_unknown(graph, device.positive), control, gain)
            _add(matrix, _node_unknown(graph, device.negative), control, -gain)
        elif isinstance(device, TimedSwitch):
            if source_value is None:
                resistance = self._value(
                    device.on_resistance if device.initially_closed else device.off_resistance,
                    device.name,
                )
            else:
                resistance = source_value
            _stamp_conductance(
                matrix,
                graph,
                device.positive,
                device.negative,
                1.0 / resistance,
            )

    def assemble_dc(self):
        matrix, rhs = self._zeros()
        times = torch.zeros((1,), device=self.device, dtype=self.dtype)
        source_values = self._dc_source_values()
        for device in self.circuit.devices:
            if isinstance(device, Capacitor | MutualInductor):
                continue
            if isinstance(device, Inductor):
                _stamp_branch(matrix, self.graph, device)
                continue
            value = (
                self._switch_resistance(device, times[0])
                if isinstance(device, TimedSwitch)
                else source_values.get(device.name)
            )
            self._stamp_static_device(matrix, rhs, device, value, dc=True)
        return matrix, rhs

    def _switch_resistance(self, device: TimedSwitch, time: torch.Tensor):
        transitions = self._parameter_cache[id(device.transition_times)]
        count = torch.searchsorted(transitions, time, right=True)
        closed = torch.remainder(count, 2).to(dtype=torch.bool)
        if device.initially_closed:
            closed = torch.logical_not(closed)
        on = self._value(device.on_resistance, device.name)
        off = self._value(device.off_resistance, device.name)
        return torch.where(closed, on, off)

    def assemble_transient(
        self,
        state: CircuitState,
        *,
        time: torch.Tensor,
        source_values: dict[str, torch.Tensor],
        step_index: int,
        integration: str | None = None,
        midpoint: bool = False,
        out=None,
        stamp_matrix: bool = True,
    ):
        if stamp_matrix:
            matrix, rhs = self._zeros(out)
        else:
            if out is None:
                matrix = torch.empty(
                    (self.unknown_count, self.unknown_count),
                    device=self.device,
                    dtype=self.dtype,
                )
                rhs = torch.zeros((self.unknown_count,), device=self.device, dtype=self.dtype)
            else:
                matrix, rhs = out
                if rhs.shape != (self.unknown_count,):
                    raise ValueError("MNA RHS buffer does not match the compiled unknown count.")
                rhs.zero_()
        integration = self.circuit.config.integration if integration is None else integration
        factor = 2.0 if integration == "trapezoidal" else 1.0
        for device in self.circuit.devices:
            if isinstance(device, Capacitor):
                capacitance = self._value(device.capacitance, device.name)
                conductance = factor * capacitance / self.dt
                if stamp_matrix:
                    _stamp_conductance(
                        matrix,
                        self.graph,
                        device.positive,
                        device.negative,
                        conductance,
                    )
                previous_voltage = state.capacitor_voltage[device.name]
                previous_current = state.capacitor_current[device.name]
                history = -conductance * previous_voltage
                if integration == "trapezoidal" and not midpoint:
                    history = history - previous_current
                _add_rhs(rhs, _node_unknown(self.graph, device.positive), -history)
                _add_rhs(rhs, _node_unknown(self.graph, device.negative), history)
                continue
            if isinstance(device, Inductor):
                branch = (
                    _stamp_branch(matrix, self.graph, device)
                    if stamp_matrix
                    else _branch_unknown(self.graph, device.name)
                )
                row = self._inductor_index[device.name]
                alpha_row = factor * self.plan.inductance_matrix[row] / self.dt
                if stamp_matrix:
                    for column, name in enumerate(self.plan.inductor_names):
                        _add(
                            matrix,
                            branch,
                            _branch_unknown(self.graph, name),
                            -alpha_row[column],
                        )
                history = -torch.dot(alpha_row, state.inductor_current)
                if integration == "trapezoidal" and not midpoint:
                    history = history - state.inductor_voltage[row]
                _add_rhs(rhs, branch, history)
                continue
            if isinstance(device, MutualInductor):
                continue
            if isinstance(device, TimedSwitch):
                value = self._switch_resistance(device, time)
            elif isinstance(device, (VoltageSource, CurrentSource)):
                value = _source_value_at(source_values[device.name], step_index)
            else:
                value = None
            if stamp_matrix:
                self._stamp_static_device(matrix, rhs, device, value, dc=False)
            elif isinstance(device, VoltageSource):
                _add_rhs(rhs, _branch_unknown(self.graph, device.name), value)
            elif isinstance(device, CurrentSource):
                _add_rhs(rhs, _node_unknown(self.graph, device.positive), -value)
                _add_rhs(rhs, _node_unknown(self.graph, device.negative), value)
        return matrix, rhs

    def _solve(self, matrix, rhs):
        factors, pivots, condition = self._factor(matrix)
        solution = torch.linalg.lu_solve(factors, pivots, rhs.unsqueeze(-1)).squeeze(-1)
        return solution, condition

    def _factor(self, matrix):
        singular_values = torch.linalg.svdvals(matrix)
        largest = singular_values[0]
        smallest = singular_values[-1]
        tolerance = self.circuit.config.pivot_tolerance
        if bool((smallest <= tolerance * largest).detach()):
            raise ValueError(
                f"Circuit {self.circuit.name!r} MNA matrix is singular or below pivot tolerance {tolerance:g}."
            )
        factors, pivots = torch.linalg.lu_factor(matrix)
        return factors, pivots, largest / smallest

    def solve_dc(self):
        self._refresh_parameter_cache()
        self._refresh_inductance_matrix()
        matrix, rhs = self.assemble_dc()
        return self._solve(matrix, rhs)

    def _initial_constraint_system(self):
        constraints: dict[tuple[float, ...], torch.Tensor] = {}

        def add_constraint(entries, target, *, override=False):
            row = [0.0] * self.unknown_count
            for index, coefficient in entries:
                if index is not None:
                    row[index] += coefficient
            first = next((value for value in row if value != 0.0), None)
            if first is None:
                return
            sign = -1.0 if first < 0.0 else 1.0
            key = tuple(sign * value for value in row)
            value = sign * target
            if override or key not in constraints:
                constraints[key] = value

        zero = self.dt.new_zeros(())
        if self.circuit.config.initialization == "zero":
            for device in self.circuit.devices:
                if isinstance(device, Capacitor):
                    add_constraint(
                        (
                            (_node_unknown(self.graph, device.positive), 1.0),
                            (_node_unknown(self.graph, device.negative), -1.0),
                        ),
                        zero,
                    )
                elif isinstance(device, Inductor):
                    add_constraint(((_branch_unknown(self.graph, device.name), 1.0),), zero)
        for node_name, (target, constraint) in self.circuit.initial_conditions.items():
            if constraint and node_name != "0":
                add_constraint(
                    ((self.graph.node_index[node_name] - 1, 1.0),),
                    self._value(target, f"initial condition {node_name}"),
                    override=True,
                )
        if not constraints:
            return None
        rows = self.dt.new_tensor(tuple(constraints))
        targets = torch.stack(tuple(constraints.values()))
        return rows, targets

    def _solve_initial(self, dc_solution, dc_condition):
        constraint_system = self._initial_constraint_system()
        if constraint_system is None:
            return dc_solution, dc_condition, 0
        rows, targets = constraint_system
        matrix, rhs = self.assemble_dc()
        count = rows.shape[0]
        augmented = matrix.new_zeros((self.unknown_count + count, self.unknown_count + count))
        augmented[: self.unknown_count, : self.unknown_count] = matrix
        augmented[: self.unknown_count, self.unknown_count :] = rows.transpose(0, 1)
        augmented[self.unknown_count :, : self.unknown_count] = rows
        augmented_rhs = torch.cat((rhs, targets))
        solution, condition = self._solve(augmented, augmented_rhs)
        return solution[: self.unknown_count], condition, 1

    def _local_backward_euler_steps(self, steps: int) -> tuple[int, ...]:
        if self.circuit.config.integration != "trapezoidal":
            return ()
        dt = float(self.dt.detach().cpu())
        duration = steps * dt
        breakpoints = [0.0]

        def scalar(value):
            return float(self._parameter_cache[id(value)].detach().cpu())

        for device in self.circuit.devices:
            waveform = getattr(device, "waveform", None)
            if isinstance(waveform, PulseWaveform):
                delay = scalar(waveform.delay)
                rise = scalar(waveform.rise)
                width = scalar(waveform.width)
                fall = scalar(waveform.fall)
                period = scalar(waveform.period)
                cycle = 0
                while True:
                    start = delay + cycle * period
                    if start > duration:
                        break
                    breakpoints.extend((start, start + rise, start + rise + width, start + rise + width + fall))
                    if period <= 0.0:
                        break
                    cycle += 1
            elif isinstance(waveform, SineWaveform):
                breakpoints.append(scalar(waveform.delay))
            elif isinstance(waveform, PiecewiseLinearWaveform):
                breakpoints.extend(
                    self._parameter_cache[id(waveform.times)].detach().cpu().tolist()
                )
            if isinstance(device, TimedSwitch):
                breakpoints.extend(
                    self._parameter_cache[id(device.transition_times)].detach().cpu().tolist()
                )
        indices = {
            max(1, math.ceil(point / dt - 1.0e-12))
            for point in breakpoints
            if 0.0 <= point <= duration
        }
        return tuple(sorted(index for index in indices if index <= steps))

    def _switch_state_keys(self, times: torch.Tensor) -> tuple[tuple[bool, ...], ...]:
        switches = tuple(device for device in self.circuit.devices if isinstance(device, TimedSwitch))
        if not switches:
            return ((),) * times.numel()
        cpu_times = times.detach().cpu()
        states = []
        for device in switches:
            transitions = self._parameter_cache[id(device.transition_times)].detach().cpu()
            counts = torch.searchsorted(transitions, cpu_times, right=True)
            states.append(
                torch.logical_xor(
                    torch.full_like(counts, device.initially_closed, dtype=torch.bool),
                    torch.remainder(counts, 2).to(dtype=torch.bool),
                ).tolist()
            )
        return tuple(tuple(state[index] for state in states) for index in range(times.numel()))

    def _initial_state(self, dc_solution):
        capacitor_voltage = {}
        capacitor_current = {}
        for device in self.circuit.devices:
            if isinstance(device, Capacitor):
                capacitor_voltage[device.name] = _terminal_voltage(
                    dc_solution, self.graph, device.positive, device.negative
                )
                capacitor_current[device.name] = dc_solution.new_zeros(())
        currents = []
        voltages = []
        for name in self.plan.inductor_names:
            device = next(device for device in self.circuit.devices if device.name == name)
            currents.append(dc_solution[_branch_unknown(self.graph, name)])
            voltages.append(_terminal_voltage(dc_solution, self.graph, device.positive, device.negative))
        return CircuitState(
            capacitor_voltage=capacitor_voltage,
            capacitor_current=capacitor_current,
            inductor_current=torch.stack(currents) if currents else dc_solution.new_zeros((0,)),
            inductor_voltage=torch.stack(voltages) if voltages else dc_solution.new_zeros((0,)),
        )

    def _update_state(
        self,
        state: CircuitState,
        solution: torch.Tensor,
        *,
        integration: str | None = None,
        midpoint: bool = False,
    ):
        integration = self.circuit.config.integration if integration is None else integration
        factor = 2.0 if integration == "trapezoidal" else 1.0
        capacitor_voltage = {}
        capacitor_current = {}
        for device in self.circuit.devices:
            if not isinstance(device, Capacitor):
                continue
            voltage = _terminal_voltage(solution, self.graph, device.positive, device.negative)
            conductance = factor * self._value(device.capacitance, device.name) / self.dt
            current = conductance * (voltage - state.capacitor_voltage[device.name])
            if integration == "trapezoidal" and midpoint:
                voltage = 2.0 * voltage - state.capacitor_voltage[device.name]
                current = 2.0 * current - state.capacitor_current[device.name]
            elif integration == "trapezoidal":
                current = current - state.capacitor_current[device.name]
            capacitor_voltage[device.name] = voltage
            capacitor_current[device.name] = current
        currents = torch.stack(
            [solution[_branch_unknown(self.graph, name)] for name in self.plan.inductor_names]
        ) if self.plan.inductor_names else solution.new_zeros((0,))
        voltages = []
        for name in self.plan.inductor_names:
            device = next(device for device in self.circuit.devices if device.name == name)
            voltages.append(_terminal_voltage(solution, self.graph, device.positive, device.negative))
        if integration == "trapezoidal" and midpoint:
            currents = 2.0 * currents - state.inductor_current
            voltages = [
                2.0 * voltage - state.inductor_voltage[index]
                for index, voltage in enumerate(voltages)
            ]
        return CircuitState(
            capacitor_voltage=capacitor_voltage,
            capacitor_current=capacitor_current,
            inductor_current=currents,
            inductor_voltage=torch.stack(voltages) if voltages else solution.new_zeros((0,)),
        )

    def _device_currents(self, solution, state, source_values, step_index, time):
        currents = {}
        for device in self.circuit.devices:
            if isinstance(device, Resistor):
                voltage = _terminal_voltage(solution, self.graph, device.positive, device.negative)
                currents[device.name] = voltage / self._value(device.resistance, device.name)
            elif isinstance(device, Capacitor):
                currents[device.name] = state.capacitor_current[device.name]
            elif isinstance(device, Inductor):
                currents[device.name] = solution[_branch_unknown(self.graph, device.name)]
            elif isinstance(device, (VoltageSource, VoltageControlledVoltageSource, CurrentControlledVoltageSource)):
                currents[device.name] = solution[_branch_unknown(self.graph, device.name)]
            elif isinstance(device, CurrentSource):
                currents[device.name] = _source_value_at(
                    source_values[device.name],
                    step_index,
                )
            elif isinstance(device, VoltageControlledCurrentSource):
                control = _terminal_voltage(
                    solution,
                    self.graph,
                    device.control_positive,
                    device.control_negative,
                )
                currents[device.name] = self._value(device.transconductance, device.name) * control
            elif isinstance(device, CurrentControlledCurrentSource):
                control = solution[
                    _branch_unknown(self.graph, self.graph.source_dependencies[device.name][0])
                ]
                currents[device.name] = self._value(device.gain, device.name) * control
            elif isinstance(device, TimedSwitch):
                voltage = _terminal_voltage(solution, self.graph, device.positive, device.negative)
                currents[device.name] = voltage / self._switch_resistance(device, time)
        return currents

    def _stored_energy(self, solution: torch.Tensor, state: CircuitState) -> torch.Tensor:
        energy = solution.new_zeros(())
        for device in self.circuit.devices:
            if isinstance(device, Capacitor):
                voltage = _terminal_voltage(solution, self.graph, device.positive, device.negative)
                energy = energy + 0.5 * self._value(
                    device.capacitance,
                    device.name,
                ) * voltage.square()
        if state.inductor_current.numel():
            energy = energy + 0.5 * torch.dot(
                state.inductor_current,
                self.plan.inductance_matrix @ state.inductor_current,
            )
        return energy

    def transient(self, steps: int) -> CircuitData:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer.")
        times = torch.arange(steps + 1, device=self.device, dtype=self.dtype) * self.dt
        dc_solution, dc_condition = self.solve_dc()
        source_values = self._source_values(times)
        initial_solution, initial_condition, initial_factorizations = self._solve_initial(
            dc_solution,
            dc_condition,
        )
        dc_state = self._initial_state(dc_solution)
        state = self._initial_state(initial_solution)
        solutions = [initial_solution]
        conditions = [initial_condition]
        device_power_series = {device.name: [] for device in self.circuit.devices}
        physical_branch_names = tuple(
            device.name
            for device in self.circuit.devices
            if not isinstance(device, MutualInductor)
        )
        branch_current_series = {name: [] for name in physical_branch_names}
        dc_sources = {
            name: value.expand_as(times)
            for name, value in self._dc_source_values().items()
        }
        initial_currents = self._device_currents(
            initial_solution,
            state,
            dc_sources,
            0,
            times[0],
        )
        for name in physical_branch_names:
            branch_current_series[name].append(initial_currents[name])
        for device in self.circuit.devices:
            if isinstance(device, MutualInductor):
                device_power_series[device.name].append(initial_solution.new_zeros(()))
            else:
                voltage = _terminal_voltage(initial_solution, self.graph, device.positive, device.negative)
                device_power_series[device.name].append(voltage * initial_currents[device.name])
        nonreactive_names = tuple(
            device.name
            for device in self.circuit.devices
            if not isinstance(device, (Capacitor, Inductor, MutualInductor))
        )
        dc_stored_energy = self._stored_energy(dc_solution, dc_state)
        previous_stored_energy = self._stored_energy(initial_solution, state)
        initial_stored_energy = previous_stored_energy
        previous_nonreactive_power = sum(
            (device_power_series[name][-1] for name in nonreactive_names),
            start=initial_solution.new_zeros(()),
        )
        energy_balance_series = [initial_solution.new_zeros(())]

        local_backward_euler_steps = self._local_backward_euler_steps(steps)
        local_backward_euler_set = set(local_backward_euler_steps)
        switch_state_keys = self._switch_state_keys(times)
        factor_cache = {}
        for step_index in range(1, steps + 1):
            integration = (
                "backward_euler"
                if step_index in local_backward_euler_set
                else self.circuit.config.integration
            )
            matrix, rhs = self.assemble_transient(
                state,
                time=times[step_index],
                source_values=source_values,
                step_index=step_index,
                integration=integration,
            )
            factor_key = (integration, switch_state_keys[step_index])
            if factor_key not in factor_cache:
                factor_cache[factor_key] = self._factor(matrix)
            factors, pivots, condition = factor_cache[factor_key]
            solution = torch.linalg.lu_solve(
                factors,
                pivots,
                rhs.unsqueeze(-1),
            ).squeeze(-1)
            state = self._update_state(state, solution, integration=integration)
            currents = self._device_currents(
                solution,
                state,
                source_values,
                step_index,
                times[step_index],
            )
            for name in physical_branch_names:
                branch_current_series[name].append(currents[name])
            solutions.append(solution)
            conditions.append(condition)
            for device in self.circuit.devices:
                if isinstance(device, MutualInductor):
                    power = solution.new_zeros(())
                else:
                    voltage = _terminal_voltage(solution, self.graph, device.positive, device.negative)
                    power = voltage * currents[device.name]
                device_power_series[device.name].append(power)
            stored_energy = self._stored_energy(solution, state)
            nonreactive_power = sum(
                (device_power_series[name][-1] for name in nonreactive_names),
                start=solution.new_zeros(()),
            )
            energy_balance_series.append(
                energy_balance_series[-1]
                + stored_energy
                - previous_stored_energy
                + 0.5 * self.dt * (previous_nonreactive_power + nonreactive_power)
            )
            previous_stored_energy = stored_energy
            previous_nonreactive_power = nonreactive_power

        stacked = torch.stack(solutions)
        node_voltages = torch.cat(
            (
                stacked.new_zeros((steps + 1, 1)),
                stacked[:, : len(self.graph.nodes) - 1],
            ),
            dim=1,
        )
        branch_currents = torch.stack(
            tuple(torch.stack(branch_current_series[name]) for name in physical_branch_names),
            dim=1,
        )
        device_powers = {
            name: torch.stack(values) for name, values in device_power_series.items()
        }
        energy_balance = torch.stack(energy_balance_series)
        condition_tensor = torch.stack(conditions)
        return CircuitData(
            circuit_name=self.circuit.name,
            times=times,
            node_names=tuple(node.name for node in self.graph.nodes),
            node_voltages=node_voltages,
            branch_names=physical_branch_names,
            branch_currents=branch_currents,
            device_powers=device_powers,
            energy_balance=energy_balance,
            diagnostics={
                "integration": self.circuit.config.integration,
                "dt": self.dt,
                "condition": condition_tensor,
                "dc_condition": dc_condition,
                "solve_count": steps + 1 + initial_factorizations,
                "factorization_count": 1 + initial_factorizations + len(factor_cache),
                "dc_factorization_count": 1,
                "initial_factorization_count": initial_factorizations,
                "transient_factorization_count": len(factor_cache),
                "local_backward_euler_steps": local_backward_euler_steps,
                "initial_stored_energy": initial_stored_energy,
                "dc_stored_energy": dc_stored_energy,
                "initialization_energy_delta": initial_stored_energy - dc_stored_energy,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "mna_branch_names": self.graph.branch_names,
            },
        )


def _compile_inductance_matrix(circuit, graph, *, device, dtype):
    inductors = tuple(device_item for device_item in circuit.devices if isinstance(device_item, Inductor))
    names = tuple(device_item.name for device_item in inductors)
    matrix = torch.zeros((len(inductors), len(inductors)), device=device, dtype=dtype)
    index = {name.casefold(): position for position, name in enumerate(names)}
    values = {}
    for position, inductor in enumerate(inductors):
        value = _parameter(
            inductor.inductance,
            device=device,
            dtype=dtype,
            name=inductor.name,
        )
        matrix[position, position] = value
        values[inductor.name.casefold()] = value
    for mutual in (device_item for device_item in circuit.devices if isinstance(device_item, MutualInductor)):
        first = index[mutual.first_inductor.casefold()]
        second = index[mutual.second_inductor.casefold()]
        coupling = _parameter(mutual.coupling, device=device, dtype=dtype, name=mutual.name)
        value = coupling * torch.sqrt(values[mutual.first_inductor.casefold()] * values[mutual.second_inductor.casefold()])
        matrix[first, second] = matrix[first, second] + value
        matrix[second, first] = matrix[second, first] + value
    if matrix.numel():
        eigenvalues = torch.linalg.eigvalsh(matrix)
        scale = torch.max(torch.abs(eigenvalues)).clamp_min(torch.finfo(dtype).tiny)
        if bool((eigenvalues[0] < -circuit.config.pivot_tolerance * scale).detach()):
            raise ValueError("Mutual inductance matrix must be positive semidefinite.")
    return names, matrix


def _compile_mna_system(
    circuit: Circuit,
    *,
    dt,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    allow_bindings: bool,
) -> LinearMNASystem:
    if not isinstance(circuit, Circuit):
        raise TypeError("compile_mna_system requires a Circuit instance.")
    if circuit.bindings and not allow_bindings:
        raise ValueError(
            "Standalone MNA cannot solve EM-bound ports without the FDTD Norton companion."
        )
    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        raise ValueError("Linear MNA is GPU-native and requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("Linear MNA requires CUDA, but torch.cuda.is_available() is False.")
    if target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    resolved_dtype = _resolve_dtype(circuit, dtype)
    for tensor in _device_tensors(circuit):
        if tensor.requires_grad and tensor.device != target:
            raise ValueError(
                f"Trainable circuit tensor is on {tensor.device}, expected {target}."
            )
    for value, _constraint in circuit.initial_conditions.values():
        if value.requires_grad and value.device != target:
            raise ValueError(
                f"Trainable circuit initial condition is on {value.device}, expected {target}."
            )
    dt_tensor = _parameter(dt, device=target, dtype=resolved_dtype, name="dt")
    if bool((dt_tensor <= 0.0).detach()):
        raise ValueError("dt must be positive.")
    graph = compile_circuit_graph(circuit)
    inductor_names, inductance_matrix = _compile_inductance_matrix(
        circuit,
        graph,
        device=target,
        dtype=resolved_dtype,
    )
    return LinearMNASystem(
        circuit,
        CompiledStampPlan(
            graph=graph,
            device=target,
            dtype=resolved_dtype,
            dt=dt_tensor,
            inductor_names=inductor_names,
            inductance_matrix=inductance_matrix,
        ),
    )


def compile_mna_system(
    circuit: Circuit,
    *,
    dt,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LinearMNASystem:
    """Compile one standalone linear circuit to the CUDA dense MNA runtime."""

    return _compile_mna_system(
        circuit,
        dt=dt,
        device=device,
        dtype=dtype,
        allow_bindings=False,
    )


def compile_coupled_mna_system(
    circuit: Circuit,
    *,
    dt,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LinearMNASystem:
    """Compile an EM-bound circuit for the FDTD Norton companion runtime."""

    if not circuit.bindings:
        raise ValueError("Coupled MNA compilation requires at least one EM port binding.")
    return _compile_mna_system(
        circuit,
        dt=dt,
        device=device,
        dtype=dtype,
        allow_bindings=True,
    )


def compile_batched_mna_factors(
    matrices: torch.Tensor,
    *,
    pivot_tolerance: float = 1.0e-10,
) -> BatchedMNAFactors:
    """Factor a fixed-shape ``[batch, unknown, unknown]`` CUDA MNA batch once."""

    if (
        matrices.ndim != 3
        or matrices.shape[0] < 1
        or matrices.shape[1] < 1
        or matrices.shape[1] != matrices.shape[2]
    ):
        raise ValueError("Batched MNA matrices must have non-empty shape [B, N, N].")
    if matrices.device.type != "cuda":
        raise ValueError("Batched MNA factorization requires CUDA matrices.")
    if matrices.dtype not in (torch.float32, torch.float64) or matrices.is_complex():
        raise ValueError("Batched MNA matrices require real float32 or float64 values.")
    if matrices.requires_grad:
        raise ValueError("Cached batched MNA factors require fixed non-trainable matrices.")
    if not math.isfinite(pivot_tolerance) or pivot_tolerance <= 0.0:
        raise ValueError("pivot_tolerance must be finite and positive.")
    singular_values = torch.linalg.svdvals(matrices)
    largest = singular_values[:, 0]
    smallest = singular_values[:, -1]
    if bool(torch.any(smallest <= pivot_tolerance * largest).detach()):
        raise ValueError("Batched MNA matrices contain a singular or below-tolerance system.")
    factors, pivots = torch.linalg.lu_factor(matrices)
    return BatchedMNAFactors(
        factors=factors,
        pivots=pivots,
        condition=largest / smallest,
    )


__all__ = [
    "BatchedMNAFactors",
    "CircuitState",
    "CompiledStampPlan",
    "LinearMNASystem",
    "compile_coupled_mna_system",
    "compile_batched_mna_factors",
    "compile_mna_system",
    "evaluate_waveform",
]

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..circuits import (
    Circuit,
    CircuitNode,
    CurrentControlledCurrentSource,
    CurrentControlledVoltageSource,
    CurrentSource,
    MutualInductor,
    PortBinding,
    TimedSwitch,
    VoltageControlledCurrentSource,
    VoltageControlledVoltageSource,
    VoltageSource,
)
from ..lumped import Inductor, Resistor
from ..circuit_devices import Diode, PiecewiseLinearIV, PolynomialIV


_BRANCH_TYPES = (
    Inductor,
    VoltageSource,
    VoltageControlledVoltageSource,
    CurrentControlledVoltageSource,
)
_CURRENT_SOURCE_TYPES = (
    CurrentSource,
    VoltageControlledCurrentSource,
    CurrentControlledCurrentSource,
)
# Nonlinear conduction devices (diode, behavioral I-V) provide a finite DC
# conductance between their terminals, so they establish a DC path exactly like a
# resistor. A VoltageDependentCapacitor is charge-only (open at DC) and is
# therefore deliberately absent, matching the linear Capacitor.
_DC_CONNECTING_TYPES = (
    Resistor,
    Inductor,
    VoltageSource,
    VoltageControlledVoltageSource,
    CurrentControlledVoltageSource,
    TimedSwitch,
    Diode,
    PiecewiseLinearIV,
    PolynomialIV,
)


@dataclass(frozen=True)
class CircuitGraph:
    circuit_name: str
    nodes: tuple[CircuitNode, ...]
    node_index: Mapping[str, int]
    devices: tuple[object, ...]
    branch_names: tuple[str, ...]
    branch_index: Mapping[str, int]
    bindings: tuple[PortBinding, ...]
    source_dependencies: Mapping[str, tuple[str, ...]]

    @property
    def unknown_count(self) -> int:
        return len(self.nodes) - 1 + len(self.branch_names)


class _DisjointSet:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, first, second) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        self.parent[second_root] = first_root
        return True


def _casefold_map(values, *, label: str):
    result = {}
    for value in values:
        key = value.name.casefold()
        if key in result:
            raise ValueError(f"Duplicate {label} name {value.name!r}.")
        result[key] = value
    return result


def _validate_control_dependencies(devices_by_name, branch_names):
    branch_keys = {name.casefold(): name for name in branch_names}
    dependencies: dict[str, tuple[str, ...]] = {}
    for device in devices_by_name.values():
        if isinstance(device, (CurrentControlledCurrentSource, CurrentControlledVoltageSource)):
            control_key = device.control_source.casefold()
            if control_key not in branch_keys:
                raise ValueError(
                    f"Controlled source {device.name!r} references missing branch source "
                    f"{device.control_source!r}."
                )
            dependencies[device.name] = (branch_keys[control_key],)
        elif isinstance(device, MutualInductor):
            resolved = []
            for inductor_name in (device.first_inductor, device.second_inductor):
                target = devices_by_name.get(inductor_name.casefold())
                if not isinstance(target, Inductor):
                    raise ValueError(
                        f"Mutual inductor {device.name!r} references missing or non-inductor "
                        f"device {inductor_name!r}."
                    )
                resolved.append(target.name)
            dependencies[device.name] = tuple(resolved)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str):
        key = name.casefold()
        if key in visited:
            return
        if key in visiting:
            raise ValueError(f"Circuit source dependency cycle includes {name!r}.")
        visiting.add(key)
        for dependency in dependencies.get(name, ()):
            dependent = devices_by_name.get(dependency.casefold())
            if dependent is not None and dependent.name in dependencies:
                visit(dependent.name)
        visiting.remove(key)
        visited.add(key)

    for name in dependencies:
        visit(name)
    return dependencies


def _validate_voltage_source_loops(nodes, devices):
    sets = _DisjointSet(node.name for node in nodes)
    for device in devices:
        if not isinstance(
            device,
            (Inductor, VoltageSource, VoltageControlledVoltageSource, CurrentControlledVoltageSource),
        ):
            continue
        if not sets.union(device.positive.name, device.negative.name):
            if isinstance(device, Inductor):
                raise ValueError(
                    f"DC ideal-voltage/inductor loop detected at device {device.name!r}."
                )
            raise ValueError(f"Ideal voltage-source loop detected at device {device.name!r}.")


def _validate_dc_reference(nodes, devices, bindings):
    adjacency = {node.name: set() for node in nodes}
    for device in devices:
        if isinstance(device, _DC_CONNECTING_TYPES):
            adjacency[device.positive.name].add(device.negative.name)
            adjacency[device.negative.name].add(device.positive.name)
    for binding in bindings:
        adjacency[binding.positive.name].add(binding.negative.name)
        adjacency[binding.negative.name].add(binding.positive.name)
    reachable = {"0"}
    frontier = ["0"]
    while frontier:
        node = frontier.pop()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    referenced = {
        terminal.name
        for device in devices
        if not isinstance(device, MutualInductor)
        for terminal in (
            device.positive,
            device.negative,
            *(
                (device.control_positive, device.control_negative)
                if isinstance(device, (VoltageControlledVoltageSource, VoltageControlledCurrentSource))
                else ()
            ),
        )
    }
    referenced.update(
        terminal.name
        for binding in bindings
        for terminal in (binding.positive, binding.negative)
    )
    floating = tuple(node.name for node in nodes if node.name in referenced and node.name not in reachable)
    if floating:
        current_sources = tuple(
            device.name
            for device in devices
            if isinstance(device, _CURRENT_SOURCE_TYPES)
            and (device.positive.name in floating or device.negative.name in floating)
        )
        detail = f"; current-source cutset includes {current_sources}" if current_sources else ""
        raise ValueError(
            f"Circuit has no DC path from nodes {floating} to ground{detail}."
        )
    if referenced and "0" not in referenced:
        raise ValueError("Circuit ground node is not referenced by any device.")


def compile_circuit_graph(
    circuit: Circuit,
    *,
    available_ports=None,
) -> CircuitGraph:
    """Normalize and structurally validate one circuit graph on the CPU control plane."""

    if not isinstance(circuit, Circuit):
        raise TypeError("compile_circuit_graph requires a Circuit instance.")
    nodes = circuit.nodes
    devices = circuit.devices
    if not devices:
        raise ValueError(f"Circuit {circuit.name!r} must contain at least one device.")
    if not nodes or nodes[0] is not circuit.ground or circuit.ground.name != "0":
        raise ValueError("Circuit ground must be the first, unique node named '0'.")
    devices_by_name = _casefold_map(devices, label="circuit device")
    branch_names = tuple(device.name for device in devices if isinstance(device, _BRANCH_TYPES))
    if len(nodes) - 1 + len(branch_names) > circuit.config.dense_unknown_limit:
        raise ValueError(
            f"Circuit {circuit.name!r} has {len(nodes) - 1 + len(branch_names)} unknowns, "
            f"exceeding the GPU dense limit {circuit.config.dense_unknown_limit}."
        )
    if available_ports is not None:
        available = {str(name) for name in available_ports}
        missing = tuple(binding.port_name for binding in circuit.bindings if binding.port_name not in available)
        if missing:
            raise ValueError(f"Circuit {circuit.name!r} binds missing EM ports {missing}.")
    dependencies = _validate_control_dependencies(devices_by_name, branch_names)
    _validate_voltage_source_loops(nodes, devices)
    _validate_dc_reference(nodes, devices, circuit.bindings)
    return CircuitGraph(
        circuit_name=circuit.name,
        nodes=nodes,
        node_index=MappingProxyType({node.name: index for index, node in enumerate(nodes)}),
        devices=devices,
        branch_names=branch_names,
        branch_index=MappingProxyType({name: index for index, name in enumerate(branch_names)}),
        bindings=circuit.bindings,
        source_dependencies=MappingProxyType(dict(dependencies)),
    )


def reject_nonlinear_devices(circuit) -> None:
    """Fail closed when a nonlinear device enters a linear executable runtime.

    ``Circuit.add`` and :func:`compile_circuit_graph` admit the nonlinear device
    family (diode / behavioral I-V / voltage-dependent capacitor) so their DC
    topology and structural validation are exercised. Their terminal ``i(v)`` /
    ``q(v)`` law is not linear, so it cannot be assembled into the single
    constant-conductance stamp that the linear MNA runtime (standalone, coupled,
    and the FDTD Norton companion) builds: those paths carry no Newton iteration,
    and would otherwise solve the network with the device silently absent. This
    rejects such a circuit at the linear runtime boundary until the nonlinear
    device-runtime integration wires the Newton loop into these paths.
    """

    from ..circuit_devices import NONLINEAR_DEVICE_TYPES

    offenders = tuple(
        device.name
        for device in circuit.devices
        if isinstance(device, NONLINEAR_DEVICE_TYPES)
    )
    if offenders:
        raise NotImplementedError(
            f"Circuit {circuit.name!r} contains nonlinear devices {offenders} whose "
            "nonlinear terminal law cannot be assembled into the constant-conductance "
            "stamp of the linear MNA / FDTD Norton-companion runtime; that path has no "
            "Newton iteration and would solve the network with the device absent. This "
            "fails closed until the nonlinear device-runtime integration lands."
        )


def compile_circuits(scene) -> tuple[CircuitGraph, ...]:
    circuits = tuple(getattr(scene, "circuits", ()))
    circuit_names = [circuit.name.casefold() for circuit in circuits]
    if len(set(circuit_names)) != len(circuit_names):
        raise ValueError("Scene circuit names must be unique.")
    ports = tuple(getattr(scene, "ports", ()))
    allowed_ports = {
        port.name: port
        for port in ports
        if type(port).__name__ in ("LumpedPort", "TerminalPort")
    }
    bound_port_names: set[str] = set()
    result = []
    for circuit in circuits:
        reject_nonlinear_devices(circuit)
        graph = compile_circuit_graph(circuit, available_ports=allowed_ports)
        for binding in graph.bindings:
            port = allowed_ports[binding.port_name]
            if getattr(port, "termination", None) is not None:
                raise ValueError(
                    f"EM port {binding.port_name!r} cannot have both a local termination "
                    "and a circuit binding."
                )
            key = binding.port_name
            if key in bound_port_names:
                raise ValueError(f"EM port {binding.port_name!r} is bound by more than one circuit.")
            bound_port_names.add(key)
        result.append(graph)
    return tuple(result)


__all__ = [
    "CircuitGraph",
    "compile_circuit_graph",
    "compile_circuits",
    "reject_nonlinear_devices",
]

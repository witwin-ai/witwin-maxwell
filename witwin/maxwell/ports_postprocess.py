"""Differentiable port functionals of caller-supplied, true-time Yee phasors."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import torch

from .compiler.ports import compile_ports
from .lumped import trapezoidal_impedance
from .ports import ModePort, TerminalWavePort, WavePort


@dataclass(frozen=True)
class PortDrive:
    """Prescribed source spectrum and resolved real Thevenin resistance."""

    source_spectrum: torch.Tensor
    source_impedance: torch.Tensor


def _resolve_port(prepared, port_name):
    solver = prepared.solver
    scene = solver.scene
    port = {port.name: port for port in scene.ports}[port_name]
    if isinstance(port, (ModePort, WavePort, TerminalWavePort)):
        raise NotImplementedError("Mode and wave ports require modal projections, not lumped field integrals.")
    if getattr(port, "wire_binding", None) is not None:
        raise NotImplementedError("Wire-bound ports require the wire charge state in addition to Yee fields.")
    if any(binding.port_name == port_name for circuit in scene.circuits for binding in circuit.bindings):
        raise NotImplementedError("Circuit-bound port currents require the circuit state in addition to Yee fields.")
    if any(port_name in network.connected_port_names for network in scene.networks):
        raise NotImplementedError("Network-bound port currents require the network state in addition to Yee fields.")
    for runtime in getattr(solver, "_port_runtimes", ()):
        if runtime.port.name == port_name:
            if runtime.sheet_feed is not None:
                raise NotImplementedError("Sheet-gap feeds require sheet trace fields in addition to Yee fields.")
            return port, runtime.geometry
    geometry, = compile_ports(scene, ports=(port,), device=scene.device)
    if getattr(geometry, "is_coaxial", False):
        from .fdtd.coaxial import harmonic_coaxial_geometry

        # FDFD owns the same material-resolved PEC edge suppression as FDTD.
        mask = torch.from_dlpack(solver._ensure_pec_mask())
        coefficients = {}
        offset = 0
        for component in ("Ex", "Ey", "Ez"):
            shape = tuple(getattr(scene, f"N{axis}_{component.lower()}") for axis in "xyz")
            count = shape[0] * shape[1] * shape[2]
            coefficients[f"c{component.lower()}_curl"] = (~mask[offset:offset + count]).reshape(shape)
            offset += count
        geometry = harmonic_coaxial_geometry(SimpleNamespace(scene=scene, **coefficients), geometry)
    return port, geometry


def _integral(fields, components, indices, weights):
    terms = []
    for component, row, weight in zip(components, indices, weights):
        values = fields[component][tuple(row.T)]
        terms.append((values * weight.to(values).reshape((-1,) + (1,) * (values.ndim - 1))).sum(dim=0))
    return sum(terms)


def _voltage(geometry, fields):
    if getattr(geometry, "is_coaxial", False):
        return _integral(fields, geometry.voltage_components, geometry.voltage_indices, geometry.voltage_weights)
    return _integral(fields, (geometry.voltage_component,), (geometry.voltage_indices,), (geometry.voltage_weights,))


def port_voltage(prepared, port_name, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Integrate E on the resolved voltage row, without a time factor.

    Fields have native Yee spatial axes and an optional trailing frequency axis.
    An unbound TerminalPort resolves to a lumped gap and takes the lumped path.
    Coaxial ports use the prepared material-resolved discrete-harmonic row.
    """
    _, geometry = _resolve_port(prepared, port_name)
    return _voltage(geometry, fields)


def port_phasors(prepared, port_name, fields, *, frequencies, dt, drive=None,
                 termination=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native port V/I from true-time-labelled Yee field phasors.

    Coupled voltage has the midpoint cosine factor; probe voltage and Ampere-loop
    current have no E/H shift. ``dt=None`` selects continuous time, with no cosine
    factor and the continuous termination impedance. ``termination`` overrides
    the scene load. An unbound TerminalPort takes the lumped-gap path.
    Excited ports require a PortDrive, normally from ``Result.port_drive``.
    """
    port, geometry = _resolve_port(prepared, port_name)
    voltage = _voltage(geometry, fields)
    omega = 2 * torch.pi * torch.as_tensor(frequencies, device=voltage.device, dtype=torch.float64)
    load = termination if termination is not None else getattr(
        prepared.solver, "_port_termination_overrides", {}).get(port_name, port.termination)
    if drive is None and any(excitation.port_name == port_name for excitation in prepared.simulation.excitations):
        raise ValueError(f"Excited port {port_name!r} requires a drive from Result.port_drive().")
    if drive is not None or load is not None:
        if dt is not None:
            voltage = torch.cos(omega * dt / 2) * voltage
        if drive is not None:
            return voltage, (drive.source_spectrum - voltage) / drive.source_impedance
        load = type(load)(**{key: None if value is None else value.to(omega)
                             for key, value in (("r", load.r), ("l", load.l), ("c", load.c))})
        impedance = load.impedance(omega) if dt is None else trapezoidal_impedance(load, omega, dt)
        return voltage, -voltage / impedance.conj()
    return voltage, _integral(fields, geometry.current_components, geometry.current_indices, geometry.current_weights)


__all__ = ["PortDrive", "port_voltage", "port_phasors", "trapezoidal_impedance"]

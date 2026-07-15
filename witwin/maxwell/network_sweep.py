from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .lumped import PortExcitation, PortSweep, SeriesRLC
from .network import NetworkData, PortData
from .sources import GaussianPulse


@dataclass(frozen=True)
class NetworkRunManifest:
    """Deterministic single-device excitation-column order for an RF sweep."""

    port_names: tuple[str, ...]
    all_rf_port_names: tuple[str, ...]
    frequencies: tuple[float, ...]
    reference_impedances: tuple[float, ...]
    source_time: object

    @property
    def valid_columns(self) -> tuple[bool, ...]:
        return (True,) * len(self.port_names)

    def metadata(self) -> dict[str, object]:
        return {
            "port_names": self.port_names,
            "all_rf_port_names": self.all_rf_port_names,
            "frequencies": self.frequencies,
            "reference_impedances": self.reference_impedances,
            "source_kind": getattr(self.source_time, "kind", None),
            "execution": "single_device_sequential",
        }


def _real_scalar_reference_impedance(port) -> float:
    value = torch.as_tensor(port.reference_impedance)
    if value.requires_grad:
        raise NotImplementedError(
            f"Port {port.name!r} has a trainable reference impedance, but PortSweep "
            "does not support RF-parameter gradients."
        )
    if value.ndim != 0:
        raise ValueError(
            f"Port {port.name!r} reference impedance must be scalar for an FDTD sweep."
        )
    if value.is_complex():
        if not bool(value.imag == 0.0):
            raise ValueError(
                f"Port {port.name!r} reference impedance must be real for an FDTD sweep."
            )
        value = value.real
    if not bool(torch.isfinite(value)) or not bool(value > 0.0):
        raise ValueError(
            f"Port {port.name!r} reference impedance must be positive and finite."
        )
    return float(value.detach().cpu())


def _default_sweep_source_time(frequencies: tuple[float, ...]):
    lower = min(frequencies)
    upper = max(frequencies)
    center = 0.5 * (lower + upper)
    width = max(upper - lower, 0.5 * lower)
    return GaussianPulse(frequency=center, fwidth=width)


def resolve_network_run_manifest(scene, sweep: PortSweep, frequencies) -> NetworkRunManifest:
    if not isinstance(sweep, PortSweep):
        raise TypeError("sweep must be a PortSweep.")
    if sweep.amplitude.requires_grad:
        raise NotImplementedError(
            "PortSweep amplitude is trainable, but PortSweep does not support "
            "RF-parameter gradients."
        )
    frequency_values = tuple(float(value) for value in frequencies)
    if not frequency_values:
        raise ValueError("PortSweep frequencies must not be empty.")
    if any(not math.isfinite(value) or value <= 0.0 for value in frequency_values):
        raise ValueError("PortSweep frequencies must be positive and finite.")
    if any(
        current <= previous
        for previous, current in zip(frequency_values, frequency_values[1:])
    ):
        raise ValueError("PortSweep frequencies must be strictly increasing.")
    rf_ports = tuple(
        port
        for port in scene.ports
        if hasattr(port, "reference_impedance") and hasattr(port, "termination")
    )
    if not rf_ports:
        raise ValueError("PortSweep requires at least one RF port in the Scene.")
    ports_by_name = {port.name: port for port in rf_ports}
    selected_names = tuple(ports_by_name) if sweep.ports is None else sweep.ports
    missing = tuple(name for name in selected_names if name not in ports_by_name)
    if missing:
        raise ValueError(f"PortSweep references missing RF ports: {missing}.")
    for port in rf_ports:
        termination = port.termination
        if termination is not None and any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in (termination.r, termination.l, termination.c)
        ):
            raise NotImplementedError(
                f"Port {port.name!r} has a trainable termination, but PortSweep "
                "does not support RF-parameter gradients."
            )
        if port.termination is not None:
            raise ValueError(
                f"Port {port.name!r} declares a termination; standard PortSweep supplies "
                "matched inactive terminations internally."
            )
        _real_scalar_reference_impedance(port)
    selected_impedances = tuple(
        _real_scalar_reference_impedance(ports_by_name[name]) for name in selected_names
    )
    source_time = sweep.source_time or _default_sweep_source_time(frequency_values)
    return NetworkRunManifest(
        port_names=selected_names,
        all_rf_port_names=tuple(port.name for port in rf_ports),
        frequencies=frequency_values,
        reference_impedances=selected_impedances,
        source_time=source_time,
    )


def build_network_column_run(scene, sweep: PortSweep, manifest: NetworkRunManifest, active_name: str):
    ports_by_name = {
        port.name: port
        for port in scene.ports
        if hasattr(port, "reference_impedance") and hasattr(port, "termination")
    }
    excitation = PortExcitation(
        active_name,
        amplitude=sweep.amplitude,
        source_impedance="matched",
        source_time=manifest.source_time,
    )
    overrides = {
        name: SeriesRLC(r=_real_scalar_reference_impedance(port))
        for name, port in ports_by_name.items()
        if name != active_name
    }
    return (excitation,), overrides


def _require_finite(value: torch.Tensor, *, label: str) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise RuntimeError(f"Network sweep {label} contains non-finite values.")


def _validate_column_contracts(
    manifest: NetworkRunManifest,
    columns: tuple[dict[str, PortData], ...],
) -> None:
    first_name = manifest.port_names[0]
    reference_entry = columns[0][first_name]
    if not isinstance(reference_entry, PortData):
        raise TypeError(
            f"Network sweep column 0 port {first_name!r} must contain PortData."
        )
    reference_frequencies = reference_entry.frequencies
    reference_device = reference_entry.voltage.device
    reference_dtype = reference_entry.voltage.dtype
    expected_frequencies = torch.as_tensor(
        manifest.frequencies,
        device=reference_frequencies.device,
        dtype=reference_frequencies.dtype,
    )
    if not torch.equal(reference_frequencies, expected_frequencies):
        raise ValueError("Network sweep column frequencies do not match the run manifest.")

    per_port_contracts = {}
    for column_index, column in enumerate(columns):
        if not isinstance(column, Mapping):
            raise TypeError(f"Network sweep column {column_index} must be a port-data mapping.")
        for port_index, port_name in enumerate(manifest.port_names):
            entry = column[port_name]
            if not isinstance(entry, PortData):
                raise TypeError(
                    f"Network sweep column {column_index} port {port_name!r} "
                    "must contain PortData."
                )
            if entry.port_name != port_name:
                raise ValueError(
                    f"Network sweep column {column_index} key {port_name!r} does not "
                    f"match PortData.port_name {entry.port_name!r}."
                )
            if entry.voltage.ndim != 1:
                raise ValueError(
                    f"Network sweep column {column_index} port {port_name!r} must "
                    "contain one-dimensional [F] voltage/current data."
                )
            if entry.voltage.device != reference_device:
                raise ValueError("Network sweep columns must be on the same device.")
            if entry.voltage.dtype != reference_dtype:
                raise TypeError("Network sweep columns must use the same signal dtype.")
            if entry.frequencies.device != reference_frequencies.device:
                raise ValueError("Network sweep column frequencies must be on the same device.")
            if entry.frequencies.dtype != reference_frequencies.dtype:
                raise TypeError("Network sweep column frequencies must use the same dtype.")
            if not torch.equal(entry.frequencies, reference_frequencies):
                raise ValueError("Network sweep columns must use identical frequencies.")
            expected_z0 = torch.full_like(entry.z0, manifest.reference_impedances[port_index])
            if not torch.equal(entry.z0, expected_z0):
                raise ValueError(
                    f"Network sweep column {column_index} port {port_name!r} reference "
                    "impedance does not match the run manifest."
                )
            _require_finite(entry.voltage, label=f"column {column_index} port {port_name!r} voltage")
            _require_finite(entry.current, label=f"column {column_index} port {port_name!r} current")

            contract = (
                entry.direction,
                entry.reference_plane,
                entry.phasor_convention,
                entry.power_wave_convention,
                entry.metadata.get("current_convention"),
            )
            previous = per_port_contracts.setdefault(port_name, contract)
            if contract != previous:
                raise ValueError(
                    f"Network sweep port {port_name!r} direction, reference plane, "
                    "or wave convention differs across excitation columns."
                )


def aggregate_network_columns(
    manifest: NetworkRunManifest,
    columns: tuple[dict[str, PortData], ...],
) -> tuple[dict[str, PortData], NetworkData]:
    port_count = len(manifest.port_names)
    if len(columns) != port_count:
        raise ValueError("Network sweep column count does not match the run manifest.")
    for column_index, column in enumerate(columns):
        if not isinstance(column, Mapping):
            raise TypeError(f"Network sweep column {column_index} must be a port-data mapping.")
        missing = tuple(name for name in manifest.port_names if name not in column)
        if missing:
            raise RuntimeError(
                f"Network sweep column {column_index} is missing port data for {missing}."
            )
    _validate_column_contracts(manifest, columns)

    s_columns = []
    for input_index, input_name in enumerate(manifest.port_names):
        driven = columns[input_index][input_name]
        incident = driven.a
        _require_finite(incident, label=f"input port {input_name!r} incident wave")
        incident_magnitude = torch.abs(incident)
        threshold = torch.clamp(
            torch.max(incident_magnitude) * 1.0e-6,
            min=torch.finfo(incident_magnitude.dtype).tiny,
        )
        if bool(torch.any(incident_magnitude < threshold)):
            weak_indices = (
                torch.nonzero(incident_magnitude < threshold)
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )
            raise RuntimeError(
                f"Network sweep input port {input_name!r} has incident-wave spectrum "
                f"below threshold at frequency indices {weak_indices}."
            )
        reflected_waves = []
        for output_name in manifest.port_names:
            reflected = columns[input_index][output_name].b
            _require_finite(
                reflected,
                label=f"column {input_index} output port {output_name!r} reflected wave",
            )
            reflected_waves.append(reflected)
        reflected = torch.stack(reflected_waves, dim=0)
        s_columns.append((reflected / incident.unsqueeze(0)).movedim(0, 1))
    scattering = torch.stack(s_columns, dim=-1)

    stacked_ports = {}
    z0_columns = []
    for port_name in manifest.port_names:
        entries = tuple(column[port_name] for column in columns)
        voltage = torch.stack([entry.voltage for entry in entries], dim=0)
        current = torch.stack([entry.current for entry in entries], dim=0)
        z0 = torch.stack([entry.z0 for entry in entries], dim=0)
        z0_columns.append(entries[0].z0)
        stacked_ports[port_name] = PortData(
            port_name=port_name,
            frequencies=entries[0].frequencies,
            voltage=voltage,
            current=current,
            z0=z0,
            direction=entries[0].direction,
            reference_plane=entries[0].reference_plane,
            metadata={
                "excitation_port_names": manifest.port_names,
                "run_manifest": manifest.metadata(),
            },
        )
    network_z0 = torch.stack(z0_columns, dim=-1)
    network = NetworkData(
        frequencies=columns[0][manifest.port_names[0]].frequencies,
        s=scattering,
        z0=network_z0,
        port_names=manifest.port_names,
        valid_columns=torch.ones(
            port_count,
            device=scattering.device,
            dtype=torch.bool,
        ),
        metadata={"run_manifest": manifest.metadata()},
    )
    return stacked_ports, network


__all__ = [
    "NetworkRunManifest",
    "aggregate_network_columns",
    "build_network_column_run",
    "resolve_network_run_manifest",
]

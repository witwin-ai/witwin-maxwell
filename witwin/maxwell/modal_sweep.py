"""Multi-port, multi-mode scattering matrices from ``ModePort`` declarations.

A ``ModalPortSweep`` builds the complete modal S-matrix of a photonic component.
Every selected ``(port, mode)`` channel is an independent input column: one FDTD
run excites only that channel's guided mode while every ``ModePort`` in the scene
is monitor-only, and the outgoing modal amplitudes at all channels are obtained
by projecting each port's monitor plane onto the requested reference modes.

Normalization. The FDTD mode solver returns unit-power reference modes, and the
overlap ``compute_mode_overlap`` amplitudes are scaled by ``sqrt(|P_mode|)`` so
each channel carries a power wave in ``sqrt(W)``: ``|a|^2`` is the guided power
travelling into the device along ``ModePort.direction`` and ``|b|^2`` the guided
power leaving it. Column ``j`` is ``S[:, :, j] = b / a_j`` with ``a_j`` the
measured incident amplitude of the driven channel, so the source amplitude, its
spectrum and the arbitrary (but deterministic) reference-mode phase cancel.
``NetworkData.z0`` is a nominal ``1`` that only labels these normalized modal
power waves; it is not a physical modal impedance.

Reference planes. S-parameters are referenced to each ``ModePort.position`` plane,
independent of ``ModePort(monitor_offset=...)``. The sampled monitor plane lies a
signed distance ``d = monitor_offset * sign(direction)`` into the device from that
reference plane, so the measured waves are de-embedded with the mode propagation
constant as ``a_ref = a_monitor * exp(-i*beta*d)`` and
``b_ref = b_monitor * exp(+i*beta*d)`` under the ``exp(-i*omega*t)`` phasor
convention. ``NetworkData.metadata["propagation_constants"]`` stores those
``beta`` values, so ``NetworkData.shift_reference_planes(...)`` moves the planes
further.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .network import NetworkData, PortData
from .ports import ModePort
from .sources import GaussianPulse, SourceTime


_POLARIZATION_TOKENS = {
    "ex": "x",
    "ey": "y",
    "ez": "z",
    "x": "x",
    "y": "y",
    "z": "z",
}


def _normalize_mode_entry(entry):
    """Return ``int`` (port polarization family) or ``(axis, index)``."""

    if isinstance(entry, bool):
        raise TypeError("ModalPortSweep mode entries must be int or (polarization, index).")
    if isinstance(entry, int):
        if entry < 0:
            raise ValueError("ModalPortSweep mode indices must be >= 0.")
        return int(entry)
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        polarization, index = entry
        axis = _POLARIZATION_TOKENS.get(str(polarization).lower())
        if axis is None:
            raise ValueError(
                "ModalPortSweep mode polarization must be one of Ex/Ey/Ez or x/y/z."
            )
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("ModalPortSweep mode indices must be integers >= 0.")
        return (axis, int(index))
    raise TypeError("ModalPortSweep mode entries must be int or (polarization, index).")


def _normalize_channel_key(key, *, role: str):
    if isinstance(key, str):
        if not key:
            raise ValueError(f"ModalPortSweep {role} channel labels must be non-empty.")
        return key
    if isinstance(key, (tuple, list)) and len(key) == 2 and isinstance(key[0], str):
        return (str(key[0]), _normalize_mode_entry(key[1]))
    raise TypeError(
        f"ModalPortSweep {role} channels must be a 'port::mode' label or a "
        "(port_name, mode) pair."
    )


def _normalize_element_mapping(mapping):
    if not isinstance(mapping, (tuple, list)) or len(mapping) != 3:
        raise TypeError(
            "ModalPortSweep element_mappings entries must be "
            "((out, in), (out, in), coefficient)."
        )
    source, destination, coefficient = mapping
    elements = []
    for element in (source, destination):
        if not isinstance(element, (tuple, list)) or len(element) != 2:
            raise TypeError(
                "ModalPortSweep element_mappings elements must be (out_channel, in_channel)."
            )
        elements.append(
            (
                _normalize_channel_key(element[0], role="element_mappings"),
                _normalize_channel_key(element[1], role="element_mappings"),
            )
        )
    if isinstance(coefficient, torch.Tensor):
        if coefficient.requires_grad or coefficient.numel() != 1:
            raise TypeError("ModalPortSweep element_mappings coefficients must be fixed scalars.")
        coefficient = coefficient.detach().cpu().item()
    resolved = complex(coefficient)
    if not (math.isfinite(resolved.real) and math.isfinite(resolved.imag)):
        raise ValueError("ModalPortSweep element_mappings coefficients must be finite.")
    return (elements[0], elements[1], resolved)


@dataclass(frozen=True)
class ModalPortSweep:
    """Request one independent FDTD run per selected ``(ModePort, mode)`` input.

    ``ports`` selects ``ModePort`` names (default: every ``ModePort`` in scene
    order). ``modes`` maps a port name to a sequence of mode entries, each an
    ``int`` mode index in that port's polarization family or a
    ``(polarization, index)`` pair, e.g. ``{"in": (0, 1, ("Ez", 0))}``; a bare
    ``int`` means one entry. Unspecified ports use their own declared mode.
    Channels are labelled ``"<port>::E<axis><index>"`` and ordered by port, then
    by mode entry. ``run_only`` restricts the driven input columns; columns that
    are neither run nor completely filled by ``element_mappings`` are reported as
    invalid through ``NetworkData.valid_columns``. Each element mapping
    ``((out, in), (out_dst, in_dst), c)`` sets ``S[out_dst, in_dst] = c * S[out, in]``
    and is applied in declaration order; its source element must already be
    known and its destination must not be.
    """

    ports: tuple[str, ...] | None = None
    modes: Mapping[str, tuple] | None = None
    run_only: tuple | None = None
    element_mappings: tuple = ()
    source_time: SourceTime | None = None
    kind: str = "modal_port_sweep"

    def __init__(
        self,
        ports=None,
        modes=None,
        run_only=None,
        element_mappings=(),
        source_time=None,
    ):
        if ports is None:
            resolved_ports = None
        else:
            resolved_ports = (ports,) if isinstance(ports, str) else tuple(str(p) for p in ports)
            if not resolved_ports or any(not name for name in resolved_ports):
                raise ValueError("ModalPortSweep ports must contain non-empty names.")
            if len(set(resolved_ports)) != len(resolved_ports):
                raise ValueError("ModalPortSweep ports must be unique.")
        if modes is None:
            resolved_modes = None
        else:
            if not isinstance(modes, Mapping):
                raise TypeError("ModalPortSweep modes must map port names to mode entries.")
            resolved_modes = {}
            for port_name, entries in modes.items():
                if isinstance(entries, int) and not isinstance(entries, bool):
                    entries = (entries,)
                normalized = tuple(_normalize_mode_entry(entry) for entry in entries)
                if not normalized:
                    raise ValueError(f"ModalPortSweep modes for port {port_name!r} must not be empty.")
                resolved_modes[str(port_name)] = normalized
        if run_only is None:
            resolved_run_only = None
        else:
            if isinstance(run_only, str):
                run_only = (run_only,)
            resolved_run_only = tuple(
                _normalize_channel_key(key, role="run_only") for key in run_only
            )
            if not resolved_run_only:
                raise ValueError("ModalPortSweep run_only must not be empty.")
        resolved_mappings = tuple(
            _normalize_element_mapping(mapping) for mapping in element_mappings
        )
        if source_time is not None and not isinstance(source_time, SourceTime):
            raise TypeError("ModalPortSweep source_time must be a SourceTime or None.")
        object.__setattr__(self, "ports", resolved_ports)
        object.__setattr__(self, "modes", resolved_modes)
        object.__setattr__(self, "run_only", resolved_run_only)
        object.__setattr__(self, "element_mappings", resolved_mappings)
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "kind", "modal_port_sweep")


@dataclass(frozen=True)
class ModalChannel:
    """One ``(ModePort, mode)`` row/column of the modal scattering matrix."""

    port_name: str
    polarization_axis: str
    mode_index: int
    direction: str
    reference_position: float
    reference_offset: float

    @property
    def mode_label(self) -> str:
        return f"E{self.polarization_axis}{self.mode_index}"

    @property
    def label(self) -> str:
        return f"{self.port_name}::{self.mode_label}"


@dataclass(frozen=True)
class ModalSweepPlan:
    """Deterministic channel order, driven columns and symmetry program."""

    channels: tuple[ModalChannel, ...]
    frequencies: tuple[float, ...]
    run_columns: tuple[int, ...]
    source_times: tuple[SourceTime, ...]
    element_mappings: tuple[tuple[int, int, int, int, complex], ...]
    known_elements: tuple[tuple[bool, ...], ...]

    @property
    def channel_labels(self) -> tuple[str, ...]:
        return tuple(channel.label for channel in self.channels)

    @property
    def port_names(self) -> tuple[str, ...]:
        names = []
        for channel in self.channels:
            if channel.port_name not in names:
                names.append(channel.port_name)
        return tuple(names)

    @property
    def valid_columns(self) -> tuple[bool, ...]:
        count = len(self.channels)
        return tuple(
            all(self.known_elements[out][column] for out in range(count))
            for column in range(count)
        )

    def metadata(self) -> dict[str, object]:
        labels = self.channel_labels
        return {
            "channel_labels": labels,
            "port_names": self.port_names,
            "frequencies": self.frequencies,
            "run_channels": tuple(labels[index] for index in self.run_columns),
            "source_kinds": tuple(getattr(source, "kind", None) for source in self.source_times),
            "element_mappings": tuple(
                ((labels[so], labels[si]), (labels[do], labels[di]), coefficient)
                for so, si, do, di, coefficient in self.element_mappings
            ),
            "valid_columns": self.valid_columns,
            "reference_positions": tuple(channel.reference_position for channel in self.channels),
            "reference_offsets": tuple(channel.reference_offset for channel in self.channels),
            "normalization": "unit-power modal power waves, sqrt(W)",
            "reference_plane": "ModePort.position (monitor_offset de-embedded with beta)",
            "execution": "single_device_sequential",
        }


def _validate_frequencies(frequencies) -> tuple[float, ...]:
    values = tuple(float(value) for value in frequencies)
    if not values:
        raise ValueError("ModalPortSweep frequencies must not be empty.")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("ModalPortSweep frequencies must be positive and finite.")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError("ModalPortSweep frequencies must be strictly increasing.")
    return values


def _default_source_time(frequencies: tuple[float, ...]) -> GaussianPulse:
    lower = min(frequencies)
    upper = max(frequencies)
    return GaussianPulse(
        frequency=0.5 * (lower + upper),
        fwidth=max(upper - lower, 0.5 * lower),
    )


def _resolve_mode(port: ModePort, entry) -> tuple[str, int]:
    if isinstance(entry, int):
        return port.polarization_axis, entry
    axis, index = entry
    if axis == port.normal_axis:
        raise ValueError(
            f"ModalPortSweep mode polarization E{axis} of port {port.name!r} must be "
            "tangential to the port plane."
        )
    return axis, index


def plan_modal_port_sweep(scene, sweep: ModalPortSweep, frequencies) -> ModalSweepPlan:
    """Resolve channels, driven columns and element mappings without running."""

    if not isinstance(sweep, ModalPortSweep):
        raise TypeError("sweep must be a ModalPortSweep.")
    frequency_values = _validate_frequencies(frequencies)
    non_modal = tuple(port.name for port in scene.ports if not isinstance(port, ModePort))
    if non_modal:
        raise ValueError(
            f"ModalPortSweep scenes may contain only ModePort ports; found {non_modal}."
        )
    if scene.sources:
        raise ValueError(
            "ModalPortSweep drives one ModePort per column; remove independent "
            "Scene.sources so they cannot contaminate the scattering columns."
        )
    ports_by_name = {port.name: port for port in scene.ports}
    if not ports_by_name:
        raise ValueError("ModalPortSweep requires at least one ModePort in the Scene.")
    selected = tuple(ports_by_name) if sweep.ports is None else sweep.ports
    missing = tuple(name for name in selected if name not in ports_by_name)
    if missing:
        raise ValueError(f"ModalPortSweep references missing ModePorts: {missing}.")
    modes = dict(sweep.modes or {})
    unknown = tuple(name for name in modes if name not in selected)
    if unknown:
        raise ValueError(f"ModalPortSweep modes reference unselected ports: {unknown}.")

    channels = []
    for name in selected:
        port = ports_by_name[name]
        entries = modes.get(name, (port.mode_index,))
        sign = 1.0 if port.direction == "+" else -1.0
        seen = set()
        for entry in entries:
            axis, index = _resolve_mode(port, entry)
            if (axis, index) in seen:
                raise ValueError(f"ModalPortSweep port {name!r} repeats mode E{axis}{index}.")
            seen.add((axis, index))
            channels.append(
                ModalChannel(
                    port_name=name,
                    polarization_axis=axis,
                    mode_index=index,
                    direction=port.direction,
                    reference_position=float(port.position["xyz".index(port.normal_axis)]),
                    reference_offset=float(port.monitor_offset) * sign,
                )
            )
    channels = tuple(channels)
    labels = tuple(channel.label for channel in channels)
    index_by_label = {label: index for index, label in enumerate(labels)}

    def resolve_key(key, *, role: str) -> int:
        if isinstance(key, str):
            label = key
        else:
            port_name, entry = key
            port = ports_by_name.get(port_name)
            if port is None or port_name not in selected:
                raise ValueError(f"ModalPortSweep {role} references unselected port {port_name!r}.")
            axis, index = _resolve_mode(port, entry)
            label = f"{port_name}::E{axis}{index}"
        if label not in index_by_label:
            raise ValueError(
                f"ModalPortSweep {role} channel {label!r} is not a sweep channel; "
                f"choices: {labels}."
            )
        return index_by_label[label]

    if sweep.run_only is None:
        run_columns = tuple(range(len(channels)))
    else:
        requested = tuple(resolve_key(key, role="run_only") for key in sweep.run_only)
        if len(set(requested)) != len(requested):
            raise ValueError("ModalPortSweep run_only channels must be unique.")
        run_columns = tuple(sorted(requested))

    for column in run_columns:
        channel = channels[column]
        if channel.reference_offset <= 0.0:
            raise ValueError(
                f"Driven ModePort {channel.port_name!r} needs its monitor plane inside "
                "the device side of the launch plane: monitor_offset must be nonzero "
                "and point along the port direction so the incident wave is measured."
            )

    count = len(channels)
    known = [[False] * count for _ in range(count)]
    for column in run_columns:
        for out in range(count):
            known[out][column] = True
    mappings = []
    for (source_out, source_in), (target_out, target_in), coefficient in sweep.element_mappings:
        so = resolve_key(source_out, role="element_mappings")
        si = resolve_key(source_in, role="element_mappings")
        do = resolve_key(target_out, role="element_mappings")
        di = resolve_key(target_in, role="element_mappings")
        if not known[so][si]:
            raise ValueError(
                f"ModalPortSweep element mapping source S[{labels[so]!r}, {labels[si]!r}] "
                "is neither computed nor filled by an earlier mapping."
            )
        if known[do][di]:
            raise ValueError(
                f"ModalPortSweep element mapping destination S[{labels[do]!r}, {labels[di]!r}] "
                "is already computed or mapped."
            )
        known[do][di] = True
        mappings.append((so, si, do, di, coefficient))

    source_times = tuple(
        sweep.source_time
        or ports_by_name[channels[column].port_name].source_time
        or _default_source_time(frequency_values)
        for column in run_columns
    )
    return ModalSweepPlan(
        channels=channels,
        frequencies=frequency_values,
        run_columns=run_columns,
        source_times=source_times,
        element_mappings=tuple(mappings),
        known_elements=tuple(tuple(row) for row in known),
    )


def _mode_port_variant(port: ModePort, *, polarization_axis, mode_index, source_time, frequencies):
    return ModePort(
        port.name,
        position=port.position,
        size=port.size,
        mode_index=mode_index,
        direction=port.direction,
        polarization=f"E{polarization_axis}",
        frequencies=frequencies,
        source_time=source_time,
        monitor_offset=port.monitor_offset,
        normal_direction=port.normal_direction,
        bend_radius=port.bend_radius,
        bend_axis=port.bend_axis,
    )


def build_modal_column_scene(scene, plan: ModalSweepPlan, run_position: int):
    """Clone ``scene`` with only the ``run_position``-th driven channel excited."""

    channel = plan.channels[plan.run_columns[run_position]]
    source_time = plan.source_times[run_position]
    ports = []
    for port in scene.ports:
        if port.name == channel.port_name:
            ports.append(
                _mode_port_variant(
                    port,
                    polarization_axis=channel.polarization_axis,
                    mode_index=channel.mode_index,
                    source_time=source_time,
                    frequencies=plan.frequencies,
                )
            )
        else:
            ports.append(
                _mode_port_variant(
                    port,
                    polarization_axis=port.polarization_axis,
                    mode_index=port.mode_index,
                    source_time=None,
                    frequencies=plan.frequencies,
                )
            )
    return scene.clone(ports=ports)


@dataclass(frozen=True)
class ModalColumnWaves:
    """Monitor-plane modal power waves of one driven column, shape ``[F, N]``."""

    incident: torch.Tensor
    outgoing: torch.Tensor
    beta: torch.Tensor


def extract_modal_column_waves(result, plan: ModalSweepPlan) -> ModalColumnWaves:
    """Project every channel's monitor plane onto its reference mode.

    The returned waves refer to the nominal monitor plane
    ``reference_position + reference_offset``; when the recorded plane snapped to
    a nearby grid plane, the residual distance is removed with the mode's own
    propagation constant so assembly sees a consistent planar geometry.
    """

    from .postprocess.modal import (
        _compute_mode_overlap_from_payload,
        _mode_source_from_mode_spec,
    )

    result = result._spectral_ratio_view()
    incident = []
    outgoing = []
    beta = []
    for frequency_index, frequency in enumerate(plan.frequencies):
        row_a = []
        row_b = []
        row_beta = []
        for channel in plan.channels:
            payload = result.raw_monitor(channel.port_name, freq_index=frequency_index)
            spec = dict(payload["mode_spec"])
            spec["mode_index"] = channel.mode_index
            spec["polarization_axis"] = channel.polarization_axis
            reference = _mode_source_from_mode_spec(spec, frequency=frequency)
            modal = _compute_mode_overlap_from_payload(
                result,
                channel.port_name,
                payload,
                mode_source=reference,
                direction=None,
            )
            beta_complex = modal.get("beta_complex")
            channel_beta = complex(beta_complex) if beta_complex is not None else complex(modal["beta"])
            sign = 1.0 if channel.direction == "+" else -1.0
            snap = (float(payload["position"]) - channel.reference_position) * sign - channel.reference_offset
            scale = torch.sqrt(torch.abs(modal["mode_power"]))
            row_a.append((modal["amplitude_forward"] * scale).to(torch.complex128) * cmath.exp(-1j * channel_beta * snap))
            row_b.append((modal["amplitude_backward"] * scale).to(torch.complex128) * cmath.exp(1j * channel_beta * snap))
            row_beta.append(channel_beta)
        incident.append(torch.stack(row_a))
        outgoing.append(torch.stack(row_b))
        beta.append(row_beta)
    incident = torch.stack(incident)
    outgoing = torch.stack(outgoing)
    return ModalColumnWaves(
        incident=incident,
        outgoing=outgoing,
        beta=torch.tensor(beta, device=incident.device, dtype=torch.complex128),
    )


def _require_finite(value: torch.Tensor, *, label: str) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise RuntimeError(f"Modal sweep {label} contains non-finite values.")


def assemble_modal_network(
    plan: ModalSweepPlan,
    columns: tuple[ModalColumnWaves, ...],
) -> tuple[dict[str, PortData], NetworkData]:
    """Assemble ``[F, out, in]`` modal S-parameters from driven-column waves."""

    if len(columns) != len(plan.run_columns):
        raise ValueError("Modal sweep column count does not match the planned driven columns.")
    frequency_count = len(plan.frequencies)
    count = len(plan.channels)
    labels = plan.channel_labels
    expected_shape = (frequency_count, count)
    for position, waves in enumerate(columns):
        for name in ("incident", "outgoing", "beta"):
            value = getattr(waves, name)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Modal sweep column {position} {name} must have shape [F, N] = {expected_shape}."
                )
            _require_finite(value, label=f"column {position} {name}")

    beta = columns[0].beta
    device = beta.device
    offsets = torch.tensor(
        [channel.reference_offset for channel in plan.channels],
        device=device,
        dtype=torch.float64,
    )
    inward = torch.exp(-1j * beta * offsets)
    outward = torch.exp(1j * beta * offsets)

    scattering = torch.zeros((frequency_count, count, count), device=device, dtype=torch.complex128)
    passive_incident = torch.zeros((frequency_count, count, count), device=device, dtype=torch.float64)
    reference_incident = []
    reference_outgoing = []
    for position, (column, waves) in enumerate(zip(plan.run_columns, columns)):
        if not torch.allclose(waves.beta.to(device), beta, rtol=1.0e-6, atol=0.0):
            raise ValueError("Modal sweep columns disagree on reference-mode propagation constants.")
        a_ref = waves.incident.to(device=device, dtype=torch.complex128) * inward
        b_ref = waves.outgoing.to(device=device, dtype=torch.complex128) * outward
        driven = a_ref[:, column]
        magnitude = torch.abs(driven)
        threshold = torch.clamp(
            torch.max(magnitude) * 1.0e-6,
            min=torch.finfo(magnitude.dtype).tiny,
        )
        weak = magnitude < threshold
        if bool(torch.any(weak)):
            raise RuntimeError(
                f"Modal sweep driven channel {labels[column]!r} has incident spectrum "
                f"below threshold at frequency indices "
                f"{torch.nonzero(weak).reshape(-1).cpu().tolist()}."
            )
        scattering[:, :, column] = b_ref / driven.unsqueeze(-1)
        passive_incident[:, :, column] = torch.abs(a_ref) / magnitude.unsqueeze(-1)
        passive_incident[:, column, column] = 0.0
        reference_incident.append(a_ref)
        reference_outgoing.append(b_ref)
    for so, si, do, di, coefficient in plan.element_mappings:
        scattering[:, do, di] = coefficient * scattering[:, so, si]

    frequency_tensor = torch.tensor(plan.frequencies, device=device, dtype=torch.float64)
    known = torch.tensor(plan.known_elements, device=device, dtype=torch.bool)
    network = NetworkData(
        frequencies=frequency_tensor,
        s=scattering,
        z0=1.0,
        port_names=labels,
        valid_columns=torch.tensor(plan.valid_columns, device=device, dtype=torch.bool),
        metadata={
            "run_manifest": plan.metadata(),
            "propagation_constants": beta,
            "known_elements": known,
            # Incident modal amplitude reaching each non-driven channel relative to
            # the driven one ([F, out, in]); large values flag re-entrant fields or
            # an under-absorbing boundary that a diagonal b/a extraction ignores.
            "passive_incident_ratio": passive_incident,
        },
    )

    ports = {}
    if reference_incident:
        incident_stack = torch.stack(reference_incident)  # [C, F, N]
        outgoing_stack = torch.stack(reference_outgoing)
        for port_name in plan.port_names:
            indices = [index for index, channel in enumerate(plan.channels) if channel.port_name == port_name]
            channel = plan.channels[indices[0]]
            ports[port_name] = PortData.from_power_waves(
                port_name=port_name,
                frequencies=frequency_tensor,
                a=incident_stack[:, :, indices].movedim(-1, -2),
                b=outgoing_stack[:, :, indices].movedim(-1, -2),
                z0=1.0,
                direction=channel.direction,
                reference_plane=channel.reference_position,
                mode_names=tuple(plan.channels[index].mode_label for index in indices),
                beta=beta[:, indices].transpose(0, 1),
                metadata={
                    "channel_labels": tuple(labels[index] for index in indices),
                    "excitation_channels": tuple(labels[index] for index in plan.run_columns),
                    "normalization": "unit-power modal power waves, sqrt(W)",
                },
            )
    return ports, network


def run_modal_port_sweep(simulation, sweep: ModalPortSweep, *, scene=None):
    """Run every planned column of ``sweep`` and return the assembled ``Result``.

    ``simulation`` is an ordinary ``Simulation.fdtd(...)`` whose scene declares the
    ``ModePort``s; it supplies the frequencies and FDTD configuration shared by all
    columns. Each column runs as an ordinary ``Simulation`` on a cloned scene.
    """

    from .result import Result
    from .simulation import SimulationMethod

    if simulation.method != SimulationMethod.FDTD:
        raise ValueError("ModalPortSweep requires a Simulation.fdtd(...) simulation.")
    if simulation.excitations or simulation.port_sweep is not None:
        raise ValueError("ModalPortSweep cannot be combined with RF port excitations.")
    run_scene = simulation.scene if scene is None else scene
    plan = plan_modal_port_sweep(run_scene, sweep, simulation.frequencies)

    columns = []
    column_stats = []
    first_result = None
    last_result = None
    for position in range(len(plan.run_columns)):
        column_scene = build_modal_column_scene(run_scene, plan, position)
        column_simulation = type(simulation)(
            scene=column_scene,
            method=simulation.method,
            frequencies=simulation.frequencies,
            config=simulation.config,
            excitations=None,
            metadata=simulation.metadata,
        )
        last_result = column_simulation.run()
        if first_result is None:
            first_result = last_result
        columns.append(extract_modal_column_waves(last_result, plan))
        column_stats.append(last_result.stats())

    ports, network = assemble_modal_network(plan, tuple(columns))
    # User monitors ride through from the first driven column, as in the RF sweeps.
    user_names = {monitor.name for monitor in run_scene.monitors}
    user_monitors = {
        name: payload for name, payload in first_result._monitors.items() if name in user_names
    }
    metadata = dict(simulation.metadata)
    metadata["modal_sweep_manifest"] = plan.metadata()
    if user_monitors:
        metadata["user_monitor_drive_channel"] = plan.channel_labels[plan.run_columns[0]]
    return Result(
        method="fdtd",
        scene=run_scene,
        prepared_scene=last_result.prepared_scene,
        frequency=plan.frequencies[0],
        frequencies=plan.frequencies,
        solver=last_result.solver,
        fields={},
        monitors=user_monitors,
        ports=ports,
        network=network,
        metadata=metadata,
        solver_stats={
            "dt": first_result.stats().get("dt"),
            "modal_sweep": plan.metadata(),
            "columns": tuple(column_stats),
        },
        raw_output={"modal_sweep_manifest": plan.metadata()},
    )


__all__ = [
    "ModalChannel",
    "ModalColumnWaves",
    "ModalPortSweep",
    "ModalSweepPlan",
    "assemble_modal_network",
    "build_modal_column_scene",
    "extract_modal_column_waves",
    "plan_modal_port_sweep",
    "run_modal_port_sweep",
]

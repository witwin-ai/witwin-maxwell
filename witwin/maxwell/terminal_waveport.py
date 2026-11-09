"""Multi-conductor TerminalWavePort: quasi-TEM terminal basis and CW column sweep.

The port cross-section is solved once for the terminal capacitance matrices
``C`` / ``C0`` and the quasi-TEM modes ``C v = eps_eff C0 v``. Every drive column
launches a superposition of those one-watt modal fields on the calibrated
per-frequency CW machinery shared with :class:`~witwin.maxwell.WavePort`; no new
FDTD kernel and no lumped circuit feed is involved.

Terminal quantities follow from the modal power waves ``a``/``b`` measured by the
modal overlap at the reference plane:

``V = Vm (a + b)``, ``I = Im (a - b)``

where column ``m`` of ``Vm`` / ``Im`` holds the terminal voltages/currents of the
unit-power mode ``m`` evaluated with the port's custom/composite integrals on the
actual mode field. With ``impedance_definition="power_voltage"`` the currents are
fixed by power, ``Im = 2 (Vm^H)^-1``, so the characteristic matrix is
``Zc = Vm Vm^H / 2``; ``"power_current"`` is the dual and ``"voltage_current"``
uses both integrals. The quasi-TEM ``C``/``C0`` impedance is reported only as a
cross-check.

Terminal power waves use a (possibly full) real symmetric reference matrix
``Zr``: ``a = F (V + Zr I)``, ``b = F (V - Zr I)``, ``F = (8 Re Zr)^-1/2``, which
reduces to the scalar peak-phasor Kurokawa waves on a diagonal reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch

from .compiler.sources import _compile_mode_source
from .compiler.terminal_waveports import CompiledTerminalWavePort, compile_terminal_waveport
from .compiler.path_integrals import audit_terminal_integral
from .constants import C_0
from .fdtd.excitation.modes import sample_mode_source_component, solve_mode_source_profile
from .lumped import PortSweep
from .monitors import ModeMonitor
from .network import NetworkData, PortData
from .ports import TerminalWavePort, WavePort
from .result import Result
from .sources import CW, ModeSource, _resolve_mode_source_polarization_axis

TERMINAL_SOURCE_PREFIX = "__terminal_waveport_source__::"

# Largest share of a mode's terminal charge that may terminate on the artificial
# (non-conductor) aperture walls before the truncated quasi-static mode is flagged.
APERTURE_WALL_CHARGE_LIMIT = 0.05


class TerminalWavePortApertureWarning(UserWarning):
    """The TerminalWavePort aperture truncates a modal field (see ``APERTURE_WALL_CHARGE_LIMIT``)."""


@dataclass(frozen=True)
class PreparedTerminalWavePort:
    port: TerminalWavePort
    compiled: CompiledTerminalWavePort
    polarization_axis: str
    mode_data: tuple
    basis: dict
    integral_audit: dict
    modal_potentials: torch.Tensor
    eps_eff: torch.Tensor
    beta: torch.Tensor
    terminal_voltage: torch.Tensor
    terminal_current: torch.Tensor
    characteristic_impedance: torch.Tensor
    modal_impedance: torch.Tensor
    quasi_tem_impedance: torch.Tensor
    loop_impedance: torch.Tensor | None
    reference_impedance: torch.Tensor | None
    drive: torch.Tensor
    wall_charge_fraction: torch.Tensor

    @property
    def channel_count(self) -> int:
        return self.port.terminal_count

    def wave_transform(self, frequency_index: int):
        """Terminal power-wave operator ``F`` and the reference matrix ``Zr``."""
        reference = self.reference_impedance[frequency_index]
        return _wave_operator(reference), reference


@dataclass(frozen=True)
class TerminalWavePortRunManifest:
    physical_port_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    frequencies: tuple[float, ...]
    prepared_ports: tuple[PreparedTerminalWavePort, ...]

    def metadata(self) -> dict[str, object]:
        return {
            "physical_port_names": self.physical_port_names,
            "channel_names": self.channel_names,
            "frequencies": self.frequencies,
            "execution": "single_device_frequency_sequential_cw",
            "normalization": "terminal power waves from one-watt quasi-TEM modal fields",
            "port_kind": "terminal_wave_port",
        }


def _wave_operator(reference: torch.Tensor) -> torch.Tensor:
    """``F = (8 Re Zr)^-1/2`` (symmetric root) for the generalized power waves."""
    resistance = torch.real(reference)
    resistance = .5 * (resistance + resistance.transpose(-2, -1))
    values, vectors = torch.linalg.eigh(resistance)
    if not bool(torch.all(values > 0)):
        raise ValueError("TerminalWavePort reference impedance must have a positive-definite real part.")
    root = vectors @ torch.diag_embed((8.0 * values).rsqrt()) @ vectors.transpose(-2, -1)
    return root.to(torch.complex128)


def _field_component_points(scene, field_name: str, indices: torch.Tensor) -> torch.Tensor:
    from .waveport_sweep import _field_component_points as native_points
    return native_points(scene, field_name, indices)


def evaluate_integral(scene, compiled_integral, mode_data) -> torch.Tensor:
    """Evaluate one compiled custom/composite integral on a modal transverse profile."""
    total = torch.zeros((), dtype=torch.complex128, device=mode_data["coords_u"].device)
    for component, indices, weights in compiled_integral.terms:
        points = _field_component_points(scene, component, indices.to(scene.device))
        values = sample_mode_source_component(mode_data, points.to(mode_data["coords_u"].dtype), component)
        total = total + torch.sum(values.to(torch.complex128) * weights.to(values.device, torch.complex128))
    return total


def _port_mode_source(port: TerminalWavePort, compiled: CompiledTerminalWavePort, polarization_axis, *,
                      mode_index, potentials, frequency, position=None, coefficient=1.0, name=None):
    source = ModeSource(
        position=port.position if position is None else position,
        size=port.size,
        mode_index=int(mode_index),
        direction=port.direction,
        polarization=f"E{polarization_axis}",
        source_time=CW(frequency=float(frequency), amplitude=abs(coefficient), phase=_phase(coefficient)),
        name=name or f"{port.name}::M{mode_index + 1}::basis",
    )
    _tag_terminal_mode(source, port, compiled, potentials)
    return source


def _phase(value) -> float:
    import cmath
    return cmath.phase(complex(value))


def _tag_terminal_mode(item, port, compiled, potentials):
    object.__setattr__(item, "_wave_family", "tem")
    object.__setattr__(item, "_conductor_model", port.conductor_model)
    object.__setattr__(item, "_terminal_mode", compiled.terminal_mode(potentials))


def _quasi_tem_impedance(capacitance, modal, eps_eff) -> torch.Tensor:
    """Cross-check ``Zc = V (I)^-1`` with ``I_m = c / sqrt(eps_eff_m) C V_m``."""
    currents = capacitance @ modal * (C_0 / torch.sqrt(eps_eff))[None, :]
    return torch.linalg.solve(currents.transpose(0, 1), modal.transpose(0, 1)).transpose(0, 1)


def _characteristic(definition, voltage, current):
    """Return ``(Vm, Im, Zc)`` for one frequency from the integral matrices."""
    if definition == "power_voltage":
        current = 2.0 * torch.linalg.inv(voltage.conj().transpose(0, 1))
    elif definition == "power_current":
        voltage = 2.0 * torch.linalg.inv(current.conj().transpose(0, 1))
    impedance = torch.linalg.solve(current.transpose(0, 1), voltage.transpose(0, 1)).transpose(0, 1)
    return voltage, current, impedance


def _resolve_reference(port: TerminalWavePort, characteristic: torch.Tensor) -> torch.Tensor | None:
    if port.basis == "modal":
        return None
    frequency_count, count, _ = characteristic.shape
    if port.reference_impedance == "characteristic":
        real = torch.real(characteristic)
        return (.5 * (real + real.transpose(-2, -1))).to(torch.complex128)
    matrix = torch.as_tensor(port.reference_impedance, dtype=torch.complex128, device=characteristic.device)
    return matrix.expand(frequency_count, count, count).clone()


def _prepare_port(context, port: TerminalWavePort, frequencies) -> PreparedTerminalWavePort:
    compiled = compile_terminal_waveport(context.scene, port)
    polarization_axis = _resolve_mode_source_polarization_axis(port.normal_axis, port.size, "auto")
    basis_source = _compile_mode_source(
        _port_mode_source(port, compiled, polarization_axis, mode_index=0, potentials=None,
                          frequency=frequencies[0]),
        default_frequency=float(frequencies[0]),
    )
    basis_data = solve_mode_source_profile(context, basis_source)
    record = basis_data["terminal_mode"]
    basis = dict(record["basis"])
    label_map = record["label_map"].cpu().numpy()
    basis["label_map"] = record["label_map"]
    modal = basis["modal_potentials"]
    count = port.terminal_count
    audits = {}
    for role, integrals in (("voltage", compiled.voltage_integrals), ("current", compiled.current_integrals)):
        if integrals is None:
            continue
        for index, integral in enumerate(integrals):
            try:
                audits[f"{role}/T{index + 1}"] = audit_terminal_integral(integral, label_map, index)
            except ValueError as error:
                raise ValueError(f"TerminalWavePort {port.name!r}: {error}") from error

    mode_rows = []
    beta_rows = []
    voltage_rows = []
    current_rows = []
    impedance_rows = []
    loop_rows = []
    for frequency in frequencies:
        modes = []
        betas = []
        voltage = torch.empty((count, count), dtype=torch.complex128, device=modal.device)
        loop_current = torch.empty_like(voltage)
        for mode in range(count):
            source = _compile_mode_source(
                _port_mode_source(port, compiled, polarization_axis, mode_index=mode,
                                  potentials=modal[:, mode], frequency=frequency),
                default_frequency=float(frequency),
            )
            data = solve_mode_source_profile(context, source)
            modes.append(data)
            beta = data.get("beta_complex")
            betas.append(complex(data["beta"]) if beta is None else complex(beta))
            for terminal, integral in enumerate(compiled.voltage_integrals):
                voltage[terminal, mode] = evaluate_integral(context.scene, integral, data).to(modal.device)
            if compiled.current_integrals is not None:
                for terminal, integral in enumerate(compiled.current_integrals):
                    loop_current[terminal, mode] = evaluate_integral(context.scene, integral, data).to(modal.device)
        current = loop_current if compiled.current_integrals is not None else None
        terminal_voltage, terminal_current, impedance = _characteristic(
            port.impedance_definition, voltage, loop_current if current is not None else voltage)
        if current is not None:
            loop_rows.append(torch.linalg.solve(current.transpose(0, 1), voltage.transpose(0, 1)).transpose(0, 1))
        if not mode_rows:
            wall_fraction = torch.as_tensor([data["terminal_mode"]["wall_charge_fraction"] for data in modes],
                                            dtype=torch.float64)
            for mode, value in enumerate(wall_fraction.tolist()):
                if value > APERTURE_WALL_CHARGE_LIMIT:
                    warnings.warn(
                        f"TerminalWavePort {port.name!r} mode M{mode + 1}: {100 * value:.1f} % of the terminal "
                        f"charge terminates on the artificial aperture walls (limit "
                        f"{100 * APERTURE_WALL_CHARGE_LIMIT:.0f} %); the grounded aperture truncates this mode "
                        "and its launch/projection loses power. Enlarge the aperture (about 7 substrate "
                        "heights above the ground and 6 beyond the outer strip edges for microstrip).",
                        TerminalWavePortApertureWarning,
                        stacklevel=3,
                    )
        mode_rows.append(tuple(modes))
        beta_rows.append(torch.as_tensor(betas, dtype=torch.complex128))
        voltage_rows.append(terminal_voltage)
        current_rows.append(terminal_current)
        impedance_rows.append(impedance)
    terminal_voltage = torch.stack(voltage_rows)
    terminal_current = torch.stack(current_rows)
    characteristic = torch.stack(impedance_rows)
    power = .5 * torch.real(torch.sum(terminal_voltage.conj() * terminal_current, dim=1))
    modal_impedance = (torch.sum(terminal_voltage.abs().square(), dim=1) / (2.0 * power)).transpose(0, 1)
    if not bool(torch.all(torch.isfinite(modal_impedance))) or not bool(torch.all(modal_impedance > 0)):
        raise RuntimeError(
            f"TerminalWavePort {port.name!r} produced a non-positive modal impedance; check the "
            "voltage path orientation (terminal to reference) and loop placement."
        )
    reference = _resolve_reference(port, characteristic)
    drive = []
    for frequency_index in range(len(frequencies)):
        if reference is None:
            drive.append(torch.eye(count, dtype=torch.complex128, device=modal.device))
            continue
        operator = _wave_operator(reference[frequency_index])
        incident = operator @ (terminal_voltage[frequency_index]
                               + reference[frequency_index] @ terminal_current[frequency_index])
        drive.append(torch.linalg.inv(incident))
    quasi_tem = _quasi_tem_impedance(basis["capacitance"], modal, basis["eps_eff"])
    return PreparedTerminalWavePort(
        port=port,
        compiled=compiled,
        polarization_axis=polarization_axis,
        mode_data=tuple(mode_rows),
        basis=basis,
        integral_audit=audits,
        modal_potentials=modal,
        eps_eff=basis["eps_eff"],
        beta=torch.stack(beta_rows).transpose(0, 1),
        terminal_voltage=terminal_voltage,
        terminal_current=terminal_current,
        characteristic_impedance=characteristic,
        modal_impedance=modal_impedance.to(torch.complex128),
        quasi_tem_impedance=quasi_tem,
        loop_impedance=torch.stack(loop_rows) if loop_rows else None,
        reference_impedance=reference,
        drive=torch.stack(drive),
        wall_charge_fraction=wall_fraction,
    )


def resolve_terminal_waveport_run_manifest(scene, sweep: PortSweep, frequencies, *, requested_dt=None,
                                           base_frequency=None, max_electric_field=None):
    from .waveport_sweep import _mode_context

    if not isinstance(sweep, PortSweep):
        raise TypeError("sweep must be a PortSweep.")
    if sweep.source_time is not None:
        raise ValueError("TerminalWavePort PortSweep uses a calibrated per-frequency CW basis; source_time must be None.")
    frequency_values = tuple(float(value) for value in frequencies)
    if not frequency_values or any(not math.isfinite(value) or value <= 0.0 for value in frequency_values):
        raise ValueError("TerminalWavePort PortSweep frequencies must be finite and positive.")
    if any(right <= left for left, right in zip(frequency_values, frequency_values[1:])):
        raise ValueError("TerminalWavePort PortSweep frequencies must be strictly increasing.")
    terminal_ports = tuple(port for port in scene.ports if isinstance(port, TerminalWavePort))
    if not terminal_ports:
        raise ValueError("TerminalWavePort PortSweep requires at least one TerminalWavePort.")
    if any(isinstance(port, WavePort) for port in scene.ports):
        raise NotImplementedError(
            "A TerminalWavePort sweep cannot yet be combined with single-mode WavePorts in one scene; "
            "declare every modal aperture as a TerminalWavePort."
        )
    unterminated = tuple(
        port.name for port in scene.ports
        if not isinstance(port, TerminalWavePort) and hasattr(port, "termination") and port.termination is None
    )
    if unterminated:
        raise ValueError(
            "A TerminalWavePort run can retain lumped/terminal ports only as explicit passive "
            f"terminations; unterminated entries are {unterminated}."
        )
    if scene.sources:
        raise ValueError("TerminalWavePort PortSweep requires a Scene without independent field sources.")
    ports_by_name = {port.name: port for port in terminal_ports}
    selected = tuple(ports_by_name) if sweep.ports is None else tuple(sweep.ports)
    missing = tuple(name for name in selected if name not in ports_by_name)
    if missing:
        raise ValueError(f"PortSweep references missing TerminalWavePort entries: {missing}.")
    context = _mode_context(scene, requested_dt=requested_dt, base_frequency=base_frequency,
                            max_electric_field=max_electric_field)
    prepared = tuple(_prepare_port(context, ports_by_name[name], frequency_values) for name in selected)
    return TerminalWavePortRunManifest(
        physical_port_names=selected,
        channel_names=tuple(name for item in prepared for name in item.port.channel_names),
        frequencies=frequency_values,
        prepared_ports=prepared,
    )


def _monitor_name(port_name: str, mode: int) -> str:
    from .waveport_sweep import WAVEPORT_MONITOR_PREFIX
    return f"{WAVEPORT_MONITOR_PREFIX}{port_name}::M{mode + 1}"


def _column_scene(scene, manifest, *, active_port: int, active_channel: int, frequency_index: int, amplitude: complex):
    from .waveport_sweep import _source_plane

    frequency = manifest.frequencies[frequency_index]
    sources = []
    monitors = []
    for port_index, prepared in enumerate(manifest.prepared_ports):
        port, compiled = prepared.port, prepared.compiled
        for mode in range(port.terminal_count):
            monitor = ModeMonitor(
                _monitor_name(port.name, mode),
                position=port.position,
                size=port.size,
                mode_index=mode,
                direction=port.direction,
                polarization=f"E{prepared.polarization_axis}",
                frequencies=(frequency,),
                normal_direction=port.direction,
            )
            _tag_terminal_mode(monitor, port, compiled, prepared.modal_potentials[:, mode])
            monitors.append(monitor)
        if port_index != active_port:
            continue
        position = _source_plane(scene, prepared)
        coefficients = prepared.drive[frequency_index][:, active_channel]
        coefficient_floor = 64 * torch.finfo(coefficients.dtype).eps * abs(amplitude) * float(coefficients.abs().max().item())
        for mode in range(port.terminal_count):
            coefficient = amplitude * complex(coefficients[mode].detach().cpu())
            if abs(coefficient) <= coefficient_floor:
                continue
            sources.append(_port_mode_source(
                port, compiled, prepared.polarization_axis, mode_index=mode,
                potentials=prepared.modal_potentials[:, mode], frequency=frequency,
                position=position, coefficient=coefficient,
                name=f"{TERMINAL_SOURCE_PREFIX}{port.name}::{active_channel}::{mode}",
            ))
    retained = tuple(port for port in scene.ports if not isinstance(port, TerminalWavePort))
    return scene.clone(ports=retained, sources=sources, monitors=[*scene.monitors, *monitors])


def _modal_waves(result, prepared, frequency_index):
    result = result._spectral_ratio_view()
    forward, backward = [], []
    for mode in range(prepared.port.terminal_count):
        modal = result.monitor(_monitor_name(prepared.port.name, mode), resolve_modal=True)
        complex_mode = prepared.mode_data[frequency_index][mode].get("beta_complex") is not None
        scale = 1.0 if complex_mode else torch.sqrt(torch.abs(modal["mode_power"]))
        forward.append(modal["amplitude_forward"] * scale)
        backward.append(modal["amplitude_backward"] * scale)
    return torch.stack(forward).to(torch.complex128).reshape(-1), torch.stack(backward).to(torch.complex128).reshape(-1)


def _channel_waves(prepared, frequency_index, forward, backward):
    """Return ``(a, b, V, I)`` in the port's channel basis."""
    forward = forward.to(prepared.terminal_voltage.device)
    backward = backward.to(prepared.terminal_voltage.device)
    voltage = prepared.terminal_voltage[frequency_index] @ (forward + backward)
    current = prepared.terminal_current[frequency_index] @ (forward - backward)
    if prepared.port.basis == "modal":
        return forward, backward, voltage, current
    operator, reference = prepared.wave_transform(frequency_index)
    return operator @ (voltage + reference @ current), operator @ (voltage - reference @ current), voltage, current


def run_terminal_waveport_sweep(simulation, scene, manifest: TerminalWavePortRunManifest) -> Result:
    from .array_execution import ArrayRunData, compact_array_column_result
    from .waveport_sweep import _user_monitor_payloads

    amplitude = complex(simulation.port_sweep.amplitude.detach().cpu())
    channel_locations = [(port_index, channel)
                         for port_index, prepared in enumerate(manifest.prepared_ports)
                         for channel in range(prepared.channel_count)]
    incident_columns, reflected_columns, voltage_columns, current_columns = [], [], [], []
    column_stats, column_results = [], []
    first_result = last_result = shared_prepared = None
    for active_port, active_channel in channel_locations:
        incident_f, reflected_f, voltage_f, current_f, stats, results = [], [], [], [], [], []
        for frequency_index, frequency in enumerate(manifest.frequencies):
            run_scene = _column_scene(scene, manifest, active_port=active_port, active_channel=active_channel,
                                      frequency_index=frequency_index, amplitude=amplitude)
            sub_simulation = type(simulation)(
                scene=run_scene,
                method=simulation.method,
                frequencies=(frequency,),
                config=simulation.config,
                excitations=None,
                metadata=simulation.metadata,
            )
            sub_simulation._fixed_waveport_mode_sources = True
            last_result = sub_simulation.run()
            if first_result is None:
                first_result = last_result
            if shared_prepared is None:
                shared_prepared = last_result.prepared_scene
            results.append(compact_array_column_result(last_result, prepared_scene=shared_prepared))
            parts = [_channel_waves(prepared, frequency_index, *_modal_waves(last_result, prepared, frequency_index))
                     for prepared in manifest.prepared_ports]
            incident_f.append(torch.cat([part[0] for part in parts]))
            reflected_f.append(torch.cat([part[1] for part in parts]))
            voltage_f.append(torch.cat([part[2] for part in parts]))
            current_f.append(torch.cat([part[3] for part in parts]))
            stats.append(last_result.stats())
        incident_columns.append(torch.stack(incident_f))
        reflected_columns.append(torch.stack(reflected_f))
        voltage_columns.append(torch.stack(voltage_f))
        current_columns.append(torch.stack(current_f))
        column_stats.append(tuple(stats))
        column_results.append(tuple(results))
    incident = torch.stack(incident_columns)    # [drive, frequency, channel]
    reflected = torch.stack(reflected_columns)
    voltages = torch.stack(voltage_columns)
    currents = torch.stack(current_columns)
    incident_matrix = incident.permute(1, 2, 0).contiguous()
    reflected_matrix = reflected.permute(1, 2, 0).contiguous()
    condition = torch.linalg.cond(incident_matrix)
    scattering = torch.linalg.solve(incident_matrix.transpose(-2, -1),
                                    reflected_matrix.transpose(-2, -1)).transpose(-2, -1)
    frequency_tensor = torch.as_tensor(manifest.frequencies, device=scattering.device, dtype=torch.float64)
    channel_count = len(manifest.channel_names)

    z0_columns, reference_blocks, transform_blocks, modal_z0 = [], [], [], []
    ports = {}
    offset = 0
    for prepared in manifest.prepared_ports:
        port = prepared.port
        count = prepared.channel_count
        window = slice(offset, offset + count)
        offset += count
        device = scattering.device
        modal_impedance = prepared.modal_impedance.to(device)
        voltage_mm = prepared.terminal_voltage.to(device)
        transform = torch.diag_embed(torch.sqrt(2.0 * modal_impedance.transpose(0, 1))) @ torch.linalg.inv(voltage_mm)
        transform_blocks.append(transform)
        modal_z0.append(modal_impedance.transpose(0, 1))
        metadata = {
            "channel_names": manifest.channel_names[window],
            "basis": port.basis,
            "terminal_names": tuple(ref.structure_name for ref in port.terminals),
            "reference_names": tuple(ref.structure_name for ref in port.reference),
            "impedance_definition": port.impedance_definition,
            "characteristic_impedance_matrix": prepared.characteristic_impedance.to(device),
            "quasi_tem_impedance_matrix": prepared.quasi_tem_impedance.to(device),
            "loop_impedance_matrix": None if prepared.loop_impedance is None else prepared.loop_impedance.to(device),
            "terminal_voltage_per_mode": voltage_mm,
            "terminal_current_per_mode": prepared.terminal_current.to(device),
            "capacitance": prepared.basis["capacitance"].to(device),
            "vacuum_capacitance": prepared.basis["vacuum_capacitance"].to(device),
            "modal_potentials": prepared.modal_potentials.to(device),
            "effective_permittivity": prepared.eps_eff.to(device),
            "integral_audit": prepared.integral_audit,
            "aperture_wall_charge_fraction": prepared.wall_charge_fraction,
            "normalization": "terminal power waves from one-watt quasi-TEM modal fields",
        }
        port_voltage = voltages[:, :, window].movedim(-1, -2)
        port_current = currents[:, :, window].movedim(-1, -2)
        if port.basis == "modal":
            z0 = modal_impedance.transpose(0, 1)                       # [F, M]
            ports[port.name] = PortData.from_power_waves(
                port_name=port.name, frequencies=frequency_tensor,
                a=incident[:, :, window].movedim(-1, -2), b=reflected[:, :, window].movedim(-1, -2),
                z0=modal_impedance, direction=port.direction, reference_plane=port.reference_plane,
                mode_names=port.mode_names, beta=prepared.beta.to(device),
                characteristic_impedance=modal_impedance,
                metadata={**metadata, "terminal_voltage": port_voltage, "terminal_current": port_current},
            )
            reference = torch.diag_embed(z0)
        else:
            reference = prepared.reference_impedance.to(device)
            diagonal = torch.diagonal(reference, dim1=-2, dim2=-1)          # [F, N]
            metadata["reference_impedance_matrix"] = reference
            ports[port.name] = PortData(
                port_name=port.name, frequencies=frequency_tensor,
                voltage=port_voltage, current=port_current,
                z0=diagonal.transpose(0, 1), direction=port.direction, reference_plane=port.reference_plane,
                metadata={**metadata, "terminal_names": port.terminal_names,
                          "power_wave_note": "PortData.a/b use the reference diagonal; the network S uses the full matrix."},
            )
            z0 = diagonal
        z0_columns.append(z0)
        reference_blocks.append(reference)
    network_z0 = torch.cat(z0_columns, dim=-1)
    reference_matrix = torch.zeros((len(manifest.frequencies), channel_count, channel_count),
                                   dtype=torch.complex128, device=scattering.device)
    transform_matrix = torch.zeros_like(reference_matrix)
    offset = 0
    for reference, transform in zip(reference_blocks, transform_blocks):
        count = reference.shape[-1]
        reference_matrix[:, offset:offset + count, offset:offset + count] = reference
        transform_matrix[:, offset:offset + count, offset:offset + count] = transform
        offset += count
    coupled = bool(torch.any(reference_matrix - torch.diag_embed(torch.diagonal(reference_matrix, dim1=-2, dim2=-1)) != 0))
    terminal_basis = all(prepared.port.basis == "terminal" for prepared in manifest.prepared_ports)
    network = NetworkData(
        frequencies=frequency_tensor,
        s=scattering,
        z0=network_z0,
        port_names=manifest.channel_names,
        valid_columns=torch.ones(channel_count, device=scattering.device, dtype=torch.bool),
        metadata={
            "run_manifest": manifest.metadata(),
            "extraction_condition_number": condition,
            "terminal_transform": {
                "voltage_basis": transform_matrix,
                "z0": torch.cat(modal_z0, dim=-1),
                "port_names": tuple(f"{prepared.port.name}::M{index + 1}"
                                    for prepared in manifest.prepared_ports
                                    for index in range(prepared.channel_count)),
            } if terminal_basis else None,
        },
        reference_matrix=reference_matrix if coupled else None,
    )
    user_monitors = _user_monitor_payloads(first_result, scene)
    note = {"user_monitor_drive_channel": manifest.channel_names[0],
            "user_monitor_frequency": manifest.frequencies[0]} if user_monitors else {}
    return Result(
        method="fdtd",
        scene=scene,
        prepared_scene=last_result.prepared_scene,
        frequency=manifest.frequencies[0],
        frequencies=manifest.frequencies,
        solver=last_result.solver,
        fields={},
        monitors=user_monitors,
        ports=ports,
        network=network,
        metadata={**simulation.metadata, "network_run_manifest": manifest.metadata(), **note},
        solver_stats={"dt": first_result.stats().get("dt"), "network_sweep": manifest.metadata(),
                      "columns": tuple(column_stats)},
        raw_output={"network_run_manifest": manifest.metadata()},
        array_run_data=ArrayRunData(
            manifest=manifest,
            column_results=tuple(column_results),
            incident=torch.stack([incident[index, :, index] for index in range(channel_count)], dim=-1),
        ),
    )


__all__ = [
    "APERTURE_WALL_CHARGE_LIMIT",
    "PreparedTerminalWavePort",
    "TerminalWavePortApertureWarning",
    "TerminalWavePortRunManifest",
    "evaluate_integral",
    "resolve_terminal_waveport_run_manifest",
    "run_terminal_waveport_sweep",
]

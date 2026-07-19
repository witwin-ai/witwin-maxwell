from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .adjoint_inputs import scene_trainable_material_tensors
from .compiler.sources import _compile_mode_source
from .fdtd.excitation.modes import (
    sample_mode_source_component,
    solve_mode_source_profile,
)
from .lumped import PortSweep
from .monitors import ModeMonitor
from .network import NetworkData, PortData
from .ports import WavePort
from .postprocess.waveports import ModeTrackingResult, track_modes
from .result import Result
from .scene import prepare_scene
from .sources import CW, ModeSource


_EPS0 = 8.8541878128e-12
_MU0 = 1.25663706212e-6
_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

# Prefix stamped on the internal per-port ModeMonitors used to extract the
# S-matrix. User-declared monitors never carry this prefix, so it is the single
# discriminator between machinery monitors (dropped from the user-facing Result)
# and monitors the user attached to the scene (which must ride through).
WAVEPORT_MONITOR_PREFIX = "__waveport__::"


@dataclass(frozen=True)
class PreparedWavePort:
    port: WavePort
    compiled: object
    mode_data: tuple[tuple[dict[str, object], ...], ...]
    tracking: ModeTrackingResult
    characteristic_impedance: torch.Tensor
    tracking_confidence: torch.Tensor


@dataclass(frozen=True)
class WavePortRunManifest:
    physical_port_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    frequencies: tuple[float, ...]
    prepared_ports: tuple[PreparedWavePort, ...]

    def metadata(self) -> dict[str, object]:
        return {
            "physical_port_names": self.physical_port_names,
            "channel_names": self.channel_names,
            "frequencies": self.frequencies,
            "execution": "single_device_frequency_sequential_cw",
            "normalization": "tracked one-watt modal power waves",
        }


def _mode_context(scene):
    prepared = prepare_scene(scene)
    return SimpleNamespace(
        scene=prepared,
        Ex=torch.empty((1,), device=prepared.device, dtype=torch.float32),
        c=299792458.0,
        boundary_kind=prepared.boundary.kind,
        _compiled_material_model=prepared.compile_materials(),
        _mode_source_rebuild_from_fields=False,
    )


def _mode_source(port: WavePort, compiled_mode, *, frequency: float, source_time=None):
    source = ModeSource(
        position=port.position,
        size=port.size,
        mode_index=compiled_mode.mode_index,
        direction=port.direction,
        polarization=f"E{compiled_mode.polarization_axis}",
        source_time=source_time or CW(frequency=frequency),
        name=f"{compiled_mode.tracking_id}::basis",
    )
    object.__setattr__(source, "_wave_family", compiled_mode.family)
    return source


def _tracking_basis(mode_data) -> torch.Tensor:
    values = torch.cat(
        [
            mode_data["component_profiles"][name]
            .to(dtype=torch.complex128)
            .reshape(-1)
            for name in _COMPONENTS[:3]
        ]
    )
    norm = torch.linalg.vector_norm(values)
    if not bool(norm > torch.finfo(norm.dtype).tiny):
        raise RuntimeError("WavePort mode solve produced a zero electric modal basis.")
    return values / norm


def _solve_port_modes(context, port: WavePort, compiled, frequencies):
    solved_by_frequency = []
    beta_rows = []
    basis_rows = []
    for frequency in frequencies:
        solved_modes = []
        beta_values = []
        basis_values = []
        for compiled_mode in compiled.modes:
            public_source = _mode_source(
                port,
                compiled_mode,
                frequency=float(frequency),
            )
            source = _compile_mode_source(public_source, default_frequency=float(frequency))
            mode_data = solve_mode_source_profile(context, source)
            solved_modes.append(mode_data)
            beta_value = mode_data.get("beta_tensor")
            if beta_value is None:
                beta_value = torch.as_tensor(
                    mode_data["beta"],
                    device=context.scene.device,
                    dtype=torch.float64,
                )
            beta_values.append(torch.real(beta_value).to(dtype=torch.float64))
            basis_values.append(_tracking_basis(mode_data))
        solved_by_frequency.append(tuple(solved_modes))
        beta_rows.append(torch.stack(beta_values))
        basis_rows.append(torch.stack(basis_values))
    beta = torch.stack(beta_rows)
    modal_basis = torch.stack(basis_rows)
    tracking = track_modes(beta, modal_basis=modal_basis)
    return tuple(solved_by_frequency), tracking


def _field_component_points(scene, field_name: str, indices: torch.Tensor) -> torch.Tensor:
    if field_name == "Ex":
        coordinates = (scene.x_half, scene.y, scene.z)
    elif field_name == "Ey":
        coordinates = (scene.x, scene.y_half, scene.z)
    elif field_name == "Ez":
        coordinates = (scene.x, scene.y, scene.z_half)
    elif field_name == "Hx":
        coordinates = (scene.x, scene.y_half, scene.z_half)
    elif field_name == "Hy":
        coordinates = (scene.x_half, scene.y, scene.z_half)
    elif field_name == "Hz":
        coordinates = (scene.x_half, scene.y_half, scene.z)
    else:
        raise ValueError(f"Unsupported wave-port field component {field_name!r}.")
    return torch.stack(
        [coordinates[axis].index_select(0, indices[:, axis]) for axis in range(3)],
        dim=-1,
    )


def _integrate_mode_geometry(scene, mode_data, compiled_mode):
    voltage = None
    if compiled_mode.voltage_component is not None:
        points = _field_component_points(
            scene,
            compiled_mode.voltage_component,
            compiled_mode.voltage_indices,
        )
        values = sample_mode_source_component(
            mode_data,
            points,
            compiled_mode.voltage_component,
        )
        voltage = torch.sum(values * compiled_mode.voltage_weights)

    current = None
    if compiled_mode.current_components:
        parts = []
        for component, indices, weights in zip(
            compiled_mode.current_components,
            compiled_mode.current_indices,
            compiled_mode.current_weights,
        ):
            points = _field_component_points(scene, component, indices)
            values = sample_mode_source_component(mode_data, points, component)
            parts.append(torch.sum(values * weights))
        current = torch.stack(parts).sum()
    if (
        mode_data.get("mode_solver_kind") == "tem_electrostatic_torch"
        and voltage is not None
        and current is not None
    ):
        if not bool(torch.abs(current) > torch.finfo(current.dtype).eps):
            raise RuntimeError("TEM WavePort current contour has zero circulation.")
        current = 2.0 / torch.conj(voltage)
    return voltage, current


def _uniform_relative_medium(context, compiled):
    model = context._compiled_material_model
    indices = compiled.aperture_indices.reshape(-1, 3)
    selector = tuple(indices[:, axis] for axis in range(3))
    pec_occupancy = model["pec_occupancy"]
    mask = (
        torch.ones(indices.shape[0], device=indices.device, dtype=torch.bool)
        if pec_occupancy is None
        else pec_occupancy[selector] < 0.5
    )
    if not bool(torch.any(mask)):
        raise ValueError(f"WavePort {compiled.port_name!r} aperture contains no dielectric nodes.")
    eps = model["eps_r"][selector][mask]
    mu = model["mu_r"][selector][mask]
    for name, value in (("eps_r", eps), ("mu_r", mu)):
        if value.is_complex() and bool(torch.any(torch.imag(value) != 0.0)):
            raise NotImplementedError(
                f"WavePort {compiled.port_name!r} {name} must be real for TE/TM impedance."
            )
        real = torch.real(value)
        if not torch.allclose(real, real[0].expand_as(real), rtol=1.0e-4, atol=1.0e-6):
            raise ValueError(
                f"WavePort {compiled.port_name!r} requires a uniformly filled aperture "
                f"for TE/TM impedance, but {name} varies across the cross-section."
            )
    return torch.real(eps[0]).to(torch.float64), torch.real(mu[0]).to(torch.float64)


def _characteristic_impedance(
    context,
    port: WavePort,
    compiled,
    mode_data,
    tracking: ModeTrackingResult,
    frequencies,
) -> torch.Tensor:
    frequency_count = len(frequencies)
    mode_count = len(compiled.modes)
    output = torch.empty(
        (mode_count, frequency_count),
        device=context.scene.device,
        dtype=torch.complex128,
    )
    eps_r = mu_r = None
    if any(mode.family in {"te", "tm"} for mode in port.modes):
        eps_r, mu_r = _uniform_relative_medium(context, compiled)

    for frequency_index, frequency in enumerate(frequencies):
        orientation = tracking.orientation[frequency_index]
        for stable_index, (mode, compiled_mode) in enumerate(
            zip(port.modes, compiled.modes)
        ):
            beta = tracking.beta[frequency_index, stable_index].to(torch.complex128)
            omega = torch.as_tensor(
                2.0 * math.pi * float(frequency),
                device=beta.device,
                dtype=torch.float64,
            )
            if mode.family == "te":
                impedance = omega * _MU0 * mu_r / beta
            elif mode.family == "tm":
                impedance = beta / (omega * _EPS0 * eps_r)
            else:
                voltage = torch.zeros((), device=beta.device, dtype=torch.complex128)
                current = torch.zeros_like(voltage)
                for raw_index, raw_mode_data in enumerate(mode_data[frequency_index]):
                    raw_voltage, raw_current = _integrate_mode_geometry(
                        context.scene,
                        raw_mode_data,
                        compiled_mode,
                    )
                    coefficient = orientation[stable_index, raw_index]
                    if raw_voltage is not None:
                        voltage = voltage + coefficient * raw_voltage.to(torch.complex128)
                    if raw_current is not None:
                        current = current + coefficient * raw_current.to(torch.complex128)
                if mode.impedance_definition == "voltage_current":
                    impedance = voltage / current
                elif mode.impedance_definition == "power_voltage":
                    impedance = torch.abs(voltage).square() / 2.0
                else:
                    impedance = 2.0 / torch.abs(current).square()
            if not bool(torch.isfinite(torch.real(impedance))) or not bool(
                torch.isfinite(torch.imag(impedance))
            ):
                raise RuntimeError(
                    f"WavePort {port.name!r} mode {mode.name!r} produced a non-finite impedance."
                )
            if not bool(torch.real(impedance) > 0.0):
                raise RuntimeError(
                    f"WavePort {port.name!r} mode {mode.name!r} characteristic impedance "
                    "does not have a positive real part; check V/I orientation."
                )
            output[stable_index, frequency_index] = impedance
    return output


def _tracking_confidence(tracking: ModeTrackingResult) -> torch.Tensor:
    frequency_count, mode_count = tracking.assignment.shape
    first = torch.ones(
        (1, mode_count),
        device=tracking.beta.device,
        dtype=tracking.beta.dtype,
    )
    return torch.cat((first, tracking.confidence), dim=0).transpose(0, 1)


def _material_cross_section_probe(model, compiled) -> torch.Tensor | None:
    aperture_indices = compiled.aperture_indices.reshape(-1, 3)
    normal_index = "xyz".index(compiled.normal_axis)
    source_indices = aperture_indices.clone()
    source_indices[:, normal_index] = (
        compiled.electric_plane_index - compiled.direction_sign
    )
    selectors = tuple(
        tuple(indices[:, axis] for axis in range(3))
        for indices in (aperture_indices, source_indices)
    )
    tensors = []
    for key in (
        "eps_components",
        "mu_components",
        "sigma_e_components",
        "sigma_m_components",
        "eps_offdiag_components",
        "mu_offdiag_components",
    ):
        values = model.get(key, {})
        if isinstance(values, dict):
            tensors.extend(value for value in values.values() if torch.is_tensor(value))
    for key in ("eps_r", "mu_r", "pec_occupancy"):
        value = model.get(key)
        if torch.is_tensor(value):
            tensors.append(value)

    probe = None
    for tensor_index, tensor in enumerate(tensors):
        if not tensor.requires_grad:
            continue
        sampled = torch.cat(
            tuple(tensor[selector].reshape(-1) for selector in selectors)
        )
        weights = torch.linspace(
            1.0 + 0.03125 * tensor_index,
            2.0 + 0.03125 * tensor_index,
            sampled.numel(),
            device=sampled.device,
            dtype=sampled.real.dtype,
        )
        term = torch.sum(sampled.real * weights)
        if sampled.is_complex():
            term = term + torch.sum(sampled.imag * torch.flip(weights, dims=(0,)))
        probe = term if probe is None else probe + term
    return probe


def _validate_fixed_trainable_cross_sections(scene, context, compiled_by_name, selected_names):
    inputs = scene_trainable_material_tensors(scene)
    if not inputs:
        raise NotImplementedError(
            "Differentiable WavePort runs require trainable inputs that contribute to "
            "the prepared material tensors."
        )
    probes = tuple(
        probe
        for name in selected_names
        if (probe := _material_cross_section_probe(
            context._compiled_material_model,
            compiled_by_name[name],
        ))
        is not None
    )
    if not probes:
        return
    dependencies = torch.autograd.grad(
        sum(probes),
        inputs,
        allow_unused=True,
        retain_graph=True,
    )
    if any(
        dependency is not None and bool(torch.any(dependency.detach() != 0))
        for dependency in dependencies
    ):
        raise NotImplementedError(
            "Differentiable WavePort runs require fixed port cross-sections and launch "
            "planes; a trainable material or geometry input affects a selected WavePort "
            "aperture or its adjacent source plane. Move the design region away from the "
            "port launch."
        )


def _detach_material_model(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_material_model(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_material_model(item) for item in value)
    if isinstance(value, list):
        return [_detach_material_model(item) for item in value]
    return value


def resolve_waveport_run_manifest(scene, sweep: PortSweep, frequencies) -> WavePortRunManifest:
    if not isinstance(sweep, PortSweep):
        raise TypeError("sweep must be a PortSweep.")
    if sweep.source_time is not None:
        raise ValueError(
            "WavePort PortSweep uses a calibrated per-frequency CW basis; source_time must be None."
        )
    if sweep.amplitude.requires_grad:
        raise NotImplementedError("WavePort PortSweep does not yet support trainable amplitude.")
    frequency_values = tuple(float(value) for value in frequencies)
    if not frequency_values:
        raise ValueError("WavePort PortSweep requires at least one frequency.")
    if any(not math.isfinite(value) or value <= 0.0 for value in frequency_values):
        raise ValueError("WavePort PortSweep frequencies must be finite and positive.")
    if any(
        right <= left for left, right in zip(frequency_values, frequency_values[1:])
    ):
        raise ValueError("WavePort PortSweep frequencies must be strictly increasing.")
    wave_ports = tuple(port for port in scene.ports if isinstance(port, WavePort))
    if not wave_ports:
        raise ValueError("WavePort PortSweep requires at least one WavePort.")
    non_wave_rf_ports = tuple(
        port
        for port in scene.ports
        if hasattr(port, "reference_impedance") and hasattr(port, "termination")
    )
    unterminated_non_wave_ports = tuple(
        port.name for port in non_wave_rf_ports if port.termination is None
    )
    if unterminated_non_wave_ports:
        raise NotImplementedError(
            "A WavePort run can retain lumped/terminal ports only as explicit passive "
            f"terminations; unterminated entries are {unterminated_non_wave_ports}."
        )
    ports_by_name = {port.name: port for port in wave_ports}
    selected_names = tuple(ports_by_name) if sweep.ports is None else sweep.ports
    missing = tuple(name for name in selected_names if name not in ports_by_name)
    if missing:
        raise ValueError(f"PortSweep references missing WavePort entries: {missing}.")
    if any(
        getattr(port, "source_time", None) is not None
        for port in scene.ports
        if not isinstance(port, WavePort)
    ) or scene.sources:
        raise ValueError("WavePort PortSweep requires a Scene without independent field sources.")

    context = _mode_context(scene)
    compiled_by_name = {entry.port_name: entry for entry in scene.compile_waveports()}
    if scene_trainable_material_tensors(scene):
        _validate_fixed_trainable_cross_sections(
            scene,
            context,
            compiled_by_name,
            selected_names,
        )
        # The selected aperture is proven independent of the design inputs above.
        # Detaching the mode context keeps the fixed mode solve on its full-vector
        # forward path instead of requesting an unnecessary mode-shape eigen-adjoint.
        context._compiled_material_model = _detach_material_model(
            context._compiled_material_model
        )
    prepared_ports = []
    channel_names = []
    for port_name in selected_names:
        port = ports_by_name[port_name]
        compiled = compiled_by_name[port_name]
        mode_data, tracking = _solve_port_modes(
            context,
            port,
            compiled,
            frequency_values,
        )
        impedance = _characteristic_impedance(
            context,
            port,
            compiled,
            mode_data,
            tracking,
            frequency_values,
        )
        prepared_ports.append(
            PreparedWavePort(
                port=port,
                compiled=compiled,
                mode_data=mode_data,
                tracking=tracking,
                characteristic_impedance=impedance,
                tracking_confidence=_tracking_confidence(tracking),
            )
        )
        channel_names.extend(port.mode_name(mode) for mode in port.modes)
    return WavePortRunManifest(
        physical_port_names=selected_names,
        channel_names=tuple(channel_names),
        frequencies=frequency_values,
        prepared_ports=tuple(prepared_ports),
    )


def _source_plane(scene, prepared_port: PreparedWavePort) -> tuple[float, float, float]:
    port = prepared_port.port
    axis_index = "xyz".index(port.normal_axis)
    grid_scene = scene if hasattr(scene, port.normal_axis) else prepare_scene(scene)
    nodes = getattr(grid_scene, port.normal_axis)
    plane_index = prepared_port.compiled.electric_plane_index
    source_index = plane_index - port.direction_sign
    if source_index <= 0 or source_index >= nodes.numel() - 1:
        raise ValueError(
            f"WavePort {port.name!r} needs one grid cell behind its reference plane for injection."
        )
    position = list(port.position)
    position[axis_index] = float(nodes[source_index].detach().cpu())
    return tuple(position)


def _coefficient_source_time(frequency: float, coefficient: complex) -> CW:
    return CW(
        frequency=float(frequency),
        amplitude=abs(coefficient),
        phase=cmath.phase(coefficient),
    )


def _monitor_name(port_name: str, raw_index: int) -> str:
    return f"{WAVEPORT_MONITOR_PREFIX}{port_name}::{raw_index}"


def _user_monitor_payloads(result, scene) -> dict:
    """Return the user-declared monitor payloads captured on a column run.

    The internal per-port ModeMonitors (``WAVEPORT_MONITOR_PREFIX``) drive the
    S-matrix extraction and are machinery, not user-facing. Every other monitor
    payload was requested by the user on the original scene and must survive into
    the assembled Result instead of being silently dropped.
    """
    user_names = {monitor.name for monitor in scene.monitors}
    return {
        name: payload
        for name, payload in result.monitors.items()
        if name in user_names and not name.startswith(WAVEPORT_MONITOR_PREFIX)
    }


def _column_scene(
    scene,
    manifest: WavePortRunManifest,
    *,
    active_port_index: int,
    active_mode_index: int,
    frequency_index: int,
    amplitude: complex,
):
    frequency = manifest.frequencies[frequency_index]
    sources = []
    monitors = []
    for port_index, prepared_port in enumerate(manifest.prepared_ports):
        port = prepared_port.port
        for raw_index, compiled_mode in enumerate(prepared_port.compiled.modes):
            monitor = ModeMonitor(
                    _monitor_name(port.name, raw_index),
                    position=port.position,
                    size=port.size,
                    mode_index=compiled_mode.mode_index,
                    direction=port.direction,
                    polarization=f"E{compiled_mode.polarization_axis}",
                    frequencies=(frequency,),
                    normal_direction=port.direction,
                )
            object.__setattr__(monitor, "_wave_family", compiled_mode.family)
            monitors.append(monitor)
        if port_index != active_port_index:
            continue
        source_position = _source_plane(scene, prepared_port)
        coefficients = prepared_port.tracking.orientation[
            frequency_index,
            active_mode_index,
        ]
        for raw_index, (compiled_mode, coefficient) in enumerate(
            zip(prepared_port.compiled.modes, coefficients)
        ):
            resolved_coefficient = amplitude * complex(coefficient.detach().cpu())
            if abs(resolved_coefficient) <= 1.0e-14:
                continue
            source = ModeSource(
                    position=source_position,
                    size=port.size,
                    mode_index=compiled_mode.mode_index,
                    direction=port.direction,
                    polarization=f"E{compiled_mode.polarization_axis}",
                    source_time=_coefficient_source_time(frequency, resolved_coefficient),
                    name=(
                        f"__waveport_source__::{port.name}::{active_mode_index}::{raw_index}"
                    ),
                )
            object.__setattr__(source, "_wave_family", compiled_mode.family)
            sources.append(source)
    passive_ports = tuple(port for port in scene.ports if not isinstance(port, WavePort))
    return scene.clone(
        ports=passive_ports,
        sources=sources,
        monitors=[*scene.monitors, *monitors],
    )


def _extract_tracked_waves(result, manifest: WavePortRunManifest, frequency_index: int):
    incident = []
    reflected = []
    for prepared_port in manifest.prepared_ports:
        raw_a = []
        raw_b = []
        for raw_index in range(len(prepared_port.compiled.modes)):
            modal = result.monitor(
                _monitor_name(prepared_port.port.name, raw_index),
                resolve_modal=True,
            )
            scale = torch.sqrt(torch.abs(modal["mode_power"]))
            raw_a.append(modal["amplitude_forward"] * scale)
            raw_b.append(modal["amplitude_backward"] * scale)
        raw_a = torch.stack(raw_a).to(torch.complex128)
        raw_b = torch.stack(raw_b).to(torch.complex128)
        orientation = prepared_port.tracking.orientation[frequency_index]
        incident.append(torch.conj(orientation) @ raw_a)
        reflected.append(torch.conj(orientation) @ raw_b)
    return torch.cat(incident), torch.cat(reflected)


def _execute_columns(
    simulation,
    scene,
    manifest: WavePortRunManifest,
    channel_locations,
    *,
    amplitude: complex,
):
    incident_columns = []
    reflected_columns = []
    column_stats = []
    column_results = []
    last_result = None
    first_result = None
    shared_prepared_scene = None
    from .array_execution import compact_array_column_result

    for active_port_index, active_mode_index in channel_locations:
        incident_frequencies = []
        reflected_frequencies = []
        stats = []
        results = []
        for frequency_index, frequency in enumerate(manifest.frequencies):
            run_scene = _column_scene(
                scene,
                manifest,
                active_port_index=active_port_index,
                active_mode_index=active_mode_index,
                frequency_index=frequency_index,
                amplitude=amplitude,
            )
            sub_simulation = type(simulation)(
                scene=run_scene,
                method=simulation.method,
                frequencies=(frequency,),
                config=simulation.config,
                excitations=None,
                metadata=simulation.metadata,
            )
            # The outer workflow has already proved that trainable material inputs
            # do not affect this fixed WavePort aperture. Reuse the initialized
            # full-vector source patches during the adjoint instead of requesting
            # an unrelated mode-shape eigen-adjoint from each inner simulation.
            sub_simulation._fixed_waveport_mode_sources = True
            last_result = sub_simulation.run()
            if first_result is None:
                first_result = last_result
            if shared_prepared_scene is None:
                shared_prepared_scene = last_result.prepared_scene
            results.append(
                compact_array_column_result(
                    last_result,
                    prepared_scene=shared_prepared_scene,
                )
            )
            incident, reflected = _extract_tracked_waves(
                last_result,
                manifest,
                frequency_index,
            )
            incident_frequencies.append(incident)
            reflected_frequencies.append(reflected)
            stats.append(last_result.stats())
        incident_columns.append(torch.stack(incident_frequencies))
        reflected_columns.append(torch.stack(reflected_frequencies))
        column_stats.append(tuple(stats))
        column_results.append(tuple(results))
    return (
        torch.stack(incident_columns),
        torch.stack(reflected_columns),
        tuple(column_stats),
        tuple(column_results),
        last_result,
        first_result,
    )


def run_waveport_sweep(simulation, scene, manifest: WavePortRunManifest) -> Result:
    channel_locations = []
    for port_index, prepared_port in enumerate(manifest.prepared_ports):
        channel_locations.extend(
            (port_index, mode_index)
            for mode_index in range(len(prepared_port.port.modes))
        )
    (
        incident,
        reflected,
        column_stats,
        column_results,
        last_result,
        first_result,
    ) = _execute_columns(
        simulation,
        scene,
        manifest,
        channel_locations,
        amplitude=complex(simulation.port_sweep.amplitude.detach().cpu()),
    )

    channel_count = len(manifest.channel_names)
    # Network S by solving the full system B = S * A across the drive columns
    # (F3). ``incident``/``reflected`` are indexed ``[drive, frequency, channel]``.
    # The per-drive ``b/a`` ratio is only correct when the passive channels carry
    # no incident wave (A diagonal); on a re-illuminated bench A has off-diagonal
    # incident amplitudes and the ratio is generically wrong. Assembling
    #   A[f, i, j] = incident[j, f, i]      (incident at channel i under drive j)
    #   B[f, i, j] = reflected[j, f, i]
    # and solving S[f] = B[f] @ A[f]^{-1} recovers the physical scattering matrix.
    # When A is diagonal this reduces to the old b/a extraction bit-for-bit.
    incident_matrix = incident.permute(1, 2, 0).contiguous()   # [freq, i, j]
    reflected_matrix = reflected.permute(1, 2, 0).contiguous()  # [freq, i, j]
    self_incident = torch.diagonal(incident_matrix, dim1=-2, dim2=-1)  # [freq, channel]
    threshold = torch.clamp(
        torch.max(torch.abs(self_incident)) * 1.0e-6,
        min=torch.finfo(self_incident.real.dtype).tiny,
    )
    if bool(torch.any(torch.abs(self_incident) < threshold)):
        raise RuntimeError(
            "WavePort drive has insufficient self-incident spectrum for the B = S*A extraction."
        )
    condition_numbers = torch.linalg.cond(incident_matrix)  # [freq]
    scattering = torch.linalg.solve(
        incident_matrix.transpose(-2, -1),
        reflected_matrix.transpose(-2, -1),
    ).transpose(-2, -1)  # S = B A^{-1} solved as S^T = (A^T)^{-1} B^T

    frequency_tensor = torch.as_tensor(
        manifest.frequencies,
        device=scattering.device,
        dtype=torch.float64,
    )
    z0_columns = []
    beta_columns = []
    ports = {}
    channel_offset = 0
    for prepared_port in manifest.prepared_ports:
        mode_count = len(prepared_port.port.modes)
        indices = slice(channel_offset, channel_offset + mode_count)
        port_a = incident[:, :, indices].movedim(-1, -2)
        port_b = reflected[:, :, indices].movedim(-1, -2)
        impedance = prepared_port.characteristic_impedance
        ports[prepared_port.port.name] = PortData.from_power_waves(
            port_name=prepared_port.port.name,
            frequencies=frequency_tensor,
            a=port_a,
            b=port_b,
            z0=impedance.unsqueeze(0),
            direction=prepared_port.port.direction,
            reference_plane=prepared_port.port.reference_plane,
            mode_names=tuple(mode.name for mode in prepared_port.port.modes),
            beta=prepared_port.tracking.beta.transpose(0, 1),
            characteristic_impedance=impedance,
            tracking_confidence=prepared_port.tracking_confidence,
            metadata={
                "channel_names": manifest.channel_names[indices],
                "normalization": "tracked one-watt modal power waves",
            },
        )
        z0_columns.extend(impedance[mode_index] for mode_index in range(mode_count))
        tracked_beta = prepared_port.tracking.beta.transpose(0, 1)
        beta_columns.extend(tracked_beta[mode_index] for mode_index in range(mode_count))
        channel_offset += mode_count
    network_z0 = torch.stack(z0_columns, dim=-1)
    network_beta = torch.stack(beta_columns, dim=-1)
    network = NetworkData(
        frequencies=frequency_tensor,
        s=scattering,
        z0=network_z0,
        port_names=manifest.channel_names,
        valid_columns=torch.ones(channel_count, device=scattering.device, dtype=torch.bool),
        metadata={
            "run_manifest": manifest.metadata(),
            "propagation_constants": network_beta,
            # Conditioning of the incident matrix A per frequency (F3). A large
            # cond(A) means the drive columns are near-collinear (a re-entrant /
            # under-driven bench) and the extracted S is untrustworthy; this is the
            # extraction-conditioning precondition that replaces the a_passive gate.
            "extraction_condition_number": condition_numbers,
        },
    )
    from .array_execution import ArrayRunData

    basis_incident = torch.stack(
        [incident[index, :, index] for index in range(channel_count)],
        dim=-1,
    )
    # User-declared monitors ride through the sweep. A PortSweep drives one channel
    # per column, so a field monitor genuinely has one payload per drive; the flat
    # top-level Result.monitors carries the FIRST drive channel at the first swept
    # frequency (recorded in metadata), while per-drive / per-frequency field data
    # is preserved column-by-column in ``array_run_data.column_results``.
    user_monitors = _user_monitor_payloads(first_result, scene)
    monitor_drive_note = {}
    if user_monitors:
        monitor_drive_note = {
            "user_monitor_drive_channel": manifest.channel_names[0],
            "user_monitor_frequency": manifest.frequencies[0],
        }
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
        array_run_data=ArrayRunData(
            manifest=manifest,
            column_results=column_results,
            incident=basis_incident,
        ),
        metadata={
            **simulation.metadata,
            "network_run_manifest": manifest.metadata(),
            **monitor_drive_note,
        },
        solver_stats={
            "network_sweep": manifest.metadata(),
            "columns": column_stats,
        },
        raw_output={"network_run_manifest": manifest.metadata()},
    )


def run_waveport_excitation(simulation, scene, excitation, manifest: WavePortRunManifest):
    if len(manifest.prepared_ports) != 1:
        raise RuntimeError("A direct WavePort excitation must resolve exactly one port.")
    prepared_port = manifest.prepared_ports[0]
    local_names = tuple(mode.name for mode in prepared_port.port.modes)
    qualified_names = tuple(
        prepared_port.port.mode_name(mode) for mode in prepared_port.port.modes
    )
    requested_name = excitation.mode_name
    if requested_name is None:
        active_mode_index = 0
    elif requested_name in local_names:
        active_mode_index = local_names.index(requested_name)
    else:
        active_mode_index = qualified_names.index(requested_name)

    incident, reflected, column_stats, _, last_result, _first_result = _execute_columns(
        simulation,
        scene,
        manifest,
        ((0, active_mode_index),),
        amplitude=complex(excitation.amplitude.detach().cpu()),
    )
    frequency_tensor = torch.as_tensor(
        manifest.frequencies,
        device=incident.device,
        dtype=torch.float64,
    )
    port_data = PortData.from_power_waves(
        port_name=prepared_port.port.name,
        frequencies=frequency_tensor,
        a=incident[0].movedim(-1, -2),
        b=reflected[0].movedim(-1, -2),
        z0=prepared_port.characteristic_impedance,
        direction=prepared_port.port.direction,
        reference_plane=prepared_port.port.reference_plane,
        mode_names=local_names,
        beta=prepared_port.tracking.beta.transpose(0, 1),
        characteristic_impedance=prepared_port.characteristic_impedance,
        tracking_confidence=prepared_port.tracking_confidence,
        metadata={
            "active_mode": local_names[active_mode_index],
            "active_channel": qualified_names[active_mode_index],
            "normalization": "tracked one-watt modal power waves",
        },
    )
    run_metadata = {
        **manifest.metadata(),
        "active_channel": qualified_names[active_mode_index],
        "kind": "direct_waveport_excitation",
    }
    # A direct WavePort excitation is a single drive column, so user-declared
    # monitors map unambiguously to this excitation and ride through unchanged --
    # identical to a plain FDTD run of the injected mode source.
    return Result(
        method="fdtd",
        scene=scene,
        prepared_scene=last_result.prepared_scene,
        frequency=manifest.frequencies[0],
        frequencies=manifest.frequencies,
        solver=last_result.solver,
        fields={},
        monitors=_user_monitor_payloads(last_result, scene),
        ports={prepared_port.port.name: port_data},
        network=None,
        metadata={**simulation.metadata, "waveport_run_manifest": run_metadata},
        solver_stats={"waveport_excitation": run_metadata, "columns": column_stats},
        raw_output={"waveport_run_manifest": run_metadata},
    )


__all__ = [
    "PreparedWavePort",
    "WavePortRunManifest",
    "resolve_waveport_run_manifest",
    "run_waveport_excitation",
    "run_waveport_sweep",
]

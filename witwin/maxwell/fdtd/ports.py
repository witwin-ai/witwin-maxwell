from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..lumped import PortExcitation
from ..network import PortData
from ..sources import CW, CustomSourceTime, GaussianPulse, RickerWavelet
from .checkpoint import lumped_state_name
from .lumped import (
    LumpedRuntime,
    apply_lumped_runtime,
    prepare_lumped_runtime,
    pullback_lumped_runtime,
    replay_lumped_runtime,
)


PhasorNormalization = Literal["peak", "rms", "none"]


class PortDFTAccumulator:
    """Accumulate staggered scalar port samples with the ``exp(+iwt)`` convention."""

    def __init__(self, frequencies: torch.Tensor) -> None:
        if not isinstance(frequencies, torch.Tensor):
            raise TypeError("frequencies must be a torch.Tensor.")
        if frequencies.ndim != 1 or frequencies.shape[0] == 0:
            raise ValueError("frequencies must have shape [F] with F > 0.")
        if frequencies.dtype not in (torch.float32, torch.float64):
            raise TypeError("frequencies must use torch.float32 or torch.float64.")

        self.frequencies = frequencies
        self._real_dtype = frequencies.dtype
        zeros = torch.zeros_like(frequencies)
        self._voltage_sum = torch.complex(zeros, zeros)
        self._current_sum = torch.complex(zeros, zeros)
        self._window_weight_sum = torch.zeros_like(frequencies)
        self._one = torch.ones((), dtype=self._real_dtype, device=frequencies.device)
        self._voltage_term = torch.empty_like(self._voltage_sum)
        self._current_term = torch.empty_like(self._current_sum)
        self._sample_count = 0

    def _scalar(
        self,
        value: torch.Tensor | float | complex,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar tensor.")
            if value.device != self.frequencies.device:
                raise ValueError(f"{name} must be on the same device as frequencies.")
            return value.to(dtype=dtype)
        return torch.as_tensor(
            value,
            dtype=dtype,
            device=self.frequencies.device,
        )

    def _phase(self, sample_time: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * torch.pi * self.frequencies * sample_time
        return torch.complex(torch.cos(angle), torch.sin(angle))

    def _weight(self, value: torch.Tensor | float | None) -> torch.Tensor:
        if value is None:
            return self._one
        if isinstance(value, torch.Tensor):
            if value.device != self.frequencies.device:
                raise ValueError("window_weight must be on the same device as frequencies.")
            if value.ndim == 0 or tuple(value.shape) == tuple(self.frequencies.shape):
                return value.to(dtype=self._real_dtype)
            raise ValueError("window_weight must be scalar or have shape [F].")
        return torch.as_tensor(value, dtype=self._real_dtype, device=self.frequencies.device)

    def accumulate(
        self,
        voltage_sample: torch.Tensor | float | complex,
        current_sample: torch.Tensor | float | complex,
        *,
        electric_sample_time: torch.Tensor | float,
        magnetic_sample_time: torch.Tensor | float,
        window_weight: torch.Tensor | float | None = None,
    ) -> None:
        """Accumulate one Yee step using each field's physical sample time."""

        voltage = self._scalar(
            voltage_sample,
            name="voltage_sample",
            dtype=self._voltage_sum.dtype,
        )
        current = self._scalar(
            current_sample,
            name="current_sample",
            dtype=self._current_sum.dtype,
        )
        electric_time = self._scalar(
            electric_sample_time,
            name="electric_sample_time",
            dtype=self._real_dtype,
        )
        magnetic_time = self._scalar(
            magnetic_sample_time,
            name="magnetic_sample_time",
            dtype=self._real_dtype,
        )
        weight = self._weight(window_weight)

        self._voltage_sum = self._voltage_sum + weight * voltage * self._phase(electric_time)
        self._current_sum = self._current_sum + weight * current * self._phase(magnetic_time)
        self._window_weight_sum = self._window_weight_sum + weight
        self._sample_count += 1

    def accumulate_precomputed(
        self,
        voltage_sample: torch.Tensor,
        current_sample: torch.Tensor | None,
        *,
        voltage_kernel: torch.Tensor,
        current_kernel: torch.Tensor,
        window_weight: torch.Tensor,
    ) -> None:
        """Accumulate one prepared FDTD step without allocating device tensors."""

        torch.mul(voltage_kernel, voltage_sample, out=self._voltage_term)
        self._voltage_sum.add_(self._voltage_term)
        if current_sample is not None:
            torch.mul(current_kernel, current_sample, out=self._current_term)
            self._current_sum.add_(self._current_term)
        self._window_weight_sum.add_(window_weight)
        self._sample_count += 1

    def phasors(
        self,
        *,
        normalization: PhasorNormalization = "peak",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return voltage and current phasors, each with explicit shape ``[F]``."""

        if normalization == "none":
            return self._voltage_sum, self._current_sum
        if normalization not in ("peak", "rms"):
            raise ValueError("normalization must be 'peak', 'rms', or 'none'.")
        if self._sample_count == 0:
            raise RuntimeError("Cannot normalize an accumulator with no samples.")

        amplitude_scale = torch.as_tensor(
            2.0,
            dtype=self._real_dtype,
            device=self.frequencies.device,
        )
        if normalization == "rms":
            amplitude_scale = torch.sqrt(amplitude_scale)
        scale = amplitude_scale / self._window_weight_sum
        return self._voltage_sum * scale, self._current_sum * scale


@dataclass
class PreparedPortRuntime:
    """Single-device FDTD state shared by a port source, load, and observer."""

    port: object
    geometry: object
    frequencies: torch.Tensor
    field_name: str
    lumped: LumpedRuntime | None
    excitation: PortExcitation | None
    source_kind: str | None
    source_frequency: float
    source_fwidth: float
    source_phase: float
    source_delay: float
    source_amplitude: torch.Tensor
    drive_buffer: torch.Tensor
    electric_time: torch.Tensor
    magnetic_time: torch.Tensor
    window_weights: torch.Tensor | None = None
    window_type: str | None = None
    accumulator: PortDFTAccumulator | None = None
    drive_accumulator: PortDFTAccumulator | None = None
    voltage_phase_weights: torch.Tensor | None = None
    current_phase_weights: torch.Tensor | None = None
    observer_current_buffer: torch.Tensor | None = None
    observer_window_buffer: torch.Tensor | None = None
    observer_voltage_kernel_buffer: torch.Tensor | None = None
    observer_current_kernel_buffer: torch.Tensor | None = None
    sample_index: int = 0
    embedded_network_name: str | None = None


def _field_map(solver) -> dict[str, torch.Tensor]:
    return {
        "Ex": solver.Ex,
        "Ey": solver.Ey,
        "Ez": solver.Ez,
        "Hx": solver.Hx,
        "Hy": solver.Hy,
        "Hz": solver.Hz,
    }


def _edge_control_volume(solver, component: str) -> torch.Tensor:
    scene = solver.scene
    dtype = getattr(solver, component).dtype
    device = solver.device
    widths = {
        "Ex": (scene.dx_primal64, scene.dy_dual64, scene.dz_dual64),
        "Ey": (scene.dx_dual64, scene.dy_primal64, scene.dz_dual64),
        "Ez": (scene.dx_dual64, scene.dy_dual64, scene.dz_primal64),
    }[component]
    x = torch.as_tensor(widths[0], device=device, dtype=dtype)
    y = torch.as_tensor(widths[1], device=device, dtype=dtype)
    z = torch.as_tensor(widths[2], device=device, dtype=dtype)
    return x[:, None, None] * y[None, :, None] * z[None, None, :]


def _real_source_resistance(port, excitation: PortExcitation, *, device, dtype) -> torch.Tensor:
    impedance = (
        port.reference_impedance
        if excitation.source_impedance == "matched"
        else excitation.source_impedance
    )
    value = torch.as_tensor(impedance, device=device)
    if value.ndim != 0:
        raise ValueError(f"Port {port.name!r} source impedance must be scalar.")
    if value.is_complex():
        if not bool(torch.isclose(value.imag, torch.zeros_like(value.imag))):
            raise ValueError(
                f"Port {port.name!r} time-domain source impedance must be real."
            )
        value = value.real
    value = value.to(dtype=dtype)
    if not bool(torch.isfinite(value)) or not bool(value > 0.0):
        raise ValueError(f"Port {port.name!r} source impedance must be positive and finite.")
    return value


def _require_forward_only_parameters(
    port,
    excitation: PortExcitation | None,
    termination=None,
) -> None:
    if (
        isinstance(port.reference_impedance, torch.Tensor)
        and port.reference_impedance.requires_grad
    ):
        raise NotImplementedError(
            f"Port {port.name!r} trainable reference_impedance is not supported by "
            "the FDTD lumped adjoint."
        )
    if excitation is not None:
        source_impedance = excitation.source_impedance
        if isinstance(source_impedance, torch.Tensor) and source_impedance.requires_grad:
            raise NotImplementedError(
                f"Port {port.name!r} trainable source_impedance is not supported by "
                "the FDTD lumped adjoint; train amplitude or series R/L/C values instead."
            )
    termination = getattr(port, "termination", None) if termination is None else termination
    if getattr(termination, "kind", None) == "parallel_rlc" and any(
        isinstance(getattr(termination, name, None), torch.Tensor)
        and getattr(termination, name).requires_grad
        for name in ("r", "l", "c")
    ):
        raise NotImplementedError(
            f"Port {port.name!r} trainable ParallelRLC is not supported by the FDTD "
            "lumped adjoint; use a SeriesRLC termination."
        )


def _validate_supported_field_coupling(solver) -> None:
    if getattr(solver, "complex_fields_enabled", False):
        raise NotImplementedError("Lumped FDTD coupling currently requires real Yee fields.")
    if getattr(solver, "conductive_enabled", False):
        raise NotImplementedError(
            "Lumped FDTD coupling in conductive media requires a conductance-aware port update coefficient."
        )
    if getattr(solver, "electric_dispersive_enabled", False):
        raise NotImplementedError(
            "Lumped FDTD coupling in electrically dispersive media requires the ADE correction in the joint circuit equation."
        )
    if getattr(solver, "nonlinear_enabled", False):
        raise NotImplementedError(
            "Lumped FDTD coupling in nonlinear media requires a field-dependent circuit update coefficient."
        )
    if getattr(solver, "modulation_enabled", False):
        raise NotImplementedError(
            "Lumped FDTD coupling in time-modulated media requires a time-dependent circuit update coefficient."
        )
    if getattr(solver, "full_aniso_enabled", False):
        raise NotImplementedError(
            "Lumped FDTD coupling in full anisotropy requires a tensor-valued circuit injection operator."
        )


def _validate_local_update_coefficient(solver, runtime: LumpedRuntime, field_name: str) -> None:
    eps = getattr(solver, f"eps_{field_name}").reshape(-1)
    actual = getattr(solver, f"c{field_name.lower()}_curl").reshape(-1)
    local_eps = torch.index_select(eps, 0, runtime.linear_indices)
    local_actual = torch.index_select(actual, 0, runtime.linear_indices)
    expected = runtime.dt / local_eps
    if not bool(torch.allclose(local_actual, expected, rtol=1.0e-5, atol=0.0)):
        raise ValueError(
            f"Lumped object {runtime.port_name!r} overlaps an absorber, conductor, "
            "or constrained Yee edge whose E update coefficient is not dt/eps."
        )


def _pec_structures(scene) -> tuple[object, ...]:
    return tuple(
        structure
        for structure in scene.structures
        if bool(getattr(structure.material, "is_pec", False))
    )


def _geometry_signed_distance(structure, points: torch.Tensor) -> torch.Tensor:
    geometry = structure.geometry
    reference = torch.as_tensor(geometry.position)
    local_points = points.to(device=reference.device, dtype=reference.dtype)
    return geometry.signed_distance(
        local_points[:, 0],
        local_points[:, 1],
        local_points[:, 2],
    )


def _validate_explicit_pec_gap(solver, geometry, lumped_object) -> None:
    structures = _pec_structures(solver.scene)
    if not structures:
        raise ValueError(
            f"Lumped object {geometry.port_name!r} has a PEC-suppressed voltage edge "
            "but no explicit PEC terminal structures."
        )
    spacing = min(
        float(min(solver.scene.dx_primal64)),
        float(min(solver.scene.dy_primal64)),
        float(min(solver.scene.dz_primal64)),
    )
    tolerance = max(1.0e-12, spacing * 1.0e-6)
    terminals = torch.as_tensor(
        (lumped_object.negative, lumped_object.positive),
        dtype=torch.float64,
    )
    terminal_distances = torch.stack(
        [_geometry_signed_distance(structure, terminals).to(dtype=torch.float64, device="cpu") for structure in structures],
        dim=0,
    )
    if bool(torch.any(terminal_distances < -tolerance)):
        raise ValueError(
            f"Lumped object {geometry.port_name!r} has a terminal embedded inside a PEC conductor."
        )
    negative_contacts = set(
        torch.nonzero(torch.abs(terminal_distances[:, 0]) <= tolerance).reshape(-1).tolist()
    )
    positive_contacts = set(
        torch.nonzero(torch.abs(terminal_distances[:, 1]) <= tolerance).reshape(-1).tolist()
    )
    if not negative_contacts or not positive_contacts:
        raise ValueError(
            f"Lumped object {geometry.port_name!r} must place both terminals on explicit PEC conductor surfaces."
        )
    if negative_contacts & positive_contacts:
        raise ValueError(
            f"Lumped object {geometry.port_name!r} connects the same PEC conductor at both terminals."
        )

    axis_index = "xyz".index(geometry.axis)
    indices = geometry.voltage_indices.detach().cpu()
    node_coordinates = (
        solver.scene.x_nodes64,
        solver.scene.y_nodes64,
        solver.scene.z_nodes64,
    )
    samples = []
    for index in indices:
        start = [float(node_coordinates[axis][int(index[axis])]) for axis in range(3)]
        end = list(start)
        end[axis_index] = float(node_coordinates[axis_index][int(index[axis_index]) + 1])
        for fraction in (0.25, 0.5, 0.75):
            samples.append(
                tuple((1.0 - fraction) * lower + fraction * upper for lower, upper in zip(start, end))
            )
    interior = torch.as_tensor(samples, dtype=torch.float64)
    for structure in structures:
        distance = _geometry_signed_distance(structure, interior)
        if bool(torch.any(distance < -tolerance)):
            raise ValueError(
                f"Lumped object {geometry.port_name!r} has a voltage edge embedded inside a PEC conductor."
            )


def _open_declared_pec_terminal_edges(solver, geometry, lumped_object) -> None:
    """Open explicitly declared lumped gaps in material PEC suppression.

    A port or standalone lumped element defines a physical feed gap.  The generic
    node-occupancy PEC stencil cannot distinguish an edge that bridges two PEC
    terminals from an edge buried in one conductor, so it suppresses both.  The
    explicit lumped declaration resolves that ambiguity for its voltage edges.
    """

    model = getattr(solver, "_compiled_material_model", None)
    if model is None or model.get("pec_occupancy") is None:
        return
    field_name = geometry.voltage_component
    curl_name = f"c{field_name.lower()}_curl"
    decay_name = f"c{field_name.lower()}_decay"
    curl = getattr(solver, curl_name)
    decay = getattr(solver, decay_name)
    indices = geometry.voltage_indices.to(device=solver.device, dtype=torch.int64)
    selector = tuple(indices[:, axis] for axis in range(3))
    local_curl = curl[selector]
    suppressed = torch.isclose(local_curl, torch.zeros_like(local_curl), rtol=0.0, atol=0.0)
    if not bool(torch.any(suppressed)):
        return
    _validate_explicit_pec_gap(solver, geometry, lumped_object)

    eps = getattr(solver, f"eps_{field_name}")
    restored_curl = solver.dt / eps[selector]
    curl[selector] = torch.where(suppressed, restored_curl, local_curl)
    local_decay = decay[selector]
    decay[selector] = torch.where(suppressed, torch.ones_like(local_decay), local_decay)
    uniformity = getattr(solver, "_coefficient_uniformity", None)
    if isinstance(uniformity, dict):
        uniformity[curl_name] = None
        uniformity[decay_name] = None


def _source_descriptor(
    excitation: PortExcitation | None,
    *,
    default_frequency: float,
    device: torch.device,
    dtype: torch.dtype,
):
    if excitation is None:
        zero = torch.zeros((), device=device, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)
        return None, 0.0, 0.0, 0.0, 0.0, zero

    source_time = excitation.source_time or CW(frequency=default_frequency)
    if isinstance(source_time, CustomSourceTime):
        raise NotImplementedError(
            "PortExcitation requires a device-native CW, GaussianPulse, or RickerWavelet source_time."
        )
    if not isinstance(source_time, (CW, GaussianPulse, RickerWavelet)):
        raise TypeError("Unsupported PortExcitation source_time type.")

    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    amplitude = excitation.amplitude.to(device=device, dtype=complex_dtype)
    amplitude = amplitude * float(source_time.amplitude)
    source_kind = source_time.kind
    frequency = float(source_time.frequency)
    fwidth = float(getattr(source_time, "fwidth", 0.0))
    phase = float(getattr(source_time, "phase", 0.0))
    delay = float(getattr(source_time, "delay", 0.0))
    if isinstance(source_time, RickerWavelet) and not bool(
        torch.isclose(amplitude.imag, torch.zeros_like(amplitude.imag))
    ):
        raise ValueError("RickerWavelet port excitation amplitude must be real.")
    return source_kind, frequency, fwidth, phase, delay, amplitude


def prepare_port_runtimes(solver, frequencies, excitations=()) -> tuple[PreparedPortRuntime, ...]:
    """Compile all lumped ports once and place their state on ``solver.device``."""

    frequency_tensor = torch.as_tensor(
        tuple(float(value) for value in frequencies),
        device=solver.device,
        dtype=torch.float64,
    )
    excitation_by_name = {excitation.port_name: excitation for excitation in excitations}
    ports_by_name = {port.name: port for port in solver.scene.ports}
    network_by_port = {
        port_name: network.name
        for network in getattr(solver.scene, "networks", ())
        for port_name in network.connected_port_names
    }
    termination_overrides = getattr(solver, "_port_termination_overrides", {})
    has_coupled_objects = bool(excitation_by_name) or any(
        termination_overrides.get(port.name, getattr(port, "termination", None)) is not None
        for port in ports_by_name.values()
    ) or bool(solver.scene.lumped_elements) or bool(network_by_port)
    if has_coupled_objects:
        _validate_supported_field_coupling(solver)
    runtimes = []
    for geometry in solver.scene.compile_ports(device=solver.device):
        port = ports_by_name[geometry.port_name]
        excitation = excitation_by_name.get(port.name)
        termination = termination_overrides.get(port.name, port.termination)
        embedded_network_name = network_by_port.get(port.name)
        _require_forward_only_parameters(port, excitation, termination)
        if excitation is not None and termination is not None:
            raise ValueError(
                f"Active port {port.name!r} cannot also declare a passive termination in the same run."
            )
        if embedded_network_name is not None and (excitation is not None or termination is not None):
            raise ValueError(
                f"Port {port.name!r} connected to embedded network "
                f"{embedded_network_name!r} cannot also declare an excitation or termination."
            )
        field = getattr(solver, geometry.voltage_component)
        lumped = None
        if excitation is not None:
            _open_declared_pec_terminal_edges(solver, geometry, port)
            resistance = _real_source_resistance(
                port,
                excitation,
                device=solver.device,
                dtype=field.dtype,
            )
            lumped = prepare_lumped_runtime(
                geometry,
                dt=solver.dt,
                eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
                yee_control_volume=_edge_control_volume(solver, geometry.voltage_component),
                resistance=resistance,
            )
            _validate_local_update_coefficient(solver, lumped, geometry.voltage_component)
        elif termination is not None:
            _open_declared_pec_terminal_edges(solver, geometry, port)
            lumped = prepare_lumped_runtime(
                geometry,
                dt=solver.dt,
                eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
                yee_control_volume=_edge_control_volume(solver, geometry.voltage_component),
                termination=termination,
            )
            _validate_local_update_coefficient(solver, lumped, geometry.voltage_component)
        elif embedded_network_name is not None:
            _open_declared_pec_terminal_edges(solver, geometry, port)
            lumped = prepare_lumped_runtime(
                geometry,
                dt=solver.dt,
                eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
                yee_control_volume=_edge_control_volume(solver, geometry.voltage_component),
                resistance=0.0,
            )
            _validate_local_update_coefficient(solver, lumped, geometry.voltage_component)

        source_kind, source_frequency, source_fwidth, source_phase, source_delay, amplitude = (
            _source_descriptor(
                excitation,
                default_frequency=float(frequency_tensor[0]),
                device=solver.device,
                dtype=field.dtype,
            )
        )
        runtimes.append(
            PreparedPortRuntime(
                port=port,
                geometry=geometry,
                frequencies=frequency_tensor,
                field_name=geometry.voltage_component,
                lumped=lumped,
                excitation=excitation,
                source_kind=source_kind,
                source_frequency=source_frequency,
                source_fwidth=source_fwidth,
                source_phase=source_phase,
                source_delay=source_delay,
                source_amplitude=amplitude,
                drive_buffer=torch.zeros((), device=solver.device, dtype=field.dtype),
                electric_time=torch.as_tensor(solver.dt, device=solver.device, dtype=field.dtype),
                magnetic_time=torch.as_tensor(0.5 * solver.dt, device=solver.device, dtype=field.dtype),
                embedded_network_name=embedded_network_name,
            )
        )
    solver._port_runtimes = tuple(runtimes)
    element_by_name = {element.name: element for element in solver.scene.lumped_elements}
    element_runtimes = []
    for geometry in solver.scene.compile_lumped_elements(device=solver.device):
        field = getattr(solver, geometry.voltage_component)
        element = element_by_name[geometry.element_name]
        _open_declared_pec_terminal_edges(solver, geometry, element)
        element_runtime = prepare_lumped_runtime(
            geometry,
            dt=solver.dt,
            eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
            yee_control_volume=_edge_control_volume(solver, geometry.voltage_component),
            termination=element,
        )
        _validate_local_update_coefficient(solver, element_runtime, geometry.voltage_component)
        element_runtimes.append((element_runtime, geometry.voltage_component))
    solver._lumped_element_runtimes = tuple(element_runtimes)
    occupied_edges = {}
    coupled = [
        (runtime.port_name, field_name, runtime.linear_indices)
        for runtime, field_name in solver._lumped_element_runtimes
    ]
    coupled.extend(
        (runtime.port.name, runtime.field_name, runtime.lumped.linear_indices)
        for runtime in solver._port_runtimes
        if runtime.lumped is not None
    )
    for name, field_name, indices in coupled:
        for index in indices.detach().cpu().tolist():
            key = (field_name, int(index))
            previous = occupied_edges.get(key)
            if previous is not None:
                raise ValueError(
                    f"Lumped objects {previous!r} and {name!r} overlap the same {field_name} Yee edge."
                )
            occupied_edges[key] = name
    return solver._port_runtimes


def _spectral_weights(
    solver,
    frequencies: torch.Tensor,
    time_steps: int,
    window_type: str,
    *,
    start_at_zero: bool = False,
) -> torch.Tensor:
    if start_at_zero:
        starts = torch.zeros_like(frequencies)
    else:
        starts = torch.as_tensor(
            [
                solver._compute_spectral_start_step(float(frequency), window_type=window_type)
                for frequency in frequencies.detach().cpu()
            ],
            device=frequencies.device,
            dtype=frequencies.dtype,
        )
    steps = torch.arange(time_steps, device=frequencies.device, dtype=frequencies.dtype)[:, None]
    end = torch.as_tensor(float(time_steps), device=frequencies.device, dtype=frequencies.dtype)
    active = (steps >= starts[None, :]) & (steps < end)
    duration = torch.clamp(end - starts, min=1.0)
    position = (steps - starts[None, :]) / duration[None, :]
    kind = str(window_type).lower()
    if kind == "none":
        weights = torch.ones_like(position)
    elif kind == "hanning":
        weights = 0.5 * (1.0 - torch.cos(2.0 * torch.pi * position))
    elif kind == "ramp":
        ramp = 0.5 * (1.0 - torch.cos(torch.pi * position / 0.1))
        weights = torch.where(position < 0.1, ramp, torch.ones_like(position))
    else:
        raise ValueError("Port DFT window must be 'none', 'hanning', or 'ramp'.")
    return torch.where(active, weights, torch.zeros_like(weights))


def _weighted_phase_table(
    frequencies: torch.Tensor,
    sample_times: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    angle = 2.0 * torch.pi * sample_times[:, None] * frequencies[None, :]
    return weights * torch.complex(torch.cos(angle), torch.sin(angle))


def prepare_port_spectral_accumulators(solver, time_steps: int, window_type: str) -> None:
    runtimes = getattr(solver, "_port_runtimes", ())
    port_pulse = any(
        runtime.source_kind is not None and runtime.source_kind != "cw"
        for runtime in runtimes
    )
    for runtime in runtimes:
        effective_window = "none" if port_pulse else window_type
        runtime.window_weights = _spectral_weights(
            solver,
            runtime.frequencies,
            time_steps,
            effective_window,
            start_at_zero=port_pulse,
        )
        runtime.window_type = effective_window
        runtime.accumulator = PortDFTAccumulator(runtime.frequencies)
        runtime.drive_accumulator = (
            PortDFTAccumulator(runtime.frequencies)
            if runtime.excitation is not None
            else None
        )
        step = torch.arange(
            time_steps,
            device=runtime.frequencies.device,
            dtype=runtime.frequencies.dtype,
        )
        dt = torch.as_tensor(
            solver.dt,
            device=runtime.frequencies.device,
            dtype=runtime.frequencies.dtype,
        )
        electric_times = (step + 1.0) * dt
        magnetic_times = (step + 0.5) * dt
        voltage_times = magnetic_times if runtime.lumped is not None else electric_times
        runtime.current_phase_weights = _weighted_phase_table(
            runtime.frequencies,
            magnetic_times,
            runtime.window_weights,
        )
        runtime.voltage_phase_weights = (
            runtime.current_phase_weights
            if runtime.lumped is not None
            else _weighted_phase_table(
                runtime.frequencies,
                voltage_times,
                runtime.window_weights,
            )
        )
        runtime.observer_current_buffer = torch.empty(
            (),
            device=solver.device,
            dtype=getattr(solver, runtime.field_name).dtype,
        )
        runtime.observer_window_buffer = torch.empty(
            (1, runtime.frequencies.numel()),
            device=solver.device,
            dtype=runtime.frequencies.dtype,
        )
        runtime.observer_voltage_kernel_buffer = torch.empty(
            (1, runtime.frequencies.numel()),
            device=solver.device,
            dtype=runtime.voltage_phase_weights.dtype,
        )
        runtime.observer_current_kernel_buffer = torch.empty_like(
            runtime.observer_voltage_kernel_buffer
        )
        runtime.sample_index = 0
    solver._port_observer_step = torch.zeros(
        (1,),
        device=solver.device,
        dtype=torch.int64,
    )
    solver._port_observer_graph_active = False


def complete_port_spectral_normalization(solver) -> None:
    """Restore planned DFT normalizers after an early field shutoff."""

    for runtime in getattr(solver, "_port_runtimes", ()):
        if runtime.window_weights is None:
            continue
        planned = runtime.window_weights.sum(dim=0)
        if runtime.accumulator is not None:
            runtime.accumulator._window_weight_sum.copy_(planned)
        if runtime.drive_accumulator is not None:
            runtime.drive_accumulator._window_weight_sum.copy_(planned)


def _drive_value(runtime: PreparedPortRuntime, time: torch.Tensor) -> torch.Tensor:
    if runtime.source_kind is None:
        return torch.zeros_like(runtime.drive_buffer)
    if runtime.source_kind == "cw":
        angle = 2.0 * torch.pi * runtime.source_frequency * time + runtime.source_phase
        value = runtime.source_amplitude.real * torch.cos(angle)
        value = value + runtime.source_amplitude.imag * torch.sin(angle)
    elif runtime.source_kind == "gaussian_pulse":
        tau = time - runtime.source_delay
        envelope = torch.exp(-0.5 * (2.0 * torch.pi * runtime.source_fwidth * tau).square())
        angle = 2.0 * torch.pi * runtime.source_frequency * tau + runtime.source_phase
        carrier = runtime.source_amplitude.real * torch.cos(angle)
        carrier = carrier + runtime.source_amplitude.imag * torch.sin(angle)
        value = envelope * carrier
    else:
        tau = time - runtime.source_delay
        alpha_sq = (torch.pi * runtime.source_frequency * tau).square()
        value = runtime.source_amplitude.real * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)
    return value


def _evaluate_drive(runtime: PreparedPortRuntime) -> torch.Tensor:
    value = _drive_value(runtime, runtime.magnetic_time)
    runtime.drive_buffer.copy_(value)
    return runtime.drive_buffer


def replay_port_runtimes(
    solver,
    electric_fields: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    time_value,
    capture=None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Replay all series branch corrections and return their next auxiliary state."""

    if not getattr(solver, "_port_runtimes", ()) and not getattr(
        solver, "_lumped_element_runtimes", ()
    ):
        return electric_fields, {}
    fields = dict(electric_fields)
    next_auxiliary = {}
    sample_time = torch.as_tensor(
        time_value + 0.5 * float(solver.dt),
        device=solver.device,
        dtype=solver.Ex.dtype,
    )
    traces = []
    for index, port_runtime in enumerate(getattr(solver, "_port_runtimes", ())):
        runtime = port_runtime.lumped
        if runtime is None or port_runtime.embedded_network_name is not None:
            continue
        inductor_name = lumped_state_name("port", index, "inductor_current")
        capacitor_name = lumped_state_name("port", index, "capacitor_voltage")
        corrected, next_inductor, next_capacitor, trace = replay_lumped_runtime(
            runtime,
            fields[port_runtime.field_name],
            inductor_current=state[inductor_name],
            capacitor_voltage=state[capacitor_name],
            drive=_drive_value(port_runtime, sample_time),
            field_name=port_runtime.field_name,
            kind="port",
            index=index,
        )
        fields[port_runtime.field_name] = corrected
        next_auxiliary[inductor_name] = next_inductor
        next_auxiliary[capacitor_name] = next_capacitor
        traces.append(trace)
    for index, (runtime, field_name) in enumerate(
        getattr(solver, "_lumped_element_runtimes", ())
    ):
        inductor_name = lumped_state_name("element", index, "inductor_current")
        capacitor_name = lumped_state_name("element", index, "capacitor_voltage")
        corrected, next_inductor, next_capacitor, trace = replay_lumped_runtime(
            runtime,
            fields[field_name],
            inductor_current=state[inductor_name],
            capacitor_voltage=state[capacitor_name],
            drive=torch.zeros_like(runtime.default_thevenin_voltage),
            field_name=field_name,
            kind="element",
            index=index,
        )
        fields[field_name] = corrected
        next_auxiliary[inductor_name] = next_inductor
        next_auxiliary[capacitor_name] = next_capacitor
        traces.append(trace)
    if capture is not None:
        capture.append(tuple(traces))
    return fields, next_auxiliary


def _source_amplitude_pullback(
    runtime: PreparedPortRuntime,
    grad_drive: torch.Tensor,
    sample_time: torch.Tensor,
) -> torch.Tensor:
    excitation = runtime.excitation
    if excitation is None:
        return torch.zeros_like(runtime.source_amplitude)
    scale = float((excitation.source_time or CW(frequency=runtime.source_frequency)).amplitude)
    if runtime.source_kind == "cw":
        angle = 2.0 * torch.pi * runtime.source_frequency * sample_time + runtime.source_phase
        real_basis = scale * torch.cos(angle)
        imag_basis = scale * torch.sin(angle)
    elif runtime.source_kind == "gaussian_pulse":
        tau = sample_time - runtime.source_delay
        envelope = torch.exp(-0.5 * (2.0 * torch.pi * runtime.source_fwidth * tau).square())
        angle = 2.0 * torch.pi * runtime.source_frequency * tau + runtime.source_phase
        real_basis = scale * envelope * torch.cos(angle)
        imag_basis = scale * envelope * torch.sin(angle)
    else:
        tau = sample_time - runtime.source_delay
        alpha_sq = (torch.pi * runtime.source_frequency * tau).square()
        real_basis = scale * (1.0 - 2.0 * alpha_sq) * torch.exp(-alpha_sq)
        imag_basis = torch.zeros_like(real_basis)
    if excitation.amplitude.is_complex():
        return torch.complex(grad_drive * real_basis, grad_drive * imag_basis).to(
            dtype=excitation.amplitude.dtype
        )
    return (grad_drive * real_basis).to(dtype=excitation.amplitude.dtype)


def pullback_port_runtimes(
    solver,
    traces,
    adjoint_state: dict[str, torch.Tensor],
    *,
    port_sample_adjoints: dict[
        int,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    eps_by_field: dict[str, torch.Tensor],
    time_value,
):
    """Reverse all local branch solves before the Maxwell reverse step."""

    updated = dict(adjoint_state)
    grad_eps = {name: torch.zeros_like(value) for name, value in eps_by_field.items()}
    semantic_grads = {}
    sample_time = torch.as_tensor(
        time_value + 0.5 * float(solver.dt),
        device=solver.device,
        dtype=solver.Ex.dtype,
    )
    for trace in reversed(tuple(traces)):
        inductor_name = lumped_state_name(trace.kind, trace.index, "inductor_current")
        capacitor_name = lumped_state_name(trace.kind, trace.index, "capacitor_voltage")
        voltage_seed, current_seed, direct_drive_seed = port_sample_adjoints.get(
            trace.index,
            (
                torch.zeros_like(trace.branch_current),
                torch.zeros_like(trace.branch_current),
                torch.zeros_like(trace.branch_current),
            ),
        ) if trace.kind == "port" else (
            torch.zeros_like(trace.branch_current),
            torch.zeros_like(trace.branch_current),
            torch.zeros_like(trace.branch_current),
        )
        voltage_seed = voltage_seed.to(
            device=trace.branch_current.device,
            dtype=trace.branch_current.dtype,
        )
        current_seed = current_seed.to(
            device=trace.branch_current.device,
            dtype=trace.branch_current.dtype,
        )
        direct_drive_seed = direct_drive_seed.to(
            device=trace.branch_current.device,
            dtype=trace.branch_current.dtype,
        )
        result = pullback_lumped_runtime(
            trace,
            updated[trace.field_name],
            inductor_current_adjoint=updated[inductor_name],
            capacitor_voltage_adjoint=updated[capacitor_name],
            voltage_sample_adjoint=voltage_seed,
            network_current_sample_adjoint=current_seed,
            eps_edge=eps_by_field[trace.field_name],
        )
        updated[trace.field_name] = result.field_adjoint
        updated[inductor_name] = result.inductor_current_adjoint
        updated[capacitor_name] = result.capacitor_voltage_adjoint
        grad_eps[trace.field_name] = grad_eps[trace.field_name] + result.grad_eps
        if trace.kind == "port":
            port_runtime = solver._port_runtimes[trace.index]
            termination = getattr(port_runtime.port, "termination", None)
            if port_runtime.excitation is not None:
                key = ("excitation", port_runtime.port.name, "amplitude")
                contribution = _source_amplitude_pullback(
                    port_runtime,
                    result.grad_drive + direct_drive_seed,
                    sample_time,
                )
                semantic_grads[key] = semantic_grads.get(key, torch.zeros_like(contribution)) + contribution
            elif termination is not None:
                for component, value in (
                    ("r", result.grad_resistance),
                    ("l", result.grad_inductance),
                    ("c", result.grad_capacitance),
                ):
                    if getattr(termination, component, None) is not None:
                        key = ("port", port_runtime.port.name, component)
                        semantic_grads[key] = semantic_grads.get(key, torch.zeros_like(value)) + value
        else:
            element = solver.scene.lumped_elements[trace.index]
            gradient = {
                "resistor": result.grad_resistance,
                "inductor": result.grad_inductance,
                "capacitor": result.grad_capacitance,
            }[element.kind]
            key = ("element", element.name, "value")
            semantic_grads[key] = semantic_grads.get(key, torch.zeros_like(gradient)) + gradient
    return updated, grad_eps, semantic_grads


def apply_port_runtimes(solver) -> None:
    """Apply prepared source/load corrections without host-device transfers."""

    for runtime in getattr(solver, "_port_runtimes", ()):
        if runtime.lumped is None or runtime.embedded_network_name is not None:
            continue
        drive = _evaluate_drive(runtime)
        apply_lumped_runtime(
            runtime.lumped,
            getattr(solver, runtime.field_name),
            thevenin_voltage=drive,
        )
    for runtime, field_name in getattr(solver, "_lumped_element_runtimes", ()):
        apply_lumped_runtime(runtime, getattr(solver, field_name))


def accumulate_port_observers(solver) -> None:
    fields = None
    for runtime in getattr(solver, "_port_runtimes", ()):
        if (
            runtime.accumulator is None
            or runtime.window_weights is None
            or runtime.voltage_phase_weights is None
            or runtime.current_phase_weights is None
            or runtime.observer_current_buffer is None
        ):
            raise RuntimeError("Port spectral accumulators were not prepared.")
        if runtime.lumped is not None:
            # Voltage and current must come from the same implicit coupling
            # state. The corrected Yee field is the post-branch voltage, whereas
            # the constitutive branch current is evaluated at the coupling
            # midpoint. Mixing them creates a false incident wave at a matched
            # passive port.
            voltage = runtime.lumped.last_voltage_midpoint
            # Branch current is positive from the field into the external
            # branch. RF network current has the opposite sign: into the field
            # network from the port reference side.
            torch.neg(
                runtime.lumped.last_branch_current,
                out=runtime.observer_current_buffer,
            )
            current = runtime.observer_current_buffer
        else:
            if fields is None:
                fields = _field_map(solver)
            voltage = runtime.geometry.integrate_voltage(fields)
            current = runtime.geometry.integrate_current(fields)
        weight = runtime.window_weights[runtime.sample_index]
        voltage_kernel = runtime.voltage_phase_weights[runtime.sample_index]
        current_kernel = runtime.current_phase_weights[runtime.sample_index]
        runtime.accumulator.accumulate_precomputed(
            voltage,
            current,
            voltage_kernel=voltage_kernel,
            current_kernel=current_kernel,
            window_weight=weight,
        )
        if runtime.drive_accumulator is not None:
            runtime.drive_accumulator.accumulate_precomputed(
                runtime.drive_buffer,
                None,
                voltage_kernel=current_kernel,
                current_kernel=current_kernel,
                window_weight=weight,
            )
        runtime.sample_index += 1
        runtime.electric_time.add_(runtime.lumped.dt if runtime.lumped is not None else solver.dt)
        runtime.magnetic_time.add_(runtime.lumped.dt if runtime.lumped is not None else solver.dt)


def _accumulate_embedded_port_observers_gpu(solver) -> None:
    step = solver._port_observer_step
    for runtime in solver._port_runtimes:
        accumulator = runtime.accumulator
        lumped = runtime.lumped
        torch.neg(
            lumped.last_branch_current,
            out=runtime.observer_current_buffer,
        )
        torch.index_select(
            runtime.window_weights,
            0,
            step,
            out=runtime.observer_window_buffer,
        )
        torch.index_select(
            runtime.voltage_phase_weights,
            0,
            step,
            out=runtime.observer_voltage_kernel_buffer,
        )
        torch.index_select(
            runtime.current_phase_weights,
            0,
            step,
            out=runtime.observer_current_kernel_buffer,
        )
        torch.mul(
            runtime.observer_voltage_kernel_buffer[0],
            lumped.last_voltage_midpoint,
            out=accumulator._voltage_term,
        )
        accumulator._voltage_sum.add_(accumulator._voltage_term)
        torch.mul(
            runtime.observer_current_kernel_buffer[0],
            runtime.observer_current_buffer,
            out=accumulator._current_term,
        )
        accumulator._current_sum.add_(accumulator._current_term)
        accumulator._window_weight_sum.add_(runtime.observer_window_buffer[0])
        runtime.electric_time.add_(lumped.dt)
        runtime.magnetic_time.add_(lumped.dt)
    step.add_(1)


def make_port_observer_runner(solver, *, use_cuda_graph: bool):
    """Capture fixed embedded-terminal DFT updates behind one graph launch."""

    def normal() -> None:
        accumulate_port_observers(solver)
    runtimes = tuple(getattr(solver, "_port_runtimes", ()))
    graphable = (
        use_cuda_graph
        and torch.cuda.is_available()
        and torch.device(solver.device).type == "cuda"
        and bool(runtimes)
        and all(
            runtime.lumped is not None
            and runtime.embedded_network_name is not None
            and runtime.excitation is None
            and runtime.drive_accumulator is None
            for runtime in runtimes
        )
        and all(runtime.window_weights.shape[0] >= 4 for runtime in runtimes)
    )
    if not graphable:
        return normal

    from .cuda.runtime.graph import CudaGraphRunner

    tensors = [solver._port_observer_step]
    for runtime in runtimes:
        accumulator = runtime.accumulator
        tensors.extend(
            (
                runtime.observer_current_buffer,
                runtime.observer_window_buffer,
                runtime.observer_voltage_kernel_buffer,
                runtime.observer_current_kernel_buffer,
                runtime.electric_time,
                runtime.magnetic_time,
                accumulator._voltage_sum,
                accumulator._current_sum,
                accumulator._window_weight_sum,
                accumulator._voltage_term,
                accumulator._current_term,
            )
        )
    saved = [tensor.clone() for tensor in tensors]

    def restore() -> None:
        for tensor, value in zip(tensors, saved):
            tensor.copy_(value)

    try:
        replay = CudaGraphRunner(enabled=True, warmup_steps=3).capture(
            lambda: _accumulate_embedded_port_observers_gpu(solver)
        )
    except Exception:
        restore()
        return normal
    restore()
    solver._port_observer_graph_active = True
    return replay


def complete_port_observer_graph(solver, sample_count: int) -> None:
    if not getattr(solver, "_port_observer_graph_active", False):
        return
    for runtime in getattr(solver, "_port_runtimes", ()):
        runtime.sample_index = int(sample_count)
        runtime.accumulator._sample_count = int(sample_count)


def finalize_port_data(solver) -> dict[str, PortData]:
    output = {}
    for runtime in getattr(solver, "_port_runtimes", ()):
        if runtime.accumulator is None:
            continue
        if bool(torch.any(runtime.accumulator._window_weight_sum <= 0.0)):
            raise RuntimeError(
                f"Port {runtime.port.name!r} has no active DFT samples; increase run_time."
            )
        voltage, current = runtime.accumulator.phasors(normalization="peak")
        available_power = None
        excitation_threshold = None
        if runtime.drive_accumulator is not None and runtime.lumped is not None:
            source_voltage, _ = runtime.drive_accumulator.phasors(normalization="peak")
            source_magnitude = torch.abs(source_voltage)
            excitation_threshold = torch.clamp(
                torch.max(source_magnitude) * 1.0e-6,
                min=torch.finfo(source_magnitude.dtype).tiny,
            )
            weak = source_magnitude < excitation_threshold
            if bool(torch.any(weak)):
                weak_frequencies = runtime.frequencies[weak].detach().cpu().tolist()
                raise RuntimeError(
                    f"Port {runtime.port.name!r} excitation spectrum is below the "
                    f"relative threshold at frequencies {weak_frequencies}."
                )
            available_power = torch.abs(source_voltage).square() / (8.0 * runtime.lumped.resistance)
        output[runtime.port.name] = PortData(
            port_name=runtime.port.name,
            frequencies=runtime.frequencies,
            voltage=voltage,
            current=current,
            z0=runtime.port.reference_impedance,
            direction="+" if runtime.geometry.direction > 0 else "-",
            reference_plane=runtime.port.reference_plane,
            available_power=available_power,
            metadata={
                "axis": runtime.geometry.axis,
                "orientation": runtime.geometry.direction,
                "current_convention": (
                    "entering_field_network"
                    if runtime.lumped is not None
                    else "positive_axis_h_contour"
                ),
                "dft_samples": runtime.sample_index,
                "window": runtime.window_type,
                "excitation_threshold": excitation_threshold,
                "units": {"voltage": "V", "current": "A", "power": "W"},
            },
        )
    return output


__all__ = [
    "PortDFTAccumulator",
    "PreparedPortRuntime",
    "accumulate_port_observers",
    "apply_port_runtimes",
    "complete_port_spectral_normalization",
    "finalize_port_data",
    "prepare_port_runtimes",
    "prepare_port_spectral_accumulators",
    "pullback_port_runtimes",
    "replay_port_runtimes",
]

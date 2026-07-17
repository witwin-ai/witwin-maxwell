from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from ..lumped import PortExcitation
from ..network import PortData
from ..sources import CW, CustomSourceTime, GaussianPulse, RickerWavelet
from .checkpoint import lumped_state_name
from .circuits import apply_circuit_runtimes, prepare_circuit_runtimes
from .lumped import (
    LumpedRuntime,
    LumpedStepPullback,
    LumpedStepTrace,
    apply_lumped_runtime,
    prepare_lumped_runtime,
    pullback_lumped_runtime,
    replay_lumped_runtime,
)
from .wire import _target_masses


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
        self._voltage_term = torch.empty_like(self._voltage_sum)
        self._current_term = torch.empty_like(self._current_sum)
        self._sample_count = 0

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
class PreparedWirePortProvider:
    """Generalized field/node coordinate owned by one wire-bound RF port."""

    geometry: object
    coordinate: torch.Tensor
    field_count: int
    node_ids: torch.Tensor
    node_capacitance: torch.Tensor
    field_control_volumes: torch.Tensor
    energy_masses: torch.Tensor
    gap_weight_indices: torch.Tensor
    component_indices: tuple[torch.Tensor, ...]
    component_offsets: tuple[torch.Tensor, ...]
    voltage_weights: torch.Tensor


@dataclass(frozen=True)
class WirePortStepTrace:
    """Packing state needed to transpose one generalized port correction."""

    lumped: object
    lumped_substeps: tuple[object, ...]
    provider: PreparedWirePortProvider
    input_charge: torch.Tensor
    output_coordinate: torch.Tensor
    exact_resistive: bool = False


@dataclass(frozen=True)
class _GeneralizedPortGeometry:
    port_name: str
    voltage_indices: torch.Tensor
    voltage_weights: torch.Tensor


@dataclass
class PreparedPortRuntime:
    """Single-device FDTD state shared by a port source, load, and observer."""

    port: object
    geometry: object
    frequencies: torch.Tensor
    field_name: str | None
    yee_control_volume: torch.Tensor | None
    lumped: LumpedRuntime | None
    wire_provider: PreparedWirePortProvider | None
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
    substeps: int = 1
    substep_buffers: dict[str, torch.Tensor] | None = None
    exact_resistive: bool = False
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
    circuit_port: object | None = None
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


def _wire_port_field_values(
    fields: dict[str, torch.Tensor],
    geometry,
) -> torch.Tensor:
    values = fields["Ex"].new_empty(geometry.edge_offsets.numel())
    for component, name in enumerate(("Ex", "Ey", "Ez")):
        selected = torch.nonzero(
            geometry.edge_components == component,
            as_tuple=False,
        ).reshape(-1)
        if selected.numel() == 0:
            continue
        values.index_copy_(
            0,
            selected,
            fields[name]
            .reshape(-1)
            .index_select(0, geometry.edge_offsets.index_select(0, selected)),
        )
    return values


def _pack_wire_port_coordinate(
    provider: PreparedWirePortProvider,
    fields: dict[str, torch.Tensor],
    charge: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    coordinate = (
        torch.empty_like(provider.coordinate)
        if out is None
        else out
    )
    flat = coordinate.reshape(-1)
    if provider.field_count:
        for name, selected, offsets in zip(
            ("Ex", "Ey", "Ez"),
            provider.component_indices,
            provider.component_offsets,
        ):
            if selected.numel():
                flat.index_copy_(
                    0,
                    selected,
                    fields[name].reshape(-1).index_select(0, offsets),
                )
    flat[provider.field_count :].copy_(
        charge.index_select(0, provider.node_ids) / provider.node_capacitance
    )
    return coordinate


def _unpack_wire_port_coordinate(
    provider: PreparedWirePortProvider,
    coordinate: torch.Tensor,
    fields: dict[str, torch.Tensor],
    charge: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    output_fields = dict(fields)
    flat_coordinate = coordinate.reshape(-1)
    for name, selected, offsets in zip(
        ("Ex", "Ey", "Ez"),
        provider.component_indices,
        provider.component_offsets,
    ):
        if selected.numel() == 0:
            continue
        flat = fields[name].reshape(-1).clone()
        flat.index_copy_(
            0,
            offsets,
            flat_coordinate.index_select(0, selected),
        )
        output_fields[name] = flat.reshape_as(fields[name])
    output_charge = charge.clone()
    output_charge.index_copy_(
        0,
        provider.node_ids,
        flat_coordinate[provider.field_count :] * provider.node_capacitance,
    )
    return output_fields, output_charge


def _unpack_wire_port_coordinate_in_place(
    provider: PreparedWirePortProvider,
    coordinate: torch.Tensor,
    fields: dict[str, torch.Tensor],
    charge: torch.Tensor,
) -> None:
    flat_coordinate = coordinate.reshape(-1)
    for name, selected, offsets in zip(
        ("Ex", "Ey", "Ez"),
        provider.component_indices,
        provider.component_offsets,
    ):
        if selected.numel():
            fields[name].reshape(-1).index_copy_(
                0,
                offsets,
                flat_coordinate.index_select(0, selected),
            )
    charge.index_copy_(
        0,
        provider.node_ids,
        flat_coordinate[provider.field_count :] * provider.node_capacitance,
    )


def _prepare_wire_port_provider(
    solver,
    geometry,
) -> tuple[PreparedWirePortProvider, _GeneralizedPortGeometry]:
    runtime = getattr(solver, "_wire_runtime", None)
    if runtime is None:
        raise RuntimeError(
            f"Wire-bound port {geometry.port_name!r} requires an initialized wire runtime."
        )
    components = geometry.edge_components.to(device=solver.device, dtype=torch.int32)
    offsets = geometry.edge_offsets.to(device=solver.device, dtype=torch.int64)
    weights = geometry.edge_weights.to(device=solver.device, dtype=solver.Ex.dtype)
    if not (components.numel() == offsets.numel() == weights.numel()):
        raise ValueError("Compiled wire-port gap row has inconsistent sparse arrays.")
    keys = tuple(zip(components.detach().cpu().tolist(), offsets.detach().cpu().tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("Compiled wire-port gap row must contain unique Yee targets.")
    node_ids = torch.as_tensor(
        (geometry.negative_node_id, geometry.positive_node_id),
        device=solver.device,
        dtype=torch.int64,
    )
    if int(node_ids[0]) == int(node_ids[1]):
        raise ValueError("Wire-bound port terminals resolve to the same global node.")
    node_capacitance = runtime.coefficients["node_capacitance"].index_select(
        0, node_ids
    )
    if not bool(torch.all(torch.isfinite(node_capacitance) & (node_capacitance > 0.0))):
        raise ValueError("Wire-bound port nodes must have positive finite capacitance.")
    field_masses = _target_masses(solver, components, offsets)
    field_eps = _wire_port_field_values(
        {"Ex": solver.eps_Ex, "Ey": solver.eps_Ey, "Ez": solver.eps_Ez},
        geometry,
    )
    field_control_volumes = field_masses / field_eps
    energy_masses = torch.cat((field_masses, node_capacitance)).reshape(-1, 1, 1)
    voltage_weights = torch.cat(
        (
            weights,
            torch.as_tensor((1.0, -1.0), device=solver.device, dtype=solver.Ex.dtype),
        )
    )
    component_indices = tuple(
        torch.nonzero(components == component, as_tuple=False).reshape(-1)
        for component in range(3)
    )
    component_offsets = tuple(
        offsets.index_select(0, selected)
        for selected in component_indices
    )
    entry_count = int(voltage_weights.numel())
    indices = torch.zeros((entry_count, 3), device=solver.device, dtype=torch.int64)
    indices[:, 0] = torch.arange(entry_count, device=solver.device, dtype=torch.int64)
    generalized = _GeneralizedPortGeometry(
        port_name=geometry.port_name,
        voltage_indices=indices,
        voltage_weights=voltage_weights,
    )
    return PreparedWirePortProvider(
        geometry=geometry,
        coordinate=torch.empty_like(energy_masses),
        field_count=int(weights.numel()),
        node_ids=node_ids,
        node_capacitance=node_capacitance,
        field_control_volumes=field_control_volumes,
        energy_masses=energy_masses,
        gap_weight_indices=torch.arange(
            geometry.gap_offset,
            geometry.gap_offset + weights.numel(),
            device=solver.device,
            dtype=torch.int64,
        ),
        component_indices=component_indices,
        component_offsets=component_offsets,
        voltage_weights=voltage_weights,
    ), generalized


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
    circuit_port_names = {
        binding.port_name
        for circuit in getattr(solver.scene, "circuits", ())
        for binding in circuit.bindings
    }
    network_by_port = {
        port_name: network.name
        for network in getattr(solver.scene, "networks", ())
        for port_name in network.connected_port_names
    }
    termination_overrides = getattr(solver, "_port_termination_overrides", {})
    has_coupled_objects = bool(excitation_by_name) or any(
        termination_overrides.get(port.name, getattr(port, "termination", None)) is not None
        for port in ports_by_name.values()
    ) or bool(solver.scene.lumped_elements) or bool(circuit_port_names) or bool(network_by_port)
    if has_coupled_objects:
        _validate_supported_field_coupling(solver)
    runtimes = []
    for geometry in solver.scene.compile_ports(device=solver.device):
        port = ports_by_name[geometry.port_name]
        excitation = excitation_by_name.get(port.name)
        termination = termination_overrides.get(port.name, port.termination)
        if port.name in circuit_port_names and excitation is not None:
            raise ValueError(
                f"Circuit-bound port {port.name!r} cannot also declare a direct PortExcitation."
            )
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
        wire_provider = None
        generalized_geometry = None
        if bool(getattr(geometry, "is_wire_bound", False)):
            wire_provider, generalized_geometry = _prepare_wire_port_provider(
                solver, geometry
            )
            field = solver.Ex
            field_name = None
        else:
            field = getattr(solver, geometry.voltage_component)
            field_name = geometry.voltage_component
        control_volume = None
        lumped = None
        if port.name in circuit_port_names:
            control_volume = _edge_control_volume(solver, geometry.voltage_component)
            _open_declared_pec_terminal_edges(solver, geometry, port)
        elif excitation is not None:
            resistance = _real_source_resistance(
                port,
                excitation,
                device=solver.device,
                dtype=field.dtype,
            )
            if wire_provider is not None:
                lumped = prepare_lumped_runtime(
                    generalized_geometry,
                    dt=0.5 * solver.dt,
                    eps_edge=wire_provider.energy_masses,
                    yee_control_volume=1.0,
                    resistance=resistance,
                )
            else:
                control_volume = _edge_control_volume(solver, geometry.voltage_component)
                _open_declared_pec_terminal_edges(solver, geometry, port)
                lumped = prepare_lumped_runtime(
                    geometry,
                    dt=solver.dt,
                    eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
                    yee_control_volume=control_volume,
                    resistance=resistance,
                )
                _validate_local_update_coefficient(
                    solver, lumped, geometry.voltage_component
                )
        elif termination is not None:
            if wire_provider is not None:
                lumped = prepare_lumped_runtime(
                    generalized_geometry,
                    dt=0.5 * solver.dt,
                    eps_edge=wire_provider.energy_masses,
                    yee_control_volume=1.0,
                    termination=termination,
                )
            else:
                control_volume = _edge_control_volume(solver, geometry.voltage_component)
                _open_declared_pec_terminal_edges(solver, geometry, port)
                lumped = prepare_lumped_runtime(
                    geometry,
                    dt=solver.dt,
                    eps_edge=getattr(solver, f"eps_{geometry.voltage_component}"),
                    yee_control_volume=control_volume,
                    termination=termination,
                )
                _validate_local_update_coefficient(
                    solver, lumped, geometry.voltage_component
                )
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
        substeps = 2 if wire_provider is not None and lumped is not None else 1
        exact_resistive = bool(
            wire_provider is not None
            and lumped is not None
            and not lumped.inductance_active
            and not lumped.capacitance_active
            and torch.isfinite(lumped.resistance)
            and lumped.resistance > 0.0
        )
        substep_buffers = (
            {
                name: torch.zeros((), device=solver.device, dtype=field.dtype)
                for name in (
                    "voltage_before",
                    "voltage_midpoint",
                    "voltage_after",
                    "model_voltage_midpoint",
                    "branch_current",
                    "resistor_current",
                    "capacitor_current",
                    "inductor_current_midpoint",
                    "dissipated_energy",
                    "stored_energy_change",
                    "source_work",
                    "field_energy_change",
                )
            }
            if substeps == 2
            else None
        )
        if exact_resistive:
            masses = wire_provider.energy_masses.reshape(-1)
            weights = lumped.voltage_weights
            gamma = torch.sum(weights.square() / masses)
            macro_dt = torch.as_tensor(
                solver.dt, device=solver.device, dtype=field.dtype
            )
            substep_buffers.update(
                {
                    "gamma": gamma,
                    "macro_dt": macro_dt,
                    "decay": torch.exp(-macro_dt * gamma / lumped.resistance),
                    "correction_direction": weights / masses,
                }
            )
        runtimes.append(
            PreparedPortRuntime(
                port=port,
                geometry=geometry,
                frequencies=frequency_tensor,
                field_name=field_name,
                yee_control_volume=(
                    control_volume if port.name in circuit_port_names else None
                ),
                lumped=lumped,
                wire_provider=wire_provider,
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
                substeps=substeps,
                substep_buffers=substep_buffers,
                exact_resistive=exact_resistive,
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
    prepare_circuit_runtimes(solver, solver._port_runtimes)
    for runtime in solver._port_runtimes:
        if runtime.circuit_port is not None:
            _validate_local_update_coefficient(
                solver,
                runtime.circuit_port.field,
                runtime.field_name,
            )
        runtime.yee_control_volume = None
    occupied_edges = {}
    coupled = [
        (runtime.port_name, field_name, runtime.linear_indices)
        for runtime, field_name in solver._lumped_element_runtimes
    ]
    coupled.extend(
        (runtime.port.name, runtime.field_name, runtime.lumped.linear_indices)
        for runtime in solver._port_runtimes
        if runtime.lumped is not None and runtime.wire_provider is None
    )
    coupled.extend(
        (
            runtime.port.name,
            runtime.field_name,
            runtime.circuit_port.field.linear_indices,
        )
        for runtime in solver._port_runtimes
        if runtime.circuit_port is not None
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
    for runtime in solver._port_runtimes:
        provider = runtime.wire_provider
        if provider is None:
            continue
        for component, offset in zip(
            provider.geometry.edge_components.detach().cpu().tolist(),
            provider.geometry.edge_offsets.detach().cpu().tolist(),
        ):
            key = (("Ex", "Ey", "Ez")[int(component)], int(offset))
            previous = occupied_edges.get(key)
            if previous is not None:
                raise ValueError(
                    f"Lumped objects {previous!r} and {runtime.port.name!r} overlap "
                    f"the same {key[0]} Yee edge."
                )
            occupied_edges[key] = runtime.port.name
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
        # Non-wire lumped ports sample voltage at the magnetic half-step (shared
        # with the current). Wire-bound ports and pure field ports keep the
        # electric full-step voltage stagger.
        voltage_at_magnetic = (
            runtime.lumped is not None and runtime.wire_provider is None
        )
        voltage_times = magnetic_times if voltage_at_magnetic else electric_times
        runtime.current_phase_weights = _weighted_phase_table(
            runtime.frequencies,
            magnetic_times,
            runtime.window_weights,
        )
        runtime.voltage_phase_weights = (
            runtime.current_phase_weights
            if voltage_at_magnetic
            else _weighted_phase_table(
                runtime.frequencies,
                voltage_times,
                runtime.window_weights,
            )
        )
        field_dtype = (
            getattr(solver, runtime.field_name).dtype
            if runtime.field_name is not None
            else solver.Ex.dtype
        )
        runtime.observer_current_buffer = torch.empty(
            (),
            device=solver.device,
            dtype=field_dtype,
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
        previous_voltage_name = lumped_state_name("port", index, "last_voltage_after")
        provider = port_runtime.wire_provider
        if provider is None:
            (
                corrected,
                next_inductor,
                next_capacitor,
                next_previous_voltage,
                trace,
            ) = replay_lumped_runtime(
                runtime,
                fields[port_runtime.field_name],
                previous_voltage=state[previous_voltage_name],
                inductor_current=state[inductor_name],
                capacitor_voltage=state[capacitor_name],
                drive=_drive_value(port_runtime, sample_time),
                field_name=port_runtime.field_name,
                kind="port",
                index=index,
            )
            fields[port_runtime.field_name] = corrected
        else:
            input_charge = next_auxiliary.get("wire_charge", state["wire_charge"])
            coordinate = _pack_wire_port_coordinate(
                provider,
                fields,
                input_charge,
            )
            drive = _drive_value(port_runtime, sample_time)
            if port_runtime.exact_resistive:
                buffers = port_runtime.substep_buffers
                flat = coordinate.reshape(-1)
                voltage_before = torch.dot(flat, runtime.voltage_weights)
                voltage_delta = voltage_before - drive
                integrated_current = (
                    voltage_delta * (1.0 - buffers["decay"]) / buffers["gamma"]
                )
                corrected = coordinate - (
                    buffers["correction_direction"] * integrated_current
                ).reshape_as(coordinate)
                average_current = integrated_current / buffers["macro_dt"]
                voltage_after = drive + buffers["decay"] * voltage_delta
                # The exact-resistive wire step carries the post-step port voltage
                # directly (no averaging recurrence), mirroring the forward
                # ``_apply_wire_resistive_exact`` scalar update.
                next_previous_voltage = voltage_after
                macro_trace = LumpedStepTrace(
                    runtime=runtime,
                    field_name="wire_port_coordinate",
                    kind="port",
                    index=index,
                    previous_voltage=state[previous_voltage_name],
                    voltage_before=voltage_before,
                    branch_current=average_current,
                    voltage_midpoint=voltage_after,
                    drive=drive,
                    old_inductor_current=state[inductor_name],
                    old_capacitor_voltage=state[capacitor_name],
                    edge_values=flat,
                )
                next_inductor = torch.zeros_like(state[inductor_name])
                next_capacitor = torch.zeros_like(state[capacitor_name])
                lumped_substeps = ()
            else:
                corrected = coordinate
                next_inductor = state[inductor_name]
                next_capacitor = state[capacitor_name]
                # Thread the port's carried post-step voltage through both
                # midpoint substeps so replay reconstructs ``last_voltage_after``
                # bit-exactly, matching the forward substep recurrence.
                next_previous_voltage = state[previous_voltage_name]
                lumped_substeps = []
                for _ in range(port_runtime.substeps):
                    (
                        corrected,
                        next_inductor,
                        next_capacitor,
                        next_previous_voltage,
                        lumped_trace,
                    ) = replay_lumped_runtime(
                        runtime,
                        corrected,
                        previous_voltage=next_previous_voltage,
                        inductor_current=next_inductor,
                        capacitor_voltage=next_capacitor,
                        drive=drive,
                        field_name="wire_port_coordinate",
                        kind="port",
                        index=index,
                    )
                    lumped_substeps.append(lumped_trace)
                macro_scale = 1.0 / float(port_runtime.substeps)
                macro_trace = replace(
                    lumped_substeps[-1],
                    voltage_before=lumped_substeps[0].voltage_before,
                    branch_current=sum(
                        item.branch_current for item in lumped_substeps
                    )
                    * macro_scale,
                    voltage_midpoint=torch.dot(
                        corrected.reshape(-1), runtime.voltage_weights
                    ),
                )
            fields, corrected_charge = _unpack_wire_port_coordinate(
                provider,
                corrected,
                fields,
                input_charge,
            )
            next_auxiliary["wire_charge"] = corrected_charge
            trace = WirePortStepTrace(
                lumped=macro_trace,
                lumped_substeps=tuple(lumped_substeps),
                provider=provider,
                input_charge=input_charge.index_select(0, provider.node_ids),
                output_coordinate=corrected,
                exact_resistive=port_runtime.exact_resistive,
            )
        next_auxiliary[inductor_name] = next_inductor
        next_auxiliary[capacitor_name] = next_capacitor
        next_auxiliary[previous_voltage_name] = next_previous_voltage
        traces.append(trace)
    for index, (runtime, field_name) in enumerate(
        getattr(solver, "_lumped_element_runtimes", ())
    ):
        inductor_name = lumped_state_name("element", index, "inductor_current")
        capacitor_name = lumped_state_name("element", index, "capacitor_voltage")
        previous_voltage_name = lumped_state_name("element", index, "last_voltage_after")
        corrected, next_inductor, next_capacitor, next_previous_voltage, trace = replay_lumped_runtime(
            runtime,
            fields[field_name],
            previous_voltage=state[previous_voltage_name],
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
        next_auxiliary[previous_voltage_name] = next_previous_voltage
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


def _pullback_wire_port_trace(
    trace: WirePortStepTrace,
    updated: dict[str, torch.Tensor],
    *,
    inductor_current_adjoint: torch.Tensor,
    capacitor_voltage_adjoint: torch.Tensor,
    previous_voltage_adjoint: torch.Tensor,
    voltage_seed: torch.Tensor,
    current_seed: torch.Tensor,
    eps_by_field: dict[str, torch.Tensor],
):
    lumped_trace = trace.lumped
    provider = trace.provider
    output_charge_adjoint = updated["wire_charge"].index_select(
        0, provider.node_ids
    )
    coordinate_adjoint = torch.cat(
        (
            _wire_port_field_values(
                {name: updated[name] for name in ("Ex", "Ey", "Ez")},
                provider.geometry,
            ),
            output_charge_adjoint * provider.node_capacitance,
        )
    ).reshape_as(provider.energy_masses)
    if trace.exact_resistive:
        runtime = lumped_trace.runtime
        with torch.enable_grad():
            coordinate_input = (
                lumped_trace.edge_values.detach().clone().requires_grad_(True)
            )
            masses = (
                provider.energy_masses.detach()
                .clone()
                .reshape(-1)
                .requires_grad_(True)
            )
            weights = (
                runtime.voltage_weights.detach().clone().requires_grad_(True)
            )
            resistance = runtime.resistance.detach().clone().requires_grad_(True)
            drive = lumped_trace.drive.detach().clone().requires_grad_(True)
            macro_dt = 2.0 * runtime.dt.detach()
            gamma = torch.sum(weights.square() / masses)
            decay = torch.exp(-macro_dt * gamma / resistance)
            voltage_before = torch.dot(coordinate_input, weights)
            integrated_current = (
                (voltage_before - drive) * (1.0 - decay) / gamma
            )
            corrected = coordinate_input - weights * integrated_current / masses
            average_current = integrated_current / macro_dt
            voltage_after = torch.dot(corrected, weights)
            # ``voltage_after`` is also carried out as the port's post-step
            # ``last_voltage_after`` scalar, so its cotangent enters the analytic
            # objective. The exact-resistive step does not read the incoming
            # ``previous_voltage`` (pure resistive analytic advance), so the
            # returned previous-voltage cotangent is zero.
            objective = torch.dot(
                corrected, coordinate_adjoint.reshape(-1)
            ) + voltage_after * (voltage_seed + previous_voltage_adjoint) - average_current * current_seed
            (
                pre_coordinate_adjoint,
                grad_eps_mass,
                grad_voltage_weights,
                grad_resistance,
                grad_drive,
            ) = torch.autograd.grad(
                objective,
                (coordinate_input, masses, weights, resistance, drive),
            )
        result = LumpedStepPullback(
            field_adjoint=pre_coordinate_adjoint.reshape_as(provider.energy_masses),
            previous_voltage_adjoint=torch.zeros_like(previous_voltage_adjoint),
            inductor_current_adjoint=torch.zeros_like(inductor_current_adjoint),
            capacitor_voltage_adjoint=torch.zeros_like(capacitor_voltage_adjoint),
            grad_eps=grad_eps_mass.reshape_as(provider.energy_masses),
            grad_resistance=grad_resistance,
            grad_inductance=torch.zeros_like(runtime.inductance),
            grad_capacitance=torch.zeros_like(runtime.capacitance),
            grad_drive=grad_drive,
            grad_voltage_weights=grad_voltage_weights,
        )
    else:
        substeps = trace.lumped_substeps
        if len(substeps) != 2:
            raise RuntimeError(
                "Wire-bound port reverse requires two recorded midpoint substeps."
            )
        sample_scale = 1.0 / float(len(substeps))
        runtime = substeps[0].runtime
        pre_coordinate_adjoint = coordinate_adjoint + (
            voltage_seed * runtime.voltage_weights
        ).reshape_as(coordinate_adjoint)
        pre_inductor_adjoint = inductor_current_adjoint
        pre_capacitor_adjoint = capacitor_voltage_adjoint
        # The macro post-step ``last_voltage_after`` scalar equals the second
        # substep's carried voltage, so its cotangent threads backward through
        # both substeps, mirroring the forward averaging recurrence.
        pre_previous_voltage_adjoint = previous_voltage_adjoint
        grad_eps_mass = torch.zeros_like(provider.energy_masses)
        grad_resistance = torch.zeros_like(runtime.resistance)
        grad_inductance = torch.zeros_like(runtime.inductance)
        grad_capacitance = torch.zeros_like(runtime.capacitance)
        grad_drive = torch.zeros_like(runtime.default_thevenin_voltage)
        grad_voltage_weights = (
            voltage_seed * trace.output_coordinate.reshape(-1)
        )
        for substep in reversed(substeps):
            substep_result = pullback_lumped_runtime(
                substep,
                pre_coordinate_adjoint,
                inductor_current_adjoint=pre_inductor_adjoint,
                capacitor_voltage_adjoint=pre_capacitor_adjoint,
                previous_voltage_adjoint=pre_previous_voltage_adjoint,
                voltage_sample_adjoint=torch.zeros_like(voltage_seed),
                network_current_sample_adjoint=current_seed * sample_scale,
                eps_edge=provider.energy_masses,
            )
            pre_coordinate_adjoint = substep_result.field_adjoint
            pre_inductor_adjoint = substep_result.inductor_current_adjoint
            pre_capacitor_adjoint = substep_result.capacitor_voltage_adjoint
            pre_previous_voltage_adjoint = substep_result.previous_voltage_adjoint
            grad_eps_mass = grad_eps_mass + substep_result.grad_eps
            grad_resistance = grad_resistance + substep_result.grad_resistance
            grad_inductance = grad_inductance + substep_result.grad_inductance
            grad_capacitance = grad_capacitance + substep_result.grad_capacitance
            grad_drive = grad_drive + substep_result.grad_drive
            grad_voltage_weights = (
                grad_voltage_weights + substep_result.grad_voltage_weights
            )
        result = LumpedStepPullback(
            field_adjoint=pre_coordinate_adjoint,
            previous_voltage_adjoint=pre_previous_voltage_adjoint,
            inductor_current_adjoint=pre_inductor_adjoint,
            capacitor_voltage_adjoint=pre_capacitor_adjoint,
            grad_eps=grad_eps_mass,
            grad_resistance=grad_resistance,
            grad_inductance=grad_inductance,
            grad_capacitance=grad_capacitance,
            grad_drive=grad_drive,
            grad_voltage_weights=grad_voltage_weights,
        )
    pre_coordinate_adjoint = result.field_adjoint.reshape(-1)
    next_updated = dict(updated)
    for component, name in enumerate(("Ex", "Ey", "Ez")):
        selected = torch.nonzero(
            provider.geometry.edge_components == component,
            as_tuple=False,
        ).reshape(-1)
        if selected.numel() == 0:
            continue
        field_adjoint = updated[name].reshape(-1).clone()
        field_adjoint.index_copy_(
            0,
            provider.geometry.edge_offsets.index_select(0, selected),
            pre_coordinate_adjoint.index_select(0, selected),
        )
        next_updated[name] = field_adjoint.reshape_as(updated[name])
    pre_charge_adjoint = updated["wire_charge"].clone()
    pre_node_coordinate_adjoint = pre_coordinate_adjoint[provider.field_count :]
    pre_charge_adjoint.index_copy_(
        0,
        provider.node_ids,
        pre_node_coordinate_adjoint / provider.node_capacitance,
    )
    next_updated["wire_charge"] = pre_charge_adjoint

    grad_eps = {name: torch.zeros_like(value) for name, value in eps_by_field.items()}
    field_mass_gradient = result.grad_eps.reshape(-1)[: provider.field_count]
    for component, name in enumerate(("Ex", "Ey", "Ez")):
        selected = torch.nonzero(
            provider.geometry.edge_components == component,
            as_tuple=False,
        ).reshape(-1)
        if selected.numel() == 0:
            continue
        grad_eps[name].reshape(-1).index_copy_(
            0,
            provider.geometry.edge_offsets.index_select(0, selected),
            field_mass_gradient.index_select(0, selected)
            * provider.field_control_volumes.index_select(0, selected),
        )

    output_node_coordinate = trace.output_coordinate.reshape(-1)[
        provider.field_count :
    ]
    grad_node_capacitance = result.grad_eps.reshape(-1)[provider.field_count :]
    grad_node_capacitance = (
        grad_node_capacitance
        + output_charge_adjoint * output_node_coordinate
        - pre_node_coordinate_adjoint
        * trace.input_charge
        / provider.node_capacitance.square()
    )
    return (
        result,
        next_updated,
        grad_eps,
        grad_node_capacitance,
        result.grad_voltage_weights.reshape(-1)[: provider.field_count],
    )


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
    wire_runtime = getattr(solver, "_wire_runtime", None)
    grad_wire_capacitance = (
        torch.zeros_like(wire_runtime.coefficients["node_capacitance"])
        if wire_runtime is not None
        else None
    )
    grad_wire_gap_weights = (
        torch.zeros_like(
            wire_runtime.network.port_gap_weights,
            device=solver.device,
            dtype=solver.Ex.dtype,
        )
        if wire_runtime is not None
        else None
    )
    semantic_grads = {}
    sample_time = torch.as_tensor(
        time_value + 0.5 * float(solver.dt),
        device=solver.device,
        dtype=solver.Ex.dtype,
    )
    for trace in reversed(tuple(traces)):
        lumped_trace = trace.lumped if isinstance(trace, WirePortStepTrace) else trace
        inductor_name = lumped_state_name(
            lumped_trace.kind, lumped_trace.index, "inductor_current"
        )
        capacitor_name = lumped_state_name(
            lumped_trace.kind, lumped_trace.index, "capacitor_voltage"
        )
        previous_voltage_name = lumped_state_name(
            lumped_trace.kind, lumped_trace.index, "last_voltage_after"
        )
        voltage_seed, current_seed, direct_drive_seed = port_sample_adjoints.get(
            lumped_trace.index,
            (
                torch.zeros_like(lumped_trace.branch_current),
                torch.zeros_like(lumped_trace.branch_current),
                torch.zeros_like(lumped_trace.branch_current),
            ),
        ) if lumped_trace.kind == "port" else (
            torch.zeros_like(lumped_trace.branch_current),
            torch.zeros_like(lumped_trace.branch_current),
            torch.zeros_like(lumped_trace.branch_current),
        )
        voltage_seed = voltage_seed.to(
            device=lumped_trace.branch_current.device,
            dtype=lumped_trace.branch_current.dtype,
        )
        current_seed = current_seed.to(
            device=lumped_trace.branch_current.device,
            dtype=lumped_trace.branch_current.dtype,
        )
        direct_drive_seed = direct_drive_seed.to(
            device=lumped_trace.branch_current.device,
            dtype=lumped_trace.branch_current.dtype,
        )
        if isinstance(trace, WirePortStepTrace):
            (
                result,
                updated,
                local_grad_eps,
                local_grad_capacitance,
                local_grad_gap_weights,
            ) = _pullback_wire_port_trace(
                    trace,
                    updated,
                    inductor_current_adjoint=updated[inductor_name],
                    capacitor_voltage_adjoint=updated[capacitor_name],
                    previous_voltage_adjoint=updated[previous_voltage_name],
                    voltage_seed=voltage_seed,
                    current_seed=current_seed,
                    eps_by_field=eps_by_field,
                )
            for name in ("Ex", "Ey", "Ez"):
                grad_eps[name] = grad_eps[name] + local_grad_eps[name]
            grad_wire_capacitance.index_add_(
                0,
                trace.provider.node_ids,
                local_grad_capacitance,
            )
            grad_wire_gap_weights.index_add_(
                0,
                trace.provider.gap_weight_indices,
                local_grad_gap_weights,
            )
        else:
            result = pullback_lumped_runtime(
                trace,
                updated[trace.field_name],
                inductor_current_adjoint=updated[inductor_name],
                capacitor_voltage_adjoint=updated[capacitor_name],
                previous_voltage_adjoint=updated[previous_voltage_name],
                voltage_sample_adjoint=voltage_seed,
                network_current_sample_adjoint=current_seed,
                eps_edge=eps_by_field[trace.field_name],
            )
            updated[trace.field_name] = result.field_adjoint
            grad_eps[trace.field_name] = grad_eps[trace.field_name] + result.grad_eps
        updated[inductor_name] = result.inductor_current_adjoint
        updated[capacitor_name] = result.capacitor_voltage_adjoint
        updated[previous_voltage_name] = result.previous_voltage_adjoint
        if lumped_trace.kind == "port":
            port_runtime = solver._port_runtimes[lumped_trace.index]
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
            element = solver.scene.lumped_elements[lumped_trace.index]
            gradient = {
                "resistor": result.grad_resistance,
                "inductor": result.grad_inductance,
                "capacitor": result.grad_capacitance,
            }[element.kind]
            key = ("element", element.name, "value")
            semantic_grads[key] = semantic_grads.get(key, torch.zeros_like(gradient)) + gradient
    return (
        updated,
        grad_eps,
        semantic_grads,
        grad_wire_capacitance,
        grad_wire_gap_weights,
    )


_WIRE_PORT_AVERAGE_DIAGNOSTICS = (
    "voltage_midpoint",
    "model_voltage_midpoint",
    "branch_current",
    "resistor_current",
    "capacitor_current",
    "inductor_current_midpoint",
)
_WIRE_PORT_SUM_DIAGNOSTICS = (
    "dissipated_energy",
    "stored_energy_change",
    "source_work",
    "field_energy_change",
)


def _apply_wire_resistive_exact(solver, runtime: PreparedPortRuntime) -> None:
    provider = runtime.wire_provider
    lumped = runtime.lumped
    buffers = runtime.substep_buffers
    if provider is None or lumped is None or buffers is None:
        raise RuntimeError("Exact wire-port resistance requires a prepared provider.")
    fields = _field_map(solver)
    coordinate = _pack_wire_port_coordinate(
        provider,
        fields,
        solver._wire_runtime.charge,
        out=provider.coordinate,
    )
    flat = coordinate.reshape(-1)
    voltage_before = torch.dot(flat, lumped.voltage_weights)
    voltage_delta = voltage_before - runtime.drive_buffer
    integrated_current = (
        voltage_delta * (1.0 - buffers["decay"]) / buffers["gamma"]
    )
    flat.addcmul_(
        buffers["correction_direction"], integrated_current, value=-1.0
    )
    average_current = integrated_current / buffers["macro_dt"]
    average_voltage = runtime.drive_buffer + lumped.resistance * average_current
    voltage_after = runtime.drive_buffer + buffers["decay"] * voltage_delta
    energy_voltage = 0.5 * (voltage_before + voltage_after)
    dissipated = (
        voltage_delta.square()
        * (1.0 - buffers["decay"].square())
        / (2.0 * buffers["gamma"])
    )
    source_work = -runtime.drive_buffer * integrated_current

    lumped.last_voltage_before.copy_(voltage_before)
    lumped.last_voltage_midpoint.copy_(energy_voltage)
    lumped.last_voltage_after.copy_(voltage_after)
    lumped.last_model_voltage_midpoint.copy_(average_voltage - runtime.drive_buffer)
    lumped.last_branch_current.copy_(average_current)
    lumped.last_resistor_current.copy_(average_current)
    lumped.last_capacitor_current.zero_()
    lumped.last_inductor_current_midpoint.zero_()
    lumped.last_dissipated_energy.copy_(dissipated)
    lumped.last_stored_energy_change.zero_()
    lumped.last_source_work.copy_(source_work)
    lumped.last_field_energy_change.copy_(source_work - dissipated)
    _unpack_wire_port_coordinate_in_place(
        provider,
        coordinate,
        fields,
        solver._wire_runtime.charge,
    )


def _apply_wire_port_substeps(solver, runtime: PreparedPortRuntime) -> None:
    provider = runtime.wire_provider
    lumped = runtime.lumped
    buffers = runtime.substep_buffers
    if provider is None or lumped is None or runtime.substeps != 2 or buffers is None:
        raise RuntimeError("Wire-bound lumped ports require two prepared midpoint substeps.")
    fields = _field_map(solver)
    _pack_wire_port_coordinate(
        provider,
        fields,
        solver._wire_runtime.charge,
        out=provider.coordinate,
    )
    apply_lumped_runtime(
        lumped,
        provider.coordinate,
        thevenin_voltage=runtime.drive_buffer,
    )
    for name in _WIRE_PORT_AVERAGE_DIAGNOSTICS + _WIRE_PORT_SUM_DIAGNOSTICS:
        buffers[name].copy_(getattr(lumped, f"last_{name}"))
    buffers["voltage_before"].copy_(lumped.last_voltage_before)
    apply_lumped_runtime(
        lumped,
        provider.coordinate,
        thevenin_voltage=runtime.drive_buffer,
    )
    for name in _WIRE_PORT_AVERAGE_DIAGNOSTICS:
        value = getattr(lumped, f"last_{name}")
        value.add_(buffers[name]).mul_(0.5)
    for name in _WIRE_PORT_SUM_DIAGNOSTICS:
        getattr(lumped, f"last_{name}").add_(buffers[name])
    lumped.last_voltage_before.copy_(buffers["voltage_before"])
    _unpack_wire_port_coordinate_in_place(
        provider,
        provider.coordinate,
        fields,
        solver._wire_runtime.charge,
    )


def apply_port_runtimes(solver) -> None:
    """Apply prepared source/load corrections without host-device transfers."""

    for runtime in getattr(solver, "_port_runtimes", ()):
        if runtime.lumped is None or runtime.embedded_network_name is not None:
            continue
        if runtime.source_kind is None and runtime.wire_provider is None:
            # Passive Yee-field termination: the drive is identically the prepared
            # default (zero). Skip the per-step waveform evaluation and its
            # allocation. Wire-bound ports have no Yee field component to read
            # (field_name is None) and must take the wire substep path below.
            apply_lumped_runtime(
                runtime.lumped,
                getattr(solver, runtime.field_name),
            )
            continue
        drive = _evaluate_drive(runtime)
        if runtime.wire_provider is None:
            apply_lumped_runtime(
                runtime.lumped,
                getattr(solver, runtime.field_name),
                thevenin_voltage=drive,
            )
            continue
        if runtime.exact_resistive:
            _apply_wire_resistive_exact(solver, runtime)
        else:
            _apply_wire_port_substeps(solver, runtime)
    for runtime, field_name in getattr(solver, "_lumped_element_runtimes", ()):
        apply_lumped_runtime(runtime, getattr(solver, field_name))
    if getattr(solver, "_circuit_runtimes", ()):
        apply_circuit_runtimes(solver)


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
        if runtime.circuit_port is not None:
            voltage = runtime.circuit_port.field.last_voltage
            current = -runtime.circuit_port.field.last_current
            # Circuit-bound ports sample voltage and current at the same instant,
            # set by the companion integration rule: trapezoidal couples at the
            # magnetic half-step, backward Euler at the electric full step. Reuse
            # the matching precomputed phase table for both terms so the DFT
            # stagger follows the coupling point instead of the default E/H split.
            if runtime.circuit_port.last_integration == "trapezoidal":
                voltage_kernel = runtime.current_phase_weights[runtime.sample_index]
            else:
                voltage_kernel = runtime.voltage_phase_weights[runtime.sample_index]
            current_kernel = voltage_kernel
        elif runtime.lumped is not None and runtime.wire_provider is not None:
            # The wire terminal coordinate is electric-time data, while its branch
            # current is the conjugate half-step current. The precomputed phase
            # tables already carry the wire electric-time voltage stagger.
            if fields is None:
                fields = _field_map(solver)
            provider = runtime.wire_provider
            coordinate = _pack_wire_port_coordinate(
                provider,
                fields,
                solver._wire_runtime.charge,
                out=provider.coordinate,
            )
            voltage = torch.dot(coordinate.reshape(-1), provider.voltage_weights)
            current = -runtime.lumped.last_branch_current
            voltage_kernel = runtime.voltage_phase_weights[runtime.sample_index]
            current_kernel = runtime.current_phase_weights[runtime.sample_index]
        elif runtime.lumped is not None:
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
            voltage_kernel = runtime.voltage_phase_weights[runtime.sample_index]
            current_kernel = runtime.current_phase_weights[runtime.sample_index]
        elif runtime.wire_provider is not None:
            if fields is None:
                fields = _field_map(solver)
            provider = runtime.wire_provider
            coordinate = _pack_wire_port_coordinate(
                provider,
                fields,
                solver._wire_runtime.charge,
                out=provider.coordinate,
            )
            voltage = torch.dot(
                coordinate.reshape(-1),
                provider.voltage_weights,
            )
            current = torch.zeros_like(voltage)
            voltage_kernel = runtime.voltage_phase_weights[runtime.sample_index]
            current_kernel = runtime.current_phase_weights[runtime.sample_index]
        else:
            if fields is None:
                fields = _field_map(solver)
            voltage = runtime.geometry.integrate_voltage(fields)
            current = runtime.geometry.integrate_current(fields)
            voltage_kernel = runtime.voltage_phase_weights[runtime.sample_index]
            current_kernel = runtime.current_phase_weights[runtime.sample_index]
        weight = runtime.window_weights[runtime.sample_index]
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
        # Observer sample times advance by the outer macro step. For wire-bound
        # ports the lumped runtime runs at a fractional substep (lumped.dt =
        # macro_dt / substeps), so the macro advance is lumped.dt * substeps;
        # for plain lumped and non-lumped ports this reduces to the single-step
        # value. The substeps == 1 case reuses the existing lumped.dt tensor so
        # the embedded-network observer hot path stays allocation-free.
        if runtime.lumped is None:
            macro_step = solver.dt
        elif runtime.substeps == 1:
            macro_step = runtime.lumped.dt
        else:
            macro_step = runtime.lumped.dt * runtime.substeps
        runtime.electric_time.add_(macro_step)
        runtime.magnetic_time.add_(macro_step)


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
            direction=(
                "+"
                if runtime.wire_provider is not None
                or runtime.geometry.direction > 0
                else "-"
            ),
            reference_plane=runtime.port.reference_plane,
            available_power=available_power,
            metadata={
                "axis": getattr(runtime.geometry, "axis", None),
                "orientation": getattr(runtime.geometry, "direction", 1),
                "provider": (
                    f"wire_{runtime.geometry.binding_kind}"
                    if runtime.wire_provider is not None
                    else "yee_contour"
                ),
                "wire_binding": (
                    dict(runtime.geometry.metadata or {})
                    if runtime.wire_provider is not None
                    else None
                ),
                "current_convention": (
                    "entering_wire_network"
                    if runtime.wire_provider is not None
                    else "entering_field_network"
                    if runtime.lumped is not None or runtime.circuit_port is not None
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

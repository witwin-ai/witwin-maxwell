from __future__ import annotations

from dataclasses import dataclass

import torch

from ..compiler.networks import CompiledNetworkBlock
from ..network import EmbeddedNetworkData
from .delay import (
    PreparedBidirectionalDelay,
    prepare_bidirectional_delay,
    read_bidirectional_delay,
    write_bidirectional_delay,
)


@dataclass
class PreparedNetworkRuntime:
    """Fixed-shape N-port state-space feedback on one FDTD device."""

    compiled: CompiledNetworkBlock
    port_runtimes: tuple[object, ...]
    electric_fields: tuple[torch.Tensor, ...]
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    state: torch.Tensor
    next_state: torch.Tensor
    state_drive: torch.Tensor
    free_voltage: torch.Tensor
    network_voltage: torch.Tensor
    branch_current: torch.Tensor
    output_buffer: torch.Tensor
    direct_drive: torch.Tensor
    delay_runtime: PreparedBidirectionalDelay | None
    reference_impedance: torch.Tensor
    sqrt_reference_impedance: torch.Tensor
    terminal_denominator: torch.Tensor
    wave_alpha: torch.Tensor
    wave_beta: torch.Tensor
    zero_indices: torch.Tensor
    zero_beta: torch.Tensor
    zero_rhs: torch.Tensor
    zero_solution: torch.Tensor
    zero_reflected: torch.Tensor
    core_incident: torch.Tensor
    core_reflected: torch.Tensor
    scene_incident: torch.Tensor
    scene_reflected: torch.Tensor
    loop_lu: torch.Tensor
    loop_pivots: torch.Tensor
    loop_permutation: torch.Tensor
    solve_workspace: torch.Tensor
    solve_scalar: torch.Tensor
    loop_denominator: torch.Tensor
    feedback_impedance: torch.Tensor
    loop_condition: float
    net_power: torch.Tensor
    power_buffer: torch.Tensor
    port_energy: torch.Tensor
    absorbed_increment: torch.Tensor
    generated_increment: torch.Tensor
    absorbed_energy: torch.Tensor
    generated_energy: torch.Tensor
    runtime_warnings: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.compiled.name

    @property
    def port_runtime(self):
        """Return the sole port runtime for one-port internal callers."""

        if len(self.port_runtimes) != 1:
            raise AttributeError("port_runtime is only defined for one-port networks.")
        return self.port_runtimes[0]

    @property
    def electric_field(self) -> torch.Tensor:
        """Return the sole field tensor for one-port internal callers."""

        if len(self.electric_fields) != 1:
            raise AttributeError("electric_field is only defined for one-port networks.")
        return self.electric_fields[0]


def _runtime_matrix(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).contiguous()


def _matvec_out(matrix: torch.Tensor, vector: torch.Tensor, output: torch.Tensor) -> None:
    """Matrix-vector product without the temporary allocated by CUDA ``mv``."""

    torch.mm(matrix, vector.unsqueeze(1), out=output.unsqueeze(1))


def _lu_solve_out(
    lu: torch.Tensor,
    permutation: torch.Tensor,
    rhs: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    scalar: torch.Tensor,
) -> None:
    """Apply a prepared pivoted LU solve with fixed scratch storage."""

    torch.index_select(rhs, 0, permutation, out=output)
    size = output.numel()
    for row in range(size):
        if row:
            torch.mul(lu[row, :row], output[:row], out=workspace[:row])
            torch.sum(workspace[:row], dim=0, out=scalar)
            output[row].sub_(scalar)
    for row in range(size - 1, -1, -1):
        if row + 1 < size:
            width = size - row - 1
            torch.mul(
                lu[row, row + 1 :],
                output[row + 1 :],
                out=workspace[:width],
            )
            torch.sum(workspace[:width], dim=0, out=scalar)
            output[row].sub_(scalar)
        output[row].div_(lu[row, row])


def _effective_source_band(source_time: dict) -> tuple[float, float]:
    kind = str(source_time["kind"])
    frequency = float(source_time["frequency"])
    if kind == "cw":
        return frequency, frequency
    if kind == "gaussian_pulse":
        width = float(source_time["fwidth"])
        return max(0.0, frequency - 3.0 * width), frequency + 3.0 * width
    if kind == "ricker_wavelet":
        return 0.0, 3.0 * frequency
    raise ValueError(
        "Embedded networks with extrapolation='reject' require an analytic "
        "CW, GaussianPulse, or RickerWavelet excitation with a finite effective band."
    )


def _solver_excitation_bands(solver) -> tuple[tuple[float, float], ...]:
    bands = [
        _effective_source_band(source["source_time"])
        for source in getattr(solver, "_compiled_sources", ())
        if source.get("source_time") is not None
    ]
    for runtime in getattr(solver, "_port_runtimes", ()):
        if getattr(runtime, "excitation", None) is None:
            continue
        bands.append(
            _effective_source_band(
                {
                    "kind": runtime.source_kind,
                    "frequency": runtime.source_frequency,
                    "fwidth": runtime.source_fwidth,
                }
            )
        )
    return tuple(bands)


def prepare_network_runtimes(solver) -> tuple[PreparedNetworkRuntime, ...]:
    """Compile embedded networks once and allocate all hot-path tensors."""

    compiled_networks = solver.scene.compile_networks(dt=solver.dt, device=solver.device)
    port_runtimes = {
        runtime.port.name: runtime
        for runtime in getattr(solver, "_port_runtimes", ())
    }
    prepared: list[PreparedNetworkRuntime] = []
    for compiled in compiled_networks:
        if compiled.extrapolation == "reject":
            lower, upper = compiled.frequency_band
            requested = tuple(getattr(solver, "_requested_port_frequencies", ()))
            outside = tuple(
                float(frequency)
                for frequency in requested
                if float(frequency) < lower or float(frequency) > upper
            )
            if outside:
                raise ValueError(
                    f"Embedded network {compiled.name!r} rejects requested frequencies "
                    f"{outside!r} outside its fitted band [{lower}, {upper}] Hz."
                )
            outside_bands = tuple(
                band
                for band in _solver_excitation_bands(solver)
                if band[0] < lower or band[1] > upper
            )
            if outside_bands:
                raise ValueError(
                    f"Embedded network {compiled.name!r} rejects excitation effective "
                    f"bands {outside_bands!r} outside its fitted band "
                    f"[{lower}, {upper}] Hz."
                )
        connected_runtimes: list[object] = []
        electric_fields: list[torch.Tensor] = []
        for port_name in compiled.connection_names:
            try:
                port_runtime = port_runtimes[port_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Compiled network {compiled.name!r} has no prepared port {port_name!r}."
                ) from exc
            lumped = port_runtime.lumped
            if lumped is None or port_runtime.embedded_network_name != compiled.name:
                raise RuntimeError(
                    f"Port {port_name!r} was not prepared for embedded network "
                    f"{compiled.name!r}."
                )
            connected_runtimes.append(port_runtime)
            electric_fields.append(getattr(solver, port_runtime.field_name))

        if not connected_runtimes:
            raise RuntimeError(f"Embedded network {compiled.name!r} has no connected ports.")
        first_lumped = connected_runtimes[0].lumped
        discrete = compiled.discrete
        matrices = (discrete.A, discrete.B, discrete.C, discrete.D)
        if any(matrix.requires_grad for matrix in matrices):
            raise NotImplementedError(
                "Trainable embedded-network state-space coefficients require the Phase 4 "
                "network adjoint; the fixed runtime accepts fixed coefficients only."
            )
        dtype = first_lumped.field_dtype
        device = first_lumped.linear_indices.device
        for port_runtime in connected_runtimes[1:]:
            lumped = port_runtime.lumped
            if lumped.field_dtype != dtype or lumped.linear_indices.device != device:
                raise ValueError(
                    "All ports of an embedded network must share one device and field dtype."
                )
        A, B, C, D = (
            _runtime_matrix(matrix, device=device, dtype=dtype)
            for matrix in matrices
        )
        port_count = len(connected_runtimes)
        state_count = A.shape[0]
        if (
            A.shape != (state_count, state_count)
            or B.shape != (state_count, port_count)
            or C.shape != (port_count, state_count)
            or D.shape != (port_count, port_count)
        ):
            raise ValueError(
                "Embedded-network matrices must match the compiled state and port counts."
            )
        feedback_impedance = torch.stack(
            tuple(runtime.lumped.discrete_port_impedance for runtime in connected_runtimes)
        ).contiguous()
        delay_runtime: PreparedBidirectionalDelay | None = None
        empty_vector = torch.empty((0,), device=device, dtype=dtype)
        empty_indices = torch.empty((0,), device=device, dtype=torch.int64)
        reference_impedance = empty_vector
        sqrt_reference_impedance = empty_vector
        terminal_denominator = empty_vector
        wave_alpha = empty_vector
        wave_beta = empty_vector
        zero_indices = empty_indices
        zero_beta = empty_vector
        zero_rhs = empty_vector
        zero_solution = empty_vector
        zero_reflected = empty_vector
        if compiled.delay is None:
            loop_denominator = torch.eye(port_count, device=device, dtype=dtype)
            loop_denominator.add_(D * feedback_impedance.unsqueeze(0))
        else:
            if compiled.reference_impedance is None:
                raise RuntimeError("A delayed network is missing its reference impedance.")
            delay_runtime = prepare_bidirectional_delay(
                compiled.delay.delay_seconds,
                dt=solver.dt,
                max_delay_steps=max(1, compiled.delay.buffer_steps),
                device=device,
                dtype=dtype,
            )
            reference_impedance = compiled.reference_impedance.to(
                device=device, dtype=dtype
            ).contiguous()
            sqrt_reference_impedance = torch.sqrt(reference_impedance)
            terminal_denominator = reference_impedance + feedback_impedance
            wave_alpha = terminal_denominator / sqrt_reference_impedance
            wave_beta = (
                reference_impedance - feedback_impedance
            ) / sqrt_reference_impedance
            zero_indices = torch.tensor(
                tuple(
                    index
                    for index, value in enumerate(compiled.delay.delay_seconds)
                    if value == 0.0
                ),
                device=device,
                dtype=torch.int64,
            )
            if zero_indices.numel():
                zero_alpha = wave_alpha.index_select(0, zero_indices)
                zero_beta = wave_beta.index_select(0, zero_indices).contiguous()
                zero_direct = D.index_select(0, zero_indices).index_select(
                    1, zero_indices
                )
                loop_denominator = torch.diag(zero_alpha)
                loop_denominator.add_(zero_beta.unsqueeze(1) * zero_direct)
                zero_count = zero_indices.numel()
                zero_rhs = torch.empty((zero_count,), device=device, dtype=dtype)
                zero_solution = torch.empty_like(zero_rhs)
                zero_reflected = torch.empty_like(zero_rhs)
            else:
                loop_denominator = torch.eye(1, device=device, dtype=dtype)
        singular_values = torch.linalg.svdvals(loop_denominator)
        largest = float(torch.max(singular_values).item())
        smallest = float(torch.min(singular_values).item())
        threshold = 64.0 * torch.finfo(dtype).eps * max(1.0, largest)
        if smallest <= threshold:
            raise ValueError(
                f"Embedded network {compiled.name!r} has a singular same-step direct loop."
            )
        loop_lu, loop_pivots, factor_info = torch.linalg.lu_factor_ex(
            loop_denominator,
            check_errors=False,
        )
        if int(factor_info.item()) != 0:
            raise ValueError(
                f"Embedded network {compiled.name!r} has a singular same-step direct loop."
            )
        permutation_values = list(range(loop_denominator.shape[0]))
        for row, pivot in enumerate(loop_pivots.detach().cpu().tolist()):
            pivot_row = int(pivot) - 1
            permutation_values[row], permutation_values[pivot_row] = (
                permutation_values[pivot_row],
                permutation_values[row],
            )
        loop_permutation = torch.tensor(
            permutation_values,
            device=device,
            dtype=torch.int64,
        )
        solve_workspace = torch.empty(
            (loop_denominator.shape[0],),
            device=device,
            dtype=dtype,
        )
        solve_scalar = torch.empty((), device=device, dtype=dtype)
        state = torch.zeros((A.shape[0],), device=device, dtype=dtype)
        port_zeros = [
            torch.zeros((port_count,), device=device, dtype=dtype)
            for _ in range(2)
        ]
        scalar_zeros = [torch.zeros((), device=device, dtype=dtype) for _ in range(5)]
        warnings: list[str] = []
        if compiled.fit_report is not None and compiled.fit_report.passivity_margin is None:
            warnings.append("The fitted model has no recorded passivity margin.")
        prepared.append(
            PreparedNetworkRuntime(
                compiled=compiled,
                port_runtimes=tuple(connected_runtimes),
                electric_fields=tuple(electric_fields),
                A=A,
                B=B,
                C=C,
                D=D,
                state=state,
                next_state=torch.empty_like(state),
                state_drive=torch.empty_like(state),
                free_voltage=torch.empty((port_count,), device=device, dtype=dtype),
                network_voltage=torch.empty((port_count,), device=device, dtype=dtype),
                branch_current=torch.empty((port_count,), device=device, dtype=dtype),
                output_buffer=torch.empty((port_count,), device=device, dtype=dtype),
                direct_drive=torch.empty((port_count,), device=device, dtype=dtype),
                delay_runtime=delay_runtime,
                reference_impedance=reference_impedance,
                sqrt_reference_impedance=sqrt_reference_impedance,
                terminal_denominator=terminal_denominator,
                wave_alpha=wave_alpha,
                wave_beta=wave_beta,
                zero_indices=zero_indices,
                zero_beta=zero_beta,
                zero_rhs=zero_rhs,
                zero_solution=zero_solution,
                zero_reflected=zero_reflected,
                core_incident=torch.empty((port_count,), device=device, dtype=dtype),
                core_reflected=torch.empty((port_count,), device=device, dtype=dtype),
                scene_incident=torch.empty((port_count,), device=device, dtype=dtype),
                scene_reflected=torch.empty((port_count,), device=device, dtype=dtype),
                loop_lu=loop_lu.contiguous(),
                loop_pivots=loop_pivots.contiguous(),
                loop_permutation=loop_permutation,
                solve_workspace=solve_workspace,
                solve_scalar=solve_scalar,
                loop_denominator=loop_denominator,
                feedback_impedance=feedback_impedance,
                loop_condition=largest / smallest,
                net_power=scalar_zeros[0],
                power_buffer=port_zeros[0],
                port_energy=port_zeros[1],
                absorbed_increment=scalar_zeros[1],
                generated_increment=scalar_zeros[2],
                absorbed_energy=scalar_zeros[3],
                generated_energy=scalar_zeros[4],
                runtime_warnings=tuple(warnings),
            )
        )
    solver._network_runtimes = tuple(prepared)
    solver._network_cuda_graph_active = False
    return solver._network_runtimes


def apply_network_runtime(runtime: PreparedNetworkRuntime) -> None:
    """Advance one implicit N-port load and correct its Yee fields in place."""

    for index, (port_runtime, electric_field) in enumerate(
        zip(runtime.port_runtimes, runtime.electric_fields)
    ):
        lumped = port_runtime.lumped
        flat_field = electric_field.view(-1)
        torch.index_select(
            flat_field,
            0,
            lumped.linear_indices,
            out=lumped.edge_buffer,
        )
        torch.mul(
            lumped.edge_buffer,
            lumped.voltage_weights,
            out=lumped.edge_buffer,
        )
        torch.sum(lumped.edge_buffer, dim=0, out=lumped.last_voltage_before)
        runtime.free_voltage[index].copy_(lumped.last_voltage_before)

    if runtime.delay_runtime is None:
        _matvec_out(runtime.C, runtime.state, runtime.output_buffer)
        _matvec_out(runtime.D, runtime.free_voltage, runtime.direct_drive)
        runtime.output_buffer.add_(runtime.direct_drive)
        _lu_solve_out(
            runtime.loop_lu,
            runtime.loop_permutation,
            runtime.output_buffer,
            runtime.branch_current,
            runtime.solve_workspace,
            runtime.solve_scalar,
        )
        runtime.network_voltage.copy_(runtime.free_voltage)
        runtime.network_voltage.addcmul_(
            runtime.feedback_impedance,
            runtime.branch_current,
            value=-1.0,
        )
        state_input = runtime.network_voltage
    else:
        read_bidirectional_delay(
            runtime.delay_runtime,
            runtime.core_incident,
            runtime.scene_reflected,
        )
        _matvec_out(runtime.C, runtime.state, runtime.core_reflected)
        _matvec_out(runtime.D, runtime.core_incident, runtime.direct_drive)
        runtime.core_reflected.add_(runtime.direct_drive)
        if runtime.zero_indices.numel():
            torch.index_select(
                runtime.free_voltage,
                0,
                runtime.zero_indices,
                out=runtime.zero_rhs,
            )
            torch.index_select(
                runtime.core_reflected,
                0,
                runtime.zero_indices,
                out=runtime.zero_reflected,
            )
            runtime.zero_reflected.mul_(runtime.zero_beta)
            runtime.zero_rhs.sub_(runtime.zero_reflected)
            _lu_solve_out(
                runtime.loop_lu,
                runtime.loop_permutation,
                runtime.zero_rhs,
                runtime.zero_solution,
                runtime.solve_workspace,
                runtime.solve_scalar,
            )
            runtime.core_incident.index_copy_(
                0,
                runtime.zero_indices,
                runtime.zero_solution,
            )
            _matvec_out(runtime.D, runtime.core_incident, runtime.direct_drive)
            _matvec_out(runtime.C, runtime.state, runtime.core_reflected)
            runtime.core_reflected.add_(runtime.direct_drive)
            torch.index_select(
                runtime.core_reflected,
                0,
                runtime.zero_indices,
                out=runtime.zero_reflected,
            )
            runtime.scene_reflected.index_copy_(
                0,
                runtime.zero_indices,
                runtime.zero_reflected,
            )

        torch.mul(
            runtime.sqrt_reference_impedance,
            runtime.scene_reflected,
            out=runtime.direct_drive,
        )
        runtime.direct_drive.mul_(2.0)
        runtime.branch_current.copy_(runtime.free_voltage)
        runtime.branch_current.sub_(runtime.direct_drive)
        runtime.branch_current.div_(runtime.terminal_denominator)
        runtime.network_voltage.copy_(runtime.free_voltage)
        runtime.network_voltage.addcmul_(
            runtime.feedback_impedance,
            runtime.branch_current,
            value=-1.0,
        )
        torch.mul(
            runtime.sqrt_reference_impedance,
            runtime.branch_current,
            out=runtime.scene_incident,
        )
        runtime.scene_incident.add_(runtime.scene_reflected)
        write_bidirectional_delay(
            runtime.delay_runtime,
            runtime.scene_incident,
            runtime.core_reflected,
        )
        state_input = runtime.core_incident

    for index, (port_runtime, electric_field) in enumerate(
        zip(runtime.port_runtimes, runtime.electric_fields)
    ):
        lumped = port_runtime.lumped
        flat_field = electric_field.view(-1)
        lumped.last_branch_current.copy_(runtime.branch_current[index])
        torch.mul(
            lumped.injection,
            lumped.last_branch_current,
            out=lumped.correction_buffer,
        )
        flat_field.index_add_(
            0,
            lumped.linear_indices,
            lumped.correction_buffer,
            alpha=-1.0,
        )
        lumped.last_voltage_midpoint.copy_(runtime.network_voltage[index])
        lumped.last_voltage_after.copy_(lumped.last_voltage_before)
        lumped.last_voltage_after.addcmul_(
            lumped.coupling_impedance,
            lumped.last_branch_current,
            value=-1.0,
        )
        lumped.last_model_voltage_midpoint.copy_(lumped.last_voltage_midpoint)

    _matvec_out(runtime.A, runtime.state, runtime.next_state)
    _matvec_out(
        runtime.B,
        state_input,
        runtime.state_drive,
    )
    runtime.next_state.add_(runtime.state_drive)
    runtime.state.copy_(runtime.next_state)

    torch.mul(
        runtime.network_voltage,
        runtime.branch_current,
        out=runtime.power_buffer,
    )
    runtime.power_buffer.mul_(runtime.port_runtimes[0].lumped.dt)
    runtime.port_energy.add_(runtime.power_buffer)
    torch.sum(runtime.power_buffer, dim=0, out=runtime.net_power)
    torch.clamp(runtime.net_power, min=0.0, out=runtime.absorbed_increment)
    torch.neg(runtime.net_power, out=runtime.generated_increment)
    torch.clamp(runtime.generated_increment, min=0.0, out=runtime.generated_increment)
    runtime.absorbed_energy.add_(runtime.absorbed_increment)
    runtime.generated_energy.add_(runtime.generated_increment)


def apply_network_runtimes(solver) -> None:
    for runtime in getattr(solver, "_network_runtimes", ()):
        apply_network_runtime(runtime)


def make_network_runner(solver, *, use_cuda_graph: bool):
    """Capture only the post-source network feedback block when requested."""

    runtimes = getattr(solver, "_network_runtimes", ())
    solver._network_cuda_graph_active = False
    if not runtimes:
        return lambda: None

    def normal():
        apply_network_runtimes(solver)

    if not use_cuda_graph or solver.device.type != "cuda":
        return normal
    from .cuda.runtime.graph import CudaGraphRunner

    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for runtime in runtimes:
        mutated = [
            *runtime.electric_fields,
            runtime.state,
            runtime.next_state,
            runtime.state_drive,
            runtime.free_voltage,
            runtime.network_voltage,
            runtime.branch_current,
            runtime.output_buffer,
            runtime.direct_drive,
            runtime.zero_rhs,
            runtime.zero_solution,
            runtime.zero_reflected,
            runtime.core_incident,
            runtime.core_reflected,
            runtime.scene_incident,
            runtime.scene_reflected,
            runtime.solve_workspace,
            runtime.solve_scalar,
            runtime.net_power,
            runtime.power_buffer,
            runtime.port_energy,
            runtime.absorbed_increment,
            runtime.generated_increment,
            runtime.absorbed_energy,
            runtime.generated_energy,
        ]
        if runtime.delay_runtime is not None:
            delay = runtime.delay_runtime
            mutated.extend(
                (
                    delay.forward_ring,
                    delay.reverse_ring,
                    delay.forward_previous_input,
                    delay.forward_previous_output,
                    delay.reverse_previous_input,
                    delay.reverse_previous_output,
                    delay.forward_integer_sample,
                    delay.forward_fractional_sample,
                    delay.forward_temp,
                    delay.reverse_integer_sample,
                    delay.reverse_fractional_sample,
                    delay.reverse_temp,
                    delay.read_positions,
                    delay.read_indices,
                    delay.write_indices,
                    delay.cursor,
                )
            )
        for port_runtime in runtime.port_runtimes:
            lumped = port_runtime.lumped
            mutated.extend(
                (
                    lumped.edge_buffer,
                    lumped.correction_buffer,
                    lumped.last_voltage_before,
                    lumped.last_voltage_midpoint,
                    lumped.last_voltage_after,
                    lumped.last_model_voltage_midpoint,
                    lumped.last_branch_current,
                )
            )
        for tensor in mutated:
            if tensor.data_ptr() not in seen:
                tensors.append(tensor)
                seen.add(tensor.data_ptr())
    saved = [tensor.clone() for tensor in tensors]

    def restore() -> None:
        for tensor, value in zip(tensors, saved):
            tensor.copy_(value)

    try:
        replay = CudaGraphRunner(enabled=True, warmup_steps=3).capture(normal)
    except Exception:
        restore()
        return normal
    restore()
    solver._network_cuda_graph_active = True
    return replay


def finalize_embedded_networks(solver, ports) -> dict[str, EmbeddedNetworkData]:
    output: dict[str, EmbeddedNetworkData] = {}
    for runtime in getattr(solver, "_network_runtimes", ()):
        connected_data = tuple(
            ports[port_name] for port_name in runtime.compiled.connection_names
        )
        reference_frequencies = connected_data[0].frequencies
        if any(
            data.frequencies.shape != reference_frequencies.shape
            or not torch.equal(data.frequencies, reference_frequencies)
            for data in connected_data[1:]
        ):
            raise RuntimeError(
                f"Embedded network {runtime.name!r} ports do not share one frequency grid."
            )
        voltage = torch.stack(tuple(data.voltage for data in connected_data), dim=0)
        current = torch.stack(tuple(-data.current for data in connected_data), dim=0)
        port_power = 0.5 * torch.real(voltage * torch.conj(current))
        net_power = torch.sum(port_power, dim=0)
        absorbed = torch.clamp(net_power, min=0.0)
        generated = torch.clamp(-net_power, min=0.0)
        report = runtime.compiled.fit_report
        output[runtime.name] = EmbeddedNetworkData(
            name=runtime.name,
            frequencies=reference_frequencies,
            port_names=runtime.compiled.port_order,
            voltage=voltage,
            current=current,
            port_power=port_power,
            absorbed_power=absorbed,
            generated_power=generated,
            state_norm=torch.linalg.vector_norm(runtime.state),
            model_id=runtime.compiled.model_id,
            fit_report=report,
            runtime_warnings=runtime.runtime_warnings,
            metadata={
                "connections": runtime.compiled.connection_names,
                "current_convention": "entering_embedded_network",
                "frequency_band": runtime.compiled.frequency_band,
                "direct_loop_condition": runtime.loop_condition,
                "delay_seconds": (
                    ()
                    if runtime.compiled.delay is None
                    else runtime.compiled.delay.delay_seconds
                ),
                "delay_phase_error_degrees": (
                    None
                    if runtime.compiled.delay is None
                    else runtime.compiled.delay.phase_error_degrees
                ),
                "delay_reembedding_max_error": (
                    None
                    if runtime.compiled.delay is None
                    else runtime.compiled.delay.reembedding_max_error
                ),
                "port_energy": tuple(
                    float(value) for value in runtime.port_energy.detach().cpu()
                ),
                "absorbed_energy": float(runtime.absorbed_energy.detach().cpu()),
                "generated_energy": float(runtime.generated_energy.detach().cpu()),
            },
        )
    return output


__all__ = [
    "PreparedNetworkRuntime",
    "apply_network_runtime",
    "apply_network_runtimes",
    "finalize_embedded_networks",
    "make_network_runner",
    "prepare_network_runtimes",
]

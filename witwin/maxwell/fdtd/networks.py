from __future__ import annotations

from dataclasses import dataclass

import torch

from ..compiler.networks import CompiledNetworkBlock
from ..network import EmbeddedNetworkData


@dataclass
class PreparedNetworkRuntime:
    """Fixed-shape single-port state-space feedback on one FDTD device."""

    compiled: CompiledNetworkBlock
    port_runtime: object
    electric_field: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    state: torch.Tensor
    next_state: torch.Tensor
    state_drive: torch.Tensor
    output_buffer: torch.Tensor
    inverse_loop: torch.Tensor
    loop_denominator: torch.Tensor
    power_buffer: torch.Tensor
    absorbed_increment: torch.Tensor
    generated_increment: torch.Tensor
    absorbed_energy: torch.Tensor
    generated_energy: torch.Tensor
    runtime_warnings: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.compiled.name


def _runtime_matrix(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).contiguous()


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
        if compiled.port_count != 1:
            raise NotImplementedError(
                "Phase 2 FDTD runtime supports one-port embedded networks; "
                "multiport coupling is Phase 3."
            )
        port_name = compiled.connection_names[0]
        try:
            port_runtime = port_runtimes[port_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Compiled network {compiled.name!r} has no prepared port {port_name!r}."
            ) from exc
        lumped = port_runtime.lumped
        if lumped is None or port_runtime.embedded_network_name != compiled.name:
            raise RuntimeError(
                f"Port {port_name!r} was not prepared for embedded network {compiled.name!r}."
            )
        discrete = compiled.discrete
        matrices = (discrete.A, discrete.B, discrete.C, discrete.D)
        if any(matrix.requires_grad for matrix in matrices):
            raise NotImplementedError(
                "Trainable embedded-network state-space coefficients require the Phase 4 "
                "network adjoint; Phase 2 accepts fixed coefficients only."
            )
        dtype = lumped.field_dtype
        device = lumped.linear_indices.device
        A, B, C, D = (
            _runtime_matrix(matrix, device=device, dtype=dtype)
            for matrix in matrices
        )
        if B.shape[1:] != (1,) or C.shape[:1] != (1,) or D.shape != (1, 1):
            raise ValueError("Phase 2 embedded-network matrices must describe one input/output.")
        loop_denominator = torch.ones((), device=device, dtype=dtype)
        loop_denominator.addcmul_(D[0, 0], lumped.discrete_port_impedance)
        scale = max(1.0, float(torch.abs(D[0, 0] * lumped.discrete_port_impedance).item()))
        if float(torch.abs(loop_denominator).item()) <= 64.0 * torch.finfo(dtype).eps * scale:
            raise ValueError(
                f"Embedded network {compiled.name!r} has a singular same-step direct loop."
            )
        inverse_loop = torch.reciprocal(loop_denominator)
        state = torch.zeros((A.shape[0],), device=device, dtype=dtype)
        scalar_zeros = [torch.zeros((), device=device, dtype=dtype) for _ in range(5)]
        warnings: list[str] = []
        if compiled.fit_report is not None and compiled.fit_report.passivity_margin is None:
            warnings.append("The fitted model has no recorded passivity margin.")
        prepared.append(
            PreparedNetworkRuntime(
                compiled=compiled,
                port_runtime=port_runtime,
                electric_field=getattr(solver, port_runtime.field_name),
                A=A,
                B=B,
                C=C,
                D=D,
                state=state,
                next_state=torch.empty_like(state),
                state_drive=torch.empty_like(state),
                output_buffer=torch.empty((1,), device=device, dtype=dtype),
                inverse_loop=inverse_loop,
                loop_denominator=loop_denominator,
                power_buffer=scalar_zeros[0],
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
    """Advance one implicit state-space load and correct its Yee field in place."""

    port_runtime = runtime.port_runtime
    lumped = port_runtime.lumped
    electric_field = runtime.electric_field
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

    torch.mv(runtime.C, runtime.state, out=runtime.output_buffer)
    lumped.last_branch_current.copy_(runtime.output_buffer[0])
    lumped.last_branch_current.addcmul_(runtime.D[0, 0], lumped.last_voltage_before)
    lumped.last_branch_current.mul_(runtime.inverse_loop)

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
    lumped.last_voltage_midpoint.copy_(lumped.last_voltage_before)
    lumped.last_voltage_midpoint.addcmul_(
        lumped.discrete_port_impedance,
        lumped.last_branch_current,
        value=-1.0,
    )
    lumped.last_voltage_after.copy_(lumped.last_voltage_before)
    lumped.last_voltage_after.addcmul_(
        lumped.coupling_impedance,
        lumped.last_branch_current,
        value=-1.0,
    )
    lumped.last_model_voltage_midpoint.copy_(lumped.last_voltage_midpoint)

    torch.mv(runtime.A, runtime.state, out=runtime.next_state)
    torch.mv(
        runtime.B,
        lumped.last_voltage_midpoint.reshape(1),
        out=runtime.state_drive,
    )
    runtime.next_state.add_(runtime.state_drive)
    runtime.state.copy_(runtime.next_state)

    torch.mul(
        lumped.last_voltage_midpoint,
        lumped.last_branch_current,
        out=runtime.power_buffer,
    )
    runtime.power_buffer.mul_(lumped.dt)
    torch.clamp(runtime.power_buffer, min=0.0, out=runtime.absorbed_increment)
    torch.neg(runtime.power_buffer, out=runtime.generated_increment)
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
        lumped = runtime.port_runtime.lumped
        mutated = (
            runtime.electric_field,
            runtime.state,
            runtime.next_state,
            runtime.state_drive,
            runtime.output_buffer,
            runtime.power_buffer,
            runtime.absorbed_increment,
            runtime.generated_increment,
            runtime.absorbed_energy,
            runtime.generated_energy,
            lumped.edge_buffer,
            lumped.correction_buffer,
            lumped.last_voltage_before,
            lumped.last_voltage_midpoint,
            lumped.last_voltage_after,
            lumped.last_model_voltage_midpoint,
            lumped.last_branch_current,
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
        port_name = runtime.compiled.connection_names[0]
        port_data = ports[port_name]
        voltage = port_data.voltage.unsqueeze(0)
        current = (-port_data.current).unsqueeze(0)
        signed_power = 0.5 * torch.real(voltage * torch.conj(current))
        absorbed = torch.clamp(signed_power, min=0.0)
        generated = torch.clamp(-signed_power, min=0.0)
        report = runtime.compiled.fit_report
        output[runtime.name] = EmbeddedNetworkData(
            name=runtime.name,
            frequencies=port_data.frequencies,
            port_names=runtime.compiled.port_order,
            voltage=voltage,
            current=current,
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
                "direct_loop_denominator": float(runtime.loop_denominator.detach().cpu()),
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

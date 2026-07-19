from __future__ import annotations

from dataclasses import dataclass

import torch

from ..compiler.networks import CompiledNetworkBlock
from ..network import EmbeddedNetworkData
from .checkpoint import network_carried_voltage_name, network_state_name
from .delay import (
    PreparedBidirectionalDelay,
    prepare_bidirectional_delay,
    read_bidirectional_delay,
    write_bidirectional_delay,
)


@dataclass
class PreparedNetworkTerminalGroup:
    """Batched terminal edges sharing one electric-field component."""

    electric_field: torch.Tensor
    linear_indices: torch.Tensor
    port_indices: torch.Tensor
    voltage_weights: torch.Tensor
    injection: torch.Tensor
    edge_buffer: torch.Tensor
    correction_buffer: torch.Tensor


@dataclass
class PreparedNetworkRuntime:
    """Fixed-shape N-port state-space feedback on one FDTD device."""

    compiled: CompiledNetworkBlock
    port_runtimes: tuple[object, ...]
    electric_fields: tuple[torch.Tensor, ...]
    terminal_groups: tuple[PreparedNetworkTerminalGroup, ...]
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    state: torch.Tensor
    next_state: torch.Tensor
    state_drive: torch.Tensor
    free_voltage: torch.Tensor
    raw_free_voltage: torch.Tensor
    carried_voltage: torch.Tensor
    coupling_impedance: torch.Tensor
    network_voltage: torch.Tensor
    voltage_after: torch.Tensor
    branch_current: torch.Tensor
    output_buffer: torch.Tensor
    direct_drive: torch.Tensor
    gain_state: torch.Tensor
    gain_voltage: torch.Tensor
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


@dataclass(frozen=True)
class NetworkStepTrace:
    """Small fixed-state trace required by the discrete network adjoint."""

    network_index: int
    state: torch.Tensor
    free_voltage: torch.Tensor
    carried_voltage: torch.Tensor
    coupling_voltage: torch.Tensor
    network_voltage: torch.Tensor
    branch_current: torch.Tensor


def _runtime_matrix(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).contiguous()


def _matvec_out(matrix: torch.Tensor, vector: torch.Tensor, output: torch.Tensor) -> None:
    """Matrix-vector product without the temporary allocated by CUDA ``mv``."""

    torch.mm(matrix, vector.unsqueeze(1), out=output.unsqueeze(1))


def _prepare_terminal_groups(
    connected_runtimes: tuple[object, ...],
    electric_fields: tuple[torch.Tensor, ...],
) -> tuple[PreparedNetworkTerminalGroup, ...]:
    """Pack terminal edges by field component for fixed-allocation GPU updates."""

    grouped: dict[str, list[tuple[int, object, torch.Tensor]]] = {}
    for port_index, (port_runtime, electric_field) in enumerate(
        zip(connected_runtimes, electric_fields)
    ):
        grouped.setdefault(port_runtime.field_name, []).append(
            (port_index, port_runtime.lumped, electric_field)
        )
    prepared = []
    for entries in grouped.values():
        field = entries[0][2]
        linear_indices = torch.cat(
            tuple(lumped.linear_indices for _, lumped, _ in entries)
        ).contiguous()
        port_indices = torch.cat(
            tuple(
                torch.full_like(lumped.linear_indices, port_index)
                for port_index, lumped, _ in entries
            )
        ).contiguous()
        voltage_weights = torch.cat(
            tuple(lumped.voltage_weights for _, lumped, _ in entries)
        ).contiguous()
        injection = torch.cat(
            tuple(lumped.injection for _, lumped, _ in entries)
        ).contiguous()
        prepared.append(
            PreparedNetworkTerminalGroup(
                electric_field=field,
                linear_indices=linear_indices,
                port_indices=port_indices,
                voltage_weights=voltage_weights,
                injection=injection,
                edge_buffer=torch.empty_like(voltage_weights),
                correction_buffer=torch.empty_like(injection),
            )
        )
    return tuple(prepared)


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


def _native_lu_solve_out(
    lu: torch.Tensor,
    pivots: torch.Tensor,
    rhs: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Use the native batched solve inside a CUDA graph capture only."""

    torch.linalg.lu_solve(lu, pivots, rhs.unsqueeze(1), out=output.unsqueeze(1))


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


def prepare_network_runtimes(
    solver, *, compiled_networks=None
) -> tuple[PreparedNetworkRuntime, ...]:
    """Compile embedded networks once and allocate all hot-path tensors.

    ``compiled_networks`` lets a caller inject already-compiled blocks instead
    of recompiling from ``solver.scene``. The distributed owner path uses this
    to build one owner-resident network runtime from the global scene while the
    owner shard's local scene carries no networks.
    """

    if compiled_networks is None:
        compiled_networks = solver.scene.compile_networks(dt=solver.dt, device=solver.device)
    else:
        compiled_networks = tuple(compiled_networks)
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
        # E4b fixed-cost reduction: the same-step direct loop matrix
        # ``M = I + D * diag(Z_f)`` is constant across the whole run, so the
        # ordinary-Y branch-current solve ``M x = C@state + D@v`` collapses to
        # ``x = (M^-1 C)@state + (M^-1 D)@v``. Applying the already-computed
        # pivoted LU to the *constant* C and D once here replaces the per-step
        # sequential triangular substitution (O(port) tiny kernels) with two
        # dense matvecs, without introducing a naive matrix inverse: the
        # composite operators are LU-solved, not formed by inversion, and the
        # ill-conditioned direct-loop parity gate is preserved. Only the
        # delay-free path uses these; the delayed reference-plane solve keeps its
        # own LU over the zero-delay subset.
        if compiled.delay is None:
            gain_state = torch.linalg.lu_solve(loop_lu, loop_pivots, C).contiguous()
            gain_voltage = torch.linalg.lu_solve(loop_lu, loop_pivots, D).contiguous()
        else:
            gain_state = torch.empty((port_count, 0), device=device, dtype=dtype)
            gain_voltage = torch.empty((port_count, 0), device=device, dtype=dtype)
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
        connected_tuple = tuple(connected_runtimes)
        field_tuple = tuple(electric_fields)
        free_voltage = torch.empty((port_count,), device=device, dtype=dtype)
        network_voltage = torch.empty_like(free_voltage)
        voltage_after = torch.empty_like(free_voltage)
        branch_current = torch.empty_like(free_voltage)
        # Slice U2: the trapezoidal network interface carries the previous step's
        # post-step port voltage; ``coupling_impedance`` = 2 * feedback (discrete)
        # is the full port coupling used for the post-step recurrence.
        raw_free_voltage = torch.empty_like(free_voltage)
        carried_voltage = torch.zeros_like(free_voltage)
        coupling_impedance = (2.0 * feedback_impedance).contiguous()
        network_runtime = PreparedNetworkRuntime(
                compiled=compiled,
                port_runtimes=connected_tuple,
                electric_fields=field_tuple,
                terminal_groups=_prepare_terminal_groups(connected_tuple, field_tuple),
                A=A,
                B=B,
                C=C,
                D=D,
                state=state,
                next_state=torch.empty_like(state),
                state_drive=torch.empty_like(state),
                free_voltage=free_voltage,
                raw_free_voltage=raw_free_voltage,
                carried_voltage=carried_voltage,
                coupling_impedance=coupling_impedance,
                network_voltage=network_voltage,
                voltage_after=voltage_after,
                branch_current=branch_current,
                output_buffer=torch.empty((port_count,), device=device, dtype=dtype),
                direct_drive=torch.empty((port_count,), device=device, dtype=dtype),
                gain_state=gain_state,
                gain_voltage=gain_voltage,
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
        for index, port_runtime in enumerate(connected_tuple):
            lumped = port_runtime.lumped
            # Slice U2 caveat: for embedded-network ports ``last_voltage_before``
            # aliases ``free_voltage``, which the forward step overwrites with the
            # trapezoidal *coupling* voltage 0.5 * (carried_voltage +
            # raw_free_voltage) -- not the raw pre-step free voltage the name
            # implies elsewhere (lumped/circuit ports keep the raw meaning). The
            # raw free voltage lives in ``raw_free_voltage``; the post-step
            # recurrence carries it via ``carried_voltage``. The only outside
            # consumer of this alias is the CUDA-graph snapshot list, which just
            # needs every mutated buffer, so the aliased semantics are harmless
            # there.
            lumped.last_voltage_before = free_voltage[index]
            lumped.last_voltage_midpoint = network_voltage[index]
            lumped.last_model_voltage_midpoint = network_voltage[index]
            lumped.last_voltage_after = voltage_after[index]
            lumped.last_branch_current = branch_current[index]
        prepared.append(network_runtime)
    solver._network_runtimes = tuple(prepared)
    solver._network_cuda_graph_active = False
    return solver._network_runtimes


def replay_network_runtimes(
    solver,
    electric_fields: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    capture=None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Replay ordinary-Y network feedback without mutating prepared state."""

    fields = dict(electric_fields)
    next_state: dict[str, torch.Tensor] = {}
    traces: list[NetworkStepTrace] = []
    for network_index, runtime in enumerate(
        getattr(solver, "_network_runtimes", ())
    ):
        if runtime.delay_runtime is not None:
            raise NotImplementedError(
                "Differentiable embedded-network replay does not support explicit "
                "delay state: the bidirectional reference-plane ring couples a "
                "step to samples written up to max_delay_steps earlier (a "
                "coupling that can span checkpoint segments), and the Thiran "
                "fractional-delay filter is an IIR recurrence over consecutive "
                "steps. The segment-local network pullback reverses only a "
                "self-contained same-step recurrence, so a correct delay adjoint "
                "requires a delay-aware reverse ring carried across segments. The "
                "delay forward state is captured in the checkpoint schema for "
                "forward resume; the reverse pass is intentionally fail-closed "
                "rather than approximate."
            )
        name = network_state_name(network_index)
        carried_name = network_carried_voltage_name(network_index)
        old_state = state[name]
        carried_prev = state[carried_name]
        free_voltage = torch.stack(
            tuple(
                torch.dot(
                    torch.index_select(
                        fields[port_runtime.field_name].reshape(-1),
                        0,
                        port_runtime.lumped.linear_indices,
                    ),
                    port_runtime.lumped.voltage_weights,
                )
                for port_runtime in runtime.port_runtimes
            )
        )
        # Slice U2: the network solve consumes the trapezoidal half-step voltage.
        coupling_voltage = 0.5 * (carried_prev + free_voltage)
        drive = runtime.C @ old_state + runtime.D @ coupling_voltage
        branch_current = torch.linalg.solve(runtime.loop_denominator, drive)
        network_voltage = (
            coupling_voltage - runtime.feedback_impedance * branch_current
        )
        advanced_state = (
            runtime.A @ old_state + runtime.B @ network_voltage
        )
        # Post-step port voltage carried to the next step (raw free voltage minus
        # the full coupling drop), matching the forward finalizer.
        next_carried = free_voltage - runtime.coupling_impedance * branch_current
        for port_index, port_runtime in enumerate(runtime.port_runtimes):
            lumped = port_runtime.lumped
            corrected = fields[port_runtime.field_name].clone()
            corrected.reshape(-1).index_add_(
                0,
                lumped.linear_indices,
                -lumped.injection * branch_current[port_index],
            )
            fields[port_runtime.field_name] = corrected
        next_state[name] = advanced_state
        next_state[carried_name] = next_carried
        traces.append(
            NetworkStepTrace(
                network_index=network_index,
                state=old_state,
                free_voltage=free_voltage,
                carried_voltage=carried_prev,
                coupling_voltage=coupling_voltage,
                network_voltage=network_voltage,
                branch_current=branch_current,
            )
        )
    if capture is not None:
        capture.append(tuple(traces))
    return fields, next_state


def pullback_network_runtimes(
    solver,
    traces,
    adjoint_state: dict[str, torch.Tensor],
    *,
    port_sample_adjoints: dict[
        int,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    eps_by_field: dict[str, torch.Tensor],
):
    """Reverse ordinary-Y state recurrence and implicit terminal coupling.

    Accuracy note (slice U2): the manual VJP matches a torch-autograd oracle to
    within 1-2 float64 ULP, not bit-for-bit. On fixed pinned seeds it is exact
    (0 ULP) for most cotangents; the worst case measured over fresh random seeds
    is ~1.2 ULP on the multiport leg and ~1.0 ULP single-port, from the
    LU-solve reassociating floating-point sums differently than autograd's
    recorded graph. The oracle tests gate this with ``assert_close`` (float64
    tolerance), which is the truthful contract, not ``torch.equal``.
    """

    updated = dict(adjoint_state)
    grad_eps = {name: torch.zeros_like(value) for name, value in eps_by_field.items()}
    semantic_grads: dict[tuple[str, str, str], torch.Tensor] = {}
    port_indices = {
        id(port_runtime): index
        for index, port_runtime in enumerate(getattr(solver, "_port_runtimes", ()))
    }
    for trace in reversed(tuple(traces)):
        runtime = solver._network_runtimes[trace.network_index]
        state_name = network_state_name(trace.network_index)
        output_field_adjoints = {
            port_runtime.field_name: updated[port_runtime.field_name]
            for port_runtime in runtime.port_runtimes
        }
        bar_current = torch.zeros_like(trace.branch_current)
        bar_voltage = torch.zeros_like(trace.network_voltage)
        bar_injection: list[torch.Tensor] = []
        for port_index, port_runtime in enumerate(runtime.port_runtimes):
            lumped = port_runtime.lumped
            field_adjoint = output_field_adjoints[port_runtime.field_name]
            local_field_adjoint = torch.index_select(
                field_adjoint.reshape(-1),
                0,
                lumped.linear_indices,
            )
            bar_injection.append(
                -trace.branch_current[port_index] * local_field_adjoint
            )
            bar_current[port_index].sub_(
                torch.dot(lumped.injection, local_field_adjoint)
            )
            voltage_seed, current_seed, _drive_seed = port_sample_adjoints.get(
                port_indices[id(port_runtime)],
                (
                    torch.zeros_like(trace.branch_current[port_index]),
                    torch.zeros_like(trace.branch_current[port_index]),
                    torch.zeros_like(trace.branch_current[port_index]),
                ),
            )
            bar_voltage[port_index].add_(
                voltage_seed.to(
                    device=bar_voltage.device,
                    dtype=bar_voltage.dtype,
                )
            )
            bar_current[port_index].sub_(
                current_seed.to(
                    device=bar_current.device,
                    dtype=bar_current.dtype,
                )
            )

        bar_next_state = updated[state_name]
        carried_name = network_carried_voltage_name(trace.network_index)
        bar_next_carried = updated[carried_name]
        grad_a = torch.outer(bar_next_state, trace.state)
        grad_b = torch.outer(bar_next_state, trace.network_voltage)
        bar_state = runtime.A.transpose(0, 1) @ bar_next_state
        bar_voltage = bar_voltage + runtime.B.transpose(0, 1) @ bar_next_state

        # Slice U2 reverse of the trapezoidal interface.
        #   nv = coupling_voltage - Zf*i      -> cotangent on the coupling voltage
        #   next_carried = free_voltage - Zc*i (Zc = coupling_impedance = 2*Zf)
        bar_coupling_voltage = bar_voltage.clone()
        bar_current = bar_current - runtime.feedback_impedance * bar_voltage
        bar_free_voltage = bar_next_carried.clone()
        bar_current = bar_current - runtime.coupling_impedance * bar_next_carried
        bar_drive = torch.linalg.solve(
            runtime.loop_denominator.transpose(0, 1),
            bar_current,
        )
        grad_loop = -torch.outer(bar_drive, trace.branch_current)
        grad_c = torch.outer(bar_drive, trace.state)
        grad_d = (
            torch.outer(bar_drive, trace.coupling_voltage)
            + grad_loop * runtime.feedback_impedance.unsqueeze(0)
        )
        bar_state = bar_state + runtime.C.transpose(0, 1) @ bar_drive
        bar_coupling_voltage = (
            bar_coupling_voltage + runtime.D.transpose(0, 1) @ bar_drive
        )
        # feedback (Zf) grad: from nv, the loop denominator, and the post-step
        # coupling drop Zc = 2*Zf carried into next_carried.
        bar_feedback = (
            -bar_voltage * trace.branch_current
            + torch.sum(grad_loop * runtime.D, dim=0)
            - 2.0 * bar_next_carried * trace.branch_current
        )
        # coupling_voltage = 0.5*(carried_prev + free_voltage): split the cotangent
        # into the carried-state cotangent (returned to the preceding step) and the
        # free-voltage (field) cotangent.
        bar_carried_prev = 0.5 * bar_coupling_voltage
        bar_free_voltage = bar_free_voltage + 0.5 * bar_coupling_voltage

        for port_index, port_runtime in enumerate(runtime.port_runtimes):
            lumped = port_runtime.lumped
            bar_injection[port_index] = (
                bar_injection[port_index]
                + 0.5 * lumped.voltage_weights * bar_feedback[port_index]
            )
            field_name = port_runtime.field_name
            field_adjoint = updated[field_name].clone()
            field_adjoint.reshape(-1).index_add_(
                0,
                lumped.linear_indices,
                lumped.voltage_weights * bar_free_voltage[port_index],
            )
            updated[field_name] = field_adjoint
            local_eps = torch.index_select(
                eps_by_field[field_name].reshape(-1),
                0,
                lumped.linear_indices,
            )
            local_grad_eps = (
                -bar_injection[port_index] * lumped.injection / local_eps
            )
            grad_eps[field_name].reshape(-1).index_add_(
                0,
                lumped.linear_indices,
                local_grad_eps,
            )

        updated[state_name] = bar_state
        updated[carried_name] = bar_carried_prev
        for matrix_name, gradient in (
            ("A", grad_a),
            ("B", grad_b),
            ("C", grad_c),
            ("D", grad_d),
        ):
            key = ("network", runtime.name, matrix_name)
            semantic_grads[key] = semantic_grads.get(
                key,
                torch.zeros_like(gradient),
            ) + gradient
    return updated, grad_eps, semantic_grads


def _gather_network_free_voltage(runtime: PreparedNetworkRuntime) -> None:
    """Sample the connected Yee edges and form the trapezoidal coupling voltage.

    Slice U2: the raw per-port free voltage V_free is gathered into
    ``raw_free_voltage`` and the network solve consumes the trapezoidal half-step
    voltage ``free_voltage`` = 0.5 * (V_after_prev + V_free), matching the native
    lumped runtime so cross-model agreement holds at the unified convention. The
    raw free voltage is retained for the post-step recurrence in the finalizer.
    """

    runtime.raw_free_voltage.zero_()
    for group in runtime.terminal_groups:
        flat_field = group.electric_field.view(-1)
        torch.index_select(
            flat_field,
            0,
            group.linear_indices,
            out=group.edge_buffer,
        )
        torch.mul(
            group.edge_buffer,
            group.voltage_weights,
            out=group.edge_buffer,
        )
        runtime.raw_free_voltage.index_add_(
            0,
            group.port_indices,
            group.edge_buffer,
        )
    torch.add(
        runtime.carried_voltage,
        runtime.raw_free_voltage,
        out=runtime.free_voltage,
    )
    runtime.free_voltage.mul_(0.5)


def _scatter_network_branch_current(runtime: PreparedNetworkRuntime) -> None:
    """Inject the solved branch currents back into the connected Yee edges."""

    for group in runtime.terminal_groups:
        flat_field = group.electric_field.view(-1)
        torch.index_select(
            runtime.branch_current,
            0,
            group.port_indices,
            out=group.edge_buffer,
        )
        torch.mul(
            group.injection,
            group.edge_buffer,
            out=group.correction_buffer,
        )
        flat_field.index_add_(
            0,
            group.linear_indices,
            group.correction_buffer,
            alpha=-1.0,
        )


def _solve_network_currents(
    runtime: PreparedNetworkRuntime,
    *,
    native_lu: bool = False,
) -> torch.Tensor:
    """Solve the same-step implicit loop from the current ``free_voltage``.

    Writes ``branch_current`` and ``network_voltage`` and returns the state
    recurrence input (``network_voltage`` for the ordinary-Y form,
    ``core_incident`` for the delayed form). Neither gather nor scatter is
    performed here so the single-device and distributed paths can share it.
    """

    if runtime.delay_runtime is None:
        # E4b: constant-loop composite solve. ``branch_current`` =
        # (M^-1 C) @ state + (M^-1 D) @ free_voltage, two dense matvecs, instead
        # of a per-step sequential triangular substitution over the prepared LU.
        # ``native_lu`` no longer changes the delay-free result: eager and
        # CUDA-graph replay now produce bitwise-identical branch currents.
        _matvec_out(runtime.gain_state, runtime.state, runtime.branch_current)
        _matvec_out(runtime.gain_voltage, runtime.free_voltage, runtime.output_buffer)
        runtime.branch_current.add_(runtime.output_buffer)
        runtime.network_voltage.copy_(runtime.free_voltage)
        runtime.network_voltage.addcmul_(
            runtime.feedback_impedance,
            runtime.branch_current,
            value=-1.0,
        )
        return runtime.network_voltage
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
            if native_lu:
                _native_lu_solve_out(
                    runtime.loop_lu,
                    runtime.loop_pivots,
                    runtime.zero_rhs,
                    runtime.zero_solution,
                )
            else:
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
        return runtime.core_incident


def _finalize_network_step(
    runtime: PreparedNetworkRuntime,
    state_input: torch.Tensor,
) -> None:
    """Advance the network state recurrence and accumulate port power."""

    # Slice U2: the post-step port voltage uses the raw free voltage and the full
    # port coupling drop (2 * feedback), matching the native runtime's
    # last_voltage_after. This is the carried state for the next step's average.
    runtime.voltage_after.copy_(runtime.raw_free_voltage)
    runtime.voltage_after.addcmul_(
        runtime.coupling_impedance,
        runtime.branch_current,
        value=-1.0,
    )
    runtime.carried_voltage.copy_(runtime.voltage_after)

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


def apply_network_runtime(
    runtime: PreparedNetworkRuntime,
    *,
    native_lu: bool = False,
) -> None:
    """Advance one implicit N-port load and correct its Yee fields in place."""

    _gather_network_free_voltage(runtime)
    state_input = _solve_network_currents(runtime, native_lu=native_lu)
    _scatter_network_branch_current(runtime)
    _finalize_network_step(runtime, state_input)


def advance_network_external(
    runtime: PreparedNetworkRuntime,
    free_voltages: tuple[torch.Tensor, ...],
    *,
    native_lu: bool = False,
) -> torch.Tensor:
    """Advance one network from externally gathered port voltages.

    The caller owns the field gather and the branch-current scatter (the
    distributed runtime performs both per shard), so this routine only runs the
    implicit loop solve, the state recurrence, and the port-power accounting on
    the owner device. It never reads or writes any Yee field, which is what lets
    a network span multiple shards behind an O(port) scalar contract. Returns
    the persistent branch-current buffer for the caller to scatter.
    """

    if len(free_voltages) != runtime.free_voltage.numel():
        raise ValueError(
            "advance_network_external requires exactly one voltage scalar per "
            "connected port."
        )
    for index, value in enumerate(free_voltages):
        runtime.raw_free_voltage[index].copy_(value)
    # Slice U2: form the trapezoidal coupling voltage from the carried post-step
    # voltage, identically to the single-device gather.
    torch.add(
        runtime.carried_voltage,
        runtime.raw_free_voltage,
        out=runtime.free_voltage,
    )
    runtime.free_voltage.mul_(0.5)
    state_input = _solve_network_currents(runtime, native_lu=native_lu)
    _finalize_network_step(runtime, state_input)
    return runtime.branch_current


def apply_network_runtimes(solver, *, native_lu: bool = False) -> None:
    for runtime in getattr(solver, "_network_runtimes", ()):
        apply_network_runtime(runtime, native_lu=native_lu)


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
            runtime.raw_free_voltage,
            runtime.carried_voltage,
            runtime.network_voltage,
            runtime.voltage_after,
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
        for group in runtime.terminal_groups:
            mutated.extend((group.edge_buffer, group.correction_buffer))
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
            # state_norm is a reported diagnostic, never an optimization
            # observable. Detach the state before the norm so the contract is
            # enforced by construction instead of relying on the runtime state
            # happening to be graph-free.
            state_norm=torch.linalg.vector_norm(runtime.state.detach()),
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
    "NetworkStepTrace",
    "PreparedNetworkRuntime",
    "PreparedNetworkTerminalGroup",
    "advance_network_external",
    "apply_network_runtime",
    "apply_network_runtimes",
    "finalize_embedded_networks",
    "make_network_runner",
    "prepare_network_runtimes",
    "pullback_network_runtimes",
    "replay_network_runtimes",
]

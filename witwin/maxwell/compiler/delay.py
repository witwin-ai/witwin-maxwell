from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Literal, Sequence

import torch

from ..network import NetworkData
from ..rational import FitReport, NetworkFitReport


@dataclass(frozen=True)
class CompiledNetworkDelay:
    """Auditable fixed-memory realization of per-port reference-plane delays."""

    delay_seconds: tuple[float, ...]
    integer_steps: tuple[int, ...]
    fractional_samples: tuple[float, ...]
    thiran_coefficients: tuple[float, ...]
    buffer_steps: int
    estimation_rank: int | None
    equation_count: int
    residual_seconds: float | None
    phase_error_degrees: float
    reembedding_max_error: float
    warnings: tuple[str, ...] = ()

    def update_report(self, report: FitReport, *, port_count: int) -> NetworkFitReport:
        values = {
            "delay_seconds": self.delay_seconds,
            "delay_estimation_rank": self.estimation_rank,
            "delay_equation_count": self.equation_count,
            "delay_residual_seconds": self.residual_seconds,
            "delay_phase_error_degrees": self.phase_error_degrees,
            "delay_reembedding_max_error": self.reembedding_max_error,
            "warnings": tuple(report.warnings) + self.warnings,
        }
        if isinstance(report, NetworkFitReport):
            return replace(report, **values)
        payload = asdict(report)
        payload.update(values)
        return NetworkFitReport(**payload, port_count=port_count)


def _delay_tensor(
    delay_seconds: Sequence[float],
    *,
    port_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    delay = torch.as_tensor(tuple(delay_seconds), device=device, dtype=dtype)
    if delay.shape != (port_count,):
        raise ValueError("delay_seconds must contain one one-way delay per network port.")
    if not bool(torch.all(torch.isfinite(delay))) or not bool(torch.all(delay >= 0.0)):
        raise ValueError("delay_seconds must contain finite non-negative values.")
    return delay


def delay_phase_matrix(
    frequencies: torch.Tensor,
    delay_seconds: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return the causal reference-plane factor for the exp(-i*omega*t) convention."""

    frequencies = torch.as_tensor(frequencies)
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not frequencies.dtype.is_floating_point or not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("frequencies must be finite real values.")
    delay = torch.as_tensor(delay_seconds, device=frequencies.device, dtype=frequencies.dtype)
    if delay.ndim != 1 or delay.numel() == 0:
        raise ValueError("delay_seconds must have non-empty shape [N].")
    if not bool(torch.all(torch.isfinite(delay))) or not bool(torch.all(delay >= 0.0)):
        raise ValueError("delay_seconds must contain finite non-negative values.")
    path_delay = delay[:, None] + delay[None, :]
    phase = 2.0 * torch.pi * frequencies[:, None, None] * path_delay[None, ...]
    return torch.complex(torch.cos(phase), torch.sin(phase))


def deembed_scattering(
    scattering: torch.Tensor,
    frequencies: torch.Tensor,
    delay_seconds: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    phase = delay_phase_matrix(frequencies, delay_seconds).to(
        device=scattering.device, dtype=scattering.dtype
    )
    if scattering.shape != phase.shape:
        raise ValueError("scattering must have shape [F, N, N] matching the delay declaration.")
    return scattering * phase.conj()


def reembed_scattering(
    scattering: torch.Tensor,
    frequencies: torch.Tensor,
    delay_seconds: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    phase = delay_phase_matrix(frequencies, delay_seconds).to(
        device=scattering.device, dtype=scattering.dtype
    )
    if scattering.shape != phase.shape:
        raise ValueError("scattering must have shape [F, N, N] matching the delay declaration.")
    return scattering * phase


def _unwrap_phase(values: torch.Tensor) -> torch.Tensor:
    phase = torch.angle(values)
    increments = torch.diff(phase)
    increments = torch.remainder(increments + torch.pi, 2.0 * torch.pi) - torch.pi
    return torch.cat((phase[:1], phase[0] + torch.cumsum(increments, dim=0)))


def _path_group_delay(
    frequencies: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    omega = 2.0 * torch.pi * frequencies
    magnitude_squared = torch.abs(values).square().real
    weight_sum = torch.sum(magnitude_squared)
    omega_mean = torch.sum(magnitude_squared * omega) / weight_sum
    phase = _unwrap_phase(values)
    phase_mean = torch.sum(magnitude_squared * phase) / weight_sum
    centered_omega = omega - omega_mean
    denominator = torch.sum(magnitude_squared * centered_omega.square())
    if float(denominator.item()) <= torch.finfo(frequencies.dtype).eps:
        raise ValueError("Delay extraction requires a nonzero frequency span.")
    slope = torch.sum(magnitude_squared * centered_omega * (phase - phase_mean)) / denominator
    fitted_phase = phase_mean + slope * centered_omega
    residual = torch.sqrt(
        torch.sum(magnitude_squared * (phase - fitted_phase).square()) / weight_sum
    )
    return slope, residual


def estimate_port_delays(
    network: NetworkData,
    *,
    band: tuple[float, float] | None = None,
    magnitude_floor: float = 1.0e-6,
) -> tuple[tuple[float, ...], int, int, float, tuple[str, ...]]:
    """Estimate non-negative one-way port delays from scattering phase slopes."""

    if magnitude_floor <= 0.0:
        raise ValueError("magnitude_floor must be positive.")
    if network.frequencies.requires_grad or network.s.requires_grad:
        raise RuntimeError(
            "Automatic delay extraction is non-differentiable; declare fixed delay_seconds instead."
        )
    # Delay extraction is a prepare-time scientific routine.  Resolve the tiny
    # rank-deficient least-squares problem in CPU float64 for deterministic
    # minimum-norm behavior across CUDA solver backends.
    frequencies = network.frequencies.to(device="cpu", dtype=torch.float64)
    scattering = network.s.to(device="cpu", dtype=torch.complex128)
    if band is not None:
        selected = (frequencies >= band[0]) & (frequencies <= band[1])
        frequencies = frequencies[selected]
        scattering = scattering[selected]
    if frequencies.numel() < 3:
        raise ValueError("Delay extraction requires at least three in-band frequency samples.")

    peak = torch.max(torch.abs(scattering))
    threshold = float(peak.item()) * magnitude_floor
    rows: list[torch.Tensor] = []
    slopes: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    port_count = len(network.port_names)
    for output in range(port_count):
        for input_ in range(port_count):
            values = scattering[:, output, input_]
            magnitude = torch.abs(values)
            if float(torch.max(magnitude).item()) <= threshold:
                continue
            reliable = magnitude > threshold
            if int(torch.count_nonzero(reliable).item()) < 3:
                continue
            slope, phase_residual = _path_group_delay(frequencies[reliable], values[reliable])
            row = torch.zeros(port_count, device=frequencies.device, dtype=frequencies.dtype)
            row[output] += 1.0
            row[input_] += 1.0
            rows.append(row)
            slopes.append(slope)
            amplitude_weight = torch.sqrt(torch.mean(magnitude[reliable].square()))
            weights.append(amplitude_weight / (1.0 + phase_residual))
    if not rows:
        raise ValueError("Automatic delay extraction found no reliable scattering path.")

    design = torch.stack(rows)
    target = torch.stack(slopes)
    weight = torch.stack(weights)
    weighted_design = design * weight[:, None]
    weighted_target = target * weight
    rank = int(torch.linalg.matrix_rank(weighted_design).item())
    active = torch.ones(port_count, device=frequencies.device, dtype=torch.bool)
    solution = torch.zeros(port_count, device=frequencies.device, dtype=frequencies.dtype)
    while bool(torch.any(active)):
        candidate = torch.linalg.pinv(weighted_design[:, active]) @ weighted_target
        if bool(torch.all(candidate >= 0.0)):
            solution[active] = candidate
            break
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        active[active_indices[candidate < 0.0]] = False
    residual = torch.sqrt(torch.mean((design @ solution - target).square()))
    tolerance = 256.0 * torch.finfo(solution.dtype).eps
    solution = torch.where(solution > tolerance, solution, torch.zeros_like(solution))
    if not bool(torch.any(solution > 0.0)):
        raise ValueError("Automatic delay extraction found no positive causal delay.")
    warnings: list[str] = []
    if rank < port_count:
        warnings.append(
            "Delay equations are rank deficient; the deterministic non-negative minimum-norm split was used."
        )
    return (
        tuple(float(value) for value in solution.detach().cpu().tolist()),
        rank,
        design.shape[0],
        float(residual.item()),
        tuple(warnings),
    )


def _discrete_delay_parameters(
    delay: torch.Tensor,
    *,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = delay / float(dt)
    integer = torch.floor(samples).to(dtype=torch.int64)
    fractional = samples - integer.to(dtype=samples.dtype)
    snap = 256.0 * torch.finfo(samples.dtype).eps
    carry = fractional >= 1.0 - snap
    integer = integer + carry.to(dtype=torch.int64)
    fractional = torch.where(carry | (fractional <= snap), torch.zeros_like(fractional), fractional)
    coefficient = torch.where(
        fractional > 0.0,
        (1.0 - fractional) / (1.0 + fractional),
        torch.zeros_like(fractional),
    )
    return integer, fractional, coefficient


def _fractional_phase_error(
    frequencies: torch.Tensor,
    delay: torch.Tensor,
    integer: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    dt: float,
) -> float:
    omega_dt = 2.0 * torch.pi * frequencies[:, None] * float(dt)
    unit_delay = torch.complex(torch.cos(omega_dt), torch.sin(omega_dt))
    integer_phase = torch.exp(1j * omega_dt * integer.to(dtype=frequencies.dtype)[None, :])
    coefficient_complex = coefficient.to(dtype=unit_delay.dtype)[None, :]
    fractional_phase = (coefficient_complex + unit_delay) / (
        1.0 + coefficient_complex * unit_delay
    )
    fractional_phase = torch.where(
        coefficient[None, :] == 0.0,
        torch.ones_like(fractional_phase),
        fractional_phase,
    )
    realized = integer_phase * fractional_phase
    target_phase = 2.0 * torch.pi * frequencies[:, None] * delay[None, :]
    target = torch.complex(torch.cos(target_phase), torch.sin(target_phase))
    one_way_error = realized * target.conj()
    path_error = torch.angle(one_way_error[:, :, None] * one_way_error[:, None, :])
    return math.degrees(float(torch.max(torch.abs(path_error)).item()))


def compile_network_delay(
    network: NetworkData,
    declaration: Literal["auto"] | Sequence[float],
    *,
    dt: float,
    max_delay_steps: int,
    band: tuple[float, float] | None = None,
    phase_tolerance_degrees: float = 3.0,
) -> tuple[CompiledNetworkDelay, torch.Tensor]:
    """Resolve, validate, and remove an explicit bounded reference-plane delay."""

    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar.")
    if not isinstance(max_delay_steps, int) or isinstance(max_delay_steps, bool) or max_delay_steps < 1:
        raise ValueError("max_delay_steps must be a positive integer.")
    if not math.isfinite(float(phase_tolerance_degrees)) or phase_tolerance_degrees <= 0.0:
        raise ValueError("phase_tolerance_degrees must be positive and finite.")
    frequencies = network.frequencies
    scattering = network.s
    selected = torch.ones_like(frequencies, dtype=torch.bool)
    if band is not None:
        if len(band) != 2 or not 0.0 <= band[0] < band[1]:
            raise ValueError("band must be an increasing non-negative frequency pair.")
        selected = (frequencies >= band[0]) & (frequencies <= band[1])
        if int(torch.count_nonzero(selected).item()) < 3:
            raise ValueError("Delay validation requires at least three in-band samples.")

    estimation_rank: int | None = None
    equation_count = 0
    residual_seconds: float | None = None
    warnings: tuple[str, ...] = ()
    if declaration == "auto":
        resolved, estimation_rank, equation_count, residual_seconds, warnings = estimate_port_delays(
            network, band=band
        )
    else:
        if isinstance(declaration, (str, bytes)):
            raise ValueError("declaration must be 'auto' or one delay per network port.")
        resolved = tuple(float(value) for value in declaration)
    delay = _delay_tensor(
        resolved,
        port_count=len(network.port_names),
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    nonzero = delay > 0.0
    if bool(torch.any(nonzero & (delay < float(dt)))):
        raise ValueError(
            "Explicit delay realization requires every nonzero one-way delay to be at least one FDTD step; "
            "shorter delay must remain in the rational core."
        )
    integer, fractional, coefficient = _discrete_delay_parameters(delay, dt=float(dt))
    required_steps = torch.ceil(delay / float(dt)).to(dtype=torch.int64)
    buffer_steps = int(torch.max(required_steps).item())
    if buffer_steps > max_delay_steps:
        raise ValueError(
            f"Delay realization requires {buffer_steps} steps, exceeding max_delay_steps={max_delay_steps}."
        )
    phase_error = _fractional_phase_error(
        frequencies[selected], delay, integer, coefficient, dt=float(dt)
    )
    if phase_error > phase_tolerance_degrees:
        raise ValueError(
            f"First-order fractional delay phase error {phase_error:.6g} deg exceeds "
            f"the {phase_tolerance_degrees:.6g} deg limit."
        )

    core = deembed_scattering(scattering, frequencies, delay)
    reconstructed = reembed_scattering(core, frequencies, delay)
    scale = max(float(torch.max(torch.abs(scattering[selected])).item()), 1.0)
    reembedding_error = float(
        torch.max(torch.abs(reconstructed[selected] - scattering[selected])).item()
    ) / scale
    tolerance = 256.0 * torch.finfo(frequencies.dtype).eps
    if reembedding_error > tolerance:
        raise RuntimeError(
            f"Delay deembedding round trip error {reembedding_error:.6g} exceeds {tolerance:.6g}."
        )
    return (
        CompiledNetworkDelay(
            delay_seconds=tuple(float(value) for value in delay.detach().cpu().tolist()),
            integer_steps=tuple(int(value) for value in integer.detach().cpu().tolist()),
            fractional_samples=tuple(float(value) for value in fractional.detach().cpu().tolist()),
            thiran_coefficients=tuple(float(value) for value in coefficient.detach().cpu().tolist()),
            buffer_steps=buffer_steps,
            estimation_rank=estimation_rank,
            equation_count=equation_count,
            residual_seconds=residual_seconds,
            phase_error_degrees=phase_error,
            reembedding_max_error=reembedding_error,
            warnings=warnings,
        ),
        core,
    )


__all__ = [
    "CompiledNetworkDelay",
    "compile_network_delay",
    "deembed_scattering",
    "delay_phase_matrix",
    "estimate_port_delays",
    "reembed_scattering",
]

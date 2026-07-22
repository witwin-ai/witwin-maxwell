"""Pure S-parameter matrix algebra shared by the network container layer.

This module holds the frequency-domain linear algebra of the RF network stack:
Kurokawa power-wave conversion, S/Z/Y representation changes, the general
multiport connection reduction, and single-port termination. It is free of
container, persistence, and metadata concerns; :mod:`witwin.maxwell.network`
owns those and calls into this module.
"""

from __future__ import annotations

import torch


def _validate_complex_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_name: str,
    second_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"{first_name} and {second_name} must be torch.Tensor instances.")
    if not first.is_complex() or not second.is_complex():
        raise TypeError(f"{first_name} and {second_name} must be complex tensors.")
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} must have identical shapes.")
    if first.ndim == 0 or first.numel() == 0:
        raise ValueError(f"{first_name} and {second_name} must be non-empty arrays.")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError(f"{first_name} and {second_name} must have the same device and dtype.")
    return first, second


def _validate_reference_impedance(z0: torch.Tensor, *, name: str = "z0") -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(torch.real(z0)))) or not bool(
        torch.all(torch.isfinite(torch.imag(z0)))
    ):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool(torch.all(torch.real(z0) > 0.0)):
        raise ValueError(f"Re({name}) must be strictly positive.")
    return z0


def _broadcast_reference_impedance(
    z0,
    *,
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(z0, device=device).to(dtype=dtype)
    try:
        value = torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        raise ValueError(f"z0 must be broadcastable to signal shape {tuple(shape)}.") from exc
    return _validate_reference_impedance(value)


def voltage_current_to_power_waves(
    voltage: torch.Tensor,
    current: torch.Tensor,
    z0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert peak-phasor voltage/current to Kurokawa power waves.

    The normalization includes the peak-to-average factor, so ``abs(a)**2`` and
    ``abs(b)**2`` are incident and reflected powers in watts and
    ``abs(a)**2 - abs(b)**2 == 0.5 * Re(V * conj(I))``.
    """

    voltage, current = _validate_complex_pair(
        voltage,
        current,
        first_name="voltage",
        second_name="current",
    )
    reference = _broadcast_reference_impedance(
        z0,
        shape=voltage.shape,
        device=voltage.device,
        dtype=voltage.dtype,
    )
    scale = 2.0 * torch.sqrt(2.0 * torch.real(reference))
    incident = (voltage + reference * current) / scale
    reflected = (voltage - torch.conj(reference) * current) / scale
    return incident, reflected


def power_waves_to_voltage_current(
    a: torch.Tensor,
    b: torch.Tensor,
    z0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Kurokawa power waves to peak-phasor voltage/current."""

    a, b = _validate_complex_pair(a, b, first_name="a", second_name="b")
    reference = _broadcast_reference_impedance(
        z0,
        shape=a.shape,
        device=a.device,
        dtype=a.dtype,
    )
    factor = torch.sqrt(2.0 / torch.real(reference))
    voltage = factor * (torch.conj(reference) * a + reference * b)
    current = factor * (a - b)
    return voltage, current


def _return_loss_db(reflection: torch.Tensor) -> torch.Tensor:
    return -20.0 * torch.log10(torch.abs(reflection))


def _vswr(reflection: torch.Tensor) -> torch.Tensor:
    magnitude = torch.abs(reflection)
    finite_ratio = (1.0 + magnitude) / (1.0 - magnitude)
    return torch.where(
        magnitude < 1.0,
        finite_ratio,
        torch.full_like(magnitude, torch.inf),
    )


def _identity_batch(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    return torch.eye(size, device=matrix.device, dtype=matrix.dtype).expand(
        matrix.shape[:-2] + (size, size)
    )


def _condition_limit(matrix: torch.Tensor) -> float:
    real_dtype = matrix.real.dtype
    return 1.0 / (100.0 * torch.finfo(real_dtype).eps * matrix.shape[-1])


def _validate_solve_matrix(matrix: torch.Tensor, *, operation: str) -> None:
    """Reject singular and numerically unsafe batched network solves."""

    condition = torch.linalg.cond(matrix)
    limit = _condition_limit(matrix)
    invalid = ~torch.isfinite(condition) | (condition > limit)
    if bool(torch.any(invalid)):
        indices = torch.nonzero(invalid, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"{operation} is singular or ill-conditioned at frequency indices {indices}; "
            f"the condition number must be finite and no greater than {limit:.3e}."
        )


def _checked_solve(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    operation: str,
) -> torch.Tensor:
    _validate_solve_matrix(matrix, operation=operation)
    result = torch.linalg.solve(matrix, rhs)
    if not bool(torch.all(torch.isfinite(torch.real(result)))) or not bool(
        torch.all(torch.isfinite(torch.imag(result)))
    ):
        raise RuntimeError(f"{operation} produced non-finite values.")
    return result


def _right_solve(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    operation: str,
) -> torch.Tensor:
    """Return ``left @ inv(right)`` using a batched linear solve."""

    return _checked_solve(
        right.transpose(-2, -1),
        left.transpose(-2, -1),
        operation=operation,
    ).transpose(-2, -1)


def _s_to_z(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    resistance_root = torch.sqrt(torch.real(z0))
    normalized_s = (
        resistance_root.unsqueeze(-1)
        * s
        / resistance_root.unsqueeze(-2)
    )
    identity = _identity_batch(s)
    z0_matrix = torch.diag_embed(z0)
    rhs = torch.diag_embed(torch.conj(z0)) + normalized_s @ z0_matrix
    return _checked_solve(
        identity - normalized_s,
        rhs,
        operation="S/Z conversion",
    )


def _z_to_s(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    z0_matrix = torch.diag_embed(z0)
    normalized_s = _right_solve(
        z - torch.diag_embed(torch.conj(z0)),
        z + z0_matrix,
        operation="Z/S conversion",
    )
    resistance_root = torch.sqrt(torch.real(z0))
    return (
        normalized_s
        * resistance_root.unsqueeze(-2)
        / resistance_root.unsqueeze(-1)
    )


def _s_to_y(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    resistance_root = torch.sqrt(torch.real(z0))
    normalization = resistance_root.unsqueeze(-1) / resistance_root.unsqueeze(-2)
    normalized_s = normalization * s
    identity = _identity_batch(s)
    coefficient = (
        torch.diag_embed(torch.conj(z0))
        + normalized_s @ torch.diag_embed(z0)
    )
    return _checked_solve(
        coefficient,
        identity - normalized_s,
        operation="S/Y conversion",
    )


def _y_to_s(y: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    identity = _identity_batch(y)
    normalized_s = _right_solve(
        identity - torch.diag_embed(torch.conj(z0)) @ y,
        identity + torch.diag_embed(z0) @ y,
        operation="Y/S conversion",
    )
    resistance_root = torch.sqrt(torch.real(z0))
    denormalization = resistance_root.unsqueeze(-2) / resistance_root.unsqueeze(-1)
    return normalized_s * denormalization


def _require_real_reference(z0: torch.Tensor, *, context: str) -> None:
    """Fail closed when a connected reference impedance is not real.

    The traveling-wave junction relation ``a_k = b_l`` used by the S-parameter
    connection algebra is only exact for real reference impedances; complex
    reference impedances must be renormalized to a real reference first.
    """

    tolerance = 1.0e-9 * torch.clamp(torch.abs(torch.real(z0)), min=1.0)
    if bool(torch.any(torch.abs(torch.imag(z0)) > tolerance)):
        raise ValueError(
            f"{context} requires a real reference impedance at the connected ports; "
            "renormalize to a real reference before connecting."
        )


def _connect_s_matrix(
    s: torch.Tensor,
    *,
    external: list[int],
    connected: list[int],
) -> torch.Tensor:
    """Close the ``connected`` port list pairwise and return the external S-block.

    ``connected`` is ordered as consecutive junction pairs ``[k0, l0, k1, l1, ...]``;
    each pair is joined by the traveling-wave relation ``a_k = b_l`` and ``a_l = b_k``.
    Implements the general multiport connection
    ``S'_EE = S_EE + S_EC P (I - S_CC P)^{-1} S_CE`` from first principles.
    """

    if len(connected) % 2 != 0:
        raise ValueError("connected ports must form an even number of junction members.")
    if not external:
        raise ValueError("A connection must leave at least one external port.")

    external_index = torch.tensor(external, device=s.device, dtype=torch.long)
    connected_index = torch.tensor(connected, device=s.device, dtype=torch.long)

    def _block(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return s.index_select(-2, rows).index_select(-1, cols)

    s_ee = _block(external_index, external_index)
    if connected_index.numel() == 0:
        return s_ee
    s_ec = _block(external_index, connected_index)
    s_ce = _block(connected_index, external_index)
    s_cc = _block(connected_index, connected_index)

    nc = connected_index.numel()
    permutation = torch.zeros((nc, nc), device=s.device, dtype=s.dtype)
    for pair in range(nc // 2):
        first = 2 * pair
        second = first + 1
        permutation[first, second] = 1.0
        permutation[second, first] = 1.0

    identity = torch.eye(nc, device=s.device, dtype=s.dtype).expand(s.shape[:-2] + (nc, nc))
    denominator = identity - s_cc @ permutation
    solved = _checked_solve(denominator, s_ce, operation="network connection")
    return s_ee + s_ec @ permutation @ solved


def _reflection_from_impedance(load: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Power-wave reflection coefficient of a load against reference ``z0``."""

    return (load - torch.conj(z0)) / (load + z0)


def _terminate_s_matrix(
    s: torch.Tensor,
    *,
    index: int,
    keep: list[int],
    reflection: torch.Tensor,
) -> torch.Tensor:
    """Reduce out one port closed by ``reflection`` and return the kept S-block.

    Implements ``S'_kk = S_kk + S_kp * gamma * S_pk / (1 - S_pp * gamma)`` for the
    kept port set ``keep`` against terminated port ``index``.
    """

    s_pp = s[:, index, index]
    s_kp = s[:, keep, index]
    s_pk = s[:, index, keep]
    s_kk = s.index_select(-2, torch.tensor(keep, device=s.device)).index_select(
        -1, torch.tensor(keep, device=s.device)
    )
    denom = 1.0 - s_pp * reflection
    if bool(torch.any(torch.abs(denom) < 1.0e3 * torch.finfo(s.real.dtype).eps)):
        raise RuntimeError(
            "termination is singular; (1 - S_pp * gamma) is numerically zero."
        )
    scale = (reflection / denom).unsqueeze(-1).unsqueeze(-1)
    return s_kk + scale * (s_kp.unsqueeze(-1) * s_pk.unsqueeze(-2))


__all__ = [
    "power_waves_to_voltage_current",
    "voltage_current_to_power_waves",
]

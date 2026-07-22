from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import EPSILON_0, MU_0, resolve_real_dtype


class ModeTrackingError(RuntimeError):
    """Raised when adjacent-frequency mode identity is not numerically reliable."""


def _te_characteristic_impedance(omega, beta, mu_r):
    """TE wave impedance ``omega * mu / beta`` for a uniformly filled aperture."""

    return omega * MU_0 * mu_r / beta


def _tm_characteristic_impedance(omega, beta, eps_r):
    """TM wave impedance ``beta / (omega * eps)`` for a uniformly filled aperture."""

    return beta / (omega * EPSILON_0 * eps_r)


def _circuit_characteristic_impedance(definition, voltage, current):
    """Characteristic impedance of a tracked one-watt mode from its V/I integrals.

    ``voltage`` and ``current`` are the modal path integrals of the one-watt
    normalized mode. ``voltage_current`` is ``V/I``; ``power_voltage`` is
    ``|V|^2 / (2P)`` and ``power_current`` is ``2P / |I|^2`` with ``P = 1`` watt.
    """

    if definition == "voltage_current":
        return voltage / current
    if definition == "power_voltage":
        return torch.abs(voltage).square() / 2.0
    return 2.0 / torch.abs(current).square()


@dataclass(frozen=True)
class ModeTrackingResult:
    """Stable mode order, orientation, and diagnostics across frequency.

    ``assignment[f, m]`` is the raw candidate index assigned to stable mode ``m``.
    ``orientation[f]`` maps raw mode rows to the tracked basis. For an ordinary
    mode it contains only its phase correction; a degenerate block may be dense.
    """

    assignment: torch.Tensor
    beta: torch.Tensor
    orientation: torch.Tensor
    phase_factors: torch.Tensor
    overlap: torch.Tensor
    margin: torch.Tensor
    subspace_singular_values: torch.Tensor
    confidence: torch.Tensor
    degenerate: torch.Tensor
    aligned_basis: torch.Tensor | None = None


def _validate_inputs(
    beta: torch.Tensor,
    overlaps: torch.Tensor | None,
    modal_basis: torch.Tensor | None,
) -> tuple[int, int]:
    if not isinstance(beta, torch.Tensor):
        raise TypeError("beta must be a torch.Tensor.")
    if beta.ndim != 2 or beta.shape[0] == 0 or beta.shape[1] == 0:
        raise ValueError("beta must have non-empty shape [F, M].")
    if not (beta.dtype.is_floating_point or beta.is_complex()):
        raise TypeError("beta must use a real or complex floating-point dtype.")
    if not bool(torch.all(torch.isfinite(beta))):
        raise ValueError("beta must contain only finite values.")
    if overlaps is None and modal_basis is None:
        raise ValueError("Pass normalized overlaps or modal_basis.")

    frequency_count, mode_count = beta.shape
    if overlaps is not None:
        if not isinstance(overlaps, torch.Tensor):
            raise TypeError("overlaps must be a torch.Tensor or None.")
        if not overlaps.is_complex():
            raise TypeError("overlaps must be complex.")
        if tuple(overlaps.shape) != (frequency_count - 1, mode_count, mode_count):
            raise ValueError("overlaps must have shape [F - 1, M, M].")
        if overlaps.device != beta.device:
            raise ValueError("beta and overlaps must be on the same device.")
        if resolve_real_dtype(overlaps) != resolve_real_dtype(beta):
            raise TypeError("beta and overlaps must use matching real precision.")
        if not bool(torch.all(torch.isfinite(overlaps))):
            raise ValueError("overlaps must contain only finite values.")
        if not bool(torch.all(torch.abs(overlaps) <= 1.0 + 1.0e-5)):
            raise ValueError("overlaps must be normalized with magnitude no greater than one.")

    if modal_basis is not None:
        if not isinstance(modal_basis, torch.Tensor):
            raise TypeError("modal_basis must be a torch.Tensor or None.")
        if not modal_basis.is_complex():
            raise TypeError("modal_basis must be complex.")
        if modal_basis.ndim < 3 or tuple(modal_basis.shape[:2]) != (
            frequency_count,
            mode_count,
        ):
            raise ValueError("modal_basis must have shape [F, M, ...].")
        if modal_basis.numel() == 0:
            raise ValueError("modal_basis feature dimensions must be non-empty.")
        if modal_basis.device != beta.device:
            raise ValueError("beta and modal_basis must be on the same device.")
        if resolve_real_dtype(modal_basis) != resolve_real_dtype(beta):
            raise TypeError("beta and modal_basis must use matching real precision.")
        if not bool(torch.all(torch.isfinite(modal_basis))):
            raise ValueError("modal_basis must contain only finite values.")
    return frequency_count, mode_count


def _overlaps_from_basis(modal_basis: torch.Tensor) -> torch.Tensor:
    flat = modal_basis.flatten(start_dim=2)
    norm = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    if bool(torch.any(norm <= torch.finfo(norm.dtype).tiny)):
        raise ValueError("modal_basis contains a numerically zero mode.")
    normalized = flat / norm
    return torch.einsum(
        "fmk,fnk->fmn",
        torch.conj(normalized[:-1]),
        normalized[1:],
    )


def _global_assignment(score: torch.Tensor) -> torch.Tensor:
    """Exact maximum-weight assignment using device-resident subset DP."""

    mode_count = score.shape[0]
    if mode_count > 16:
        raise ValueError("Device-resident exact mode assignment supports at most 16 modes.")
    state_count = 1 << mode_count
    masks = torch.arange(state_count, device=score.device, dtype=torch.int64)
    bits = torch.bitwise_left_shift(
        torch.ones((mode_count,), device=score.device, dtype=torch.int64),
        torch.arange(mode_count, device=score.device, dtype=torch.int64),
    )
    negative_infinity = torch.full((), -torch.inf, device=score.device, dtype=score.dtype)
    dynamic = torch.full((state_count,), -torch.inf, device=score.device, dtype=score.dtype)
    dynamic[0] = 0.0
    history = [dynamic]
    for row in range(mode_count):
        next_dynamic = torch.full_like(dynamic, -torch.inf)
        for column in range(mode_count):
            available = torch.bitwise_and(masks, bits[column]) == 0
            targets = torch.bitwise_or(masks, bits[column])
            values = torch.where(available, dynamic + score[row, column], negative_infinity)
            next_dynamic.scatter_reduce_(
                0,
                targets,
                values,
                reduce="amax",
                include_self=True,
            )
        dynamic = next_dynamic
        history.append(dynamic)

    assignment = torch.empty((mode_count,), device=score.device, dtype=torch.int64)
    current_mask = torch.full((), state_count - 1, device=score.device, dtype=torch.int64)
    for row in range(mode_count - 1, -1, -1):
        present = torch.bitwise_and(current_mask, bits) != 0
        previous_masks = torch.bitwise_xor(current_mask, bits)
        candidates = history[row][previous_masks] + score[row]
        selected = torch.argmax(torch.where(present, candidates, negative_infinity))
        assignment[row] = selected
        current_mask = previous_masks[selected]
    return assignment


def _assignment_score(
    previous_beta: torch.Tensor,
    current_beta: torch.Tensor,
    overlap: torch.Tensor,
    *,
    beta_weight: float,
) -> torch.Tensor:
    distance = torch.abs(previous_beta[:, None] - current_beta[None, :])
    scale = torch.maximum(
        torch.abs(previous_beta)[:, None],
        torch.abs(current_beta)[None, :],
    ).clamp_min(torch.finfo(distance.dtype).eps)
    beta_similarity = torch.reciprocal(1.0 + distance / scale)
    return torch.abs(overlap) + beta_weight * beta_similarity


def _degenerate_labels(
    previous_beta: torch.Tensor,
    current_beta: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    mode_count = previous_beta.shape[0]
    previous_distance = torch.abs(previous_beta[:, None] - previous_beta[None, :])
    current_distance = torch.abs(current_beta[:, None] - current_beta[None, :])
    previous_scale = torch.maximum(
        torch.abs(previous_beta)[:, None],
        torch.abs(previous_beta)[None, :],
    ).clamp_min(1.0)
    current_scale = torch.maximum(
        torch.abs(current_beta)[:, None],
        torch.abs(current_beta)[None, :],
    ).clamp_min(1.0)
    connected = (previous_distance <= tolerance * previous_scale) | (
        current_distance <= tolerance * current_scale
    )
    for bridge in range(mode_count):
        connected = connected | (
            connected[:, bridge].unsqueeze(1) & connected[bridge, :].unsqueeze(0)
        )
    indices = torch.arange(mode_count, device=previous_beta.device, dtype=torch.int64)
    sentinel = torch.full_like(indices, mode_count)
    return torch.where(connected, indices.unsqueeze(0), sentinel.unsqueeze(0)).amin(dim=1)


def _index_block(
    matrix: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    row_grid = rows[:, None].expand(rows.numel(), columns.numel()).reshape(-1)
    column_grid = columns[None, :].expand(rows.numel(), columns.numel()).reshape(-1)
    return matrix.index_put((row_grid, column_grid), values.reshape(-1))


def track_modes(
    beta: torch.Tensor,
    overlaps: torch.Tensor | None = None,
    *,
    modal_basis: torch.Tensor | None = None,
    beta_weight: float = 0.05,
    degeneracy_tolerance: float = 1.0e-3,
    min_overlap: float = 0.5,
    min_margin: float = 0.05,
    min_subspace_singular_value: float = 0.5,
) -> ModeTrackingResult:
    """Track candidate waveguide modes across frequency on one torch device.

    ``overlaps[k, i, j]`` is the normalized complex inner product between raw
    candidate ``i`` at frequency ``k`` and raw candidate ``j`` at ``k + 1``.
    Alternatively, pass complex ``modal_basis[F, M, ...]`` and overlaps are
    computed after vector normalization. Low-confidence ordinary matches and
    degenerate subspaces raise :class:`ModeTrackingError`.
    """

    frequency_count, mode_count = _validate_inputs(beta, overlaps, modal_basis)
    for name, value in (
        ("beta_weight", beta_weight),
        ("degeneracy_tolerance", degeneracy_tolerance),
        ("min_overlap", min_overlap),
        ("min_margin", min_margin),
        ("min_subspace_singular_value", min_subspace_singular_value),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    resolved_overlaps = _overlaps_from_basis(modal_basis) if overlaps is None else overlaps
    complex_dtype = resolved_overlaps.dtype
    real_dtype = resolved_overlaps.real.dtype
    device = beta.device
    identity_assignment = torch.arange(mode_count, device=device, dtype=torch.int64)
    identity_orientation = torch.eye(mode_count, device=device, dtype=complex_dtype)
    assignments = [identity_assignment]
    orientations = [identity_orientation]
    phases = [torch.ones((mode_count,), device=device, dtype=complex_dtype)]
    tracked_beta = [beta[0]]
    overlap_diagnostics = []
    margin_diagnostics = []
    singular_diagnostics = []
    confidence_diagnostics = []
    degenerate_diagnostics = []

    for transition in range(frequency_count - 1):
        previous_assignment = assignments[-1]
        previous_orientation = orientations[-1]
        previous_raw_beta = beta[transition]
        current_raw_beta = beta[transition + 1]
        raw_overlap = resolved_overlaps[transition]
        tracked_previous_beta = previous_raw_beta.index_select(0, previous_assignment)
        tracked_overlap = torch.conj(previous_orientation) @ raw_overlap
        score = _assignment_score(
            tracked_previous_beta,
            current_raw_beta,
            tracked_overlap,
            beta_weight=float(beta_weight),
        )
        with torch.no_grad():
            assignment = _global_assignment(score.detach())
        current_tracked_beta = current_raw_beta.index_select(0, assignment)
        labels = _degenerate_labels(
            tracked_previous_beta,
            current_tracked_beta,
            tolerance=float(degeneracy_tolerance),
        )

        assigned_score = score.gather(1, assignment[:, None]).squeeze(1)
        if mode_count == 1:
            margin = torch.full_like(assigned_score, torch.inf)
        else:
            alternatives = score.masked_fill(
                torch.nn.functional.one_hot(assignment, mode_count).to(dtype=torch.bool),
                -torch.inf,
            )
            margin = assigned_score - torch.amax(alternatives, dim=1)

        orientation = torch.zeros((mode_count, mode_count), device=device, dtype=complex_dtype)
        phase_factors = torch.ones((mode_count,), device=device, dtype=complex_dtype)
        aligned_overlap = torch.zeros((mode_count,), device=device, dtype=complex_dtype)
        singular_values = torch.zeros((mode_count,), device=device, dtype=real_dtype)
        confidence = torch.zeros((mode_count,), device=device, dtype=real_dtype)
        degenerate = torch.zeros((mode_count,), device=device, dtype=torch.bool)
        for root in range(mode_count):
            rows = torch.nonzero(labels == root).reshape(-1)
            if rows.numel() == 0:
                continue
            columns = assignment.index_select(0, rows)
            block = tracked_overlap.index_select(0, rows).index_select(1, columns)
            if rows.numel() == 1:
                magnitude = torch.abs(block[0, 0])
                phase = torch.conj(block[0, 0]) / magnitude.clamp_min(
                    torch.finfo(magnitude.dtype).tiny
                )
                block_orientation = phase.reshape(1, 1)
                block_singular_values = magnitude.reshape(1)
            else:
                with torch.no_grad():
                    left, _, right_h = torch.linalg.svd(block.detach(), full_matrices=False)
                    block_orientation = (
                        right_h.mH @ left.mH
                    ).transpose(-2, -1)
                block_singular_values = torch.linalg.svdvals(block)

            oriented = block @ block_orientation.transpose(-2, -1)
            diagonal = torch.diagonal(oriented)
            orientation = _index_block(orientation, rows, columns, block_orientation)
            aligned_overlap = aligned_overlap.index_copy(0, rows, diagonal)
            singular_values = singular_values.index_copy(0, rows, block_singular_values)
            diagonal_orientation = torch.diagonal(block_orientation)
            block_phases = diagonal_orientation / torch.abs(diagonal_orientation).clamp_min(
                torch.finfo(real_dtype).tiny
            )
            phase_factors = phase_factors.index_copy(0, rows, block_phases)

            if rows.numel() == 1:
                row_margin = margin.index_select(0, rows)
                if bool(torch.any(torch.abs(diagonal) < float(min_overlap))) or bool(
                    torch.any(row_margin < float(min_margin))
                ):
                    raise ModeTrackingError(
                        f"Low-confidence mode match at transition {transition} for mode {root}."
                    )
                relative_margin = row_margin / torch.abs(
                    assigned_score.index_select(0, rows)
                ).clamp_min(torch.finfo(real_dtype).eps)
                block_confidence = torch.minimum(
                    torch.abs(diagonal),
                    relative_margin.clamp(0.0, 1.0),
                )
            else:
                degenerate = degenerate.index_fill(0, rows, True)
                subspace_confidence = torch.amin(block_singular_values)
                if bool(subspace_confidence < float(min_subspace_singular_value)):
                    raise ModeTrackingError(
                        f"Low-confidence degenerate subspace at transition {transition} "
                        f"for group rooted at mode {root}."
                    )
                block_confidence = subspace_confidence.expand(rows.numel())
            confidence = confidence.index_copy(0, rows, block_confidence)

        assignments.append(assignment)
        orientations.append(orientation)
        phases.append(phase_factors)
        tracked_beta.append(current_tracked_beta)
        overlap_diagnostics.append(aligned_overlap)
        margin_diagnostics.append(margin)
        singular_diagnostics.append(singular_values)
        confidence_diagnostics.append(confidence)
        degenerate_diagnostics.append(degenerate)

    assignment_tensor = torch.stack(assignments, dim=0)
    orientation_tensor = torch.stack(orientations, dim=0)
    phase_tensor = torch.stack(phases, dim=0)
    beta_tensor = torch.stack(tracked_beta, dim=0)
    empty_complex = torch.empty((0, mode_count), device=device, dtype=complex_dtype)
    empty_real = torch.empty((0, mode_count), device=device, dtype=real_dtype)
    empty_bool = torch.empty((0, mode_count), device=device, dtype=torch.bool)
    aligned_basis = None
    if modal_basis is not None:
        aligned_basis = torch.stack(
            [
                torch.einsum("ij,j...->i...", orientation_tensor[index], modal_basis[index])
                for index in range(frequency_count)
            ],
            dim=0,
        )
    return ModeTrackingResult(
        assignment=assignment_tensor,
        beta=beta_tensor,
        orientation=orientation_tensor,
        phase_factors=phase_tensor,
        overlap=(
            torch.stack(overlap_diagnostics, dim=0)
            if overlap_diagnostics
            else empty_complex
        ),
        margin=torch.stack(margin_diagnostics, dim=0) if margin_diagnostics else empty_real,
        subspace_singular_values=(
            torch.stack(singular_diagnostics, dim=0)
            if singular_diagnostics
            else empty_real
        ),
        confidence=(
            torch.stack(confidence_diagnostics, dim=0)
            if confidence_diagnostics
            else empty_real
        ),
        degenerate=(
            torch.stack(degenerate_diagnostics, dim=0)
            if degenerate_diagnostics
            else empty_bool
        ),
        aligned_basis=aligned_basis,
    )


__all__ = ["ModeTrackingError", "ModeTrackingResult", "track_modes"]

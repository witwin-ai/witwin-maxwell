"""Scalar-operator builders and dense/sparse eigenpair solvers."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import linalg as scipy_linalg
from scipy import sparse

from .common import (
    _LOBPCG_MAX_ITER,
    _LOBPCG_REQUEST_PADDING,
    _LOBPCG_TOL,
)


def _build_scalar_operator(index_sq: np.ndarray, du: float, dv: float):
    nu = int(index_sq.shape[0])
    nv = int(index_sq.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    main_u = np.full((interior_u,), -2.0 / (du * du), dtype=np.float64)
    off_u = np.full((max(interior_u - 1, 0),), 1.0 / (du * du), dtype=np.float64)
    d2_u = sparse.diags((off_u, main_u, off_u), offsets=(-1, 0, 1), format="csr")

    main_v = np.full((interior_v,), -2.0 / (dv * dv), dtype=np.float64)
    off_v = np.full((max(interior_v - 1, 0),), 1.0 / (dv * dv), dtype=np.float64)
    d2_v = sparse.diags((off_v, main_v, off_v), offsets=(-1, 0, 1), format="csr")

    laplacian = sparse.kron(sparse.eye(interior_v, format="csr"), d2_u, format="csr")
    laplacian = laplacian + sparse.kron(d2_v, sparse.eye(interior_u, format="csr"), format="csr")
    potential = sparse.diags(index_sq[1:-1, 1:-1].reshape(-1), offsets=0, format="csr")
    return laplacian + potential, interior_u, interior_v


def _build_scalar_operator_torch_dense(index_sq: torch.Tensor, du: float, dv: float):
    nu = int(index_sq.shape[0])
    nv = int(index_sq.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    dtype = index_sq.dtype
    device = index_sq.device

    main_u = torch.full((interior_u,), -2.0 / (du * du), device=device, dtype=dtype)
    off_u = torch.full((max(interior_u - 1, 0),), 1.0 / (du * du), device=device, dtype=dtype)
    d2_u = torch.diag(main_u)
    if off_u.numel() > 0:
        d2_u = d2_u + torch.diag(off_u, diagonal=1) + torch.diag(off_u, diagonal=-1)

    main_v = torch.full((interior_v,), -2.0 / (dv * dv), device=device, dtype=dtype)
    off_v = torch.full((max(interior_v - 1, 0),), 1.0 / (dv * dv), device=device, dtype=dtype)
    d2_v = torch.diag(main_v)
    if off_v.numel() > 0:
        d2_v = d2_v + torch.diag(off_v, diagonal=1) + torch.diag(off_v, diagonal=-1)

    identity_u = torch.eye(interior_u, device=device, dtype=dtype)
    identity_v = torch.eye(interior_v, device=device, dtype=dtype)
    laplacian = torch.kron(identity_v, d2_u) + torch.kron(d2_v, identity_u)
    potential = torch.diag(index_sq[1:-1, 1:-1].reshape(-1))
    return laplacian + potential, interior_u, interior_v


def _build_scalar_operator_torch_sparse(index_sq: torch.Tensor, du: float, dv: float):
    nu = int(index_sq.shape[0])
    nv = int(index_sq.shape[1])
    interior_u = nu - 2
    interior_v = nv - 2
    unknowns = interior_u * interior_v
    if unknowns <= 0:
        raise ValueError(
            "ModeSource aperture must contain at least one interior node after applying zero boundary conditions."
        )

    device = index_sq.device
    dtype = index_sq.dtype
    flat_indices = torch.arange(unknowns, device=device, dtype=torch.int64).reshape((interior_v, interior_u))
    center = flat_indices.reshape(-1)

    row_chunks = [center]
    col_chunks = [center]
    value_chunks = [
        (
            index_sq[1:-1, 1:-1].reshape(-1)
            + float(-2.0 / (du * du) - 2.0 / (dv * dv))
        ).to(device=device, dtype=dtype)
    ]

    if interior_u > 1:
        lower = flat_indices[:, :-1].reshape(-1)
        upper = flat_indices[:, 1:].reshape(-1)
        coupling = torch.full((lower.numel(),), 1.0 / (du * du), device=device, dtype=dtype)
        row_chunks.extend((lower, upper))
        col_chunks.extend((upper, lower))
        value_chunks.extend((coupling, coupling))

    if interior_v > 1:
        lower = flat_indices[:-1, :].reshape(-1)
        upper = flat_indices[1:, :].reshape(-1)
        coupling = torch.full((lower.numel(),), 1.0 / (dv * dv), device=device, dtype=dtype)
        row_chunks.extend((lower, upper))
        col_chunks.extend((upper, lower))
        value_chunks.extend((coupling, coupling))

    indices = torch.stack((torch.cat(row_chunks), torch.cat(col_chunks)), dim=0)
    values = torch.cat(value_chunks)
    operator = torch.sparse_coo_tensor(indices, values, (unknowns, unknowns), device=device, dtype=dtype)
    return operator.coalesce(), interior_u, interior_v


def _solve_mode_eigenpair(operator, *, mode_index: int):
    dense = operator.toarray()
    eigenvalues, eigenvectors = scipy_linalg.eigh(dense)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    valid = np.where(eigenvalues > 0.0)[0]
    if valid.size <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {valid.size} positive-beta modes were found."
        )
    mode_slot = int(valid[int(mode_index)])
    return float(eigenvalues[mode_slot]), eigenvectors[:, mode_slot]


def _real_profile_from_complex_eigenvector(eigenvector, *, interior_u: int, interior_v: int, shape):
    """Real transverse amplitude of a complex eigenmode, phase-aligned at its peak.

    A complex eigenvector carries an arbitrary global phase; removing the phase at
    the amplitude peak yields the real standing transverse shape used for soft
    injection (the complex propagation constant, not the transverse phase, carries
    the loss). Returns a peak-normalized, sign-oriented float64 numpy array on the
    full aperture grid with zeroed Dirichlet borders.
    """
    peak = int(np.argmax(np.abs(eigenvector)))
    peak_value = eigenvector[peak]
    if abs(peak_value) <= 0.0:
        raise RuntimeError("ModeSource complex eigenmode solve returned a zero profile.")
    aligned = eigenvector * (np.conjugate(peak_value) / abs(peak_value))
    profile = np.zeros(tuple(int(dim) for dim in shape), dtype=np.float64)
    profile[1:-1, 1:-1] = np.real(aligned).reshape((int(interior_u), int(interior_v)))
    scale = float(np.max(np.abs(profile)))
    if scale <= 0.0:
        raise RuntimeError("ModeSource complex eigenmode solve returned a zero profile.")
    profile /= scale
    peak_index = np.unravel_index(np.argmax(np.abs(profile)), profile.shape)
    if profile[peak_index] < 0.0:
        profile *= -1.0
    return profile


def _solve_mode_eigenpair_torch(operator: torch.Tensor, *, mode_index: int):
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    valid = torch.nonzero(eigenvalues > 0.0, as_tuple=False).flatten()
    if valid.numel() <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {int(valid.numel())} positive-beta modes were found."
        )
    mode_slot = int(valid[int(mode_index)].item())
    return eigenvalues[mode_slot], eigenvectors[:, mode_slot]


def _lobpcg_request_count(unknowns: int, *, mode_index: int) -> int:
    requested = max(int(mode_index) + _LOBPCG_REQUEST_PADDING, 1)
    max_requested = max(1, (int(unknowns) - 1) // 3)
    requested = min(requested, max_requested)
    if requested <= int(mode_index):
        raise ValueError(
            "ModeSource mode_index is too large for this aperture's resolution: the sparse "
            "LOBPCG eigensolver needs at least three interior aperture unknowns per requested "
            f"mode to span a valid iteration subspace, but mode_index={mode_index} on an "
            f"aperture with {unknowns} interior unknowns leaves fewer. Refine the aperture grid "
            "or request a lower mode_index."
        )
    return requested


def _lobpcg_initial_guess(interior_u: int, interior_v: int, *, k: int, device, dtype) -> torch.Tensor:
    coord_u = torch.arange(1, interior_u + 1, device=device, dtype=dtype)
    coord_v = torch.arange(1, interior_v + 1, device=device, dtype=dtype)
    basis_pairs: list[tuple[int, int]] = []
    total_order = 2
    while len(basis_pairs) < int(k):
        for order_u in range(1, total_order):
            order_v = total_order - order_u
            basis_pairs.append((order_u, order_v))
            if len(basis_pairs) >= int(k):
                break
        total_order += 1

    vectors = []
    for order_u, order_v in basis_pairs:
        candidate = torch.sin(math.pi * float(order_v) * coord_v / float(interior_v + 1))[:, None]
        candidate = candidate * torch.sin(math.pi * float(order_u) * coord_u / float(interior_u + 1))[None, :]
        vectors.append(candidate.reshape(-1))
    return torch.stack(vectors, dim=1).contiguous()


def _solve_mode_eigenpair_torch_sparse(operator: torch.Tensor, *, interior_u: int, interior_v: int, mode_index: int):
    unknowns = int(operator.shape[0])
    requested = _lobpcg_request_count(unknowns, mode_index=int(mode_index))
    initial_guess = _lobpcg_initial_guess(
        interior_u,
        interior_v,
        k=requested,
        device=operator.device,
        dtype=operator.dtype,
    )
    eigenvalues, eigenvectors = torch.lobpcg(
        operator,
        k=requested,
        X=initial_guess,
        niter=max(_LOBPCG_MAX_ITER, requested * 8),
        tol=_LOBPCG_TOL,
        largest=True,
    )
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    valid = torch.nonzero(eigenvalues > 0.0, as_tuple=False).flatten()
    if valid.numel() <= int(mode_index):
        raise ValueError(
            f"ModeSource requested mode_index={mode_index}, but only {int(valid.numel())} positive-beta modes were found."
        )
    mode_slot = int(valid[int(mode_index)].item())
    return eigenvalues[mode_slot], eigenvectors[:, mode_slot]


def _orient_unit_eigenvector(eigenvector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(eigenvector)
    if float(norm.item()) <= 0.0:
        raise RuntimeError("ModeSource eigenmode solve returned a zero eigenvector.")
    normalized = eigenvector / norm
    peak_index = int(torch.argmax(torch.abs(normalized)).item())
    if float(normalized[peak_index].item()) < 0.0:
        normalized = -normalized
    return normalized


def _sparse_matvec(operator: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector for sparse matvec, got shape {tuple(vector.shape)}.")
    return torch.sparse.mm(operator, vector[:, None]).squeeze(1)


def _project_orthogonal(vector: torch.Tensor, eigenvector: torch.Tensor) -> torch.Tensor:
    return vector - eigenvector * torch.dot(eigenvector, vector)


def _apply_shifted_square_operator(
    operator: torch.Tensor,
    vector: torch.Tensor,
    *,
    eigenvalue: torch.Tensor,
    eigenvector: torch.Tensor,
) -> torch.Tensor:
    shifted = _sparse_matvec(operator, vector) - eigenvalue * vector
    shifted = _sparse_matvec(operator, shifted) - eigenvalue * shifted
    return shifted + eigenvector * torch.dot(eigenvector, vector)


def _conjugate_gradient_solve(
    apply_operator,
    rhs: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
) -> torch.Tensor:
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = torch.dot(r, r)
    rhs_norm = torch.sqrt(rr)
    threshold = max(float(rhs_norm.item()) * float(tol), float(tol))
    if float(rhs_norm.item()) <= threshold:
        return x

    for _ in range(int(max_iter)):
        ap = apply_operator(p)
        denom = torch.dot(p, ap)
        if abs(float(denom.item())) <= 1.0e-30:
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * ap
        r_norm = torch.linalg.vector_norm(r)
        if float(r_norm.item()) <= threshold:
            return x
        rr_new = torch.dot(r, r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return x

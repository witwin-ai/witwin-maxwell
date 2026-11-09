"""GPU-native Krylov engines for the FDFD solver.

cupyx.scipy.sparse.linalg ships only gmres/cg/cgs/minres. This module adds
the short-recurrence engines from the FDFD performance plan (§10.2–10.3):

- ``bicgstab`` — general non-Hermitian, no minimization property.
- ``tfqmr``    — transpose-free quasi-minimal residual (right-preconditioned).
- ``idr_s``    — IDR(s) with biorthogonalization (van Gijzen & Sonneveld).
- ``sqmr``     — simplified QMR for complex-symmetric systems (Freund);
                 valid for the symmetrized UPML assembly where A = A^T.

All engines run entirely on GPU (sparse matvec plus vector work), accept a
CSR matrix or LinearOperator for ``A`` and an optional approximate-inverse
preconditioner ``M`` (applied as ``M @ v``), and follow the cupyx return
convention: ``(x, info)`` with ``info == 0`` on convergence (relative
residual ``<= tol``) and the iteration count otherwise.
"""

from __future__ import annotations

import math

import cupy as cp
import torch


def _matvec(operator):
    if operator is None:
        return lambda v: v
    if hasattr(operator, "matvec"):
        return operator.matvec
    return lambda v: operator @ v


def _bilinear(a, b):
    """Unconjugated bilinear form a^T b used by complex-symmetric methods."""
    return cp.sum(a * b)


def solve(kind, A, b, M=None, tol=1e-6, maxiter=1000, s=4):
    """Dispatch to one of the engines by name ('bicgstab'|'tfqmr'|'idr'|'sqmr')."""
    if kind == "bicgstab":
        return bicgstab(A, b, M=M, tol=tol, maxiter=maxiter)
    if kind == "tfqmr":
        return tfqmr(A, b, M=M, tol=tol, maxiter=maxiter)
    if kind == "idr":
        return idr_s(A, b, M=M, tol=tol, maxiter=maxiter, s=s)
    if kind == "sqmr":
        return sqmr(A, b, M=M, tol=tol, maxiter=maxiter)
    raise ValueError(f"Unknown Krylov engine {kind!r}.")


def bicgstab(A, b, M=None, tol=1e-6, maxiter=1000):
    matvec = _matvec(A)
    psolve = _matvec(M)
    x = cp.zeros_like(b)
    r = b.copy()
    r_shadow = r.copy()
    b_norm = float(cp.linalg.norm(b))
    if b_norm == 0.0:
        return x, 0
    target = tol * b_norm

    rho = alpha = omega = 1.0 + 0.0j
    v = cp.zeros_like(b)
    p = cp.zeros_like(b)
    for iteration in range(1, maxiter + 1):
        rho_next = complex(cp.vdot(r_shadow, r))
        if rho_next == 0.0:
            return x, iteration
        if iteration == 1:
            p = r.copy()
        else:
            beta = (rho_next / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        p_hat = psolve(p)
        v = matvec(p_hat)
        denominator = complex(cp.vdot(r_shadow, v))
        if denominator == 0.0:
            return x, iteration
        alpha = rho_next / denominator
        residual_half = r - alpha * v
        if float(cp.linalg.norm(residual_half)) <= target:
            x += alpha * p_hat
            return x, 0
        s_hat = psolve(residual_half)
        t = matvec(s_hat)
        tt = complex(cp.vdot(t, t))
        if tt == 0.0:
            return x, iteration
        omega = complex(cp.vdot(t, residual_half)) / tt
        x += alpha * p_hat + omega * s_hat
        r = residual_half - omega * t
        if float(cp.linalg.norm(r)) <= target:
            return x, 0
        if omega == 0.0:
            return x, iteration
        rho = rho_next
    return x, maxiter


def sqmr(A, b, M=None, tol=1e-6, maxiter=1000, callback=None):
    """Freund's simplified QMR for complex-symmetric A (A = A^T, bilinear
    inner products, no transpose matvec). M should also be symmetric
    (Jacobi and SSOR of a symmetric matrix are). ``callback``, if given, is
    invoked as ``callback(iteration, residual_norm)`` once per iteration."""
    matvec = _matvec(A)
    psolve = _matvec(M)
    x = cp.zeros_like(b)
    r = b.copy()
    b_norm = float(cp.linalg.norm(b))
    if b_norm == 0.0:
        return x, 0
    target = tol * b_norm

    z = psolve(r)
    tau = float(cp.linalg.norm(r))
    rho = complex(_bilinear(r, z))
    q = z.copy()
    theta = 0.0
    d = cp.zeros_like(b)
    for iteration in range(1, maxiter + 1):
        Aq = matvec(q)
        sigma = complex(_bilinear(q, Aq))
        if sigma == 0.0 or rho == 0.0:
            return x, iteration
        alpha = rho / sigma
        r = r - alpha * Aq
        r_norm = float(cp.linalg.norm(r))
        theta_next = r_norm / tau
        c = 1.0 / (1.0 + theta_next * theta_next) ** 0.5
        tau = tau * theta_next * c
        d = (c * c * theta * theta) * d + (c * c * alpha) * q
        x = x + d
        if callback is not None:
            callback(iteration, r_norm)
        if r_norm <= target:
            return x, 0
        z = psolve(r)
        rho_next = complex(_bilinear(r, z))
        beta = rho_next / rho
        q = z + beta * q
        rho = rho_next
        theta = theta_next
    return x, maxiter


def tfqmr(A, b, M=None, tol=1e-6, maxiter=1000):
    """Right-preconditioned transpose-free QMR with a true-residual check.

    ``maxiter`` counts QMR half-steps. Numerical breakdown returns a finite
    iterate and nonzero information; it never switches iterative engines.
    """
    matvec = _matvec(A)
    psolve = _matvec(M)

    def op(v):
        return matvec(psolve(v))

    b_norm = float(cp.linalg.norm(b))
    if b_norm == 0.0:
        return cp.zeros_like(b), 0
    target = tol * b_norm

    z = cp.zeros_like(b)  # solution of (A M^-1) z = b; x = M^-1 z
    r = b.copy()
    w = r.copy()
    u1 = r.copy()
    r_shadow = r.copy()
    Au1 = op(u1)
    v = Au1.copy()
    d = cp.zeros_like(b)
    tau = float(cp.linalg.norm(r))
    # Store theta**2 * eta as sin(rotation)**2 * alpha. Forming theta
    # explicitly can overflow although the corresponding rotation is bounded.
    direction_weight = 0.0 + 0.0j
    rho = complex(cp.vdot(r_shadow, r))
    iteration = 0
    while iteration < maxiter:
        sigma = complex(cp.vdot(r_shadow, v))
        if (sigma == 0.0 or rho == 0.0
                or not all(math.isfinite(value) for value in
                           (sigma.real, sigma.imag, rho.real, rho.imag))):
            break
        alpha = rho / sigma
        if alpha == 0.0 or not (math.isfinite(alpha.real) and math.isfinite(alpha.imag)):
            break
        u2 = u1 - alpha * v
        Au2 = op(u2)
        stopped = False
        for u, Au in ((u1, Au1), (u2, Au2)):
            if iteration >= maxiter:
                stopped = True
                break
            w = w - alpha * Au
            w_norm = float(cp.linalg.norm(w))
            rotation_norm = math.hypot(tau, w_norm)
            if not math.isfinite(rotation_norm) or rotation_norm == 0.0:
                stopped = True
                break
            c = tau / rotation_norm
            sine = w_norm / rotation_norm
            tau = w_norm * c
            eta_next = c * c * alpha
            d_next = u + (direction_weight / alpha) * d
            z_next = z + eta_next * d_next
            if not bool(cp.all(cp.isfinite(z_next))):
                stopped = True
                break
            d, z = d_next, z_next
            direction_weight = sine * sine * alpha
            iteration += 1
            # This auxiliary bound triggers a final true-residual check.
            if tau * (iteration + 1) ** 0.5 <= target:
                stopped = True
                break
        if stopped:
            break
        rho_next = complex(cp.vdot(r_shadow, w))
        beta = rho_next / rho
        u1 = w + beta * u2
        Au1 = op(u1)
        v = Au1 + beta * (Au2 + beta * v)
        rho = rho_next
    x = psolve(z)
    if not bool(cp.all(cp.isfinite(x))):
        # A nonfinite preconditioner application cannot be a valid solution.
        # The initial zero vector is the last known finite physical iterate.
        return cp.zeros_like(b), max(iteration, 1)
    residual = float(cp.linalg.norm(b - matvec(x)))
    return x, 0 if residual <= target else max(iteration, 1)


def idr_s(A, b, M=None, tol=1e-6, maxiter=1000, s=4, seed=0):
    """IDR(s) with biorthogonalization (van Gijzen & Sonneveld, TOMS 2011).

    ``maxiter`` counts matvecs, comparable with the other engines. The shadow
    space is a fixed orthonormalized complex random block (deterministic
    ``seed``)."""
    matvec = _matvec(A)
    psolve = _matvec(M)
    n = b.size
    x = cp.zeros_like(b)
    r = b.copy()
    b_norm = float(cp.linalg.norm(b))
    if b_norm == 0.0:
        return x, 0
    target = tol * b_norm

    rng = cp.random.default_rng(seed)
    shadow = (rng.standard_normal((n, s), dtype=cp.float32)
              + 1j * rng.standard_normal((n, s), dtype=cp.float32)).astype(b.dtype)
    shadow_torch, _ = torch.linalg.qr(torch.from_dlpack(shadow), mode="reduced")
    shadow = cp.from_dlpack(shadow_torch)

    G = cp.zeros((n, s), dtype=b.dtype)
    U = cp.zeros((n, s), dtype=b.dtype)
    Ms = cp.eye(s, dtype=b.dtype)
    omega = 1.0 + 0.0j
    kappa = 0.7
    iteration = 0

    r_norm = float(cp.linalg.norm(r))
    while r_norm > target and iteration < maxiter:
        f = shadow.conj().T @ r
        for k in range(s):
            c = cp.from_dlpack(torch.linalg.solve(
                torch.from_dlpack(Ms[k:, k:]), torch.from_dlpack(f[k:]),
            ))
            v = r - G[:, k:] @ c
            v = psolve(v)
            U[:, k] = U[:, k:] @ c + omega * v
            G[:, k] = matvec(U[:, k])
            iteration += 1
            for i in range(k):
                alpha = complex(cp.vdot(shadow[:, i], G[:, k])) / complex(Ms[i, i])
                G[:, k] -= alpha * G[:, i]
                U[:, k] -= alpha * U[:, i]
            Ms[k:, k] = shadow[:, k:].conj().T @ G[:, k]
            if complex(Ms[k, k]) == 0.0:
                return x, iteration
            beta = complex(f[k]) / complex(Ms[k, k])
            r = r - beta * G[:, k]
            x = x + beta * U[:, k]
            r_norm = float(cp.linalg.norm(r))
            if r_norm <= target or iteration >= maxiter:
                break
            if k + 1 < s:
                f[k + 1:] = f[k + 1:] - beta * Ms[k + 1:, k]
        if r_norm <= target or iteration >= maxiter:
            break
        # Dimension-reduction step
        v = psolve(r)
        t = matvec(v)
        iteration += 1
        t_norm = float(cp.linalg.norm(t))
        if t_norm == 0.0:
            return x, iteration
        ts = complex(cp.vdot(t, r))
        angle = abs(ts) / (t_norm * r_norm)
        omega = ts / (t_norm * t_norm)
        if angle < kappa and angle > 0.0:
            omega = omega * kappa / angle
        if omega == 0.0:
            return x, iteration
        x = x + omega * v
        r = r - omega * t
        r_norm = float(cp.linalg.norm(r))
    return x, 0 if r_norm <= target else iteration

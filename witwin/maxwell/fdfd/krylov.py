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

import cupy as cp


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


def sqmr(A, b, M=None, tol=1e-6, maxiter=1000):
    """Freund's simplified QMR for complex-symmetric A (A = A^T, bilinear
    inner products, no transpose matvec). M should also be symmetric
    (Jacobi and SSOR of a symmetric matrix are)."""
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
    """Transpose-free QMR (Freund 1993), right-preconditioned so the
    recurrence residual matches the true residual."""
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
    theta = 0.0
    eta = 0.0 + 0.0j
    rho = complex(cp.vdot(r_shadow, r))
    iteration = 0
    while iteration < maxiter:
        sigma = complex(cp.vdot(r_shadow, v))
        if sigma == 0.0 or rho == 0.0:
            break
        alpha = rho / sigma
        u2 = u1 - alpha * v
        Au2 = op(u2)
        converged = False
        for u, Au in ((u1, Au1), (u2, Au2)):
            w = w - alpha * Au
            theta_next = float(cp.linalg.norm(w)) / tau
            c = 1.0 / (1.0 + theta_next * theta_next) ** 0.5
            tau = tau * theta_next * c
            eta_next = c * c * alpha
            d = u + (theta * theta * eta / alpha) * d
            z = z + eta_next * d
            theta = theta_next
            eta = eta_next
            iteration += 1
            # tau * sqrt(iteration + 1) bounds the true residual
            if tau * (iteration + 1) ** 0.5 <= target:
                converged = True
                break
        if converged:
            break
        rho_next = complex(cp.vdot(r_shadow, w))
        beta = rho_next / rho
        u1 = w + beta * u2
        Au1 = op(u1)
        v = Au1 + beta * (Au2 + beta * v)
        rho = rho_next
    x = psolve(z)
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
    shadow, _ = cp.linalg.qr(shadow)

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
            c = cp.linalg.solve(Ms[k:, k:], f[k:])
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

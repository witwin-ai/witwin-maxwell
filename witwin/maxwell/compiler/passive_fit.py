"""Prepare-time nonnegative fits on caller-sampled fixed Foster/RL columns."""

from __future__ import annotations

import numpy as np


def fit_passive_foster(
    frequencies, target_complex, poles, weights=None, *, fixed_terms=None, equalities=None,
):
    """Return nonnegative coefficients shared across all response channels.

    ``poles`` contains sampled complex fixed-pole columns, shaped ``[F, ..., K]``;
    ``target_complex`` and optional additive ``fixed_terms`` have shape ``[F, ...]``.
    Sampling stays with the caller to preserve its pole normalization, convention,
    and arithmetic order. A column may include a paired pole in each channel or a
    nonnegative direct term. ``weights`` multiplies each complex residual before
    stacking all real rows followed by all imaginary rows, in frequency order.
    Callers with preweighted columns/targets can omit it.

    ``equalities=(A, b)`` imposes ``A @ coefficients == b`` using an augmented
    NNLS seed followed by the constrained least-squares solve. Fixed terms (such
    as an exact integrator) are subtracted before fitting. Band certification and
    any active-set cleanup remain with the caller.
    """
    from scipy.optimize import minimize, nnls

    columns = np.asarray(poles)
    target = np.asarray(target_complex)
    if fixed_terms is not None:
        target = target - fixed_terms
    if weights is not None:
        columns = columns * np.asarray(weights)[..., None]
        target = target * weights
    count = columns.shape[-1]
    rows = len(frequencies) * int(np.prod(target.shape[1:]))
    matrix = np.concatenate((columns.real.reshape(rows, count),
                             columns.imag.reshape(rows, count)))
    rhs = np.r_[target.real.reshape(-1), target.imag.reshape(-1)]
    if equalities is None:
        return nnls(matrix, rhs)[0]
    equality, moments = equalities
    initial = nnls(np.vstack((matrix, equality)), np.r_[rhs, moments])[0]
    solution = minimize(lambda x: .5 * np.sum((matrix @ x - rhs)**2), initial,
        jac=lambda x: matrix.T @ (matrix @ x - rhs), method="SLSQP",
        bounds=[(0., None)] * count,
        constraints=[dict(type="eq", fun=lambda x: equality @ x - moments,
                          jac=lambda x: equality)], options=dict(ftol=1e-15, maxiter=1000))
    return solution.x

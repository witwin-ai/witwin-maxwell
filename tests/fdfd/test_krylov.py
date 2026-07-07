import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpl
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdfd import krylov

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _manufactured(symmetric):
    rng = np.random.default_rng(0)
    n = 2000
    base = sp.random(n, n, density=0.002, random_state=1, dtype=np.float64)
    if symmetric:
        core = (base + base.T) * 0.5
    else:
        core = base + 0.3 * sp.random(n, n, density=0.002, random_state=2, dtype=np.float64)
    matrix = (core + sp.eye(n) * 8.0).astype(np.complex64) + 1j * 0.5 * sp.eye(n, dtype=np.complex64)
    A = cpsp.csr_matrix(sp.csr_matrix(matrix))
    b = (cp.asarray(rng.standard_normal(n), dtype=cp.float32)
         + 1j * cp.asarray(rng.standard_normal(n), dtype=cp.float32)).astype(cp.complex64)
    return A, b


def _jacobi(A):
    inv_diag = cp.reciprocal(A.diagonal())
    return cpl.LinearOperator(A.shape, matvec=lambda v: inv_diag * v, dtype=A.dtype)


@requires_cuda
@pytest.mark.parametrize("engine", ["bicgstab", "tfqmr", "idr"])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_general_engines_converge_on_nonsymmetric_system(engine, preconditioned):
    A, b = _manufactured(symmetric=False)
    M = _jacobi(A) if preconditioned else None
    x, _info = krylov.solve(engine, A, b, M=M, tol=1e-6, maxiter=2000)
    residual = float(cp.linalg.norm(A @ x - b) / cp.linalg.norm(b))
    assert residual < 1e-5


@requires_cuda
@pytest.mark.parametrize("preconditioned", [False, True])
def test_sqmr_converges_on_complex_symmetric_system(preconditioned):
    A, b = _manufactured(symmetric=True)
    M = _jacobi(A) if preconditioned else None
    x, info = krylov.sqmr(A, b, M=M, tol=1e-6, maxiter=2000)
    residual = float(cp.linalg.norm(A @ x - b) / cp.linalg.norm(b))
    assert info == 0
    assert residual < 1e-5


@requires_cuda
@pytest.mark.parametrize("engine", ["bicgstab", "tfqmr", "idr", "sqmr"])
def test_engines_run_through_public_fdfd_path(engine):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.16, 0.16), (-0.16, 0.16), (-0.16, 0.16))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=4.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
            name="src",
        )
    )
    result = mw.Simulation.fdfd(
        scene,
        frequency=1e9,
        solver=mw.GMRES(max_iter=400, tol=1e-6, solver_type=engine),
    ).run()
    stats = result.stats()
    assert stats["solver"]["type"] == engine
    assert stats["final_residual"] is not None
    for name in ("EX", "EY", "EZ"):
        assert torch.isfinite(result.fields[name].abs()).all()

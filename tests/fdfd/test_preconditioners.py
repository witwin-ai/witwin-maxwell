import cupy as cp
import pytest
import torch

import witwin.maxwell as mw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9


def _solver(preconditioner):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1e6),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            width=0.05,
            source_time=mw.CW(frequency=FREQ, amplitude=100.0),
            name="src",
        )
    )
    return mw.Simulation.fdfd(
        scene,
        frequency=FREQ,
        solver=mw.GMRES(max_iter=100, tol=1e-6, restart=25, preconditioner=preconditioner),
    ).prepare().solver


@requires_cuda
@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "ssor", "ilu"])
def test_preconditioner_apply_is_finite_and_typed(preconditioner):
    solver = _solver(preconditioner)
    solver._ensure_system_matrix()
    operator = solver._ensure_preconditioner()
    if preconditioner == "none":
        assert operator is None
        return
    v = cp.ones(solver.A_matrix.shape[0], dtype=cp.complex64)
    result = operator.matvec(v)
    assert result.dtype == cp.complex64
    assert bool(cp.isfinite(result).all())
    # Cached per matrix
    assert solver._ensure_preconditioner() is operator


@requires_cuda
@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "ssor"])
def test_solve_runs_with_each_preconditioner(preconditioner):
    solver = _solver(preconditioner)
    solver.solve(max_iter=100, tol=1e-6, restart=25)
    assert solver.E_field is not None
    for component in solver.E_field:
        assert torch.isfinite(component.abs()).all()


@requires_cuda
def test_rejects_unknown_preconditioner_and_solver_type():
    with pytest.raises(ValueError):
        _solver("cholesky")
    scene_solver = _solver("jacobi")
    from witwin.maxwell.fdfd import FDFD

    with pytest.raises(ValueError):
        FDFD(scene_solver.scene, frequency=FREQ, solver_type="bicgstab")

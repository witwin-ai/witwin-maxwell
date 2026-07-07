import cupy as cp
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdfd import ams

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _solver():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=4.0),
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
    return mw.Simulation.fdfd(
        scene,
        frequency=1e9,
        solver=mw.GMRES(solver_type="gmres", preconditioner="ams", precision="double"),
    ).prepare().solver


@requires_cuda
def test_gradient_space_is_exact_null_space_in_interior():
    # A (P phi) = k0^2 eps (P phi) holds exactly away from the outer wall;
    # the outermost cell layer carries an inherent truncation defect.
    solver = _solver()
    solver._ensure_system_matrix()
    pre = ams.AMSPreconditioner(solver)
    A = solver._iteration_matrix()
    eps_face = ams._eps_face_vector(solver, A.dtype)
    rng = cp.random.default_rng(0)
    n_nodes = pre.P_grad.shape[1]
    phi = (rng.standard_normal(n_nodes) + 1j * rng.standard_normal(n_nodes)).astype(A.dtype)
    u = pre.P_grad @ phi
    defect = (A @ u) - solver.k0**2 * eps_face * u

    scene = solver.scene
    shapes = (
        (scene.Nx_ex, scene.Ny_ex, scene.Nz_ex),
        (scene.Nx_ey, scene.Ny_ey, scene.Nz_ey),
        (scene.Nx_ez, scene.Ny_ez, scene.Nz_ez),
    )
    offsets = (0, scene.N_ex, scene.N_ex + scene.N_ey)
    interior_sq = 0.0
    for c in range(3):
        count = shapes[c][0] * shapes[c][1] * shapes[c][2]
        block = defect[offsets[c]:offsets[c] + count].reshape(shapes[c])
        interior_sq += float(cp.linalg.norm(block[1:-1, 1:-1, 1:-1])) ** 2
    relative = interior_sq**0.5 / float(cp.linalg.norm(A @ u))
    assert relative < 1e-5


@requires_cuda
def test_ams_apply_is_symmetric_and_finite():
    solver = _solver()
    solver._ensure_system_matrix()
    operator = solver._ensure_preconditioner()
    n = operator.shape[0]
    rng = cp.random.default_rng(1)
    u = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(cp.complex128)
    v = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(cp.complex128)
    Mu = operator.matvec(u)
    Mv = operator.matvec(v)
    assert bool(cp.isfinite(Mu).all()) and bool(cp.isfinite(Mv).all())
    lhs = complex(cp.sum(u * Mv))
    rhs = complex(cp.sum(v * Mu))
    # low-order noise from the complex64-sourced PML/scale weights
    assert abs(lhs - rhs) / abs(lhs) < 5e-6

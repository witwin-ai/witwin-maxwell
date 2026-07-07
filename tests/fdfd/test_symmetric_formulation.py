import cupy as cp
import pytest
import torch

import witwin.maxwell as mw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9
RES = 0.02


def _vacuum_scene(cells_half, strength=4.0):
    half = cells_half * RES
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(RES),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=strength),
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
    return scene


def _direct_solver(scene):
    return mw.Simulation.fdfd(
        scene, frequency=FREQ, solver=mw.GMRES(solver_type="direct")
    ).prepare().solver


@requires_cuda
def test_assembled_matrix_is_complex_symmetric():
    solver = _direct_solver(_vacuum_scene(12, strength=1e6))
    solver._ensure_system_matrix()
    A = solver.A_matrix
    asymmetry = A - A.transpose().tocsr()
    relative = float(cp.abs(asymmetry.data).max() / cp.abs(A.data).max()) if asymmetry.nnz else 0.0
    assert relative < 1e-6


@requires_cuda
def test_solve_adjoint_satisfies_transpose_system():
    solver = _direct_solver(_vacuum_scene(12))
    solver.solve()
    scale = solver._ensure_symmetrization_scale()
    rng = cp.random.default_rng(0)
    rhs = (rng.standard_normal(solver.A_matrix.shape[0], dtype=cp.float32)
           + 1j * rng.standard_normal(solver.A_matrix.shape[0], dtype=cp.float32)).astype(cp.complex64)
    lam = solver.solve_adjoint(rhs)
    # A^T lam = rhs  <=>  S (lam/d) = rhs/d with S = D A D^-1 symmetric
    residual = float(
        cp.linalg.norm(solver.A_matrix @ (lam / scale) - rhs / scale) / cp.linalg.norm(rhs / scale)
    )
    assert residual < 1e-3


@requires_cuda
def test_interior_matches_large_domain_reference():
    # PML quality: the interior of a small domain must match the same window
    # of a domain twice as large (whose walls are too far to contaminate it).
    reference = _direct_solver(_vacuum_scene(24))
    reference.solve()
    ref_ez = reference.E_field[2].clone()
    reference._release_direct_solver()

    small = _direct_solver(_vacuum_scene(12))
    small.solve()
    ez = small.E_field[2]

    margin = 10  # 8 PML layers + 2 cells
    offset = 12  # center alignment between the 24- and 48-cell grids
    window = ez[margin:-margin, margin:-margin, margin:-margin]
    ref_window = ref_ez[
        margin + offset:-(margin + offset),
        margin + offset:-(margin + offset),
        margin + offset:-(margin + offset),
    ]
    assert window.shape == ref_window.shape
    error = float(torch.linalg.vector_norm(window - ref_window) / torch.linalg.vector_norm(ref_window))
    # The pre-symmetrization PML discretization measured ~24% here; the
    # symmetric UPML stencil measures ~2% on this small window (0.4% on a
    # 32^3-vs-64^3 pairing). 5% catches any regression of that class.
    assert error < 0.05, error

import pytest
import torch

import witwin.maxwell as mw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9


def _dipole(name, position=(0.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0)):
    return mw.PointDipole(
        position=position,
        polarization=polarization,
        width=0.05,
        source_time=mw.CW(frequency=FREQ, amplitude=100.0),
        name=name,
    )


def _direct_solver():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1e6),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="cube",
            geometry=mw.Box(position=(0.0, 0.05, 0.0), size=(0.1, 0.1, 0.1)),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(_dipole("src"))
    return mw.Simulation.fdfd(
        scene,
        frequency=FREQ,
        solver=mw.GMRES(solver_type="direct"),
    ).prepare().solver


@requires_cuda
def test_direct_solve_is_accurate_and_caches_factorization():
    solver = _direct_solver()
    solver.solve()
    assert solver.converged
    assert solver.final_residual < 1e-4
    for component in solver.E_field:
        assert component.is_cuda
        assert torch.isfinite(component.abs()).all()
    assert solver._direct_solver is not None


@requires_cuda
def test_direct_factorization_reused_across_sources():
    solver = _direct_solver()
    solver.solve()
    factorization = solver._direct_solver
    first_ez = solver.E_field[2].clone()

    solver.scene.sources = [_dipole("src2", position=(0.05, 0.0, 0.0), polarization=(1.0, 0.0, 0.0))]
    solver.solve()
    assert solver._direct_solver is factorization
    assert solver.final_residual < 1e-4
    assert not torch.allclose(solver.E_field[2], first_ez)

    fresh = _direct_solver()
    fresh.scene.sources = [_dipole("src2", position=(0.05, 0.0, 0.0), polarization=(1.0, 0.0, 0.0))]
    fresh.solve()
    for reused, reference in zip(solver.E_field, fresh.E_field):
        # Iterative refinement warm-starts from the previous solution, so the
        # reused-factorization solve differs from a fresh solve by solver
        # tolerance, not bitwise: compare in norm.
        error = torch.linalg.vector_norm(reused - reference) / torch.linalg.vector_norm(reference)
        assert float(error) < 1e-3


@requires_cuda
def test_direct_solutions_do_not_alias_previous_fields():
    solver = _direct_solver()
    solver.solve()
    first_ez = solver.E_field[2]
    snapshot = first_ez.clone()

    solver.scene.sources = [_dipole("src2", position=(0.05, 0.0, 0.0), polarization=(1.0, 0.0, 0.0))]
    solver.solve()
    assert torch.equal(first_ez, snapshot)


@requires_cuda
def test_set_frequency_releases_direct_factorization():
    solver = _direct_solver()
    solver.solve()
    assert solver._direct_solver is not None
    solver.set_frequency(1.5e9)
    assert solver._direct_solver is None
    solver.solve()
    assert solver._direct_solver is not None
    assert solver.final_residual < 1e-4

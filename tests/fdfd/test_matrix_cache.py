import pytest
import torch

import witwin.maxwell as mw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9


def _build_scene(material=None):
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
            material=material or mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(_dipole("src"))
    return scene


def _dipole(name, position=(0.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0)):
    return mw.PointDipole(
        position=position,
        polarization=polarization,
        width=0.05,
        source_time=mw.CW(frequency=FREQ, amplitude=100.0),
        name=name,
    )


def _solver(scene, frequency=FREQ):
    return mw.Simulation.fdfd(
        scene,
        frequency=frequency,
        solver=mw.GMRES(max_iter=100, tol=1e-6, restart=25),
    ).prepare().solver


@requires_cuda
def test_solve_reuses_matrix_across_sources():
    solver = _solver(_build_scene())
    solver.solve(max_iter=100, tol=1e-6, restart=25)
    matrix_first = solver.A_matrix
    assert matrix_first is not None

    solver.scene.sources = [_dipole("src2", position=(0.05, 0.0, 0.0), polarization=(1.0, 0.0, 0.0))]
    solver.solve(max_iter=100, tol=1e-6, restart=25)
    assert solver.A_matrix is matrix_first

    # The cached-matrix solve matches a fresh solver on the same scene.
    fresh = _solver(_build_scene())
    fresh.scene.sources = [_dipole("src2", position=(0.05, 0.0, 0.0), polarization=(1.0, 0.0, 0.0))]
    fresh.solve(max_iter=100, tol=1e-6, restart=25)
    for cached, reference in zip(solver.E_field, fresh.E_field):
        assert torch.allclose(cached, reference, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_set_frequency_rebuilds_matrix_and_reuses_nondispersive_materials():
    solver = _solver(_build_scene())
    solver.solve(max_iter=50, tol=1e-6, restart=25)
    matrix_first = solver.A_matrix
    eps_first = solver.material_eps_components

    solver.set_frequency(1.5e9)
    assert solver.A_matrix is None
    assert solver.material_eps_components is eps_first  # non-dispersive: reused

    solver.solve(max_iter=50, tol=1e-6, restart=25)
    assert solver.A_matrix is not None
    assert solver.A_matrix is not matrix_first

    fresh = _solver(_build_scene(), frequency=1.5e9)
    fresh.solve(max_iter=50, tol=1e-6, restart=25)
    for swept, reference in zip(solver.E_field, fresh.E_field):
        assert torch.allclose(swept, reference, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_set_frequency_recompiles_dispersive_materials():
    material = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1e-10),),
    )
    solver = _solver(_build_scene(material=material))
    solver.solve(max_iter=50, tol=1e-6, restart=25)
    eps_first = solver.material_eps_components
    assert eps_first is not None

    solver.set_frequency(1.5e9)
    assert solver.material_eps_components is None

    solver.solve(max_iter=50, tol=1e-6, restart=25)
    assert solver.material_eps_components is not eps_first

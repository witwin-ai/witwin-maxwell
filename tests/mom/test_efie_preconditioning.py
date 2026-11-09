"""Buffa-Christiansen duals and the Calderon EFIE preconditioner on open sheets (CUDA)."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def dual_for(mesh, rule='natural'):
    from witwin.maxwell.mom.efie_preconditioning import DualBasis
    from witwin.maxwell.mom.open_surface import OpenRWGBasis
    return DualBasis(OpenRWGBasis(mesh), open_edge_rule=rule)


@pytest.mark.parametrize('kind, rule', [('plate', 'natural'), ('disk', 'natural'), ('sphere', 'natural'),
                                        ('plate', 'zero_flux')])
def test_bc_functions_carry_uniform_unit_cell_charges_and_dual_loops_are_solenoidal(kind, rule):
    from witwin.maxwell.mom.efie_preconditioning import dual_star_incidence
    from witwin.maxwell.mom.mesh import sphere_mesh
    from witwin.maxwell.mom.open_surface import OpenSurfaceMesh, disk_mesh, rectangular_plate_mesh
    if kind == 'plate':
        mesh = rectangular_plate_mesh((1.0, 0.7), (5, 4))
    elif kind == 'disk':
        mesh = disk_mesh(0.3, 3)
    else:
        closed = sphere_mesh(0.2, 1)
        mesh = OpenSurfaceMesh(closed.vertices, closed.faces)
    dual = dual_for(mesh, rule)
    charge = dual.divergence.to_dense()*dual.bary.mesh.areas[:, None]  # [bary faces, BC]
    children = dual.bary.mesh.faces
    owner = torch.where(children[:, 0] < len(mesh.vertices), children[:, 0], children[:, 1])
    cell_charge = torch.zeros((len(mesh.vertices), dual.n_functions), device=charge.device, dtype=charge.dtype)
    cell_charge.index_add_(0, owner, charge)
    # Natural rule: boundary-vertex cells are uncharged; zero flux: every cell holds +-1.
    expected = dual_star_incidence(dual.primal, rule).T
    torch.testing.assert_close(cell_charge, expected, rtol=0, atol=1e-12)
    # Each fan triangle holds the same share of its cell charge.
    fan = torch.bincount(owner, minlength=len(mesh.vertices)).to(charge.dtype)
    torch.testing.assert_close(charge, expected[owner]/fan[owner, None], rtol=0, atol=1e-12)
    # Oriented sums of BC functions around a primal face with three interior
    # edges (dual loops) have no charge; boundary faces lack one BC function.
    from witwin.maxwell.mom.efie_preconditioning import primal_star_incidence
    interior = (dual.primal.face_signs != 0).all(-1)
    loops = primal_star_incidence(dual.primal)[:, interior]
    assert loops.shape[1] > 0
    assert float((charge@loops).abs().max()) < 1e-12


def test_mixed_gram_is_well_conditioned_and_matches_rotation():
    from witwin.maxwell.mom.efie_preconditioning import mixed_gram
    from witwin.maxwell.mom.open_surface import rectangular_plate_mesh
    for cells in (4, 8, 16):
        dual = dual_for(rectangular_plate_mesh((1.0, 1.0), (cells, cells)))
        gram = mixed_gram(dual)
        singular = torch.linalg.svdvals(gram)
        # b_e ~ n x f_e, so G = <f, n x b> ~ -<f, f>: negative diagonal, bounded condition.
        assert bool((torch.diagonal(gram) < 0).all())
        assert float(singular[0]/singular[-1]) < 30


def run(scene, frequency, side, cells, factory, *, rtol=1e-6, maxiter=1200):
    stopping = StoppingRule(method='gmres', rtol=rtol, maxiter=maxiter, restart=maxiter, preconditioner='none',
                            stagnation_window=maxiter)
    config = mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=side/cells), stopping=stopping,
                          right_preconditioner=factory)
    return mw.Simulation.mom(scene, frequency=frequency, config=config).run()


def plate(side, frequency):
    return mw.Scene(structures=[mw.Structure(mw.Box(size=(side, side, 0)), mw.Material.pec())],
                    sources=[mw.PlaneWave(direction=(0.0, 0.6, -0.8), polarization=(1.0, 0.0, 0.0),
                                          source_time=mw.CW(frequency=frequency))], device='cuda')


def test_calderon_iterations_stay_flat_under_refinement_and_match_the_direct_solution():
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner, preconditioned_residual
    frequency = 3e8
    side = C_0/frequency
    counts = {}
    for cells in (6, 12):
        plain = run(plate(side, frequency), frequency, side, cells, None)
        preconditioned = run(plate(side, frequency), frequency, side, cells, CalderonPreconditioner())
        assert preconditioned.solver_stats['converged']
        counts[cells] = (plain.solver_stats['iterations'], preconditioned.solver_stats['iterations'])
        error = (plain.surface_currents.J-preconditioned.surface_currents.J).norm()/plain.surface_currents.J.norm()
        assert float(error) < 1e-4
        runtime = preconditioned.solver
        report = preconditioned_residual(runtime.handle, preconditioned.surface_currents.J, runtime.rhs,
                                         runtime.right_preconditioner.apply)
        assert report['true_relative_residual'] <= 1e-6 and report['preconditioned_relative_residual'] > 0
        assert preconditioned.solver_stats['right_preconditioner']['version'].startswith('open_efie_calderon')
    assert counts[12][1] < counts[12][0]
    assert counts[12][1] <= counts[6][1]+6  # natural open-edge rule: flat under refinement


def test_calderon_stays_accurate_and_flat_at_low_frequency():
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner
    from witwin.maxwell.mom.linalg import solve_reference
    side, cells = 0.1, 10
    for size in (0.1, 0.01):
        frequency = size*C_0/side
        result = run(plate(side, frequency), frequency, side, cells, CalderonPreconditioner())
        runtime = result.solver
        reference, _ = solve_reference(runtime.handle, runtime.rhs, StoppingRule(method='direct'))
        error = (result.surface_currents.J-reference).norm()/reference.norm()
        assert result.solver_stats['converged'] and result.solver_stats['iterations'] <= 25
        assert float(error) < 1e-4


def test_open_surface_preconditioners_reject_the_pmchwt_layout():
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=0.1), mw.Material(eps_r=2))],
                     sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8))], device='cuda')
    config = mw.MoMConfig(stopping=StoppingRule(method='gmres'), right_preconditioner=CalderonPreconditioner())
    with pytest.raises(ValueError, match='single-current'):
        mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()


def test_calderon_rejects_multi_sheet_scenes_explicitly():
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner
    frequency = 3e8
    side = 0.4*C_0/frequency
    plates = [mw.Structure(mw.Box(position=(0.0, 0.0, z), size=(side, side, 0)), mw.Material.pec())
              for z in (0.075*C_0/frequency, -0.075*C_0/frequency)]
    scene = mw.Scene(structures=plates, sources=[mw.PlaneWave(direction=(0.0, 0.0, -1.0), polarization=(1.0, 0.0, 0.0),
                                                              source_time=mw.CW(frequency=frequency))], device='cuda')
    config = mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=side/6),
                          stopping=StoppingRule(method='gmres'), right_preconditioner=CalderonPreconditioner())
    with pytest.raises(NotImplementedError, match='exactly one sheet; this scene has 2'):
        mw.Simulation.mom(scene, frequency=frequency, config=config).prepare()

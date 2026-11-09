"""Small real-operator interaction control for combined native storage changes."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference, residual_loss
from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs
from witwin.maxwell.mom.fields import decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compact_real_eight_rhs_restore_repeat_fields_gradient_and_persistence(tmp_path):
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.01))],
                     sources=[mw.PlaneWave()])
    rule = StoppingRule(method='direct', direct_storage='sealed_host_restore_v1', rtol=1e-10)
    config = mw.MoMConfig(subdivisions=0, riesz_storage='compact_rwg_v1', stopping=rule,
                         integration=IntegrationConfig(order=3, singular_order=4))
    prepared = mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()
    handle = prepared.solver.handle
    original = handle.matrix.clone(); gram = handle.test_riesz.blocks[0].gram.clone()
    pointer = handle.matrix.data_ptr(); factor = handle._riesz_factor
    result = prepared.run()
    assert result.solver_stats['converged'] and result.solver_stats['direct_storage']['matrix_restored_exact']
    orientations = [((0,0,1),(1,0,0)), ((0,0,1),(0,1,0)), ((0,0,-1),(1,0,0)),
                    ((0,0,-1),(0,1,0)), ((1,0,0),(0,1,0)), ((1,0,0),(0,0,1)),
                    ((0,1,0),(1,0,0)), ((0,1,0),(0,0,1))]
    sources = [mw.PlaneWave(direction=d, polarization=p, source_time=mw.CW(frequency=3e8, phase=.17*i))
               for i,(d,p) in enumerate(orientations)]
    rhs = torch.stack([build_rhs(handle.basis, handle.exterior, [source], order=3) for source in sources])
    oracle = torch.linalg.solve(original, rhs.mT).mT
    for repeat in range(2):
        solved, log = solve_reference(handle, rhs, rule)
        assert log['converged'] and log['rhs_count']==8 and all(row['converged'] for row in log['rhs_logs'])
        assert handle.matrix.data_ptr()==pointer and torch.equal(handle.matrix, original)
        assert torch.equal(handle.test_riesz.blocks[0].gram, gram) and handle._riesz_factor is factor
        assert handle._lu is None and handle._direct_storage_backup is None
        assert log['direct_storage']['matrix_restored_exact'] and not log['direct_storage']['backup_descriptor_open']
        torch.testing.assert_close(solved, oracle, rtol=2e-11, atol=2e-12)
        actual = handle.apply(solved)
        assert bool((torch.linalg.vector_norm(actual-rhs, dim=-1)/torch.linalg.vector_norm(rhs, dim=-1)<1e-10).all())
        torch.testing.assert_close(handle.adjoint(solved), solved@original.conj(), rtol=2e-12, atol=2e-12)
    points = torch.tensor([[.3,.2,.4]], dtype=torch.float64, device='cuda')
    for actual, expected in zip(decode(handle, solved[0], points, 'exterior', 'scattered'),
                                decode(handle, oracle[0], points, 'exterior', 'scattered')):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    before_fields = result.query_fields(points, region_id='exterior', field_kind='total')
    path = tmp_path/'combined-sealed-compact.pt';result.save(path)
    loaded = mw.Result.load(path, scene=scene)
    assert loaded.surface_currents.config.stopping.direct_storage=='sealed_host_restore_v1'
    assert loaded.surface_currents.config.riesz_storage=='compact_rwg_v1'
    assert loaded.surface_currents.handle.metadata['riesz_storage']=='block_compact_rwg_v1'
    torch.testing.assert_close(loaded.surface_currents.scattered_coefficients, result.surface_currents.scattered_coefficients)
    for actual, expected in zip(loaded.query_fields(points, region_id='exterior', field_kind='total'), before_fields):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    x = (solved*1.01).detach().requires_grad_()
    gradient, = torch.autograd.grad(residual_loss(handle, x, rhs), x)
    r = x.detach()@original.mT-rhs
    metric = torch.block_diag(gram/handle.test_riesz.blocks[0].impedance_squared, gram).to(original.dtype)
    expected_gradient = torch.linalg.solve(metric, r.mT).mT@original.conj()
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-11, atol=2e-12)
    assert torch.equal(handle.matrix, original) and torch.equal(handle.test_riesz.blocks[0].gram, gram) and handle._lu is None

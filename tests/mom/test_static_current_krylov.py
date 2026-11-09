"""DC nonlinear roots and material gradients without global tangent matrices."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.static_current import (
    NodalCurrentBalance,FaceCurrentBalance,StaticCurrentConvergenceError,
    FixedStaticCurrentInverse,StaticTangentConvergenceError)
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('kind',[NodalCurrentBalance,FaceCurrentBalance])
def test_dc_krylov_root_and_parameter_gradients(kind,monkeypatch):
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block=kind(volume)
    amplitude=volume.vertices.new_tensor(.3,requires_grad=True)
    sigma=volume.vertices.new_tensor(.2,requires_grad=True)
    axis=volume.vertices.new_tensor([1.,.1,-.2])
    ac=torch.stack((amplitude.new_tensor(1.4),amplitude))[:,None,None]*axis[None,None,:]
    ac=ac.expand(-1,len(volume.cells),-1).to(torch.complex128)
    expected=block.solve(ac,conductivity=sigma,tpa_scale=.7,absolute_tolerance=1e-15)
    expected_grad=torch.autograd.grad(expected.electric.square().sum(),(sigma,amplitude),retain_graph=True)
    def forbidden(*args,**kwargs):raise AssertionError('Global DC matrix/direct solve/eigendecomposition forbidden')
    monkeypatch.setattr(block,'_matrix',forbidden)
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch.linalg,'eigvalsh',forbidden)
    monkeypatch.setattr(torch.autograd.functional,'jacobian',forbidden)
    rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-20,maxiter=150,restart=40)
    actual=block.solve(ac,conductivity=sigma,tpa_scale=.7,absolute_tolerance=1e-15,
        tangent_rule=rule,tangent_column_batch=3)
    torch.testing.assert_close(actual.electric,expected.electric,rtol=1e-9,atol=1e-11)
    torch.testing.assert_close(actual.potential,expected.potential,rtol=1e-9,atol=1e-11)
    gradients=torch.autograd.grad(actual.electric.square().sum(),(sigma,amplitude))
    for a,b in zip(gradients,expected_grad):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    assert not actual.receipt['global_current_jacobian_stored']
    checks=actual.receipt['tangent_checks']
    assert [c['phase'] for c in checks[-2:]]==['implicit','implicit']
    assert all(c['bound']['admitted'] and c['solve']['converged'] for c in checks)
    assert not actual.receipt['fast_work_qualified'] and not actual.receipt['physical_qualification']
    failed=StoppingRule(preconditioner='none',rtol=1e-15,atol=1e-25,maxiter=1,restart=1)
    with pytest.raises(StaticCurrentConvergenceError,match='current_tangent_bound_inconclusive'):
        block.solve(ac,conductivity=sigma,tpa_scale=.7,tangent_rule=failed)


def test_dc_fixed_inverse_transpose_and_rhs_derivatives(monkeypatch):
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block=NodalCurrentBalance(volume)
    tangent=volume.vertices.new_tensor([[2.,.3,0.],[-.2,1.,.1],[.1,0.,3.]])
    tangent=tangent.expand(len(volume.cells),3,3)
    matrix=block._matrix(tangent)[1:,1:]
    inverse=FixedStaticCurrentInverse(block.tangent_action(tangent),StoppingRule(
        preconditioner='none',rtol=1e-12,atol=1e-15,maxiter=100,restart=40))
    rhs=torch.arange(2*inverse.shape[0],device='cuda',dtype=torch.float64).reshape(2,-1).requires_grad_()
    expected=torch.linalg.solve(matrix,rhs.unsqueeze(-1)).squeeze(-1)
    transpose=torch.linalg.solve(matrix.mT,rhs.unsqueeze(-1)).squeeze(-1)
    gradient=torch.linalg.solve(matrix.mT,(2*expected).unsqueeze(-1)).squeeze(-1)
    second=torch.linalg.solve(matrix.mT,2*torch.linalg.solve(matrix,torch.ones_like(rhs).unsqueeze(-1))).squeeze(-1)
    def forbidden(*args,**kwargs):raise AssertionError('Dense solve forbidden')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    actual=inverse.apply(rhs)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(inverse.adjoint(rhs),transpose,rtol=1e-10,atol=1e-10)
    g,=torch.autograd.grad(actual.square().sum(),rhs,create_graph=True)
    torch.testing.assert_close(g,gradient,rtol=1e-10,atol=1e-8)
    gg,=torch.autograd.grad(g.sum(),rhs)
    torch.testing.assert_close(gg,second,rtol=1e-10,atol=1e-8)
    failed=FixedStaticCurrentInverse(block.tangent_action(tangent),StoppingRule(
        preconditioner='none',rtol=1e-15,atol=1e-20,maxiter=1),refinement_steps=0)
    with pytest.raises(StaticTangentConvergenceError):failed.apply(rhs)


def test_dc_overflowed_threshold_rejected():
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block=NodalCurrentBalance(volume)
    ac=torch.zeros(0,len(volume.cells),3,device='cuda',dtype=torch.complex128)
    with pytest.raises(ValueError,match='threshold must be finite'):
        block.solve(ac,conductivity=1.,source_current=torch.ones(len(volume.cells),3,device='cuda',dtype=torch.float64)*1e4,
            relative_tolerance=1e308)


def test_dc_krylov_singular_final_tangent_is_rejected():
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block=NodalCurrentBalance(volume)
    # Nonzero AC bypasses the unexcited-material shortcut; the zero current law
    # still has no isolated DC root and must fail the final tangent check.
    ac=torch.ones(1,len(volume.cells),3,device='cuda',dtype=torch.complex128)
    rule=StoppingRule(preconditioner='none',rtol=1e-10,atol=1e-15,maxiter=10)
    with pytest.raises(StaticCurrentConvergenceError,match='current_tangent_bound_inconclusive') as error:
        block.solve(ac,tangent_rule=rule)
    assert error.value.receipt['tangent_checks'][0]['phase']=='implicit'
    assert not error.value.receipt['tangent_checks'][0]['bound']['admitted']


def test_nonfinite_arnoldi_solution_retains_failure_receipt(monkeypatch):
    from types import SimpleNamespace
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block=NodalCurrentBalance(volume)
    tangent=torch.eye(3,device='cuda',dtype=torch.float64).expand(len(volume.cells),3,3)
    inverse=FixedStaticCurrentInverse(block.tangent_action(tangent),StoppingRule(preconditioner='none'))
    def invalid(matrix,rhs):
        return SimpleNamespace(solution=rhs.new_full((matrix.shape[1],),float('nan')))
    monkeypatch.setattr(torch.linalg,'lstsq',invalid)
    rhs=volume.vertices.new_ones(inverse.shape[0])
    with pytest.raises(StaticTangentConvergenceError) as error:inverse.apply(rhs)
    entry=error.value.receipt['rhs'][0]
    assert not error.value.receipt['converged']
    assert len(entry['attempts'])==3
    assert all(a['reason']=='nonfinite_iterate' for a in entry['attempts'])
    assert entry['absolute_residual']==float(torch.linalg.vector_norm(rhs))

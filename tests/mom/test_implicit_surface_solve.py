"""Fixed inverse implicit gradients are checked against actual surface matrices."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse,SurfaceInverseConvergenceError
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def problem():
    dense,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    provider=ScalarSystemBlocks(dense.system,dense.exterior,dense.integration)
    operator=HierarchicalOperator(provider,HMatrixConfig(leaf_size=64))
    return dense,operator


@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('method',['gmres','block_gmres'])
def test_actual_surface_inverse_forward_adjoint_and_second_rhs_derivative(method,nested):
    dense,operator=problem()
    if nested:
        from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
        operator=HierarchicalH2Operator(operator)
    inverse=FixedSurfaceInverse(operator,StoppingRule(method=method,preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,maxiter=200,restart=48,stagnation_window=48))
    torch.manual_seed(73)
    b=torch.randn(2,operator.shape[0],device='cuda',dtype=operator.dtype,requires_grad=True)
    y=torch.randn_like(b)
    x=inverse.apply(b)
    expected=torch.linalg.solve(dense.matrix,b.mT).mT
    adjoint=inverse.adjoint(y)
    expected_adjoint=torch.linalg.solve(dense.matrix.mH,y.mT).mT
    torch.testing.assert_close(x,expected,rtol=2e-8,atol=1e-9)
    torch.testing.assert_close(adjoint,expected_adjoint,rtol=2e-8,atol=1e-9)
    torch.testing.assert_close((x*y.conj()).sum(),(b*adjoint.conj()).sum(),rtol=2e-8,atol=1e-9)
    actual_grad,=torch.autograd.grad(x.abs().square().sum(),b,create_graph=True)
    expected_grad,=torch.autograd.grad(expected.abs().square().sum(),b,create_graph=True)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=3e-8,atol=1e-8)
    actual_hvp,=torch.autograd.grad((actual_grad*y.conj()).real.sum(),b)
    expected_hvp,=torch.autograd.grad((expected_grad*y.conj()).real.sum(),b)
    torch.testing.assert_close(actual_hvp,expected_hvp,rtol=3e-8,atol=1e-8)
    assert inverse.receipt['last_forward']['converged']
    assert inverse.receipt['last_adjoint']['converged']
    delayed=inverse.apply(b)
    with torch.no_grad():operator.provider.system.bodies[0].mesh.vertices[0,0]+=1e-5
    with pytest.raises(ValueError,match='changed'):
        torch.autograd.grad(delayed.real.sum(),b)


def test_inverse_failure_is_explicit_and_does_not_fallback():
    _,operator=problem()
    inverse=FixedSurfaceInverse(operator,StoppingRule(maxiter=1,restart=1,rtol=1e-14,atol=0,preconditioner='none'))
    b=torch.arange(operator.shape[0],device='cuda',dtype=torch.float64).to(operator.dtype)
    with pytest.raises(SurfaceInverseConvergenceError) as info:inverse.apply(b)
    assert not info.value.receipt['converged']
    for rule in (StoppingRule(method='direct'),StoppingRule(recycle_size=2),StoppingRule(fallback='direct')):
        with pytest.raises(ValueError):FixedSurfaceInverse(operator,rule)


def test_refreshable_dense_operator_cannot_enter_fixed_inverse():
    dense,_=problem()
    for replacement in (False,True):
        if replacement:dense.matrix=dense.matrix.clone()
        else:
            with torch.no_grad():dense.matrix.mul_(1.01)
        dense.refresh()
        with pytest.raises(TypeError,match='immutable'):FixedSurfaceInverse(dense)


def test_nonfinite_original_threshold_cannot_accept_solution():
    _,operator=problem()
    inverse=FixedSurfaceInverse(operator,StoppingRule(rtol=1e308,preconditioner='none'),refinement_steps=0)
    b=torch.full((operator.shape[0],),1e4,device='cuda',dtype=operator.dtype)
    with pytest.raises(SurfaceInverseConvergenceError):inverse.apply(b)

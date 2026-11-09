"""Physical H/H2 handles retain fixed inverse and delayed-gradient guards."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.hierarchical_runtime import HierarchicalSystemOperator,HierarchicalH2SystemOperator
from witwin.maxwell.mom.hmatrix import HMatrixConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('backend,kind',[('hmatrix',HierarchicalSystemOperator),('h2',HierarchicalH2SystemOperator)])
def test_physical_inverse_original_solve_gradient_and_decoder_mutation(backend,kind):
    dense,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    config=mw.MoMConfig(operator_backend=backend,integration=dense.integration,
        hierarchical=HMatrixConfig(leaf_size=64),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'))
    operator=kind(dense.system,dense.exterior,dense.sources,config)
    inverse=FixedSurfaceInverse(operator,StoppingRule(method='block_gmres',
        preconditioner='row_column_norm',rtol=1e-11,atol=1e-13,stagnation_window=48))
    torch.manual_seed(313)
    rhs=torch.randn(2,operator.shape[0],device='cuda',dtype=operator.dtype,requires_grad=True)
    passed,log=inverse._dual_gate(rhs.detach(),rhs.detach())
    assert not passed and not log['dual_converged']
    assert inverse._dual_gate(torch.zeros_like(rhs),rhs.detach())[0]
    actual=inverse.apply(rhs)
    assert inverse.receipt['last_forward']['dual_converged']
    expected=torch.linalg.solve(dense.matrix,rhs.mT).mT
    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=1e-9)
    scales=rhs.real.new_tensor([1e-18,1e-28])[:,None]
    tiny=(rhs.detach()*scales).requires_grad_(True)
    tiny_actual=inverse.apply(tiny)
    assert inverse.receipt['last_forward']['dual_converged']
    assert inverse.receipt['last_forward']['refinement_steps_used']>0
    torch.testing.assert_close(tiny_actual/scales,expected.detach(),rtol=2e-8,atol=1e-9)
    tiny_gradient,=torch.autograd.grad(tiny_actual.real.sum(),tiny)
    adjoint_reference=torch.linalg.solve(dense.matrix.mH,torch.ones_like(rhs).mT).mT
    torch.testing.assert_close(tiny_gradient,adjoint_reference,rtol=3e-8,atol=1e-8)
    ga,=torch.autograd.grad(actual.abs().square().sum(),rhs)
    gb,=torch.autograd.grad(expected.abs().square().sum(),rhs)
    torch.testing.assert_close(ga,gb,rtol=3e-8,atol=1e-8)
    for name,replacement in [('integration',IntegrationConfig(order=4,singular_order=4)),
                             ('incident_carrier',None)]:
        delayed=inverse.apply(rhs)
        handle=operator.body_handles[0];previous=getattr(handle,name)
        setattr(handle,name,replacement)
        with pytest.raises(ValueError,match='changed'):
            torch.autograd.grad(delayed.real.sum(),rhs)
        setattr(handle,name,previous)
        inverse.assert_current()
    previous=operator.sources;operator.sources=()
    with pytest.raises(ValueError,match='changed'):inverse.apply(rhs)
    operator.sources=previous
    inverse.assert_current()
    class Unverified(kind):
        pass
    unverified=object.__new__(Unverified)
    with pytest.raises(TypeError,match='immutable'):FixedSurfaceInverse(unverified)

"""Actual scalar-limit tensor-periodic selected and hierarchical algebra."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0,MU_0
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.periodic_dyadic import LossyDyadicImageReference
from witwin.maxwell.mom.periodic_dyadic_nearfield import PeriodicDyadicSystemBlocks
from witwin.maxwell.mom.periodic_dyadic_operator import assemble_periodic_dyadic_system
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from test_generalized_system_blocks import basis_at

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def problem(spheres=False,subdivisions=0):
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    scalar=compile_medium(mw.Material(sigma_e=.04,sigma_m=2000.),3e8)
    medium=TensorMedium(3e8,scalar.epsilon_r*eye,scalar.mu_r*eye)
    interior=TensorMedium(3e8,2*eye,1.3*eye)
    other=compile_medium(mw.Material(eps_r=3.,mu_r=1.1),3e8)
    bases=[RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),subdivisions=subdivisions,device='cuda'))
           if spheres else basis_at(x) for x in (-.2,.2)]
    system=SurfaceSystem.create([(0,'tensor',bases[0],interior),(1,'scalar',bases[1],other)])
    sample=bases[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[1.,0.,0.]]),sample.new_tensor([.2,0.,0.]))
    reference=LossyDyadicImageReference(lattice,medium,1)
    config=IntegrationConfig(order=2,singular_order=3,pair_batch=32)
    exterior=DyadicGreenKernel(medium)
    interiors={1:DyadicGreenKernel(interior)}
    provider=PeriodicDyadicSystemBlocks(system,reference,config,exterior_kernel=exterior,interior_kernels=interiors)
    dense,_=assemble_periodic_dyadic_system(system,reference=reference,config=config,
        exterior_kernel=exterior,interior_kernels=interiors)
    return provider,dense


def test_global_periodic_selection_streamed_action_and_current_gradient():
    provider,dense=problem()
    n=provider.system.total_edges
    rows=torch.tensor([n-1,0,n//2],device='cuda');cols=torch.arange(n-1,-1,-1,device='cuda')
    full_rows=torch.cat((rows,rows+n));full_cols=torch.cat((cols,cols+n))
    torch.testing.assert_close(provider(rows,cols),dense[full_rows[:,None],full_cols],rtol=1e-11,atol=2e-13)
    generator=torch.Generator(device='cuda').manual_seed(914)
    current=torch.randn((2,2*n),device='cuda',dtype=dense.dtype,generator=generator,requires_grad=True)
    actual=provider.sampled_action(rows,current,source_batch=3)
    torch.testing.assert_close(actual,current@dense[full_rows].T,rtol=1e-11,atol=2e-13)
    probe=torch.randn(actual.shape,device='cuda',dtype=dense.dtype,generator=generator)
    gradient,=torch.autograd.grad(actual,current,probe)
    torch.testing.assert_close(gradient,probe@dense[full_rows].conj(),rtol=1e-11,atol=2e-13)
    assert provider.metadata['nonprimary_panel_pairs']>0
    assert provider.metadata['maximum_nonprimary_panel_pairs']<=provider.config.pair_batch
    assert any(value is provider.reference.lattice.vectors for value in provider.input_tensors())
    with pytest.raises(ValueError,match='explicit residual proxy'):
        HierarchicalOperator(provider)
    old=provider.reference
    provider.reference=LossyDyadicImageReference(old.lattice,old.medium,old.shells)
    with pytest.raises(ValueError,match='binding changed'):provider.assert_current()


@pytest.mark.parametrize('subdivisions',[0,1])
def test_tensor_exterior_h_and_h2_use_explicit_metric_for_actions_and_gradients(subdivisions):
    provider,dense=problem(spheres=True,subdivisions=subdivisions)
    eta=math.sqrt(MU_0/EPSILON_0)
    h=HierarchicalOperator(provider,HMatrixConfig(pivoting='adaptive',
        leaf_size=32 if subdivisions==0 else 128,max_rank=32 if subdivisions==0 else 64),metric_impedance=eta)
    print({'hierarchical_construction':h.metadata},flush=True)
    if subdivisions==0:
        # Independent SVD diagnostic needs rank 40 of 60 at 1e-8; factors
        # cannot reduce storage. Preserve exact fallback, not a fast claim.
        assert h.metadata['far_blocks']==0 and h.metadata['exact_blocks']==4
        assert len(h.metadata['compression_rejections'])==2
        assert all(not value['storage_reduced'] for value in h.metadata['compression_rejections'])
    else:
        assert h.metadata['far_blocks']>0
    h2=HierarchicalH2Operator(h)
    assert bool(h2.actions.couplings)==bool(subdivisions)
    generator=torch.Generator(device='cuda').manual_seed(915)
    x=torch.randn((2,len(dense)),device='cuda',dtype=dense.dtype,generator=generator,requires_grad=True)
    y=torch.randn(x.shape,device='cuda',dtype=dense.dtype,generator=generator)
    for operator in (h,h2):
        actual=operator.apply(x);expected=x@dense.T
        for got,want in zip(actual.chunk(2,-1),expected.chunk(2,-1)):
            assert float(((got-want).norm()/want.norm()).detach())<1e-8
        actual=operator.adjoint(y);expected=y@dense.conj()
        for got,want in zip(actual.chunk(2,-1),expected.chunk(2,-1)):
            assert float(((got-want).norm()/want.norm()).detach())<1e-8
        gradient,=torch.autograd.grad(operator.apply(x),x,y)
        torch.testing.assert_close(gradient,operator.adjoint(y),rtol=1e-11,atol=2e-13)
        assert operator.metric.impedance_squared==eta**2
        assert not operator.metadata['physical_qualification']

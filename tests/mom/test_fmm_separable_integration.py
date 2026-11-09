"""Full free/periodic action storage and configuration identity for tree shifts."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig,assemble_system
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors,storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def provider(*,absorbing=False):
    exterior=compile_medium(mw.Material(eps_r=1.5,sigma_e=.02) if absorbing else mw.Material(),3e8)
    bodies=[(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),subdivisions=0,device='cuda')),
        compile_medium(mw.Material(eps_r=2.+i,mu_r=1.2+.1*i),3e8)) for i,x in enumerate((-.2,.2))]
    system=SurfaceSystem.create(bodies)
    return ScalarSystemBlocks(system,exterior,IntegrationConfig(order=3,singular_order=3))


def test_separable_config_identity_defaults_and_strict_binding():
    from witwin.maxwell.mom.runtime import _numerical_identity
    from witwin.maxwell.mom.linalg import StoppingRule
    default=mw.FMMConfig()
    assert default.payload()['version']=='cartesian_fmm_shared_compact_config_v1'
    assert default.payload()['parameters']['shift_storage']=='separable_v1'
    selected=mw.FMMConfig(table_storage='compact',near_storage='full_matrix',shift_storage='separable_v1')
    assert selected.payload()['version']=='cartesian_fmm_separable_config_v1'
    assert mw.FMMConfig.restore(default.payload())==default
    assert mw.FMMConfig.restore(selected.payload())==selected
    assert selected.construction()['shift_storage']=='separable_v1'
    options=dict(operator_backend='fmm',riesz_storage='local_rwg_v1',stopping=StoppingRule(rtol=1e-9, preconditioner='rwg_diagonal'))
    assert _numerical_identity(mw.MoMConfig(fmm=default,**options))!=_numerical_identity(mw.MoMConfig(fmm=selected,**options))
    for version in ('cartesian_fmm_config_v7','unknown'):
        bad=copy.deepcopy(selected.payload());bad['version']=version
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    for field in selected.payload()['parameters']:
        bad=copy.deepcopy(selected.payload());del bad['parameters'][field]
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    with pytest.raises(ValueError):mw.FMMConfig(shift_storage='unknown')


def test_separable_full_free_action_original_matrix_and_capacity():
    p=provider()
    op=MultipoleSystemActions(p,order=6,leaf_size=30,table_storage='compact',shift_storage='separable_v1')
    assert len(op.exterior.far)>0
    plan=op.capacity_plan
    assert plan['planned_bytes']==storage_bytes(multipole_action_tensors(op))
    assert sum(plan['categories'][k] for k in ('centers_radii','leaf_maps','leaf_moments','shifts'))==op.projection_bytes
    dense,_=assemble_system(p.system,p.exterior,p.config)
    n=p.system.total_edges;eta=abs(p.exterior.eta)
    actual=op.apply(torch.eye(2*n,device=p.system.device,dtype=torch.complex128)).mT
    left=actual.real.new_tensor([eta]*n+[1.]*n);right=actual.real.new_tensor([1.]*n+[eta]*n)
    error=(left[:,None]*(actual-dense)*right).norm()/(left[:,None]*dense*right).norm()
    assert float(error)<1e-8
    x=torch.randn((2,2*n),device=p.system.device,dtype=torch.complex128,requires_grad=True)
    y=torch.randn_like(x)
    gradient,=torch.autograd.grad(op.apply(x),x,y)
    torch.testing.assert_close(gradient,op.adjoint(y),rtol=2e-11,atol=1e-14)
    torch.testing.assert_close(op.adjoint(y),y@actual.conj(),rtol=2e-11,atol=1e-14)
    print(dict(full_free_scaled_error=float(error),planned_bytes=plan['planned_bytes'],categories=plan['categories']),flush=True)


@pytest.mark.parametrize('dimension',(1,2,3))
def test_separable_periodic_translated_projections_capacity_and_dense_shift_action(dimension):
    from witwin.maxwell.mom.periodic import LatticeSpec
    from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
    from witwin.maxwell.mom.fmm_periodic_capacity import periodic_action_tensors
    p=provider(absorbing=True)
    lattice=LatticeSpec(torch.eye(3,device=p.system.device,dtype=torch.float64)[:dimension],
        torch.tensor([.37,.13 if dimension>=2 else 0.,.07 if dimension==3 else 0.],device=p.system.device,dtype=torch.float64))
    options=dict(order=6,shells=1,leaf_size=30,separation_ratio=.1,table_storage='compact')
    original=PeriodicMultipoleSystemActions(p,lattice,**options)
    selected=PeriodicMultipoleSystemActions(p,lattice,shift_storage='separable_v1',**options)
    assert selected.capacity_plan['planned_bytes']==storage_bytes(periodic_action_tensors(selected))
    assert selected.capacity_plan['categories']['projections']==selected.projection_bytes
    assert selected.capacity_plan['translated_images']>0
    for region in selected.exterior.regions:
        assert region.source.shift_storage=='separable_v1'
    x=torch.randn((2,*selected.shape[:1]),device=p.system.device,dtype=torch.complex128,requires_grad=True)
    y=torch.randn_like(x)
    torch.testing.assert_close(selected.apply(x),original.apply(x),rtol=2e-11,atol=1e-14)
    torch.testing.assert_close(selected.adjoint(y),original.adjoint(y),rtol=2e-11,atol=1e-14)
    derivative,=torch.autograd.grad(selected.apply(x),x,y)
    torch.testing.assert_close(derivative,selected.adjoint(y),rtol=2e-11,atol=1e-14)
    print(dict(periodic_dimension=dimension,original_projection_bytes=original.projection_bytes,
        separable_projection_bytes=selected.projection_bytes,translated_images=selected.capacity_plan['translated_images']),flush=True)

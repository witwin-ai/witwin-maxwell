"""Compressed harmonic residuals and roots match original dense physics maps."""
import math
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.action_harmonic_balance import ActionHarmonicBalance
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.harmonic_preparation import prepare_static_cell_response
from witwin.maxwell.mom.compressed_background import prepare_compressed_background
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.hmatrix import HierarchicalOperator,HMatrixConfig
from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def maps(volumes,frequency,nested=False,solve_atol=1e-13):
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),compile_medium(mw.Material(eps_r=2.+i),frequency)) for i,v in enumerate(volumes)])
    exterior=compile_medium(mw.Material(),frequency);config=IntegrationConfig(order=3,singular_order=3)
    op=HierarchicalOperator(ScalarSystemBlocks(system,exterior,config),HMatrixConfig(leaf_size=64))
    if nested:op=HierarchicalH2Operator(op)
    inverse=FixedSurfaceInverse(op,StoppingRule(method='block_gmres',preconditioner='row_column_norm',rtol=1e-11,atol=solve_atol,stagnation_window=48))
    bindings=[bind_volume_region(v,system) for v in volumes]
    source=[SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(b,order=3,inner_order=3),SurfaceVolumeHConfig(leaf_size=64)) for b in bindings]
    field=[InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(b,order=3),InteriorVolumeHConfig(leaf_size=64)) for b in bindings]
    direct=[VolumeHierarchicalAction(VolumeCellBlocks(b.volume,system.bodies[b.volume.region_id-1].interior,order=3),VolumeHConfig(leaf_size=64)) for b in bindings]
    return CoupledVolumeAction(inverse,source,field,direct),bindings,exterior,config


@pytest.mark.parametrize('tensor',[False,True],ids=['scalar','tensor'])
def test_actual_two_harmonic_residual_root_and_parameter_gradients(tensor):
    _,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    volumes=[b.volume for b in bindings]
    data=[maps(volumes,n*3e8,nested=n==2) for n in (1,2)]
    actions=[d[0] for d in data]
    original=runpy.run_path('tests/mom/test_coupled_volume_action.py')['original_matrix']
    response=[original(*d)[0] for d in data]
    a0,bindings,exterior,_=data[0]
    static=prepare_static_cell_response(a0.inverse.operator.provider.system,bindings,EPSILON_0,[2*EPSILON_0,3*EPSILON_0],testing_order=3)
    zero=torch.zeros_like(response[0]);dc=static['electric_response'].to(zero.dtype)
    bank=prepare_compressed_background(a0.inverse,bindings,(mw.PlaneWave(),))
    scale=torch.tensor(.6,device='cuda',dtype=torch.float64,requires_grad=True)
    background=torch.stack((bank.electric[0]*0,bank.combine(scale[None])['background_electric'],bank.electric[0]*0))
    weights=torch.cat([v.volumes for v in volumes])
    settings=dict(chi2=.002,chi3=.001,tpa_scale=.02)
    if tensor:
        coefficient=torch.tensor(.001,device='cuda',dtype=torch.float64,requires_grad=True)
        axis=coefficient.new_tensor([1.,2.,3.]);axis=axis/axis.norm()
        settings.update(chi2_tensor=coefficient*torch.einsum('i,j,k->ijk',axis,axis,axis),
                        chi3_tensor=coefficient*torch.einsum('i,j,k,l->ijkl',axis,axis,axis,axis))
    actual=ActionHarmonicBalance(background,actions,weights,3e8,dc_polarization_response=dc,dc_current_response=zero,**settings)
    dense=DenseHarmonicBalance(background,torch.stack((dc,*[-1j*(2*math.pi*3e8*n)*v for n,v in enumerate(response,1)])),
        torch.stack((zero,*response)),weights,3e8,**settings)
    electric=(background.detach()+.001).requires_grad_()
    a=actual.residual(electric);b=dense.residual(electric)
    torch.testing.assert_close(a,b,rtol=2e-8,atol=1e-10)
    ga,=torch.autograd.grad(a.abs().square().sum(),electric)
    gb,=torch.autograd.grad(b.abs().square().sum(),electric)
    torch.testing.assert_close(ga,gb,rtol=3e-8,atol=1e-10)
    solution=actual.solve(max_iterations=10)
    reference=dense.solve(max_iterations=10)
    torch.testing.assert_close(solution.electric,reference.electric,rtol=2e-7,atol=1e-9)
    ga,=torch.autograd.grad(solution.electric.abs().square().sum(),scale,retain_graph=True)
    gb,=torch.autograd.grad(reference.electric.abs().square().sum(),scale,retain_graph=True)
    torch.testing.assert_close(ga,gb,rtol=3e-7,atol=1e-9)
    if tensor:
        ga,=torch.autograd.grad(solution.electric.abs().square().sum(),coefficient,retain_graph=True)
        gb,=torch.autograd.grad(reference.electric.abs().square().sum(),coefficient,retain_graph=True)
        torch.testing.assert_close(ga,gb,rtol=3e-7,atol=1e-9)

    assert solution.receipt['converged'] and not solution.receipt['physical_qualification']
    assert not hasattr(actual,'current_response') and not hasattr(actual,'polarization_response')
    with pytest.raises(ValueError,match='frequency'):
        ActionHarmonicBalance(background,actions[::-1],weights,3e8,dc_polarization_response=dc,dc_current_response=zero)
    with torch.no_grad():actions[0].indices[0][0]+=1
    with pytest.raises(ValueError,match='changed'):actual.residual(electric)

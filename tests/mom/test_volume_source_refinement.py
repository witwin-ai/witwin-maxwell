"""Original weak columns, independent dual norms and explicit order exhaustion."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.volume_source_refinement import check_source_columns,prepare_source_action
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction,SurfaceVolumeHConfig
from witwin.maxwell.mom.riesz import LocalRWGRiesz
from witwin.maxwell.mom.volume_response_matrix import VolumeMatrixConvergenceError
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.volume_mesh import bind_volume_region

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(refine=False):
    op,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    if refine:
        volumes=[bindings[0].volume,bindings[1].volume.refine(1)]
        system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),op.system.bodies[i].interior) for i,v in enumerate(volumes)])
        bindings=[bind_volume_region(v,system) for v in volumes]
    else:system=op.system
    metric=LocalRWGRiesz(system,op.exterior.eta)
    return bindings[1],metric


def action(binding,order,inner):
    return SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=order,inner_order=inner),SurfaceVolumeHConfig(leaf_size=64))


def test_streamed_original_and_axis_norms_with_partial_cell_batch(monkeypatch):
    binding,metric=setup(True)
    old=action(binding,2,3);fine=action(binding,4,3)
    actual=check_source_columns(fine,metric,cell_batch=3)
    assert actual['passed'] and len(actual['checks'])==3 and actual['checks'][-1]['cell_stop']==8
    old_full=old.apply(torch.eye(old.shape[1],device='cuda',dtype=old.dtype))
    full=fine.apply(torch.eye(fine.shape[1],device='cuda',dtype=fine.dtype))
    basis=binding.surface_system.bodies[1].basis
    gram=basis.mass_matrix().to(fine.dtype)
    metric_dense=torch.block_diag(gram/metric.impedance_squared,gram)
    def norms(v):return (v.conj()*torch.linalg.solve(metric_dense,v.mT).mT).real.sum(-1).clamp_min(0).sqrt()
    error=norms(full-old_full);threshold=.5*(1e-12+1e-4*norms(full))
    result=check_source_columns(fine,metric,previous=old,cell_batch=3)
    assert result['passed']==bool((error<=threshold).all())
    assert sum(c['failed_columns'] for c in result['checks'])==int((error>threshold).sum())
    assert max(c['maximum_column_difference_dual_proxy'] for c in result['checks'])==pytest.approx(float(error.max()),rel=1e-9,abs=1e-13)
    original=fine.apply
    monkeypatch.setattr(fine,'apply',lambda value:1.01*original(value))
    assert not check_source_columns(fine,metric,cell_batch=3)['passed']
    with pytest.raises(ValueError,match='exactly one'):check_source_columns(old,metric,previous=old)


def test_actual_adaptive_source_default_tolerances_and_exhaustion():
    binding,metric=setup()
    config=SurfaceVolumeHConfig(leaf_size=64)
    prepared,receipt=prepare_source_action(binding,metric,config=config)
    assert receipt['converged'] and receipt['checks'][-1]['axis']=='outer'
    assert all(v['compression']['passed'] for v in receipt['evaluations'])
    assert prepared.provider.order==receipt['final_outer_order']
    assert prepared.provider.inner_order==receipt['final_inner_order']
    assert not receipt['physical_qualification']
    with pytest.raises(VolumeMatrixConvergenceError) as info:
        prepare_source_action(binding,metric,config=config,max_order=3,max_inner_order=3,relative_tolerance=1e-12,absolute_tolerance=1e-20)
    assert not info.value.receipt['converged'] and info.value.receipt['failure_reason']=='inner_order_budget_exhausted'


@pytest.mark.parametrize('settings',[{'rtol':1.},{'atol':1e30}])
def test_loose_metric_cannot_erase_source_corruption(monkeypatch,settings):
    binding,metric=setup()
    loose=LocalRWGRiesz(metric.system,metric.impedance_squared**.5,**settings)
    candidate=action(binding,3,3)
    original=candidate.apply
    monkeypatch.setattr(candidate,'apply',lambda value:1.01*original(value))
    with pytest.raises(ValueError,match='independent true residual'):
        check_source_columns(candidate,loose)

"""Regional nested volume actions preserve H and original field contracts."""
import gc
import runpy
import weakref
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHConfig,VolumeHierarchicalAction
from witwin.maxwell.mom.volume_h2 import VolumeNestedH2Action
from witwin.maxwell.mom.h2 import H2Config
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('field',['E','H'])
def test_actual_volume_h2_far_conversion_original_adjoint_gradient_and_release(field):
    mesh=runpy.run_path('tests/mom/test_volume_hmatrix.py')['elongated_volume']()
    provider=VolumeCellBlocks(mesh,compile_medium(mw.Material(eps_r=2.,mu_r=1.2),3e8),field=field,order=3)
    source=VolumeHierarchicalAction(provider,VolumeHConfig(leaf_size=6,admissibility=1.,max_rank=24,tolerance=1e-8,row_batch=12))
    assert source.receipt['far_blocks']>0
    action=VolumeNestedH2Action(source,H2Config(max_rank=64,tolerance=1e-10))
    assert action.receipt['nested']['far_couplings']==source.receipt['far_blocks']
    assert action.receipt['retained_bytes']==storage_bytes(action.tensors())
    assert not action.receipt['source_h_far_factors_retained']
    torch.manual_seed(75)
    x=torch.randn(2,3*len(mesh.cells),device='cuda',dtype=provider.dtype,requires_grad=True)
    y=torch.randn_like(x)
    actual=action.apply(x)
    torch.testing.assert_close(actual,source.apply(x),rtol=2e-9,atol=1e-10)
    torch.testing.assert_close(action.adjoint(y),source.adjoint(y),rtol=2e-9,atol=1e-10)
    reference=weakref.ref(source);del source;gc.collect()
    assert reference() is None
    ids=torch.arange(len(mesh.cells),device='cuda')
    dense=provider(ids,ids)
    torch.testing.assert_close(actual,x@dense.mT,rtol=2e-7,atol=1e-9)
    torch.testing.assert_close(action.adjoint(y),y@dense.conj(),rtol=2e-7,atol=1e-9)
    torch.testing.assert_close((actual*y.conj()).sum(),(x*action.adjoint(y).conj()).sum(),rtol=1e-11,atol=1e-10)
    grad,=torch.autograd.grad((actual*y.conj()).real.sum(),x)
    torch.testing.assert_close(grad,action.adjoint(y),rtol=2e-12,atol=1e-12)
    with torch.no_grad():action.nested.couplings[0][2][0,0]+=1
    with pytest.raises(ValueError,match='changed'):action.apply(x)


def test_all_near_volume_h2_and_capacity_rejection(monkeypatch):
    import witwin.maxwell.mom.volume_h2 as module
    mesh=runpy.run_path('tests/mom/test_volume_green.py')['mesh']()
    provider=VolumeCellBlocks(mesh,compile_medium(mw.Material(),3e8),order=3)
    source=VolumeHierarchicalAction(provider)
    action=VolumeNestedH2Action(source)
    x=torch.ones(3,device='cuda',dtype=provider.dtype)
    torch.testing.assert_close(action.apply(x),source.apply(x),rtol=0,atol=0)
    with torch.no_grad():action.nested.near.groups[0][2][0,0,0]+=1
    with pytest.raises(ValueError,match='changed'):action.apply(x)
    def forbidden(*args,**kwargs):raise AssertionError('Budget rejection must precede nested allocation')
    monkeypatch.setattr(module,'NestedH2Actions',forbidden)
    with pytest.raises(MemoryError,match='budget'):
        VolumeNestedH2Action(source,H2Config(retained_budget_bytes=1))

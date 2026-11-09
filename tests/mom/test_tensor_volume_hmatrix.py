"""Tensor volume H/H2 actions and exact factors without kernel reevaluation."""
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel,DyadicIntegrationConfig
from witwin.maxwell.mom.tensor_volume import cell_tensor_current_fields
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from witwin.maxwell.mom.volume_h2 import VolumeNestedH2Action
from witwin.maxwell.mom import volume_h_snapshot,volume_h2_snapshot

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def provider(field):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.05,0.],[0.,0.,.06],
                           [0.,0.,-.04]],device='cuda',dtype=torch.float64)
    mesh=TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3],[0,2,1,4]],device='cuda'),region_id=1)
    a=torch.diag(vertices.new_tensor([1.25,1.,.75])).to(torch.complex128)
    medium=TensorMedium(3e7,a,a)
    kernel=DyadicGreenKernel(medium,DyadicIntegrationConfig(radial_map='matched_spd_branch_v1',
        relative_tolerance=2e-4,max_radial_order=1024,max_azimuthal_order=256))
    return VolumeCellBlocks(mesh,medium,field=field,order=2,tensor_kernel=kernel,
                            radial_order=2,kernel_batch=8192)


@pytest.mark.parametrize('field',['E','H'])
def test_tensor_volume_actions_and_exact_restore(field,monkeypatch,tmp_path):
    p=provider(field)
    action=VolumeHierarchicalAction(p,VolumeHConfig(leaf_size=8))
    nested=VolumeNestedH2Action(action)
    # Two distinct cells exercise component-major ordering. This near-only
    # control does not qualify tensor far compression or quadrature accuracy.
    torch.manual_seed(21)
    x=torch.randn(6,device='cuda',dtype=torch.complex128,requires_grad=True)
    y=torch.randn_like(x)
    expected,_=cell_tensor_current_fields(p.mesh,x.reshape(3,2).mT,p.points,p.tensor_kernel,
        order=2,radial_order=2,kernel_batch=8192)
    reference=expected[field=='H'].mT.flatten()
    for a in (action,nested):
        torch.testing.assert_close(a.apply(x),reference,rtol=2e-12,atol=1e-12)
        gradient,=torch.autograd.grad((a.apply(x)*y.conj()).real.sum(),x,retain_graph=True)
        torch.testing.assert_close(gradient,a.adjoint(y),rtol=2e-12,atol=1e-12)
    for a,codec in ((action,volume_h_snapshot),(nested,volume_h2_snapshot)):
        payload=codec.encode(a)
        path=tmp_path/(field+codec.__name__.split('.')[-1]+'.pt')
        torch.save(payload,path);saved=torch.load(path,weights_only=True)
        fresh=provider(field)
        def forbidden(*args,**kwargs):raise AssertionError('Restore must not sample kernels or rebuild factors')
        with monkeypatch.context() as patch:
            patch.setattr(VolumeCellBlocks,'__call__',forbidden)
            patch.setattr(VolumeHierarchicalAction,'__init__',forbidden)
            patch.setattr(VolumeNestedH2Action,'__init__',forbidden)
            restored=codec.restore(saved,fresh)
        torch.testing.assert_close(restored.apply(x),a.apply(x),rtol=0,atol=0)
        torch.testing.assert_close(restored.adjoint(y),a.adjoint(y),rtol=0,atol=0)
        assert codec.encode(restored)['sha256']==payload['sha256']
        wrong=provider(field);wrong.radial_order=3;wrong._binding=wrong._identity()
        with pytest.raises(ValueError,match='physical binding'):codec.restore(saved,wrong)
    object.__setattr__(p.tensor_kernel,'cache_max_bytes',p.tensor_kernel.cache_max_bytes+1)
    with pytest.raises(ValueError,match='binding changed'):action.apply(x)


def test_tensor_volume_cache_policy_binding():
    p=provider('E')
    kernel=p.tensor_kernel
    before=kernel.mode_cache.max_bytes
    kernel.mode_cache.max_bytes=before+1
    with pytest.raises(ValueError,match='binding changed'):p.assert_current()
    with pytest.raises(ValueError,match='declared material and capacity'):
        VolumeCellBlocks(p.mesh,p.medium,tensor_kernel=kernel)
    kernel.mode_cache.max_bytes=before
    p.assert_current()
    kernel.mode_cache.medium=provider('E').medium
    with pytest.raises(ValueError,match='binding changed'):p.assert_current()
    with pytest.raises(ValueError,match='declared material and capacity'):
        VolumeCellBlocks(p.mesh,p.medium,tensor_kernel=kernel)

"""Regional tensor rectangular actions and exact stored-factor bindings."""
from dataclasses import replace
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_block
from witwin.maxwell.mom.fields import decode_interior_columns
from witwin.maxwell.mom import volume_h_snapshot as codec

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def make(kind):
    op,binding=setup();kernel=op.body_handles[0].tensor_kernel
    if kind=='source':
        p=SurfaceVolumeBlocks(binding,order=2,inner_order=2,tensor_kernel=kernel,
                              tensor_kernel_batch=8192)
    else:p=InteriorVolumeBlocks(binding,field=kind,order=2,tensor_kernel=kernel)
    return op,binding,p


@pytest.mark.parametrize('kind',['source','E','H'])
def test_tensor_rectangular_actions_exact_restore_and_policy(kind,monkeypatch,tmp_path):
    op,binding,p=make(kind)
    cls=SurfaceVolumeHierarchicalAction if kind=='source' else InteriorVolumeHierarchicalAction
    a=cls(p)
    if kind=='source':
        edges=torch.arange(p.row_size,device=p.device);cells=edges.new_tensor([0])
        expected,_=build_volume_source_block(op,binding,edges,cells,order=2,inner_order=2,
                                             tensor_kernel_batch=8192)
        expected=expected.reshape(2*len(edges),1,3).permute(0,2,1).reshape(2*len(edges),3)
    else:
        h=op.body_handles[0];h.integration=replace(h.integration,order=2,singular_order=2)
        identity=torch.eye(2*h.basis.n_edges,device=p.device,dtype=p.dtype)
        expected=decode_interior_columns(h,identity,p.points)[kind=='H'].permute(1,0,2).reshape(3,-1)
    torch.manual_seed(28)
    x=torch.randn(a.shape[1],device=a.device,dtype=a.dtype,requires_grad=True)
    y=torch.randn(a.shape[0],device=a.device,dtype=a.dtype)
    torch.testing.assert_close(a.apply(x),expected@x,rtol=3e-12,atol=1e-10)
    gradient,=torch.autograd.grad((a.apply(x)*y.conj()).real.sum(),x)
    torch.testing.assert_close(gradient,a.adjoint(y),rtol=2e-12,atol=1e-10)
    payload=codec.encode(a);path=tmp_path/'cross.pt';torch.save(payload,path)
    _,_,fresh=make(kind)
    def forbidden(*args,**kwargs):raise AssertionError('Restore cannot sample or rebuild factors')
    with monkeypatch.context() as patch:
        patch.setattr(type(p),'__call__',forbidden);patch.setattr(cls,'__init__',forbidden)
        restored=codec.restore(torch.load(path,weights_only=True),fresh)
    torch.testing.assert_close(restored.apply(x),a.apply(x),rtol=0,atol=0)
    assert codec.encode(restored)['sha256']==payload['sha256']
    assert a.receipt['impedance_proxy']=='vacuum_eta'
    geometry=fresh if kind=='source' else fresh.geometry
    geometry.tensor_kernel.mode_cache.max_bytes+=1
    with pytest.raises(ValueError,match='binding changed'):restored.apply(x)
    _,_,wrong=make(kind)
    geometry=wrong if kind=='source' else wrong.geometry
    geometry.tensor_kernel_batch+=1;geometry._binding=geometry._identity()
    with pytest.raises(ValueError,match='physical binding'):codec.restore(payload,wrong)

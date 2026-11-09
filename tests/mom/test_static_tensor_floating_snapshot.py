"""Owned floating tensor preparation and response, restored without assembly."""
import copy
import pytest
import torch
from test_tensor_harmonic_current_closure import setup
from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator
from witwin.maxwell.mom.static_tensor_snapshot import (
    snapshot_static_tensor_preparation,restore_static_tensor_preparation,
    snapshot_static_tensor_response,restore_static_tensor_response)
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('two_current',[False,True])
def test_floating_preparation_and_response_exact_restore(monkeypatch,tmp_path,two_current):
    op,bindings=setup()
    if two_current:
        op=StaticTensorFloatingOperator(op.system,op.exterior.coefficient,
                                       (op.interiors[0].coefficient,None,None))
    selected=() if two_current else (bindings[2],)
    polarizations=() if two_current else (op.centers.new_tensor([[.003,-.001,.002]])*1e-11,)
    traces={r:torch.arange(op.face_counts[r-1],device=op.system.device,dtype=op.system.dtype)*.01*r
            for r in op.current_regions}
    response=op.solve_uniform(op.centers.new_tensor([.2,-.1,.3]),boundary_potentials=traces,
                              bindings=selected,polarizations=polarizations)
    preparation,preparation_binding=snapshot_static_tensor_preparation(op)
    state,state_binding=snapshot_static_tensor_response(response)
    path=tmp_path/'floating.pt';torch.save(dict(preparation=preparation,state=state),path)
    loaded=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Restore repeated interface assembly or solve')
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorFloatingOperator,'__init__',forbidden)
        patch.setattr(StaticTensorFloatingOperator,'_layers',forbidden)
        patch.setattr(StaticTensorFloatingOperator,'solve_uniform',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=restore_static_tensor_preparation(loaded['preparation'],op.system,expected_binding=preparation_binding)
        decoded=restore_static_tensor_response(loaded['state'],restored,selected,expected_binding=state_binding)
    assert restored.current_regions==op.current_regions
    assert decoded.receipt==response.receipt
    for name in ('matrix','areas','flux_scale'):
        torch.testing.assert_close(getattr(restored,name),getattr(op,name),rtol=0,atol=0)
    for name in ('field','potential','conormal','normal_p','offsets'):
        torch.testing.assert_close(getattr(decoded,name),getattr(response,name),rtol=0,atol=0)
    for region in (0,1)+(() if two_current else (3,)):
        point=op.centers.new_tensor([[.8,.2,.1]]) if region==0 else bindings[region-1].volume.vertices.mean(0,keepdim=True)
        for got,want in zip(decoded.fields(point,region),response.fields(point,region)):
            torch.testing.assert_close(got,want,rtol=0,atol=0)
    with pytest.raises(NotImplementedError,match='current closure'):
        decoded.fields(bindings[1].volume.vertices.mean(0,keepdim=True),2)
    with pytest.raises(ValueError,match='budget'):
        restore_static_tensor_preparation(preparation,op.system,expected_binding=preparation_binding,max_matrix_bytes=1)
    for section,key in (('state','offsets'),('state','conormal')):
        bad=copy.deepcopy(state);bad[section][key].add_(1.)
        with pytest.raises(ValueError,match='owning state binding'):
            restore_static_tensor_response(bad,restored,selected,expected_binding=state_binding)
    bad=copy.deepcopy(preparation);bad['kernels'][0]=None
    with pytest.raises(ValueError,match='inventory'):
        restore_static_tensor_preparation(bad,op.system,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    bad=copy.deepcopy(state);bad['state']['offsets']=bad['state']['offsets'][:0]
    with pytest.raises(ValueError,match='array'):
        restore_static_tensor_response(bad,restored,selected,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    with pytest.raises(ValueError,match='conductive'):
        restore_static_tensor_response(state,restored,(bindings[1],),expected_binding=state_binding)
    state['state']['offsets'].add_(1)
    torch.testing.assert_close(decoded.offsets,response.offsets,rtol=0,atol=0)
    decoded.offsets.add_(1)
    with pytest.raises(ValueError,match='changed'):decoded.assert_current()

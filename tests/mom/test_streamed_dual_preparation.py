"""Complete streamed dual face/floating preparation and bound state replay."""
import copy
import runpy
import pytest
import torch
from witwin.maxwell.mom.harmonic_dual_current import DualHarmonicCurrentClosure
from witwin.maxwell.mom.dual_preparation_snapshot import snapshot_dual_preparation,restore_dual_preparation,_streamed_binding
from witwin.maxwell.mom.streamed_dual_state_snapshot import snapshot_streamed_dual_state,restore_streamed_dual_state
from witwin.maxwell.mom.dual_static_current import DualStaticExterior
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
from witwin.maxwell.mom.fixed_tangent_inverse import _FixedTangentInverse
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.linalg import StoppingRule
from test_dual_preparation_snapshot import bindings_for
from test_dual_action_snapshot import deterministic

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_complete_streamed_preparation_and_state_without_solves(tmp_path,monkeypatch,deterministic):
    v=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    bindings=bindings_for((v,))
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-22,maxiter=100,restart=20)
    closure=DualHarmonicCurrentClosure((v,),[2.],[5.],streamed_rule=rule,streamed_column_batch=3)
    laplace=FloatingPotentialOperator(bindings[0].surface_system,testing_order=6)
    source=torch.arange(24,device='cuda',dtype=torch.float64).cos().reshape(2,4,3)
    result=closure.current.solve(*source)
    exterior=DualStaticExterior(closure.current,result,bindings,laplace)
    prepared,identity=snapshot_dual_preparation(closure,bindings,laplace)
    state,state_identity=snapshot_streamed_dual_state(closure.current,result,exterior,*source)
    path=tmp_path/'prepared-state.pt'
    torch.save({'prepared':prepared,'binding':identity,'state':state,'state_binding':state_identity,
        'vertices':v.vertices.cpu(),'cells':v.cells.cpu(),'region_id':v.region_id},path)
    raw=torch.load(path,weights_only=True)
    fresh=TetrahedralVolumeMesh(v.vertices.clone(),v.cells.clone(),region_id=v.region_id)
    new_bindings=bindings_for((fresh,))
    points=v.vertices.new_tensor([[.5,.6,.7],[-.3,-.2,.4]])
    expected=exterior.fields(points)
    def forbidden(*args,**kwargs):raise AssertionError('Complete preparation decode cannot assemble or solve')
    with monkeypatch.context() as patch:
        for cls in (DualHarmonicCurrentClosure,FaceCurrentBalance,FloatingPotentialOperator,DualStaticExterior):
            patch.setattr(cls,'__init__',forbidden)
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        for name in ('inv','solve','svdvals'):patch.setattr(torch.linalg,name,forbidden)
        import witwin.maxwell.mom.static_volume_curl as curl
        patch.setattr(curl,'cell_green_potentials',forbidden)
        restored,floating=restore_dual_preparation(raw['prepared'],new_bindings,
            sigma_e=[v.vertices.new_tensor(2.,requires_grad=True)],
            sigma_m=[v.vertices.new_tensor(5.,requires_grad=True)],expected_binding=raw['binding'])
        decoded,fields=restore_streamed_dual_state(raw['state'],restored.current,new_bindings,floating,expected_binding=raw['state_binding'])
        for name in ('blocks','laplace','nonlinear'):
            changed=copy.deepcopy(prepared)
            if name=='blocks':changed[name][0]['gradients'][0,0,0]+=.1
            elif name=='laplace':changed[name]['single_layer'][0,0]+=.1
            else:changed[name]['column_batch']+=1
            with pytest.raises(ValueError,match='retained binding'):
                restore_dual_preparation(changed,new_bindings,sigma_e=[2.],sigma_m=[5.],expected_binding=identity)
        changed=copy.deepcopy(prepared);changed['blocks'][0]['gradients'][0,0,0]+=.1
        with pytest.raises(ValueError,match='discrete geometry'):
            restore_dual_preparation(changed,new_bindings,sigma_e=[2.],sigma_m=[5.],expected_binding=_streamed_binding(changed))
    for key,value in fields.fields(points).items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    assert torch.equal(decoded['electric'],result['electric'])
    from test_streamed_dual_bridge import check_streamed_bridge
    check_streamed_bridge(restored,new_bindings,floating,monkeypatch)
    from test_streamed_dual_bridge import check_streamed_recovery
    check_streamed_recovery(restored,new_bindings,floating,monkeypatch)

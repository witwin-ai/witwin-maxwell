"""Bound DC response restoration without assembly, solve or source inference."""
import copy
import pytest
import torch
from test_static_tensor_polarization import source
from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
from witwin.maxwell.mom.static_tensor_snapshot import snapshot_static_tensor_response,restore_static_tensor_response
from witwin.maxwell.mom.volume_h_snapshot import _digest

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def prepared():
    particular,a,p=source()
    system=particular.binding.surface_system
    op=StaticTensorInterfaceOperator(system,torch.eye(3,device=a.device,dtype=a.dtype),(a,))
    response=op.solve_uniform(a.new_tensor([.2,-.1,.3]),offset=.7,
                             bindings=(particular.binding,),polarizations=(p,))
    return response,particular.binding


def test_static_response_exact_decode_without_assembly_or_solve(monkeypatch,tmp_path):
    response,volume=prepared()
    payload,binding=snapshot_static_tensor_response(response)
    path=tmp_path/'static.pt';torch.save(dict(payload=payload,binding=binding),path)
    loaded=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Decode attempted assembly or solve')
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'_layers',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'solve_uniform',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=restore_static_tensor_response(loaded['payload'],response.operator,(volume,),expected_binding=loaded['binding'])
    for region,coords in ((1,[.021,.019,.023]),(0,[.2,.2,.2])):
        query=response.field.new_tensor([coords])
        for got,expected in zip(restored.fields(query,region),response.fields(query,region)):
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
    for name in ('field','potential','conormal','normal_p'):
        torch.testing.assert_close(getattr(restored,name),getattr(response,name),rtol=0,atol=0)
    payload['state']['potential'].add_(1)
    assert torch.equal(restored.potential,response.potential)
    restored.sources[0].polarization.add_(1)
    with pytest.raises(ValueError,match='changed'):restored.fields(response.field.new_tensor([[.02,.02,.02]]),1)


def test_static_response_rejects_rehashed_state_and_foreign_preparation():
    response,volume=prepared();payload,binding=snapshot_static_tensor_response(response)
    for component in ('potential','normal_p','conormal'):
        bad=copy.deepcopy(payload);bad['state'][component].add_(.01)
        with pytest.raises(ValueError,match='owning state binding'):
            restore_static_tensor_response(bad,response.operator,(volume,),expected_binding=binding)
    bad=copy.deepcopy(payload);bad['state']['normal_p'].add_(.01)
    with pytest.raises(ValueError,match='boundary polarization'):
        restore_static_tensor_response(bad,response.operator,(volume,),expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    other,_=prepared()
    other.operator.matrix.add_(1)
    with pytest.raises(ValueError,match='changed'):
        restore_static_tensor_response(payload,other.operator,(volume,),expected_binding=binding)
    with pytest.raises(ValueError,match='inventory'):
        restore_static_tensor_response(payload,response.operator,(),expected_binding=binding)


def test_static_response_rejects_valid_different_operator_and_volume_numbering():
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh,bind_volume_region
    response,volume=prepared();payload,binding=snapshot_static_tensor_response(response)
    op=response.operator
    changed=StaticTensorInterfaceOperator(op.system,1.1*op.exterior.coefficient,
                                         tuple(k.coefficient for k in op.interiors))
    with pytest.raises(ValueError,match='preparation differs'):
        restore_static_tensor_response(payload,changed,(volume,),expected_binding=binding)
    mesh=volume.volume
    permutation=mesh.cells.new_tensor([1,0,2,3])
    inverse=torch.argsort(permutation)
    renumbered=TetrahedralVolumeMesh(mesh.vertices[permutation],inverse[mesh.cells],region_id=1)
    alternate=bind_volume_region(renumbered,op.system)
    with pytest.raises(ValueError,match='volume geometry differs'):
        restore_static_tensor_response(payload,op,(alternate,),expected_binding=binding)
    malformed=copy.deepcopy(payload);malformed['receipt']['physical_qualification']=True
    with pytest.raises(ValueError,match='diagnostics'):
        restore_static_tensor_response(malformed,op,(volume,),expected_binding=dict(
            version=malformed['version'],sha256=_digest(malformed)))


def test_preparation_and_response_restore_without_original_operator(monkeypatch,tmp_path):
    from witwin.maxwell.mom.static_tensor_snapshot import snapshot_static_tensor_preparation,restore_static_tensor_preparation
    from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
    response,volume=prepared()
    preparation,preparation_binding=snapshot_static_tensor_preparation(response.operator)
    state,state_binding=snapshot_static_tensor_response(response)
    path=tmp_path/'prepared-state.pt'
    torch.save(dict(preparation=preparation,state=state),path)
    loaded=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Restore attempted numerical preparation')
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'_layers',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        # Kernel construction is bypassed; coefficient SPD admission remains required.
        patch.setattr(StaticTensorGreen,'__init__',forbidden)
        restored=restore_static_tensor_preparation(loaded['preparation'],response.operator.system,expected_binding=preparation_binding)
    assert restored is not response.operator
    original_init=StaticTensorGreen.__init__
    def rounded_factor(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.factor=self.factor+torch.eye(3,device=self.factor.device,dtype=self.factor.dtype)*1e-15
        self.root_determinant=self.factor.diagonal().prod();self._binding=self._state()
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorGreen,'__init__',rounded_factor)
        decoded=restore_static_tensor_response(loaded['state'],restored,(volume,),expected_binding=state_binding)
    torch.testing.assert_close(decoded.sources[0].kernel.factor,restored.interiors[0].factor,rtol=0,atol=0)
    torch.testing.assert_close(restored.matrix,response.operator.matrix,rtol=0,atol=0)
    for old,new in zip((response.operator.exterior,*response.operator.interiors),(restored.exterior,*restored.interiors)):
        torch.testing.assert_close(new.factor,old.factor,rtol=0,atol=0)
    for region,coords in ((1,[.021,.019,.023]),(0,[.2,.2,.2])):
        point=response.field.new_tensor([coords])
        for got,want in zip(decoded.fields(point,region),response.fields(point,region)):
            torch.testing.assert_close(got,want,rtol=0,atol=0)
    with pytest.raises(ValueError,match='budget'):
        restore_static_tensor_preparation(preparation,restored.system,expected_binding=preparation_binding,max_matrix_bytes=1)
    bad=copy.deepcopy(preparation);bad['kernels'][0]['factor'][0,0]*=2
    with pytest.raises(ValueError,match='owning state binding'):
        restore_static_tensor_preparation(bad,restored.system,expected_binding=preparation_binding)
    with pytest.raises(ValueError,match='factors do not match'):
        restore_static_tensor_preparation(bad,restored.system,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    preparation['matrix'].add_(1)
    torch.testing.assert_close(restored.matrix,response.operator.matrix,rtol=0,atol=0)


def test_preparation_independent_spd_and_response_factor_identity():
    from witwin.maxwell.mom.static_tensor_snapshot import snapshot_static_tensor_preparation,restore_static_tensor_preparation
    response,volume=prepared();payload,binding=snapshot_static_tensor_preparation(response.operator)
    state,state_binding=snapshot_static_tensor_response(response)
    bad=copy.deepcopy(payload);record=bad['kernels'][0]
    record['coefficient']=torch.diag(torch.tensor([1.,1.,-1e-16],dtype=torch.float64))
    record['factor']=torch.diag(torch.tensor([1.,1.,1e-8],dtype=torch.float64))
    record['root_determinant']=record['factor'].diagonal().prod()
    with pytest.raises(ValueError,match='independently be positive definite'):
        restore_static_tensor_preparation(bad,response.operator.system,expected_binding=dict(version=bad['version'],sha256=_digest(bad)))
    changed=copy.deepcopy(payload)
    changed['kernels'][0]['factor'][0,0]+=1e-15
    changed['kernels'][0]['root_determinant']=changed['kernels'][0]['factor'].diagonal().prod()
    restored=restore_static_tensor_preparation(changed,response.operator.system,
        expected_binding=dict(version=changed['version'],sha256=_digest(changed)))
    with pytest.raises(ValueError,match='preparation differs'):
        restore_static_tensor_response(state,restored,(volume,),expected_binding=state_binding)

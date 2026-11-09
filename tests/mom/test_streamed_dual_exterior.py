"""Exterior neutral-layer recovery reuses admitted streamed current preparation."""
import pytest
import torch
from witwin.maxwell.mom.dual_action_snapshot import restore_dual_action
from witwin.maxwell.mom.dual_static_current import DualStaticCurrentBalance,DualStaticExterior
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from test_dual_preparation_snapshot import bindings_for

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def check_saved_streamed_exterior(path):
    """Reuse a retained current preparation; do not repeat its admission."""
    import runpy
    first=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    second=TetrahedralVolumeMesh(1.2*first.vertices+first.vertices.new_tensor([.25,.03,-.02]),first.cells,region_id=2)
    volumes=(second,first)
    raw=torch.load(path,weights_only=True)
    balance=restore_dual_action(raw['record'],tuple(FaceCurrentBalance(v) for v in volumes),expected_binding=raw['binding'])
    dense=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    source=torch.arange(48,device='cuda',dtype=torch.float64).cos().reshape(2,8,3).requires_grad_()
    reference=dense.solve(*source)
    actual=balance.solve(*source)
    by_region={b.volume.region_id:b for b in bindings_for(tuple(sorted(volumes,key=lambda v:v.region_id)))}
    bindings=tuple(by_region[v.region_id] for v in volumes)
    laplace=FloatingPotentialOperator(bindings[0].surface_system,testing_order=6)
    exterior=DualStaticExterior(balance,actual,bindings,laplace)
    expected=DualStaticExterior(dense,reference,bindings,laplace)
    points=first.vertices.new_tensor([[.8,.7,.9],[-.3,-.2,.4]])
    fields=exterior.fields(points);target=expected.fields(points)
    for key in fields:torch.testing.assert_close(fields[key],target[key],rtol=1e-8,atol=1e-11)
    gradient,=torch.autograd.grad(sum(v.square().sum() for v in fields.values()),source)
    reference_gradient,=torch.autograd.grad(sum(v.square().sum() for v in target.values()),source)
    torch.testing.assert_close(gradient,reference_gradient,rtol=1e-8,atol=1e-11)
    assert not exterior.receipt['physical_qualification']
    signature=balance._state()
    assert signature==balance._state()
    balance.action.curl.points[0,0]+=.01
    with pytest.raises(ValueError,match='binding changed'):exterior.fields(points)
    return {'field_and_source_gradient_match':True,'physical_qualification':False}


def check_saved_streamed_state(path,*,output_path=None):
    import copy
    import runpy
    from witwin.maxwell.mom.streamed_dual_state_snapshot import (
        snapshot_streamed_dual_state,restore_streamed_dual_state,validate_response_witness)
    from witwin.maxwell.mom.fixed_tangent_inverse import _FixedTangentInverse
    first=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    second=TetrahedralVolumeMesh(1.2*first.vertices+first.vertices.new_tensor([.25,.03,-.02]),first.cells,region_id=2)
    volumes=(second,first)
    raw=torch.load(path,weights_only=True)
    balance=restore_dual_action(raw['record'],tuple(FaceCurrentBalance(v) for v in volumes),expected_binding=raw['binding'])
    by_region={b.volume.region_id:b for b in bindings_for((first,second))}
    bindings=tuple(by_region[v.region_id] for v in volumes)
    laplace=FloatingPotentialOperator(bindings[0].surface_system,testing_order=6)
    source=torch.arange(48,device='cuda',dtype=torch.float64).cos().reshape(2,8,3)
    state=balance.solve(*source)
    exterior=DualStaticExterior(balance,state,bindings,laplace)
    payload,identity=snapshot_streamed_dual_state(balance,state,exterior,*source)
    points=first.vertices.new_tensor([[.8,.7,.9],[-.3,-.2,.4]])
    expected=exterior.fields(points)
    def forbidden(*args,**kwargs):raise AssertionError('State replay must not integrate, admit or solve')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        patch.setattr(FloatingPotentialOperator,'__init__',forbidden)
        patch.setattr(DualStaticExterior,'__init__',forbidden)
        import witwin.maxwell.mom.static_volume_curl as curl
        patch.setattr(curl,'cell_green_potentials',forbidden)
        result,decoded=restore_streamed_dual_state(payload,balance,bindings,laplace,expected_binding=identity)
        for kind in ('current','curl','response','source','receipt'):
            changed=copy.deepcopy(payload)
            if kind=='current':changed['state']['state']['electric_current'][0,0]+=.001
            elif kind=='receipt':changed['state']['receipt']['physical_qualification']=True
            else:changed['witness'][{'curl':'curl_m','response':'electric_response','source':'electric_source'}[kind]][0,0]+=.001
            with pytest.raises(ValueError,match='owning identity'):
                restore_streamed_dual_state(changed,balance,bindings,laplace,expected_binding=identity)
        owning_field=state['electric'].flatten()+1e-10
        response=validate_response_witness(result,*source,owning_field,1e-8)
        assert float((owning_field-response).norm())>0
        assert torch.equal(response,payload['witness']['electric_response'].to(response.device).flatten())
        with pytest.raises(ValueError,match='coupled equation'):
            validate_response_witness(result,*source,owning_field+1e-3,1e-8)
        with pytest.raises(ValueError,match='owning HB source'):
            validate_response_witness(result,source[0]+1e-5,source[1],owning_field,1e-8)
    for key,value in decoded.fields(points).items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    if output_path is not None:torch.save({'payload':payload,'binding':identity},output_path)
    return {'decode_only_fields_match':True,'original_response_residual_preserved':True,'physical_qualification':False}

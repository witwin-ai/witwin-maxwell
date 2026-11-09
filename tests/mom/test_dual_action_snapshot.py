"""Decode-only dual action preparation and exact native field replay."""
import copy
import runpy
import pytest
import torch
from witwin.maxwell.mom.dual_current_action import DualStaticCurrentAction,StreamedDualCurrentBalance
from witwin.maxwell.mom.dual_action_snapshot import snapshot_dual_action,restore_dual_action,_binding
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.static_current_projector import StaticCurrentProjector
from witwin.maxwell.mom.fixed_tangent_inverse import _FixedTangentInverse
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(autouse=True)
def deterministic(monkeypatch):
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    try:yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


def test_dual_preparation_decode_without_kernels_or_admission(tmp_path,monkeypatch):
    volume=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    other=TetrahedralVolumeMesh(1.2*volume.vertices+volume.vertices.new_tensor([.25,.03,-.02]),volume.cells,region_id=2)
    volumes=(other,volume)
    fresh=tuple(FaceCurrentBalance(TetrahedralVolumeMesh(v.vertices.clone(),v.cells.clone(),region_id=v.region_id)) for v in volumes)
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-22,maxiter=100,restart=20)
    action=DualStaticCurrentAction(volumes,[2.,3.],[5.,7.],projection_rule=rule,column_batch=4,target_batch=5,source_batch=3)
    balance=StreamedDualCurrentBalance(action,rule,column_batch=8)
    source=torch.arange(48,device='cuda',dtype=torch.float64).cos().reshape(2,8,3)
    expected=balance.solve(*source)
    record,binding=snapshot_dual_action(balance)
    path=tmp_path/'dual-action.pt';torch.save({'record':record,'binding':binding},path)
    raw=torch.load(path,weights_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Decode-only preparation must not assemble, admit or solve')
    with monkeypatch.context() as patch:
        for cls in (DualStaticCurrentAction,StreamedDualCurrentBalance,StaticCurrentProjector,FaceCurrentBalance):
            patch.setattr(cls,'__init__',forbidden)
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        import witwin.maxwell.mom.static_volume_curl as curl
        patch.setattr(curl,'cell_green_potentials',forbidden)
        for name in ('inv','solve','svdvals'):patch.setattr(torch.linalg,name,forbidden)
        restored=restore_dual_action(raw['record'],fresh,expected_binding=raw['binding'])
        assert restored.bound==balance.bound
        for key in ('electric','rule','bound','curl'):
            changed=copy.deepcopy(record)
            if key=='electric':changed[key][0]+=1
            elif key=='rule':changed[key]['rtol']*=2
            elif key=='bound':changed[key]['sigma_min_lower']*=.5
            else:changed[key]['source_batch']+=1
            with pytest.raises(ValueError,match='binding'):
                restore_dual_action(changed,fresh,expected_binding=binding)
        changed=copy.deepcopy(record);changed['bound']['sigma_min_lower']*=.5
        with pytest.raises(ValueError,match='admission'):
            restore_dual_action(changed,fresh,expected_binding=_binding(changed))
        with pytest.raises(ValueError,match='geometry'):
            restore_dual_action(record,fresh[::-1],expected_binding=binding)
    actual=restored.solve(*source)
    for key in ('electric','magnetic','electric_current','magnetic_current'):
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    for key in ('electric_potentials','magnetic_potentials'):
        for a,b in zip(actual[key],expected[key]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    from test_streamed_dual_exterior import check_saved_streamed_exterior,check_saved_streamed_state
    check_saved_streamed_exterior(path)
    check_saved_streamed_state(path)
    restored.action.magnetic[0]+=1
    with pytest.raises(ValueError,match='binding changed'):restored.solve(*source)


def test_nonlinear_preparation_rebinds_live_materials_without_resolve(monkeypatch):
    from witwin.maxwell.mom.harmonic_dual_current import DualHarmonicCurrentClosure
    from witwin.maxwell.mom.dual_action_snapshot import snapshot_dual_closure,restore_dual_closure,_closure_binding
    volume=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    fresh=FaceCurrentBalance(TetrahedralVolumeMesh(volume.vertices.clone(),volume.cells.clone(),region_id=volume.region_id))
    parameters=volume.vertices.new_tensor([2.,5.],requires_grad=True)
    rule=StoppingRule(preconditioner='none',rtol=1e-13,atol=1e-22,maxiter=100,restart=20)
    original=DualHarmonicCurrentClosure((volume,),[parameters[0]],[parameters[1]],streamed_rule=rule,streamed_column_batch=3)
    source=torch.arange(12,device='cuda',dtype=torch.float64).cos().requires_grad_()
    reference=original.material_inverse.electric_response(source,*original.material_coefficients)
    expected_grad=torch.autograd.grad(reference.square().sum(),(source,parameters))
    record,binding=snapshot_dual_closure(original)
    live=parameters.detach().clone().requires_grad_()
    def forbidden(*args,**kwargs):raise AssertionError('Nonlinear decode must not prepare or solve')
    with monkeypatch.context() as patch:
        for cls in (DualHarmonicCurrentClosure,DualStaticCurrentAction,StreamedDualCurrentBalance,StaticCurrentProjector):
            patch.setattr(cls,'__init__',forbidden)
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        import witwin.maxwell.mom.static_volume_curl as curl
        patch.setattr(curl,'cell_green_potentials',forbidden)
        restored=restore_dual_closure(record,(fresh,),sigma_e=[live[0]],sigma_m=[live[1]],expected_binding=binding)
        with pytest.raises(ValueError,match='sampled materials'):
            restore_dual_closure(record,(fresh,),sigma_e=[2.1],sigma_m=[5.],expected_binding=binding)
        changed=copy.deepcopy(record);changed['column_batch']+=1
        with pytest.raises(ValueError,match='policies differ'):
            restore_dual_closure(changed,(fresh,),sigma_e=[2.],sigma_m=[5.],expected_binding=_closure_binding(changed))
    actual=restored.material_inverse.electric_response(source,*restored.material_coefficients)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    gradients=torch.autograd.grad(actual.square().sum(),(source,live),retain_graph=True)
    for a,b in zip(gradients,expected_grad):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with torch.no_grad():live[0]+=.1
    with pytest.raises(RuntimeError,match='mutated'):torch.autograd.grad(actual.sum(),live)

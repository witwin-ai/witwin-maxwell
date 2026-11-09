"""HB bridge checks using an already admitted streamed preparation."""
import copy
from dataclasses import replace
import pytest
import torch
from witwin.maxwell.mom.harmonic_state import HarmonicCoupledState
from witwin.maxwell.mom.harmonic_dual_bridge import validate_dual,query_dual
from witwin.maxwell.mom.streamed_dual_state_snapshot import snapshot_streamed_dual_state
from witwin.maxwell.mom.dual_static_current import DualStaticExterior
from witwin.maxwell.mom.fixed_tangent_inverse import _FixedTangentInverse


def check_streamed_bridge(closure,bindings,laplace,monkeypatch):
    sample=closure.current.points
    source=torch.arange(sample.numel(),device=sample.device,dtype=sample.dtype).cos().reshape_as(sample)
    result=closure.current.solve(source,torch.zeros_like(source))
    exterior=DualStaticExterior(closure.current,result,bindings,laplace)
    payload,identity=snapshot_streamed_dual_state(closure.current,result,exterior,source,torch.zeros_like(source))
    prepared={'static':{'current_closure':closure,'floating_operator':laplace},'bindings':(bindings,)}
    field=result['electric'][None].to(torch.complex128)
    state=HarmonicCoupledState(sample.new_zeros(1),field,torch.zeros_like(field),
        source[None].to(field.dtype),result['electric_current'][None].to(field.dtype),(),
        exterior.electric_layer.density.clone(),{'pump_scale':1.},id(prepared),
        dc_dual_state=payload,dc_dual_binding=identity)
    points=sample.new_tensor([[.5,.6,.7]])
    expected=exterior.fields(points)
    threshold=1e-10
    def forbidden(*args,**kwargs):raise AssertionError('HB bridge replay cannot integrate or solve')
    with monkeypatch.context() as patch:
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        patch.setattr(DualStaticExterior,'__init__',forbidden)
        import witwin.maxwell.mom.static_volume_curl as curl
        patch.setattr(curl,'cell_green_potentials',forbidden)
        response=validate_dual(prepared,state,threshold)
        torch.testing.assert_close(response,result['electric'].flatten(),rtol=0,atol=0)
        fields,decoded=query_dual(prepared,state,points,0)
        for key,value in expected.items():torch.testing.assert_close(fields[key],value,rtol=0,atol=0)
        assert decoded[0]['replay_receipt']['inverse_recomputed'] is False
        changed=copy.deepcopy(payload);changed['witness']['electric_response'][0,0]+=1.
        with pytest.raises(ValueError,match='independent owning identity'):
            validate_dual(prepared,replace(state,dc_dual_state=changed),threshold)
        with pytest.raises(ValueError,match='independent owning identity'):
            query_dual(prepared,replace(state,dc_dual_binding=None),points,0)
        changed_source=state.dissipative_current.clone();changed_source[0,0,0]+=1e-12
        with pytest.raises(ValueError,match='owning HB source'):
            validate_dual(prepared,replace(state,dissipative_current=changed_source),threshold)
        changed_field=field.clone();changed_field[0,0,0]+=2*threshold
        with pytest.raises(ValueError,match='witnessed coupled equation'):
            validate_dual(prepared,replace(state,cell_electric=changed_field),threshold)
        # Return the witnessed response, including a nonzero accepted defect,
        # so the outer cross-harmonic norm does not silently lose its DC term.
        changed_field=field.clone();changed_field[0,0,0]+=threshold/4
        accepted=validate_dual(prepared,replace(state,cell_electric=changed_field),threshold)
        assert bool((accepted-changed_field[0].real.flatten()).norm()>0)
    for factor in (1e-22,1e-30):
        tiny=closure.current.solve(factor*source,torch.zeros_like(source))
        for name in ('electric','magnetic','electric_current','magnetic_current'):
            torch.testing.assert_close(tiny[name]/factor,result[name],rtol=1e-8,atol=1e-12)
        assert tiny['receipt']['coupled_current_residual']<=tiny['receipt']['threshold']
    state.dc_dual_binding['sha256']='mutated'
    with pytest.raises(RuntimeError,match='mutat'):state.assert_current()


def check_streamed_recovery(closure,bindings,laplace,monkeypatch):
    from witwin.maxwell.mom.harmonic_dual_bridge import recover_dual,dual_ohmic_current
    from witwin.maxwell.mom.streamed_dual_state_snapshot import restore_streamed_dual_state
    sample=closure.current.points
    source=torch.arange(sample.numel(),device=sample.device,dtype=sample.dtype).sin().reshape_as(sample).requires_grad_()
    reference=closure.current.solve(source,torch.zeros_like(source),conductivities=closure.material_coefficients)
    field=torch.stack((reference['electric'].to(torch.complex128),torch.zeros_like(source,dtype=torch.complex128)))
    prepared={'static':{'current_closure':closure,'floating_operator':laplace},'bindings':(bindings,)}
    def forbidden(*args,**kwargs):raise AssertionError('Recovery must not resolve the nonlinear root')
    with monkeypatch.context() as patch:
        patch.setattr(type(closure),'solve',forbidden)
        payload,charge,live,identity=recover_dual(closure,prepared,field,0.,electric_source=source,threshold=1e-10)
        assert torch.equal(payload['witness']['electric_source'],source.detach().cpu())
        assert payload['state']['receipt']['hb_recovery']['nonlinear_root_recomputed'] is False
        query=sample.new_tensor([[.5,.6,.7]])
        value=live[1].fields(query)['electric_V_per_m'].square().sum()
        gradient=torch.autograd.grad(value,source,retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and bool(gradient.abs().max()>0)
        material_inputs=tuple(v for v in closure.material_inputs if isinstance(v,torch.Tensor) and v.requires_grad)
        if material_inputs:
            recovered=dual_ohmic_current(closure,live[0]['electric'])+source
            native=live[0]['electric_current']
            torch.testing.assert_close(recovered,native,rtol=0,
                atol=8*torch.finfo(sample.dtype).eps*float(source.detach().abs().max()))
            weight=torch.arange(source.numel(),device=sample.device,dtype=sample.dtype).cos().reshape_as(source)
            actual=torch.autograd.grad((weight*recovered).sum(),material_inputs,retain_graph=True)
            expected=torch.autograd.grad((weight*native).sum(),material_inputs,retain_graph=True)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-14)
        changed=field.clone();changed[0,0,0]+=1e-6
        with pytest.raises(ValueError,match='owning HB threshold'):
            recover_dual(closure,prepared,changed,0.,electric_source=source,threshold=1e-10)
        with pytest.raises(ValueError,match='owning HB threshold'):
            recover_dual(closure,prepared,field,0.,electric_source=source)
    with monkeypatch.context() as patch:
        patch.setattr(_FixedTangentInverse,'_solve',forbidden)
        restored,exterior=restore_streamed_dual_state(payload,closure.current,bindings,laplace,expected_binding=identity)
        assert torch.equal(charge,exterior.electric_layer.density)
        changed=copy.deepcopy(payload)
        changed['state']['receipt']['hb_recovery']['threshold_V_per_m_l2']=1.
        with pytest.raises(ValueError,match='independent owning identity'):
            restore_streamed_dual_state(changed,closure.current,bindings,laplace,expected_binding=identity)

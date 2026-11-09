"""Actual factorized carrier points, complex maps, storage and native rejection."""
import json
import weakref
import pytest
import torch
import test_fmm_changing_carrier_shift as original
from witwin.maxwell.mom.fmm_directional_shift import DirectionalChangingCarrierShift,ChangingCarrierPhaseLayout
from witwin.maxwell.mom.fmm_shift import SeparableMultipoleShift

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_center_axis_temporaries_released_before_next_allocation(monkeypatch,tmp_path):
    layout=SeparableMultipoleShift(3,device='cuda',dtype=torch.float64)
    displacement=torch.tensor([.03,-.02,.01],device='cuda',dtype=torch.float64)
    value=torch.randn((2,layout.count,4),device='cuda',dtype=torch.complex128)
    native_getitem,native_mul=torch.Tensor.__getitem__,torch.Tensor.__mul__
    observed=[]
    for adjoint in (False,True):
        for reject_underflow in (False,True):
            gathers,products=[],[]
            def getitem(tensor,key):
                if (isinstance(key,tuple) and len(key)==3 and key[0] is Ellipsis
                        and isinstance(key[1],torch.Tensor) and key[1].shape==(layout.count,layout.order+1)):
                    assert all(ref() is None for ref in gathers+products)
                    result=native_getitem(tensor,key)
                    gathers.append(weakref.ref(result))
                    return result
                return native_getitem(tensor,key)
            def multiply(tensor,other):
                result=native_mul(tensor,other)
                if gathers and tensor is gathers[-1]():products.append(weakref.ref(result))
                return result
            with monkeypatch.context() as patch:
                patch.setattr(torch.Tensor,'__getitem__',getitem)
                patch.setattr(torch.Tensor,'__mul__',multiply)
                result=layout.apply(displacement,value,adjoint=adjoint,reject_underflow=reject_underflow)
            assert result.shape==value.shape
            assert len(gathers)==len(products)==3
            assert all(ref() is None for ref in gathers+products)
            observed.append(dict(adjoint=adjoint,reject_underflow=reject_underflow,
                observed_gathers=len(gathers),observed_products=len(products)))
    (tmp_path/'axis_lifetime.json').write_text(json.dumps(observed,indent=2)+'\n')
    print(dict(axis_temporary_lifetimes=observed),flush=True)


@pytest.mark.parametrize('ps,pt,L,wave,angle,radius',[
    (0,0,0,0.,.5,.05),(3,3,0,5.+.7j,0.,.02),
    (6,3,3,5.+.7j,.02,.01),(12,3,9,80.+2j,.05,.03),(12,0,12,2000.+2j,.002,.05)])
def test_compact_independent_actual_points_four_H_current_higher(ps,pt,L,wave,angle,radius,monkeypatch,tmp_path):
    # Reuse the independent physical oracle and its unchanged strict gates for
    # a new storage path, without collecting/rerunning the dense test module.
    def compact(**kw):return DirectionalChangingCarrierShift(**kw,phase_storage='compact_phase_v1')
    monkeypatch.setattr(original,'DirectionalChangingCarrierShift',compact)
    original.test_actual_padded_source_and_receiver_points_full_H_current_and_higher_derivatives(
        ps,pt,L,wave,angle,radius,tmp_path)


def test_compact_unique_owned_shared_exact_dense_columns_preflight_and_saved_layout_guards(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_shift as module
    args=original._args()
    dense=DirectionalChangingCarrierShift(**args)
    owned=DirectionalChangingCarrierShift(**args,phase_storage='compact_phase_v1')
    shared=DirectionalChangingCarrierShift(**args,phase_storage='compact_phase_v1',
        phase_layout=owned.phase_layout,center_layout=owned.center_layout)
    phase_bytes=sum(t.untyped_storage().nbytes() for t in owned.phase_layout.tensors())
    center_bytes=sum(t.untyped_storage().nbytes() for t in owned.center_layout.tensors())
    expected_shared=128+32*owned.phase_count
    for op,expected in [(owned,expected_shared+phase_bytes+center_bytes),(shared,expected_shared)]:
        tensors=op.tensors()
        assert len({t.untyped_storage().data_ptr() for t in tensors})==len(tensors)
        assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
        assert sum(t.untyped_storage().nbytes() for t in tensors)==op.retained_bytes==op.planned_retained_bytes==expected
    for name in ('phase_layout','center_layout'):
        for a,b in zip(getattr(owned,name).tensors(),getattr(shared,name).tensors()):
            assert torch.equal(a,b)
    for name in ('source_coefficients','receiver_coefficients','source_phase','receiver_phase'):
        assert torch.equal(getattr(owned,name),getattr(shared,name))
    errors,shared_errors,repeated_errors={},{},{}
    for name,count in [('source',dense.source_count),('receiver',dense.target_count)]:
        eye=torch.eye(count,device='cuda',dtype=torch.complex128)[...,None].expand(-1,-1,4)
        errors[name]=original._relative(getattr(owned,name)(eye),getattr(dense,name)(eye))
        assert errors[name]<1e-11
        actual=getattr(owned,name)(eye)
        shared_errors[name]=original._relative(actual,getattr(shared,name)(eye))
        repeated_errors[name]=original._relative(actual,getattr(owned,name)(eye))
        assert shared_errors[name]<1e-11 and repeated_errors[name]<1e-11
    def forbidden(*a,**kw):raise AssertionError('Rejected compact transfer allocated layouts')
    with monkeypatch.context() as patch:
        patch.setattr(module,'ChangingCarrierPhaseLayout',forbidden);patch.setattr(module,'SeparableMultipoleShift',forbidden)
        for kw,exc in [({'retained_budget_bytes':owned.retained_bytes-1},MemoryError),
                ({'tensor_workspace_budget_bytes':owned.planned_tensor_workspace_bytes-1},MemoryError),
                ({'phase_relative_budget':1e-30},ValueError),({'source_order':11},ValueError)]:
            altered=dict(args);altered.update(kw)
            with pytest.raises(exc):DirectionalChangingCarrierShift(**altered,phase_storage='compact_phase_v1')
    for name,count,layout_name in [('source',shared.source_count,'phase_layout'),('receiver',shared.target_count,'center_layout')]:
        x=torch.randn((count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
        output=getattr(shared,name)(x)
        getattr(shared,layout_name).tensors()[0].add_(0)
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,x,torch.ones_like(output))
        owned=DirectionalChangingCarrierShift(**args,phase_storage='compact_phase_v1')
        shared=DirectionalChangingCarrierShift(**args,phase_storage='compact_phase_v1',
            phase_layout=owned.phase_layout,center_layout=owned.center_layout)
    record=dict(dense_owned_bytes=dense.retained_bytes,compact_owned_bytes=owned.retained_bytes,
        compact_shared_metadata_bytes=shared.retained_bytes,phase_layout_bytes=phase_bytes,center_layout_bytes=center_bytes,
        dense_all_column_errors=errors,owned_shared_action_errors=shared_errors,
        same_instance_repeated_action_errors=repeated_errors,layouts_and_coefficients_exact=True,
        bit_exact_actions_claimed=False,all_unique_storages=True,qualification=False)
    (tmp_path/'ownership.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_compact_native_component_coefficient_current_underflow_overflow_and_bounded_empty_maps(monkeypatch,tmp_path):
    args=original._args()
    op=DirectionalChangingCarrierShift(**args,phase_storage='compact_phase_v1')
    for change in [dict(wave_number=1e-100,target_order=0,phase_order=12),
            dict(displacement=args['displacement'].new_zeros(3),direction=args['direction'].new_tensor([0.,0.,1.]),
                target_direction=args['direction'].new_tensor([1e-124,1e-100,1.]),wave_number=1e-200,
                child_radius=args['child_radius'].new_tensor(1e290),source_order=1,target_order=0,phase_order=1)]:
        kw=dict(args);kw.update(change)
        with pytest.raises(ValueError,match='precision'):DirectionalChangingCarrierShift(**kw,phase_storage='compact_phase_v1')
    with pytest.raises(ValueError,match='underflow'):
        op.source(torch.full((op.source_count,4),1e-320,device='cuda',dtype=torch.complex128))
    with pytest.raises(ValueError,match='precision|nonfinite'):
        op.receiver(torch.full((op.target_count,4),1e308,device='cuda',dtype=torch.complex128))
    native=torch.isfinite;observed=[]
    def bounded(value):
        if value.is_complex() and value.ndim>=3:observed.append(value.shape[0]);assert value.shape[0]<=4
        return native(value)
    with monkeypatch.context() as patch:
        patch.setattr(torch,'isfinite',bounded)
        for name,ni,no in [('source',op.source_count,op.target_count),('receiver',op.target_count,op.source_count)]:
            for H in (False,True):
                a,b=(no,ni) if H else (ni,no)
                value=torch.randn((3,5,a,4),device='cuda',dtype=torch.complex128).transpose(0,1)
                assert getattr(op,name)(value,adjoint=H).shape==(5,3,b,4)
                assert getattr(op,name)(value[:0],adjoint=H).shape==(0,3,b,4)
                with pytest.raises(ValueError,match='finite'):getattr(op,name)(value*float('nan'),adjoint=H)
    assert observed and max(observed)==4
    record=dict(maximum_finite_check_RHS=max(observed),rhs_batch=4,
        nonzero_component_and_coefficient_underflow_rejected=True,nonzero_current_underflow_rejected=True,
        native_admission_identical_to_dense=False,qualification=False)
    (tmp_path/'native.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)

"""Independent high-precision continuous tails and native configured FFT admission."""
from dataclasses import replace
import json
import math
import mpmath as mp
import pytest
import torch
from witwin.maxwell.mom.fmm_error import (directional_multipole_truncation_bound,
    directional_multipole_truncation_bounds_many,directional_bound_workspace_bytes,MultipoleTruncationError)
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget,KernelBoundUnavailable
from witwin.maxwell.mom.fmm_directional_fft import CartesianDirectionalFFTTranslation
from witwin.maxwell.mom.fmm_expansion import monomials,multipole_moments
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _relative(a,b):return float(((a-b).norm()/b.norm()).detach())


def _mp_polynomial_fields(center,direction,x,y,wave,order):
    # Independent oracle only: arbitrary-precision scalar bivariate derivatives
    # do not use the CUDA Taylor recurrence, FFT, phase tables or native bounds.
    with mp.workdps(100):
        R,u,xx,yy=[tuple(mp.mpf(v) for v in values) for values in (center,direction,x,y)]
        k=mp.mpc(wave)
        def field(t,s,j):
            offset=tuple(t*a-s*b for a,b in zip(xx,yy))
            z=tuple(a+b for a,b in zip(R,offset));r=mp.sqrt(sum(a*a for a in z))
            g=mp.exp(1j*k*(r-sum(a*b for a,b in zip(u,offset))))/(4*mp.pi*r)
            return g if j==0 else g*(1j*k-1/r)*z[j-1]/r
        phase=mp.exp(1j*k*sum(a*(b-c) for a,b,c in zip(u,xx,yy)))
        approximate=[];exact=[]
        for j in range(4):
            value=mp.mpc(0)
            for n in range(order+1):
                coefficients=mp.taylor(lambda t:mp.diff(lambda s:field(t,s,j),0,n)/mp.factorial(n),0,order)
                value+=sum(coefficients)
            approximate.append(value*phase);exact.append(field(1,1,j)*phase)
        return [complex(v) for v in approximate],[float(abs(a-b)) for a,b in zip(approximate,exact)]


@pytest.mark.parametrize('order,wave,center,radius,offset',[
    (0,0.,[1.,.2,-.1],.03,.02),(3,5.+.7j,[1.,.2,-.1],.04,.01),
    (6,100.,[2000.,0.,0.],.5,0.),(6,100.+.002j,[2000.,400.,-200.],.5,.0002)])
def test_continuous_bound_mp_bivariate_polynomial_actual_fft_four_physical_fields(order,wave,center,radius,offset,tmp_path,monkeypatch):
    R=torch.tensor(center,device='cuda',dtype=torch.float64);u=R/R.norm()
    u=u+R.new_tensor([offset,-2*offset,3*offset]);u=u/u.norm()
    r=R.new_tensor(radius)
    bound=directional_multipole_truncation_bound(R,u,r,r,wave,order)
    if order>=6:
        budget=KernelErrorBudget(1e-8,1e-8)
        if order==6 and wave==100. and center==[2000.,0.,0.] and radius==.5 and offset==0.:
            # Preserve the original over-budget case as rejected at p6. Its
            # sampled physical accuracy is not continuous-budget admission.
            assert bound['gradient_component_absolute']>budget.gradient_component_absolute
            import witwin.maxwell.mom.fmm_directional_fft as module
            def forbidden(*a,**kw):raise AssertionError('Over-budget p6 reached translation preparation')
            with monkeypatch.context() as patch:
                patch.setattr(module.CartesianDirectionalFFTTranslation,'_prepare',forbidden)
                patch.setattr(module,'directional_helmholtz_taylor',forbidden)
                with pytest.raises(MultipoleTruncationError) as failure:
                    CartesianDirectionalFFTTranslation(R,u,wave,r,order,test_radius=r,source_radius=r,kernel_error_budget=budget)
            assert failure.value.receipt['admitted'] is False
            record=dict(order=order,wave=str(wave),continuous_bound=bound,budget=dict(green_absolute=1e-8,
                gradient_component_absolute=1e-8),admitted=False,refused_before_translation_allocation=True,
                original_positive_expectation_failed=True,actual_error_exceeds_budget_proven=False,qualification=False)
            (tmp_path/'rejected_p6.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
            return
        assert max(bound['green_absolute'],bound['gradient_component_absolute'])<1e-8
        assert bound['green_contour_fraction']<.05
    else:budget=KernelErrorBudget(bound['green_absolute']*1.01,bound['gradient_component_absolute']*1.01)
    op=CartesianDirectionalFFTTranslation(R,u,wave,r,order,test_radius=r,source_radius=r,kernel_error_budget=budget)
    x=R.new_tensor([[.3,.4,.5]])*r;y=R.new_tensor([[-.2,.1,.6]])*r
    assert bool((x.norm(dim=-1)<=r).all() & (y.norm(dim=-1)<=r).all())
    source_phase=torch.exp(-1j*wave*(y@u));receiver_phase=torch.exp(1j*wave*(x@u))
    moments=multipole_moments(y,source_phase[None],order)
    actual=torch.einsum('tb,sfb->tsf',monomials(x,order).to(torch.complex128),op.apply(moments))*receiver_phase[:,None,None]
    polynomial,ideal_errors=_mp_polynomial_fields(R.tolist(),u.tolist(),x[0].tolist(),y[0].tolist(),wave,order)
    assert ideal_errors[0]<=bound['green_absolute']
    assert max(ideal_errors[1:])<=bound['gradient_component_absolute']
    polynomial=actual.new_tensor(polynomial).reshape_as(actual)
    native_polynomial_error=_relative(actual,polynomial)
    assert bool((polynomial.abs()>0).all())
    native_components=[_relative(actual[...,j],polynomial[...,j]) for j in range(4)]
    assert max(native_components)<1e-8
    delta=R+x[:,None]-y[None];distance=delta.norm(dim=-1)
    G=torch.exp(1j*wave*distance)/(4*math.pi*distance)
    grad=G[...,None]*(1j*wave-1/distance)[...,None]*delta/distance[...,None]
    original=torch.cat((G[...,None],grad),-1)
    actual_error=_relative(actual,original)
    assert bool((original.abs()>0).all())
    physical_components=[_relative(actual[...,j],original[...,j]) for j in range(4)]
    if order>=6:assert max(physical_components)<1e-8
    record=dict(order=order,wave=str(wave),ideal_mp_absolute_errors=ideal_errors,continuous_bound=bound,
        mp_precision_digits=100,actual_fft_vs_mp_polynomial_relative=native_polynomial_error,
        actual_fft_vs_mp_polynomial_per_component=native_components,
        actual_fft_vs_direct_physical_relative=actual_error,
        actual_fft_vs_direct_physical_per_component=physical_components,receipt=op.receipt(),qualification=False)
    (tmp_path/'continuous.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_same_axial_geometry_order7_admits_unchanged_budget_mp_and_four_fields(tmp_path,monkeypatch):
    test_continuous_bound_mp_bivariate_polynomial_actual_fft_four_physical_fields(
        7,100.,[2000.,0.,0.],.5,0.,tmp_path,monkeypatch)


def test_scalar_batch_zero_sides_orders_native_range_empty_and_positive_floor(tmp_path):
    R=torch.tensor([[1.,.2,-.1]]*4,device='cuda',dtype=torch.float64);u=R/R.norm(dim=-1,keepdim=True)
    rt=R.new_tensor([.02,0.,.02,0.]);rs=R.new_tensor([.03,.03,0.,0.]);orders=(0,3,6,12)
    values=directional_multipole_truncation_bounds_many(R,u,rt,rs,5.+.7j,orders)
    assert bool(values['available'].all())
    assert bool((values['green_absolute'][:3]>0).all() & (values['gradient_component_absolute'][:3]>0).all())
    assert bool((values['green_absolute'][3]==0).all() & (values['gradient_component_absolute'][3]==0).all())
    for row in range(4):
        for col,order in enumerate(orders):
            scalar=directional_multipole_truncation_bound(R[row],u[row],rt[row],rs[row],5.+.7j,order)
            for name in ('green_absolute','gradient_component_absolute'):
                torch.testing.assert_close(R.new_tensor(scalar[name]),values[name][row,col],rtol=1e-11,atol=0)
    empty=directional_multipole_truncation_bounds_many(R[:0],u[:0],rt[:0],rs[:0],5.,orders)
    assert all(v.shape==(0,4) for v in empty.values())
    bad=directional_multipole_truncation_bounds_many(R[:1],u[:1],R.new_tensor([1.]),R.new_tensor([1.]),5.,(3,))
    assert not bool(bad['available'].any())
    with pytest.raises(KernelBoundUnavailable):directional_multipole_truncation_bound(R[0],u[0],R.new_tensor(1.),R.new_tensor(1.),5.,3)
    # Nonzero support ratio lost to native division must not become exact zero.
    extremes=[]
    for distance,radius,wave in [(1e100,1e-300,5.),(1.,.01,5e-324),(1e100,.01,1e300)]:
        c=R.new_tensor([[distance,0.,0.]]);d=R.new_tensor([[1.,0.,0.]]);r=R.new_tensor([radius])
        v=directional_multipole_truncation_bounds_many(c,d,r,r,wave,(3,))
        assert not bool(v['available'].any());extremes.append(dict(distance=distance,radius=radius,wave=wave,available=False))
    decayed=directional_multipole_truncation_bounds_many(R[:1],u[:1],rt[:1],rs[:1],1000j,(12,))
    assert bool(decayed['available'].all())
    assert 0<float(decayed['green_absolute'][0,0])<=torch.finfo(torch.float64).tiny*1.0000001
    record=dict(orders=orders,single_and_double_zero_sides=True,scalar_batch_relative_gate=1e-11,
        extreme_native_unavailable=extremes,positive_floor=float(decayed['green_absolute'][0,0]),qualification=False)
    (tmp_path/'native_bounds.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_bound_and_translation_policy_capacity_rejection_before_factories(monkeypatch):
    import witwin.maxwell.mom.fmm_directional_fft as module
    R=torch.tensor([1.,.2,-.1],device='cuda',dtype=torch.float64);u=R/R.norm();r=R.new_tensor(.03)
    bound=directional_multipole_truncation_bound(R,u,r,r,5.,3)
    budget=KernelErrorBudget(bound['green_absolute']*1.01,bound['gradient_component_absolute']*1.01)
    def forbidden(*a,**kw):raise AssertionError('Rejected directional admission reached translation allocation')
    with monkeypatch.context() as patch:
        patch.setattr(module.CartesianDirectionalFFTTranslation,'_prepare',forbidden)
        patch.setattr(module,'directional_helmholtz_taylor',forbidden)
        for kw,exc in [({'kernel_error_budget':replace(budget,green_absolute=bound['green_absolute']/2)},MultipoleTruncationError),
                ({'source_radius':R.new_tensor(2.)},KernelBoundUnavailable),({'source_radius':None},ValueError),
                ({'kernel_error_budget':None},ValueError),({'retained_budget_bytes':1},MemoryError),
                ({'tensor_workspace_budget_bytes':1},MemoryError)]:
            args=dict(test_radius=r,source_radius=r,kernel_error_budget=budget);args.update(kw)
            with pytest.raises(exc):CartesianDirectionalFFTTranslation(R,u,5.,r,3,**args)
        centers=R[None].repeat(33,1);dirs=u[None].repeat(33,1);radii=r.repeat(33);scales=r.repeat(33)
        for last,exc in [(2.,KernelBoundUnavailable),(.15,MultipoleTruncationError)]:
            bad=radii.clone();bad[-1]=last
            with pytest.raises(exc):CartesianDirectionalFFTTranslation.create_many(centers,dirs,5.,scales,3,
                test_radii=radii,source_radii=bad,kernel_error_budget=budget,retained_budget_bytes=16*1024**2)
    with pytest.raises(MemoryError):directional_multipole_truncation_bounds_many(R[None],u[None],r[None],r[None],5.,(3,),
        tensor_workspace_budget_bytes=directional_bound_workspace_bytes(1,1)-1)
    for c,d,k,orders in [(R.float(),u.float(),5.,(3,)),(R,u*2,5.,(3,)),(R,u,5.-1j,(3,)),(R,u,5.,(13,))]:
        with pytest.raises(ValueError):directional_multipole_truncation_bounds_many(c[None],d[None],c.new_tensor([.03]),c.new_tensor([.03]),k,orders)


def test_declared_batch_unique_owned_radius_exact_capacity_budget_and_saved_backward_guards(monkeypatch,tmp_path):
    R=torch.tensor([[1.+.01*i,.2,-.1] for i in range(33)],device='cuda',dtype=torch.float64)
    u=R/R.norm(dim=-1,keepdim=True);r=R.new_full((33,),.03)
    budget=KernelErrorBudget(1.,1.)
    expected=m2l_storage_bytes(3,8,'fft_v1')+24+16
    objects=CartesianDirectionalFFTTranslation.create_many(R,u,5.+.7j,r,3,test_radii=r,source_radii=r,
        kernel_error_budget=budget,retained_budget_bytes=33*expected)
    assert len(objects)==33
    assert sum(op.retained_bytes for op in objects)==33*expected
    tensors=[t for op in objects for t in op.tensors()]
    assert len({t.untyped_storage().data_ptr() for t in tensors})==len(tensors)
    assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
    for op in objects:
        assert op.retained_bytes==op.planned_retained_bytes==expected
        assert op.support_radii.untyped_storage().nbytes()==16
        assert op.receipt()['ideal_truncation_admission']['admitted']
    R.add_(1);u.add_(1);r.add_(1)
    for op in objects:op.assert_current()
    x=torch.randn((objects[0].count,),device='cuda',dtype=torch.complex128,requires_grad=True)
    for index,name in enumerate(('radius','budget','receipt')):
        op=objects[index]
        output=op.apply(x)
        if name=='radius':op.support_radii.add_(0)
        elif name=='budget':op.kernel_error_budget=replace(budget,green_absolute=.9)
        else:op._ideal_bound['green_absolute']=op._ideal_bound['green_absolute']*.9
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,x,torch.ones_like(output))
    record=dict(objects=33,per_object_owned_bytes=expected,total_owned_bytes=33*expected,
        independent_radius_bytes=16,all_unique_storages=True,bound_budget_radius_saved_backward_guard=True,
        actual_support_verified=False,qualification=False)
    (tmp_path/'declared_ownership.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


@pytest.mark.parametrize('lossy',(False,True))
def test_declared_em_original_weak_full_H_and_actual_support(lossy,monkeypatch,tmp_path):
    import test_fmm_directional_rwg as original
    from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L
    native_basis=original._basis;prepared=[];couplings=[]
    def basis(*args,**kwargs):
        result=native_basis(*args,**kwargs);prepared.append(result);return result
    def coupling(displacement,direction,medium,scale,order,**kwargs):
        radius=scale*1.01
        bound=directional_multipole_truncation_bound(displacement,direction,radius,radius,medium.k,order)
        budget=KernelErrorBudget(bound['green_absolute']*1.01,bound['gradient_component_absolute']*1.01)
        op=DirectionalFFTElectromagneticM2L(displacement,direction,medium,scale,order,
            test_radius=radius,source_radius=radius,kernel_error_budget=budget,**kwargs)
        couplings.append(op);return op
    monkeypatch.setattr(original,'_basis',basis)
    monkeypatch.setattr(original,'DirectionalFFTElectromagneticM2L',coupling)
    original.test_directional_em_actual_original_weak_all_columns_adjoint_and_current_gradient(lossy,tmp_path)
    assert len(prepared)==2 and len(couplings)==1
    for index,space in enumerate(prepared):
        assert bool(((space.mesh.vertices-space.mesh.vertices.mean(0)).norm(dim=-1)
                     <=couplings[0].scalar.support_radii[index]).all())
    record=json.loads((tmp_path/'weak.json').read_text())
    assert record['receipt']['scalar_translation']['ideal_truncation_admission']['admitted']
    record['declared_case_actual_full_support_verified']=True
    record['global_operator_budget_qualified']=False
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n')

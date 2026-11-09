"""Complete-support and independent material-weighted directional admission."""
import json
import math
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.quadrature import triangle_rule
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_directional_rwg import (DirectionalRWGProjection,
    DirectionalFFTElectromagneticM2L, directional_rwg_admission_workspace_bytes)

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _pair(lossy=False, order=6):
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01 if lossy else 0.,
        sigma_m=.02 if lossy else 0.),3e8)
    bases=[RWGBasis(sphere_mesh(radius=.02,center=c,subdivisions=0,device='cuda',dtype=torch.float64))
        for c in ((-.5,.1,0.),(.5,-.1,.05))]
    centers=[b.mesh.vertices.mean(0)+b.mesh.vertices.new_tensor([.0007,-.0003,.0002]) for b in bases]
    u=centers[0]-centers[1];u=u/u.norm()+u.new_tensor([.003,-.002,.001]);u=u/u.norm()
    selected=([17,2,11,29,7],[8,1,22])
    projections=[DirectionalRWGProjection(b,torch.tensor(e,device='cuda'),c,u,medium.k,order,
        quadrature_order=12) for b,e,c in zip(bases,selected,centers)]
    return *projections,medium


def _construct(test,source,medium,**kwargs):
    args=dict(kernel_error_budget=KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,
        balancing_impedance=abs(medium.eta));args.update(kwargs)
    return DirectionalFFTElectromagneticM2L.from_projections(test,source,medium,
        test.center.new_tensor(.02),**args)


def _action(op,test,source,value,adjoint=False):
    n=len(test.edges) if adjoint else len(source.edges)
    currents=torch.stack((value[...,:n],value[...,n:]),dim=-2)
    if adjoint:
        output=source.source_adjoint(op.adjoint(test.receiver_adjoint(currents)))
    else:
        output=test.receiver_evaluate(op.apply(source.source_moments(currents)))
    return torch.cat((output[...,0,:],output[...,1,:]),dim=-1)


def _independent_integral_majorants(projection):
    # Independent full plus/minus face integration and Cauchy-Schwarz L2 bounds,
    # without rwg_integral_logs, statistics or production block_error_log.
    b,g=projection.basis,projection.geometry
    values=[]
    for edge in projection.edges.tolist():
        U=projection.center.new_zeros(());Q=U.clone();L1=U.clone()
        for face in (int(b.plus_faces[edge]),int(b.minus_faces[edge])):
            triangle=g.triangles[face:face+1]
            points,weights=triangle_rule(triangle,16);points,weights=points[0],weights[0]
            position=int((b.face_edges[face]==edge).nonzero()[0]);slope=g.slopes[face,position]
            f=slope*(points-triangle[0,position])
            U+=torch.sqrt(weights.sum()*(f.square().sum(-1)*weights).sum())
            L1+=(f.norm(dim=-1)*weights).sum();Q+=2*slope.abs()*weights.sum()
        assert bool(L1<=U*(1+1e-12));values.append(torch.stack((U,Q)))
    return torch.stack(values)


@pytest.mark.parametrize('lossy',(False,True))
def test_prescribed_complete_support_weighted_bound_original_weak_H_and_current(lossy,tmp_path):
    test,source,medium=_pair(lossy);op=_construct(test,source,medium)
    receipt=op.receipt();bound=op.ideal_rwg_truncation_bound();kernel=op.scalar.receipt()['ideal_truncation_admission']
    for projection,radius in zip((test,source),op.scalar.support_radii):
        b=projection.basis;triangles=b.mesh.vertices[b.mesh.faces]
        corners=torch.cat((triangles[b.plus_faces[projection.edges]],triangles[b.minus_faces[projection.edges]]),dim=1)
        assert bool(((corners-projection.center).norm(dim=-1)<=radius).all())
    u,q=_independent_integral_majorants(test).unbind(-1)
    v,r=_independent_integral_majorants(source).unbind(-1)
    D=kernel['green_absolute']*(u[:,None]*v[None]+q[:,None]*r[None]/abs(medium.k)**2)
    K=math.sqrt(3)*kernel['gradient_component_absolute']*u[:,None]*v[None]
    eta=abs(medium.eta);e=1j*medium.omega*medium.epsilon;m=-1j*medium.omega*medium.mu
    independent=torch.sqrt((2*eta**2*K.square()+(eta**4*abs(e)**2+abs(m)**2)*D.square()).sum())
    torch.testing.assert_close(independent,independent.new_tensor(bound['absolute_bound']),rtol=1e-11,atol=0)
    rows,cols=len(test.edges),len(source.edges)
    actual=_action(op,test,source,torch.eye(2*cols,device='cuda',dtype=torch.complex128)).mT
    reference,_=scalar_regional_block(test.basis,source.basis,test.edges,source.edges,(medium,),IntegrationConfig(order=16,singular_order=3))
    left=actual.real.new_tensor([eta]*rows+[1.]*rows);right=actual.real.new_tensor([1.]*cols+[eta]*cols)
    absolute=(left[:,None]*(actual-reference)*right).norm()
    relative=float(absolute/(left[:,None]*reference*right).norm())
    diagnostic=dict(lossy=lossy,phase='original_weak_computed',receipt=receipt,
        absolute_original_weak_error=float(absolute),balanced_original_weak_relative=relative,
        physical_relative_gate=1e-8,ideal_absolute_budget=1e-4,production_qualification=False)
    (tmp_path/'weak_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    torch.save(dict(actual=actual.detach(),reference=reference.detach()),tmp_path/'weak_diagnostic.pt')
    print(diagnostic,flush=True)
    assert relative<1e-8 and float(absolute)<=bound['absolute_bound']
    generator=torch.Generator(device='cuda').manual_seed(1739)
    current=torch.randn((3,5,2*cols),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    dual=torch.randn((*current.shape[:-1],2*rows),device='cuda',dtype=torch.complex128,generator=generator)
    applied=_action(op,test,source,current);adjoint=_action(op,test,source,dual,True)
    original_adjoint=dual@reference.conj();physical_h=float((adjoint-original_adjoint).norm()/original_adjoint.norm())
    assert physical_h<1e-8
    derivative,=torch.autograd.grad(applied,current,dual,retain_graph=True)
    native_h=float((adjoint-dual@actual.conj()).norm()/(dual@actual.conj()).norm())
    gradient=float((derivative-adjoint).norm()/adjoint.norm())
    lhs=(applied.conj()*dual).sum();rhs=(current.conj()*adjoint).sum();dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    loss=applied.abs().square().sum();first,=torch.autograd.grad(loss,current,create_graph=True)
    second,=torch.autograd.grad(first,current,torch.ones_like(first))
    expected=2*_action(op,test,source,_action(op,test,source,torch.ones_like(current)),True)
    higher=float((second-expected).norm()/expected.norm())
    assert max(native_h,gradient,dot,higher)<1e-11
    assert op.scalar.retained_bytes==sum(t.numel()*t.element_size() for t in op.tensors())
    assert not any(t.untyped_storage().data_ptr() in {x.untyped_storage().data_ptr() for p in (test,source) for x in p.tensors()} for t in op.tensors())
    assert receipt['caller_prescribed_projection_storage_excluded'] and bound['complete_selected_plus_minus_support_verified']
    assert not bound['quadrature_included'] and not bound['implemented_operator_error_certified']
    record=dict(lossy=lossy,test_edges=test.edges.tolist(),source_edges=source.edges.tolist(),
        independent_l2_integral_and_edge_majorant=True,complete_plus_minus_support_verified=True,
        absolute_original_weak_error=float(absolute),balanced_original_weak_relative=relative,
        original_physical_adjoint_relative=physical_h,native_adjoint_relative=native_h,
        current_gradient_relative=gradient,dot=dot,higher_current_relative=higher,
        receipt=receipt,production_qualification=False)
    (tmp_path/'weighted.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_weighted_kernel_capacity_and_policy_reject_before_translation(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as module
    import witwin.maxwell.mom.rwg_interpolation_error as error
    test,source,medium=_pair(True)
    def forbidden(*a,**kw):raise AssertionError('Rejected RWG admission reached coefficient allocation')
    native_translation=module.CartesianDirectionalFFTTranslation
    monkeypatch.setattr(module,'CartesianDirectionalFFTTranslation',forbidden)
    for kwargs,exc,match in [({'ideal_rwg_error_budget':1e-30},ValueError,'absolute budget'),
            ({'kernel_error_budget':KernelErrorBudget(1e-30,1e-30)},ValueError,'kernel budget'),
            ({'ideal_rwg_error_budget':0.},ValueError,'positive'),({'balancing_impedance':0.},ValueError,'positive'),
            ({'test_radius':test.center.new_tensor(.001)},ValueError,'derived'),
            ({'tensor_workspace_budget_bytes':1},MemoryError,'workspace')]:
        with pytest.raises(exc,match=match):_construct(test,source,medium,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(error,'rwg_integral_logs',forbidden)
        patch.setattr(module,'_selected_edges',forbidden)
        with pytest.raises(MemoryError,match='workspace'):_construct(test,source,medium,
            tensor_workspace_budget_bytes=directional_rwg_admission_workspace_bytes(test,source)-1)
    with monkeypatch.context() as patch:
        patch.setattr(module,'CartesianDirectionalFFTTranslation',native_translation)
        patch.setattr(native_translation,'_prepare',forbidden)
        with pytest.raises(MemoryError,match='retained'):_construct(test,source,medium,retained_budget_bytes=1)
    wrong=[(test,object(),TypeError),(test,test,ValueError)]
    # Coincident centers have no separated directional contour.
    for a,b,exc in wrong:
        with pytest.raises(exc):_construct(a,b,medium)
    old=source.wave_number;source.wave_number=old+1
    with pytest.raises(ValueError,match='changed'):_construct(test,source,medium)
    source.wave_number=old
    with pytest.raises(ValueError):_construct(test,source,compile_medium(mw.Material(),3e8))
    # Distinct currently valid but wrong order/carrier maps must reject.
    for order,direction in [(3,source.direction),(6,-source.direction)]:
        other=DirectionalRWGProjection(source.basis,source.edges,source.center,direction,medium.k,order,quadrature_order=12)
        with pytest.raises(ValueError,match='order and carrier'):_construct(test,other,medium)
    with pytest.raises(ValueError,match='nonempty'):
        DirectionalRWGProjection(source.basis,source.edges[:0],source.center,source.direction,medium.k,6,quadrature_order=12)
    record=dict(ideal_budget_and_kernel_reject_before_coefficients=True,
        full_basis_workspace_preflight=True,empty_selection_rejected=True,
        strict_prescribed_maps=True,production_qualification=False)
    (tmp_path/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')


def test_prescribed_projection_and_semantic_guards_on_action_receipt_saved_backward(tmp_path):
    test,source,medium=_pair(True);op=_construct(test,source,medium)
    value=torch.ones((2,op.count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
    edits=[(op,'ideal_rwg_error_budget',op.ideal_rwg_error_budget*.9),
        (op,'balancing_impedance',op.balancing_impedance*.9),
        (op,'_rwg_context',(source,test)),
        (op.scalar,'kernel_error_budget',replace(op.scalar.kernel_error_budget,green_absolute=9e-7)),
        (op,'_rwg_bound',{**op._rwg_bound,'absolute_bound':op._rwg_bound['absolute_bound']*.9})]
    for owner,name,new in edits:
        old=getattr(owner,name);output=op.apply(value);setattr(owner,name,new)
        with pytest.raises(ValueError,match='changed'):op.apply(value)
        with pytest.raises(ValueError,match='changed'):op.receipt()
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,value,torch.ones_like(output))
        setattr(owner,name,old);op.assert_current()
    for index in (0,1):
        fresh=_construct(test,source,medium);output=fresh.apply(value)
        (test,source)[index].receiver_vector.add_(0)
        with pytest.raises(ValueError,match='changed'):fresh.receipt()
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,value,torch.ones_like(output))
        # A new pair is required after mutation; do not restore stale versions.
        test,source,medium=_pair(True)
    op=_construct(test,source,medium);output=op.apply(value)
    source.geometry.triangles.add_(0)
    with pytest.raises(ValueError,match='changed'):op.apply(value)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,value,torch.ones_like(output))
    (tmp_path/'guards.json').write_text(json.dumps(dict(external_projection_and_geometry_guards=True,
        impedance_two_budgets_receipt_identity_saved_backward_guards=True,
        production_qualification=False),indent=2)+'\n')

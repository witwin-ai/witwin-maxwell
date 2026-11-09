"""Complete original near/FAR PMCHWT and independent overlapping-error norms."""
from dataclasses import replace
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks,scalar_regional_block
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalNestedFarActions
from witwin.maxwell.mom.fmm_regional import DirectionalRegionalOperator
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.rwg_interpolation_error import directional_system_ideal_bound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _basis(x):
    vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        device='cuda',dtype=torch.float64)*.01
    vertices[:,0]+=x
    faces = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
    return RWGBasis(SurfaceMesh(vertices,faces))


def _owner(basis,tree,direction,medium,geometry=None,layout=None):
    return NestedDirectionalRWGProjection(basis,tree,direction,medium.k,8,
        quadrature_order=12,geometry=geometry,layout=layout,face_batch=4,rhs_batch=4)


def _coupling(test,a,source,b,medium,eta):
    return DirectionalFFTElectromagneticM2L.from_nested_nodes(test,a,source,b,medium,
        test.centers.new_tensor(.01),kernel_error_budget=KernelErrorBudget(1e-6,1e-4),
        ideal_rwg_error_budget=1e-4,balancing_impedance=eta)


@pytest.fixture(scope='module')
def case():
    exterior = compile_medium(mw.Material(),3e8)
    materials = [mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01,sigma_m=.02),
                 mw.Material(eps_r=3.,mu_r=1.5,sigma_e=.015,sigma_m=.005)]
    system = SurfaceSystem.create([(i,str(i),_basis(x),compile_medium(m,3e8))
        for i,(x,m) in enumerate(zip((-.5,.5),materials,strict=True))])
    config = IntegrationConfig(order=4,singular_order=3,pair_batch=4,far_pair_batch=8)
    provider = ScalarSystemBlocks(system,exterior,config)
    lower,upper = provider.support_boxes();tree = build_support_octree(lower,upper,leaf_size=6)
    minus = lower.new_tensor([-1.,0.,0.]);left = _owner(provider,tree,minus,exterior)
    right = _owner(provider,tree,-minus,exterior,layout=left.layout)
    assert tree.node_count==3 and len(left.children[0])==2
    a,b = left.children[0]
    eta = abs(exterior.eta)
    couplings = (_coupling(left,a,left,b,exterior,eta),_coupling(right,b,right,a,exterior,eta))
    far = DirectionalNestedFarActions(left,left,exterior,couplings,ideal_rwg_error_budget=1e-4)
    regional = DirectionalRegionalOperator(far,((a,a),(b,b)),integration=config)
    interiors = []
    for i,body in enumerate(system.bodies):
        begin,end = body.edge_offset,body.edge_offset+body.basis.n_edges
        local_tree = build_support_octree(lower[begin:end],upper[begin:end],leaf_size=6)
        owner = _owner(body.basis,local_tree,minus,body.interior,geometry=provider.geometry[i])
        empty_far = DirectionalNestedFarActions(owner,owner,body.interior,(),
            ideal_rwg_error_budget=1e-4,balancing_impedance=eta)
        interiors.append(DirectionalRegionalOperator(empty_far,((0,0),),integration=config))
    actions = DirectionalSystemActions(provider,regional,interiors,ideal_rwg_error_budget=1e-4)
    return actions


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def test_complete_multibody_magnetic_lossy_original_PMCHWT_H_current_and_higher(case,tmp_path):
    actions = case;provider = actions.provider;n = provider.system.total_edges
    reference = torch.zeros(actions.shape,device='cuda',dtype=torch.complex128)
    for i,target in enumerate(provider.system.bodies):
        for j,source in enumerate(provider.system.bodies):
            rows = torch.arange(target.basis.n_edges,device='cuda');cols = torch.arange(source.basis.n_edges,device='cuda')
            media = (provider.exterior,target.interior) if i==j else (provider.exterior,)
            config = provider.config if i==j else IntegrationConfig(order=16,singular_order=3)
            block,_ = scalar_regional_block(target.basis,source.basis,rows,cols,media,config,
                geometry=(provider.geometry[i],provider.geometry[j]))
            rr,cc = rows+target.edge_offset,cols+source.edge_offset
            rr,cc = torch.cat((rr,rr+n)),torch.cat((cc,cc+n))
            reference[rr[:,None],cc] = block
    actual = actions.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)).mT
    eta = actions.balancing_impedance
    left = actual.real.new_tensor([eta]*n+[1.]*n);right = actual.real.new_tensor([1.]*n+[eta]*n)
    weak = float((left[:,None]*(actual-reference)*right).norm()/(left[:,None]*reference*right).norm())
    diagnostic = dict(phase='original_full_weak_computed',balanced_weak_relative=weak,
        receiver_source_carriers_distinct=True,original_cross_quadrature=16,original_near_quadrature=4,
        body_media=[dict(epsilon_r=str(b.interior.epsilon_r),mu_r=str(b.interior.mu_r)) for b in provider.system.bodies],
        receipt=actions.receipt(),production_qualification=False)
    (tmp_path/'weak_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    torch.save(dict(actual=actual,reference=reference),tmp_path/'weak_diagnostic.pt');print(diagnostic,flush=True)
    assert weak<1e-8
    generator = torch.Generator(device='cuda').manual_seed(2513)
    current = torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        output = actions.apply(current)
    assert not saved and not current.is_contiguous()
    H = actions.adjoint(dual)
    derivative, = torch.autograd.grad(output,current,dual,retain_graph=True)
    differentiable_dual = dual.detach().requires_grad_();h_output = actions.adjoint(differentiable_dual)
    h_derivative, = torch.autograd.grad(h_output,differentiable_dual,current.detach())
    first, = torch.autograd.grad(output.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn_like(current);second, = torch.autograd.grad(first,current,tangent)
    lhs,rhs = (output.conj()*dual).sum(),(current.conj()*H).sum()
    errors = dict(original_H=_relative(H,dual@reference.conj()),native_H=_relative(H,dual@actual.conj()),
        native_forward=_relative(output,current@actual.mT),current=_relative(derivative,H),
        adjoint_current=_relative(h_derivative,actions.apply(current.detach())),
        higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    inventories = []
    for owner in (actions.exterior,*actions.interiors,actions):
        tensors = owner.tensors();owned = sum(t.untyped_storage().nbytes() for t in tensors)
        assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
        assert owned==owner.retained_bytes==owner.planned_retained_bytes
        inventories.append(owned)
    assert actions.retained_bytes==16*n
    assert actions.exterior.near is not None and len(actions.exterior.far.couplings)==2
    assert all(not r.far.couplings and r.near is not None for r in actions.interiors)
    diagnostic.update(phase='all_current_checks_done',errors=errors,owned_component_bytes=inventories,
        physical_rhs=15,saved_internal_tensors=len(saved),
        interior_far_physical_actions_covered=False,interior_original_near_actions_covered=True)
    (tmp_path/'system.json').write_text(json.dumps(diagnostic,indent=2)+'\n');print(diagnostic,flush=True)
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11
    assert not actions.ideal_rwg_truncation_bound()['implemented_operator_error_certified']


def test_independent_common_impedance_overlap_triangle_and_disjoint_RSS_math(tmp_path):
    # Independent native error matrices give a case where RSS across overlapping
    # media underbounds the true error. These are norm controls, not physical
    # positive-interior-FAR qualification for the all-near body fixtures above.
    generator = torch.Generator(device='cuda').manual_seed(2521)
    n,eta = 6,73.
    interiors = torch.zeros((2,2*n,2*n),device='cuda',dtype=torch.complex128)
    body_ids = (torch.tensor([0,1,6,7],device='cuda'),torch.tensor([2,3,4,5,8,9,10,11],device='cuda'))
    for i,ids in enumerate(body_ids):
        interiors[i,ids[:,None],ids] = torch.randn((len(ids),len(ids)),device='cuda',dtype=torch.complex128,generator=generator)
    exterior = .7*interiors.sum(0)
    left = exterior.real.new_tensor([eta]*n+[1.]*n);right = exterior.real.new_tensor([1.]*n+[eta]*n)
    errors = (exterior,*interiors)
    receipts = []
    with mp.workdps(100):
        exact = []
        for error in errors:
            values = error.tolist()
            value = mp.sqrt(sum(abs(mp.mpc(values[i][j]))**2
                *mp.mpf(float(left[i]))**2*mp.mpf(float(right[j]))**2
                for i in range(2*n) for j in range(2*n)))
            exact.append(value)
            receipts.append(dict(absolute_bound=float(value),log_absolute_bound=float(mp.log(value)),balancing_impedance=eta))
        expected = float(exact[0]+mp.sqrt(sum(v*v for v in exact[1:])))
    bound = directional_system_ideal_bound(receipts[0],receipts[1:],device=exterior.device,balancing_impedance=eta)
    actual = float((left[:,None]*(exterior+interiors.sum(0))*right).norm())
    naive = math.sqrt(sum(b['absolute_bound']**2 for b in receipts))
    assert bound['absolute_bound']==pytest.approx(expected,rel=1e-11,abs=0)
    assert actual==pytest.approx(expected,rel=1e-11,abs=0)
    assert naive<.9*actual
    zero = dict(absolute_bound=0.,log_absolute_bound=None,balancing_impedance=eta)
    assert directional_system_ideal_bound(zero,[zero,zero],device='cuda',balancing_impedance=eta)['absolute_bound']==0
    assert directional_system_ideal_bound(receipts[0],[],device='cuda',balancing_impedance=eta)['absolute_bound']==pytest.approx(receipts[0]['absolute_bound'],rel=1e-11,abs=0)
    for bad in ({**zero,'absolute_bound':1.},{**zero,'log_absolute_bound':float('nan')},
                {**receipts[0],'balancing_impedance':eta+1}):
        with pytest.raises(ValueError):directional_system_ideal_bound(bad,[zero],device='cuda',balancing_impedance=eta)
    (tmp_path/'overlap_math.json').write_text(json.dumps(dict(independent_MP100_expected=expected,
        actual_balanced_coaligned_error=actual,native_triangle_RSS=bound['absolute_bound'],
        wrong_RSS_across_media=naive,absolute_comparison_tolerance=0,relative_gate=1e-11,
        numerical_norm_control_only=True,physical_interior_far_qualification=False),indent=2)+'\n')


def test_complete_all_far_cross_domain_and_empty_rhs_without_packed_allocation(case,monkeypatch,tmp_path):
    actions = case;provider = actions.provider;medium = provider.exterior;eta = actions.balancing_impedance
    owners = []
    for i,body in enumerate(provider.system.bodies):
        b = body.basis;triangles = b.mesh.triangles
        supports = torch.cat((triangles[b.plus_faces],triangles[b.minus_faces]),1)
        tree = build_support_octree(supports.amin(1),supports.amax(1),leaf_size=b.n_edges)
        owners.append(_owner(b,tree,supports.new_tensor([-1.,0.,0.]),medium,geometry=provider.geometry[i]))
    test,source = owners;coupling = _coupling(test,0,source,0,medium,eta)
    far = DirectionalNestedFarActions(test,source,medium,(coupling,),ideal_rwg_error_budget=1e-4)
    def forbidden(*a,**kw):raise AssertionError('All FAR cross-domain region allocated packed near storage')
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_regional.PackedScalarNearChannels.from_layout',forbidden)
        region = DirectionalRegionalOperator(far,(),integration=provider.config)
    assert region.near is None and region.retained_bytes==0
    current = torch.ones((3,5,region.shape[1]),device='cuda',dtype=torch.complex128).transpose(0,1).requires_grad_()
    assert _relative(region.apply(current),far.apply(current))<1e-11
    for owner in (region,actions.exterior,*actions.interiors,actions):
        empty = torch.empty((0,3,owner.shape[1]),device='cuda',dtype=torch.complex128,requires_grad=True)
        output = owner.apply(empty);assert output.shape==(0,3,owner.shape[0])
        derivative, = torch.autograd.grad(output,empty,torch.ones_like(output));assert derivative.shape==empty.shape
    (tmp_path/'all_far_empty.json').write_text(json.dumps(dict(cross_domain_complete_all_far=True,
        near_storage_allocation_not_reached=True,new_near_owned_bytes=0,
        region_and_system_empty_current_backward=True,production_qualification=False),indent=2)+'\n')


def test_before_integration_coverage_capacity_and_complete_saved_backward_guards(case,monkeypatch,tmp_path):
    actions = case;region = actions.exterior;provider = actions.provider
    def forbidden(*a,**kw):raise AssertionError('Rejected partition/capacity reached original integration or packed allocation')
    near = region.near_nodes
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_regional.PackedScalarNearChannels.from_layout',forbidden)
        for nodes,match in ((near[:-1],'every physical'),((*near,near[0]),'overlap or duplicate'),
                            (((0,near[0][1]),),'terminal'),(((-1,near[0][1]),),'outside')):
            with pytest.raises(ValueError,match=match):DirectionalRegionalOperator(region.far,nodes,integration=provider.config)
        for change in ({'near_retained_budget_bytes':region.retained_bytes-1},
                       {'tensor_workspace_budget_bytes':region.planned_tensor_workspace_bytes-1}):
            with pytest.raises(MemoryError):DirectionalRegionalOperator(region.far,near,integration=provider.config,**change)
        with pytest.raises(ValueError,match='integration identities'):
            DirectionalRegionalOperator(region.far,near,integration=replace(provider.config))
    with monkeypatch.context() as patch:
        patch.setattr(torch,'arange',forbidden)
        for change,exc,match in (({'retained_budget_bytes':actions.retained_bytes-1},MemoryError,'index budget'),
                ({'tensor_workspace_budget_bytes':actions.planned_tensor_workspace_bytes-1},MemoryError,'workspace'),
                ({'ideal_rwg_error_budget':actions.ideal_rwg_truncation_bound()['absolute_bound']/2},ValueError,'global absolute'),
                ({'balancing_impedance':actions.balancing_impedance*1.01},ValueError,'common global')):
            args=dict(ideal_rwg_error_budget=1e-4);args.update(change)
            with pytest.raises(exc,match=match):DirectionalSystemActions(provider,region,actions.interiors,**args)
        with pytest.raises(ValueError,match='actual body'):
            DirectionalSystemActions(provider,region,actions.interiors[::-1],ideal_rwg_error_budget=1e-4)
    current = torch.ones((2,actions.shape[0]),device='cuda',dtype=torch.complex128,requires_grad=True)
    variants = ((actions,'indices',actions.indices[::-1]),(actions,'interiors',actions.interiors[::-1]),
        (actions,'ideal_rwg_error_budget',.9),(actions,'_ideal_bound',{**actions._ideal_bound,'absolute_bound':0.}),
        (region,'near_nodes',near[::-1]),(region,'integration',replace(provider.config)),
        (region,'planned_retained_bytes',region.planned_retained_bytes+1))
    for owner,name,new in variants:
        old = getattr(owner,name);output = actions.apply(current);setattr(owner,name,new)
        with pytest.raises(ValueError,match='changed'):actions.receipt()
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,torch.ones_like(output))
        setattr(owner,name,old);actions.assert_current()
    for method in (actions.apply,actions.adjoint,region.apply,region.adjoint):
        with pytest.raises(ValueError,match='finite'):method(current.detach()*float('nan'))
        with pytest.raises(ValueError,match='complex128'):method(current.real)
    output = actions.apply(current);actions.indices[0].add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,torch.ones_like(output))
    # Separately validate the caller's original near numeric storage after the
    # embedding guard, using the regional custom backward saved before mutation.
    local = torch.ones((2,region.shape[0]),device='cuda',dtype=torch.complex128,requires_grad=True)
    output = region.apply(local);region.near.groups[0][2].add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,local,torch.ones_like(output))
    (tmp_path/'guards.json').write_text(json.dumps(dict(coverage_before_original_integration=True,
        exact_near_and_embedding_budgets_preflighted=True,common_impedance_and_body_identity_required=True,
        semantic_partition_integration_budget_and_external_numeric_saved_backward=True,
        owned_embedding_index_saved_backward=True,production_qualification=False),indent=2)+'\n')

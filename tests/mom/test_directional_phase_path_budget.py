"""Independent ideal phase products and actual prescribed node weak/current maps."""
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_directional_moments import (NestedDirectionalRWGProjection,
    directional_phase_path_workspace_bytes)
from witwin.maxwell.mom.fmm_directional_rwg import (DirectionalFFTElectromagneticM2L,
    directional_nested_admission_workspace_bytes)
from test_fmm_directional_moments import _SelectedTreeEM

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _tree(basis):
    triangles=basis.mesh.triangles
    corners=torch.cat((triangles[basis.plus_faces],triangles[basis.minus_faces]),1)
    return build_support_octree(corners.amin(1),corners.amax(1),leaf_size=2)


def _owner(basis,tree,direction,wave,root_order=6,L=2,step=1e-4,anchor=1):
    orders=tuple(root_order+L*int(d) for d in tree.depth.tolist());phases=(0,)+(L,)*(tree.node_count-1)
    dirs=direction[None]+(tree.depth-anchor)[:,None]*step*direction.new_tensor([.2,-.4,.7])
    dirs=dirs/dirs.norm(dim=-1,keepdim=True);dirs[tree.depth==anchor]=direction
    parents={c:p for p,row in enumerate(tree.children.tolist()) for c in row if c>=0}
    exact=next((c for c,p in parents.items() if not bool((tree.children[c]>=0).any()) and int(tree.depth[c])>anchor),None)
    if exact is not None:dirs[exact]=dirs[parents[exact]]
    return NestedDirectionalRWGProjection(basis,tree,dirs[0],wave,orders[0],quadrature_order=12,
        face_batch=4,node_directions=dirs,node_orders=orders,phase_orders=phases,
        phase_storage='compact_phase_v1',phase_relative_budget=1e-8)


def _pair(lossy):
    bases=[RWGBasis(sphere_mesh(radius=.02,center=c,subdivisions=0,device='cuda',dtype=torch.float64))
        for c in ((-.5,.1,0.),(.5,-.1,.05))]
    trees=[_tree(b) for b in bases]
    nodes=[next(i for i,row in enumerate(t.children.tolist()) if i>0 and any(c>=0 for c in row)) for t in trees]
    R=(trees[0].lower[nodes[0]]+trees[0].upper[nodes[0]]-trees[1].lower[nodes[1]]-trees[1].upper[nodes[1]])*.5
    direction=R/R.norm()
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01 if lossy else 0.,sigma_m=.02 if lossy else 0.),3e8)
    owners=[_owner(b,t,direction,medium.k) for b,t in zip(bases,trees)]
    return *owners,*nodes,medium


def _coupling(test,source,a,b,medium,**kwargs):
    args=dict(kernel_error_budget=KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,
        balancing_impedance=abs(medium.eta));args.update(kwargs)
    return DirectionalFFTElectromagneticM2L.from_nested_nodes(test,a,source,b,medium,
        test.centers.new_tensor(.02),**args)


def _mp_descendant_phase_products(owner,node):
    # Independent real support vertices, scalar Taylor exponential products and
    # explicit leaf-to-node routes, not production transfer tables or path sums.
    parents={c:p for p,row in enumerate(owner.children) for c in row}
    maximum=0.;nonexact=0
    with mp.workdps(100):
        for leaf in owner.leaves:
            path=[];current=leaf
            while current!=node and current in parents:
                path.append(current);current=parents[current]
            if current!=node:continue
            p=owner.leaves[leaf][0][1];edge=int(p.edges[0]);face=int(p.basis.plus_faces[edge])
            point=tuple(mp.mpf(v) for v in p.geometry.triangles[face,0].tolist())
            for sign in (-1,1):
                product=mp.mpc(1)
                for child in path:
                    parent=parents[child]
                    child_u=owner.node_directions[child].tolist();parent_u=owner.node_directions[parent].tolist()
                    center=owner.centers[child].tolist();k=mp.mpc(owner.wave_number)
                    z=sign*1j*k*sum((mp.mpf(v)-mp.mpf(u))*(x-mp.mpf(c))
                        for u,v,x,c in zip(child_u,parent_u,point,center))
                    L=owner.phase_orders[child]
                    approx=sum(z**n/mp.factorial(n) for n in range(L+1))
                    product*=approx/mp.exp(z)
                    nonexact+=int(z!=0)
                maximum=max(maximum,float(abs(product-1)))
    return maximum,nonexact


def _mp_path_bound_products(owner,node):
    parents={c:p for p,row in enumerate(owner.children) for c in row}
    values=[]
    with mp.workdps(100):
        for leaf in owner.leaves:
            product=mp.mpf(1);current=leaf
            while current!=node and current in parents:
                shift=owner.transfers[current]
                delta=mp.mpf(0) if shift.exact_phase else mp.exp(mp.mpf(float(shift.log_phase_bound)))
                product*=1+delta;current=parents[current]
            if current==node:values.append(float(product-1))
    return max(values),values


@pytest.mark.parametrize('lossy',(False,True))
def test_actual_nonroot_compact_phase_paths_combined_bound_weak_H_current_higher(lossy,tmp_path):
    test,source,a,b,medium=_pair(lossy);coupling=_coupling(test,source,a,b,medium)
    path_receipts=[p.ideal_phase_path_bound(node) for p,node in ((test,a),(source,b))]
    sampled=[]
    for owner,node,path in zip((test,source),(a,b),path_receipts):
        error,nonexact=_mp_descendant_phase_products(owner,node)
        assert nonexact>0 and 0<path['relative_bound']
        assert error<=path['relative_bound'];sampled.append(error)
        assert any(t.exact_phase for t in owner.transfers.values())
        for check_node in (0,node):
            maximum,branches=_mp_path_bound_products(owner,check_node)
            assert maximum==pytest.approx(owner.ideal_phase_path_bound(check_node)['relative_bound'],rel=1e-11,abs=0)
            if check_node==0:assert len(set(branches))>1
    bound=coupling.ideal_rwg_truncation_bound();scalar=coupling.scalar.receipt()['ideal_truncation_admission']
    with mp.workdps(100):
        D=mp.sqrt(sum(mp.mpf(v)**2 for v in coupling.scalar.displacement.tolist()))
        r=sum(mp.mpf(v) for v in coupling.scalar.support_radii.tolist());d=D-r
        delta=mp.expm1(sum(mp.mpf(p['max_log_one_plus_relative_bound']) for p in path_receipts))
        G=mp.exp(-mp.mpf(medium.k.imag)*d)/(4*mp.pi*d)
        grad=G*(abs(mp.mpc(medium.k))+1/d)
        for name,majorant in [('green_absolute',G),('gradient_component_absolute',grad)]:
            E=mp.mpf(scalar[name]);expected=float(E+delta*(majorant+E))
            assert bound['effective_'+name]==pytest.approx(expected,rel=1e-11,abs=0)
            assert bound['effective_'+name]>=scalar[name]
    op=_SelectedTreeEM(test,source,a,b,coupling)
    actual=op.apply(torch.eye(2*len(op.cols),device='cuda',dtype=torch.complex128)).mT
    reference,_=scalar_regional_block(test.basis,source.basis,op.rows,op.cols,(medium,),IntegrationConfig(order=16,singular_order=3))
    nr,nc=len(op.rows),len(op.cols);eta=abs(medium.eta)
    left=actual.real.new_tensor([eta]*nr+[1.]*nr);right=actual.real.new_tensor([1.]*nc+[eta]*nc)
    absolute=(left[:,None]*(actual-reference)*right).norm();weak=float(absolute/(left[:,None]*reference*right).norm())
    initial=dict(lossy=lossy,phase='weak_computed',balanced_original_weak_relative=weak,
        absolute_original_weak_error=float(absolute),path_receipts=path_receipts,
        independent_mp_phase_relative=sampled,coupling_receipt=coupling.receipt(),production_qualification=False)
    (tmp_path/'weak_diagnostic.json').write_text(json.dumps(initial,indent=2)+'\n')
    torch.save(dict(actual=actual.detach(),reference=reference.detach()),tmp_path/'weak_diagnostic.pt')
    print(initial,flush=True)
    assert weak<1e-8 and float(absolute)<=bound['absolute_bound']
    g=torch.Generator(device='cuda').manual_seed(2313)
    current=torch.randn((3,5,2*nc),device='cuda',dtype=torch.complex128,generator=g).transpose(0,1).requires_grad_()
    dual=torch.randn((*current.shape[:-1],2*nr),device='cuda',dtype=torch.complex128,generator=g)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):out=op.apply(current)
    assert not saved
    H=op.adjoint(dual);original_H=dual@reference.conj()
    derivative,=torch.autograd.grad(out,current,dual,retain_graph=True)
    loss=out.abs().square().sum();first,=torch.autograd.grad(loss,current,create_graph=True)
    second,=torch.autograd.grad(first,current,torch.ones_like(first));expected=2*op.adjoint(op.apply(torch.ones_like(current)))
    relative=lambda x,y:float(((x-y).norm()/y.norm()).detach())
    lhs=(out.conj()*dual).sum();rhs=(current.conj()*H).sum()
    errors=dict(original_H=relative(H,original_H),actual_H=relative(H,dual@actual.conj()),
        forward=relative(out,current@actual.mT),current=relative(derivative,H),higher=relative(second,expected),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    initial.update(phase='current_checks_done',errors=errors,owned_scalar_bytes=coupling.scalar.retained_bytes)
    (tmp_path/'nested.json').write_text(json.dumps(initial,indent=2)+'\n');print(initial,flush=True)
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11
    assert coupling.scalar.retained_bytes==sum(t.numel()*t.element_size() for t in coupling.tensors())
    assert bound['phase_path_included'] and not bound['quadrature_included'] and not bound['floating_point_certified']


def _small_pair(wave=None):
    medium=compile_medium(mw.Material(),3e8)
    wave=medium.k if wave is None else wave
    bases=[RWGBasis(sphere_mesh(radius=.04,center=(c,0.,0.),subdivisions=0,device='cuda',dtype=torch.float64)) for c in (-2.,2.)]
    direction=bases[0].mesh.vertices.new_tensor([-1.,0.,0.])
    owners=[_owner(b,_tree(b),direction,wave,root_order=1,L=1,step=1e-5,anchor=0) for b in bases]
    return *owners,medium


def test_exact_edges_fixed_paths_positive_floor_and_workspace_before_accumulation(monkeypatch,tmp_path):
    owner,other,_=_small_pair(1e-180)
    before=owner.retained_bytes
    bounds=[owner.ideal_phase_path_bound(n) for n in range(owner.tree.node_count)]
    assert bounds[0]['relative_bound']>0 and bounds[0]['relative_bound']>=torch.finfo(torch.float64).tiny
    assert all(p['relative_bound']==0 for n,p in enumerate(bounds) if not owner.children[n])
    fixed=NestedDirectionalRWGProjection(owner.basis,owner.tree,owner.direction,owner.wave_number,1,quadrature_order=4)
    assert all(fixed.ideal_phase_path_bound(n)['relative_bound']==0 for n in range(fixed.tree.node_count))
    assert owner.retained_bytes==before
    for node in (-1,True,owner.tree.node_count):
        with pytest.raises(ValueError,match='node'):owner.ideal_phase_path_bound(node)
    def forbidden(*a,**kw):raise AssertionError('Rejected phase workspace reached native accumulation')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,'new_zeros',forbidden)
        with pytest.raises(MemoryError,match='workspace'):owner.ideal_phase_path_bound(0,
            tensor_workspace_budget_bytes=directional_phase_path_workspace_bytes(owner.tree.node_count)-1)
    (tmp_path/'path_range.json').write_text(json.dumps(dict(positive_nonexact_floor_preserved=True,
        root_relative_bound=bounds[0]['relative_bound'],fixed_exact_paths_zero=True,
        exact_edges=[c for c,t in owner.transfers.items() if t.exact_phase],
        no_persistent_path_arrays=True,production_qualification=False),indent=2)+'\n')


def test_nested_combined_budget_node_preflight_owner_and_saved_backward_guards(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as module
    test,source,medium=_small_pair();a=b=0
    kwargs=dict(kernel_error_budget=KernelErrorBudget(1.,1.),ideal_rwg_error_budget=1.,balancing_impedance=abs(medium.eta))
    coupling=_coupling(test,source,a,b,medium,**kwargs)
    bound=coupling.ideal_rwg_truncation_bound()
    # A budget accepting the pure kernel but below combined paths must reject.
    pure=coupling.scalar.receipt()['ideal_truncation_admission']
    assert bound['combined_phase_relative_bound']>0
    assert bound['effective_green_absolute']>pure['green_absolute']
    middle=(bound['effective_green_absolute']+pure['green_absolute'])/2
    native=module.CartesianDirectionalFFTTranslation
    def forbidden(*a,**kw):raise AssertionError('Rejected nested admission reached coefficient allocation')
    with monkeypatch.context() as patch:
        patch.setattr(module,'CartesianDirectionalFFTTranslation',forbidden)
        for change,exc,match in [({'kernel_error_budget':KernelErrorBudget(middle,1.)},ValueError,'kernel budget'),
                ({'ideal_rwg_error_budget':bound['absolute_bound']/2},ValueError,'absolute budget'),
                ({'tensor_workspace_budget_bytes':directional_nested_admission_workspace_bytes(test,source)-1},MemoryError,'workspace')]:
            args=dict(kwargs);args.update(change)
            with pytest.raises(exc,match=match):_coupling(test,source,a,b,medium,**args)
        for node in (-1,test.tree.node_count,True):
            with pytest.raises(ValueError,match='node'):_coupling(test,source,node,b,medium,**kwargs)
        with pytest.raises(ValueError,match='order and carrier'):_coupling(test,source,1,b,medium,**kwargs)
    value=torch.ones((2,coupling.count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
    for name,new in [('ideal_rwg_error_budget',.9),('balancing_impedance',coupling.balancing_impedance*.9),
            ('_rwg_context',(source,test)),('_rwg_bound',{**bound,'test_node':1})]:
        old=getattr(coupling,name);output=coupling.apply(value);setattr(coupling,name,new)
        with pytest.raises(ValueError,match='changed'):coupling.receipt()
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,value,torch.ones_like(output))
        setattr(coupling,name,old);coupling.assert_current()
    output=coupling.apply(value);transfer=next(iter(source.transfers.values()));transfer.log_phase_bound.add_(0)
    with pytest.raises(ValueError,match='changed'):coupling.apply(value)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,value,torch.ones_like(output))
    (tmp_path/'nested_guards.json').write_text(json.dumps(dict(combined_not_pure_kernel_admitted=True,
        before_coefficients_and_fulltree_workspace_checked=True,
        external_transfer_and_prescribed_owner_saved_backward_guard=True,production_qualification=False),indent=2)+'\n')


@pytest.mark.parametrize('lossy',(False,True))
def test_strict_relative_mp_path_maximum_and_combined_physical_bounds_only(lossy,tmp_path):
    # Close the late-reviewed implicit-absolute-tolerance gap without repeating
    # any covered physical matrix/action/current/higher or guard controls.
    test,source,a,b,medium=_pair(lossy)
    paths=[]
    for owner,node in ((test,a),(source,b)):
        for check_node in (0,node):
            maximum,branches=_mp_path_bound_products(owner,check_node)
            native=owner.ideal_phase_path_bound(check_node)
            assert maximum==pytest.approx(native['relative_bound'],rel=1e-11,abs=0)
            assert maximum>0
            paths.append(dict(node=check_node,independent_mp_maximum=maximum,
                native=native,branch_bounds=branches,relative_error=abs(maximum-native['relative_bound'])/maximum))
    coupling=_coupling(test,source,a,b,medium)
    bound=coupling.ideal_rwg_truncation_bound();scalar=coupling.scalar.receipt()['ideal_truncation_admission']
    with mp.workdps(100):
        D=mp.sqrt(sum(mp.mpf(v)**2 for v in coupling.scalar.displacement.tolist()))
        radius=sum(mp.mpf(v) for v in coupling.scalar.support_radii.tolist());d=D-radius
        delta=mp.expm1(sum(mp.mpf(p.ideal_phase_path_bound(n)['max_log_one_plus_relative_bound'])
            for p,n in ((test,a),(source,b))))
        G=mp.exp(-mp.mpf(medium.k.imag)*d)/(4*mp.pi*d);grad=G*(abs(mp.mpc(medium.k))+1/d)
        comparisons=[]
        for name,M in [('green_absolute',G),('gradient_component_absolute',grad)]:
            E=mp.mpf(scalar[name]);expected=float(E+delta*(M+E));actual=bound['effective_'+name]
            assert actual==pytest.approx(expected,rel=1e-11,abs=0)
            # Ensure a 1% discrepancy cannot be accepted by an absolute floor.
            assert not (actual*1.01==pytest.approx(expected,rel=1e-11,abs=0))
            comparisons.append(dict(component=name,expected=expected,actual=actual,relative_error=abs(actual-expected)/expected))
    record=dict(lossy=lossy,paths=paths,combined_components=comparisons,
        absolute_comparison_tolerance=0,relative_gate=1e-11,physical_current_controls_repeated=False,
        production_unchanged=True,production_qualification=False)
    (tmp_path/'strict_math.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)

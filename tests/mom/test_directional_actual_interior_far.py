"""Actual connected-body internal directional FAR, not complete system physics."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks,scalar_regional_block
from witwin.maxwell.mom.cluster import SupportOctree,build_support_octree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget,KernelBoundUnavailable
from witwin.maxwell.mom.fmm_error import directional_multipole_truncation_bounds_many,directional_multipole_truncation_bound
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalNestedFarActions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _support_geometry(basis):
    triangles = basis.mesh.triangles
    supports = torch.cat((triangles[basis.plus_faces],triangles[basis.minus_faces]),1)
    lower,upper = supports.amin(1),supports.amax(1)
    tree = build_support_octree(lower,upper,leaf_size=1)
    centers = tree.lower*.5+tree.upper*.5
    leaves = torch.nonzero((tree.children<0).all(-1)).flatten()
    a,b = leaves[centers[leaves,0].argmax()],leaves[centers[leaves,0].argmin()]
    return lower,upper,tree,int(a),int(b)


def _complete_prescribed_tree(lower,upper,original,a,b):
    # All remaining physical edges stay in complete contiguous ranges. This
    # coarsens unused leaf groups without shrinking the actual basis/supports.
    starts = original.starts;stops = original.stops
    assert int(stops[a]-starts[a])==int(stops[b]-starts[b])==1
    cuts = sorted({0,original.size,int(starts[a]),int(stops[a]),int(starts[b]),int(stops[b])})
    ranges = [(0,original.size),*zip(cuts[:-1],cuts[1:])]
    permutation = original.permutation.clone()
    lowers = torch.stack([lower[permutation[x:y]].amin(0) for x,y in ranges])
    uppers = torch.stack([upper[permutation[x:y]].amax(0) for x,y in ranges])
    children = torch.full((len(ranges),8),-1,device=lower.device,dtype=torch.int64)
    children[0,:len(ranges)-1] = torch.arange(1,len(ranges),device=lower.device)
    tree = SupportOctree(permutation,torch.tensor([x for x,_ in ranges],device=lower.device),
        torch.tensor([y for _,y in ranges],device=lower.device),lowers,uppers,children,
        torch.tensor([0]+[1]*(len(ranges)-1),device=lower.device),original.size,original.morton_bits)
    aa = ranges.index((int(starts[a]),int(stops[a])))
    bb = ranges.index((int(starts[b]),int(stops[b])))
    assert tree.size==480 and len(ranges)<=6
    return tree,aa,bb


def _relative(actual,reference):return float(((actual-reference).norm()/reference.norm()).detach())


@pytest.mark.parametrize('lossy',[False,True])
def test_actual_connected_body_interior_original_weak_H_current_higher(lossy,tmp_path):
    exterior = compile_medium(mw.Material(),3e8)
    interior = compile_medium(mw.Material(eps_r=2.,mu_r=1.3,
        sigma_e=.01 if lossy else 0.,sigma_m=.02 if lossy else 0.),3e8)
    basis = RWGBasis(sphere_mesh(radius=.01,subdivisions=2,device='cuda',dtype=torch.float64))
    system = SurfaceSystem.create([(0,'body',basis,interior)])
    integration = IntegrationConfig(order=4,singular_order=3)
    provider = ScalarSystemBlocks(system,exterior,integration)
    lower,upper,original,a,b = _support_geometry(basis)
    tree,a,b = _complete_prescribed_tree(lower,upper,original,a,b)
    centers = tree.lower*.5+tree.upper*.5
    direction = centers[a]-centers[b];direction = direction/direction.norm()
    def owner(d,layout=None):
        return NestedDirectionalRWGProjection(basis,tree,d,interior.k,12,quadrature_order=16,
            geometry=provider.geometry[0],layout=layout,face_batch=4,rhs_batch=4)
    positive = owner(direction);negative = owner(-direction,positive.layout)
    eta = abs(exterior.eta)
    assert abs(eta-abs(interior.eta))/eta>.1
    kernel_budget = KernelErrorBudget(100.,1e5)
    couplings = tuple(DirectionalFFTElectromagneticM2L.from_nested_nodes(p,x,p,y,interior,
        (centers[a]-centers[b]).norm()/(2*12+2),kernel_error_budget=kernel_budget,
        ideal_rwg_error_budget=.1,balancing_impedance=eta) for p,x,y in ((positive,a,b),(negative,b,a)))
    actions = DirectionalNestedFarActions(positive,positive,interior,couplings,
        ideal_rwg_error_budget=.1,balancing_impedance=eta,tensor_workspace_budget_bytes=1024**3)
    assert actions.medium is system.bodies[0].interior
    assert all(p.basis is system.bodies[0].basis and p.tree is tree
        for c in couplings for p in c._rwg_context)
    n = basis.n_edges
    ea = tree.permutation[positive.starts[a]:positive.stops[a]]
    eb = tree.permutation[positive.starts[b]:positive.stops[b]]
    edges = torch.cat((ea,eb));ids = torch.cat((edges,edges+n))
    reference = torch.zeros((4,4),device='cuda',dtype=torch.complex128)
    config = IntegrationConfig(order=20,singular_order=3,pair_batch=8,far_pair_batch=8)
    for i,j,rows,cols in ((0,1,ea,eb),(1,0,eb,ea)):
        block,_ = scalar_regional_block(basis,basis,rows,cols,(interior,),config,
            geometry=(provider.geometry[0],provider.geometry[0]))
        rr,cc = torch.tensor([i,i+2],device='cuda'),torch.tensor([j,j+2],device='cuda')
        reference[rr[:,None],cc] = block
    inputs = torch.zeros((4,2*n),device='cuda',dtype=torch.complex128)
    inputs[torch.arange(4,device='cuda'),ids] = 1
    all_columns = actions.apply(inputs);actual = all_columns[:,ids].mT
    selected = torch.zeros(2*n,device='cuda',dtype=torch.bool);selected[ids] = True
    assert bool((all_columns[:,~selected]==0).all())
    l = lower.new_tensor([eta,eta,1.,1.]);r = lower.new_tensor([1.,1.,eta,eta])
    weak = float((l[:,None]*(actual-reference)*r).norm()/(l[:,None]*reference*r).norm())
    diagnostic = dict(lossy=lossy,phase='original_interior_weak_computed',actual_body_basis_edges=n,
        complete_prescribed_tree_nodes=tree.node_count,test_node=a,source_node=b,
        directed_selected_RWG_blocks=2,physical_domain_shape=actions.shape,
        selected_physical_indices=ids.tolist(),balanced_weak_relative=weak,
        actual_medium_is_body_interior=True,balancing_impedance_from_actual_exterior=eta,
        actual_interior_impedance=str(interior.eta),original_weak_quadrature=20,projection_quadrature=16,
        kernel_control_budget=dict(green_absolute=100.,gradient_component_absolute=1e5),
        ideal_absolute_control_budget=.1,receipt=actions.receipt(),
        selected_interior_far_only=True,complete_interior_system_physics_covered=False,
        production_qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    torch.save(dict(actual=actual,reference=reference,ids=ids),tmp_path/'weak.pt');print(diagnostic,flush=True)
    assert weak<1e-8
    generator = torch.Generator(device='cuda').manual_seed(2633)
    current = torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        output = actions.apply(current)
    H = actions.adjoint(dual)
    original_H = torch.zeros_like(H);original_H[...,ids] = dual[...,ids]@reference.conj()
    native_output = torch.zeros_like(output);native_output[...,ids] = current[...,ids]@actual.mT
    native_H = torch.zeros_like(H);native_H[...,ids] = dual[...,ids]@actual.conj()
    derivative, = torch.autograd.grad(output,current,dual,retain_graph=True)
    first, = torch.autograd.grad(output.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn_like(current);second, = torch.autograd.grad(first,current,tangent)
    lhs,rhs = (output.conj()*dual).sum(),(current.conj()*H).sum()
    errors = dict(original_H=_relative(H,original_H),native_H=_relative(H,native_H),
        native_forward=_relative(output,native_output),current=_relative(derivative,H),
        higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    assert not saved and not current.is_contiguous()
    assert bool((output[...,~selected]==0).all() & (H[...,~selected]==0).all())
    owned = []
    for p in (positive,negative):
        amount = sum(t.numel()*t.element_size() for t in p.tensors())
        assert amount==p.retained_bytes==p.planned_retained_bytes
        owned.append(amount)
    diagnostic.update(phase='all_interior_current_checks_done',errors=errors,
        physical_rhs=15,saved_internal_tensors=len(saved),caller_projection_owned_bytes=owned,
        newly_owned_far_bytes=actions.retained_bytes,omitted_physical_outputs_exact_zero=True)
    (tmp_path/'interior.json').write_text(json.dumps(diagnostic,indent=2)+'\n');print(diagnostic,flush=True)
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11
    assert not actions.receipt()['production_qualification']


def test_actual_coarse_body_support_unavailable_does_not_accept_zero_placeholders(tmp_path):
    basis = RWGBasis(sphere_mesh(radius=.01,subdivisions=1,device='cuda',dtype=torch.float64))
    _,_,tree,a,b = _support_geometry(basis)
    centers = tree.lower*.5+tree.upper*.5
    radii = torch.maximum((tree.lower-centers).abs(),(tree.upper-centers).abs()).norm(dim=-1)
    radii = radii*(1+512*torch.finfo(torch.float64).eps)
    displacement = centers[a]-centers[b];direction = displacement/displacement.norm()
    medium = compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.01,sigma_m=.02),3e8)
    bounds = directional_multipole_truncation_bounds_many(displacement[None],direction[None],
        radii[a][None],radii[b][None],medium.k,(8,10,12))
    assert not bool(bounds['available'].any())
    assert bool((bounds['green_absolute']==0).all() & (bounds['gradient_component_absolute']==0).all())
    with pytest.raises(KernelBoundUnavailable):directional_multipole_truncation_bound(
        displacement,direction,radii[a],radii[b],medium.k,12)
    (tmp_path/'unavailable_actual_support.json').write_text(json.dumps(dict(actual_connected_body_edges=basis.n_edges,
        support_ratio=float((radii[a]+radii[b])/displacement.norm()),
        available=bounds['available'].tolist(),zero_placeholders_not_admitted=True,
        selected_negative_control=True,production_qualification=False),indent=2)+'\n')

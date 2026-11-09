"""Complete sparse exterior/interior coverage and actual shared Krylov solves."""
import itertools
import os
import hashlib
from pathlib import Path
from dataclasses import asdict
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig,assemble_system
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks,scalar_regional_block
from witwin.maxwell.mom.cluster import SupportOctree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.direct_capacity import storage_bytes
from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalSparseFarActions
from witwin.maxwell.mom.fmm_regional import DirectionalRegionalOperator
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import StoppingRule,residual_loss
from witwin.maxwell.mom.runtime import SystemDecodeHandle,MoMConfig
from witwin.maxwell.mom.lifting import build_lifted_rhs
from test_directional_far_actions import _original_global_far_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _basis(z):
    # Refine the same closed slender box longitudinally; caps and physical
    # dimensions are unchanged. Coarse18-edge near-convergence failure stays
    # recorded rather than relaxing the near quadrature tolerance.
    segments=16
    corners=torch.tensor([[-.006,-.006],[.006,-.006],[.006,.006],[-.006,.006]],device='cuda',dtype=torch.float64)
    x=torch.linspace(-.5,.5,segments+1,device='cuda',dtype=torch.float64)
    vertices=torch.cat((x[:,None,None].expand(-1,4,1),corners[None].expand(segments+1,-1,-1)),dim=-1).reshape(-1,3)
    vertices[:,2]+=z
    faces=[[0,2,1],[0,3,2]]
    last=4*segments;faces.extend(((last,last+1,last+2),(last,last+2,last+3)))
    for ring in range(segments):
        for j in range(4):
            a=4*ring+j;b=a+4;c=4*(ring+1)+(j+1)%4;d=4*ring+(j+1)%4
            faces.extend(((a,c,b),(a,d,c)))
    return RWGBasis(SurfaceMesh(vertices,torch.tensor(faces,device='cuda')))


def _caps(basis):
    x=basis.mesh.triangles[:,:,0]
    support=torch.cat((x[basis.plus_faces],x[basis.minus_faces]),dim=-1)
    caps=(support.amin(-1)==support.amax(-1)).nonzero().flatten()
    assert len(caps)==2
    caps=caps[support[caps,0].argsort()]
    return caps.tolist()


def _physical_tree(lower,upper,groups):
    # Prescribed exact physical partition of every RWG edge. This does not
    # claim automatic Morton/cone/admissibility construction.
    permutation = torch.cat(groups)
    assert torch.equal(permutation.sort().values,torch.arange(len(lower),device=lower.device))
    cuts = [0]
    for group in groups:cuts.append(cuts[-1]+len(group))
    ranges = [(0,len(lower)),*zip(cuts[:-1],cuts[1:])]
    lows = torch.stack([lower[permutation[a:b]].amin(0) for a,b in ranges])
    highs = torch.stack([upper[permutation[a:b]].amax(0) for a,b in ranges])
    children = torch.full((len(ranges),8),-1,device=lower.device,dtype=torch.int64)
    children[0,:len(groups)] = torch.arange(1,len(ranges),device=lower.device)
    return SupportOctree(permutation,torch.tensor([a for a,b in ranges],device=lower.device),
        torch.tensor([b for a,b in ranges],device=lower.device),lows,highs,children,
        torch.tensor([0]+[1]*len(groups),device=lower.device),max(map(len,groups)),10)


def _regional(basis,tree,medium,pairs,eta,config,geometry=None):
    centers = tree.lower*.5+tree.upper*.5
    direction = centers[pairs[0][0]]-centers[pairs[0][1]];direction=direction/direction.norm()
    carriers = torch.stack((direction,-direction))
    physical = sorted({n for pair in pairs for n in pair})
    requests = tuple((n,c) for c in range(2) for n in physical)
    owner = SparseDirectionalRWGProjection(basis,tree,carriers,(8,8),requests,medium.k,
        geometry=geometry,quadrature_order=12,face_batch=4,rhs_batch=4)
    indices = {key:i for i,key in enumerate(owner.context_keys)}
    couplings = []
    for slot,(a,b) in enumerate(pairs):
        x,y = indices[a,slot],indices[b,slot]
        scale = torch.minimum((centers[a]-centers[b]).norm()/18,torch.maximum(owner.radii[x],owner.radii[y]))
        couplings.append(DirectionalFFTElectromagneticM2L.from_sparse_contexts(owner,x,owner,y,medium,scale,
            kernel_error_budget=KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,
            balancing_impedance=eta))
    far = DirectionalSparseFarActions(owner,owner,medium,tuple(couplings),ideal_rwg_error_budget=1e-4,
        balancing_impedance=eta,tensor_workspace_budget_bytes=512*1024**2)
    leaves = tuple(range(1,tree.node_count))
    near = tuple(pair for pair in itertools.product(leaves,repeat=2) if pair not in pairs)
    region = DirectionalRegionalOperator(far,near,integration=config)
    assert any(node>=len(owner.context_keys) for pair in near for node in pair) or tree.node_count==4
    return region


def _reference_identity(provider):
    def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return dict(integration=asdict(provider.config),frequency=provider.exterior.frequency,
        exterior=dict(epsilon=str(provider.exterior.epsilon_r),mu=str(provider.exterior.mu_r),k=str(provider.exterior.k)),
        bodies=[dict(offset=b.edge_offset,edges=b.basis.n_edges,vertices_sha=digest(b.basis.mesh.vertices),
            faces_sha=digest(b.basis.mesh.faces),epsilon=str(b.interior.epsilon_r),mu=str(b.interior.mu_r),k=str(b.interior.k))
            for b in provider.system.bodies])


def _load_reference(provider,manifest):
    data=json.loads(Path(manifest).read_text())
    if data.get('version')!='sparse_complete_original_reference_v1' or data.get('identity')!=_reference_identity(provider):
        raise ValueError('Explicit original reference has a different physical/configuration identity')
    path=Path(data['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=data['sha256']:
        raise ValueError('Explicit original reference matrix SHA mismatch')
    payload=torch.load(path,map_location=provider.system.device,weights_only=True)
    matrix=payload['reference'];n=2*provider.system.total_edges
    if (matrix.shape!=(n,n) or matrix.dtype!=torch.complex128 or matrix.device!=provider.system.device
            or matrix.requires_grad or not bool(torch.isfinite(matrix).all())):
        raise ValueError('Explicit original reference requires the complete native physical matrix')
    return matrix


def _original_reference(provider):
    manifest=os.environ.get('GSP_DIRECTIONAL_SPARSE_SYSTEM_REFERENCE_MANIFEST')
    if manifest:
        matrix=_load_reference(provider,manifest)
        print(dict(original_reference='explicit_identity_SHA_bound_reuse',manifest=manifest),flush=True)
        return matrix
    matrix,_=assemble_system(provider.system,provider.exterior,provider.config)
    return matrix


@pytest.fixture(scope='module')
def case():
    frequency = 3e8;exterior = compile_medium(mw.Material(),frequency)
    entries = []
    for i,(z,eps,mu,se,sm) in enumerate(((-.3,2.,1.3,.01,.02),(.3,3.,1.5,.015,.005))):
        basis = _basis(z); assert basis.n_edges==198
        medium = compile_medium(mw.Material(eps_r=eps,mu_r=mu,sigma_e=se,sigma_m=sm),frequency)
        entries.append((i,str(i),basis,medium))
    system = SurfaceSystem.create(entries)
    config = IntegrationConfig(order=6,singular_order=3,pair_batch=4,far_pair_batch=16,near_max_order=32)
    provider = ScalarSystemBlocks(system,exterior,config)
    lower,upper = provider.support_boxes();groups=[]
    for body in system.bodies:
        caps = _caps(body.basis)
        groups.extend([torch.tensor([edge+body.edge_offset],device='cuda') for edge in caps])
        groups.append(torch.tensor([e+body.edge_offset for e in range(body.basis.n_edges) if e not in caps],device='cuda'))
    tree = _physical_tree(lower,upper,groups);eta=abs(exterior.eta)
    regional = _regional(provider,tree,exterior,((1,5),(5,1)),eta,config)
    interiors = []
    for i,body in enumerate(system.bodies):
        start,stop = body.edge_offset,body.edge_offset+body.basis.n_edges
        local_groups = [group-start for group in groups[3*i:3*i+3]]
        local_tree = _physical_tree(lower[start:stop],upper[start:stop],local_groups)
        interiors.append(_regional(body.basis,local_tree,body.interior,((1,2),(2,1)),eta,config,provider.geometry[i]))
    actions = DirectionalSystemActions(provider,regional,interiors,ideal_rwg_error_budget=1e-4)
    print(dict(stage='complete_sparse_system_constructed',full_edges=system.total_edges,
        exterior_near=len(regional.near_nodes),interior_near=[len(r.near_nodes) for r in interiors],
        positive_actual_interior_far=[len(r.far.couplings) for r in interiors]),flush=True)
    dense = _original_reference(provider)
    rhs_values=[]
    for polarization in ('Ex','Ey'):
        decoder = SystemDecodeHandle(system,exterior,[mw.PlaneWave(polarization=polarization)],MoMConfig(integration=config),{})
        rhs = torch.empty(2*system.total_edges,device='cuda',dtype=torch.complex128)
        for i,handle in enumerate(decoder.body_handles):
            local,_ = build_lifted_rhs(handle);n=handle.basis.n_edges
            rhs[system.J_slices[i]],rhs[system.M_slices[i]] = local[:n],local[n:]
        rhs_values.append(rhs)
    rhs=torch.stack(rhs_values);reference=torch.linalg.solve(dense,rhs.mT).mT
    return provider,actions,dense,rhs,reference


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def _forbid(*args,**kwargs):raise AssertionError('Sparse complete adapter reconstructed actions or used dense fallback')


def test_complete_sparse_actual_body_near_far_full_weak_H_current_higher_and_ideal(case,tmp_path):
    provider,actions,dense,_,_=case;n=provider.system.total_edges
    columns=actions.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)).mT
    eta=abs(provider.exterior.eta)
    l,r=dense.real.new_tensor([eta]*n+[1.]*n),dense.real.new_tensor([1.]*n+[eta]*n)
    weak=float((l[:,None]*(columns-dense)*r).norm()/(l[:,None]*dense*r).norm())
    # Independently integrate the actual selected far pairs at q16, including
    # positive intra-body far in both different lossy magnetic materials.
    far_errors=[]
    for region in (actions.exterior,*actions.interiors):
        owner=region.test;size=owner.tree.size
        for coupling,rectangle in zip(region.far.couplings,region.far.rectangles,strict=True):
            start,stop,low,high,_,_=rectangle
            rows,cols=owner.tree.permutation[start:stop],owner.tree.permutation[low:high]
            if isinstance(owner.basis,ScalarSystemBlocks):original=_original_global_far_block(owner.basis,rows,cols,region.medium)
            else:
                original,_=scalar_regional_block(owner.basis,owner.basis,rows,cols,(region.medium,),
                    IntegrationConfig(order=16,singular_order=3),geometry=(owner.geometry,owner.geometry))
            rr,cc=torch.cat((rows,rows+size)),torch.cat((cols,cols+size))
            inputs=torch.eye(2*size,device='cuda',dtype=torch.complex128)[cc]
            block=region.far.apply(inputs)[:,rr].mT
            left,right=block.real.new_tensor([eta]*len(rows)+[1.]*len(rows)),block.real.new_tensor([1.]*len(cols)+[eta]*len(cols))
            far_errors.append(float((left[:,None]*(block-original)*right).norm()/(left[:,None]*original*right).norm()))
    record=dict(phase='complete_original_weak',whole_system_weak_relative=weak,selected_original_q16_far_relative=far_errors,
        positive_actual_interior_far_both_bodies=True,full_original_discrete_system=True,
        near_integration_physics_certified=False,full_edges=n,system_receipt=actions.receipt(),qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n');torch.save(dict(actual=columns.detach(),reference=dense.detach()),tmp_path/'weak.pt');print(record,flush=True)
    assert weak<1e-8 and max(far_errors)<1e-8
    with mp.workdps(100):
        bounds=[r.ideal_rwg_truncation_bound() for r in (actions.exterior,*actions.interiors)]
        expected=mp.mpf(bounds[0]['absolute_bound'])+mp.sqrt(sum(mp.mpf(b['absolute_bound'])**2 for b in bounds[1:]))
        actual=actions.ideal_rwg_truncation_bound()['absolute_bound']
        assert actual==pytest.approx(float(expected),rel=1e-11,abs=0)
        assert not (actual==pytest.approx(float(expected*mp.mpf('1.01')),rel=1e-11,abs=0))
    g=torch.Generator(device='cuda').manual_seed(55271)
    current=torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=g).transpose(0,1).requires_grad_()
    dual=torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=g).requires_grad_()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):out=actions.apply(current)
    H=actions.adjoint(dual);derivative,=torch.autograd.grad(out,current,dual,retain_graph=True)
    H_derivative,=torch.autograd.grad(H,dual,current,retain_graph=True)
    first,=torch.autograd.grad(out.abs().square().sum(),current,create_graph=True)
    tangent=torch.randn_like(current);second,=torch.autograd.grad(first,current,tangent)
    lhs,rhs=(out.conj()*dual).sum(),(current.conj()*H).sum()
    errors=dict(original_H=_relative(H,dual@dense.conj()),native_H=_relative(H,dual@columns.conj()),
        forward=_relative(out,current@columns.mT),current=_relative(derivative,H),
        adjoint_current=_relative(H_derivative,actions.apply(current)),
        higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    regions=(actions.exterior,*actions.interiors)
    inventories=dict(near=[storage_bytes(r.tensors()) for r in regions],maps=[storage_bytes(r.test.tensors()) for r in regions],
        couplings=[sum(storage_bytes(c.tensors()) for c in r.far.couplings) for r in regions],embeddings=storage_bytes(actions.tensors()))
    assert inventories['embeddings']==actions.retained_bytes==16*n
    assert all(amount==region.retained_bytes for amount,region in zip(inventories['near'],regions))
    assert not saved
    record.update(phase='complete',errors=errors,physical_rhs=15,noncontiguous=True,saved_internal_tensors=len(saved),inventories=inventories)
    (tmp_path/'system.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11


def test_sparse_complete_native_Riesz_multi_rhs_Krylov_original_residual_and_failure(case,monkeypatch,tmp_path):
    provider,actions,dense,rhs,reference=case
    with monkeypatch.context() as patch:
        for path in ('witwin.maxwell.mom.hmatrix.HierarchicalOperator.__init__',
                     'witwin.maxwell.mom.direct_system.DirectSystemActions.__init__',
                     'witwin.maxwell.mom.fmm_system.MultipoleSystemActions.__init__',
                     'witwin.maxwell.mom.linalg.DenseOperator.__init__','witwin.maxwell.mom.linalg._one_shot_direct'):
            patch.setattr(path,_forbid)
        op=DirectionalMultipoleOperator(provider,actions)
        assert isinstance(op,SurfaceOperatorProtocol) and op.actions is actions and not op.supports_direct
        assert not hasattr(op,'matrix')
        x=torch.randn_like(rhs,requires_grad=True)
        grad,=torch.autograd.grad(residual_loss(op,x,rhs),x)
        gradient_error=_relative(grad,op.adjoint(op.riesz_inverse(op.apply(x)-rhs)));assert gradient_error<1e-11
        n=provider.system.total_edges
        # The complete original matrix was independently assembled once. Reuse
        # it only for original residual samples and bounded principal factors;
        # Krylov actions still use the complete sparse near/FAR physical owner.
        from witwin.maxwell.mom.block_preconditioner import LeafBlockInverse
        native_selection=ScalarSystemBlocks.__call__
        reference_selections=[]
        def original_selection(current_provider,rows,cols):
            if current_provider is not provider:return native_selection(current_provider,rows,cols)
            provider.assert_current()
            rr,cc=torch.cat((rows,rows+n)),torch.cat((cols,cols+n))
            reference_selections.append((len(rows),len(cols)))
            return dense[rr[:,None],cc]
        patch.setattr(ScalarSystemBlocks,'__call__',original_selection)
        inverse=LeafBlockInverse(provider,actions.exterior.test.tree,max_block_edges=32)
        assert inverse.provider is provider and inverse.max_block_edges==32
        assert inverse.retained_bytes==storage_bytes(inverse.tensors())
        assert max(len(ids)//2 for ids in inverse.indices)<=32
        def dual(value):return (value.conj()*op.riesz_inverse(value)).sum(-1).real.clamp_min(0).sqrt()
        scale=torch.cat((torch.ones(n,device='cuda'),torch.full((n,),abs(provider.exterior.eta),device='cuda')))
        logs=[]
        for method in ('gmres','block_gmres'):
            solved,log=op.solve(rhs,StoppingRule(method=method,preconditioner='rwg_diagonal',
                rtol=1e-12,atol=0.,maxiter=768,restart=384),original_edges=torch.arange(n,device='cuda'),source_batch=4,
                right_inverse=inverse)
            original=float((dual(solved@dense.mT-rhs)/dual(rhs)).max())
            current_error=float((torch.linalg.vector_norm((solved-reference)/scale,dim=-1)/torch.linalg.vector_norm(reference/scale,dim=-1)).max())
            data=dict(method=method,original_dual_relative=original,dense_current_relative=current_error,log=log)
            logs.append(data);print(data,flush=True)
            (tmp_path/'solve.json').write_text(json.dumps(dict(methods=logs,native_Riesz_current_gradient=gradient_error,
                actual_original_full_edges=n,physical_rhs=2,positive_both_interiors_far=True,
                original_reference_reused_only_for_samples_and_bounded_factors=True,
                original_leaf_inverse=inverse.receipt(),qualification=False),indent=2)+'\n')
            assert log['converged'] and log['inner_converged'] and log['error_budget']['checked'],log
            assert log['error_budget']['compressed_dual_pass'] and log['error_budget']['original_sampled_pass']
            assert not log['fallback_used'] and not log['original_global_residual_certified']
            assert original<1e-8 and current_error<1e-7
        _,failure=op.solve(rhs,StoppingRule(method='gmres',preconditioner='rwg_diagonal',rtol=1e-12,atol=0.,maxiter=1,restart=1))
        assert not failure['converged'] and not failure['error_budget']['checked'] and not failure['fallback_used']
        for rule in (StoppingRule(method='direct'),StoppingRule(fallback='direct')):
            with pytest.raises(ValueError,match='requires a dense operator'):op.solve(rhs,rule)
        (tmp_path/'failure.json').write_text(json.dumps(dict(one_step_failure=failure,explicit_direct_and_fallback_refused=True,
            snapshot=op.persistence_receipt(),reference_selection_count=len(reference_selections),
            bounded_original_factors_only=True,qualification=False),indent=2)+'\n')


def test_sparse_region_physical_coverage_preflight_and_original_entry_guards(case,monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_regional as module
    _,actions,_,_,_=case;region=actions.exterior;far=region.far
    assert max(n for pair in region.near_nodes for n in pair)>=len(region.test.context_keys)
    assert region.receipt()['near_node_index_space']=='physical_support_tree'
    def create(near,**kw):return DirectionalRegionalOperator(far,near,integration=region.integration,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(module.PackedScalarNearChannels,'from_layout',_forbid)
        for near in (region.near_nodes[:-1],region.near_nodes+region.near_nodes[:1],((0,0),),((True,1),)):
            with pytest.raises(ValueError):create(near)
        tn,sn=far.couplings[0].ideal_rwg_truncation_bound()['test_physical_node'],far.couplings[0].ideal_rwg_truncation_bound()['source_physical_node']
        with pytest.raises(ValueError):create(region.near_nodes+((tn,sn),))
        with pytest.raises(MemoryError):create(region.near_nodes,near_retained_budget_bytes=region.retained_bytes-1)
        with pytest.raises(MemoryError):create(region.near_nodes,tensor_workspace_budget_bytes=region.planned_tensor_workspace_bytes-1)
        with pytest.raises(TypeError):DirectionalRegionalOperator(object(),(),integration=region.integration)
    provider=actions.provider;interiors=actions.interiors
    with pytest.raises(ValueError):DirectionalSystemActions(provider,region,interiors[::-1],ideal_rwg_error_budget=1e-4)
    with pytest.raises(ValueError):DirectionalSystemActions(provider,region,interiors[:1],ideal_rwg_error_budget=1e-4)
    with pytest.raises(ValueError):DirectionalSystemActions(provider,region,interiors,ideal_rwg_error_budget=1e-4,balancing_impedance=abs(interiors[0].medium.eta))
    with pytest.raises(MemoryError):DirectionalSystemActions(provider,region,interiors,ideal_rwg_error_budget=1e-4,retained_budget_bytes=actions.retained_bytes-1)
    (tmp_path/'coverage.json').write_text(json.dumps(dict(near_ids_are_physical_not_sparse=True,
        all_original_coefficients_covered=True,missing_duplicate_overlap_internal_bool_rejected_before_near_integration=True,
        capacities_before_layout=True,actual_body_medium_and_common_impedance_preserved=True,qualification=False),indent=2)+'\n')


def test_sparse_saved_full_system_and_Riesz_guards_without_reconstruction(case,monkeypatch,tmp_path):
    provider,actions,_,rhs,_=case
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.direct_operator.LocalRWGRiesz',_forbid)
        with pytest.raises(TypeError):DirectionalMultipoleOperator(provider,actions.exterior.far)
        with pytest.raises(ValueError):DirectionalMultipoleOperator(object(),actions)
    op=DirectionalMultipoleOperator(provider,actions)
    trips=[]
    for kind in ('near_budget','context_keys','owned_modes','coupling_budget','body_order','system_budget'):
        current=torch.ones_like(rhs,requires_grad=True);out=actions.adjoint(current)
        if kind=='near_budget':obj,name,new=actions.interiors[0],'near_retained_budget_bytes',1
        elif kind=='context_keys':obj,name,new=actions.exterior.test,'context_keys',actions.exterior.test.context_keys[::-1]
        elif kind=='owned_modes':
            tensor=actions.interiors[0].test.carriers;tensor.requires_grad_(True)
            with pytest.raises(ValueError):torch.autograd.grad(out,current,torch.ones_like(out))
            tensor.requires_grad_(False);actions.assert_current();trips.append(kind);continue
        elif kind=='coupling_budget':obj,name,new=actions.interiors[1].far.couplings[0],'ideal_rwg_error_budget',.9
        elif kind=='body_order':obj,name,new=actions,'interiors',actions.interiors[::-1]
        else:obj,name,new=actions,'ideal_rwg_error_budget',.9
        old=getattr(obj,name);setattr(obj,name,new)
        with pytest.raises(ValueError):actions.receipt()
        with pytest.raises(ValueError):torch.autograd.grad(out,current,torch.ones_like(out))
        setattr(obj,name,old);actions.assert_current();trips.append(kind)
    x=torch.randn_like(rhs,requires_grad=True);loss=residual_loss(op,x,rhs)
    op.metric.diagonal.add_(0)
    with pytest.raises(ValueError):torch.autograd.grad(loss,x)
    (tmp_path/'saved_guards.json').write_text(json.dumps(dict(pre_Riesz_identity_preserved=True,
        saved_full_system_adjoint_tripwires=trips,saved_Riesz_metric_mutation_guard=True,qualification=False),indent=2)+'\n')


def test_explicit_original_reference_identity_SHA_and_portable_default_lifecycle(case,monkeypatch,tmp_path):
    provider,_,dense,_,_=case
    manifest=os.environ.get('GSP_DIRECTIONAL_SPARSE_SYSTEM_REFERENCE_MANIFEST')
    if manifest:
        actual=_load_reference(provider,manifest);assert torch.equal(actual,dense)
        data=json.loads(Path(manifest).read_text())
    else:
        original=tmp_path/'original.pt';torch.save(dict(reference=dense.detach()),original)
        data=dict(version='sparse_complete_original_reference_v1',identity=_reference_identity(provider),
            path=str(original),sha256=hashlib.sha256(original.read_bytes()).hexdigest())
        manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(data))
        assert torch.equal(_load_reference(provider,manifest),dense)
    errors=[]
    for kind in ('identity','sha','missing','partial'):
        changed={**data}
        if kind=='identity':changed['identity']={**data['identity'],'frequency':1.}
        elif kind=='sha':changed['sha256']='0'*64
        elif kind=='missing':changed['path']=str(tmp_path/'absent.pt')
        else:
            path=tmp_path/'partial.pt';torch.save(dict(reference=dense[:1].detach()),path)
            changed.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path=tmp_path/(kind+'.json');path.write_text(json.dumps(changed))
        with pytest.raises((ValueError,FileNotFoundError)):_load_reference(provider,path)
        errors.append(kind)
    assert 'GSP_DIRECTIONAL_SPARSE_SYSTEM_REFERENCE_MANIFEST' in _original_reference.__code__.co_consts
    record=dict(optional_explicit_SHA_identity_bound_reference=True,strict_reuse_failures=errors,
        default_fresh_same_original_assembly=True,no_mandatory_historical_artifact=True,
        old3PASS_physics_guards_not_repeated=True,qualification=False)
    (tmp_path/'reference_lifecycle.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)

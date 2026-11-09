"""Complete prescribed changing-carrier DAG exterior and magnetic interiors."""
import itertools
import json
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks,scalar_regional_block
from witwin.maxwell.mom.cluster import SupportOctree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.direct_capacity import storage_bytes
from witwin.maxwell.mom.fmm_directional_moments import DirectionalRWGDAGProjection
from witwin.maxwell.mom.fmm_directional_rwg import DirectionalFFTElectromagneticM2L,DirectionalDAGFarActions
from witwin.maxwell.mom.fmm_regional import DirectionalRegionalOperator
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.linalg import StoppingRule,residual_loss
from witwin.maxwell.mom.runtime import SystemDecodeHandle,MoMConfig
from witwin.maxwell.mom.lifting import build_lifted_rhs
from test_directional_sparse_system import _basis,_caps,_physical_tree,_original_reference
from test_directional_far_actions import _original_global_far_block
from test_directional_dag_coupling import _mp_products

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _branch_caps(tree,cap_nodes):
    # Add one physical child with the complete same support beneath each cap.
    # Nearby requested parent carriers share its once-integrated child context.
    count=tree.node_count;caps=torch.tensor(cap_nodes,device=tree.device)
    children=torch.cat((tree.children.clone(),tree.children.new_full((len(caps),8),-1)))
    descendants={node:count+i for i,node in enumerate(cap_nodes)}
    for node,child in descendants.items():children[node,0]=child
    refined=SupportOctree(tree.permutation.clone(),torch.cat((tree.starts,tree.starts[caps])),
        torch.cat((tree.stops,tree.stops[caps])),torch.cat((tree.lower,tree.lower[caps])),
        torch.cat((tree.upper,tree.upper[caps])),children,torch.cat((tree.depth,tree.depth[caps]+1)),
        tree.leaf_size,tree.morton_bits)
    return refined,descendants


def _regional(basis,tree,descendants,medium,pairs,eta,config,geometry=None):
    centers=tree.lower*.5+tree.upper*.5
    direction=centers[pairs[0][0]]-centers[pairs[0][1]];direction=direction/direction.norm()
    step=direction.new_tensor([.2,-.4,.7])*1e-4
    carriers=torch.stack((direction+step,direction-step,direction,-direction+step,-direction-step,-direction))
    carriers=carriers/carriers.norm(dim=-1,keepdim=True)
    physical=sorted({node for pair in pairs for node in pair})
    requests=tuple((node,family) for node in physical for family in (0,1,3,4))
    links=tuple(((node,family),(descendants[node],2 if family<3 else 5),4) for node,family in requests)
    owner=DirectionalRWGDAGProjection(basis,tree,carriers,(8,8,12,8,8,12),requests,links,medium.k,
        geometry=geometry,quadrature_order=12,face_batch=4,rhs_batch=4,phase_relative_budget=1e-12)
    indices={key:i for i,key in enumerate(owner.context_keys)}
    couplings=[]
    for family,(a,b) in zip((0,3),pairs,strict=True):
        x,y=indices[a,family],indices[b,family]
        scale=torch.minimum((centers[a]-centers[b]).norm()/18,torch.maximum(owner.radii[x],owner.radii[y]))
        couplings.append(DirectionalFFTElectromagneticM2L.from_dag_contexts(owner,x,owner,y,medium,scale,
            kernel_error_budget=KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,balancing_impedance=eta))
    far=DirectionalDAGFarActions(owner,owner,medium,tuple(couplings),ideal_rwg_error_budget=1e-4,
        balancing_impedance=eta,tensor_workspace_budget_bytes=512*1024**2)
    leaves=[node for node,row in enumerate(tree.children.tolist()) if all(child<0 for child in row)]
    excluded={(descendants[a],descendants[b]) for a,b in pairs}
    near=tuple(pair for pair in itertools.product(leaves,repeat=2) if pair not in excluded)
    return DirectionalRegionalOperator(far,near,integration=config)


@pytest.fixture(scope='module')
def case():
    frequency=3e8;exterior=compile_medium(mw.Material(),frequency)
    entries=[]
    for i,(z,eps,mu,se,sm) in enumerate(((-.3,2.,1.3,.01,.02),(.3,3.,1.5,.015,.005))):
        basis=_basis(z);assert basis.n_edges==198
        medium=compile_medium(mw.Material(eps_r=eps,mu_r=mu,sigma_e=se,sigma_m=sm),frequency)
        entries.append((i,str(i),basis,medium))
    system=SurfaceSystem.create(entries)
    config=IntegrationConfig(order=6,singular_order=3,pair_batch=4,far_pair_batch=16,near_max_order=32)
    provider=ScalarSystemBlocks(system,exterior,config)
    lower,upper=provider.support_boxes();groups=[]
    for body in system.bodies:
        caps=_caps(body.basis)
        groups.extend([torch.tensor([edge+body.edge_offset],device='cuda') for edge in caps])
        groups.append(torch.tensor([edge+body.edge_offset for edge in range(body.basis.n_edges) if edge not in caps],device='cuda'))
    tree,descendants=_branch_caps(_physical_tree(lower,upper,groups),(1,2,4,5));eta=abs(exterior.eta)
    regional=_regional(provider,tree,descendants,exterior,((1,5),(5,1)),eta,config)
    interiors=[]
    for i,body in enumerate(system.bodies):
        start,stop=body.edge_offset,body.edge_offset+body.basis.n_edges
        local_groups=[group-start for group in groups[3*i:3*i+3]]
        local_tree,local_descendants=_branch_caps(_physical_tree(lower[start:stop],upper[start:stop],local_groups),(1,2))
        interiors.append(_regional(body.basis,local_tree,local_descendants,body.interior,((1,2),(2,1)),eta,config,provider.geometry[i]))
    actions=DirectionalSystemActions(provider,regional,interiors,ideal_rwg_error_budget=1e-4)
    print(dict(stage='complete_DAG_system_constructed',full_edges=system.total_edges,
        exterior_near=len(regional.near_nodes),interior_near=[len(region.near_nodes) for region in interiors],
        positive_actual_interior_far=[len(region.far.couplings) for region in interiors],
        shared_children=[region.test.receipt()['shared_child_contexts'] for region in (regional,*interiors)]),flush=True)
    dense=_original_reference(provider)
    rhs_values=[]
    for polarization in ('Ex','Ey'):
        decoder=SystemDecodeHandle(system,exterior,[mw.PlaneWave(polarization=polarization)],MoMConfig(integration=config),{})
        rhs=torch.empty(2*system.total_edges,device='cuda',dtype=torch.complex128)
        for i,handle in enumerate(decoder.body_handles):
            local,_=build_lifted_rhs(handle);n=handle.basis.n_edges
            rhs[system.J_slices[i]],rhs[system.M_slices[i]]=local[:n],local[n:]
        rhs_values.append(rhs)
    rhs=torch.stack(rhs_values);reference=torch.linalg.solve(dense,rhs.mT).mT
    return provider,actions,dense,rhs,reference


def _relative(x,y):return float(((x-y).norm()/y.norm()).detach())


def _forbid(*args,**kwargs):raise AssertionError('DAG complete adapter reconstructed actions or used dense fallback')


def test_complete_dag_actual_body_near_far_full_weak_H_current_higher_and_ideal(case,tmp_path):
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
    record=dict(phase='complete_DAG_original_weak',whole_system_weak_relative=weak,selected_original_q16_far_relative=far_errors,
        positive_actual_interior_far_both_bodies=True,full_original_discrete_system=True,
        near_integration_physics_certified=False,full_edges=n,system_receipt=actions.receipt(),qualification=False)
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert weak<1e-8 and max(far_errors)<1e-8
    phase_errors=[]
    with mp.workdps(100):
        for region in (actions.exterior,*actions.interiors):
            owner=region.test;products=_mp_products(owner)
            assert owner.receipt()['shared_child_contexts']>0 and len(region.far.couplings)==2
            for coupling in region.far.couplings:
                bound=coupling.ideal_rwg_truncation_bound()
                a,b=bound['test_context'],bound['source_context']
                expected=products[a]*products[b]-1
                assert expected>0 and bound['changing_carrier_DAG']
                phase_errors.append(float(abs(mp.mpf(bound['combined_phase_relative_bound'])/expected-1)))
        assert max(phase_errors)<1e-11
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
    record.update(phase='complete',errors=errors,physical_rhs=15,noncontiguous=True,saved_internal_tensors=len(saved),inventories=inventories,mp100_phase_relative_errors=phase_errors,mp_absolute_tolerance=0,all_regions_shared_changing_DAG=True)
    (tmp_path/'system.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert errors['original_H']<1e-8 and max(v for k,v in errors.items() if k!='original_H')<1e-11



def test_dag_complete_native_Riesz_multi_rhs_Krylov_original_residual_and_failure(case,monkeypatch,tmp_path):
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
        # Krylov actions still use the complete DAG near/FAR physical owner.
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



def test_dag_region_physical_coverage_preflight_and_original_entry_guards(case,monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_regional as module
    _,actions,_,_,_=case;region=actions.exterior;far=region.far
    assert any(node not in {key[0] for key in region.test.context_keys} for pair in region.near_nodes for node in pair)
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



def test_dag_saved_full_system_and_Riesz_guards_without_reconstruction(case,monkeypatch,tmp_path):
    provider,actions,_,rhs,_=case
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.direct_operator.LocalRWGRiesz',_forbid)
        with pytest.raises(TypeError):DirectionalMultipoleOperator(provider,actions.exterior.far)
        with pytest.raises(ValueError):DirectionalMultipoleOperator(object(),actions)
    op=DirectionalMultipoleOperator(provider,actions)
    trips=[]
    for kind in ('near_budget','context_keys','owned_modes','coupling_budget','body_order','system_budget','dag_links','owned_phase_mode'):
        current=torch.ones_like(rhs,requires_grad=True);out=actions.adjoint(current)
        if kind=='near_budget':obj,name,new=actions.interiors[0],'near_retained_budget_bytes',1
        elif kind=='context_keys':obj,name,new=actions.exterior.test,'context_keys',actions.exterior.test.context_keys[::-1]
        elif kind=='owned_modes':
            tensor=actions.interiors[0].test.carriers;tensor.requires_grad_(True)
            with pytest.raises(ValueError):torch.autograd.grad(out,current,torch.ones_like(out))
            tensor.requires_grad_(False);actions.assert_current();trips.append(kind);continue
        elif kind=='coupling_budget':obj,name,new=actions.interiors[1].far.couplings[0],'ideal_rwg_error_budget',.9
        elif kind=='dag_links':obj,name,new=actions.interiors[0].test,'dag_links',actions.interiors[0].test.dag_links[::-1]
        elif kind=='owned_phase_mode':
            tensor=next(iter(actions.interiors[1].test.transfers.values())).source_phase;tensor.requires_grad_(True)
            with pytest.raises(ValueError):torch.autograd.grad(out,current,torch.ones_like(out))
            tensor.requires_grad_(False);actions.assert_current();trips.append(kind);continue
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


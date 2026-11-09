"""Disjoint global physical FAR components, exact ownership and current adjoints."""
from dataclasses import replace
import itertools
import json
import math
import random
import weakref
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_block
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_directional_rwg import (
    DirectionalFFTElectromagneticM2L, DirectionalNestedFarActions)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _case(lossy=False, order=6):
    physical = order>=6
    medium = compile_medium(mw.Material(eps_r=2., mu_r=1.3,
        sigma_e=.01 if lossy else 0., sigma_m=.02 if lossy else 0.), 3e8)
    bases = [RWGBasis(sphere_mesh(radius=.02, center=(x,0.,0.), subdivisions=0,
        device='cuda', dtype=torch.float64)) for x in (-.5,.5)]
    system = SurfaceSystem.create([(i,str(i),basis,medium) for i,basis in enumerate(bases)])
    provider = ScalarSystemBlocks(system, medium, IntegrationConfig(order=3, singular_order=3))
    lower, upper = provider.support_boxes()
    tree = build_support_octree(lower, upper, leaf_size=2)
    minus = lower.new_tensor([-1.,0.,0.])
    left = NestedDirectionalRWGProjection(provider, tree, minus, medium.k, order,
        quadrature_order=12 if physical else 4, face_batch=4, rhs_batch=4)
    right = NestedDirectionalRWGProjection(provider, tree, -minus, medium.k, order,
        quadrature_order=12 if physical else 4, face_batch=4, rhs_batch=4, layout=left.layout)
    # Integer physical body IDs select complete nonroot subtrees. Both source
    # children share target rows; the opposite direction uses the other carrier.
    permutation = tree.permutation.tolist()
    selected = []
    for body in system.bodies:
        begin, end = body.edge_offset, body.edge_offset+body.basis.n_edges
        selected.append(next(n for n,children in enumerate(left.children)
            if n>0 and len(children)>=2 and all(begin<=edge<end
                for edge in permutation[left.starts[n]:left.stops[n]])))
    a,b = selected
    source_children = left.children[b][:2]
    records = [(left,left,a,c) for c in source_children]+[(right,right,b,a)]
    couplings = tuple(DirectionalFFTElectromagneticM2L.from_nested_nodes(test,i,source,j,
        medium, lower.new_tensor(.02), kernel_error_budget=KernelErrorBudget(
            1e-6 if physical else 1., 1e-4 if physical else 1.),
        ideal_rwg_error_budget=1e-4 if physical else 1., balancing_impedance=abs(medium.eta))
        for test,source,i,j in records)
    workspace_budget = 512*1024**2 if physical else 128*1024**2
    print(dict(stage='explicit_far_workspace_preflight',order=order,
        tree_nodes=tree.node_count,owner_map_workspace_bytes=[p.planned_tensor_workspace_bytes for p in (left,right)],
        coupling_workspace_bytes=[c.planned_tensor_workspace_bytes for c in couplings],
        fullnode_counts=sum(left.node_counts),configured_action_workspace_budget_bytes=workspace_budget),flush=True)
    actions = DirectionalNestedFarActions(left,left,medium,couplings,
        ideal_rwg_error_budget=1e-4 if physical else 1., rhs_batch=4,
        tensor_workspace_budget_bytes=workspace_budget)
    return actions


def _relative(x,y):
    return float(((x-y).norm()/y.norm()).detach())


def _original_global_far_block(provider,rows,cols,medium):
    nr,nc = len(rows),len(cols)
    original = torch.zeros((2*nr,2*nc),device='cuda',dtype=torch.complex128)
    for i,target in enumerate(provider.system.bodies):
        ri = ((rows>=target.edge_offset)&(rows<target.edge_offset+target.basis.n_edges)).nonzero().flatten()
        if not len(ri):continue
        for j,source in enumerate(provider.system.bodies):
            ci = ((cols>=source.edge_offset)&(cols<source.edge_offset+source.basis.n_edges)).nonzero().flatten()
            if not len(ci):continue
            block,_ = scalar_regional_block(target.basis,source.basis,rows[ri]-target.edge_offset,
                cols[ci]-source.edge_offset,(medium,),IntegrationConfig(order=16,singular_order=3),
                geometry=(provider.geometry[i],provider.geometry[j]))
            rr,cc = torch.cat((ri,ri+nr)),torch.cat((ci,ci+nc))
            original[rr[:,None],cc] = block
    return original


@pytest.mark.parametrize('lossy', (False,True))
def test_multiple_carrier_global_original_far_weak_H_current_and_higher(lossy,tmp_path):
    actions = _case(lossy,order=8)
    n = actions.test.tree.size
    reference = torch.zeros(actions.shape, device='cuda', dtype=torch.complex128)
    active_rows, active_cols = [], []
    for coupling, rectangle in zip(actions.couplings,actions.rectangles,strict=True):
        target, origin = coupling._rwg_context
        start, stop, low, high, _, _ = rectangle
        rows, cols = target.tree.permutation[start:stop], origin.tree.permutation[low:high]
        row_ids, col_ids = torch.cat((rows,rows+n)), torch.cat((cols,cols+n))
        original = _original_global_far_block(target.basis,rows,cols,actions.medium)
        reference[row_ids[:,None],col_ids] = original
        active_rows.append(row_ids);active_cols.append(col_ids)
    columns = torch.cat(active_cols).unique()
    rows = torch.cat(active_rows).unique()
    actual = torch.zeros_like(reference)
    actual[:,columns] = actions.apply(torch.eye(2*n,device='cuda',dtype=torch.complex128)[columns]).mT
    eta = actions.balancing_impedance
    left = actual.real.new_tensor([eta]*n+[1.]*n)
    right = actual.real.new_tensor([1.]*n+[eta]*n)
    absolute = (left[:,None]*(actual-reference)*right).norm()
    weak = float(absolute/(left[:,None]*reference*right).norm())
    aggregate = actions.ideal_rwg_truncation_bound()
    blocks = [c.ideal_rwg_truncation_bound() for c in actions.couplings]
    with mp.workdps(100):
        expected_log = float(mp.log(mp.sqrt(sum(mp.exp(2*mp.mpf(b['log_absolute_bound'])) for b in blocks))))
        expected_bound = float(mp.sqrt(sum(mp.mpf(b['absolute_bound'])**2 for b in blocks)))
    assert aggregate['log_absolute_bound']==pytest.approx(expected_log,rel=1e-11,abs=0)
    assert aggregate['absolute_bound']==pytest.approx(expected_bound,rel=1e-11,abs=0)
    assert not (aggregate['absolute_bound']*1.01==pytest.approx(expected_bound,rel=1e-11,abs=0))
    inactive_rows = torch.ones(2*n,device='cuda',dtype=torch.bool);inactive_rows[rows] = False
    assert torch.equal(actual[inactive_rows],torch.zeros_like(actual[inactive_rows]))
    diagnostic = dict(lossy=lossy,phase='original_weak_computed',rectangles=actions.rectangles,
        balancing_impedance=eta,original_weak_relative=weak,original_weak_absolute=float(absolute),
        aggregate_expected_bound=expected_bound,aggregate_expected_log=expected_log,
        block_bounds=blocks,receipt=actions.receipt(),production_qualification=False)
    (tmp_path/'weak_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    torch.save(dict(actual=actual,reference=reference),tmp_path/'weak_diagnostic.pt')
    print(diagnostic,flush=True)
    assert weak<1e-8 and float(absolute)<=aggregate['absolute_bound']
    generator = torch.Generator(device='cuda').manual_seed(2437)
    current = torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,
        generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        forward = actions.apply(current)
    assert not saved and not current.is_contiguous()
    H = actions.adjoint(dual)
    derivative, = torch.autograd.grad(forward,current,dual,retain_graph=True)
    differentiable_dual = dual.detach().requires_grad_()
    H_differentiable = actions.adjoint(differentiable_dual)
    H_derivative, = torch.autograd.grad(H_differentiable,differentiable_dual,current.detach())
    first, = torch.autograd.grad(forward.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator)
    second, = torch.autograd.grad(first,current,tangent)
    lhs, rhs = (forward.conj()*dual).sum(), (current.conj()*H).sum()
    errors = dict(original_H=_relative(H,dual@reference.conj()),
        native_H=_relative(H,dual@actual.conj()),native_forward=_relative(forward,current@actual.mT),
        current=_relative(derivative,H),adjoint_current=_relative(H_derivative,actions.apply(current.detach())),
        higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    # All omitted physical source columns must contribute zero, rather than
    # silently leaking through the full global source/receiver moment caches.
    inactive = current.detach().clone();inactive[...,columns] = 0
    assert torch.equal(actions.apply(inactive),torch.zeros_like(inactive))
    assert torch.equal(forward[...,inactive_rows],torch.zeros_like(forward[...,inactive_rows]))
    diagnostic.update(phase='current_checks_done',errors=errors,physical_rhs=15,
        selected_original_columns=len(columns),saved_internal_tensors=len(saved),
        independent_relative_comparison_absolute_tolerance=0,omitted_columns_and_rows_exact_zero=True)
    (tmp_path/'far.json').write_text(json.dumps(diagnostic,indent=2)+'\n');print(diagnostic,flush=True)
    assert errors['original_H']<1e-8
    assert max(v for k,v in errors.items() if k!='original_H')<1e-11
    assert len(actions.targets)==len(actions.origins)==2
    assert actions.tensors()==() and actions.retained_bytes==actions.planned_retained_bytes==0
    assert not aggregate['implemented_operator_error_certified']


def test_half_open_overlap_sweep_matches_independent_integer_cell_oracle(tmp_path):
    rectangles = [(a,b,c,d,0,0) for a in range(3) for b in range(a+1,4)
        for c in range(3) for d in range(c+1,4)]
    randomizer = random.Random(2441)
    cases = [(a,b) for a,b in itertools.combinations_with_replacement(rectangles,2)]
    cases.extend(tuple(randomizer.choices(rectangles,k=randomizer.randrange(3,9))) for _ in range(100))
    for records in cases:
        cells = [set(itertools.product(range(a,b),range(c,d))) for a,b,c,d,_,_ in records]
        overlap = any(x & y for x,y in itertools.combinations(cells,2))
        if overlap:
            with pytest.raises(ValueError,match='overlap or duplicate'):
                DirectionalNestedFarActions._validate_disjoint_rectangles(records)
        else:
            DirectionalNestedFarActions._validate_disjoint_rectangles(records)
    for bad in ((0,0,1,2,0,0),(-1,2,1,2,0,0),(0,2,3,3,0,0)):
        with pytest.raises(ValueError,match='nonempty half-open'):
            DirectionalNestedFarActions._validate_disjoint_rectangles((bad,))
    (tmp_path/'sweep.json').write_text(json.dumps(dict(independent_integer_cell_cases=len(cases),
        adjacent_half_open_boundaries_checked=True,no_physical_pair_mask_in_production=True),indent=2)+'\n')


def test_cache_lifetime_one_map_per_physical_chunk_and_empty_far(monkeypatch,tmp_path):
    actions = _case(True,order=2)
    counts, max_rhs = {}, 0
    refs = dict(up=[],down=[])
    # Check every old generated cache at the beginning of the next physical
    # chunk, before a new owner map has allocated another full-tree product.
    for adjoint in (False,True):
        inputs, outputs = (actions.targets,actions.origins) if adjoint else (actions.origins,actions.targets)
        up_name, down_name = ('receiver_upward','source_downward') if adjoint else ('source_upward','receiver_downward')
        with monkeypatch.context() as patch:
            first_owner = inputs[0]
            for owner in inputs:
                native = getattr(owner,up_name)
                def upward(currents,native=native,owner=owner):
                    nonlocal max_rhs
                    if owner is first_owner:
                        assert all(ref() is None for ref in refs['up']+refs['down'])
                        refs['up'].clear();refs['down'].clear()
                    assert currents.shape[0]<=actions.rhs_batch
                    max_rhs = max(max_rhs,currents.shape[0])
                    key=(adjoint,up_name,id(owner));counts[key]=counts.get(key,0)+1
                    values = native(currents)
                    refs['up'].extend(weakref.ref(t) for t in values)
                    return values
                patch.setattr(owner,up_name,upward)
            for owner in outputs:
                native = getattr(owner,down_name)
                def downward(values,native=native,owner=owner):
                    key=(adjoint,down_name,id(owner));counts[key]=counts.get(key,0)+1
                    refs['down'].extend(weakref.ref(t) for t in values)
                    return native(values)
                patch.setattr(owner,down_name,downward)
            currents = torch.ones((3,5,actions.shape[0]),device='cuda',dtype=torch.complex128).transpose(0,1)
            result = actions.adjoint(currents) if adjoint else actions.apply(currents)
            assert result.shape==currents.shape
        assert all(ref() is None for ref in refs['up']+refs['down'])
        refs['up'].clear();refs['down'].clear()
    assert len(counts)==8 and set(counts.values())=={4} and max_rhs==actions.rhs_batch
    assert actions.tensors()==() and actions.retained_bytes==0
    empty = DirectionalNestedFarActions(actions.test,actions.source,actions.medium,(),ideal_rwg_error_budget=1.)
    assert empty.balancing_impedance==abs(actions.medium.eta)
    assert empty.ideal_rwg_truncation_bound()['absolute_bound']==0
    assert empty.ideal_rwg_truncation_bound()['log_absolute_bound'] is None
    def forbidden(*args,**kwargs):raise AssertionError('Empty FAR component reached caller maps or couplings')
    with monkeypatch.context() as patch:
        for owner in (*actions.targets,*actions.origins):
            for name in ('source_upward','receiver_upward','source_downward','receiver_downward'):
                patch.setattr(owner,name,forbidden)
        for method in (empty.apply,empty.adjoint):
            assert torch.equal(method(currents),torch.zeros_like(currents))
            zero = method(currents[:0]);assert zero.shape==currents[:0].shape
            scalar = method(currents[0,0]);assert scalar.shape==(actions.shape[0],) and not bool(scalar.any())
            with pytest.raises(ValueError,match='finite'):method(currents*float('nan'))
        empty_input = currents[:0].requires_grad_()
        value = empty.apply(empty_input)
        derivative, = torch.autograd.grad(value,empty_input,torch.ones_like(value))
        assert derivative.shape==empty_input.shape
    (tmp_path/'lifetime.json').write_text(json.dumps(dict(physical_rhs=15,rhs_batch=actions.rhs_batch,
        owner_passes_per_direction=list(counts.values()),max_physical_rhs=max_rhs,
        previous_chunk_and_completed_action_generated_caches_released=True,
        caller_storage_excluded=True,new_owned_bytes=0,empty_far_and_empty_rhs_checked=True,
        empty_default_impedance=empty.balancing_impedance,production_qualification=False),indent=2)+'\n')


def test_partition_admission_preflight_semantic_external_and_saved_backward_guards(monkeypatch,tmp_path):
    actions = _case(order=2)
    prototype, medium = actions.test, actions.medium
    args = dict(ideal_rwg_error_budget=1.,rhs_batch=4)
    def construct(couplings=actions.couplings,**changes):
        options = dict(args);options.update(changes)
        return DirectionalNestedFarActions(prototype,prototype,medium,couplings,**options)
    def forbidden(*a,**kw):raise AssertionError('Rejected FAR admission reached log tensors or current maps')
    equivalent_medium = compile_medium(mw.Material(eps_r=2.,mu_r=1.3),3e8)
    with monkeypatch.context() as patch:
        patch.setattr(torch,'tensor',forbidden)
        for owner in (*actions.targets,*actions.origins):patch.setattr(owner,'source_upward',forbidden)
        with pytest.raises(MemoryError,match='workspace'):
            construct(tensor_workspace_budget_bytes=actions.planned_tensor_workspace_bytes-1)
        with pytest.raises(ValueError,match='overlap or duplicate'):
            construct(couplings=(*actions.couplings,actions.couplings[0]))
        with pytest.raises(ValueError,match='common medium and balancing'):
            construct(balancing_impedance=actions.balancing_impedance*1.01)
        with pytest.raises(ValueError,match='common medium and balancing'):
            DirectionalNestedFarActions(prototype,prototype,equivalent_medium,actions.couplings,**args)
        with pytest.raises(TypeError,match='admitted prescribed-node'):construct(couplings=(object(),))
    with monkeypatch.context() as patch:
        for owner in (*actions.targets,*actions.origins):patch.setattr(owner,'source_upward',forbidden)
        with pytest.raises(ValueError,match='regional absolute budget'):
            construct(ideal_rwg_error_budget=actions.ideal_rwg_truncation_bound()['absolute_bound']/2)
    # Same coordinates/storage do not establish tree or physical basis identity.
    alien_tree = NestedDirectionalRWGProjection(prototype.basis,replace(prototype.tree),prototype.direction,
        medium.k,2,quadrature_order=4,layout=prototype.layout)
    alien_provider = ScalarSystemBlocks(prototype.basis.system,medium,prototype.basis.config)
    alien_basis = NestedDirectionalRWGProjection(alien_provider,prototype.tree,prototype.direction,
        medium.k,2,quadrature_order=4,layout=prototype.layout)
    for alien in (alien_tree,alien_basis):
        with pytest.raises(ValueError,match='basis and tree identities'):
            DirectionalNestedFarActions(alien,prototype,medium,actions.couplings,**args)
    unadmitted = DirectionalFFTElectromagneticM2L(prototype.centers.new_tensor([-1.,0.,0.]),
        prototype.direction,medium,prototype.centers.new_tensor(.02),0)
    with pytest.raises(ValueError,match='common medium and balancing'):construct(couplings=(unadmitted,))
    for name,value in (('rhs_batch',True),('ideal_rwg_error_budget',0),('balancing_impedance',float('nan'))):
        with pytest.raises(ValueError):construct(**{name:value})
    current = torch.ones((2,actions.shape[1]),device='cuda',dtype=torch.complex128,requires_grad=True)
    mutations = dict(rhs_batch=3,tensor_workspace_budget_bytes=actions.tensor_workspace_budget_bytes+1,
        planned_tensor_workspace_bytes=actions.planned_tensor_workspace_bytes+1,
        ideal_rwg_error_budget=.9,balancing_impedance=actions.balancing_impedance*.9,
        rectangles=actions.rectangles[::-1],couplings=actions.couplings[::-1],targets=actions.targets[::-1],
        origins=actions.origins[::-1],shape=(1,1),retained_bytes=1,planned_retained_bytes=1,
        _ideal_bound={**actions._ideal_bound,'absolute_bound':0})
    for name,new in mutations.items():
        old = getattr(actions,name);output = actions.apply(current);setattr(actions,name,new)
        with pytest.raises(ValueError,match='changed'):actions.receipt()
        with pytest.raises(ValueError,match='changed'):
            torch.autograd.grad(output,current,torch.ones_like(output))
        setattr(actions,name,old);actions.assert_current()
    coupling = actions.couplings[0]
    old = coupling._rwg_bound;output = actions.apply(current)
    coupling._rwg_bound = {**old,'test_node':0}
    with pytest.raises(ValueError,match='changed'):actions.apply(current)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,torch.ones_like(output))
    coupling._rwg_bound = old;actions.assert_current()
    for method in (actions.apply,actions.adjoint):
        with pytest.raises(ValueError,match='complex128'):method(current.real)
        with pytest.raises(ValueError,match='complex128'):method(current[...,:-1])
        with pytest.raises(ValueError,match='finite'):method(current.detach()*float('nan'))
        assert method(current[:0]).shape==current[:0].shape
    # A controlled failed caller map must not escape as a successful FAR output.
    # This checks the composition boundary, not an unspecified overflow input.
    for method,owners,name in ((actions.apply,actions.targets,'receiver_downward'),
                               (actions.adjoint,actions.origins,'source_downward')):
        with monkeypatch.context() as patch:
            for owner in owners:
                native = getattr(owner,name)
                patch.setattr(owner,name,lambda values,native=native:
                    torch.full_like(native(values),complex(math.inf,0)))
            with pytest.raises(ValueError,match='native precision'):method(current.detach())
    output = actions.apply(current)
    transfer = next(iter(actions.origins[-1].transfers.values()))
    transfer.tensors()[0].add_(0)
    with pytest.raises(ValueError,match='changed'):actions.receipt()
    with pytest.raises(ValueError,match='changed'):
        torch.autograd.grad(output,current,torch.ones_like(output))
    (tmp_path/'guards.json').write_text(json.dumps(dict(semantic_mutations=list(mutations),
        workspace_and_overlap_before_log_or_current_buffers=True,
        aggregate_budget_before_current_actions=True,caller_basis_tree_medium_identity_required=True,
        prescribed_node_receipt_and_external_transfer_saved_backward_guard=True,
        nonfinite_input_and_injected_failed_map_output_rejected=True,production_qualification=False),indent=2)+'\n')

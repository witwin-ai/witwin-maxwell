"""Original scalar reciprocity, compact native near actions and lifecycle guards."""
import json
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks, scalar_regional_channels
from witwin.maxwell.mom.packed_near import PackedReciprocalScalarNearChannels
from witwin.maxwell.mom.direct_capacity import storage_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _relative(actual,reference):
    return float(((actual-reference).norm()/reference.norm()).detach())


def _layout(parts,n):
    return tuple((a,b+len(parts),torch.cat((rows,rows+n)),torch.cat((cols,cols+n))+2*n)
        for a,rows in enumerate(parts) for b,cols in enumerate(parts))


@pytest.fixture(scope='module')
def problems():
    bodies = []
    for i,x in enumerate((-.3,.3)):
        vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
            device='cuda',dtype=torch.float64)*.025
        vertices[:,0] += x
        faces = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
        basis = RWGBasis(SurfaceMesh(vertices,faces))
        medium = compile_medium(mw.Material(eps_r=2.+i,mu_r=1.4+.2*i,sigma_e=.01,sigma_m=.02),3e8)
        bodies.append((i,str(i),basis,medium))
    config = IntegrationConfig(order=3,singular_order=3,pair_batch=4,far_pair_batch=8)
    provider = ScalarSystemBlocks(SurfaceSystem.create(bodies),compile_medium(mw.Material(),3e8),config)
    n = provider.system.total_edges
    assert n==12
    parts = [torch.tensor(v,device='cuda') for v in ((7,2,10,0,5),(1,8,4,11),(6,9,3))]
    global_layout = _layout(parts,n)
    def exterior(a,b,rows,cols):
        return provider.exterior_channels(rows[:len(rows)//2],cols[:len(cols)//2]-2*n)
    basis,medium = provider.system.bodies[0].basis,provider.system.bodies[0].interior
    interior_n = basis.n_edges
    parts = [torch.tensor(v,device='cuda') for v in ((2,5),(1,4),(0,3))]
    interior_layout = _layout(parts,interior_n)
    def interior(a,b,rows,cols):
        d,k,_ = scalar_regional_channels(basis,basis,rows[:len(rows)//2],cols[:len(cols)//2]-2*interior_n,
            medium,config,geometry=(provider.geometry[0],provider.geometry[0]))
        return d,k
    return dict(exterior=(global_layout,n,exterior,provider.exterior,config),
        magnetic_interior=(interior_layout,interior_n,interior,medium,config))


def _construct(problem,factory=None,budget=None):
    layout,n,original,medium,config = problem
    return PackedReciprocalScalarNearChannels.from_layout(layout,2*n,2*n,
        original if factory is None else factory,source_node_offset=3,medium=medium,
        integration=config,device='cuda',dtype=torch.complex128,action_batch=2,
        retained_budget_bytes=budget)


def _expected(problem):
    layout,n,_,_,_ = problem
    canonical = [record for record in layout if record[0]<=record[1]-3]
    return 8*(2*len({a for a,_,_,_ in canonical})+1+3*len(canonical))+sum(
        32*(len(rows)//2)*(len(cols)//2)+8*(len(rows)+len(cols)) for _,_,rows,cols in canonical)


@pytest.fixture(scope='module')
def operators(problems):
    result = {}
    for label,problem in problems.items():
        layout,n,factory,medium,_ = problem
        calls = []
        def counted(*args):
            calls.append(args[:2])
            return factory(*args)
        op = _construct(problem,counted,_expected(problem))
        assert len(calls)==6
        original = torch.zeros((4*n,4*n),device='cuda',dtype=torch.complex128)
        for a,b,rows,cols in layout:
            d,k = factory(a,b,rows,cols)
            block = torch.cat((torch.cat((k,op.electric*d),1),torch.cat((op.magnetic*d,k),1)),0)
            original[rows[:,None],cols[None,:]] = block
        expanded = torch.zeros_like(original)
        for rows,cols,fields,transposed in op._action_groups():
            for rr,cc,(d,k) in zip(rows,cols,fields,strict=True):
                if transposed:d,k = d.mT,k.mT
                block = torch.cat((torch.cat((k,op.electric*d),1),torch.cat((op.magnetic*d,k),1)),0)
                expanded[rr[:,None],cc[None,:]] = block
        result[label] = op,original,expanded,calls
    return result


@pytest.mark.parametrize('label',('exterior','magnetic_interior'))
def test_original_reciprocal_variable_blocks_H_current_higher_and_storage(label,problems,operators,tmp_path):
    op,original,expanded,calls = operators[label]
    _,n,_,_,_ = problems[label]
    assert op.receipt()['logical_blocks']==9 and op.receipt()['stored_blocks']==6
    assert storage_bytes(op.tensors())==op._retained_bytes==_expected(problems[label])
    assert all(t._base is None and t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in op.tensors())
    assert len({t.untyped_storage().data_ptr() for t in op.tensors()})==len(op.tensors())
    eta = abs(op.medium.eta)
    left = original.real.new_tensor([1.]*n+[eta]*n+[1.]*n+[eta]*n)
    right = original.real.new_tensor([eta]*n+[1.]*n+[eta]*n+[1.]*n)
    weak = _relative(left[:,None]*expanded*right,left[:,None]*original*right)
    assert weak<1e-8
    generator = torch.Generator(device='cuda').manual_seed(410523)
    current = torch.randn((3,5,op.size),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(current.shape,device='cuda',dtype=op.dtype,generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        forward = op.apply(current)
    assert not saved and not current.is_contiguous()
    H = op.adjoint(dual)
    gradient, = torch.autograd.grad(forward,current,dual,retain_graph=True)
    H_gradient, = torch.autograd.grad(H,dual,current,retain_graph=True)
    first, = torch.autograd.grad(forward.abs().square().sum(),current,create_graph=True)
    tangent = torch.randn_like(current)
    second, = torch.autograd.grad(first,current,tangent)
    lhs,rhs = (forward.conj()*dual).sum(),(current.conj()*H).sum()
    errors = dict(forward=_relative(forward,current@expanded.mT),H=_relative(H,dual@expanded.conj()),
        gradient=_relative(gradient,H),adjoint_gradient=_relative(H_gradient,forward),
        higher=_relative(second,2*op.adjoint(op.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    original_H = _relative(H,dual@original.conj())
    assert original_H<1e-8 and max(errors.values())<1e-11
    values = dict(case=label,physical_edges=n,logical_blocks=9,stored_blocks=6,
        factory_calls=calls,original_balanced_weak_relative=weak,original_H_relative=original_H,
        native_errors=errors,saved_internal_tensors=len(saved),retained_bytes=op._retained_bytes,
        rectangular_blocks=label=='exterior',diagonal_and_reverse_same_shape_group=label=='magnetic_interior',
        reverse_values_not_conjugated=True,full_system_physics_and_resources_qualified=False,qualification=False)
    (tmp_path/'reciprocal.json').write_text(json.dumps(values,indent=2)+'\n');print(values,flush=True)


def test_reciprocal_preallocation_and_complete_reverse_mapping_rejections(problems,tmp_path):
    problem = problems['exterior'];layout,n,_,medium,config = problem
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('Rejected reciprocal layout reached original integration')
    with pytest.raises(MemoryError,match='before allocation'):
        _construct(problem,forbidden,_expected(problem)-1)
    bad = list(layout);bad.pop(1)
    with pytest.raises(ValueError,match='every reverse pair'):
        _construct((bad,n,forbidden,medium,config))
    bad = list(layout);a,b,rows,cols = bad[1]
    bad[1] = a,b,torch.cat((rows[:len(rows)//2].flip(0),rows[len(rows)//2:].flip(0))),cols
    with pytest.raises(ValueError,match='physical node maps|reverse maps'):
        _construct((bad,n,forbidden,medium,config))
    # Without diagonals, namespace-local checks miss inconsistent node1 maps
    # in node0/node1 and node1/node2 pairs, even though each reverse matches.
    bad = [layout[i] for i in (1,3,5,7)]
    a,b,rows,cols = bad[2]
    bad[2] = a,b,torch.cat((rows[:len(rows)//2].flip(0),rows[len(rows)//2:].flip(0))),cols
    a,b,rows,cols = bad[3]
    bad[3] = a,b,rows,torch.cat((cols[:len(cols)//2].flip(0),cols[len(cols)//2:].flip(0)))
    with pytest.raises(ValueError,match='physical node maps'):
        _construct((bad,n,forbidden,medium,config))
    with pytest.raises(ValueError,match='unique aligned'):
        _construct(((*layout,layout[0]),n,forbidden,medium,config))
    for offset in (0,True,1.5):
        with pytest.raises(ValueError):
            PackedReciprocalScalarNearChannels.from_layout(layout,2*n,2*n,forbidden,
                source_node_offset=offset,medium=medium,integration=config,device='cuda',dtype=torch.complex128)
    assert not calls
    (tmp_path/'capacity.json').write_text(json.dumps(dict(one_byte_below_exact_preflight_rejected=True,
        missing_reverse_rejected=True,reverse_JM_order_mismatch_rejected=True,duplicate_pair_rejected=True,
        pairwise_consistent_but_global_node_map_mismatch_rejected=True,
        strict_offset_validation=True,original_integration_calls=0,qualification=False),indent=2)+'\n')


def test_saved_reciprocal_policy_tensor_identity_mode_and_value_guards(operators,tmp_path):
    op,_,_,_ = operators['magnetic_interior']
    current = torch.ones(op.size,device='cuda',dtype=op.dtype,requires_grad=True)
    def rejected():
        output = op.apply(current)
        return output
    checked = []
    for field,changed in (('_reciprocal_node_offset',4),('_logical_blocks',10),
            ('_reverse_group_counts',(0,)),('action_batch',3)):
        output = rejected();original = getattr(op,field);setattr(op,field,changed)
        for check in (op.receipt,lambda:op.adjoint(current),lambda:torch.autograd.grad(output,current,torch.ones_like(output))):
            with pytest.raises(ValueError,match='changed'):check()
        setattr(op,field,original);op.assert_current();checked.append(field)
    output = rejected();original = op.groups
    op.groups = ((original[0][0],original[0][1],original[0][2].clone()),*original[1:])
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,torch.ones_like(output))
    op.groups = original;op.assert_current();checked.append('field_identity')
    output = rejected();fields = op.groups[0][2];fields.requires_grad_(True)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,torch.ones_like(output))
    fields.requires_grad_(False);op.assert_current();checked.append('field_requires_grad_mode')
    output = rejected();op.groups[0][2].add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(output,current,torch.ones_like(output))
    checked.append('field_version')
    (tmp_path/'guards.json').write_text(json.dumps(dict(saved_current_guards=checked,qualification=False),indent=2)+'\n')

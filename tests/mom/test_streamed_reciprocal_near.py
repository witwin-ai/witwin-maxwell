"""New exact recomputation policy: original columns, native H and owner guards."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.operator import IntegrationConfig, PreparedRWGGeometry
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import scalar_regional_block, scalar_regional_channels
from witwin.maxwell.mom.packed_near import StreamedReciprocalScalarNearChannels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _relative(actual, expected):
    return float(((actual-expected).norm()/expected.norm()).detach())


def _layout(groups, edges):
    maps = [torch.cat((g,g+edges)) for g in groups]
    return [(a,b+len(groups),ra,cb+2*edges)
            for a,ra in enumerate(maps) for b,cb in enumerate(maps)]


def test_new_irregular_tetra_all_original_columns_native_H_and_two_current_derivatives(tmp_path):
    vertices = torch.tensor([[.003,.001,-.002],[.026,.002,.001],
                             [.005,.031,.002],[.004,.006,.033]], device='cuda',dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=17)
    basis = RWGBasis(volume.boundary); assert basis.n_edges==6
    geometry = PreparedRWGGeometry(basis)
    medium = compile_medium(mw.Material(eps_r=2.35,mu_r=1.47,sigma_e=.006,sigma_m=.013),137e6)
    integration = IntegrationConfig(order=4,singular_order=3,pair_batch=16,far_pair_batch=32)
    edges = torch.arange(6,device='cuda')
    reference, metadata = scalar_regional_block(basis,basis,edges,edges,(medium,),
        integration,geometry=(geometry,geometry))
    saved_path = tmp_path/'streamed-original-reference.pt'
    torch.save(dict(reference=reference,metadata=metadata),saved_path)
    calls=[]
    def factory(a,b,rows,cols):
        calls.append((a,b))
        d,k,_ = scalar_regional_channels(basis,basis,rows[:len(rows)//2],
            cols[:len(cols)//2]-12,medium,integration,geometry=(geometry,geometry))
        return d,k
    groups = [torch.tensor(g,device='cuda') for g in ([5],[1,3],[4,0,2])]
    layout = _layout(groups,6)
    # Maps require584B; the old retained D/K arrays require another800B.
    expected_maps = 8*(2*3+1+3*6)+sum(16*(len(a)+len(b)) for a,b in
        ((groups[i],groups[j]) for i in range(3) for j in range(i,3)))
    expected_coefficients = sum(32*len(groups[i])*len(groups[j])
                                for i in range(3) for j in range(i,3))
    operator = StreamedReciprocalScalarNearChannels.from_layout(layout,12,12,factory,
        source_node_offset=3,medium=medium,integration=integration,device='cuda',dtype=torch.complex128,
        physical_guard=lambda:geometry.assert_current(basis),
        physical_inputs=(*geometry.inputs(),*geometry.tensors()), action_batch=4,
        coefficient_workspace_budget_bytes=3*2*3*3*16,retained_budget_bytes=expected_maps)
    assert calls==[] and operator.receipt()['retained_bytes']==expected_maps
    assert sum(t.untyped_storage().nbytes() for t in operator.tensors())==expected_maps
    def apply(value,adjoint=False):
        zero=torch.zeros_like(value)
        embedded=torch.cat((value,zero) if adjoint else (zero,value),dim=-1)
        result=operator.action(embedded,adjoint=adjoint)
        return result[...,12:] if adjoint else result[...,:12]
    columns=apply(torch.eye(12,device='cuda',dtype=torch.complex128)).mT
    assert len(calls)==6
    generator=torch.Generator(device='cuda').manual_seed(13837)
    current=torch.randn((3,5,12),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    dual=torch.randn(current.shape,device='cuda',dtype=torch.complex128,generator=generator)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        output=apply(current)
    assert saved==[]
    native_H=apply(dual,True)
    grad,=torch.autograd.grad(output,current,dual)
    norm_output=apply(current)
    first,=torch.autograd.grad(norm_output.abs().square().sum(),current,create_graph=True)
    second,=torch.autograd.grad(first,current,dual)
    expected_second=2*(dual@columns.mT)@columns.conj()
    lhs=(output.conj()*dual).sum();rhs=(current.conj()*native_H).sum()
    errors=dict(columns=_relative(columns,reference),forward=_relative(output,current@reference.mT),
        original_H=_relative(native_H,dual@reference.conj()),
        native_H=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()),
        current_gradient=_relative(grad,native_H),higher_current_gradient=_relative(second,expected_second))
    gates={k:(1e-8 if k in ('columns','forward','original_H') else 1e-11) for k in errors}
    torch.save(dict(reference=reference,native_columns=columns,original_metadata=metadata,
        current=current.detach(),dual=dual,native_output=output.detach(),native_H=native_H.detach()),saved_path)
    record=dict(completed=True,errors=errors,gates=gates,logical_blocks=9,stored_canonical_blocks=6,
        constructor_factory_calls=0,factory_calls=len(calls),retained_maps_bytes=expected_maps,
        omitted_retained_original_coefficients_bytes=expected_coefficients,
        current_forward_saved_tensors=len(saved),receipt=operator.receipt(),
        full_original_reference_saved_before_actions=True,physical_qualification=False,production_qualification=False)
    (tmp_path/'streamed-original.json').write_text(json.dumps(record,indent=2)+'\n')
    assert all(errors[k]<gates[k] for k in errors),errors
    assert operator.receipt()['retained_kernel_bytes']==0


def _synthetic(factory=None,**kwargs):
    physical=torch.tensor([.41,.23],device='cuda',dtype=torch.float64)
    calls=[]
    def good(a,b,rows,cols):
        calls.append((a,b))
        n,m=len(rows)//2,len(cols)//2
        return (torch.full((n,m),.7+.2j,device='cuda',dtype=torch.complex128),
                torch.full((n,m),.3-.1j,device='cuda',dtype=torch.complex128))
    groups=[torch.tensor([0],device='cuda'),torch.tensor([1,2],device='cuda')]
    medium=compile_medium(mw.Material(eps_r=1.8,mu_r=1.21),121e6)
    defaults=dict(source_node_offset=2,medium=medium,integration=IntegrationConfig(),
        device='cuda',dtype=torch.complex128,physical_guard=lambda:None,
        physical_inputs=(physical,),action_batch=3,coefficient_workspace_budget_bytes=384)
    defaults.update(kwargs)
    op=StreamedReciprocalScalarNearChannels.from_layout(_layout(groups,3),6,6,
        good if factory is None else factory,**defaults)
    return op,physical,calls


@pytest.mark.parametrize('mutation',('physical_version','physical_grad','map_grad','factory','workspace'))
def test_new_streamed_owner_changes_refuse_before_factory(mutation):
    op,physical,calls=_synthetic()
    if mutation=='physical_version':physical.add_(.01)
    elif mutation=='physical_grad':physical.requires_grad_()
    elif mutation=='map_grad':op.groups[0][2].requires_grad_()
    elif mutation=='factory':op._channel_factory=lambda *args:None
    else:op.coefficient_workspace_budget_bytes+=1
    with pytest.raises(ValueError):op.action(torch.zeros(12,device='cuda',dtype=torch.complex128))
    assert calls==[]


@pytest.mark.parametrize('failure',('missing_guard','workspace_too_small','retained_too_small'))
def test_new_streamed_setup_refuses_without_original_integration(failure):
    options={'missing_guard':{'physical_guard':None},
             'workspace_too_small':{'coefficient_workspace_budget_bytes':383},
             'retained_too_small':{'retained_budget_bytes':1}}[failure]
    with pytest.raises((TypeError,MemoryError)):_synthetic(**options)


@pytest.mark.parametrize('failure',('nonfinite','gradient','oversized_backing'))
def test_new_streamed_factory_fields_refuse(failure):
    def factory(a,b,rows,cols):
        n,m=len(rows)//2,len(cols)//2
        d=torch.ones((n,m),device='cuda',dtype=torch.complex128)
        k=d.clone()
        if failure=='nonfinite':k[0,0]=complex(float('nan'),0)
        elif failure=='gradient':d.requires_grad_()
        else:d=torch.ones((n+4,m+4),device='cuda',dtype=torch.complex128)[:n,:m]
        return d,k
    op,_,_=_synthetic(factory)
    with pytest.raises(ValueError):op.action(torch.ones(12,device='cuda',dtype=torch.complex128))


def test_new_same_shape_coefficient_chunks_shrink_without_reintegrating_reverse(monkeypatch,tmp_path):
    groups=[torch.tensor(g,device='cuda') for g in ([4,1],[5,0],[3,2])]
    layout=_layout(groups,6)
    medium=compile_medium(mw.Material(eps_r=2.1,mu_r=1.31),149e6)
    physical=torch.ones(1,device='cuda',dtype=torch.float64)
    calls=[]
    def factory(a,b,rows,cols):
        calls.append((a,b))
        r=rows[:2].to(torch.float64)+1
        c=(cols[:2]-12).to(torch.float64)+1
        return ((r[:,None]*c[None]+.2j*(r[:,None]+c[None])).to(torch.complex128),
                (.3*(r[:,None]+c[None])+.1j*r[:,None]*c[None]).to(torch.complex128))
    operator=StreamedReciprocalScalarNearChannels.from_layout(layout,12,12,factory,
        source_node_offset=3,medium=medium,integration=IntegrationConfig(),device='cuda',
        dtype=torch.complex128,physical_guard=lambda:None,physical_inputs=(physical,),
        action_batch=8,coefficient_workspace_budget_bytes=384)
    assert len(operator.groups)==1 and calls==[]
    observed=[]
    original=operator._action_groups
    def observed_groups():
        for rows,cols,fields,transposed in original():
            observed.append(dict(entries=len(fields),transposed=transposed,
                three_array_bytes=3*fields.numel()*fields.element_size()))
            yield rows,cols,fields,transposed
    monkeypatch.setattr(operator,'_action_groups',observed_groups)
    # Independent closed-form synthetic kernels; this is a chunk-policy oracle,
    # not another electromagnetic reference or an original-physics replay.
    r=torch.arange(1,7,device='cuda',dtype=torch.float64)
    d=r[:,None]*r[None]+.2j*(r[:,None]+r[None])
    k=.3*(r[:,None]+r[None])+.1j*r[:,None]*r[None]
    a,b=1j*medium.omega*medium.epsilon,-1j*medium.omega*medium.mu
    reference=torch.cat((torch.cat((k,a*d),dim=1),torch.cat((b*d,k),dim=1)),dim=0)
    eye=torch.eye(12,device='cuda',dtype=torch.complex128)
    embedded=torch.cat((torch.zeros_like(eye),eye),dim=1)
    native=operator.apply(embedded)[:,:12].mT
    error=_relative(native,reference)
    record=dict(completed=True,coefficient_budget_bytes=384,requested_action_batch=8,
        actual_chunk_entries=[d['entries'] for d in observed],chunks=observed,
        canonical_factory_calls=calls,logical_block_count=9,relative_error=error,
        gate=1e-11,synthetic_scope_only=True,production_qualification=False)
    (tmp_path/'same-shape-chunks.json').write_text(json.dumps(record,indent=2)+'\n')
    assert len(calls)==6 and len(set(calls))==6
    assert len(observed)==9 and all(d['entries']==1 and d['three_array_bytes']==384 for d in observed)
    assert error<1e-11

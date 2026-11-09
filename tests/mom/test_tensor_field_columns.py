"""Shared tensor multi-RHS fields, scalar reduction and current derivatives."""
from dataclasses import replace
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom.anisotropic_fields import tensor_current_potentials,tensor_current_potential_columns
from witwin.maxwell.mom.fields import decode_interior_columns

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_shared_columns_and_current_gradient_with_owned_interior_decoder(monkeypatch):
    op,binding=setup();handle=op.body_handles[0];kernel=handle.tensor_kernel
    basis=handle.basis
    generator=torch.Generator(device='cuda').manual_seed(891)
    c=torch.randn(2*basis.n_edges,3,device='cuda',dtype=torch.complex128,generator=generator,requires_grad=True)
    points=binding.volume.vertices.new_tensor([[.013,.017,.011],[.3,.2,.4]])
    calls=[];original=type(kernel).evaluate_regular
    def measured(self,delta):
        if self is kernel:calls.append(len(delta))
        return original(self,delta)
    monkeypatch.setattr(type(kernel),'evaluate_regular',measured)
    e,h,r=tensor_current_potential_columns(basis,c,points,kernel,order=4,pair_batch=2,query_batch=1)
    shared=list(calls);calls.clear()
    expected=[tensor_current_potentials(basis,c[:,i],points,kernel,order=4,pair_batch=2,query_batch=1) for i in range(3)]
    assert len(calls)==3*len(shared) and sum(calls)==3*sum(shared)
    reference=[torch.stack([x[k] for x in expected],-1) for k in range(2)]
    for actual,ref in zip((e,h),reference):torch.testing.assert_close(actual,ref,rtol=2e-12,atol=1e-10)
    gradient,=torch.autograd.grad(e.abs().square().sum()+h.abs().square().sum(),c)
    ref_gradient,=torch.autograd.grad(sum(x.abs().square().sum() for x in reference),c)
    torch.testing.assert_close(gradient,ref_gradient,rtol=2e-11,atol=1e-8)
    assert r['shared_column_kernel'] and r['column_count']==3 and not r['physical_qualification']
    handle.integration=replace(handle.integration,order=4,singular_order=4)
    interior=decode_interior_columns(handle,c.detach(),points[:1])
    for actual,ref in zip(interior,(e[:1],h[:1])):torch.testing.assert_close(actual,-ref,rtol=2e-12,atol=1e-10)
    assert handle.last_column_field_receipt['shared_column_kernel']
    with pytest.raises(ValueError,match='requested region'):
        decode_interior_columns(handle,c,points[1:])
    saved=handle.tensor_kernel;handle.tensor_kernel=None
    with pytest.raises(ValueError,match='owning regional'):
        decode_interior_columns(handle,c,points[:1])
    handle.tensor_kernel=saved
    empty=tensor_current_potential_columns(basis,c,points[:0],kernel)
    assert empty[0].shape==(0,3,3)
    with pytest.raises(ValueError,match='trace_side'):
        tensor_current_potential_columns(basis,c,basis.mesh.triangles[:1].mean(1),kernel)
    broken=c.detach().clone();broken[0,2]=complex(float('nan'),0)
    with pytest.raises(ValueError,match='finite'):
        tensor_current_potential_columns(basis,broken,points,kernel)
    with pytest.raises(ValueError,match='columns'):
        tensor_current_potential_columns(basis,c[:,:0],points,kernel)


def test_actual_anisotropic_columns_match_independent_column_calls():
    op,binding=setup(anisotropic=True);handle=op.body_handles[0]
    basis=handle.basis
    c=torch.zeros(2*basis.n_edges,2,device='cuda',dtype=torch.complex128)
    c[0,0]=1.;c[basis.n_edges+1,1]=.01j
    points=binding.volume.vertices.new_tensor([[.013,.017,.011]])
    actual=tensor_current_potential_columns(basis,c,points,handle.tensor_kernel,order=2)
    expected=[tensor_current_potentials(basis,c[:,i],points,handle.tensor_kernel,order=2) for i in range(2)]
    for k in range(2):
        torch.testing.assert_close(actual[k],torch.stack([x[k] for x in expected],-1),rtol=2e-11,atol=1e-10)
    assert all(x['kernel']['converged'] for x in actual[2]['batches'])
    assert not actual[2]['physical_qualification']

"""Borrowed action protocol against one pinned closed original native reference."""
import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_compact import (CompactElectromagneticM2L,
    SignedAxisCompactElectromagneticM2L,signed_axis_compact_moment_actions)
from witwin.maxwell.mom.scalar_moment_action import scalar_moment_action
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_new_borrowed_protocol_actions_and_guard_reuse_closed_reference(tmp_path,monkeypatch):
    source=os.environ.get('GSP_SIGNED_AXIS_REFERENCE')
    digest=os.environ.get('GSP_SIGNED_AXIS_REFERENCE_SHA256')
    if source is None and digest is None:pytest.skip('Requires the pinned closed original native reference')
    assert source and digest and hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    # Declare the original physical/numerical controls before opening evidence.
    m=compile_medium(mw.Material(eps_r=2.6+.017j,mu_r=1.41,sigma_m=.009),2.11e8)
    order,row_batch=4,7
    r=torch.tensor([.4375,-.28125,.15625],device='cuda',dtype=torch.float64)
    axes,signs=(2,0,1),(-1,1,-1)
    transforms=tuple((p,s) for p in itertools.permutations(range(3)) for s in itertools.product((-1,1),repeat=3))
    owner_index=transforms.index(((0,1,2),(1,1,1)));reference_index=transforms.index((axes,signs))
    saved=torch.load(source,map_location='cuda',weights_only=True)
    assert saved['order']==order and saved['frequency_Hz']==2.11e8
    torch.testing.assert_close(saved['displacements'][owner_index],r,rtol=0,atol=0)
    torch.testing.assert_close(saved['displacements'][reference_index],r[list(axes)]*r.new_tensor(signs),rtol=0,atol=0)
    import witwin.maxwell.mom.fmm_compact as module
    def forbidden(*args,**kwargs):raise AssertionError('Borrowed protocol regenerated reference or owner coefficients')
    monkeypatch.setattr(module,'helmholtz_taylor',forbidden)
    owner=CompactElectromagneticM2L.__new__(CompactElectromagneticM2L)
    owner._bind_coefficients(saved['original_independent_coefficients'][owner_index].clone(),m,order,row_batch)
    view=SignedAxisCompactElectromagneticM2L(owner,axes,signs)
    with pytest.raises(TypeError):SignedAxisCompactElectromagneticM2L(view,axes,signs)
    assert storage_bytes([*owner.tensors(),*view.tensors()])==owner.retained_bytes
    assert view.receipt()['additional_owned_tensor_bytes']==0
    green=saved['green'][reference_index];gradient=saved['gradient'][reference_index]
    generator=torch.Generator(device='cuda').manual_seed(22417)
    x=torch.randn((2,3,2,owner.count,4),device='cuda',dtype=torch.complex128,generator=generator).transpose(0,1).requires_grad_()
    y=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
    direction=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=generator)
    relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
    ef=scalar_moment_action(green,gradient,m,x.detach())
    eh=scalar_moment_action(green,gradient,m,y,adjoint=True)
    captured=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:captured.append(t) or t,lambda t:t):f=view.apply(x)
    assert captured==[]
    h=view.adjoint(y);derivative=torch.autograd.grad(f,x,y)[0]
    first=torch.autograd.grad(view.apply(x).abs().square().sum(),x,create_graph=True)[0]
    second=torch.autograd.grad(first,x,direction)[0]
    expected_second=2*scalar_moment_action(green,gradient,m,scalar_moment_action(green,gradient,m,direction),adjoint=True)
    bx=x.detach()[:2,0,None];by=y[:2,0,None]
    with torch.no_grad():bf=signed_axis_compact_moment_actions([view],bx);bh=signed_axis_compact_moment_actions([view],by,adjoint=True)
    errors=dict(forward=relative(f,ef),original_H=relative(h,eh),current=relative(derivative,eh),
        higher=relative(second,expected_second),batch_F=relative(bf,ef[:2,0,None]),batch_H=relative(bh,eh[:2,0,None]))
    after=view.apply(x);owner.coefficients.add_(0)
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(after,x,torch.ones_like(after))
    assert max(errors.values())<1e-11
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    (tmp_path/'borrowed-protocol.json').write_text(json.dumps(dict(completed=True,errors=errors,
        reference_sha256=digest,no_reference_coefficient_regeneration=True,
        new_RHS_seed=22417,saved_forward_tensors=0,after_forward_mutation_refused=True,
        additional_owned_view_bytes=0,original_eleven_PASS_not_replayed=True,
        public_or_f30_qualified=False,production_qualification=False),indent=2)+'\n')

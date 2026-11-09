"""Bounded compact coefficient construction and exact retained ownership."""
import pytest
import weakref
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fmm_compact import CompactElectromagneticM2L
from witwin.maxwell.mom.fmm_expansion import multipole_to_local
from witwin.maxwell.mom.fmm_storage import m2l_storage_bytes
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.fmm_capacity import storage_bytes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
def test_compact_batch_coefficients_actual_tables_action_gradient_and_ownership(dtype,monkeypatch):
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,sigma_m=.002),3e8)
    d=torch.tensor([[1.,.2,-.1],[.8,-.3,.1],[1.5,.1,-.2],[.7,-.1,.2]],device='cuda',dtype=dtype)
    order=6
    batch=CompactElectromagneticM2L.create_many(d,medium,order,row_batch=11)
    single=tuple(CompactElectromagneticM2L(v,medium,order,row_batch=11) for v in d)
    tol=4e-5 if dtype==torch.float32 else 2e-13
    for obj,ref,center in zip(batch,single,d,strict=True):
        torch.testing.assert_close(obj.coefficients,ref.coefficients,rtol=tol,atol=0)
        assert obj.coefficients.untyped_storage().nbytes()==obj.coefficients.numel()*obj.coefficients.element_size()
        assert obj.coefficients._base is None
        dense_g,dense_h=multipole_to_local(center,medium.k,order)
        g,h=zip(*(obj._tables(start,min(start+11,obj.count)) for start in range(0,obj.count,11)))
        torch.testing.assert_close(torch.cat(g),dense_g,rtol=tol,atol=tol)
        torch.testing.assert_close(torch.cat(h).movedim(-1,0),dense_h,rtol=tol,atol=tol)
        x=torch.randn((2,2,obj.count,4),device=d.device,dtype=obj.dtype,requires_grad=True)
        y=torch.randn_like(x)
        torch.testing.assert_close(obj.apply(x),ref.apply(x),rtol=tol,atol=tol)
        grad,=torch.autograd.grad(obj.apply(x),x,y)
        torch.testing.assert_close(grad,obj.adjoint(y),rtol=tol,atol=tol)
    expected=len(d)*m2l_storage_bytes(order,d.element_size(),'compact')
    assert storage_bytes(t for obj in batch for t in obj.tensors())==expected
    assert len({obj.coefficients.untyped_storage().data_ptr() for obj in batch})==len(batch)
    versions=[tuple((id(t),t._version) for t in obj.tensors()) for obj in batch]
    batch[0].coefficients.add_(1)
    with pytest.raises(ValueError,match='changed'):batch[0].assert_current()
    for index,obj in enumerate(batch[1:],start=1):
        obj.assert_current();assert tuple((id(t),t._version) for t in obj.tensors())==versions[index]


def test_compact_coefficient_batch32_workspace_and_preallocation(monkeypatch):
    import witwin.maxwell.mom.fmm_compact as module
    medium=compile_medium(mw.Material(),3e8)
    d=torch.tensor([[1.,.2,-.1]]*65,device='cuda',dtype=torch.float64)
    primitive=module.helmholtz_taylor;calls=[];batch_outputs=[]
    def observed(points,*a,**k):
        assert all(ref() is None for ref in batch_outputs)
        calls.append(tuple(points.shape))
        result=primitive(points,*a,**k);batch_outputs.append(weakref.ref(result))
        return result
    monkeypatch.setattr(module,'helmholtz_taylor',observed)
    objects=CompactElectromagneticM2L.create_many(d,medium,2,row_batch=7)
    assert calls==[(32,3),(32,3),(1,3)]
    assert len(objects)==65
    assert all(ref() is None for ref in batch_outputs)
    assert sum(obj.retained_bytes for obj in objects)==65*m2l_storage_bytes(2,8,'compact')
    calls.clear()
    with pytest.raises(MemoryError,match='before coefficient'):
        CompactElectromagneticM2L.create_many(d,medium,2,retained_budget_bytes=1)
    assert calls==[]
    assert CompactElectromagneticM2L.create_many(d[:0],medium,2)==()
    assert calls==[]
    for value in (d[0],d.cpu(),d.clone().requires_grad_(),d*torch.nan):
        with pytest.raises(ValueError):CompactElectromagneticM2L.create_many(value,medium,2)
    for change in ({'order':True},{'row_batch':False},{'retained_budget_bytes':0}):
        with pytest.raises(ValueError):CompactElectromagneticM2L.create_many(d,medium,**{'order':2,**change})

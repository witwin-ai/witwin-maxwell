"""Selected tensor carrier rows: real scalar limit and isolated pair routing."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_anisotropic_lifting import handle_for
from witwin.maxwell.mom import anisotropic_lifting as module
from witwin.maxwell.mom.tensor_medium import TensorMedium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_selected_tensor_rhs_scalar_limit_and_order():
    handle=handle_for(2.)
    ids=handle.basis.face_edges.new_tensor([4,0,2])
    full,_=module.build_tensor_lifted_rhs(handle)
    selected,receipt=module.build_tensor_lifted_rhs(handle,test_edges=ids)
    expected=full[torch.cat((ids,ids+handle.basis.n_edges))]
    torch.testing.assert_close(selected,expected,rtol=2e-12,atol=1e-12)
    assert not receipt['global_pair_classification_retained']
    assert receipt['selected_edges']==3
    for bad in (ids[:0],ids.new_tensor([0,0]),ids.new_tensor([-1])):
        with pytest.raises(ValueError):module.build_tensor_lifted_rhs(handle,test_edges=bad)


def test_selected_tensor_generic_pair_routing_is_bounded(monkeypatch):
    # Synthetic pair contributions test routing only, not electromagnetic
    # quadrature or physical accuracy. Actual scalar-limit integration is above.
    handle=handle_for(2.);eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    handle.interior=TensorMedium(handle.exterior.frequency,torch.diag(eye.new_tensor([2.,2.5,3.])),eye)
    handle.integration=replace(handle.integration,pair_batch=3,far_pair_batch=5)
    exterior=TensorMedium(handle.exterior.frequency,eye,eye)
    contrast=SimpleNamespace(exterior=exterior,interior=handle.interior)
    def pairs(handle,rows,cols,category,order,kernel):
        value=(1+rows[:,None,None]*7+cols[:,None,None]*3+
               torch.arange(2,device='cuda')[None,:,None]*2+
               torch.arange(3,device='cuda')[None,None,:]).to(torch.complex128)
        return value,{'category':category,'order':order}
    monkeypatch.setattr(module,'_pair_rhs',pairs)
    full,_=module.build_tensor_lifted_rhs(handle,contrast_kernel=contrast)
    def forbidden(*args,**kwargs):raise AssertionError('Selected rows cannot classify global pairs')
    monkeypatch.setattr(module,'classify_all_pairs',forbidden)
    ids=handle.basis.face_edges.new_tensor([4,0,2])
    value,receipt=module.build_tensor_lifted_rhs(handle,contrast_kernel=contrast,test_edges=ids)
    torch.testing.assert_close(value,full[torch.cat((ids,ids+handle.basis.n_edges))],rtol=0,atol=0)
    assert receipt['maximum_classification_pairs']<=5
    assert not receipt['global_pair_classification_retained']
    assert sum(receipt['rhs_triangle_pair_counts'])<len(handle.basis.mesh.faces)**2

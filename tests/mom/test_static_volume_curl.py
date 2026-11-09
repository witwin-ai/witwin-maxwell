"""Global multi-body static curls preserve native integration and transposes."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.vie_operator import cell_green_potentials
from witwin.maxwell.mom.static_volume_curl import StaticVolumeCurlAction

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_blocked_global_curl_selected_cells_and_rhs_gradients(monkeypatch):
    first=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    second=TetrahedralVolumeMesh(first.vertices[:4]+first.vertices.new_tensor([.23,.02,-.01]),
        torch.tensor([[0,1,2,3]],device='cuda'),region_id=2)
    volumes=(second,first)
    points=torch.cat([v.vertices[v.cells].mean(1) for v in volumes])
    basis=torch.eye(3,device='cuda',dtype=torch.float64)
    parts=[]
    for volume in volumes:
        _,gradient,_=cell_green_potentials(volume,points,0.,order=2)
        # Independent cross-product construction in source-vector basis order.
        cross=torch.linalg.cross(gradient.real[...,None,:].expand(-1,-1,3,-1),
            basis[None,None].expand(len(points),len(volume.cells),-1,-1),dim=-1)
        parts.append(cross.permute(0,3,1,2).reshape(3*len(points),-1))
    matrix=torch.cat(parts,dim=1)
    assert float(matrix[:3,3:].norm())>0 and float(matrix[3:,:3].norm())>0
    rhs=torch.arange(30,device='cuda',dtype=torch.float64).reshape(2,15).cos().requires_grad_()
    expected=rhs@matrix.T
    grad_expected=2*expected@matrix
    second_expected=2*torch.ones_like(rhs)@matrix.T@matrix
    import witwin.maxwell.mom.static_volume_curl as module
    original=module.cell_green_potentials;calls=[]
    def selected(mesh,targets,k,**kwargs):
        assert len(targets)<=2 and len(kwargs['cell_indices'])<=2
        assert kwargs['include_hessian'] is False
        calls.append((mesh.region_id,len(targets),len(kwargs['cell_indices'])))
        return original(mesh,targets,k,**kwargs)
    monkeypatch.setattr(module,'cell_green_potentials',selected)
    action=StaticVolumeCurlAction(volumes,target_batch=2,source_batch=2)
    amax,amin=torch.Tensor.amax,torch.Tensor.amin
    def guarded_max(value,*args,**kwargs):
        if any(value is v.vertices for v in volumes):raise AssertionError('Repeated complete-mesh extent scan')
        return amax(value,*args,**kwargs)
    def guarded_min(value,*args,**kwargs):
        if any(value is v.vertices for v in volumes):raise AssertionError('Repeated complete-mesh extent scan')
        return amin(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'amax',guarded_max)
    monkeypatch.setattr(torch.Tensor,'amin',guarded_min)
    actual=action.apply(rhs)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-14)
    torch.testing.assert_close(action.adjoint(rhs),rhs@matrix,rtol=2e-12,atol=2e-14)
    gradient,=torch.autograd.grad(actual.square().sum(),rhs,create_graph=True)
    torch.testing.assert_close(gradient,grad_expected,rtol=2e-11,atol=2e-14)
    derivative,=torch.autograd.grad(gradient.sum(),rhs)
    torch.testing.assert_close(derivative,second_expected,rtol=2e-11,atol=2e-14)
    assert {region for region,_,_ in calls}=={1,2}
    assert action.apply(rhs[:0]).shape==rhs[:0].shape
    assert not action.receipt['global_curl_stored'] and not action.receipt['fast_work_qualified']
    delayed=action.apply(rhs)
    action.target_batch=1
    with pytest.raises(ValueError,match='binding changed'):torch.autograd.grad(delayed.sum(),rhs)


def test_curl_geometry_and_policy_guards():
    volume=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    with pytest.raises(ValueError,match='positive integers'):StaticVolumeCurlAction([volume],target_batch=True)
    with pytest.raises(ValueError,match='unique regions'):StaticVolumeCurlAction([volume,volume])
    action=StaticVolumeCurlAction([volume])
    volume.vertices.add_(.01)
    with pytest.raises(ValueError,match='geometry changed'):action.apply(torch.zeros(12,device='cuda',dtype=torch.float64))


def test_prepared_extent_preserves_full_mesh_trace_semantics():
    from witwin.maxwell.mom.vie_operator import PreparedVolumeGreenGeometry
    volume=runpy.run_path('tests/mom/test_static_current_projector.py')['block']().volume
    prepared=PreparedVolumeGreenGeometry(volume)
    points=volume.vertices[volume.cells].mean(1)
    indices=torch.tensor([3,0,3],device='cuda')
    expected=cell_green_potentials(volume,points,0.,cell_indices=indices)
    actual=cell_green_potentials(volume,points,0.,cell_indices=indices,geometry=prepared)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    face=volume.vertices[volume.cells[0,:3]].mean(0,keepdim=True)
    for geometry in (None,prepared):
        with pytest.raises(ValueError,match='cell-face'):
            cell_green_potentials(volume,face,0.,cell_indices=indices,geometry=geometry)
    prepared.extent.add_(.01)
    with pytest.raises(ValueError,match='geometry changed'):
        cell_green_potentials(volume,points,0.,geometry=prepared)

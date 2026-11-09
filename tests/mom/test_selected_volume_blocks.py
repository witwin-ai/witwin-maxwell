"""Selected volume blocks preserve self terms, ordering and native gradients."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom import vie_operator as vie
from witwin.maxwell.mom.physics import compile_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def volume():
    return runpy.run_path('tests/mom/test_volume_green.py')['mesh']().refine(1)


@pytest.mark.parametrize('k',[0.,7+.4j])
def test_selected_potentials_and_query_gradient(k,monkeypatch):
    mesh=volume()
    points=mesh.vertices[mesh.cells[[2,0]]].mean(1).requires_grad_()
    ids=mesh.cells.new_tensor([6,2,6,0])
    full=vie.cell_green_potentials(mesh,points,k,order=5)
    expected=tuple(v[:,ids] for v in full)
    actual=vie.cell_green_potentials(mesh,points,k,order=5,cell_indices=ids)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-12)
    gradients=[torch.autograd.grad(sum(v.real.square().sum() for v in values),points)[0] for values in (actual,expected)]
    torch.testing.assert_close(*gradients,rtol=2e-11,atol=1e-9)
    seen=[];original=vie._static_panel_gradient
    def capture(observers,panels,**kwargs):
        seen.append(len(panels));return original(observers,panels,**kwargs)
    monkeypatch.setattr(vie,'_static_panel_gradient',capture)
    vie.cell_green_potentials(mesh,points[:1],k,order=5,cell_indices=ids[:1])
    assert sum(seen)==4


def test_selected_field_blocks_scatter_action_adjoint_and_gradient():
    mesh=volume();medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3),3e8)
    points=mesh.vertices[mesh.cells[[1,5]]].mean(1)
    ids=mesh.cells.new_tensor([7,1,7])
    e,h=vie.cell_current_field_blocks(mesh,points,medium,cell_indices=ids,order=5)
    assert e.shape==h.shape==(2,3,3,3)
    torch.manual_seed(91)
    current=torch.randn(3,3,device='cuda',dtype=torch.complex128,requires_grad=True)
    full=current.new_zeros(len(mesh.cells),3).index_add(0,ids,current)
    expected=vie.cell_current_fields(mesh,full,points,medium,order=5)
    actual=tuple(block.reshape(6,9)@current.flatten() for block in (e,h))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b.flatten(),rtol=2e-12,atol=1e-12)
    gradients=[torch.autograd.grad(sum(v.abs().square().sum() for v in values),current)[0] for values in (actual,expected)]
    torch.testing.assert_close(*gradients,rtol=2e-12,atol=1e-10)
    for block,action in zip((e,h),actual):
        dual=torch.randn(6,device='cuda',dtype=torch.complex128)
        torch.testing.assert_close(torch.vdot(dual,action),torch.vdot(block.reshape(6,9).mH@dual,current.flatten()),rtol=1e-12,atol=1e-12)


def test_selection_validation_empty_and_trace_guards():
    mesh=volume();point=mesh.vertices[mesh.cells[:1]].mean(1)
    for ids in ([0],mesh.cells.new_tensor([-1]),mesh.cells.new_tensor([8]),torch.tensor([0]),mesh.cells[:1],mesh.vertices.new_tensor([0])):
        with pytest.raises(ValueError,match='Selected volume cells'):
            vie.cell_green_potentials(mesh,point,0.,cell_indices=ids)
    empty=mesh.cells.new_empty(0)
    outputs=vie.cell_green_potentials(mesh,point,0.,cell_indices=empty)
    assert [v.shape for v in outputs]==[(1,0),(1,0,3),(1,0,3,3)]
    assert vie.cell_green_potentials(mesh,point[:0],0.,cell_indices=empty,include_hessian=False)[2] is None
    with pytest.raises(ValueError,match='edge'):
        vie.cell_green_potentials(mesh,mesh.vertices[mesh.cells[0,:1]],0.,cell_indices=mesh.cells.new_tensor([0]),principal_value=True)

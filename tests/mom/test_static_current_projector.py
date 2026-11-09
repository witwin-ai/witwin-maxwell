"""Weighted conservative projector and recovery without global arrays."""
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.static_current_projector import StaticCurrentProjector,CurrentProjectionAdmissionError
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def block():
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1],[.013,.011,.017]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]],device='cuda')
    return FaceCurrentBalance(TetrahedralVolumeMesh(vertices,cells,region_id=1))


def test_weighted_projection_recovery_adjoint_and_gradients(monkeypatch):
    native=block();volume=native.volume
    gradient=volume.vertices.new_zeros(len(volume.cells),3,native.n_dofs)
    gradient.scatter_(2,native.cell_dofs[:,None,:].expand(-1,3,-1),native.gradients.transpose(1,2))
    gradient=gradient.reshape(-1,native.n_dofs)[:,1:]
    weights=volume.volumes.repeat_interleave(3)
    weak=gradient.T*weights
    recovery=torch.linalg.solve(weak@gradient,weak)
    matrix=torch.eye(len(gradient),device='cuda',dtype=torch.float64)-gradient@recovery
    assert float((matrix-matrix.T).norm())>.01
    rhs=torch.arange(72,device='cuda',dtype=torch.float64).reshape(2,3,12).sin().requires_grad_()
    expected=rhs@matrix.T
    expected_gradient=2*expected@matrix
    expected_second=2*torch.ones_like(rhs)@matrix.T@matrix
    def forbidden(*args,**kwargs):raise AssertionError('Global projection/stiffness solve forbidden')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(native,'_matrix',forbidden)
    projector=StaticCurrentProjector(native,StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-20,maxiter=100,restart=20),column_batch=3)
    actual=projector.apply(rhs)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(projector.adjoint(rhs),rhs@matrix,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(projector.recover(rhs),rhs@recovery.T,rtol=1e-10,atol=1e-11)
    potential=projector.recover(rhs)
    torch.testing.assert_close(projector.recover_adjoint(potential),potential@recovery,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(projector.apply(actual),actual,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(projector.adjoint(rhs*weights),actual*weights,rtol=1e-10,atol=1e-13)
    for row in actual.detach().reshape(-1,4,3):assert float(native.weak_current(row).norm())<1e-12
    g,=torch.autograd.grad(actual.square().sum(),rhs,create_graph=True)
    torch.testing.assert_close(g,expected_gradient,rtol=1e-9,atol=1e-10)
    gg,=torch.autograd.grad(g.sum(),rhs)
    torch.testing.assert_close(gg,expected_second,rtol=1e-9,atol=1e-10)
    rg,=torch.autograd.grad(potential.square().sum(),rhs)
    torch.testing.assert_close(rg,2*(rhs@recovery.T)@recovery,rtol=1e-9,atol=1e-11)
    assert projector.apply(rhs[:0]).shape==rhs[:0].shape
    assert projector.recover(rhs[:0]).shape==(0,3,native.n_dofs-1)
    assert not projector.receipt['global_projector_stored'] and not projector.receipt['physical_qualification']
    delayed=projector.apply(rhs)
    with torch.no_grad():projector.inverse.tangent.tangent[0,0,0]+=1
    with pytest.raises(ValueError,match='changed'):torch.autograd.grad(delayed.sum(),rhs)


def test_projection_admission_budget_and_geometry_rejection():
    native=block()
    with pytest.raises(CurrentProjectionAdmissionError) as error:
        StaticCurrentProjector(native,StoppingRule(preconditioner='none',rtol=1e-15,atol=1e-25,maxiter=1,restart=1))
    assert not error.value.receipt['admitted'] and error.value.receipt['inconclusive']
    projector=StaticCurrentProjector(native,StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-20,maxiter=100,restart=20))
    native.volume.vertices.add_(.01)
    with pytest.raises(ValueError,match='geometry changed'):projector.apply(torch.zeros(12,device='cuda',dtype=torch.float64))

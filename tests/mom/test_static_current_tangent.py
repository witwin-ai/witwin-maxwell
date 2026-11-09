"""Cellwise DC tangent products against assembled independent contractions."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.static_current import NodalCurrentBalance,FaceCurrentBalance

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('kind',[NodalCurrentBalance,FaceCurrentBalance])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cellwise_reduced_tangent_products_and_gradients(kind,dtype,monkeypatch):
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh'](dtype)
    block=kind(volume)
    generator=torch.Generator(device='cuda').manual_seed(43)
    tensor=torch.randn(len(volume.cells),3,3,device='cuda',dtype=dtype,generator=generator)
    # Nonsymmetric heterogeneous derivatives distinguish forward and transpose.
    matrix=block._matrix(tensor)[1:,1:]
    action=block.tangent_action(tensor)
    value=torch.randn(2,3,block.n_dofs-1,device='cuda',dtype=dtype,generator=generator,requires_grad=True)
    expected=value@matrix.mT
    expected_gradient=2*expected@matrix
    expected_second=2*torch.ones_like(value)@matrix.mT@matrix
    tolerance=5e-5 if dtype==torch.float32 else 2e-12
    def forbidden(*args,**kwargs):raise AssertionError('Global DC matrix or Jacobian forbidden')
    monkeypatch.setattr(block,'_matrix',forbidden)
    monkeypatch.setattr(torch.autograd.functional,'jacobian',forbidden)
    with torch.no_grad():tensor.zero_()
    actual=action.apply(value)
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(action.adjoint(value),value@matrix,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(action.diagonal(),matrix.diagonal(),rtol=tolerance,atol=tolerance)
    gradient,=torch.autograd.grad(actual.square().sum(),value,create_graph=True)
    torch.testing.assert_close(gradient,expected_gradient,rtol=tolerance,atol=tolerance)
    second,=torch.autograd.grad(gradient.sum(),value)
    torch.testing.assert_close(second,expected_second,rtol=tolerance,atol=tolerance)
    assert action.apply(value[:0]).shape==value[:0].shape
    torch.testing.assert_close(action.apply(torch.zeros_like(value)),torch.zeros_like(value),rtol=0,atol=0)
    with pytest.raises(ValueError,match='matching finite'):action.apply(value.to(torch.complex128))
    delayed=action.apply(value)
    with torch.no_grad():action.tangent[0,0,0]+=1
    with pytest.raises(ValueError,match='binding changed'):torch.autograd.grad(delayed.sum(),value)


def test_cellwise_tangent_gauge_and_geometry_guard():
    volume=runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block=FaceCurrentBalance(volume)
    tangent=torch.eye(3,device='cuda',dtype=torch.float64).expand(len(volume.cells),3,3)
    action=block.tangent_action(tangent)
    vector=torch.arange(block.n_dofs-1,device='cuda',dtype=torch.float64)/block.n_dofs
    pinned=torch.cat((vector.new_zeros(1),vector))
    # Direct weak current differentiation has the opposite potential sign.
    expected=-block.weak_current(block.electric(pinned))[1:]
    torch.testing.assert_close(action.apply(vector),expected,rtol=1e-12,atol=1e-14)
    assert float(vector@action.apply(vector))>0
    singular=block.tangent_action(torch.zeros_like(tangent))
    torch.testing.assert_close(singular.apply(vector),torch.zeros_like(vector),rtol=0,atol=0)
    volume.vertices.add_(.01)
    with pytest.raises(ValueError,match='geometry changed'):action.apply(vector)

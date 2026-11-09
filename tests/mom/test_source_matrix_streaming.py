"""Streaming preserves source ownership; unused Hessians are not evaluated."""
import runpy
import pytest
import torch
from witwin.maxwell.mom import vie_operator as vie
from witwin.maxwell.mom.volume_response_matrix import build_volume_source_matrix

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_optional_hessian_preserves_potentials_and_query_derivatives(monkeypatch):
    volume=runpy.run_path('tests/mom/test_volume_green.py')['mesh']()
    points=volume.vertices.new_tensor([[.013,.019,.031],[.2,-.1,.3]],requires_grad=True)
    expected=vie.cell_green_potentials(volume,points,7+.4j,order=8)
    reference,=torch.autograd.grad(sum(v.real.square().sum() for v in expected[:2]),points)
    def forbidden(*args,**kwargs):raise AssertionError('Unused dynamic Hessian must not be evaluated')
    monkeypatch.setattr(vie,'_smooth_background_gradient',forbidden)
    actual=vie.cell_green_potentials(volume,points,7+.4j,order=8,include_hessian=False)
    assert actual[2] is None
    for a,b in zip(actual[:2],expected[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    gradient,=torch.autograd.grad(sum(v.real.square().sum() for v in actual[:2]),points)
    torch.testing.assert_close(gradient,reference,rtol=0,atol=0)
    empty=vie.cell_green_potentials(volume,points[:0],7+.4j,include_hessian=False)
    assert empty[0].shape==(0,1) and empty[1].shape==(0,1,3) and empty[2] is None
    with pytest.raises(TypeError,match='include_hessian'):
        vie.cell_green_potentials(volume,points,7+.4j,include_hessian=0)


def test_streaming_query_chunks_and_reversed_binding_columns():
    operator,bindings=runpy.run_path('tests/mom/test_surface_volume.py')['problem']((2.,3.))
    a,_=build_volume_source_matrix(operator,bindings,order=5,inner_order=6)
    b,receipt=build_volume_source_matrix(operator,bindings[::-1],order=5,inner_order=6,query_batch_size=3)
    torch.testing.assert_close(b,torch.cat((a[:,3:],a[:,:3]),dim=1),rtol=2e-12,atol=1e-13)
    assert not receipt['hessian_evaluated'] and not receipt['physical_qualification']
    assert [body['region_id'] for body in receipt['bodies']]==[2,1]
    assert all(body['query_batch_size']==3 for body in receipt['bodies'])
    for invalid in (0,True,1.5):
        with pytest.raises(ValueError,match='batch size'):
            build_volume_source_matrix(operator,bindings,query_batch_size=invalid)

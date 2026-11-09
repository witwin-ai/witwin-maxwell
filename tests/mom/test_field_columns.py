"""Shared geometric field quadrature preserves independent response columns."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fields import decode,decode_interior_columns,FieldQueryConfig
from witwin.maxwell.mom.mesh import RWGBasis,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.kernels import FREE_SPACE
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_columns_match_individual_fields_and_gradients(dtype):
    basis=RWGBasis(sphere_mesh(radius=.08,subdivisions=0,dtype=dtype))
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3),4e8)
    handle=SimpleNamespace(basis=basis,interior=medium,exterior=medium,kernel=FREE_SPACE,
        sources=(),integration=IntegrationConfig(order=4,singular_order=6))
    complex_dtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    generator=torch.Generator(device='cuda').manual_seed(17)
    coefficients=torch.randn(2*basis.n_edges,7,device='cuda',dtype=complex_dtype,
                              generator=generator,requires_grad=True)
    points=torch.tensor([[0.,0.,0.],[.06,0.,0.],[-.02,.017,-.01]],device='cuda',dtype=dtype)
    actual=decode_interior_columns(handle,coefficients,points)
    individual=[decode(handle,coefficients[:,i],points,1,'total',
                       query_config=FieldQueryConfig(adaptive=False)) for i in range(7)]
    expected=tuple(torch.stack([value[k] for value in individual],dim=-1) for k in range(2))
    rtol,atol=(2e-12,1e-10) if dtype==torch.float64 else (2e-5,1e-4)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    weights=[torch.randn(value.shape,device='cuda',dtype=complex_dtype,generator=generator) for value in actual]
    loss=lambda values:sum((a*w.conj()).real.sum() for a,w in zip(values,weights))
    a,=torch.autograd.grad(loss(actual),coefficients,retain_graph=True)
    b,=torch.autograd.grad(loss(expected),coefficients)
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    assert all(v.shape==(0,3,7) for v in decode_interior_columns(handle,coefficients,points[:0]))
    bad=coefficients.detach().clone();bad[0,-1]=float('nan')
    with pytest.raises(ValueError,match='finite'):decode_interior_columns(handle,bad,points)
    with pytest.raises(ValueError,match='geometry'):
        decode_interior_columns(handle,coefficients,points.new_tensor([[.5,0.,0.]]))
    with pytest.raises(ValueError,match='trace'):
        decode_interior_columns(handle,coefficients,basis.mesh.triangles[:1].mean(1))

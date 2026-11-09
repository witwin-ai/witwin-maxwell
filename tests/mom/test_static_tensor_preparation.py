"""Fixed tensor DC response columns against nonzero direct transmission solves."""
import pytest
import torch
from test_static_tensor_polarization import source
from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
from witwin.maxwell.mom.static_tensor_preparation import prepare_static_tensor_cell_response

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('refine', (0, 1))
def test_tensor_dc_preparation_direct_nonzero_response_and_tangent(refine):
    particular, a, p = source(refine=refine)
    p = (p*torch.linspace(.7,1.4,len(p),device=p.device,dtype=p.dtype)[:,None]).detach().requires_grad_()
    binding = particular.binding
    op = StaticTensorInterfaceOperator(binding.surface_system, torch.eye(3,device=a.device,dtype=a.dtype), (a,))
    prepared = prepare_static_tensor_cell_response(op, (binding,))
    direct = op.solve_uniform(a.new_zeros(3), bindings=(binding,), polarizations=(p,))
    field = direct.fields(prepared['points'], 1)[1]
    for name, expected in (('electric', field.flatten()), ('potential', direct.potential),
                           ('conormal', direct.conormal), ('normal_p', direct.normal_p)):
        torch.testing.assert_close(prepared[name+'_response']@p.flatten(), expected, rtol=2e-11, atol=1e-12)
    tangent, = torch.autograd.grad(field.sum(), p)
    torch.testing.assert_close(prepared['electric_response'].sum(0).reshape_as(p), tangent, rtol=2e-11, atol=1e-12)
    assert prepared['receipt']['physical_qualification'] is False
    with pytest.raises(ValueError, match='budget'):
        prepare_static_tensor_cell_response(op, (binding,), max_response_bytes=1)
    with pytest.raises(ValueError, match='unique'):
        prepare_static_tensor_cell_response(op, (binding,binding))


def test_tensor_dc_preparation_multibody_binding_order_and_cross_response():
    from witwin.maxwell.mom.mesh import SurfaceSystem, RWGBasis
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh, bind_volume_region
    from witwin.maxwell.mom.physics import compile_medium
    import witwin.maxwell as mw
    particular,a,p = source()
    first = particular.binding.volume
    second = TetrahedralVolumeMesh(first.vertices+a.new_tensor([.3,0.,0.]),first.cells,region_id=2)
    system = SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),compile_medium(mw.Material(),3e8))
                                   for i,v in enumerate((first,second))])
    bindings = tuple(bind_volume_region(v,system) for v in (second,first))
    op = StaticTensorInterfaceOperator(system,torch.eye(3,device=a.device,dtype=a.dtype),(a,1.3*a))
    prepared = prepare_static_tensor_cell_response(op,bindings)
    polarization = torch.cat((.4*p,p))
    direct = op.solve_uniform(a.new_zeros(3),bindings=bindings,polarizations=(.4*p,p))
    expected = torch.cat([direct.fields(query[None],region)[1]
                          for query,region in zip(prepared['points'],(2,1))]).flatten()
    torch.testing.assert_close(prepared['electric_response']@polarization.flatten(),expected,rtol=2e-11,atol=1e-12)
    assert prepared['regions']==(2,1)
    assert bool(prepared['electric_response'][:3,3:].abs().max()>1e-5)
    with pytest.raises(ValueError,match='exact surface'):
        prepare_static_tensor_cell_response(op,(particular.binding,))

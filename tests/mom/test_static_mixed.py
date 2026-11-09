"""Mixed static interface limits, charge ownership and native source gradients."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
from witwin.maxwell.mom.static_coupling import StaticVolumeResponse
from witwin.maxwell.mom.static_mixed import MixedStaticOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup():
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    volumes=tuple(mw.TetrahedralVolumeMesh(vertices+vertices.new_tensor([i*.24,0.,0.]),cells,region_id=i+1)
                  for i in range(2))
    system=SurfaceSystem.create([(i,f'body{i}',RWGBasis(v.boundary),compile_medium(mw.Material(eps_r=2+i),3e8))
                                 for i,v in enumerate(volumes)])
    laplace=FloatingPotentialOperator(system,testing_order=4)
    return laplace,tuple(bind_volume_region(v,system) for v in volumes)


def test_mixed_static_reduces_to_existing_dielectric_and_floating_systems():
    laplace,bindings=setup()
    sample=laplace.triangles
    polarization=[sample.new_tensor([[.2,-.1,.3]]),sample.new_tensor([[-.1,.4,.2]])]
    mixed=MixedStaticOperator(laplace,1.,[2.,3.]).solve(bindings,polarization,[])
    dielectric=StaticVolumeResponse(laplace.system,1.,[2.,3.],bindings,polarization,testing_order=4)
    torch.testing.assert_close(mixed.density,dielectric.charge,rtol=1e-12,atol=1e-14)
    boundary=[sample.new_tensor([.2,.1,-.1,.3]),sample.new_tensor([-.4,.2,.5,.1])]
    charges=sample.new_tensor([.003,-.002])
    mixed=MixedStaticOperator(laplace,1.,[None,None]).solve([],[],boundary,net_charges=charges)
    floating=laplace.solve_face_means(torch.cat(boundary),epsilon_exterior=1.,net_charges=charges)
    torch.testing.assert_close(mixed.density,floating.density,rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(mixed.offsets,floating.offsets,rtol=1e-12,atol=1e-14)


def test_mixed_current_dielectric_charge_coupling_and_native_gradient(monkeypatch):
    laplace,bindings=setup()
    sample=laplace.triangles
    amplitude=sample.new_tensor(.7,requires_grad=True)
    boundary=sample.new_tensor([.2,.1,-.1,.3])
    base_p=sample.new_tensor([[.1,-.2,.3]])
    operator=MixedStaticOperator(laplace,1.,[None,3.])
    def forbidden(*args,**kwargs):
        raise AssertionError('Repeated mixed responses must reuse the prepared factorization')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(torch.linalg,'lu_factor',forbidden)
    def solve(value):
        return operator.solve([bindings[1]],[value*base_p],[boundary])
    response=solve(amplitude)
    assert response.current_regions==(1,)
    torch.testing.assert_close(operator.charge_test@response.density,sample.new_zeros(1),rtol=0,atol=1e-16)
    current_faces=laplace.body_faces[0]
    # Dielectric polarization changes the current body's floating voltage.
    zero=solve(sample.new_zeros(()))
    roundoff=512*torch.finfo(sample.dtype).eps*(response.offsets.norm()+zero.offsets.norm())
    assert (response.offsets-zero.offsets).norm()>roundoff
    assert response.density[current_faces:].norm()>0
    point=bindings[1].volume.vertices[bindings[1].volume.cells].mean(1)
    value=response.fields(point,2)[1].square().sum()
    derivative,=torch.autograd.grad(value,amplitude)
    step=1e-5
    difference=(solve(amplitude.detach()+step).fields(point,2)[1].square().sum()
                -solve(amplitude.detach()-step).fields(point,2)[1].square().sum())/(2*step)
    torch.testing.assert_close(derivative,difference,rtol=1e-8,atol=1e-10)
    with pytest.raises(ValueError,match='physical current closure'):
        response.fields(bindings[0].volume.vertices[bindings[0].volume.cells].mean(1),1)
    assert not response.receipt['physical_qualification']
    maps=operator.prepare_cell_response([bindings[1]])
    packed=torch.cat(((amplitude*base_p).flatten(),boundary))
    torch.testing.assert_close(maps['electric_response']@packed,response.fields(point,2)[1].flatten(),rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(maps['surface_response']@packed,response.density,rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(maps['offset_response']@packed,response.offsets,rtol=1e-12,atol=1e-14)
    assert maps['source_regions']==(2,) and maps['current_regions']==(1,)
    assert response.receipt['reused_factorization']
    operator.lu.add_(1.)
    with pytest.raises(RuntimeError,match='operator was mutated'):
        response.fields(point,2)

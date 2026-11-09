"""Floating charge constraints, DC current coupling and exterior sphere oracle."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem,sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_current import NodalCurrentBalance
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def system(meshes):
    medium = compile_medium(mw.Material(eps_r=2.),3e8)
    return SurfaceSystem.create([(i,str(i),RWGBasis(mesh),medium) for i,mesh in enumerate(meshes)])


def tetrahedron():
    vertices = torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    return TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1).refine(1)


def test_current_potential_couples_to_neutral_exterior_with_gauge_and_native_gradient():
    volume = tetrahedron()
    current = NodalCurrentBalance(volume)
    operator = FloatingPotentialOperator(system([volume.boundary]),testing_order=6)
    positive = volume.vertices.new_tensor([[1.4,0.,0.],[.3,0.,0.]]).to(torch.complex128)
    ac = positive[:,None].expand(-1,len(volume.cells),-1)
    sigma = volume.vertices.new_tensor(.2,requires_grad=True)
    points = volume.vertices.new_tensor([[.2,.12,.13],[-.12,.08,.11]])
    def evaluate(conductivity,shift=0.):
        interior = current.solve(ac,conductivity=conductivity,tpa_scale=.7)
        boundary = interior.potential[volume.boundary_vertex_ids]+shift
        exterior = operator.solve([boundary],epsilon_exterior=EPSILON_0)
        return interior,exterior,exterior.fields(points)
    interior,exterior,(potential,electric) = evaluate(sigma)
    assert potential.norm() > 1e-5
    assert not exterior.receipt['physical_interface_charge_recovered']
    assert (operator.charge_test@exterior.density).abs().max() < 1e-15
    _,shifted,shifted_fields = evaluate(sigma,1.7)
    torch.testing.assert_close(shifted.density,exterior.density,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(shifted.offsets,exterior.offsets-1.7,rtol=1e-11,atol=1e-13)
    for a,b in zip((potential,electric),shifted_fields):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    derivative, = torch.autograd.grad(electric.square().sum(),sigma)
    delta = 1e-5
    lower = evaluate(sigma.detach()-delta)[2][1].square().sum()
    upper = evaluate(sigma.detach()+delta)[2][1].square().sum()
    torch.testing.assert_close(derivative,(upper-lower)/(2*delta),rtol=1e-6,atol=1e-12)
    with pytest.raises(ValueError,match='only exterior'):
        exterior.fields(volume.vertices.mean(0,keepdim=True))
    exterior.density.add_(1.)
    with pytest.raises(RuntimeError,match='mutated'):
        exterior.fields(points)


def test_separate_body_charges_fix_offsets_and_couple_through_exterior():
    volume = tetrahedron()
    first = volume.boundary
    second = SurfaceMesh(first.vertices+first.vertices.new_tensor([.35,0.,0.]),first.faces)
    operator = FloatingPotentialOperator(system([first,second]),testing_order=6)
    zero = [mesh.vertices.new_zeros(len(mesh.vertices)) for mesh in (first,second)]
    charge = first.vertices.new_tensor([1.,-1.])*1e-12
    response = operator.solve(zero,epsilon_exterior=EPSILON_0,net_charges=charge)
    torch.testing.assert_close(EPSILON_0*(operator.charge_test@response.density),charge,rtol=1e-12,atol=1e-26)
    assert response.offsets[0] > 0 and response.offsets[1] < 0
    one_charged = operator.solve(zero,epsilon_exterior=EPSILON_0,net_charges=charge.new_tensor([1e-12,0.]))
    # The neutral body's floating voltage is induced by the charged body.
    assert one_charged.offsets[1] > 0
    split = operator.body_faces[0]
    assert one_charged.density[split:].norm() > 0
    neutral = operator.solve(zero,epsilon_exterior=EPSILON_0)
    assert torch.count_nonzero(neutral.density) == 0
    assert torch.count_nonzero(neutral.offsets) == 0


def test_charged_sphere_exterior_refines_toward_coulomb_potential_and_field():
    errors = []
    radius = .1
    for level in (1,2):
        mesh = sphere_mesh(radius=radius,subdivisions=level)
        operator = FloatingPotentialOperator(system([mesh]),testing_order=6)
        charge = mesh.vertices.new_tensor([4*torch.pi*EPSILON_0*radius])
        response = operator.solve([mesh.vertices.new_zeros(len(mesh.vertices))],
                                  epsilon_exterior=EPSILON_0,net_charges=charge)
        points = mesh.vertices.new_tensor([[.16,.04,.03],[-.05,.19,.02],[.02,.03,-.22]])
        potential,electric = response.fields(points)
        distance = points.norm(dim=-1)
        expected_potential = radius/distance
        expected_electric = radius*points/distance[:,None]**3
        voltage_error = abs(float(response.offsets[0])-1.)
        field_error = float((electric-expected_electric).norm()/expected_electric.norm())
        potential_error = float((potential-expected_potential).norm()/expected_potential.norm())
        boundary = response.boundary_fields(interior_displacement=torch.zeros_like(operator.normals))
        torch.testing.assert_close(boundary['potential_V'],response.offsets.expand_as(boundary['potential_V']),
                                   rtol=1e-12,atol=1e-13)
        assert torch.count_nonzero(boundary['bulk_free_charge_C']) == 0
        torch.testing.assert_close(boundary['surface_free_charge_C'],boundary['gauss_total_charge_C'],rtol=0,atol=0)
        gauss_error = float((boundary['gauss_total_charge_C']-charge).abs().max()/charge.abs().max())
        assert gauss_error < .01
        errors.append((voltage_error,potential_error,field_error))
    print({'sphere_relative_errors_voltage_potential_electric':errors,'physical_qualification':False})
    assert all(fine < coarse for coarse,fine in zip(errors[0],errors[1]))

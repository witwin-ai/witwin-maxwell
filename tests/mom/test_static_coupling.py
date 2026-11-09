"""DC interface jumps, matched media and polarized-sphere refinement."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh, bind_volume_region
from witwin.maxwell.mom.static_coupling import StaticVolumeResponse
from witwin.maxwell.mom.vie_operator import cell_static_polarization_fields

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def system_from_volume(volume):
    mesh = SurfaceMesh(volume.boundary.vertices.clone(),volume.boundary.faces.clone())
    system = SurfaceSystem.create([(0,'dielectric',RWGBasis(mesh),compile_medium(mw.Material(eps_r=2.),3e8))])
    return system,bind_volume_region(volume,system)


def test_dc_dielectric_normal_jump_and_matched_medium_reduction():
    vertices = torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    system,binding = system_from_volume(volume)
    polarization = torch.tensor([[1.,-.3,.5]],device='cuda',dtype=torch.float64,requires_grad=True)*1e-9
    response = StaticVolumeResponse(system,EPSILON_0,[2*EPSILON_0],[binding],[polarization])
    outside,inside = response.boundary_fields()
    jump_d = EPSILON_0*(outside*response.normals).sum(-1)-2*EPSILON_0*(inside*response.normals).sum(-1)
    expected = (polarization*response.normals).sum(-1)
    torch.testing.assert_close(jump_d,expected,rtol=1e-12,atol=1e-21)
    torch.testing.assert_close(torch.linalg.cross(outside-inside,response.normals),torch.zeros_like(inside),rtol=0,atol=1e-12)
    point = vertices.mean(0,keepdim=True).requires_grad_()
    potential,electric = response.fields(point,1)
    gradient, = torch.autograd.grad(potential.sum(),point,retain_graph=True)
    torch.testing.assert_close(electric,-gradient,rtol=1e-10,atol=1e-11)
    sensitivity, = torch.autograd.grad(electric.sum(),polarization)
    assert torch.isfinite(sensitivity).all() and sensitivity.abs().max() > 0
    matched = StaticVolumeResponse(system,EPSILON_0,[EPSILON_0],[binding],[polarization])
    assert torch.count_nonzero(matched.charge) == 0
    actual = matched.fields(point,1)
    expected,_ = cell_static_polarization_fields(volume,polarization,point,EPSILON_0)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_uniformly_polarized_sphere_dc_refines_toward_analytic_internal_field():
    errors = []
    for level in (1,2):
        surface = sphere_mesh(radius=.1,subdivisions=level)
        vertices = torch.cat((surface.vertices,surface.vertices.new_zeros(1,3)))
        cells = torch.cat((torch.full((len(surface.faces),1),len(surface.vertices),device='cuda',dtype=torch.int64),surface.faces),dim=1)
        volume = TetrahedralVolumeMesh(vertices,cells,region_id=1)
        system,binding = system_from_volume(volume)
        polarization = vertices.new_tensor([[1.,-.3,.5]])*1e-9
        response = StaticVolumeResponse(system,EPSILON_0,[2*EPSILON_0],[binding],
            [polarization.expand(len(cells),3)])
        point = vertices.new_tensor([[.013,.007,.019]])
        _,actual = response.fields(point,1)
        expected = -polarization/(4*EPSILON_0)
        error = float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
        errors.append(error)
    print({'sphere_dc_relative_errors':errors,'physical_qualification':False})
    assert errors[1] < errors[0]/2


def test_prepared_dc_columns_match_nonzero_actual_interface_solution():
    from witwin.maxwell.mom.harmonic_preparation import prepare_static_cell_response
    vertices = torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    system,binding = system_from_volume(volume)
    prepared = prepare_static_cell_response(system,[binding],EPSILON_0,[2*EPSILON_0])
    polarization = vertices.new_tensor([[1.,-.3,.5]])*1e-9
    response = StaticVolumeResponse(system,EPSILON_0,[2*EPSILON_0],[binding],[polarization])
    expected = response.fields(prepared['points'],1)[1]
    actual = (prepared['electric_response']@polarization.flatten()).reshape_as(expected)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(prepared['surface_response']@polarization.flatten(),response.charge,rtol=1e-12,atol=1e-12)

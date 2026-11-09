"""Independent magnetic sphere oracle and conservative-current interface jumps."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import sphere_mesh,RWGBasis,SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
from witwin.maxwell.mom.static_magnetic import MagnetostaticInterfaceResponse
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.magnetostatic import cell_magnetostatic_current_fields

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def system(mesh):
    return SurfaceSystem.create([(0,'body',RWGBasis(mesh),compile_medium(mw.Material(eps_r=2.,mu_r=3.),3e8))])


def test_magnetic_sphere_refinement_and_permeability_gradient():
    errors=[]
    exterior_errors=[]
    exterior_gradient_errors=[]
    for level in (1,2):
        mesh=sphere_mesh(radius=.1,subdivisions=level)
        op=FloatingPotentialOperator(system(mesh),testing_order=6)
        mu=mesh.vertices.new_tensor(3.,requires_grad=True)
        background=mesh.vertices.new_tensor([.3,-.2,1.])
        response=MagnetostaticInterfaceResponse(op,[],[],1.,[mu],background_h=background)
        points=mesh.vertices.new_tensor([[.013,.007,.019],[-.031,.011,-.023],[.041,-.033,.017]])
        magnetic,flux=response.fields(points,1)
        expected=3/(mu+2)*background
        error=float(((magnetic-expected).norm()/expected.expand_as(magnetic).norm()).detach())
        errors.append(error)
        exterior_points=mesh.vertices.new_tensor([[.13,.02,-.01],[-.03,.17,.04],[.06,-.04,.25]])
        radius=exterior_points.norm(dim=-1)
        direction=exterior_points/radius[:,None]
        dipole=(.1/radius[:,None])**3*(3*direction*(direction*background).sum(-1)[:,None]-background)
        expected_scattered=(mu-1)/(mu+2)*dipole
        exterior_h,_=response.fields(exterior_points,0)
        scattered=exterior_h-background
        exterior_errors.append(float(((scattered-expected_scattered).norm()/expected_scattered.norm()).detach()))
        exterior_derivative,=torch.autograd.grad(scattered[0,2],mu,retain_graph=True)
        expected_exterior_derivative=3/(mu+2)**2*dipole[0,2]
        exterior_gradient_errors.append(abs(float(((exterior_derivative-expected_exterior_derivative)/expected_exterior_derivative).detach())))
        derivative,=torch.autograd.grad(magnetic[0,2],mu)
        expected_derivative=-3/(mu+2)**2
        assert abs(float((derivative-expected_derivative).detach()))<.01
        torch.testing.assert_close(flux,mu*magnetic,rtol=0,atol=0)
    print({'sphere_internal_H_relative_errors':errors,'sphere_scattered_H_relative_errors':exterior_errors,'sphere_scattered_H_gradient_relative_errors':exterior_gradient_errors,
           'final_field_threshold':.005,'field_threshold_passed':max(errors[-1],exterior_errors[-1])<=.005,
           'physical_qualification':False})
    assert errors[1]<errors[0]
    assert exterior_errors[1]<exterior_errors[0]
    assert exterior_gradient_errors[1]<exterior_gradient_errors[0]


def test_conservative_volume_current_mu_interface_and_matched_reduction():
    volume=runpy.run_path('tests/mom/test_magnetostatic.py')['mesh']()
    sys=system(volume.boundary)
    binding=bind_volume_region(volume,sys)
    op=FloatingPotentialOperator(sys,testing_order=4)
    generator=torch.Generator(device='cuda').manual_seed(27)
    ac=torch.randn((2,len(volume.cells),3),device='cuda',dtype=torch.complex128,generator=generator)*.4
    current=FaceCurrentBalance(volume).solve(ac,conductivity=.05,tpa_scale=.8,absolute_tolerance=1e-16).mean_current
    response=MagnetostaticInterfaceResponse(op,[binding],[current],1.,[3.])
    outside,inside=response.boundary_fields()
    torch.testing.assert_close((outside*op.normals).sum(-1),3*(inside*op.normals).sum(-1),rtol=1e-10,atol=1e-13)
    jump=outside-inside
    torch.testing.assert_close(jump,op.normals*(jump*op.normals).sum(-1)[:,None],rtol=1e-10,atol=1e-13)
    matched=MagnetostaticInterfaceResponse(op,[binding],[current],1.,[1.])
    assert torch.count_nonzero(matched.density)==0
    points=volume.vertices.new_tensor([[.2,.12,.14]])
    expected,_=cell_magnetostatic_current_fields(volume,current,points)
    actual,_=matched.fields(points,0)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    response.density.add_(1.)
    with pytest.raises(RuntimeError,match='mutated'):
        response.fields(points,0)

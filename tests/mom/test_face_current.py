"""Conservative nonlinear cell current without postprocessing projection."""
import runpy
import pytest
import torch
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.magnetostatic import cell_current_continuity,cell_magnetostatic_current_fields

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_face_current_affine_ohmic_solution_and_native_gradient():
    volume = runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block = FaceCurrentBalance(volume)
    sigma = volume.vertices.new_tensor(.8,requires_grad=True)
    gradient = volume.vertices.new_tensor([.7,-.3,.4])
    source = (sigma.detach()*gradient).expand(len(volume.cells),3)
    ac = torch.zeros((0,len(volume.cells),3),device='cuda',dtype=torch.complex128)
    result = block.solve(ac,conductivity=sigma,source_current=source)
    centroid = (volume.volumes[:,None]*volume.vertices[volume.cells].mean(1)).sum(0)/volume.volumes.sum()
    expected = (volume.vertices[block.faces].mean(1)-centroid)@gradient
    torch.testing.assert_close(result.potential,expected,rtol=1e-11,atol=1e-13)
    torch.testing.assert_close(result.electric,-gradient.expand_as(source),rtol=1e-11,atol=1e-13)
    derivative, = torch.autograd.grad(result.electric[:,0].mean(),sigma)
    torch.testing.assert_close(derivative,gradient[0]/sigma,rtol=1e-11,atol=1e-13)


def test_nonuniform_tpa_current_conserves_every_face_and_obeys_ampere():
    volume = runpy.run_path('tests/mom/test_magnetostatic.py')['mesh']()
    generator = torch.Generator(device='cuda').manual_seed(27)
    ac = torch.randn((2,len(volume.cells),3),device='cuda',dtype=torch.complex128,generator=generator)*.4
    block = FaceCurrentBalance(volume)
    tpa = volume.vertices.new_tensor(.8,requires_grad=True)
    result = block.solve(ac,conductivity=.05,tpa_scale=tpa,absolute_tolerance=1e-16)
    phase = torch.arange(1024,device='cuda',dtype=torch.float64)*(2*torch.pi/1024)
    time_ac = torch.einsum('hn,hci->nci',torch.exp(-1j*torch.arange(1,3,device='cuda')[:,None]*phase),ac).real
    total = time_ac+result.electric[None]
    constitutive = .05*result.electric+tpa*(total.square().sum(-1)[...,None]*total).mean(0)
    torch.testing.assert_close(result.mean_current,constitutive,rtol=1e-10,atol=1e-13)
    diagnostic = cell_current_continuity(volume,result.mean_current)
    assert diagnostic['normal_current_jump_A_per_m2'].abs().max() < 1e-11
    # Independently integrated oriented face flux equals each assembled equation.
    torch.testing.assert_close(block.weak_current(result.mean_current),diagnostic['face_flux_defect_A'],
                               rtol=1e-8,atol=1e-16)
    points = torch.cat((volume.vertices[volume.cells].mean(1),volume.vertices.new_tensor([[.2,.12,.14]])))
    magnetic,receipt = cell_magnetostatic_current_fields(volume,result.mean_current,points)
    assert magnetic.norm() > 0
    expected = torch.cat((result.mean_current,torch.zeros_like(points[-1:])))
    torch.testing.assert_close(receipt['curl_H_A_per_m2'],expected,rtol=1e-8,atol=1e-11)
    derivative, = torch.autograd.grad(result.electric.square().sum(),tpa)
    step = 1e-5
    values = [block.solve(ac,conductivity=.05,tpa_scale=float(tpa.detach())+shift,
                         absolute_tolerance=1e-16).electric.square().sum() for shift in (-step,step)]
    torch.testing.assert_close(derivative,(values[1]-values[0])/(2*step),rtol=1e-5,atol=1e-9)
    print({'face_jump_rms':float(diagnostic['area_weighted_jump_rms_A_per_m2'].detach()),
           'ampere_defect_norm':float(receipt['ampere_defect_A_per_m2'].norm().detach()),
           'current_norm':float(result.mean_current.norm().detach()),'physical_qualification':False})


def test_face_potential_mapping_preserves_permuted_surface_order():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import bind_volume_region
    from witwin.maxwell.mom.harmonic_current import surface_face_potentials
    from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
    volume = runpy.run_path('tests/mom/test_static_current.py')['mesh']()
    block = FaceCurrentBalance(volume)
    original = volume.boundary
    permutation = torch.arange(len(original.vertices),device='cuda').flip(0)
    surface = SurfaceMesh(original.vertices[permutation],permutation[original.faces.flip(0)])
    medium = compile_medium(mw.Material(eps_r=2.),3e8)
    system = SurfaceSystem.create([(0,'body',RWGBasis(surface),medium)])
    binding = bind_volume_region(volume,system)
    gradient = volume.vertices.new_tensor([.7,-.3,.4])
    potentials = volume.vertices[block.faces].mean(1)@gradient
    means = surface_face_potentials(block,binding,potentials)
    torch.testing.assert_close(means,surface.triangles.mean(1)@gradient,rtol=1e-13,atol=1e-14)
    operator = FloatingPotentialOperator(system)
    face_result = operator.solve_face_means(means,epsilon_exterior=1.)
    nodal_result = operator.solve([surface.vertices@gradient],epsilon_exterior=1.)
    torch.testing.assert_close(face_result.density,nodal_result.density,rtol=1e-12,atol=1e-13)
    torch.testing.assert_close(face_result.offsets,nodal_result.offsets,rtol=1e-12,atol=1e-13)

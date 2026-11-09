"""Independent Biot-Savart cubature and closed-current Maxwell identities."""
import math
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.magnetostatic import cell_current_continuity,cell_magnetostatic_current_fields
from witwin.maxwell.mom.static_current import NodalCurrentBalance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def mesh():
    vertices = torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    return TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1).refine(1)


def test_static_biot_savart_matches_direct_cubature_and_native_gradient():
    volume = mesh()
    scale = volume.vertices.new_tensor(.7,requires_grad=True)
    current = scale*volume.vertices.new_tensor([1.,-.4,.3]).expand(len(volume.cells),3)
    points = volume.vertices.new_tensor([[.2,-.12,.16],[-.14,.11,.22]])
    magnetic,receipt = cell_magnetostatic_current_fields(volume,current,points)
    nodes,weights = volume.quadrature(12)
    displacement = points[:,None,None,:]-nodes[None]
    expected = (torch.linalg.cross(current[None,:,None,:].expand_as(displacement),displacement)
                /displacement.norm(dim=-1)[...,None]**3*weights[None,:,:,None]).sum((1,2))/(4*math.pi)
    torch.testing.assert_close(magnetic,expected,rtol=1e-10,atol=1e-14)
    derivative, = torch.autograd.grad(magnetic.square().sum(),scale)
    torch.testing.assert_close(derivative,2*magnetic.square().sum()/scale,rtol=1e-12,atol=1e-15)
    assert receipt['ampere_defect_A_per_m2'].norm() > 1e-5
    assert cell_current_continuity(volume,current)['normal_current_jump_A_per_m2'].norm() > .1


def test_closed_cell_current_obeys_ampere_inside_and_outside():
    volume = mesh()
    zero = volume.vertices.new_zeros(len(volume.cells),3)
    matrix = torch.autograd.functional.jacobian(
        lambda j:cell_current_continuity(volume,j)['face_flux_defect_A'],zero).flatten(1)
    _,singular,right = torch.linalg.svd(matrix,full_matrices=True)
    rank = int((singular > 1e-12*singular.max()).sum())
    assert rank < matrix.shape[1]
    current = right[rank].reshape_as(zero)
    continuity = cell_current_continuity(volume,current)
    assert continuity['normal_current_jump_A_per_m2'].abs().max() < 1e-13
    inside = volume.vertices[volume.cells].mean(1)
    outside = volume.vertices.new_tensor([[.2,.12,.14],[-.1,.08,.11]])
    magnetic,receipt = cell_magnetostatic_current_fields(volume,current,torch.cat((inside,outside)))
    expected = torch.cat((current,torch.zeros_like(outside)))
    torch.testing.assert_close(receipt['curl_H_A_per_m2'],expected,rtol=1e-10,atol=1e-12)
    assert receipt['ampere_defect_A_per_m2'].norm() < 1e-12
    assert receipt['divergence_H_A_per_m2'].norm() < 1e-12
    assert magnetic.norm() > 0


def test_nodal_tpa_balance_exposes_unresolved_face_current_jumps():
    volume = mesh()
    generator = torch.Generator(device='cuda').manual_seed(27)
    ac = torch.randn((2,len(volume.cells),3),device='cuda',dtype=torch.complex128,generator=generator)*.4
    block = NodalCurrentBalance(volume)
    solution = block.solve(ac,conductivity=.05,tpa_scale=.8)
    assert block.weak_current(solution.mean_current).norm() < 1e-13
    continuity = cell_current_continuity(volume,solution.mean_current)
    assert continuity['area_weighted_jump_rms_A_per_m2'] > 1e-3
    _,receipt = cell_magnetostatic_current_fields(volume,solution.mean_current,
        volume.vertices.new_tensor([[.2,.12,.14]]))
    assert receipt['ampere_defect_A_per_m2'].norm() > 1e-6
    print({'nodal_residual':float(block.weak_current(solution.mean_current).norm()),
           'normal_jump_rms':float(continuity['area_weighted_jump_rms_A_per_m2']),
           'exterior_ampere_defect':float(receipt['ampere_defect_A_per_m2'].norm()),
           'physical_qualification':False})

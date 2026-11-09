"""Anisotropic current fields: direct cubature and distributional Maxwell tests."""
import math
import pytest
import torch
from test_magnetostatic import mesh
from witwin.maxwell.constants import MU_0
from witwin.maxwell.mom.magnetostatic import (
    cell_current_continuity,cell_magnetostatic_current_fields,cell_tensor_magnetostatic_current_fields)

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def coefficient(sample):
    return MU_0*sample.new_tensor([[2.,.3,.1],[.3,1.5,-.2],[.1,-.2,3.]])


def test_tensor_magnetic_cubature_scale_and_native_derivatives():
    volume=mesh();sample=volume.vertices
    amplitude=sample.new_tensor(.7,requires_grad=True)
    current=amplitude*sample.new_tensor([1.,-.4,.3]).expand(len(volume.cells),3)
    points=sample.new_tensor([[.2,-.12,.16],[-.14,.11,.22]],requires_grad=True)
    anisotropy=sample.new_tensor(.2,requires_grad=True)
    mu=coefficient(sample)+MU_0*anisotropy*torch.diag(sample.new_tensor([1.,-.2,.3]))
    h,b,receipt=cell_tensor_magnetostatic_current_fields(volume,current,points,mu,pair_batch=13)
    nodes,weights=volume.quadrature(12)
    displacement=points[:,None,None,:]-nodes[None]
    inverse=torch.linalg.inv(mu)
    radius=torch.einsum('...i,ij,...j->...',displacement,inverse,displacement).sqrt()
    direct=(torch.linalg.cross(current[None,:,None,:].expand_as(displacement),displacement)
        /radius[...,None]**3*weights[None,:,:,None]).sum((1,2))/(4*math.pi*torch.linalg.det(mu).sqrt())
    torch.testing.assert_close(h,direct,rtol=1e-9,atol=1e-14)
    torch.testing.assert_close(b,h@mu.T,rtol=0,atol=0)
    observed=torch.stack([torch.autograd.grad(h[:,axis].sum(),points,retain_graph=True,create_graph=True)[0]
                          for axis in range(3)],dim=1)
    torch.testing.assert_close(observed,receipt['gradient_H_A_per_m2'],rtol=1e-10,atol=1e-12)
    derivative=torch.autograd.grad(h.square().sum(),anisotropy,retain_graph=True)[0]
    step=1e-5
    objectives=[]
    for value in (anisotropy.detach()-step,anisotropy.detach()+step):
        perturbed=coefficient(sample)+MU_0*value*torch.diag(sample.new_tensor([1.,-.2,.3]))
        objectives.append(cell_tensor_magnetostatic_current_fields(volume,current,points,perturbed)[0].square().sum())
    torch.testing.assert_close(derivative,(objectives[1]-objectives[0])/(2*step),rtol=2e-6,atol=1e-13)
    source_derivative=torch.autograd.grad(h.square().sum(),amplitude)[0]
    torch.testing.assert_close(source_derivative,2*h.square().sum()/amplitude,rtol=1e-12,atol=1e-14)
    scaled=cell_tensor_magnetostatic_current_fields(volume,current,points,17*mu)
    torch.testing.assert_close(scaled[0],h,rtol=1e-10,atol=1e-14)
    torch.testing.assert_close(scaled[1],17*b,rtol=1e-10,atol=1e-18)
    scalar=cell_tensor_magnetostatic_current_fields(volume,current,points,MU_0*torch.eye(3,device='cuda',dtype=sample.dtype))
    reference=cell_magnetostatic_current_fields(volume,current,points)
    torch.testing.assert_close(scalar[0],reference[0],rtol=1e-10,atol=1e-14)
    assert receipt['ampere_defect_A_per_m2'].norm()>1e-5
    assert not receipt['physical_qualification'] and not receipt['interface_conditions_solved']


def test_conservative_tensor_current_ampere_contact_and_flux_divergence():
    volume=mesh();zero=volume.vertices.new_zeros(len(volume.cells),3)
    matrix=torch.autograd.functional.jacobian(lambda j:cell_current_continuity(volume,j)['face_flux_defect_A'],zero).flatten(1)
    _,singular,right=torch.linalg.svd(matrix,full_matrices=True)
    rank=int((singular>1e-12*singular.max()).sum())
    current=right[rank].reshape_as(zero)
    assert cell_current_continuity(volume,current)['normal_current_jump_A_per_m2'].abs().max()<1e-13
    inside=volume.vertices[volume.cells].mean(1)
    outside=volume.vertices.new_tensor([[.2,.12,.14],[-.1,.08,.11]])
    h,b,receipt=cell_tensor_magnetostatic_current_fields(volume,current,torch.cat((inside,outside)),coefficient(zero))
    expected=torch.cat((current,torch.zeros_like(outside)))
    torch.testing.assert_close(receipt['contact_current_A_per_m2'],expected,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(receipt['curl_H_A_per_m2'],expected,rtol=1e-10,atol=1e-12)
    assert receipt['ampere_defect_A_per_m2'].norm()<1e-12
    assert receipt['divergence_B_T_per_m'].norm()<1e-17
    assert h.norm()>0 and b.norm()>0
    with pytest.raises(ValueError,match='principal values'):
        cell_tensor_magnetostatic_current_fields(volume,current,volume.vertices[volume.cells[:1,[1,2,3]]].mean(1),coefficient(zero))
    with pytest.raises(ValueError,match='positive definite'):
        cell_tensor_magnetostatic_current_fields(volume,current,outside,-coefficient(zero))
    with pytest.raises(ValueError,match='budget'):
        cell_tensor_magnetostatic_current_fields(volume,current,outside,coefficient(zero),pair_batch=0)

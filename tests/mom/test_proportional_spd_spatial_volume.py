"""Opt-in spatial transform with independent scalar physical volume integrals."""
import pytest
import torch
from test_proportional_spd_spatial import medium, kernel
from test_refined_lossy_anisotropic_volume_reference import errors
from witwin.maxwell.mom.tensor_medium import TensorMedium, scalar_medium_if_isotropic
from witwin.maxwell.mom.tensor_volume import cell_tensor_current_blocks
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.vie_operator import cell_current_field_blocks

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_proportional_closed_form_physical_volume_and_static_contact(tmp_path):
    m=medium();a=m.epsilon_r.real
    vertices=.07*a.new_tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    cells=torch.tensor([[0,1,2,3]],device=a.device)
    mesh=TetrahedralVolumeMesh(vertices,cells,region_id=1)
    queries=a.new_tensor([[.10,.07,.11],[.015,.012,.018]])
    actual,receipt=cell_tensor_current_blocks(mesh,queries,kernel(m),order=12,radial_order=12,
        face_rule='distance_scaled_duffy',kernel_batch=8192)
    # A symmetric square root supplies a distinct coordinate frame. The scalar
    # reference uses exact boundary potentials rather than radial volume fans.
    eigenvalues,eigenvectors=torch.linalg.eigh(a)
    transform=(eigenvectors*eigenvalues.sqrt())@eigenvectors.T/eigenvalues.prod().sqrt()
    inverse=torch.linalg.inv(transform)
    reference_mesh=TetrahedralVolumeMesh(vertices@inverse.T,cells,region_id=1)
    eye=torch.eye(3,device=a.device,dtype=m.epsilon_r.dtype)
    scalar=scalar_medium_if_isotropic(TensorMedium(m.frequency,eye,.5*eye))
    e,h=cell_current_field_blocks(reference_mesh,queries@inverse.T,scalar,order=40)
    e,h=e.permute(0,2,1,3),h.permute(0,2,1,3)
    blocks=torch.cat((torch.cat((e,-h),-1),torch.cat((h,e*scalar.epsilon/scalar.mu),-1)),-2)
    dual=torch.block_diag(inverse,inverse).to(m.epsilon_r.dtype)
    expected=torch.linalg.det(transform)*dual.T@blocks@dual
    relative_errors=errors(actual,expected)
    torch.save(dict(actual=actual,reference=expected,epsilon_r=m.epsilon_r,mu_r=m.mu_r,
        vertices=vertices,cells=cells,queries=queries,errors=relative_errors,receipt=receipt),
        tmp_path/'volume-comparison.pt')
    print(dict(block_errors=relative_errors,physical_qualification=False),flush=True)
    assert max(relative_errors)<.005
    assert receipt['contact_distribution_included'] and receipt['static_diagonal_included']
    assert all(item['kernel_version']=='analytic_proportional_spd_spatial_v1'
               for item in receipt['spectral_evaluations'])
    assert not receipt['volume_quadrature_converged'] and not receipt['physical_qualification']

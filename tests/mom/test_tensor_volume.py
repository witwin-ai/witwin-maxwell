"""AC volume self integration against the independent scalar boundary kernel."""
import pytest
import torch
from test_static_tensor_volume import fixture
from witwin.maxwell.mom.tensor_medium import TensorMedium,scalar_medium_if_isotropic
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.tensor_volume import cell_tensor_current_blocks
from witwin.maxwell.mom.vie_operator import cell_current_field_blocks

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_isotropic_full_current_blocks_interior_exterior_and_pv():
    mesh,a=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    medium=TensorMedium(3e7,2*eye,1.3*eye);kernel=DyadicGreenKernel(medium)
    scalar=scalar_medium_if_isotropic(medium)
    points=a.new_tensor([[.2,.2,.2],[1.5,.7,1.1],[.2,.3,0.]])
    result,receipt=cell_tensor_current_blocks(mesh,points,kernel,order=20,radial_order=8,
        principal_value=True,kernel_batch=1024)
    e,h=cell_current_field_blocks(mesh,points,scalar,order=32,principal_value=True)
    e,h=e.permute(0,2,1,3),h.permute(0,2,1,3)
    torch.testing.assert_close(result[:,:,:3,:3],e,rtol=4e-5,atol=1e-6)
    torch.testing.assert_close(result[:,:,3:,:3],h,rtol=4e-5,atol=1e-8)
    torch.testing.assert_close(result[:,:,:3,3:],-h,rtol=4e-5,atol=1e-8)
    torch.testing.assert_close(result[:,:,3:,3:],e*(scalar.epsilon/scalar.mu),rtol=4e-5,atol=1e-10)
    assert receipt['contact_distribution_included'] and not receipt['volume_quadrature_converged']
    assert not receipt['physical_qualification']


def test_selection_budget_admission_and_recorded_orders():
    mesh,a=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    kernel=DyadicGreenKernel(TensorMedium(3e7,2*eye,eye))
    x=a.new_tensor([[.2,.2,.2]])
    first,_=cell_tensor_current_blocks(mesh,x,kernel,order=3,radial_order=2,kernel_batch=17)
    selected,r=cell_tensor_current_blocks(mesh,x,kernel,cell_indices=mesh.cells.new_tensor([0,0]),
        order=3,radial_order=2,kernel_batch=31)
    torch.testing.assert_close(selected,first.expand_as(selected),rtol=1e-13,atol=1e-13)
    assert r['kernel_nodes']==2*4*3**2*2
    with pytest.raises(ValueError,match='kernel batch'):
        cell_tensor_current_blocks(mesh,x,kernel,kernel_batch=0)
    characteristic=DyadicGreenKernel(TensorMedium(3e7,torch.diag(eye.new_tensor([1.,-1.,1.])),eye))
    with pytest.raises(NotImplementedError,match='definite real or strictly lossy'):
        cell_tensor_current_blocks(mesh,x,characteristic)


def test_matched_anisotropic_medium_against_transformed_scalar_volume():
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    mesh,a=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    tensor=torch.diag(a.new_tensor([1.2,1.1,1.05]))
    medium=TensorMedium(3e7,tensor.to(eye.dtype),tensor.to(eye.dtype))
    config=DyadicIntegrationConfig(radial_order=16,azimuthal_order=16,
        max_radial_order=1024,max_azimuthal_order=256,relative_tolerance=2e-4,
        absolute_tolerance=1e-9,radial_map='matched_spd_branch_v1')
    points=a.new_tensor([[1.5,.7,1.1]])
    actual,receipt=cell_tensor_current_blocks(mesh,points,DyadicGreenKernel(medium,config),
        order=4,radial_order=4,kernel_batch=128)
    # Transformation optics: A=L L^T/det(L), x=L y. The vacuum volume
    # solution uses exact scalar face integrals, independent of radial fans
    # and the Weyl modal kernel under test.
    transform=torch.linalg.cholesky(tensor)/torch.linalg.det(tensor).sqrt()
    inverse=torch.linalg.inv(transform)
    volume=TetrahedralVolumeMesh(mesh.vertices@inverse.T,mesh.cells,region_id=1)
    scalar=scalar_medium_if_isotropic(TensorMedium(3e7,eye,eye))
    e,h=cell_current_field_blocks(volume,points@inverse.T,scalar,order=32)
    e,h=e.permute(0,2,1,3),h.permute(0,2,1,3)
    vacuum=torch.cat((torch.cat((e,-h),-1),torch.cat((h,e*scalar.epsilon/scalar.mu),-1)),-2)
    dual=torch.block_diag(inverse,inverse).to(eye.dtype)
    expected=torch.linalg.det(transform)*dual.T@vacuum@dual
    eta=abs(scalar.eta);balance=actual.new_tensor([1.]*3+[eta]*3)
    error=((actual-expected)*balance[:,None]*balance[None,:]).norm()/(
        expected*balance[:,None]*balance[None,:]).norm()
    print({'matched_tensor_volume_relative_error':float(error),'physical_qualification':False})
    assert error<.005
    assert all(r['converged'] for r in receipt['spectral_evaluations'])
    assert not receipt['volume_quadrature_converged']


def test_rotated_face_pv_matching_nodes_and_translation():
    from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
    from witwin.maxwell.mom.quadrature import triangle_rule
    mesh,a=fixture()
    eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    kernel=DyadicGreenKernel(TensorMedium(3e7,2*eye,1.3*eye))
    rotation=torch.linalg.qr(a.new_tensor([[1.,2.,3.],[3.,1.,2.],[2.,3.,1.]]))[0]
    if torch.linalg.det(rotation)<0:rotation[:,0]=-rotation[:,0]
    shift=a.new_tensor([.37,-.23,.41])
    moved=TetrahedralVolumeMesh(mesh.vertices@rotation.T+shift,mesh.cells,region_id=1)
    face=mesh.cells.new_tensor([0,2,1])
    points,_=triangle_rule(mesh.vertices[face][None],3)
    moved_points,_=triangle_rule(moved.vertices[face][None],3)
    original,r=cell_tensor_current_blocks(mesh,points[0],kernel,order=3,radial_order=3,
        principal_value=True,kernel_batch=128)
    actual,s=cell_tensor_current_blocks(moved,moved_points[0],kernel,order=3,radial_order=3,
        principal_value=True,kernel_batch=128)
    dual=torch.block_diag(rotation,rotation).to(eye.dtype)
    torch.testing.assert_close(actual,dual@original@dual.T,rtol=2e-11,atol=1e-10)
    # Exactly one face fan is degenerate for every query, including rotated
    # nodes constructed by the same rule as the volume integrator.
    assert r['kernel_nodes']==s['kernel_nodes']==len(points[0])*3*3**3
    with pytest.raises(ValueError,match='explicit principal'):
        cell_tensor_current_blocks(moved,moved_points[0],kernel,order=3,radial_order=3)


def test_projected_face_rule_near_field_refinement_and_signed_exterior():
    mesh,a=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    medium=TensorMedium(3e7,2*eye,1.3*eye);kernel=DyadicGreenKernel(medium)
    scalar=scalar_medium_if_isotropic(medium)
    points=a.new_tensor([[.07,.11,0.],[.07,.11,.001],[1.3,-.2,.1]])
    e,h=cell_current_field_blocks(mesh,points,scalar,order=40,principal_value=True)
    e,h=e.permute(0,2,1,3),h.permute(0,2,1,3)
    errors=[]
    for order in (3,6,12):
        actual,record=cell_tensor_current_blocks(mesh,points,kernel,order=order,radial_order=6,
            principal_value=True,face_rule='projected_duffy',kernel_batch=1024)
        errors.append([float((actual[:,:,:3,:3]-e).norm()/e.norm()),
                       float((actual[:,:,3:,:3]-h).norm()/h.norm())])
    print({'projected_face_rule_errors':errors,'physical_qualification':False})
    assert record['face_rule']=='projected_duffy'
    assert max(errors[-1])<.005
    per_query=torch.stack(((actual[:,:,:3,:3]-e).flatten(1).norm(dim=-1)/e.flatten(1).norm(dim=-1),
                           (actual[:,:,3:,:3]-h).flatten(1).norm(dim=-1)/h.flatten(1).norm(dim=-1)))
    print({'projected_face_rule_per_query_errors':per_query.tolist()})
    assert bool((per_query<.005).all())
    selected,_=cell_tensor_current_blocks(mesh,points[:1],kernel,order=3,radial_order=3,
        principal_value=True,face_rule='projected_duffy',cell_indices=mesh.cells.new_tensor([0,0]),kernel_batch=17)
    single,_=cell_tensor_current_blocks(mesh,points[:1],kernel,order=3,radial_order=3,
        principal_value=True,face_rule='projected_duffy',kernel_batch=128)
    torch.testing.assert_close(selected,single.expand_as(selected),rtol=2e-12,atol=1e-12)
    with pytest.raises(ValueError,match='face rule'):
        cell_tensor_current_blocks(mesh,points,kernel,face_rule='invalid')


def test_regular_volume_parts_reconstruct_static_contact_without_subtraction():
    from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
    from witwin.maxwell.mom.static_tensor_volume import cell_tensor_static_potentials
    mesh,a=fixture();eye=torch.eye(3,device=a.device,dtype=torch.complex128)
    medium=TensorMedium(3e7,2*eye,1.3*eye);kernel=DyadicGreenKernel(medium)
    points=a.new_tensor([[.2,.2,.2],[.2,.3,0.]])
    options=dict(order=4,radial_order=3,principal_value=True,face_rule='projected_duffy')
    full,_=cell_tensor_current_blocks(mesh,points,kernel,**options)
    regular,receipt=cell_tensor_current_blocks(mesh,points,kernel,regular_only=True,**options)
    restored=regular.clone()
    for section,coefficient in ((slice(0,3),medium.epsilon),(slice(3,6),medium.mu)):
        hessian=cell_tensor_static_potentials(mesh,points,StaticTensorGreen(coefficient.real),principal_value=True)[2]
        restored[:,:,section,section]+=1j/medium.omega*hessian
    torch.testing.assert_close(restored,full,rtol=2e-12,atol=1e-12)
    assert not receipt['static_diagonal_included'] and not receipt['contact_distribution_included']
    assert receipt['version']=='tensor_cell_radial_remainder_v1'
    with pytest.raises(ValueError,match='boolean'):
        cell_tensor_current_blocks(mesh,points,kernel,regular_only=1)
    with pytest.raises(ValueError,match='explicit principal'):
        cell_tensor_current_blocks(mesh,points,kernel,regular_only=True)

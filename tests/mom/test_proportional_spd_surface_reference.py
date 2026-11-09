"""Actual tensor weak panels against an independent transformed scalar kernel."""
import pytest
import torch
from witwin.maxwell.mom.mesh import RWGBasis
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.anisotropic_operator import weak_dyadic_pairs
import witwin.maxwell.mom.dyadic_green as dg

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('category',[0,4],ids=['self','separated'])
def test_proportional_surface_blocks_against_transformed_scalar(category,tmp_path):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    basis=RWGBasis(TetrahedralVolumeMesh(vertices,cells,region_id=1).boundary)
    source=basis if category==0 else RWGBasis(TetrahedralVolumeMesh(vertices+.2,cells,region_id=2).boundary)
    metric=torch.diag(vertices.new_tensor([1.25,1.,.75]))
    medium=TensorMedium(3e7,(2*metric).to(torch.complex128),(1.3*metric).to(torch.complex128))
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    scalar=TensorMedium(medium.frequency,2*eye,1.3*eye)
    transform=torch.linalg.cholesky(metric)/torch.linalg.det(metric).sqrt()
    inverse=torch.linalg.inv(transform);dual=torch.block_diag(inverse,inverse).to(eye.dtype)
    class Reference:
        def __init__(self):self.medium=medium
        def evaluate_regular(self,points):
            value,_=dg.isotropic_dyadic_green(points@inverse.T,scalar,regular_part=True)
            return dual.T@value@dual,dict(converged=True,static_diagonal_restored=False,
                static_diagonal_singularity_subtracted=True)
    kernel=dg.DyadicGreenKernel(medium,dg.DyadicIntegrationConfig(radial_map='proportional_spd_branch_v1',
        relative_tolerance=1e-5,absolute_tolerance=1e-10,max_radial_order=256,max_azimuthal_order=128))
    indices=cells.new_tensor([0])
    expected,_=weak_dyadic_pairs(basis,source,indices,indices,Reference(),category=category,order=3)
    scales=expected.norm(dim=(-2,-1))
    if category==0:
        # The two self curl blocks vanish pointwise by coplanarity. Normalize
        # against an independently computed unsigned integrand bound, not the
        # roundoff-sized signed reference or the observed spectral error.
        from witwin.maxwell.mom.operator import local_geometry
        from witwin.maxwell.mom.quadrature import touching_pair_rule
        triangles=basis.mesh.triangles[indices];faces=basis.mesh.faces[indices]
        obs,src,weights=touching_pair_rule(triangles,triangles,faces,faces,0,3)
        _,free,slopes=local_geometry(basis)
        left=slopes[indices,None,:,None]*(obs[:,:,None]-free[indices,None])
        right=slopes[indices,None,:,None]*(src[:,:,None]-free[indices,None])
        regular,_=Reference().evaluate_regular((obs-src).reshape(-1,3))
        regular=regular.reshape(*weights.shape,6,6)
        basis_weight=left.norm(dim=(-2,-1))*right.norm(dim=(-2,-1))*weights.abs()
        for row,offset,column in ((0,3,0),(1,0,3)):
            scale=(basis_weight*regular[:,:,offset:offset+3,column:column+3].norm(dim=(-2,-1))).sum(-1)
            assert bool((torch.isfinite(scale)&(scale>0)).all())
            assert bool((scales[:,row,row]<=128*torch.finfo(vertices.dtype).eps*scale).all())
            scales[:,row,row]=scale
            expected[:,row,row]=0
    actual,receipt=weak_dyadic_pairs(basis,source,indices,indices,kernel,category=category,order=3)
    torch.save(dict(actual=actual,expected=expected,scales=scales,receipt=receipt),tmp_path/'weak_blocks.pt')
    errors=(actual-expected).norm(dim=(-2,-1))/scales
    assert bool(torch.isfinite(errors).all()) and bool((errors<.005).all())
    assert receipt['kernel']['converged']
    print({'category':category,'block_relative_errors':errors.tolist(),
        'self_zero_blocks_use_unsigned_integrand_scale':category==0,
        'shared_static_potential_and_panel_quadrature':True,'physical_qualification':False},flush=True)

"""Targeted algebra controls for the test-only regional density experiment."""
import pytest
import torch
from static_regional_density_diagnostic import project_density_system, RegionalDensityDiagnostic, density_reference_mass
from test_static_coupling import system_from_volume
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_patch_projection_restores_nonuniform_area_and_preserves_linear_fields():
    # Coplanar triangles with areas 1 and 2. The shared edge uses opposite order.
    vertices=torch.tensor([[0.,0,0],[2,0,0],[0,1,0],[2,-2,0]],device='cuda',dtype=torch.float64)
    faces=torch.tensor([[0,1,2],[1,0,3]],device='cuda')
    triangles=vertices[faces]
    areas=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]).norm(dim=-1)/2
    mass=(torch.ones(3,3,device='cuda',dtype=torch.float64)+torch.eye(3,device='cuda',dtype=torch.float64))/12
    local=torch.block_diag(mass,mass)
    matrix=torch.block_diag(local,3*local)
    reduced,lift,weights=project_density_system(matrix,areas,faces,torch.zeros(2,device='cuda',dtype=torch.int64))
    p=lift[:6,:4]
    # Assemble the true integral analytically by global vertex IDs, independently
    # of the helper's projection and weighting implementation.
    expected=matrix.new_zeros(4,4)
    for f in range(2):
        for i in range(3):
            for j in range(3):
                expected[faces[f,i],faces[f,j]]+=areas[f]*mass[i,j]
    torch.testing.assert_close(reduced,torch.block_diag(expected,3*expected))
    nodal=1+2*vertices[:,0]-3*vertices[:,1]
    torch.testing.assert_close(p@nodal,(1+2*triangles[:,:,0]-3*triangles[:,:,1]).reshape(-1))
    assert not torch.allclose(reduced,lift.T@matrix@lift)
    rhs=matrix@torch.cat((p@nodal,p@(2*nodal)))
    solution=torch.linalg.solve(reduced,lift.T@(weights*rhs))
    torch.testing.assert_close(solution,torch.cat((nodal,2*nodal)))
    # Arbitrary positive face-row normalization must disappear after restoring it.
    factor=areas.repeat_interleave(3).repeat(2)
    alternate,_,_=project_density_system(matrix/factor[:,None],areas.square(),faces,torch.zeros(2,device='cuda',dtype=torch.int64))
    torch.testing.assert_close(alternate,reduced)
    separate,independent,_=project_density_system(matrix,areas,faces,torch.arange(2,device='cuda'))
    assert independent.shape==(12,12)
    torch.testing.assert_close(independent.T@independent,torch.eye(12,device='cuda',dtype=matrix.dtype))
    assert separate.shape==(12,12)


@pytest.mark.parametrize('degree',[1,2])
@torch.no_grad()
def test_regional_density_no_contrast_and_dg_projection_equivalence(degree):
    surface=sphere_mesh(radius=.1,subdivisions=0)
    vertices=torch.cat((surface.vertices,surface.vertices.new_zeros(1,3)))
    cells=torch.cat((torch.full((len(surface.faces),1),len(surface.vertices),device='cuda',dtype=torch.int64),surface.faces),1)
    system,_=system_from_volume(TetrahedralVolumeMesh(vertices,cells,region_id=1))
    exterior=torch.eye(3,device='cuda',dtype=torch.float64)
    operator=RegionalDensityDiagnostic(system,exterior,(2*exterior,),degree=degree)
    drive=exterior.new_tensor([.3,-.2,.7])
    operator.solve_uniform(drive)
    expected_ext=operator.external.clone();expected_int=operator.internal.clone()
    operator.solve_uniform(drive,patch_ids=torch.arange(operator.nf,device='cuda'))
    torch.testing.assert_close(operator.external,expected_ext,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(operator.internal,expected_int,rtol=2e-11,atol=2e-12)
    # Reuse raw blocks in the exact no-contrast case; no second quadrature run.
    n=operator.nb*operator.nf
    mass=density_reference_mass(exterior,degree)
    jump=torch.kron(torch.eye(operator.nf,device='cuda',dtype=exterior.dtype),mass)
    operator.matrix[:n,n:]=-operator.matrix[:n,:n]
    operator.matrix[n:,n:]=-operator.matrix[n:,:n]-jump/operator.scale
    operator.interior=operator.exterior
    operator.solve_uniform(drive,patch_ids=torch.zeros(operator.nf,device='cuda',dtype=torch.int64))
    assert operator.residual==0 and operator.local_residual==0
    assert torch.count_nonzero(operator.external)==0
    assert torch.count_nonzero(operator.internal)==0
    points=exterior.new_tensor([[0.,0,0],[.4,.2,-.3]])
    for region in (0,1):
        potential,field=operator.fields(points,region)
        torch.testing.assert_close(potential,-points@drive)
        torch.testing.assert_close(field,drive.expand_as(points))


def test_quadratic_mass_and_patch_projection_against_polynomial_integrals():
    from witwin.maxwell.mom.quadrature import triangle_rule
    from test_static_tensor_quadratic_weak import physical_basis
    vertices=torch.tensor([[0.,0,0],[2,0,0],[0,1,0],[2,-2,0]],device='cuda',dtype=torch.float64)
    faces=torch.tensor([[0,1,2],[1,0,3]],device='cuda')
    edges,inverse=torch.unique(faces[:,[[0,1],[1,2],[2,0]]].sort(-1).values.reshape(-1,2),dim=0,return_inverse=True)
    coords=torch.cat((vertices,vertices[edges].mean(1)))
    nodes=torch.cat((faces,4+inverse.reshape(-1,3)),1)
    triangles=vertices[faces]
    areas=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]).norm(dim=-1)/2
    x,w=triangle_rule(triangles,6);basis=physical_basis(x,triangles)
    reference=torch.einsum('pq,pqa,pqb->pab',w,basis,basis)/areas[:,None,None]
    mass=density_reference_mass(vertices,2)
    torch.testing.assert_close(reference,mass.expand_as(reference),rtol=1e-12,atol=1e-14)
    torch.testing.assert_close((w[:,:,None]*basis).sum(1)/areas[:,None],vertices.new_tensor([0,0,0,1/3,1/3,1/3]).expand(2,-1),atol=1e-14,rtol=1e-12)
    local=torch.block_diag(mass,mass);matrix=torch.block_diag(local,3*local)
    reduced,lift,weights=project_density_system(matrix,areas,nodes,torch.zeros(2,device='cuda',dtype=torch.int64))
    def polynomial(p):
        a,b=p[...,0],p[...,1]
        return 1+2*a-3*b+.5*a*a+.7*a*b+b*b
    nodal=polynomial(coords);solution=torch.cat((nodal,2*nodal))
    torch.testing.assert_close(solution@(reduced@solution),13*(w*polynomial(x).square()).sum(),rtol=1e-12,atol=1e-12)
    lifted=lift@solution
    torch.testing.assert_close(lifted[:12],polynomial(coords[nodes]).reshape(-1))
    rhs=matrix@lifted
    torch.testing.assert_close(torch.linalg.solve(reduced,lift.T@(weights*rhs)),solution,rtol=1e-12,atol=1e-12)
    assert reduced.shape==(18,18)

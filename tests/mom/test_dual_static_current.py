"""Conservation, native field reconstruction and electric-only limiting closure."""
import pytest
import torch
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.dual_static_current import DualStaticCurrentBalance
from witwin.maxwell.mom.static_current import FaceCurrentBalance
from witwin.maxwell.mom.magnetostatic import cell_magnetostatic_current_fields

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def volume(region=1,offset=0.):
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    mesh=TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=region)
    vertices=torch.cat((vertices,vertices.mean(0,keepdim=True)))+offset
    cells=torch.cat((torch.full((4,1),4,device='cuda',dtype=torch.int64),mesh.boundary.faces),1)
    return TetrahedralVolumeMesh(vertices,cells,region_id=region)


def test_two_body_dual_currents_conserve_and_reconstruct_fields():
    volumes=(volume(),volume(2,.3)); closure=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    source=torch.arange(24,device='cuda',dtype=torch.float64).reshape(8,3).sin().requires_grad_()
    other=source.detach().cos()
    result=closure.solve(source,other)
    j,m=result['electric_current'],result['magnetic_current']
    assert float((j.norm()*m.norm()).detach())>1e-6
    for index,block in enumerate(closure.blocks):
        assert float(block.weak_current(j[index*4:(index+1)*4]).norm().detach())<1e-14
        assert float(block.weak_current(m[index*4:(index+1)*4]).norm().detach())<1e-14
    cj=sum(cell_magnetostatic_current_fields(v,c,closure.points)[0] for v,c in zip(volumes,j.split(4)))
    cm=sum(cell_magnetostatic_current_fields(v,c,closure.points)[0] for v,c in zip(volumes,m.split(4)))
    ep=torch.cat([b.electric(p) for b,p in zip(closure.blocks,result['electric_potentials'])])
    hp=torch.cat([b.electric(p) for b,p in zip(closure.blocks,result['magnetic_potentials'])])
    torch.testing.assert_close(result['electric'],ep-cm,rtol=1e-11,atol=1e-13)
    torch.testing.assert_close(result['magnetic'],hp+cj,rtol=1e-11,atol=1e-13)
    grad=torch.autograd.grad(result['electric'].square().sum(),source)[0]
    assert bool(torch.isfinite(grad).all()) and float(grad.norm())>0
    assert not result['receipt']['physical_qualification']


def test_zero_magnetic_conductivity_limit_recovers_existing_electric_closure():
    v=volume(); source=torch.arange(12,device='cuda',dtype=torch.float64).reshape(4,3).cos()
    native=FaceCurrentBalance(v).solve(torch.zeros((1,4,3),device='cuda',dtype=torch.complex128),
                                    conductivity=2.,source_current=source)
    errors=[]
    for sigma in (1.,.01,.0001):
        result=DualStaticCurrentBalance((v,),[2.],[sigma]).solve(source,torch.zeros_like(source))
        errors.append(float((result['electric_current']-native.mean_current).norm()))
    assert errors[1]<errors[0]/90 and errors[2]<errors[1]/90
    assert errors[-1]<1e-6


def test_budget_and_geometry_mutation_reject():
    v=volume()
    with pytest.raises(MemoryError): DualStaticCurrentBalance((v,),[2.],[3.],max_matrix_bytes=1)
    closure=DualStaticCurrentBalance((v,),[2.],[3.])
    v.vertices.add_(.01)
    with pytest.raises(ValueError,match='changed'): closure.solve(torch.zeros((4,3),device='cuda',dtype=torch.float64),torch.zeros((4,3),device='cuda',dtype=torch.float64))


def test_neutral_dual_exterior_gauge_and_gradients():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.dual_static_current import DualStaticExterior
    from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import bind_volume_region
    from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
    volumes=(volume(),volume(2,.3))
    medium=compile_medium(mw.Material(eps_r=2.),3e8)
    system=SurfaceSystem.create([(i,str(i),RWGBasis(v.boundary),medium) for i,v in enumerate(volumes)])
    bindings=tuple(bind_volume_region(v,system) for v in volumes)
    laplace=FloatingPotentialOperator(system,testing_order=6)
    closure=DualStaticCurrentBalance(volumes,[2.,3.],[5.,7.])
    source=torch.arange(24,device='cuda',dtype=torch.float64).reshape(8,3).sin().requires_grad_()
    result=closure.solve(source,source.detach().cos())
    exterior=DualStaticExterior(closure,result,bindings,laplace)
    points=source.new_tensor([[.8,.9,.7],[-.3,-.1,.2]])
    fields=exterior.fields(points)
    for layer in (exterior.electric_layer,exterior.magnetic_layer):
        assert float((laplace.charge_test@layer.density).abs().max().detach())<1e-12
    shifted=dict(result)
    shifted['electric_potentials']=tuple(p+3 for p in result['electric_potentials'])
    shifted['magnetic_potentials']=tuple(p-2 for p in result['magnetic_potentials'])
    other=DualStaticExterior(closure,shifted,bindings,laplace).fields(points)
    for key in fields: torch.testing.assert_close(fields[key],other[key],rtol=1e-9,atol=1e-12)
    grad=torch.autograd.grad(fields['electric_V_per_m'].square().sum()+fields['magnetic_A_per_m'].square().sum(),source)[0]
    assert bool(torch.isfinite(grad).all()) and float(grad.norm().detach())>0
    original=exterior.electric_layer
    exterior.electric_layer=exterior.magnetic_layer
    with pytest.raises(RuntimeError,match='mutated'): exterior.fields(points)
    exterior.electric_layer=original
    exterior.j.add_(1.)
    with pytest.raises(RuntimeError,match='mutated'): exterior.fields(points)

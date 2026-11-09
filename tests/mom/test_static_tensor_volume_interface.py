"""Induced polarization transmission and independent anisotropic sphere checks."""
import pytest
import torch
from test_static_tensor_polarization import source
from test_static_coupling import system_from_volume
from test_static_tensor_interface import tensor
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_particular_boundary_trace_and_interface_source_ownership():
    particular,a,p=source()
    psi,q_inside,normal_p=particular.boundary_trace()
    body=particular.binding.surface_system.bodies[0]
    points=body.mesh.triangles.mean(1);normal=body.mesh.normals
    _,inside=particular.fields(points-1e-7*normal)
    _,closer=particular.fields(points-5e-8*normal)
    expected=-((2*closer-inside)@a.T*normal).sum(-1)
    torch.testing.assert_close(q_inside,expected,rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(normal_p,(p*normal).sum(-1),rtol=0,atol=0)
    op=StaticTensorInterfaceOperator(particular.binding.surface_system,torch.eye(3,device=a.device,dtype=a.dtype),(a,))
    drive=a.new_tensor([.1,-.2,.3]);zero=torch.zeros_like(p)
    baseline=op.solve_uniform(drive)
    neutral=op.solve_uniform(drive,bindings=(particular.binding,),polarizations=(zero,))
    torch.testing.assert_close(neutral.potential,baseline.potential,rtol=0,atol=0)
    torch.testing.assert_close(neutral.conormal,baseline.conormal,rtol=0,atol=0)
    with pytest.raises(ValueError,match='Duplicate'):
        op.solve_uniform(drive,bindings=(particular.binding,particular.binding),polarizations=(p,p))
    other,_,_=source()
    with pytest.raises(ValueError,match='exact surface'):
        op.solve_uniform(drive,bindings=(other.binding,),polarizations=(p,))
    response=op.solve_uniform(drive,bindings=(particular.binding,),polarizations=(p,))
    potential,field=response.fields(a.new_tensor([[.021,.019,.023]],requires_grad=True),1)
    sensitivity,=torch.autograd.grad(field.sum(),p)
    assert bool(torch.isfinite(sensitivity).all()) and bool(sensitivity.abs().max()>0)
    response.sources[0].polarization.detach().add_(1)
    with pytest.raises(ValueError,match='changed'):response.fields(a.new_tensor([[.02,.02,.02]]),1)


def test_polarized_anisotropic_sphere_refinement_and_uniform_drive_superposition():
    a=tensor();eye=torch.eye(3,device=a.device,dtype=a.dtype)
    polarization=a.new_tensor([[1.,-.3,.5]])
    expected=-torch.linalg.solve(a+2*eye,polarization[0])
    errors=[]
    for level in (1,2):
        surface=sphere_mesh(radius=.1,subdivisions=level)
        vertices=torch.cat((surface.vertices,surface.vertices.new_zeros(1,3)))
        cells=torch.cat((torch.full((len(surface.faces),1),len(surface.vertices),device='cuda',dtype=torch.int64),surface.faces),dim=1)
        volume=TetrahedralVolumeMesh(vertices,cells,region_id=1)
        system,binding=system_from_volume(volume)
        op=StaticTensorInterfaceOperator(system,eye,(a,))
        p=polarization.expand(len(cells),3)
        response=op.solve_uniform(a.new_zeros(3),bindings=(binding,),polarizations=(p,))
        point=a.new_tensor([[.013,.007,.019]])
        _,field=response.fields(point,1)
        errors.append(float(((field[0]-expected).norm()/expected.norm()).detach()))
        assert response.receipt['relative_algebraic_residual']<1e-12
        if level==1:
            drive=a.new_tensor([.2,.1,-.3])
            incident=op.solve_uniform(drive)
            combined=op.solve_uniform(drive,bindings=(binding,),polarizations=(p,))
            for region,query in ((1,point),(0,a.new_tensor([[.3,.2,.1]]))):
                for total,left,right in zip(combined.fields(query,region),response.fields(query,region),incident.fields(query,region)):
                    torch.testing.assert_close(total,left+right,rtol=2e-11,atol=1e-12)
    print({'polarized_tensor_sphere_relative_errors':errors,'physical_threshold':.005,'selected_physical_pass':errors[-1]<.005,'refinement_control_threshold':.05,'physical_qualification':False})
    assert errors[1]<errors[0] and errors[1]<.05


def test_nonzero_polarization_multibody_superposition_and_source_order():
    from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import bind_volume_region
    import witwin.maxwell as mw
    base,a,p=source()
    first=base.binding.volume
    second=TetrahedralVolumeMesh(first.vertices+a.new_tensor([.3,0.,0.]),first.cells,region_id=2)
    system=SurfaceSystem.create([(i,'body'+str(i),RWGBasis(v.boundary),compile_medium(mw.Material(),3e8))
                                 for i,v in enumerate((first,second))])
    b1,b2=(bind_volume_region(v,system) for v in (first,second))
    op=StaticTensorInterfaceOperator(system,torch.eye(3,device=a.device,dtype=a.dtype),(a,1.3*a))
    zero=a.new_zeros(3);p2=.4*p
    left=op.solve_uniform(zero,bindings=(b1,),polarizations=(p,))
    right=op.solve_uniform(zero,bindings=(b2,),polarizations=(p2,))
    both=op.solve_uniform(zero,bindings=(b2,b1),polarizations=(p2,p))
    assert both.receipt['induced_regions']==(1,2)
    assert float(left.fields(a.new_tensor([[.321,.019,.023]]),2)[1].abs().max().detach())>1e-5
    for region,point in ((0,[.2,.2,.2]),(1,[.021,.019,.023]),(2,[.321,.019,.023])):
        for total,l,r in zip(both.fields(a.new_tensor([point]),region),left.fields(a.new_tensor([point]),region),right.fields(a.new_tensor([point]),region)):
            torch.testing.assert_close(total,l+r,rtol=2e-10,atol=1e-12)


def test_nonuniform_polarization_matched_medium_interface_refinement():
    from witwin.maxwell.mom.vie_operator import cell_static_polarization_fields
    base,a,p=source(refine=1,scalar=True)
    original=base.binding.volume
    p=p*torch.linspace(.7,1.4,len(p),device=p.device,dtype=p.dtype)[:,None]
    query=p.new_tensor([[.021,.019,.023],[.13,.12,.11]])
    expected,_=cell_static_polarization_fields(original,p,query,2.)
    errors=[]
    volume=original
    for level in (0,1):
        if level:
            volume=volume.refine();p=p.repeat_interleave(8,dim=0)
        system,binding=system_from_volume(volume)
        op=StaticTensorInterfaceOperator(system,a,(a,))
        response=op.solve_uniform(a.new_zeros(3),bindings=(binding,),polarizations=(p,))
        fields=torch.cat([response.fields(query[i:i+1],region)[1] for i,region in enumerate((1,0))])
        errors.append(float(((fields-expected[1]).norm()/expected[1].norm()).detach()))
    print({'nonuniform_matched_medium_errors':errors,'physical_threshold':.005,
           'selected_physical_pass':errors[-1]<.005,'physical_qualification':False})
    assert errors[1]<errors[0]


def test_empty_internal_charge_retains_finite_zero_and_query_gradient():
    particular,a,_=source()
    points=a.new_tensor([[1e308,1e308,1e308]],requires_grad=True)
    potential,field=particular.fields(points,interior_sources_only=True)
    assert torch.equal(potential,torch.zeros_like(potential))
    assert torch.equal(field,torch.zeros_like(field))
    gradient,=torch.autograd.grad(potential.sum()+field.sum(),points)
    assert torch.equal(gradient,torch.zeros_like(gradient))

"""Magnetic contrast VIE: actual currents, tensor coupling and sphere control."""
import pytest
import torch
from test_magnetostatic import mesh
from test_static_coupling import system_from_volume
from test_tensor_magnetostatic import coefficient
from witwin.maxwell.constants import MU_0
from witwin.maxwell.mom.magnetostatic import cell_tensor_magnetostatic_current_fields
from witwin.maxwell.mom.static_magnetic_volume import MagneticVolumeContrastResponse
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_matched_current_and_anisotropic_contrast_native_derivative():
    volume=mesh();system,binding=system_from_volume(volume);sample=volume.vertices
    mu=coefficient(sample);current=sample.new_tensor([1.,-.4,.3]).expand(len(volume.cells),3)
    points=sample.new_tensor([[.2,.12,.14]])
    response=MagneticVolumeContrastResponse(system,(binding,),(current,),mu,(mu,))
    actual=response.fields(points,0);expected=cell_tensor_magnetostatic_current_fields(volume,current,points,mu)
    for a,b in zip(actual[:2],expected[:2]):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-18)
    assert actual[2]['ampere_defect_A_per_m2'].norm()>0
    strength=sample.new_tensor(.4,requires_grad=True)
    def solve(value):
        return MagneticVolumeContrastResponse(system,(binding,),(current,),mu,
            (mu+MU_0*value*torch.diag(sample.new_tensor([.7,1.,.3])),),pair_batch=53)
    coupled=solve(strength)
    query=volume.vertices[volume.cells].mean(1)
    torch.testing.assert_close(coupled.fields(query,1)[0],coupled.cell_fields,rtol=1e-11,atol=1e-13)
    objective=coupled.fields(points,0)[0].square().sum()
    derivative=torch.autograd.grad(objective,strength)[0]
    step=1e-5
    finite=(solve(strength.detach()+step).fields(points,0)[0].square().sum()
            -solve(strength.detach()-step).fields(points,0)[0].square().sum())/(2*step)
    torch.testing.assert_close(derivative,finite,rtol=1e-5,atol=1e-14)
    assert not coupled.receipt['interface_continuity_verified']
    with pytest.raises(ValueError,match='declared region'):coupled.fields(points,1)
    with pytest.raises(ValueError,match='budget'):
        MagneticVolumeContrastResponse(system,(binding,),(current,),mu,(mu,),max_matrix_bytes=1)
    coupled.sources[0].polarization.detach().add_(1)
    with pytest.raises(ValueError,match='changed'):coupled.fields(points,0)


def test_tensor_sphere_uniform_magnetic_drive_refinement():
    errors=[]
    for level in (0,1,2):
        surface=sphere_mesh(radius=.1,subdivisions=level)
        vertices=torch.cat((surface.vertices,surface.vertices.new_zeros(1,3)))
        cells=torch.cat((torch.full((len(surface.faces),1),len(surface.vertices),device='cuda',dtype=torch.int64),surface.faces),dim=1)
        volume=TetrahedralVolumeMesh(vertices,cells,region_id=1)
        system,binding=system_from_volume(volume);mu=coefficient(vertices)
        exterior=MU_0*torch.eye(3,device='cuda',dtype=vertices.dtype)
        drive=vertices.new_tensor([.3,-.2,.7])
        response=MagneticVolumeContrastResponse(system,(binding,),(vertices.new_zeros(len(cells),3),),exterior,(mu,),background_h=drive)
        expected=torch.linalg.solve(mu+2*exterior,3*exterior@drive)
        actual=response.fields(vertices.new_tensor([[.013,.007,.019]]),1)[0][0]
        errors.append(float(((actual-expected).norm()/expected.norm()).detach()))
    print({'magnetic_tensor_sphere_errors':errors,'physical_threshold':.005,'physical_qualification':False})
    assert errors[2]<errors[1]<errors[0]


def test_two_body_contrast_action_source_superposition_and_ownership():
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.volume_mesh import bind_volume_region
    from witwin.maxwell.mom.static_tensor_polarization import StaticTensorPolarization
    first=mesh();sample=first.vertices
    second=TetrahedralVolumeMesh(first.vertices+sample.new_tensor([.3,0.,0.]),first.cells,region_id=2)
    system=SurfaceSystem.create([(i,'body'+str(i),RWGBasis(v.boundary),compile_medium(mw.Material(),3e8))
                                for i,v in enumerate((first,second))])
    bindings=tuple(bind_volume_region(v,system) for v in (first,second))
    ext=MU_0*torch.eye(3,device=sample.device,dtype=sample.dtype)
    interiors=(coefficient(sample),coefficient(sample)*1.4)
    a=sample.new_tensor([1.,-.4,.3]).expand(len(first.cells),3)
    b=sample.new_tensor([-.1,.8,.2]).expand_as(a)
    zero=torch.zeros_like(a)
    def solve(currents):return MagneticVolumeContrastResponse(system,bindings,currents,ext,interiors)
    left,right,both=solve((a,zero)),solve((zero,b)),solve((a,b))
    torch.testing.assert_close(both.cell_fields,left.cell_fields+right.cell_fields,rtol=1e-11,atol=1e-13)
    trial=torch.arange(both.cell_fields.numel(),device=sample.device,dtype=sample.dtype).reshape_as(both.cell_fields)/17
    correction=torch.zeros_like(trial)
    for binding,mu,h in zip(bindings,interiors,trial.split(both.counts)):
        p=h@(mu-ext).T
        correction=correction+StaticTensorPolarization(binding,ext,p).fields(both.centers)[1]
    torch.testing.assert_close((both.matrix@trial.flatten()).reshape_as(trial),trial-correction,rtol=1e-11,atol=1e-12)
    assert left.cell_fields[len(a):].norm()>1e-5
    for region,query in ((0,[[.2,.2,.1]]),(1,[[.021,.019,.023]]),(2,[[.321,.019,.023]])):
        q=sample.new_tensor(query)
        for total,l,r in zip(both.fields(q,region)[:2],left.fields(q,region)[:2],right.fields(q,region)[:2]):
            torch.testing.assert_close(total,l+r,rtol=1e-11,atol=1e-14)
    with pytest.raises(ValueError,match='region order'):
        MagneticVolumeContrastResponse(system,bindings[::-1],(a,b),ext,interiors)
    with pytest.raises(ValueError,match='representable'):
        MagneticVolumeContrastResponse(system,bindings,(zero,zero),ext,(ext,ext),background_h=sample.new_full((3,),1e308))
    both.matrix.detach().add_(1)
    with pytest.raises(ValueError,match='changed'):both.fields(sample.new_tensor([[.2,.2,.1]]),0)

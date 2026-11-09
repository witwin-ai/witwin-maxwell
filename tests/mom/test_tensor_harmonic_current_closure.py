"""Actual face-current closure with tensor surfaces and dielectric polarization."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.volume_mesh import bind_volume_region
from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator
from witwin.maxwell.mom.harmonic_mixed import MixedHarmonicClosure
from witwin.maxwell.mom.harmonic_current import surface_face_potentials
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup():
    vertices=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    volumes=tuple(mw.TetrahedralVolumeMesh(vertices+vertices.new_tensor([i*.24,0.,0.]),cells,region_id=i+1) for i in range(3))
    system=SurfaceSystem.create([(i,f'body{i}',RWGBasis(v.boundary),compile_medium(mw.Material(),3e8)) for i,v in enumerate(volumes)])
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    tensor=eye.new_tensor([[2.,.3,.1],[.3,3.,.2],[.1,.2,1.5]])
    op=StaticTensorFloatingOperator(system,EPSILON_0*eye,(EPSILON_0*tensor,None,3*EPSILON_0*eye))
    return op,tuple(bind_volume_region(v,system) for v in volumes)


@pytest.mark.parametrize('tensor_current',[False,True])
def test_tensor_surface_current_hb_reversed_volumes_and_implicit_gradient(monkeypatch,tensor_current):
    op,bindings=setup()
    sigma=op.centers.new_tensor([[.05,.01,0.],[.01,.08,.005],[0.,.005,.04]]) if tensor_current else .05
    closure=MixedHarmonicClosure(op,(bindings[2],bindings[1]),{2:sigma})
    generator=torch.Generator(device='cuda').manual_seed(39)
    background=torch.randn((3,2,3),device='cuda',dtype=torch.complex128,generator=generator)*.4
    background[0]=0;maps=background.new_zeros(3,6,6)
    weights=torch.cat([b.volume.volumes for b in (bindings[2],bindings[1])])
    chi=op.centers.new_tensor(.004,requires_grad=True)
    def model(value):return DenseHarmonicBalance(background,maps,maps,weights,3e8,
        chi2=torch.stack((value,value*0+.002)),chi3=.001,tpa_scale=chi.new_tensor([0.,.8]),dc_current_closure=closure)
    def forbidden(*args,**kwargs):raise AssertionError('HB must reuse prepared static response maps')
    with monkeypatch.context() as patch:
        patch.setattr(StaticTensorFloatingOperator,'solve_uniform',forbidden)
        base=model(chi);solved=base.solve(relative_tolerance=1e-10,absolute_tolerance=1e-12)
        assert solved.receipt['dc_formulation']=='mixed_tensor_dielectric_face_current_balance'
        assert not solved.receipt['physical_qualification']
        torch.testing.assert_close(solved.electric[1:],background[1:],rtol=0,atol=1e-14)
        derivative,=torch.autograd.grad(solved.electric[0,0].real.square().sum(),chi)
        step=1e-5
        values=[model(value).solve(solved.electric.detach(),relative_tolerance=1e-10,absolute_tolerance=1e-12).electric[0,0].real.square().sum()
                for value in (chi.detach()-step,chi.detach()+step)]
        torch.testing.assert_close(derivative,(values[1]-values[0])/(2*step),rtol=1e-5,atol=1e-11)
    dc=closure.solve(solved.electric[1:],base.tpa_scale)[0]
    boundary=surface_face_potentials(closure.current.blocks[0],bindings[1],dc.potential)
    direct=op.solve_uniform(chi.new_zeros(3),bindings=(bindings[2],),
        polarizations=(solved.sources.polarization[0,0:1].real,),boundary_potentials={2:boundary})
    point=bindings[2].volume.vertices[bindings[2].volume.cells].mean(1)
    torch.testing.assert_close(solved.electric[0,0:1].real,direct.fields(point,3)[1],rtol=1e-8,atol=1e-11)
    torch.testing.assert_close(solved.electric[0,1:2].real,dc.electric,rtol=1e-8,atol=1e-11)
    n=op.face_counts[0];m=op.face_counts[1]
    assert abs(float((direct.conormal[n:n+m]*op.areas[n:n+m]).sum().detach()))<1e-20
    assert direct.fields(bindings[0].volume.vertices.mean(0,keepdim=True),1)[1].norm()>0
    with pytest.raises(ValueError,match='budget'):op.prepare_cell_response((bindings[2],),max_response_bytes=1)
    with pytest.raises(ValueError,match='dielectric region'):
        op.solve_uniform(chi.new_zeros(3),bindings=(bindings[1],),polarizations=(chi.new_zeros(1,3),))
    with pytest.raises(ValueError,match='TPA current law'):closure.solve(background[1:],chi.new_tensor([.1,.8]))
    closure.maps['conormal_response'].add_(1)
    with pytest.raises(RuntimeError,match='mutated'):base.residual(solved.electric)
    print({'fixed_ac_maps':'zero_for_DC_closure_control','volume_regions':[3,2],
           'dc_residual':solved.receipt['final_residual'],'physical_qualification':False})


def test_current_only_volume_keeps_linear_tensor_surfaces():
    op,bindings=setup();closure=MixedHarmonicClosure(op,(bindings[1],),{2:.05})
    ac=torch.tensor([[[.4,.1,.2]],[[.1,.2,-.1]]],device='cuda',dtype=torch.complex128)
    tpa=op.centers.new_tensor([.8])
    actual=closure.electric(ac,tpa,polarization=op.centers.new_zeros(1,3))
    torch.testing.assert_close(actual,closure.solve(ac,tpa)[0].electric,rtol=0,atol=0)
    assert closure.dielectric_indices.numel()==0 and closure.maps['electric_response'].shape[0]==0

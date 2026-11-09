"""Mixed DC residual mapping and implicit derivatives in retained HB algebra."""
import runpy
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.static_mixed import MixedStaticOperator
from witwin.maxwell.mom.harmonic_mixed import MixedHarmonicClosure
from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
from witwin.maxwell.mom.harmonic_current import surface_face_potentials

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_mixed_dc_harmonic_residual_reversed_cells_and_implicit_gradient(monkeypatch):
    laplace,bindings=runpy.run_path('tests/mom/test_static_mixed.py')['setup']()
    operator=MixedStaticOperator(laplace,EPSILON_0,[None,3*EPSILON_0])
    closure=MixedHarmonicClosure(operator,bindings[::-1],{1:.05})
    sample=laplace.triangles
    generator=torch.Generator(device='cuda').manual_seed(39)
    background=torch.randn((3,2,3),device='cuda',dtype=torch.complex128,generator=generator)*.4
    background[0]=0
    maps=background.new_zeros(3,6,6)
    weights=torch.cat([binding.volume.volumes for binding in bindings[::-1]])
    chi=sample.new_tensor(.004,requires_grad=True)
    def model(value):
        coefficients=torch.stack((value,value*0+.002))
        return DenseHarmonicBalance(background,maps,maps,weights,3e8,chi2=coefficients,chi3=.001,
            tpa_scale=sample.new_tensor([0.,.8]),dc_current_closure=closure)
    def forbidden(*args,**kwargs):
        raise AssertionError('Mixed HB iterations must apply prepared static maps')
    with monkeypatch.context() as patch:
        patch.setattr(MixedStaticOperator,'solve',forbidden)
        base=model(chi)
        solved=base.solve(relative_tolerance=1e-10,absolute_tolerance=1e-12)
        assert solved.receipt['dc_formulation']=='mixed_dielectric_face_current_balance'
        assert not solved.receipt['physical_qualification']
        assert solved.electric[0].norm()>0
        torch.testing.assert_close(solved.electric[1:],background[1:],rtol=0,atol=1e-14)
        objective=solved.electric[0,0].real.square().sum()
        derivative,=torch.autograd.grad(objective,chi)
        step=1e-5
        values=[]
        for value in (chi.detach()-step,chi.detach()+step):
            result=model(value).solve(solved.electric.detach(),relative_tolerance=1e-10,absolute_tolerance=1e-12)
            values.append(result.electric[0,0].real.square().sum())
        torch.testing.assert_close(derivative,(values[1]-values[0])/(2*step),rtol=1e-5,atol=1e-11)
    dc=closure.solve(solved.electric[1:],base.tpa_scale)[0]
    boundary=surface_face_potentials(closure.current.blocks[0],bindings[0],dc.potential)
    direct=operator.solve([bindings[1]],[solved.sources.polarization[0,0:1].real],[boundary])
    point=bindings[1].volume.vertices[bindings[1].volume.cells].mean(1)
    torch.testing.assert_close(solved.electric[0,0:1].real,direct.fields(point,2)[1],rtol=1e-8,atol=1e-11)
    torch.testing.assert_close(solved.electric[0,1:2].real,dc.electric,rtol=1e-8,atol=1e-11)
    with pytest.raises(ValueError,match='TPA current law'):
        closure.solve(background[1:],sample.new_tensor([.1,.8]))
    closure.maps['electric_response'].add_(1.)
    with pytest.raises(RuntimeError,match='closure was mutated'):
        base.residual(solved.electric)
    print({'mixed_residual':solved.receipt['final_residual'],'dielectric_chi2_gradient':float(derivative),
           'cell_order':[2,1],'ac_maps':'zero_for_dc_algebra_test','physical_qualification':False})

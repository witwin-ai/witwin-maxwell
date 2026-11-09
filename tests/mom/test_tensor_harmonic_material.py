"""Tensor AC harmonics and analytic DC channels preserve public constitutive laws."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material, compile_tensor_harmonic_material
from witwin.maxwell.mom.tensor_medium import compile_tensor_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_scalar_nonlinear_background_and_sources_match_scalar_harmonics():
    material=mw.Material(eps_r=3.,mu_r=1.4,debye_poles=(mw.DebyePole(.3,1e-10),),
        nonlinearity=mw.NonlinearSusceptibility(chi2=.2,chi3=.1))
    scalar=compile_harmonic_material(material,3e8,3)
    tensor=compile_tensor_harmonic_material(material,3e8,3,device='cuda')
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    for left,right in zip(tensor.linear_samples,scalar.linear_samples):
        torch.testing.assert_close(left.epsilon_r,right.epsilon_r*eye,rtol=0,atol=0)
        torch.testing.assert_close(left.mu_r,right.mu_r*eye,rtol=0,atol=0)
    for name,scale in (('electric_dc',EPSILON_0),('magnetic_dc',MU_0)):
        torch.testing.assert_close(getattr(tensor,name).dielectric_sample(),
            getattr(scalar,name).dielectric_sample(scale)*eye.real,rtol=0,atol=0)
    field=eye.new_tensor([[[.2,.1,.3]],[[.1j,.2,.4]],[[.2,.3j,.1]],[[.05,.02j,.01]]])
    left,right=tensor.sources(field),scalar.sources(field)
    for name in ('polarization','dissipative_current','average_dissipation'):
        torch.testing.assert_close(getattr(left,name),getattr(right,name),rtol=0,atol=0)


def test_reciprocal_and_ferrite_harmonics_keep_full_tensor_samples():
    materials=(mw.Material(epsilon_tensor=mw.Tensor3x3(((2.,.2,0.),(.2,3.,.1),(0.,.1,4.)))),
        mw.GyromagneticFerrite(saturation_magnetization=1e5,bias_field=(3e4,4e4,0.),gilbert_damping=.03))
    for material in materials:
        compiled=compile_tensor_harmonic_material(material,3e8,3,device='cuda')
        for n,actual in enumerate(compiled.linear_samples,1):
            expected=compile_tensor_medium(material,n*3e8,device='cuda')
            torch.testing.assert_close(actual.epsilon_r,expected.epsilon_r,rtol=0,atol=0)
            torch.testing.assert_close(actual.mu_r,expected.mu_r,rtol=0,atol=0)
            assert actual.reciprocal==expected.reciprocal
        assert compiled.electric_dc.kind==compiled.magnetic_dc.kind=='finite_positive_dielectric'
        assert compiled.chi2==compiled.chi3==compiled.tpa_scale==0


def test_singular_dc_classification_and_sampling_admission():
    material=mw.Material(sigma_e_tensor=mw.DiagonalTensor3(0.,.2,0.),mu_drude_poles=(mw.DrudePole(1e8,0.),))
    compiled=compile_tensor_harmonic_material(material,3e8,2,device='cuda')
    assert compiled.electric_dc.kind=='conductive_limit'
    assert compiled.magnetic_dc.kind=='lossless_plasma_limit'
    for channel in (compiled.electric_dc,compiled.magnetic_dc):
        with pytest.raises(NotImplementedError):channel.dielectric_sample()
    for frequency,count in ((0.,1),(3e8,True),(float('inf'),2)):
        with pytest.raises(ValueError):compile_tensor_harmonic_material(material,frequency,count,device='cuda')
    with pytest.raises(ValueError,match='CUDA'):
        compile_tensor_harmonic_material(material,3e8,1,device='cpu')


def test_nonreciprocal_static_declaration_and_nonzero_tpa_source():
    gyro=mw.Material(epsilon_tensor=mw.ComplexTensor3x3(
        ((2.,.2j,0.),(-.2j,2.,0.),(0.,0.,2.)),reciprocal=False))
    sampled=compile_tensor_harmonic_material(gyro,3e8,2,device='cuda')
    assert all(not m.reciprocal for m in sampled.linear_samples)
    assert sampled.electric_dc.kind=='unresolved_complex_dc'
    with pytest.raises(NotImplementedError,match='unresolved_complex_dc'):
        sampled.electric_dc.dielectric_sample()
    material=mw.Material(eps_r=3.,kerr_chi3=.1,
        nonlinearity=(mw.NonlinearSusceptibility(chi2=.2,chi3=.3),mw.TwoPhotonAbsorption(beta=1e-9)))
    tensor=compile_tensor_harmonic_material(material,3e8,2,device='cuda')
    scalar=compile_harmonic_material(material,3e8,2)
    assert tensor.chi3==.4 and tensor.tpa_scale==scalar.tpa_scale>0
    field=torch.zeros(3,1,3,device='cuda',dtype=torch.complex128)
    field[1,0,0]=2+.5j
    left,right=tensor.sources(field),scalar.sources(field)
    assert bool(left.average_dissipation>0)
    for name in ('polarization','dissipative_current','average_dissipation'):
        torch.testing.assert_close(getattr(left,name),getattr(right,name),rtol=0,atol=0)

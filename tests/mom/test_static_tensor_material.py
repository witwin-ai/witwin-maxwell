"""Analytic material DC expansion, tensor replacement and singular admission."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0,MU_0
from witwin.maxwell.mom.static_tensor_material import compile_static_tensor_material
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.tensor_medium import compile_tensor_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def compile(material):return compile_static_tensor_material(material,device='cuda')


def test_tensor_bases_poles_and_scalar_nonlinear_background():
    rows=((2.,.3,.1),(.3,3.,.2),(.1,.2,1.5))
    material=mw.Material(eps_r=9.,mu_r=7.,epsilon_tensor=mw.Tensor3x3(rows),
        mu_tensor=mw.DiagonalTensor3(2.,3.,4.),debye_poles=(mw.DebyePole(.4,1e-10),),
        lorentz_poles=(mw.LorentzPole(.5,1e9,1e7),),mu_debye_poles=(mw.DebyePole(.3,1e-10),))
    sample=compile(material);eye=torch.eye(3,device='cuda',dtype=torch.float64)
    torch.testing.assert_close(sample.electric_dc.dielectric_sample(),EPSILON_0*(eye.new_tensor(rows)+.9*eye),rtol=1e-15,atol=0)
    torch.testing.assert_close(sample.magnetic_dc.dielectric_sample(),MU_0*torch.diag(eye.new_tensor([2.3,3.3,4.3])),rtol=1e-15,atol=0)
    nonlinear=mw.Material(eps_r=3.,nonlinearity=mw.NonlinearSusceptibility(chi2=.2))
    dc=compile(nonlinear).electric_dc.dielectric_sample()
    expected=compile_harmonic_material(nonlinear,3e8,1).electric_dc.dielectric_sample(EPSILON_0)
    torch.testing.assert_close(dc,expected*eye,rtol=0,atol=0)
    sample.electric_dc.finite_relative_part.add_(1)
    with pytest.raises(ValueError,match='changed'):sample.electric_dc.dielectric_sample()


def test_singular_channels_and_complex_dc_are_never_projected_to_spd():
    material=mw.Material(sigma_e=10.,sigma_e_tensor=mw.DiagonalTensor3(0.,.2,0.),
        drude_poles=(mw.DrudePole(1e8,2e7),),mu_drude_poles=(mw.DrudePole(1e8,0.),))
    sample=compile(material)
    pole=material.drude_poles[0]
    expected=torch.diag(torch.tensor([0.,.2,0.],device='cuda',dtype=torch.float64))+EPSILON_0*pole.plasma_angular_frequency**2/pole.gamma_angular_frequency*torch.eye(3,device='cuda')
    torch.testing.assert_close(sample.electric_dc.conductivity,expected,rtol=1e-7,atol=0)
    assert sample.electric_dc.kind=='conductive_limit'
    assert sample.magnetic_dc.kind=='lossless_plasma_limit'
    for channel in (sample.electric_dc,sample.magnetic_dc):
        with pytest.raises(NotImplementedError):channel.dielectric_sample()
    complex_material=mw.Material(epsilon_tensor=mw.ComplexTensor3x3(((2.,.2j,0.),(-.2j,2.,0.),(0.,0.,2.)),reciprocal=False))
    channel=compile(complex_material).electric_dc
    assert channel.kind=='unresolved_complex_dc'
    with pytest.raises(NotImplementedError,match='unresolved_complex_dc'):channel.dielectric_sample()


def test_ferrite_analytic_dc_and_small_frequency_limit():
    material=mw.GyromagneticFerrite(saturation_magnetization=1e5,bias_field=(3e4,4e4,0.),gilbert_damping=.03,mu_infinity=1.2)
    sample=compile(material);mu=sample.magnetic_dc.dielectric_sample()/MU_0
    bias=mu.new_tensor([.6,.8,0.]);eye=torch.eye(3,device='cuda',dtype=mu.dtype)
    expected=1.2*eye+2*(eye-bias[:,None]*bias[None,:])
    torch.testing.assert_close(mu,expected,rtol=1e-15,atol=1e-15)
    errors=[]
    for frequency in (1e5,1e4):
        ac=compile_tensor_medium(material,frequency,device='cuda')
        errors.append(float((ac.mu_r-mu).norm()/mu.norm()))
    assert errors[1]<errors[0]/5 and errors[1]<1e-4


def test_public_material_samples_drive_actual_static_interface():
    from test_static_tensor_interface import system
    from witwin.maxwell.mom.static_tensor_material import compile_static_tensor_interface
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    geometry=system()
    body=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,3.,4.),mu_tensor=mw.DiagonalTensor3(3.,4.,5.))
    for channel,scale,diagonal in (('electric',EPSILON_0,[2.,3.,4.]),('magnetic',MU_0,[3.,4.,5.])):
        op=compile_static_tensor_interface(geometry,mw.Material(),(body,),channel=channel)
        eye=torch.eye(3,device=geometry.device,dtype=geometry.dtype)
        explicit=StaticTensorInterfaceOperator(geometry,scale*eye,(scale*torch.diag(eye.new_tensor(diagonal)),))
        torch.testing.assert_close(op.matrix,explicit.matrix,rtol=0,atol=0)
        drive=eye.new_tensor([1.,-.3,.5])
        torch.testing.assert_close(op.solve_uniform(drive).potential,explicit.solve_uniform(drive).potential,rtol=0,atol=0)
    with pytest.raises(NotImplementedError,match='conductive_limit'):
        compile_static_tensor_interface(geometry,mw.Material(),(mw.Material(sigma_e=.1),))

"""Public nonlinear channels and unambiguous linear/DC material splitting."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material


def test_public_material_harmonics_sample_linear_background_once_and_classify_dc():
    material = mw.Material(eps_r=2.,sigma_e=.02,kerr_chi3=.1,
        nonlinearity=(mw.NonlinearSusceptibility(chi2=.2,chi3=.3),mw.TwoPhotonAbsorption(beta=1e-9)),
        debye_poles=(mw.DebyePole(.4,1e-10),),drude_poles=(mw.DrudePole(1e8,2e7),),
        lorentz_poles=(mw.LorentzPole(.5,1e9,1e7),))
    model = compile_harmonic_material(material,3e8,3)
    assert model.chi2 == .2 and model.chi3 == .4 and model.tpa_scale == material.tpa_sigma_scale
    for n,sample in enumerate(model.linear_samples,1):
        expected = compile_medium(mw.Material(eps_r=2.,sigma_e=.02,
            debye_poles=material.debye_poles,drude_poles=material.drude_poles,
            lorentz_poles=material.lorentz_poles),n*3e8)
        assert sample.epsilon_r == expected.epsilon_r and sample.mu_r == expected.mu_r
    expected_sigma = .02+EPSILON_0*(2*math.pi*1e8)**2/(2*math.pi*2e7)
    assert model.electric_dc.conductivity == pytest.approx(expected_sigma)
    assert model.electric_dc.finite_relative_part.real == pytest.approx(2.+.4+.5-25.)
    assert model.electric_dc.kind == 'conductive_limit'
    with pytest.raises(NotImplementedError,match='conductive_limit'):
        model.electric_dc.dielectric_sample(EPSILON_0)
    plasma = compile_harmonic_material(mw.Material(drude_poles=(mw.DrudePole(1e8,0.),)),3e8,1)
    assert plasma.electric_dc.kind == 'lossless_plasma_limit'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_public_nonlinearity_source_and_zero_chi_linear_contrast_falsification():
    field = torch.zeros(3,1,3,device='cuda',dtype=torch.complex128)
    field[1,0,0] = 2+.5j
    linear = compile_harmonic_material(mw.Material(eps_r=3.),3e8,2)
    sources = linear.sources(field)
    assert torch.count_nonzero(sources.polarization) == 0
    wrong_linear_contrast = EPSILON_0*(3.-1)*field
    assert torch.linalg.vector_norm(wrong_linear_contrast) > 0
    nonlinear = compile_harmonic_material(mw.Material(eps_r=3.,
        nonlinearity=mw.NonlinearSusceptibility(chi2=.2)),3e8,2)
    sources = nonlinear.sources(field)
    expected_dc = EPSILON_0*.2*abs(2+.5j)**2/2
    torch.testing.assert_close(sources.polarization[0,0,0].real,field.real.new_tensor(expected_dc),rtol=1e-12,atol=0)
    assert nonlinear.electric_dc.dielectric_sample(EPSILON_0) == 3*EPSILON_0

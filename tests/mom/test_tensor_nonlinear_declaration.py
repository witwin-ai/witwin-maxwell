"""Backend-independent tensor/nonlinear data with explicit solver admission."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
import witwin.maxwell.fdtd.runtime.materials as fdtd_materials
from witwin.maxwell.fdtd.runtime.materials import build_materials
from witwin.maxwell.mom.harmonic_material import compile_tensor_harmonic_material,compile_harmonic_material

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('tensor_kind',('epsilon_tensor','mu_tensor','sigma_e_tensor'))
@pytest.mark.parametrize('flavor',('chi2','chi3','tpa'))
def test_tensor_nonlinear_declaration_preserves_data_and_fdtd_rejects_before_compile(tensor_kind,flavor,monkeypatch):
    nonlinear=(mw.media.NonlinearSusceptibility(chi2=.002) if flavor=='chi2' else
               mw.media.NonlinearSusceptibility(chi3=.001) if flavor=='chi3' else
               mw.media.TwoPhotonAbsorption(beta=1e-12,n0=1.5))
    coefficient=mw.DiagonalTensor3(1.2,1.1,1.05)
    material=mw.Material(**{tensor_kind:coefficient},nonlinearity=nonlinear)
    assert material.is_nonlinear and material.is_anisotropic
    assert getattr(material,tensor_kind) is coefficient
    sampled=compile_tensor_harmonic_material(material,3e7,2,device='cuda')
    assert sampled.chi2==material.nonlinear_chi2
    assert sampled.chi3==material.nonlinear_chi3
    assert sampled.tpa_scale==material.tpa_sigma_scale
    assert all(sample.frequency==(n+1)*3e7 for n,sample in enumerate(sampled.linear_samples))
    if tensor_kind=='epsilon_tensor':
        torch.testing.assert_close(sampled.linear_samples[0].epsilon_r.real.diag(),
                                   torch.tensor([1.2,1.1,1.05],device='cuda',dtype=torch.float64))
    elif tensor_kind=='mu_tensor':
        torch.testing.assert_close(sampled.linear_samples[0].mu_r.real.diag(),
                                   torch.tensor([1.2,1.1,1.05],device='cuda',dtype=torch.float64))
    else:assert sampled.electric_dc.kind=='conductive_limit'
    def forbidden(*args,**kwargs):raise AssertionError('Unsupported FDTD combination compiled its grids')
    monkeypatch.setattr(fdtd_materials,'compile_materials',forbidden)
    # Prepared-scene stub: FDTD reads the declarations through prepared.scene.
    prepared=SimpleNamespace(scene=SimpleNamespace(structures=(SimpleNamespace(material=material),)))
    with pytest.raises(NotImplementedError,match='FDTD nonlinear anisotropic'):
        build_materials(SimpleNamespace(_compiled_sources=()),prepared)


def test_fdfd_and_public_mom_tensor_nonlinear_admission_stay_closed():
    from witwin.maxwell.capability import check_scene
    from witwin.maxwell.mom.tensor_harmonic_runtime import _admit
    from witwin.maxwell.mom.harmonic_config import HarmonicBalanceConfig
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    from test_tensor_volume_source import setup
    material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.,2.),
        nonlinearity=mw.media.NonlinearSusceptibility(chi2=.002))
    with pytest.raises(NotImplementedError,match='FDFD.*nonlinear'):
        check_scene(mw.Scene(structures=[mw.Structure(geometry=mw.Box(size=(.1,.1,.1)),material=material)]),'fdfd',frequencies=(1e9,)).raise_if_unsupported()
    _,binding=setup()
    sampled=compile_tensor_harmonic_material(material,3e7,1,device='cuda')
    exterior=compile_harmonic_material(mw.Material(),3e7,1)
    config=mw.mom.MoMConfig(dyadic_integration=DyadicIntegrationConfig(),
        harmonic_balance=HarmonicBalanceConfig(volumes=(binding.volume,),harmonics=1,branch_id='tensor-declaration'))
    with pytest.raises(NotImplementedError,match='scalar regional AC kernels'):
        _admit(config,(sampled,),exterior)


def test_tensor_tpa_requires_explicit_intensity_reference():
    with pytest.raises(ValueError,match='explicit reference.*n0'):
        mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.5,3.),
                    nonlinearity=mw.media.TwoPhotonAbsorption(beta=1e-12))

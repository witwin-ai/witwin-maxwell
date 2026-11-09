"""Independent beta units from actual compiled real-field cycle dissipation.

These tests evaluate material coefficients, not CPU Maxwell propagation.
"""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import C_0,EPSILON_0
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.runtime.materials import build_nonlinear_channels


def cycle_heat(kappa, peak):
    phase=np.arange(4096,dtype=np.float64)*(2*np.pi/4096)
    electric=peak*np.cos(phase)
    return float(np.mean(kappa*electric**4))


def reference_heat(specs, peak, host_index):
    # beta is defined by loss of intensity per length; dissipation density has
    # W/m^3 units. This oracle does not repeat the coefficient conversion.
    return sum(spec.beta*(.5*(host_index if spec.n0 is None else spec.n0)*EPSILON_0*C_0*peak**2)**2
        for spec in specs if isinstance(spec,mw.TwoPhotonAbsorption))


def compiled_material(material):
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.2,0.2),)*3),grid=mw.GridSpec.uniform(.1),
        boundary=mw.BoundarySpec.pec(),device='cpu',structures=[mw.Structure(
            mw.Box(position=torch.zeros(3,dtype=torch.float64),size=torch.full((3,),.4,dtype=torch.float64)),material)])
    prepared=prepare_scene(scene);model=prepared.compile_materials();edge=model['edge_components']['eps']
    runtime=SimpleNamespace(**{'eps_'+name:edge[name]*EPSILON_0 for name in ('Ex','Ey','Ez')})
    build_nonlinear_channels(runtime,model)
    return prepared,model,runtime


@pytest.mark.parametrize('n0',[None,1.,2.])
@pytest.mark.parametrize('peak',[25.,100.,400.])
def test_descriptor_cycle_heat_matches_declared_beta_intensity_law(n0,peak):
    spec=mw.TwoPhotonAbsorption(beta=4e-5,n0=n0)
    material=mw.Material(eps_r=2.25,nonlinearity=spec)
    assert cycle_heat(material.tpa_sigma_scale,peak)==pytest.approx(reference_heat([spec],peak,1.5),rel=2e-14)


@pytest.mark.parametrize('specs',[
    (mw.TwoPhotonAbsorption(beta=4e-5),),
    (mw.TwoPhotonAbsorption(beta=1e-5,n0=2.),),
    (mw.TwoPhotonAbsorption(beta=1e-5),mw.TwoPhotonAbsorption(beta=3e-5)),
    (mw.TwoPhotonAbsorption(beta=1e-5,n0=1.),mw.TwoPhotonAbsorption(beta=3e-5,n0=2.)),
    (mw.NonlinearSusceptibility(chi2=1e-7,chi3=1e-9),mw.TwoPhotonAbsorption(beta=4e-5)),
])
def test_actual_compiled_runtime_cycle_heat_preserves_explicit_index_and_additive_terms(specs):
    initialized=torch.cuda.is_initialized()
    _,model,runtime=compiled_material(mw.Material(eps_r=2.25,nonlinearity=specs))
    target=reference_heat(specs,100.,1.5)
    assert runtime.tpa_enabled and runtime.nonlinear_general_enabled
    for name in ('Ex','Ey','Ez'):
        coefficient=getattr(runtime,'tpa_sigma_'+name)
        kappa=float(coefficient[1,1,1])
        assert kappa>0
        # Runtime coefficient storage is float32; this bound is a stated dtype
        # budget, independent of the observed pass/fail value.
        assert cycle_heat(kappa,100.)==pytest.approx(target,rel=8*torch.finfo(coefficient.dtype).eps)
    assert torch.cuda.is_initialized()==initialized


@pytest.mark.parametrize('n0',[None,2.])
def test_physical_cycle_heat_beta_derivative_matches_independent_finite_difference(n0):
    beta=4e-5;step=beta*1e-4;peak=100.
    def work(value):return cycle_heat(mw.Material(eps_r=2.25,
        nonlinearity=mw.TwoPhotonAbsorption(beta=value,n0=n0)).tpa_sigma_scale,peak)
    measured=(work(beta+step)-work(beta-step))/(2*step)
    index=1.5 if n0 is None else n0
    expected=(.5*index*EPSILON_0*C_0*peak**2)**2
    assert measured==pytest.approx(expected,rel=2e-12)


def test_zero_tpa_material_has_no_fabricated_cycle_loss():
    _,model,runtime=compiled_material(mw.Material(eps_r=2.25))
    assert not runtime.tpa_enabled and not runtime.nonlinear_general_enabled
    assert not bool((model['tpa_sigma']!=0).any())
    assert runtime.tpa_sigma_Ez is None

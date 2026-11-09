"""Compiler/runtime coefficient, constitutive-work and AD-unit CPU controls.

No Maxwell field evolution, CUDA emulation or physical transmission is executed.
"""
from pathlib import Path
from decimal import Decimal, localcontext
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0, C_0
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.fdtd.runtime.materials import build_nonlinear_channels, build_materials
from witwin.maxwell.fdtd.material_pullback import component_node_gradients_from_yee_permittivity
from witwin.maxwell.fdtd.adjoint.core import _general_nonlinear_electric_coefficients
from witwin.maxwell.fdtd.runtime.stepping import update_electric_fields_bloch,update_electric_fields_bloch_xy_standard_z

BETA=4e-5



def compiled(beta=BETA):
    scene=mw.Scene(domain=mw.Domain(bounds=((-1.,1.),(-.3,.3),(-.3,.3))),grid=mw.GridSpec.uniform(.02),
        boundary=mw.BoundarySpec.pml(num_layers=8,strength=1.),device='cpu',
        structures=[mw.Structure(mw.Box(position=(-.2,0.,0.),size=(.3,.8,.8)),
            mw.Material(eps_r=2.25,nonlinearity=None if beta==0 else mw.TwoPhotonAbsorption(beta=beta)))])
    prepared=prepare_scene(scene);model=prepared.compile_materials()
    edge=model['edge_components']['eps']
    runtime=SimpleNamespace(eps_Ex=edge['Ex']*EPSILON_0,eps_Ey=edge['Ey']*EPSILON_0,eps_Ez=edge['Ez']*EPSILON_0)
    build_nonlinear_channels(runtime,model)
    return scene,prepared,model,runtime


def test_compiled_tpa_preserves_linear_background_and_active_real_runtime_channel():
    initialized=torch.cuda.is_initialized()
    _,_,model,runtime=compiled()
    _,_,linear_model,linear_runtime=compiled(beta=0.)
    assert runtime.tpa_enabled and runtime.nonlinear_general_enabled
    assert not linear_runtime.tpa_enabled and not linear_runtime.nonlinear_general_enabled
    for name in ('Ex','Ey','Ez'):
        torch.testing.assert_close(getattr(runtime,'eps_'+name),getattr(linear_runtime,'eps_'+name),rtol=0,atol=0)
    for key in ('eps','mu','sigma_e','sigma_m'):
        for name,value in model['edge_components'][key].items():
            torch.testing.assert_close(value,linear_model['edge_components'][key][name],rtol=0,atol=0)
    assert bool((model['tpa_sigma']>0).any()) and not bool((linear_model['tpa_sigma']!=0).any())
    assert torch.cuda.is_initialized()==initialized


@pytest.mark.parametrize('amplitude',[25.,100.,400.])
def test_independent_cycle_work_from_corrected_CPU_compiled_coefficient(amplitude):
    scene,_,_,runtime=compiled()
    actual_kappa=float(runtime.tpa_sigma_Ez.max())
    # The compiler sums prod(samples) identical FP32 sub-sample values: allow that accumulation plus one rounding.
    rel=(int(np.prod(scene.subpixel.samples))+1)*float(np.finfo(np.float32).eps)
    phase=np.arange(16384,dtype=np.float64)*(2*np.pi/16384)
    E=amplitude*np.cos(phase)
    actual_work=np.mean(actual_kappa*E**4)
    I=.5*1.5*EPSILON_0*C_0*amplitude**2
    ratio=actual_work/(BETA*I**2)
    # New CPU compilation is checked against beta's independent physical units.
    assert ratio==pytest.approx(1.,rel=rel)
    assert actual_kappa==(pytest.approx((2/3)*BETA*(1.5*EPSILON_0*C_0)**2,rel=rel))


def _coefficient_context(kappa,amplitude=400.):
    shape=(4,4,4)
    state={'Ex':torch.zeros((3,4,4),dtype=torch.float64),
           'Ey':torch.zeros((4,3,4),dtype=torch.float64),
           'Ez':torch.full((4,4,3),amplitude,dtype=torch.float64)}
    eps={key:torch.full_like(value,2.25*EPSILON_0) for key,value in state.items()}
    solver=SimpleNamespace(eps0=EPSILON_0,dt=3.4664996853903976e-11)
    for key,value in state.items():
        setattr(solver,'sigma_e_'+key,torch.zeros_like(value))
        setattr(solver,'c'+key.lower()+'_decay_external',torch.ones_like(value))
    zeros={key:torch.zeros_like(value) for key,value in state.items()}
    tpa={key:torch.ones_like(value)*kappa for key,value in state.items()}
    kwargs={prefix+'_'+key.lower():values[key] for prefix,values in [('eps',eps),('chi2',zeros),('chi3',zeros),('tpa',tpa)] for key in state}
    return solver,state,kwargs


def test_AD_replica_consumes_same_tpa_coefficient_and_preserves_derivative_units():
    _,_,_,runtime=compiled()
    kappa=runtime.tpa_sigma_Ez.max().double().detach().requires_grad_(True)
    solver,state,kwargs=_coefficient_context(kappa)
    coefficients=_general_nonlinear_electric_coefficients(solver,state,**kwargs)
    decay,curl=coefficients['Ez'];E=400.;eps=2.25*EPSILON_0
    half=.5*float(kappa.detach())*E**2*solver.dt/eps
    assert float(decay[1,1,1].detach())==pytest.approx((1-half)/(1+half),rel=2e-15)
    assert float(curl[1,1,1].detach())==pytest.approx((solver.dt/eps)/(1+half),rel=2e-15)
    derivative=torch.autograd.grad(decay[1,1,1],kappa)[0]
    expected=-(solver.dt*E**2/eps)/(1+half)**2
    assert float(derivative)==pytest.approx(expected,rel=2e-15)
    # This is local coefficient algebra only, not CPU FDTD time advancement.
    # Verify the independent mathematical CN energy identity in high precision;
    # subtracting two almost equal binary64 E^2 values creates an unrelated
    # cancellation floor. This is not an actual discrete-field energy audit.
    with localcontext() as context:
        context.prec=60
        D=Decimal.from_float
        dtD,epsD,ED,kD=D(solver.dt),D(eps),D(E),D(float(kappa.detach()))
        sigmaD=kD*ED**2
        halfD=sigmaD*dtD/(2*epsD)
        afterD=(1-halfD)/(1+halfD)*ED
        energy_drop=epsD*(ED**2-afterD**2)/(2*dtD)
        heat=sigmaD*((ED+afterD)/2)**2
        assert abs((energy_drop-heat)/heat)<Decimal('1e-45')


def test_node_to_edge_tpa_pullback_is_exact_transpose_without_eps0_rescale():
    node=torch.arange(60,dtype=torch.float64).reshape(3,4,5)/61
    node.requires_grad_(True)
    rng=torch.Generator().manual_seed(19)
    runtime=SimpleNamespace(eps_Ex=torch.empty((2,4,5)),eps_Ey=torch.empty((3,3,5)),eps_Ez=torch.empty((3,4,4)))
    model={key:torch.zeros_like(node) for key in ('kerr_chi3','chi2')};model['tpa_sigma']=node
    build_nonlinear_channels(runtime,model)
    grads={key:torch.randint(-5,6,getattr(runtime,'tpa_sigma_'+key).shape,generator=rng).double() for key in ('Ex','Ey','Ez')}
    value=sum((getattr(runtime,'tpa_sigma_'+key)*grads[key]).sum() for key in grads)
    true=torch.autograd.grad(value,node)[0]
    scatter=component_node_gradients_from_yee_permittivity(SimpleNamespace(Nx=3,Ny=4,Nz=5,device='cpu'),
        grad_eps_ex=grads['Ex'],grad_eps_ey=grads['Ey'],grad_eps_ez=grads['Ez'],eps0=1.)
    torch.testing.assert_close(true,scatter['x']+scatter['y']+scatter['z'],rtol=0,atol=0)


@pytest.mark.parametrize('function',[update_electric_fields_bloch,update_electric_fields_bloch_xy_standard_z])
def test_complex_nonlinear_evolution_rejects_before_any_field_or_kernel_access(function):
    with pytest.raises(NotImplementedError,match='instantaneous nonlinear media cannot evolve'):
        function(SimpleNamespace(nonlinear_enabled=True))


def test_actual_Bloch_material_admission_rejects_nonlinear_on_CPU_before_native_loading():
    scene=mw.Scene(domain=mw.Domain(bounds=((-1.,1.),)*3),grid=mw.GridSpec.uniform(.25),device='cpu',
        boundary=mw.BoundarySpec.bloch(wavevector=(.2,0.,0.)),
        structures=[mw.Structure(mw.Box(size=(.5,.5,.5)),mw.Material(eps_r=2.25,nonlinearity=mw.TwoPhotonAbsorption(beta=BETA)))])
    with pytest.raises(NotImplementedError,match='nonlinear media require the real-valued update kernels'):
        build_materials(SimpleNamespace(_compiled_sources=()),prepare_scene(scene))

"""Complex TEM controls independent of a time-domain propagation result."""
from types import SimpleNamespace
import cmath
import math
import pytest
import torch

from witwin.maxwell.constants import C_0
from witwin.maxwell.fdtd.excitation.modes.tem import _solve_pec_tem_mode_torch
from witwin.maxwell.postprocess import modal


def _cross_section(epsilon=2.25+.2j, permeability=1.+0j):
    occupancy = torch.zeros((17, 17),dtype=torch.float64)
    occupancy[0,:] = occupancy[-1,:] = 1
    occupancy[:,0] = occupancy[:,-1] = 1
    occupancy[7:10,7:10] = 1
    eps = tuple(torch.full_like(occupancy,epsilon,dtype=torch.complex128) for _ in range(3))
    mu = tuple(torch.full_like(occupancy,permeability,dtype=torch.complex128) for _ in range(3))
    return eps,mu,occupancy


@pytest.mark.parametrize('dtype', (torch.float32, torch.complex64))
def test_tem_phase_precision_is_independent_of_material_storage_dtype(dtype):
    eps,mu,occupancy=_cross_section(2.25+.2j if dtype.is_complex else 2.25,
                                   1.4+.03j if dtype.is_complex else 1.4)
    def stored(values):
        return tuple((value if dtype.is_complex else value.real).to(dtype) for value in values)
    eps,mu=stored(eps),stored(mu)
    k0=2*math.pi*1.23456789e9/C_0
    beta,_,_=_solve_pec_tem_mode_torch(eps,mu,occupancy,k0=k0,du=.001,dv=.001,
        mode_index=0,field_names=('Ey','Ez','Hy','Hz'))
    # The physical coefficients are the declared stored values, while phase
    # arithmetic must not incur a second float32 rounding of sqrt(eps*mu)*k0.
    expected=k0*cmath.sqrt(complex(eps[0][0,0])*complex(mu[0][0,0]))
    assert beta.dtype == (torch.complex128 if dtype.is_complex else torch.float64)
    assert complex(beta)==pytest.approx(expected,rel=1e-13)


@pytest.mark.parametrize('epsilon,permeability',[(2.25+.2j,1.+0j),(2.25+.2j,1.4+.03j)])
def test_homogeneous_complex_tem_uses_one_real_potential_and_passive_native_beta(epsilon,permeability):
    eps,mu,occupancy = _cross_section(epsilon,permeability)
    frequency,dt,dx = 1e9,2e-12,.0025
    k0 = 2*math.pi*frequency/C_0
    args=dict(k0=k0,du=.001,dv=.001,mode_index=0,field_names=('Ey','Ez','Hy','Hz'))
    beta,fields,_ = _solve_pec_tem_mode_torch(eps,mu,occupancy,time_step=dt,normal_spacing=dx,**args)
    _,vacuum,_ = _solve_pec_tem_mode_torch(*_cross_section(1.+0j,1.+0j),**args)
    # Constant complex media leave Laplace's boundary-value solution unchanged.
    for name in ('Ey','Ez'):
        torch.testing.assert_close(fields[name],vacuum[name])
    for name in ('Hy','Hz'):
        torch.testing.assert_close(fields[name],vacuum[name]*cmath.sqrt(epsilon/permeability))
    expected=2/dx*cmath.asin(dx/(C_0*dt)*math.sin(math.pi*frequency*dt)*cmath.sqrt(epsilon*permeability))
    assert complex(beta)==pytest.approx(expected,rel=1e-13)
    assert beta.real>0 and beta.imag>0


def test_homogeneous_complex_tem_rejects_layered_or_anisotropic_fill():
    eps,mu,occupancy=_cross_section()
    eps[0][3:6,3:6] += .01j
    with pytest.raises(NotImplementedError,match='homogeneous isotropic'):
        _solve_pec_tem_mode_torch(eps,mu,occupancy,k0=20.,du=.001,dv=.001,mode_index=0,field_names=('Ey','Ez','Hy','Hz'))


@pytest.mark.parametrize('forward,backward',[(1.+0j,0j),(0j,1.+0j),(.7+.2j,-.3+.6j)])
def test_complex_modal_projection_recovers_physical_counterpropagating_fields(monkeypatch,forward,backward):
    coordinate=torch.linspace(0.,1.,5,dtype=torch.float64)
    electric=torch.zeros((5,5,3),dtype=torch.complex128)
    magnetic=torch.zeros_like(electric)
    impedance=40.-12.j
    electric[...,1]=3.
    magnetic[...,2]=3./impedance
    total_e=(forward+backward)*electric
    total_h=(forward-backward)*magnetic
    monitor={'axis':'x','position':0.,'normal_direction':'+','frequency':1e9,'y':coordinate,'z':coordinate}
    for index,axis in enumerate('xyz'):
        monitor['E'+axis]=total_e[...,index]
        monitor['H'+axis]=total_h[...,index]
    source=SimpleNamespace(direction='+',mode_index=0,polarization_axis='y')
    monkeypatch.setattr(modal,'_mode_overlap_context',lambda result: None)
    monkeypatch.setattr(modal,'_resolve_mode_source',lambda *args,**kwargs:source)
    monkeypatch.setattr(modal,'_reference_mode_fields',lambda *args,**kwargs:(coordinate,coordinate,electric,magnetic,{'beta':20.,'effective_index':1.}))
    result=modal._compute_mode_overlap_from_payload(SimpleNamespace(scene=None), 'plane',monitor)
    assert complex(result['amplitude_forward'])==pytest.approx(forward,abs=1e-14)
    assert complex(result['amplitude_backward'])==pytest.approx(backward,abs=1e-14)
    # Reactive interference contributes to net physical power for complex Z.
    expected=.5*((3*(forward+backward))*(3/impedance*(forward-backward)).conjugate()).real
    assert float(result['total_power'])==pytest.approx(expected,abs=1e-14)


def test_complex_cw_mode_patch_preserves_admittance_phase(monkeypatch):
    from witwin.maxwell.fdtd.excitation import tfsf_common
    captured={}
    monkeypatch.setattr(tfsf_common,'build_source_term',lambda solver,**kwargs:captured.update(kwargs) or kwargs)
    scale=torch.tensor([1.-.2j,-.7+.8j],dtype=torch.complex128)
    delay=torch.tensor([.13,.29],dtype=torch.float64)
    omega=3.1
    solver=SimpleNamespace(Ex=torch.empty(0,dtype=torch.float64))
    tfsf_common.build_term_from_profile(solver,field_name='Ez',offsets=(0,0,0),scale=scale,
        delay_patch=delay,activation_delay_patch=None,source_time={'kind':'cw'},omega=omega)
    for time in (0.,.27,1.6):
        actual=captured['cw_cos_patch']*math.cos(omega*time)+captured['cw_sin_patch']*math.sin(omega*time)
        expected=(scale*torch.exp(1j*omega*(delay-time))).real
        torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)


def test_complex_port_extraction_preserves_physical_vi_and_kurokawa_power(monkeypatch):
    from witwin.maxwell import waveport_sweep
    from witwin.maxwell.network_math import power_waves_to_voltage_current
    # A mode phase rotation and reactive Z expose both projection and power-wave errors.
    orientation=torch.tensor([[[cmath.exp(.37j)]]],dtype=torch.complex128)
    vm,im=3.+.2j,(3.+.2j)/(40.-12j)
    forward,backward=.7+.4j,-.2+.3j
    prepared=SimpleNamespace(
        port=SimpleNamespace(name='port'),compiled=SimpleNamespace(modes=(object(),)),
        mode_data=(({'beta_complex':20.+1j},),),
        tracking=SimpleNamespace(orientation=orientation),
        characteristic_impedance=torch.tensor([[40.-12j]],dtype=torch.complex128))
    result=SimpleNamespace(prepared_scene=None,monitor=lambda *args,**kwargs:{
        'mode_power':torch.tensor(1.),'amplitude_forward':torch.tensor(forward,dtype=torch.complex128),
        'amplitude_backward':torch.tensor(backward,dtype=torch.complex128)})
    # Ratio reductions read through Result._spectral_ratio_view(); a non-mixed result views itself.
    result._spectral_ratio_view=lambda:result
    monkeypatch.setattr(waveport_sweep,'_integrate_mode_geometry',lambda *args:(torch.tensor(vm,dtype=torch.complex128),torch.tensor(im,dtype=torch.complex128)))
    a,b=waveport_sweep._extract_tracked_waves(result,SimpleNamespace(prepared_ports=(prepared,)),0)
    voltage,current=power_waves_to_voltage_current(a,b,40.-12j)
    torch.testing.assert_close(voltage,torch.tensor([vm*(forward+backward)],dtype=torch.complex128))
    torch.testing.assert_close(current,torch.tensor([im*(forward-backward)],dtype=torch.complex128))
    torch.testing.assert_close(a.abs().square()-b.abs().square(),.5*(voltage*current.conj()).real)

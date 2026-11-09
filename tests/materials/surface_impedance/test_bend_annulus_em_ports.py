"""Independent TE Hankel phase, flux and source-free radial port controls."""
import math
import numpy as np
import pytest
from scripts.dev.fdtd.bend_annulus_em_audit import em_hankel_basis,project_em,modal_power_unit
from scripts.dev.fdtd.copper_cylinder_reference import shell_scattering,MU_0


@pytest.mark.parametrize('order',(0,1,2))
@pytest.mark.parametrize('radius',(.23,.3,.37))
def test_orientation_complex_phase_and_hankel_wronskian_power(order,radius):
    reference=.3;frequency=1e9;area=.012**2
    amplitudes=np.asarray((1.2+.7j,-.3+.2j))*np.exp(.83j)
    electric,magnetic=em_hankel_basis(radius,reference,frequency,order)@amplitudes
    recovered=project_em(electric,magnetic,radius,reference,frequency,order)
    np.testing.assert_allclose(recovered,amplitudes,rtol=3e-15,atol=3e-15)
    physical_flux=-.5*np.real(electric*np.conj(magnetic))*area
    exact=modal_power_unit(reference,frequency,area)*(abs(amplitudes[0])**2-abs(amplitudes[1])**2)
    assert physical_flux==pytest.approx(exact,rel=3e-14)
    for component,sign in ((0,1),(1,-1)):
        e,h=em_hankel_basis(radius,reference,frequency,order)[:,component]
        assert -np.real(e*np.conj(h))*sign>0


@pytest.mark.parametrize('order',(0,1,2))
def test_two_port_em_reconstruction_preserves_small_absorption_and_coherent_phase(order):
    frequency=1e9;reference=.3;sigma=5.8e7
    thickness=math.sqrt(2/(2*math.pi*frequency*MU_0*sigma))
    exact=shell_scattering(reference,thickness,frequency,sigma,order=order,polarization='te')
    drives=np.asarray(((1.,.2+.1j),(.3j,1-.2j)))
    output=exact@drives;recovered_input=[];recovered_output=[]
    for a,b in zip(drives.T,output.T):
        inner=np.asarray((a[0],b[0]));outer=np.asarray((b[1],a[1]))
        traces=[]
        for radial,amplitudes in ((reference-reference/6,inner),(reference+reference/6,outer)):
            electric,magnetic=em_hankel_basis(radial,reference,frequency,order)@amplitudes
            traces.append(project_em(electric,magnetic,radial,reference,frequency,order))
        recovered_input.append((traces[0][0],traces[1][1]));recovered_output.append((traces[0][1],traces[1][0]))
    actual=np.linalg.solve(np.asarray(recovered_input),np.asarray(recovered_output)).T
    np.testing.assert_allclose(actual,exact,rtol=4e-14,atol=4e-15)
    expected_absorption=1-(abs(exact)**2).sum(0)
    absorption=1-(abs(actual)**2).sum(0)
    np.testing.assert_allclose(absorption,expected_absorption,rtol=2e-10,atol=1e-15)
    assert np.all(absorption>0)


def test_real_stagger_interpolation_preserves_lossless_standing_wave_zero_flux():
    reference=.3;frequency=1e9;order=1;radius=.35
    amplitudes=np.asarray((1.,np.exp(.7j)))
    electric,_=em_hankel_basis(radius,reference,frequency,order)@amplitudes
    for width in (.005,.0025,.00125):
        # Native E_s at the radial node and H_z at its two adjacent halves.
        magnetic=sum((em_hankel_basis(radius+shift,reference,frequency,order)@amplitudes)[1]
            for shift in (-width/2,width/2))/2
        assert abs(np.real(electric*np.conj(magnetic)))<2e-13
        recovered=project_em(electric,magnetic,radius,reference,frequency,order)
        assert abs(abs(recovered[0])**2-abs(recovered[1])**2)<2e-15


def test_retained_native_E_H_time_staggers_have_one_shared_complex128_phase():
    import torch
    from scripts.dev.fdtd.bend_annulus_time_control import gpu_phasor
    frequency=1e9;dt=1/(256*frequency);steps=np.arange(8192)
    amplitude=1.2+.7j
    values=[]
    for component,offset in (('Ex',1.),('Hz',.5)):
        raw=(amplitude*np.cos(2*np.pi*frequency*dt*(steps+offset))).astype(np.complex64)
        values.append(gpu_phasor(torch.from_numpy(raw),steps,3072,8192,dt,frequency,component))
    expected=amplitude*np.exp(-2j*np.pi*frequency*dt)
    for value in values:
        assert complex(value)==pytest.approx(expected,rel=3e-8,abs=3e-8)
    assert complex(values[0])==pytest.approx(complex(values[1]),rel=3e-8,abs=3e-8)

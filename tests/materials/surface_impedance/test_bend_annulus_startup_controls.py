"""Independent clock and finite-startup contracts for private diagnostics."""
import math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import jnp_zeros
from scripts.dev.fdtd.bend_annulus_smooth_start import envelope
from scripts.dev.fdtd.bend_annulus_window_audit import phasor


@pytest.mark.parametrize('component,offset',(('Ex',0.),('Hz',-.5)))
def test_fixed_window_fourier_recovers_exact_native_complex_carrier(component,offset):
    frequency=1e9;dt=1/(256*frequency);n=torch.arange(44*256,dtype=torch.float64)
    expected=1.2+.7j
    values=(expected*.5*torch.exp(-2j*torch.pi*frequency*dt*(n+offset))).to(torch.complex64)
    record=dict(component=component,steps=n.to(torch.int64),values=values)
    for start,end in ((0,20),(24,44),(36,44)):
        actual=phasor(record,start,end,dt,frequency)
        assert complex(actual)==pytest.approx(expected,rel=3e-8,abs=3e-8)
    with pytest.raises(ValueError,match='lacks retained'):
        phasor(record,40,50,dt,frequency)


@pytest.mark.parametrize('duration_ns,maximum',((20.,1e-4),(40.,2e-5)))
def test_predeclared_c2_startup_suppresses_independent_cavity_sideband(duration_ns,maximum):
    # The analytic derivative of the unit ramp is the Beta(3,3) density. Its
    # Fourier transform predicts startup excitation relative to an abrupt step.
    cavity_frequency=jnp_zeros(1,1)[0]*299792458/(2*math.pi*.3)
    argument=2*math.pi*(1e9-cavity_frequency)*duration_ns*1e-9
    real=quad(lambda q:30*q*q*(1-q)**2*math.cos(argument*q),0,1,epsabs=1e-14)[0]
    imag=quad(lambda q:30*q*q*(1-q)**2*math.sin(argument*q),0,1,epsabs=1e-14)[0]
    assert abs(complex(real,imag))<maximum
    # No discontinuity at the physical endpoint, no overshoot, fixed plateau.
    duration=duration_ns*1e-9
    assert envelope(-duration,duration)==0.
    assert envelope(0.,duration)==0.
    assert envelope(duration,duration)==1.
    assert envelope(2*duration,duration)==1.
    samples=np.asarray([envelope(t,duration)for t in np.linspace(0,duration,101)])
    assert np.all(np.diff(samples)>=0)

"""Independent finite-copper two-port transformation and passive CN storage."""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.curved_copper import compile_copper_channels
from tests.materials.surface_impedance.test_copper_foil import DELTA, FREQUENCY, _film_transfer


@pytest.mark.parametrize("ratio", (.01,.1,1.,3.,10.,30.))
@pytest.mark.parametrize("surroundings", ((1.,1.),(1.,4.),(4.,4.)))
def test_separate_electric_magnetic_channels_match_independent_full_maxwell_film(ratio,surroundings):
    material=mw.CopperFoil(thickness=ratio*DELTA,frequency_range=(.5e9,2e9))
    # The transmission floor is stricter than the slab model's default band
    # contract, so request a correspondingly tighter analytic mode reduction.
    channels=compile_copper_channels(material,tolerance=2e-5)
    ye,zm=(value.item() for value in channels.evaluate([FREQUENCY]))
    reflection,transmission,zi,zt=_film_transfer(material.thickness,incident_eps=surroundings[0],outgoing_eps=surroundings[1])
    matrix=np.asarray(((-1/zi-ye/2,-1/zt-ye/2),(-1-zm/(2*zi),1+zm/(2*zt))))
    rhs=np.asarray((ye/2-1/zi,1-zm/(2*zi)))
    actual_r,actual_t=np.linalg.solve(matrix,rhs)
    np.testing.assert_allclose((actual_r,actual_t),(reflection,transmission),rtol=5e-4,atol=2e-9)
    physical_absorption=1-abs(reflection)**2-zi/zt*abs(transmission)**2
    channel_absorption=1-abs(actual_r)**2-zi/zt*abs(actual_t)**2
    assert channel_absorption > 0
    assert abs(channel_absorption-physical_absorption)/physical_absorption < 5e-4
    assert ye.real >= 0 and zm.real >= 0
    assert channels.electric_direct >= 0 and channels.magnetic_inductance >= 0


@pytest.mark.parametrize("device",("cpu","cuda"))
def test_both_channel_midpoint_states_have_exact_positive_storage_and_heat(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    material=mw.CopperFoil(thickness=torch.tensor(2e-6,device=device,dtype=torch.float64),
        frequency_range=(.5e9,2e9))
    channels=compile_copper_channels(material)
    dt=5e-12
    coefficients=channels.midpoint_coefficients(dt)
    ze=torch.linspace(-1.,1.,len(channels.rates),device=device,dtype=torch.float64)*1e-4
    zm=-ze.clone()
    r,a,b=channels.rates,channels.electric_residue,channels.magnetic_residue
    for step in range(64):
        ve=ze.new_tensor(math.sin(.37*step))
        vh=ze.new_tensor(math.cos(.21*step)*.01)
        next_e=coefficients["decay"]*ze+coefficients["electric_drive"]*ve
        next_m=coefficients["decay"]*zm+coefficients["magnetic_drive"]*vh
        mean_e,mean_m=(next_e+ze)/2,(next_m+zm)/2
        ie=channels.electric_direct*ve+(a.sqrt()*mean_e).sum()
        im=(b*vh-(b*r).sqrt()*mean_m).sum()
        heat_e=dt*(channels.electric_direct*ve.square()+(r*mean_e.square()).sum())
        heat_m=dt*((b.sqrt()*vh-r.sqrt()*mean_m).square()).sum()
        change_e=(next_e.square()-ze.square()).sum()/2
        change_m=(next_m.square()-zm.square()).sum()/2
        torch.testing.assert_close(change_e+heat_e,dt*ve*ie,rtol=1e-10,atol=1e-20)
        torch.testing.assert_close(change_m+heat_m,dt*vh*im,rtol=1e-10,atol=1e-20)
        assert heat_e >= 0 and heat_m >= 0
        # The proper current history must agree with the eliminated recurrence.
        torch.testing.assert_close(ie,coefficients["electric_gain"]*ve+(coefficients["electric_history"]*ze).sum(),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(im,coefficients["magnetic_gain"]*vh+(coefficients["magnetic_history"]*zm).sum(),rtol=1e-12,atol=1e-12)
        ze,zm=next_e,next_m


def test_channel_parameters_retain_native_continuous_material_gradients():
    thickness=torch.tensor(2e-6,dtype=torch.float64,requires_grad=True)
    conductivity=torch.tensor(5.8e7,dtype=torch.float64,requires_grad=True)
    def objective(t,sigma):
        channel=compile_copper_channels(mw.CopperFoil(thickness=t,conductivity=sigma,
            frequency_range=(.5e9,2e9),mode_count=32))
        electric,magnetic=channel.evaluate([1e9])
        return electric.real.sum()+1e4*magnetic.real.sum()
    value=objective(thickness,conductivity)
    gradients=torch.autograd.grad(value,(thickness,conductivity))
    for parameter,gradient in zip((thickness,conductivity),gradients):
        relative_errors=[]
        for relative_step in (1e-3,3e-4,1e-4):
            delta=parameter.detach()*relative_step
            arguments=[thickness.detach(),conductivity.detach()]
            index=0 if parameter is thickness else 1
            arguments[index]=parameter.detach()+delta
            upper=objective(*arguments)
            arguments[index]=parameter.detach()-delta
            lower=objective(*arguments)
            finite_difference=(upper-lower)/(2*delta)
            relative_errors.append(float((finite_difference-gradient).abs()/gradient.abs()))
        assert min(relative_errors) < 2e-7

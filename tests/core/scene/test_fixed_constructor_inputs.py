"""Unsupported scalar inputs reject before coercion; supported AD stays intact."""
from dataclasses import dataclass

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


@dataclass(frozen=True)
class Case:
    name: str
    construct: object
    value: object


CASES = (
    Case('cw_frequency',lambda x:mw.CW(3e9*x),lambda x:x.frequency),
    Case('cw_amplitude',lambda x:mw.CW(3e9,amplitude=x),lambda x:x.amplitude),
    Case('cw_phase',lambda x:mw.CW(3e9,phase=x),lambda x:x.phase),
    Case('pulse_frequency',lambda x:mw.GaussianPulse(3e9*x,1e9),lambda x:x.frequency),
    Case('pulse_fwidth',lambda x:mw.GaussianPulse(3e9,1e9*x),lambda x:x.fwidth),
    Case('pulse_amplitude',lambda x:mw.GaussianPulse(3e9,1e9,amplitude=x),lambda x:x.amplitude),
    Case('pulse_phase',lambda x:mw.GaussianPulse(3e9,1e9,phase=x),lambda x:x.phase),
    Case('pulse_delay',lambda x:mw.GaussianPulse(3e9,1e9,delay=x*1e-9),lambda x:x.delay),
    Case('ricker_frequency',lambda x:mw.RickerWavelet(3e9*x),lambda x:x.frequency),
    Case('ricker_amplitude',lambda x:mw.RickerWavelet(3e9,amplitude=x),lambda x:x.amplitude),
    Case('ricker_delay',lambda x:mw.RickerWavelet(3e9,delay=x*1e-9),lambda x:x.delay),
    Case('custom_amplitude',lambda x:mw.CustomSourceTime((0.,1.),(0.,1.),amplitude=x),lambda x:x.amplitude),
    Case('custom_frequency',lambda x:mw.CustomSourceTime((0.,1.),(0.,1.),characteristic_frequency=x*3e9),lambda x:x.characteristic_frequency),
    Case('custom_times',lambda x:mw.CustomSourceTime((0.,x),(0.,1.)),lambda x:x.times),
    Case('custom_samples',lambda x:mw.CustomSourceTime((0.,1.),(0.,x)),lambda x:x.amplitudes),
    Case('sigma_m',lambda x:mw.Material(sigma_m=x),lambda x:x.sigma_m),
    Case('domain',lambda x:mw.Domain(bounds=((-x,x),(-.02,.02),(-.02,.02))),lambda x:x.bounds),
    Case('uniform',lambda x:mw.GridSpec.uniform(x*.01),lambda x:(x.dx,x.dy,x.dz)),
    Case('anisotropic',lambda x:mw.GridSpec.anisotropic(.01,.02*x,.03),lambda x:(x.dx,x.dy,x.dz)),
    Case('custom_x',lambda x:mw.GridSpec.custom(x*torch.tensor((-1.,0.,1.)),(-1.,0.,1.),(-1.,0.,1.)),lambda x:x.x_coords),
    Case('custom_y',lambda x:mw.GridSpec.custom((-1.,0.,1.),(-x,0.,x),(-1.,0.,1.)),lambda x:x.y_coords),
    Case('custom_z',lambda x:mw.GridSpec.custom((-1.,0.,1.),(-1.,0.,1.),x*torch.tensor((-1.,0.,1.))),lambda x:x.z_coords),
    Case('auto_steps',lambda x:mw.GridSpec.auto(min_steps_per_wavelength=10*x),lambda x:x.min_steps_per_wavelength),
    Case('auto_wavelength',lambda x:mw.GridSpec.auto(wavelength=.1*x),lambda x:x.wavelength),
    Case('auto_ratio',lambda x:mw.GridSpec.auto(max_ratio=1+.4*x),lambda x:x.max_ratio),
    Case('observer',lambda x:mw.PointMonitor('point',(x,0.,0.)),lambda x:x.position),
)


@pytest.mark.parametrize('case',CASES,ids=lambda case:case.name)
@pytest.mark.parametrize('shared_epsilon',(False,True),ids=('direct_leaf','shared_epsilon_leaf'))
def test_trainable_scalar_and_shared_material_dependency_reject_before_coercion(case,shared_epsilon):
    parameter=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    value=parameter
    if shared_epsilon:
        material=mw.Material(eps_r=parameter.exp())
        assert material.eps_r.requires_grad
        value=1+parameter*0  # A zero numerical derivative must not hide the unsupported path.
    with pytest.raises(NotImplementedError,match='Trainable'):
        case.construct(value)


@pytest.mark.parametrize('case',CASES,ids=lambda case:case.name)
def test_fixed_float_and_nongrad_tensor_keep_same_public_value(case):
    expected=case.value(case.construct(1.))
    actual=case.value(case.construct(torch.tensor(1.,dtype=torch.float64)))
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=0)


def test_registered_electric_and_copper_material_graphs_remain_differentiable():
    parameter=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    material=mw.Material(eps_r=parameter.exp(),sigma_e=parameter.square())
    copper=mw.CopperFoil(thickness=parameter*1e-6,conductivity=parameter*1e6,
        frequency_range=(2e9,4e9),mode_count=16)
    value=material.eps_r+material.sigma_e+copper.thickness+copper.conductivity
    actual,=torch.autograd.grad(value,parameter)
    torch.testing.assert_close(actual,parameter.exp()+2*parameter+1e-6+1e6)


def test_registered_port_excitation_and_batch_scale_graphs_remain_differentiable():
    parameter=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    excitation=mw.PortExcitation('feed',amplitude=parameter,source_time=mw.CW(3e9))
    batch=mw.BatchParameters(source_scales={'drive':parameter.square().expand(2)})
    actual,=torch.autograd.grad(excitation.amplitude.abs().square().sum()+batch.source_scales['drive'].sum(),parameter)
    torch.testing.assert_close(actual,6*parameter)


def test_registered_circuit_waveform_keeps_tensor_ancestry():
    parameter=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    waveform=mw.SineWaveform(0.,parameter,3e9)
    actual,=torch.autograd.grad(waveform.amplitude.square(),parameter)
    torch.testing.assert_close(actual,2*parameter)


def test_fixed_current_dataset_keeps_existing_numpy_contract():
    dataset=mw.CurrentDataset(((-.01,.01),)*3,{'Jx':torch.ones(2,2,2)})
    np.testing.assert_array_equal(dataset.components['Jx'],np.ones((2,2,2)))


def test_fixed_source_time_waveform_compiles_without_numeric_change():
    from witwin.maxwell.sources import compile_source_time,evaluate_source_time
    for factory in (lambda value:mw.CW(3e9,amplitude=value,phase=.37),
                    lambda value:mw.GaussianPulse(3e9,1e9,amplitude=value,phase=.37),
                    lambda value:mw.RickerWavelet(3e9,amplitude=value)):
        fixed=compile_source_time(factory(2.),default_frequency=3e9)
        tensor=compile_source_time(factory(torch.tensor(2.)),default_frequency=3e9)
        for time in (0.,1e-10,3e-9):
            assert evaluate_source_time(tensor,time)==pytest.approx(evaluate_source_time(fixed,time),rel=1e-14,abs=0)


@pytest.mark.parametrize('kind',('observer','grid_nodes','custom_samples'))
def test_object_array_cannot_hide_a_trainable_scalar(kind):
    values=np.empty(3,dtype=object)
    values[0]=0.
    values[1]=torch.tensor(1.,requires_grad=True)
    values[2]=2.
    with pytest.raises(NotImplementedError,match='Trainable'):
        if kind=='observer':mw.PointMonitor('point',values)
        elif kind=='grid_nodes':mw.GridSpec.custom(values,(-1.,0.,1.),(-1.,0.,1.))
        else:mw.CustomSourceTime((0.,1.,2.),values)

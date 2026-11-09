"""Assignment audit of the admitted declarative fixed-control families."""
import pytest
import torch
import witwin.maxwell as mw


@pytest.mark.parametrize('kind',('direct','pml','faces'))
@pytest.mark.parametrize('slot',('num_layers','strength','bloch_wavevector'))
def test_boundary_received_trainable_controls_reject_before_replacement(kind,slot):
    value=torch.tensor(4. if slot=='num_layers' else 1.,requires_grad=True)
    options={slot:(value*0,0.,0.) if slot=='bloch_wavevector' else value}
    with pytest.raises(NotImplementedError,match='fixed'):
        if kind=='pml' and slot=='bloch_wavevector':mw.BoundarySpec(kind='pml',num_layers=4,**options)
        elif kind=='pml':mw.BoundarySpec.pml(**options)
        elif kind=='faces':mw.BoundarySpec.faces(default='pml',num_layers=options.pop('num_layers',4),**options)
        else:mw.BoundarySpec(kind='pml',num_layers=options.pop('num_layers',4),**options)


@pytest.mark.parametrize('slot',('verbose','lazy_meshgrid'))
def test_scene_received_trainable_execution_flags_reject_before_bool(slot):
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.Scene(domain=mw.Domain(bounds=((- .02,.02),)*3),grid=mw.GridSpec.uniform(.005),device='cpu',**{slot:torch.tensor(0.,requires_grad=True)})


def test_received_trainable_pec_switch_cannot_disappear_as_false():
    value=torch.tensor(2.,requires_grad=True)
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.Material(eps_r=value.exp(),pec=value*0)


def test_received_trainable_copper_band_rejects_without_touching_registered_parameters():
    value=torch.tensor(2.,requires_grad=True)
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.CopperFoil(thickness=value.exp()*1e-7,conductivity=value.exp()*1e6,mode_count=16,frequency_range=(value*1e9,4e9))


def test_received_trainable_lumped_reference_plane_rejects_before_float():
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.LumpedPort('feed',positive=(0.,0.,.005),negative=(0.,0.,-.005),voltage_path=mw.AxisPath('z'),
            current_surface=mw.Box(size=(.01,.01,0.)),reference_plane=torch.tensor(0.,requires_grad=True))


def test_existing_nongrad_tensor_controls_and_registered_rf_values_remain_valid():
    boundary=mw.BoundarySpec.pml(num_layers=torch.tensor(4.),strength=torch.tensor(1e6,dtype=torch.float64))
    assert boundary.num_layers==4
    assert not mw.Material(pec=torch.tensor(0.)).pec
    value=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    copper=mw.CopperFoil(thickness=value.exp()*1e-7,conductivity=value.exp()*1e6,mode_count=16,
        frequency_range=(torch.tensor(2e9,dtype=torch.float64),torch.tensor(4e9,dtype=torch.float64)))
    assert copper.thickness.requires_grad and copper.conductivity.requires_grad
    port=mw.LumpedPort('feed',positive=(0.,0.,.005),negative=(0.,0.,-.005),voltage_path=mw.AxisPath('z'),
        current_surface=mw.Box(size=(.01,.01,0.)),reference_plane=torch.tensor(0.),reference_impedance=value*50)
    assert port.reference_impedance.requires_grad and port.reference_plane==0.


@pytest.mark.parametrize('kind',('finite','plane'))
def test_trainable_flux_reporting_flag_rejects_before_bool(kind):
    value=torch.tensor(0.,requires_grad=True)
    with pytest.raises(NotImplementedError,match='fixed'):
        if kind=='finite':mw.FinitePlaneMonitor('m',position=(0.,0.,0.),size=(.02,.02,0.),compute_flux=value)
        else:mw.PlaneMonitor('m',compute_flux=value)


def test_nested_object_array_cannot_hide_received_chart_curvature_gradient():
    import numpy as np
    theta=torch.tensor(2.,requires_grad=True)
    curvature=np.empty((),dtype=object);curvature[()]=theta
    with pytest.raises(NotImplementedError,match='fixed chart'):
        mw.BendCoordinates(curvature=curvature)

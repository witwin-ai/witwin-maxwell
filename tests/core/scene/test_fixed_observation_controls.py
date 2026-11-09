"""Fixed observation geometry/frequencies and material quadrature topology."""
import pytest
import torch
import witwin.maxwell as mw


def monitor(kind,**options):
    if kind=='finite':return mw.FinitePlaneMonitor('m',**({'position':(0.,0.,0.),'size':(.02,.02,0.)}|options))
    if kind=='closed':return mw.ClosedSurfaceMonitor.box('m',**({'position':(0.,0.,0.),'size':(.02,.02,.02)}|options))
    return mw.PowerLossMonitor('m',**({'position':(0.,0.,0.),'size':(.02,.02,.02)}|options))


@pytest.mark.parametrize('kind',('finite','closed','bulk'))
@pytest.mark.parametrize('slot',('position','size','frequencies'))
def test_received_trainable_observation_geometry_or_frequency_rejects(kind,slot):
    theta=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    kwargs={slot:(theta*.001,0.,0.) if slot=='position' else (theta*.01,.02,0. if kind=='finite' else .02) if slot=='size' else (theta*1e9,)}
    with pytest.raises(NotImplementedError,match='fixed'):
        monitor(kind,**kwargs)


@pytest.mark.parametrize('kind',('finite','closed'))
def test_trainable_observation_quadrature_rejects_before_float(kind):
    theta=torch.tensor(.001,requires_grad=True)
    with pytest.raises(NotImplementedError,match='fixed'):
        monitor(kind,sample_spacing=theta)


@pytest.mark.parametrize('declaration',('spec','scene'))
def test_trainable_material_sampling_count_rejects_before_int(declaration):
    theta=torch.tensor(2.,requires_grad=True)
    with pytest.raises(NotImplementedError,match='fixed'):
        if declaration=='spec':mw.SubpixelSpec(samples=(theta,2,2), averaging="arithmetic")
        else:mw.Scene(domain=mw.Domain(bounds=((- .02,.02),)*3),grid=mw.GridSpec.uniform(.005),device='cpu',subpixel_samples=(theta,2,2))


@pytest.mark.parametrize('kind',('finite','closed','bulk'))
def test_existing_fixed_tensor_observation_arguments_remain_valid(kind):
    value=monitor(kind,position=(torch.tensor(.001,dtype=torch.float64),0.,0.),frequencies=(torch.tensor(3e9,dtype=torch.float64),))
    assert value.frequencies==(3e9,)


def test_fixed_tensor_sampling_and_supported_box_material_rf_gradients_unchanged():
    assert mw.SubpixelSpec(samples=(torch.tensor(2),2,2), averaging="arithmetic").samples==(2,2,2)
    theta=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    geometry=mw.Box(position=(theta*.001,0.,0.),size=(.02,.02,.02))
    assert geometry.position.requires_grad
    assert mw.Material(eps_r=theta).eps_r is theta
    excitation=mw.PortExcitation('feed',amplitude=theta)
    assert excitation.amplitude.requires_grad


def test_observation_frequency_generator_consumed_once():
    visits=[]
    def frequencies():
        for value in (2e9,3e9):visits.append(value);yield torch.tensor(value,dtype=torch.float64)
    value=monitor('finite',frequencies=frequencies())
    assert visits==[2e9,3e9] and value.frequencies==(2e9,3e9)

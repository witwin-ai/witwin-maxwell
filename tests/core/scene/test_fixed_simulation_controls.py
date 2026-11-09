"""Received fixed physical controls must not disappear before admission."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.simulation import FDTDConfig
def scene(material=None):
    return mw.Scene(domain=mw.Domain(bounds=((- .02,.02),)*3),grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.pml(num_layers=4),device='cpu',coordinates=mw.BendCoordinates(curvature=-3.),
        structures=(mw.Structure(mw.Box(size=(.02,.02,.02)),material or mw.Material()),))


@pytest.mark.parametrize('shared_material',(False,True))
@pytest.mark.parametrize('kind',('frequency','frequencies','generator','direct','zero_shutoff'))
def test_shared_epsilon_leaf_in_fixed_control_rejects_before_conversion(kind,shared_material):
    theta=torch.tensor(2.,requires_grad=True,dtype=torch.float64)
    value=scene(material=mw.Material(eps_r=theta.exp()) if shared_material else None)
    common=dict(run_time=mw.TimeConfig(dt=5e-12,time_steps=96))
    with pytest.raises(NotImplementedError,match='fixed'):
        if kind=='direct':
            mw.Simulation(scene=value,method='fdtd',frequencies=(theta*3e9,),config=FDTDConfig(**common))
        elif kind=='zero_shutoff':mw.Simulation.fdtd(value,frequency=3e9,shutoff=theta*0,**common)
        elif kind=='frequency':mw.Simulation.fdtd(value,frequency=theta*3e9,**common)
        elif kind=='frequencies':mw.Simulation.fdtd(value,frequencies=(3e9,theta*3e9),**common)
        else:mw.Simulation.fdtd(value,frequencies=(item for item in (theta*3e9,)),**common)


@pytest.mark.parametrize('value',(3e9,torch.tensor(3e9),torch.tensor(3e9,dtype=torch.float64)))
def test_fixed_scalar_frequency_values_keep_existing_api(value):
    result=mw.Simulation.fdtd(scene(),frequency=value,shutoff=torch.tensor(0.))
    assert result.frequencies==(float(value),)


@pytest.mark.parametrize('name,value',(('enable_plot',0.),('full_field_dft',0.),('cuda_graph',0.),('shutoff_check_interval',100.),('adjoint_checkpoint_stride',7.)))
def test_trainable_execution_control_is_rejected_before_bool_or_int_conversion(name,value):
    with pytest.raises(NotImplementedError,match='fixed'):
        FDTDConfig(**{name:torch.tensor(value,requires_grad=True)})


def test_trainable_source_normalization_control_is_rejected_before_bool_conversion():
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.SpectralSampler(normalize_source=torch.tensor(0.,requires_grad=True))


def test_fixed_tensor_execution_controls_retain_existing_api():
    value=FDTDConfig(enable_plot=torch.tensor(0.),full_field_dft=torch.tensor(0.),cuda_graph=torch.tensor(0.),
        shutoff=torch.tensor(0.),shutoff_check_interval=torch.tensor(100.),adjoint_checkpoint_stride=torch.tensor(7.))
    assert not value.enable_plot and not value.full_field_dft and not value.cuda_graph
    assert value.shutoff_check_interval==100 and value.adjoint_checkpoint_stride==7
    assert not mw.SpectralSampler(normalize_source=torch.tensor(0.)).normalize_source


def test_frequency_generator_is_consumed_exactly_once():
    visits=[]
    def frequencies():
        for value in (2e9,3e9):
            visits.append(value)
            yield torch.tensor(value,dtype=torch.float64)
    value=mw.Simulation.fdtd(scene(),frequencies=frequencies())
    assert visits==[2e9,3e9] and value.frequencies==(2e9,3e9)


@pytest.mark.parametrize('name',('time_steps','steady_cycles','transient_cycles','dt','max_electric_field'))
def test_trainable_time_configuration_is_rejected_before_validation_conversion(name):
    value=5e-12 if name=='dt' else 96. if name=='time_steps' else 1.
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.TimeConfig(**{name:torch.tensor(value,requires_grad=True)})


@pytest.mark.parametrize('name',('window','cpml_config','absorber'))
def test_other_fixed_metadata_rejects_trainable_values_before_normalization(name):
    value=torch.tensor(1.,requires_grad=True)
    with pytest.raises(NotImplementedError,match='fixed'):
        if name=='window':mw.SpectralSampler(window=value)
        elif name=='cpml_config':FDTDConfig(cpml_config={'kappa_max':value})
        else:FDTDConfig(absorber=value)


@pytest.mark.parametrize('name',('time_steps','steady_cycles','transient_cycles','dt','max_electric_field',
    'enable_plot','full_field_dft','shutoff','shutoff_check_interval','cuda_graph','adjoint_checkpoint_stride',
    'cpml_config','normalize_source','window','frequency','frequencies'))
def test_prepare_rechecks_direct_fixed_config_mutation_without_scene_parameters(name):
    value=mw.Simulation.fdtd(scene(),frequency=3e9,run_time=mw.TimeConfig(dt=5e-12,time_steps=96))
    assert not value.has_trainable_parameters
    parameter=torch.tensor(1.,requires_grad=True)
    if name in ('time_steps','steady_cycles','transient_cycles','dt','max_electric_field'):
        object.__setattr__(value.config.run_time,name,parameter*5e-12 if name=='dt' else parameter)
    elif name in ('normalize_source','window'):setattr(value.config.spectral_sampler,name,parameter*0)
    elif name=='frequency':value.frequency=parameter*3e9
    elif name=='frequencies':value.frequencies=(parameter*3e9,)
    elif name=='cpml_config':value.config.cpml_config={'kappa_max':parameter}
    else:setattr(value.config,name,parameter*0 if name in ('shutoff','cuda_graph','full_field_dft','enable_plot') else parameter)
    with pytest.raises(NotImplementedError,match='fixed'):
        value.prepare()


def test_fixed_tensor_time_values_preserve_tensor_identity_and_precision():
    dt=torch.tensor(5e-12,dtype=torch.float64)
    value=mw.TimeConfig(time_steps=96,dt=dt,steady_cycles=torch.tensor(1.),transient_cycles=torch.tensor(0.))
    assert value.dt is dt and value.dt.dtype==torch.float64
    with pytest.raises(ValueError,match='positive integer'):
        mw.TimeConfig(time_steps=torch.tensor(96))

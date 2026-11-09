"""Actual public constructor admission in an isolated candidate; no CUDA solve."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint.mapped_admission import tensor_leaves
from tests.core.scene.fp64_public_scene import scene as fixed_scene


def leaf(value=2.):return torch.tensor(value,dtype=torch.float64,requires_grad=True)


def simulation(value,**kwargs):
    options=dict(frequency=3e9,run_time=mw.TimeConfig(dt=5e-12,time_steps=96),cuda_graph=False,
        excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9)),full_field_dft=True)
    options.update(kwargs)
    return mw.Simulation.fdtd(value,**options)


@pytest.mark.parametrize('kind',('epsilon','sigma','shared','zero_vjp','copper'))
def test_actual_public_constructor_admits_registered_material_dependencies(kind):
    theta=leaf()
    if kind=='copper':
        value=fixed_scene();old=value.structures[1]
        value=value.clone(structures=(value.structures[0],replace(old,material=mw.CopperFoil(thickness=theta.exp()*1e-7,conductivity=theta.exp()*1e6,frequency_range=(2e9,4e9),mode_count=16))))
    else:
        material=mw.Material(eps_r=theta.exp() if kind in ('epsilon','shared') else 2.,sigma_e=theta.square() if kind in ('sigma','shared') else theta*0 if kind=='zero_vjp' else 0.)
        value=fixed_scene(material=material)
    model=simulation(value)
    assert model.has_trainable_parameters
    assert model._mapped_material_admission.leaf_ids=={id(theta)}


@pytest.mark.parametrize('name',('mu_r','mass_density'))
def test_direct_magnetic_or_density_leaf_cannot_hide_from_legacy_detection(name):
    theta=leaf();material=mw.Material(**{name:theta if name!='mass_density' else theta.reshape(1,1,1)})
    with pytest.raises(NotImplementedError,match='Unregistered'):
        simulation(fixed_scene(material=material))


@pytest.mark.parametrize('factor',(.1,0.))
def test_shared_epsilon_leaf_forbidden_magnetic_use_even_zero_vjp(factor):
    theta=leaf()
    with pytest.raises(NotImplementedError,match='Unregistered'):
        simulation(fixed_scene(material=mw.Material(eps_r=theta.exp(),mu_r=1+theta*factor)))


@pytest.mark.parametrize('representation',('stack','tuple','list'))
def test_core_preserved_nested_geometry_dependency_is_rejected(representation):
    theta=leaf();parts=(theta*.001,theta*0,theta*0)
    position=torch.stack(parts) if representation=='stack' else list(parts) if representation=='list' else parts
    value=fixed_scene(material=mw.Material(eps_r=theta.exp()))
    structure=replace(value.structures[0],geometry=mw.Box(position=position,size=(.04,.02,.04)))
    with pytest.raises(NotImplementedError,match='Unregistered'):
        simulation(value.clone(structures=(structure,value.structures[1])))


@pytest.mark.parametrize('role',('rf_amplitude','rf_impedance','time_step'))
def test_shared_electrical_leaf_forbidden_fixed_input_use(role):
    theta=leaf();value=fixed_scene(material=mw.Material(eps_r=theta.exp()))
    options={}
    if role=='rf_amplitude':options['excitations']=mw.PortExcitation('feed',amplitude=theta)
    elif role=='rf_impedance':options['excitations']=mw.PortExcitation('feed',source_impedance=theta*50)
    with pytest.raises(NotImplementedError,match='Unregistered|fixed'):
        if role=='time_step':options['run_time']=mw.TimeConfig(dt=theta*1e-12,time_steps=96)
        simulation(value,**options)


def test_distinct_material_leaf_storage_alias_is_rejected():
    first=torch.nn.Parameter(torch.tensor(2.));second=torch.nn.Parameter(first.data)
    with pytest.raises(NotImplementedError,match='share storage'):
        simulation(fixed_scene(material=mw.Material(eps_r=first,sigma_e=second)))


def test_complex_latent_leaf_is_not_misclassified_real_electric_parameter():
    theta=torch.tensor(2.+.2j,requires_grad=True)
    with pytest.raises(NotImplementedError,match='Complex'):
        simulation(fixed_scene(material=mw.Material(eps_r=theta.real)))


@pytest.mark.parametrize('configuration',(None,mw.TimeConfig(dt=5e-12),mw.TimeConfig(time_steps=96)))
def test_adaptive_time_problem_is_rejected(configuration):
    with pytest.raises(NotImplementedError,match='explicit fixed'):
        simulation(fixed_scene(material=mw.Material(eps_r=leaf())),run_time=configuration)


def test_auto_grid_is_not_silently_differentiated():
    with pytest.raises(NotImplementedError,match='fixed grid'):
        simulation(fixed_scene(material=mw.Material(eps_r=leaf())).clone(grid=mw.GridSpec.auto(wavelength=.1)))


@pytest.mark.parametrize('slot',('phase','amplitude','frequency'))
def test_received_trainable_cw_input_rejects_before_float_conversion(slot):
    kwargs={'frequency':3e9,slot:leaf()}
    with pytest.raises(NotImplementedError,match='fixed'):
        mw.CW(**kwargs)


def test_scalar_parameter_view_ancestry_is_structural():
    value=leaf();assert set(tensor_leaves(value.view(())))=={id(value)}


def test_detached_mapped_legacy_default_does_not_need_explicit_time():
    model=mw.Simulation.fdtd(fixed_scene(),frequency=3e9)
    assert not model.has_trainable_parameters and model._mapped_material_admission is None


class MaterialModule(mw.SceneModule):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.epsilon=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
        if kind in ('unused','unused_shared_storage'):
            self.unused=torch.nn.Parameter(self.epsilon.data if kind=='unused_shared_storage' else torch.tensor(.2))
        if kind in ('buffer','unused_buffer'):
            other=torch.tensor(.1,requires_grad=True)
            self.register_buffer('latent',other.exp())
    def to_scene(self):
        sigma=self.latent.square() if self.kind=='buffer' else 0.
        value=fixed_scene(material=mw.Material(eps_r=self.epsilon.exp(),sigma_e=sigma))
        if self.kind=='disabled':
            value=value.clone(structures=(replace(value.structures[0],enabled=False),value.structures[1]))
        return value


@pytest.mark.parametrize('kind',('normal','buffer'))
def test_scene_module_ownership_matches_actual_registered_parameter_frontier(kind):
    module=MaterialModule(kind)
    if kind=='buffer':
        with pytest.raises(NotImplementedError,match='registered SceneModule parameters'):
            simulation(module)
    else:
        model=simulation(module)
        assert id(module.epsilon) in model._mapped_material_admission.leaf_ids


@pytest.mark.parametrize('kind',('unused','unused_shared_storage','disabled','unused_buffer'))
def test_module_unused_or_disabled_trainables_are_not_excused_by_epsilon(kind):
    with pytest.raises(NotImplementedError,match='Unaccounted|Unregistered'):
        simulation(MaterialModule(kind))


@pytest.mark.parametrize('option,value',(('field_dtype',torch.float64),('shutoff',.01)))
def test_material_admission_rejects_unqualified_precision_or_duration(option,value):
    with pytest.raises(NotImplementedError,match='FP32|fixed-duration'):
        simulation(fixed_scene(material=mw.Material(eps_r=leaf())),**{option:value})


def test_mixed_boundary_not_qualified_by_all_pml_proof():
    value=fixed_scene(material=mw.Material(eps_r=leaf()),boundary=mw.BoundarySpec.faces(default='pml',num_layers=4,x_low='pec'))
    with pytest.raises(NotImplementedError,match='six PML'):
        simulation(value)


def test_module_to_scene_refresh_rechecks_new_semantic_input_dependencies():
    module=MaterialModule('normal');model=simulation(module)
    module.kind='unused';module.unused=torch.nn.Parameter(torch.tensor(.2))
    with pytest.raises(NotImplementedError,match='Unaccounted'):
        model._refresh_scene()



def test_multifrequency_adjoint_has_no_admission_from_single_frequency_proof():
    with pytest.raises(NotImplementedError,match='one fixed frequency'):
        simulation(fixed_scene(material=mw.Material(eps_r=leaf())),frequency=None,frequencies=(3e9,3.1e9))


def test_fixed_terminal_feed_remains_an_explicit_unqualified_combination():
    port=mw.TerminalPort('feed',positive_terminal=mw.TerminalRef('positive'),negative_terminal=mw.TerminalRef('negative'),
        integration_path=mw.AxisPath('z'),reference_plane=0.)
    value=fixed_scene(material=mw.Material(eps_r=leaf()))
    terminals=tuple(mw.Structure(name=name,geometry=mw.Box(position=( -.015,-.015,z),size=(.01,.01,.005)),
        material=mw.Material.pec(name=name)) for name,z in (('positive',.0125),('negative',-.0125)))
    value=value.clone(structures=(*value.structures,*terminals),ports=(port,))
    with pytest.raises(NotImplementedError,match='LumpedPort Norton'):
        simulation(value)


def test_static_phase_screen_cannot_be_misclassified_as_bulk_electrical_material():
    from witwin.core import PhaseScreen
    value=fixed_scene(material=mw.Material(eps_r=leaf()));screen=PhaseScreen(height=torch.zeros(2,2),height_scale_m=1e-6)
    value=value.clone(structures=(replace(value.structures[0],phase_screen=screen),value.structures[1]))
    with pytest.raises(NotImplementedError,match='phase-screen'):
        simulation(value)


def test_core_assignment_switch_cannot_share_material_leaf_and_disappear():
    theta=leaf()
    with pytest.raises(NotImplementedError,match='fixed assignment'):
        mw.Structure(mw.Box(size=(.02,.02,.02)),mw.Material(eps_r=theta.exp()),enabled=theta*0)


def test_subpixel_sampling_cannot_share_material_leaf_and_disappear():
    theta=leaf()
    with pytest.raises(NotImplementedError,match='fixed'):
        fixed_scene(material=mw.Material(eps_r=theta.exp())).clone(subpixel_samples=mw.SubpixelSpec(samples=(theta,2,2), averaging="arithmetic"))

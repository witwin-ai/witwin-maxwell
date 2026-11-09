"""Internal custom-autograd/no_grad lifecycle and unchanged-model contracts."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from tests.core.scene.fp64_public_scene import scene as fixed_scene


class LifecycleModule(mw.SceneModule):
    def __init__(self,kind='normal'):
        super().__init__();self.kind=kind
        self.epsilon=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
    def to_scene(self):
        factor=2. if self.kind=='grad_material' and not torch.is_grad_enabled() else 1.
        material=mw.Material(eps_r=self.epsilon.exp()*factor,
            mu_r=1+self.epsilon*0 if self.kind=='forbidden_mu' else 1.)
        value=fixed_scene(material=material)
        if self.kind in ('grad_geometry','forbidden_geometry'):
            x=(.001 if torch.is_grad_enabled() else .002) if self.kind=='grad_geometry' else self.epsilon*0
            geometry=mw.Box(position=(x,.005,0.),size=(.04,.02,.04))
            value=value.clone(structures=(replace(value.structures[0],geometry=geometry),value.structures[1]))
        if self.kind=='grad_monitor':
            x=.001 if torch.is_grad_enabled() else .002
            value=value.clone(monitors=(*value.monitors,mw.PointMonitor('mode-audit',position=(x,.02,.02),fields=('Ez',))))
        if self.kind=='grad_source' and not torch.is_grad_enabled():
            value=value.clone(sources=(mw.PointDipole(position=(-.02,-.02,-.02),source_time=mw.CW(3e9)),))
        return value


def simulation(module):
    return mw.Simulation.fdtd(module,frequency=3e9,run_time=mw.TimeConfig(dt=5e-12,time_steps=96),
        cuda_graph=False,excitations=mw.PortExcitation('feed',source_time=mw.CW(3e9)))


def test_internal_no_grad_refresh_keeps_numerical_scene_detached_and_census_complete():
    module=LifecycleModule();value=simulation(module)
    with torch.no_grad():
        value._refresh_scene()
        assert not torch.is_grad_enabled()
        assert not value.scene.structures[0].material.eps_r.requires_grad
        assert value.has_trainable_parameters
        assert value._mapped_material_admission.leaf_ids=={id(module.epsilon)}
    assert torch.is_grad_enabled()


class RefreshInsideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,parameter,simulation):
        assert not torch.is_grad_enabled()
        simulation._refresh_scene()
        assert not simulation.scene.structures[0].material.eps_r.requires_grad
        return simulation.scene.structures[0].material.eps_r.detach().clone()


def test_actual_custom_autograd_forward_can_refresh_without_bypassing_admission():
    module=LifecycleModule();value=simulation(module)
    output=RefreshInsideFunction.apply(module.epsilon,value)
    torch.testing.assert_close(output,module.epsilon.detach().exp())
    assert value._mapped_material_admission.leaf_ids=={id(module.epsilon)}


@pytest.mark.parametrize('kind',('forbidden_mu','forbidden_geometry'))
def test_internal_no_grad_does_not_hide_forbidden_shared_leaf_even_with_zero_vjp(kind):
    module=LifecycleModule();value=simulation(module);module.kind=kind
    with torch.no_grad(),pytest.raises(NotImplementedError,match='Unregistered'):
        value._refresh_scene()


def test_internal_no_grad_still_rejects_unused_module_parameter():
    module=LifecycleModule();value=simulation(module)
    module.unused=torch.nn.Parameter(torch.tensor(.1))
    with torch.no_grad(),pytest.raises(NotImplementedError,match='Unaccounted'):
        value._refresh_scene()


@pytest.mark.parametrize('kind',('grad_material','grad_geometry','grad_monitor','grad_source'))
def test_grad_context_dependent_physical_declarations_reject_instead_of_changing_forward(kind):
    module=LifecycleModule(kind);value=simulation(module)
    with torch.no_grad(),pytest.raises(NotImplementedError,match='grad mode'):
        value._refresh_scene()


class IndependentTensorModule(mw.SceneModule):
    def __init__(self,buffer):
        super().__init__();self.epsilon=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
        extra=torch.tensor(.1,requires_grad=True,dtype=torch.float64)
        if buffer:self.register_buffer('extra',extra)
        else:self.extra=extra
    def to_scene(self):
        return fixed_scene(material=mw.Material(eps_r=self.epsilon.exp(),sigma_e=self.extra.exp()))


@pytest.mark.parametrize('buffer',(False,True))
@pytest.mark.parametrize('grad_enabled',(False,True))
def test_independent_buffer_and_plain_tensor_leaves_are_outside_actual_bridge_frontier(buffer,grad_enabled):
    with torch.set_grad_enabled(grad_enabled),pytest.raises(NotImplementedError,match='registered SceneModule parameters'):
        simulation(IndependentTensorModule(buffer))


def test_custom_temporal_callable_does_not_hide_material_parameter_dependency():
    module=LifecycleModule()
    waveform=mw.CustomSourceTime(lambda t:float(module.epsilon.detach())*t,characteristic_frequency=3e9)
    with pytest.raises(NotImplementedError,match='declarative fixed temporal'):
        mw.Simulation.fdtd(module,frequency=3e9,run_time=mw.TimeConfig(dt=5e-12,time_steps=96),
            excitations=mw.PortExcitation('feed',source_time=waveform))


class CachedSceneModule(LifecycleModule):
    def to_scene(self):
        if not hasattr(self,'cached'):
            self.cached=fixed_scene()
        factor=2. if self.kind=='cached_branch' and not torch.is_grad_enabled() else 1.
        material=mw.Material(eps_r=self.epsilon.exp()*factor)
        self.cached.structures[0]=replace(self.cached.structures[0],material=material)
        return self.cached


@pytest.mark.parametrize('kind',('cached_branch','cached_same_values'))
def test_rematerialization_cannot_mutate_the_numerical_scene_through_a_cached_alias(kind):
    module=CachedSceneModule(kind);value=simulation(module)
    with torch.no_grad(),pytest.raises(NotImplementedError,match='grad mode'):
        value._refresh_scene()

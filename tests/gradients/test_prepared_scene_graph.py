"""Result material declarations must stay outside the mode launch replay graph."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint import bridge as bridge_module
from witwin.maxwell.fdtd.adjoint import core, dispatch
from witwin.maxwell.fdtd.excitation.injection import initialize_source_terms
from witwin.maxwell.fdtd.runtime.materials import pole_component_weight
from witwin.maxwell.postprocess.fdtd_loss import automatic_electric_dispersion_density
from witwin.maxwell.postprocess.modal import _mode_overlap_context


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _Design(mw.SceneModule):
    def __init__(self, launch):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(.4))
        self.launch = launch

    def to_scene(self):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
            grid=mw.GridSpec.uniform(.08), boundary=mw.BoundarySpec.none(), device='cpu',
            subpixel_samples=mw.SubpixelSpec(samples=1, averaging='arithmetic'))
        scene.add_material_region(mw.MaterialRegion(
            'design', mw.Box(position=(0, 0, 0), size=(.48, .24, .32)),
            self.value.sigmoid().expand(2, 2, 2), eps_bounds=(1., 12.)))
        kwargs = dict(position=(0, 0, 0), size=(0, .48, .48), polarization='Ez',
                      source_time=mw.GaussianPulse(frequency=1e9, fwidth=.2e9), name='launch')
        if self.launch == 'source':
            scene.add_source(mw.ModeSource(**kwargs))
        elif self.launch == 'port':
            scene.add_port(mw.ModePort(**kwargs))
        else:
            scene.add_source(mw.PointDipole(position=(0, 0, 0), source_time=mw.CW(1e9)))
            scene.add_structure(mw.Structure(
                mw.Box(position=(.32, 0, 0), size=(.16, .24, .32)),
                mw.Material(eps_r=2., debye_poles=(mw.DebyePole(
                    delta_eps=self.value.square(), tau=1e-10),))))
        for cls, name in ((mw.PermittivityMonitor, 'eps'), (mw.MediumMonitor, 'medium')):
            scene.add_monitor(cls(name, position=(0, 0, 0), size=(.64, .48, .48)))
        return scene


def _host_bridge_result(monkeypatch, model):
    # Exercise production bridge scene selection, material/source initialization,
    # custom-forward grad mode and Result construction. Native stepping/adjoints
    # require CUDA; only those parts and CPU admission are replaced here.
    from witwin.maxwell.fdtd import cpu_profile
    monkeypatch.setattr(cpu_profile, 'admit_cpu_reference_scene',
                        lambda simulation, scene: setattr(scene, '_fdtd_cpu_reference', True))
    cls = bridge_module._FDTDGradientBridge

    def init(self, simulation):
        self.simulation = simulation
        self.trainable_inputs = tuple(model.parameters())
        self._last_embedded_networks = {}
        self._last_solver_stats = {}

    def accept(self, solver, raw, checkpoints, **kwargs):
        self._last_solver = solver
        return (torch.zeros(1),)

    monkeypatch.setattr(cls, '__init__', init)
    monkeypatch.setattr(cls, '_validate_supported_configuration', lambda *args: None)
    monkeypatch.setattr(dispatch, 'validate_native_adjoint_preparation',
                        lambda *args: SimpleNamespace(name='MATERIAL'))
    monkeypatch.setattr(cls, '_run_forward_with_checkpoints', lambda *args, **kwargs: ({}, ()))
    monkeypatch.setattr(cls, 'accept_forward', accept)
    monkeypatch.setattr(cls, 'rebuild_forward_outputs',
                        lambda self, outputs: ({'Ex': outputs[0]}, {}, {}, {}, {}))
    simulation = mw.Simulation.fdtd(model, frequency=1e9, backend='cpu_reference',
                                   run_time=mw.TimeConfig(time_steps=12))
    return bridge_module.run_fdtd_with_gradient_bridge(simulation)


@pytest.mark.parametrize('launch', ['none', 'source', 'port'])
def test_bridge_result_materials_are_numerical_but_launch_replay_has_gradients(monkeypatch, launch):
    model = _Design(launch)
    result = _host_bridge_result(monkeypatch, model)
    assert not result.scene.material_regions[0].density.requires_grad
    assert not result.prepared_scene.material_regions[0].density.requires_grad
    for name in ('eps', 'medium'):
        payload = result.monitor(name)
        assert not payload['eps'].requires_grad
        assert payload['eps'].grad_fn is None
    context = _mode_overlap_context(result)
    assert not context._compiled_material_model['eps_r'].requires_grad
    if launch == 'none':
        assert result.prepared_scene is result.solver.scene
        return
    assert result.prepared_scene is not result.solver.scene
    solver = result.solver
    source = solver._compiled_sources[0]
    assert source['mode_solver_kind'] == 'vector_dense_torch'
    bridge_module._validate_mode_source_replay(solver, model.to_scene(), (model.value,))
    replay = core._build_source_replay_solver(
        solver, solver._compiled_sources, solver.eps_Ex, solver.eps_Ey, solver.eps_Ez)
    initialize_source_terms(replay)
    for actual, expected in zip(replay._electric_source_terms, solver._electric_source_terms):
        assert torch.equal(actual['patch'], expected['patch'])
    objective = sum(term['patch'].square().sum() for term in replay._electric_source_terms)
    gradient, = torch.autograd.grad(objective, model.value)
    assert torch.isfinite(gradient) and gradient != 0


def test_pole_work_keeps_field_gradients_without_a_declaration_graph(monkeypatch):
    result = _host_bridge_result(monkeypatch, _Design('none'))
    scene = result.prepared_scene
    entry = scene.compile_materials()['debye_poles'][0]
    masks = {c: torch.ones_like(pole_component_weight(entry, c), dtype=torch.bool)
             for c in ('Ex', 'Ey', 'Ez')}
    fields = {c: torch.ones((1, *mask.shape), dtype=torch.complex128, requires_grad=True)
              for c, mask in masks.items()}
    density = automatic_electric_dispersion_density(
        scene, SimpleNamespace(component_masks=masks), torch.tensor([1e9]), fields, 1e-11)
    assert not entry['pole'].delta_eps.requires_grad
    objective = sum(value.sum() for value in density.values())
    gradients = torch.autograd.grad(objective, tuple(fields.values()))
    assert objective > 0
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)

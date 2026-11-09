"""Finite-volume source products across deposition, replay and ownership readers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from tests.fdtd.test_custom_current_volume_support import _COMPONENTS, _prepare, _scene
from witwin.maxwell.fdtd.adjoint.core import _apply_source_term_list, _resolved_source_term_lists
from witwin.maxwell.fdtd.distributed.sources import _crop_term_to_owned_x
from witwin.maxwell.fdtd.excitation.temporal import apply_compiled_source_terms, apply_generic_source_terms


@pytest.fixture
def solver():
    scene = _scene(mw.GridSpec.uniform(.08))
    axes = ([-.079, .083], [-.061, .093], [-.33, -.17])
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(axes, {
        key: np.full((2, 2, 2), (-1.)**i * (i + 1)) for i, key in enumerate(_COMPONENTS)
    }), source_time=mw.CW(frequency=2e9, phase=.37)))
    return _prepare(scene)


def _terms(solver):
    return solver._source_terms + solver._magnetic_source_terms


def test_electric_step_and_batch_table_use_the_same_injection_instant(solver):
    from witwin.maxwell.fdtd.excitation.temporal import BatchSourceEngine
    from witwin.maxwell.sources import evaluate_source_time

    buffers = {field: getattr(solver, field).unsqueeze(0) for field in ("Ex", "Ey", "Ez")}
    slot = SimpleNamespace(capacity=1, solvers=(solver,), buffers=buffers, device=solver.device,
                           active=torch.ones(1, dtype=torch.int32, device=solver.device))
    engine = BatchSourceEngine(slot, 4)
    # Explicit time_value is an injection instant; integer steps apply the electric half-step offset.
    instant = 2 * solver.dt + 0.5 * float(solver.dt)
    signal = evaluate_source_time(solver._source_time, instant)
    for group in engine.groups:
        assert group[5][2, 0, 0] == torch.tensor(signal, dtype=torch.float32)
    solver.add_source(n=2)
    expected = {field: value.clone() for field, value in buffers.items()}
    for value in buffers.values():
        value.zero_()
    solver.add_source(time_value=instant)
    for field in buffers:
        assert torch.equal(buffers[field], expected[field])


@pytest.mark.parametrize("time", [0., 3.7e-11])
def test_generic_compiled_and_reverse_replay_deposit_identical_currents(solver, time):
    arguments = dict(source_time=solver._source_time, omega=solver.source_omega, time_value=time)
    zero = {field: torch.zeros_like(getattr(solver, field)) for field in _COMPONENTS.values()}
    expected = _apply_source_term_list(zero, terms=_terms(solver), **arguments)
    for apply in (apply_generic_source_terms, apply_compiled_source_terms):
        for field in zero:
            getattr(solver, field).zero_()
        apply(solver, _terms(solver), **arguments)
        for field in zero:
            assert torch.equal(getattr(solver, field), expected[field])


def test_source_rematerialization_keeps_support_and_material_denominators(solver):
    eps = [getattr(solver, 'eps_' + field).detach().clone().requires_grad_() for field in ('Ex', 'Ey', 'Ez')]
    rebuilt = _resolved_source_term_lists(solver, *eps)
    forward = (solver._source_terms, solver._electric_source_terms, solver._magnetic_source_terms)
    for original, rematerialized in zip(forward, rebuilt):
        assert len(original) == len(rematerialized)
        for a, b in zip(original, rematerialized):
            assert a['field_name'] == b['field_name'] and a['offsets'] == b['offsets']
            assert torch.equal(a['patch'], b['patch'])
    sum(term['patch'].sum() for term in rebuilt[0]).backward()
    for leaf, term in zip(eps, rebuilt[0]):
        expected = torch.zeros_like(leaf)
        region = tuple(slice(i, i + n) for i, n in zip(term['offsets'], term['patch'].shape))
        expected[region] = -term['patch'].detach() / leaf.detach()[region]
        torch.testing.assert_close(leaf.grad, expected, rtol=1e-6, atol=0.)


def test_ownership_crops_partition_every_prepared_current_sample_once(solver):
    for term in _terms(solver):
        field = getattr(solver, term['field_name'])
        split = int(term['offsets'][0]) + 1
        recovered = torch.zeros_like(field)
        for owned in (slice(0, split), slice(split, field.shape[0])):
            layout = SimpleNamespace(storage_cell_owned=owned, storage_node_owned=owned)
            cropped = _crop_term_to_owned_x(term, layout)
            if cropped is None:
                continue
            region = tuple(slice(i, i + n) for i, n in zip(cropped['offsets'], cropped['patch'].shape))
            recovered[region] += cropped['patch']
        expected = torch.zeros_like(field)
        region = tuple(slice(i, i + n) for i, n in zip(term['offsets'], term['patch'].shape))
        expected[region] = term['patch']
        assert torch.equal(recovered, expected)


@pytest.mark.parametrize('reader', ['cut_surface', 'copper_foil', 'thin_sheet'])
def test_intersection_guards_read_only_nonzero_finite_support(solver, reader):
    from witwin.maxwell.fdtd.cut_surface import _source_support_overlap
    from witwin.maxwell.fdtd.copper_foil import validate_copper_port_coupling
    from witwin.maxwell.fdtd.thin_sheet import _reject_overlaps

    term = solver._source_terms[0]
    inside = torch.tensor([term['offsets']], dtype=torch.long)
    outside = inside.clone()
    outside[0, 0] -= 1
    for point, overlaps in ((outside, False), (inside, True)):
        if reader == 'cut_surface':
            linear = (point[:, 0] * solver.Ex.shape[1] + point[:, 1]) * solver.Ex.shape[2] + point[:, 2]
            check = lambda: _source_support_overlap(solver, {'Ex': {'indices': point, 'linear_indices': linear}})
        elif reader == 'copper_foil':
            solver._copper_foils = [dict(electric_name='Ex', imaginary=False, indices=point, edge_geometry=None)]
            check = lambda: validate_copper_port_coupling(solver)
        else:
            complex_ = SimpleNamespace(e_component=[0], e_index=point, e_split=[False],
                                       h_component=[], h_index=[], h_extra=np.array([], dtype=int))
            runtime = SimpleNamespace(complex=complex_, patch_e=[0], patch_h=[])
            check = lambda: _reject_overlaps(solver, [runtime])
        if overlaps:
            with pytest.raises(NotImplementedError, match='source.*overlap'):
                check()
        else:
            check()


def test_magnetic_terms_keep_native_batch_unsupported_guard(solver):
    from witwin.maxwell.fdtd.runtime.batch import validate_native_solver
    with pytest.raises(NotImplementedError, match='_magnetic_source_terms'):
        validate_native_solver(solver)


def test_custom_currents_keep_public_native_batch_admission_guard(solver):
    from witwin.maxwell.execution.batch_runtime import PreparedBatch
    simulation = mw.Simulation.fdtd(solver.scene._public_scene, frequency=2e9, backend='cpu_reference')
    batch = SimpleNamespace(simulations=(simulation,), outputs=SimpleNamespace(result_device='cuda'))
    with pytest.raises(NotImplementedError, match='supports PointDipole sources'):
        PreparedBatch._validate_inputs(SimpleNamespace(batch=batch), None)


def test_custom_magnetic_terms_keep_distributed_adjoint_guard(solver):
    from witwin.maxwell.fdtd.distributed.adjoint import require_distributed_adjoint_support
    distributed = SimpleNamespace(_initialized=True, shards=(SimpleNamespace(rank=0, solver=solver),))
    with pytest.raises(ValueError, match='magnetic.*source terms'):
        require_distributed_adjoint_support(distributed)


def test_ideal_point_shard_correction_does_not_rescale_custom_currents(solver):
    from witwin.maxwell.fdtd.distributed.source_corrections import correct_ideal_point_ex_control_volume
    term = solver._source_terms[0]
    edge = int(term['offsets'][0]) + 1
    layout = SimpleNamespace(owns_physical_face=lambda *args: False,
                             storage_cell_owned=slice(0, edge + 1), global_cell_owned=slice(0, edge + 1))
    before = [term['patch'].clone() for term in _terms(solver)]
    correct_ideal_point_ex_control_volume(solver, layout, solver.scene)
    for term, patch in zip(_terms(solver), before):
        assert torch.equal(term['patch'], patch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native periodic deposition requires CUDA')
@pytest.mark.parametrize('field', ['Ex', 'Ey', 'Ez'])
def test_periodic_finite_box_owns_duplicated_nodes_once(field):
    from witwin.maxwell.fdtd.excitation.injection import inject_source_terms
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.64, 0.64),) * 3), grid=mw.GridSpec.uniform(.08),
                     boundary=mw.BoundarySpec.periodic(), device='cuda')
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(([-.64, .64],) * 3,
        {'J' + field[-1]: np.ones((2, 2, 2))}), source_time=mw.CW(frequency=2e9)))
    runtime = mw.Simulation.fdtd(scene, frequency=2e9, cuda_graph=False).prepare().solver
    inject_source_terms(runtime, time_value=0.)
    density = -getattr(runtime, field) * getattr(runtime, 'eps_' + field) / runtime.dt
    torch.testing.assert_close(density, torch.ones_like(density), rtol=2e-7, atol=0.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native current propagation requires CUDA')
@pytest.mark.parametrize('component', list(_COMPONENTS))
def test_native_current_preparation_and_fields_match_cpu_reference(component):
    scene = _scene(mw.GridSpec.uniform(.08))
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(
        ([-.079, .083], [-.061, .093], [-.33, -.17]), {component: np.ones((2, 2, 2))}),
        # A pulse records spectral samples from the first step, so a short run returns output.
        source_time=mw.GaussianPulse(frequency=2e9, fwidth=.5e9)))
    solvers = []
    for device, backend in (('cpu', 'cpu_reference'), ('cuda', 'native')):
        prepared = mw.Simulation.fdtd(scene.clone(device=device), frequency=2e9, backend=backend,
            full_field_dft=True, cuda_graph=False, run_time=mw.TimeConfig(time_steps=32)).prepare()
        prepared.run()
        solvers.append(prepared.solver)
    cpu, native = solvers
    for a, b in zip(_terms(cpu), _terms(native)):
        assert a['field_name'] == b['field_name'] and a['offsets'] == b['offsets']
        torch.testing.assert_close(a['patch'], b['patch'].cpu(), rtol=2e-7, atol=0.)
    for field in _COMPONENTS.values():
        torch.testing.assert_close(getattr(native, field).cpu(), getattr(cpu, field), rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='current graph execution requires CUDA')
def test_finite_volume_electric_current_graph_matches_eager():
    scene = _scene(mw.GridSpec.uniform(.08)).clone(device='cuda')
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(
        ([-.079, .083], [-.061, .093], [-.33, -.17]), {'Jx': np.ones((2, 2, 2))}),
        source_time=mw.GaussianPulse(frequency=2e9, fwidth=.5e9)))
    fields = []
    for graph in (False, True):
        prepared = mw.Simulation.fdtd(scene, frequency=2e9, full_field_dft=True, cuda_graph=graph,
            run_time=mw.TimeConfig(time_steps=32)).prepare()
        prepared.run()
        fields.append({field: getattr(prepared.solver, field).clone() for field in _COMPONENTS.values()})
    for field in _COMPONENTS.values():
        assert torch.equal(fields[0][field], fields[1][field])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='batched native current deposition requires CUDA')
def test_batched_source_engine_consumes_prepared_volume_patches():
    from witwin.maxwell.fdtd.cuda.backend import get_compiled_extension
    from witwin.maxwell.fdtd.excitation.temporal import BatchSourceEngine
    scene = _scene(mw.GridSpec.uniform(.08)).clone(device='cuda')
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(
        ([-.079, .083], [-.061, .093], [-.33, -.17]),
        {'J' + axis: np.full((2, 2, 2), i + 1.) for i, axis in enumerate('xyz')}),
        source_time=mw.CW(frequency=2e9, phase=.37)))
    solver = mw.Simulation.fdtd(scene, frequency=2e9, cuda_graph=False).prepare().solver
    buffers = {field: getattr(solver, field).unsqueeze(0) for field in ('Ex', 'Ey', 'Ez')}
    slot = SimpleNamespace(capacity=1, solvers=(solver,), buffers=buffers, device=solver.device,
                           native=get_compiled_extension(), active=torch.ones(1, dtype=torch.int32, device=solver.device))
    engine = BatchSourceEngine(slot, 4)
    # Step n samples additive electric current at (n + 1/2) dt.
    solver.add_source(n=2)
    expected = {field: value.clone() for field, value in buffers.items()}
    for value in buffers.values():
        value.zero_()
    engine.advance(torch.tensor([2], device=solver.device))
    for field in buffers:
        assert torch.equal(buffers[field], expected[field])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='joint material current gradients require CUDA')
@pytest.mark.parametrize('component', ['Jx', 'My'])
def test_volume_current_joint_material_gradients_match_finite_differences(component):
    from tests.gradients.test_fdtd_conductive_port_adjoint import ConductiveDesign

    class CurrentDesign(ConductiveDesign):
        def to_scene(self):
            scene = super().to_scene().clone(ports=())
            scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(([-.006, .006],) * 3,
                {component: np.full((2, 2, 2), 50.)}), source_time=mw.CW(frequency=3e9)))
            scene.add_monitor(mw.PointMonitor('probe', (.005, .005, .01), fields=('Ex', 'Ey', 'Ez')))
            return scene

    model = CurrentDesign()

    def objective():
        result = mw.Simulation.fdtd(model, frequency=3e9, cuda_graph=False,
            run_time=mw.TimeConfig(time_steps=48), spectral_sampler=mw.SpectralSampler(window='none')).run()
        fields = result.monitor('probe')['components']
        return sum(fields[field].abs().square().sum() for field in ('Ex', 'Ey', 'Ez'))

    objective().backward()
    actual = {name: getattr(model, name).grad.detach().clone() for name in ('log_sigma', 'log_eps')}
    model.requires_grad_(False)
    for name, gradient in actual.items():
        parameter = getattr(model, name)
        saved = parameter.detach().clone()
        assert torch.isfinite(gradient) and abs(float(gradient)) > 1e-12
        for delta in (.01, .005):
            parameter.copy_(saved + delta)
            plus = objective()
            parameter.copy_(saved - delta)
            minus = objective()
            parameter.copy_(saved)
            torch.testing.assert_close(gradient, ((plus - minus) / (2 * delta)).to(gradient), rtol=.01, atol=1e-10)

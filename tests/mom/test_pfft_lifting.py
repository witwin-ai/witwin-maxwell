"""Shared-grid analytic lifting: dense agreement, weak contrast and body locality."""
from dataclasses import replace
import json

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.lifting import build_lifted_rhs
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
FREQUENCY = 299792458.
INTEGRATION = IntegrationConfig(order=4, singular_order=4)


def _scene(frequency_mesh, bodies):
    structures = []
    for centre, epsilon in bodies:
        surface = sphere_mesh(radius=.5, frequency=frequency_mesh, center=centre, device='cuda')
        structures.append(mw.Structure(mw.Mesh(surface.vertices, surface.faces, recenter=False, fill_mode='solid'),
                                       mw.Material(eps_r=epsilon)))
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()])


def _config(pfft=mw.PFFTConfig()):
    return mw.MoMConfig(operator_backend='pfft', riesz_storage='local_rwg_v1', integration=INTEGRATION,
                        pfft=pfft, stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal'))


def _relative(actual, reference):
    return float(torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference))


@pytest.mark.parametrize('frequency_mesh', [2, 3, 4])
@pytest.mark.parametrize('epsilon', [4., 1.+1e-6])
def test_shared_grid_rhs_dense_agreement_and_order(frequency_mesh, epsilon):
    scene = _scene(frequency_mesh, [((0., 0., 0.), epsilon)])
    errors = []
    config = mw.PFFTConfig()
    for pfft in (config, replace(config, order=config.stencil_order+1)):
        prepared = mw.Simulation.mom(scene, frequency=FREQUENCY, config=_config(pfft)).prepare()
        handle = prepared.solver.handle
        dense, _ = build_lifted_rhs(handle.body_handles[0])
        error = _relative(prepared.solver.rhs, dense)
        errors.append(error)
        record = handle.metadata['rhs_bodies'][0]
        assert record['rhs_far_acceleration'] == 'pfft_shared_grid_stable_difference'
        assert record['rhs_lifting_grid']['grid_region'] == 'interior'
        assert sum(record['rhs_triangle_pair_counts']) == record['rhs_near_face_pairs']
        if frequency_mesh == 4:
            assert record['rhs_lifting_grid']['far_face_pairs'] > 0
        del prepared, handle, dense
    print(json.dumps({'frequency_mesh': frequency_mesh, 'eps_r': epsilon,
                      'order': config.stencil_order, 'rhs_relative_error': errors[0],
                      'order_plus_one_rhs_relative_error': errors[1]}), flush=True)
    # The RHS follows the operator's action accuracy (matvec tests allow 3x target_accuracy)
    # and converges with stencil order.
    assert errors[0] <= 3*config.target_accuracy and errors[1] <= 3e-4, errors


@pytest.mark.parametrize('grids', ['single', 'clusters'])
def test_shared_grid_rhs_body_local_and_multiple_carriers(grids):
    # The weak-contrast body's rows must not receive the stronger body's carrier.
    scene = _scene(3, [((-1.25, 0., 0.), 4.), ((1.25, 0., 0.), 1.+1e-6)])
    prepared = mw.Simulation.mom(scene, frequency=FREQUENCY,
                                config=_config(mw.PFFTConfig(exterior_grids=grids))).prepare()
    handle = prepared.solver.handle
    second = (mw.PlaneWave(direction=(1., 0., 0.), polarization=(0., 0., 1.),
                           source_time=mw.CW(frequency=FREQUENCY, phase=.37)),)
    batched, decoders = handle.build_rhs_batch((scene.sources, second, ()))
    assert _relative(batched[0], prepared.solver.rhs) < 1e-14
    assert torch.count_nonzero(batched[2]) == 0
    errors = []
    for carrier, decoder in enumerate(decoders[:2]):
        for i, body_handle in enumerate(decoder.body_handles):
            dense, _ = build_lifted_rhs(body_handle)
            local = torch.cat((batched[carrier, handle.system.J_slices[i]],
                               batched[carrier, handle.system.M_slices[i]]))
            errors.append(_relative(local, dense))
        assert decoder.metadata['rhs_bodies'][0]['rhs_source_batch'] == 3
    print(json.dumps({'exterior_grids': grids, 'body_carrier_rhs_relative_errors': errors}), flush=True)
    assert max(errors) <= 3*mw.PFFTConfig().target_accuracy, errors


@pytest.mark.parametrize('epsilon', [.25, 1.])
def test_shared_grid_rhs_finer_background_and_matched_medium(epsilon):
    scene = _scene(3, [((0., 0., 0.), epsilon)])
    prepared = mw.Simulation.mom(scene, frequency=FREQUENCY, config=_config()).prepare()
    handle = prepared.solver.handle
    grid = handle.metadata['rhs_bodies'][0]['rhs_lifting_grid']
    if epsilon == 1.:
        assert torch.count_nonzero(prepared.solver.rhs) == 0
    else:
        assert grid['grid_region'] == 'exterior_finer_spacing'
        assert grid['spacing_m'] == handle.actions.exterior.region.config.spacing
        dense, _ = build_lifted_rhs(handle.body_handles[0])
        assert _relative(prepared.solver.rhs, dense) <= 3*mw.PFFTConfig().target_accuracy

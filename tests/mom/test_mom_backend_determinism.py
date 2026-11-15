"""Repeated runs of each homogeneous MoM backend are bitwise identical without deterministic-mode switches."""
from dataclasses import replace
import runpy

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.fmm_config import DirectionalFMMConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
REPEATS = 3
EXTERIOR = ((.2, .3, .4),)


def _replay(run):
    first = run()
    for _ in range(REPEATS-1):
        again = run()
        assert again.keys() == first.keys()
        for key, value in first.items():
            assert torch.equal(again[key], value), key


def _linear(scene, config, points=EXTERIOR):
    def run():
        result = mw.Simulation.mom(scene, frequencies=3e8, config=config).run()
        electric, magnetic = result.query_fields(torch.tensor(points, device='cuda', dtype=torch.float64),
                                                 region_id='exterior', field_kind='total')
        return {'coefficients': mw.mom.surface_currents(result).scattered_coefficients.clone(), 'E': electric, 'H': magnetic}
    _replay(run)


@pytest.mark.parametrize('backend', ['hmatrix', 'h2'])
def test_hierarchical_backends_replay_bitwise(backend):
    spheres = (mw.Sphere(radius=.1),) if backend == 'hmatrix' else tuple(
        mw.Sphere(radius=.01, position=(x, 0., 0.)) for x in (-.5, .5))
    scene = mw.Scene(structures=[mw.Structure(s, mw.Material(eps_r=2.)) for s in spheres], sources=[mw.PlaneWave()])
    config = mw.mom.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2 if backend == 'hmatrix' else 3, singular_order=3),
        stopping=StoppingRule(method='gmres', preconditioner='row_column_norm', rtol=1e-12, atol=0, maxiter=240, restart=120),
        query_config=FieldQueryConfig(adaptive=False), operator_backend=backend, riesz_storage='local_rwg_v1',
        hierarchical=mw.mom.HMatrixConfig(pivoting='adaptive'))
    _linear(scene, config)


def test_pfft_backend_replay_bitwise():
    module = runpy.run_path('tests/mom/test_pfft_public.py')
    _linear(module['_scene'](), module['_pfft_config'](), points=((1.5, .3, .4),))


def test_directional_fmm_backend_replay_bitwise():
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.02, position=(i, 1e-6*(i % 2), 0.)),
        mw.Material(eps_r=2.+i, mu_r=1.3+.1*i, sigma_e=.01, sigma_m=.02)) for i in range(3)], sources=[mw.PlaneWave()])
    fmm = DirectionalFMMConfig(leaf_size=4, separation_ratio=.15, cone_subdivisions=(2**20,), max_contexts=4096,
        tensor_workspace_budget_bytes=512*1024**2, leaf_factor_kernel='original', leaf_factor_max_edges=16)
    config = mw.mom.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=6, singular_order=3, pair_batch=4,
        far_pair_batch=16, near_max_order=32), operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=1e-12, atol=0., maxiter=240, restart=120))

    # Preparation builds the shift tables; forward and adjoint actions cover
    # the packed near field and the directional far field without a solve.
    def run():
        actions = mw.Simulation.mom(scene, frequencies=3e8, config=config).prepare().solver.handle.actions
        generator = torch.Generator(device='cuda').manual_seed(7)
        x = torch.randn((2, actions.shape[1]), generator=generator, device='cuda', dtype=torch.complex128)
        return {'apply': actions.apply(x), 'adjoint': actions.adjoint(x)}
    _replay(run)


@pytest.mark.parametrize('backend', ['dense', 'hmatrix'])
def test_anisotropic_tensor_interior_replay_bitwise(backend):
    from test_linear_volume_state import system_and_volumes
    _, volumes = system_and_volumes(torch.float64)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(v.boundary.vertices, v.boundary.faces, recenter=False,
        fill_mode='solid'), mw.Material(epsilon_tensor=mw.DiagonalTensor3(*([2.+i]*3)),
        mu_tensor=mw.DiagonalTensor3(*([1.2+i*.1]*3)))) for i, v in enumerate(volumes)],
        sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8))])
    config = mw.mom.MoMConfig(integration=IntegrationConfig(order=2, singular_order=2),
        query_config=FieldQueryConfig(adaptive=False), operator_backend=backend,
        riesz_storage='compact_rwg_v1' if backend == 'dense' else 'local_rwg_v1',
        stopping=StoppingRule(method='direct') if backend == 'dense' else
            StoppingRule(method='gmres', fallback='none', rtol=1e-12, maxiter=240, restart=120),
        dyadic_integration=mw.mom.DyadicIntegrationConfig())
    _linear(scene, config, points=((1., 1., 1.),))


def test_harmonic_dc_current_replay_bitwise():
    scene, config = runpy.run_path('tests/mom/test_compressed_hb_public.py')['setup']('current', 'hmatrix')
    config = replace(config, operator_backend='dense', hierarchical=None, riesz_storage='compact_rwg_v1',
                     stopping=StoppingRule(method='direct'))
    point = config.harmonic_balance.volumes[0].vertices.new_tensor([[.16, .1, .09]])

    def run():
        result = mw.Simulation.mom(scene, frequencies=3e8, config=config).run()
        state = mw.mom.harmonic_state(result)
        values = {name: getattr(state, name).clone() for name in
                  ('cell_electric', 'polarization', 'total_volume_current', 'dc_surface_charge')}
        values.update({f'surface{i}': value.clone() for i, value in enumerate(state.surface_coefficients)})
        values.update({f'dc{i}': value for i, value in enumerate(mw.mom.query_dc(result, point, region_id=0))})
        return values
    _replay(run)


def test_periodic_backend_replay_bitwise():
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.07), mw.Material(eps_r=2.))], sources=[mw.PlaneWave()])
    periodic = mw.mom.MoMPeriodicConfig(vectors_m=((1., 0., 0.),), reciprocal_shells=4, quadrature_order=128)
    config = mw.mom.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2, singular_order=3),
        query_config=FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1', periodic=periodic)
    _linear(scene, config)

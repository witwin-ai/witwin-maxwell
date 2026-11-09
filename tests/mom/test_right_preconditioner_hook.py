"""User-supplied right preconditioners through Scene -> Simulation -> Result."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

FREQUENCY = 3e8
INTEGRATION = IntegrationConfig(order=4, singular_order=4)


def _scene():
    radius = 0.15
    structures = [mw.Structure(mw.Sphere(radius=radius, position=(x, 0., 0.)), mw.Material(eps_r=eps))
                  for x, eps in ((-0.4, 4.0), (0.4, 2.5))]
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()])


def _config(preconditioner=None, *, scaling='none', rtol=1e-10, **kwargs):
    return mw.MoMConfig(subdivisions=1, integration=INTEGRATION, riesz_storage='compact_rwg_v1',
                        stopping=StoppingRule(method='gmres', preconditioner=scaling, rtol=rtol, atol=0.,
                                              maxiter=600, restart=600, stagnation_window=200),
                        right_preconditioner=preconditioner, **kwargs)


def _run(config):
    prepared = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=config).prepare()
    return prepared, prepared.run()


def _block_jacobi(context):
    """A user-written block-Jacobi action built only from the public context."""
    groups = context.leaf_partition(16)
    blocks = []
    for edges in groups:
        index = context.coefficient_indices(edges)
        blocks.append((index, torch.linalg.inv(context.principal_block(edges))))

    def apply(residual):
        result = torch.empty_like(residual)
        for index, inverse in blocks:
            result[index] = inverse @ residual[index]
        return result
    return apply


@pytest.fixture(scope='module')
def plain():
    return _run(_config(None))


@pytest.fixture(scope='module')
def reference():
    config = mw.MoMConfig(subdivisions=1, integration=INTEGRATION, riesz_storage='compact_rwg_v1',
                          stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))
    prepared, result = _run(config)
    return prepared, result.surface_currents.scattered_coefficients


def _relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


def test_default_path_is_the_unchanged_gmres(plain):
    prepared, result = plain
    stats = result.stats()
    assert 'krylov_method' not in stats and 'right_preconditioner' not in stats
    handle = prepared.solver.handle
    direct, _ = solve_reference(handle, prepared.solver.rhs, prepared.simulation.config.stopping)
    assert torch.equal(direct, result.surface_currents.scattered_coefficients)


def test_identity_reproduces_unpreconditioned_gmres(reference, plain):
    _, plain = plain
    _, identity = _run(_config(mw.IdentityRightPreconditioner()))
    a, b = plain.stats(), identity.stats()
    assert a['converged'] and b['converged']
    assert b['krylov_method'] == 'flexible_right_gmres'
    assert b['right_preconditioner']['version'] == 'identity_right_preconditioner_v1'
    assert abs(a['iterations']-b['iterations']) <= 1
    assert b['preconditioner_applications'] == b['iterations']
    x_plain = plain.surface_currents.scattered_coefficients
    x_identity = identity.surface_currents.scattered_coefficients
    assert _relative(x_identity, x_plain) < 1e-8
    assert _relative(x_identity, reference[1]) < 1e-7


@pytest.mark.parametrize('factory', [
    mw.FunctionRightPreconditioner(_block_jacobi, name='test_block_jacobi'),
    mw.LeafBlockRightPreconditioner(16)])
def test_block_jacobi_reduces_iterations_with_true_residual_admission(reference, plain, factory):
    _, plain = plain
    prepared, result = _run(_config(factory))
    stats = result.stats()
    assert stats['converged'] and stats['relative_residual'] <= 1e-10
    assert stats['iterations'] < plain.stats()['iterations']
    assert stats['right_preconditioner']['principal_block_requests'] > 0
    x = result.surface_currents.scattered_coefficients
    handle = prepared.solver.handle
    true = torch.linalg.vector_norm(prepared.solver.rhs-handle.apply(x))/torch.linalg.vector_norm(prepared.solver.rhs)
    assert float(true) <= 1e-10
    assert _relative(x, reference[1]) < 1e-7


def test_variable_preconditioner_still_converges(reference):
    calls = []

    def factory(context):
        inner = _block_jacobi(context)

        def apply(residual):
            calls.append(None)
            # A deliberately non-stationary action: alternating scale and a
            # nonlinear magnitude clip. Plain right GMRES would be inconsistent.
            scale = 0.5 if len(calls) % 2 else 2.0
            z = scale*inner(residual)
            return z/(1+1e-3*torch.tanh(z.abs()))
        return apply
    _, result = _run(_config(mw.FunctionRightPreconditioner(factory, name='variable')))
    stats = result.stats()
    assert stats['converged'] and stats['relative_residual'] <= 1e-10
    assert _relative(result.surface_currents.scattered_coefficients, reference[1]) < 1e-7


def test_fast_backend_keeps_its_residual_admission():
    """FMM (shared HierarchicalOperator.solve) keeps compressed and sampled-original admission."""
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
                                              mw.Material(eps_r=eps, mu_r=mu))
                                 for x, eps, mu in ((-.2, 2., 1.2), (.2, 3., 1.3))], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=4, singular_order=3),
                          operator_backend='fmm', riesz_storage='local_rwg_v1',
                          fmm=mw.FMMConfig(order=6, leaf_size=8, separation_ratio=.25),
                          stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=1e-10, atol=0.,
                                                maxiter=400, restart=400, stagnation_window=200),
                          right_preconditioner=mw.LeafBlockRightPreconditioner(8))
    result = mw.Simulation.mom(scene, frequency=9e7, config=config).run()
    stats = result.stats()
    assert stats['converged'] and stats['error_budget']['checked']
    assert stats['error_budget']['compressed_dual_pass'] and stats['error_budget']['original_sampled_pass']
    assert stats['krylov_method'] == 'flexible_right_gmres'
    assert stats['right_preconditioner']['block_source'] == 'hierarchical'


def test_configuration_admission():
    identity = mw.IdentityRightPreconditioner()
    with pytest.raises(TypeError):
        mw.MoMConfig(stopping=StoppingRule(method='gmres'), right_preconditioner=object())
    for rule in (StoppingRule(method='direct'), StoppingRule(method='block_gmres'),
                 StoppingRule(method='gmres', recycle_size=2), StoppingRule(method='gmres', fallback='direct')):
        with pytest.raises(ValueError):
            mw.MoMConfig(stopping=rule, right_preconditioner=identity)
    with pytest.raises(ValueError):
        mw.MoMConfig(operator_backend='fmm', riesz_storage='local_rwg_v1',
                     fmm=mw.FMMConfig(leaf_factor_kernel='primary'),
                     stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal'),
                     right_preconditioner=identity)
    with pytest.raises(ValueError):
        _run(_config(mw.FunctionRightPreconditioner(lambda context: (lambda r: r[:-1]), name='bad_shape')))

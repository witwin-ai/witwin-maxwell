"""Public tensor-interior configuration, scalar limit, exact storage and guards."""
from dataclasses import replace
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.runtime import SurfaceData
from witwin.maxwell.mom.tensor_medium import TensorMedium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('backend', ('dense', 'hmatrix', 'h2'))
def test_public_tensor_scalar_limit_multibody_restore(backend, tmp_path, monkeypatch):
    def scene(tensor):
        return mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01, position=(x, 0., 0.)),
            mw.Material(epsilon_tensor=mw.DiagonalTensor3(eps, eps, eps),
                        mu_tensor=mw.DiagonalTensor3(mu, mu, mu)) if tensor else mw.Material(eps_r=eps, mu_r=mu),
            name=f'body{i}') for i, (x, eps, mu) in enumerate(((-.5, 2., 1.3), (.5, 3., 1.5)))],
            sources=[mw.PlaneWave()])
    physical = scene(True)
    base = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=3),
                        query_config=mw.FieldQueryConfig(adaptive=False), riesz_storage='compact_rwg_v1')
    reference = mw.Simulation.mom(scene(False), frequency=3e8, config=base).run()
    config = replace(base, dyadic_integration=mw.DyadicIntegrationConfig(), dyadic_cache_max_bytes=1024**2,
        operator_backend=backend, riesz_storage='compact_rwg_v1' if backend == 'dense' else 'local_rwg_v1',
        hierarchical=None if backend == 'dense' else mw.HMatrixConfig(pivoting='adaptive'),
        stopping=StoppingRule(method='direct') if backend == 'dense' else StoppingRule(method='gmres',
            preconditioner='row_column_norm', rtol=1e-12, atol=0., maxiter=240, restart=120))
    result = mw.Simulation.mom(physical, frequency=3e8, config=config).run()
    assert result.stats()['converged'], result.stats()
    data = result.surface_currents
    assert all(isinstance(b.interior, TensorMedium) for b in data.handle.system.bodies)
    for got, expected in zip(data.scattered_coefficients.chunk(2), reference.surface_currents.scattered_coefficients.chunk(2)):
        assert float((got-expected).norm()/expected.norm()) < 1e-8
    if backend == 'hmatrix':
        assert data.handle.metadata['far_blocks'] > 0
    if backend == 'h2':
        assert len(data.handle.actions.couplings) > 0
    points = torch.tensor([[-.5, 0., 0.]], device='cuda', dtype=torch.float64)
    fields = result.query_fields(points, region_id='body0', field_kind='total')
    for got, expected in zip(fields, reference.query_fields(points, region_id='body0', field_kind='total')):
        assert float((got-expected).norm()/expected.norm()) < 1e-8
    path = tmp_path/'tensor.pt'
    result.save(path)
    payload = data.snapshot()
    assert payload['schema_version'] == 13
    assert payload['formulation_id'] == 'anisotropic_dgf_sie'
    def forbidden(*args, **kwargs):
        raise AssertionError('Exact tensor restore must not assemble a matrix or rebuild H')
    import witwin.maxwell.mom.anisotropic_operator as assembly
    import witwin.maxwell.mom.hmatrix as hmatrix
    monkeypatch.setattr(assembly, 'assemble_generalized_system', forbidden)
    monkeypatch.setattr(hmatrix.HierarchicalOperator, '__init__', forbidden)
    if backend in ('hmatrix', 'h2'):
        from witwin.maxwell.mom.surface_h_snapshot import encode, restore
        from witwin.maxwell.mom.hierarchical_runtime import _physical_provider
        from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
        factors = encode(data.handle)
        provider = _physical_provider(data.handle.system, data.handle.exterior, config)
        vector = torch.randn_like(data.scattered_coefficients)
        actions = (data.handle.apply(vector), data.handle.adjoint(vector))
        with monkeypatch.context() as patch:
            patch.setattr(HierarchicalH2Operator, '__init__', forbidden)
            patch.setattr(type(provider), '__call__', forbidden)
            patch.setattr(torch.linalg, 'solve', forbidden)
            decoded = restore(factors, provider, sources=physical.sources)
            for actual, expected in zip((decoded.apply(vector), decoded.adjoint(vector)), actions):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = mw.Result.load(path, scene=physical)
    assert restored.surface_currents.config == config
    assert torch.equal(restored.surface_currents.scattered_coefficients, data.scattered_coefficients)
    for got, expected in zip(restored.query_fields(points, region_id='body0', field_kind='total'), fields):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    for modify in ('budget', 'material', 'qualification', 'operator', 'rhs'):
        damaged = copy.deepcopy(payload)
        if modify == 'budget':
            damaged['dyadic_cache_max_bytes'] += 1
        elif modify == 'qualification':
            damaged['operator_metadata']['physical_qualification'] = True
        elif modify in ('operator', 'rhs'):
            damaged['operator_metadata'][modify+'_identity'] = 'changed'
        else:
            damaged['body_table'][0]['interior']['mu_r_real_imag'][0][0][0] += .1
        with pytest.raises(ValueError, match='identity|material'):
            SurfaceData.restore(damaged, physical, 3e8)
    outer = torch.load(path, weights_only=False)
    outer['metadata']['physical_qualification'] = True
    damaged_path = tmp_path/'wrong_outer.pt'
    torch.save(outer, damaged_path)
    with pytest.raises(ValueError, match='metadata'):
        mw.Result.load(damaged_path, scene=physical)
    local = data.handle.body_handles[0]
    contrast = local.tensor_contrast_kernel
    local.tensor_contrast_kernel = replace(contrast, integration=replace(contrast.integration, relative_tolerance=1e-3))
    with pytest.raises(ValueError, match='binding changed'):
        data.handle.build_rhs()
    local.tensor_contrast_kernel = contrast
    if backend != 'dense':
        assert local.tensor_kernel is data.handle.provider.interior_kernels[0]
    kernel = restored.surface_currents.handle.body_handles[0].tensor_kernel
    restored.surface_currents.handle.body_handles[0].tensor_kernel = replace(kernel)
    with pytest.raises(ValueError, match='binding changed'):
        restored.query_fields(points, region_id='body0', field_kind='total')


@pytest.mark.parametrize('periodic,backend',((False,'dense'),(True,'dense'),(True,'hmatrix'),(True,'h2')))
def test_public_anisotropic_kernel_failure_is_not_a_scalar_fallback(periodic,backend):
    from witwin.maxwell.mom.dyadic_green import DyadicConvergenceError
    material = mw.Material(epsilon_tensor=mw.DiagonalTensor3(2., 3., 4.), sigma_e=.01)
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01), material)], sources=[mw.PlaneWave()])
    config = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=1, singular_order=1, pair_batch=1),
        periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)) if periodic else None,
        dyadic_integration=mw.DyadicIntegrationConfig(radial_order=2, azimuthal_order=2,
            max_radial_order=4, max_azimuthal_order=4, relative_tolerance=1e-12, absolute_tolerance=1e-15))
    if backend != 'dense':
        from witwin.maxwell.mom.linalg import StoppingRule
        config = replace(config, operator_backend=backend, riesz_storage='local_rwg_v1',
            stopping=StoppingRule(method='gmres', fallback='none'))
    with pytest.raises(DyadicConvergenceError):
        mw.Simulation.mom(scene, frequency=3e8, config=config).prepare()


def test_tensor_combinations_reject_before_construction():
    for backend in ('fmm', 'direct_h2'):
        with pytest.raises(NotImplementedError, match='Tensor interiors'):
            mw.MoMConfig(operator_backend=backend, dyadic_integration=mw.DyadicIntegrationConfig())
    for backend in ('fmm','direct_h2'):
        with pytest.raises(NotImplementedError, match='Tensor interiors'):
            mw.MoMConfig(operator_backend=backend,periodic=mw.MoMPeriodicConfig(((1., 0., 0.),)),
                         dyadic_integration=mw.DyadicIntegrationConfig())

    from witwin.maxwell.mom.linalg import StoppingRule
    for backend in ('hmatrix','h2'):
        config = mw.MoMConfig(operator_backend=backend, riesz_storage='local_rwg_v1',
            stopping=StoppingRule(method='gmres', fallback='none'),
            periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)),
            dyadic_integration=mw.DyadicIntegrationConfig())
        assert config.operator_backend == backend and config.hierarchical is not None


def test_tensor_volume_observation_uses_loss_quadratic_form_and_field_gradients():
    from witwin.maxwell.constants import EPSILON_0
    from witwin.maxwell.mom.observables import integrate_volume_dissipation
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    factor = eye.new_tensor([[.2, .1j, 0], [.1, .3, 0], [0, .1j, .4]])
    loss = factor.mH@factor
    mu = eye.clone()
    mu[0, 1], mu[1, 0] = .3j, -.3j
    medium = TensorMedium(3e8, 2*eye+1j*loss, mu, reciprocal=False)
    generator = torch.Generator(device='cuda').manual_seed(112)
    e = torch.randn((4, 3), generator=generator, device='cuda', dtype=eye.dtype, requires_grad=True)
    h = torch.randn((4, 3), generator=generator, device='cuda', dtype=eye.dtype, requires_grad=True)
    weights = torch.tensor([.1, .2, .3, .4], device='cuda', dtype=torch.float64)
    actual = integrate_volume_dissipation(medium, e, h, weights)
    expected = .5*medium.omega*EPSILON_0*((e@factor.T).abs().square().sum(-1)*weights).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
    ge, gh = torch.autograd.grad(actual, (e, h))
    reference, = torch.autograd.grad(expected, e)
    torch.testing.assert_close(ge, reference, rtol=1e-13, atol=1e-15)
    assert torch.count_nonzero(gh) == 0


def test_complex_constitutive_declaration_public_scalar_limit_and_restore(tmp_path):
    from witwin.maxwell.constants import EPSILON_0, MU_0
    import math
    frequency = 3e8
    omega = 2*math.pi*frequency
    def diagonal(value):
        return mw.ComplexTensor3x3(tuple(tuple(value if i == j else 0 for j in range(3)) for i in range(3)))
    def scene(material):
        return mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01), material)], sources=[mw.PlaneWave()])
    physical = scene(mw.Material(epsilon_tensor=diagonal(2+.2j), mu_tensor=diagonal(1.3+.1j)))
    scalar = scene(mw.Material(eps_r=2, mu_r=1.3, sigma_e=.2*omega*EPSILON_0, sigma_m=.1*omega*MU_0))
    base = mw.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=3, singular_order=3),
                        query_config=mw.FieldQueryConfig(adaptive=False))
    expected = mw.Simulation.mom(scalar, frequency=frequency, config=base).run()
    result = mw.Simulation.mom(physical, frequency=frequency,
        config=replace(base, dyadic_integration=mw.DyadicIntegrationConfig())).run()
    assert result.stats()['converged']
    for actual, reference in zip(result.surface_currents.scattered_coefficients.chunk(2),
                                 expected.surface_currents.scattered_coefficients.chunk(2)):
        assert float((actual-reference).norm()/reference.norm()) < 1e-8
    result.save(tmp_path/'complex.pt')
    restored = mw.Result.load(tmp_path/'complex.pt', scene=physical)
    assert torch.equal(restored.surface_currents.scattered_coefficients, result.surface_currents.scattered_coefficients)
    assert restored.surface_currents.snapshot()['operator_metadata']['physical_qualification'] is False

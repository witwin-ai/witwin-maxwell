"""Checked spectral regular-part routing, budgets and contribution ownership."""
from dataclasses import replace
import runpy
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel
from witwin.maxwell.mom.periodic_dyadic import (QuasiperiodicDyadicGreenKernel,
    QuasiperiodicDyadicRegularKernel, LossyDyadicImageReference, PeriodicDyadicConvergenceError)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(dimension=2):
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    vectors = eye.real.new_tensor([[.7, 0., 0.], [0., .8, 0.], [.05, .1, .9]])[:dimension]
    lattice = LatticeSpec(vectors, vectors.new_tensor([.2, .1 if dimension >= 2 else 0., .05 if dimension == 3 else 0.]))
    kernel = QuasiperiodicDyadicGreenKernel(lattice, TensorMedium(3e8, (1+2j)*eye, (1+2j)*eye), mode_batch=256)
    return QuasiperiodicDyadicRegularKernel(kernel, 8, .12, 3, 1e-4, 16,
        normal=vectors.new_tensor([0., 0., 1.]) if dimension == 1 else None,
        transverse_cutoff=64., quadrature_order=64, transverse_map='sinh')


@pytest.mark.parametrize('dimension,location', [(1, 'plane'), (2, 'plane'), (3, 'plane'),
    (1, 'offplane'), (2, 'offplane'), (3, 'offplane'), (3, 'translated')])
def test_actual_primary_construction_receives_regular_cache_policy(monkeypatch, dimension, location):
    regular = replace(problem(dimension), primary_cache_max_bytes=0)
    payload = regular.payload()
    assert payload['version'] == 'periodic_dyadic_regular_config_v2'
    restored = QuasiperiodicDyadicRegularKernel.from_payload(payload,
        medium=regular.medium, lattice=regular.lattice)
    assert restored.primary_cache_max_bytes == 0 and restored.identity == regular.identity
    observed = []
    original = DyadicGreenKernel.__post_init__
    class ReachedPrimary(RuntimeError):
        pass
    def constructed(self):
        original(self)
        observed.append(self.cache_max_bytes)
        raise ReachedPrimary('Actual free-space primary construction reached')
    monkeypatch.setattr(DyadicGreenKernel, '__post_init__', constructed)
    def spectrum(self, points, **kwargs):
        return self.medium.epsilon_r.new_zeros(len(points), 6, 6), {'converged': True}
    point = regular.lattice.vectors.new_tensor([[.1, .07, 0.]])
    if location == 'offplane':
        point[:, 2] = .2
        monkeypatch.setattr(QuasiperiodicDyadicGreenKernel,
            {1: 'estimate_line', 2: 'estimate_open', 3: 'estimate_stacked'}[dimension], spectrum)
    elif location == 'translated':
        point = point+regular.lattice.vectors[2]
        monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'estimate_stacked_plane_regular', spectrum)
    with pytest.raises(PeriodicDyadicConvergenceError) as failure:
        restored.evaluate(point, exclude_primary=True)
    assert isinstance(failure.value.__cause__, ReachedPrimary)
    assert observed == [0]


@pytest.mark.parametrize('dimension', (1, 2, 3))
def test_mixed_regular_queries_keep_actual_offsets_and_absolute_budgets(monkeypatch, dimension):
    regular = problem(dimension)
    sample = regular.lattice.vectors
    points = sample.new_tensor([[0., 0., 0.], [.1, .07, .2], [.1, .07, -1e-8]])
    observed = []
    def plane(self, queries, **kwargs):
        assert torch.equal(queries, points[:1])
        assert kwargs['absolute_tolerance'] == regular.absolute_tolerance
        observed.append('plane')
        return self.medium.epsilon_r.new_full((len(queries), 6, 6), 2), {'converged': True}
    def spectrum(self, queries, **kwargs):
        assert torch.equal(queries, points[1:])
        assert kwargs['relative_tolerance'] == 0
        assert kwargs['absolute_tolerance'] == regular.absolute_tolerance/8
        observed.append('spectrum')
        return self.medium.epsilon_r.new_full((len(queries), 6, 6), 7), {'cutoff_converged': True}
    def primary(self, queries):
        assert torch.equal(queries, points[1:])
        assert self.integration.relative_tolerance == 0
        assert self.integration.absolute_tolerance == regular.absolute_tolerance/8
        observed.append('primary')
        return self.medium.epsilon_r.new_full((len(queries), 6, 6), 3), {'converged': True}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel,
        {1: 'estimate_line_plane_regular', 2: 'estimate_open_plane_regular', 3: 'estimate_stacked_plane_regular'}[dimension], plane)
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel,
        {1: 'estimate_line', 2: 'estimate_open', 3: 'estimate_stacked'}[dimension], spectrum)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', primary)
    value, receipt = regular.evaluate(points, exclude_primary=True)
    assert observed == ['plane', 'spectrum', 'primary']
    assert torch.equal(value[:, 0, 0], value.new_tensor([2, 4, 4]))
    assert receipt['converged'] and receipt['primary_image_excluded']
    assert not receipt['physical_qualification'] and not receipt['certified_error_bound']


def test_regular_budget_failures_retain_nested_receipts_and_precision_guard(monkeypatch):
    regular = problem()
    points = regular.lattice.vectors.new_tensor([[.1, .1, .2]])
    def failed(self, queries, **kwargs):
        raise PeriodicDyadicConvergenceError({'failed_axis': 'reciprocal_tail'})
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'estimate_open', failed)
    with pytest.raises(PeriodicDyadicConvergenceError) as error:
        regular.evaluate(points, exclude_primary=True)
    assert error.value.receipt['component_failure']['failed_axis'] == 'reciprocal_tail'
    def huge(self, queries, **kwargs):
        return self.medium.epsilon_r.new_full((len(queries), 6, 6), 1e20), {'converged': True}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'estimate_open', huge)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', huge)
    with pytest.raises(PeriodicDyadicConvergenceError) as error:
        regular.evaluate(points, exclude_primary=True)
    assert error.value.receipt['failure_reason'] == 'primary_subtraction_precision_budget_exhausted'


@pytest.mark.parametrize('orientation', (1, -1))
def test_stacked_regular_translation_preserves_original_primary(monkeypatch, orientation):
    regular = problem(3)
    vectors = regular.lattice.vectors.clone()
    vectors[2] *= orientation
    regular = replace(regular, kernel=replace(regular.kernel,
        lattice=LatticeSpec(vectors, regular.lattice.bloch)))
    base = vectors.new_tensor([[.11, .07, 0.]])
    layers = vectors.new_tensor([0., 1., -2.])
    points = base+layers[:, None]*vectors[2]
    # The manufactured full Green value has exact Bloch covariance. The
    # nonperiodic primary makes incorrect phase-only regular wrapping visible.
    pattern = regular.medium.epsilon_r.new_zeros(6, 6)
    pattern[0, 0] = 1
    def full(queries):
        return torch.exp(1j*(queries@regular.lattice.bloch))[:, None, None]*pattern
    def primary(queries):
        return (2+queries.square().sum(-1))[:, None, None]*pattern
    budgets = []
    def plane(self, queries, **settings):
        budgets.append(settings['absolute_tolerance'])
        assert bool(((queries@regular.normal).abs() < 1e-14).all())
        return full(queries)-primary(queries), {'converged': True}
    def free(self, queries):
        assert self.integration.absolute_tolerance == regular.absolute_tolerance/8
        return primary(queries), {'converged': True}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'estimate_stacked_plane_regular', plane)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', free)
    value, receipt = regular.evaluate(points, exclude_primary=True)
    torch.testing.assert_close(value, full(points)-primary(points), rtol=1e-13, atol=1e-13)
    assert budgets == [regular.absolute_tolerance, regular.absolute_tolerance/2]
    assert receipt['source_plane_count'] == 3
    assert receipt['source_plane']['translated_query_count'] == 2
    assert receipt['source_plane']['converged'] and receipt['converged']
    assert not receipt['physical_qualification']


def test_stacked_regular_translation_rejects_precision_loss_and_nonprimary_singularities(monkeypatch):
    regular = problem(3)
    vectors = regular.lattice.vectors
    with pytest.raises(ValueError, match='half the lattice-plane spacing'):
        replace(regular, height=.45)
    with pytest.raises(ValueError, match='Nonprimary lattice singularity'):
        regular.evaluate(-100*vectors[2:], exclude_primary=True)
    def plane(self, queries, **settings):
        return self.medium.epsilon_r.new_zeros(len(queries), 6, 6), {'converged': True}
    def huge(self, queries):
        return self.medium.epsilon_r.new_full((len(queries), 6, 6), 1e20), {'converged': True}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel, 'estimate_stacked_plane_regular', plane)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', huge)
    points = vectors.new_tensor([[.11, .07, 0.]])+vectors[2]
    with pytest.raises(PeriodicDyadicConvergenceError) as error:
        regular.evaluate(points, exclude_primary=True)
    assert error.value.receipt['failure_reason'] == 'primary_translation_precision_budget_exhausted'
    assert not error.value.receipt['source_plane']['converged']


@pytest.mark.parametrize('dimension', (1, 2, 3))
def test_regular_input_singularity_ownership_and_empty_queries(dimension):
    regular = problem(dimension)
    sample = regular.lattice.vectors
    with pytest.raises(ValueError, match='primary exclusion'):
        regular.evaluate(sample[:1])
    with pytest.raises(ValueError, match='Nonprimary lattice singularity'):
        regular.evaluate(100*sample[:1], exclude_primary=True)
    value, receipt = regular.evaluate(sample.new_empty(0, 3), exclude_primary=True)
    assert value.shape == (0, 6, 6) and receipt['query_count'] == 0
    clone = replace(regular, absolute_tolerance=2e-4)
    assert clone.integration.absolute_tolerance == 2e-4/8
    regular.normal[0] += .1
    with pytest.raises(ValueError, match='normal changed'):
        regular.evaluate(sample.new_empty(0, 3), exclude_primary=True)
    for update in ({'shells': 0}, {'max_halvings': 2}, {'absolute_tolerance': float('inf')}, {'height': 0}):
        with pytest.raises(ValueError):
            replace(clone, **update)


@pytest.mark.parametrize('dimension', (1, 2))
def test_actual_off_plane_regular_values_match_independent_images(dimension):
    regular = problem(dimension)
    points = regular.lattice.vectors.new_tensor([[.1, .07, .4], [.12, -.03, -.4]])
    value, receipt = regular.evaluate(points, exclude_primary=True)
    expected, reference_receipt = LossyDyadicImageReference(regular.lattice, regular.medium, 4).estimate_regular(
        points, relative_tolerance=0., absolute_tolerance=1e-7)
    from witwin.maxwell.mom.periodic_dyadic import _cutoff_check
    assert _cutoff_check(expected, value, 0., regular.absolute_tolerance)['passed']
    assert reference_receipt['primary_image_excluded'] and receipt['source_plane_count'] == 0


@pytest.mark.parametrize('dimension', (1, 3))
def test_spectral_regular_adapter_reaches_matrix_and_field_convolution(monkeypatch, dimension):
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic_operator import assemble_periodic_dyadic_correction
    from witwin.maxwell.mom.periodic_dyadic_fields import regular_dyadic_surface_fields
    handle, _ = runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    regular = problem(dimension)
    regular = replace(regular, kernel=replace(regular.kernel,
        lattice=LatticeSpec(regular.lattice.vectors.new_tensor(
            [[1., 0., 0.], [0., 1.1, 0.], [.05, .1, 1.2]])[:dimension], regular.lattice.bloch)))
    image = LossyDyadicImageReference(regular.lattice, regular.medium, 1)
    seen = []
    def value(self, points, *, exclude_primary=False):
        assert exclude_primary is True
        seen.append(type(self))
        green = self.medium.epsilon_r.new_ones(len(points), 6, 6)
        return green, {'primary_image_excluded': True, 'manufactured_routing_control': True}
    monkeypatch.setattr(QuasiperiodicDyadicRegularKernel, 'evaluate', value)
    monkeypatch.setattr(LossyDyadicImageReference, 'evaluate', value)
    config = IntegrationConfig(order=1, singular_order=1)
    matrix, receipt = assemble_periodic_dyadic_correction(handle.system, reference=regular, config=config)
    expected, _ = assemble_periodic_dyadic_correction(handle.system, reference=image, config=config)
    torch.testing.assert_close(matrix, expected, rtol=0, atol=0)
    assert not receipt['primary_included'] and not receipt['interiors_periodic']
    torch.manual_seed(97)
    coefficients = torch.randn(2*handle.system.total_edges, device=matrix.device, dtype=matrix.dtype, requires_grad=True)
    points = regular.lattice.vectors.new_tensor([[.1, .1, .3]])
    actual, _ = regular_dyadic_surface_fields(handle.system, coefficients, points, reference=regular, order=1)
    expected, _ = regular_dyadic_surface_fields(handle.system, coefficients, points, reference=image, order=1)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    derivative, = torch.autograd.grad(sum(v.abs().square().sum() for v in actual), coefficients)
    assert bool(torch.isfinite(derivative).all()) and bool(derivative.abs().max() > 0)
    assert QuasiperiodicDyadicRegularKernel in seen and LossyDyadicImageReference in seen
    from witwin.maxwell.mom import anisotropic_operator
    from witwin.maxwell.mom.periodic_dyadic_operator import assemble_periodic_dyadic_system
    primary = 3*torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype)
    def primary_system(system, medium, config, **kwargs):
        assert system is handle.system and medium is regular.medium
        return primary, {'manufactured_primary_ownership_control': True}
    monkeypatch.setattr(anisotropic_operator, 'assemble_generalized_system', primary_system)
    complete, complete_receipt = assemble_periodic_dyadic_system(handle.system, reference=regular, config=config)
    torch.testing.assert_close(complete, primary+matrix, rtol=0, atol=0)
    assert complete_receipt['kernel_id'] == regular.kernel_id
    assert complete_receipt['primary_included'] and not complete_receipt['interiors_periodic']
    from witwin.maxwell.mom.periodic_dyadic_fields import exterior_dyadic_surface_fields
    actual, exterior_receipt = exterior_dyadic_surface_fields(handle.system, coefficients, points,
        reference=regular, order=1)
    expected, _ = exterior_dyadic_surface_fields(handle.system, coefficients, points, reference=image, order=1)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert exterior_receipt['primary_included'] and not exterior_receipt['interiors_included']


@pytest.mark.parametrize('dimension', (2, 3))
@pytest.mark.parametrize('outcome', ('converged', 'exhausted', 'pole'))
def test_offplane_refinement_preserves_budget_and_failed_attempts(monkeypatch, dimension, outcome):
    # An odd, non-power-of-two cap must still bound every doubled comparison.
    regular = replace(problem(dimension), max_shells=70)
    points = regular.lattice.vectors.new_tensor([[.01, .02, .003]])
    calls = []
    def spectrum(self, queries, **settings):
        assert torch.equal(queries, points)
        assert settings['relative_tolerance'] == 0
        assert settings['absolute_tolerance'] == regular.absolute_tolerance/8
        shells = settings['shells']
        calls.append(shells)
        assert 2*shells <= regular.max_shells
        if outcome == 'pole':
            raise PeriodicDyadicConvergenceError({'failure_reason': 'spectral_pole'})
        passed = outcome == 'converged' and shells == 35
        record = dict(cutoff_converged=passed, cutoff_check={'passed': passed},
            compared_shells=[shells, 2*shells])
        if not passed:
            raise PeriodicDyadicConvergenceError(record)
        return self.medium.epsilon_r.new_zeros(len(queries), 6, 6), record
    primary_calls = []
    def primary(self, queries):
        primary_calls.append(1)
        return self.medium.epsilon_r.new_zeros(len(queries), 6, 6), {'converged': True}
    monkeypatch.setattr(QuasiperiodicDyadicGreenKernel,
        'estimate_open' if dimension == 2 else 'estimate_stacked', spectrum)
    monkeypatch.setattr(DyadicGreenKernel, 'evaluate', primary)
    if outcome == 'converged':
        _, receipt = regular.evaluate(points, exclude_primary=True)
        assert receipt['converged'] and primary_calls == [1]
        assert receipt['off_plane_spectrum']['compared_shells'] == [35, 70]
    else:
        with pytest.raises(PeriodicDyadicConvergenceError) as failure:
            regular.evaluate(points, exclude_primary=True)
        receipt = failure.value.receipt
        assert not receipt['converged'] and primary_calls == []
        assert receipt['component_failure'] == receipt['off_plane_spectral_attempts'][-1]
    assert calls == ([8] if outcome == 'pole' else [8, 16, 32, 35])
    assert len(receipt['off_plane_spectral_attempts']) == len(calls)
    assert not receipt['physical_qualification']

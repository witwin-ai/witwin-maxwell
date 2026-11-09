"""New shared-grid paired spatial references and complete radial coordinates."""
import json
import math
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig, integrate_spatial_green
from witwin.maxwell.mom.dyadic_contrast import DyadicContrastKernel
from mp_matched_spd_spatial_reference import mp_matched_spd_spatial_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
OUTSIDE = [1.4, 1.1, .85]
INSIDE = [2.7, 1.9, 1.35]
POINTS = [[.052, .018, .061], [.049, -.021, .063]]


def write_raw(path, value):
    def encode(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {'nonfinite_value': repr(item)}
        if isinstance(item, dict):
            return {key: encode(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(v) for v in item]
        return item
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False)+'\n')


def medium(diagonal=INSIDE):
    tensor = torch.diag(torch.tensor(diagonal, device='cuda', dtype=torch.complex128))
    return TensorMedium(3.6e7, tensor, tensor)


def config():
    return DyadicIntegrationConfig(radial_order=8, azimuthal_order=8,
        max_radial_order=256, max_azimuthal_order=128, relative_tolerance=2e-4,
        radial_map='paired_spd_reference_branch_v1', spatial_rule='factored_spd_pair_tail_v1')


def test_new_paired_spatial_MP100_and_final_grid_cache_reuse(tmp_path, monkeypatch):
    from witwin.maxwell.mom import dyadic_contrast, dyadic_green
    raw = dict(version='new_two_factor_paired_spatial_MP100_v1', MP_digits=100,
        points=POINTS, epsilon_mu_outside=OUTSIDE, epsilon_mu_inside=INSIDE,
        frequency_Hz=3.6e7, blocks=[], completed=False,
        old_plane_single_spatial_or_original_public_FAIL_not_replayed=True,
        production_qualification=False)
    path = tmp_path/'paired-spatial.json'
    write_raw(path, raw)
    with mp.workdps(100):
        values = []
        for point in POINTS:
            i = mp_matched_spd_spatial_reference(point, INSIDE, 3.6e7, regular_part=True, return_mp=True)
            e = mp_matched_spd_spatial_reference(point, OUTSIDE, 3.6e7, regular_part=True, return_mp=True)
            difference = i-e
            values.append([[complex(difference[a, b]) for b in range(6)] for a in range(6)])
        expected = torch.tensor(values, dtype=torch.complex128)
    artifact = tmp_path/'paired-spatial-native-reference.pt'
    saved = dict(expected=expected, points=POINTS, outside=OUTSIDE, inside=INSIDE)
    torch.save(saved, artifact)
    exterior, interior = medium(OUTSIDE), medium()
    kernel = DyadicContrastKernel(exterior, interior, config())
    points = torch.tensor(POINTS, device='cuda', dtype=torch.float64)
    try:
        actual, receipt = kernel.evaluate_regular(points)
    except (ValueError, RuntimeError) as error:
        raw.update(failure=str(error), receipt=getattr(error, 'receipt', None))
        write_raw(path, raw)
        raise
    reference = expected.to('cuda')
    for query in range(2):
        for name, a, b in (('EE', 0, 0), ('EH', 0, 3), ('HE', 3, 0), ('HH', 3, 3)):
            target = reference[query, a:a+3, b:b+3]
            norm = float(target.norm())
            error = float((actual[query, a:a+3, b:b+3]-target).norm()/target.norm())
            raw['blocks'].append(dict(query=query, block=name, reference_norm=norm if math.isfinite(norm) else repr(norm),
                relative_error=error if math.isfinite(error) else repr(error), gate=.005,
                passed=math.isfinite(norm) and norm > 0 and math.isfinite(error) and error < .005))
    saved.update(native=actual.detach(), receipt=receipt)
    torch.save(saved, artifact)
    raw.update(receipt=receipt, spatial_evaluation_completed=True)
    write_raw(path, raw)
    assert receipt['converged'] and receipt['radial_intervals'] == 3
    assert receipt['kernel_version'] == 'dyadic_contrast_two_factor_spd_tail_v1'
    assert all(block['passed'] for block in raw['blocks'])
    cache = kernel.mode_cache
    storages = {}
    for entry, _, _ in cache._entries.values():
        for tensor in cache._tensors(entry):
            storages[(str(tensor.device), tensor.untyped_storage().data_ptr())] = tensor.untyped_storage().nbytes()
    assert sum(storages.values()) == cache.retained_bytes <= cache.max_bytes
    radial, angular = receipt['final_orders']
    key = (2, 1, -5, radial, angular, True, config().radial_map, config().spatial_rule)
    grid = cache._entries[key][0][0]
    assert grid.tail is not None
    # This is a new cache action on the one saved final grid, not a replay of
    # reference construction, adaptation, or a previous successful workflow.
    def forbidden(*args, **kwargs):
        raise AssertionError('Paired final-grid reuse must not compile again')
    for name in ('compile_plane_contrast', 'compile_electrostatic_plane_contrast',
                 '_compile_factored_plane_remainder_contrast', '_compile_regular_plane_contrast_grid'):
        monkeypatch.setattr(dyadic_contrast, name, forbidden)
    monkeypatch.setattr(dyadic_green, '_compile_regular_plane_grid', forbidden)
    reused = integrate_spatial_green(points, interior, _exterior=exterior,
        radial_order=radial, azimuthal_order=angular, restore_static=False,
        radial_map=config().radial_map, spatial_rule=config().spatial_rule, mode_cache=cache)
    torch.testing.assert_close(reused, actual, rtol=0, atol=0)
    before = grid.tail.state['projector']._version
    grid.tail.state['projector'].requires_grad_(True)
    assert grid.tail.state['projector']._version == before
    with pytest.raises(ValueError): cache.get(key, forbidden)
    with pytest.raises(ValueError): grid.integrate(points, torch.ones(len(grid.transverse), device='cuda', dtype=torch.float64))
    raw.update(completed=True, cache_unique_retained_bytes=sum(storages.values()),
        final_grid_reused_without_compile=True, gradient_flag_mutation_rejected=True)
    write_raw(path, raw)


@pytest.mark.parametrize('equal', [False, True])
def test_new_complete_paired_radial_coordinates_analytic_moments(equal, tmp_path):
    from witwin.maxwell.mom.quadrature import gauss_rule
    from witwin.maxwell.mom.matched_spd_green import paired_spd_reference_transverse_rule
    nodes, weights = gauss_rule(64, device='cuda', dtype=torch.float64)
    phi = torch.arange(16, device='cuda', dtype=torch.float64)*(2*math.pi/16)
    circle = torch.stack((phi.cos(), phi.sin()), -1)
    exterior, interior = medium(OUTSIDE), medium(OUTSIDE if equal else INSIDE)
    transverse, physical_weights = paired_spd_reference_transverse_rule(nodes, weights, 1., .8, exterior, interior, circle)
    radius = transverse.norm(dim=-1)
    errors = []
    for power in range(3):
        native = (physical_weights*radius.pow(power)*torch.exp(-radius)).sum()
        with mp.workdps(100): target = float(mp.factorial(power+1)/(2*mp.pi))
        error = float((native-target).abs()/target)
        errors.append(error)
        assert math.isfinite(error) and error < 1e-9
    if equal:
        middle = physical_weights.reshape(3, 64, 16)[1]
        assert torch.count_nonzero(middle) == 0
    (tmp_path/'paired-radial-moments.json').write_text(json.dumps(dict(
        version='new_complete_paired_reference_radial_moments_v1', equal_branches=equal,
        moment_relative_errors=errors, gate=1e-9, radial_intervals=3,
        completed=True, production_qualification=False), indent=2)+'\n')


def test_new_pair_rule_config_public_guards_and_zero_spatial(monkeypatch, tmp_path):
    from witwin.maxwell.mom import dyadic_contrast
    cfg = config()
    assert DyadicIntegrationConfig(**cfg.payload()) == cfg
    assert 'spatial_rule' not in DyadicIntegrationConfig().payload()
    with pytest.raises(ValueError): DyadicIntegrationConfig(radial_map=cfg.radial_map)
    assert mw.MoMConfig(dyadic_integration=cfg).dyadic_integration == cfg
    with pytest.raises(NotImplementedError, match='free-space'):
        mw.MoMConfig(dyadic_integration=cfg, periodic=mw.MoMPeriodicConfig(((1., 0., 0.),)))
    def forbidden(*args, **kwargs):
        raise AssertionError('New strict pair guards and exact-zero spatial path must precede modes')
    for name in ('compile_plane_contrast', 'compile_electrostatic_plane_contrast',
                 '_compile_factored_plane_remainder_contrast'):
        monkeypatch.setattr(dyadic_contrast, name, forbidden)
    sample = medium()
    from witwin.maxwell.mom.periodic import LatticeSpec
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel, QuasiperiodicDyadicRegularKernel
    lattice = LatticeSpec(sample.epsilon_r.real.new_tensor([[1., 0., 0.], [0., 1., 0.]]),
                          sample.epsilon_r.real.new_zeros(3))
    periodic_kernel = QuasiperiodicDyadicGreenKernel(lattice, sample)
    with pytest.raises(NotImplementedError, match='periodic primary composition'):
        QuasiperiodicDyadicRegularKernel(periodic_kernel, shells=1, height=.02, max_halvings=3,
            absolute_tolerance=1e-5, max_shells=2, integration=cfg)
    lossy = TensorMedium(sample.frequency, sample.epsilon_r+1e-20j*torch.eye(3, device='cuda', dtype=torch.complex128), sample.mu_r)
    with pytest.raises(ValueError, match='exact real symmetric'):
        DyadicContrastKernel(lossy, sample, cfg)
    zero = DyadicContrastKernel(sample, sample, cfg)
    value, receipt = zero.evaluate_regular(torch.tensor(POINTS, device='cuda', dtype=torch.float64))
    assert receipt['converged'] and torch.count_nonzero(value) == 0
    (tmp_path/'paired-spatial-zero.json').write_text(json.dumps(dict(
        version='new_exact_zero_paired_spatial_rule_v1', zero_without_plane_compilation=True,
        new_config_identity=cfg.payload(), receipt=receipt, completed=True,
        production_qualification=False), indent=2)+'\n')

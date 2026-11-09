"""NEW spatial reference and compiled-grid/cache identity controls."""
import json
import math
import pytest
import torch
from mp_matched_spd_spatial_reference import mp_matched_spd_spatial_reference
from witwin.maxwell.mom import dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_new_factored_spatial_regular_reference_and_cached_compilation(tmp_path, monkeypatch):
    diagonal = [2.4, 1.7, 1.2]
    tensor = torch.diag(torch.tensor(diagonal, device='cuda', dtype=torch.complex128))
    medium = TensorMedium(3e7, tensor, tensor)
    points = tensor.real.new_tensor([[.043, -.027, .054], [.047, .024, .058]])
    expected = tensor.new_tensor([mp_matched_spd_spatial_reference(p, diagonal, medium.frequency,
                                regular_part=True) for p in points.tolist()])
    config = dg.DyadicIntegrationConfig(radial_order=8, azimuthal_order=8,
        max_radial_order=256, max_azimuthal_order=128, relative_tolerance=2e-4,
        radial_map='matched_spd_branch_v1', spatial_rule='factored_spd_tail_v1')
    cache = dg.SpectralModeCache(medium)
    record = dict(version='new_factored_spatial_regular_mp100_v2', MP_digits=100,
                  points=points.tolist(), diagonal=diagonal, blocks=[],
                  completed=False, production_qualification=False,
                  old_plane_or_spatial_PASS_not_replayed=True)
    def save():
        def safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return dict(nonfinite_value=repr(value))
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(v) for v in value]
            return value
        (tmp_path/'spatial-reference.json').write_text(json.dumps(safe(record), indent=2, allow_nan=False)+'\n')
    save()
    try:
        value, receipt = dg.evaluate_spatial_green(points, medium, config, regular_part=True, mode_cache=cache)
    except (ValueError, RuntimeError) as error:
        record.update(failure=str(error), receipt=getattr(error, 'receipt', None))
        save()
        raise
    blocks = []
    for query in range(len(points)):
        for name, i, j in [('EE', 0, 0), ('EH', 0, 3), ('HE', 3, 0), ('HH', 3, 3)]:
            wanted = expected[query, i:i+3, j:j+3]
            norm = float(wanted.norm())
            error = float((value[query, i:i+3, j:j+3]-wanted).norm()/wanted.norm())
            blocks.append(dict(query=query, block=name, reference_norm=norm,
                               relative_error=error if math.isfinite(error) else None,
                               gate=.005, passed=math.isfinite(norm) and norm > 0
                               and math.isfinite(error) and error < .005))
    record.update(blocks=blocks, receipt=receipt, spatial_evaluation_completed=True)
    save()
    torch.save(dict(native=value, expected=expected, points=points), tmp_path/'spatial-native-reference.pt')
    assert receipt['converged'] and receipt['kernel_version'] == 'dyadic_weyl_factored_spd_tail_v1'
    assert len(blocks) == 8 and all(b['passed'] for b in blocks)
    entries = [entry for entry, _, _ in cache._entries.values()]
    assert entries and any(len(entry[0].tail_indices) for entry in entries)
    unique = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
              for entry in entries for t in cache._tensors(entry)}
    assert cache.retained_bytes == sum(unique.values()) <= cache.max_bytes
    selected = receipt['evaluated_orders'][-1]
    def forbidden(*args, **kwargs):
        raise AssertionError('A cache hit must not compile modal or sign matrices')
    monkeypatch.setattr(dg, '_compile_factored_evanescent_plane', forbidden)
    monkeypatch.setattr(dg, 'compile_plane_modes', forbidden)
    cached = dg.integrate_spatial_green(points, medium, radial_order=selected[0],
        azimuthal_order=selected[1], restore_static=False, radial_map=config.radial_map,
        spatial_rule=config.spatial_rule, mode_cache=cache)
    torch.testing.assert_close(cached, value, rtol=0, atol=0)
    assert cache.hits >= 1
    key, (entry, _, _) = next(iter(cache._entries.items()))
    assert key[-1] == config.spatial_rule
    tail = entry[0].tail
    version = tail.state['a']._version
    tail.state['a'].requires_grad_(True)
    assert tail.state['a']._version == version
    with pytest.raises(ValueError, match='changed'):
        cache.get(key, forbidden)
    with pytest.raises(ValueError, match='changed'):
        tail._assert_current()
    tail.state['a'].requires_grad_(False)
    tail._assert_current()
    cache.get(key, lambda: entry)
    with pytest.raises(ValueError, match='query_batch'):
        tail.integrate(points, torch.ones(len(tail.transverse), device='cuda', dtype=torch.float64), query_batch=0)
    entry[0].tail.state['a'].add_(1.)
    with pytest.raises(ValueError, match='changed'):
        cache.get(key, forbidden)
    record.update(completed=True, cache_unique_owned_bytes=sum(unique.values()),
                  cache_reuse_without_compilation=True, gradient_flag_mutation_rejected=True,
                  inplace_matrix_mutation_rejected=True, invalid_query_batch_rejected=True)
    save()


def test_new_rule_payload_and_unsupported_domain_refusal_before_modes(monkeypatch):
    config = dg.DyadicIntegrationConfig(spatial_rule='factored_spd_tail_v1')
    payload = config.payload()
    assert payload['spatial_rule'] == config.spatial_rule and dg.DyadicIntegrationConfig(**payload) == config
    assert 'spatial_rule' not in dg.DyadicIntegrationConfig().payload()
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsupported material must refuse before modal construction')
    monkeypatch.setattr(dg, 'compile_plane_modes', forbidden)
    identity = torch.eye(3, device='cuda', dtype=torch.complex128)
    loss = TensorMedium(3e7, (2+1e-20j)*identity, identity)
    with pytest.raises(ValueError, match='exact real symmetric'):
        dg.DyadicGreenKernel(loss, config)
    real = TensorMedium(3e7, 2*identity, identity)
    from witwin.maxwell.mom.dyadic_contrast import DyadicContrastKernel
    with pytest.raises(NotImplementedError, match='paired regional contrasts'):
        DyadicContrastKernel(real, real, config)
    points = identity.real.new_tensor([[.01, .02, .03]])
    with pytest.raises(ValueError, match='regional contrasts'):
        dg.evaluate_spatial_green(points, real, config, _exterior=real)
    from witwin.maxwell.mom.runtime import MoMConfig
    from witwin.maxwell.mom.periodic_config import MoMPeriodicConfig
    periodic = MoMPeriodicConfig(vectors_m=((1., 0., 0.), (0., 1., 0.)))
    with pytest.raises(NotImplementedError, match='free-space execution'):
        MoMConfig(periodic=periodic, dyadic_integration=config)
    from witwin.maxwell.mom.periodic import LatticeSpec
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicGreenKernel, QuasiperiodicDyadicRegularKernel
    lattice = LatticeSpec(identity.real.new_tensor([[1., 0., 0.], [0., 1., 0.]]), identity.real.new_zeros(3))
    periodic_kernel = QuasiperiodicDyadicGreenKernel(lattice, real)
    with pytest.raises(NotImplementedError, match='periodic primary composition'):
        QuasiperiodicDyadicRegularKernel(periodic_kernel, shells=1, height=.02, max_halvings=3,
            absolute_tolerance=1e-5, max_shells=2, integration=config)


def test_affected_native_mode_cache_gradient_flag_mutation_refuses():
    identity = torch.eye(3, device='cuda', dtype=torch.complex128)
    medium = TensorMedium(3e7, 2*identity, identity)
    modes = dg.PlaneModes(identity.real.new_zeros((2, 2)), identity.new_ones((2, 2))*1j,
                          identity.new_zeros((2, 2, 6, 6)), medium)
    entry = modes, None, identity.real.new_ones(2)
    cache = dg.SpectralModeCache(medium)
    cache.get('new_gradient_mode_control', lambda: entry)
    version = modes.residues._version
    modes.residues.requires_grad_(True)
    assert modes.residues._version == version
    with pytest.raises(ValueError, match='changed'):
        cache.get('new_gradient_mode_control', lambda: pytest.fail('Must reject before callback'))
    assert cache.retained_bytes == 0

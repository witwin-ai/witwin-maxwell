"""Bounded spectral reuse, exact storage accounting and stale-source rejection."""
import pytest
import torch
from witwin.maxwell.mom import dyadic_green as dg
from witwin.maxwell.mom.tensor_medium import TensorMedium

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def medium():
    identity = torch.eye(3, device='cuda', dtype=torch.complex128)
    return TensorMedium(3e8, (2+.8j)*identity, (1.3+.5j)*identity)


def entry(m):
    transverse = torch.zeros(2, 2, device='cuda', dtype=torch.float64)
    longitudinal = torch.ones(2, 2, device='cuda', dtype=torch.complex128)*1j
    residues = torch.zeros(2, 2, 6, 6, device='cuda', dtype=torch.complex128)
    return dg.PlaneModes(transverse, longitudinal, residues, m), None, torch.ones(2, device='cuda', dtype=torch.float64)


def test_cache_exact_unique_storage_lru_and_mutation():
    m = medium()
    first, second = entry(m), entry(m)
    def bytes_in(*entries):
        tensors = [t for e in entries for t in dg.SpectralModeCache._tensors(e)]
        return sum({(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}.values())
    cache = dg.SpectralModeCache(m, bytes_in(first, second))
    cache.get('first', lambda: first)
    cache.get('second', lambda: second)
    assert cache.retained_bytes == bytes_in(first, second)
    assert cache.get('first', lambda: pytest.fail('must reuse')) is first
    third = entry(m)
    cache.get('third', lambda: third)
    assert list(cache._entries) == ['first', 'third'] and cache.evictions == 1
    assert cache.retained_bytes == bytes_in(first, third) <= cache.max_bytes
    first[2].add_(1.)
    with pytest.raises(ValueError, match='changed'): cache.get('first', lambda: first)
    assert cache.retained_bytes == bytes_in(third)
    m.mu_r.add_(1.)
    with pytest.raises(ValueError, match='changed'): cache.get('third', lambda: third)
    assert cache.retained_bytes == 0


def test_oversized_and_autograd_entries_are_not_retained():
    m = medium()
    cache = dg.SpectralModeCache(m, 0)
    cache.get('x', lambda: entry(m))
    assert cache.bypasses == 1 and cache.retained_bytes == 0
    differentiable = entry(m)
    differentiable[0].residues.requires_grad_()
    cache = dg.SpectralModeCache(m)
    cache.get('x', lambda: differentiable)
    assert cache.bypasses == 1 and not cache._entries
    with pytest.raises(ValueError, match='different'): cache.validate(medium())


@pytest.mark.parametrize('paired', [False, True])
def test_actual_spatial_cache_reuses_grid_across_query_batches(monkeypatch, paired):
    m = medium()
    options = {}
    if paired:
        from witwin.maxwell.mom.dyadic_contrast import SpectralContrastCache
        eye = torch.eye(3, device='cuda', dtype=torch.complex128)
        exterior = TensorMedium(m.frequency, eye, eye)
        options['_exterior'] = exterior
        cache = SpectralContrastCache(exterior, m)
    else:
        cache = dg.SpectralModeCache(m)
    points = torch.tensor([[.01, -.02, .20], [.02, .01, .22]], device='cuda', dtype=torch.float64)
    together = dg.integrate_spatial_green(points, m, radial_order=4, azimuthal_order=4, mode_cache=cache, **options)
    assert cache.misses == 1
    assert cache.stats()['modal_matrices_built'] == (64 if paired else 16)
    assert 0 <= cache.stats()['native_eigen_fallbacks'] <= (64 if paired else 16)
    def forbidden(*args, **kwargs): raise AssertionError('must reuse spectral modes across calls')
    monkeypatch.setattr(dg, 'compile_plane_modes', forbidden)
    monkeypatch.setattr(TensorMedium, '__post_init__', forbidden)
    separate = torch.cat([dg.integrate_spatial_green(p[None], m, radial_order=4, azimuthal_order=4,
                                                     mode_cache=cache, **options) for p in points])
    torch.testing.assert_close(separate, together, rtol=1e-12, atol=1e-12)
    assert cache.hits == 2
    assert cache.stats()['modal_matrices_built'] == (64 if paired else 16)


def test_concurrent_miss_builds_once_and_keeps_accounting_consistent():
    from concurrent.futures import ThreadPoolExecutor
    import time
    m = medium()
    value = entry(m)
    cache = dg.SpectralModeCache(m)
    calls = []
    def build():
        calls.append(1)
        time.sleep(.02)
        return value
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: cache.get('shared', build), range(2)))
    assert len(calls) == 1 and results[0] is results[1]
    assert cache.hits == 1 and cache.misses == 1 and cache.retained_bytes <= cache.max_bytes

"""Bounded reuse of outgoing periodic modal grids, without retained graphs."""
from dataclasses import replace
import runpy
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(dimension, **kwargs):
    namespace = runpy.run_path('tests/mom/test_periodic_dyadic_nonreciprocal.py')
    return replace(namespace['problem'](dimension, **kwargs), mode_batch=64)


def evaluate(kernel, points, **kwargs):
    if kernel.lattice.dimension == 2:
        return kernel.evaluate_open(points, shells=kwargs.get('shells', 1))
    return kernel.evaluate_line(points, normal=kwargs.get('normal', points.new_tensor([0., 0., 1.])),
        shells=1, transverse_cutoff=16., quadrature_order=12,
        transverse_map=kwargs.get('transverse_map', 'sinh'))


@pytest.mark.parametrize('dimension', (1, 2))
def test_periodic_modes_reused_across_heights_with_exact_values(dimension):
    kernel = problem(dimension)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .4], [.08, -.03, -.35]])
    evaluate(kernel, points)
    first = kernel.mode_cache.stats()
    assert first['entries'] == 2 and first['modal_matrices_built'] > 0
    shifted = points.clone()
    shifted[:, 2] *= .8
    actual, receipt = evaluate(kernel, shifted)
    stats = receipt['spectral_cache']
    assert stats['hits'] == 2 and stats['modal_matrices_built'] == first['modal_matrices_built']
    expected, _ = evaluate(replace(kernel, cache_max_bytes=0), shifted)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert 0 < stats['retained_cuda_bytes'] <= kernel.cache_max_bytes
    assert not receipt['physical_qualification']
    kwargs = {'normal': points.new_tensor([0., 1., 0.])} if dimension == 1 else {'shells': 2}
    changed, _ = evaluate(kernel, points, **kwargs)
    expected, _ = evaluate(replace(kernel, cache_max_bytes=0), points, **kwargs)
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)
    assert kernel.mode_cache.stats()['misses'] > first['misses']
    if dimension == 1:
        changed, _ = evaluate(kernel, points, transverse_map='linear')
        expected, _ = evaluate(replace(kernel, cache_max_bytes=0), points, transverse_map='linear')
        torch.testing.assert_close(changed, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dimension', (1, 2))
def test_periodic_mode_cache_never_reuses_material_autograd_graph(dimension):
    bias = torch.tensor(.2, device='cuda', dtype=torch.float64, requires_grad=True)
    kernel = problem(dimension, bias=bias)
    points = bias.new_tensor([[.12, .07, .4]])
    # Backward through newly built modes, retaining only the common material graph.
    derivatives = []
    for _ in range(2):
        value, _ = evaluate(kernel, points)
        derivatives.append(torch.autograd.grad(value.real.sum(), bias, retain_graph=True)[0])
    torch.testing.assert_close(*derivatives, rtol=0, atol=0)
    stats = kernel.mode_cache.stats()
    assert stats['entries'] == 0 and stats['retained_cuda_bytes'] == 0
    assert stats['hits'] == 0 and stats['bypasses'] == 2
    # Construct a fresh no-cache kernel for the independent centered difference.
    def objective(b):
        return evaluate(replace(problem(dimension, bias=b), cache_max_bytes=0), points)[0].real.sum()
    step = 1e-5
    expected = (objective(bias.detach()+step)-objective(bias.detach()-step))/(2*step)
    torch.testing.assert_close(derivatives[0], expected, rtol=2e-5, atol=2e-5)


def test_periodic_cache_budget_and_owned_state_guards():
    kernel = problem(2)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .4]])
    tiny = replace(kernel, cache_max_bytes=1)
    evaluate(tiny, points)
    assert tiny.mode_cache.stats()['entries'] == 0 and tiny.mode_cache.stats()['bypasses'] == 1
    evaluate(kernel, points)
    entry = next(iter(kernel.mode_cache._entries.values()))[0]
    entry[0].residues.add_(1.)
    with pytest.raises(ValueError, match='Cached spectral tensors changed'):
        evaluate(kernel, points)
    assert kernel.mode_cache.stats()['retained_cuda_bytes'] == 0
    fresh = replace(kernel)
    evaluate(fresh, points)
    fresh.lattice.bloch.add_(.1)
    with pytest.raises(RuntimeError, match='mutated'):
        evaluate(fresh, points)
    with pytest.raises(ValueError, match='max_bytes'):
        replace(problem(2), cache_max_bytes=-1)


def test_trainable_normal_bypasses_same_value_detached_cache():
    kernel = problem(1)
    points = kernel.lattice.vectors.new_tensor([[.12, .07, .4]])
    evaluate(kernel, points)
    assert kernel.mode_cache.stats()['entries'] == 1
    angle = points.new_tensor(0., requires_grad=True)
    normal = torch.stack((0*angle, torch.sin(angle), torch.cos(angle)))
    value, _ = evaluate(kernel, points, normal=normal)
    derivative, = torch.autograd.grad(value.real.sum(), angle)
    uncached_angle = points.new_tensor(0., requires_grad=True)
    uncached_normal = torch.stack((0*uncached_angle, torch.sin(uncached_angle), torch.cos(uncached_angle)))
    expected, _ = evaluate(replace(kernel, cache_max_bytes=0), points, normal=uncached_normal)
    expected_derivative, = torch.autograd.grad(expected.real.sum(), uncached_angle)
    torch.testing.assert_close(derivative, expected_derivative, rtol=0, atol=0)
    stats = kernel.mode_cache.stats()
    assert stats['hits'] == 0 and stats['bypasses'] == 1


@pytest.mark.parametrize('dimension', (1, 2))
def test_no_grad_warmup_does_not_capture_trainable_material(dimension):
    bias = torch.tensor(.2, device='cuda', dtype=torch.float64, requires_grad=True)
    kernel = problem(dimension, bias=bias)
    points = bias.new_tensor([[.12, .07, .4]])
    with torch.no_grad():
        evaluate(kernel, points)
    assert kernel.mode_cache.stats()['entries'] == 0
    value, _ = evaluate(kernel, points)
    derivative, = torch.autograd.grad(value.real.sum(), bias, retain_graph=True)
    expected, _ = evaluate(replace(kernel, cache_max_bytes=0), points)
    target, = torch.autograd.grad(expected.real.sum(), bias)
    torch.testing.assert_close(derivative, target, rtol=0, atol=0)
    assert kernel.mode_cache.stats()['hits'] == 0

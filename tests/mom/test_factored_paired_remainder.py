"""New paired evanescent remainders with two independently small differences."""
import json
import math
import mpmath as mp
import pytest
import torch
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_contrast import (
    _compile_factored_plane_remainder_contrast, SpectralContrastCache)
from mp_plane_remainder_reference import mp_plane_remainder_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
EPS = [[2.2, .125, .08], [.125, 1.8, -.04], [.08, -.04, 1.3]]
MU = [[1.6, -.07, .09], [-.07, 1.1, .06], [.09, .06, 1.4]]
FINITE_EPS = [[3., .2, -.13], [.2, 2.4, .1], [-.13, .1, 1.7]]
FINITE_MU = [[1.35, .04, -.08], [.04, 1.55, .03], [-.08, .03, 1.9]]
DE = [[.7, .3, -.2], [.3, -.4, .1], [-.2, .1, .2]]
DM = [[.2, -.25, .3], [-.25, .6, -.15], [.3, -.15, -.35]]
CASES = [('finite_low', (1700., 2900.), None),
         ('finite_high', (-6e7, 2e7), None),
         ('near40', (3e7, -8e7), -40),
         ('near50', (-7e7, -4e7), -50)]


def medium(eps=EPS, mu=MU, frequency=3e7):
    return TensorMedium(frequency,
        torch.tensor(eps, device='cuda', dtype=torch.complex128),
        torch.tensor(mu, device='cuda', dtype=torch.complex128))


def inside_data(exponent):
    if exponent is None:
        return FINITE_EPS, FINITE_MU
    h = math.ldexp(1., exponent)
    return tuple([[base[i][j]+h*delta[i][j] for j in range(3)] for i in range(3)]
                 for base, delta in ((EPS, DE), (MU, DM)))


@pytest.mark.parametrize('case,q,exponent', CASES, ids=[row[0] for row in CASES])
def test_new_paired_plane_MP100_individual_blocks(case, q, exponent, tmp_path):
    eps_i, mu_i = inside_data(exponent)
    distance = 1/(2*math.pi*3e7*math.sqrt(EPSILON_0*MU_0)*math.hypot(*q))
    raw = dict(version='new_paired_factored_plane_MP100_v1', case=case,
        MP_digits=100, transverse=q, distance_m=distance, epsilon_e=EPS, mu_e=MU,
        epsilon_i=eps_i, mu_i=mu_i, blocks=[], completed=False,
        old_single_medium_or_spatial_PASS_not_replayed=True, production_qualification=False)
    path = tmp_path/'paired-remainder.json'
    path.write_text(json.dumps(raw, indent=2)+'\n')
    with mp.workdps(100):
        e = mp_plane_remainder_reference(q, EPS, MU, 3e7, distance, return_mp=True)
        i = mp_plane_remainder_reference(q, eps_i, mu_i, 3e7, distance, return_mp=True)
        expected_mp = i['remainder']-e['remainder']
        expected = torch.tensor([[complex(expected_mp[a, b]) for b in range(6)]
                                 for a in range(6)], dtype=torch.complex128)
    artifact = tmp_path/'paired-native-reference.pt'
    # Save reference before compilation so an actual native failure can reuse
    # the exact MP100 result without repeating its calculation.
    saved = dict(expected=expected, transverse=q, distance_m=distance,
                 epsilon_e=EPS, mu_e=MU, epsilon_i=eps_i, mu_i=mu_i)
    torch.save(saved, artifact)
    exterior, interior = medium(), medium(eps_i, mu_i)
    transverse = torch.tensor([q], device='cuda', dtype=torch.float64)
    try:
        grid = _compile_factored_plane_remainder_contrast(transverse, exterior, interior)
        actual = grid.spectrum(distance)[0]
    except (ValueError, RuntimeError) as error:
        raw['native_failure'] = str(error)
        path.write_text(json.dumps(raw, indent=2)+'\n')
        raise
    reference = expected.to(device='cuda')
    for name, row, column in (('EE', 0, 0), ('EH', 0, 3), ('HE', 3, 0), ('HH', 3, 3)):
        target = reference[row:row+3, column:column+3]
        norm = float(target.norm())
        error = float((actual[row:row+3, column:column+3]-target).norm()/target.norm())
        finite = math.isfinite(norm) and norm > 0 and math.isfinite(error)
        raw['blocks'].append(dict(block=name, reference_norm=norm if math.isfinite(norm) else repr(norm),
            relative_error=error if math.isfinite(error) else repr(error), gate=.005,
            passed=finite and error < .005))
    raw.update(completed=True, sign_iterations=grid.sign_iterations)
    saved['native'] = actual.detach()
    torch.save(saved, artifact)
    path.write_text(json.dumps(raw, indent=2)+'\n')
    assert all(block['passed'] for block in raw['blocks']), raw['blocks']


def test_new_exact_zero_pair_bounded_batches_cache_and_mutation(monkeypatch, tmp_path):
    from witwin.maxwell.mom import dyadic_contrast as contrast
    sample = medium()
    nodes = torch.arange(129, device='cuda', dtype=torch.float64)
    transverse = torch.stack((2500.+3*nodes, -3800.-7*nodes), -1)
    grid = _compile_factored_plane_remainder_contrast(transverse, sample, sample)
    cache = SpectralContrastCache(sample, sample, 16*1024**2)
    weights = torch.ones(129, device='cuda', dtype=torch.float64)
    entry = (grid, None, weights)
    key = ('new_paired_plane_cache', 129)
    cache.get(key, lambda: entry)
    tensors = cache._tensors(entry)
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    assert cache.retained_bytes == sum(storages.values()) and 0 < cache.retained_bytes <= cache.max_bytes
    shapes = []
    original_exp = torch.linalg.matrix_exp
    def bounded_exp(value):
        shapes.append(tuple(value.shape))
        assert value.shape[-2:] == (16, 16) and len(value) <= 64
        return original_exp(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('The new paired cache control must not compile again')
    monkeypatch.setattr(torch.linalg, 'matrix_exp', bounded_exp)
    monkeypatch.setattr(contrast, '_compile_factored_plane_remainder_contrast', forbidden)
    assert cache.get(key, forbidden) is entry
    spectrum = grid.spectrum(1e-4)
    assert torch.count_nonzero(spectrum) == 0 and [s[0] for s in shapes] == [64, 64, 1]
    points = torch.tensor([[.01, .02, 1e-4], [-.02, .03, 2e-4]], device='cuda', dtype=torch.float64)
    assert torch.count_nonzero(grid.integrate(points, weights)) == 0
    before = grid.state['projector']._version
    grid.state['projector'].requires_grad_(True)
    assert grid.state['projector']._version == before
    with pytest.raises(ValueError): cache.get(key, forbidden)
    assert cache.retained_bytes == 0
    with pytest.raises(ValueError): grid.spectrum(1e-4)
    grid.state['projector'].requires_grad_(False)
    cache.get(key, lambda: entry)
    # Version-only mutation is a metadata negative: numerical values stay exact.
    grid.state['generator'].add_(0)
    with pytest.raises(ValueError): cache.get(key, forbidden)
    with pytest.raises(ValueError): grid.spectrum(1e-4)
    assert cache.retained_bytes == 0
    (tmp_path/'paired-zero-cache.json').write_text(json.dumps(dict(
        version='new_exact_zero_paired_plane_control_v1', nodes=129,
        exact_zero_spectrum=True, exact_zero_integral=True, exponential_batch_shapes=shapes,
        retained_unique_owned_bytes=sum(storages.values()), retained_budget_bytes=16*1024**2,
        cache_hit_without_compile=True, gradient_mode_and_version_mutation_refused=True,
        completed=True, production_qualification=False), indent=2)+'\n')


@pytest.mark.parametrize('kind', ['frequency', 'exterior_bound'])
def test_new_pair_specific_refusal_before_polynomials(kind, monkeypatch):
    from witwin.maxwell.mom import dyadic_green
    exterior = medium(frequency=3.1e7) if kind == 'frequency' else medium(
        eps=[[1000., 0., 0.], [0., 1000., 0.], [0., 0., 1000.]],
        mu=[[1000., 0., 0.], [0., 1000., 0.], [0., 0., 1000.]])
    interior = medium()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pair-specific refusal must precede polynomial compilation')
    monkeypatch.setattr(dyadic_green, '_factored_plane_polynomial_components', forbidden)
    with pytest.raises(ValueError, match='common frequency|both media beyond'):
        _compile_factored_plane_remainder_contrast(
            torch.tensor([[100., 30.]], device='cuda', dtype=torch.float64), exterior, interior)

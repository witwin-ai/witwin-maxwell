"""New noncommuting real-SPD controls and strict physical domain refusals."""
import json
import math
import pytest
import torch
from mp_plane_remainder_reference import mp_plane_remainder_reference
from witwin.maxwell.constants import EPSILON_0, MU_0
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import _factored_evanescent_plane_spectrum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_new_noncommuting_spd_regular_planes_mp100(tmp_path):
    eps = torch.tensor([[3.1, .3, .2], [.3, 2.2, -.1], [.2, -.1, 1.6]],
                       device='cuda', dtype=torch.complex128)
    mu = eps.new_tensor([[1.7, -.15, .12], [-.15, 1.2, .2], [.12, .2, 1.45]])
    assert not torch.equal(eps@mu, mu@eps)
    medium = TensorMedium(3e7, eps, mu)
    before = (medium.epsilon_r.clone(), medium.mu_r.clone())
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    receipt = dict(version='new_noncommuting_spd_remainder_mp100_v1', MP_digits=100,
                   only_new_material_and_wavevectors=True, old_PASS_not_replayed=True,
                   production_qualification=False, points=[])
    samples = []
    def save():
        (tmp_path/'general-spd-remainder.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    save()
    for q in [(2500., 4300.), (8e7, -3e7)]:
        distance = 1/(k0*math.hypot(*q))
        reference = mp_plane_remainder_reference(q, eps.tolist(), mu.tolist(), medium.frequency, distance)
        expected = eps.new_tensor(reference['remainder'])
        value, algorithm = _factored_evanescent_plane_spectrum(eps.real.new_tensor([q]), distance, medium)
        blocks = []
        for name, i, j in [('EE', 0, 0), ('EH', 0, 3), ('HE', 3, 0), ('HH', 3, 3)]:
            wanted, actual = expected[i:i+3, j:j+3], value[0, i:i+3, j:j+3]
            reference_norm = float(wanted.norm())
            error = float((actual-wanted).norm()/wanted.norm())
            blocks.append(dict(block=name, relative_error=error if math.isfinite(error) else None,
                               gate=.005, passed=math.isfinite(reference_norm) and reference_norm > 0
                               and math.isfinite(error) and error < .005))
        item = dict(transverse=list(q), distance_m=distance, algorithm=algorithm, blocks=blocks,
                    passed=all(b['passed'] for b in blocks))
        receipt['points'].append(item)
        samples.append(dict(transverse=list(q), distance_m=distance, expected=expected, native=value[0]))
        save()
        print(item, flush=True)
    torch.save(samples, tmp_path/'general-spd-native-reference.pt')
    receipt['completed'] = True
    save()
    assert torch.equal(medium.epsilon_r, before[0]) and torch.equal(medium.mu_r, before[1])
    assert len(receipt['points']) == 2 and all(p['passed'] for p in receipt['points'])


@pytest.mark.parametrize('case', ['physical_loss', 'roundoff_asymmetry', 'indefinite', 'material_gradient'])
def test_new_physical_domain_refusals_preserve_actual_tensors(case):
    eps = torch.diag(torch.tensor([3., 2., 1.5], device='cuda', dtype=torch.complex128))
    mu = torch.eye(3, device='cuda', dtype=torch.complex128)
    if case == 'physical_loss':
        eps = eps+1e-20j*mu
    elif case == 'roundoff_asymmetry':
        eps[0, 1] = 1e-16
    elif case == 'indefinite':
        eps[0, 0] = -3.
    elif case == 'material_gradient':
        eps.requires_grad_()
    medium = TensorMedium(3e7, eps, mu)
    before = (medium.epsilon_r.detach().clone(), medium.mu_r.detach().clone())
    with pytest.raises(ValueError, match='exact real symmetric|strictly positive definite'):
        _factored_evanescent_plane_spectrum(eps.real.new_tensor([[1e6, 1e6]]), 1e-6, medium)
    assert torch.equal(medium.epsilon_r, before[0]) and torch.equal(medium.mu_r, before[1])

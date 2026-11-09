"""Exact fixed tensor modes restored through JSON and rebound to an exterior."""
import copy
import json
import pytest
import torch
from witwin.maxwell.mom.tensor_carrier import TensorIncidentCarrier
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import compile_plane_modes
from witwin.maxwell.mom.periodic import LatticeSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def carrier(reciprocal=True, dtype=torch.complex128):
    eye = torch.eye(3, device='cuda', dtype=dtype)
    epsilon = eye.new_tensor([[2., .2, .1], [.2, 3., .15], [.1, .15, 4.]])+.2j*eye
    mu = eye.new_tensor([[1.1, .1, 0.], [.1, 1.3, .05], [0., .05, 1.6]])+.1j*eye
    if not reciprocal:
        mu = mu+eye.new_tensor([[0., .08j, 0.], [-.08j, 0., .03j], [0., -.03j, 0.]])
    medium = TensorMedium(3e8, epsilon, mu, reciprocal=reciprocal)
    modes = compile_plane_modes(eye.real.new_tensor([[.2, .1]]), medium)
    fields, wavevectors = [], []
    for index in range(2):
        residue = modes.residues[0, index]
        field = residue[:, residue.norm(dim=0).argmax()]
        fields.append(field/field[:3].norm())
        wavevectors.append(torch.cat((modes.transverse[0].to(dtype), modes.longitudinal[0, index:index+1])))
    fields = torch.stack(fields)
    return TensorIncidentCarrier(medium, torch.stack(wavevectors), fields[:, :3], fields[:, 3:],
                                 eye.real.new_tensor([.07, -.03, .01]))


@pytest.mark.parametrize('reciprocal', [True, False])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_json_modes_restore_exact_fields_bloch_identity_and_medium_binding(reciprocal, dtype):
    original = carrier(reciprocal, dtype)
    payload = json.loads(json.dumps(original.payload(), allow_nan=False))
    restored = TensorIncidentCarrier.from_payload(payload, device='cuda')
    rebound = TensorIncidentCarrier.from_payload(payload, device='cuda', medium=original.medium)
    assert restored.medium is not original.medium
    assert rebound.medium is original.medium
    bloch = torch.cat((original.wavevectors[0, :2].real, original.origin.new_zeros(1)))
    lattice = LatticeSpec(original.origin.new_tensor([[1., 0., 0.], [0., 1., 0.]]), bloch)
    points = original.origin.new_tensor([[.1, -.2, .3], [0., .2, -.1]])
    for value in (restored, rebound):
        assert value.identity == original.identity
        assert value.payload() == payload
        value.validate_lattice(lattice)
        for actual, expected in zip(value.evaluate(points, value.medium),
                                    original.evaluate(points, original.medium)):
            assert torch.equal(actual, expected)
        assert not value.electric.requires_grad
    payload['electric_V_per_m'][0][0][0] += 1.
    assert restored.identity == original.identity
    restored.electric.mul_(2.)
    with pytest.raises(ValueError, match='changed'):
        restored.payload()


def test_restoration_rejects_forged_modes_and_wrong_exterior_or_sector():
    original = carrier(False)
    payload = original.payload()
    wrong = TensorMedium(original.medium.frequency, original.medium.epsilon_r*1.01,
                         original.medium.mu_r, reciprocal=False)
    with pytest.raises(ValueError, match='bound exterior'):
        TensorIncidentCarrier.from_payload(payload, device='cuda', medium=wrong)
    forged = copy.deepcopy(payload)
    forged['magnetic_A_per_m'][0][0][0] += .1
    with pytest.raises(ValueError, match='Maxwell curl'):
        TensorIncidentCarrier.from_payload(forged, device='cuda')
    forged = copy.deepcopy(payload)
    forged['epsilon_r'][0][0][1] = -1.
    with pytest.raises(ValueError, match='passive'):
        TensorIncidentCarrier.from_payload(forged, device='cuda')
    restored = TensorIncidentCarrier.from_payload(payload, device='cuda')
    lattice = LatticeSpec(original.origin.new_tensor([[1., 0., 0.]]), original.origin.new_zeros(3))
    with pytest.raises(ValueError, match='Bloch sector'):
        restored.validate_lattice(lattice)


def test_restoration_rejects_malformed_version_precision_shapes_and_nonfinite_data():
    payload = carrier().payload()
    for key, value in [('version', 'unknown'), ('phasor', 'exp(+i omega t)'),
                       ('dtype', 'torch.float64'), ('frequency_Hz', True),
                       ('reciprocal', 1), ('origin_m', [0., 0.]),
                       ('wavevectors_per_m', []), ('epsilon_r', [[[True, 0.]]*3]*3),
                       ('magnetic_A_per_m', [[[float('nan'), 0.]]*3]*2)]:
        malformed = dict(payload, **{key: value})
        with pytest.raises(ValueError):
            TensorIncidentCarrier.from_payload(malformed, device='cuda')
    for malformed in (dict(payload, extra=0), {k: v for k, v in payload.items() if k != 'mu_r'}):
        with pytest.raises(ValueError, match='fields'):
            TensorIncidentCarrier.from_payload(malformed, device='cuda')
    with pytest.raises(ValueError, match='CUDA'):
        TensorIncidentCarrier.from_payload(payload, device='cpu')

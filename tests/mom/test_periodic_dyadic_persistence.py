"""Exact periodic tensor settings rebound without trusting cached acceptance."""
import copy
import json
from dataclasses import replace
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.periodic_dyadic import (
    QuasiperiodicDyadicGreenKernel, QuasiperiodicDyadicRegularKernel)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def problem(dimension, dtype=torch.complex128, reciprocal=True):
    eye = torch.eye(3, dtype=dtype, device='cuda')
    epsilon = eye.new_tensor([[2., .2, .1], [.2, 3., .15], [.1, .15, 4.]])+.2j*eye
    mu = eye.new_tensor([[1.1, .1, 0.], [.1, 1.3, .05], [0., .05, 1.6]])+.1j*eye
    if not reciprocal:
        mu = mu+eye.new_tensor([[0., .08j, 0.], [-.08j, 0., .03j], [0., -.03j, 0.]])
    medium = TensorMedium(3e8, epsilon, mu, reciprocal=reciprocal)
    vectors = eye.real.new_tensor([[.7, 0., 0.], [.1, .8, 0.], [.05, .1, .9]])[:dimension]
    bloch = vectors.new_tensor([.2, .1 if dimension > 1 else 0., .05 if dimension == 3 else 0.])
    lattice = LatticeSpec(vectors, bloch)
    kernel = QuasiperiodicDyadicGreenKernel(lattice, medium, mode_batch=16, query_batch=7,
                                          cache_max_bytes=1024*1024)
    return QuasiperiodicDyadicRegularKernel(kernel, 8, .12, 3, .02, 16,
        normal=eye.real.new_tensor([0., 0., 1.]) if dimension == 1 else None,
        transverse_cutoff=128., quadrature_order=32, transverse_map='sinh')


@pytest.mark.parametrize('dimension', (1, 2, 3))
@pytest.mark.parametrize('dtype,reciprocal', [(torch.complex64, True), (torch.complex128, False)])
def test_json_exact_configuration_binding_and_owned_normal(dimension, dtype, reciprocal):
    original = problem(dimension, dtype, reciprocal)
    payload = json.loads(json.dumps(original.payload(), allow_nan=False))
    medium = TensorMedium(original.medium.frequency, original.medium.epsilon_r,
                          original.medium.mu_r, reciprocal=reciprocal)
    lattice = LatticeSpec(original.lattice.vectors, original.lattice.bloch)
    restored = QuasiperiodicDyadicRegularKernel.from_payload(payload, medium=medium, lattice=lattice)
    assert restored.medium is medium and restored.lattice is lattice
    assert restored.payload() == payload and restored.identity == original.identity
    assert restored.kernel.mode_cache is not original.kernel.mode_cache
    if dimension == 3:
        orders = torch.tensor([[0, 0, 0], [1, -1, 2]], device='cuda', dtype=torch.int64)
        expected = original.kernel.spectral_modes(orders)
        actual = restored.kernel.spectral_modes(orders)
        assert torch.equal(actual['current_to_field'], expected['current_to_field'])
        assert not actual['physical_qualification']
    payload['normal'][0] = .3
    assert restored.identity == original.identity
    restored.normal.add_(.1)
    with pytest.raises(ValueError, match='normal changed'):
        restored.payload()


def test_restore_rejects_wrong_binding_versions_budget_drift_and_nonfinite_data():
    original = problem(1)
    def restore(payload):
        return QuasiperiodicDyadicRegularKernel.from_payload(payload,
            medium=original.medium, lattice=original.lattice)
    base = original.payload()
    for path, value in [
        (('version',), 'unknown'), (('kernel_id',), 'unknown'),
        (('binding', 'frequency_Hz'), 1.), (('binding', 'reciprocal'), 0),
        (('binding', 'dtype'), 'torch.complex64'),
        (('spectral', 'mode_batch'), True), (('spectral', 'cache_max_bytes'), -1),
        (('regular', 'max_shells'), 8), (('regular', 'absolute_tolerance'), float('nan')),
        (('regular', 'transverse_map'), 'unknown'),
        (('integration', 'relative_tolerance'), .1),
        (('integration', 'absolute_tolerance'), .01),
        (('normal',), [True, 0., 1.]), (('normal',), [0., 0., 2.]),
        (('extra',), 'unrecognized'), (('spectral', 'extra'), 1),
    ]:
        payload = copy.deepcopy(base)
        target = payload
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            restore(payload)
    bad = copy.deepcopy(base)
    del bad['integration']['radial_order']
    with pytest.raises(ValueError, match='fields'):
        restore(bad)
    wrong = LatticeSpec(original.lattice.vectors, original.lattice.bloch*2)
    with pytest.raises(ValueError, match='bindings'):
        QuasiperiodicDyadicRegularKernel.from_payload(base, medium=original.medium, lattice=wrong)
    original.medium.mu_r.add_(.01)
    with pytest.raises(ValueError, match='Tensor medium changed'):
        restore(base)


def test_restored_actual_regular_values_and_failure_gates_match():
    from witwin.maxwell.mom.periodic_dyadic import PeriodicDyadicConvergenceError
    original = problem(2)
    eye = torch.eye(3, device='cuda', dtype=torch.complex128)
    original = replace(original, kernel=replace(original.kernel,
        medium=TensorMedium(3e8, (1+2j)*eye, (1+2j)*eye)))
    restored = QuasiperiodicDyadicRegularKernel.from_payload(original.payload(),
        medium=original.medium, lattice=original.lattice)
    points = original.lattice.vectors.new_tensor([[.12, .07, .4], [.08, -.03, -.35]])
    expected, expected_receipt = original.evaluate(points, exclude_primary=True)
    actual, actual_receipt = restored.evaluate(points, exclude_primary=True)
    assert torch.equal(actual, expected)
    assert actual_receipt == expected_receipt and actual_receipt['converged']
    assert not actual_receipt['physical_qualification']
    # A restored configuration must still execute admission, not reuse a pass.
    inadequate = replace(original, shells=1, max_shells=2, absolute_tolerance=1e-12)
    restored_bad = QuasiperiodicDyadicRegularKernel.from_payload(inadequate.payload(),
        medium=inadequate.medium, lattice=inadequate.lattice)
    with pytest.raises(PeriodicDyadicConvergenceError):
        restored_bad.evaluate(points, exclude_primary=True)

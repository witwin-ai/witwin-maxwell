"""Conditioned checks retain thresholds and reject actual amplitude/phase faults."""
import copy
import numpy as np
import pytest
import torch

from scripts.dev.fdtd.printed_line_loss_assertions import superposition_metrics, flux_metrics


@pytest.mark.parametrize('phase', [-1., 1j])
def test_source_roundoff_near_output_zero_uses_actual_nonzero_basis_scale(phase):
    a = torch.tensor([1 + 2j, 3 - 1j, -.5 + .7j], dtype=torch.complex128)
    combined = a + phase * a
    direct = combined + 1e-6 * a
    data = superposition_metrics(combined, direct, [a, a], [1., phase])
    assert data['passed'] and data['rtol'] == 1e-4
    assert data['l2_actual_basis_scale'] == pytest.approx(2 * float(a.norm()))
    assert data['l2_relative_to_actual_basis'] == pytest.approx(5e-7, rel=1e-8)
    if phase == -1:
        assert not data['original_pointwise_passed']
        assert data['mixed_output_condition_number_l2'] is None


@pytest.mark.parametrize('phase', [-1., 1j])
@pytest.mark.parametrize('fault', ['amplitude', 'phase'])
def test_conditioned_superposition_rejects_wrong_physical_source_amplitude_or_phase(phase, fault):
    a = torch.tensor([1 + 2j, 3 - 1j, -.5 + .7j], dtype=torch.complex128)
    combined = a + phase * a
    wrong = phase * (1.001 if fault == 'amplitude' else np.exp(.001j))
    direct = a + wrong * a
    data = superposition_metrics(combined, direct, [a, a], [1., phase])
    assert not data['passed']
    assert data['l2_relative_to_actual_basis'] > 1e-4


def test_zero_input_basis_keeps_original_absolute_floor_and_rejects_spurious_output():
    zero = torch.zeros(5, dtype=torch.complex128)
    assert superposition_metrics(zero, zero + .5e-11, [zero, zero], [1, -1])['passed']
    assert not superposition_metrics(zero, zero + 2e-11, [zero, zero], [1, -1])['passed']


def powers(negative=-1., positive=.99):
    faces = {'x_neg': negative, 'x_pos': positive, 'y_neg': 0., 'y_pos': 0., 'z_neg': 0., 'z_pos': 0.}
    return {'frequencies_hz': [1e9], 'native_signed_face_flux_w': [faces], 'net_outward_flux_w': [sum(faces.values())]}


def test_small_net_cancellation_is_reported_with_unchanged_entering_power_threshold():
    direct = powers()
    combined = powers(positive=.990002)
    metric = flux_metrics(combined, direct)
    assert metric['passed'] and metric['rtol'] == 3e-5
    assert not metric['original_net_relative_passed']
    assert metric['frequency_records'][0]['net_error_relative_to_entering'] == pytest.approx(2e-6)


def test_compensating_signed_face_fault_cannot_hide_in_a_correct_net_sum():
    direct = powers()
    combined = powers(negative=-.999, positive=.989)
    metric = flux_metrics(combined, direct)
    assert abs(metric['frequency_records'][0]['combined_net_power_w'] - metric['frequency_records'][0]['direct_net_power_w']) < 1e-15
    assert not metric['passed']
    assert metric['frequency_records'][0]['face_l1_error_relative_to_entering'] == pytest.approx(.002)


@pytest.mark.parametrize('fault', ['amplitude', 'phase'])
def test_power_criterion_rejects_substantive_wrong_source_amplitude_or_phase(fault):
    direct = powers()
    # Two equal physical phasors with reference relative phase pi/2 have unit
    # power in these units. No constitutive/runtime power formula is copied.
    factor = 1.001 ** 2 if fault == 'amplitude' else abs(1 + np.exp(1j * (np.pi / 2 + .001))) ** 2 / 2
    combined = powers(negative=-factor, positive=.99 * factor)
    assert not flux_metrics(combined, direct)['passed']


@pytest.mark.parametrize('fault', ['frequency', 'faces', 'nonfinite', 'declared_net'])
def test_flux_metadata_requires_complete_finite_actual_signed_faces(fault):
    direct = powers(); combined = copy.deepcopy(direct)
    if fault == 'frequency': combined['frequencies_hz'] = []
    elif fault == 'faces': combined['native_signed_face_flux_w'][0].pop('z_pos')
    elif fault == 'nonfinite': combined['native_signed_face_flux_w'][0]['z_pos'] = float('nan')
    elif fault == 'declared_net': combined['net_outward_flux_w'][0] += .001
    with pytest.raises(ValueError): flux_metrics(combined, direct)

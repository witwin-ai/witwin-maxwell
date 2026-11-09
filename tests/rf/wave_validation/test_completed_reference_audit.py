"""Complete frequency/station metadata and unfitted parity diagnostics."""
import copy

import numpy as np
import pytest

from scripts.dev.fdtd.printed_line_completed_reference_audit import compare_modes, symmetry_diagnostic


def reference():
    row = {'positions_m': [-.015, .015], 'forward_characteristic_impedance_ohm': {'real': 50., 'imag': 1.},
        'beta_rad_per_m': 30., 'alpha_np_per_m': -.001}
    return {'frequencies_hz': [.8e9, 1e9], 'state_transfer': [[copy.deepcopy(row)], [copy.deepcopy(row)]]}


def test_complete_physical_comparison_keeps_complex_Z_signed_alpha_and_declared_denominator():
    coarse, fine = reference(), reference()
    for row in fine['state_transfer']:
        row[0]['forward_characteristic_impedance_ohm'] = {'real': 51., 'imag': 2.}
        row[0]['beta_rad_per_m'] = 30.3
    result = compare_modes(coarse, fine, scope='test')
    assert result['max_complex_Z_relative_difference'] == pytest.approx(abs(1 + 1j) / abs(50 + 1j))
    assert result['max_beta_relative_difference'] == pytest.approx(.01)
    assert result['frequency_records'][0]['station_pairs'][0]['modes']['single']['fine_reference_signed_alpha_np_per_m'] == -.001
    alternative = compare_modes(coarse, fine, scope='test', relative_to_fine=True)
    assert alternative['max_complex_Z_relative_difference'] == pytest.approx(abs(1 + 1j) / abs(51 + 2j))


@pytest.mark.parametrize('change', ['both_truncated', 'one_truncated', 'duplicate_frequency', 'reordered', 'duplicate_pair', 'empty_pair', 'nonfinite'])
def test_physical_reducer_rejects_incomplete_or_invalid_retained_modes(change):
    coarse, fine = reference(), reference()
    if change == 'both_truncated':
        coarse['state_transfer'].pop(); fine['state_transfer'].pop()
    elif change == 'one_truncated': fine['state_transfer'].pop()
    elif change == 'duplicate_frequency':
        coarse['frequencies_hz'] = fine['frequencies_hz'] = [1e9, 1e9]
    elif change == 'reordered': coarse['frequencies_hz'].reverse()
    elif change == 'duplicate_pair': fine['state_transfer'][0] *= 2
    elif change == 'empty_pair': fine['state_transfer'][0] = []
    elif change == 'nonfinite': fine['state_transfer'][0][0]['beta_rad_per_m'] = float('nan')
    with pytest.raises(ValueError): compare_modes(coarse, fine, scope='test')


@pytest.mark.parametrize('kind,component,axis', [('stripline', 'Hy', 2), ('cpw', 'Hz', 1)])
def test_native_ground_parity_reports_actual_even_contamination_without_fitted_phase(kind, component, axis):
    plus, minus = [0., 0., 0.], [0., 0., 0.]
    plus[axis], minus[axis] = .003, -.003
    records = {'plus': {'component': component, 'physical_component_position_m': plus},
        'minus': {'component': component, 'physical_component_position_m': minus}}
    spectrum = np.array([1 + 2j, 3 - 1j])
    clean = symmetry_diagnostic(records, {'plus': spectrum, 'minus': -spectrum}, kind, [1e9, 1.2e9])
    assert clean['even_to_target_odd_magnetic_l2'].tolist() == [0., 0.]
    # A known 25% even component added to the odd component is preserved.
    contaminated = symmetry_diagnostic(records, {'plus': 1.25 * spectrum, 'minus': -.75 * spectrum}, kind, [1e9, 1.2e9])
    np.testing.assert_allclose(contaminated['even_to_target_odd_magnetic_l2'], [.25, .25], rtol=1e-15)

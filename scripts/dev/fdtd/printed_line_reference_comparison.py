"""Compare independent physical line transfer without modal impedance or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(native, reference):
    """Require complete data and fixed geometry before comparing physical modes."""
    for data, guards in ((native, ('source_unchanged', 'fixture_unchanged', 'driver_unchanged',
                                   'helper_unchanged', 'state_helper_unchanged')),
                         (reference, ('source_unchanged', 'fixture_unchanged', 'driver_unchanged',
                                      'helper_unchanged', 'state_helper_unchanged', 'reference_unchanged'))):
        if not all(data[key] for key in guards):
            raise ValueError('A numerical or reference-build identity guard failed.')
    nc, rc = native['configuration'], reference['configuration']
    if nc['kind'] != reference['kind'] or nc['metal'] != 'pec':
        raise ValueError('This independent reference control requires the same PEC line kind.')
    for key in ('dx', 'longitudinal_dl', 'padding', 'separation', 'axial_half_length',
                'transverse_refinement_radius', 'duration', 'sigma'):
        if nc[key] != rc[key]:
            raise ValueError('Independent physical configuration differs: '+key)
    if native['fixture_sha256'] != reference['fixture_sha256'] or native['native_contour_helper_sha256'] != reference['contour_driver_sha256'] or native['state_helper_sha256'] != reference['state_helper_sha256']:
        raise ValueError('Geometry or native physical integration helper identity differs.')
    frequencies = native['frequencies_hz']
    if not frequencies or frequencies != reference['frequencies_hz'] or len(set(frequencies)) != len(frequencies):
        raise ValueError('Declared nonempty unique frequency lists differ.')
    if [r['frequency_hz'] for r in native['frequency_records']] != frequencies or len(reference['state_transfer']) != len(frequencies):
        raise ValueError('Retained frequency records are incomplete or out of order.')
    expected_drives = 4 if nc['kind'] == 'differential_pair' else 2
    if len(native['runs']) != expected_drives or len(reference['runs']) != expected_drives:
        raise ValueError('Physical independent drive basis is incomplete.')
    pairs = [tuple(p) for p in zip(reference['positions_m'][:-1], reference['positions_m'][1:])]
    native_pairs = [tuple(p) for p in zip(nc['positions'][:-1], nc['positions'][1:])]
    if not pairs or not set(pairs).issubset(native_pairs):
        raise ValueError('The declared native/reference physical station pairs do not overlap.')
    def require_shape(values, shape, label):
        if not shape:
            if not isinstance(values, dict) or set(values) != {'real', 'imag'}:
                raise ValueError('Malformed retained complex sample: '+label)
            return
        if not isinstance(values, list) or len(values) != shape[0]:
            raise ValueError('Retained physical state is incomplete: '+label)
        for value in values:
            require_shape(value, shape[1:], label)

    state_count = 4 if nc['kind'] == 'differential_pair' else 2
    for row in native['frequency_records']:
        require_shape(row['actual_physical_state'], (len(nc['positions']), state_count, expected_drives), 'native V/I')
    shape = (len(frequencies), len(reference['positions_m']))
    if nc['kind'] == 'differential_pair':
        shape += (2,)
    shape += (expected_drives,)
    for key in ('voltage_V', 'current_A'):
        require_shape(reference[key], shape, key)
    if not native['native_probe_spectra']:
        raise ValueError('No retained native field spectra.')
    for probe in native['native_probe_spectra'].values():
        require_shape(probe['all_drive_frequency_phasors'], (expected_drives, len(frequencies)), 'native probe')
    output = []
    for index, frequency in enumerate(frequencies):
        nr = native['frequency_records'][index]['state_transfer']
        rr = reference['state_transfer'][index]
        if [tuple(row['positions_m']) for row in nr] != native_pairs or [tuple(row['positions_m']) for row in rr] != pairs:
            raise ValueError('Physical station records are incomplete or reordered.')
        native_by_pair = {tuple(row['positions_m']): row for row in nr}
        records = []
        for right in rr:
            left = native_by_pair[tuple(right['positions_m'])]
            modes = ('common', 'differential') if nc['kind'] == 'differential_pair' else ('single',)
            mode_records = {}
            for mode in modes:
                a, b = (left, right) if mode == 'single' else (left['modes'][mode], right['modes'][mode])
                key = 'forward_characteristic_impedance_ohm' if mode == 'single' else 'characteristic_impedance_ohm'
                z = [complex(row[key]['real'], row[key]['imag']) for row in (a, b)]
                mode_records[mode] = {'native_impedance_ohm': a[key], 'reference_impedance_ohm': b[key],
                    'complex_impedance_relative_difference': abs(z[0]-z[1])/abs(z[1]),
                    'native_beta_rad_per_m': a['beta_rad_per_m'], 'reference_beta_rad_per_m': b['beta_rad_per_m'],
                    'beta_relative_difference': abs(a['beta_rad_per_m']/b['beta_rad_per_m']-1),
                    'native_signed_alpha_np_per_m': a['alpha_np_per_m'], 'reference_signed_alpha_np_per_m': b['alpha_np_per_m']}
            records.append({'positions_m': right['positions_m'], 'modes': mode_records,
                'native_condition_number_50ohm': left['state_condition_number_50ohm'],
                'reference_condition_number_50ohm': right['state_condition_number_50ohm']})
        output.append({'frequency_hz': frequency, 'station_pairs': records})
    metrics = [mode for record in output for pair in record['station_pairs'] for mode in pair['modes'].values()]
    return {'frequency_records': output,
        'max_complex_impedance_relative_difference': max(m['complex_impedance_relative_difference'] for m in metrics),
        'max_beta_relative_difference': max(m['beta_relative_difference'] for m in metrics),
        'scope': 'Physical native V/I transfer eigenvectors, no Network.z0 or modal beta and no fitted amplitude/phase. Different complete real feed bases are allowed outside the fixed interior ROI. Signed near-zero PEC attenuation is retained without a relative-loss qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous independent comparisons: '+str(args.output))
    driver_before = digest(Path(__file__))
    paths = (args.native, args.reference)
    before = {str(path): digest(path) for path in paths}
    native, reference = [json.loads(path.read_text()) for path in paths]
    record = compare(native, reference)
    record.update(schema='independent-physical-line-transfer-comparison/1', input_sha256=before,
        inputs_unchanged=before == {str(path): digest(path) for path in paths},
        native_field_source_sha256=native['source']['source_sha256'],
        reference_geometry_package_sha256=reference['maxwell_source_sha256'],
        driver_sha256=driver_before, driver_unchanged=driver_before == digest(Path(__file__)))
    if not record['inputs_unchanged'] or not record['driver_unchanged']:
        raise RuntimeError('Retained independent field inputs changed during reduction.')
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print({key: value for key, value in record.items() if key.startswith('max_')})


if __name__ == '__main__':
    main()

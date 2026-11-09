"""Compare fixed-geometry local outer-grid controls using retained native fields."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--uniform', type=Path, required=True)
parser.add_argument('--graded', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Preserve previous retained-field comparisons: '+str(args.output))
paths = [args.uniform, args.graded]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

hashes = [sha(path) for path in paths]
a, b = [json.loads(path.read_text()) for path in paths]
for key in ['source', 'fixture_sha256', 'native_contour_helper_sha256', 'state_helper_sha256',
            'driver_sha256', 'frequencies_hz', 'source_geometry', 'coherent_drive_groups']:
    if a[key] != b[key]:
        raise ValueError('Input physical/source contract differs: '+key)
allowed = {'transverse_refinement_radius', 'output'}
if set(a['configuration']) != set(b['configuration']) or any(
        a['configuration'][k] != b['configuration'][k] for k in a['configuration'] if k not in allowed):
    raise ValueError('Physical configuration changed beyond the explicit outer-grid option.')
if a['configuration']['transverse_refinement_radius'] is not None or b['configuration']['transverse_refinement_radius'] != .02:
    raise ValueError('This control compares uniform spacing against the declared 20 mm local refinement radius.')
frequencies = a['frequencies_hz']
if not frequencies or len(set(frequencies)) != len(frequencies):
    raise ValueError('Declared frequencies must be nonempty and unique.')
positions = a['configuration']['positions']
expected_pairs = [list(pair) for pair in zip(positions[:-1], positions[1:])]
if not expected_pairs:
    raise ValueError('At least two physical stations are required.')
if set(a['native_probe_spectra']) != set(b['native_probe_spectra']) or not a['native_probe_spectra']:
    raise ValueError('Native probe keysets differ or are empty.')
drive_count = len(a['coherent_drive_groups'])
for data in (a, b):
    if not all(data[key] for key in ['source_unchanged', 'fixture_unchanged', 'driver_unchanged',
                                    'helper_unchanged', 'state_helper_unchanged']):
        raise ValueError('A retained numerical identity guard failed.')
    if len(data['runs']) != drive_count or [r['frequency_hz'] for r in data['frequency_records']] != frequencies:
        raise ValueError('Retained drive or ordered frequency records are incomplete.')
    for record in data['frequency_records']:
        if [pair['positions_m'] for pair in record['state_transfer']] != expected_pairs:
            raise ValueError('Native station-pair records do not cover the declared positions in order.')
    for name, probe in data['native_probe_spectra'].items():
        spectra = probe['all_drive_frequency_phasors']
        if len(spectra) != drive_count or any(len(drive) != len(frequencies) for drive in spectra):
            raise ValueError('Native probe drive/frequency spectra are incomplete: '+name)
        other = a['native_probe_spectra'][name]
        if probe['component'] != other['component']:
            raise ValueError('Native field component changed: '+name)
        np.testing.assert_allclose(probe['position_m'], other['position_m'], rtol=0, atol=1e-16)

def complex_value(value):
    return complex(value['real'], value['imag'])

metrics = []
for index, (ar, br) in enumerate(zip(a['frequency_records'], b['frequency_records'])):
    records = []
    for left, right in zip(ar['state_transfer'], br['state_transfer']):
        z0, z1 = [complex_value(item['forward_characteristic_impedance_ohm']) for item in (left, right)]
        records.append({'positions_m': left['positions_m'],
            'complex_impedance_relative_change': abs(z1-z0)/abs(z0),
            'beta_relative_change': abs(right['beta_rad_per_m']/left['beta_rad_per_m']-1)})
    fields = {}
    for component in sorted({r['component'] for r in a['native_probe_spectra'].values()}):
        samples = []
        for name, left in a['native_probe_spectra'].items():
            if left['component'] != component:
                continue
            right = b['native_probe_spectra'][name]
            samples.append(([complex_value(drive[index]) for drive in left['all_drive_frequency_phasors']],
                            [complex_value(drive[index]) for drive in right['all_drive_frequency_phasors']]))
        x = np.array([s[0] for s in samples]); y = np.array([s[1] for s in samples])
        fields[component] = float(np.linalg.norm(y-x)/np.linalg.norm(x))
    metrics.append({'frequency_hz': ar['frequency_hz'], 'station_pair_comparisons': records,
                    'unfitted_native_component_relative_l2': fields})
record = {'schema': 'printed-line-outer-grid-budget/1', 'input_sha256': dict(zip(map(str, paths), hashes)),
    'inputs_unchanged': hashes == [sha(path) for path in paths], 'field_source_sha256': a['source']['source_sha256'],
    'fixed_contract': 'Same physical domain, conductors, source apertures, source pulse, duration, stations and axial spacing. Only the transverse grid beyond the explicit 20 mm local refinement radius changes. No fitted amplitude, phase or modal impedance.',
    'frequency_records': metrics,
    'max_complex_impedance_relative_change': max(q['complex_impedance_relative_change'] for r in metrics for q in r['station_pair_comparisons']),
    'max_beta_relative_change': max(q['beta_relative_change'] for r in metrics for q in r['station_pair_comparisons']),
    'max_unfitted_native_component_relative_l2': max(q for r in metrics for q in r['unfitted_native_component_relative_l2'].values()),
    'scope': 'Outer-grid coarsening control only; continuum, physical loss and modal S gates remain separate.'}
if not record['inputs_unchanged']:
    raise RuntimeError('Retained-field inputs changed during comparison.')
args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
print({key: value for key, value in record.items() if key.startswith('max_')})

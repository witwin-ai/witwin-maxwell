"""Save explicit untuned antenna candidates from the retained physical input families."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.dev.fdtd.pcb_application_campaign import FIXTURES, NAMES


def candidate(definition, *, finite_copper=False):
    """Declare a complete connected radiator; qualification is a subsequent task."""
    data = copy.deepcopy(definition)
    name = data['id']
    pcb = any(c['id'] == 'positive-pad' for c in data['conductors'])
    selected = []
    for record in data['conductors']:
        if record['role'] != 'design':
            continue
        bounds = np.asarray(record['bounds_m']).reshape(3, 2)
        center = bounds.mean(1)
        if pcb:
            keep = ((center[0] < .030 and .0006 < center[1] < .0024)
                    or (center[0] < .0084 and center[1] < 0))
        else:
            # A top-loaded front-face monopole keeps its explicit 0.5 mm feed
            # gap. The open-top family uses its front stem alone.
            keep = (record['id'].startswith('front:') and abs(center[0]) < .00051
                    and center[2] >= .0005)
            keep |= record['id'].startswith('top:') and abs(center[0]) < .00051
        if keep:
            selected.append(record['id'])
    data['initial_conductor_ids'] = selected
    data['candidate_status'] = 'complete untuned fixed candidate; no accuracy or matching qualification'
    data['candidate_provenance'] = {'original_id': name,
        'original_sha256': hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
        'design': 'printed monopole and partial counterpoise' if pcb else 'front monopole with optional top loading',
        'selected_original_design_ids': selected}
    if finite_copper:
        if not pcb:
            raise ValueError('Finite-copper conversion is defined only for the two planar PCB families.')
        original = data['conductors'][0]['bounds_m']
        bottom, top = original[4:6]
        thickness = top-bottom
        # This transformation is saved as a NEW input. It removes the physical
        # copper interval and translates the exterior and feed to its support;
        # substrate thickness and permittivity are preserved exactly.
        coordinates = np.asarray(data['mesh']['axes_m']['z'])
        transformed = np.where(coordinates > top, coordinates-thickness,
                               np.where(coordinates >= bottom, bottom, coordinates))
        data['mesh']['axes_m']['z'] = np.unique(transformed).tolist()
        model = {'kind': 'copper_foil', 'thickness_m': thickness,
                 'conductivity_s_per_m': 5.8e7, 'frequency_range_hz': [1e9, 4e9]}
        data['conductors'] = [
            {'id': 'positive-pad', 'kind': 'sheet', 'role': 'fixed',
             'bounds_m': [0., .030, .0006, .0024, bottom, bottom], 'constitutive_model': dict(model)},
            {'id': 'negative-pad', 'kind': 'sheet', 'role': 'fixed',
             'bounds_m': [0., .0084, -.0048, 0., bottom, bottom], 'constitutive_model': dict(model)}]
        for port in data['ports']:
            port['sheet_bounds_m'][4:6] = [bottom, bottom]
            for point in port['integration_line_m']:
                point[2] = bottom
        data['initial_conductor_ids'] = []
        data['candidate_provenance']['surface_mapping'] = {
            'kind': 'explicit input transformation', 'old_copper_bounds_m': [bottom, top],
            'support_coordinate_m': bottom, 'thickness_parameter_m': thickness,
            'z_mapping': 'z - clamp(z - copper_bottom, 0, physical_copper_thickness)',
            'substrate_thickness_preserved': True,
            'scope': 'surface-model candidate, distinct from original volumetric PEC input'}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--finite-copper', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        if args.finite_copper and not name.startswith('pcb-'):
            continue
        definition = json.loads((FIXTURES/f'metahz-scene-{name}.json').read_text())
        data = candidate(definition, finite_copper=args.finite_copper)
        (args.output/f'metahz-scene-{name}.json').write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()

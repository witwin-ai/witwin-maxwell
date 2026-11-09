"""Prepare explicit fixed-geometry PCB mesh controls with fixed RF references."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from witwin.maxwell.adapters.pcb import import_pcb_scene


def refined_definition(definition, *, axes, factor, normal_samples=512):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1 or factor % 2 != 1:
        raise ValueError('Refinement requires a positive odd integer to preserve native magnetic half planes.')
    if not axes or set(axes)-set('xyz') or len(set(axes)) != len(axes):
        raise ValueError('Refinement axes must be distinct members of xyz.')
    if isinstance(normal_samples, bool) or not isinstance(normal_samples, int) or normal_samples < 1:
        raise ValueError('Normal material sample count must be a positive integer.')
    result = copy.deepcopy(definition)
    original = import_pcb_scene(definition, frequency=2.45e9, device='cpu')
    for record, port in zip(result['ports'], original.ports):
        if 'current_contour_bounds_m' in record:
            continue
        surface = port.current_surface
        bounds = torch.stack((surface.position-surface.size/2, surface.position+surface.size/2), -1)
        record['current_contour_bounds_m'] = bounds.reshape(-1).tolist()
    active = set(original.metadata['pcb_import']['active_conductor_ids'])
    records = definition['dielectrics']+[r for r in definition['conductors'] if r['id'] in active]
    boxes = np.asarray([r['bounds_m'] for r in records], dtype=np.float64).reshape(-1, 3, 2)
    bounds = np.stack((boxes[..., 0].min(0), boxes[..., 1].max(0)), -1)
    for axis in axes:
        dim = 'xyz'.index(axis)
        nodes = definition['mesh']['axes_m'][axis]
        refined = [nodes[0]]
        for low, high in zip(nodes, nodes[1:]):
            # Keep all original endpoints and the six exterior/PML intervals.
            # Splitting an intersecting interval also retains its old midpoint.
            count = factor if high > bounds[dim, 0] and low < bounds[dim, 1] else 1
            refined.extend(low+(high-low)*i/count for i in range(1, count))
            refined.append(high)
        result['mesh']['axes_m'][axis] = refined
    result['mesh']['subpixel_samples'] = {'samples': [1, 1, normal_samples], 'averaging': 'polarized'}
    result.setdefault('metadata', {})['fixed_geometry_refinement'] = {
        'axes': axes, 'odd_subdivision_factor': factor, 'normal_material_samples': normal_samples,
        'parent_definition_sha256': hashlib.sha256(json.dumps(definition,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        'physical_geometry_and_materials_unchanged': True,
        'reference_contract': 'Explicit original physical current contours; original nodes and half planes retained.',
        'qualification': 'Prepared input only; physical execution and convergence gates are separate.',
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axes', choices=('x', 'y', 'z', 'xy', 'xyz'), required=True)
    parser.add_argument('--factor', type=int, required=True)
    parser.add_argument('--normal-samples', type=int, default=512)
    args = parser.parse_args()
    data = args.input.read_bytes()
    result = refined_definition(json.loads(data), axes=args.axes, factor=args.factor,
                                normal_samples=args.normal_samples)
    result['metadata']['fixed_geometry_refinement']['baseline_input_sha256'] = hashlib.sha256(data).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'axis_nodes': {k: len(v) for k,v in result['mesh']['axes_m'].items()}}))


if __name__ == '__main__':
    main()

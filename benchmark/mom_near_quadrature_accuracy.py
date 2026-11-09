"""Near-disjoint panel-pair accuracy of the dense MoM assembly rules.

Every case is one observer/source triangle pair. The production rule
(``_adaptive_near_integrals``: product Gauss up to ``near_max_order``, then
the gap-aware fallback) and the plain order-16 product rule are compared with
a high-order reference: observer leaves bisected isotropically until each is
at least two diameters from every source edge and vertex (depth up to 28),
order-10 outer Gauss and order-20 edge-graded distance-scaled inner fans. The
production fallback refines anisotropically, so the reference is a distinct
refinement family. The reference is checked
against orders 8/16 on the same leaves; a profiled ratio-3, order 10/24
reference agreed with both to 1e-10 or better on the hardest cases. Errors use the
assembly norm: max of |dD|/|D| and |dK|/max(|K|, |k| |D|).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from witwin.maxwell.mom import operator as op
from witwin.maxwell.mom.operator import IntegrationConfig


SIZE = .05  # panel size in exterior wavelengths (lambda = 1 m)


def _tensor(points):
    return torch.tensor(points, device='cuda', dtype=torch.float64)


def parallel(gap):
    """Slab-like stacked panels at height gap*SIZE, laterally offset."""
    source = _tensor([(0, 0, 0), (SIZE, 0, 0), (0, SIZE, 0)])
    observer = source[[0, 2, 1]]+_tensor([.3*SIZE, .2*SIZE, gap*SIZE])
    return observer, source


def perpendicular(gap):
    """Box-edge-like orthogonal panels separated by gap*SIZE without a shared vertex."""
    source = _tensor([(0, 0, 0), (SIZE, 0, 0), (0, SIZE, 0)])
    observer = _tensor([(.2*SIZE, -gap*SIZE, 0), (SIZE, -gap*SIZE, 0), (.5*SIZE, -gap*SIZE, SIZE)])
    return observer, source


def coplanar_sliver(gap, aspect):
    """Neighbouring parallel strips of width SIZE/aspect, gap relative to the width."""
    width = SIZE/aspect
    source = _tensor([(0, 0, 0), (SIZE, 0, 0), (SIZE/2, width, 0)])
    base = width*(1+gap)
    observer = _tensor([(0, base, 0), (SIZE, base, 0), (SIZE/2, base+width, 0)])
    return observer, source


def hinged_sliver(gap, aspect, angle=math.radians(15)):
    """Strips meeting at a 15 degree crease, as on a sampled rounding arc."""
    width = SIZE/aspect
    source = _tensor([(0, 0, 0), (SIZE, 0, 0), (SIZE/2, width, 0)])
    start = _tensor([0, width*(1+gap), 0])
    direction = _tensor([0, math.cos(angle), math.sin(angle)])
    observer = torch.stack((start, start+_tensor([SIZE, 0, 0]), start+_tensor([SIZE/2, 0, 0])+width*direction))
    return observer, source


def cases():
    out = []
    for gap in (1., .3, .1, .05, .02, 1e-2, 1e-3):
        out.append(('parallel', gap, 1, parallel(gap)))
    for gap in (1., .1, 1e-2, 1e-3):
        out.append(('perpendicular', gap, 1, perpendicular(gap)))
    for aspect in (1, 10, 30, 100):
        for gap in (1., .1, 1e-2, 1e-3):
            out.append(('coplanar_sliver', gap, aspect, coplanar_sliver(gap, aspect)))
        for gap in (1., .1, 1e-2):
            out.append(('hinged_sliver', gap, aspect, hinged_sliver(gap, aspect)))
    return out


def pair_basis(observer, source):
    """Minimal RWG geometry view: two faces with unit signs and local opposite-edge lengths."""
    triangles = torch.stack((observer, source))
    lengths = torch.linalg.vector_norm(triangles.roll(-1, 1)-triangles.roll(1, 1), dim=-1)
    areas = torch.linalg.vector_norm(torch.linalg.cross(triangles[:, 1]-triangles[:, 0],
                                                        triangles[:, 2]-triangles[:, 0]), dim=-1)/2
    mesh = SimpleNamespace(triangles=triangles, areas=areas, vertices=triangles.reshape(-1, 3),
                           faces=torch.arange(6, device='cuda').reshape(2, 3), dtype=torch.float64)
    return SimpleNamespace(mesh=mesh, face_edges=torch.arange(6, device='cuda').reshape(2, 3),
                           face_signs=torch.ones((2, 3), device='cuda', dtype=torch.long),
                           edge_lengths=lengths.reshape(-1))


def error(value, reference, k):
    (d, magnetic), (rd, rk) = value, reference
    return float(op._near_relative_difference(d, magnetic, rd, rk, SimpleNamespace(k=k), torch.float64).max())


def measure(name, gap, aspect, geometry, k, config=IntegrationConfig()):
    basis = pair_basis(*geometry)
    medium = SimpleNamespace(k=k)
    rows, cols = torch.tensor([0], device='cuda'), torch.tensor([1], device='cuda')
    # Isotropic leaves: an independent refinement family from the anisotropic production rule.
    reference = op._fan_near_integrals(basis, rows, cols, medium, 10, 20, 2., isotropic=True)[:2]
    check = op._fan_near_integrals(basis, rows, cols, medium, 8, 16, 2., isotropic=True)[:2]
    plain = op._pair_integrals(basis, rows, cols, medium, 16, 16, False, 3)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        d, magnetic, stats = op._adaptive_near_integrals(basis, rows, cols, medium, config)
        torch.cuda.synchronize()
        production = {'error': error((d, magnetic), reference, k), 'rule': 'fallback' if 'fallback' in stats else 'product',
                      'seconds': time.perf_counter()-start, 'stats': stats}
    except RuntimeError as failure:
        production = {'error': None, 'rule': 'failed_closed', 'message': str(failure)}
    return {'geometry': name, 'separation_over_size': gap, 'aspect_ratio': aspect, 'k': k,
            'reference_self_difference': error(check, reference, k), 'product16_error': error(plain, reference, k),
            'production': production}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometries', nargs='+', default=['parallel', 'perpendicular', 'coplanar_sliver', 'hinged_sliver'])
    parser.add_argument('--wavenumbers', nargs='+', type=float, default=[2*math.pi, 2*math.pi*math.sqrt(6.)])
    args = parser.parse_args(argv)
    rows = json.loads(args.output.read_text()) if args.output.exists() else []
    for k in args.wavenumbers:
        for name, gap, aspect, geometry in cases():
            if name not in args.geometries or any(r['geometry'] == name and r['separation_over_size'] == gap
                                                  and r['aspect_ratio'] == aspect and r['k'] == k for r in rows):
                continue
            row = measure(name, gap, aspect, geometry, k)
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items()}, default=str), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

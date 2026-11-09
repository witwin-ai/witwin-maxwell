"""Touching panel-pair accuracy (self, shared edge, shared vertex) on slivers.

Each case is one pair taken from a strip of right-triangle slivers of aspect
H/w, as on sampled rounding arcs: the self pair, the pair across the long
shared edge, the pair across the diagonal, a shared-vertex pair, a pair
across a 15 degree crease and a box-corner pair across a right angle. Every
geometry is also rotated arbitrarily, so coplanar pairs carry roundoff
heights. The assembly path (joint Duffy rules below the aspect threshold,
the anisotropy-aware rule above it) and the fixed order-8 Duffy rules are
compared with an independent reference: isotropically bisected observer
leaves (depth 18) with order-8 outer Gauss and order-16 edge-graded
distance-scaled inner fans of the full kernel, checked against depth 16 with
orders 6/12. The production rule instead integrates the static kernel in
closed form and its remainder with joint Duffy rules. For aspect 1 the
order-24 Duffy rules give a second, unrelated reference.
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
from witwin.maxwell.mom.quadrature import ANISOTROPIC_TOUCHING_ASPECT, triangle_aspect


SIZE = .05
ROTATION = ((.36, .48, -.8), (-.8, .6, 0.), (.48, .64, .6))
PAIRS = {'self': (0, 0, 0), 'edge_long': (1, 2, 1), 'edge_diagonal': (0, 1, 1), 'vertex': (0, 2, 2),
         'edge_crease': (2, 4, 1), 'edge_corner': (0, 5, 1)}


def strip(aspect, rotated):
    """Faces: 0,1 one cell; 2,3 the next coplanar cell; 4 creased 15 deg; 5 folded 90 deg over edge (0,0)-(H,0)."""
    h, w = SIZE, SIZE/aspect
    c, s = math.cos(math.radians(15)), math.sin(math.radians(15))
    vertices = [(0, 0, 0), (h, 0, 0), (h, w, 0), (0, w, 0), (0, 2*w, 0), (h, 2*w, 0),
                (0, w+w*c, w*s), (0, 0, -w)]
    faces = [(0, 1, 2), (0, 2, 3), (3, 2, 5), (3, 5, 4), (2, 6, 3), (0, 7, 1)]
    v = torch.tensor(vertices, device='cuda', dtype=torch.float64)
    if rotated:
        v = v@torch.tensor(ROTATION, device='cuda', dtype=torch.float64).T+torch.tensor([.11, -.07, .05], device='cuda', dtype=torch.float64)
    f = torch.tensor(faces, device='cuda')
    triangles = v[f]
    lengths = torch.linalg.vector_norm(triangles.roll(-1, 1)-triangles.roll(1, 1), dim=-1)
    areas = torch.linalg.vector_norm(torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), dim=-1)/2
    mesh = SimpleNamespace(triangles=triangles, areas=areas, vertices=v, faces=f, dtype=torch.float64)
    n = len(faces)
    return SimpleNamespace(mesh=mesh, face_edges=torch.arange(3*n, device='cuda').reshape(n, 3),
                           face_signs=torch.ones((n, 3), device='cuda', dtype=torch.long), edge_lengths=lengths.reshape(-1))


def error(value, reference, k):
    return float(op._near_relative_difference(value[0], value[1], reference[0], reference[1], SimpleNamespace(k=k), torch.float64).max())


def measure(name, aspect, rotated, k, config=op.IntegrationConfig()):
    basis = strip(aspect, rotated)
    medium = SimpleNamespace(k=k)
    r, c, category = PAIRS[name]
    rows, cols = torch.tensor([r], device='cuda'), torch.tensor([c], device='cuda')
    planar = category == 0

    def fan(*args, **kw):
        d, m, info = op._fan_near_integrals(basis, rows, cols, medium, *args, isotropic=True, **kw)
        return (d, torch.zeros_like(m) if planar else m), info
    reference, info = fan(8, 16, 2., max_level=18)
    check, _ = fan(6, 12, 2., max_level=16)
    duffy8 = op._pair_integrals(basis, rows, cols, medium, 8, 8, True, category)
    row = {'pair': name, 'aspect': aspect, 'normalized_aspect': float(triangle_aspect(basis.mesh.triangles[[r, c]]).max()),
           'rotated': rotated, 'k': k, 'reference_leaves': info['leaves'],
           'reference_self_difference': error(check, reference, k), 'duffy8_error': error(duffy8, reference, k)}
    if aspect == 1:
        row['duffy24_vs_reference'] = error(op._pair_integrals(basis, rows, cols, medium, 24, 24, True, category), reference, k)
    torch.cuda.synchronize()
    start = time.perf_counter()
    if row['normalized_aspect'] > ANISOTROPIC_TOUCHING_ASPECT:
        d, m, stats = op._anisotropic_touching_integrals(basis, rows, cols, medium, config, category)
        row['production'] = {'rule': 'anisotropic', 'error': error((d, m), reference, k), 'level': stats['maximum_level'],
                             'leaves': stats['maximum_leaves_per_call']}
    else:
        value = op._pair_integrals(basis, rows, cols, medium, config.singular_order, config.singular_order, True, category)
        row['production'] = {'rule': 'duffy', 'error': error(value, reference, k)}
    torch.cuda.synchronize()
    row['production']['seconds'] = time.perf_counter()-start
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', nargs='+', default=list(PAIRS))
    parser.add_argument('--aspects', nargs='+', type=float, default=[1, 3, 10, 30, 100])
    parser.add_argument('--wavenumbers', nargs='+', type=float, default=[2*math.pi*math.sqrt(6.)])
    args = parser.parse_args(argv)
    rows = json.loads(args.output.read_text()) if args.output.exists() else []
    for k in args.wavenumbers:
        for name in args.pairs:
            for aspect in args.aspects:
                for rotated in ((False, True) if aspect == 30 else (False,)):
                    if any(r['pair'] == name and r['aspect'] == aspect and r['rotated'] == rotated and r['k'] == k for r in rows):
                        continue
                    row = measure(name, aspect, rotated, k)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Surface preparation time versus face count: pruned validation against all-pairs.

Builds icosahedral spheres (20 n^2 faces) and measures, CUDA-synchronised:
SurfaceMesh construction (host topology plus validation), its coincident-vertex
and self-intersection stages, RWG basis construction, and the two-body
separation check of SurfaceSystem. The all-pairs reference is the original
``mesh.py`` loaded verbatim from git revision ``--reference-revision`` and is run
only up to ``--reference-max-faces``; both paths must make the same decisions
and report the same gap. No operator is assembled and nothing is solved.

    python -m benchmark.mom_validation_scaling --frequencies 8 16 32 64 128 158 --output prep.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

from witwin.maxwell.mom import mesh as current


def _load_reference(revision):
    source = subprocess.check_output(['git', 'show', f'{revision}:witwin/maxwell/mom/mesh.py'], text=True)
    module = types.ModuleType('witwin.maxwell.mom._reference_mesh')
    module.__package__ = 'witwin.maxwell.mom'
    module.__file__ = f'<{revision}:mesh.py>'
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def _timed(function, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function(*args)
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


def measure(module, frequency, *, separation=True):
    sphere = current.sphere_mesh(radius=1., frequency=frequency, device='cuda')
    vertices, faces = sphere.vertices, sphere.faces
    record = {'faces': len(faces), 'vertices': len(vertices)}
    (host, _), record['topology_seconds'] = _timed(module._topology, faces, len(vertices))
    extent = torch.linalg.vector_norm(vertices.amax(0)-vertices.amin(0))
    tolerance = 128*torch.finfo(vertices.dtype).eps*extent
    if hasattr(module, '_has_coincident_vertices'):
        coincident, record['coincident_seconds'] = _timed(module._has_coincident_vertices, vertices, tolerance)
    else:
        def scan():
            ids = torch.arange(len(vertices), device=vertices.device)
            for start in range(0, len(vertices), 128):
                close = (vertices[start:start+128, None]-vertices[None]).abs().amax(-1) <= tolerance
                if bool((close & (ids[start:start+128, None] < ids[None])).any()):
                    return True
            return False
        coincident, record['coincident_seconds'] = _timed(scan)
    intersects, record['intersection_seconds'] = _timed(module._has_intersections, vertices[faces])
    surface, record['surface_mesh_seconds'] = _timed(module.SurfaceMesh, vertices, faces)
    _, record['rwg_basis_seconds'] = _timed(module.RWGBasis, surface)
    record.update(coincident=bool(coincident), intersects=bool(intersects))
    del host
    if separation:
        other = current.sphere_mesh(radius=1., frequency=frequency, center=(2.1, 0.3, 0.), device='cuda')
        left, right = vertices[faces], other.vertices[other.faces]
        if hasattr(module, '_minimum_triangle_distance'):
            gap, record['separation_seconds'] = _timed(module._minimum_triangle_distance, left, right)
        else:
            def all_pairs():
                minimum = torch.tensor(torch.inf, dtype=left.dtype, device=left.device)
                for start in range(0, len(left)*len(right), 4096):
                    ids = torch.arange(start, min(start+4096, len(left)*len(right)), device=left.device)
                    minimum = torch.minimum(minimum, module.triangle_pair_distances(
                        left[ids//len(right)], right[ids % len(right)]).min())
                return minimum
            gap, record['separation_seconds'] = _timed(all_pairs)
        record['gap'] = float(gap)
    # SurfaceMesh construction already contains topology and both validation stages.
    record['total_seconds'] = sum(record.get(k, 0.) for k in ('surface_mesh_seconds', 'rwg_basis_seconds',
                                                               'separation_seconds'))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frequencies', type=int, nargs='+', default=[8, 16, 32, 64, 128, 158])
    parser.add_argument('--reference-revision', default='c90c8346')
    parser.add_argument('--reference-max-faces', type=int, default=21000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    reference = _load_reference(args.reference_revision)
    rows = []
    for frequency in args.frequencies:
        row = {'frequency': frequency, 'pruned': measure(current, frequency)}
        if row['pruned']['faces'] <= args.reference_max_faces:
            row['all_pairs'] = measure(reference, frequency)
            for key in ('coincident', 'intersects', 'gap'):
                if row['pruned'][key] != row['all_pairs'][key]:
                    raise AssertionError(f'{key} differs at {frequency}: {row["pruned"][key]} vs {row["all_pairs"][key]}')
        rows.append(row)
        print(json.dumps({'faces': row['pruned']['faces'], 'pruned_s': round(row['pruned']['total_seconds'], 2),
                          'all_pairs_s': round(row['all_pairs']['total_seconds'], 2) if 'all_pairs' in row else None}),
              flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'schema': 'mom_validation_scaling_v1', 'rows': rows,
                                           'reference_revision': args.reference_revision,
                                           'device': torch.cuda.get_device_name(0)}, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

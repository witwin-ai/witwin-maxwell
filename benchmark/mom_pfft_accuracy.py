"""pFFT operator-action accuracy, memory and matvec time against original rows.

For one closed body (icosahedral sphere or structured cube) and one medium,
this samples R test rows, integrates those rows of the original D (single
layer) and K channels exactly with the near-field quadrature, and compares
``(A x)[rows]`` with the pFFT action for each grid spacing and stencil order.
The pFFT result is precorrected: grid entries of pairs closer than the
precorrection radius are replaced by the exact entries of the sampled rows.
Both precorrected and raw relative errors are reported.

    python -m benchmark.mom_pfft_accuracy --shape sphere --size 2 --frequency-mesh 18 \\
        --grid-ppw 8 10 15 --orders 2 3 4 --output pfft.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh, sphere_mesh
from witwin.maxwell.mom.nearfield import scalar_edge_pair_channels
from witwin.maxwell.mom.operator import IntegrationConfig, PreparedRWGGeometry
from witwin.maxwell.mom.pfft_grid import PFFTGridSpec
from witwin.maxwell.mom.pfft_operator import PFFTRegion
from witwin.maxwell.mom.physics import compile_medium

C0 = 299792458.0


def build_basis(args, wavelength):
    size = args.size*wavelength
    if args.shape == 'sphere':
        mesh = sphere_mesh(radius=size/2, frequency=args.frequency_mesh, device='cuda')
        return RWGBasis(SurfaceMesh(mesh.vertices, mesh.faces))
    from benchmark.mom_large_scale_l1 import _box_surface
    v, f = _box_surface((size, size, size), size/args.frequency_mesh, (0., 0., 0.))
    return RWGBasis(SurfaceMesh(torch.tensor(v, device='cuda'), torch.tensor(np.asarray(f), device='cuda')))


def exact_rows(basis, medium, rows, config, chunk):
    geometry = PreparedRWGGeometry(basis)
    n = basis.n_edges
    columns = torch.arange(n, device=rows.device)
    d = torch.empty((len(rows), n), device=rows.device, dtype=torch.complex128)
    k = torch.empty_like(d)
    for start in range(0, len(rows), chunk):
        r = rows[start:start+chunk]
        er = r[:, None].expand(-1, n).reshape(-1)
        ec = columns[None].expand(len(r), -1).reshape(-1)
        dv, kv = scalar_edge_pair_channels(basis, basis, er, ec, medium, config, geometry=(geometry, geometry))
        d[start:start+chunk], k[start:start+chunk] = dv.reshape(len(r), n), kv.reshape(len(r), n)
    return d, k


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def evaluate(region, basis, rows, exact_d, exact_k, vectors, repeats, *, support_aware=False):
    midpoints = (basis.mesh.vertices[basis.edges[:, 0]]+basis.mesh.vertices[basis.edges[:, 1]])/2
    distance = torch.cdist(midpoints[rows], midpoints)
    radius = region.config.near_radius
    if support_aware:
        # Same radius as the public backend: grid reach plus twice the largest RWG support.
        radius = radius+2*float(region.projection.support_radius.max())
    near = distance < radius
    pr, pc = near.nonzero(as_tuple=True)
    grid_d, grid_k = region.grid_entries(rows[pr], pc)
    out = {'near_pairs_per_row': float(near.sum(1).double().mean()),
           'near_radius_m': radius}
    for name, exact, grid, action in (('single_layer_D', exact_d, grid_d, region.apply_single_layer),
                                      ('double_layer_K', exact_k, grid_k, region.apply_double_layer)):
        errors, raw = [], []
        for x in vectors:
            reference = exact@x
            approximate = action(x).to(torch.complex128)[rows]
            correction = torch.zeros_like(reference).index_add_(0, pr, (exact[pr, pc]-grid)*x[pc])
            corrected = approximate+correction
            scale = torch.linalg.vector_norm(reference)
            errors.append(float(torch.linalg.vector_norm(corrected-reference)/scale))
            raw.append(float(torch.linalg.vector_norm(approximate-reference)/scale))
        times = []
        for _ in range(repeats):
            start = _sync()
            action(vectors[0])
            times.append(_sync()-start)
        far = ~near
        far_scale = float(torch.linalg.vector_norm(exact[far]))
        far_error = None
        if far_scale > 0:
            # Entry-level far-zone error from a bounded sample of far pairs.
            fr, fc = far.nonzero(as_tuple=True)
            pick = torch.randperm(len(fr), device=fr.device)[:20000]
            sample_d, sample_k = region.grid_entries(rows[fr[pick]], fc[pick])
            sample = sample_d if name.startswith('single') else sample_k
            reference = exact[fr[pick], fc[pick]]
            far_error = float(torch.linalg.vector_norm(sample-reference)/torch.linalg.vector_norm(reference))
        out[name] = {'relative_error_precorrected': max(errors), 'relative_error_raw': max(raw),
                     'far_entry_relative_error': far_error, 'matvec_seconds_median': sorted(times)[len(times)//2]}
    return out


def fft_timing(sizes, precisions, repeats):
    """Time one forward FFT, kernel product and inverse FFT on padded cubic grids."""
    rows = []
    for precision in precisions:
        dtype = getattr(torch, precision)
        for n in sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            kernel = torch.randn((n, n, n), device='cuda', dtype=dtype)
            grid = torch.randn((n, n, n), device='cuda', dtype=dtype)
            times = []
            for _ in range(repeats):
                start = _sync()
                torch.fft.ifftn(torch.fft.fftn(grid)*kernel)
                times.append(_sync()-start)
            rows.append({'precision': precision, 'padded_points': n**3, 'side': n,
                         'array_bytes': grid.numel()*grid.element_size(),
                         'convolution_seconds_median': sorted(times)[len(times)//2],
                         'peak_bytes': torch.cuda.max_memory_allocated()})
            print(json.dumps(rows[-1]), flush=True)
            del kernel, grid
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shape', choices=('sphere', 'cube'), default='sphere')
    parser.add_argument('--size', type=float, default=1.)
    parser.add_argument('--frequency-mesh', type=int, default=9)
    parser.add_argument('--frequency', type=float, default=C0)
    parser.add_argument('--eps', type=float, default=1., help='medium relative permittivity (1 = exterior vacuum)')
    parser.add_argument('--grid-ppw', type=float, nargs='+', default=[8., 10., 15.],
                        help='grid points per wavelength of the medium')
    parser.add_argument('--orders', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--precisions', nargs='+', default=['complex128'])
    parser.add_argument('--rows', type=int, default=512)
    parser.add_argument('--row-chunk', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--support-aware', action='store_true',
                        help='precorrect within (M+2) h plus twice the largest RWG support radius')
    parser.add_argument('--memory-fraction', type=float, default=0.)
    parser.add_argument('--fft-sizes', type=int, nargs='*', default=None,
                        help='only time padded-grid convolutions of these cubic side lengths')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    if args.fft_sizes:
        rows = fft_timing(args.fft_sizes, args.precisions, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'schema': 'mom_pfft_fft_timing_v1', 'rows': rows,
                                           'device': torch.cuda.get_device_name(0)}, indent=2)+'\n')
        return 0
    wavelength = C0/args.frequency
    basis = build_basis(args, wavelength)
    medium = compile_medium(mw.Material(eps_r=args.eps), args.frequency, background=args.eps == 1.)
    medium_wavelength = 2*math.pi/abs(complex(medium.k))
    integration = IntegrationConfig(order=4, singular_order=4)
    generator = torch.Generator(device='cuda').manual_seed(7)
    rows = torch.randperm(basis.n_edges, generator=generator, device='cuda')[:args.rows].sort().values
    start = _sync()
    exact_d, exact_k = exact_rows(basis, medium, rows, integration, args.row_chunk)
    exact_seconds = _sync()-start
    vectors = [torch.randn(basis.n_edges, generator=generator, device='cuda', dtype=torch.complex128)
               for _ in range(2)]
    lengths = basis.edge_lengths
    report = {'schema': 'mom_pfft_accuracy_v1', 'arguments': {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items() if k != 'output'},
              'edges': basis.n_edges, 'faces': len(basis.mesh.faces),
              'mean_edge_medium_wavelengths': float(lengths.mean())/medium_wavelength,
              'exact_row_seconds': exact_seconds, 'device': torch.cuda.get_device_name(0), 'cases': []}
    for precision in args.precisions:
        for ppw in args.grid_ppw:
            for order in args.orders:
                config = PFFTGridSpec(spacing=medium_wavelength/ppw, order=order, precision=precision)
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                region = PFFTRegion.from_basis(basis, medium, config)
                retained = torch.cuda.memory_allocated()-base
                record = {'precision': precision, 'grid_ppw': ppw, 'order': order, **region.receipt(),
                          'retained_bytes_measured': retained}
                record.update(evaluate(region, basis, rows, exact_d, exact_k, vectors, args.repeats,
                                       support_aware=args.support_aware))
                record['grid_to_mean_edge'] = (medium_wavelength/ppw)/float(lengths.mean())
                record['peak_bytes_above_base'] = torch.cuda.max_memory_allocated()-base
                report['cases'].append(record)
                print(json.dumps({k: record[k] for k in ('precision', 'grid_ppw', 'order', 'padded_shape')}
                                 | {c: round(record[c]['relative_error_precorrected'], 7)
                                    for c in ('single_layer_D', 'double_layer_K')}
                                 | {'h_g/h': round(record['grid_to_mean_edge'], 3),
                                    'near/row': round(record['near_pairs_per_row'], 1)}
                                 | {'matvec_D_s': round(record['single_layer_D']['matvec_seconds_median'], 4)}),
                      flush=True)
                del region
                torch.cuda.empty_cache()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

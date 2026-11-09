"""Sparse multi-body pFFT acceptance: single exterior grid versus per-cluster grids.

Dielectric spheres (icosahedral ``sphere_mesh``) are placed at the given
centres. The same scene is prepared with ``exterior_grids='single'`` and
``'clusters'`` (identical PFFTConfig otherwise) through Scene -> Simulation ->
Result. For each variant the driver records preparation time and peak memory,
retained operator bytes, matvec time and the relative error of A x against
original rows integrated exactly (``ScalarSystemBlocks.sampled_action``) on a
random row sample, plus the difference between the two variants on the full
vector. The clustered variant is solved with exact per-body LU right
preconditioning; its far field is compared with the independent multi-sphere
T-matrix reference (``mom_multisphere_reference``) and optionally a dense LU
solve. ``--stage dense|clusters|single|compare`` splits the same study into
resumable chunks (each saves its far field under ``--far-dir``; ``compare``
adds the T-matrix reference on the host) so no GPU hold has to cover the whole
study. ``--skip-single`` replaces the single-grid preparation by its analytic
grid-memory estimate (for boxes where it does not fit).

    python -m benchmark.mom_pfft_sparse --size 1 --frequency-mesh 9 \\
        --centres="-2,-2,0;2,-2,0;-2,2,0;2,2,0" --eps 4 2.5 --output sparse_5l.json
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
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.mesh import sphere_mesh
from benchmark.mom_pfft_acceptance import INTEGRATION, _body_lu, _peak_reset, _relative, _sync, angular_rule

C0 = 299792458.0


def build_scene(args):
    wavelength = C0/args.frequency
    structures = []
    for i, centre in enumerate(args.centres):
        mesh = sphere_mesh(radius=args.size*wavelength/2, frequency=args.frequency_mesh,
                           center=tuple(v*wavelength for v in centre), device='cuda')
        structures.append(mw.Structure(mw.Mesh(mesh.vertices, mesh.faces, recenter=False, fill_mode='solid'),
                                       mw.Material(eps_r=args.eps[i % len(args.eps)]), name=f'body{i}'))
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()])


def config(args, grids, preconditioner=None):
    pfft = mw.PFFTConfig(points_per_wavelength=args.grid_ppw, order=args.order, target_accuracy=args.target,
                         grid_precision=args.grid_precision, exterior_grids=grids,
                         cluster_separation=args.separation, solve_dual_tolerance=args.admission_rtol,
                         original_sample_tolerance=args.original_rtol)
    return mw.MoMConfig(integration=INTEGRATION, operator_backend='pfft', riesz_storage='local_rwg_v1', pfft=pfft,
                        right_preconditioner=preconditioner,
                        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=args.rtol, atol=0.,
                                              maxiter=args.maxiter, restart=args.restart,
                                              stagnation_window=args.restart))


def prepare(args, scene, grids, preconditioner=None):
    _peak_reset()
    start = _sync()
    prepared = mw.Simulation.mom(scene, frequency=args.frequency, config=config(args, grids, preconditioner)).prepare()
    seconds = _sync()-start
    peak = torch.cuda.max_memory_allocated()
    handle = prepared.solver.handle
    receipt = handle.actions.receipt()
    return prepared, {'prepare_seconds': seconds, 'prepare_peak_bytes': peak,
                      'retained_bytes': receipt['retained_bytes'],
                      'exterior_retained_bytes': receipt['regions']['exterior']['retained_bytes'],
                      'exterior': receipt['regions']['exterior']}


def action_record(handle, x, rows, exact):
    times = []
    for _ in range(3):
        start = _sync()
        y = handle.apply(x)
        times.append(_sync()-start)
    total = handle.shape[0]//2
    sampled = y[torch.cat((rows, rows+total))]
    return y, {'matvec_seconds_median': sorted(times)[1], 'sampled_row_relative_error': _relative(sampled, exact)}


def single_grid_estimate(args, scene):
    """Padded exterior grid of the single-grid variant and its kernel plus work-grid bytes."""
    from witwin.maxwell.mom.pfft_operator import _fast_size
    prepared_scene = scene
    vertices = torch.cat([s.geometry.vertices for s in prepared_scene.structures])
    wavelength = C0/args.frequency
    spacing = wavelength/(args.grid_ppw or 15.)
    order = args.order or 3
    shape = [int(math.ceil(float(e)/spacing))+order+3 for e in (vertices.amax(0)-vertices.amin(0))]
    padded = [_fast_size(2*n-1) for n in shape]
    element = 8 if args.grid_precision == 'complex64' else 16
    return {'grid_shape': shape, 'padded_shape': padded, 'kernel_bytes': 4*math.prod(padded)*element,
            'work_grid_bytes': math.prod(padded)*element, 'spacing_m': spacing}


def tmatrix_far_field(args, directions):
    from benchmark.mom_multisphere_reference import multisphere_far_field
    wavelength = C0/args.frequency
    spheres = [(tuple(v*wavelength for v in c), args.size*wavelength/2, args.eps[i % len(args.eps)])
               for i, c in enumerate(args.centres)]
    start = time.perf_counter()
    field, record = multisphere_far_field(spheres, 2*math.pi/wavelength, (0., 0., 1.), (1., 0., 0.),
                                          directions.cpu().numpy())
    record['seconds'] = time.perf_counter()-start
    return torch.tensor(field, device='cuda'), record


def staged(args, scene, directions, weights):
    """One resumable chunk: solve one variant and save its far field, or compare saved ones."""
    args.far_dir.mkdir(parents=True, exist_ok=True)
    path = args.far_dir/f'{args.stage}.pt'
    if args.stage == 'compare':
        fields = {name: torch.load(args.far_dir/f'{name}.pt') for name in ('dense', 'clusters', 'single')
                  if (args.far_dir/f'{name}.pt').exists()}
        reference, record = tmatrix_far_field(args, directions)
        out = {'tmatrix': record, 'records': {n: f['record'] for n, f in fields.items()}}
        for name, item in fields.items():
            far = item['far'].to('cuda')
            out[f'{name}_vs_tmatrix'] = _relative(far, reference)
            for other, second in fields.items():
                if other < name:
                    out[f'{name}_vs_{other}'] = _relative(far, second['far'].to('cuda'))
        return out
    _peak_reset()
    start = _sync()
    if args.stage == 'dense':
        prepared = mw.Simulation.mom(scene, frequency=args.frequency, config=mw.MoMConfig(
            integration=INTEGRATION, riesz_storage='compact_rwg_v1',
            stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))).prepare()
    else:
        prepared, _ = prepare(args, scene, args.stage, mw.FunctionRightPreconditioner(_body_lu, name='body_lu'))
    prepare_seconds = _sync()-start
    result = prepared.run()
    stats = result.stats()
    far, _ = result.surface_currents.far_field(directions)
    record = {'stage': args.stage, 'unknowns': prepared.solver.handle.shape[0], 'prepare_seconds': prepare_seconds,
              'total_seconds': _sync()-start, 'peak_bytes': torch.cuda.max_memory_allocated(),
              'iterations': stats.get('iterations'), 'converged': stats.get('converged')}
    torch.save({'far': far.cpu(), 'record': record}, path)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--size', type=float, default=1., help='sphere diameter in wavelengths')
    parser.add_argument('--centres', required=True, help="'x,y,z;...' in wavelengths (use --centres=...)")
    parser.add_argument('--frequency-mesh', type=int, default=9)
    parser.add_argument('--eps', type=float, nargs='+', default=[4.])
    parser.add_argument('--frequency', type=float, default=C0)
    parser.add_argument('--grid-ppw', type=float, default=None)
    parser.add_argument('--order', type=int, default=None)
    parser.add_argument('--target', type=float, default=1e-3)
    parser.add_argument('--grid-precision', default='complex64')
    parser.add_argument('--separation', type=float, default=2.)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--admission-rtol', type=float, default=1e-4)
    parser.add_argument('--original-rtol', type=float, default=3e-2)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--restart', type=int, default=300)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--skip-single', action='store_true')
    parser.add_argument('--single-solve', action='store_true', help='also solve the single-grid variant')
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--tmatrix', action='store_true')
    parser.add_argument('--far-theta', type=int, default=24)
    parser.add_argument('--stage', choices=('all', 'dense', 'clusters', 'single', 'compare'), default='all')
    parser.add_argument('--far-dir', type=Path, default=None)
    parser.add_argument('--memory-fraction', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage != 'all' and args.far_dir is None:
        raise SystemExit('--stage requires --far-dir')
    args.centres = [tuple(float(v) for v in c.split(',')) for c in args.centres.split(';')]
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    report = {'schema': 'mom_pfft_sparse_v1', 'arguments': {k: v for k, v in vars(args).items() if k != 'output'},
              'device': torch.cuda.get_device_name(0)}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')

    scene = build_scene(args)
    directions, weights = angular_rule(args.far_theta)
    if args.stage != 'all':
        report['stage'] = staged(args, scene, directions, weights)
        save()
        print(json.dumps(report['stage'], default=str), flush=True)
        return 0
    generator = torch.Generator(device='cuda').manual_seed(11)
    clustered, record = prepare(args, scene, 'clusters', mw.FunctionRightPreconditioner(_body_lu, name='body_lu'))
    handle = clustered.solver.handle
    n = handle.shape[0]
    total = n//2
    x = torch.randn(n, generator=generator, device='cuda', dtype=torch.complex128)
    rows = torch.randperm(total, generator=generator, device='cuda')[:args.rows].sort().values
    start = _sync()
    exact = handle.provider.sampled_action(rows, x)
    report['exact_rows'] = {'rows': len(rows), 'seconds': _sync()-start}
    y_clusters, action = action_record(handle, x, rows, exact)
    report['clusters'] = {**record, **action, 'unknowns': n}
    save()
    print(json.dumps({'clusters': {k: report['clusters'][k] for k in ('prepare_seconds', 'retained_bytes',
                                                                         'sampled_row_relative_error')}}), flush=True)
    single = None
    if not args.skip_single:
        try:
            single, record = prepare(args, scene, 'single')
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            report['single'] = {'estimate': single_grid_estimate(args, scene), 'prepared': False,
                                'out_of_memory': str(error).splitlines()[0]}
    if single is None:
        report.setdefault('single', {'estimate': single_grid_estimate(args, scene), 'prepared': False})
    else:
        y_single, action = action_record(single.solver.handle, x, rows, exact)
        report['single'] = {**record, **action, 'clusters_vs_single_relative': _relative(y_clusters, y_single)}
        if args.single_solve:
            _peak_reset()
            start = _sync()
            result = single.run()
            stats = result.stats()
            report['single']['solve'] = {'seconds': _sync()-start, 'iterations': stats['iterations'],
                                         'converged': stats['converged']}
            report['single']['far_field'] = result.surface_currents.far_field(directions)[0]
        del single
        torch.cuda.empty_cache()
        save()
        print(json.dumps({'single': {k: report['single'][k] for k in ('retained_bytes', 'sampled_row_relative_error',
                                                                       'clusters_vs_single_relative')}}), flush=True)
    _peak_reset()
    start = _sync()
    result = clustered.run()
    stats = result.stats()
    far, _ = result.surface_currents.far_field(directions)
    sections = result.surface_currents.plane_wave_cross_sections(directions, weights)
    report['clusters']['solve'] = {
        'seconds': _sync()-start, 'peak_bytes': torch.cuda.max_memory_allocated(), 'iterations': stats['iterations'],
        'converged': stats['converged'], 'reason': stats['reason'], 'relative_residual': stats.get('relative_residual'),
        'preconditioner_setup_seconds': stats.get('right_preconditioner', {}).get('setup_seconds'),
        'error_budget': {k: stats['error_budget'].get(k) for k in ('checked', 'compressed_dual_pass',
                                                                   'original_sampled_pass')},
        'optical_theorem_defect': abs(float(sections['extinction_m2'])-float(sections['scattering_m2']))
        / abs(float(sections['extinction_m2']))}
    save()
    if isinstance(report['single'].get('far_field'), torch.Tensor):
        report['single']['far_field_vs_clusters'] = _relative(report['single'].pop('far_field'), far)
    if args.dense:
        _peak_reset()
        start = _sync()
        dense = mw.Simulation.mom(scene, frequency=args.frequency, config=mw.MoMConfig(
            integration=INTEGRATION, riesz_storage='compact_rwg_v1',
            stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))).run()
        far_dense, _ = dense.surface_currents.far_field(directions)
        report['dense'] = {'seconds': _sync()-start, 'peak_bytes': torch.cuda.max_memory_allocated(),
                           'clusters_far_field_relative_difference': _relative(far, far_dense)}
        del dense
        save()
    if args.tmatrix:
        reference, record = tmatrix_far_field(args, directions)
        report['tmatrix'] = {**record, 'clusters_far_field_relative_error': _relative(far, reference)}
        if args.dense:
            report['tmatrix']['dense_far_field_relative_error'] = _relative(far_dense, reference)
        save()
    print(json.dumps({k: report[k] for k in ('dense', 'tmatrix') if k in report}, default=str))
    print(json.dumps({'solve': report['clusters']['solve']}, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

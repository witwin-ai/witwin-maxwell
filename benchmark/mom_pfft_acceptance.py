"""Public pFFT backend acceptance: dense LU, Mie, refinement and preconditioners.

Bodies are icosahedral spheres (``sphere_mesh`` frequency n, 20 n^2 faces) or
structured cubes as explicit solid meshes, placed at the given centres. For
one scene the driver solves with ``operator_backend='pfft'`` through
Scene -> Simulation -> Result and optionally compares with:

- a dense LU solve of the same scene (currents and far field),
- Mie theory (single sphere at the origin),
- a refined mesh (h/sqrt(2)) and a tighter accelerator setting (order+1),
- right preconditioners (none, leaf blocks, exact per-body LU) on the same
  prepared pFFT operator.

    python -m benchmark.mom_pfft_acceptance --shape sphere --size 1 --frequency-mesh 15 \\
        --dense --mie --output pfft_sphere1.json
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
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.preconditioning import bind_right_preconditioner

C0 = 299792458.0
INTEGRATION = IntegrationConfig(order=4, singular_order=4, far_pair_batch=16384)


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def _peak_reset():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def build_scene(args, frequency_mesh):
    wavelength = C0/args.frequency
    size = args.size*wavelength
    centres = [tuple(float(v)*wavelength for v in c.split(',')) for c in args.centres]
    structures = []
    for i, centre in enumerate(centres):
        if args.shape == 'sphere':
            mesh = sphere_mesh(radius=size/2, frequency=frequency_mesh, center=centre, device='cuda')
            vertices, faces = mesh.vertices, mesh.faces
        else:
            from benchmark.mom_large_scale_l1 import _box_surface
            v, f = _box_surface((size,)*3, size/frequency_mesh, centre)
            vertices, faces = torch.tensor(v, device='cuda'), torch.tensor(np.asarray(f), device='cuda')
        structures.append(mw.Structure(mw.Mesh(vertices, faces, recenter=False, fill_mode='solid'),
                                       mw.Material(eps_r=args.eps[i % len(args.eps)]), name=f'body{i}'))
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()]), centres


def rule(args):
    stop = dict(rtol=args.rtol, atol=0., maxiter=args.maxiter, restart=args.restart, stagnation_window=args.restart)
    return stop


def pfft_config(args, order=None, ppw=None, **extra):
    extra.setdefault('right_preconditioner', preconditioner_factory(args.solve_preconditioner))
    pfft = mw.PFFTConfig(points_per_wavelength=ppw or args.grid_ppw, order=order or args.order,
                         grid_precision=args.grid_precision, solve_dual_tolerance=args.admission_rtol,
                         original_sample_tolerance=args.original_rtol)
    return mw.MoMConfig(integration=INTEGRATION, operator_backend='pfft', riesz_storage='local_rwg_v1', pfft=pfft,
                        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', **rule(args)), **extra)


def angular_rule(n_theta):
    mu, w = np.polynomial.legendre.leggauss(n_theta)
    n_phi = 2*n_theta
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    theta = np.arccos(mu)
    d = np.stack([np.outer(np.sin(theta), np.cos(phi)), np.outer(np.sin(theta), np.sin(phi)),
                  np.outer(np.cos(theta), np.ones(n_phi))], -1).reshape(-1, 3)
    return (torch.tensor(d, device='cuda'), torch.tensor(np.repeat(w, n_phi)*2*np.pi/n_phi, device='cuda'))


def _relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


def solve_pfft(args, scene, config, directions, weights):
    _peak_reset()
    start = _sync()
    prepared = mw.Simulation.mom(scene, frequency=args.frequency, config=config).prepare()
    prepare_seconds = _sync()-start
    prepare_peak = torch.cuda.max_memory_allocated()
    handle = prepared.solver.handle
    x = torch.randn(handle.shape[0], device='cuda', dtype=torch.complex128)
    times = []
    for _ in range(3):
        start = _sync()
        handle.apply(x)
        times.append(_sync()-start)
    _peak_reset()
    start = _sync()
    result = prepared.run()
    solve_seconds = _sync()-start
    stats = result.stats()
    far, _ = result.surface_currents.far_field(directions)
    sections = result.surface_currents.plane_wave_cross_sections(directions, weights)
    receipt = handle.actions.receipt()
    record = {'unknowns': handle.shape[0], 'prepare_seconds': prepare_seconds, 'prepare_peak_bytes': prepare_peak,
              'solve_seconds': solve_seconds, 'solve_peak_bytes': torch.cuda.max_memory_allocated(),
              'matvec_seconds_median': sorted(times)[1], 'retained_bytes': receipt['retained_bytes'],
              'regions': {name: {'padded_shape': r['padded_shape'], 'kernel_bytes': r['kernel_bytes'],
                                 'projection_bytes': r['projection']['retained_bytes'],
                                 'precorrection': r['precorrection']} for name, r in receipt['regions'].items()},
              'converged': stats['converged'], 'reason': stats['reason'], 'iterations': stats['iterations'],
              'relative_residual': stats.get('relative_residual'),
              'error_budget': {k: stats['error_budget'].get(k) for k in ('checked', 'compressed_dual_pass',
                                                                         'original_sampled_pass')},
              'residual_report': stats['error_budget'].get('residual_report'),
              'optical_theorem_defect': abs(float(sections['extinction_m2'])-float(sections['scattering_m2']))
                  / abs(float(sections['extinction_m2']))}
    return prepared, result, far, record


def _body_lu(context):
    blocks = []
    for body in context.system.bodies:
        edges = torch.arange(body.edge_offset, body.edge_offset+body.basis.n_edges, device=context.device)
        blocks.append((context.coefficient_indices(edges), torch.linalg.lu_factor(context.principal_block(edges))))

    def apply(residual):
        result = torch.empty_like(residual)
        for index, (factor, pivot) in blocks:
            result[index] = torch.linalg.lu_solve(factor, pivot, residual[index, None])[:, 0]
        return result
    return apply


def preconditioner_factory(name):
    if name in (None, 'none', 'diag'):
        return None
    if name.startswith('leaf:'):
        return mw.LeafBlockRightPreconditioner(int(name.split(':')[1]), budget_bytes=4*1024**3)
    if name == 'body':
        return mw.FunctionRightPreconditioner(_body_lu, name='body_block_lu')
    if name == 'sparse-lu':
        return mw.NearFieldRightPreconditioner()
    raise ValueError(f'Unknown preconditioner {name}')


def preconditioner_runs(args, prepared, names):
    handle = prepared.solver.handle
    rhs = prepared.solver.rhs
    stopping = prepared.simulation.config.stopping
    out = {}
    for name in names:
        factory = preconditioner_factory(name)
        _peak_reset()
        setup = 0.
        start = _sync()
        if factory is None:
            _, log = handle.solve(rhs, stopping)
        else:
            bound, receipt = bind_right_preconditioner(factory, handle, args.frequency)
            setup = receipt['setup_seconds']
            start = _sync()
            _, log = handle.solve(rhs, stopping, right_preconditioner=bound.apply)
        out[name] = {'converged': log['converged'], 'reason': log['reason'], 'iterations': log['iterations'],
                     'relative_residual': log.get('relative_residual'), 'solve_seconds': _sync()-start,
                     'preconditioner_setup_seconds': setup, 'peak_bytes': torch.cuda.max_memory_allocated()}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shape', choices=('sphere', 'cube'), default='sphere')
    parser.add_argument('--size', type=float, default=1., help='diameter or edge in wavelengths')
    parser.add_argument('--centres', default='0,0,0',
                        help="body centres in wavelengths, 'x,y,z;x,y,z;...' (use --centres=... for negatives)")
    parser.add_argument('--frequency-mesh', type=int, default=15)
    parser.add_argument('--eps', type=float, nargs='+', default=[4.])
    parser.add_argument('--frequency', type=float, default=C0)
    parser.add_argument('--grid-ppw', type=float, default=15.)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--grid-precision', default='complex64')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--admission-rtol', type=float, default=1e-4,
                        help='compressed dual residual budget (pFFT operator itself)')
    parser.add_argument('--original-rtol', type=float, default=3e-2,
                        help='sampled original-row residual budget (sits above the accelerator error)')
    parser.add_argument('--maxiter', type=int, default=1500)
    parser.add_argument('--restart', type=int, default=300)
    parser.add_argument('--dense', action='store_true', help='dense LU reference on the same mesh')
    parser.add_argument('--mie', action='store_true')
    parser.add_argument('--refine', action='store_true', help='also solve at mesh frequency round(n*sqrt(2))')
    parser.add_argument('--tighter', action='store_true', help='also solve with order+1 on the same mesh')
    parser.add_argument('--preconditioners', nargs='*', default=[])
    parser.add_argument('--solve-preconditioner', default='none',
                        help='right preconditioner of the reported pFFT solves: none | leaf:N | body | sparse-lu')
    parser.add_argument('--far-theta', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.centres = args.centres.split(';')
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    wavelength = C0/args.frequency
    extent = max(abs(float(v)) for c in args.centres for v in c.split(','))*2+args.size
    directions, weights = angular_rule(args.far_theta or max(16, math.ceil(math.pi*extent)+12))
    report = {'schema': 'mom_pfft_acceptance_v1', 'arguments': {k: v for k, v in vars(args).items() if k != 'output'},
              'device': torch.cuda.get_device_name(0)}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')

    scene, centres = build_scene(args, args.frequency_mesh)
    prepared, result, far, record = solve_pfft(args, scene, pfft_config(args), directions, weights)
    report['pfft'] = record
    x = result.surface_currents.scattered_coefficients
    save()
    print(json.dumps({'pfft': {k: record[k] for k in ('unknowns', 'prepare_seconds', 'solve_seconds', 'iterations',
                                                      'converged', 'matvec_seconds_median')}}), flush=True)
    if args.mie and args.shape == 'sphere' and len(centres) == 1:
        from benchmark.sie_reference import sphere_far_field
        k = 2*math.pi/wavelength
        mie = torch.tensor(sphere_far_field(args.size*wavelength/2, k, complex(args.eps[0]),
                                            directions.cpu().numpy()), device='cuda')
        report['mie'] = {'pfft_far_field_relative_error': _relative(far, mie)}
    if args.dense:
        _peak_reset()
        start = _sync()
        dense = mw.Simulation.mom(scene, frequency=args.frequency, config=mw.MoMConfig(
            integration=INTEGRATION, riesz_storage='compact_rwg_v1',
            stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))).run()
        far_dense, _ = dense.surface_currents.far_field(directions)
        x_dense = dense.surface_currents.scattered_coefficients
        report['dense'] = {'seconds': _sync()-start, 'peak_bytes': torch.cuda.max_memory_allocated(),
                           'current_relative_difference': _relative(x, x_dense),
                           'far_field_relative_difference': _relative(far, far_dense)}
        if 'mie' in report:
            report['mie']['dense_far_field_relative_error'] = _relative(far_dense,
                torch.tensor(sphere_far_field(args.size*wavelength/2, 2*math.pi/wavelength, complex(args.eps[0]),
                                              directions.cpu().numpy()), device='cuda'))
        del dense
        save()
    if args.preconditioners:
        report['preconditioners'] = preconditioner_runs(args, prepared, args.preconditioners)
        save()
    del prepared, result
    torch.cuda.empty_cache()
    if args.tighter:
        _, tight, far_tight, rec = solve_pfft(args, scene, pfft_config(args, order=args.order+1), directions, weights)
        report['tighter'] = {**{k: rec[k] for k in ('iterations', 'converged', 'prepare_seconds', 'retained_bytes')},
                             'order': args.order+1, 'far_field_relative_difference': _relative(far, far_tight)}
        del tight
        save()
    if args.refine:
        refined_mesh = round(args.frequency_mesh*math.sqrt(2))
        fine_scene, _ = build_scene(args, refined_mesh)
        _, fine, far_fine, rec = solve_pfft(args, fine_scene, pfft_config(args), directions, weights)
        report['refined'] = {**{k: rec[k] for k in ('unknowns', 'iterations', 'converged', 'prepare_seconds',
                                                    'solve_seconds', 'retained_bytes')},
                             'frequency_mesh': refined_mesh, 'far_field_relative_difference': _relative(far, far_fine)}
        if 'mie' in report:
            from benchmark.sie_reference import sphere_far_field
            mie = torch.tensor(sphere_far_field(args.size*wavelength/2, 2*math.pi/wavelength, complex(args.eps[0]),
                                                directions.cpu().numpy()), device='cuda')
            report['refined']['mie_far_field_relative_error'] = _relative(far_fine, mie)
        save()
    print(json.dumps({k: v for k, v in report.items() if k in ('mie', 'dense', 'tighter', 'refined', 'preconditioners')},
                     default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

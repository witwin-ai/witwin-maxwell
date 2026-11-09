"""Open-sheet PEC EFIE on the pFFT backend against the dense S-1 operator.

Square 1λ plate, mesh side/cells. Two studies on the same meshes:

- RCS: normal incidence, x polarization (the S-1 ``plate`` study): monostatic
  and 30° bistatic RCS and total scattering cross section from dense LU and
  from pFFT GMRES (rtol 1e-8, no right preconditioner).
- iterations: oblique incidence (0, 0.6, -0.8), GMRES rtol 1e-6 without
  restart and without diagonal scaling (the S-1 ``iterations`` study), with no
  right preconditioner and with the Calderón preconditioner (whose setup
  stays dense); solution error against dense LU of the dense operator.

    python -m benchmark.mom_pfft_efie --cells 8 16 32 --output pfft_efie.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference

C0 = 299792458.0


def sphere_rule(n_theta=32, n_phi=64):
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2*math.pi*np.arange(n_phi)/n_phi
    ct, ph = np.meshgrid(x, phi, indexing='ij')
    st = np.sqrt(1-ct**2)
    directions = np.stack((st*np.cos(ph), st*np.sin(ph), ct), -1).reshape(-1, 3)
    return (torch.tensor(directions, device='cuda'),
            torch.tensor(np.repeat(w, n_phi)*2*math.pi/n_phi, device='cuda'))


def scene(side, frequency, direction, polarization):
    return mw.Scene(structures=[mw.Structure(mw.Box(size=(side, side, 0)), mw.Material.pec())],
                    sources=[mw.PlaneWave(direction=direction, polarization=polarization,
                                          source_time=mw.CW(frequency=frequency))], device='cuda')


PFFT = {}


def config(backend, edge, stopping, factory=None):
    extra = {}
    if backend == 'pfft':
        extra = dict(operator_backend='pfft', riesz_storage='local_rwg_v1',
                     pfft=mw.PFFTConfig(solve_dual_tolerance=1e-4, **PFFT))
    return mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=edge), stopping=stopping,
                        right_preconditioner=factory, **extra)


def run(s, frequency, cfg):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prepared = mw.Simulation.mom(s, frequency=frequency, config=cfg).prepare()
    torch.cuda.synchronize()
    setup = time.perf_counter()-start
    result = prepared.run()
    torch.cuda.synchronize()
    return prepared, result, {'setup_seconds': setup, 'solve_seconds': result.solver_stats.get('solve_seconds'),
                              'iterations': result.solver_stats['iterations'],
                              'converged': result.solver_stats['converged'],
                              'peak_bytes': torch.cuda.max_memory_allocated()}


def rcs_study(args, side, cells, directions, weights):
    frequency = args.frequency
    backscatter = torch.tensor([[0., 0., 1.]], device='cuda', dtype=torch.float64)
    bistatic = torch.tensor([[math.sin(math.radians(30)), 0., math.cos(math.radians(30))]], device='cuda',
                            dtype=torch.float64)
    s = scene(side, frequency, (0., 0., -1.), (1., 0., 0.))
    out = {}
    for backend in ('dense', 'pfft'):
        stopping = (StoppingRule(method='direct') if backend == 'dense' else
                    StoppingRule(method='gmres', rtol=1e-8, maxiter=3000, restart=3000, preconditioner='none',
                                 stagnation_window=3000))
        prepared, result, record = run(s, frequency, config(backend, side/cells, stopping))
        data = result.surface_currents
        sections = data.plane_wave_cross_sections(directions, weights)
        record.update(monostatic_rcs_m2=float(data.radar_cross_section(backscatter)[0]),
                      bistatic30_rcs_m2=float(data.radar_cross_section(bistatic)[0]),
                      scattering_m2=float(sections['scattering_m2']), edges=len(data.J))
        if backend == 'pfft':
            receipt = prepared.solver.handle.receipt()
            record.update(retained_bytes=receipt['retained_bytes'],
                          padded_shape=receipt['region']['padded_shape'],
                          spacing_rule=receipt['region']['projection']['spacing_rule'],
                          spacing_to_mean_edge=receipt['region']['projection']['spacing_to_mean_edge'],
                          precorrection_pairs=receipt['precorrection']['pairs_stored'])
        out[backend] = record
        out[backend+'_J'] = data.J
    dense_j, fast_j = out.pop('dense_J'), out.pop('pfft_J')
    out['current_relative_difference'] = float((fast_j-dense_j).norm()/dense_j.norm())
    for key in ('monostatic_rcs_m2', 'bistatic30_rcs_m2', 'scattering_m2'):
        out[key+'_relative_difference'] = abs(out['pfft'][key]-out['dense'][key])/abs(out['dense'][key])
    return out


def iteration_study(args, side, cells):
    from witwin.maxwell.mom.efie_preconditioning import CalderonPreconditioner
    frequency = args.frequency
    s = scene(side, frequency, (0.0, 0.6, -0.8), (1.0, 0.0, 0.0))
    stopping = StoppingRule(method='gmres', rtol=args.rtol, maxiter=args.maxiter, restart=args.maxiter,
                            preconditioner='none', stagnation_window=args.maxiter)
    dense_handle = mw.Simulation.mom(s, frequency=frequency, config=config('dense', side/cells, stopping)).prepare()
    reference = solve_reference(dense_handle.solver.handle, dense_handle.solver.rhs, StoppingRule(method='direct'))[0]
    del dense_handle
    out = {}
    for backend in ('dense', 'pfft'):
        for name, factory in (('none', None), ('calderon', CalderonPreconditioner())):
            if f'{backend}_{name}' not in args.cases:
                continue
            prepared, result, record = run(s, frequency, config(backend, side/cells, stopping, factory))
            record['solution_relative_error'] = float((result.surface_currents.J-reference).norm()/reference.norm())
            if prepared.solver.right_preconditioner_receipt is not None:
                record['preconditioner_setup_seconds'] = prepared.solver.right_preconditioner_receipt.get(
                    'setup_seconds')
            out[f'{backend}_{name}'] = record
            print(json.dumps({'cells': cells, 'case': f'{backend}_{name}', **record}), flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--side', type=float, default=1.)
    parser.add_argument('--cells', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--frequency', type=float, default=3e8)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=1500)
    parser.add_argument('--cases', nargs='+', default=['dense_none', 'dense_calderon', 'pfft_none', 'pfft_calderon'])
    parser.add_argument('--skip-rcs', action='store_true')
    parser.add_argument('--grid-precision', default='complex64')
    parser.add_argument('--target', type=float, default=1e-3)
    parser.add_argument('--memory-fraction', type=float, default=0.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    PFFT.update(grid_precision=args.grid_precision, target_accuracy=args.target)
    side = args.side*C0/args.frequency
    directions, weights = sphere_rule()
    report = {'schema': 'mom_pfft_efie_v1', 'arguments': vars(args), 'device': torch.cuda.get_device_name(0),
              'records': []}
    for cells in args.cells:
        record = {'cells': cells}
        if not args.skip_rcs:
            record['rcs'] = rcs_study(args, side, cells, directions, weights)
            print(json.dumps({'cells': cells, 'rcs': {k: v for k, v in record['rcs'].items()
                                                      if not isinstance(v, dict)}}), flush=True)
        record['iterations'] = iteration_study(args, side, cells)
        report['records'].append(record)
        with open(args.output, 'w') as stream:
            json.dump(report, stream, indent=2, default=str)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

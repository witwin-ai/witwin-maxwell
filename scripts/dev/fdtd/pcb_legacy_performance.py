"""Interleave source-pinned legacy field benchmarks with unchanged physical inputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import witwin.maxwell as mw

CASES = ('vacuum_dipole', 'dielectric_dipole', 'two_frequency_dipole')


def source_identity():
    root = Path(mw.__file__).resolve().parent
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    return {'root': str(root), 'python_sources': hashes}


def processes():
    row = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                          '--format=csv,noheader'], text=True, capture_output=True)
    return {'returncode': row.returncode, 'stdout': row.stdout, 'stderr': row.stderr}


def run_one(args):
    identity = source_identity()
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),)*3), grid=mw.GridSpec.uniform(.002),
        boundary=mw.BoundarySpec.pml(num_layers=6), device='cuda',
        sources=[mw.PointDipole(position=(0., 0., 0.), polarization='Ez',
                               source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9))],
        monitors=[mw.PointMonitor('probe', position=(.01, 0., 0.), fields=('Ez',))])
    if args.case == 'dielectric_dipole':
        scene.add_structure(mw.Structure(mw.Box(size=(.02, .02, .02)), mw.Material(eps_r=2.2)))
    frequencies = (0.8e9, 1.2e9) if args.case == 'two_frequency_dipole' else (1e9,)
    simulation = mw.Simulation.fdtd(scene, frequencies=frequencies, full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=args.steps), spectral_sampler=mw.SpectralSampler(window='none'))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prepared = simulation.prepare()
    torch.cuda.synchronize()
    preparation = time.perf_counter()-start
    before = processes()
    start = time.perf_counter()
    result = prepared.run()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    after = processes()
    peak = torch.cuda.max_memory_allocated()
    fields = {key: value.detach().cpu() for key, value in result.fields.items()}
    fields['point_Ez'] = result.monitor('probe')['data'].detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fields, args.output.with_suffix('.pt'))
    report = {'case': args.case, 'pid': os.getpid(), 'source': identity,
              'source_unchanged': source_identity() == identity, 'steps': args.steps,
              'dt_s': prepared.solver.dt, 'duration_s': args.steps*prepared.solver.dt,
              'gpu': torch.cuda.get_device_name(), 'preparation_seconds': preparation,
              'run_seconds': seconds, 'peak_allocated_bytes': peak,
              'processes_before': before, 'processes_after': after}
    args.output.write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--source-roots', type=Path, nargs=2)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.case:
        run_one(args)
        return
    if not args.source_roots or args.repeats < 5:
        parser.error('Two frozen source roots and at least five measured repeats are required.')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'legacy-field-performance/1', 'samples': [],
              'scope': 'Synchronized prepare/run/peak allocation; spectral export included, persistence excluded.'}
    for case in CASES:
        reference = None
        for repeat in range(-1, args.repeats):
            for version in ((0, 1) if repeat % 2 else (1, 0)):
                path = args.output/f'{case}-{repeat+1}-{version}.json'
                env = dict(os.environ, PYTHONPATH=str(args.source_roots[version].resolve()))
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--case', case,
                    '--output', str(path.resolve()), '--steps', str(args.steps)],
                    cwd=args.source_roots[version], env=env, check=True)
                row = json.loads(path.read_text())
                actual = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
                if reference is None:
                    reference = actual
                errors = {}
                for name, expected in reference.items():
                    torch.testing.assert_close(actual[name], expected, rtol=2e-4, atol=1e-10)
                    errors[name] = float(torch.linalg.vector_norm(actual[name]-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-30))
                row.update(version=version, warmup=repeat == -1, output_relative_l2_errors=errors)
                if not row['source_unchanged']:
                    raise RuntimeError('A frozen numerical source changed during a timing sample.')
                report['samples'].append(row)
                (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps({key: row[key] for key in ('case', 'version', 'warmup', 'run_seconds', 'peak_allocated_bytes')}), flush=True)
    report['comparisons'] = {}
    for case in CASES:
        rows = [r for r in report['samples'] if r['case'] == case and not r['warmup']]
        timing = [[r['run_seconds'] for r in rows if r['version'] == version] for version in (0, 1)]
        memory = [[r['peak_allocated_bytes'] for r in rows if r['version'] == version] for version in (0, 1)]
        report['comparisons'][case] = {'median_run_seconds': [statistics.median(v) for v in timing],
            'run_range_seconds': [[min(v), max(v)] for v in timing],
            'time_ratio': statistics.median(timing[1])/statistics.median(timing[0]),
            'peak_memory_ratio': max(memory[1])/max(memory[0])}
    report['qualification'] = 'Measured; process/exclusivity and the frozen five-percent gates require explicit review.'
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

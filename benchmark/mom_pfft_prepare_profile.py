"""Profile preparation with the pFFT acceptance driver's scene and configuration.

Run under the CUDA sharing locks, with CUDA_VISIBLE_DEVICES=1. For example::

    python -m benchmark.mom_pfft_prepare_profile --size 2 --frequency-mesh 18 \
        --solve-preconditioner leaf:64 --output /tmp/pfft-prepare-2.json

Stage times include their children. Only preparation is executed; no solve or
field query is needed. A host profile resolves work inside the timed stages
without adding a CUDA synchronization to every block or quadrature call.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path

import torch

import witwin.maxwell as mw
from benchmark.mom_fmm_prepare_profile import StageTracer
from benchmark.mom_pfft_acceptance import C0, build_scene, pfft_config


class PrepareTracer(StageTracer):
    """Print completed stages so a bounded GPU run retains partial timings."""

    def _exit(self, name, frame):
        super()._exit(name, frame)
        print(json.dumps({'stage': name, **self.stages[name]}), flush=True)


def install(tracer):
    from witwin.maxwell.mom import (cluster, nearfield, pfft_grid, pfft_lifting, pfft_operator,
                                    pfft_system, preconditioning, riesz, runtime)
    for owner, attribute, name in (
        (runtime, 'compile_surfaces', 'surfaces'),
        (nearfield.ScalarSystemBlocks, '__init__', 'provider_geometry'),
        (pfft_system.PFFTSystemActions, '__init__', 'actions'),
        (pfft_grid.PFFTProjection, '__init__', 'projection'),
        (pfft_operator.PFFTRegion, '__init__', 'grid_kernels'),
        (pfft_system, 'near_pairs', 'precorrection.pair_layout'),
        (pfft_system, 'pmchwt_pair_entries', 'precorrection.exact_entries'),
        (pfft_operator.PFFTRegion, 'grid_entries', 'precorrection.grid_entries'),
        (cluster, 'build_support_octree', 'support_tree'),
        (riesz.LocalRWGRiesz, '__init__', 'riesz_metric'),
        (pfft_system._system_operator_class(), '_bind_physical', 'physical_binding'),
        (preconditioning, 'bind_right_preconditioner', 'right_preconditioner'),
        (runtime.SystemOperator, 'build_rhs', 'rhs_lifting'),
        (pfft_lifting.PFFTLifting, '__init__', 'rhs_lifting.shared_grid_setup'),
        (runtime, '_signature', 'scene_signature'),
    ):
        tracer.wrap(owner, attribute, name)
    # The acceptance configuration has one exterior followed by each body's interior.
    original = pfft_system.PFFTPrecorrection.__init__
    regions = iter(('exterior', 'interior'))

    def precorrection(self, *args, **kwargs):
        with tracer.stage('precorrection.'+next(regions)):
            original(self, *args, **kwargs)

    tracer.patches.append((pfft_system.PFFTPrecorrection, '__init__', original))
    pfft_system.PFFTPrecorrection.__init__ = precorrection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shape', choices=('sphere', 'cube'), default='sphere')
    parser.add_argument('--size', type=float, default=2.)
    parser.add_argument('--frequency-mesh', type=int, default=18)
    parser.add_argument('--eps', type=float, default=4.)
    parser.add_argument('--solve-preconditioner', default='leaf:64')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # Match the acceptance command, while capping this process at 20 GiB.
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
    args.frequency, args.centres, args.eps = C0, ['0,0,0'], [args.eps]
    args.grid_ppw, args.order, args.grid_precision = 15., 3, 'complex64'
    args.admission_rtol, args.original_rtol = 1e-4, 3e-2
    args.rtol, args.maxiter, args.restart = 1e-6, 1500, 300
    scene, _ = build_scene(args, args.frequency_mesh)
    config = pfft_config(args)
    tracer = PrepareTracer()
    install(tracer)
    profiler = cProfile.Profile()
    base = torch.cuda.memory_allocated()
    with tracer.stage('prepare_total'):
        prepared = profiler.runcall(mw.Simulation.mom(scene, frequency=args.frequency, config=config).prepare)
    tracer.restore()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(50)
    handle = prepared.solver.handle
    report = {'schema': 'mom_pfft_prepare_profile_v1',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'unknowns': handle.shape[0], 'stages': tracer.stages,
              'construction': handle.actions.receipt(),
              'rhs_bodies': handle.metadata['rhs_bodies'],
              'right_preconditioner': prepared.solver.right_preconditioner_receipt,
              'peak_allocated_bytes': base+tracer.stages['prepare_total']['transient_peak_bytes'],
              'host_profile_cumulative': stream.getvalue()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()

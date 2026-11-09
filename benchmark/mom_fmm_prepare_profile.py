"""Stage-by-stage time and CUDA memory profile of fast MoM preparation.

Wraps the preparation stages of the public ``Simulation.prepare()`` path
(validation, basis, provider, trees, order planning, capacity planning,
expansions, M2L tables, near-field quadrature, Riesz metric, leaf factors,
RHS lifting) with CUDA-synchronised timers and allocator snapshots, then
optionally runs the solve. Each stage records calls, seconds, retained bytes
(allocated after minus before, summed over calls) and the transient peak above
the allocation at entry. Nested stages are reported separately; parent totals
include their children. A host cProfile of the whole preparation is attached.

Example:
    python -m benchmark.mom_fmm_prepare_profile --shape sphere --size 1 \\
        --frequency-mesh 9 --backend fmm --config default --output prof.json
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import functools
import io
import json
import math
import pstats
import time
import traceback
from pathlib import Path

import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig

C0 = 299792458.0


class StageTracer:
    """CUDA-synchronised timing and allocator deltas for wrapped callables."""

    def __init__(self):
        self.stages, self.stack, self.patches = {}, [], []

    def _enter(self):
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        for frame in self.stack:
            frame['peak'] = max(frame['peak'], peak)
        torch.cuda.reset_peak_memory_stats()
        frame = {'allocated': torch.cuda.memory_allocated(), 'peak': 0, 'start': time.perf_counter()}
        self.stack.append(frame)
        return frame

    def _exit(self, name, frame):
        torch.cuda.synchronize()
        frame['peak'] = max(frame['peak'], torch.cuda.max_memory_allocated())
        self.stack.pop()
        for parent in self.stack:
            parent['peak'] = max(parent['peak'], frame['peak'])
        record = self.stages.setdefault(name, {'calls': 0, 'seconds': 0., 'retained_bytes': 0,
                                               'transient_peak_bytes': 0, 'depth': len(self.stack)})
        record['calls'] += 1
        record['seconds'] += time.perf_counter()-frame['start']
        record['retained_bytes'] += torch.cuda.memory_allocated()-frame['allocated']
        record['transient_peak_bytes'] = max(record['transient_peak_bytes'], frame['peak']-frame['allocated'])
        record['depth'] = min(record['depth'], len(self.stack))

    def wrap(self, owner, attribute, name):
        original = owner.__dict__[attribute] if isinstance(owner, type) else getattr(owner, attribute)
        unwrap = original.__func__ if isinstance(original, (classmethod, staticmethod)) else original
        tracer = self

        @functools.wraps(unwrap)
        def wrapped(*args, **kwargs):
            frame = tracer._enter()
            try:
                return unwrap(*args, **kwargs)
            finally:
                tracer._exit(name, frame)
        value = classmethod(wrapped) if isinstance(original, classmethod) else (
            staticmethod(wrapped) if isinstance(original, staticmethod) else wrapped)
        setattr(owner, attribute, value)
        self.patches.append((owner, attribute, original))

    @contextlib.contextmanager
    def stage(self, name):
        frame = self._enter()
        try:
            yield
        finally:
            self._exit(name, frame)

    def restore(self):
        for owner, attribute, original in reversed(self.patches):
            setattr(owner, attribute, original)
        self.patches.clear()


def install(tracer):
    from witwin.maxwell.mom import (runtime, mesh, nearfield, fmm_system, fmm_moments, fmm_regional,
                                    fmm_translation, packed_near, riesz, block_preconditioner,
                                    hierarchical_runtime, cluster)
    plan = [
        (runtime, 'compile_surfaces', 'validation_and_basis'),
        (mesh, 'validate_mesh', 'validation.mesh'),
        (mesh, '_has_intersections', 'validation.self_intersection'),
        (mesh.SurfaceSystem, '_validate_separation', 'validation.body_separation'),
        (mesh.RWGBasis, '__init__', 'basis.rwg_topology'),
        (nearfield.ScalarSystemBlocks, '__init__', 'provider.geometry'),
        (fmm_system, 'build_support_octree', 'trees'),
        (fmm_system, 'plan_multipole_order', 'planner.order'),
        (fmm_system, 'plan_multipole_storage', 'planner.capacity'),
        (fmm_moments.NestedRWGMultipoles, '__init__', 'expansions.moments_and_shifts'),
        (fmm_regional.MultipoleRegionalOperator, '__init__', 'regional'),
        (fmm_regional, 'sphere_block_partition', 'regional.partition'),
        (fmm_translation.ElectromagneticM2L, 'create_many', 'regional.m2l_tables'),
        (riesz.LocalRWGRiesz, '__init__', 'riesz_metric'),
        (block_preconditioner.LeafBlockInverse, '__init__', 'leaf_factors'),
        (hierarchical_runtime.HierarchicalSystemOperator, 'build_rhs', 'rhs_lifting'),
    ]
    for cls in (packed_near.PackedNearBlocks, packed_near.PackedScalarNearChannels,
                packed_near.PackedReciprocalScalarNearChannels, packed_near.StreamedReciprocalScalarNearChannels,
                packed_near.BoundedCachedReciprocalScalarNearChannels):
        plan.append((cls, 'from_layout', 'regional.near_field'))
    if hasattr(fmm_system, 'DirectionalSystemActions'):
        plan.append((fmm_system.DirectionalSystemActions, 'from_automatic', 'directional.automatic_plan_and_build'))
    for owner, attribute, name in plan:
        if (owner.__dict__ if isinstance(owner, type) else vars(owner)).get(attribute) is not None:
            tracer.wrap(owner, attribute, name)
    del cluster


def build_geometry(args, wavelength):
    from witwin.maxwell.mom.mesh import sphere_mesh
    size = args.size*wavelength
    if args.shape == 'sphere':
        surface = sphere_mesh(radius=size/2, frequency=args.frequency_mesh, device='cuda')
        vertices, faces = surface.vertices.cpu(), surface.faces.cpu()
    else:
        from benchmark.mom_large_scale_l1 import _box_surface
        import numpy as np
        v, f = _box_surface((size, size, size), size/args.frequency_mesh, (0., 0., 0.))
        vertices, faces = torch.tensor(v), torch.tensor(np.asarray(f))
    return mw.Mesh(vertices, faces, recenter=False, fill_mode='solid'), len(faces)


def build_config(args):
    budget = int(args.budget_gb*1024**3)
    integration = IntegrationConfig(order=args.order, singular_order=args.singular_order)
    rule = StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=args.rtol, atol=0.,
                        maxiter=args.maxiter, restart=args.restart, stagnation_window=args.restart)
    leaf = dict(leaf_factor_kernel='primary', leaf_factor_max_edges=32) if args.leaf_factors else {}
    if args.backend == 'directional':
        fmm = mw.DirectionalFMMConfig(**({'leaf_factor_kernel': 'original'} if args.leaf_factors else {}))
    elif args.config == 'default':
        fmm = mw.FMMConfig(**leaf)
    elif args.config == 'budgeted':
        fmm = mw.FMMConfig(action_retained_budget_bytes=budget, table_budget_bytes=budget//2,
                           near_budget_bytes=budget//2, projection_budget_bytes=budget//2, **leaf)
    else:
        fmm = mw.FMMConfig(order=args.fmm_order, table_storage='compact_shared_displacement_v1',
                           shift_storage='separable_v1', near_storage='scalar_reciprocal_dk_v1',
                           leaf_size=args.leaf_size, separation_ratio=args.separation_ratio,
                           action_retained_budget_bytes=budget, table_budget_bytes=budget//2,
                           near_budget_bytes=budget//2, projection_budget_bytes=budget//2, **leaf)
    return mw.MoMConfig(integration=integration, operator_backend='fmm', fmm=fmm,
                        riesz_storage='local_rwg_v1', stopping=rule)


def plan_only(args, scene, config, report):
    """Structural retained-storage plan of the configured Cartesian FMM (verified equal to
    the constructed tensors at small sizes); no quadrature, expansions or tables are built."""
    from witwin.maxwell.mom.runtime import compile_surfaces
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.cluster import build_support_octree
    from witwin.maxwell.mom.fmm_capacity import plan_multipole_storage
    start = time.perf_counter()
    system, exterior = compile_surfaces(scene, args.frequency, config)
    provider = ScalarSystemBlocks(system, exterior, config.integration)
    fmm = config.fmm
    lower, upper = provider.support_boxes()
    trees = (build_support_octree(lower, upper, leaf_size=fmm.leaf_size),
             *(build_support_octree(lower[b.edge_offset:b.edge_offset+b.basis.n_edges],
                                    upper[b.edge_offset:b.edge_offset+b.basis.n_edges], leaf_size=fmm.leaf_size)
               for b in system.bodies))
    plan = plan_multipole_storage(trees, order=fmm.order, separation_ratio=fmm.separation_ratio,
        budget_bytes=1 << 62, table_storage=fmm.table_storage, shift_storage=fmm.shift_storage,
        near_storage=fmm.near_storage)
    report.update(status='planned', capacity_plan={'categories': plan['categories'],
                  'planned_bytes': plan['planned_bytes'], 'version': plan['version']},
                  plan_seconds=time.perf_counter()-start, prepare_seconds=0., prepare_retained_bytes=0,
                  stages={}, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  tree_leaves=[int((t.children < 0).all(1).sum()) for t in trees])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps({'status': 'planned', 'faces': report['faces'], 'planned_bytes': plan['planned_bytes'],
                      'plan_seconds': report['plan_seconds']}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shape', choices=('sphere', 'cube'), default='sphere')
    parser.add_argument('--size', type=float, default=1., help='diameter or edge in exterior wavelengths')
    parser.add_argument('--frequency-mesh', type=int, default=9,
                        help='sphere: icosahedral edge frequency (20 n^2 faces); cube: divisions per edge')
    parser.add_argument('--eps', type=float, default=4.)
    parser.add_argument('--frequency', type=float, default=3e8)
    parser.add_argument('--backend', choices=('fmm', 'directional'), default='fmm')
    parser.add_argument('--config', choices=('default', 'budgeted', 'compact'), default='default')
    parser.add_argument('--fmm-order', type=int, default=6)
    parser.add_argument('--leaf-size', type=int, default=32)
    parser.add_argument('--separation-ratio', type=float, default=.5)
    parser.add_argument('--budget-gb', type=float, default=12.)
    parser.add_argument('--leaf-factors', action='store_true')
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--singular-order', type=int, default=4)
    parser.add_argument('--solve', action='store_true')
    parser.add_argument('--skip-rhs', action='store_true',
                        help='replace RHS lifting by zeros (operator memory/time only; incompatible with --solve)')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--restart', type=int, default=500)
    parser.add_argument('--memory-fraction', type=float, default=0.)
    parser.add_argument('--plan-only', action='store_true',
                        help='only build geometry/trees and the structural retained-storage plan (no integration)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    wavelength = C0/args.frequency
    geometry, faces = build_geometry(args, wavelength)
    scene = mw.Scene(structures=[mw.Structure(geometry, mw.Material(eps_r=args.eps))], sources=[mw.PlaneWave()])
    config = build_config(args)
    report = {'schema': 'mom_fmm_prepare_profile_v1',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'faces': faces, 'edges': 3*faces//2, 'unknowns': 3*faces,
              'device': torch.cuda.get_device_name(0)}
    if args.plan_only:
        return plan_only(args, scene, config, report)
    if args.skip_rhs and args.solve:
        raise SystemExit('--skip-rhs cannot be combined with --solve')
    tracer = StageTracer()
    install(tracer)
    if args.skip_rhs:
        from witwin.maxwell.mom import hierarchical_runtime as hr
        # Each runtime class binds build_rhs as its own attribute, so patch every one.
        for cls in (hr.HierarchicalSystemOperator, hr.HierarchicalH2SystemOperator, hr.DirectH2SystemOperator,
                    hr.MultipoleSystemOperator, hr.DirectionalSystemOperator):
            tracer.patches.append((cls, 'build_rhs', cls.__dict__['build_rhs']))
            cls.build_rhs = lambda self: torch.zeros(self.shape[0], device=self.device, dtype=self.dtype)
        report['rhs_skipped'] = True
    profiler = cProfile.Profile()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    prepared = None
    try:
        profiler.enable()
        with tracer.stage('prepare_total'):
            prepared = mw.Simulation.mom(scene, frequency=args.frequency, config=config).prepare()
        profiler.disable()
        report['status'] = 'prepared'
    except Exception as error:  # Profiling records the failure instead of hiding it.
        profiler.disable()
        report['status'] = 'prepare_failed'
        report['error'] = f'{type(error).__name__}: {error}'[:4000]
        report['traceback'] = traceback.format_exc()[-8000:]
    report['prepare_seconds'] = time.perf_counter()-start
    report['prepare_retained_bytes'] = torch.cuda.memory_allocated()-base
    if prepared is not None:
        handle = prepared.solver.handle
        actions = getattr(handle, 'actions', None)
        if actions is not None and hasattr(actions, 'capacity_plan'):
            plan = actions.capacity_plan
            report['capacity_plan'] = {'categories': plan['categories'], 'planned_bytes': plan['planned_bytes'],
                                       'version': plan['version']}
            report['order_plan'] = actions.order_plan
            report['regions'] = []
            for label, region in (('exterior', actions.exterior), *(('interior', r) for r in actions.interiors)):
                record = {'region': label, 'far_pairs': len(region.far), 'table_bytes': region.table_bytes,
                          'near_bytes': 0 if region.near is None else region.near.receipt().get('retained_bytes'),
                          'order': region.test.order, 'tree_nodes': region.test.tree.node_count,
                          'leaves': int((region.test.tree.children < 0).all(1).sum())}
                record['unique_m2l_owners'] = len({id(c) for _, _, c in region.far})
                report['regions'].append(record)
        elif actions is not None and hasattr(actions, 'receipt'):
            with contextlib.suppress(Exception):
                receipt = actions.receipt()
                report['directional_receipt_keys'] = sorted(receipt)[:50]
        report['riesz_bytes'] = sum(t.numel()*t.element_size() for t in handle.metric.tensors())
        report['rhs_bytes'] = prepared.solver.rhs.numel()*prepared.solver.rhs.element_size()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(40)
    report['host_profile_cumulative'] = stream.getvalue()[-20000:]
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('tottime').print_stats(25)
    report['host_profile_tottime'] = stream.getvalue()[-12000:]
    if prepared is not None and args.solve:
        before = torch.cuda.memory_allocated()
        try:
            with tracer.stage('solve'):
                result = prepared.run()
            stats = result.stats()
            report['solve'] = {k: stats.get(k) for k in ('converged', 'reason', 'iterations', 'matvecs',
                                                          'relative_residual', 'solve_seconds', 'setup_seconds')}
            report['solve']['workspace_above_prepared_bytes'] = tracer.stages['solve']['transient_peak_bytes']
            report['solve']['gmres_basis_bytes_model'] = (min(args.restart, args.maxiter)+1)*report['unknowns']*16
            del before
        except Exception as error:
            report['solve_error'] = f'{type(error).__name__}: {error}'[:4000]
            report['solve_traceback'] = traceback.format_exc()[-8000:]
    tracer.restore()
    report['stages'] = tracer.stages
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
    summary = {k: report.get(k) for k in ('status', 'faces', 'prepare_seconds', 'error')}
    summary['stages'] = {k: (round(v['seconds'], 2), round(v['retained_bytes']/1e6, 1)) for k, v in tracer.stages.items()}
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

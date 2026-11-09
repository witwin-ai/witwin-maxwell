"""L-1 classical baseline: multi-body and sharp-edged scalar PMCHWT scenes.

Each case builds one Scene, prepares it once through Scene -> Simulation ->
prepare(), and runs the baseline through ``prepared.run()``. Further
preconditioners reuse the same prepared operator and RHS (operator setup is
measured once) and are bound with the same function the runtime uses. Every
solve stops on the true unpreconditioned residual. The JSON report records
iterations, operator actions, setup/solve/query seconds, per-phase CUDA peaks,
current and far-field errors against an optional reference solve, the optical
theorem defect, and isolated-sphere Mie errors where they apply.

Example (dense tier, two spheres, 3 lambda extent):
    python -m benchmark.mom_large_scale_l1 --family spheres --bodies 2 \
        --extent 3 --body-size 1 --backend dense --output l1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.preconditioning import bind_right_preconditioner

C0 = 299792458.0
FAMILIES = ('spheres', 'boxes', 'cylinders', 'mixed')
BACKENDS = ('dense', 'hmatrix', 'h2', 'direct_h2', 'fmm', 'directional')


# ---------------------------------------------------------------- geometry

def _box_surface(size, h, center):
    """Structured closed box triangulation, outward oriented, edge length <= h."""
    size = np.asarray(size, float)
    counts = np.maximum(1, np.ceil(size/h).astype(int))
    vertices, index, faces = [], {}, []

    def vertex(p):
        key = tuple(np.round(p/(h*1e-6)).astype(np.int64))
        if key not in index:
            index[key] = len(vertices)
            vertices.append(p)
        return index[key]

    lower = -size/2
    for axis in range(3):
        u, v = (axis+1) % 3, (axis+2) % 3
        for side in (0, 1):
            nu, nv = counts[u], counts[v]
            grid = np.empty((nu+1, nv+1), dtype=np.int64)
            for i in range(nu+1):
                for j in range(nv+1):
                    p = lower.copy()
                    p[axis] += side*size[axis]
                    p[u] += size[u]*i/nu
                    p[v] += size[v]*j/nv
                    grid[i, j] = vertex(p)
            for i in range(nu):
                for j in range(nv):
                    a, b, c, d = grid[i, j], grid[i+1, j], grid[i+1, j+1], grid[i, j+1]
                    # (u, v, axis) is right handed; side 1 faces +axis.
                    quad = ((a, b, c), (a, c, d)) if side else ((a, c, b), (a, d, c))
                    faces.extend(quad)
    return np.asarray(vertices)+np.asarray(center), np.asarray(faces)


def _ring_zip(inner, outer, angles_inner, angles_outer, flip):
    """Triangulate between two closed counter-clockwise rings that both start at angle 0."""
    a, b = len(inner), len(outer)
    if a == 1:
        faces = [(inner[0], outer[j], outer[(j+1) % b]) for j in range(b)]
        return [f[::-1] for f in faces] if flip else faces

    def angle(values, k):
        return values[k % len(values)]+(2*math.pi if k >= len(values) else 0.)
    faces, i, j = [], 0, 0
    while i < a or j < b:
        if j < b and (i == a or angle(angles_outer, j+1) <= angle(angles_inner, i+1)):
            tri = (inner[i % a], outer[j], outer[(j+1) % b])
            j += 1
        else:
            tri = (inner[i], outer[j % b], inner[(i+1) % a])
            i += 1
        faces.append(tri[::-1] if flip else tri)
    return faces


def _cylinder_surface(radius, height, h, center):
    """Closed z-axis cylinder: structured side wall and ring-zipped caps."""
    n_side = max(8, math.ceil(2*math.pi*radius/h))
    n_z = max(1, math.ceil(height/h))
    n_rings = max(1, math.ceil(radius/h))
    vertices, faces = [], []

    def ring(r, z, n):
        start = len(vertices)
        phases = [2*math.pi*k/n for k in range(n)]
        vertices.extend((r*math.cos(t), r*math.sin(t), z) for t in phases)
        return list(range(start, start+n)), phases

    wall = [ring(radius, -height/2+height*k/n_z, n_side) for k in range(n_z+1)]
    for (lo, _), (hi, _) in zip(wall, wall[1:]):
        for k in range(n_side):
            a, b, c, d = lo[k], lo[(k+1) % n_side], hi[(k+1) % n_side], hi[k]
            faces.extend(((a, b, c), (a, c, d)))
    for z, rim, outward_up in ((height/2, wall[-1], True), (-height/2, wall[0], False)):
        rings = []
        for m in range(1, n_rings):
            r = radius*m/n_rings
            rings.append(ring(r, z, max(6, math.ceil(2*math.pi*r/h))))
        rings.append(rim)
        vertices.append((0., 0., z))
        centre = ([len(vertices)-1], [0.])
        previous = centre
        for current in rings:
            # (inner, outer_j, outer_j+1) on counter-clockwise rings faces +z.
            faces.extend(_ring_zip(previous[0], current[0], previous[1], current[1], flip=not outward_up))
            previous = current
    return np.asarray(vertices)+np.asarray(center), np.asarray(faces)


def _sphere_subdivisions(radius, h):
    # Icosphere mean edge is about 1.1*radius/2**s after projection.
    return max(0, math.ceil(math.log2(max(1e-12, 1.1*radius/h))))


def body_specs(args, wavelength):
    """Bodies along x with equal pitch; extent and body size in wavelengths. Host-only."""
    h = wavelength/args.ppw
    size = args.body_size*wavelength
    extent = args.extent*wavelength
    count = args.bodies
    pitch = 0. if count == 1 else (extent-size)/(count-1)
    if count > 1 and pitch-size < args.min_gap*wavelength:
        raise ValueError('Bodies are closer than --min-gap; enlarge --extent or reduce --bodies/--body-size')
    kinds = {'spheres': ['sphere'], 'boxes': ['box'], 'cylinders': ['cylinder'],
             'mixed': ['sphere', 'box', 'cylinder']}[args.family]
    specs = []
    for i in range(count):
        centre = (-(extent-size)/2+i*pitch if count > 1 else 0., 0., 0.)
        spec = {'kind': kinds[i % len(kinds)], 'centre_m': centre, 'size_m': size, 'eps_r': args.eps[i % len(args.eps)]}
        if spec['kind'] == 'sphere':
            spec['subdivisions'] = _sphere_subdivisions(size/2, h)
            spec['faces'] = 20*4**spec['subdivisions']
        else:
            vertices, faces = (_box_surface((size, size, size*args.box_aspect), h, centre) if spec['kind'] == 'box'
                               else _cylinder_surface(size/2, size, h, centre))
            spec.update(vertices=vertices, faces_array=faces, faces=len(faces))
        specs.append(spec)
    # One MoMConfig.subdivisions applies to every sphere; use the finest.
    subdivisions = max((p['subdivisions'] for p in specs if p['kind'] == 'sphere'), default=0)
    for spec in specs:
        if spec['kind'] == 'sphere':
            spec['subdivisions'], spec['faces'] = subdivisions, 20*4**subdivisions
    return specs, subdivisions


def build_scene(specs):
    structures = []
    for i, spec in enumerate(specs):
        if spec['kind'] == 'sphere':
            geometry = mw.Sphere(radius=spec['size_m']/2, position=spec['centre_m'])
        else:
            geometry = mw.Mesh(torch.tensor(spec['vertices']), torch.tensor(spec['faces_array']),
                               recenter=False, fill_mode='solid')
        structures.append(mw.Structure(geometry, mw.Material(eps_r=spec['eps_r']), name=f'body{i}'))
    bodies = [{k: spec[k] for k in ('kind', 'centre_m', 'size_m', 'eps_r', 'faces')} for spec in specs]
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()]), bodies


def estimate(args):
    """Host-only size and memory estimate; no CUDA context is created."""
    wavelength = C0/args.frequency
    specs, subdivisions = body_specs(args, wavelength)
    faces = sum(p['faces'] for p in specs)
    edges = 3*faces//2
    n = 2*edges
    gib = 1024**3
    restart = args.restart
    pair_checks = sum(a['faces']*b['faces'] for i, a in enumerate(specs) for b in specs[i+1:])
    return {'faces': faces, 'edges': edges, 'unknowns': n, 'subdivisions': subdivisions,
            'per_body_faces': [p['faces'] for p in specs],
            'interior_ppw': [args.ppw/abs(complex(p['eps_r']))**.5 for p in specs],
            'dense_matrix_gib': 16*n*n/gib,
            'gmres_basis_gib': 16*n*(restart+1)/gib, 'fgmres_basis_gib': 16*n*(2*restart+1)/gib,
            'cartesian_fmm_owned_gib_range': [150e3*edges/gib, 550e3*edges/gib],
            'cartesian_fmm_estimate_basis': 'measured 150-175 kB/edge (3840 edges, two small spheres) to '
                                            '550 kB/edge (27000 edges, resonant sphere, minimum p6)',
            'leaf_block_lu_gib': (2*args.leaf_size)**2*16*(edges/args.leaf_size)/gib,
            'body_pair_separation_face_pairs': pair_checks,
            'body_pair_separation_python_batches': pair_checks//4096}


# ---------------------------------------------------------------- configuration

def build_config(args, subdivisions):
    integration = IntegrationConfig(order=args.order, singular_order=args.singular_order)
    common = dict(subdivisions=subdivisions, integration=integration)
    rule = dict(rtol=args.rtol, atol=0., maxiter=args.maxiter, restart=args.restart,
                stagnation_window=args.stagnation_window)
    budget = int(args.budget_gb*1024**3)
    if args.backend == 'dense':
        # Row-only scaling mixes the J (A/m) and M (V/m) test rows badly; the
        # balanced row/column scaling is the dense diagonal baseline.
        return mw.MoMConfig(**common, riesz_storage='compact_rwg_v1',
                            stopping=StoppingRule(method='gmres', preconditioner='row_column_norm', **rule))
    tolerances = dict(solve_dual_tolerance=args.admission_rtol, original_sample_tolerance=args.admission_rtol)
    if args.backend in ('hmatrix', 'h2'):
        extra = {'h2': mw.H2Config(retained_budget_bytes=budget, **tolerances)} if args.backend == 'h2' else {}
        return mw.MoMConfig(**common, operator_backend=args.backend, riesz_storage='local_rwg_v1',
                            hierarchical=mw.HMatrixConfig(leaf_size=args.leaf_size, tolerance=args.compression_tol,
                                                          retained_budget_bytes=budget, **tolerances),
                            stopping=StoppingRule(method='gmres', preconditioner='none', **rule), **extra)
    if args.backend == 'direct_h2':
        return mw.MoMConfig(**common, operator_backend='direct_h2', riesz_storage='local_rwg_v1',
                            direct_h2=mw.DirectH2Config(order=args.fmm_order, leaf_size=args.leaf_size,
                                                        action_retained_budget_bytes=budget, **tolerances),
                            stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', **rule))
    if args.backend == 'fmm':
        fmm = mw.FMMConfig(order=args.fmm_order, table_storage='compact_shared_displacement_v1',
                           shift_storage='separable_v1', near_storage='scalar_reciprocal_dk_v1',
                           leaf_size=args.leaf_size, separation_ratio=args.separation_ratio,
                           action_retained_budget_bytes=budget, table_budget_bytes=budget//2,
                           near_budget_bytes=budget//2, projection_budget_bytes=budget//2, **tolerances)
    else:
        fmm = mw.DirectionalFMMConfig(near_storage='scalar_reciprocal_dk_v1', regional_near_budget_bytes=budget//2,
                                      regional_table_budget_bytes=budget//2, **tolerances)
    return mw.MoMConfig(**common, operator_backend='fmm', fmm=fmm, riesz_storage='local_rwg_v1',
                        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', **rule))


def preconditioner_factory(name):
    """'none' | 'diag' (backend scaling, plain GMRES) | 'leaf:<edges>' | 'body'."""
    if name == 'diag':
        return None
    if name == 'none':
        return mw.IdentityRightPreconditioner()
    if name.startswith('leaf:'):
        # Dense handles build their support tree here, so the leaf size equals
        # the block bound; hierarchical handles keep their own tree leaves.
        edges = int(name.split(':')[1])
        return mw.LeafBlockRightPreconditioner(edges, budget_bytes=4*1024**3, leaf_size=edges)
    if name == 'body':
        def factory(context):
            blocks = []
            for i, body in enumerate(context.system.bodies):
                edges = torch.arange(body.edge_offset, body.edge_offset+body.basis.n_edges, device=context.device)
                index = context.coefficient_indices(edges)
                blocks.append((index, torch.linalg.lu_factor(context.principal_block(edges))))

            def apply(residual):
                result = torch.empty_like(residual)
                for index, (factor, pivot) in blocks:
                    result[index] = torch.linalg.lu_solve(factor, pivot, residual[index, None])[:, 0]
                return result
            return apply
        return mw.FunctionRightPreconditioner(factory, name='body_block_lu')
    raise ValueError(f'Unknown preconditioner {name}')


# ---------------------------------------------------------------- measurement

def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def _phase_reset():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _peak():
    return {'allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.max_memory_reserved()}


def angular_rule(n_theta, device, dtype):
    mu, w = np.polynomial.legendre.leggauss(n_theta)
    n_phi = 2*n_theta
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    theta = np.arccos(mu)
    directions = np.stack([np.outer(np.sin(theta), np.cos(phi)), np.outer(np.sin(theta), np.sin(phi)),
                           np.outer(np.cos(theta), np.ones(n_phi))], -1).reshape(-1, 3)
    weights = np.repeat(w, n_phi)*2*np.pi/n_phi
    return (torch.tensor(directions, device=device, dtype=dtype), torch.tensor(weights, device=device, dtype=dtype))


def _summary(log):
    keys = ('converged', 'reason', 'iterations', 'matvecs', 'restarts', 'relative_residual', 'absolute_residual',
            'threshold', 'setup_seconds', 'solve_seconds', 'total_seconds', 'krylov_method',
            'preconditioner_applications', 'preconditioner_seconds', 'inner_converged')
    out = {k: log[k] for k in keys if k in log}
    if 'error_budget' in log:
        budget = log['error_budget']
        out['error_budget'] = {k: budget.get(k) for k in ('checked', 'compressed_dual_pass', 'original_sampled_pass')}
    out['residual_history'] = [entry['relative'] for entry in log.get('residual_history', [])]
    return out


def mie_errors(bodies, handle, far, directions, wavelength):
    """Isolated-sphere Mie far fields; a reference only for a single sphere."""
    from benchmark.sie_reference import sphere_far_field
    k = 2*math.pi/wavelength
    wave = handle.incident_carrier.waves[0]
    if tuple(wave.direction) != (0., 0., 1.) or tuple(wave.polarization) != (1., 0., 0.):
        return None
    d = directions.cpu().numpy()
    total = np.zeros((len(d), 3), complex)
    for body in bodies:
        if body['kind'] != 'sphere':
            return None
        c = np.asarray(body['centre_m'])
        shift = np.exp(1j*k*((np.array([0., 0., 1.])-d)@c))
        total += shift[:, None]*sphere_far_field(body['size_m']/2, k, complex(body['eps_r']), d)
    numeric = far.cpu().numpy()
    error = float(np.linalg.norm(numeric-total)/np.linalg.norm(total))
    if len(bodies) == 1:
        return {'mie_far_field_relative_error': error}
    return {'independent_mie_sum_relative_difference': error,
            'note': 'multiple scattering strength diagnostic, not a reference'}


def run_case(args):
    wavelength = C0/args.frequency
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    specs, subdivisions = body_specs(args, wavelength)
    scene, bodies = build_scene(specs)
    config = build_config(args, subdivisions)
    record = {'family': args.family, 'bodies': bodies, 'extent_lambda': args.extent,
              'body_size_lambda': args.body_size, 'ppw_target': args.ppw, 'frequency_hz': args.frequency,
              'backend': args.backend, 'subdivisions': subdivisions, 'rtol': args.rtol,
              'maxiter': args.maxiter, 'restart': args.restart}
    _phase_reset()
    start = _sync()
    prepared = mw.Simulation.mom(scene, frequency=args.frequency, config=config).prepare()
    record['prepare_seconds'] = _sync()-start
    record['prepare_peak_bytes'] = _peak()
    runtime = prepared.solver
    handle = runtime.handle
    system = handle.system
    lengths = torch.cat([b.basis.edge_lengths for b in system.bodies])
    record.update(unknowns=handle.shape[0], edges=system.total_edges, faces=system.total_faces,
                  mean_edge_lambda=float(lengths.mean())/wavelength, max_edge_lambda=float(lengths.max())/wavelength,
                  pair_gaps_lambda=[(a, b, g/wavelength) for a, b, g in system.pair_gaps],
                  assembly_seconds=handle.metadata.get('assembly_seconds'))
    directions, weights = angular_rule(args.far_theta or max(16, math.ceil(math.pi*args.extent)+12),
                                       system.device, system.dtype)

    _phase_reset()
    start = _sync()
    baseline = prepared.run()
    baseline_seconds = _sync()-start
    rhs = runtime.rhs
    runs = {'diag': {**_summary(baseline.stats()), 'wall_seconds': baseline_seconds, 'peak_bytes': _peak(),
                     'preconditioner_setup_seconds': 0.}}
    solutions = {'diag': baseline.surface_currents.scattered_coefficients}
    for name in args.preconditioners:
        if name == 'diag':
            continue
        _phase_reset()
        bound, receipt = bind_right_preconditioner(preconditioner_factory(name), handle, args.frequency)
        start = _sync()
        if args.backend == 'dense':
            x, log = solve_reference(handle, rhs, config.stopping, right_preconditioner=bound.apply)
        else:
            x, log = handle.solve(rhs, config.stopping, right_preconditioner=bound.apply)
        runs[name] = {**_summary(log), 'wall_seconds': _sync()-start, 'peak_bytes': _peak(),
                      'preconditioner_setup_seconds': receipt['setup_seconds'], 'receipt': receipt}
        solutions[name] = x
        del bound
    reference = None
    if args.reference == 'lu' and args.backend == 'dense':
        _phase_reset()
        start = _sync()
        reference, log = solve_reference(handle, rhs, StoppingRule(method='direct', rtol=1e-6, atol=0.))
        record['reference'] = {'kind': 'dense_lu_same_mesh', 'seconds': _sync()-start, 'peak_bytes': _peak(),
                               'relative_residual': log['relative_residual']}
        handle._lu = None
        torch.cuda.empty_cache()
    elif args.reference == 'tight':
        stopping = StoppingRule(method='gmres', preconditioner=config.stopping.preconditioner, rtol=1e-10, atol=0.,
                                maxiter=4*args.maxiter, restart=args.restart, stagnation_window=args.stagnation_window)
        bound, _ = bind_right_preconditioner(mw.LeafBlockRightPreconditioner(32, budget_bytes=4*1024**3),
                                             handle, args.frequency)
        solver = (lambda: solve_reference(handle, rhs, stopping, right_preconditioner=bound.apply)) \
            if args.backend == 'dense' else (lambda: handle.solve(rhs, stopping, right_preconditioner=bound.apply))
        reference, log = solver()
        record['reference'] = {'kind': 'tight_fgmres_same_operator', 'relative_residual': log.get('relative_residual'),
                               'converged': log.get('converged')}
    if reference is not None:
        from witwin.maxwell.mom.lifting_fields import far_field_system
        far_reference, _ = far_field_system(handle, reference, directions)
        for name, x in solutions.items():
            far, _ = far_field_system(handle, x, directions)
            runs[name]['current_relative_error'] = float(torch.linalg.vector_norm(x-reference)
                                                          / torch.linalg.vector_norm(reference))
            runs[name]['far_field_relative_error'] = float(torch.linalg.vector_norm(far-far_reference)
                                                            / torch.linalg.vector_norm(far_reference))
    record['runs'] = runs

    # Observables use the reference, else the converged run with the smallest residual.
    from witwin.maxwell.mom.lifting_fields import far_field_system
    from witwin.maxwell.mom.observables import plane_wave_cross_sections
    if reference is not None:
        source, best = reference, 'reference'
    else:
        converged = [k for k, r in runs.items() if r.get('converged')]
        best = min(converged or runs, key=lambda k: runs[k].get('relative_residual') or math.inf)
        source = solutions[best]
    _phase_reset()
    start = _sync()
    far_function = lambda d: far_field_system(handle, source, d)
    far, _ = far_function(directions)
    sections = plane_wave_cross_sections(handle, source, directions, weights, far_field_function=far_function)
    sections = {k: float(v) for k, v in sections.items()}
    sections['solution'] = best
    lossless = all(complex(b['eps_r']).imag == 0 for b in bodies)
    record['observables'] = {**sections, 'far_seconds': _sync()-start,
                             'angular_rule': {'gauss_theta': int(len(weights)**.5/2**.5), 'directions': len(weights)},
                             'optical_theorem_defect': abs(sections['extinction_m2']-sections['scattering_m2'])
                                 / abs(sections['extinction_m2']) if lossless else None}
    record['mie'] = mie_errors(bodies, handle, far, directions, wavelength)

    if args.queries:
        generator = torch.Generator(device='cuda').manual_seed(1729)
        points = torch.randn((args.queries, 3), generator=generator, device=system.device, dtype=system.dtype)
        points = points/torch.linalg.vector_norm(points, dim=-1, keepdim=True)*(0.5*args.extent+args.query_offset)*wavelength
        _phase_reset()
        start = _sync()
        # Direct (default adaptive) evaluation; cost is independent of which solution is decoded.
        baseline.surface_currents.query_fields(points, 'exterior', 'total')
        record['query'] = {'points': args.queries, 'seconds': _sync()-start, 'peak_bytes': _peak(),
                           'radius_lambda': 0.5*args.extent+args.query_offset,
                           'log': {k: v for k, v in handle.last_field_query_log.items() if k != 'last_body_queries'}}
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--family', choices=FAMILIES, default='spheres')
    parser.add_argument('--bodies', type=int, default=2)
    parser.add_argument('--extent', type=float, default=3., help='total x extent in exterior wavelengths')
    parser.add_argument('--body-size', type=float, default=1., help='body diameter/edge in wavelengths')
    parser.add_argument('--box-aspect', type=float, default=1.)
    parser.add_argument('--min-gap', type=float, default=.25, help='minimum body gap in wavelengths')
    parser.add_argument('--eps', type=float, nargs='+', default=[4.0, 2.5])
    parser.add_argument('--ppw', type=float, default=10., help='target points per exterior wavelength')
    parser.add_argument('--frequency', type=float, default=3e8)
    parser.add_argument('--backend', choices=BACKENDS, default='dense')
    parser.add_argument('--preconditioners', nargs='+', default=['diag', 'none', 'leaf:32'])
    parser.add_argument('--reference', choices=('lu', 'tight', 'none'), default='lu')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=1000)
    parser.add_argument('--restart', type=int, default=200)
    parser.add_argument('--stagnation-window', type=int, default=200)
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--singular-order', type=int, default=4)
    parser.add_argument('--leaf-size', type=int, default=32)
    parser.add_argument('--fmm-order', type=int, default=8)
    parser.add_argument('--separation-ratio', type=float, default=.5)
    parser.add_argument('--compression-tol', type=float, default=1e-6)
    parser.add_argument('--admission-rtol', type=float, default=1e-5)
    parser.add_argument('--budget-gb', type=float, default=6.)
    parser.add_argument('--queries', type=int, default=1000)
    parser.add_argument('--query-offset', type=float, default=1., help='query shell radius beyond extent/2, wavelengths')
    parser.add_argument('--far-theta', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=0.)
    parser.add_argument('--dry-run', action='store_true', help='host-only size/memory estimate, no CUDA')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.dry_run:
        record = {'schema': 'mom_large_scale_l1_estimate_v1', 'arguments': {k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, 'estimate': estimate(args)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record['estimate']))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU backend is substituted')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', 'witwin', 'benchmark'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    sources = sorted(Path(mw.__file__).parent.joinpath('mom').glob('*.py'))
    report = {'schema': 'mom_large_scale_l1_v1', 'arguments': {k: str(v) if isinstance(v, Path) else v
                                                               for k, v in vars(args).items()},
              'source_revision': revision, 'source_dirty': dirty,
              'mom_source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
              'package_path': str(Path(mw.__file__).parent), 'torch': torch.__version__,
              'torch_cuda': torch.version.cuda, 'python': platform.python_version(),
              'device': torch.cuda.get_device_name(0)}
    start = time.perf_counter()
    try:
        report['case'] = run_case(args)
        report['status'] = 'completed'
    except (torch.cuda.OutOfMemoryError, MemoryError, ValueError, RuntimeError, NotImplementedError) as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'[:4000]
        report['peak_bytes_at_failure'] = _peak()
    report['wall_seconds'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'wall_seconds': report['wall_seconds'],
                      **({'error': report['error']} if 'error' in report else {})}))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

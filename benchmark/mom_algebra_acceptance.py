"""CUDA medium-size algebra, precision and execution-evidence acceptance.

This is a fixed-geometry numerical acceptance driver, not an independent
physical-reference or H2/G0 certificate. Its profiler samples the real kernels
used by assembly and the resident algebra/decoder; it does not record the full
dense preparation trace. Run ``python -m benchmark.mom_algebra_acceptance``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.fields import decode, far_field
from witwin.maxwell.mom.linalg import (
    StoppingRule, apply_A, apply_A_adjoint, apply_test_riesz_inverse,
    correct, residual_loss, solve_reference,
)
from witwin.maxwell.mom.observables import volume_absorption
from witwin.maxwell.mom.operator import IntegrationConfig, _pair_integrals
from witwin.maxwell.mom.quadrature import classify_pairs


def _provenance():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *[root/'witwin/maxwell/mom'/name for name in
        ('runtime.py', 'mesh.py', 'physics.py', 'operator.py', 'quadrature.py', 'linalg.py', 'fields.py', 'observables.py')]]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True, check=False)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=root, text=True, capture_output=True, check=False)
    return {'captured_at_process_start_unix': time.time(),
            'git_commit': revision.stdout.strip() if revision.returncode == 0 else None,
            'git_working_tree_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
            'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'python': platform.python_version(), 'platform': platform.platform(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_home': os.environ.get('CUDA_HOME'),
            'timing_environment': 'other validation jobs may share the GPU; these are not isolated throughput timings'}


def _json_default(value):
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(type(value).__name__)


def _compact_log(value):
    if isinstance(value, dict):
        result = {key: _compact_log(item) for key, item in value.items() if key != 'residual_history'}
        if 'residual_history' in value:
            history = value['residual_history']
            stride = max(1, math.ceil(len(history)/16))
            selected = sorted(set(range(0, len(history), stride)) | {len(history)-1})
            result['residual_history_sample'] = [history[index] for index in selected if index >= 0]
            result['residual_history_total_checks'] = len(history)
            result['residual_history_sampling'] = 'at most 17 uniformly spaced checks including final'
        return result
    if isinstance(value, list):
        return [_compact_log(item) for item in value]
    return value


def _relative(actual, reference):
    return float((torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-30)).detach())


def _timed(device, operation, *, repeats=1):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = operation()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter()-start)/repeats


def _weighted_current_error(actual, reference, handle):
    ne = handle.basis.n_edges
    a = torch.cat((actual[:ne]*abs(handle.exterior.eta), actual[ne:]))
    b = torch.cat((reference[:ne]*abs(handle.exterior.eta), reference[ne:]))
    return _relative(a, b)


def _tensor_record(tensor):
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'device': str(tensor.device),
            'bytes': tensor.numel()*tensor.element_size()}


def _profile_kernels(handle, rhs, reference, points):
    """Bounded representative profiling, with no full dense host trace."""
    from torch.profiler import ProfilerActivity, profile, record_function
    count = len(handle.basis.mesh.faces)
    rows = torch.zeros(count, dtype=torch.long, device=handle.device)
    cols = torch.arange(count, device=handle.device)
    categories = classify_pairs(handle.basis.mesh, rows, cols)
    query = mw.FieldQueryConfig(relative_tolerance=1e-4, max_order=32)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as capture:
        for category in range(5):
            selected = cols[categories == category][:1]
            if not len(selected):
                continue
            with record_function(f'mom_acceptance.integration_class_{category}'):
                order = handle.integration.order if category == 4 else handle.integration.singular_order
                _pair_integrals(handle.basis, rows[:1], selected, handle.interior, order, order, category != 4, category)
        with record_function('mom_acceptance.apply_A_and_adjoint'):
            image = apply_A(handle, reference)
            apply_A_adjoint(handle, image)
        with record_function('mom_acceptance.riesz_inverse'):
            apply_test_riesz_inverse(handle, rhs)
        with record_function('mom_acceptance.gmres_bounded_sample'):
            correct(handle, rhs, torch.zeros_like(rhs), StoppingRule(
                maxiter=3, restart=3, preconditioner='row_column_norm'))
        with record_function('mom_acceptance.near_decode'):
            decode(handle, reference, points[:1], 'exterior', 'scattered', query_config=query)
        torch.cuda.synchronize(handle.device)
    events = capture.events()
    cuda_events = [event for event in events if str(event.device_type).endswith('CUDA')]
    kernels = Counter(event.name for event in cuda_events)
    transfers = Counter(event.name for event in events if 'memcpy' in event.name.lower())
    scalar_reads = sum(event.count for event in capture.key_averages() if '_local_scalar_dense' in event.key)
    regions = [event for event in events if event.name.startswith('mom_acceptance.')]
    return {
        'scope': 'five representative actual pair batches plus A/A*, Riesz, three GMRES steps and one near query',
        'full_dense_setup_profiled': False,
        'cuda_event_count': len(cuda_events),
        'cuda_device_indices': sorted({event.device_index for event in cuda_events}),
        'cuda_kernel_counts_top_20': kernels.most_common(20),
        'transfer_event_counts': dict(transfers),
        'scalar_status_read_count': scalar_reads,
        'regions': [{'name': event.name, 'cpu_total_us': event.cpu_time_total,
                     'device_total_us': event.device_time_total} for event in regions],
        'interpretation': 'CUDA kernel events verify sampled numerical execution. Host scalar reads are validation/stopping decisions; CPU operator dispatch is not a CPU numerical fallback.',
        'setup_host_work_not_profiled': ['configuration', 'mesh connectivity', 'scene signature'],
        'output_host_transfers': ['compact scalar metrics', 'small decoded arrays for JSON'],
        'large_vectors_or_matrices_copied_to_cpu_for_solving': False,
    }


def run_case(level, dtype, *, profile_kernels=False, correction_guesses=True, maxiter=1000):
    device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats(device)
    initial_allocated = torch.cuda.memory_allocated(device)
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.004))],
                     sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8, amplitude=1., phase=.23))], device=device)
    integration = IntegrationConfig(order=4, singular_order=6, pair_batch=16, near_tolerance=1e-4, near_max_order=16)
    query = mw.FieldQueryConfig(relative_tolerance=2e-4, absolute_tolerance=1e-8, max_order=48)
    config = mw.MoMConfig(subdivisions=level, dtype=dtype, integration=integration, query_config=query)
    print(json.dumps({'stage': 'preparation_started', 'level': level, 'dtype': str(dtype)}), flush=True)
    prepared, preparation_seconds = _timed(device, lambda: mw.Simulation.mom(scene, frequency=3e8, config=config).prepare())
    print(json.dumps({'stage': 'preparation_finished', 'level': level, 'seconds': preparation_seconds}), flush=True)
    runtime = prepared.solver
    handle, rhs = runtime.handle, runtime.rhs
    # Larger decoder pair batches change memory scheduling only, not integration
    # orders, tolerance, physical q, assembled A or its adjoint.
    handle.integration = replace(handle.integration, pair_batch=256)
    tolerance = 1e-8 if dtype == torch.complex128 else 2e-5
    reference, direct_log = solve_reference(handle, rhs, StoppingRule(method='direct', rtol=tolerance))
    if not direct_log['converged']:
        raise RuntimeError(f'Direct algebraic reference failed: {_compact_log(direct_log)}')
    generator = torch.Generator(device=device).manual_seed(51091)
    vector = torch.randn(reference.shape, device=device, dtype=dtype, generator=generator)
    other = torch.randn(reference.shape, device=device, dtype=dtype, generator=generator)
    image, apply_seconds = _timed(device, lambda: apply_A(handle, vector), repeats=8)
    adjoint, adjoint_seconds = _timed(device, lambda: apply_A_adjoint(handle, other), repeats=8)
    _, riesz_seconds = _timed(device, lambda: apply_test_riesz_inverse(handle, rhs), repeats=4)
    adjoint_defect = float(abs(torch.vdot(image, other)-torch.vdot(vector, adjoint))/
                          (torch.linalg.vector_norm(image)*torch.linalg.vector_norm(other)).clamp_min(1e-30))
    trainable = vector.detach().requires_grad_(True)
    gradient, = torch.autograd.grad(residual_loss(handle, trainable, rhs), trainable)
    expected = apply_A_adjoint(handle, apply_test_riesz_inverse(handle, apply_A(handle, trainable)-rhs))
    gradient_error = _relative(gradient, expected)
    guesses = {'zero': torch.zeros_like(reference), 'random': vector,
               'perturbed_reference': reference+.1*torch.linalg.vector_norm(reference)/math.sqrt(reference.numel())*other}
    corrections = {}
    if correction_guesses:
        for name, initial in guesses.items():
            solution, log = correct(handle, rhs, initial, StoppingRule(rtol=tolerance,
                maxiter=maxiter, restart=min(160, len(rhs)), stagnation_window=80,
                preconditioner='row_column_norm', fallback='direct'))
            corrections[name] = {**_compact_log(log), 'physical_residual_rechecked':
                float(torch.linalg.vector_norm(apply_A(handle, solution)-rhs)/torch.linalg.vector_norm(rhs)),
                'scaled_current_error_vs_LU': _weighted_current_error(solution, reference, handle)}
            print(json.dumps({'stage': 'correction_finished', 'level': level, 'guess': name,
                              'iterations': log['iterations'], 'fallback_used': log['fallback_used']}), flush=True)
    real_dtype = handle.basis.mesh.dtype
    points = torch.tensor([[.107, 0., 0.], [-.108, .012, .003]], dtype=real_dtype, device=device)
    directions = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]], dtype=real_dtype, device=device)
    near, near_seconds = _timed(device, lambda: decode(handle, reference, points, 'exterior', 'scattered', query_config=query))
    far, far_seconds = _timed(device, lambda: far_field(handle, reference, directions))
    near_log = dict(handle.last_field_query_log)
    absorption, absorption_seconds = None, None
    if level == 2:
        triangles = handle.basis.mesh.triangles
        volume_points = triangles.sum(1)/4
        volume_weights = (triangles[:, 0]*torch.linalg.cross(triangles[:, 1], triangles[:, 2])).sum(-1)/6
        absorption, absorption_seconds = _timed(device, lambda: volume_absorption(
            handle, reference, volume_points, volume_weights, query_config=query))
    print(json.dumps({'stage': 'decoders_finished', 'level': level}), flush=True)
    fingerprint = hashlib.sha256(json.dumps(runtime.scene_signature, default=_json_default, sort_keys=True).encode()).hexdigest()
    record = {
        'level': level, 'edges': handle.basis.n_edges, 'dtype': str(dtype), 'device': str(device),
        'scene_fingerprint_sha256': fingerprint, 'scene_signature': runtime.scene_signature,
        'integration': asdict(integration), 'decoder_pair_batch': handle.integration.pair_batch,
        'near_query_config': asdict(query), 'preparation_seconds': preparation_seconds,
        'assembly_metadata': _compact_log(handle.metadata), 'direct_solve': _compact_log(direct_log),
        'reference_factor_cache_available_to_fallback': True,
        'reference_factorization_cost_accounting': 'charged in direct_solve; fallback_log explicitly reports reuse of this same-q cache',
        'corrections': corrections, 'timings_seconds': {'apply_A': apply_seconds,
            'apply_A_adjoint': adjoint_seconds, 'riesz_inverse': riesz_seconds,
            'near_decode': near_seconds, 'far_decode': far_seconds, 'volume_absorption': absorption_seconds},
        'adjoint_relative_inner_product_defect': adjoint_defect, 'fixed_q_gradient_relative_error': gradient_error,
        'scaled_current_error_norm': 'Euclidean coefficient norm after scaling J by exterior impedance; not a trace-space norm',
        'near_query_log': near_log,
        'native_tensors': {name: _tensor_record(tensor) for name, tensor in {
            'matrix': handle.matrix, 'test_riesz': handle.test_riesz, 'rhs': rhs,
            'coefficients': reference, 'near_E': near[0], 'far_E': far[0]}.items()},
        'absorption_W': None if absorption is None else float(absorption),
        'absorption_quadrature': 'one centroid per mesh tetrahedron; precision diagnostic only, volume convergence not certified' if absorption is not None else 'evaluated at level 2 only',
        'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(device),
        'initial_allocated_cuda_bytes': initial_allocated,
        'peak_increment_cuda_bytes': torch.cuda.max_memory_allocated(device)-initial_allocated,
        'peak_memory_scope': 'this case after explicit process-allocator reset, including preparation',
        'reference_precision_rtol': tolerance,
    }
    if profile_kernels:
        record['profiler'] = _profile_kernels(handle, rhs, reference, points)
        if record['profiler']['cuda_event_count'] == 0:
            raise RuntimeError('Profiler captured no CUDA events; native execution evidence is unavailable')
    arrays = {'current': reference.detach(), 'forward': image.detach(), 'adjoint': adjoint.detach(),
              'near_E': near[0].detach(), 'near_H': near[1].detach(),
              'far_E': far[0].detach(), 'far_H': far[1].detach(), 'absorption': absorption,
              'probe_vector': vector.detach(), 'probe_other': other.detach(), 'matrix': handle.matrix,
              'eta': abs(handle.exterior.eta)}
    return record, arrays


def precision_comparison(low, high):
    # Use exactly the same complex probe values for both precision operators;
    # dtype-specific RNG streams need not produce the same normal samples.
    x, y = high['probe_vector'], high['probe_other']
    metrics = {
        'forward_same_probe': _relative((low['matrix']@x.to(torch.complex64)).to(torch.complex128), high['matrix']@x),
        'adjoint_same_probe': _relative((low['matrix'].mH@y.to(torch.complex64)).to(torch.complex128), high['matrix'].mH@y),
    }
    ne = len(high['current'])//2
    scale = torch.cat((torch.full((ne,), high['eta'], device=x.device), torch.ones(ne, device=x.device)))
    metrics['impedance_scaled_current'] = _relative(low['current'].to(torch.complex128)*scale, high['current']*scale)
    for name in ('near_E', 'near_H', 'far_E', 'far_H'):
        metrics[name] = _relative(low[name].to(torch.complex128), high[name])
    metrics['volume_absorption'] = abs(float(low['absorption']/high['absorption'])-1)
    return {'reference': 'complex128 with same mesh family and quadrature tolerances',
            'relative_differences': metrics, 'independent_physical_accuracy_certified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/mom-algebra-acceptance.json'))
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--maxiter', type=int, default=1000)
    parser.add_argument('--no-profiler', action='store_true')
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU numerical fallback exists')
    report = {'schema_version': 1, 'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'run_provenance': _provenance(),
              'gpu_name': torch.cuda.get_device_name(), 'H2_G0_certified': False, 'cases': [],
              'scope': 'native algebra, actual-operator precision comparison and measured correction costs',
              'host_transfer_policy': 'configuration/topology/I/O and scalar status only; no CPU linear algebra'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint():
        args.output.write_text(json.dumps(report, default=_json_default, indent=2))
    high = None
    for level in args.levels:
        try:
            row, arrays = run_case(level, torch.complex128, profile_kernels=level == 2 and not args.no_profiler, maxiter=args.maxiter)
            row['status'] = 'measured'
            report['cases'].append(row)
            if level == 2:
                high = arrays
            else:
                del arrays
        except Exception as error:
            report['cases'].append({'level': level, 'dtype': 'torch.complex128', 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        checkpoint()
        print(json.dumps({'completed_level': level, 'status': report['cases'][-1]['status']}), flush=True)
    if high is not None:
        try:
            row, low = run_case(2, torch.complex64, correction_guesses=False, maxiter=args.maxiter)
            row['status'] = 'measured'
            report['cases'].append(row)
            report['precision_comparison_480_edges'] = precision_comparison(low, high)
        except Exception as error:
            report['cases'].append({'level': 2, 'dtype': 'torch.complex64', 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
    report['all_requested_cases_executed'] = all(row['status'] == 'measured' for row in report['cases'])
    report['all_reported_corrections_converged'] = all(
        correction['converged'] for row in report['cases'] for correction in row.get('corrections', {}).values())
    checkpoint()
    print(json.dumps({'output': str(args.output), 'cases': len(report['cases']), 'all_requested_cases_executed': report['all_requested_cases_executed']}))
    return 0 if report['all_requested_cases_executed'] and report['all_reported_corrections_converged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

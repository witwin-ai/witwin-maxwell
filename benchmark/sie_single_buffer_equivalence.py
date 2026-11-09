"""Source-bound CUDA assembly equivalence; run this file in a fresh process.

This validation driver contains no solver implementation. Its only provider
substitution is the reviewed historical assemble_weak callable during prepare.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

BASE_COMMIT = 'e95ab35a87e4259f329fbf8b0dfaecbea45eb0c7'
BASE_OPERATOR_SHA256 = 'b6d44b2414fd8f44e4ab73b259cd2bb3befe9ebb07b3a67af712946845f019d9'
CANDIDATE_OPERATOR_SHA256 = 'eb491a09d5858d92acaddd6b79be20943e8fbcda2e7dc9fac0c99f42638b87d4'
SOURCE_SUFFIXES = {'.py', '.cpp', '.cu', '.cuh', '.h', '.so'}
PROTOCOL = {
    'version': 'single-buffer-equivalence-v1',
    'shapes': ['sphere', 'ellipsoid', 'harmonic'],
    'dtypes': ['complex64', 'complex128'],
    'batches': [[7, 19], [23, 61]],
    'subdivisions': 1, 'frequency_hz': 3e8, 'rhs_count': 8,
    'integration': {'order': 4, 'singular_order': 4,
                    'near_tolerance': 1e-4, 'near_max_order': 16},
    'tolerances': {
        'complex64': {'comparison': 3e-5, 'direct_residual': 3e-5},
        'complex128': {'comparison': 3e-12, 'direct_residual': 1e-10}},
    'scope': 'storage equivalence, not reference accuracy or H2',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_map(root):
    root = Path(root).resolve()
    return {str(p.relative_to(root)): {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts}


def verify_provider_pair(candidate, baseline):
    """Reject unreviewed changes, even if both working trees share them."""
    candidate, baseline = Path(candidate).resolve(), Path(baseline).resolve()
    for root in (candidate, baseline):
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if head != BASE_COMMIT:
            raise ValueError('provider HEAD differs from reviewed e95')
    old = source_map(baseline / 'witwin/maxwell')
    new = source_map(candidate / 'witwin/maxwell')
    if set(old) != set(new):
        raise ValueError('provider source inventory differs')
    tracked = subprocess.check_output(['git', '-C', str(baseline), 'ls-tree', '-r', '--name-only', BASE_COMMIT], text=True).splitlines()
    expected = {p.removeprefix('witwin/maxwell/') for p in tracked
                if p.startswith('witwin/maxwell/') and Path(p).suffix in SOURCE_SUFFIXES}
    if set(old) != expected:
        raise ValueError('provider contains untracked or absent scientific sources')
    for name in old:
        blob = subprocess.check_output(['git', '-C', str(baseline), 'show', f'{BASE_COMMIT}:witwin/maxwell/{name}'])
        if digest(blob) != old[name]['sha256']:
            raise ValueError(f'baseline source differs from commit: {name}')
        if name != 'sie/operator.py' and new[name] != old[name]:
            raise ValueError(f'unreviewed candidate dependency: {name}')
    if old['sie/operator.py']['sha256'] != BASE_OPERATOR_SHA256:
        raise ValueError('wrong historical operator')
    if new['sie/operator.py']['sha256'] != CANDIDATE_OPERATOR_SHA256:
        raise ValueError('wrong candidate operator')
    return {'baseline': old, 'candidate': new}


class BoundSources(importlib.abc.MetaPathFinder):
    """Compile archived bytes at first import; never accept preloaded providers."""
    def __init__(self, paths, archive):
        if any(n == 'witwin.core' or n.startswith(('witwin.core.', 'witwin.maxwell')) for n in sys.modules):
            raise ValueError('provider/core already imported; use a fresh process')
        self.files, self.modules = {}, {}
        archive = Path(archive)
        archive.mkdir()
        for p in paths:
            p = Path(p).resolve()
            data = p.read_bytes()
            sha = digest(data)
            target = archive / sha
            if not target.exists():
                target.write_bytes(data)
            self.files[str(p)] = {'sha256': sha, 'bytes': len(data), 'archive': str(target)}

    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == 'witwin.core' or fullname.startswith(('witwin.core.', 'witwin.maxwell'))):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            return spec
        origin = str(Path(spec.origin).resolve())
        if origin not in self.files or not origin.endswith('.py'):
            raise ValueError(f'unarchived imported provider: {fullname}: {origin}')
        owner = self
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                row = owner.files[origin]
                data = Path(row['archive']).read_bytes()
                if digest(data) != row['sha256'] or digest(Path(origin).read_bytes()) != row['sha256']:
                    raise ValueError('source changed before first execution')
                module.__file__ = origin
                exec(compile(data, origin, 'exec'), module.__dict__)
                owner.modules[fullname] = (module, origin)
        spec.loader = Loader()
        return spec

    def verify(self):
        for path, row in self.files.items():
            if digest(Path(path).read_bytes()) != row['sha256'] or digest(Path(row['archive']).read_bytes()) != row['sha256']:
                raise ValueError(f'source/archive changed: {path}')
        for name, (module, _) in self.modules.items():
            if sys.modules.get(name) is not module:
                raise ValueError(f'loaded module replaced: {name}')
        return {'files': self.files,
                'loaded_modules': {name: {'origin': origin, 'sha256': self.files[origin]['sha256']}
                                   for name, (_, origin) in self.modules.items()}}


def load_baseline(path):
    data = Path(path).read_bytes()
    if digest(data) != BASE_OPERATOR_SHA256:
        raise ValueError('baseline operator hash mismatch')
    name = 'witwin.maxwell.sie._reviewed_e95_operator'
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = 'witwin.maxwell.sie'
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


@contextlib.contextmanager
def assembly_binding(runtime, replacement):
    original = runtime.assemble_weak
    runtime.assemble_weak = replacement
    try:
        yield
    finally:
        runtime.assemble_weak = original


def compare_cuda(left, right, tolerance):
    import torch
    if left.device.type != 'cuda' or right.device != left.device or left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('comparison requires matching CUDA tensors')
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError('nonfinite numerical evidence')
    scale = right.abs().max()
    if float(scale) == 0:
        maximum = float(left.abs().max())
        return {'relative_l2': None, 'relative_max': None, 'absolute_max': maximum, 'passed': maximum == 0}
    a, b = left / scale, right / scale
    relative_l2 = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b))
    relative_max = float((a-b).abs().max())
    return {'relative_l2': relative_l2, 'relative_max': relative_max,
            'passed': relative_l2 <= tolerance and relative_max <= tolerance}


def validate_metadata(old, new):
    ignored = {'assembly_seconds'}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        raise ValueError('scientific assembly metadata differs')
    counts = new['triangle_pair_counts']
    if len(counts) != 5 or any(type(v) is not int or v <= 0 for v in counts):
        raise ValueError('every near/touching/far pair category must actually execute')
    return counts


def observe_assembly(assembler, measurements):
    def measured(basis, exterior, interior, config):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        edges = basis.n_edges
        stores = {}
        class MatrixStorageObserver(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                def visit(value):
                    if isinstance(value, torch.Tensor) and value.is_cuda and value.is_complex() and value.numel() == 4*edges*edges:
                        storage = value.untyped_storage()
                        stores.setdefault(storage.data_ptr(), {'bytes': storage.nbytes(), 'first_operation': str(func), 'first_shape': list(value.shape)})
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            visit(item)
                visit(out)
                return out
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with MatrixStorageObserver():
            result = assembler(basis, exterior, interior, config)
        torch.cuda.synchronize()
        measurements.update({'assembly_wall_seconds': time.perf_counter()-started,
            'allocated_before': before, 'allocated_after': torch.cuda.memory_allocated(),
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'matrix_sized_storage_observations': list(stores.values()),
            'observer_scope': 'assembly only; allocator pointers identify buffers, not total allocation events'})
        return result
    return measured


def declared_sources(mw):
    orientations = [((0,0,1),(1,0,0)), ((0,0,1),(0,1,0)),
        ((0,0,-1),(1,0,0)), ((0,0,-1),(0,1,0)), ((1,0,0),(0,1,0)),
        ((1,0,0),(0,0,1)), ((0,1,0),(1,0,0)), ((0,1,0),(0,0,1))]
    return [mw.PlaneWave(direction=d, polarization=p,
        source_time=mw.CW(frequency=3e8, amplitude=1+.1*i, phase=.17*i))
        for i, (d, p) in enumerate(orientations)]


def run_case(shape, dtype_name, batches, baseline, output):
    import numpy as np
    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.sie import runtime, operator
    from witwin.maxwell.sie.mesh import deformed_sphere_mesh
    from witwin.maxwell.sie.linalg import apply_A, apply_A_adjoint, residual_loss, solve_reference, StoppingRule
    dtype = getattr(torch, dtype_name)
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    tol = PROTOCOL['tolerances'][dtype_name]
    position = (.013, -.009, .006)
    if shape == 'sphere':
        geometry = mw.Sphere(radius=.1, position=position, device='cuda')
    elif shape == 'ellipsoid':
        geometry = mw.Ellipsoid(radii=(.14, .09, .07), position=position, device='cuda')
    elif shape == 'harmonic':
        mesh = deformed_sphere_mesh(radius=.1, subdivisions=1, harmonics={(2, 0): .08, (3, 2): .035}, dtype=real)
        geometry = mw.Mesh(mesh.vertices, mesh.faces, position=position, recenter=False, fill_mode='solid', device='cuda')
    else:
        raise ValueError('unknown shape')
    sources = declared_sources(mw)
    scene = mw.Scene(structures=[mw.Structure(geometry, mw.Material(eps_r=2.7, sigma_e=.015))],
                     sources=[sources[0]], device='cuda')
    integration = operator.IntegrationConfig(**PROTOCOL['integration'], pair_batch=batches[0], far_pair_batch=batches[1])
    stopping = StoppingRule(method='direct', rtol=tol['direct_residual'], atol=1e-12, fallback='none')
    config = mw.SIEConfig(subdivisions=1, dtype=dtype, integration=integration, stopping=stopping, riesz_storage='dense')
    prepared, allocation = {}, {}
    for name, assembler in [('old', baseline.assemble_weak), ('new', operator.assemble_weak)]:
        allocation[name] = {}
        with assembly_binding(runtime, observe_assembly(assembler, allocation[name])):
            prepared[name] = mw.Simulation.sie(scene, frequency=3e8, config=config).prepare()
    old, new = prepared['old'].solver.handle, prepared['new'].solver.handle
    if not torch.equal(old.basis.mesh.vertices, new.basis.mesh.vertices) or not torch.equal(old.basis.mesh.faces, new.basis.mesh.faces):
        raise ValueError('old/new physical mesh differs')
    counts = validate_metadata(old.metadata, new.metadata)
    if not new.matrix.is_contiguous():
        raise ValueError('candidate matrix is not contiguous')
    e = old.basis.n_edges
    checks = {f'block_{i}{j}': compare_cuda(new.matrix[i*e:(i+1)*e, j*e:(j+1)*e], old.matrix[i*e:(i+1)*e, j*e:(j+1)*e], tol['comparison']) for i in range(2) for j in range(2)}
    repeat_tolerance = 32*torch.finfo(real).eps
    checks['riesz'] = compare_cuda(new.test_riesz, old.test_riesz, repeat_tolerance)
    rhs = {name: torch.stack([operator.build_rhs(p.solver.handle.basis, p.solver.handle.exterior, [s], order=integration.order) for s in sources]) for name, p in prepared.items()}
    checks['rhs'] = compare_cuda(rhs['new'], rhs['old'], repeat_tolerance)
    for source_index in range(8):
        checks[f'rhs_source_{source_index}'] = compare_cuda(rhs['new'][source_index], rhs['old'][source_index], repeat_tolerance)
    checks['prepared_rhs'] = compare_cuda(prepared['new'].solver.rhs, prepared['old'].solver.rhs, repeat_tolerance)
    for name, expected_buffers in [('old', 2), ('new', 1)]:
        buffers = allocation[name]['matrix_sized_storage_observations']
        expected_bytes = new.matrix.numel()*new.matrix.element_size()
        checks[f'{name}_assembly_matrix_buffers'] = {
            'observed': len(buffers), 'expected': expected_buffers,
            'passed': len(buffers) == expected_buffers and all(b['bytes'] == expected_bytes for b in buffers)}
    torch.manual_seed(73019)
    x = torch.randn(8, 2*e, dtype=dtype, device='cuda')
    y = torch.randn_like(x)
    actions, adjoints, gradients, solutions, residuals, solve_logs = {}, {}, {}, {}, {}, {}
    for name, handle in [('old', old), ('new', new)]:
        actions[name], adjoints[name] = apply_A(handle, x), apply_A_adjoint(handle, y)
        trial = x.clone().requires_grad_()
        gradients[name], = torch.autograd.grad(residual_loss(handle, trial, rhs[name]), trial)
        solutions[name], solve_logs[name] = solve_reference(handle, rhs[name], stopping)
        residuals[name] = torch.linalg.vector_norm(apply_A(handle, solutions[name])-rhs[name], dim=-1) / torch.linalg.vector_norm(rhs[name], dim=-1)
        left = (actions[name].conj()*y).sum(-1)
        right = (x.conj()*adjoints[name]).sum(-1)
        checks[f'{name}_adjoint_identity'] = compare_cuda(left, right, tol['comparison'])
        checks[f'{name}_direct_residual'] = {'per_source': residuals[name].tolist(), 'passed': bool((residuals[name] <= tol['direct_residual']).all()) and solve_logs[name]['converged']}
    for key, data in [('action', actions), ('adjoint', adjoints), ('fixed_current_gradient', gradients), ('direct_solution', solutions)]:
        checks[key] = compare_cuda(data['new'], data['old'], tol['comparison'])
        for source_index in range(8):
            checks[f'{key}_source_{source_index}'] = compare_cuda(data['new'][source_index], data['old'][source_index], tol['comparison'])
    # Save exact tensors before reporting acceptance, including failed comparisons.
    raw = {'vertices': old.basis.mesh.vertices, 'faces': old.basis.mesh.faces, 'x': x, 'y': y}
    for name, handle in [('old', old), ('new', new)]:
        raw.update({f'{name}_{k}': v for k, v in dict(matrix=handle.matrix, rhs=rhs[name],
            action=actions[name], adjoint=adjoints[name], gradient=gradients[name],
            solution=solutions[name], residual=residuals[name]).items()})
    raw_path = Path(output) / 'raw.npz'
    with raw_path.open('xb') as stream:
        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in raw.items()})
    result = {'shape': shape, 'dtype': dtype_name, 'batches': batches, 'rhs_count': 8,
        'checks': checks, 'allocation': allocation, 'triangle_pair_counts': counts,
        'operator_metadata': {'old': old.metadata, 'new': new.metadata}, 'solve_logs': solve_logs,
        'raw_sha256': digest(raw_path.read_bytes()),
        'passed': all(c['passed'] for c in checks.values()), 'H2': 'not_assessed'}
    write_json(Path(output) / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'protocol': PROTOCOL, 'pid': os.getpid(), 'interpreter': sys.executable,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'status': 'failed', 'cases': []}
    bound = None
    try:
        pair = verify_provider_pair(args.candidate, args.baseline)
        write_json(args.output / 'provider-pair.json', pair)
        sys.path.insert(0, str(args.candidate.resolve()))
        core = importlib.util.find_spec('witwin.core')
        if core is None or core.origin is None:
            raise ValueError('installed core unavailable')
        core_root = Path(core.origin).parent
        core_inventory = source_map(core_root)
        paths = [args.candidate/'witwin/maxwell'/p for p in pair['candidate']]
        paths += [core_root/p for p in source_map(core_root)]
        paths += [Path(__file__).resolve(), args.baseline/'witwin/maxwell/sie/operator.py']
        paths += [args.candidate/'tests/sie/test_single_buffer_assembly.py',
                  args.candidate/'docs/assessments/sie_single_buffer_equivalence_protocol.md']
        bound = BoundSources(paths, args.output / 'source')
        sys.meta_path.insert(0, bound)
        import torch
        import witwin.maxwell as mw
        from witwin.maxwell.sie import runtime, operator
        if Path(operator.__file__).resolve() != (args.candidate/'witwin/maxwell/sie/operator.py').resolve():
            raise ValueError('wrong candidate import origin')
        baseline = load_baseline(args.baseline/'witwin/maxwell/sie/operator.py')
        bound.modules[baseline.__name__] = (baseline, str(Path(baseline.__file__).resolve()))
        report['physical_sources'] = [repr(source) for source in declared_sources(mw)]
        report['torch_version'] = torch.__version__
        report['cuda_initialized_before'] = torch.cuda.is_initialized()
        write_json(args.output/'source-before.json', bound.verify())
        if args.preflight:
            if torch.cuda.is_initialized():
                raise ValueError('preflight initialized CUDA')
            report['status'] = 'preflight_passed_no_numerical_execution'
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; no numerical fallback')
            report['device'] = {'name': torch.cuda.get_device_name(), 'total_memory': torch.cuda.get_device_properties(0).total_memory, 'cuda': torch.version.cuda}
            for shape in PROTOCOL['shapes']:
                for dtype in PROTOCOL['dtypes']:
                    for batch in PROTOCOL['batches']:
                        name = f'{shape}-{dtype}-batch{batch[0]}-{batch[1]}'
                        target = args.output/name
                        target.mkdir()
                        print(name, flush=True)
                        bound.verify()
                        result = run_case(shape, dtype, batch, baseline, target)
                        report['cases'].append({'id': name, 'passed': result['passed'], 'result_sha256': digest((target/'result.json').read_bytes())})
                        bound.verify()
            report['status'] = 'passed' if len(report['cases']) == 12 and all(c['passed'] for c in report['cases']) else 'failed'
        verify_provider_pair(args.candidate, args.baseline)
        if source_map(core_root) != core_inventory:
            raise ValueError('actual installed core inventory changed')
        write_json(args.output/'source-after.json', bound.verify())
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        report['invocation_wall_seconds'] = time.perf_counter()-started
        report['H2'] = 'not_assessed'
        write_json(args.output/'execution.json', report)
    print(json.dumps({'status': report['status'], 'failure': report.get('failure')}), flush=True)
    return 0 if report['status'] in ('passed', 'preflight_passed_no_numerical_execution') else 1


if __name__ == '__main__':
    raise SystemExit(main())

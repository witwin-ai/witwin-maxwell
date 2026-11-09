"""Actual PMCHWT multi-source block Krylov acceptance and cost evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule, apply_A, correct, solve_reference
from witwin.maxwell.mom.operator import IntegrationConfig, build_rhs


def encode(value):
    array = value.detach().cpu().numpy()
    return dict(real=array.real.tolist(), imag=array.imag.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subdivisions', type=int, default=2)
    parser.add_argument('--rtol', type=float, default=1e-8,
        help='True residual tolerance; coefficient accuracy is assessed independently')
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root/'witwin/maxwell').rglob('*.py'))+[Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    snapshot = args.output.parent/'source_snapshots'/digest
    for path in sources:
        target = snapshot/path.relative_to(root)
        data = path.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError('Immutable source snapshot changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    rule = StoppingRule(method='block_gmres', preconditioner='row_column_norm',
                        restart=125, maxiter=180, stagnation_window=150, rtol=args.rtol, atol=1e-12)
    integration = IntegrationConfig(order=4, singular_order=4, pair_batch=16, far_pair_batch=1024)
    report = dict(source_sha256=hashes, source_snapshot_directory=str(snapshot),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True).strip()),
        source_scope='all local Maxwell Python sources and this driver; installed dependencies described by environment',
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), dtype='complex128'),
        physical_scene=dict(frequency_hz=3e8, radius_m=.1, epsilon_r=2.25, sigma_S_per_m=.004,
                            exterior='vacuum', convention='exp(-i omega t)'),
        configuration=dict(stopping=asdict(rule), integration=asdict(integration), subdivisions=args.subdivisions),
        coefficient_relative_budget=1e-6,
        scope='actual same-q eight-RHS algebra and cost; no physical accuracy or speedup certificate', cases=[])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report))
    save()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    scene = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.25, sigma_e=.004))],
                     sources=[mw.PlaneWave()])
    prepared = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(
        subdivisions=args.subdivisions, integration=integration)).prepare()
    op = prepared.solver.handle
    report['edges'] = op.basis.n_edges
    torch.cuda.synchronize(); report['prepare_seconds'] = time.perf_counter()-start
    rhs = []
    source_contracts = []
    for index in range(8):
        angle = 2*math.pi*index/8
        direction = (math.sin(angle), 0., math.cos(angle))
        source = mw.PlaneWave(direction=direction, polarization=(0.,1.,0.),
            source_time=mw.CW(frequency=3e8, amplitude=1., phase=index*.17))
        rhs.append(build_rhs(op.basis, op.exterior, [source], order=4))
        source_contracts.append(dict(direction=direction, polarization=[0.,1.,0.], amplitude=1., phase=index*.17))
    b = torch.stack(rhs)
    exact, direct_log = solve_reference(op, b)
    report.update(sources=source_contracts, rhs=encode(b), direct_coefficients=encode(exact), direct_log=direct_log)
    save()
    torch.manual_seed(918)
    guesses = [('zero', torch.zeros_like(b)), ('random', .1*torch.randn_like(b))]
    for name, initial in guesses:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        x, log = correct(op, b, initial, rule)
        torch.cuda.synchronize(); elapsed = time.perf_counter()-begin
        raw = torch.linalg.vector_norm(apply_A(op, x)-b, dim=-1)
        relative = raw/torch.linalg.vector_norm(b, dim=-1)
        coefficient_error = torch.linalg.vector_norm(x-exact, dim=-1)/torch.linalg.vector_norm(exact, dim=-1)
        balanced = torch.cat((torch.full((op.basis.n_edges,), abs(op.exterior.eta), device=op.device, dtype=torch.float64),
                              torch.ones(op.basis.n_edges, device=op.device, dtype=torch.float64)))
        balanced_error = torch.linalg.vector_norm((x-exact)*balanced, dim=-1)/torch.linalg.vector_norm(exact*balanced, dim=-1)
        row = dict(initial_guess=name, initial_coefficients=encode(initial), coefficients=encode(x),
            solver=log, independent_true_relative_residuals=relative.tolist(),
            coefficient_relative_errors=coefficient_error.tolist(), balanced_coefficient_relative_errors=balanced_error.tolist(),
            measured_solve_seconds=elapsed, peak_cuda_allocator_bytes=torch.cuda.max_memory_allocated(),
            accepted=bool(log['converged'] and torch.all(raw<=torch.maximum(
                torch.full_like(raw, rule.atol), rule.rtol*torch.linalg.vector_norm(b, dim=-1)))
                and torch.all(balanced_error<1e-6)))
        report['cases'].append(row); save()
        print(json.dumps(dict(initial_guess=name, accepted=row['accepted'], iterations=log['iterations'],
            matvecs=log['matvecs'], block_products=log['block_operator_products'],
            residual_max=float(relative.max()), coefficient_error_max=float(balanced_error.max()), seconds=elapsed)), flush=True)
    report['accepted'] = bool(direct_log['converged'] and len(report['cases'])==2 and all(row['accepted'] for row in report['cases']))
    save()
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

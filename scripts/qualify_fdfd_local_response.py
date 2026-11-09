"""Independent full-operator qualification and synchronized GPU timing.

Run with CUDA_VISIBLE_DEVICES=1. The reference assembles diagonal changes and
eliminates PEC rows/columns in the full sparse operator, then refactorizes it.
It does not use reduced response equations or Woodbury updates.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import time
import cupy as cp
from cupyx.scipy import sparse
from nvmath.sparse.advanced import DirectSolver, ExecutionCUDA
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdfd.observations import compile_projection


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def relative(predicted, reference):
    return float(torch.linalg.vector_norm(predicted-reference) /
                 torch.linalg.vector_norm(reference).clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=18)
    parser.add_argument('--volume-design', action='store_true')
    args = parser.parse_args()
    if args.trials < 3:
        raise ValueError('At least three trials are required.')
    if not torch.cuda.is_available():
        raise RuntimeError('Qualification requires native CUDA.')
    simulation = runpy.run_path('examples/fdfd/local_response_antenna.py')['build']()
    scene = simulation.scene
    frequency = simulation.frequencies[0]
    cfg = simulation.config.local_response
    if args.volume_design:
        cfg = replace(cfg,design=mw.LocalDesignRegion(bounds=((-0.06,.06),(-0.06,.06),(.02,.08))))
    settings = mw.GMRES(solver_type='direct', precision='double')
    # Prepare a separate full-domain reference before response timing.
    full = mw.Simulation.fdfd(scene, frequency=frequency, solver=settings).prepare().solver
    full._ensure_system_matrix()
    base_matrix = full.A_matrix.copy()
    scale = full._ensure_symmetrization_scale()
    source = full._build_source_vector_yee().copy()
    projections = compile_projection(cfg.observations, full.scene, torch.complex128, frequency)
    torch.cuda.reset_peak_memory_stats()
    free_before = cp.cuda.runtime.memGetInfo()[0]
    response, cold_seconds = timed(lambda: mw.Simulation.fdfd(scene,frequency=frequency,
                                               solver=settings,local_response=cfg).prepare())
    free_after = cp.cuda.runtime.memGetInfo()[0]
    design = response.design_indices
    cp_design = cp.from_dlpack(design)
    fixed_pec = full._ensure_pec_mask().copy()
    n = design.numel()
    # Keep the prescribed current gap open, as in the antenna search example.
    eligible = torch.from_dlpack(cp.flatnonzero(source[cp_design] == 0).copy())

    def reference(delta, pec):
        diagonal = cp.zeros(base_matrix.shape[0], dtype=base_matrix.dtype)
        diagonal[cp_design] = cp.from_dlpack(delta)
        matrix = (base_matrix + sparse.diags(diagonal)).tocoo()
        mask = fixed_pec.copy()
        mask[cp_design] = cp.from_dlpack(pec)
        keep = ~(mask[matrix.row] | mask[matrix.col])
        constrained = cp.flatnonzero(mask).astype(cp.int32)
        matrix = sparse.coo_matrix((cp.concatenate((matrix.data[keep],cp.ones(constrained.size,dtype=matrix.dtype))),
                                   (cp.concatenate((matrix.row[keep],constrained)),cp.concatenate((matrix.col[keep],constrained)))),
                                  shape=matrix.shape).tocsr()
        rhs = source.copy()
        rhs[mask] = 0
        direct = DirectSolver(matrix,rhs,execution=ExecutionCUDA())
        try:
            direct.plan()
            direct.factorize()
            solution = direct.solve().copy()
        finally:
            direct.free()
        residual = float(cp.linalg.norm((matrix@solution-rhs)/scale)/cp.linalg.norm(rhs/scale))
        field = torch.from_dlpack((solution/scale).copy())
        observed = torch.sparse.mm(projections,field[:,None])[:,0]
        return field, observed, residual

    generator = torch.Generator(device='cuda').manual_seed(73021)
    delta = torch.zeros(n, device='cuda', dtype=torch.complex128)
    pec = torch.zeros(n, device='cuda', dtype=torch.bool)
    # Warm both numerical library paths; initialization is reported separately.
    _, reference_warmup = timed(lambda: reference(delta,pec))
    response.trial(delta=delta,pec=pec)
    records = []
    for index in range(args.trials):
        rank = (1,4,12)[index%3]
        selected = eligible[torch.randperm(eligible.numel(), generator=generator, device='cuda')[:rank]]
        candidate_delta, candidate_pec = delta.clone(), pec.clone()
        candidate_delta[selected] = (torch.rand(rank,device='cuda',generator=generator,dtype=torch.float64)*150
                                     + 1j*torch.rand(rank,device='cuda',generator=generator,dtype=torch.float64)*4)
        candidate_pec[selected] = ~candidate_pec[selected] if index%2 else False
        trial, trial_seconds = timed(lambda: response.trial(delta=candidate_delta,pec=candidate_pec))
        (field, observed, residual), full_seconds = timed(lambda: reference(candidate_delta,candidate_pec))
        projection_error = relative(trial.projections,observed)
        local_error = relative(trial.result().raw_output['local_field'],field[design])
        accepted = index%3 != 1
        accept_seconds = None
        if accepted:
            _, accept_seconds = timed(lambda: response.accept(trial))
            delta, pec = candidate_delta, candidate_pec
        records.append(dict(rank=trial.rank,accepted=accepted,trial_seconds=trial_seconds,
                            accept_seconds=accept_seconds,full_refactor_seconds=full_seconds,
                            projection_relative_error=projection_error,local_relative_error=local_error,
                            full_physical_residual=residual))
        if not all(__import__('math').isfinite(v) and v <= 1e-7 for v in (projection_error,local_error,residual)):
            raise AssertionError(records[-1])
    # A material gradient is compared against fresh full-system central differences.
    amplitude = torch.tensor(.8,device='cuda',dtype=torch.float64,requires_grad=True)
    direction = torch.rand(n,device='cuda',dtype=torch.float64,generator=generator)*100
    result = response.run(delta=amplitude*direction)
    loss = result.raw_output['projections'].abs().square().sum()
    gradient = torch.autograd.grad(loss,amplitude)[0]
    step = 1e-4
    values = [reference(((float(amplitude.detach())+sign*step)*direction).to(torch.complex128),torch.zeros_like(pec))[1]
              for sign in (1,-1)]
    difference = (values[0].abs().square().sum()-values[1].abs().square().sum())/(2*step)
    gradient_error = float((gradient-difference).abs()/difference.abs().clamp_min(1e-30))
    if gradient_error > 1e-5:
        raise AssertionError(f'Material gradient relative error {gradient_error}')
    trial_median = statistics.median(r['trial_seconds'] for r in records)
    full_median = statistics.median(r['full_refactor_seconds'] for r in records)
    report = dict(device=torch.cuda.get_device_name(),visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  tracked_source_dirty=subprocess.run(['git','diff','--quiet'],check=False).returncode != 0,
                  torch_version=torch.__version__,cupy_version=cp.__version__,frequency=frequency,
                  full_unknowns=base_matrix.shape[0],design_unknowns=n,observations=len(cfg.observations),
                  volume_design=args.volume_design,
                  cold_prepare_seconds=cold_seconds,selected_inverse_seconds=response.precompute_seconds,
                  reference_warmup_seconds=reference_warmup,trial_median_seconds=trial_median,
                  accept_median_seconds=statistics.median(r['accept_seconds'] for r in records if r['accepted']),
                  full_refactor_median_seconds=full_median,warm_trial_speedup=full_median/trial_median,
                  estimated_break_even_trials=(cold_seconds/max(full_median-trial_median,1e-30)
                                               if full_median>trial_median else None),
                  memory=dict(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              driver_free_change_after_prepare_bytes=free_before-free_after,
                              response_bytes_estimate=response.response_bytes_estimate),
                  gradient=dict(autograd=float(gradient),central_difference=float(difference),relative_error=gradient_error),
                  trials=records,passed=True,
                  scope='Compact graded substrate/ground/feed antenna; independent full sparse reassembly of diagonal/PEC changes. '
                        'Reference timing includes factorization, solve, physical residual and projection, excluding scene compilation. '
                        'Driver memory change is retained memory, not process peak; torch excludes external library allocations. '
                        'No published 8064-unknown device or upstream CPU runtime speed claim.')
    response.close()
    full._release_direct_solver()
    gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('passed','full_unknowns','design_unknowns','cold_prepare_seconds',
          'trial_median_seconds','accept_median_seconds','full_refactor_median_seconds','warm_trial_speedup','gradient')},indent=2))


if __name__ == '__main__':
    main()

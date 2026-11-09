"""Compare the external original optimizer with the public CUDA response API.

The C++ driver includes upstream headers from an external checkout. Inputs are
physical Green responses computed by Maxwell; this is not an APF benchmark.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import time
import numpy as np
import torch
import witwin.maxwell as mw


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter()-start


def relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[6,12,24])
    parser.add_argument('--trials', type=int, default=90)
    args = parser.parse_args()
    if args.trials < 3 or any(m < 6 for m in args.sizes):
        parser.error('Use at least three trials and at least six cells per axis.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Run this qualification on physical GPU 1.')
    template = runpy.run_path('examples/fdfd/local_response_antenna.py')['build']()
    report = {'scope':'Original PNGF CPU optimizer vs Maxwell CUDA public trial/accept, shared physical responses; no upstream APF timing',
              'upstream_revision':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
              'maxwell_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'maxwell_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
              'driver_sha256':hashlib.sha256(args.driver.read_bytes()).hexdigest(),
              'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
              'gpu':torch.cuda.get_device_name(), 'precision':'complex128',
              'cpu':subprocess.check_output(['lscpu'],text=True),
              'torch':torch.__version__, 'cases':[]}
    for m in args.sizes:
        directory = args.work / str(m)
        directory.mkdir(parents=True,exist_ok=True)
        lateral = [-.2,-.15,-.12,-.08] + np.linspace(-.06,.06,m+1).tolist() + [.08,.12,.15,.2]
        vertical = [-.2,-.15,-.12,-.08,-.04,-.02,0.,.02,.04,.08,.12,.15,.2]
        scene = mw.Scene(domain=template.scene.domain, grid=mw.GridSpec.custom(lateral,lateral,vertical),
                         boundary=template.scene.boundary,device='cuda')
        for structure in template.scene.structures:
            scene.add_structure(structure)
        for source in template.scene.sources:
            scene.add_source(source)
        prepared_grid = mw.Simulation.fdfd(scene,frequency=.9e9).prepare().solver.scene
        nx, ny, nz = prepared_grid.Nx, prepared_grid.Ny, prepared_grid.Nz
        start = int((prepared_grid.x+.06).abs().argmin())
        plane = int(prepared_grid.z.abs().argmin())
        # Match the external optimizer's x-fast planar Ex, then Ey ordering.
        indices = [(start+i)*ny*nz+(start+j)*nz+plane for j in range(m+1) for i in range(m)]
        indices += [(nx-1)*ny*nz+(start+i)*(ny-1)*nz+(start+j)*nz+plane for j in range(m) for i in range(m+1)]
        design = torch.tensor(indices,device='cuda',dtype=torch.int64)
        config = replace(template.config.local_response,design=mw.LocalDesignRegion(indices=design),rhs_batch_size=32)
        response,cold = timed(lambda: mw.Simulation.fdfd(scene,frequency=.9e9,solver=mw.GMRES(solver_type='direct',precision='double'),local_response=config).prepare())
        with response:
            n=design.numel()
            permutation=torch.searchsorted(response.design_indices,design)
            for name, array in [('green',response._green[permutation][:,permutation]),('observation',response._observation_green[:,permutation]),
                                ('incident',response._u0[permutation]),('background',response._y0)]:
                array.cpu().numpy().ravel(order='F').tofile(directory/f'{name}.bin')
            groups=[]
            for j in range(m):
                for i in range(m):
                    groups.append([i+j*m,i+(j+1)*m,m*(m+1)+i+j*(m+1),m*(m+1)+i+1+j*(m+1)])
            groups=permutation[torch.tensor(groups,device='cuda')].cpu().tolist()
            source_support=(response._rhs[response.design_indices].abs()>0).cpu().numpy()
            eligible=[i for i,g in enumerate(groups) if not source_support[g].any()]
            rng=np.random.default_rng(71)
            tiles=np.zeros(m*m,dtype=bool)
            def mask_for(state):
                mask=np.zeros(n,dtype=bool)
                for t in np.flatnonzero(state):
                    mask[groups[t]]=True
                return mask
            sequence=[]; masks=[]; previous=mask_for(tiles)
            while len(sequence)<args.trials:
                tile=int(rng.choice(eligible)); candidate=tiles.copy(); candidate[tile]^=True
                mask=mask_for(candidate)
                if np.array_equal(mask,previous):
                    continue
                accept=int(len(sequence)%3!=2)
                sequence.append((tile,accept)); masks.append(torch.tensor(mask,device='cuda'))
                if accept:
                    tiles=candidate; previous=mask
            np.savetxt(directory/'sequence.txt',sequence,fmt='%d')
            delta=torch.zeros(n,device='cuda',dtype=torch.complex128)
            for _ in range(10):
                response.trial(delta=delta,pec=masks[0])
            gpu_values=[]; gpu_trial=[]; gpu_accept=[]; dense_errors=[]; ranks=[]
            for (_,accept),mask in zip(sequence,masks):
                trial,seconds=timed(lambda:response.trial(delta=delta,pec=mask))
                gpu_trial.append(seconds); gpu_values.append(trial.projections); ranks.append(trial._changed.numel())
                accepted=0.
                if accept:
                    _,accepted=timed(lambda:response.accept(trial))
                gpu_accept.append(accepted)
                # Independent fresh reduced LU checks accumulated update drift.
                matrix=torch.where(mask[:,None],response._green,response._identity)
                current=torch.linalg.solve(matrix,mask*response._u0)
                exact=response._y0-response._observation_green@current
                dense_errors.append(relative(trial.projections,exact))
            values=torch.stack(gpu_values)
            case={'cells_per_axis':m,'design_dofs':n,'full_dofs':response.scene.N_vector_total,'observations':3,
                  'trials':args.trials,'accepted':sum(a for _,a in sequence),'rank_min':min(ranks),'rank_max':max(ranks),
                  'cold_maxwell_prepare_s':cold,'precompute_residual':response.precompute_residual,
                  'gpu_trial_median_s':float(np.median(gpu_trial)),
                  'gpu_accept_median_s':float(np.median([s for s in gpu_accept if s])),
                  'gpu_updates_total_s':sum(gpu_trial)+sum(gpu_accept),
                  'gpu_trial_p95_s':float(np.percentile(gpu_trial,95)),
                  'gpu_trial_max_s':max(gpu_trial),
                  'gpu_accept_p95_s':float(np.percentile([s for s in gpu_accept if s],95)),
                  'gpu_accept_max_s':max(gpu_accept),
                  'rebuild_interval':response.config.rebuild_interval,
                  'rebuild_count':response._rebuild_count,
                  'max_gpu_vs_fresh_reduced_lu_relative':max(dense_errors),'cpu':[],
                  'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.bin')}}
            np.savetxt(directory/'gpu.csv',np.column_stack([gpu_trial,gpu_accept,ranks,values.cpu().numpy().view(float)]),delimiter=',')
            for threads in (1,4):
                env=dict(os.environ,OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads))
                completed=subprocess.run([str(args.driver),str(directory),str(m),str(threads)],env=env,text=True,capture_output=True,check=True)
                rows=np.loadtxt(directory/f'cpu-{threads}.csv',delimiter=',',skiprows=1,ndmin=2)
                observed=torch.tensor(rows[:,3::2]+1j*rows[:,4::2],device='cuda')
                if rows.shape != (args.trials,9) or not np.isfinite(rows).all():
                    raise RuntimeError('Invalid or nonfinite external reference output.')
                errors=[relative(a,b) for a,b in zip(observed,values)]
                record={'threads':threads,'trial_median_s':float(np.median(rows[:,0])),
                        'trial_p95_s':float(np.percentile(rows[:,0],95)),
                        'accept_p95_s':float(np.percentile(rows[rows[:,1]>0,1],95)),
                        'accept_median_s':float(np.median(rows[rows[:,1]>0,1])),
                        'updates_total_s':float(rows[:,:2].sum()),'max_projection_relative':max(errors),
                        'per_observation_relative':[relative(observed[:,j],values[:,j]) for j in range(3)],
                        'rank_matches':bool(np.array_equal(rows[:,2],ranks)), 'stdout':completed.stdout}
                record['cpu_over_gpu_trial_ratio']=record['trial_median_s']/case['gpu_trial_median_s']
                case['cpu'].append(record)
            # Separate same-generation throughput probe; never changes accepted state.
            batch_indices=torch.tensor([groups[i] for i in eligible[:16]],device='cuda')
            batch_pec=~response._accepted_pec[batch_indices]
            def singles():
                return tuple(response.trial_updates(row,pec=mask) for row,mask in zip(batch_indices,batch_pec))
            def batched():
                return response.trial_updates_batch(batch_indices,pec=batch_pec)
            expected=singles()
            actual=batched()
            batch_error=max(relative(a.projections,b.projections) for a,b in zip(actual,expected))
            for _ in range(5):
                singles();batched()
            single_times=[];batch_times=[]
            for _ in range(20):
                _,seconds=timed(singles);single_times.append(seconds)
                _,seconds=timed(batched);batch_times.append(seconds)
            case['batch_probe']={'scope':'Independent same-generation PEC row proposals, GPU serial vs GPU batch; no CPU or search timing comparison',
                'count':len(expected),'max_projection_relative':batch_error,
                'serial_group_median_s':float(np.median(single_times)),
                'batch_group_median_s':float(np.median(batch_times)),
                'batch_per_candidate_s':float(np.median(batch_times))/len(expected)}
            report['cases'].append(case)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(case,indent=2),flush=True)
            if not np.isfinite(batch_error) or batch_error>1e-7 or not np.isfinite(dense_errors).all() or max(dense_errors)>1e-7 or any(c['max_projection_relative']>1e-7 or not c['rank_matches'] for c in case['cpu']):
                raise RuntimeError('Numerical comparison failed; evidence saved.')


if __name__=='__main__':
    main()

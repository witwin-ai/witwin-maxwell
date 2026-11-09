import json
import argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description='Independently audit saved optimizer measurement CSV files.')
parser.add_argument('--report',type=Path,default=Path('docs/assessments/fdfd-pngf-optimizer.json'))
parser.add_argument('--data',type=Path,default=Path('docs/assessments/fdfd-pngf-optimizer-data'))
parser.add_argument('--output',type=Path,default=Path('docs/assessments/fdfd-pngf-optimizer-audit.json'))
args=parser.parse_args()
d=json.loads(args.report.read_text())
assert d['maxwell_dirty'] is False
checks=[]
for c in d['cases']:
    n=c['design_dofs']; directory=args.data
    gpu=np.loadtxt(directory/f'n{n}-gpu.csv',delimiter=',')
    sequence=np.loadtxt(directory/f'n{n}-sequence.txt',dtype=int)
    assert gpu.shape==(90,9) and np.isfinite(gpu).all()
    assert sequence[:,1].sum()==60 and np.array_equal(gpu[:,1]>0,sequence[:,1].astype(bool))
    assert np.isclose(gpu[:,:2].sum(),c['gpu_updates_total_s'])
    assert np.isclose(np.median(gpu[:,0]),c['gpu_trial_median_s'])
    values=gpu[:,3::2]+1j*gpu[:,4::2]
    for cpu in c['cpu']:
        rows=np.loadtxt(directory/f'n{n}-cpu-{cpu["threads"]}.csv',delimiter=',',skiprows=1)
        assert rows.shape==gpu.shape and np.isfinite(rows).all()
        assert np.array_equal(rows[:,2],gpu[:,2])
        assert np.array_equal(rows[:,1]>0,sequence[:,1].astype(bool))
        observed=rows[:,3::2]+1j*rows[:,4::2]
        error=np.max(np.linalg.norm(values-observed,axis=1)/np.linalg.norm(values,axis=1))
        assert error<=1e-7
        assert np.isclose(error,cpu['max_projection_relative'],rtol=1e-6,atol=1e-16)
        assert np.isclose(np.median(rows[:,0]),cpu['trial_median_s'])
        assert np.isclose(rows[:,:2].sum(),cpu['updates_total_s'])
        checks.append({'n':n,'cpu_threads':cpu['threads'],'rows':len(rows),'max_projection_relative_recomputed':float(error),'passed':True})
result={'clean_source_verified':True,'checks':checks,'all_passed':True,
        'scope':'Independent NumPy readback of saved CSV: finiteness, accept counts, ranks, errors, timing aggregation; not another physical solver.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

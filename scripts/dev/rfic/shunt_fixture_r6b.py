"""Acquire isolated plate-to-ring silicon admittance through independent methods."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tests.support.rfic_r6b_references import plate_fixture,electrostatic,lumped_estimate,FREQUENCIES

def native(directory, maximum, edge):
    import torch
    import witwin.maxwell as mw
    scene,config=plate_fixture(maximum,edge)
    start=time.perf_counter()
    result=mw.Simulation.mom(scene,frequencies=FREQUENCIES,config=config).run()
    y=result.network.to_y()[:,0,0].cpu().numpy()
    data=dict(maximum_um=maximum,edge_um=edge,frequency_Hz=list(FREQUENCIES),
              y_real=y.real.tolist(),y_imag=y.imag.tolist(),
              seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),
              unknowns=result.network.metadata["unknowns"])
    print(json.dumps(data),flush=True)
    (directory/f"native-{maximum:g}-{edge:g}.json").write_text(json.dumps(data,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("static","native"))
    p.add_argument("directory",type=Path)
    p.add_argument("--pitch",type=float,default=5.)
    p.add_argument("--gap",type=float,default=0.)
    p.add_argument("--maximum",type=float,default=20.)
    p.add_argument("--edge",type=float,default=5.)
    args=p.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    if args.mode=="native":
        native(args.directory,args.maximum,args.edge)
    else:
        os.nice(10)
        start=time.perf_counter()
        data=electrostatic(args.pitch,args.gap)
        data.update(seconds=time.perf_counter()-start,lumped_estimate=lumped_estimate())
        print(json.dumps(data),flush=True)
        (args.directory/f"static-{args.pitch:g}-gap{args.gap:g}.json").write_text(json.dumps(data,indent=2))

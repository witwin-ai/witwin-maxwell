"""Acquire public-path shield sweeps on GPU 0 and fit the single-pi model.

Invoke through the shared gpu0.py runner. This is evidence acquisition outside
pytest, including meshes that exceed the test allocator budget.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from tests.support.rfic_fixtures import fixture_fa,SWEEP
from tests.support.rfic_r6_references import fit_pi,metrics


def acquire(directory,variant,mesh,grounded=True,sigma=10.,frequencies=SWEEP,label="",lossy=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fixture = fixture_fa(shield=variant,mesh=mesh,shield_grounded=grounded,substrate_sigma=sigma,lossy=lossy)
    simulation = fixture.simulation(frequencies)
    prepared = simulation.prepare()
    network = prepared.run().network
    torch.cuda.synchronize()
    y = network.to_y().cpu().numpy()
    f = network.frequencies.cpu().numpy()
    s = network.s.cpu().numpy()
    data = dict(variant=variant,mesh=mesh,grounded=grounded,substrate_sigma_S_m=sigma,lossy_metal=lossy,
                frequency_Hz=f.tolist(),y_real=y.real.tolist(),y_imag=y.imag.tolist(),
                s_real=s.real.tolist(),s_imag=s.imag.tolist(),metrics=metrics(f,y),
                unknowns=network.metadata["unknowns"],faces=network.metadata["faces"],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                mesh_quality=network.metadata["mesh_quality"],
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ("tests/support/rfic_fixtures.py","witwin/maxwell/mom/layered/runtime.py",
                     "witwin/maxwell/mom/layered/network.py","witwin/maxwell/mom/layered/mesh.py")})
    if len(f)>8:
        data["pi_fit"] = fit_pi(f,y)
    name = f"{variant}-{mesh}-grounded{int(grounded)}-sigma{sigma:g}" + (f"-{label}" if label else "") + ".json"
    directory.mkdir(parents=True,exist_ok=True)
    (directory/name).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:data[k] for k in ("variant","mesh","unknowns","peak_allocated_bytes")}),flush=True)
    print(json.dumps({k:data["metrics"][k] for k in
                     ("L_1GHz_H","Q_peak","peak_frequency_Hz","self_resonance_Hz")}),flush=True)
    del prepared,simulation,fixture,network
    gc.collect()
    torch.cuda.empty_cache()
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--variant",choices=("none","pgs","solid","all"),default="all")
    parser.add_argument("--mesh",default="shield")
    parser.add_argument("--floating",action="store_true")
    parser.add_argument("--sigma",type=float,default=10.)
    parser.add_argument("--frequencies",type=float,nargs="+")
    parser.add_argument("--label",default="")
    parser.add_argument("--pec-metal",action="store_true",
                        help="Use PEC for winding, ring and vias; retain the declared shield material")
    args=parser.parse_args()
    frequencies=tuple(args.frequencies) if args.frequencies else tuple(sorted(set((*SWEEP,1e9))))
    variants=("none","pgs","solid") if args.variant=="all" else (args.variant,)
    for variant in variants:
        acquire(args.directory,variant,args.mesh,not args.floating,args.sigma,frequencies,args.label,not args.pec_metal)


if __name__=="__main__":
    main()

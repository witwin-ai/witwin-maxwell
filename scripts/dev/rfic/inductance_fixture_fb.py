"""Resolved-copper union-grid quasi-static inductance reference for F-B."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from tests.support.rfic_fixtures import fixture_fb
from tests.support.rfic_r6_references import geometry_digest
from witwin.maxwell.mom.layered.runtime import compile_layered_scene

BINARY=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106/tools/fasthenry-build/bin/fasthenry"


def write_model(directory,pitch,nw,nh,thin=False,longitudinal=1,fmin=1e6,fmax=1e9):
    f=fixture_fb(thin=thin)
    _,rectangles,ports,materials,_,vias,vm=compile_layered_scene(f.scene)
    digest=geometry_digest(rectangles,materials,vias,vm)
    rectangles=[tuple(round(v*1e6,8) for v in r) for r in rectangles]
    vias=[tuple(round(v*1e6,8) for v in r) for r in vias]
    sheets=rectangles+[(205.,215.,y-10,y+10,305.) for y in (170.,70.,145.,45.)]
    def axis(a):
        edges=sorted({v for r in sheets for v in r[2*a:2*a+2]}|
                     {v for r in vias for v in r[2*a:2*a+2]}|
                     ({210.} if a==0 else set()))
        return [float(v) for lo,hi in zip(edges[:-1],edges[1:])
                for v in np.linspace(lo,hi,math.ceil((hi-lo)/pitch)+1)[:-1]]+[edges[-1]]
    xs,ys=axis(0),axis(1)
    nodes,segments,coordinates={},[],{}
    def node(point,label=""):
        key=tuple(round(float(v),10) for v in point),label
        if key not in nodes:
            nodes[key]="N"+str(len(nodes))
            coordinates[nodes[key]]=key[0]
        return nodes[key]
    def seg(a,b,w,h,sigma):
        pa,pb=coordinates[a],coordinates[b]
        previous=a
        for q in range(1,longitudinal+1):
            endpoint=(b if q==longitudinal else node(
                tuple(x+(y-x)*q/longitudinal for x,y in zip(pa,pb)),
                "split"+str(len(segments))))
            segments.append((previous,endpoint,w,h,sigma))
            previous=endpoint
    cells={}
    for z in (302.,305.):
        for i,(x0,x1) in enumerate(zip(xs[:-1],xs[1:])):
            for j,(y0,y1) in enumerate(zip(ys[:-1],ys[1:])):
                x,y=(x0+x1)/2,(y0+y1)/2
                owners=[k for k,r in enumerate(sheets) if r[4]==z and r[0]<x<r[1] and r[2]<y<r[3]]
                if owners:
                    physical=any(k<len(rectangles) for k in owners)
                    h=.1 if thin else (2. if z==305. else 1.)
                    cells[z,i,j]=(node((x,y,z)),x,y,x1-x0,y1-y0,h,58. if physical else 1e10)
    cuts={(p,s):[] for p in range(1,5) for s in ("s","g")}
    ys_port=(170.,70.,145.,45.)
    for (z,i,j),(name,x,y,dx,dy,h,sigma) in cells.items():
        for a,(di,dj) in enumerate(((1,0),(0,1))):
            if (z,i+di,j+dj) not in cells:
                continue
            other,ox,oy,odx,ody,oh,osigma=cells[z,i+di,j+dj]
            border=xs[i+1] if a==0 else ys[j+1]
            on_port=a==0 and border==210. and z==305. and any(abs(y-v)<10 for v in ys_port)
            point=(border,y,z) if a==0 else (x,border,z)
            width=dy if a==0 else dx
            if on_port:
                number=next(p for p,v in enumerate(ys_port,1) if abs(y-v)<10)
                left,right=node(point,f"p{number}s"),node(point,f"p{number}g")
                cuts[number,"s"].append(left);cuts[number,"g"].append(right)
            elif h!=oh or sigma!=osigma:
                left=right=node(point)
            else:
                seg(name,other,width,h,sigma)
                continue
            seg(name,left,width,h,sigma);seg(right,other,width,oh,osigma)
    for x0,x1,y0,y1,z0,z1 in vias:
        for (z,i,j),(name,x,y,dx,dy,h,sigma) in cells.items():
            if z==z0 and x0<x<x1 and y0<y<y1:
                seg(name,cells[z1,i,j][0],dx,dy,58.)
    lines=["* Four-port compiled F-B metal union",".units um",
           f".default sigma=58 nwinc={nw} nhinc={nh} rw=2 rh=2"]
    for (point,label),name in nodes.items():
        lines.append(name+" "+" ".join(f"{a}={v:.12g}" for a,v in zip("xyz",point)))
    for i,(a,b,w,h,sigma) in enumerate(segments):
        lines.append(f"E{i} {a} {b} w={w:.12g} h={h:.12g} sigma={sigma:.12g}")
    for p in range(1,5):
        for side in ("s","g"):
            assert cuts[p,side]
            lines.append(".equiv "+" ".join(cuts[p,side]))
        lines.append(f".external {cuts[p,'s'][0]} {cuts[p,'g'][0]} p{p}")
    lines+=[f".freq fmin={fmin:.16g} fmax={fmax:.16g} ndec=1",".end"]
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"model.inp"
    path.write_text("\n".join(lines)+"\n")
    (directory/"model.json").write_text(json.dumps(dict(geometry_sha256=digest,pitch_um=pitch,
        nw=nw,nh=nh,thin=thin,longitudinal_subdivisions=longitudinal,fmin_Hz=fmin,fmax_Hz=fmax,cells=len(cells),segments=len(segments),nodes=len(nodes),
        compiled_rectangles_um=rectangles,compiled_vias_um=vias,
        ports=[dict(name=p.name,positive=list(p.positive),negative=list(p.negative)) for p in ports],
        bridge_conductivity_S_m=1e16,contact="ideal bridge midplane cuts, no deembedding",
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    return path


def parse(directory,expected_count=4):
    data=[]
    for block in (directory/"Zc.mat").read_text().split("ADMITTANCE matrix for frequency = ")[1:]:
        f=float(block.split()[0])
        rows=[]
        for line in block.splitlines()[1:5]:
            n=re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",line)
            assert len(n)==4,(line,n)
            rows.append([complex(float(n[i]),-float(n[i+1])) for i in (0,2)])
        y=np.asarray(rows)
        # Active columns p1 and p3: Vd1=V1-V2, Vd2=V3-V4.
        zw=np.linalg.inv(y[[0,2],:])
        inductance=-zw.imag/(2*np.pi*f)
        data.append(dict(frequency_Hz=f,L1_H=inductance[0,0],L2_H=inductance[1,1],
            M_H=inductance[0,1],k=inductance[0,1]/np.sqrt(inductance[0,0]*inductance[1,1]),
            R1_ohm=zw[0,0].real,R2_ohm=zw[1,1].real,
            z_w_real=zw.real.tolist(),z_w_imag=zw.imag.tolist(),
            y_real=y.real.tolist(),y_imag=y.imag.tolist(),
            kcl_relative=float(np.max(np.abs(y[[0,2]]+y[[1,3]]))/np.max(np.abs(y))),
            reciprocity_relative=float(np.max(np.abs(zw-zw.T))/np.max(np.abs(zw)))))
    assert len(data)==expected_count
    return data


def acquire(directory,pitch,nw,nh,thin=False,method="multi",order=6,longitudinal=1,
            preconditioner="seg",fmin=1e6,fmax=1e9):
    path=write_model(directory,pitch,nw,nh,thin,longitudinal,fmin,fmax)
    args=["nice","-n","10",str(BINARY),"-a","off","-m",method,
          "-o",str(order),"-p",preconditioner,"-t","1e-9","-c","600","-x","p1","-x","p3",path.name]
    command=["/usr/bin/time","-v","-o",str(directory/"resources.txt"),*args]
    start=time.perf_counter()
    with (directory/"run.log").open("w") as log:
        done=subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"))
    print(directory,"rc="+str(done.returncode),flush=True)
    (directory/"receipt.json").write_text(json.dumps(dict(returncode=done.returncode,command=command,
        seconds=time.perf_counter()-start,binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        log_sha256=hashlib.sha256((directory/"run.log").read_bytes()).hexdigest()),indent=2))
    assert done.returncode==0
    assert "Convergence not reached" not in (directory/"run.log").read_text()
    result=parse(directory,int(math.floor(math.log10(1.001*fmax/fmin)))+1)
    (directory/"network.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--pitch",type=float,default=5.)
    p.add_argument("--nw",type=int,default=1)
    p.add_argument("--nh",type=int,default=3)
    p.add_argument("--thin",action="store_true")
    p.add_argument("--method",choices=("direct","multi"),default="multi")
    p.add_argument("--order",type=int,default=6)
    p.add_argument("--longitudinal",type=int,choices=(1,2),default=1)
    p.add_argument("--preconditioner",choices=("cube","seg"),default="seg")
    p.add_argument("--fmin",type=float,default=1e6)
    p.add_argument("--fmax",type=float,default=1e9)
    a=p.parse_args()
    acquire(a.directory,a.pitch,a.nw,a.nh,a.thin,a.method,a.order,a.longitudinal,
            a.preconditioner,a.fmin,a.fmax)

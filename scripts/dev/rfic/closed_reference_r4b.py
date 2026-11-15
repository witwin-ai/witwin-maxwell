"""Independent quasi-static strip-and-explicit-return reference acquisition."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

BINARY=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106/tools/fasthenry-build/bin/fasthenry"


def grid_model(pitch,nw,nh,thickness,fmin,fmax):
    """Orthogonal volume-current graph of the exact rectangle union in um.

    Each cell owns its conductivity and thickness. Half-cell junction segments
    keep cross-section changes explicit. The high-conductivity source strip
    retains the physical thickness; its mid-plane contact difference is
    recorded rather than hidden in a filament-convergence claim.
    """
    import numpy as np
    rectangles=[(0.,200.,-5.,5.,58.),(-10.,0.,-115.,5.,1e10),
                (200.,210.,-115.,5.,1e10),(0.,95.,-115.,-105.,1e10),
                (105.,200.,-115.,-105.,1e10),(95.,105.,-115.,-105.,1e10)]
    def axis(a):
        edges=sorted({v for r in rectangles for v in r[2*a:2*a+2]}|({100.} if a==0 else set()))
        return [float(v) for lo,hi in zip(edges[:-1],edges[1:])
                for v in np.linspace(lo,hi,math.ceil((hi-lo)/pitch)+1)[:-1]]+[edges[-1]]
    xs,ys=axis(0),axis(1)
    nodes={};segments=[];equiv={side:[] for side in ("left","right")}
    def node(x,y,label=""):
        key=(round(float(x),10),round(float(y),10),label)
        if key not in nodes:
            nodes[key]="N"+str(len(nodes))
        return nodes[key]
    def seg(a,b,w,h,sigma):
        segments.append((a,b,w,h,sigma))
    cells={}
    for i,(x0,x1) in enumerate(zip(xs[:-1],xs[1:])):
        for j,(y0,y1) in enumerate(zip(ys[:-1],ys[1:])):
            x,y=(x0+x1)/2,(y0+y1)/2
            owners=[r for r in rectangles if r[0]<x<r[1] and r[2]<y<r[3]]
            if owners:
                cells[i,j]=(node(x,y),x,y,x1-x0,y1-y0,owners[0][4])
    for (i,j),(name,x,y,dx,dy,sigma) in cells.items():
        for axis,(di,dj) in enumerate(((1,0),(0,1))):
            if (i+di,j+dj) not in cells:
                continue
            other,ox,oy,odx,ody,osigma=cells[i+di,j+dj]
            cut=axis==0 and xs[i+1]==100 and y<-105
            border=xs[i+1] if axis==0 else ys[j+1]
            px,py=(border,y) if axis==0 else (x,border)
            width=dy if axis==0 else dx
            if cut:
                left,right=node(px,py,"left"),node(px,py,"right")
                equiv["left"].append(left);equiv["right"].append(right)
            elif sigma!=osigma:
                left=right=node(px,py)
            else:
                seg(name,other,width,thickness,sigma)
                continue
            seg(name,left,width,thickness,sigma)
            seg(right,other,width,thickness,osigma)
    lines=["* Exact strip/return union quasi-static volume graph",".units um",
           f".default sigma=58 nwinc={nw} nhinc={nh} rw=2 rh=2"]
    for (x,y,label),name in nodes.items():
        lines.append(f"{name} x={x:.14g} y={y:.14g} z=100")
    for i,(a,b,w,h,sigma) in enumerate(segments):
        lines.append(f"E{i} {a} {b} w={w:.14g} h={h:.14g} sigma={sigma:.14g}")
    for side in ("left","right"):
        lines.append(".equiv "+" ".join(equiv[side]))
    lines+=[f".external {equiv['left'][0]} {equiv['right'][0]} p",
            f".freq fmin={fmin:.16g} fmax={fmax:.16g} ndec=1",".end"]
    return lines,dict(cells=len(cells),segments=len(segments),nodes=len(nodes),pitch_um=pitch,
        contact="midplane bridge-cut graph; native ideal strip is on the top face")


def acquire(directory,nw,nh,maximum,fmin=1e4,fmax=1e11,thickness=2.,grid=False):
    directory.mkdir(parents=True,exist_ok=True)
    nodes=[]
    segments=[]
    def node(point):
        nodes.append(point)
        return "N"+str(len(nodes)-1)
    def segment(a,b,width,thickness,sigma):
        count=math.ceil(math.dist(a,b)/maximum)
        a_name=node(a)
        for i in range(1,count+1):
            point=tuple(x+(y-x)*i/count for x,y in zip(a,b))
            b_name=node(point)
            segments.append((a_name,b_name,width,thickness,sigma))
            a_name=b_name
        return len(nodes)-count-1,len(nodes)-1
    # Ideal return follows the same union's strip centrelines. Its horizontal
    # source bridge is at the top face; vertical ideal arms join the mid-plane.
    paths=[((0.,0.,100.),(200.,0.,100.),58.),
           ((200.,0.,100.),(205.,0.,100.),1e10),
           ((205.,0.,100.),(205.,-110.,100.),1e10),
           ((205.,-110.,100.),(105.,-110.,100.),1e10),
           ((105.,-110.,100.),(105.,-110.,101.),1e10),
           ((105.,-110.,101.),(100.,-110.,101.),1e10),
           ((100.,-110.,101.),(95.,-110.,101.),1e10),
           ((95.,-110.,101.),(95.,-110.,100.),1e10),
           ((95.,-110.,100.),(-5.,-110.,100.),1e10),
           ((-5.,-110.,100.),(-5.,0.,100.),1e10),
           ((-5.,0.,100.),(0.,0.,100.),1e10)]
    paths=[(tuple(100+thickness/2 if d==2 and v==101 else v for d,v in enumerate(a)),
            tuple(100+thickness/2 if d==2 and v==101 else v for d,v in enumerate(b)),sigma)
           for a,b,sigma in paths]
    endpoints=[segment(a,b,10.,thickness,sigma) for a,b,sigma in paths]
    lines=["* Straight copper strip with an explicit ideal return",".units um",
           f".default sigma=58 nwinc={nw} nhinc={nh} rw=2 rh=2"]
    for i,point in enumerate(nodes):
        lines.append(f"N{i} "+" ".join(f"{a}={v:.14g}" for a,v in zip("xyz",point)))
    for i,(a,b,w,h,sigma) in enumerate(segments):
        lines.append(f"E{i} {a} {b} w={w} h={h} sigma={sigma}")
    for i in range(len(endpoints)):
        if i==5:
            continue
        lines.append(f".equiv N{endpoints[i][1]} N{endpoints[(i+1)%len(endpoints)][0]}")
    lines.append(f".external N{endpoints[5][1]} N{endpoints[6][0]} p")
    # 10 kHz, 100 kHz, 1 MHz, ... 100 GHz; copper delta/t reaches 0.105.
    lines+=[f".freq fmin={fmin:.16g} fmax={fmax:.16g} ndec=1",".end"]
    geometry=dict(model="centreline",contact="top-face ideal strip")
    if grid:
        lines,geometry=grid_model(maximum,nw,nh,thickness,fmin,fmax)
        geometry["model"]="union_grid"
    model=directory/"model.inp"
    model.write_text("\n".join(lines)+"\n")
    command=["nice","-n","10",str(BINARY),"-a","off","-m","direct","-o","4","-t","1e-10","-c","600",model.name]
    start=time.perf_counter()
    with (directory/"run.log").open("w") as log:
        done=subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
    print(directory,"rc="+str(done.returncode),flush=True)
    (directory/"receipt.json").write_text(json.dumps(dict(returncode=done.returncode,
        seconds=time.perf_counter()-start,command=command,binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),nw=nw,nh=nh,maximum_um=maximum,
        fmin_Hz=fmin,fmax_Hz=fmax,thickness_um=thickness,geometry=geometry),indent=2))
    assert done.returncode==0
    text=(directory/"Zc.mat").read_text()
    data=[]
    for block in text.split("Impedance matrix for frequency = ")[1:]:
        f=float(block.split()[0])
        # Output uses exp(+i omega t); retain native negative reactance.
        row=block.splitlines()[1]
        numbers=re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",row)
        r,x=map(float,numbers[:2])
        data.append(dict(frequency_Hz=f,R_ohm=r,L_H=x/(2*math.pi*f)))
    assert len(data)==round(math.log10(fmax/fmin))+1,text[:1000]
    (directory/"network.json").write_text(json.dumps(data,indent=2))
    print(data,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--binary",type=Path,default=BINARY)
    p.add_argument("--nw",type=int,default=9)
    p.add_argument("--nh",type=int,default=5)
    p.add_argument("--maximum",type=float,default=20.)
    p.add_argument("--fmin",type=float,default=1e4)
    p.add_argument("--fmax",type=float,default=1e11)
    p.add_argument("--thickness",type=float,default=2.)
    p.add_argument("--grid",action="store_true")
    a=p.parse_args()
    BINARY=a.binary
    acquire(a.directory,a.nw,a.nh,a.maximum,a.fmin,a.fmax,a.thickness,a.grid)

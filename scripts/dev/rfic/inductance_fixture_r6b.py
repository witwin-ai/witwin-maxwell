"""Quasi-static inductance arbitration for the compiled two-port F-A layout."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from tests.support.rfic_fixtures import fixture_fa
from witwin.maxwell.mom.layered.runtime import compile_layered_scene

def write_model(directory, maximum, nw, nh, thin=False, terminal_ports=False):
    fixture = fixture_fa(device="cpu")
    stack, rectangles, ports, materials, _, vias, _ = compile_layered_scene(fixture.scene)
    nodes, segments = {}, []
    def node(point, label=""):
        key = (tuple(round(v, 8) for v in point), label)
        if key not in nodes:
            nodes[key] = f"N{len(nodes)}"
        return nodes[key]
    def segment(a, b, width, thickness, first="", last=""):
        n = math.ceil(math.dist(a, b)/maximum)
        points = [tuple(x+(y-x)*i/n for x,y in zip(a,b)) for i in range(n+1)]
        for i, (u,v) in enumerate(zip(points[:-1],points[1:])):
            segments.append((node(u, first if i==0 else ""), node(v, last if i==n-1 else ""),
                             width, thickness))
    for a,b,width,thickness in fixture.segments:
        a,b = tuple(v*1e6 for v in a),tuple(v*1e6 for v in b)
        if thickness:
            segment(a,b,width*1e6,.1 if thin else thickness*1e6)
        else:
            segment(a,b,4.,4.)
    # The finite ring is a closed 10-um strip, with the two terminal taps retained.
    ring = ((150.,-150.,305.),(150.,80.,305.),(150.,95.,305.),(150.,150.,305.),
            (-150.,150.,305.),(-150.,-150.,305.),(150.,-150.,305.))
    for a,b in zip(ring[:-1],ring[1:]):
        segment(a,b,10.,.1 if thin else 2.)
    external = []
    for port in ports:
        x,y,z = (float(v)*1e6 for v in port.positive)
        xn = float(port.negative[0])*1e6
        cut = (x+xn)/2
        signal_x,ground_x=(x,xn) if terminal_ports else (cut,cut)
        segment((125.,y,z),(signal_x,y,z),10.,.1 if thin else 2.,last=port.name+"s")
        segment((150.,y,z),(ground_x,y,z),10.,.1 if thin else 2.,last=port.name+"g")
        external.append((node((signal_x,y,z),port.name+"s"),
                         node((ground_x,y,z),port.name+"g"),port.name))
    lines = ["* Two-port compiled F-A quasi-static reference", ".units um",
             f".default sigma=58 nwinc={nw} nhinc={nh} rw=2 rh=2"]
    for (point,label),name in nodes.items():
        lines.append(name+" "+" ".join(f"{a}={v:.12g}" for a,v in zip("xyz",point)))
    for i,(a,b,w,h) in enumerate(segments):
        lines.append(f"E{i} {a} {b} w={w:.12g} h={h:.12g}")
    lines.extend(f".external {a} {b} {name}" for a,b,name in external)
    lines.extend((".freq fmin=1e8 fmax=1e9 ndec=1", ".end"))
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/"model.inp"
    path.write_text("\n".join(lines)+"\n")
    metadata = dict(maximum_segment_um=maximum,nwinc=nw,nhinc=nh,thin=thin,
        segments=len(segments),nodes=len(nodes),compiled_rectangles=rectangles,compiled_vias=vias,
        ports=[dict(name=p.name,positive=list(p.positive),negative=list(p.negative)) for p in ports],
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note="Rectangular strips with overlapping square bends; two ideal zero-length bridge cuts. "
             "The solver writes Y directly because its magnetoquasistatic common mode is singular.")
    (directory/"model.json").write_text(json.dumps(metadata,indent=2))
    return path


def write_grid(directory, maximum, nw, nh, thin=False, terminal_ports=False):
    fixture=fixture_fa(device="cpu")
    _,rectangles,ports,materials,_,vias,_=compile_layered_scene(fixture.scene)
    rectangles=[tuple(round(v*1e6,8) for v in r) for r in rectangles]
    vias=[tuple(round(v*1e6,8) for v in r) for r in vias]
    sheets=rectangles if terminal_ports else rectangles+[(130.,145.,y-5,y+5,305.) for y in (95.,80.)]
    def axis_edges(axis):
        values={v for r in sheets for v in r[2*axis:2*axis+2]}
        values.update(v for r in vias for v in r[2*axis:2*axis+2])
        if axis==0:
            values.add(137.5)
        values=sorted(values)
        return [float(v) for a,b in zip(values[:-1],values[1:])
                for v in np.linspace(a,b,math.ceil((b-a)/maximum)+1)[:-1]]+[values[-1]]
    xs,ys=axis_edges(0),axis_edges(1)
    nodes,segments={},[]
    def node(point,label=""):
        key=tuple(round(float(v),10) for v in point),label
        if key not in nodes:
            nodes[key]=f"N{len(nodes)}"
        return nodes[key]
    def seg(a,b,w,h,sigma=58.):
        segments.append((a,b,w,h,sigma))
    cells={}
    for z in (302.,305.):
        for i,(x0,x1) in enumerate(zip(xs[:-1],xs[1:])):
            for j,(y0,y1) in enumerate(zip(ys[:-1],ys[1:])):
                x,y=(x0+x1)/2,(y0+y1)/2
                owners=[k for k,r in enumerate(sheets) if r[4]==z and r[0]<x<r[1] and r[2]<y<r[3]]
                if owners:
                    physical=any(k<len(rectangles) for k in owners)
                    thickness=.1 if thin else (2. if z==305. else 1.)
                    cells[z,i,j]=(node((x,y,z)),x,y,x1-x0,y1-y0,thickness,58. if physical else 1e10)
    cut_nodes={(p,side):[] for p in (1,2) for side in ("s","g")}
    for (z,i,j),cell in cells.items():
        name,x,y,dx,dy,h,sigma=cell
        for axis,(di,dj) in enumerate(((1,0),(0,1))):
            key=z,i+di,j+dj
            if key not in cells:
                continue
            other,ox,oy,odx,ody,oh,osigma=cells[key]
            border=xs[i+1] if axis==0 else ys[j+1]
            width=dy if axis==0 else dx
            on_port=axis==0 and border==137.5 and z==305. and (75<y<85 or 90<y<100)
            point=(border,y,z) if axis==0 else (x,border,z)
            if on_port:
                port=1 if y>90 else 2
                left,right=node(point,f"p{port}s"),node(point,f"p{port}g")
                cut_nodes[port,"s"].append(left)
                cut_nodes[port,"g"].append(right)
            elif h!=oh or sigma!=osigma:
                left=right=node(point)
            if not on_port and h==oh and sigma==osigma:
                seg(name,other,width,h,sigma)
            else:
                seg(name,left,width,h,sigma)
                seg(right,other,width,oh,osigma)
    if terminal_ports:
        # Nominal terminal nodes identify equipotential contacts on the metal boundary.
        for number,port in enumerate(ports,1):
            for side,point in (("s",port.positive),("g",port.negative)):
                tx,ty,tz=(round(float(v)*1e6,8) for v in point)
                nominal=node((tx,ty,tz),f"p{number}{side}_terminal")
                cut_nodes[number,side].append(nominal)
                for (z,i,j),(name,x,y,dx,dy,h,sigma) in cells.items():
                    edge=x+dx/2 if side=="s" else x-dx/2
                    if z==tz and edge==tx and abs(y-ty)<5:
                        contact=node((tx,y,z),f"p{number}{side}_contact")
                        seg(name,contact,dy,h,sigma)
                        cut_nodes[number,side].append(contact)
    # Solid via cross sections join every occupied footprint cell on both metal planes.
    for x0,x1,y0,y1,z0,z1 in vias:
        for (z,i,j),(name,x,y,dx,dy,h,sigma) in cells.items():
            if z==z0 and x0<x<x1 and y0<y<y1:
                other=cells[z1,i,j][0]
                seg(name,other,dx,dy)
    lines=["* Compiled metal-union two-port quasi-static reference",".units um",
           f".default sigma=58 nwinc={nw} nhinc={nh} rw=2 rh=2"]
    for (point,label),name in nodes.items():
        lines.append(name+" "+" ".join(f"{a}={v:.12g}" for a,v in zip("xyz",point)))
    for i,(a,b,w,h,sigma) in enumerate(segments):
        lines.append(f"E{i} {a} {b} w={w:.12g} h={h:.12g} sigma={sigma:.12g}")
    for port in (1,2):
        for side in ("s","g"):
            lines.append(".equiv "+" ".join(cut_nodes[port,side]))
        lines.append(f".external {cut_nodes[port,'s'][0]} {cut_nodes[port,'g'][0]} p{port}")
    lines.extend((".freq fmin=1e8 fmax=1e9 ndec=1",".end"))
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"model.inp"
    path.write_text("\n".join(lines)+"\n")
    (directory/"model.json").write_text(json.dumps(dict(maximum_um=maximum,nwinc=nw,nhinc=nh,thin=thin,
        cells=len(cells),segments=len(segments),nodes=len(nodes),longitudinal_cell_halves_merged=True,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),compiled_rectangles_um=rectangles,
        compiled_vias_um=vias,port_terminals=[dict(name=p.name,positive=list(p.positive),
        negative=list(p.negative)) for p in ports],bridge_conductivity_S_m=1e16,
        note="Orthogonal ground-plane-style volume grid on the exact compiled union; "
             "corner omissions shrink with pitch. Direct admittance column avoids a singular common mode."),
        indent=2))
    return path


def parse(directory):
    text=(directory/"Zc.mat").read_text()
    matrices=[]
    for block in text.split("ADMITTANCE matrix for frequency = ")[1:]:
        frequency=float(block.split()[0])
        lines=block.splitlines()[1:3]
        values=[]
        for line in lines:
            numbers=re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",line)
            assert len(numbers)==2,(line,numbers)
            values.append([complex(float(numbers[i]),float(numbers[i+1])) for i in (0,)])
        y=np.conj(np.array(values))
        series=-1/y[1,0]
        matrices.append(dict(frequency_Hz=frequency,y_real=y.real.tolist(),y_imag=y.imag.tolist(),
                             series_L_H=-series.imag/(2*np.pi*frequency),series_R_ohm=series.real))
    assert len(matrices)==2,text[:500]
    return matrices

def run(directory, binary, maximum, nw, nh, thin=False, grid=False, multi=False, conductivity=5.8e7, terminal_ports=False):
    path=(write_grid if grid else write_model)(directory,maximum,nw,nh,thin,terminal_ports)
    lines=[" ".join(f"sigma={conductivity*1e-6:.12g}" if token=="sigma=58" else token
                    for token in line.split()) for line in path.read_text().splitlines()]
    path.write_text("\n".join(lines)+"\n")
    metadata=json.loads((directory/"model.json").read_text())
    metadata.update(conductivity_S_m=conductivity,terminal_ports=terminal_ports,
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (directory/"model.json").write_text(json.dumps(metadata,indent=2))
    command=["nice","-n","10",str(binary),"-a","off","-m","multi" if multi else "direct","-o","4","-t","1e-8","-c","600",
             "-x","p1",path.name]
    start=time.perf_counter()
    with (directory/"run.log").open("w") as log:
        done=subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
    print(f"{directory}: rc={done.returncode}",flush=True)
    receipt=dict(command=command,returncode=done.returncode,seconds=time.perf_counter()-start,
                 child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (directory/"receipt.json").write_text(json.dumps(receipt,indent=2))
    assert done.returncode==0
    result=parse(directory)
    (directory/"network.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--binary",type=Path,default=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106/tools/fasthenry-build/bin/fasthenry")
    p.add_argument("--maximum",type=float,default=20.)
    p.add_argument("--nw",type=int,default=3)
    p.add_argument("--nh",type=int,default=3)
    p.add_argument("--thin",action="store_true")
    p.add_argument("--grid",action="store_true")
    p.add_argument("--multi",action="store_true")
    p.add_argument("--conductivity",type=float,default=5.8e7)
    p.add_argument("--terminal-ports",action="store_true")
    args=p.parse_args()
    run(args.directory,args.binary,args.maximum,args.nw,args.nh,args.thin,args.grid,args.multi,args.conductivity,args.terminal_ports)

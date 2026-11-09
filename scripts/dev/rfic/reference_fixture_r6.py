"""Independent open-source FDTD reference for the shared F-A shield fixtures.

The installed binary reads generated XML; no bindings or build are needed.
Run outside pytest. Geometry comes from the same public fixture, whereas the
rectilinear field solve and V/I extraction are independent of MPIE numerics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ["OMP_NUM_THREADS"]="8"
os.environ["OPENBLAS_NUM_THREADS"]="8"
os.environ["MKL_NUM_THREADS"]="8"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.support.rfic_fixtures import fixture_fa
from witwin.maxwell.mom.layered.runtime import compile_layered_scene


def box(parent, start, stop, priority=10):
    primitives = parent.find("Primitives")
    if primitives is None:
        primitives = ET.SubElement(parent, "Primitives")
    primitive = ET.SubElement(primitives, "Box", Priority=str(priority))
    for tag, point in (("P1", start), ("P2", stop)):
        ET.SubElement(primitive, tag, **dict(zip("XYZ", map(str, point))))


def graded_axis(core, extent, maximum=150., ratio=1.3):
    core = sorted(set(round(float(x), 8) for x in core))
    low, high = core[0], core[-1]
    first_step = min(np.diff(core))
    for sign in (-1, 1):
        point = low if sign < 0 else high
        step = first_step
        while abs(point) < extent:
            step = min(step*ratio, maximum)
            point += sign*step
            core.append(point)
    return sorted(core)


def write_model(directory, variant, resolution, driven, *, max_time=3e-10, extent=1500., lossy=True,
                z_refinement=1,port_gap_um=5.):
    fixture = fixture_fa(shield=variant, device="cpu",lossy=lossy)
    stack, rectangles, ports, materials, _, vias, via_materials = compile_layered_scene(fixture.scene)
    root = ET.Element("openEMS")
    fdtd = ET.SubElement(root, "FDTD", NumberOfTimesteps="2000000", endCriteria="1e-6",
                         MaxTime=str(max_time), OverSampling="8", f_max="100000000000")
    ET.SubElement(fdtd, "Excitation", Type="0", f0="50000000000", fc="50000000000")
    ET.SubElement(fdtd, "BoundaryCond", **{k: "PML_8" for k in
                  ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")})
    csx = ET.SubElement(root, "ContinuousStructure", CoordSystem="0")
    props = ET.SubElement(csx, "Properties")
    silicon = ET.SubElement(props, "Material", Name="silicon")
    ET.SubElement(silicon, "Property", Epsilon="11.9", Kappa="10")
    box(silicon, (-10000,-10000,-10000), (10000,10000,300), 0)
    oxide = ET.SubElement(props, "Material", Name="oxide")
    ET.SubElement(oxide, "Property", Epsilon="4.1")
    box(oxide, (-10000,-10000,300), (10000,10000,306), 1)
    groups = {}
    for rectangle, (sigma, thickness, _) in zip(rectangles, materials):
        key = (sigma, thickness)
        if key not in groups:
            groups[key] = (ET.SubElement(props, "Metal", Name=f"sheet_{len(groups)}") if sigma is None else
                           ET.SubElement(props, "ConductingSheet", Name=f"sheet_{len(groups)}",
                                         Conductivity=str(sigma), Thickness=str(thickness)))
        x0,x1,y0,y1,z = (round(v*1e6, 8) for v in rectangle)
        box(groups[key], (x0,y0,z), (x1,y1,z))
    conductor = ET.SubElement(props, "Material" if lossy else "Metal", Name="via_bulk")
    if lossy:
        ET.SubElement(conductor, "Property", Epsilon="1", Kappa="58000000")
    for x0,x1,y0,y1,z0,z1 in vias:
        box(conductor, (x0*1e6,y0*1e6,z0*1e6), (x1*1e6,y1*1e6,z1*1e6))
    bridge = ET.SubElement(props, "Metal", Name="ideal_port_bridges")
    gap_left,gap_right=137.5-port_gap_um/2,137.5+port_gap_um/2
    for number, y in enumerate((95.,80.), 1):
        # The finite source gap is centred on the same ideal bridge cut.
        box(bridge, (130,y-5,305), (gap_left,y+5,305), 20)
        box(bridge, (gap_right,y-5,305), (145,y+5,305), 20)
        lump = ET.SubElement(props, "LumpedElement", Name=f"port_r{number}",
                             Direction="0", R="50", Caps="1")
        box(lump, (gap_left,y-5,305), (gap_right,y+5,305), 30)
        if number == driven:
            excitation = ET.SubElement(props, "Excitation", Name=f"drive{number}",
                                       Type="0", Excite="-1,0,0")
            box(excitation, (gap_left,y-5,305), (gap_right,y+5,305), 30)
        voltage = ET.SubElement(props, "ProbeBox", Name=f"port_ut{number}", Type="0", Weight="-1")
        box(voltage, (gap_left,y,305), (gap_right,y,305))
        current = ET.SubElement(props, "ProbeBox", Name=f"port_it{number}", Type="1",
                                Weight="1", NormDir="0")
        box(current, (137.5,y-5,304.25), (137.5,y+5,305.5))
    xy = np.arange(-160.,160.+resolution/2,resolution).tolist()
    # Preserve every feature line before subdivision, including narrow turn gaps.
    xy += [round(v*1e6,8) for r in rectangles for v in r[:4]]
    xy += [round(v*1e6,8) for via in vias for v in via[:4]]
    xy += [-152,-148,148,152,-52,-48,123,127,130,135,137.5,140,145,155,-155]
    edges = sorted(set(xy))
    xy = [v for a,b in zip(edges[:-1],edges[1:])
          for v in np.linspace(a,b,int(np.ceil((b-a)/resolution))+1)[:-1]] + edges[-1:]
    xy = graded_axis(xy, extent)
    # Preserve all primary mesh nodes; add only the new x-directed gap edges.
    x=sorted(set(xy+[gap_left,gap_right]))
    z_edges=[299,300,300.5,301.25,302,303.5,304.25,305,305.5,306,307]
    z_core=[v for a,b in zip(z_edges[:-1],z_edges[1:])
            for v in np.linspace(a,b,z_refinement+1)[:-1]]+z_edges[-1:]
    # Keep exterior nodes and PML entrances fixed during the vertical control.
    z = sorted(set(graded_axis(z_edges,extent)+z_core))
    grid = ET.SubElement(csx, "RectilinearGrid", DeltaUnit="1e-6", CoordSystem="0")
    for name, values in (("XLines",x),("YLines",xy),("ZLines",z)):
        ET.SubElement(grid, name).text = ",".join(map(str,values))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"model.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    metadata = dict(variant=variant, resolution_um=resolution, driven_port=driven,lossy_metal=lossy,
                    max_time_s=max_time, extent_um=extent,z_refinement=z_refinement,grid_shape=[len(x),len(xy),len(z)],
                    fixture_sha256=hashlib.sha256(Path("tests/support/rfic_fixtures.py").read_bytes()).hexdigest(),
                    xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sheet_records=[list(k) for k in groups], via_count=len(vias),
                    port_gap_um=port_gap_um, boundary="PML_8", end_criteria=1e-6,
                    geometry_sha256=hashlib.sha256(json.dumps([rectangles,materials,vias,via_materials],
                        sort_keys=True).encode()).hexdigest())
    (directory/"model.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return path


def run_model(path, tools):
    libraries = [tools/"open-fdtd-install/lib", tools/"open-fdtd-deps/usr/lib/x86_64-linux-gnu",
                 tools/"open-fdtd-deps/usr/lib"]
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(map(str,libraries)), OMP_NUM_THREADS="8")
    args = ["nice", "-n", "10", str(tools/"open-fdtd-install/bin/openEMS"), path.name,
            "--engine=multithreaded", "--numThreads=8", "--dump-statistics", "--exact-endcriteria"]
    # A prior controlled stop must not terminate the next acquisition immediately.
    stop_file=path.parent/"ABORT"
    if stop_file.exists():
        stop_file.unlink()
    with (path.parent/"run.log").open("w") as log:
        rc = subprocess.call(args, cwd=path.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"{path.parent}: rc={rc}", flush=True)
    (path.parent/"run_receipt.json").write_text(json.dumps(dict(returncode=rc,command=args,
        environment={k:env[k] for k in ("LD_LIBRARY_PATH","OMP_NUM_THREADS")}),indent=2)+"\n")
    assert rc == 0


def extract(directories, frequencies, stop_time=None):
    voltage, current, termination_errors = [], [], []
    for driven,directory in enumerate(directories):
        u, i = [], []
        for port in (1,2):
            for prefix, target in (("ut",u),("it",i)):
                samples = np.loadtxt(directory/f"port_{prefix}{port}", comments="%")
                if stop_time is not None:
                    samples = samples[samples[:,0]<=stop_time]
                t, value = samples[:,0], samples[:,1]
                # Positive Fourier sign yields the runtime exp(-i omega t) convention.
                target.append(2*(t[1]-t[0])*(np.exp(2j*np.pi*frequencies[:,None]*t)@value))
        quiet=1-driven
        termination_errors.append(float(np.max(np.abs(u[quiet]/i[quiet]+50)/50)))
        voltage.append(np.stack(u,-1))
        current.append(np.stack(i,-1))
    u = np.stack(voltage,-1)
    i = np.stack(current,-1)
    y = i@np.linalg.inv(u)
    z = np.linalg.inv(y)
    identity = np.broadcast_to(np.eye(2),y.shape)
    s = (identity-50*y)@np.linalg.inv(identity+50*y)
    return dict(frequency_Hz=frequencies.tolist(),quiet_termination_relative_errors=termination_errors,
                y_real=y.real.tolist(),y_imag=y.imag.tolist(),
                z_real=z.real.tolist(),z_imag=z.imag.tolist(),s_real=s.real.tolist(),s_imag=s.imag.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--tools",type=Path,default=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106" / "tools")
    parser.add_argument("--variant",choices=("none","pgs","solid"),default="none")
    parser.add_argument("--resolution",type=float,default=5.)
    parser.add_argument("--max-time",type=float,default=3e-10)
    parser.add_argument("--extent",type=float,default=1500.)
    parser.add_argument("--z-refinement",type=int,choices=(1,2),default=1)
    parser.add_argument("--port-gap",type=float,choices=(2.5,5.),default=5.,metavar="UM")
    parser.add_argument("--extract-only",action="store_true")
    parser.add_argument("--pec-metal",action="store_true",
                        help="Use PEC for winding, ring and vias; retain the declared shield material")
    args = parser.parse_args()
    directories = [args.directory/f"port{p}" for p in (1,2)]
    if not args.extract_only:
        for port,directory in enumerate(directories,1):
            path = write_model(directory,args.variant,args.resolution,port,
                               max_time=args.max_time,extent=args.extent,lossy=not args.pec_metal,
                               z_refinement=args.z_refinement,port_gap_um=args.port_gap)
            run_model(path,args.tools)
    frequencies=np.unique(np.concatenate((np.linspace(.1e9,20e9,400),[1e9,3.4e9])))
    data = extract(directories,frequencies)
    (args.directory/"network.json").write_text(json.dumps(data,indent=2)+"\n")


if __name__ == "__main__":
    main()

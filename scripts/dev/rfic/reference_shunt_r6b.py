"""Open-source FDTD controls for isolated silicon shunt admittance."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from tests.support.rfic_r6b_references import plate_fixture,FREQUENCIES
from witwin.maxwell.mom.layered.runtime import compile_layered_scene
from reference_fixture_r6 import box,graded_axis,run_model

def write_plate(directory,resolution,gap=5.,end_criteria=1e-11,max_time=3e-10):
    scene,_=plate_fixture(device="cpu")
    stack,rectangles,ports,materials,_,vias,via_materials=compile_layered_scene(scene)
    root=ET.Element("openEMS")
    fdtd=ET.SubElement(root,"FDTD",NumberOfTimesteps="2000000",endCriteria=str(end_criteria),
                      MaxTime=str(max_time),OverSampling="8",f_max="100000000000")
    ET.SubElement(fdtd,"Excitation",Type="0",f0="50000000000",fc="50000000000")
    ET.SubElement(fdtd,"BoundaryCond",**{key:"PML_8" for key in
        ("xmin","xmax","ymin","ymax","zmin","zmax")})
    csx=ET.SubElement(root,"ContinuousStructure",CoordSystem="0")
    props=ET.SubElement(csx,"Properties")
    si=ET.SubElement(props,"Material",Name="silicon")
    ET.SubElement(si,"Property",Epsilon="11.9",Kappa="10")
    box(si,(-10000,-10000,-10000),(10000,10000,300),0)
    ox=ET.SubElement(props,"Material",Name="oxide")
    ET.SubElement(ox,"Property",Epsilon="4.1")
    box(ox,(-10000,-10000,300),(10000,10000,306),1)
    metal=ET.SubElement(props,"Metal",Name="metal")
    for x0,x1,y0,y1,z in rectangles:
        box(metal,(round(x0*1e6,8),round(y0*1e6,8),round(z*1e6,8)),
                  (round(x1*1e6,8),round(y1*1e6,8),round(z*1e6,8)))
    left,right=55-gap/2,55+gap/2
    box(metal,(50,-5,305),(left,5,305),20)
    box(metal,(right,-5,305),(60,5,305),20)
    lump=ET.SubElement(props,"LumpedElement",Name="port_r1",Direction="0",R="50",Caps="1")
    box(lump,(left,-5,305),(right,5,305),30)
    drive=ET.SubElement(props,"Excitation",Name="drive1",Type="0",Excite="-1,0,0")
    box(drive,(left,-5,305),(right,5,305),30)
    voltage=ET.SubElement(props,"ProbeBox",Name="port_ut1",Type="0",Weight="-1")
    box(voltage,(left,0,305),(right,0,305))
    current=ET.SubElement(props,"ProbeBox",Name="port_it1",Type="1",Weight="1",NormDir="0")
    box(current,(55,-5,304.25),(55,5,305.5))
    xy=graded_axis(np.unique(np.concatenate((np.arange(-75,75+resolution/2,resolution),
         [-70,-60,-50,-5,0,5,50,60,70]))),1500.)
    x=sorted(set(xy+[left,right,55.]))
    z=graded_axis([299,300,300.5,301.25,302,303.5,304.25,305,305.5,306,307],1500.)
    grid=ET.SubElement(csx,"RectilinearGrid",DeltaUnit="1e-6",CoordSystem="0")
    for name,values in (("XLines",x),("YLines",xy),("ZLines",z)):
        ET.SubElement(grid,name).text=",".join(map(str,values))
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"model.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    from tests.support.rfic_r6_references import geometry_digest
    (directory/"model.json").write_text(json.dumps(dict(resolution_um=resolution,gap_um=gap,
        xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        geometry_sha256=geometry_digest(rectangles,materials,vias,via_materials),
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ("scripts/dev/rfic/reference_shunt_r6b.py","tests/support/rfic_r6b_references.py")},
        ports=[dict(name=p.name,positive=list(p.positive),negative=list(p.negative)) for p in ports],
        conductor_model="zero_volume_PEC",bridge_width_um=10.,stack_payload=asdict(stack),
        grid_shape=[len(x),len(xy),len(z)],end_criteria=end_criteria,max_time_s=max_time,stack=dict(eps_si=11.9,sigma_si=10.,eps_oxide=4.1),
        current_probe=dict(y_min_um=-5,y_max_um=5,z_min_um=304.25,z_max_um=305.5)),indent=2))
    return path

def extract(directory):
    values=[]
    f=np.asarray(FREQUENCIES)
    for prefix in ("ut","it"):
        data=np.loadtxt(directory/f"port_{prefix}1",comments="%")
        t,v=data.T
        values.append(2*(t[1]-t[0])*(np.exp(2j*np.pi*f[:,None]*t)@v))
    y=values[1]/values[0]
    return dict(frequency_Hz=f.tolist(),y_real=y.real.tolist(),y_imag=y.imag.tolist())

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--resolution",type=float,default=5.)
    p.add_argument("--gap",type=float,default=5.)
    p.add_argument("--end-criteria",type=float,default=1e-11)
    p.add_argument("--max-time",type=float,default=3e-10)
    args=p.parse_args()
    path=write_plate(args.directory,args.resolution,args.gap,args.end_criteria,args.max_time)
    run_model(path,Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "rfic-mom-20261106" / "tools")
    data=extract(args.directory)
    (args.directory/"network.json").write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)

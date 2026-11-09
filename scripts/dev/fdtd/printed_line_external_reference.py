"""Independent openEMS physical V/I line transfer, with exact Maxwell mesh.

Two finite resistor feeds lie outside the measurement region. Their matching
and amplitudes do not enter the two-drive physical state transfer. Raw native
E/H point probes use the reference backend's NO_INTERPOLATION field probes;
their actual integer addresses are checked before applying the geometric SI
voltage/current weights. This script never uses a prepared modal impedance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene,WIDTH,HEIGHT,GAP,PAIR_WIDTH,PAIR_OFFSET
from scripts.dev.fdtd.printed_line_physical_modes import coupled_physical_state,coupled_transfer
from scripts.dev.fdtd.pcb_acceptance import package_identity
from witwin.maxwell.constants import C_0
from witwin.maxwell.scene import prepare_scene


def contour_driver(path):
    spec=importlib.util.spec_from_file_location("physical_contours",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def reference_identity(source,package):
    pins={str(path):{"head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=path,text=True).strip(),
        "working_diff_sha256":hashlib.sha256(subprocess.check_output(["git","diff","HEAD","--binary"],cwd=path)).hexdigest()}
        for path in (source,source/"CSXCAD",source/"openEMS")}
    bindings={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(package.__file__).parent.glob("*.so")}
    libraries={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source.parent/"install/lib").glob("*.so*") if path.is_file()}
    return {"pins":pins,"python_package":package.__file__,"binding_sha256":bindings,"library_sha256":libraries}


def export_structures(csx,scene):
    records=[]
    for structure in scene.structures:
        if not structure.enabled:continue
        geometry,material=structure.geometry,structure.material
        if isinstance(geometry,mw.PlanarPEC):
            if geometry.holes or len(geometry.vertices)!=4:
                raise NotImplementedError("Independent line export accepts explicit rectangular PEC supports.")
            normal="xyz".index(geometry.normal_axis);tangent=[i for i in range(3) if i!=normal]
            vertices=geometry.vertices.detach().cpu().numpy()
            bounds=np.zeros((3,2));bounds[normal]=geometry.coordinate
            bounds[tangent,0]=vertices.min(0);bounds[tangent,1]=vertices.max(0)
        elif isinstance(geometry,mw.Box):
            center=geometry.position.detach().cpu().double().numpy();size=geometry.size.detach().cpu().double().numpy()
            bounds=np.column_stack((center-size/2,center+size/2))
        else:raise NotImplementedError("Independent line export requires fixed PEC rectangles and dielectric Boxes.")
        if getattr(material,"is_copper_foil",False):
            raise NotImplementedError("This independent full-wave reference qualifies ideal-conductor lines only.")
        pec=bool(getattr(material,"is_pec",False))
        prop=csx.AddMetal(structure.name) if pec else csx.AddMaterial(structure.name,
            epsilon=float(material.eps_r),kappa=float(material.sigma_e))
        prop.AddBox(start=bounds[:,0],stop=bounds[:,1],priority=10 if pec else 0)
        records.append({"name":structure.name,"bounds_m":bounds.tolist(),"pec":pec,
            "eps_r":None if pec else float(material.eps_r),"sigma_e_S_per_m":None if pec else float(material.sigma_e)})
    return records


def native_addresses(prepared,expected):
    """Field probes address all-primary(E)/all-dual(H) grid indices.

    openEMS then returns each raw vector component at its own Yee offset. A
    component's *physical* position differs from this grid-address coordinate.
    Keeping that distinction avoids an implicit node/cell interpolation.
    """
    records={}
    for name,(component,index) in expected.items():
        dual=component.startswith("H")
        address=[float(getattr(prepared,a+("_half64" if dual else "_nodes64"))[index[i]]) for i,a in enumerate("xyz")]
        component_axis="xyz".index(component[-1].lower())
        physical=[]
        for i,a in enumerate("xyz"):
            half=(i==component_axis) if not dual else (i!=component_axis)
            physical.append(float(getattr(prepared,a+("_half64" if half else "_nodes64"))[index[i]]))
        records[name]={"component":component,"index":list(index),"address_m":address,
                       "physical_component_position_m":physical,"probe_type":3 if dual else 2}
    return records


def read_native_spectra(directory,records,frequencies):
    spectra={};tails={}
    for name,record in records.items():
        path=directory/name
        header="\n".join(path.read_text().splitlines()[:4])
        indices=re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",header)
        if not indices or list(map(int,indices[0]))!=record["index"]:
            raise ValueError(f"Independent native point {name} snapped to a different grid address: {header}")
        values=np.loadtxt(path,comments="%")
        times=values[:,0];field=values[:,1+"xyz".index(record["component"][-1].lower())]
        # Integrate actual recorded E/H timestamps: the backend already writes
        # magnetic half-step times, so no empirical phase correction is needed.
        phase=np.exp(2j*np.pi*np.asarray(frequencies)[:,None]*times)
        spectra[name]=2*np.trapezoid(phase*field,times,axis=-1)
        scale=float(np.max(abs(field)))
        tails[name]=float(np.max(abs(field[-max(2,len(field)//10):]))/scale) if scale else 0.
    return spectra,tails


def reference_line_scene(args):
    """Keep axial spacing and physical outer boundaries separate from local h."""
    extra={} if args.transverse_refinement_radius is None else {
        "transverse_refinement_radius":args.transverse_refinement_radius}
    return printed_line_scene(args.kind,dx=args.dx,longitudinal_dl=args.longitudinal_dl,
        port_separation=args.separation,axial_half_length=args.axial_half_length,
        transverse_padding=args.padding,sigma_e=args.sigma,device="cpu",**extra)


def reference_feeds(kind,separation,balanced=False):
    """Independent physical sources; symmetric grounds share one drive column.

    The reference port orients its electric excitation opposite start→stop.
    Ground→signal arrows therefore give symmetric top/bottom or left/right
    electric fields. Coupled conductors retain four independent columns.
    """
    feeds=[];groups=[]
    for x in (-separation/2,separation/2):
        if kind=='differential_pair':
            for y in (PAIR_OFFSET,-PAIR_OFFSET):
                groups.append([len(feeds)])
                feeds.append(((x,y-PAIR_WIDTH/2,-HEIGHT),(x,y+PAIR_WIDTH/2,0.),'z'))
            continue
        groups.append([len(feeds)])
        if kind=='cpw':
            feeds.append(((x,WIDTH/2+GAP,0.),(x,WIDTH/2,0.),'y'))
            if balanced:
                groups[-1].append(len(feeds))
                feeds.append(((x,-WIDTH/2-GAP,0.),(x,-WIDTH/2,0.),'y'))
        else:
            feeds.append(((x,-WIDTH/2,-HEIGHT),(x,WIDTH/2,0.),'z'))
            if balanced and kind=='stripline':
                groups[-1].append(len(feeds))
                feeds.append(((x,-WIDTH/2,HEIGHT),(x,WIDTH/2,0.),'z'))
    # Preserve the original unbalanced CPW source arrow and byte-identical XML.
    if kind=='cpw' and not balanced:
        feeds=[(stop,start,axis) for start,stop,axis in feeds]
    return feeds,groups


def completed_drive_action(directory,*,reuse_raw=False,reuse_complete=False):
    """Reuse only an auditable completed drive; never append partial traces."""
    directory=Path(directory)
    xml=directory/'simulation.xml';execution=directory/'execution.json'
    if reuse_raw and reuse_complete:
        raise ValueError('Choose one explicit raw-reuse mode.')
    if reuse_raw or reuse_complete and execution.exists():
        if not xml.exists() or not execution.exists():
            raise ValueError('Raw reuse requires both retained XML and execution timing.')
        return 'reuse'
    if reuse_complete and (xml.exists() or (directory/'raw').exists()):
        raise ValueError('An incomplete drive cannot be appended; use a fresh directory without its partial data.')
    return 'run'


def validate_completed_execution(execution,path,dt,steps):
    if (execution['xml_sha256']!=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        or execution['dt_s']!=dt or execution['steps']!=steps
        or not math.isfinite(execution['seconds']) or execution['seconds']<=0):
        raise ValueError('Retained execution identity/timestep/count/timing differs from the requested drive.')


def completed_drive_provenance(directory,current):
    directory=Path(directory);provenance_path=directory/'reuse-provenance.json'
    if not provenance_path.exists():raise ValueError('Completed-drive resumption requires explicit original source/build/driver provenance.')
    provenance=json.loads(provenance_path.read_text())
    for key,value in current.items():
        if provenance.get(key)!=value:raise ValueError('Completed-drive original provenance differs: '+key)
    if provenance.get('execution_sha256')!=hashlib.sha256((directory/'execution.json').read_bytes()).hexdigest():
        raise ValueError('Completed-drive original timing sidecar changed.')
    original_driver=Path(provenance['executed_driver_path'])
    if not original_driver.exists() or hashlib.sha256(original_driver.read_bytes()).hexdigest()!=provenance['executed_driver_sha256']:
        raise ValueError('Completed-drive original executed driver is unavailable or changed.')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("stripline","cpw","microstrip","differential_pair"),default="microstrip")
    parser.add_argument("--balanced-feeds",action="store_true",help="Drive both physical ground gaps coherently for stripline/CPW.")
    parser.add_argument("--dx",type=float,default=.0025)
    parser.add_argument("--longitudinal-dl","--axial-dl",type=float,default=.0025)
    parser.add_argument("--transverse-refinement-radius",type=float)
    parser.add_argument("--padding",type=float,default=.020)
    parser.add_argument("--separation",type=float,default=.40)
    parser.add_argument("--axial-half-length",type=float,default=.30)
    parser.add_argument("--sigma",type=float,default=0.)
    parser.add_argument("--duration",type=float,default=32e-9)
    parser.add_argument("--dt-factor",type=float,default=.4)
    parser.add_argument("--threads",type=int,default=6)
    parser.add_argument("--positions",type=float,nargs="+",default=[-.015,0.,.015])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference-source",type=Path,required=True)
    parser.add_argument("--reuse-raw",action="store_true")
    parser.add_argument('--reuse-complete-drives',action='store_true',
        help='Reuse explicitly proven completed drive directories and execute only fresh missing drives; never append partial traces.')
    parser.add_argument("--prepare-only",action="store_true",help="Write auditable XML/probe contracts without running external FDTD.")
    parser.add_argument("--contour-driver",type=Path)
    args=parser.parse_args()
    if args.reuse_raw and args.reuse_complete_drives:parser.error('Choose one explicit raw-reuse mode.')
    if args.contour_driver is None:
        filename='coupled_contours.py' if args.kind=='differential_pair' else 'physical_contours.py'
        args.contour_driver=Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-22'/filename
    if len(args.positions)<2 or any(b<=a for a,b in zip(args.positions,args.positions[1:])):
        parser.error("Measurement positions must be strictly increasing and contain at least two stations.")
    if max(map(abs,args.positions))>=args.separation/2:
        parser.error("Measurement positions must lie strictly between the independent source planes.")
    if not 0<args.dt_factor<=.9:raise ValueError("dt-factor must be positive and <=0.9 of vacuum CFL.")
    if args.separation<=.06:raise ValueError("Independent feeds must lie strictly outside the measured ROI.")
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    import openEMS as reference_package
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    identity=package_identity(mw,"witwin-maxwell")
    reference_before=reference_identity(args.reference_source,reference_package)
    fixture=Path(inspect.getfile(printed_line_scene));fixture_sha=hashlib.sha256(fixture.read_bytes()).hexdigest()
    helper_sha=hashlib.sha256(args.contour_driver.read_bytes()).hexdigest()
    driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state_helper=Path(inspect.getfile(coupled_transfer))
    state_sha=hashlib.sha256(state_helper.read_bytes()).hexdigest()
    helper=contour_driver(args.contour_driver)
    scene=reference_line_scene(args)
    prepared=prepare_scene(scene);positions=args.positions
    if args.kind=='differential_pair':
        _monitors,integrals,expected=helper.probes(scene,args.dx,args.longitudinal_dl,args.padding,positions,'pec')
    else:
        _monitors,integrals,expected=helper.native_probe_monitors(scene,args.kind,args.dx,args.longitudinal_dl,args.padding,positions,"pec")
    records=native_addresses(prepared,expected)
    nodes=[np.asarray(getattr(prepared,a+"_nodes64")) for a in "xyz"]
    dt=args.dt_factor/(C_0*math.sqrt(sum(float(np.min(np.diff(values)))**-2 for values in nodes)))
    steps=math.ceil(args.duration/dt);frequencies=np.linspace(.8e9,1.2e9,5)
    values=[];runs=[]
    feeds,drive_groups=reference_feeds(args.kind,args.separation,args.balanced_feeds)
    for active,active_feeds in enumerate(drive_groups):
        directory=output/f"drive-{active}";directory.mkdir(exist_ok=True)
        backend=openEMS(NrTS=steps,EndCriteria=0.,TimeStep=dt)
        backend.SetGaussExcite(1e9,.8e9)
        backend.SetBoundaryCond([f"PML_{prepared.boundary.num_layers}"]*6)
        csx=ContinuousStructure();backend.SetCSX(csx);grid=csx.GetGrid();grid.SetDeltaUnit(1.)
        for axis,coordinates in zip("xyz",nodes):grid.AddLine(axis,coordinates)
        geometry=export_structures(csx,scene)
        source_contract=[]
        for number,(start,stop,axis) in enumerate(feeds):
            backend.AddLumpedPort(number+1,50.,start,stop,axis,excite=float(number in active_feeds),priority=20)
            source_contract.append({"reference_impedance_ohm":50.,"start_m":start,"stop_m":stop,
                "voltage_axis":axis,"active":number in active_feeds,"role":"outside-ROI independent drive and passive load"})
        for name,record in records.items():
            probe=csx.AddProbe(name,p_type=record["probe_type"],over_sampling=4)
            probe.AddBox(start=record["address_m"],stop=record["address_m"])
        path=directory/"simulation.xml"
        action=completed_drive_action(directory,reuse_raw=args.reuse_raw,reuse_complete=args.reuse_complete_drives)
        reuse_provenance=None
        if args.reuse_complete_drives and action=='reuse':
            reuse_provenance=completed_drive_provenance(directory,{
                'maxwell_source_sha256':identity['source_sha256'],'fixture_sha256':fixture_sha,
                'contour_driver_sha256':helper_sha,'state_helper_sha256':state_sha,'reference':reference_before})
        if action=='reuse':
            if not path.exists():
                raise ValueError("Raw reuse requires the retained independent XML.")
            requested=directory/"requested.xml"
            try:
                backend.Write2XML(str(requested))
                if requested.read_bytes()!=path.read_bytes():
                    raise ValueError("Independent configuration changed; raw trace reuse is invalid.")
            finally:
                requested.unlink(missing_ok=True)
        else:
            backend.Write2XML(str(path))
        if args.prepare_only:
            declaration={"status":"prepared only; no external field solve executed","kind":args.kind,
                "active_drive":active,"dt_s":dt,"steps":steps,"geometry":geometry,
                "balanced_feeds":args.balanced_feeds,"drive_groups":drive_groups,
                "transverse_refinement_radius_m":args.transverse_refinement_radius,"longitudinal_dl_m":args.longitudinal_dl,
                "grid_nodes_m":{axis:coordinates.tolist() for axis,coordinates in zip("xyz",nodes)},
                "source_contract":source_contract,"probe_contract":records,
                "maxwell_source_sha256":identity["source_sha256"],"fixture_sha256":fixture_sha,
                "reference":reference_before,"contour_driver_sha256":helper_sha,
                "state_helper_sha256":state_sha,"driver_sha256":driver_sha}
            (directory/"preparation.json").write_text(json.dumps(declaration,indent=2)+"\n")
            continue
        execution_path=directory/"execution.json"
        if action=='reuse' and not execution_path.exists():
            raise ValueError("Raw reuse requires retained execution timing; postprocess time cannot replace it.")
        original=Path.cwd()
        try:
            if action=='run':
                start_time=time.perf_counter()
                backend.Run(str(directory/"raw"),cleanup=False,verbose=0,numThreads=args.threads)
                execution_path.write_text(json.dumps({"seconds":time.perf_counter()-start_time,
                    "dt_s":dt,"steps":steps,"xml_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    'executed_driver_sha256':driver_sha},indent=2)+"\n")
        finally:os.chdir(original)
        execution=json.loads(execution_path.read_text())
        validate_completed_execution(execution,path,dt,steps)
        postprocess_start=time.perf_counter()
        spectra,tails=read_native_spectra(directory/"raw",records,frequencies)
        def integrate(rows):return sum(spectra[name]*weight for name,_component,weight in rows)
        if args.kind=='differential_pair':
            values.append(np.asarray([coupled_physical_state(integrate(common[0]),integrate(odd[0]),
                integrate(common[1]),integrate(odd[1])) for common,odd in integrals]))
        else:values.append(np.asarray([(integrate(v),integrate(i)) for v,i in integrals]))
        run={"active_drive":active,"seconds":execution["seconds"],"postprocess_seconds":time.perf_counter()-postprocess_start,"raw_directory":str(directory/"raw"),
             "source_contract":source_contract,"maximum_late_trace_relative":max(tails.values()),"late_trace_relative":tails,
             'reused_completed_drive':action=='reuse','reuse_provenance':reuse_provenance,
             'field_execution_driver_sha256':reuse_provenance['executed_driver_sha256'] if reuse_provenance is not None else execution.get('executed_driver_sha256'),
             'orchestration_driver_sha256':driver_sha}
        runs.append(run);print(json.dumps({k:run[k] for k in ("active_drive","seconds","maximum_late_trace_relative")}),flush=True)
    if args.prepare_only:
        print(json.dumps({"status":"independent drive/probe XML configurations prepared; no field solve","drive_count":len(drive_groups),"output":str(output)}))
        return
    values=np.asarray(values).transpose(3,1,2,0)
    transfers=[coupled_transfer(row[:,:2],row[:,2:],positions) if args.kind=='differential_pair'
        else helper.state_transfer(row[:,0],row[:,1],positions) for row in values]
    record={"schema":"independent-native-line-reference/1","kind":args.kind,
        "configuration":{key:getattr(args,key) for key in ("dx","longitudinal_dl","transverse_refinement_radius","balanced_feeds","padding","separation","axial_half_length","sigma","duration","dt_factor")},
        "frequencies_hz":frequencies.tolist(),"dt_s":dt,"steps":steps,"actual_duration_s":steps*dt,
        "reference":reference_before,"reference_unchanged":reference_before==reference_identity(args.reference_source,reference_package),"geometry":geometry,
        "grid_nodes_m":{axis:coordinates.tolist() for axis,coordinates in zip("xyz",nodes)},
        "probe_contract":records,"positions_m":positions,
        "voltage_V":values[:,:,:2] if args.kind=='differential_pair' else values[:,:,0],
        "current_A":values[:,:,2:] if args.kind=='differential_pair' else values[:,:,1],
        "state_transfer":transfers,"runs":runs,"phasor_convention":"direct integral of raw real E/H against exp(+i omega t), using actual native timestamps",
        "maxwell_source_sha256":identity["source_sha256"],"maxwell_source_sha256_after_run":package_identity(mw,"witwin-maxwell")["source_sha256"],
        "fixture_sha256":fixture_sha,"fixture_unchanged":fixture_sha==hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "contour_driver_sha256":helper_sha,"helper_unchanged":helper_sha==hashlib.sha256(args.contour_driver.read_bytes()).hexdigest(),
        "state_helper_sha256":state_sha,"state_helper_unchanged":state_sha==hashlib.sha256(state_helper.read_bytes()).hexdigest(),
        "driver_sha256":driver_sha,"driver_unchanged":driver_sha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"independent full-wave actual field transfer; no modal/port characteristic impedance enters V/I extraction"}
    record["source_unchanged"]=record["maxwell_source_sha256"]==record["maxwell_source_sha256_after_run"]
    (output/"report.json").write_text(json.dumps(helper.encode(record),indent=2,allow_nan=False)+"\n")
    if not all(record[key] for key in ("source_unchanged","fixture_unchanged","helper_unchanged","state_helper_unchanged","driver_unchanged","reference_unchanged")):
        raise RuntimeError("Reference campaign source identity changed; raw data retained but physical qualification rejected.")


if __name__=="__main__":main()

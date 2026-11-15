"""Freeze compact F-B independent-reference evidence after all acquisitions succeed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from tests.support.rfic_fixtures import fixture_fb
from tests.support.rfic_fb_references import winding_metrics,unpack,reference_checks,observed_grid_order
from tests.support.rfic_r6_references import geometry_digest
from witwin.maxwell.mom.layered.runtime import compile_layered_scene

QS=("qs-10-1-3","qs-5-1-3","qs-5-2-3","qs-5-1-5","qs-2p5-1-3",
    "qs-thin-5-1-3","qs-thin-2p5-1-3","qs-5-1-3-multi","qs-2p5-1-3-order8","qs-5-1-3-long2")
FDTD=("fdtd-10","fdtd-5","fdtd-2p5","fdtd-boundary","fdtd-z","fdtd-gap")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory):
    return {str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name!="ABORT"}



def resources(path):
    fields=dict(line.strip().split(": ",1) for line in path.read_text().splitlines() if ": " in line)
    return dict(max_rss_KiB=int(fields["Maximum resident set size (kbytes)"]),
        elapsed_wall=fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"],
        user_CPU_s=float(fields["User time (seconds)"]),
        system_CPU_s=float(fields["System time (seconds)"]))


def observation(directory):
    log=(directory/"run.log").read_text()
    progress=[line for line in log.splitlines() if "Timestep:" in line]
    samples=np.loadtxt(directory/"port_ut1",comments="%")
    grid=ET.parse(directory/"model.xml").find(".//RectilinearGrid")
    lines={axis:np.fromstring(grid.find(axis+"Lines").text,sep=",") for axis in ("X","Y","Z")}
    bounds={axis:[float(values[8]),float(values[-9])] for axis,values in lines.items()}
    clearance=dict(x=min(-245-bounds["X"][0],bounds["X"][1]-245),
        y=min(-245-bounds["Y"][0],bounds["Y"][1]-245),
        top=bounds["Z"][1]-305,bottom=302-bounds["Z"][0])
    alignment=[bool(np.any(lines["Y"]==y)) for y in (170.,70.,145.,45.)]
    return dict(probe_y_on_grid=alignment,pml_inner_bounds_um=bounds,device_to_pml_clearance_um=clearance,timestep_s=float(re.search(r"FDTD timestep is: (\S+) s",log).group(1)),
        maximum_steps=int(re.search(r"Max. number of timesteps: (\d+)",log).group(1)),
        actual_last_probe_time_s=float(samples[-1,0]),last_progress_line=progress[-1])

def peak_resonance(f,l,q):
    f,l,q=map(np.asarray,(f,l,q))
    positive=np.flatnonzero(l>0)
    peak=positive[np.argmax(q[positive])]
    crossings=np.flatnonzero((l[:-1]>0)&(l[1:]<=0))
    resonances=[]
    for i in crossings:
        x0,x1=-2*np.pi*f[i]*l[i],-2*np.pi*f[i+1]*l[i+1]
        resonances.append(float(f[i]+(f[i+1]-f[i])*(-x0)/(x1-x0)))
    return dict(Q_peak=float(q[peak]),peak_frequency_Hz=float(f[peak]),
        self_resonance_Hz=resonances[0] if resonances else None,
        self_resonance_brackets_Hz=[[float(f[i]),float(f[i+1])] for i in crossings])


def collect(root,destination):
    evidence=dict(fixture="F-B",geometry_sha256={},quasi_static={},fdtd={},
        port_convention="p1-p2 primary, p3-p4 secondary; signal positive, ring negative; 50 ohm; exp(-i omega t)",
        winding_extraction="D=((1,0),(-1,0),(0,1),(0,-1)); Zw=D.T Z D, zero common-mode currents; no deembedding",
        accuracy_targets=dict(L_relative=.02,Q_peak_relative=.05,S_magnitude_absolute=.01,
                              S_phase_degrees=1.,k_absolute=.01),
        S_qualification_band_Hz=[1e8,3e10],
        Q_qualification_scope="each winding peak below its first self-resonance",
        k_qualification_frequencies_Hz=[1e8,1e9],
        L_qualification_frequencies_Hz=[1e8,1e9],L_qualification_quantities=["L1_H","L2_H","M_H"],
        k_target_reason="0.01 absolute resolves coupling changes about 1.5% at k near 0.65 and must exceed measured reference uncertainty",
        native_full_qualification="pending_dense_memory_task",
        reference_status=dict(quasi_static="finite_discretization_spreads_reported",
            fdtd="not_converged_at_qualification_targets"),
        stack=dict(silicon_z_um=[0,300],silicon_eps_r=11.9,silicon_sigma_S_m=10.,
            oxide_z_um=[300,306],oxide_eps_r=4.1,bottom="matching silicon half-space",top="vacuum"),
        reference_models=dict(quasi_static="orthogonal resolved-copper volume graph in vacuum",
            fdtd="conducting sheets and bulk conductive vias on the lossy stack"))
    for lossy in (False,True):
        _,r,_,m,_,v,vm=compile_layered_scene(fixture_fb(lossy=lossy).scene)
        evidence["geometry_sha256"]["Cu" if lossy else "PEC"]=geometry_digest(r,m,v,vm)
    evidence["fdtd_geometry_raw_sha256"]=hashlib.sha256(
        json.dumps([r,m,v,vm],sort_keys=True).encode()).hexdigest()
    _,r,_,m,_,v,vm=compile_layered_scene(fixture_fb(thin=True).scene)
    evidence["geometry_sha256"]["Cu_thin"]=geometry_digest(r,m,v,vm)
    for name in QS:
        directory=root/name
        receipt=json.loads((directory/"receipt.json").read_text())
        assert receipt["returncode"]==0
        assert "Convergence not reached" not in (directory/"run.log").read_text()
        assert receipt["input_sha256"]==digest(directory/"model.inp")
        assert receipt["log_sha256"]==digest(directory/"run.log")
        rows=json.loads((directory/"network.json").read_text())
        model=json.loads((directory/"model.json").read_text())
        assert model["geometry_sha256"]==evidence["geometry_sha256"]["Cu_thin" if model["thin"] else "Cu"]
        # Keep every acquired frequency and the complete 4x2 direct admittance.
        evidence["quasi_static"][name]=dict(table=rows,receipt=receipt,
            model=model,raw_sha256=hashes(directory))
    directory=root/"qs-preconditioner-seg"
    receipt=json.loads((directory/"receipt.json").read_text())
    assert receipt["returncode"]==0
    assert receipt["input_sha256"]==digest(directory/"model.inp")
    row=json.loads((directory/"network.json").read_text())[0]
    baseline_row=next(r for r in evidence["quasi_static"]["qs-5-1-3-multi"]["table"] if r["frequency_Hz"]==1e9)
    evidence["preconditioner_control"]=dict(receipt=receipt,table=[row],
        relative_changes={key:abs(row[key]/baseline_row[key]-1) for key in ("L1_H","L2_H","M_H","R1_ohm","R2_ohm")},
        k_absolute_change=abs(row["k"]-baseline_row["k"]),raw_sha256=hashes(directory))
    for name in FDTD[:3]:
        directory=root/name
        network=json.loads((directory/"network.json").read_text())
        f=np.asarray(network["frequency_Hz"])
        values=winding_metrics(f,unpack(network,"z"))
        reference_metrics={str(w):peak_resonance(f,values[f"L{w}_H"],values[f"Q{w}"]) for w in (1,2)}
        receipts=[]
        models=[]
        usage=[]
        observations=[]
        for p in range(1,5):
            driven=directory/f"port{p}"
            receipt=json.loads((driven/"run_receipt.json").read_text())
            assert receipt["returncode"]==0
            assert receipt["input_sha256"]==digest(driven/"model.xml")
            assert receipt["log_sha256"]==digest(driven/"run.log")
            receipts.append(receipt)
            model=json.loads((driven/"model.json").read_text())
            assert model["geometry_sha256"]==evidence["fdtd_geometry_raw_sha256"]
            xml=ET.parse(driven/"model.xml")
            assert xml.find(".//Material[@Name='silicon']/Property").get("Epsilon")=="11.9"
            assert xml.find(".//Material[@Name='silicon']/Property").get("Kappa")=="10"
            assert xml.find(".//Material[@Name='oxide']/Property").get("Epsilon")=="4.1"
            models.append(model)
            usage.append(resources(driven/"resources.txt"))
            observations.append(observation(driven))
        target=np.array((.1,1.,2.,3.4,5.,8.,10.,12.,15.,18.,20.,25.,30.))*1e9
        selection=sorted(set(np.argmin(abs(f[:,None]-target[None,:]),axis=0).tolist()))
        samples=dict(frequency_Hz=f[selection].tolist(),
            s_real=np.asarray(network["s_real"])[selection].tolist(),
            s_imag=np.asarray(network["s_imag"])[selection].tolist())
        table={key:[data[i] for i in selection] for key,data in values.items()}
        table["shunt_G_S"]=unpack(network,"y").real.sum(axis=(-2,-1))[selection].tolist()
        evidence["fdtd"][name]=dict(samples=samples,table=table,metrics=reference_metrics,
            checks=reference_checks(network),receipts=receipts,models=models,
            resources=usage,observations=observations,raw_sha256=hashes(directory),
            quiet_termination_relative_errors=network["quiet_termination_relative_errors"])
    from scripts.dev.rfic.reference_fixture_fb import extract
    evidence["fdtd_temporal"]={}
    for name in FDTD[:3]:
        windows={}
        for horizon in (240e-12,270e-12):
            network=extract([root/name/f"port{p}" for p in range(1,5)],
                            np.array((1e9,3.4e9)),stop_time=horizon)
            values=winding_metrics(network["frequency_Hz"],unpack(network,"z"))
            values["shunt_G_S"]=unpack(network,"y").real.sum(axis=(-2,-1)).tolist()
            windows[str(horizon)]=values
        evidence["fdtd_temporal"][name]=windows
    evidence["fdtd_controls"]={}
    from scripts.dev.rfic.reference_fixture_fb import extract_column
    for name in FDTD[3:]:
        directory=root/name
        network=json.loads((directory/"network.json").read_text())
        frequencies=np.asarray(network["frequency_Hz"])
        baseline_column=extract_column(root/"fdtd-5"/"port1",frequencies)
        column=unpack(network,"loaded_column")
        original=unpack(baseline_column,"loaded_column")
        driven=directory/"port1"
        receipt=json.loads((driven/"run_receipt.json").read_text())
        assert receipt["returncode"]==0
        assert receipt["input_sha256"]==digest(driven/"model.xml")
        assert receipt["log_sha256"]==digest(driven/"run.log")
        assert json.loads((driven/"model.json").read_text())["geometry_sha256"]==evidence["fdtd_geometry_raw_sha256"]
        evidence["fdtd_controls"][name]=dict(
            max_loaded_column_absolute_change=float(np.max(abs(column-original))),
            quiet_incoming_relative=network["quiet_incoming_relative"],
            receipt=receipt,model=json.loads((driven/"model.json").read_text()),
            resources=resources(driven/"resources.txt"),observation=observation(driven),raw_sha256=hashes(directory),
            convention="Drive p1, terminate all four ports with 50 ohm; compare direct loaded-wave responses, not a full S matrix")
    evidence["quasi_static_convergence"]={}
    comparisons=dict(grid_10_to_5=("qs-10-1-3","qs-5-1-3"),
        grid_5_to_2p5=("qs-5-1-3","qs-2p5-1-3"),
        width_filaments=("qs-5-1-3-multi","qs-5-2-3"),
        height_filaments=("qs-5-1-3-multi","qs-5-1-5"),
        longitudinal=("qs-5-1-3-multi","qs-5-1-3-long2"),
        direct_vs_accelerated=("qs-5-1-3","qs-5-1-3-multi"),
        order_6_to_8=("qs-2p5-1-3","qs-2p5-1-3-order8"),
        thin_grid=("qs-thin-5-1-3","qs-thin-2p5-1-3"))
    for label,(coarse,fine) in comparisons.items():
        rows=[]
        for frequency in (1e6,1e8,1e9):
            a=next(row for row in evidence["quasi_static"][coarse]["table"] if row["frequency_Hz"]==frequency)
            b=next(row for row in evidence["quasi_static"][fine]["table"] if row["frequency_Hz"]==frequency)
            changes={key:abs(a[key]/b[key]-1) for key in ("L1_H","L2_H","M_H","R1_ohm","R2_ohm")}
            changes["k_absolute"]=abs(a["k"]-b["k"])
            rows.append(dict(frequency_Hz=frequency,changes=changes))
        evidence["quasi_static_convergence"][label]=dict(coarse=coarse,fine=fine,rows=rows)
    baseline=[evidence["fdtd"][name] for name in FDTD[:3]]
    convergence={}
    for key in ("L1_H","L2_H","M_H","k","R1_ohm","R2_ohm","shunt_G_S"):
        query=3.4e9 if key=="shunt_G_S" else 1e9
        values=[c["table"][key][c["table"]["frequency_Hz"].index(query)] for c in baseline]
        convergence[key]=dict(frequency_Hz=query,values=values,**observed_grid_order(values))
    evidence["S_mesh_spreads"]={}
    for coarse,fine in zip(FDTD[:2],FDTD[1:3]):
        a=json.loads((root/coarse/"network.json").read_text())
        b=json.loads((root/fine/"network.json").read_text())
        assert a["frequency_Hz"]==b["frequency_Hz"]
        sa,sb=unpack(a,"s"),unpack(b,"s")
        phase=np.abs(np.angle(sa*np.conj(sb),deg=True))
        index=np.unravel_index(np.argmax(phase),phase.shape)
        evidence["S_mesh_spreads"][coarse+"_to_"+fine]=dict(
            max_complex_absolute=float(np.max(np.abs(sa-sb))),
            max_magnitude_absolute=float(np.max(np.abs(abs(sa)-abs(sb)))),
            max_phase_degrees=float(phase[index]),phase_max_frequency_Hz=a["frequency_Hz"][index[0]],
            phase_max_ports=[int(index[1]+1),int(index[2]+1)],
            min_magnitude_at_phase_max=float(min(abs(sa[index]),abs(sb[index]))))
    evidence["fdtd_convergence"]=convergence
    evidence["fdtd_finest_grid_changes"]=dict(
        relative={key:abs(value["values"][-2]/value["values"][-1]-1)
                  for key,value in convergence.items() if key!="k"},
        k_absolute=abs(convergence["k"]["values"][-2]-convergence["k"]["values"][-1]))
    spreads={}
    for label in ("grid_5_to_2p5","width_filaments","height_filaments","longitudinal","order_6_to_8","thin_grid"):
        rows=evidence["quasi_static_convergence"][label]["rows"]
        spreads[label]=dict(
            L_relative=max(row["changes"][key] for row in rows for key in ("L1_H","L2_H","M_H")),
            k_absolute=max(row["changes"]["k_absolute"] for row in rows),
            R_relative=max(row["changes"][key] for row in rows for key in ("R1_ohm","R2_ohm")))
    s_spread=evidence["S_mesh_spreads"]["fdtd-5_to_fdtd-2p5"]
    evidence["reference_roles"]=dict(
        quasi_static=dict(role="PRIMARY",quantities=["L1_H","L2_H","M_H","k","R1_ohm","R2_ohm"],
            scope="Resolved copper in vacuum at its acquired frequencies; measured finite discretizations.",
            discretization_spreads=spreads,R_relative_target=.03,
            R_spread_exceeds_target=spreads["width_filaments"]["R_relative"]>.03,
            R_note="Winding R is primary, but its width-filament spread exceeds the 3% R target."),
        fdtd=dict(role="EVIDENCE-ONLY",quantities=["S","Q","SRF","shunt_G_S"],
            status="not_converged",not_targets_for=["L1_H","L2_H","M_H","k"],
            finest_grid_changes=evidence["fdtd_finest_grid_changes"],
            max_S_magnitude_absolute_change=s_spread["max_magnitude_absolute"],
            max_S_phase_degrees=s_spread["max_phase_degrees"],
            self_L_observed_orders=[convergence[key]["observed_order"] for key in ("L1_H","L2_H")]),
        limit_statement="No Richardson or other extrapolated limit is claimed for either reference.")
    evidence["native_meshes"]=json.loads((root/"counts.json").read_text())
    evidence["source_sha256"]={str(p):digest(p) for p in map(Path,(
        "tests/support/rfic_fixtures.py","tests/support/rfic_fb_references.py",
        "scripts/dev/rfic/reference_fixture_fb.py","scripts/dev/rfic/inductance_fixture_fb.py",
        "scripts/dev/rfic/reference_fixture_r6.py","tests/support/rfic_r6_references.py",
        "scripts/dev/rfic/summarize_fixture_fb.py"))}
    evidence_root=Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence"))
    serialized=json.dumps(evidence,separators=(",",":"),allow_nan=False)
    for prefix in (str(evidence_root.resolve()),str(evidence_root)):
        serialized=serialized.replace(prefix.rstrip("/")+"/","<evidence-root>/")
    destination.write_text(serialized+"\n")
    assert destination.stat().st_size<200_000
    print(destination,destination.stat().st_size,"bytes",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--destination",type=Path,
        default=Path("tests/mom/layered/data/fixture_fb_reference.json"))
    args=parser.parse_args()
    os.nice(10)
    collect(args.directory,args.destination)

"""Summarize measured R6 evidence and freeze small independent reference samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

os.environ["OMP_NUM_THREADS"]="8"
os.environ["OPENBLAS_NUM_THREADS"]="8"
os.environ["MKL_NUM_THREADS"]="8"

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.dev.rfic.reference_fixture_r6 import extract
from tests.support.rfic_r6_references import fit_pi,metrics,pi_attribution,pi_terms,pi_admittance,unpack_network,geometry_digest

SCALARS=("L_1GHz_H","Q_peak","peak_frequency_Hz","self_resonance_Hz")
TARGETS=dict(L_1GHz_H=.03,Q_peak=.10,peak_frequency_Hz=.10,self_resonance_Hz=.05)


def scalars(data):
    return {key:data[key] for key in SCALARS}


def relative(left,right):
    return {key:None if left[key] is None or right[key] is None else abs(left[key]/right[key]-1)
            for key in SCALARS}


def interpolate(frequencies,y,samples):
    result=np.empty((len(samples),2,2),complex)
    for i in range(2):
        for j in range(2):
            result[:,i,j]=np.interp(samples,frequencies,y[:,i,j].real)+1j*np.interp(samples,frequencies,y[:,i,j].imag)
    return result


def collect_mpie(directory):
    groups={}
    for path in sorted(directory.glob("*.json")):
        data=json.loads(path.read_text())
        key="{}/{}".format(data["variant"],data["mesh"])
        groups.setdefault(key,[]).append((path,data))
    result={}
    for key,parts in groups.items():
        values={}
        for _,data in parts:
            values.update(zip(data["frequency_Hz"],unpack_network(data)))
        f=np.array(sorted(values))
        y=np.stack([values[x] for x in f])
        result[key]=dict(metrics=metrics(f,y),pi_fit=fit_pi(f,y),frequency_Hz=f.tolist(),
                         y_real=y.real.tolist(),y_imag=y.imag.tolist(),
                         sources=[str(path) for path,_ in parts],unknowns=parts[0][1]["unknowns"])
    return result


def xml_geometry_digest(path):
    props=ET.parse(path).getroot().find("ContinuousStructure/Properties")
    rectangles,materials,vias,via_materials=[],[],[],[]
    for prop in props.findall("ConductingSheet"):
        material=(float(prop.attrib["Conductivity"]),float(prop.attrib["Thickness"]),None)
        for primitive in prop.findall("Primitives/Box"):
            a,b=primitive.find("P1").attrib,primitive.find("P2").attrib
            rectangles.append(tuple(float(v)*1e-6 for v in (a["X"],b["X"],a["Y"],b["Y"],a["Z"])))
            materials.append(material)
    for prop in props.findall("Material"):
        if prop.attrib["Name"]=="via_bulk":
            for primitive in prop.findall("Primitives/Box"):
                a,b=primitive.find("P1").attrib,primitive.find("P2").attrib
                vias.append(tuple(float(v)*1e-6 for v in (a["X"],b["X"],a["Y"],b["Y"],a["Z"],b["Z"])))
                via_materials.append((float(prop.find("Property").attrib["Kappa"]),0.,None))
    return geometry_digest(rectangles,materials,vias,via_materials)


def collect_reference(directory):
    data=json.loads((directory/"network.json").read_text())
    f=np.unique(np.concatenate((np.array(data["frequency_Hz"]),[1e9,3.4e9])))
    data=extract([directory/f"port{i}" for i in (1,2)],f)
    (directory/"network.json").write_text(json.dumps(data,indent=2)+"\n")
    y=unpack_network(data)
    props=ET.parse(directory/"port1/model.xml").getroot().find("ContinuousStructure/Properties")
    media={p.attrib["Name"]:p for p in props.findall("Material")}
    result=dict(metrics=metrics(f,y),sources=[],dielectrics=dict(
        silicon_epsilon_r=float(media["silicon"].find("Property").attrib["Epsilon"]),
        silicon_sigma_S_m=float(media["silicon"].find("Property").attrib["Kappa"]),
        oxide_epsilon_r=float(media["oxide"].find("Property").attrib["Epsilon"]),
        silicon_top_um=float(media["silicon"].find("Primitives/Box/P2").attrib["Z"]),
        oxide_top_um=float(media["oxide"].find("Primitives/Box/P2").attrib["Z"])))
    bridge=next(p for p in props.findall("Metal") if p.attrib["Name"]=="ideal_port_bridges")
    result["ports"]=[]
    for port in (1,2):
        probe=next(p for p in props.findall("ProbeBox") if p.attrib["Name"]==f"port_ut{port}")
        resistor=next(p for p in props.findall("LumpedElement") if p.attrib["Name"]==f"port_r{port}")
        port_y,port_z=(float(probe.find("Primitives/Box/P1").attrib[a]) for a in ("Y","Z"))
        pieces=[b for b in bridge.findall("Primitives/Box")
                if float(b.find("P1").attrib["Y"])<port_y<float(b.find("P2").attrib["Y"])]
        left=min(float(b.find("P1").attrib["X"]) for b in pieces)
        right=max(float(b.find("P2").attrib["X"]) for b in pieces)
        result["ports"].append(dict(name=f"p{port}",terminal_pair_um=[[left,port_y,port_z],[right,port_y,port_z]],
                                    axis="x",reference_impedance=float(resistor.attrib["R"])))
    windows={}
    for stop in (240e-12,270e-12,300e-12):
        subset=extract([directory/f"port{i}" for i in (1,2)],f,stop_time=stop)
        windows[str(stop)]=scalars(metrics(f,unpack_network(subset)))
    result["quiet_termination_relative_errors"]=extract(
        [directory/f"port{i}" for i in (1,2)],f)["quiet_termination_relative_errors"]
    result["temporal_windows"]=windows
    result["temporal_spread_240_to_300ps"]=relative(windows[str(300e-12)],windows[str(240e-12)])
    result["temporal_spread_270_to_300ps"]=relative(windows[str(300e-12)],windows[str(270e-12)])
    s=np.array(data["s_real"])+1j*np.array(data["s_imag"])
    result["max_singular_S"]=float(np.linalg.svd(s,compute_uv=False)[:,0].max())
    result["max_relative_Y_nonreciprocity"]=float((np.linalg.norm(y-y.transpose(0,2,1),axis=(1,2))/
                                                    np.linalg.norm(y,axis=(1,2))).max())
    for port in (1,2):
        path=directory/f"port{port}"
        metadata=json.loads((path/"model.json").read_text())
        log=(path/"run.log").read_text()
        probe=np.loadtxt(path/"port_ut1",comments="%")
        metadata.update(observed_time_s=float(probe[-1,0]),
                        solver_revision=re.search(r"64bit -- version (\w+)",log).group(1),
                        geometry_library_revision=re.search(r"CSXCAD -- Version: (\w+)",log).group(1),
                        requested_decay_reached="before the end-criteria" not in log,
                        log_sha256=hashlib.sha256(log.encode()).hexdigest(),
                        input_xml=str(path/"model.xml"),log=str(path/"run.log"))
        matches=re.findall(r"Energy: ~([0-9.e+-]+) \((-?[ 0-9.]+)dB\)",log)
        metadata["last_logged_energy_dB"]=float(matches[-1][1].replace(" ",""))
        metadata["declared_geometry_sha256"]=xml_geometry_digest(path/"model.xml")
        result["sources"].append(metadata)
    assert result["sources"][0]["declared_geometry_sha256"]==result["sources"][1]["declared_geometry_sha256"]
    result["geometry_sha256"]=result["sources"][0]["declared_geometry_sha256"]
    samples=np.array([1e9,2e9,3e9,3.4e9,4e9,6e9,10e9,12e9,14e9,16e9,20e9])
    sampled=unpack_network(extract([directory/f"port{i}" for i in (1,2)],samples))
    result["samples"]=dict(frequency_Hz=samples.tolist(),y_real=sampled.real.tolist(),y_imag=sampled.imag.tolist())
    return result


def freeze_native(root,mpie,conductance):
    from tests.support.rfic_fixtures import fixture_fa
    from witwin.maxwell.mom.layered.runtime import compile_layered_scene
    source_paths=sorted(p for p in Path("witwin/maxwell/mom/layered").rglob("*")
                        if p.suffix in (".py",".cu",".cuh",".cpp",".h"))
    source_paths += [Path("witwin/maxwell/mom")/name for name in ("open_surface.py","topology.py")]
    source_paths += [Path("witwin/maxwell")/name for name in
                     ("scene.py","simulation.py","network.py","network_math.py","constants.py","media.py","layers.py","ports.py")]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    analysis_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                     ("tests/support/rfic_fa_references.py","tests/support/rfic_r6_references.py")}
    evidence=dict(solver_source_sha256=hashes,analysis_source_sha256=analysis_hashes,
                  coarse={},finest={},pi_model_source_mesh="shield_fine",
                  pi_effective_shunt_conductance_3p4GHz_S=conductance)
    for variant in ("none","pgs","solid"):
        for level,mesh in (("coarse","shield"),("finest","full" if variant=="none" else "shield_fine")):
            data=mpie[f"{variant}/{mesh}"]
            fixture=fixture_fa(shield=variant,mesh=mesh,device="cpu")
            _,rectangles,ports,materials,_,vias,via_materials=compile_layered_scene(fixture.scene)
            producers=[json.loads(Path(path).read_text()) for path in data["sources"]]
            for producer in producers:
                for name,digest in producer["source_sha256"].items():
                    if name!="tests/support/rfic_fixtures.py":
                        assert hashes[name]==digest
            evidence[level][variant]=dict(metrics=scalars(data["metrics"]),mesh=mesh,
                unknowns=data["unknowns"],config=fixture.config.layered.payload(),
                ports=[dict(name=p.name,terminal_pair_um=[[round(float(v)*1e6,6) for v in point]
                           for point in (p.positive,p.negative)],axis=p.voltage_path.axis,
                           reference_impedance=p.reference_impedance) for p in ports],
                geometry_sha256=geometry_digest(rectangles,materials,vias,via_materials),
                producer_fixture_sha256=sorted({x["source_sha256"]["tests/support/rfic_fixtures.py"] for x in producers}),
                sources=data["sources"])
    Path("tests/mom/layered/data/fixture_r6_native_evidence.json").write_text(
        json.dumps(evidence,indent=2)+"\n")


def summarize(root):
    mpie=collect_mpie(root/"mpie")
    reference={f"{variant}/{resolution}":collect_reference(root/f"reference-{variant}-{resolution}")
               for variant in ("none","pgs","solid") for resolution in (10,5)}
    for variant in ("none","pgs","solid"):
        assert reference[f"{variant}/10"]["geometry_sha256"]==reference[f"{variant}/5"]["geometry_sha256"]
    resolution_spreads={v:relative(reference[f"{v}/10"]["metrics"],reference[f"{v}/5"]["metrics"])
                        for v in ("none","pgs","solid")}
    comparisons={}
    for variant,mesh in (("none","full"),("pgs","shield_fine"),("solid","shield_fine")):
        ours=mpie[f"{variant}/{mesh}"]["metrics"]
        ref=reference[f"{variant}/5"]["metrics"]
        errors=relative(ours,ref)
        comparisons[variant]=dict(relative_errors=errors,
            strict_target_pass={k:errors[k] is not None and errors[k]<=TARGETS[k] for k in TARGETS},
            reference_resolution_spread=resolution_spreads[variant],
            mesh_spread=relative(mpie[f"{variant}/shield"]["metrics"],ours))
    attribution,common_fits={},{}
    for mesh in ("shield","shield_fine"):
        groups={v:mpie[f"{v}/{mesh}"] for v in ("none","pgs","solid")}
        common=sorted(set.intersection(*(set(d["frequency_Hz"]) for d in groups.values())))
        fitted={}
        for variant,data in groups.items():
            values=dict(zip(data["frequency_Hz"],unpack_network(data)))
            fitted[variant]=fit_pi(common,np.stack([values[f] for f in common]))
        common_fits[mesh]=fitted
        band=fitted["none"]["sampled_fit_band_Hz"]
        for variant in ("pgs","solid"):
            attribution[f"{variant}/{mesh}"]=pi_attribution(
                np.linspace(*band,2000),fitted["none"],fitted[variant])
    effects={}
    base=mpie["none/full"]["metrics"]
    for variant in ("pgs","solid"):
        ours=mpie[f"{variant}/shield_fine"]["metrics"]
        deltas={resolution:reference[f"{variant}/{resolution}"]["metrics"]["Q_peak"]-
                reference[f"none/{resolution}"]["metrics"]["Q_peak"] for resolution in (5,10)}
        delta=ours["Q_peak"]-base["Q_peak"]
        allowance=abs(deltas[10]-deltas[5])+.1*abs(deltas[5])
        effects[variant]=dict(native_delta_Q=delta,reference_delta_Q=deltas,
            reference_delta_Q_spatial_spread=abs(deltas[10]-deltas[5]),
            absolute_delta_Q_mismatch=abs(delta-deltas[5]),magnitude_allowance=allowance,
            same_direction=bool(np.sign(delta)==np.sign(deltas[5])),
            magnitude_pass=abs(delta-deltas[5])<=allowance,
            native_fractional_Q_change=ours["Q_peak"]/base["Q_peak"]-1,
            reference_fractional_Q_change=reference[f"{variant}/5"]["metrics"]["Q_peak"]/
                                          reference["none/5"]["metrics"]["Q_peak"]-1,
            native_fractional_L_change=ours["L_1GHz_H"]/base["L_1GHz_H"]-1,
            reference_fractional_L_change=reference[f"{variant}/5"]["metrics"]["L_1GHz_H"]/
                                          reference["none/5"]["metrics"]["L_1GHz_H"]-1)
    conductance={}
    for variant in ("none","pgs","solid"):
        parameters=list(common_fits["shield_fine"][variant]["parameters"].values())
        conductance[variant]=float(pi_terms(pi_admittance(np.array([3.4e9]),parameters))[0,1:].real.sum())
    gates={**{f"A_{key}":value for key,value in comparisons["none"]["strict_target_pass"].items()},
        "B_pgs_Q_increases_native":effects["pgs"]["native_delta_Q"]>0,
        "B_pgs_Q_increases_reference":effects["pgs"]["reference_delta_Q"][5]>0,
        "B_pgs_L_change_below_2pct_native":abs(effects["pgs"]["native_fractional_L_change"])<.02,
        "B_pgs_L_change_below_2pct_reference":abs(effects["pgs"]["reference_fractional_L_change"])<.02,
        "B_pgs_same_direction":effects["pgs"]["same_direction"],
        "B_pgs_delta_Q_magnitude":effects["pgs"]["magnitude_pass"],
        "B_fitted_effective_shunt_conductance_reduced":conductance["pgs"]<conductance["none"],
        "B_solid_L_and_Q_drop_native":effects["solid"]["native_fractional_L_change"]<0 and
                                         effects["solid"]["native_delta_Q"]<0,
        "B_solid_L_and_Q_drop_reference":effects["solid"]["reference_fractional_L_change"]<0 and
                                            effects["solid"]["reference_delta_Q"][5]<0}
    result=dict(mpie=mpie,reference=reference,resolution_spreads=resolution_spreads,
                comparisons=comparisons,pi_attribution=attribution,pi_fits_common=common_fits,
                shield_effects=effects,physical_gates=gates,shunt_conductance_3p4GHz_S=conductance,
                physical_gate_rc=int(not all(gates.values())),targets=TARGETS)
    (root/"qualification.json").write_text(json.dumps(dict(gates=gates,shield_effects=effects,
        physical_gate_rc=result["physical_gate_rc"]),indent=2)+"\n")
    freeze_native(root,mpie,conductance)
    (root/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    frozen=dict(reference="open-source FDTD reference",convention="exp(-i omega t)",
                port_orientation="Common sign reversal of native V and I cancels in Y/Z/S",
                extraction="Y=I@inv(V), Z=inv(Y), S=(I-50Y)@inv(I+50Y)",
                metric_frequency_grid=dict(linear_start_Hz=.1e9,linear_stop_Hz=20e9,linear_count=400,
                                           additional_exact_Hz=[1e9,3.4e9]),
                resonance_extraction="First inductive-to-capacitive Im(1/Y11) zero; linear interpolation",
                reference_variants={key:{**value,"metrics":scalars(value["metrics"]),
                                         "self_resonance_brackets_Hz":value["metrics"]["self_resonance_brackets_Hz"]}
                                    for key,value in reference.items()},
                resolution_spreads=resolution_spreads,targets=TARGETS)
    path=Path("tests/mom/layered/data/fixture_r6_reference.json")
    path.write_text(json.dumps(frozen,indent=2)+"\n")
    print("Frozen reference:",path)
    return result



def plot_evidence(root,summary):
    os.environ["MPLCONFIGDIR"]=str(root/"mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from tests.support.rfic_fixtures import fixture_fa
    from witwin.maxwell.mom.layered.runtime import compile_layered_scene
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharex=True,sharey=True)
    for ax,variant in zip(axes,("none","pgs","solid")):
        _,rectangles,_,_,_,vias,_=compile_layered_scene(fixture_fa(shield=variant,device="cpu").scene)
        for z,color,alpha in ((300.5e-6,"#3886c6",.8),(302e-6,"#db9c36",.65),(305e-6,"#242a32",.45)):
            for x0,x1,y0,y1,height in rectangles:
                if abs(height-z)<1e-12:
                    ax.add_patch(Rectangle((x0*1e6,y0*1e6),(x1-x0)*1e6,(y1-y0)*1e6,
                                           facecolor=color,alpha=alpha,edgecolor="none"))
        for x0,x1,y0,y1,_,_ in vias:
            ax.add_patch(Rectangle((x0*1e6,y0*1e6),(x1-x0)*1e6,(y1-y0)*1e6,
                                   facecolor="#c83d45",edgecolor="none"))
        ax.set(title=variant,xlim=(-165,165),ylim=(-165,165),aspect="equal",xlabel="x (um)")
    axes[0].set_ylabel("y (um)")
    fig.suptitle("F-A: shield blue, underpass amber, top metal gray, vias red")
    fig.tight_layout()
    fig.savefig(root/"fixtures.pdf")
    fig.savefig(root/"fixtures.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex="col")
    for column,variant in enumerate(("none","pgs","solid")):
        for mesh,style,color in (("shield","--","#7b7e87"),("full" if variant=="none" else "shield_fine","-","#255c95")):
            data=summary["mpie"][f"{variant}/{mesh}"]["metrics"]
            for row,key in ((0,"Q"),(1,"L_H")):
                values=np.array(data[key])*(1e9 if row else 1)
                label="coarse" if mesh=="shield" else "fine"
                axes[row,column].plot(np.array(data["frequency_Hz"])*1e-9,values,style,color=color,
                                      marker="o",markersize=2.2,label=f"MPIE {label}")
        for resolution,color in ((10,"#c77a24"),(5,"#9c3644")):
            data=summary["reference"][f"{variant}/{resolution}"]["metrics"]
            for row,key in ((0,"Q"),(1,"L_H")):
                values=np.array(data[key])*(1e9 if row else 1)
                axes[row,column].plot(np.array(data["frequency_Hz"])*1e-9,values,":",color=color,label=f"FDTD {resolution} um")
        axes[0,column].set(title=variant,ylim=(-5,23))
        axes[1,column].set(xlim=(.5,20),ylim=(0,6),xlabel="Frequency (GHz)")
        axes[0,column].legend(fontsize=7)
        for ax in axes[:,column]: ax.grid(alpha=.25)
    axes[0,0].set_ylabel("Q11 (port 2 shorted)")
    axes[1,0].set_ylabel("Series L (nH)")
    fig.tight_layout()
    fig.savefig(root/"networks.pdf")
    fig.savefig(root/"networks.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--check-targets",action="store_true")
    args=parser.parse_args()
    result=summarize(args.directory)
    plot_evidence(args.directory,result)
    for key,value in result["comparisons"].items():
        print(key,json.dumps(value))
    if args.check_targets:
        print("physical_gate_rc="+str(result["physical_gate_rc"]))
        raise SystemExit(result["physical_gate_rc"])


if __name__=="__main__":
    main()

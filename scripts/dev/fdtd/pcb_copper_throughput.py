"""Bound the real edge-refined PCB workload with a short native field pilot.

This preserves geometry, physical material conductivity, carrier and RF drive.
It omits field/face DFT to isolate the field workload. No
short-time power response or shared-device timing qualifies a physical speedup.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.pcb_copper_campaign import scene_control, validation_identity
from scripts.dev.fdtd.pcb_copper_grid import refine_pcb_edges
from scripts.dev.fdtd.pcb_legacy_performance import processes


def physical_snapshot(scene,frequency):
    structures = []
    for structure in scene.structures:
        material = structure.material
        physical = {"type":type(material).__name__}
        if isinstance(material,mw.CopperFoil):
            physical.update(thickness=float(material.thickness),conductivity=float(material.conductivity),
                frequency_range=list(material.frequency_range),mode_count=material.mode_count)
        else:
            physical.update(eps_r=float(material.eps_r),mu_r=float(material.mu_r),sigma_e=float(material.sigma_e))
        structures.append({"name":structure.name,"position_m":structure.geometry.position.detach().cpu().tolist(),
            "size_m":structure.geometry.size.detach().cpu().tolist(),"material":physical})
    ports = []
    for port in scene.ports:
        ports.append({"name":port.name,"positive_m":list(port.positive),"negative_m":list(port.negative),
            "reference_impedance_ohm":float(port.reference_impedance),
            "voltage_axis":port.voltage_path.axis,
            "aperture_position_m":port.voltage_aperture.position.detach().cpu().tolist(),
            "aperture_size_m":port.voltage_aperture.size.detach().cpu().tolist(),
            "contour_position_m":port.current_surface.position.detach().cpu().tolist(),
            "contour_size_m":port.current_surface.size.detach().cpu().tolist()})
    return {"grid_nodes_m":{axis:getattr(scene.grid,axis+"_coords").tolist() for axis in "xyz"},
        "structures":structures,"ports":ports,"carrier_hz":frequency,
        "drive":{"kind":"GaussianPulse","frequency":frequency,"fwidth":frequency,"amplitude":1.},
        "boundary":{"kind":"pml","num_layers":scene.boundary.num_layers},
        "subpixel":{"samples":list(scene.subpixel.samples),"averaging":scene.subpixel.averaging}}


def field_digests(solver):
    values = {}
    for name in ("Ex","Ey","Ez","Hx","Hy","Hz"):
        array = getattr(solver,name).detach().cpu().numpy()
        values[name] = {"shape":list(array.shape),"dtype":str(array.dtype),
            "sha256":hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            "nonzero_fraction":float(np.count_nonzero(array)/array.size),
            "maximum_absolute_value":float(np.max(abs(array)))}
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",choices=("flexible","rigid"),required=True)
    parser.add_argument("--kind",choices=("volume","surface"),default="volume")
    parser.add_argument("--frequency",type=float,default=2.45e9)
    parser.add_argument("--normal-cells-per-skin",type=int,default=4)
    parser.add_argument("--edge-cells-per-skin",type=int,default=4)
    parser.add_argument("--steps",type=int,choices=(1000,5000),default=1000)
    parser.add_argument("--minimum-free-gib",type=float,default=16.)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    free,total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib*2**30:
        raise SystemExit("Resource hold: less than required shared-GPU free headroom; no preparation started.")
    scene,_ = scene_control(args.family,args.kind,args.frequency,args.normal_cells_per_skin)
    scene = refine_pcb_edges(scene,args.frequency,args.edge_cells_per_skin).clone(monitors=())
    identity = validation_identity()
    identity[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"schema":"pcb-copper-field-pilot/1","source":package_identity(mw,"witwin-maxwell"),
        "validation_sources":identity,"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "free_bytes_before":free,"device_capacity_bytes":total,"processes_before":processes(),
        "physical_input":physical_snapshot(scene,args.frequency),"steps":args.steps,
        "cold_compilation_measured":False,
        "equal_error_qualified":False,"full_duration_performance_qualified":False,
        "scope":"short native field/RF pilot; no field/face DFT; excludes digests and persistence from run timing; extrapolation is conditional on the observed shared-device rate, not an exclusive forecast or rigorous lower bound"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    simulation = mw.Simulation.fdtd(scene,frequency=args.frequency,
        excitations=mw.PortExcitation(scene.ports[0].name,
            source_time=mw.GaussianPulse(args.frequency,fwidth=args.frequency)),
        run_time=mw.TimeConfig(time_steps=args.steps,transient_cycles=0),full_field_dft=False)
    batch = mw.Simulation.batch([simulation],outputs=mw.BatchOutput(),
        execution=mw.BatchExecution(devices=("cuda:0",),batch_size=1,graph="required",graph_steps=256))
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with batch.prepare() as prepared:
        torch.cuda.synchronize()
        report["native_prepare_seconds"] = time.perf_counter()-start
        solver = prepared._slot.solvers[0]
        report.update(dt_s=solver.dt,grid_shape=[solver.Nx,solver.Ny,solver.Nz],
            actual_duration_s=args.steps*solver.dt,prepared_allocated_bytes=torch.cuda.memory_allocated())
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        runs = []
        for repeat in range(2):
            start = time.perf_counter()
            result = prepared.run().results()[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            port = result.port(scene.ports[0].name)
            runs.append({"seconds":elapsed,"field_digests":field_digests(solver),
                "port_voltage":[[float(v.real),float(v.imag)] for v in port.voltage.flatten()],
                "port_current":[[float(v.real),float(v.imag)] for v in port.current.flatten()],
                "execution_tier":prepared.report.execution_tier})
            report["runs"] = runs
            args.output.write_text(json.dumps(report,indent=2)+"\n")
        report["reset_fields_bitwise_equal"] = runs[0]["field_digests"] == runs[1]["field_digests"]
        report["projected_12ns_steps"] = math.ceil(12e-9/solver.dt)
        report["projected_12ns_field_only_seconds"] = runs[1]["seconds"]*report["projected_12ns_steps"]/args.steps
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["processes_after"] = processes()
    report["source_unchanged"] = package_identity(mw,"witwin-maxwell")["source_sha256"] == report["source"]["source_sha256"]
    report["validation_unchanged"] = all(Path(path).exists() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path,digest in identity.items())
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key not in ("source","physical_input","validation_sources")},indent=2),flush=True)


if __name__ == "__main__":
    main()

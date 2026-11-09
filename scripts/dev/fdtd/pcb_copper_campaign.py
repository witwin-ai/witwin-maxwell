"""Run source-pinned physical copper PCB controls before repeated timings.

The default prepares only. Full execution is explicit because a skin-resolved
12 ns volume control contains millions of actual native GPU timesteps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.pcb_application_campaign import FIXTURES, power_observations
from scripts.dev.fdtd.pcb_candidate_inputs import candidate
from scripts.dev.fdtd.pcb_copper_performance import pcb_volume_scene
from scripts.dev.fdtd.pcb_copper_grid import refine_pcb_edges
from scripts.dev.fdtd.pcb_legacy_performance import processes


def validation_identity():
    names = ("pcb_acceptance","pcb_application_campaign","pcb_candidate_inputs",
             "pcb_copper_performance","pcb_copper_grid","pcb_legacy_performance")
    files = [Path(__file__)] + [Path(sys.modules["scripts.dev.fdtd."+name].__file__) for name in names]
    return {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def scene_control(family, kind, frequency, cells_per_skin, *, device="cuda:0"):
    from witwin.maxwell.adapters.pcb import import_pcb_scene
    original = json.loads((FIXTURES/f"metahz-scene-pcb-{family}.json").read_text())
    definition = candidate(original, finite_copper=True)
    scene = (pcb_volume_scene(family,frequency,cells_per_skin,device=device) if kind == "volume"
             else import_pcb_scene(definition,frequency=frequency,device=device))
    if kind == "volume":
        for record in definition["conductors"]:
            record["bounds_m"][5] += record["constitutive_model"]["thickness_m"]
    monitors = power_observations(definition,scene,frequency)
    return scene.clone(monitors=monitors), definition


def volume_material_heat(result, *, copper):
    """Integrate actual fields against separately compiled material conductivity.

    Copper is declared last and has no overlapping conductor masks, so its
    isolated conductivity is its actual contribution. A separately isolated
    substrate is only a diagnostic: sequential soft masks can change interface
    occupancy. The native remaining loss is total bulk loss minus copper.
    """
    from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
    from witwin.maxwell.postprocess.power_loss import compute_power_loss_data
    from witwin.maxwell.scene import prepare_scene
    structures = tuple(s for s in result.scene.structures
        if (s.name in ("positive-pad","negative-pad")) == copper)
    scene = result.scene.clone(structures=structures,ports=(),sources=())
    monitor = next(m for m in scene.monitors if m.name == "bulk_loss")
    compiled = compile_power_loss_monitor(prepare_scene(scene,lazy_meshgrid=True),monitor)
    fields = {component:torch.stack([result.tensor(component,frequency=f) for f in result.frequencies])
              for component in ("Ex","Ey","Ez")}
    return compute_power_loss_data(compiled,torch.as_tensor(result.frequencies,device=compiled.device,
        dtype=torch.float64),electric_fields=fields).conduction


def run(args):
    free,total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib*2**30:
        raise RuntimeError("Insufficient free GPU headroom for this physical preparation; retry after other work completes.")
    scene,definition = scene_control(args.family,args.kind,args.frequency,args.cells_per_skin)
    if args.edge_cells_per_skin is not None:
        scene = refine_pcb_edges(scene,args.frequency,args.edge_cells_per_skin)
    simulation = mw.Simulation.fdtd(scene,frequency=args.frequency,
        excitations=mw.PortExcitation(scene.ports[0].name,
            source_time=mw.GaussianPulse(args.frequency,fwidth=args.frequency)),
        run_time=mw.TimeConfig(time_steps=1,transient_cycles=0),
        spectral_sampler=mw.SpectralSampler(window="none"),full_field_dft=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    preflight = simulation.prepare()
    torch.cuda.synchronize()
    solver = preflight.solver
    dt = solver.dt
    steps = math.ceil(args.duration/dt)
    report = {"schema":"pcb-copper-physical-campaign/1","family":args.family,"kind":args.kind,
        "source":package_identity(mw,"witwin-maxwell"),"validation_sources":validation_identity(),
        "torch":torch.__version__,
        "gpu":torch.cuda.get_device_name(),"grid_shape":[solver.Nx,solver.Ny,solver.Nz],
        "free_bytes_before":free,"device_capacity_bytes":total,
        "dt_s":dt,"steps":steps,"requested_duration_s":args.duration,"physical_duration_s":steps*dt,
        "preflight_prepare_seconds":time.perf_counter()-start,
        "prepared_allocated_bytes":torch.cuda.memory_allocated(),
        "physical_input_sha256":hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest(),
        "equal_error_qualified":False,"exclusive_repeated_performance_qualified":False,
        "edge_cells_per_skin":args.edge_cells_per_skin,
        "scope":"physical control; actual normal/optional edge grid; full response convergence pending"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if not args.execute:
        return report
    del preflight,solver
    simulation.config.run_time = mw.TimeConfig(time_steps=steps,transient_cycles=0)
    batch = mw.Simulation.batch([simulation],outputs=mw.BatchOutput(fields=("Ex","Ey","Ez")),
        execution=mw.BatchExecution(devices=("cuda:0",),batch_size=1,graph="required",graph_steps=256))
    report["processes_before"] = processes()
    start = time.perf_counter()
    with batch.prepare() as execution:
        torch.cuda.synchronize()
        report["native_prepare_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        result = execution.run().results()[0]
        torch.cuda.synchronize()
        report["run_and_export_seconds"] = time.perf_counter()-start
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["processes_after"] = processes()
    start = time.perf_counter()
    total_bulk = result.power_loss("bulk_loss").conduction
    copper = (volume_material_heat(result,copper=True) if args.kind == "volume" else
              sum((row["surface_power"].to(total_bulk.device) for row in result.conductor_losses.values()),
                  torch.zeros_like(total_bulk)))
    substrate = total_bulk-copper if args.kind == "volume" else total_bulk
    report["bulk_loss_partition"] = "actual total native bulk work minus last-declared copper contribution" if args.kind == "volume" else "bulk work plus separately retained two-trace conductor work"
    report["copper_w"],report["substrate_w"] = copper.detach().cpu().tolist(),substrate.detach().cpu().tolist()
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    antenna = result.antenna(surface="radiation",driven_port=scene.ports[0].name,
        theta_points=91,phi_points=73,radius=10.)
    port = result.port(scene.ports[0].name)
    report.update({"reflection":[[float(v.real),float(v.imag)] for v in port.reflection_coefficient.flatten()],
        "accepted_w":antenna.p_accepted.detach().cpu().tolist(),
        "radiated_w":antenna.p_rad.detach().cpu().tolist(),
        "copper_w":copper.detach().cpu().tolist(),"substrate_w":substrate.detach().cpu().tolist(),
        "relative_power_closure":((antenna.p_rad+substrate+copper-antenna.p_accepted).abs()
            /antenna.p_accepted.abs()).detach().cpu().tolist()})
    torch.cuda.synchronize()
    report["power_postprocess_seconds"] = time.perf_counter()-start
    start = time.perf_counter()
    result.save(args.output.with_suffix(".pt"))
    report["persistence_seconds"] = time.perf_counter()-start
    report["source_unchanged"] = package_identity(mw,"witwin-maxwell")["source_sha256"] == report["source"]["source_sha256"]
    report["validation_unchanged"] = validation_identity() == report["validation_sources"]
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",choices=("flexible","rigid"),required=True)
    parser.add_argument("--kind",choices=("volume","surface"),required=True)
    parser.add_argument("--frequency",type=float,default=2.45e9)
    parser.add_argument("--cells-per-skin",type=int,default=4)
    parser.add_argument("--duration",type=float,default=12e-9)
    parser.add_argument("--edge-cells-per-skin",type=int)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--minimum-free-gib",type=float,default=16.)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.cells_per_skin < 1 or (args.edge_cells_per_skin is not None and args.edge_cells_per_skin < 1):
        parser.error("Positive physical duration and skin resolution are required.")
    print(json.dumps(run(args),indent=2),flush=True)


if __name__ == "__main__":
    main()

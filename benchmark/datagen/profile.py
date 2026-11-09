"""Profile one dataset sample by stage: python -m benchmark.datagen.profile.

Cold prepare populates a fresh public material cache; warm prepare measures its
reuse on the same scene. Unsupported cache workloads are explicitly reported.
Postprocessing/serialization precede the monitor-free control so two solvers
never coexist. Control preparation is reported separately from control run.

The control preserves all physics, sources, ports, frequency bands and steps.
It removes Scene.monitors and full-field readout requests. Intrinsic port and
copper-sheet spectra remain: the public API cannot disable those independently
of their physical coupling. Consequently the signed run difference measures
removable monitor/DFT cost, not all instrumentation cost.
"""

from __future__ import annotations

import argparse
import gc
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

import witwin.maxwell as mw
from .scenes import FAMILIES, SceneBuilder, scene_builders

STAGES = ("build", "configure", "prepare_cold", "prepare_warm", "run",
          "run_no_monitors", "postprocess", "serialize")
SAMPLE_STAGES = ("build", "configure", "prepare_cold", "run", "postprocess", "serialize")


def _measure(operation, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    return value, dict(seconds=seconds, peak_allocated_bytes=peak,
                       allocated_before_bytes=allocated, peak_increment_bytes=peak - allocated)


def _serialize(readouts):
    buffer = BytesIO()
    torch.save(readouts, buffer)
    return buffer.tell()


def _release():
    gc.collect()
    torch.cuda.empty_cache()


def _repeat(builder, device, repeat_index):
    _release()
    stages = {}
    sample, stages["build"] = _measure(lambda: builder.build(device=device), device)
    cache = (mw.FDTDMaterialCache(max_entries=1, max_bytes=4 * 1024**3)
             if sample.cache_unavailable is None else None)
    simulation, stages["configure"] = _measure(
        lambda: mw.Simulation.fdtd(sample.scene, material_cache=cache, **sample.fdtd_kwargs),
        device,
    )
    prepared, stages["prepare_cold"] = _measure(simulation.prepare, device)
    # The solver's endpoint-inclusive primal grid includes external PML nodes.
    grid_shape = [prepared.solver.scene.Nx - 1, prepared.solver.scene.Ny - 1,
                  prepared.solver.scene.Nz - 1]
    cells = math.prod(grid_shape)
    if cache is not None:
        del prepared
        _release()
        prepared, stages["prepare_warm"] = _measure(simulation.prepare, device)
        cache_stats = dict(cache.stats)
        stages["prepare_warm"]["cache_stats"] = cache_stats
    else:
        stages["prepare_warm"] = dict(status="unsupported", reason=sample.cache_unavailable)

    result, stages["run"] = _measure(prepared.run, device)
    stats = result.solver_stats
    run_stats = {key: stats[key] for key in ("dt", "time_steps", "steps_run")}
    run_stats["cell_updates_per_second"] = cells * stats["steps_run"] / stages["run"]["seconds"]
    run_stats["cuda_graph_active"] = stats["cuda_graph_active"]
    run_stats["material_cache"] = stats["material_cache"]
    readouts, stages["postprocess"] = _measure(
        lambda: {name: readout(result) for name, readout in sample.readouts}, device,
    )
    serialized_bytes, stages["serialize"] = _measure(lambda: _serialize(readouts), device)
    sample.readouts.clear()
    del readouts, result, prepared, simulation
    _release()

    sample.scene.monitors.clear()
    # FDTD refuses a run without outputs; one single-component point probe is the
    # minimal observable and its per-step cost is negligible.
    center = tuple(sum(bounds) / 2 for bounds in sample.scene.domain.bounds)
    sample.scene.add_monitor(mw.PointMonitor("control_probe", position=center, fields=("Ez",)))
    control_kwargs = dict(sample.fdtd_kwargs)
    control_kwargs["full_field_dft"] = False
    # Freeze the measured time plan: deleting monitor requests must not change
    # the integration duration, even when automatic planning uses those requests.
    control_kwargs["run_time"] = mw.TimeConfig(time_steps=run_stats["time_steps"], dt=run_stats["dt"])

    def prepare_control():
        return mw.Simulation.fdtd(sample.scene, material_cache=cache, **control_kwargs).prepare()

    control, control_setup = _measure(prepare_control, device)
    control_result, stages["run_no_monitors"] = _measure(control.run, device)
    control_stats = {key: control_result.solver_stats[key]
                     for key in ("dt", "time_steps", "steps_run")}
    control_stats["cell_updates_per_second"] = (
        cells * control_stats["steps_run"] / stages["run_no_monitors"]["seconds"])
    control_stats["intrinsic_spectra_retained"] = bool(sample.scene.ports)
    delta = stages["run"]["seconds"] - stages["run_no_monitors"]["seconds"]
    record = dict(
        repeat=repeat_index + 1,
        cold_including_native_build_or_jit_if_needed=repeat_index == 0,
        stages=stages, grid_shape_including_pml=grid_shape, cells=cells,
        run=run_stats, run_no_monitors=control_stats, serialized_bytes=serialized_bytes,
        monitor_dft_seconds=delta, control_configure_prepare=control_setup,
    )
    del control_result, control, sample, cache
    _release()
    return record


def _median_records(records):
    return {key: statistics.median(record[key] for record in records) for key in records[0]}


def _profile(builder: SceneBuilder, repeats, device):
    records = []
    for repeat_index in range(repeats):
        print(f"{builder.name}: repeat {repeat_index + 1}/{repeats}", file=sys.stderr, flush=True)
        records.append(_repeat(builder, device, repeat_index))
    stages = {}
    for stage in STAGES:
        entries = [record["stages"][stage] for record in records]
        if stage == "prepare_warm" and "reason" in entries[0]:
            stages[stage] = entries[0]
        else:
            stages[stage] = _median_records([
                {key: entry[key] for key in ("seconds", "peak_allocated_bytes",
                                            "allocated_before_bytes", "peak_increment_bytes")}
                for entry in entries
            ])
    run = _median_records([{key: record["run"][key]
                            for key in ("dt", "time_steps", "steps_run", "cell_updates_per_second")}
                           for record in records])
    bare_run = _median_records([{key: record["run_no_monitors"][key]
                                 for key in ("dt", "time_steps", "steps_run", "cell_updates_per_second")}
                                for record in records])
    return dict(
        name=builder.name, family=builder.family, expected_cells=builder.expected_cells,
        cells=records[0]["cells"], grid_shape_including_pml=records[0]["grid_shape_including_pml"],
        stages=stages, run=run, run_no_monitors=bare_run,
        serialized_bytes=statistics.median(record["serialized_bytes"] for record in records),
        monitor_dft_seconds=statistics.median(record["monitor_dft_seconds"] for record in records),
        cold=records[0], repeats=records,
    )


def _summaries(scenes):
    summaries = {}
    for family in FAMILIES:
        members = [scene for scene in scenes if scene["family"] == family]
        if not members:
            continue
        totals = [sum(scene["stages"][stage]["seconds"] for stage in SAMPLE_STAGES)
                  for scene in members]
        # All-eight-stage shares describe the measurement workload. The separate
        # sample shares exclude duplicate preparation and the diagnostic control.
        measured_totals = [sum(entry["seconds"] for entry in scene["stages"].values()
                               if "seconds" in entry) for scene in members]
        stage_shares = {}
        for stage in STAGES:
            shares = [scene["stages"][stage]["seconds"] / total
                      for scene, total in zip(members, measured_totals)
                      if "seconds" in scene["stages"][stage]]
            stage_shares[stage] = dict(median=statistics.median(shares) if shares else None,
                                       supported_scenes=len(shares))
        summaries[family] = dict(
            scenes=len(members), median_sample_seconds=statistics.median(totals),
            median_stage_share=stage_shares,
            median_sample_stage_share={
                stage: statistics.median(scene["stages"][stage]["seconds"] / total
                                         for scene, total in zip(members, totals))
                for stage in SAMPLE_STAGES
            },
            median_removable_monitor_dft_run_share=statistics.median(
                scene["monitor_dft_seconds"] / scene["stages"]["run"]["seconds"]
                for scene in members),
        )
    return summaries


def _table(scenes, summaries):
    print("\nTimes in seconds; warm '—' means the public cache does not support the scene.")
    print("\n| Scene | Cells | Build | Config | Cold | Warm | Run | Bare | Post | Save | Peak GiB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for scene in scenes:
        values = [f"{scene['stages'][stage]['seconds']:.3f}"
                  if "seconds" in scene["stages"][stage] else "—" for stage in STAGES]
        peak = max(stage["peak_allocated_bytes"] for stage in scene["stages"].values()
                   if "peak_allocated_bytes" in stage) / 1024**3
        print(f"| {scene['name']} | {scene['cells']:,} | " + " | ".join(values) + f" | {peak:.2f} |")
    print("\nMedian single-sample shares (cold prepare; diagnostic reruns excluded):")
    print("\n| Family | Build | Config | Prepare | Run | Post | Save | Removable run cost |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for family, summary in summaries.items():
        values = [f"{summary['median_sample_stage_share'][stage]:.1%}" for stage in SAMPLE_STAGES]
        print(f"| {family} | " + " | ".join(values) +
              f" | {summary['median_removable_monitor_dft_run_share']:.1%} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--scenes", default="all", help="Comma-separated catalog names, or all")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--smoke", action="store_true", help="Smallest selected scene per family")
    args = parser.parse_args()
    catalog = {builder.name: builder for builder in scene_builders()}
    selected = list(catalog.values()) if args.scenes == "all" else [catalog[name] for name in args.scenes.split(",")]
    selected = [builder for builder in selected if args.family in ("all", builder.family)]
    if args.smoke:
        selected = [min((builder for builder in selected if builder.family == family),
                        key=lambda builder: builder.expected_cells)
                    for family in FAMILIES if any(builder.family == family for builder in selected)]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    # Initialize the CUDA context outside the timed scene stages. Native solver
    # compilation remains cold and is included in the first measured repeat.
    torch.cuda.synchronize(device)
    scenes = [_profile(builder, args.repeats, device) for builder in selected]
    summaries = _summaries(scenes)
    report = dict(
        schema="datagen-fdtd-cost-profile/1", device=str(device),
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        repeats=args.repeats, smoke=args.smoke,
        methodology=dict(
            timer="CUDA synchronize before/after perf_counter wall time",
            memory="per-stage absolute peak allocated bytes and increment over entry allocation; excludes reserved memory",
            cold="first repeat per scene; includes native build/JIT if needed; process/disk caches are not purged",
            prepare_cold="fresh material-cache miss and population where supported",
            prepare_warm="same scene, same cache, fresh solver; unsupported cases explicitly marked",
            run="warm prepared solver where supported, otherwise the cold prepared solver",
            control="same physics/band, monitors replaced by one Ez point probe, full_field_dft=False, identical dt and planned steps",
            residual_control_cost="F3/F4 and loaded F5 retain intrinsic port spectra; F3 also retains copper-sheet spectra; no public independent disable",
            execution_order="build, configure, prepare_cold, prepare_warm, run, postprocess, serialize, control setup, run_no_monitors",
            shares="median of per-scene shares; sample shares exclude warm preparation and control reruns",
            qualification="cost measurements only; no accuracy or convergence claim",
        ),
        catalog=[dict(name=builder.name, family=builder.family, expected_cells=builder.expected_cells)
                 for builder in catalog.values()],
        scenes=scenes, families=summaries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _table(scenes, summaries)


if __name__ == "__main__":
    main()

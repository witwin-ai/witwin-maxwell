"""Setup stage wall times and warmed, graph-enabled whole-step CUDA event timing.

Dry runs construct CPU Scene objects only. Actual timing requires a coordinator's
exclusive GPU window. Each selected scene executes in a fresh interpreter.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

from .gates import THREAD_ENV, make_simulation
from .scenes import CORE_SCENES, build_core_case, build_ladder_case, ladder_names


OUTPUT = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "core-cuda-g5"


def build_case(name):
    return build_core_case(name) if name in CORE_SCENES else build_ladder_case(name)


def timed_prepare(simulation):
    import torch
    from witwin.maxwell.fdtd.solver import FDTD

    simulation_module = importlib.import_module("witwin.maxwell.simulation")
    solver_module = importlib.import_module("witwin.maxwell.fdtd.solver")
    initialization = importlib.import_module("witwin.maxwell.fdtd.runtime.initialization")
    materials = importlib.import_module("witwin.maxwell.fdtd.runtime.materials")
    stepping = importlib.import_module("witwin.maxwell.fdtd.runtime.stepping")
    stages = {name: 0.0 for name in (
        "geometry_material_compile_s", "cpml_build_s", "source_monitor_compile_s", "extension_load_s",
    )}

    calls = {}
    depth = 0

    def measure(function, stage, target):
        def wrapped(*args, **kwargs):
            nonlocal depth
            assert depth == 0, f"Nested prepare timing target: {target}"
            depth += 1
            calls[target] += 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            stages[stage] += time.perf_counter() - start
            depth -= 1
            return result
        return wrapped

    # Both modules bind prepare_scene at import time. The material-coefficient
    # import in init_field resolves against materials at call time. Class methods
    # and the other runtime aliases are patched where their callers resolve them.
    # These regions do not nest; boundary initialization includes CPML allocation.
    targets = [
        (simulation_module, "prepare_scene", "geometry_material_compile_s"),
        (solver_module, "prepare_scene", "geometry_material_compile_s"),
        (FDTD, "build_materials", "geometry_material_compile_s"),
        (materials, "build_material_update_coefficients", "geometry_material_compile_s"),
        (FDTD, "_build_update_coefficients", "geometry_material_compile_s"),
        (stepping, "initialize_boundary_state", "cpml_build_s"),
        (initialization, "compile_fdtd_sources", "source_monitor_compile_s"),
        (initialization, "compile_fdtd_observers", "source_monitor_compile_s"),
        (initialization, "compile_fdtd_time_observers", "source_monitor_compile_s"),
        (initialization, "compile_fdtd_breakdown_observers", "source_monitor_compile_s"),
        (stepping, "initialize_tfsf_state", "source_monitor_compile_s"),
        (stepping, "initialize_source_terms", "source_monitor_compile_s"),
    ]
    # FDTD.__init__ copies this alias into initialization.get_fdtd_module.
    targets.append((solver_module, "_get_fdtd_module", "extension_load_s"))
    torch.cuda.synchronize()
    start = time.perf_counter()
    with ExitStack() as stack:
        for owner, name, stage in targets:
            owner_name = f"{owner.__module__}.{owner.__qualname__}" if owner is FDTD else owner.__name__
            target = f"{owner_name}.{name}"
            calls[target] = 0
            stack.enter_context(patch.object(owner, name, measure(getattr(owner, name), stage, target)))
        prepared = simulation.prepare()
    assert all(calls.values()), f"Uncalled prepare timing targets: {[name for name, count in calls.items() if count == 0]}"
    torch.cuda.synchronize()
    stages["prepare_total_s"] = time.perf_counter() - start
    stages["prepare_other_s"] = stages["prepare_total_s"] - sum(
        stages[name] for name in stages if name != "prepare_total_s"
    )
    stages["target_calls"] = calls
    return prepared, stages


def run_case(args):
    import torch
    from witwin.maxwell.fdtd.runtime import stepping

    case = build_case(args.scene)
    # The extra step supplies an end boundary before finalization/field export.
    simulation = make_simulation(case, arm=args.arm, steps=args.warmup + args.steps + 1,
                                 full_field_dft=False)
    prepared, setup = timed_prepare(simulation)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    factory = stepping._make_field_update_runner
    calls = 0

    def timed_factory(solver, use_cuda_graph):
        runner = factory(solver, use_cuda_graph)

        def run(time_value):
            nonlocal calls
            if calls == args.warmup:
                if args.profile_range:
                    torch.cuda.cudart().cudaProfilerStart()
                begin.record()
            if calls == args.warmup + args.steps:
                end.record()
                if args.profile_range:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
            runner(time_value)
            calls += 1
        return run

    with patch.object(stepping, "_make_field_update_runner", timed_factory):
        result = prepared.run()
    end.synchronize()
    assert calls == args.warmup + args.steps + 1
    assert result.stats()["cuda_graph_active"], "Timing requires active CUDA graphs"
    shape = (prepared.solver.Nx, prepared.solver.Ny, prepared.solver.Nz)
    elapsed_ms = begin.elapsed_time(end)
    payload = {
        "scene": case.plan(), "arm": args.arm, "setup": setup,
        "computational_primal_shape": shape, "warmup_steps": args.warmup,
        "measured_steps": args.steps, "cuda_graph_active": True,
        "elapsed_ms": elapsed_ms, "ms_per_step": elapsed_ms / args.steps,
        "gcell_step_per_s": math.prod(shape) * args.steps / (elapsed_ms * 1e6),
        "scope": "whole steps including source, boundary, DFT and host launch gaps; excludes setup, graph capture, warmup and finalization",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.scene}-{args.arm}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def profiler_commands(args, name):
    command = [sys.executable, "-m", "benchmark.core_path.timing", "--worker", "--scene", name,
               "--steps", str(args.steps), "--warmup", str(args.warmup), "--arm", args.arm,
               "--output", str(args.output), "--profile-range"]
    environment = [f"{key}={value}" for key, value in THREAD_ENV.items()]
    for profiler in (
        ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--capture-range=cudaProfilerApi",
         "--capture-range-end=stop", "-o", str(args.output / f"{name}-nsys")],
        ["ncu", "--profile-from-start", "off", "--graph-profiling", "node",
         "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum", "--csv",
         "--log-file", str(args.output / f"{name}-dram.csv")],
    ):
        print(shlex.join(["env", *environment, "nice", "-n", "10", *profiler, *command]))
    print(f"DRAM normalization: sum read+write bytes over the captured range / ({args.steps} * Nx * Ny * Nz).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=CORE_SCENES + ladder_names())
    parser.add_argument("--core-scenes", action="store_true", help="Time setup and stepping of the correctness set")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emit-profiler-commands", action="store_true")
    parser.add_argument("--profile-range", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    names = (args.scene,) if args.scene else CORE_SCENES if args.core_scenes else ladder_names()
    if args.emit_profiler_commands:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in names:
            print(json.dumps(build_case(name).plan()))
            profiler_commands(args, name)
    elif args.dry_run:
        for name in names:
            print(json.dumps(build_case(name).plan()))
        print(json.dumps({"setup_stages": ["geometry/material", "CPML", "source/monitor", "extension", "other"],
                          "warmup_steps": args.warmup, "measured_steps": args.steps, "cuda_graph": True,
                          "fresh_interpreters": True, "output": str(args.output)}))
    elif args.worker:
        run_case(args)
    else:
        for name in names:
            subprocess.run([sys.executable, "-m", "benchmark.core_path.timing", "--worker",
                            "--scene", name, "--steps", str(args.steps), "--warmup", str(args.warmup),
                            "--arm", args.arm, "--output", str(args.output)],
                           check=True, env=os.environ | THREAD_ENV)


if __name__ == "__main__":
    main()

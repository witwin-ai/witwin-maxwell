"""D3b measurement only: source-derived ranges, eager attribution, graph timings.

Run through the prescribed GPU-0 Python launcher. The baseline subcommand runs
the unchanged V15 driver. Profile labels come from temporary instance methods
compiled from thin_sheet.py's AST; no library file or arithmetic is changed.
An exact ordered GPU-operation match transfers the ranged eager call sites onto
one steady production CUDA-graph replay (whose CPU ranges are not replayed).
Captured memcpy32_post kernels match eager device copies by position and bytes.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
import time
from types import MethodType

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev/fdtd"))
import thin_sheet_overhead as v15
import witwin.maxwell as mw
from witwin.maxwell.fdtd import thin_sheet
from witwin.maxwell.fdtd.runtime import stepping

SCRATCH = Path("/tmp/pcb-d3b-build")
BUCKETS = [
    "gather/scatter patch_h", "gather/scatter patch_e",
    "gather/scatter h_cols", "gather/scatter e_rows",
    "h_matrix mv", "e_matrix mv", "copper predictor/h_chord",
    "copper rhs history/Acx/Acc", "chord_solve S mv",
    "chord_solve R mv", "chord_solve vector ops", "pair 2x2 solves",
    "state/loss update", "feed loops", "non-sheet field update", "other",
]


def gpu_sample():
    result = v15._gpu_load()
    result["own_pid"] = os.getpid()
    result["pmon"] = subprocess.check_output(
        ["nvidia-smi", "pmon", "-i", "0", "-c", "3", "-d", "1"], text=True)
    busy = []
    for line in result["pmon"].splitlines():
        fields = line.split()
        if fields and fields[0] == "0" and fields[3] != "-":
            if int(fields[1]) != os.getpid() and int(fields[3]) > 5:
                busy.append(int(fields[1]))
    result["other_busy_pids"] = sorted(set(busy))
    result["unix_time"] = time.time()
    return result


@contextmanager
def monitor_gpu(name):
    path = SCRATCH / f"{name}-pmon.txt"
    with path.open("w") as handle:
        process = subprocess.Popen(["nvidia-smi", "pmon", "-i", "0", "-d", "1", "-o", "T"], stdout=handle)
        try:
            yield path
        finally:
            process.terminate()
            process.wait()


def wait_clean(output):
    samples = []
    start = time.monotonic()
    for attempt in range(16):
        if attempt:
            time.sleep(max(0, start + 120 * attempt - time.monotonic()))
        sample = gpu_sample()
        samples.append(sample)
        output.write_text(json.dumps(samples, indent=2))
        print(json.dumps(sample), flush=True)
        if not sample["other_busy_pids"]:
            return


def statement_bucket(method, node):
    expression = ast.unparse(node)
    if method == "chord_solve":
        return "chord_solve vector ops"
    for plan in ("patch_h", "patch_e", "h_cols", "e_rows"):
        if "self.plan_" + plan in expression:
            return "gather/scatter " + plan
    if method == "magnetic":
        if "predictor" in expression:
            return "copper predictor/h_chord"
        return "other"
    if method == "electric":
        line = node.lineno
        if 621 <= line <= 639:
            return "copper rhs history/Acx/Acc"
        if 640 <= line <= 642:
            return "chord_solve vector ops"
        if line == 646:
            return "copper predictor/h_chord"
        if 648 <= line <= 661:
            return "pair 2x2 solves"
        if 662 <= line <= 683 or line == 644:
            return "state/loss update"
    return "other"


class Ranges(ast.NodeTransformer):
    def __init__(self, method):
        self.method = method

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "torch.mv":
            matrix = ast.unparse(node.args[0])
            bucket = None
            if self.method == "chord_solve" or "['S']" in matrix:
                bucket = "chord_solve R mv" if "['R']" in matrix else "chord_solve S mv"
            elif matrix == "self.h_matrix":
                bucket = "h_matrix mv"
            elif matrix == "self.e_matrix":
                bucket = "e_matrix mv"
            if bucket:
                node.func = ast.Name(id="_phase0_mv", ctx=ast.Load())
                node.args.insert(0, ast.Constant(f"{bucket}|thin_sheet.py:{node.lineno}|{matrix}"))
        return node

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                return node
            bucket = statement_bucket(self.method, node)
            label = f"{bucket}|thin_sheet.py:{node.lineno}|{self.method}"
            context = ast.Call(func=ast.Name(id="_phase0_range", ctx=ast.Load()),
                               args=[ast.Constant(label)], keywords=[])
            return ast.copy_location(ast.With(items=[ast.withitem(context_expr=context)], body=[node]), node)
        return node


def ranged_mv(label, *args, **kwargs):
    with record_function(label):
        return torch.mv(*args, **kwargs)


@contextmanager
def instrument(sheet):
    namespace = dict(vars(thin_sheet), _phase0_range=record_function, _phase0_mv=ranged_mv)
    methods = ("begin", "magnetic", "electric", "chord_solve")
    for name in methods:
        original = getattr(type(sheet), name)
        lines, start = inspect.getsourcelines(original)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        ast.increment_lineno(tree, start - 1)
        tree = Ranges(name).visit(tree)
        exec(compile(ast.fix_missing_locations(tree), thin_sheet.__file__, "exec"), namespace)
        setattr(sheet, name, MethodType(namespace[name], sheet))
    try:
        yield
    finally:
        for name in methods:
            delattr(sheet, name)


def collect(run, label, path):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(label):
            run(0.0)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())["traceEvents"]
    kernels = sorted((e for e in trace if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    return kernels, trace


def attribute(kernels, trace):
    cpu = {e["args"]["External id"]: e for e in trace
           if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    ranges = [e for e in trace if e.get("cat") == "user_annotation" and "|" in e["name"]]
    sites = []
    for kernel in kernels:
        op = cpu[kernel["args"]["External id"]]
        enclosing = [r for r in ranges if r["tid"] == op["tid"]
                     and r["ts"] <= op["ts"] and op["ts"] + op["dur"] <= r["ts"] + r["dur"] + .01]
        site = min(enclosing, key=lambda r: r["dur"])["name"]
        sites.append(site)
    return sites


def run_profile(name, output):
    before = gpu_sample()
    prepared = mw.Simulation.fdtd(v15._scene(v15.SCENARIOS[name], True), frequencies=[2.4e9],
                                 run_time=mw.TimeConfig(time_steps=1)).prepare()
    result = prepared.run()
    solver = result.solver
    sheet = solver._thin_sheets[0]
    cu = sheet.copper
    matrices = {key: dict(rows=int(matrix.shape[0]), cols=int(matrix.shape[1]), nnz=matrix._nnz())
                for key, matrix in (("S", cu["S"]), ("R", cu["R"]),
                                    ("h_matrix", sheet.h_matrix), ("e_matrix", sheet.e_matrix))}
    output.write_text(json.dumps(dict(scenario=name, matrices=matrices, chords=cu["count"],
                                     chebyshev_degree=cu["chebyshev"][2]), indent=2))
    replay = stepping._make_field_update_runner(solver, True)
    assert solver._cuda_graph_active
    for _ in range(20):
        replay(0.0)
    event_ms = v15._event_ms(replay, 200)
    from witwin.maxwell.fdtd.checkpoint import graph_state_tensors
    state = graph_state_tensors(solver)
    saved = {key: tensor.clone() for key, tensor in state.items()}
    graph_kernels, graph_trace = collect(replay, "steady captured step", SCRATCH / f"{name}-graph.json")
    assert sum(e.get("name") == "cudaGraphLaunch" for e in graph_trace) == 1
    assert all(e["args"]["graph id"] != 0 for e in graph_kernels)
    graph_state = {key: tensor.clone() for key, tensor in state.items()}
    for key, tensor in state.items():
        tensor.copy_(saved[key])
    with instrument(sheet):
        eager = stepping._make_field_update_runner(solver, False)
        eager_kernels, eager_trace = collect(
            eager, "non-sheet field update|runtime/stepping.py:1859|field block", SCRATCH / f"{name}-ranged.json")
    assert all(torch.equal(tensor, graph_state[key]) for key, tensor in state.items()), "Instrumentation changed step state"
    eager_operations = sorted((e for e in eager_trace if e.get("cat") in ("kernel", "gpu_memcpy")),
                              key=lambda e: e["ts"])
    names = ["memcpy32_post" if e["cat"] == "gpu_memcpy" else e["name"] for e in eager_operations]
    assert names == [e["name"] for e in graph_kernels], "Ambiguous graph/eager operation mapping"
    copies = []
    for index, (kernel, operation) in enumerate(zip(graph_kernels, eager_operations)):
        if operation["cat"] == "gpu_memcpy":
            assert operation["name"] == "Memcpy DtoD (Device -> Device)"
            assert kernel["args"]["block"] == [256, 1, 1]
            assert kernel["args"]["grid"] == [(operation["args"]["bytes"] + 1023) // 1024, 1, 1]
            copies.append(dict(kernel_index=index, bytes=operation["args"]["bytes"]))
    sites = attribute(eager_operations, eager_trace)
    buckets = {b: dict(kernels=0, us=0.0) for b in BUCKETS}
    callsites = defaultdict(lambda: dict(kernels=0, us=0.0))
    inventory = []
    for index, (kernel, site) in enumerate(zip(graph_kernels, sites)):
        bucket = site.split("|")[0]
        for row in (buckets[bucket], callsites[site]):
            row["kernels"] += 1
            row["us"] += kernel["dur"]
        inventory.append(dict(index=index, name=kernel["name"], us=kernel["dur"], site=site,
                              graph_node_id=kernel["args"]["graph node id"]))
    record = dict(scenario=name, before=before, after=gpu_sample(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(0),
                  source_sha256=hashlib.sha256(Path(thin_sheet.__file__).read_bytes()).hexdigest(),
                  graph_active=True, graph_launches=sum(e.get("name") == "cudaGraphLaunch" for e in graph_trace),
                  graph_ids=sorted({e["args"]["graph id"] for e in graph_kernels}),
                  event_ms_per_step=event_ms, instrumented_state_bitwise_equal=True,
                  exact_ordered_operation_match=True, captured_copy_kernels=copies,
                  eager_kernel_count=len(eager_kernels), chords=cu["count"], feeds=len(sheet.feeds),
                  pair_count=int(cu["pair_position"].shape[0]), grid=[solver.Nx, solver.Ny, solver.Nz],
                  chebyshev_degree=cu["chebyshev"][2], iterations=2 * cu["chebyshev"][2], matrices=matrices,
                  buckets=buckets, callsites=dict(callsites), kernel_inventory=inventory,
                  total_kernels=len(graph_kernels), total_kernel_us=sum(e["dur"] for e in graph_kernels),
                  graph_span_us=max(e["ts"] + e["dur"] for e in graph_kernels) - min(e["ts"] for e in graph_kernels),
                  nonkernel_gpu_events=[e for e in graph_trace if e.get("cat") in ("gpu_memcpy", "gpu_memset")],
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k not in ("kernel_inventory", "callsites", "nonkernel_gpu_events")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("wait", "baseline", "profile"))
    parser.add_argument("--scenario", choices=("copper_patch", "copper_board_60k"))
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(SCRATCH)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(SCRATCH)
    if args.mode == "wait":
        wait_clean(args.json)
    else:
        torch.cuda.set_per_process_memory_fraction(5.0 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        with monitor_gpu(args.json.stem):
            if args.mode == "baseline":
                before = gpu_sample()
                sys.argv = [str(ROOT / "scripts/dev/fdtd/thin_sheet_overhead.py"), "--scenario",
                            "copper_patch", "copper_board_60k", "--json", str(args.json), "--memory-gib", "5"]
                runpy.run_path(sys.argv[0], run_name="__main__")
                args.json.with_suffix(".gpu.json").write_text(json.dumps(dict(before=before, after=gpu_sample()), indent=2))
            else:
                run_profile(args.scenario, args.json)


if __name__ == "__main__":
    main()

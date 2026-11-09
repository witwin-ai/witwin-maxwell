"""Profile the printed-dipole step floor without changing installed solver code.

Four matched-grid controls isolate foil and feed costs. Preparation, graph capture,
warmup and result extraction are outside the measured window. A second, explicitly
eager inventory attributes kernels to blocks that disappear behind graph replay.
All runtime instrumentation is temporary and confined to this diagnostic process.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import functools
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
VARIANTS = ("copper_port", "pec_port", "copper_point", "pec_point")


def partition(anchors, step, *, fine=(), half_planes=()):
    """Copied from the data-plan builder; retain its exact physical anchors."""
    contour_step = min(step, .0004)
    brackets = tuple((p - contour_step / 2, p + contour_step / 2) for p in half_planes)
    required = sorted(set((*anchors, *(x for lo, hi, _ in fine for x in (lo, hi)),
                           *(x for pair in brackets for x in pair))))
    nodes = [required[0]]
    for lo, hi in zip(required, required[1:]):
        target = step
        for first, last, local_step in fine:
            if first <= (lo + hi) / 2 <= last:
                target = min(target, local_step)
        count = 1 if (lo, hi) in brackets else max(1, math.ceil((hi - lo) / target - 1e-12))
        nodes.extend(np.linspace(lo, hi, count + 1)[1:])
    return np.asarray(nodes)


def build_scene(variant, *, device="cuda:0"):
    import witwin.maxwell as mw

    lateral, air, finest, height = .002, .004, .0002, .0014
    x = partition((-.065, -.045, -.040, -.001, 0., .001, .040, .045, .065), air,
                  fine=((-.045, .045, lateral), (-.002, .002, finest)))
    y = partition((-.025, -.0055, -.001, 0., .001, .0055, .025), air,
                  fine=((-.0055, .0055, lateral),), half_planes=(-.0035, .0035))
    z = partition((-.025, -height, 0., .025), air,
                  fine=((-height, 0., min(height / 7, finest)),),
                  half_planes=(-.0025, .0025))

    def box(position, size):
        return mw.Box(position=torch.tensor(position, dtype=torch.float64),
                      size=torch.tensor(size, dtype=torch.float64))

    frequencies = tuple(np.linspace(1.8e9, 3.0e9, 11))
    scene = mw.Scene(domain=mw.Domain(tuple((float(a[0]), float(a[-1])) for a in (x, y, z))),
                     grid=mw.GridSpec.custom(x, y, z),
                     boundary=mw.BoundarySpec.pml(num_layers=8), device=device)
    scene.add_structure(mw.Structure(box((0., 0., -height / 2), (.090, .011, height)),
                                     mw.Material(eps_r=3.5), name="substrate"))
    for name, lo, hi in (("negative", -.040, -.001), ("positive", .001, .040)):
        material = (mw.CopperFoil(thickness=35e-6, conductivity=5.8e7,
                                 frequency_range=(.1e9, 10e9))
                    if variant.startswith("copper") else mw.Material.pec())
        scene.add_structure(mw.Structure(box(((lo + hi) / 2, 0., 0.), (hi - lo, .002, 0.)),
                                         material, name=name))
    pulse = mw.GaussianPulse(frequency=2.4e9, fwidth=1.92e9)
    excitations = ()
    if variant.endswith("port"):
        scene.add_port(mw.LumpedPort(
            "feed", positive=(.001, 0., 0.), negative=(-.001, 0., 0.),
            voltage_path=mw.AxisPath("x"), reference_impedance=50.,
            current_surface=box((x[np.searchsorted(x, 0.) + 1] / 2, 0., 0.), (0., .007, .005)),
            voltage_aperture=box((0., 0., 0.), (.002, .002, 0.))))
        excitations = (mw.PortExcitation("feed", source_time=pulse),)
    else:
        scene.add_source(mw.PointDipole(position=(0., 0., 0.), polarization="Ex",
                                       profile="ideal", source_time=pulse, name="feed"))
    scene.add_monitor(mw.PointMonitor("probe", position=(0., 0., .01), fields=("Ex",)))
    return scene, frequencies, excitations


def union_us(intervals):
    total, end = 0., -math.inf
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(prof, trace_path, steps, scope):
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    ranges = [e for e in events if e.get("name") == scope and e.get("ph") == "X"
              and e.get("cat") == "user_annotation"]
    if not ranges:
        raise RuntimeError(f"Missing profiler scope {scope}")
    lo = min(e["ts"] for e in ranges)
    hi = max(e["ts"] + e["dur"] for e in ranges)
    cpu = [e for e in events if e.get("ph") == "X" and lo <= e.get("ts", -1) < hi
           and e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver", "user_annotation")]
    # Device work may finish after the CPU submission window. The profiler is
    # started after warmup synchronization and stopped after measured completion.
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    copies = [e for e in events if e.get("cat") == "gpu_memcpy" and e.get("ph") == "X"]
    runtime = [e for e in cpu if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    api_counts = Counter(e["name"] for e in runtime)
    sync = Counter({name: count for name, count in api_counts.items() if "Synchronize" in name})
    host_reads = Counter(e["name"] for e in cpu
                         if e["name"] in ("aten::item", "aten::_local_scalar_dense", "aten::tolist", "aten::cpu"))
    dtoh = [e for e in copies if "dtoh" in e["name"].lower() or "device to host" in e["name"].lower()]
    gpu_ops = defaultdict(lambda: [0, 0.])
    for event in kernels:
        row = gpu_ops[event["name"]]
        row[0] += 1
        row[1] += event["dur"]
    top_gpu = [{"name": name, "calls_per_step": n / steps, "us_per_step": duration / steps}
               for name, (n, duration) in sorted(gpu_ops.items(), key=lambda kv: -kv[1][1])[:20]]
    # Scope CPU rankings to loop descendants, excluding the measurement-end
    # event wait and profiler teardown. CPU/GPU annotations can share a name.
    cpu_ops = defaultdict(lambda: [0, 0., 0.])
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        parent = event
        while parent is not None and parent.name != scope:
            parent = parent.cpu_parent
        if parent is None:
            continue
        row = cpu_ops[event.name]
        row[0] += 1
        row[1] += event.self_cpu_time_total
        row[2] += event.cpu_time_total
    def cpu_row(name, values):
        n, own, total = values
        return dict(name=name, calls_per_step=n / steps,
                    self_cpu_us_per_step=own / steps, total_cpu_us_per_step=total / steps)
    top_cpu = [cpu_row(name, row) for name, row in
               sorted(cpu_ops.items(), key=lambda kv: -kv[1][1])[:20]]
    blocks = {name: cpu_row(name, row) for name, row in cpu_ops.items() if name.startswith("diag/")}
    # Correlate eager launches with the innermost diagnostic CPU range. Graph
    # replays are labeled by capture owner; graph-node visibility is checked below.
    annotations = [e for e in cpu if e["name"].startswith("diag/")]
    correlations = {}
    for event in runtime:
        parents = [p for p in annotations if p["ts"] <= event["ts"] < p["ts"] + p["dur"]]
        if parents:
            owner = min(parents, key=lambda p: p["dur"])["name"]
            correlations[event.get("args", {}).get("correlation")] = owner
    owned = Counter(correlations.get(e.get("args", {}).get("correlation"), "unattributed") for e in kernels)
    owned_us = Counter()
    for event in kernels:
        owned_us[correlations.get(event.get("args", {}).get("correlation"), "unattributed")] += event["dur"]
    return dict(
        steps=steps, device_kernels_per_step=len(kernels) / steps,
        host_kernel_launch_calls_per_step=sum(n for k, n in api_counts.items()
                                              if "LaunchKernel" in k) / steps,
        graph_launch_calls_per_step=sum(n for k, n in api_counts.items() if "GraphLaunch" in k) / steps,
        synchronization_calls_per_step={k: n / steps for k, n in sync.items()},
        dtoh_copies_per_step=len(dtoh) / steps,
        scalar_read_ops_per_step={k: n / steps for k, n in host_reads.items()},
        cuda_api_calls_per_step={k: n / steps for k, n in api_counts.items()},
        gpu_busy_union_ms_per_step=union_us((e["ts"], e["ts"] + e["dur"]) for e in kernels) / steps / 1000,
        gpu_kernel_sum_ms_per_step=sum(e["dur"] for e in kernels) / steps / 1000,
        gpu_copy_union_ms_per_step=union_us((e["ts"], e["ts"] + e["dur"]) for e in copies) / steps / 1000,
        kernels_by_correlated_block_per_step={k: n / steps for k, n in owned.items()},
        gpu_us_by_correlated_block_per_step={k: duration / steps for k, duration in owned_us.items()},
        top_cpu=top_cpu, top_gpu=top_gpu, blocks=blocks,
        trace=str(trace_path),
        caveat="GPU durations and elapsed wall reflect contention; CPU launch waits can too. "
               "Sync APIs, scalar ops and DtoH copies overlap and must not be added together.")


class StepWindow:
    def __init__(self, args, trace_path):
        self.args, self.trace_path = args, trace_path
        self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA])
        self.started = self.stopped = False

    @contextmanager
    def step(self, n):
        if n == self.args.warmup:
            torch.cuda.synchronize()
            self.prof.start()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.start = time.perf_counter()
            self.started = True
        measured = self.args.warmup <= n < self.args.warmup + self.args.steps
        with torch.profiler.record_function("diag/step") if measured else _nullcontext():
            yield
        if n + 1 == self.args.warmup + self.args.steps:
            self.submit_end = time.perf_counter()
            self.end_event.record()
            # This is a measurement-boundary wait, outside diag/step. It is
            # deliberately excluded from the per-step synchronization count.
            self.end_event.synchronize()
            self.complete_end = time.perf_counter()
            self.prof.stop()
            self.stopped = True

    def report(self):
        if not self.stopped:
            raise RuntimeError("The solver did not complete the requested profiling window")
        result = summarize(self.prof, self.trace_path, self.args.steps, "diag/step")
        result.update(cpu_submit_wall_ms_per_step=(self.submit_end - self.start) * 1000 / self.args.steps,
                      completion_wall_ms_per_step=(self.complete_end - self.start) * 1000 / self.args.steps,
                      cuda_event_elapsed_ms_per_step=self.start_event.elapsed_time(self.end_event) / self.args.steps)
        if result["device_kernels_per_step"] == 0:
            raise RuntimeError("CUPTI did not expose kernels; zero is not a valid launch-count result")
        return result


@contextmanager
def _nullcontext():
    yield


@contextmanager
def instrument(stepping, window, capture_log, *, eager_port=False, captured_calls=None):
    """Wrap the existing loop body, without reimplementing its step sequence."""
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner

    source = ast.parse(inspect.getsource(stepping.solve_steps))
    loops = [node for node in ast.walk(source) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "n"
             and isinstance(node.iter, ast.Name) and node.iter.id == "iterator"]
    if len(loops) != 1:
        raise RuntimeError("Expected exactly one ordinary step loop; inspect the updated runtime")
    loop = loops[0]
    loop.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Attribute(value=ast.Name(id="_diag_window", ctx=ast.Load()), attr="step", ctx=ast.Load()),
        args=[ast.Name(id="n", ctx=ast.Load())], keywords=[]))], body=loop.body)]
    namespace = dict(vars(stepping), _diag_window=window)
    exec(compile(ast.fix_missing_locations(source), "<copper-diagnostic-loop>", "exec"), namespace)
    originals = [(stepping, "solve_steps", stepping.solve_steps),
                 (CudaGraphRunner, "capture", CudaGraphRunner.capture)]
    stepping.solve_steps = namespace["solve_steps"]
    original_capture = CudaGraphRunner.capture

    def capture(runner, fn):
        caller = inspect.currentframe().f_back.f_code.co_name
        label = f"diag/graph/{caller}"
        entry = dict(caller=caller, callable=getattr(fn, "__qualname__", repr(fn)))
        if caller == "run" and "_make_port_step_runner" in entry["callable"]:
            label = "diag/graph/_make_port_step_runner"
        capture_log.append(entry)
        try:
            replay = original_capture(runner, fn)
        except Exception as error:
            entry.update(success=False, error=repr(error))
            raise
        entry["success"] = True
        if captured_calls is not None:
            captured_calls[label] = fn

        def run():
            with torch.profiler.record_function(label):
                return replay()
        return run

    CudaGraphRunner.capture = capture
    if eager_port:
        originals.append((stepping, "_make_port_step_runner", stepping._make_port_step_runner))
        stepping._make_port_step_runner = lambda *a, **k: None
    # The clone resolves symbols in namespace, so wrappers must be installed in
    # both namespaces. These scopes also label the separate eager inventory.
    names = ("begin_copper_step", "correct_copper_magnetic", "advance_copper_electric",
             "accumulate_copper_spectra", "_field_update_block", "_port_step_block",
             "apply_port_runtimes", "accumulate_port_observers", "_post_source_block",
             "update_magnetic_fields", "update_electric_fields", "enforce_pec_boundaries")
    for name in names:
        original = getattr(stepping, name)
        originals.append((stepping, name, original))

        def wrap(fn, label):
            @functools.wraps(fn)
            def call(*args, **kwargs):
                with torch.profiler.record_function(f"diag/block/{label}"):
                    return fn(*args, **kwargs)
            return call

        setattr(stepping, name, wrap(original, name))
    namespace.update(vars(stepping))
    try:
        yield
    finally:
        if window.started and not window.stopped:
            window.prof.stop()
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def metadata(solver):
    return dict(grid=[solver.Nx, solver.Ny, solver.Nz], cells=solver.Nx * solver.Ny * solver.Nz,
                dt_s=solver.dt, banks=[dict(component=e["electric_name"], edges=e["state"].shape[1],
                    state_rows=e["state"].shape[0], modes_per_parity=(len(e["rates"]) - 1) // 2,
                    state_dtype=str(e["state"].dtype), trace_shape=list(e["old_e"].shape),
                    adjacent_material_poles=len(e["material_ade"])) for e in solver._copper_foils],
                graph_flags={name: bool(getattr(solver, name, False)) for name in
                    ("_cuda_graph_active", "_port_cuda_graph_active", "_observer_cuda_graph_active", "_tail_graph_active")})


@contextmanager
def inventory_details(stepping):
    """Label electric subexpressions in the eager inventory, preserving operators."""
    import witwin.maxwell.fdtd.copper_foil as copper

    original = stepping.advance_copper_electric
    tree = ast.parse(inspect.getsource(copper.advance_copper_electric))

    class LabelAssignments(ast.NodeTransformer):
        def visit_Assign(self, node):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            selected = set(targets) & {"history", "rhs", "updated", "next_state", "loss"}
            if not selected:
                return node
            label = "diag/detail/" + sorted(selected)[0]
            return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="profiler", ctx=ast.Load()),
                attr="record_function", ctx=ast.Load()), args=[ast.Constant(label)], keywords=[]))], body=[node])

    tree = LabelAssignments().visit(tree)
    namespace = dict(vars(copper))
    exec(compile(ast.fix_missing_locations(tree), "<copper-inventory-details>", "exec"), namespace)
    detailed = namespace["advance_copper_electric"]

    def run(*args, **kwargs):
        with torch.profiler.record_function("diag/block/advance_copper_electric"):
            return detailed(*args, **kwargs)

    stepping.advance_copper_electric = run
    try:
        yield
    finally:
        stepping.advance_copper_electric = original


@contextmanager
def copper_call_sites(stepping):
    """Label copper statements in the diagnostic process, preserving arithmetic."""
    from witwin.maxwell.fdtd import copper_foil as copper

    names = ("begin_copper_step", "correct_copper_magnetic", "correct_copper_magnetic_support",
             "advance_copper_electric", "copper_spectral_weights", "accumulate_copper_spectra")
    originals, replacements = [], {}
    namespace = dict(vars(copper))
    for name in names:
        function = getattr(copper, name)
        source, first = inspect.getsourcelines(function)

        class Sites(ast.NodeTransformer):
            def generic_visit(self, node):
                node = super().generic_visit(node)
                if not isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr)):
                    return node
                label = f"diag/site/copper_foil.py:{first + node.lineno - 1}/{name}"
                return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                                        attr="profiler", ctx=ast.Load()),
                    attr="record_function", ctx=ast.Load()), args=[ast.Constant(label)], keywords=[]))],
                    body=[node])

        tree = Sites().visit(ast.parse("".join(source)))
        exec(compile(ast.fix_missing_locations(tree), "<copper-call-sites>", "exec"), namespace)
        replacements[name] = namespace[name]
    namespace.update(replacements)
    for name, replacement in replacements.items():
        originals.append((copper, name, getattr(copper, name)))
        setattr(copper, name, replacement)
        if hasattr(stepping, name):
            originals.append((stepping, name, getattr(stepping, name)))
            setattr(stepping, name, replacement)
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def attribute_captured_step(steady_path, eager_path, output):
    """Map replay nodes to their identical eager callable's Python launch sites."""
    def read(path):
        events = json.loads(path.read_text())["traceEvents"]
        scopes = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"]
        owners = {}
        for event in events:
            if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
                continue
            parents = [p for p in scopes if p["ts"] <= event["ts"] < p["ts"] + p["dur"]]
            owners[event.get("args", {}).get("correlation")] = sorted(parents, key=lambda p: -p["dur"])
        kernels = sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")),
                         key=lambda e: e["ts"])
        return [(e, [p["name"] for p in owners.get(e["args"]["correlation"], [])]) for e in kernels]

    eager = defaultdict(list)
    for event, sites in read(eager_path):
        label = next(s.removeprefix("diag/call/") for s in sites if s.startswith("diag/call/"))
        eager[label].append((event, sites[-1]))
    positions, rows, buckets = Counter(), [], defaultdict(
        lambda: dict(kernels=0, microseconds=0., memcopies=0, copy_microseconds=0.))
    for event, sites in read(steady_path):
        label = next((s for s in sites if s.startswith("diag/graph/")), None)
        if label:
            reference, site = eager[label][positions[label]]
            positions[label] += 1
            if reference["cat"] == "gpu_memcpy":
                if event["cat"] == "gpu_memcpy":
                    assert reference["name"] == event["name"]
                    assert reference["args"]["bytes"] == event["args"]["bytes"]
                else:
                    # CUDA can replace scalar DtoD copies with graph memcpy kernels.
                    assert reference["args"]["bytes"] == 4 and event["name"] == "memcpy32_post"
            else:
                assert reference["name"] == event["name"], (label, positions[label], "name")
                for key in ("grid", "block"):
                    assert reference["args"][key] == event["args"][key], (label, positions[label], key)
        else:
            site = sites[-1] if sites else "other/step control"
        if "begin_copper_step" in site:
            bucket = "copper begin"
        elif "correct_copper_magnetic" in site:
            bucket = "copper magnetic correction"
        elif "spectr" in site and "copper" in site:
            bucket = "copper spectra (clock, weights, real, imag, norm)"
        elif "advance_copper_electric" in site:
            bucket = "copper electric (H gathers, state, loss)"
        elif "port" in (label or site):
            bucket = "port"
        elif "update_magnetic_fields" in site or "update_electric_fields" in site:
            bucket = "field update"
        elif "enforce_pec_boundaries" in site:
            bucket = "PEC"
        else:
            bucket = "other (observer / step control)"
        if event["cat"] == "kernel":
            buckets[bucket]["kernels"] += 1
            buckets[bucket]["microseconds"] += event["dur"]
        else:
            buckets[bucket]["memcopies"] += 1
            buckets[bucket]["copy_microseconds"] += event["dur"]
        rows.append(dict(kernel=event["name"], microseconds=event["dur"], bucket=bucket,
                         kind=event["cat"], site=site, graph=label, node=event["args"].get("graph node id")))
    assert positions == Counter({label: len(values) for label, values in eager.items()})
    result = dict(kernels=sum(row["kind"] == "kernel" for row in rows),
                  memcopies=sum(row["kind"] == "gpu_memcpy" for row in rows), buckets=dict(buckets), launches=rows,
                  method="Captured nodes matched in stream order to the same callable's eager kernels; "
                         "kernel names and grid/block dimensions are asserted identical; four-byte DtoD copies "
                         "match CUDA's memcpy32_post nodes. Times are captured replay times.")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def history_layout_probe(solver, args, variant):
    """Measure storage-layout candidates only; never replace the solver operator.

    Equality here is evidence on sampled operands, not an equivalence proof or
    permission to change reduction order in the library.
    """
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner

    observations = []
    for bank, entry in enumerate(solver._copper_foils):
        matrix = entry["B"].T * ((1 + entry["decay"]) / 2)[None, :]
        state = entry["state"].clone()
        contiguous = matrix.contiguous()
        reference = matrix @ state
        candidate = contiguous @ state
        generator = torch.Generator(device=solver.device).manual_seed(20261030 + bank)
        sampled_state = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
        sampled_reference = matrix @ sampled_state
        sampled_candidate = contiguous @ sampled_state
        observation = dict(bank=bank, matrix_shape=list(matrix.shape), state_shape=list(state.shape),
            original_stride=list(matrix.stride()), contiguous_stride=list(contiguous.stride()),
            actual_state_equal=bool(torch.equal(reference, candidate)),
            sampled_state_equal=bool(torch.equal(sampled_reference, sampled_candidate)),
            sampled_max_abs_difference=float((sampled_reference - sampled_candidate).abs().max()), layouts={})
        # Fixed output and input tensors: benchmark only the matrix product.
        for name, operand in (("original", matrix), ("contiguous", contiguous)):
            output = torch.empty_like(reference)
            replay = CudaGraphRunner(device=solver.device, warmup_steps=3).capture(
                lambda: torch.mm(operand, state, out=output))
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function("diag/history-layout"):
                    for _ in range(100):
                        replay()
                torch.cuda.synchronize()
            observation["layouts"][name] = summarize(prof,
                args.output / f"{variant}-history-{bank}-{name}.json", 100, "diag/history-layout")
        observations.append(observation)
    return observations


def profile_variant(args, variant):
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.runtime import stepping

    scene, frequencies, excitations = build_scene(variant)
    simulation = mw.Simulation.fdtd(scene, frequencies=frequencies, excitations=excitations,
        run_time=mw.TimeConfig(time_steps=args.warmup + args.steps),
        full_field_dft=False, shutoff=0., cuda_graph=True)
    prepared = simulation.prepare()
    solver = prepared.solver
    solver.verbose = False
    capture_log = []
    captured_calls = {} if getattr(args, "attribute_step", False) else None
    window = StepWindow(args, args.output / f"{variant}-trace.json")
    with copper_call_sites(stepping) if captured_calls is not None else _nullcontext(), \
            instrument(stepping, window, capture_log, eager_port=args.eager_port,
                       captured_calls=captured_calls), torch.no_grad():
        # Pin full-field DFT off in all four cases. The public multi-frequency
        # no-port convenience policy otherwise forces a volume DFT only in the
        # point controls, defeating the ablation. Point/port/copper DFT stay on.
        solver.solve(time_steps=args.warmup + args.steps, dft_frequency=frequencies,
                     full_field_dft=False, use_cuda_graph=True, shutoff=0., enable_plot=False)
        result = dict(variant=variant, metadata=metadata(solver), captures=capture_log,
                      steady=window.report())
        if captured_calls is not None:
            assert args.steps == 1
            eager_path = args.output / f"{variant}-call-sites.json"
            if solver._port_runtimes:
                solver._port_observer_step.zero_()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as calls:
                for label, function in captured_calls.items():
                    with torch.profiler.record_function("diag/call/" + label):
                        function()
                torch.cuda.synchronize()
            calls.export_chrome_trace(str(eager_path))
            result["attribution"] = attribute_captured_step(window.trace_path, eager_path,
                args.output / f"{variant}-attribution.json")
        # Inventory is deliberately eager and runs AFTER the physical solve on
        # the same prepared solver. Its state is discarded, never used as output.
        # A fixed valid port-table index avoids indexing beyond the run horizon.
        counter = torch.zeros(1, device=solver.device, dtype=torch.int64)
        torch.cuda.synchronize()
        with inventory_details(stepping), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as inventory:
            with torch.profiler.record_function("diag/inventory"):
                for _ in range(args.inventory_steps):
                    stepping._field_update_block(solver, 0.)
                    if solver._port_runtimes:
                        stepping._port_step_block(solver, counter)
            torch.cuda.synchronize()
        result["eager_inventory"] = summarize(inventory, args.output / f"{variant}-inventory.json",
                                               args.inventory_steps, "diag/inventory")
    if args.probe_history_layout:
        with torch.no_grad():
            result["history_layout_probe"] = history_layout_probe(solver, args, variant)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--inventory-steps", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--eager-port", action="store_true", help="Disable only the R1 port-step graph in this process")
    parser.add_argument("--probe-history-layout", action="store_true",
                        help="Profile a contiguous FP64 history coefficient copy; report sampled equality")
    parser.add_argument("--attribute-step", action="store_true",
                        help="Attribute every kernel of --steps 1 to its Python call site")
    parser.add_argument("--output", type=Path, default=Path("/tmp/copper-step-floor-profile"))
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 3 or args.inventory_steps < 1:
        parser.error("steps/inventory-steps must be positive and warmup must be at least 3")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  steps=args.steps, warmup=args.warmup, eager_port=args.eager_port,
                  policy="Shared GPU 0: absolute elapsed and GPU-busy times are not exclusive performance claims.",
                  variants=[])
    if not torch.cuda.is_available():
        report.update(status="blocked", reason="CUDA is unavailable; no GPU measurements were made.")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 2
    report["gpu"] = torch.cuda.get_device_name(0)
    for variant in args.variants:
        print(f"Profiling {variant}", flush=True)
        result = profile_variant(args, variant)
        report["variants"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"variant": variant, "metadata": result["metadata"],
                          "steady": {k: v for k, v in result["steady"].items()
                                     if k not in ("top_cpu", "top_gpu", "blocks")}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

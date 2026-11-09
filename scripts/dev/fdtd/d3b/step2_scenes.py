"""Step 2 scene gates and captured kernel counts, without timing measurements.

Run each snapshot/count invocation in its own GPU-0 process. The 3 GiB Torch
allocator cap leaves context/library headroom inside the 4 GiB process limit.
Only scene construction is imported from the V15 driver; none of its measurement
functions are called. Profiler durations are never read or exported.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev/fdtd"))
from thin_sheet_overhead import SCENARIOS, _scene

import witwin.maxwell as mw
from witwin.maxwell.fdtd.cuda.build import STABLE_ABI_VERSION
from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner
from tests.materials.thin_sheet.test_thin_sheet_feeds import _simulation

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
SCENES = ("copper_patch", "copper_board_60k", "fed_dipole", "copper_patch_ricker")


def simulation(scenario, steps, graph=True):
    if scenario == "fed_dipole":
        return _simulation(device="cuda", tilted=True, steps=steps, graph=graph)
    if scenario == "copper_patch_ricker":
        # Well-conditioned control: a zero-mean Ricker current leaves no static charge residue,
        # so the strict E2 bounds apply (user decision 2026-11-01).
        scene = _scene(SCENARIOS["copper_patch"], True)
        scene.sources = [mw.PointDipole(position=source.position, polarization=source.polarization,
                                        width=source.width, profile=source.profile,
                                        source_time=mw.RickerWavelet(frequency=2.4e9))
                         for source in scene.sources]
        return mw.Simulation.fdtd(scene, frequencies=[2.4e9], run_time=mw.TimeConfig(time_steps=steps),
                                  cuda_graph=graph)
    return mw.Simulation.fdtd(_scene(SCENARIOS[scenario], True), frequencies=[2.4e9],
                              run_time=mw.TimeConfig(time_steps=steps), cuda_graph=graph)


def _add_roundoff_noise(prepared):
    """Calibration control: one FP64 ulp of random-sign noise per step on every FP64 tensor the fused
    copper kernels write and the torch path also holds: chord_e (via the rhs/solve), chord_f (forcing
    carried into the next rhs), the modal state and the loss accumulator."""
    import types
    sheet = prepared.solver._thin_sheets[0]
    generator = torch.Generator(device="cuda").manual_seed(7)
    original = sheet.electric

    def noisy(self, solver):
        original(solver)
        for tensor in (self.chord_e, self.chord_f, self.copper["state"], self.loss):
            sign = torch.randint(0, 2, tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)
            tensor.mul_(1 + (2 * sign - 1) * 2.0**-52)
    sheet.electric = types.MethodType(noisy, sheet)


def snapshot(args):
    noise = args.kernel == "torch-noise"
    os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = "torch" if noise else args.kernel
    prepared = simulation(args.scenario, 2000, graph=not noise).prepare()
    if noise:
        _add_roundoff_noise(prepared)
    result = prepared.run()
    solver = result.solver
    sheet = solver._thin_sheets[0]
    assert solver._cuda_graph_active != noise
    state = {name: getattr(solver, name).cpu() for name in FIELDS}
    state["loss"] = sheet.loss.cpu()
    for name, port in result.ports.items():
        for key in ("voltage", "current"):
            state[name+"_"+key] = getattr(port, key).cpu()
        for key in ("column_voltage", "variance_power", "field_accepted_power"):
            state[name+"_"+key] = port.metadata["sheet_gap"][key].cpu()
    assert any(bool(state[name].any()) for name in FIELDS)
    assert float(state["loss"]) > 0
    torch.save(dict(scenario=args.scenario, kernel=args.kernel, steps=2000, state=state,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3), args.out)
    print(dict(snapshot=str(args.out), scenario=args.scenario, kernel=args.kernel, steps=2000), flush=True)


def compare(args):
    fused = torch.load(args.fused, map_location="cpu", weights_only=True)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    assert fused["scenario"] == reference["scenario"] and fused["steps"] == reference["steps"] == 2000
    left, right = fused["state"], reference["state"]
    assert left.keys() == right.keys()
    metrics = {}
    for name in left:
        a, b = left[name].to(torch.complex128), right[name].to(torch.complex128)
        scale = torch.maximum(torch.linalg.vector_norm(a.reshape(-1)), torch.linalg.vector_norm(b.reshape(-1)))
        error = torch.linalg.vector_norm((a-b).reshape(-1))
        metrics[name] = float(torch.where(scale == 0, 0., error/scale))
    # Combine fields in physical units exactly as the requested relative L2 gate.
    a = torch.cat([left[name].double().reshape(-1) for name in FIELDS])
    b = torch.cat([right[name].double().reshape(-1) for name in FIELDS])
    metrics["field_l2"] = float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
    result = dict(scenario=fused["scenario"], steps=2000, errors=metrics,
                  peaks={kind: {key: data[key] for key in ("peak_allocated_gib", "peak_reserved_gib")}
                         for kind, data in (("fused", fused), ("torch", reference))})
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if args.control is None:
        # Strict bounds, applied per field component as well as to the aggregate.
        for name, value in metrics.items():
            assert value <= (1e-6 if name in FIELDS or name == "field_l2" else 1e-10), (name, value)
    else:
        # Self-calibrated (user decision 2026-11-01): within 2x the reference's own divergence
        # under one-ulp-per-step roundoff noise on the same scene.
        control = json.loads(args.control.read_text())["errors"]
        for name, value in metrics.items():
            assert value <= 2 * control[name], (name, value, control[name])


def _smooth_simulation(steps):
    """Well-conditioned E2 scene (user decision 2026-11-02): the V15 patch driven by a smooth
    1.5-cell Gaussian-profile dipole with a zero-mean Ricker pulse."""
    scene = _scene(SCENARIOS["copper_patch"], True)
    scene.sources = [mw.PointDipole(position=source.position, polarization=source.polarization, profile="gaussian",
                                    width=1.5 * SCENARIOS["copper_patch"]["cell"],
                                    source_time=mw.RickerWavelet(frequency=2.4e9)) for source in scene.sources]
    return mw.Simulation.fdtd(scene, frequencies=[2.4e9], run_time=mw.TimeConfig(time_steps=steps), cuda_graph=True)


def _fields(kernel, simulation_at, steps):
    os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = kernel
    solver = simulation_at(steps).run().solver
    return {name: getattr(solver, name).clone() for name in FIELDS} | {"loss": solver._thin_sheets[0].loss.clone()}


def smooth(args):
    """E2: peak-normalized per component over the run, sampled at 300/600/1000/2000 steps.
    The bounds sit above the fp32-field floor (E ~5e-7, H ~10x) that any non-bitwise path reaches."""
    samples = (300, 600, 1000, 2000)
    runs = {steps: (_fields("fused", _smooth_simulation, steps), _fields("torch", _smooth_simulation, steps))
            for steps in samples}
    peak = {name: max(float(torch.linalg.vector_norm(reference[name].double())) for _, reference in runs.values())
            for name in (*FIELDS, "loss")}
    errors = {steps: {name: float(torch.linalg.vector_norm((fused[name].double() - reference[name].double())))
                      / peak[name] for name in (*FIELDS, "loss")} for steps, (fused, reference) in runs.items()}
    bounds = {name: (1e-6 if name.startswith("E") else 1e-5 if name.startswith("H") else 1e-6)
              for name in (*FIELDS, "loss")}
    result = dict(scenario="copper_patch_smooth", samples=samples, peak=peak, errors=errors, bounds=bounds)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    for steps, row in errors.items():
        for name, value in row.items():
            assert value <= bounds[name], (steps, name, value)


def early(args):
    """Tight early check: fp32 fields bitwise equal for the first 50 steps on the V15 patch."""
    fused = _fields("fused", lambda steps: simulation("copper_patch", steps), 50)
    reference = _fields("torch", lambda steps: simulation("copper_patch", steps), 50)
    equal = {name: bool(torch.equal(fused[name], reference[name])) for name in FIELDS}
    args.out.write_text(json.dumps(dict(scenario="copper_patch", steps=50, bitwise=equal), indent=2) + "\n")
    print(dict(early_bitwise=equal), flush=True)
    assert all(equal.values()), equal


def counts(args):
    os.environ.pop("WITWIN_MAXWELL_THIN_SHEET_KERNEL", None)
    solver = simulation(args.scenario, 1).prepare().solver
    replay = _make_field_update_runner(solver, True)
    assert solver._cuda_graph_active
    for _ in range(20):
        replay(0.)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        replay(0.)
        torch.cuda.synchronize()
    events = prof.events()
    names = [event.name for event in events
             if event.device_type == torch.autograd.DeviceType.CUDA and not event.is_user_annotation]
    by_name = Counter(names)
    degree = solver._thin_sheets[0].copper["chebyshev"][2]
    expected = dict(thin_sheet_gather=4, thin_sheet_scatter=3, thin_sheet_copper_coupling=1,
                    thin_sheet_copper_rhs=1, thin_sheet_chord_forcing=1, thin_sheet_copper_state=1,
                    thin_sheet_chord_initialize=1,
                    thin_sheet_chord_residual=2 * degree, thin_sheet_chord_matvec=2 * degree,
                    thin_sheet_chord_direction=2 * degree)
    native = {part: sum(count for name, count in by_name.items() if part+"_kernel" in name)
              for part in expected}
    assert native == expected, (native, expected)
    assert sum(event.name == "cudaGraphLaunch" for event in events) == 1
    result = dict(scenario=args.scenario, abi=STABLE_ABI_VERSION, selector="default/fused", device=0,
                  captured=True, profiled_steps=1, total_kernels=len(names), phase0_total_kernels=263,
                  step1_total_kernels=193, fused_kernels=native, kernel_counts=dict(by_name),
                  chebyshev_degree=degree, expected_chord_solve_kernels=6 * degree + 1,
                  chords=solver._thin_sheets[0].copper["count"],
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("snapshot", "counts"):
        command = sub.add_parser(mode)
        command.add_argument("--scenario", choices=SCENES if mode == "snapshot" else SCENES[:2], required=True)
        command.add_argument("--out", type=Path, required=True)
        if mode == "snapshot":
            command.add_argument("--kernel", choices=("fused", "torch", "torch-noise"), required=True)
    for mode in ("smooth", "early"):
        sub.add_parser(mode).add_argument("--out", type=Path, required=True)
    command = sub.add_parser("compare")
    command.add_argument("--fused", type=Path, required=True)
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--control", type=Path, help="compare JSON of the torch-noise control (calibrated gate)")
    args = parser.parse_args()
    if args.mode in ("smooth", "early"):
        os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = "/tmp/pcb-d3b-build"
        os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/pcb-d3b-build"
    elif args.mode != "compare":
        os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = "/tmp/pcb-d3b-build"
        os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/pcb-d3b-build"
        torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    globals()[args.mode](args)


if __name__ == "__main__":
    main()

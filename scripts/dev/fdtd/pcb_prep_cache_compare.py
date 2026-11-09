"""Diagnose PCB material replay, cache lifetime and uncached repeatability."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

from witwin.maxwell.compiler.materials import _PCBConductivityReplay
from witwin.maxwell.scene import prepare_scene
from scripts.dev.fdtd import pcb_application_campaign as campaign


ROW_TIMINGS = {"prepare_seconds", "run_seconds", "cached_prepare_seconds", "prep_cache"}


def material_tensors(value, path="model"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from material_tensors(item, f"{path}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from material_tensors(item, f"{path}/{index}")
    elif is_dataclass(value):
        for field in fields(value):
            yield from material_tensors(getattr(value, field.name), f"{path}/{field.name}")


def sequence(args):
    """Audit the cache before/after prepare, stepping and power postprocessing."""
    audits, rows = [], []

    class AuditedPreparation(campaign.FamilyPreparation):
        def __init__(self):
            super().__init__()
            self.snapshots = [None, None]

        def audit(self, index, stage):
            changed = []
            for name, tensor in material_tensors(self.materials[index].model):
                original = self.snapshots[index][name]
                if not torch.equal(tensor, original):
                    changed.append({"tensor": name, "different": int(torch.count_nonzero(tensor != original))})
            audits.append({"frequency_hz": frequency, "path": ("direct", "adapter")[index],
                           "stage": stage, "changed": changed})
            print(json.dumps(audits[-1]), flush=True)

        def prepare(self, config, index):
            initialize = _PCBConductivityReplay.__init__

            def capture(replay, scene):
                initialize(replay, scene)
                self.snapshots[index] = {name: tensor.clone()
                                         for name, tensor in material_tensors(replay.model)}
                print(f"Captured {('direct', 'adapter')[index]} cache snapshot", flush=True)

            with patch.object(_PCBConductivityReplay, "__init__", capture):
                prepared = super().prepare(config, index)
            self.audit(index, "prepared")
            return prepared

    cache = AuditedPreparation()
    args.output.mkdir()
    for frequency in args.frequencies:
        destination = args.output / f"{frequency:.0f}"
        destination.mkdir()
        print(f"Preparing sequential target {frequency:.12g} Hz", flush=True)
        try:
            row = campaign.run("metamaterial-solid", output=destination,
                               duration=args.final_duration if frequency == args.frequencies[-1] else args.duration,
                               frequency=frequency, selections={}, input_dir=args.input_dir,
                               with_power=True, prep_cache=cache)
            rows.append(row)
        finally:
            for index, replay in enumerate(cache.materials):
                if replay is not None:
                    cache.audit(index, "completed")
            (args.output / "audit.json").write_text(json.dumps({"audits": audits, "runs": rows}, indent=2) + "\n")
        print(f"Sequential gate: {frequency:.12g} Hz passed", flush=True)
    assert not any(row["changed"] for row in audits), f"Cached tensors changed; see {args.output / 'audit.json'}"


def conductivity_tensors(model):
    for axis, tensor in model["sigma_e_components"].items():
        yield f"sigma_e_components/{axis}", tensor
    yield "sigma_e", model["sigma_e"]
    if model["edge_components"] is not None:
        for component, tensor in model["edge_components"]["sigma_e"].items():
            yield f"edge_components/sigma_e/{component}", tensor


def conductivity_differences(replayed, compiled):
    """Report ULP distances for the nonnegative float conductivity fields."""
    result = {}
    assert replayed.keys() == compiled.keys()
    for name, left in replayed.items():
        right = compiled[name]
        assert left.shape == right.shape and left.dtype == right.dtype, name
        integer = {torch.float32: torch.int32, torch.float64: torch.int64}[left.dtype]
        distance = (left.contiguous().view(integer).to(torch.int64)
                    - right.contiguous().view(integer).to(torch.int64)).abs()
        result[name] = {"different": int(torch.count_nonzero(distance)), "max_ulp": int(distance.max()),
                        "dtype": str(left.dtype)}
    return result


def materials(args):
    """Compile only materials, with each path's replay seeded at its first target."""
    rows = []
    for family in args.names:
        definition = json.loads((args.input_dir / f"metahz-scene-{family}.json").read_bytes())
        selected = definition["initial_conductor_ids"]
        declared = definition["frequencies_hz"]
        frequencies = args.frequencies
        if frequencies is None:
            frequencies = declared[::4] if family.startswith("metamaterial-") else declared
            if family == "metamaterial-solid":
                frequencies = sorted(set(frequencies) | {6.4e9})
        for index, path in enumerate(("direct", "adapter")):
            print(f"Building {family}/{path} replay at {declared[0]:.12g} Hz", flush=True)
            scenes = campaign.application_scenes(definition, declared[0], selected, with_power=False)
            replay = _PCBConductivityReplay(prepare_scene(scenes[index]))
            del scenes
            equal = 0
            for frequency in frequencies:
                print(f"Compiling {family}/{path} at {frequency:.12g} Hz", flush=True)
                scenes = campaign.application_scenes(definition, frequency, selected, with_power=False)
                scene = prepare_scene(scenes[index])
                values = {structure.name: float(structure.material.sigma_e) for structure in scene.structures
                          if not structure.material.is_pec and not isinstance(structure.material, campaign.mw.CopperFoil)}
                compiled, replayed = scene.compile_materials(), replay.compile(values)
                tensors = conductivity_differences(dict(conductivity_tensors(replayed)),
                                                   dict(conductivity_tensors(compiled)))
                passed = all(value["different"] == 0 for value in tensors.values())
                equal += passed
                rows.append({"family": family, "path": path, "frequency_hz": frequency,
                             "seed_frequency_hz": declared[0], "equal": passed, "tensors": tensors})
                print(json.dumps(rows[-1]), flush=True)
                del scenes, scene, compiled, replayed
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
            print(f"{family}/{path}: {equal}/{len(frequencies)} equal; "
                  f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB", flush=True)
            del replay
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["equal"] for row in rows), f"Unequal material tensors; see {args.output}"


def deterministic_trial(args, index):
    """Fresh adapter Scene and full preparation; no preparation cache or reuse."""
    raw = (args.input_dir / "metahz-scene-metamaterial-solid.json").read_bytes()
    definition = json.loads(raw)
    scene = campaign.import_pcb_scene(definition, frequency=args.frequency,
                                      selected_conductors=definition["initial_conductor_ids"], device="cuda")
    scene = scene.clone(monitors=campaign.power_observations(definition, scene, args.frequency))
    config = campaign.simulation(scene, args.frequency, with_power=True)
    print(f"Trial {index}: full uncached adapter preparation", flush=True)
    prepared = config.prepare()
    solver = prepared.solver
    steps = math.ceil(args.duration / solver.dt)
    config.config.run_time = campaign.mw.TimeConfig(time_steps=steps)
    # Retain references only: no extra device allocations or synchronization
    # before stepping. These constitutive tensors remain read-only during a run.
    coefficients = {name: value for name, value in vars(solver).items()
                    if isinstance(value, torch.Tensor) and name.startswith(("ce", "ch", "eps_", "mu_", "sigma_"))}
    coefficients["model"] = solver._compiled_material_model
    coefficients["foil"] = [{name: value for name, value in entry.items()
                             if isinstance(value, torch.Tensor)
                             and name not in ("state", "right_e", "old_e", "cumulative_loss")}
                            for entry in solver._copper_foils]
    mutable = {"edge_buffer", "correction_buffer", "fast_state_scratch", "inductor_current", "capacitor_voltage"}
    coefficients["ports"] = [{name: value for name, value in vars(runtime.lumped).items()
                              if isinstance(value, torch.Tensor) and not name.startswith("last_") and name not in mutable}
                             for runtime in solver._port_runtimes]
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = prepared.run()
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - start
    result.save(args.output / f"result-{index:02d}.pt")
    ports = {}
    for declared in scene.ports:
        port = result.port(declared.name)
        ports[declared.name] = {name: getattr(port, name) for name in (
            "voltage", "current", "z_in", "reflection_coefficient", "a", "b", "incident_power",
            "reflected_power", "accepted_power", "delivered_power", "available_power")}
    antenna = result.antenna(surface="radiation", driven_port=scene.ports[0].name,
                             theta_points=91, phi_points=73, radius=10.)
    loss = result.power_loss("bulk_loss")
    copper = sum((row["surface_power"].to(loss.total.device) for row in result.conductor_losses.values()),
                 torch.zeros_like(loss.total))
    power = {"accepted_w": antenna.p_accepted, "incident_w": antenna.p_incident, "radiated_w": antenna.p_rad,
             "eirp_w": antenna.eirp, "radiation_intensity_w_sr": antenna.radiation_intensity,
             "bulk_channels_w": loss.channel_power, "bulk_total_w": loss.total,
             "bulk_density_w_m3": loss.volume_density, "copper_w": copper,
             "conductor_w": {name: row["surface_power"] for name, row in result.conductor_losses.items()},
             "relative_power_closure": (antenna.p_rad + loss.conduction + copper - antenna.p_accepted).abs()
                                       / antenna.p_accepted.abs()}
    payload = {"outputs": {name: tensor.detach().cpu() for name, tensor in
                           material_tensors({"ports": ports, "power": power}, "outputs")},
               "prepared": {name: tensor.detach().cpu() for name, tensor in material_tensors(coefficients, "prepared")},
               "metadata": {"pid": os.getpid(), "frequency_hz": args.frequency, "duration_s": steps * solver.dt,
                            "steps": steps, "dt_s": solver.dt, "run_seconds": run_seconds,
                            "prep_cache": False, "path": "adapter", "input_sha256": hashlib.sha256(raw).hexdigest(),
                            "torch": torch.__version__, "source": campaign.mw.__file__,
                            "core": sys.modules["witwin.core"].__file__,
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    torch.save(payload, args.output / f"run-{index:02d}.pt")
    print(json.dumps(payload["metadata"]), flush=True)


def determinism(args):
    args.output.mkdir()
    for index in range(args.runs):
        deterministic_trial(args, index)


def tensor_differences(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    first = left.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
    second = right.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
    first, second = first.reshape(left.numel(), left.element_size()), second.reshape(right.numel(), right.element_size())
    different = int(torch.count_nonzero(torch.any(first != second, dim=1)))
    dtype = torch.complex128 if left.is_complex() else torch.float64
    left, right = left.to(dtype), right.to(dtype)
    delta = (left - right).abs()
    relative = delta / right.abs().clamp_min(torch.finfo(delta.dtype).tiny)
    return {"different": different, "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "max_relative": float(relative.max()) if delta.numel() else 0.0}


def determinism_compare(args):
    files = [path for directory in args.inputs for path in sorted(directory.glob("run-*.pt"))]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    report = {"files": list(map(str, files)), "runs": len(files), "metadata": [row["metadata"] for row in payloads]}
    for section in ("outputs", "prepared"):
        report[section] = {}
        for name in payloads[0][section]:
            comparisons = [tensor_differences(left[section][name], right[section][name])
                           for index, left in enumerate(payloads) for right in payloads[index + 1:]]
            report[section][name] = {
                "equal_pairs": sum(row["different"] == 0 for row in comparisons), "total_pairs": len(comparisons),
                "max_abs": max(row["max_abs"] for row in comparisons),
                "max_relative": max(row["max_relative"] for row in comparisons)}
            if section == "outputs" or any(row["different"] for row in comparisons):
                print(name, json.dumps(report[section][name]), flush=True)
    report["all_outputs_equal"] = all(row["equal_pairs"] == row["total_pairs"] for row in report["outputs"].values())
    report["all_prepared_equal"] = all(row["equal_pairs"] == row["total_pairs"] for row in report["prepared"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("runs", "all_outputs_equal", "all_prepared_equal")}), flush=True)


def run_targets(args):
    """Use the campaign CLI unchanged, measuring the whole target externally."""
    definition = json.loads((args.input_dir / f"metahz-scene-{args.family}.json").read_bytes())
    midpoint = len(definition["frequencies_hz"]) // 2
    wall_times = []
    original_run = campaign.run

    def timed_run(name, **kwargs):
        start = time.perf_counter()
        row = original_run(name, **kwargs)
        wall_times.append({"family": name, "frequency_hz": kwargs["frequency"],
                           "wall_seconds": time.perf_counter() - start})
        return row

    campaign.run = timed_run
    sys.argv = [str(Path(campaign.__file__)), "--output", str(args.output),
                "--input-dir", str(args.input_dir), "--names", args.family,
                "--declared-band", "--power", "--duration", "12e-9", "--fail-fast",
                "--shard-count", str(midpoint), "--shard-index", "0"]
    if args.no_prep_cache:
        sys.argv.append("--no-prep-cache")
    campaign.main()
    assert [row["frequency_hz"] for row in wall_times] == [
        definition["frequencies_hz"][index] for index in (0, midpoint, 2 * midpoint)]
    (args.output / "wall-times.json").write_text(json.dumps(wall_times, indent=2) + "\n")


def equal_tensors(left, right, path):
    """Check every saved tensor, including nested field, port and power payloads."""
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
        return 1
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return sum(equal_tensors(left[key], right[key], f"{path}/{key}") for key in left)
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        return sum(equal_tensors(a, b, f"{path}/{index}")
                   for index, (a, b) in enumerate(zip(left, right)))
    return 0


def compare(args):
    timings = []
    for family in campaign.NAMES:
        cached, uncached = args.cached / family, args.uncached / family
        reports = [json.loads((directory / "report.json").read_text()) for directory in (cached, uncached)]
        for report in reports:
            assert report["failures"] == [] and len(report["runs"]) == report["expected_runs"] == 3
            assert all(report[key] for key in ("inputs_unchanged", "source_unchanged", "driver_unchanged"))
        for a, b in zip(reports[0]["runs"], reports[1]["runs"]):
            assert a["prep_cache"] is True and b["prep_cache"] is False
            assert {key: value for key, value in a.items() if key not in ROW_TIMINGS} == {
                key: value for key, value in b.items() if key not in ROW_TIMINGS}, (family, a["frequency_hz"])
        assert {key: value for key, value in reports[0].items() if key != "runs"} == {
            key: value for key, value in reports[1].items() if key != "runs"}, family
        files = sorted(cached.rglob("*.pt"))
        assert [path.relative_to(cached) for path in files] == [
            path.relative_to(uncached) for path in sorted(uncached.rglob("*.pt"))]
        assert len(files) == 6, family
        tensors = 0
        for path in files:
            other = uncached / path.relative_to(cached)
            tensors += equal_tensors(torch.load(path, map_location="cpu", weights_only=False),
                                     torch.load(other, map_location="cpu", weights_only=False), str(path))
        assert tensors > 0, family
        wall = [json.loads((directory / "wall-times.json").read_text()) for directory in (cached, uncached)]
        for a, b in zip(*wall):
            assert a["family"] == b["family"] == family and a["frequency_hz"] == b["frequency_hz"]
            timings.append({"family": family, "frequency_hz": a["frequency_hz"],
                            "cached_wall_seconds": a["wall_seconds"],
                            "uncached_wall_seconds": b["wall_seconds"]})
        print(f"{family}: 3 report rows and {tensors} saved tensors bitwise equal", flush=True)
    print("family,frequency_hz,cached_wall_seconds,uncached_wall_seconds")
    for row in timings:
        print(f"{row['family']},{row['frequency_hz']:.12g},"
              f"{row['cached_wall_seconds']:.3f},{row['uncached_wall_seconds']:.3f}")
    (args.cached.parent / "prep-cache-comparison.json").write_text(json.dumps(timings, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run exactly three targets in one family process.")
    run.add_argument("--family", choices=campaign.NAMES, required=True)
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--no-prep-cache", action="store_true")
    check = commands.add_parser("compare", help="Compare five cached and uncached family directories on CPU.")
    check.add_argument("--cached", type=Path, required=True)
    check.add_argument("--uncached", type=Path, required=True)
    material = commands.add_parser("materials", help="Compare replay/full conductivity tensors without FDTD steps.")
    material.add_argument("--input-dir", type=Path, required=True)
    material.add_argument("--output", type=Path, required=True)
    material.add_argument("--names", nargs="+", choices=campaign.NAMES, default=campaign.NAMES)
    material.add_argument("--frequencies", nargs="+", type=float,
                          help="Override the dense declared-band sample (Hz).")
    sequential = commands.add_parser("sequence", help="Audit cached tensors across sequential solid solves.")
    sequential.add_argument("--input-dir", type=Path, required=True)
    sequential.add_argument("--output", type=Path, required=True)
    sequential.add_argument("--frequencies", nargs="+", type=float, required=True)
    sequential.add_argument("--duration", type=float, default=1e-9)
    sequential.add_argument("--final-duration", type=float, default=12e-9)
    repeat = commands.add_parser("determinism", help="Repeat fresh uncached solid adapter solves with power.")
    repeat.add_argument("--input-dir", type=Path, required=True)
    repeat.add_argument("--output", type=Path, required=True)
    repeat.add_argument("--runs", type=int, default=3)
    repeat.add_argument("--frequency", type=float, default=6.4e9)
    repeat.add_argument("--duration", type=float, default=12e-9)
    repeat_compare = commands.add_parser("determinism-compare", help="Compare every trial pair across processes on CPU.")
    repeat_compare.add_argument("--inputs", nargs="+", type=Path, required=True)
    repeat_compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_targets(args)
    elif args.command == "materials":
        materials(args)
    elif args.command == "sequence":
        sequence(args)
    elif args.command == "determinism":
        determinism(args)
    elif args.command == "determinism-compare":
        determinism_compare(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

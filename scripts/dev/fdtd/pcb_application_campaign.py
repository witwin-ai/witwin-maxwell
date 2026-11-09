"""Execute saved physical application inputs through direct and adapter Scene paths."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.adapters.pcb import import_pcb_scene
from witwin.maxwell.scene import prepare_scene
from scripts.dev.fdtd.pcb_acceptance import package_identity

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "docs/assessments/thin-copper-2026-10-22/site-scenes"
NAMES = ("metamaterial-solid", "metamaterial-hollow", "metamaterial-open-top", "pcb-flexible", "pcb-rigid")


def direct_scene(definition, adapted, frequency, selected):
    """Build physical structures directly, independently of the import implementation."""
    materials = {record["id"]: mw.Material(eps_r=record["epsilon_r"], mu_r=record.get("mu_r", 1),
        sigma_e=2 * math.pi * frequency * 8.8541878128e-12 * record["epsilon_r"] * record.get("loss", {}).get("tan_delta", 0))
        for record in definition["materials"]}
    structures = []
    for record in definition["dielectrics"] + definition["conductors"]:
        metal = record in definition["conductors"]
        if metal and record["role"] != "fixed" and record["id"] not in selected:
            continue
        bounds = np.asarray(record["bounds_m"], dtype=np.float64).reshape(3, 2)
        center, size = torch.tensor(bounds.mean(1)), torch.tensor(np.diff(bounds).ravel())
        model = record.get("constitutive_model", {"kind": "pec"})
        material = mw.Material.pec() if metal else materials[record["material_id"]]
        if metal and model["kind"] == "copper_foil":
            material = mw.CopperFoil(thickness=model["thickness_m"], conductivity=model["conductivity_s_per_m"],
                                      frequency_range=model["frequency_range_hz"])
            geometry = mw.Box(position=center, size=size)
        elif metal and record["kind"] == "sheet":
            normal = int(np.flatnonzero(size.numpy() == 0)[0])
            tangential = np.delete(bounds, normal, axis=0)
            geometry = mw.PlanarPEC.rectangle(tuple(map(tuple, tangential)),
                normal_axis="xyz"[normal], coordinate=float(bounds[normal, 0]))
        else:
            geometry = mw.Box(position=center, size=size)
        structures.append(mw.Structure(geometry, material, name=record["id"]))
    axes = [np.asarray(definition["mesh"]["axes_m"][axis]) for axis in "xyz"]
    # Build declared apertures and integration lines independently. An omitted
    # magnetic contour uses the adapter's native automatically resolved contour.
    ports = []
    for record, imported in zip(definition["ports"], adapted.ports):
        endpoints = np.asarray(record["integration_line_m"])
        bounds = np.asarray(record["sheet_bounds_m"]).reshape(3, 2)
        aperture = mw.Box(position=torch.tensor(bounds.mean(1)), size=torch.tensor(np.diff(bounds).ravel()))
        current_surface = imported.current_surface
        if "current_contour_bounds_m" in record:
            contour = np.asarray(record["current_contour_bounds_m"], dtype=np.float64).reshape(3, 2)
            current_surface = mw.Box(position=torch.tensor(contour.mean(1)),
                                     size=torch.tensor(np.diff(contour).ravel()))
        ports.append(mw.LumpedPort(record["id"], positive=endpoints[0], negative=endpoints[1],
            voltage_path=mw.AxisPath("xyz"[np.flatnonzero(endpoints[0] != endpoints[1])[0]]),
            voltage_aperture=aperture, current_surface=current_surface,
            reference_impedance=record["reference_ohm"]))
    return mw.Scene(domain=mw.Domain(bounds=[(float(a[0]), float(a[-1])) for a in axes]),
        grid=mw.GridSpec.custom(*axes), boundary=mw.BoundarySpec.pml(num_layers=definition["mesh"]["pml_cells"]),
        subpixel_samples=mw.SubpixelSpec(**definition['mesh']['subpixel_samples'])
            if 'subpixel_samples' in definition['mesh'] else 1,
        device=adapted.device, structures=structures, ports=ports,
        metadata={"application_path": "direct physical declarations", "selection": selected})


def power_observations(definition, scene, frequency):
    """Declare a closed vacuum surface between active geometry and the PML."""
    from witwin.maxwell.scene import prepare_scene
    grid = prepare_scene(scene)
    active = {structure.name for structure in scene.structures}
    records = [r for r in definition["dielectrics"] + definition["conductors"] if r["id"] in active]
    boxes = np.asarray([r["bounds_m"] for r in records]).reshape(-1, 3, 2)
    bounds = np.stack((boxes[:, :, 0].min(0), boxes[:, :, 1].max(0)), axis=-1)
    surface = np.empty_like(bounds)
    for dim, axis in enumerate("xyz"):
        nodes = getattr(grid, axis + "_nodes64")
        low = grid.pml_thickness_for_face(axis, "low")
        high = grid.pml_thickness_for_face(axis, "high")
        interior = (nodes[low+1], nodes[len(nodes)-high-2])
        if bounds[dim, 0] <= interior[0] or bounds[dim, 1] >= interior[1]:
            raise ValueError("Application geometry leaves no enclosing vacuum observation surface outside PML.")
        surface[dim] = ((interior[0]+bounds[dim, 0])/2, (interior[1]+bounds[dim, 1])/2)
    center, size = surface.mean(1), np.diff(surface).ravel()
    return [mw.ClosedSurfaceMonitor.box("radiation", position=center, size=size, frequencies=(frequency,)),
            mw.PowerLossMonitor("bulk_loss", position=center, size=size,
                                frequencies=(frequency,), channels=("conduction",))]


def simulation(scene, frequency, *, with_power=False):
    return mw.Simulation.fdtd(scene, frequency=frequency,
        excitations=mw.PortExcitation(scene.ports[0].name, source_time=mw.GaussianPulse(frequency=frequency, fwidth=frequency)),
        run_time=mw.TimeConfig(time_steps=1), full_field_dft=with_power,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))


def application_scenes(definition, frequency, selected, *, with_power, device="cuda"):
    adapted = import_pcb_scene(definition, frequency=frequency, selected_conductors=selected, device=device)
    direct = direct_scene(definition, adapted, frequency, selected)
    if with_power:
        direct = direct.clone(monitors=power_observations(definition, direct, frequency))
        adapted = adapted.clone(monitors=power_observations(definition, adapted, frequency))
    return direct, adapted


class FamilyPreparation:
    """Material replay for one shard's current family, separately for each path."""

    def __init__(self):
        self.materials = [None, None]

    def prepare(self, config, index):
        from witwin.maxwell.compiler.materials import _PCBConductivityReplay
        if self.materials[index] is None:
            self.materials[index] = _PCBConductivityReplay(config.scene)
        else:
            config.scene._material_model_override = self.materials[index].compile({
                structure.name: float(structure.material.sigma_e)
                for structure in config.scene.structures
                if not structure.material.is_pec and not isinstance(structure.material, mw.CopperFoil)})
        # Simulation still validates the target's imported loss-reference frequency.
        return config.prepare()


def declared_frequencies(definition, frequency, *, full_band):
    """Retain declared physical frequencies rather than relabeling a center solve."""
    values = tuple(float(value) for value in definition["frequencies_hz"]) if full_band else (float(frequency),)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Application observation frequencies must be finite and positive.")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Application observation frequencies must be strictly increasing.")
    return values


class _BitwiseGateFailure(AssertionError):
    def __init__(self, message, max_rel_diff):
        super().__init__(message)
        self.max_rel_diff = max_rel_diff
        self.gate_recheck = None


def _assert_bitwise_gate(outputs, powers):
    pairs = [(outputs[0][key], outputs[1][key]) for key in outputs[0]]
    if powers:
        pairs.extend((powers[0][key], powers[1][key]) for key in powers[0])
    try:
        for left, right in pairs:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    except AssertionError as error:
        max_relative = max(float(((left - right).abs().double()
                                 / right.abs().double().clamp_min(torch.finfo(right.real.dtype).tiny)).max())
                           for left, right in pairs)
        raise _BitwiseGateFailure(str(error), max_relative) from error


def _run_once(name, *, output, duration, frequency, selections, input_dir=FIXTURES, with_power=False,
              prep_cache=None):
    preparation_start = time.perf_counter()
    path = input_dir / f"metahz-scene-{name}.json"
    input_bytes = path.read_bytes()
    definition = json.loads(input_bytes)
    selected = selections.get(name, definition["initial_conductor_ids"])
    direct, adapted = application_scenes(definition, frequency, selected, with_power=with_power)
    configs = [simulation(prepare_scene(scene) if prep_cache is not None else scene,
                          frequency, with_power=with_power) for scene in (direct, adapted)]
    prepared, preparation_seconds = [], []
    for index, config in enumerate(configs):
        start = time.perf_counter()
        prepared.append(config.prepare() if prep_cache is None else prep_cache.prepare(config, index))
        preparation_seconds.append(time.perf_counter() - start)
    cached_prepare_seconds = time.perf_counter() - preparation_start
    a, b = [item.solver for item in prepared]
    coefficients_equal = all(torch.equal(getattr(a, key), getattr(b, key)) for key in (
        "cex_decay", "cex_curl", "cey_decay", "cey_curl", "cez_decay", "cez_curl"))
    pa, pb = a._port_runtimes[0].geometry, b._port_runtimes[0].geometry
    geometry_equal = torch.equal(pa.voltage_indices, pb.voltage_indices) and torch.equal(pa.voltage_weights, pb.voltage_weights)
    geometry_equal &= all(torch.equal(x, y) for x, y in zip(pa.current_indices + pa.current_weights,
                                                                         pb.current_indices + pb.current_weights))
    if not coefficients_equal or not geometry_equal or a.dt != b.dt:
        raise AssertionError("Direct and imported discrete physical contracts disagree.")
    steps = math.ceil(duration / a.dt)
    outputs, run_seconds, powers = [], [], []
    for config, item, label in zip(configs, prepared, ("direct", "adapter")):
        config.config.run_time = mw.TimeConfig(time_steps=steps)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = item.run()
        torch.cuda.synchronize()
        run_seconds.append(time.perf_counter() - start)
        result.save(output / f"{name}-{label}.pt")
        port = result.port(item.simulation.scene.ports[0].name)
        if not bool(torch.isfinite(port.voltage).all() and torch.isfinite(port.current).all()):
            raise AssertionError("Nonfinite port response.")
        outputs.append({"voltage": port.voltage.detach().cpu(), "current": port.current.detach().cpu(),
                        "reflection": port.reflection_coefficient.detach().cpu()})
        if with_power:
            antenna = result.antenna(surface="radiation", driven_port=port.port_name,
                                    theta_points=91, phi_points=73, radius=10.)
            bulk = result.power_loss("bulk_loss").conduction
            copper = sum((row["surface_power"].to(bulk.device) for row in result.conductor_losses.values()),
                         torch.zeros_like(bulk))
            powers.append({"accepted_w": antenna.p_accepted.detach().cpu(),
                           "radiated_w": antenna.p_rad.detach().cpu(),
                           "bulk_conduction_w": bulk.detach().cpu(), "copper_w": copper.detach().cpu(),
                           "relative_power_closure": ((antenna.p_rad+bulk+copper-antenna.p_accepted).abs()
                                                      /antenna.p_accepted.abs()).detach().cpu()})
    _assert_bitwise_gate(outputs, powers)
    serial = {key: {"real": value.real.tolist(), "imag": value.imag.tolist()} for key, value in outputs[1].items()}
    return {"family": name, "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "input_unchanged": input_bytes == path.read_bytes(),
        "selected_conductor_ids": selected, "active_conductor_ids": adapted.metadata["pcb_import"]["active_conductor_ids"],
        "candidate_kind": definition.get("candidate_status", "original draft" if name not in selections else "explicit caller selection; physical qualification pending"),
        "frequency_hz": frequency, "frequencies_hz": (frequency,),
        "spectral_scope": "one physical frequency with independently calibrated material conductivity",
        "loss_model": "Static conductivity calibrated from declared loss tangent at frequency_hz; no constant-loss-tangent dispersion claim across the band.",
        "physical_duration_s": steps * a.dt, "steps": steps, "dt_s": a.dt,
        "diagnostics": prepared[1].diagnostics, "prepare_seconds": preparation_seconds, "run_seconds": run_seconds,
        "prep_cache": prep_cache is not None, "cached_prepare_seconds": cached_prepare_seconds,
        "compiled_coefficients_equal": coefficients_equal, "port_geometry_equal": bool(geometry_equal),
        "outputs_bitwise_equal": True, "port": serial,
        "power": {key: value.tolist() for key, value in powers[1].items()} if powers else None,
        "qualification": "full-duration path parity only; spatial/time/PML convergence and radiated/loss closure remain required"}


def run(name, *, output, duration, frequency, selections, input_dir=FIXTURES, with_power=False,
        prep_cache=None):
    options = dict(output=output, duration=duration, frequency=frequency, selections=selections,
                   input_dir=input_dir, with_power=with_power, prep_cache=prep_cache)
    try:
        return _run_once(name, **options)
    except _BitwiseGateFailure as first:
        if prep_cache is None:
            raise
        first.gate_recheck = {"cached_first": False, "max_rel_diff_first": first.max_rel_diff}
        # Release failed solver frames before allocating the two re-check runs.
        traceback.clear_frames(first.__traceback__)
        rows = []
        for label, cache in (("uncached", None), ("cached_second", prep_cache)):
            destination = output / f"gate-recheck-{label}"
            destination.mkdir()
            try:
                row = _run_once(name, **dict(options, output=destination, prep_cache=cache))
            except Exception as error:
                first.add_note(f"{label} re-check failed:\n{traceback.format_exc()}")
                traceback.clear_frames(error.__traceback__)
                row = None
            first.gate_recheck[label] = row is not None
            rows.append(row)
        if not first.gate_recheck["uncached"] or not first.gate_recheck["cached_second"]:
            raise
        for path in (output / "gate-recheck-uncached").glob("*.pt"):
            shutil.copy2(path, output / path.name)
        recovered = rows[0]
        recovered.update(gate_recheck=first.gate_recheck, recovered_after_recheck=True)
        return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--names", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--duration", type=float, default=12e-9)
    parser.add_argument("--frequency", type=float, default=2.45e9)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--input-dir", type=Path, default=FIXTURES)
    parser.add_argument("--power", action="store_true")
    parser.add_argument("--declared-band", action="store_true",
                        help="Reimport and run each declared frequency separately, retaining its physical tangent-loss calibration.")
    parser.add_argument("--no-prep-cache", action="store_true",
                        help="Repeat all preparation for every declared-band target.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Run only targets whose position in the ordered (family, frequency) list is congruent to this index.")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at the first failed target; record it and exit nonzero without running later targets.")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < --shard-count.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Preserve prior evidence; choose a new or empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    selections = json.loads(args.selections.read_text()) if args.selections else {}
    inputs = {}
    for name in args.names:
        path = args.input_dir/f"metahz-scene-{name}.json"
        raw = path.read_bytes()
        definition = json.loads(raw)
        inputs[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "frequencies_hz": declared_frequencies(definition, args.frequency, full_band=args.declared_band)}
    report = {"schema": "pcb-application-campaign/1", "source": package_identity(mw, "witwin-maxwell"),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "requested_duration_s": args.duration,
        "inputs": inputs, "runs": [], "failures": []}
    report['declared_band'] = args.declared_band
    report['frequency_contract'] = 'Each target is separately reimported at its own loss-reference frequency; no metadata bypass or automatic dispersive approximation.'
    targets = [(name, index, frequency) for name in args.names
               for index, frequency in enumerate(inputs[name]['frequencies_hz'])]
    selected_targets = targets[args.shard_index::args.shard_count]
    expected_runs = len(selected_targets)
    report['expected_runs'] = expected_runs
    report['shard'] = {"index": args.shard_index, "count": args.shard_count, "total_targets": len(targets),
                       "fail_fast": args.fail_fast,
                       "targets": [{"family": n, "index": i, "frequency_hz": f} for n, i, f in selected_targets]}
    current_family, prep_cache = None, None
    for name, index, frequency in selected_targets:
        if name != current_family:
            current_family = name
            prep_cache = FamilyPreparation() if args.declared_band and not args.no_prep_cache else None
        destination = args.output/name/f"frequency-{index:03d}-{frequency:.12g}" if args.declared_band else args.output
        destination.mkdir(parents=True, exist_ok=True)
        try:
            row = run(name, output=destination, duration=args.duration, frequency=frequency,
                      selections=selections, input_dir=args.input_dir, with_power=args.power, prep_cache=prep_cache)
            report["runs"].append(row)
            print(json.dumps({k: row[k] for k in ("family", "frequency_hz", "steps", "run_seconds", "outputs_bitwise_equal")}), flush=True)
        except Exception as error:
            failure = {"family": name, "frequency_hz": frequency, "traceback": traceback.format_exc()}
            if isinstance(error, _BitwiseGateFailure) and error.gate_recheck is not None:
                failure["gate_recheck"] = error.gate_recheck
            report["failures"].append(failure)
            print(traceback.format_exc(), flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Finished solvers sit in a reference cycle through their single-observer
        # engine, so release them before the next target instead of waiting for GC.
        gc.collect()
        torch.cuda.empty_cache()
        if args.fail_fast and report["failures"]:
            report["stopped_early"] = True
            break
    report['inputs_unchanged'] = all(
        record['sha256'] == hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
        for record in inputs.values())
    report["source_unchanged"] = report["source"]["source_sha256"] == package_identity(mw, "witwin-maxwell")["source_sha256"]
    report['driver_unchanged'] = report['driver_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if (report["failures"] or len(report["runs"]) != expected_runs or not report["source_unchanged"]
            or not report['driver_unchanged'] or not report['inputs_unchanged']
            or not all(row['input_unchanged'] for row in report['runs'])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

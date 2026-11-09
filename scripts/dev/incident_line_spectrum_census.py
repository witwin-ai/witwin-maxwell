"""CPU-only trigger census; construct/prepare scenes without time stepping.

Run with the project interpreter and CUDA_VISIBLE_DEVICES=''. JSON records actual
prepared timing where CPU preparation is supported. Otherwise diagnostics use
the source-resolution and vacuum Courant/grid rule, explicitly labelled as an
estimate (material/wire bounds can further shorten dt). Benchmark records use
their declared run_time_factor * longest_domain_side / c observation horizon.
No reference backend, fixed-k provider, solve, or optimization is invoked.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/incident-census/matplotlib")

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.scenes import SCENARIOS, build_scene
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.constants import C_0
from witwin.maxwell.fdtd.excitation.incident_line_spectrum import _spectrum_fractions, _warn_spectrum
from witwin.maxwell.fdtd.excitation.tfsf_common import solve_auxiliary_step, solve_numerical_wavenumber
from witwin.maxwell.fdtd.runtime.initialization import maximum_declared_frequency, _scene_material_characteristic_frequency
from witwin.maxwell.fdtd.runtime.timeplan import source_time_offset
from witwin.maxwell.scene import prepare_scene


def _estimated_inputs(scene, frequencies):
    grid = prepare_scene(scene, lazy_meshgrid=True)
    spacings = {axis: float(getattr(grid, f"d{axis}_primal64").min()) for axis in "xyz"}
    smallest = [min(spacings[axis], float(getattr(grid, f"d{axis}_dual64").min())) for axis in "xyz"]
    frequency = max(maximum_declared_frequency(scene, max(frequencies)),
                    _scene_material_characteristic_frequency(scene))
    dt = min(1 / (30 * frequency), .9 / (C_0 * math.sqrt(sum(1 / h**2 for h in smallest))))
    temporal_sources = [source for source in scene.resolved_sources() if hasattr(source, "source_time")]
    sources = compile_fdtd_sources(
        SimpleNamespace(sources=temporal_sources), default_frequency=frequencies[0])
    return dt, spacings, sources


def inspect_scene(scene, *, frequencies, run_time_factor=8.0, simulation=None, solver_kind="fdtd"):
    """Return a serializable record, retaining a reason for every fallback."""
    scene = scene.clone(device="cpu")
    row = dict(solver=solver_kind, preparation="direct_estimate", checks=[], warnings=[],
        skipped_sources=[dict(source=getattr(source, "name", None), kind=type(source).__name__,
                              reason="Source has no declared temporal waveform.")
                         for source in scene.resolved_sources() if not hasattr(source, "source_time")])
    prepared = None
    if solver_kind == "fdtd":
        try:
            if simulation is None:
                simulation = mw.Simulation.fdtd(scene, frequencies=frequencies, backend="cpu_reference",
                    run_time=mw.TimeConfig(time_steps=1), full_field_dft=False,
                    spectral_sampler=mw.SpectralSampler(normalize_source=False))
            # Prepare only; suppress unrelated warnings and the one-step probe's
            # deliberately truncated spectrum. Diagnose the full record below.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prepared = simulation.prepare()
            dt = prepared.solver.dt
            sources = prepared.solver._compiled_sources
            row["preparation"] = "cpu_reference"
        except Exception as exc:
            row["prepare_error"] = f"{type(exc).__name__}: {exc}"
    else:
        row["prepare_error"] = "Frequency-domain scene: temporal diagnostic evaluated directly."
    if prepared is None:
        dt, spacings, sources = _estimated_inputs(scene, frequencies)
    if simulation is not None and simulation.config.run_time.time_steps not in (None, 1):
        steps = simulation.config.run_time.time_steps
    else:
        horizon = run_time_factor * max(b - a for a, b in scene.domain.bounds) / C_0
        steps = max(1, math.ceil(horizon / dt))
    row.update(dt=dt, record_steps=steps,
               threshold=max(1e-6, float(np.finfo(np.float32).eps) * math.sqrt(steps)))
    for index, source in enumerate(sources):
        waveform = source["source_time"]
        name = source.get("name") or f"{source['kind']}[{index}]"
        is_tfsf = source.get("injection", {}).get("kind") == "tfsf"
        ds, f_edge = None, None
        offset = source_time_offset(source)
        if is_tfsf and waveform["kind"] != "cw" and source["kind"] == "plane_wave":
            if prepared is not None:
                aux = (prepared.solver._tfsf_state or {}).get("auxiliary_grid")
                if aux is not None:
                    ds = float(aux.ds)
            elif not scene.boundary.uses_kind("bloch"):
                direction = source["direction"]
                axis = int(np.argmax(np.abs(direction)))
                ds = spacings["xyz"[axis]]
                if abs(direction[axis]) < 1 - 1e-12:
                    context = SimpleNamespace(c=C_0, dt=dt, source_omega=2 * math.pi * waveform["frequency"])
                    k_numeric = solve_numerical_wavenumber(context, direction, spacings)
                    ds = solve_auxiliary_step(context, k_numeric)
            if ds is not None:
                f_edge = math.asin(min(1., C_0 * dt / ds)) / (math.pi * dt)
                offset = 1.0
        fractions = _spectrum_fractions(waveform, f_edge=f_edge, dt=dt,
                                        record_steps=steps, sample_offset=offset)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            _warn_spectrum(fractions, f_edge=f_edge, line_name=name,
                           record_steps=steps, pole_period_modulus=1.)
        check = dict(source=name, kind=source["kind"], waveform=waveform["kind"], ds=ds,
                     f_edge=f_edge, sample_offset=offset, **fractions)
        row["checks"].append(check)
        for item in caught:
            row["warnings"].append(dict(**check, hazard="edge" if "axial band edge" in str(item.message) else "dc",
                                         message=str(item.message)))
    return row


def example_builders():
    """Discover named builders and SceneModule constructors, excluding main/run."""
    for path in sorted((ROOT / "examples").rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                 and (node.name == "build" or node.name.startswith("build_scene"))]
        modules = [node.name for node in tree.body if isinstance(node, ast.ClassDef)
                   and any(isinstance(base, ast.Attribute) and base.attr == "SceneModule" for base in node.bases)]
        if not names and not modules:
            yield path, None, "No callable scene builder or SceneModule constructor."
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"incident_census_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            for name in names + modules:
                builder = getattr(module, name)
                signature = inspect.signature(builder)
                required = [p.name for p in signature.parameters.values()
                            if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
                if required:
                    yield path, name, f"Required builder arguments: {required}"
                else:
                    yield path, builder, None
        except Exception as exc:
            yield path, None, f"{type(exc).__name__}: {exc}"


def _benchmark_entry(name):
    torch.set_num_threads(1)
    scenario = SCENARIOS[name]
    try:
        scene = build_scene(name)
        return inspect_scene(scene, frequencies=scenario.frequencies,
                             run_time_factor=scenario.run_time_factor, solver_kind=scenario.solver)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}", warnings=[])
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/incident-census/census.json"))
    parser.add_argument("--workers", type=int, default=4, help="Independent CPU preparation processes.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(scenes={}, skipped_examples=[])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_benchmark_entry, name): name for name in SCENARIOS}
        for future in as_completed(futures):
            key = f"benchmark/{futures[future]}"
            report["scenes"][key] = future.result()
            print(key, report["scenes"][key].get("preparation", "error"), flush=True)
    original_scene = mw.Scene

    def cpu_scene(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_scene(*args, **kwargs)

    for path, builder, reason in example_builders():
        key = str(path.relative_to(ROOT))
        if reason:
            report["skipped_examples"].append(dict(path=key, builder=builder, reason=reason))
            continue
        key += f"::{builder.__name__}"
        try:
            kwargs = {"device": "cpu"} if "device" in inspect.signature(builder).parameters else {}
            with patch.object(mw, "Scene", cpu_scene):
                value = builder(**kwargs)
                if isinstance(value, mw.SceneModule):
                    value = value.to_scene()
            simulation = value if isinstance(value, mw.Simulation) else None
            scene = simulation.scene if simulation is not None else value
            solver_kind = str(simulation.method.value) if simulation is not None else "fdtd"
            frequencies = simulation.frequencies if simulation is not None else (float(getattr(sys.modules[builder.__module__], "FREQUENCY", 1e9)),)
            report["scenes"][key] = inspect_scene(scene, frequencies=frequencies, solver_kind=solver_kind)
        except Exception as exc:
            report["scenes"][key] = dict(error=f"{type(exc).__name__}: {exc}", warnings=[])
        print(key, report["scenes"][key].get("preparation", "error"), flush=True)
    report["scenes"] = dict(sorted(report["scenes"].items()))
    rows = list(report["scenes"].values())
    hazards = Counter(w["hazard"] for row in rows for w in row["warnings"])
    report["summary"] = dict(benchmark_scenes=len(SCENARIOS), example_builders=len(rows) - len(SCENARIOS),
        skipped_examples=len(report["skipped_examples"]), errors=sum("error" in row for row in rows),
        preparation=dict(Counter(row.get("preparation", "error") for row in rows)),
        scenes_with_warnings=sum(bool(row["warnings"]) for row in rows), warnings=dict(hazards))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return int(report["summary"]["errors"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())

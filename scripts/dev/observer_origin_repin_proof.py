#!/usr/bin/env python3
"""Audit historical BITID payloads without changing their pins or expectations.

Run with the maxwell interpreter on CUDA. ``--self-test`` and ``--analyze-only``
are CPU-only. Original test helpers launch the original probes, including their
separate copper baselines and environment isolation. A read-only Python profile
hook records dt, frequencies and prepared provider tags in sidecar JSON; it does
not replace any solver operation or change the saved payload.

The residual is ||current - predicted||_inf / ||baseline||_inf, per key (not
relative to the phase increment). A pointwise relative maximum is also reported;
zeros have no relative phase. The fixed acceptance budget is 64 FP32 epsilons.
Spectrum-derived invariants use the same budget; trajectory/state invariants
and documented trace-label predictions compare bytes. The owning roughness
test excludes fitted coefficients from exact equality and checks their responses
instead; coefficient residuals are diagnostic only. A FAIL is evidence against
re-pinning that key.
"""
from __future__ import annotations

import argparse
import ast
import atexit
from contextlib import contextmanager
from functools import cache
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "coverage-observer-origin" / "repin-proof"
DESIGN = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "instamax-line-classc-triage-20261101/timing/origin-design/CORRECTION_TABLE.md"
REL_LIMIT = 64 * torch.finfo(torch.float32).eps
TESTS = {
    "boundary": "tests/boundaries/absorbers/test_boundary_bit_identity.py",
    "roughness": "tests/materials/surface_impedance/test_roughness_opt_in_bit_identity.py",
    "monitors": "tests/monitors/in_simulation/test_existing_monitor_bit_identity.py",
    "graph": "tests/rf/lumped/test_port_step_graph_bit_identity.py",
    "rf": "tests/rf/lumped/test_rf_port_bit_identity.py",
    "terminal": "tests/rf/terminal_wave/test_terminal_wave_bit_identity.py",
}
PROBES = (
    "tests/boundaries/absorbers/_boundary_state_probe.py",
    "tests/materials/surface_impedance/smooth_surface_probe.py",
    "tests/monitors/in_simulation/_bit_identity_runner.py",
    "tests/support/port_step_graph_bit_identity_scene.py",
    "tests/support/rf_port_bit_identity_scene.py",
    "tests/support/rf_port_bit_identity_terminal_wave_scene.py",
)
PROVENANCE = {"module", "__module", "__module__", "__file__", "module_file"}
COMPONENTS = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "EX", "EY", "EZ", "HX", "HY", "HZ"}
FITTED_CLASS = "fitted (excluded by the test)"


@cache
def roughness_fitted_keys():
    """Read the owning test's exact scope without importing the solver package."""
    tree = ast.parse((ROOT / TESTS["roughness"]).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_FITTED_KEYS" for target in node.targets):
            return frozenset(ast.literal_eval(node.value))
    raise ValueError("Roughness test has no _FITTED_KEYS comparison scope")


def provider_beta(provider, waveform):
    """Independent transcription of CORRECTION_TABLE, not a runtime call."""
    if provider in {"additive_J", "terminal_drive"}:
        return 0.5
    if provider in {"magnetic_M", "mixed_JM", "soft_equivalent_sheet", "mode_surface", "custom_field_surface"}:
        return 0.0
    if provider in {"plane_wave_axis_aligned", "plane_wave_ref_x_ez", "plane_wave_aux", "plane_wave_discrete_cw"}:
        return 1.0
    if provider in {"analytic_profile", "plane_wave_grating_slab_cw"}:
        return 1.0 if waveform == "cw" else 0.0
    raise ValueError(f"Unrecognized prepared provider: {provider!r}")


def result_clock(result=None, solver=None, prepared_clock=None):
    """Read scalar metadata only; never copy or evaluate field tensors."""
    stats = (result.solver_stats or {}) if result is not None else {}
    solver = getattr(result, "solver", None) if solver is None else solver
    candidates, leaves = {}, []

    def visit(value, path):
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
        elif isinstance(value, dict):
            if value.get("dt") is not None:
                candidates[path + "/dt"] = float(value["dt"])
            if value.get("columns"):
                visit(value["columns"], path + "/columns")
            else:
                leaves.append(value)
            for name in ("time_plan", "time_step"):
                if isinstance(value.get(name), dict) and value[name].get("dt") is not None:
                    candidates[f"{path}/{name}/dt"] = float(value[name]["dt"])

    visit(stats, "solver_stats")
    if solver is not None:
        candidates["solver.dt"] = float(solver.dt)
    if prepared_clock is not None:
        candidates["prepared_export.dt"] = float(prepared_clock["dt"])
    if not candidates or len(set(candidates.values())) != 1:
        raise ValueError(f"Result has no unambiguous executed dt: {candidates}")
    dt = next(iter(candidates.values()))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid executed dt: {candidates}")
    # Sweep columns must share the execution policy as well as the clock.
    policies = {(bool(item.get("normalize_source", False)), bool(item.get("full_field_dft", False)))
                for item in leaves}
    if len(policies) != 1:
        raise ValueError(f"Ambiguous column spectral policies: {policies}")
    normalize, full_field = next(iter(policies))
    prepared_clock = prepared_clock or {}

    def stamp(attribute, key):
        return getattr(solver, attribute, prepared_clock.get(key))

    return {
        "dt": dt,
        "dt_sources": candidates,
        "frequencies": [float(f) for f in result.frequencies] if result is not None else [],
        "normalize_source": bool(getattr(solver, "_normalize_source", normalize)),
        "full_field_dft": full_field,
        "provider": stamp("_normalization_reference_provider", "provider"),
        "waveform": stamp("_normalization_reference_waveform_kind", "waveform"),
        "beta": stamp("_normalization_label_offset", "beta"),
        "alpha": stamp("_normalization_time_offset", "alpha"),
        "compiled_providers": [
            {k: source[k] for k in ("reference_provider_kind", "reference_waveform_kind") if k in source}
            for source in getattr(solver, "_compiled_sources", ())
        ],
    }


def install_metadata_hook():
    """Loaded only in probe children through the temporary sitecustomize file."""
    destination = Path(os.environ["OBSERVER_REPIN_METADATA"])
    records, errors, exports = {}, [], {}
    probe_paths = {str(ROOT / path) for path in PROBES}
    batch_export_path = str(ROOT / "witwin/maxwell/execution/batch_runtime.py")

    def profile(frame, event, arg):
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        is_export = filename == batch_export_path and name == "_export"
        if filename not in probe_paths and not is_export:
            return
        local = frame.f_locals
        try:
            if is_export and event == "return" and arg is not None:
                # Retain the object with its scalar stamp to prevent id reuse.
                exports[id(arg)] = (arg, result_clock(arg, solver=local["solver"]))
            elif event == "call" and name == "_record" and "result" in local:
                result = local["result"]
                prepared = exports.get(id(result), (None, None))[1]
                records[local["prefix"]] = result_clock(result, prepared_clock=prepared)
            elif event == "return" and name in {"main", "collect"}:
                if "result" in local and not isinstance(local["result"], list):
                    records.setdefault("", result_clock(local["result"]))
                elif "solver" in local:
                    records.setdefault("", result_clock(solver=local["solver"]))
        except Exception:
            errors.append(traceback.format_exc())

    sys.setprofile(profile)

    def finish():
        sys.setprofile(None)
        destination.write_text(json.dumps({"records": records, "errors": errors}, indent=2) + "\n")

    atexit.register(finish)


@contextmanager
def metadata_environment(module, output, hook_dir):
    """Keep the helper's subprocess argv/cwd and baseline-build policy intact."""
    original = module.probe_environment

    def environment(tree, *, baseline):
        env = original(tree, baseline=baseline)
        # Only current probes need prepared provider stamps. Historical payloads
        # are collected with exactly the helper's original environment.
        if not baseline:
            env["PYTHONPATH"] += os.pathsep + str(hook_dir)
            env["OBSERVER_REPIN_METADATA"] = str(output.with_suffix(".metadata.json"))
        return env

    module.probe_environment = environment
    try:
        yield
    finally:
        module.probe_environment = original


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def bitwise(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
            return False
        # Compare bytes: torch.equal alone equates +0 and -0 and rejects equal NaNs.
        def bytes_of(value):
            # contiguous() may retain a non-unit stride on singleton views.
            # copy_ into fresh storage guarantees a unit-stride byte view.
            flat = torch.empty(value.numel(), dtype=value.dtype, device=value.device)
            flat.copy_(value.resolve_conj().resolve_neg().reshape(-1))
            return flat.view(torch.uint8)
        return torch.equal(bytes_of(left), bytes_of(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        import struct
        return struct.pack("!d", left) == struct.pack("!d", right)
    if isinstance(left, complex):
        return bitwise(left.real, right.real) and bitwise(left.imag, right.imag)
    return left == right


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        if not value and prefix:
            out[prefix] = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        if not value:
            out[prefix] = value
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}/{index}"))
    else:
        out[prefix] = value
    return out


def complex_value(value):
    return isinstance(value, complex) or isinstance(value, torch.Tensor) and value.is_complex()


def clock_for(key, clocks):
    matches = [prefix for prefix in clocks if not prefix or key.startswith(prefix + "/")]
    if not matches:
        raise ValueError(f"No executed clock for {key}")
    clock = clocks[max(matches, key=len)]
    if not (math.isfinite(clock["dt"]) and clock["dt"] > 0):
        raise ValueError("Invalid executed dt")
    if clock.get("provider") is not None:
        beta = provider_beta(clock["provider"], clock["waveform"])
        if clock.get("beta") != beta:
            raise ValueError(f"Prepared beta disagrees with design: {clock}")
    return clock


def spectral_rule(kind, key, value, clock):
    """Whitelist physical quantities, never infer movement from complex dtype."""
    parts = key.split("/")
    leaf = parts[-1]
    exponent = 1.0
    label, axis = None, 0
    if "/source_normalization_spectra/" in key and complex_value(value):
        beta = provider_beta(clock["provider"], clock["waveform"])
        return "provider denominator: exp(+i beta theta), before safe division", beta, 0
    if kind == "roughness":
        if key in {"run.foil.dft.EX", "run.foil.dft.EY", "run.foil.dft.EZ",
                   "run.foil.port.voltage", "run.foil.port.current"}:
            label = "ordinary raw electric/port: exp(+i theta)"
            axis = -1 if ".port." in key else 0
    elif kind in {"rf", "terminal"}:
        if len(parts) == 3 and leaf in {"voltage", "current"}:
            label, axis = "ordinary port V/I: exp(+i theta), H stagger retained", -1
    elif kind == "monitors":
        if len(parts) > 2 and parts[1] not in {"time_point", "time_flux"}:
            if leaf in COMPONENTS | {"data", "amplitude_forward", "amplitude_backward"} or (
                    "components" in parts and leaf in COMPONENTS):
                label = "ordinary field/modal: exp(+i theta), H -1/2 offset retained"
    elif kind == "graph":
        if "/port_runtime/" in key and leaf in {"_voltage_sum", "_current_sum"}:
            return ("raw port accumulator (including drive): exp(+i theta), H stagger retained", 1.0, 0)
        if "/fields/" in key and leaf in COMPONENTS and clock.get("full_field_dft"):
            label = "ordinary full-field spectrum: exp(+i theta)"
        elif "/monitors/" in key and not ({"t_e", "t_h"} & set(parts)) and (
                leaf in COMPONENTS | {"data", "amplitude_forward", "amplitude_backward"}):
            label = "ordinary monitor spectrum: exp(+i theta), H stagger retained"
        elif ("/ports/" in key or "/embedded_networks/" in key) and leaf in {"voltage", "current"}:
            label, axis = "ordinary port/network V/I: exp(+i theta), H stagger retained", -1
        elif "/ports/" in key and leaf in {"column_voltage", "column_current"} and "/sheet_gap/" in key:
            label = "sheet-gap electric/port spectrum: exp(+i theta)"
        elif "/monitors/" in key and leaf in {"current", "charge"} and "/wire" in key:
            exponent = 0.0
            label = "independent raw wire I/q clock unchanged"
    if label is None:
        return None
    if not complex_value(value):
        # A final time-domain E tensor is not a spectrum just because its key is 'fields'.
        return None
    if clock.get("normalize_source"):
        beta = provider_beta(clock["provider"], clock["waveform"])
        exponent -= beta
        label += f"; normalized denominator exp(+i beta theta), beta={beta:g}"
    return label, exponent, axis


def frequency_vector(kind, key, clock):
    frequencies = clock["frequencies"]
    if kind == "monitors" and key.startswith("monitor/"):
        parts = key.split("/")
        if parts[2].isdigit():
            frequencies = [frequencies[int(parts[2])]]
    if not frequencies:
        raise ValueError(f"No executed frequencies for {key}")
    return torch.tensor(frequencies, dtype=torch.float64)


def rotate(baseline, frequencies, dt, exponent, axis):
    value = torch.as_tensor(baseline).to(torch.complex128)
    phase = torch.exp(2j * math.pi * frequencies * dt * exponent)
    if len(frequencies) == 1:
        return value * phase[0]
    if value.ndim == 0 or value.shape[axis] != len(frequencies):
        raise ValueError(f"Frequency axis {axis} inconsistent with shape {tuple(value.shape)}")
    shape = [1] * value.ndim
    shape[axis] = len(frequencies)
    return value * phase.reshape(shape)


def residual(baseline, current, predicted):
    baseline = torch.as_tensor(baseline).to(torch.complex128)
    current = torch.as_tensor(current).to(torch.complex128)
    if current.shape != predicted.shape or baseline.shape != current.shape:
        raise ValueError("Prediction shape mismatch")
    if not bool(torch.isfinite(baseline).all() and torch.isfinite(current).all() and torch.isfinite(predicted).all()):
        raise ValueError("Nonfinite spectral value")
    error = (current - predicted).abs()
    maximum = float(error.max()) if error.numel() else 0.0
    scale = float(baseline.abs().max()) if baseline.numel() else 0.0
    relative = maximum / scale if scale else (0.0 if maximum == 0 else None)
    nonzero = baseline.abs() > 0
    pointwise = float((error[nonzero] / baseline.abs()[nonzero]).max()) if bool(nonzero.any()) else None
    return {"max_abs_residual": maximum, "baseline_max_abs": scale,
            "max_rel_residual": relative, "max_pointwise_rel_residual_nonzero": pointwise,
            "zero_baseline_entries": int((~nonzero).sum()),
            "zero_baseline_max_abs_residual": float(error[~nonzero].max()) if bool((~nonzero).any()) else 0.0,
            "passes": relative is not None and relative <= REL_LIMIT}


def derived_invariant_rule(kind, key):
    """Whitelist actual spectrum consumers, excluding trace and reference data."""
    parts = key.split("/")
    leaf = parts[-1]
    monitor = (kind == "monitors" and len(parts) > 2 and parts[0] in {"raw", "payload", "monitor"}
               and parts[1] not in {"time_point", "time_flux"}) or (
                   kind == "graph" and "/monitors/" in key and not {"t_e", "t_h"}.intersection(parts))
    if monitor:
        if leaf in {"flux", "power"}:
            return ("Poynting product E x conj(H), integrated after spectral rounding",
                    ["witwin/maxwell/fdtd/observers.py:_assemble_plane_monitor_payload",
                     "witwin/maxwell/fdtd/quadrature.py:plane_normal_poynting",
                     "witwin/maxwell/fdtd/quadrature.py:_compute_plane_flux"])
        if leaf in {"total_power", "power_forward", "power_backward", "power_fraction_forward", "power_fraction_backward"}:
            return ("modal overlap powers |a|^2 and ratios to integrated spectral power",
                    ["witwin/maxwell/postprocess/modal.py:_compute_mode_overlap_from_payload"])
    port = (kind in {"rf", "terminal"} and len(parts) == 3 and parts[1] not in {"field", "network"}) or (
        kind == "graph" and "/ports/" in key)
    if port and parts[-2:] == ["metadata", "excitation_threshold"]:
        return ("clamp(1e-6 * max(abs(drive-accumulator source-voltage phasor)), min=tiny)",
                ["witwin/maxwell/fdtd/ports.py:finalize_port_data"])
    if port and leaf == "available_power":
        return ("drive-accumulator source-voltage spectrum magnitude squared / (8 R)",
                ["witwin/maxwell/fdtd/ports.py:finalize_port_data"])
    if port and leaf in {"z_in", "reflection_coefficient", "return_loss", "return_loss_db", "vswr",
                         "incident_power", "reflected_power", "accepted_power", "delivered_power"}:
        return ("PortData V/I or b/a ratio, its scalar transforms, or power-wave magnitude squared",
                ["witwin/maxwell/network.py:PortData"])
    if kind == "graph" and len(parts) >= 4 and parts[-3] == "embedded_networks" and leaf in {
            "port_power", "absorbed_power", "generated_power"}:
        return ("0.5 * real(V * conj(I)); summed and clamped for absorbed/generated spectral power",
                ["witwin/maxwell/fdtd/networks.py:finalize_embedded_networks"])
    if port and "/metadata/sheet_gap/" in key and leaf in {
            "variance_power", "port_accepted_power", "field_accepted_power", "power_correction"}:
        return ("sheet-column spectral voltage variance or conjugate V/I power products",
                ["witwin/maxwell/fdtd/ports.py:finalize_port_data"])
    if kind in {"rf", "terminal", "graph"} and "/network/" in key and leaf == "s":
        return ("network b_out/a_in ratios or the B = S A solve cancel the common spectral phase",
                ["witwin/maxwell/network_sweep.py:aggregate_network_columns",
                 "witwin/maxwell/waveport_sweep.py:run_waveport_sweep"])
    return None


def spectrum_independent_rule(kind, key):
    """Producer audit for every graph/roughness key still requiring exact bytes."""
    parts = key.split("/")
    leaf = parts[-1]
    if kind == "roughness":
        probe = PROBES[1]
        if key.startswith(("compiled.", "layout.", "prepared.")):
            return ("Material/layout compilation or initialized solver state, before spectral accumulation",
                    [probe + ":collect", probe + ":_solver_tensors"])
        if key.startswith("run.") and key.rsplit(".", 1)[-1] in COMPONENTS:
            return ("Final time-domain Yee field; no spectral postprocessing",
                    [probe + ":collect", "witwin/maxwell/fdtd/runtime/stepping.py:solve_steps"])
        if key.startswith(("lossy.", "foil.")) or key == "rational.model.poles":
            return ("Analytic material response or material-fit reference data; no relabelled spectra",
                    [probe + ":collect"])
    elif kind == "graph":
        probe = PROBES[3]
        if "/fields/" in key and leaf in COMPONENTS:
            return ("Final time-domain electric field exported with full_field_dft=False; no spectral input",
                    ["witwin/maxwell/simulation.py:Simulation._fdtd_last_step_field_payload",
                     "witwin/maxwell/simulation.py:Simulation._run_fdtd_from_solver"])
        if "/solver/" in key or "/history/" in key or leaf == "steps_run":
            return ("Final Yee field, per-step V/I/drive history or executed step count; no spectral input",
                    [probe + ":_record_solver", probe + ":_record_history", probe + ":_record"])
        if "/port_runtime/" in key and leaf not in {"_voltage_sum", "_current_sum"}:
            return ("Live port/branch time-step state, sample clock/count or window-weight sum; no spectral input",
                    [probe + ":_record_solver", "witwin/maxwell/fdtd/ports.py:PortDFTAccumulator"])
        if "/copper/" in key and leaf in {"state", "right_e", "cumulative_loss"}:
            return ("Copper diffusion time-step state or accumulated time-domain loss; no spectral input",
                    ["witwin/maxwell/fdtd/copper_foil.py:advance_copper_electric", probe + ":_record_solver"])
        if "/circuits/" in key:
            return ("Circuit step samples, instantaneous powers, energy accounting or solve diagnostics; no DFT input",
                    ["witwin/maxwell/fdtd/circuits.py:EMCircuitRuntime.finalize"])
        if "/embedded_networks/" in key:
            return ("Network time-step state norm/energy totals or compiled model metadata; no spectral power input",
                    ["witwin/maxwell/fdtd/networks.py:_finalize_network_step",
                     "witwin/maxwell/fdtd/networks.py:finalize_embedded_networks"])
        if "/monitors/" in key:
            if {"t_e", "t_h"}.intersection(parts):
                return ("Raw time-domain monitor field samples or configured trace geometry/sampling metadata",
                        ["witwin/maxwell/fdtd/observers.py:get_time_observer_results"])
            if leaf in {"current", "charge"} and "/wire" in key:
                return ("Independent raw wire I/q clock; does not consume relabelled field/port spectra",
                        ["witwin/maxwell/fdtd/wire.py:finalize_wire_data"])
            return ("Configured monitor coordinates/indices/frequencies or sample counts, independent of DFT values",
                    ["witwin/maxwell/fdtd/observers.py:get_observer_results"])
        if "/network/" in key:
            return ("Sweep manifest, reference impedances/frequencies or constant all-valid column flags",
                    ["witwin/maxwell/network_sweep.py:aggregate_network_columns",
                     "witwin/maxwell/waveport_sweep.py:run_waveport_sweep"])
        if "/ports/" in key:
            return ("Configured port geometry/reference impedance, sampling metadata or prepared reference modes",
                    ["witwin/maxwell/fdtd/ports.py:finalize_port_data",
                     "witwin/maxwell/network_sweep.py:aggregate_network_columns",
                     "witwin/maxwell/waveport_sweep.py:resolve_waveport_run_manifest"])
    raise ValueError(f"No spectrum-independent producer classification for {kind}: {key}")


def analyze(kind, baseline, current, clocks, predictions=None):
    """All keys are accounted for, including additions, removals and unknown movers."""
    rows, unchanged, failures = [], [], []
    for key in sorted(set(baseline) | set(current)):
        old, new = baseline.get(key), current.get(key)
        present = key in baseline and key in current
        equal = present and bitwise(old, new)
        if equal:
            unchanged.append(key)
        row = {"key": key, "moved": not equal, "class": "bitwise", "rule": "bitwise invariant", "verdict": "PASS",
               "classification": "Probe trajectory/state, prepared/reference data, geometry or scalar metadata; no relabelled-spectrum computation",
               "code": [TESTS.get(kind, kind), PROBES[list(TESTS).index(kind)] if kind in TESTS else kind]}
        try:
            if not present:
                # The only new observer-origin payload keys are explicit trace offsets.
                if key not in baseline and re.search(r"/(time_point|time_flux|t_e|t_h)/.*field_time_offsets_steps/[EH]$", key):
                    expected = -0.5 if key.endswith("/H") else 0.0
                    if new != expected:
                        raise ValueError("Incorrect new trace offset")
                    row["rule"] = "new physical trace offset: E=0, H=-1/2"
                    row["class"] = "trace-offset"
                else:
                    raise ValueError("Unexpected key addition/removal")
            elif kind == "roughness" and key in roughness_fitted_keys():
                row.update(rule=FITTED_CLASS, **{"class": FITTED_CLASS},
                           classification="Non-unique material fit; the test checks dtype/shape and fitted responses, not exact coefficients",
                           code=[TESTS["roughness"] + ":_FITTED_KEYS",
                                 TESTS["roughness"] + ":_compare_fitted_responses"])
                predicted = torch.as_tensor(old).to(torch.complex128)
                row.update(residual(old, new, predicted))
                row.pop("passes")  # Coefficient residuals are measured, with no coefficient acceptance threshold.
                if old.dtype != new.dtype:
                    raise ValueError("Fitted entry dtype changed")
            elif kind == "graph" and re.search(r"/copper/\d+/spectral_(real|imag)$", key):
                clock = clock_for(key, clocks)
                partner = key.rsplit("_", 1)[0]
                b = torch.complex(baseline[partner + "_real"], baseline[partner + "_imag"])
                c = torch.complex(current[partner + "_real"], current[partner + "_imag"])
                predicted = rotate(b, frequency_vector(kind, key, clock), clock["dt"], 1.0, 0)
                row["rule"] = "copper electric Re/Im pair: exp(+i theta); joint complex residual"
                row["class"] = "rotated"
                row["paired_keys"] = [partner + "_real", partner + "_imag"]
                row.update(dt=clock["dt"], frequencies=clock["frequencies"], exponent_steps=1.0, frequency_axis=0)
                if predictions is not None:
                    predictions[key] = predicted.real if key.endswith("_real") else predicted.imag
                row.update(residual(b, c, predicted))
                if old.dtype != new.dtype or not row.pop("passes"):
                    raise ValueError("Copper rotation residual exceeds FP32 budget")
            elif re.search(r"/(time_point|time_flux|t_e|t_h)/(?:.*/)?t$", key):
                clock = clock_for(key, clocks)
                # Build the new labels as (n+1)*dt, avoiding a different addition order.
                value = torch.as_tensor(old)
                n = torch.round(value.to(torch.float64) / clock["dt"])
                if not bitwise((n * clock["dt"]).to(value.dtype), value):
                    raise ValueError("Historical trace labels do not lie on n*dt")
                predicted = ((n + 1) * clock["dt"]).to(value.dtype)
                row["rule"] = "trace labels (n dt) -> ((n+1)dt); trace values invariant"
                row["class"] = "trace-label"
                if predictions is not None:
                    predictions[key] = predicted
                row.update(residual(old, new, predicted))
                row.pop("passes")
                if not bitwise(predicted, new):
                    raise ValueError("Trace time is not the exact documented label advance")
            else:
                matches = [prefix for prefix in clocks if not prefix or key.startswith(prefix + "/")]
                candidate_clock = clocks[max(matches, key=len)] if matches else {}
                rule = spectral_rule(kind, key, old, candidate_clock) if kind != "boundary" else None
                derived = derived_invariant_rule(kind, key)
                if derived is not None:
                    reason, code = derived
                    predicted = torch.as_tensor(old).to(torch.complex128)
                    row.update(rule="derived-invariant: identity within 64 eps32", **{"class": "derived-invariant"},
                               classification=reason, code=code)
                    if predictions is not None:
                        predictions[key] = predicted
                    row.update(residual(old, new, predicted))
                    if type(old) is not type(new) or isinstance(old, torch.Tensor) and old.dtype != new.dtype:
                        raise ValueError("Derived-invariant dtype changed")
                    if not row.pop("passes"):
                        raise ValueError("Derived-invariant residual exceeds FP32 budget")
                elif rule is not None:
                    label, exponent, axis = rule
                    clock = clock_for(key, clocks)
                    predicted = rotate(old, frequency_vector(kind, key, clock), clock["dt"], exponent, axis)
                    row.update(rule=label, exponent_steps=exponent, frequency_axis=axis,
                               dt=clock["dt"], frequencies=frequency_vector(kind, key, clock).tolist())
                    row["class"] = "rotated" if exponent else "derived-invariant"
                    if label == "independent raw wire I/q clock unchanged":
                        row["class"] = "bitwise"
                    if predictions is not None:
                        predictions[key] = predicted
                    row.update(residual(old, new, predicted))
                    if isinstance(old, torch.Tensor) and (not isinstance(new, torch.Tensor) or old.dtype != new.dtype):
                        raise ValueError("Spectral dtype changed")
                    if not row.pop("passes"):
                        raise ValueError("Rotation residual exceeds FP32 budget")
                    if row["class"] == "bitwise" and not equal:
                        raise ValueError("Predicted unchanged category is not bitwise equal")
                elif not equal:
                    if complex_value(old):
                        predicted = torch.as_tensor(old).to(torch.complex128)
                        row["rule"] = "bitwise invariant; identity prediction (no permitted phase change)"
                        row.update(residual(old, new, predicted))
                        row.pop("passes")
                        if predictions is not None:
                            predictions[key] = predicted
                    raise ValueError("Non-predicted category changed; bitwise invariant failed")
            if row["class"] == "bitwise" and kind in {"roughness", "graph"}:
                reason, code = spectrum_independent_rule(kind, key)
                row.update(classification=reason, code=code)
        except Exception as exc:
            row.update(verdict="FAIL", error=str(exc))
            failures.append(key)
        if row["rule"] != "bitwise invariant" and row["class"] != FITTED_CLASS:
            row["design"] = str(DESIGN)
            if row["rule"].startswith("derived-invariant:"):
                pass  # The spectrum consumer supplies its own code citations.
            elif "identity prediction" in row["rule"]:
                row["code"] = [TESTS.get(kind, kind)]
            elif "copper" in row["rule"]:
                row["code"] = ["witwin/maxwell/fdtd/copper_foil.py:accumulate_copper_spectra"]
            elif "trace" in row["rule"]:
                row["code"] = ["witwin/maxwell/fdtd/observers.py:get_time_observer_results"]
            elif "independent raw wire" in row["rule"]:
                row["code"] = ["witwin/maxwell/fdtd/wire.py:finalize_wire_data"]
            elif "port" in row["rule"] or "network" in row["rule"]:
                row["code"] = ["witwin/maxwell/fdtd/ports.py:prepare_port_spectral_accumulators",
                               "witwin/maxwell/fdtd/ports.py:finalize_port_data"]
            else:
                row["code"] = ["witwin/maxwell/fdtd/runtime/spectral.py:build_dft_step_tables",
                               "witwin/maxwell/fdtd/observers.py:get_observer_results"]
            if key.rsplit("/", 1)[-1] in {"amplitude_forward", "amplitude_backward"}:
                row["code"] += ["witwin/maxwell/postprocess/modal.py:_compute_mode_overlap_from_payload"]
            if "normalized" in row["rule"] or "denominator" in row["rule"]:
                row["code"] += ["witwin/maxwell/fdtd/runtime/timeplan.py:compile_fdtd_time_plan",
                                "witwin/maxwell/fdtd/runtime/spectral.py:source_normalization_spectrum",
                                "witwin/maxwell/fdtd/runtime/spectral.py:convert_source_spectrum"]
            if not row["rule"].startswith("derived-invariant:"):
                row["classification"] = row["rule"]
        rows.append(row)
    relative = [row["max_rel_residual"] for row in rows if row.get("max_rel_residual") is not None]
    return {"file": TESTS.get(kind, kind), "kind": kind, "verdict": "FAIL" if failures else "PASS",
            "max_rel_residual": max(relative) if relative else None,
            "classes": summarize_classes(rows),
            "moved_keys": [row["key"] for row in rows if row["moved"]],
            "unmoved_keys_verified_bitwise": len(unchanged), "unmoved_keys": unchanged,
            "failed_keys": failures, "keys": rows}


def summarize_classes(rows):
    summaries = {}
    for category in ("rotated", "derived-invariant", FITTED_CLASS, "bitwise", "trace-label", "trace-offset"):
        group = [row for row in rows if row["class"] == category]
        if not group:
            continue
        values = [row["max_rel_residual"] for row in group if row.get("max_rel_residual") is not None]
        summaries[category] = {"keys": len(group), "failed": sum(row["verdict"] != "PASS" for row in group),
                               "max_rel_residual": max(values) if values else None,
                               "comparison": (
                                   "diagnostic residual only; owning test checks fitted responses" if category == FITTED_CLASS else
                                   "exact bytes" if category == "bitwise" else
                                   "exact documented value" if category in {"trace-label", "trace-offset"} else "64 eps32")}
    return summaries


def module_for(kind):
    return importlib.import_module(TESTS[kind][:-3].replace("/", "."))


def archive(module, ref, directory):
    directory.mkdir(parents=True, exist_ok=True)
    return module._baseline_tree(ref, directory)


def remove_provenance(payload, tree):
    removed = {}
    for key in PROVENANCE & payload.keys():
        value = payload.pop(key)
        if isinstance(value, torch.Tensor):
            value = bytes(value.tolist()).decode()
        if not Path(value).resolve().is_relative_to(tree.resolve()):
            raise ValueError(f"Probe imported the wrong tree: {value}; expected {tree}")
        removed[key] = value
    return removed


def same_tree(left, right):
    if left.keys() != right.keys():
        raise AssertionError(f"Same-tree key difference: {sorted(left.keys() ^ right.keys())}")
    different = [key for key in left if not bitwise(left[key], right[key])]
    if different:
        raise AssertionError(f"Same-tree bitwise changes: {different}")
    return len(left)


def collect_payloads(kind, directory, hook_dir):
    """Mirror each owning test, retaining original unfiltered payload files."""
    module = module_for(kind)
    directory.mkdir(parents=True)
    notes = {"baseline_refs": [], "provenance": {}, "excluded_from_historical_comparison": []}
    current_path = directory / "current.pt"

    def baseline(ref, name="base", *extra):
        notes["baseline_refs"].append(ref)
        tree = archive(module, ref, directory / name)
        run = getattr(module, "_run", None) or module._run_probe
        data = run(tree, directory / f"{name}.pt", *extra)
        notes["provenance"][name] = remove_provenance(data, tree)
        return data

    def current(*extra, name="current"):
        path = directory / f"{name}.pt"
        run = getattr(module, "_run", None) or module._run_probe
        with metadata_environment(module, path, hook_dir):
            data = run(ROOT, path, *extra)
        notes["provenance"][name] = remove_provenance(data, ROOT)
        return data

    if kind == "roughness":
        # This environment variable is the owning test's copper_torch_kernel fixture.
        with temporary_env("WITWIN_MAXWELL_COPPER_KERNEL", "torch"):
            base = baseline(module._BASELINE_REV)
            branch = current()
            copper = baseline(module._COPPER_BASELINE_REV, "copper")
        base.update({key: value for key, value in copper.items()
                     if key.startswith(("foil.response.", "prepared.foil.", "run.foil."))})
        added = set(branch) - set(base)
        assert set(base) <= set(branch)
        assert {key.rsplit(".", 1)[1] for key in added} <= {"admittance_scale", "effective", "trace_conductance"}, added
        notes["excluded_from_historical_comparison"] = sorted(added)
        notes["fitted_response_control"] = module._compare_fitted_responses(base, branch)
        # Keep fitted entries for diagnostic residuals while respecting the
        # owning test's response-based comparison rather than exact coefficients.
        branch = {key: value for key, value in branch.items() if key not in added}
        assert len(branch) > 100 and "run.foil.dft.EX" in branch, "CUDA copper payload missing"
    elif kind == "monitors":
        base = baseline(module.BASELINE_COMMIT)
        branch = current()
        extended = current("with-new-monitors", name="extended")
        notes["same_tree_extended_keys_verified_bitwise"] = same_tree(branch, extended)
        excluded = {key for key in set(base) | set(branch)
                    if key.split("/")[1].split("::")[0] in module._CHANGED_SINCE_BASELINE}
        notes["excluded_from_historical_comparison"] = sorted(excluded)
        notes["excluded_historical_differences"] = sorted(
            key for key in excluded if key not in base or key not in branch or not bitwise(base[key], branch[key]))
        base = {key: value for key, value in base.items() if key not in excluded}
        branch = {key: value for key, value in branch.items() if key not in excluded}
    elif kind == "graph":
        with temporary_env("WITWIN_MAXWELL_COPPER_KERNEL", "torch"):
            base = baseline(module.BASE_COMMIT)
            branch = current()
            copper = baseline(module.COPPER_BASE_COMMIT, "copper", module.COPPER_FIXTURES)
        base.update(copper)
        for payload in (base, branch):
            for fixture in ("circuit", "circuit_ongoing", "circuit_transient"):
                assert payload[f"{fixture}/__field_graph"] and payload[f"{fixture}/__circuit_graph"]
            for key in module.CIRCUIT_ABSOLUTE_KEYS:
                assert f"circuit/{key}" not in payload
                for fixture in ("circuit_ongoing", "circuit_transient"):
                    assert f"{fixture}/{key}" in payload
        excluded = {key for key in base.keys() | branch.keys() if key.rsplit("/", 1)[-1].startswith("__")}
        notes["excluded_from_historical_comparison"] = sorted(excluded)
        notes["excluded_historical_differences"] = sorted(
            key for key in excluded if key not in base or key not in branch or not bitwise(base[key], branch[key]))
        module._split_flags(base)
        graph = module._split_flags(branch)
        for fixture in ("lumped1", "lumped2_one_driven", "terminal_direct", "coax", "wire_gap", "copper_terminal",
                        "resume_copper_terminal_150", "resume_lumped_100"):
            assert any(value for key, value in graph.items() if key.startswith(fixture + "/")), fixture
        for fixture in ("lumped1_nograph", "circuit", "circuit_ongoing", "circuit_transient", "network_embedded", "dipole"):
            assert not any(value for key, value in graph.items() if key.startswith(fixture + "/")), fixture
        notes["current_graph_flags"] = graph
        assert any("/history/" in key for key in base)
    else:
        base = baseline(module.BASE_COMMIT)
        branch = current()

    metadata = json.loads(current_path.with_suffix(".metadata.json").read_text())
    if metadata["errors"]:
        raise RuntimeError("Metadata capture failed: " + "\n".join(metadata["errors"]))
    if not metadata["records"]:
        raise RuntimeError("No executed clocks were captured")
    for prefix in metadata["records"]:
        clock_for(prefix + "/", metadata["records"])
    torch.save(base, directory / "comparison_baseline.pt")
    torch.save(branch, directory / "comparison_current.pt")
    spec = {"kind": kind, "notes": notes, "clocks": metadata["records"]}
    (directory / "comparison.json").write_text(json.dumps(spec, indent=2) + "\n")
    predictions = {}
    result = analyze(kind, base, branch, spec["clocks"], predictions)
    torch.save(predictions, directory / "predicted.pt")
    return result | {"collection": notes, "predicted_payload": str(directory / "predicted.pt")}


@contextmanager
def temporary_env(key, value):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def collect_boundary(directory):
    module = module_for("boundary")
    directory.mkdir(parents=True)
    tree = archive(module, module._BASELINE_REF, directory / "source")
    base, branch = {}, {}
    for config in module._CONFIGS:
        print(f"boundary: {config}", flush=True)
        left = module._run_probe(tree, config, directory / f"{config}.baseline.pt")
        right = module._run_probe(ROOT, config, directory / f"{config}.current.pt")
        remove_provenance(left, tree)
        remove_provenance(right, ROOT)
        assert left["prepared_state"] and left["fields"]["Ez"].abs().max() > 0
        base.update(flatten(left, config))
        branch.update(flatten(right, config))

    config = "mur_cpml_mixed"
    checkpoint = directory / "baseline_checkpoint.pt"
    module._run_probe(tree, config, directory / "saved.pt", "save", str(checkpoint))
    schema = load(checkpoint)["schema_version"]
    expected_error = "Unsupported FDTD resume checkpoint schema_version 5"
    if schema != 5:
        raise AssertionError(f"Expected historical format 5, found {schema!r}")
    resumed_path = directory / "refused_resume.pt"
    command = [sys.executable, str(module._PROBE), config, str(resumed_path), "resume", str(checkpoint)]
    # Identical to _run_probe's child invocation, with captured stderr so the
    # proof can distinguish the intended load guard from other child failures.
    completed = subprocess.run(command, cwd=str(directory), env=module.probe_environment(ROOT, baseline=False),
                               capture_output=True, text=True, check=False)
    (directory / "schema_refusal.stdout.log").write_text(completed.stdout)
    (directory / "schema_refusal.stderr.log").write_text(completed.stderr)
    assert completed.returncode != 0 and not resumed_path.exists(), "Historical resume was not refused"
    assert (expected_error in completed.stderr and "resume.py" in completed.stderr
            and "in load" in completed.stderr
            and completed.stderr.rstrip().splitlines()[-1] == f"ValueError: {expected_error}."), completed.stderr

    from witwin.maxwell.fdtd.resume import _RESUME_SCHEMA_VERSION
    assert _RESUME_SCHEMA_VERSION == 6
    fresh = directory / "fresh_checkpoint.pt"
    module._run_probe(ROOT, config, directory / "fresh_saved.pt", "save", str(fresh))
    fresh_payload = load(fresh)
    assert fresh_payload["schema_version"] == 6
    assert fresh_payload["signature"]["observer_spectral_origin"] == "physical_yee_v1"
    assert fresh_payload["signature"]["source_reference_label"] == "provider_reference_v1"
    assert fresh_payload["signature"]["normalization_label_offset"] == 0.5
    resumed = module._run_probe(ROOT, config, directory / "fresh_resumed.pt", "resume", str(fresh))
    current_fields = {key: value for key, value in branch.items() if key.startswith(config + "/fields/")}
    count = same_tree(current_fields, flatten(resumed["fields"], config + "/fields"))
    notes = {
        "baseline_refs": [module._BASELINE_REF], "historical_schema": schema, "current_schema": 6,
        "refusal_command": command, "refusal_returncode": completed.returncode,
        "refusal_error": completed.stderr.rstrip().splitlines()[-1],
        "checkpoint_verdict": "EXPECTED_SCHEMA_REFUSAL",
        "guard": "FDTDResumeCheckpoint.load schema-version guard, before solver preparation or state mutation",
        "required_fresh_baseline": "Re-run this unchanged probe on a tree with physical_yee_v1/provider_reference_v1, "
                                   "format 6 and the selected provider/beta metadata. Generate a NEW step-37 checkpoint "
                                   "and its 80-step field reference; do not relabel or convert format-5 retained sums.",
        "fresh_resume_fields_verified_bitwise": count,
        "fresh_signature_origin": {key: fresh_payload["signature"][key] for key in
                                   ("observer_spectral_origin", "source_reference_label", "normalization_label_offset")},
    }
    torch.save(base, directory / "comparison_baseline.pt")
    torch.save(branch, directory / "comparison_current.pt")
    (directory / "comparison.json").write_text(json.dumps({"kind": "boundary", "notes": notes, "clocks": {}}, indent=2) + "\n")
    return analyze("boundary", base, branch, {}) | {"collection": notes}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(output, results, provenance, mode):
    results = sorted(results, key=lambda item: list(TESTS).index(item["kind"]))
    failures = [result for result in results if result["verdict"] != "PASS"]
    report = {"mode": mode, "verdict": "FAIL" if failures else "PASS", "fp32_relative_limit": REL_LIMIT,
              "residual_definition": "max(abs(current-predicted)) / max(abs(baseline)); zero baseline requires exact zero",
              "provenance": provenance, "results": results,
              "qualification": "Analytical re-pin evidence only. No pins or expectations were modified. "
                               "Spectrum-derived invariants use 64 eps32; trajectory/state invariants compare bytes. "
                               "Trace labels follow the exact documented advance. Roughness fitted entries are excluded "
                               "from exact equality by their owning test; coefficient residuals are diagnostic only, "
                               "and fitted responses use the owning test's 1e-9 limit, equal orders and passivity flags."}
    (output / "proof.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [f"Observer-origin BITID re-pin proof: {report['verdict']}", "",
             f"Mode: `{mode}`. Fixed spectral and derived-invariant residual limit: `{REL_LIMIT:.9g}` (64 FP32 epsilons).",
             "Residual = max |current - predicted| / max |baseline| per key. Derived invariants predict baseline itself; "
             "a zero baseline requires exact zero. No tolerance is applied to trajectory/state bytes or trace-label predictions.", "",
             "The common E/H phase cancels algebraically in conjugate products, magnitudes and ratios. "
             "These operations consume rounded FP32 spectra, so their outputs are only invariant within the fixed rounding budget. "
             "Modal amplitudes themselves are linear overlaps and rotate; their powers and phase-cancelling ratios are derived invariants. "
             "Port excitation thresholds and embedded-network port/absorbed/generated powers also consume relabelled spectra. "
             "Reference-mode powers/impedances, geometry, coordinates, configured normalization policy and prepared tensors do not consume "
             "the relabelled spectra and remain bitwise. Raw field/time-flux trace values and native state also remain bitwise. "
             "Time labels must equal ((n+1)*dt) exactly, with E=0/H=-1/2 trace offsets.", "",
             "The roughness test's eight `_FITTED_KEYS` are reported as **fitted (excluded by the test)**. "
             "Their coefficient residuals are diagnostic, with no coefficient tolerance or bitwise requirement. "
             "The owning test's fitted-response comparison is rerun on CPU from the retained payloads: equal order/state count, "
             "equal passivity flags and relative response differences <= 1e-9.", "",
             "Executed clocks are read from `Result.solver_stats['dt']` "
             "(`witwin/maxwell/simulation.py:Simulation._build_fdtd_solver_stats`) and the live solver when retained. "
             "Sweep columns, including grouped frequency columns, must agree exactly. Resumed runs use the executed solver clock. "
             "Solver-free batch lanes retain their own stats; a profile of "
             "`witwin/maxwell/execution/batch_runtime.py:PreparedBatch._export` also captures the prepared solver's clock/provider "
             "before export and checks it against each lane's result. Each new sidecar records `dt_sources`; no clock is inferred "
             "from a requested CFL or borrowed from another fixture.", "",
             "| File | Class | Keys | Max relative residual | Failed | Verdict |",
             "|---|---|---:|---:|---:|---|"]
    for result in results:
        if not result.get("classes"):
            lines.append(f"| {result['kind']} | collection incomplete; CUDA rerun required | — | — | — | {result['verdict']} |")
        for category, summary in result.get("classes", {}).items():
            value = summary["max_rel_residual"]
            value = f"{value:.9g}" if value is not None else "— (exact comparison)"
            lines.append(f"| {result['kind']} | {category} | {summary['keys']} | {value} | {summary['failed']} | "
                         f"{'FAIL' if summary['failed'] else 'PASS'} |")
    lines += ["", "Retained CPU replay inputs (all collection artifact hashes verified before analysis):", "",
              "| File | Collection | Fitted-response residuals (limit 1e-9) |",
              "|---|---|---|"]
    for result in results:
        controls = result.get("collection", {}).get("fitted_response_control", {})
        values = ", ".join(f"{name}: {value:.9g}" for name, value in controls.items()) or "—"
        lines.append(f"| {result['kind']} | `{result.get('retained_collection', 'collected directly')}` | {values} |")
    if provenance.get("replay_recipe"):
        lines += ["", "Reproduce with the maxwell environment interpreter from the repository root, with CUDA hidden:",
                  "", "```python", provenance["replay_recipe"], "```"]
    lines += ["", "Classification justifications and computing code paths (each key's category and citations are also in JSON):", "",
              "| Class | Category / reason | Code path |", "|---|---|---|"]
    categories = sorted({(row["class"], row["classification"], tuple(row["code"]))
                         for result in results for row in result.get("keys", []) if "class" in row})
    for category, reason, code in categories:
        lines.append(f"| {category} | {reason.replace('|', '&#124;')} | " + "<br>".join(f"`{name}`" for name in code) + " |")
    lines += ["",
             "| File | Moved keys | Rule applied | Max rel residual | Unmoved keys verified bitwise | Verdict |",
             "|---|---|---|---:|---:|---|"]
    for result in results:
        rows = result.get("keys", [])
        moved = [row for row in rows if row["moved"]]
        names = "<br>".join(f"`{row['key']}`" for row in moved) or "none"
        rules = "<br>".join(sorted({row["rule"] for row in rows if row["moved"] or "exponent_steps" in row})) or "bitwise invariants"
        values = [row["max_rel_residual"] for row in rows if row.get("max_rel_residual") is not None]
        maximum = f"{max(values):.9g}" if values else "—"
        if result["kind"] == "boundary" and result.get("collection"):
            names += "<br>format-5 disk checkpoint refused"
            rules += "<br>schema 5 -> refusal; fresh format-6 resume checked"
        if result.get("error"):
            rules = result["error"].replace("|", "\\|").replace("\n", "<br>")
            if result.get("analysis_status"):
                rules += "<br>" + result["analysis_status"]
        lines.append(f"| `{result['file']}` | {names} | {rules} | {maximum} | "
                     f"{result.get('unmoved_keys_verified_bitwise', 0)} | {result['verdict']} |")
    lines += ["", "Per-key evidence (including every unchanged key) and collection scopes are in `proof.json`.",
              "Original `.pt` payloads and prepared-clock sidecars are retained by file. Historical monitor box keys,",
              "graph admission flags and roughness additions retain their owning tests' comparison scopes.",
              "The current monitor run with extra monitors must match the ordinary current run bitwise, including box keys.", ""]
    lines += ["| File | Moved or predicted key | Rule | Rel residual | Verdict |",
              "|---|---|---|---:|---|"]
    for result in results:
        for row in result.get("keys", []):
            if row["moved"] or "exponent_steps" in row or "paired_keys" in row or row.get("class") in {"derived-invariant", FITTED_CLASS}:
                value = row.get("max_rel_residual")
                value = "—" if value is None else f"{value:.9g}"
                lines.append(f"| {result['kind']} | `{row['key']}` | {row['rule']} | {value} | {row['verdict']} |")
    lines.append("")
    for result in results:
        if result.get("collection", {}).get("required_fresh_baseline"):
            lines += [result["collection"]["required_fresh_baseline"], ""]
        if result.get("failed_keys"):
            lines += [f"Failed invariant/prediction keys for `{result['kind']}`: " + ", ".join(f"`{key}`" for key in result["failed_keys"]), ""]
    (output / "proof.md").write_text("\n".join(lines))
    print(f"{report['verdict']}: {output / 'proof.json'} and {output / 'proof.md'}", flush=True)
    return 1 if failures else 0


def make_hook_directory(output):
    directory = output / "metadata_hook"
    directory.mkdir(exist_ok=True)
    # runpy leaves the baseline tree first on sys.path. Importing this file for
    # its hook does not import the current Maxwell package into a historical run.
    (directory / "sitecustomize.py").write_text(
        "import runpy\n" + f"runpy.run_path({str(Path(__file__).resolve())!r}, "
        "run_name='observer_repin_hook')['install_metadata_hook']()\n")
    return directory


def self_test(output, hook_dir):
    """Independent finite-record examples and negative controls, all on CPU."""
    dt, frequencies = 0.037, [0.4, 0.9]
    clock = {"dt": dt, "frequencies": frequencies, "normalize_source": False,
             "provider": "additive_J", "waveform": "gaussian_pulse", "beta": 0.5,
             "full_field_dft": True}
    angles = 2 * math.pi * torch.tensor(frequencies, dtype=torch.float64) * dt
    wave = torch.tensor([1 + 2j, -0.4 + 0.7j], dtype=torch.complex64)

    def independent_rotation(value, factor=1.0, axis=0, selected=None):
        theta = angles if selected is None else angles[selected:selected + 1]
        multiplier = torch.cos(factor * theta) + 1j * torch.sin(factor * theta)
        if len(multiplier) > 1:
            shape = [1] * value.ndim
            shape[axis] = len(multiplier)
            multiplier = multiplier.reshape(shape)
        else:
            multiplier = multiplier[0]
        return (value.to(torch.complex128) * multiplier).to(value.dtype)

    results, controls = [], []
    from types import SimpleNamespace

    # Singleton slicing is considered contiguous despite its non-unit stride.
    strided = torch.arange(14, dtype=torch.float64).to(torch.complex128)[::7][1:]
    assert strided.stride() == (7,) and bitwise(strided, strided.clone())
    matrix = torch.tensor([[1 + 2j, -0j], [3j, 4j]], dtype=torch.complex128).T
    assert bitwise(matrix, matrix.clone()) and bitwise(matrix.conj(), matrix.conj().resolve_conj())
    assert not bitwise(torch.tensor([0.]), torch.tensor([-0.]))
    controls.append("noncontiguous, singleton-strided and conjugate byte comparisons; signed zeros stay distinct")
    for stats in ({"dt": dt}, {"columns": ({"dt": dt}, {"dt": dt})},
                  {"columns": (({"dt": dt},), ({"dt": dt},))}):
        item = SimpleNamespace(solver=None, solver_stats=stats, frequencies=frequencies)
        assert result_clock(item)["dt"] == dt
    for stats in ({"columns": ({"dt": dt}, {"dt": 2 * dt})}, {}):
        try:
            result_clock(SimpleNamespace(solver=None, solver_stats=stats, frequencies=frequencies))
        except ValueError:
            pass
        else:
            raise AssertionError("Missing or inconsistent executed clocks accepted")
    controls.append("solver-free batch, sweep and nested-column clocks; missing and mixed clocks rejected")

    def positive(kind, base, current, clocks):
        result = analyze(kind, base, current, clocks)
        assert result["verdict"] == "PASS", result
        results.append(result)
        controls.append(f"{kind}: positive payload, predictions and invariants")
        return result

    def negative(name, kind, base, current, clocks, key):
        result = analyze(kind, base, current, clocks)
        assert result["verdict"] == "FAIL" and key in result["failed_keys"], (name, result)
        controls.append(name)

    baseline = {f"run.foil.dft.{name}": wave.clone() for name in ("EX", "EY", "EZ")}
    baseline.update({f"run.foil.port.{name}": wave.clone() for name in ("voltage", "current")})
    baseline["run.foil.Ex"] = torch.tensor([0.0, 1.0], dtype=torch.float32)
    current = {key: independent_rotation(value) if value.is_complex() else value.clone() for key, value in baseline.items()}
    positive("roughness", baseline, current, {"": clock})
    fitted = {key: torch.ones(2, dtype=torch.float64) for key in roughness_fitted_keys()}
    # Coefficients have no acceptance tolerance; only the owning test's responses do.
    fit_result = positive("roughness", fitted, {key: value * 1.01 for key, value in fitted.items()}, {})
    assert fit_result["classes"][FITTED_CLASS]["keys"] == 8
    assert all(row["class"] == FITTED_CLASS and row["max_rel_residual"] > REL_LIMIT
               for row in fit_result["keys"])
    fit_key = "rational.model.direct"
    negative("fitted dtype mutation rejected", "roughness", fitted,
             fitted | {fit_key: fitted[fit_key].float()}, {}, fit_key)
    negative("fitted shape mutation rejected", "roughness", fitted,
             fitted | {fit_key: fitted[fit_key][:1]}, {}, fit_key)
    negative("non-excluded model poles remain bitwise", "roughness",
             {"rational.model.poles": wave}, {"rational.model.poles": wave * 1.0000001}, {}, "rational.model.poles")
    key = "run.foil.dft.EX"
    negative("wrong rotation sign rejected", "roughness", baseline,
             current | {key: independent_rotation(wave, -1)}, {"": clock}, key)
    negative("half-step substituted for raw full step rejected", "roughness", baseline,
             current | {key: independent_rotation(wave, 0.5)}, {"": clock}, key)
    negative("unchanged nonzero spectrum rejected", "roughness", baseline,
             current | {key: baseline[key]}, {"": clock}, key)
    negative("field trajectory drift rejected", "roughness", baseline,
             current | {"run.foil.Ex": torch.tensor([0.0, 1.0000001])}, {"": clock}, "run.foil.Ex")
    negative("signed-zero field mutation rejected", "roughness", baseline,
             current | {"run.foil.Ex": torch.tensor([-0.0, 1.0])}, {"": clock}, "run.foil.Ex")
    negative("spectral dtype mutation rejected", "roughness", baseline,
             current | {key: current[key].to(torch.complex128)}, {"": clock}, key)
    negative("spectral shape mutation rejected", "roughness", baseline,
             current | {key: current[key][:1]}, {"": clock}, key)
    negative("NaN spectrum rejected", "roughness", baseline,
             current | {key: torch.full_like(wave, complex(float('nan'), 0))}, {"": clock}, key)
    negative("missing key rejected", "roughness", baseline,
             {k: v for k, v in current.items() if k != key}, {"": clock}, key)
    negative("unknown complex mover rejected", "roughness", baseline | {"unknown": wave},
             current | {"unknown": independent_rotation(wave)}, {"": clock}, "unknown")
    negative("unexpected new key rejected", "roughness", baseline,
             current | {"new": wave}, {"": clock}, "new")
    zero = torch.zeros_like(wave)
    positive("roughness", {key: zero}, {key: zero.clone()}, {"": clock})
    negative("nonzero result from zero baseline rejected", "roughness", {key: zero},
             {key: torch.full_like(wave, 1e-30)}, {"": clock}, key)

    time = torch.arange(0, 12, 3, dtype=torch.float64) * dt
    h = wave * torch.exp(-0.5j * angles).to(wave.dtype)
    baseline = {"raw/point/Hy": h, "monitor/point/1/Ez": wave[1],
                "raw/time_point/t": time, "raw/time_point/data": torch.arange(4.),
                "raw/time_flux/t": time, "raw/time_flux/flux": torch.arange(4.),
                "monitor/mode_out/0/power_forward": torch.tensor(3.0),
                "raw/flux/flux": torch.tensor([0.3, 0.7])}
    current = baseline | {"raw/point/Hy": independent_rotation(h),
                          "monitor/point/1/Ez": independent_rotation(wave[1], selected=1),
                          "raw/time_point/t": (torch.arange(0, 12, 3, dtype=torch.float64) + 1) * dt,
                          "raw/time_flux/t": (torch.arange(0, 12, 3, dtype=torch.float64) + 1) * dt,
                          "raw/time_point/field_time_offsets_steps/E": 0.0,
                          "raw/time_point/field_time_offsets_steps/H": -0.5}
    positive("monitors", baseline, current, {"": clock})
    negative("double H stagger correction rejected", "monitors", baseline,
             current | {"raw/point/Hy": independent_rotation(h, 1.5)}, {"": clock}, "raw/point/Hy")
    negative("unchanged trace label rejected", "monitors", baseline,
             current | {"raw/time_point/t": time}, {"": clock}, "raw/time_point/t")
    positive("monitors", baseline,
             current | {"raw/flux/flux": torch.tensor([0.3000001, 0.7])}, {"": clock})
    negative("flux drift beyond fixed budget rejected", "monitors", baseline,
             current | {"raw/flux/flux": torch.tensor([0.301, 0.7])}, {"": clock}, "raw/flux/flux")
    negative("derived-invariant dtype mutation rejected", "monitors", baseline,
             current | {"raw/flux/flux": baseline["raw/flux/flux"].double()}, {"": clock}, "raw/flux/flux")
    positive("monitors", {"raw/flux/flux": torch.zeros(2)}, {"raw/flux/flux": torch.zeros(2)}, {"": clock})
    negative("nonzero derived invariant from zero baseline rejected", "monitors",
             {"raw/flux/flux": torch.zeros(2)}, {"raw/flux/flux": torch.full((2,), 1e-30)}, {"": clock}, "raw/flux/flux")
    negative("NaN derived invariant rejected", "monitors", baseline,
             current | {"raw/flux/flux": torch.full((2,), float("nan"))}, {"": clock}, "raw/flux/flux")
    negative("raw time-flux drift remains bitwise rejected", "monitors", baseline,
             current | {"raw/time_flux/flux": baseline["raw/time_flux/flux"] + 1e-7},
             {"": clock}, "raw/time_flux/flux")

    # PortData has frequency LAST, including mode and sweep excitation axes.
    port = torch.stack([wave, 2 * wave, 3 * wave])
    for kind in ("rf", "terminal"):
        baseline = {"pair/left/voltage": port, "pair/right/current": port,
                    "pair/field/Ez": torch.ones(3), "pair/network/s": torch.eye(2, dtype=torch.complex64),
                    "pair/left/characteristic_impedance": wave, "pair/left/available_power": torch.ones(2)}
        current = baseline | {"pair/left/voltage": independent_rotation(port, axis=-1),
                              "pair/right/current": independent_rotation(port, axis=-1)}
        positive(kind, baseline, current, {"pair": clock})
        positive(kind, baseline, current | {"pair/network/s": 1.0000001 * baseline["pair/network/s"]}, {"pair": clock})
        negative(f"{kind}: S drift beyond fixed budget rejected", kind, baseline,
                 current | {"pair/network/s": 1.001 * baseline["pair/network/s"]}, {"pair": clock}, "pair/network/s")

    baseline = {"lumped/port_runtime/0/accumulator/_voltage_sum": wave,
                "lumped/port_runtime/0/drive_accumulator/_current_sum": wave,
                "lumped/ports/p1/voltage": port,
                "lumped/fields/EX": torch.stack([wave, wave, wave], dim=1),
                "lumped/copper/0/spectral_real": wave.real.to(torch.float64),
                "lumped/copper/0/spectral_imag": wave.imag.to(torch.float64),
                "lumped/monitors/t_h/t": time,
                "lumped/history/run0/port0/last_branch_current": torch.ones(4),
                "lumped/solver/Hx": torch.ones(4)}
    cu = independent_rotation(wave.to(torch.complex128))
    current = baseline | {
        "lumped/port_runtime/0/accumulator/_voltage_sum": independent_rotation(wave),
        "lumped/port_runtime/0/drive_accumulator/_current_sum": independent_rotation(wave),
        "lumped/ports/p1/voltage": independent_rotation(port, axis=-1),
        "lumped/fields/EX": independent_rotation(baseline["lumped/fields/EX"]),
        "lumped/copper/0/spectral_real": cu.real, "lumped/copper/0/spectral_imag": cu.imag,
        "lumped/monitors/t_h/t": (torch.arange(0, 12, 3, dtype=torch.float64) + 1) * dt}
    positive("graph", baseline, current, {"lumped": clock})
    final_fields = {"run/fields/EX": torch.ones(3)}
    positive("graph", final_fields, final_fields, {"run": clock | {"full_field_dft": False}})
    negative("exported time-domain field drift rejected", "graph", final_fields,
             {"run/fields/EX": torch.ones(3) * 1.0000001}, {"run": clock | {"full_field_dft": False}}, "run/fields/EX")
    consumers = {
        "lumped/ports/p1/metadata/excitation_threshold": torch.tensor(1e-9, dtype=torch.float64),
        "load/embedded_networks/block/port_power": torch.ones(2, dtype=torch.float64),
        "load/embedded_networks/block/absorbed_power": torch.ones(2, dtype=torch.float64),
        "load/embedded_networks/block/generated_power": torch.zeros(2, dtype=torch.float64),
        "sheet/ports/p1/metadata/sheet_gap/variance_power": torch.ones(2, dtype=torch.float64),
        "sheet/ports/p1/metadata/sheet_gap/port_accepted_power": torch.ones(2, dtype=torch.float64),
        "sheet/ports/p1/metadata/sheet_gap/field_accepted_power": torch.ones(2, dtype=torch.float64),
        "sheet/ports/p1/metadata/sheet_gap/power_correction": -torch.ones(2, dtype=torch.float64),
    }
    consumer_result = positive("graph", consumers, {key: value * (1 + REL_LIMIT / 2) for key, value in consumers.items()}, {})
    assert consumer_result["classes"]["derived-invariant"]["keys"] == len(consumers)
    for key, value in consumers.items():
        negative(f"spectrum consumer budget enforced: {key}", "graph", consumers,
                 consumers | {key: value * (1 + 2 * REL_LIMIT) if bool(value.abs().max()) else value + 1e-30}, {}, key)
    independent = {
        "load/embedded_networks/block/metadata/absorbed_energy": torch.ones(1),
        "circuit/circuits/driven/diagnostics/port_powers": torch.ones(1),
        "lumped/ports/p1/metadata/dft_samples": torch.ones(1),
    }
    for key, value in independent.items():
        negative(f"time-step/metadata drift remains bitwise: {key}", "graph", independent,
                 independent | {key: value * 1.0000001}, {}, key)
    negative("native state drift rejected", "graph", baseline,
             current | {"lumped/solver/Hx": 1.0000001 * baseline["lumped/solver/Hx"]},
             {"lumped": clock}, "lumped/solver/Hx")
    negative("only one copper quadrature moved rejected", "graph", baseline,
             current | {"lumped/copper/0/spectral_imag": baseline["lumped/copper/0/spectral_imag"]},
             {"lumped": clock}, "lumped/copper/0/spectral_imag")

    for provider, waveform, beta in (("additive_J", "cw", 0.5), ("terminal_drive", "gaussian_pulse", 0.5),
                                     ("mode_surface", "cw", 0.0), ("magnetic_M", "cw", 0.0),
                                     ("plane_wave_aux", "gaussian_pulse", 1.0),
                                     ("analytic_profile", "cw", 1.0), ("analytic_profile", "gaussian_pulse", 0.0),
                                     ("plane_wave_grating_slab_cw", "gaussian_pulse", 0.0)):
        normalized_clock = clock | {"normalize_source": True, "provider": provider, "waveform": waveform, "beta": beta}
        baseline = {"run/monitors/probe/Ez": wave, "run/monitors/wire1/current": wave,
                    "run/monitors/probe/source_normalization_spectra/Ez": wave}
        current = {"run/monitors/probe/Ez": independent_rotation(wave, 1 - beta),
                   "run/monitors/wire1/current": independent_rotation(wave, -beta),
                   "run/monitors/probe/source_normalization_spectra/Ez": independent_rotation(wave, beta)}
        positive("graph", baseline, current, {"run": normalized_clock})
    negative("provider beta disagreement rejected", "graph", baseline, current,
             {"run": normalized_clock | {"beta": 0.5}}, "run/monitors/probe/Ez")
    positive("boundary", {"fingerprint": "a" * 64, "state": torch.ones(3)},
             {"fingerprint": "a" * 64, "state": torch.ones(3)}, {})

    # Exercise the actual child sitecustomize/profile transport without CUDA or
    # the Maxwell runtime. The compiled frame uses an original probe filename.
    metadata_path = output / "synthetic.metadata.json"
    env = dict(os.environ, OBSERVER_REPIN_METADATA=str(metadata_path), PYTHONPATH=str(hook_dir))
    export_source = (
        "def _export(result, solver):\n"
        "    return result\n")
    source = ("from types import SimpleNamespace\n"
              "class R:\n"
              "    solver_stats = {'dt': 0.037}\n"
              "    frequencies = (0.4, 0.9)\n"
              "    solver = SimpleNamespace(dt=0.037, _normalize_source=False)\n"
              "def _record(prefix, result, payload):\n    pass\n"
              "_record('fixture', R(), {})\n"
              "batch = SimpleNamespace(solver=None, solver_stats={'dt': 0.037}, frequencies=(0.4, 0.9))\n"
              "prepared = SimpleNamespace(dt=0.037, _normalize_source=False, "
              "_normalization_reference_provider='terminal_drive', _normalization_reference_waveform_kind='cw', "
              "_normalization_label_offset=0.5, _normalization_time_offset=0.5)\n"
              f"exec(compile({export_source!r}, {str(ROOT / 'witwin/maxwell/execution/batch_runtime.py')!r}, 'exec'))\n"
              "_export(batch, prepared)\n"
              "_record('batch/lane0', batch, {})\n")
    code = f"exec(compile({source!r}, {str(ROOT / PROBES[3])!r}, 'exec'))"
    subprocess.run([sys.executable, "-c", code], env=env, cwd=str(output), check=True)
    metadata = json.loads(metadata_path.read_text())
    assert not metadata["errors"] and metadata["records"]["fixture"]["dt"] == dt, metadata
    batch_clock = metadata["records"]["batch/lane0"]
    assert batch_clock["dt"] == dt and batch_clock["provider"] == "terminal_drive", metadata
    assert batch_clock["dt_sources"] == {"solver_stats/dt": dt, "prepared_export.dt": dt}, metadata
    controls.append("original-probe profile frame / child sidecar transport")
    controls.append("solver-free batch prepared-export clock/provider transport per lane")
    replay = output / "synthetic_replay"
    replay.mkdir(exist_ok=True)
    torch.save({"state": torch.ones(3)}, replay / "comparison_baseline.pt")
    torch.save({"state": torch.ones(3)}, replay / "comparison_current.pt")
    (replay / "comparison.json").write_text(json.dumps({"kind": "boundary", "clocks": {}, "notes": {}}))
    seal_collection(replay)
    assert replay_collection(replay)["verdict"] == "PASS"
    torch.save({"state": torch.zeros(3)}, replay / "comparison_current.pt")
    try:
        replay_collection(replay)
    except ValueError as exc:
        assert "artifact changed" in str(exc)
    else:
        raise AssertionError("Artifact corruption went undetected")
    controls.append("hash-verified CPU artifact replay and corruption rejection")
    provenance = {"synthetic_controls": controls, "control_count": len(controls),
                  "cuda_probes_executed": False, "script_sha256": sha256(Path(__file__))}
    return write_report(output, results, provenance, "synthetic CPU self-test; not GPU re-pin evidence")


def seal_collection(directory):
    files = [path for path in directory.iterdir() if path.is_file() and
             (path.suffix in {".pt", ".log"} or path.name.endswith(".metadata.json") or path.name == "comparison.json")]
    (directory / "collection_hashes.json").write_text(json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n")


def replay_collection(directory, prediction_path=None):
    hashes = json.loads((directory / "collection_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Collected artifact changed: {directory / name}")
    spec = json.loads((directory / "comparison.json").read_text())
    predictions = {} if prediction_path is not None else None
    baseline = load(directory / "comparison_baseline.pt")
    current = load(directory / "comparison_current.pt")
    notes = dict(spec["notes"])
    if spec["kind"] == "roughness":
        notes["fitted_response_control"] = module_for("roughness")._compare_fitted_responses(baseline, current)
        notes["fitted_response_control_recomputed_on_cpu"] = True
    result = analyze(spec["kind"], baseline, current, spec["clocks"], predictions)
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(predictions, prediction_path)
        result["predicted_payload"] = str(prediction_path)
    return result | {"collection": notes, "retained_collection": str(directory),
                     "verified_collection_hashes": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-dir", type=Path, help="Retained collection root for CPU analysis; defaults to output-dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true", help="CPU synthetic positive and negative controls; no CUDA probes")
    mode.add_argument("--analyze-only", action="store_true", help="CPU replay of hash-verified payloads in input-dir (default: output-dir)")
    parser.add_argument("--files", "--only", dest="files", nargs="+", choices=TESTS, default=list(TESTS),
                        help="Select files for collection/replay; default is all six. CUDA collection requires a fresh output-dir.")
    args = parser.parse_args()
    if args.input_dir is not None and not args.analyze_only:
        parser.error("--input-dir requires --analyze-only")
    output = args.output_dir.expanduser().resolve()
    source = args.input_dir.expanduser().resolve() if args.input_dir is not None else output
    output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        return self_test(output, make_hook_directory(output))
    if not args.analyze_only and not torch.cuda.is_available():
        parser.error("CUDA is required for the original probes. Use --self-test for CPU validation.")
    # Script execution from scripts/dev must resolve repository-local test helpers.
    sys.path.insert(0, str(ROOT))
    provenance_path = source / "collection_provenance.json" if args.analyze_only else output / "collection_provenance.json"
    if args.analyze_only:
        provenance = json.loads(provenance_path.read_text())
        provenance["analysis_script_sha256"] = sha256(Path(__file__))
        provenance["analysis_selected_files"] = args.files
        provenance["retained_input_dir"] = str(source)
    else:
        if provenance_path.exists() or any((output / kind).exists() for kind in args.files):
            parser.error("Output already contains a collection. Use --analyze-only or a new --output-dir.")
        def git(*argv):
            return subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True, text=True, check=True).stdout.strip()
        sources = [*TESTS.values(), *PROBES, "tests/support/baseline_tree_env.py", "tests/support/pinned_baselines.py",
                   "tests/conftest.py", "docs/assessments/observer-origin-2026-11-02.md",
                   "witwin/maxwell/fdtd/runtime/spectral.py", "witwin/maxwell/fdtd/runtime/timeplan.py",
                   "witwin/maxwell/fdtd/observers.py", "witwin/maxwell/fdtd/ports.py",
                   "witwin/maxwell/fdtd/copper_foil.py", "witwin/maxwell/fdtd/resume.py",
                   "witwin/maxwell/fdtd/wire.py", "witwin/maxwell/postprocess/modal.py",
                   "witwin/maxwell/simulation.py", "witwin/maxwell/waveport_sweep.py",
                   "witwin/maxwell/fdtd/quadrature.py", "witwin/maxwell/network.py",
                   "witwin/maxwell/network_sweep.py", "witwin/maxwell/execution/batch_runtime.py"]
        provenance = {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                      "worktree_status": git("status", "--short"), "script_sha256": sha256(Path(__file__)),
                      "source_hashes": {name: sha256(ROOT / name) for name in sources},
                      "design_path": str(DESIGN), "design_sha256": sha256(DESIGN) if DESIGN.exists() else None,
                      "python": sys.executable, "torch": torch.__version__, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                      "selected_files": args.files,
                      "baseline_environment": {key: value for key, value in os.environ.items()
                                               if key.startswith("WITWIN_") and ("BASELINE" in key or "COPPER_KERNEL" in key)}}
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    hook_dir = None if args.analyze_only else make_hook_directory(output)
    results = []
    if args.analyze_only and (source / "proof.json").exists():
        previous = json.loads((source / "proof.json").read_text())
        results = [item | {"analysis_status": "retained previous result; not reanalyzed; CUDA rerun required",
                           "error_log": str(source / item["error_log"]) if item.get("error_log") else None}
                   for item in previous["results"] if item["kind"] not in args.files]
    with temporary_env("WITWIN_REQUIRE_BASELINES", "1"):
        for kind in args.files:
            print(f"{'Analyze' if args.analyze_only else 'Collect'}: {TESTS[kind]}", flush=True)
            try:
                if args.analyze_only:
                    result = replay_collection(source / kind, output / kind / "analysis_predicted.pt")
                else:
                    result = collect_boundary(output / kind) if kind == "boundary" else collect_payloads(kind, output / kind, hook_dir)
                    seal_collection(output / kind)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                # pytest's missing-baseline failure is a BaseException too; never
                # turn it into a skipped file or an incomplete PASS report.
                detail = traceback.format_exc()
                (output / f"{kind}.error.log").write_text(detail)
                result = {"kind": kind, "file": TESTS[kind], "verdict": "FAIL",
                          "error": f"{type(exc).__name__}: {exc}", "error_log": f"{kind}.error.log"}
            results.append(result)
            print(f"{kind}: {result['verdict']}", flush=True)
            write_report(output, results, provenance, "CPU artifact replay" if args.analyze_only else "CUDA original probes")
    return write_report(output, results, provenance, "CPU artifact replay" if args.analyze_only else "CUDA original probes")


if __name__ == "__main__":
    raise SystemExit(main())

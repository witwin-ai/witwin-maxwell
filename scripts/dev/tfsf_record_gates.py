"""Fresh-package TFSF record gates; CPU is a frozen-kernel diagnostic only.

Run with the maxwell interpreter. No GPU is used unless --device cuda is given.
The combined report preserves old measured checks as diagnostics. Only the
accepted replacement gates determine --check's exit status.
"""
from __future__ import annotations

import argparse
from copy import copy
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = "d7b7cdf3"
DEFAULT_OUT = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "coverage-tfsf-raw-phase" / "gates"
PULSES = ("axis_pulse", "special_pulse", "oblique_pulse", "slab_pulse")
CWS = ("axis_cw", "special_cw", "slab_cw", "oblique_cw", "beam_cw", "astigmatic_cw")
EPS = float(np.finfo(np.float32).eps)
ULPS = 4
SCHEMA_VERSION = 2
PHYSICS_RTOL = 1e-12
ENDPOINT_RTOL = 5e-9
FLOOR_K = 3
PULSE_LENGTHS = (512, 2048, 8192)


def diagnostic_execution(args, case, *, normalized=True, **kwargs):
    """FP64 TFSF admission override is confined to this frozen-kernel CPU probe.

    Production admission excludes TFSF and normalized FP64 observation. The
    ordinary CPU observer uses Python-double source sums and preserves complex128
    field exports, so enable its normalization only after public preparation.
    Full-field DFT is unused by these point gates and has FP32 phase weights.
    """
    from tests.fdtd.test_observer_origin_providers import execution
    dtype = getattr(torch, args.field_dtype)
    fp64 = dtype == torch.float64
    with ExitStack() as context:
        if fp64:
            context.enter_context(patch("witwin.maxwell.fdtd.runtime.precision.validate_fp64_forward"))
        if args.device == "cpu" and case == "grating_cw":
            # This frozen-kernel diagnostic also supplies the CPU FP64 Bloch
            # oracle; public CPU boundary admission remains unchanged.
            context.enter_context(patch("witwin.maxwell.fdtd.cpu_profile.admit_cpu_reference_scene",
                side_effect=lambda simulation, scene: setattr(scene, "_fdtd_cpu_reference", True)))
        prepared, electric, magnetic = execution(
            case, device=args.device, normalized=normalized and not fp64,
            field_dtype=dtype, observer_dtype=dtype, full_field_dft=not fp64, **kwargs)
    if fp64:
        prepared.simulation.config.spectral_sampler.normalize_source = normalized
        prepared.solver._normalize_source = normalized
    if args.device == "cpu":
        from tests.fdtd.cuda import torch_reference
        torch_reference._require_cuda_tensors = lambda *values: None
        prepared.solver.fdtd_module = torch_reference.NativeFDTDModule()
    if fp64 and case == "grating_cw":
        solver = prepared.solver
        prepare_observers = solver._prepare_observers

        def prepare_grating_observers(frequencies, window_type, steps):
            # Public FP64 complex observers are unqualified. This point-only
            # diagnostic allocates both replica groups via ordinary setup,
            # then promotes their ZERO accumulators before the first sample.
            solver._observer_dtype = torch.float32
            try:
                prepare_observers(frequencies, window_type, steps)
            finally:
                solver._observer_dtype = torch.float64
            assert not solver._plane_observer_groups
            for group in solver._point_observer_groups.values():
                for name in ("real", "imag", "aux_real", "aux_imag"):
                    group[name] = group[name].to(torch.float64)
            solver._sync_observer_legacy_state()

        solver._prepare_observers = prepare_grating_observers
    return prepared, electric, magnetic


def precision_inventory(solver):
    """Retain actual tensor precision, including source/CPML/auxiliary state."""
    found = {}
    seen = set()
    def visit(name, value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                found[name] = str(value.dtype)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{name}.{key}", child)
        elif isinstance(value, (tuple, list)):
            for key, child in enumerate(value):
                visit(f"{name}.{key}", child)
        elif type(value).__module__.startswith("witwin.") and hasattr(value, "__dict__"):
            for key, child in vars(value).items():
                visit(f"{name}.{key}", child)
    for name, value in vars(solver).items():
        visit(name, value)
    return found


def dft(values, theta, *, label=1., start=0, stop=None):
    """Unscaled rectangular exp(+iwt) sum, with actual global sample labels."""
    values = np.asarray(values)
    stop = len(values) if stop is None else stop
    n = np.arange(start, stop)
    return np.einsum("n,n...->...", np.exp(1j * theta * (n + label)), values[start:stop])


def advance_spectrum(values, theta, *, label=1., stop=512):
    """Endpoint identity for samples x[1:stop+1] on labels n+label."""
    p = np.exp(1j * theta)
    return (dft(values, theta, label=label, stop=stop)
            - values[0] * p**label + values[stop] * p**(stop + label)) / p


def transfer(row, *, stop, start=0):
    theta = row["theta"]
    y = np.array([dft(row["y"][:, k], theta, label=label, start=start, stop=stop)
                  for k, label in enumerate((1., .5))])
    return y / dft(row["r"], theta, label=row["beta"], start=start, stop=stop)


def max_relative(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))


def capture(args):
    # A parent retry can wait for a capture already running in another process.
    # This utility targets the Linux CPU/native validation environments.
    import fcntl
    with args.capture_file.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _capture(args)


def _capture(args):
    # Import the selected package BEFORE adding the shared test helper directory.
    import witwin.maxwell as mw
    sys.path.insert(0, str(ROOT))
    from witwin.maxwell.fdtd.runtime.representation import normalization_sample_time, normalization_signal

    torch.set_num_threads(1)
    payload = (torch.load(args.capture_file, weights_only=False) if args.capture_file.exists()
               else {"module_file": mw.__file__, "records": {}})
    assert Path(payload["module_file"]).resolve() == Path(mw.__file__).resolve()
    cases = args.cases or (*PULSES, *CWS, *(("grating_cw",) if args.device == "cuda" else ()))
    for case in cases:
        # Keep the original short-record dt. A separate CW run uses an exact
        # 105 samples/period so its steady window contains integer periods.
        variants = (("record", args.pulse_steps, 4.7619e-12),) if case in PULSES else (
            ("record", 513, 4.7619e-12), ("steady", args.cw_periods * 105, 1 / (105 * 2e9)),
            ("decomposition", 513 if args.smoke else 8193, 4.7619e-12))
        for variant, steps, dt in variants:
            if variant == "decomposition" and Path(mw.__file__).resolve() == (ROOT / "witwin/maxwell/__init__.py").resolve():
                continue  # Only A supplies the long carrier fit for b2.
            key = f"{case}/{variant}"
            if key in payload["records"]:
                continue
            print(f"capture {key}: {steps} steps, {mw.__file__}", flush=True)
            # FP64 field stepping retains the public absolute-export contract.
            # Replay below normalizes the SAME FP64 accumulators/exports.
            prepared, electric, magnetic = diagnostic_execution(
                args, case, steps=steps, dt=dt, normalized=args.field_dtype == "float32")
            result = prepared.run()
            solver = result.solver
            trace = result.raw_monitor("trace")
            y = np.stack([trace["components"][f].cpu().numpy().reshape(steps, -1)[:, 0]
                          for f in (electric, magnetic)], axis=-1)
            r = np.array([normalization_signal(solver, solver._normalization_source_time,
                         normalization_sample_time(solver, n)) for n in range(steps)])
            row = dict(y=y, r=r, theta=2 * np.pi * 2e9 * solver.dt,
                       beta=solver._normalization_label_offset, dt=solver.dt,
                       provider=solver._normalization_reference_provider,
                       precision=precision_inventory(solver),
                       trace_dtype=str(y.dtype),
                       export_dtype=str(result.raw_monitor("point")["components"][electric].dtype),
                       analytic=bool(solver.complex_fields_enabled), fields=(electric, magnetic),
                       exported=np.array([result.raw_monitor("point")["components"][f].cpu().numpy().item()
                                          for f in (electric, magnetic)]))
            payload["records"][key] = row
            torch.save(payload, args.capture_file)


def capture_defects(args):
    """Predict startup/activation defects from A's code, without B samples.

    In exact arithmetic q'_0 - q_1 evolves under the homogeneous step map.
    The only nonzero initial B entry is its explicitly imposed auxiliary W(0).
    Run that initial-state difference through A with every drive set to zero.
    """
    from dataclasses import replace
    import witwin.maxwell as mw
    sys.path.insert(0, str(ROOT))
    from witwin.maxwell.fdtd.checkpoint import restore_checkpoint_state
    from witwin.maxwell.sources import evaluate_source_time

    torch.set_num_threads(1)
    payload = {"module_file": mw.__file__, "records": {}}

    def prepare(case, steps):
        return diagnostic_execution(args, case, normalized=False, steps=steps)

    def zero_drives(value):
        if isinstance(value, dict):
            if "amplitude" in value and "frequency" in value:
                value["amplitude"] = 0.
            for child in value.values():
                zero_drives(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                zero_drives(child)

    cases = args.cases or (*PULSES, *CWS, *(("grating_cw",) if args.device == "cuda" else ()))
    for case in cases:
        print(f"startup prediction {case}: A one-step state and state/activation defect evolution", flush=True)
        prepared, electric, magnetic = prepare(case, 513)
        aux = prepared.solver._tfsf_state.get("auxiliary_grid")
        seed = 0. if aux is None else float(evaluate_source_time(aux.source_time, 0.))
        checkpoint = prepared.run_until(1).physics
        tensors = {name: -value for name, value in checkpoint.tensors.items()}
        if aux is not None:
            tensors["tfsf_aux_electric"][aux.source_index] += seed
        initial = replace(checkpoint, step=0, tensors=tensors)
        steps = args.pulse_steps - 1 if case in PULSES else 512
        homogeneous, _, _ = prepare(case, steps)
        solver = homogeneous.solver
        aux = solver._tfsf_state.get("auxiliary_grid")
        gated_amplitudes = {term["patch"].data_ptr(): float(term["source_time"]["amplitude"])
                            for group in ("electric_terms", "magnetic_terms")
                            for term in solver._tfsf_state[group]
                            if term.get("activation_delay_patch") is not None}
        for value in (solver._source_time, solver._normalization_source_time,
                      solver._tfsf_state, solver._electric_source_terms, solver._magnetic_source_terms):
            zero_drives(value)
        if aux is not None:
            zero_drives(aux.source_time)
        if gated_amplitudes:
            # Carrier phase advances, but the beam's spatial causal mask does
            # not. At each M/E Yee stage only newly activated cells differ:
            # [H(t-a)-H(t+dt-a)] * carrier(t, phase_A+theta).
            class ActivationDefectKernels:
                def __init__(self, module):
                    self.module = module

                def __getattr__(self, name):
                    return getattr(self.module, name)

                def addTimeShiftedSourcePatch3D(self, **kwargs):
                    time = kwargs["time"]
                    magnetic = kwargs["field"].data_ptr() in {
                        getattr(solver, field).data_ptr() for field in ("Hx", "Hy", "Hz")}
                    offset = 0. if magnetic else .5
                    step = round(time / solver.dt - offset)
                    future_time = (step + 1) * solver.dt + offset * solver.dt
                    activation = kwargs["activationDelayPatch"]
                    newly_active = (activation > time) & (activation <= future_time)
                    original = kwargs["sourcePatch"]
                    kwargs.update(sourcePatch=-original * newly_active,
                                  amplitude=gated_amplitudes[original.data_ptr()],
                                  phase=kwargs["phase"] + 2 * np.pi * 2e9 * solver.dt,
                                  causalGate=0)
                    return self.module.addTimeShiftedSourcePatch3D(**kwargs)

            solver.fdtd_module = ActivationDefectKernels(solver.fdtd_module)
        restore_checkpoint_state(solver, initial)
        result = homogeneous.run()
        trace = result.raw_monitor("trace")
        payload["records"][case] = dict(seed=seed, y=np.stack([
            trace["components"][field].cpu().numpy().reshape(steps, -1)[:, 0]
            for field in (electric, magnetic)], axis=-1),
            initial_state_max={name: float(value.abs().max()) for name, value in tensors.items()},
            activation_defect_terms=len(gated_amplitudes))
        torch.save(payload, args.defect_file)


def replay_observers(args):
    """Replay saved point samples through the selected package's REAL accumulator.

    No field stepping is repeated. The final export must be bitwise identical
    to the original full-scene export, validating the replay at all prefixes.
    """
    import witwin.maxwell as mw
    sys.path.insert(0, str(ROOT))
    from witwin.maxwell.fdtd.observers import get_observer_results

    torch.set_num_threads(1)
    traces = torch.load(args.traces_file, weights_only=False)
    payload = {"module_file": mw.__file__, "records": {}}
    records = traces["records"]
    if args.prediction_file:
        defects = torch.load(args.defects_file, weights_only=False)
        payload["input_module"] = traces["module_file"]
        records = {key: dict(row, y=row["y"][1:513] + defects["records"][key.split("/")[0]]["y"][:512])
                   for key, row in records.items() if key.endswith("/record")}
    for key, row in records.items():
        if not key.endswith(("/record", "/steady")):
            continue
        case = key.split("/")[0]
        steps = len(row["y"])
        prepared, electric, magnetic = diagnostic_execution(args, case, steps=steps, dt=row["dt"])
        solver = prepared.solver
        solver._normalize_source = True
        solver._prepare_observers((2e9,), "none", steps)
        snapshots = {}
        indices = {observer["component"].capitalize(): observer["field_index"] for observer in solver.observers}
        for n, values in enumerate(row["y"]):
            # The time and spectral point monitors share the same nearest node.
            for field, value in zip((electric, magnetic), values):
                getattr(solver, field)[tuple(indices[field])] = float(value.real)
                if solver.complex_fields_enabled:
                    getattr(solver, field + "_imag")[tuple(indices[field])] = float(value.imag)
            solver.accumulate_observers(n)
            if n + 1 in {512, 2048, 8192, steps} or (case in PULSES and n + 1 >= 8192 and (n + 1) % 512 == 0):
                result = get_observer_results(solver)["point"]
                snapshots[n + 1] = np.array([result["components"][field].cpu().numpy().item()
                                             for field in (electric, magnetic)])
        if not args.prediction_file:
            # FP64 captures are absolute. Check the absolute replay bitwise,
            # while retaining normalized snapshots for the A/(ii) comparisons.
            if args.field_dtype == "float64":
                solver._normalize_source = False
                result = get_observer_results(solver)["point"]
                full_export = np.array([result["components"][field].cpu().numpy().item()
                                        for field in (electric, magnetic)])
                solver._normalize_source = True
            else:
                full_export = snapshots[steps]
            exact = np.array_equal(full_export, row["exported"])
            print(f"{key}: observer replay final export bitwise equal = {exact}", flush=True)
            if not exact:
                raise AssertionError(f"observer replay differs for {key}: {full_export} != {row['exported']}")
        else:
            print(f"{key}: exported A-based predicted trajectory through current {args.field_dtype} accumulator", flush=True)
        if key.endswith("/steady"):
            # Explicit half-open rectangular windows, preserving the original
            # global phase recurrence and each package's reference conversion.
            window_steps = min(2100, steps // 2)
            for start, stop in ((steps - window_steps, steps), (steps - 2 * window_steps, steps - window_steps)):
                solver._prepare_observers((2e9,), "none", steps)
                for entry in solver._observer_spectral_entries:
                    entry["start_step"], entry["end_step"] = start, stop
                for n, values in enumerate(row["y"][:stop]):
                    if n >= start:
                        for field, value in zip((electric, magnetic), values):
                            getattr(solver, field)[tuple(indices[field])] = float(value.real)
                            if solver.complex_fields_enabled:
                                getattr(solver, field + "_imag")[tuple(indices[field])] = float(value.imag)
                    solver.accumulate_observers(n)
                result = get_observer_results(solver)["point"]
                snapshots[f"{start}:{stop}"] = np.array([result["components"][field].cpu().numpy().item()
                                                         for field in (electric, magnetic)])
        payload["records"][key] = snapshots
        torch.save(payload, args.prediction_file or args.replay_file)


def endpoint_prediction(a, b, stop=512, defect=None):
    """Prediction uses A alone; B only supplies declared reference labels."""
    theta = a["theta"]
    numerator = np.array([advance_spectrum(a["y"][:, k], theta, label=label, stop=stop)
                          for k, label in enumerate((1., .5))])
    if defect is not None:
        numerator += np.array([dft(defect[:, k], theta, label=label, stop=stop)
                               for k, label in enumerate((1., .5))])
    denominator = predicted_reference(a, b, stop)
    return numerator / denominator


def predicted_reference(a, b, stop=512):
    # Only auxiliary alpha moves from 0 to 1. Analytic/discrete CW changes beta
    # alone; advancing its real reference samples would add a spurious endpoint.
    theta = a["theta"]
    if a["provider"] in {"plane_wave_axis_aligned", "plane_wave_ref_x_ez", "plane_wave_aux"}:
        denominator = advance_spectrum(a["r"], theta, label=a["beta"], stop=stop)
    else:
        denominator = dft(a["r"], theta, label=a["beta"], stop=stop)
    denominator *= np.exp(1j * theta * (b["beta"] - a["beta"]))
    return denominator


def prediction_metrics(a, b, predicted, stop=512, exported=None):
    old, observed = (transfer(a, stop=stop), transfer(b, stop=stop)) if exported is None else exported
    change = np.abs(observed - old)
    error = np.abs(predicted - observed)
    def pairs(values):
        return [[float(z.real), float(z.imag)] for z in values]
    return dict(baseline=pairs(old), observed=pairs(observed), predicted=pairs(predicted),
                change_relative=max_relative(old, observed),
                error_relative=max_relative(observed, predicted),
                error_over_spectrum=float(np.max(error / np.maximum(np.abs(observed), 1e-300))),
                component_error_over_spectrum=(error / np.maximum(np.abs(observed), 1e-300)).tolist(),
                component_error_over_change=(error / np.maximum(change, 1e-300)).tolist(),
                error_over_change=float(np.max(error / np.maximum(change, 1e-300))),
                fp32_epsilon_of_change=float(np.max(error / np.maximum(change, 1e-300)) / EPS),
                fp64_epsilon_of_change=float(np.max(error / np.maximum(change, 1e-300)) / np.finfo(np.float64).eps),
                error_over_fp32_sqrt_steps_spectrum=float(np.max(error / np.maximum(np.abs(observed), 1e-300)) / (EPS * np.sqrt(stop))),
                error_over_fp64_sqrt_steps_spectrum=float(np.max(error / np.maximum(np.abs(observed), 1e-300)) / (np.finfo(np.float64).eps * np.sqrt(stop))),
                closes=bool(np.all(error <= ULPS * EPS * change)))


def harmonic_parts(row, *, stop=512, fit_start=6090, fit_stop=8190):
    """Determine A's steady +/- carriers on a declared late window only.

    Startup is the residual to these carriers. No B samples enter this fit.
    The residual contains numerical noise and any unresolved late transient.
    """
    theta = row["theta"]
    n = np.arange(len(row["y"]))
    basis = np.stack([np.exp(-1j * theta * n), np.exp(1j * theta * n)], axis=-1)
    coeff = np.linalg.lstsq(basis[fit_start:fit_stop], row["y"][fit_start:fit_stop], rcond=None)[0]
    positive = basis[:, :1] * coeff[:1]
    negative = basis[:, 1:] * coeff[1:]
    startup = row["y"] - positive - negative
    return positive[:stop + 1], negative[:stop + 1], startup[:stop + 1], coeff


def analyze(a_payload, b_payload, defects, a_observers, b_observers, rounded_predictions=None):
    report = dict(baseline=BASE, baseline_module=a_payload["module_file"],
                  current_module=b_payload["module_file"], fp32_epsilon=EPS,
                  ulps=ULPS, pulses={}, cw={})
    aa, bb = a_payload["records"], b_payload["records"]
    for case in PULSES:
        if f"{case}/record" not in aa:
            continue
        a, b = aa[f"{case}/record"], bb[f"{case}/record"]
        ao, bo = a_observers["records"][f"{case}/record"], b_observers["records"][f"{case}/record"]
        lengths = tuple(n for n in (512, 2048, 8192, 32768) if n in ao and n < len(a["y"]))
        differences = [max_relative(ao[n], bo[n]) for n in lengths]
        # A terminal full carrier period avoids mistaking a zero crossing for decay.
        tail_steps = int(np.ceil(2 * np.pi / a["theta"]))
        decay_lengths = sorted(n for n in ao if n <= len(a["y"]) - 1)
        tails = {}
        for n in decay_lengths:
            tails[n] = {f"{label}_{signal}": float(np.max(np.max(np.abs(row[signal][n-tail_steps:n]), axis=0)
                        / np.maximum(np.max(np.abs(row[signal]), axis=0), 1e-300)))
                        for label, row in (("a", a), ("ii", b)) for signal in ("y", "r")}
        # Also require EVERY E/H component separately, despite differing units.
        decayed = {}
        for n in decay_lengths:
            decayed[n] = all(np.all(np.max(np.abs(row[signal][n-tail_steps:n]), axis=0)
                              <= 1e-6 * np.max(np.abs(row[signal]), axis=0))
                             for row in (a, b) for signal in ("y", "r"))
        chosen = next((n for n in decay_lengths if decayed[n]), None)
        decay_difference = None if chosen is None else max_relative(ao[chosen], bo[chosen])
        a1 = dict(lengths=lengths, relative=differences, terminal_period_peak_ratios=tails,
                  decayed=decayed, decay_length=chosen,
                  decay_search_lengths=decay_lengths, decay_relative=decay_difference,
                  decay_tail_steps=tail_steps,
                  trace_sum_relative=[max_relative(transfer(a, stop=n), transfer(b, stop=n)) for n in lengths],
                  passes=bool(all(x > y for x, y in zip(differences, differences[1:]))
                              and chosen is not None and decay_difference <= 1e-6))
        defect = defects["records"][case]["y"][:512]
        pred = endpoint_prediction(a, b, defect=defect)
        a2 = prediction_metrics(a, b, pred, exported=(ao[512], bo[512]))
        a2["trace_sum"] = prediction_metrics(a, b, pred)
        if rounded_predictions is not None:
            rounded = rounded_predictions["records"][f"{case}/record"][512]
            a2["observer_replayed_prediction"] = prediction_metrics(a, b, rounded, exported=(ao[512], bo[512]))
        a2["endpoint_only"] = prediction_metrics(a, b, endpoint_prediction(a, b), exported=(ao[512], bo[512]))
        a2["initial_entry_seed"] = defects["records"][case]["seed"]
        a2["reference_prediction_relative"] = max_relative(
            np.asarray(dft(b["r"], b["theta"], label=b["beta"], stop=512)),
            np.asarray(predicted_reference(a, b)))
        a2["corrected_trajectory_peak_relative"] = float(np.max(
            np.max(np.abs(b["y"][:512] - a["y"][1:513] - defect), axis=0)
            / np.max(np.abs(a["y"][:513]), axis=0)))
        # Use peak scaling as well: relative errors at zeros are not useful.
        a2["trajectory_translation_peak_relative"] = float(np.max(
            np.max(np.abs(b["y"][:512] - a["y"][1:513]), axis=0)
            / np.max(np.abs(a["y"][:513]), axis=0)))
        report["pulses"][case] = dict(provider=a["provider"], a1=a1, a2=a2)
    for key in aa:
        if not key.endswith("/steady"):
            continue
        case = key.split("/")[0]
        a, b = aa[key], bb[key]
        # Twenty whole periods after at least 58 periods of settling.
        stop = len(a["y"])
        window_steps = min(2100, stop // 2)
        start = stop - window_steps
        old, observed = (observers["records"][key][f"{start}:{stop}"] for observers in (a_observers, b_observers))
        relative = max_relative(old, observed)
        # Adjacent 20-period windows expose unresolved startup without fitting B.
        stability = max(max_relative(transfer(row, start=start-window_steps, stop=start),
                                     transfer(row, start=start, stop=stop)) for row in (a, b))
        b1 = dict(window=[start, stop], periods=window_steps / 105, samples_per_period=105,
                  dt=a["dt"], analytic=a["analytic"], relative=relative,
                  fp32_epsilon=relative / EPS, adjacent_window_relative=stability,
                  trace_sum_relative=max_relative(transfer(a, start=start, stop=stop), transfer(b, start=start, stop=stop)),
                  startup_excluded=bool(stability <= ULPS * EPS),
                  passes=bool(relative <= ULPS * EPS and stability <= ULPS * EPS))
        ar, br = aa[f"{case}/record"], bb[f"{case}/record"]
        # Fit A's short-dt steady carriers in a separate long run at that SAME dt.
        # These are captured separately; never transfer fitted phases between dts.
        long_key = f"{case}/decomposition"
        if long_key not in aa:
            raise ValueError(f"missing {long_key}")
        al = aa[long_key]
        fit_start, fit_stop = (6090, 8190) if len(al["y"]) >= 8193 else (len(al["y"]) // 2, len(al["y"]) - 1)
        pos, neg, startup, _ = harmonic_parts(al, fit_start=fit_start, fit_stop=fit_stop)
        theta = ar["theta"]
        r_old = dft(ar["r"], theta, label=ar["beta"], stop=512)
        r_new = predicted_reference(ar, br)
        defect = defects["records"][case]["y"]
        contributions = {}
        predicted_change = np.zeros(2, dtype=complex)
        for name, values in (("positive", pos), ("counter_rotating", neg), ("startup", startup)):
            old_part = np.array([dft(values[:, k], theta, label=label, stop=512)
                                 for k, label in enumerate((1., .5))]) / r_old
            new_part = np.array([advance_spectrum(values[:, k], theta, label=label)
                                 for k, label in enumerate((1., .5))]) / r_new
            if name == "startup":
                new_part += np.array([dft(defect[:, k], theta, label=label)
                                      for k, label in enumerate((1., .5))]) / r_new
            delta = new_part - old_part
            predicted_change += delta
            contributions[name] = [[float(z.real), float(z.imag)] for z in delta]
        predicted = transfer(ar, stop=512) + predicted_change
        exports = (a_observers["records"][f"{case}/record"][512], b_observers["records"][f"{case}/record"][512])
        b2 = prediction_metrics(ar, br, predicted, exported=exports)
        b2["trace_sum"] = prediction_metrics(ar, br, predicted)
        if rounded_predictions is not None:
            rounded = rounded_predictions["records"][f"{case}/record"][512]
            b2["observer_replayed_prediction"] = prediction_metrics(ar, br, rounded, exported=exports)
        b2["decomposition_change"] = prediction_metrics(ar, br, exports[0] + predicted_change, exported=exports)
        b2["endpoint_only"] = prediction_metrics(ar, br, endpoint_prediction(ar, br), exported=exports)
        b2["reference_prediction_relative"] = max_relative(
            np.asarray(dft(br["r"], br["theta"], label=br["beta"], stop=512)), np.asarray(r_new))
        b2["corrected_trajectory_peak_relative"] = float(np.max(
            np.max(np.abs(br["y"][:512] - ar["y"][1:513] - defect), axis=0)
            / np.max(np.abs(ar["y"][:513]), axis=0)))
        b2["activation_defect_terms"] = defects["records"][case].get("activation_defect_terms", 0)
        b2.update(contributions=contributions,
                  contribution_relative={name: max_relative(transfer(ar, stop=512), transfer(ar, stop=512)
                          + np.array([complex(*z) for z in values])) for name, values in contributions.items()},
                  fit_window=[fit_start, fit_stop], fit_dt=al["dt"],
                  short_long_prefix_equal=bool(np.array_equal(ar["y"], al["y"][:513])))
        report["cw"][case] = dict(provider=a["provider"], b1=b1, b2=b2)
    report["diagnostics_pass"] = all(v["a1"]["passes"] and v["a2"]["closes"] for v in report["pulses"].values()) and all(
        v["b1"]["passes"] for v in report["cw"].values())
    return report


def gate_result(value, bound, **metadata):
    """One scalar acceptance check; positive margin means room to spare."""
    value, bound = float(value), float(bound)
    return dict(value=value, bound=bound, margin=bound - value,
                passes=bool(np.isfinite(value) and np.isfinite(bound) and value <= bound), **metadata)


def physics_gate(observed, predicted, *, endpoint_only=False):
    tolerance = ENDPOINT_RTOL if endpoint_only else PHYSICS_RTOL
    return gate_result(abs(observed - predicted), tolerance * abs(observed),
                       relative_error=float(abs(observed - predicted) / max(abs(observed), 1e-300)),
                       relative_bound=tolerance)


def floor_gate(a32, ii32, a64, ii64):
    floor_a, floor_ii = abs(a32 - a64), abs(ii32 - ii64)
    return floor_rule(abs((a32 - ii32) - (a64 - ii64)), floor_a, floor_ii)


def floor_rule(error, floor_a, floor_ii):
    return gate_result(error, FLOOR_K * (floor_a + floor_ii),
                       floor_a=float(floor_a), floor_ii=float(floor_ii), k=FLOOR_K)


def load_leg(directory, dtype):
    """Consume captures and their diagnostic receipt without field stepping."""
    diagnostics = json.loads((directory / "report.json").read_text())
    if diagnostics["baseline"] != BASE or diagnostics["field_dtype"] != dtype:
        raise ValueError(f"{directory}: expected baseline {BASE} and {dtype}")
    if dtype == "float64" and diagnostics["device"] != "cpu":
        raise ValueError("the FP64 oracle must be a CPU capture")
    files = {name: torch.load(directory / filename, weights_only=False) for name, filename in (
        ("a", "a.pt"), ("ii", "ii.pt"), ("defects", "defects.pt"),
        ("a_observers", "a-observers.pt"), ("ii_observers", "ii-observers.pt"))}
    for tree in ("a", "ii"):
        if files[tree]["module_file"] != files[f"{tree}_observers"]["module_file"]:
            raise ValueError(f"{directory}: observer/capture package mismatch for {tree}")
        for key, row in files[tree]["records"].items():
            expected = np.dtype("float64" if dtype == "float64" else "float32")
            if row["y"].real.dtype != expected:
                raise ValueError(f"{directory}: {tree}/{key} has wrong field dtype")
    return dict(**files, diagnostics=diagnostics)


def replacement_report(fp64, fp32, *, smoke=False):
    """Evaluate componentwise fixed bounds on the same records and windows."""
    legs = (fp64, fp32)
    if any(leg["diagnostics"].get("smoke", False) != smoke for leg in legs):
        raise ValueError("smoke scope must match both precision captures")
    keys = set(fp64["a"]["records"])
    for leg in legs:
        # Decomposition is A-only. All observation records must match across
        # trees and precisions; in particular never silently drop a grating.
        for tree in ("a", "ii"):
            actual = {key for key in leg[tree]["records"] if not key.endswith("/decomposition")}
            expected = {key for key in keys if not key.endswith("/decomposition")}
            if actual != expected:
                raise ValueError("case/variant mismatch across precision legs or trees")
    gates = {"a_prime": [], "a1_endpoint": [], "a_double_prime": [], "b_prime": []}
    windows = []

    def append_components(group, observed, predicted, row, case, stop, **kwargs):
        for k, field in enumerate(row["fields"]):
            gates[group].append(dict(case=case, component=field, frequency_hz=2e9,
                window=[0, stop], **physics_gate(observed[k], predicted[k], **kwargs)))

    for key in sorted(keys):
        if not key.endswith("/record"):
            continue
        case = key.split("/")[0]
        a, b = (fp64[tree]["records"][key] for tree in ("a", "ii"))
        for leg in legs:
            for tree in ("a", "ii"):
                row = leg[tree]["records"][key]
                if (row["fields"] != a["fields"] or row["theta"] != a["theta"]
                        or row["dt"] != a["dt"] or len(row["y"]) != len(a["y"])):
                    raise ValueError(f"{key}: scene/record mismatch across precision legs")
        lengths = ((512,) if smoke else PULSE_LENGTHS) if case in PULSES else (512,)
        for stop in lengths:
            if len(a["y"]) <= stop or len(b["y"]) <= stop:
                raise ValueError(f"{key}: missing endpoint at {stop}")
            defect = fp64["defects"]["records"][case]["y"][:stop]
            if len(defect) != stop:
                raise ValueError(f"{case}: recapture A startup defects through {stop} steps")
            observed = fp64["ii_observers"]["records"][key][stop]
            predicted = endpoint_prediction(a, b, stop=stop, defect=defect)
            if case in PULSES:
                append_components("a_prime", observed, predicted, a, case, stop)
            else:
                # Retain the A-only positive/counter-rotating/startup split.
                # Startup includes the independently evolved activation defect.
                al = fp64["a"]["records"][f"{case}/decomposition"]
                fit = (6090, 8190) if not smoke else (len(al["y"]) // 2, len(al["y"]) - 1)
                parts = harmonic_parts(al, stop=stop, fit_start=fit[0], fit_stop=fit[1])[:3]
                if not np.array_equal(al["y"][:stop + 1], a["y"][:stop + 1]):
                    raise ValueError(f"{case}: decomposition and record prefixes differ")
                numerator = np.zeros(len(a["fields"]), dtype=complex)
                contributions = {}
                denominator = predicted_reference(a, b, stop)
                for name, values in zip(("positive", "counter_rotating", "startup"), parts):
                    piece = np.array([advance_spectrum(values[:, k], a["theta"], label=label, stop=stop)
                                      for k, label in enumerate((1., .5))])
                    if name == "startup":
                        piece += np.array([dft(defect[:, k], a["theta"], label=label, stop=stop)
                                           for k, label in enumerate((1., .5))])
                    numerator += piece
                    contributions[name] = [[float(z.real), float(z.imag)] for z in piece / denominator]
                append_components("b_prime", observed, numerator / denominator, a, case, stop)
                for gate in gates["b_prime"][-len(a["fields"]):]:
                    gate.update(fit_window=list(fit), normalized_contributions=contributions)
            windows.append((key, 0, stop, stop))
        if case in PULSES:
            # The accepted a1 check explains the surviving long-record mode;
            # short-record startup belongs to a_prime, not this endpoint-only check.
            stop = lengths[-1]
            append_components("a1_endpoint", fp64["ii_observers"]["records"][key][stop],
                              endpoint_prediction(a, b, stop=stop), a, case, stop, endpoint_only=True)

    # Production floors cover every normalized observer snapshot/window used
    # in the original report, including the two CW candidate steady windows.
    for key in sorted(keys):
        if not key.endswith(("/record", "/steady")):
            continue
        snapshots = fp64["a_observers"]["records"][key]
        for snapshot in snapshots:
            if key.endswith("/record") and snapshot == len(fp64["a"]["records"][key]["y"]):
                # Original pulse diagnostics exclude the extra prediction endpoint.
                if key.split("/")[0] in PULSES:
                    continue
            start, stop = (0, snapshot) if isinstance(snapshot, int) else map(int, snapshot.split(":"))
            windows.append((key, start, stop, snapshot))
    for key, start, stop, snapshot in dict.fromkeys(windows):
        row = fp64["a"]["records"][key]
        values = []
        for leg in (fp32, fp64):
            for tree in ("a", "ii"):
                record = leg[tree]["records"][key]
                if record["dt"] != row["dt"] or len(record["y"]) != len(row["y"]) or record["fields"] != row["fields"]:
                    raise ValueError(f"{key}: floor requires the same scene, record and window")
                values.append(leg[f"{tree}_observers"]["records"][key][snapshot])
        for k, field in enumerate(row["fields"]):
            gates["a_double_prime"].append(dict(case=key.split("/")[0], variant=key.split("/")[1],
                component=field, frequency_hz=2e9, window=[start, stop],
                **floor_gate(*(value[k] for value in values))))
    if not any(gates.values()):
        raise ValueError("no replacement gate observations")
    return dict(schema_version=SCHEMA_VERSION, baseline=BASE, smoke=smoke,
        qualification_scope="plumbing smoke only" if smoke else "selected cases at required record lengths",
        constants=dict(physics_rtol=PHYSICS_RTOL, endpoint_rtol=ENDPOINT_RTOL, floor_k=FLOOR_K),
        assumptions=["a1 is the endpoint-only prediction residual at 8192 steps for each pulse; smoke uses 512",
                     "N is the observed normalized ii spectrum, with E/H bounded separately",
                     "startup/activation corrections are evolved independently using A only",
                     "FP32-to-FP64 floors include device differences when FP32 runs on CUDA"],
        cases=sorted(key.split("/")[0] for key in keys if key.endswith("/record")),
        gates=gates, passes=all(gate["passes"] for group in gates.values() for gate in group),
        diagnostics={dtype: leg["diagnostics"] for dtype, leg in zip(("float64", "float32"), legs)})


def check_report(report, check):
    if check and not report["passes"]:
        raise SystemExit(1)


def run_replacement(args):
    args.out.mkdir(parents=True, exist_ok=True)
    directories = (args.fp64_out or args.out / "fp64", args.fp32_out or args.out / "fp32")
    if not args.analyze_only:
        for dtype, directory in zip(("float64", "float32"), directories):
            leg_args = copy(args)
            leg_args.out = directory.resolve()
            leg_args.field_dtype = dtype
            leg_args.device = "cpu" if dtype == "float64" else args.device
            leg_args.cases = args.cases or (*PULSES, *CWS, *(("grating_cw",) if args.device == "cuda" else ()))
            run(leg_args)
    report = replacement_report(load_leg(directories[0], "float64"), load_leg(directories[1], "float32"), smoke=args.smoke)
    report["input_directories"] = {dtype: str(directory.resolve()) for dtype, directory in zip(("float64", "float32"), directories)}
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# TFSF replacement gates", "", f"Scope: {report['qualification_scope']}.", "",
             "| Gate | Case | Component | Window | Value | Bound | Margin | Pass |",
             "|---|---|---|---|---:|---:|---:|---|"]
    for group, gates in report["gates"].items():
        for gate in gates:
            lines.append(f"| {group} | {gate['case']} | {gate['component']} | {gate['window']} | "
                         f"{gate['value']:.9g} | {gate['bound']:.9g} | {gate['margin']:.9g} | {gate['passes']} |")
    lines += ["", "Original decay, few-ulp and steady-window diagnostics are preserved in report.json."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(report=str(args.out / "report.json"), passes=report["passes"],
                          gate_counts={name: len(group) for name, group in report["gates"].items()})))
    check_report(report, args.check)


def render_report(report):
    dtype = report["field_dtype"]
    lengths = next(iter(report["pulses"].values()))["a1"]["lengths"] if report["pulses"] else []
    lines = ["# TFSF finite-record diagnostics", "", f"A: `{report['baseline']}`; device: `{report['device']}`; precision: `{dtype}`.", "",
             "N denotes a normalized E/H spectrum. Each maximum is taken componentwise before reducing over E/H.",
             "The original decay, four-epsilon prediction and steady-window checks are diagnostics only.", "",
             "| Pulse | " + " | ".join(map(str, lengths)) + " | Decay length | a1 | a2 error/change | a2 error/abs(N) |",
             "|---|" + "---:|" * len(lengths) + "---:|---|---:|---:|"]
    for case, row in report["pulses"].items():
        a1, a2 = row["a1"], row["a2"]
        values = " | ".join(f"{v:.9g}" for v in a1["relative"])
        lines.append(f"| {case} | {values} | {a1['decay_length']} | {a1['passes']} | {a2['error_over_change']:.9g} | "
                     f"{a2['error_over_spectrum']:.9g} |")
    window = next(iter(report["cw"].values()))["b1"]["window"] if report["cw"] else None
    lines += ["", f"CW candidate steady window: {window}, dt=1/(105*2 GHz), 20 whole periods; adjacent-window startup check required.", "",
              "| CW | b1 relative | b1 eps32 | Adjacent window relative | b1 | b2 short change | b2 error/change | b2 error/abs(N) | b2 closes |",
              "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for case, row in report["cw"].items():
        b1, b2 = row["b1"], row["b2"]
        lines.append(f"| {case} | {b1['relative']:.9g} | {b1['fp32_epsilon']:.9g} | {b1['adjacent_window_relative']:.9g} | "
                     f"{b1['passes']} | {b2['change_relative']:.9g} | {b2['error_over_change']:.9g} | "
                     f"{b2['error_over_spectrum']:.9g} | {b2['closes']} |")
    lines += ["", "| CW | Positive-carrier delta | Counter-rotating delta | Startup delta | Trace-sum eps(change) |",
              "|---|---:|---:|---:|---:|"]
    for case, row in report["cw"].items():
        b2 = row["b2"]
        parts = b2["contribution_relative"]
        lines.append(f"| {case} | {parts['positive']:.9g} | {parts['counter_rotating']:.9g} | {parts['startup']:.9g} | "
                     f"{b2['trace_sum']['fp32_epsilon_of_change']:.9g} |")
    lines += ["", f"Additional diagnostic: A-based trajectory replayed through the current {dtype} accumulator; no observed current fields used.", "",
              "| Provider case | Replayed error/change | Replayed error/abs(N) |", "|---|---:|---:|"]
    for group, key in ((report["pulses"], "a2"), (report["cw"], "b2")):
        for case, row in group.items():
            rounded = row[key].get("observer_replayed_prediction")
            if rounded is not None:
                lines.append(f"| {case} | {rounded['error_over_change']:.9g} | {rounded['error_over_spectrum']:.9g} |")
    return "\n".join(lines) + "\n"


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        baseline = args.out / "baseline"
        if baseline.exists():
            raise FileExistsError(f"use a fresh --out or --analyze-only: {baseline}")
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", BASE, "witwin"], check=True, capture_output=True)
        baseline.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(baseline, filter="data")
        for label, tree in (("a", baseline), ("ii", ROOT)):
            environment = dict(os.environ, PYTHONPATH=str(tree), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if args.device == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            if label == "a":
                for name in ("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT", "WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT"):
                    environment.pop(name, None)
                environment["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(args.out / "a-native")
            command = [sys.executable, str(Path(__file__).resolve()), "--field-dtype", args.field_dtype,
                       "--capture-file", str(args.out / f"{label}.pt"),
                       "--device", args.device, "--pulse-steps", str(args.pulse_steps), "--cw-periods", str(args.cw_periods)]
            if args.cases:
                command += ["--cases", *args.cases]
            if args.smoke:
                command += ["--smoke"]
            with (args.out / f"{label}.log").open("w") as log:
                subprocess.run(command, cwd=args.out, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            replay_command = [sys.executable, str(Path(__file__).resolve()), "--field-dtype", args.field_dtype, "--replay-file",
                              str(args.out / f"{label}-observers.pt"), "--traces-file", str(args.out / f"{label}.pt"),
                              "--device", args.device]
            with (args.out / f"{label}-observers.log").open("w") as log:
                subprocess.run(replay_command, cwd=args.out, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--field-dtype", args.field_dtype, "--defect-file", str(args.out / "defects.pt"),
                   "--device", args.device, "--pulse-steps", str(args.pulse_steps)]
        if args.smoke:
            command += ["--smoke"]
        if args.cases:
            command += ["--cases", *args.cases]
        environment["PYTHONPATH"] = str(baseline)
        for name in ("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT", "WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT"):
            environment.pop(name, None)
        environment["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = str(args.out / "a-native")
        with (args.out / "defects.log").open("w") as log:
            subprocess.run(command, cwd=args.out, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--field-dtype", args.field_dtype, "--prediction-file", str(args.out / "predictions.pt"),
                   "--traces-file", str(args.out / "a.pt"), "--defects-file", str(args.out / "defects.pt"),
                   "--device", args.device]
        environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if args.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        with (args.out / "predictions.log").open("w") as log:
            subprocess.run(command, cwd=args.out, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    a, b = [torch.load(args.out / f"{label}.pt", weights_only=False) for label in ("a", "ii")]
    assert Path(a["module_file"]).resolve().is_relative_to((args.out / "baseline").resolve())
    assert Path(b["module_file"]).resolve() == (ROOT / "witwin/maxwell/__init__.py").resolve()
    defects = torch.load(args.out / "defects.pt", weights_only=False)
    assert Path(defects["module_file"]).resolve().is_relative_to((args.out / "baseline").resolve())
    a_observers, b_observers = [torch.load(args.out / f"{label}-observers.pt", weights_only=False)
                              for label in ("a", "ii")]
    assert a_observers["module_file"] == a["module_file"]
    assert b_observers["module_file"] == b["module_file"]
    rounded_predictions = torch.load(args.out / "predictions.pt", weights_only=False)
    assert rounded_predictions["module_file"] == b["module_file"]
    assert rounded_predictions["input_module"] == a["module_file"]
    report = analyze(a, b, defects, a_observers, b_observers, rounded_predictions)
    report["device"] = args.device
    report["schema_version"] = SCHEMA_VERSION
    report["role"] = "diagnostics_only"
    report["smoke"] = args.smoke
    report["field_dtype"] = args.field_dtype
    report["full_scene_export_normalized"] = args.field_dtype == "float32"
    report["gate_exports_normalized"] = True
    report["record_lengths"] = {key: len(row["y"]) for key, row in a["records"].items()}
    report["precision"], report["precision_inventories"] = {}, {}
    for label, payload in (("a", a), ("ii", b)):
        report["precision"][label] = {}
        for key, row in payload["records"].items():
            metadata = {name: row[name] for name in ("trace_dtype", "export_dtype") if name in row}
            if "precision" in row:
                inventory = row["precision"]
                digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
                report["precision_inventories"][digest] = inventory
                metadata["inventory_sha256"] = digest
            report["precision"][label][key] = metadata
    report["observer_replay_full_exports_bitwise"] = True
    report["defect_module"] = defects["module_file"]
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["current_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "report.md").write_text(render_report(report))
    print(json.dumps(dict(diagnostics=str(args.out / "report.json"), diagnostics_pass=report["diagnostics_pass"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--field-dtype", choices=("float32", "float64"), default="float32",
                        help="float64 uses a diagnostic-only CPU admission override and FP64 point observers")
    parser.add_argument("--cases", nargs="+", choices=(*PULSES, *CWS, "grating_cw"))
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--replacement-gates", action="store_true",
                        help="run both precision legs or combine existing captures; only replacement gates decide acceptance")
    parser.add_argument("--fp64-out", type=Path, help="existing FP64 CPU leg directory (default: OUT/fp64)")
    parser.add_argument("--fp32-out", type=Path, help="existing FP32 production leg directory (default: OUT/fp32)")
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing only: 513 pulse/CW samples, 4 CW periods, short carrier fit; no full qualification")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--pulse-steps", type=int, default=8193,
                        help="pulse updates including the extra prediction endpoint; extend decay search beyond 8192")
    parser.add_argument("--cw-periods", type=int, default=78,
                        help="CW steady-run length in periods; the last two 20-period windows check settling")
    parser.add_argument("--capture-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--defect-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--replay-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--traces-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--prediction-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--defects-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.field_dtype == "float64" and args.device != "cpu" and not args.replacement_gates:
        parser.error("the FP64 TFSF admission override is CPU-only")
    if args.smoke:
        args.pulse_steps, args.cw_periods = 513, 4
    if args.pulse_steps < 8193 and not args.smoke:
        parser.error("--pulse-steps must be at least 8193")
    if args.cw_periods < 78 and not args.smoke:
        parser.error("--cw-periods must be at least 78")
    args.out = args.out.resolve()
    if args.check and not args.replacement_gates:
        parser.error("--check requires --replacement-gates and both precision legs")
    action = capture if args.capture_file else capture_defects if args.defect_file else replay_observers if (args.replay_file or args.prediction_file) else run_replacement if args.replacement_gates else run
    action(args)


if __name__ == "__main__":
    main()

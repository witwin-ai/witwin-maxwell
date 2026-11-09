"""Fresh-process G1/G1-prime/G2/G4 capture and separate CPU comparison.

Examples::

    python -m benchmark.core_path.gates capture --dry-run
    python -m benchmark.core_path.gates capture --gate g1 --output <evidence-root>/core-cuda-gates/g1
    python -m benchmark.core_path.gates compare --gate g1 --output <evidence-root>/core-cuda-gates/g1

G1-prime requires a JSON mapping of exact tensor names to {"kind": "fields"
or "phasors" or "flux", "reason": "pre-declared numerical change"}. All other
items remain bit-identity checks. O0 declares no bounded exceptions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import witwin.maxwell as mw

from .scenes import (GATE_SCENES, G2_SCENES, W1_SCENES, W2_SCENES, W3_SCENES, DISPERSIVE_SCENES,
                     build_core_case)


OUTPUT = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "core-cuda-gates"
# Cached external references live only in the main checkout (untracked); G2 reads them there.


def _reference_cache():
    # Reference caches are untracked and live in the main checkout, shared by its worktrees.
    if "WITWIN_BENCHMARK_CACHE" in os.environ:
        return Path(os.environ["WITWIN_BENCHMARK_CACHE"])
    root = Path(__file__).resolve().parents[2]
    common = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True)
    main = (root / common.stdout.strip()).resolve().parent if common.returncode == 0 else root
    return main / "benchmark" / "cache"


REFERENCE_CACHE = _reference_cache()
THREAD_ENV = {f"{name}_NUM_THREADS": "8" for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR")}


def discrete_leaf(name, value):
    """Names describing discrete decisions remain exact even in floating storage."""
    return (not (value.is_floating_point() or value.is_complex())
            or any(token in name.lower() for token in
                   ("indices", "index", "mask", "owner", "count", "starts", "targets")))


def run_families(names):
    """Keep field, spectral and individual monitor scales independent."""
    family = {}
    for name in names:
        parts = name.split("/")
        if parts[0] in ("fields", "phasors", "dft_accumulators") and len(parts) > 1:
            if re.fullmatch(r"[EH][XYZ]", parts[1].upper()):
                family[name] = f"{parts[0]}/{parts[1][0].upper()}"
        elif parts[0] == "monitors" and len(parts) > 2:
            component = any(re.fullmatch(r"[EH][XYZ]", part.upper())
                            and parts[index + 1:] in ([], ["data"])
                            for index, part in enumerate(parts[2:], start=2))
            if component or (len(parts) == 3 and parts[2] in ("data", "field")):
                family[name] = f"monitors/{parts[1]}"
        elif parts[0] in ("physics", "outputs"):
            # Physics records get no cross-record structural-zero allowance.
            family[name] = name
    return family


def rule2_rows(reference, candidate, *, level, ref_half=None, new_half=None):
    """Collect shared-comparator receipts, including rejected T2 rows.

    Suppress only its final assertion so every rejection can be written to the
    gate receipt. All classification, bounds and growth checks remain shared.
    """
    from tests.support import numerical_equivalence as equivalence

    assert reference.keys() == candidate.keys() and reference
    if level == "run":
        assert ref_half is not None and new_half is not None, "Rule 2 G1 needs half-horizon captures"
        assert ref_half.keys() == new_half.keys() == reference.keys()
    rows = {}
    with patch.object(equivalence, "_assert_report"):
        for discrete in (False, True):
            keys = [key for key in reference if discrete_leaf(key, reference[key]) is discrete]
            if not keys:
                continue
            left = {key: reference[key] for key in keys}
            right = {key: candidate[key] for key in keys}
            if level == "run":
                report = equivalence.assert_run_level_equivalent(
                    {key: ref_half[key] for key in keys}, {key: new_half[key] for key in keys},
                    left, right, discrete=discrete, family=run_families(keys))
            else:
                report = equivalence.assert_numerically_equivalent(
                    left, right, level=level, discrete=discrete)
            for row in report["tensors"]:
                row.setdefault("structural_zero", False)
                rows[row["name"]] = row
    return rows


def tier_summary(rows):
    return {tier: sum(row["tier"] == tier for row in rows.values()) for tier in ("T0", "T1", "T2")}


def o1_candidate_simulation_options(arm: str) -> dict:
    """Pin both controls so the baseline remains the existing implementation."""
    return {"fdtd_core_path": arm == "candidate", "accelerated_setup": False}


def make_simulation(case, *, arm: str, steps: int, full_field_dft: bool):
    return mw.Simulation.fdtd(
        case.scene.clone(device="cuda"), frequencies=case.frequencies,
        run_time=mw.TimeConfig(time_steps=max(1, case.steps if case.steps is not None else steps)), full_field_dft=full_field_dft,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        cpml_config={"memory_mode": case.cpml_memory, "kappa_max": case.cpml_kappa_max}, cuda_graph=True,
        shutoff=case.shutoff, shutoff_check_interval=7 if case.shutoff else 100,
        enable_plot=case.boundaries,
        excitations=case.excitations,
        **({"absorber": case.absorber} if case.absorber is not None else {}),
        **o1_candidate_simulation_options(arm),
    )


def tensors(value, prefix: str) -> dict[str, torch.Tensor]:
    """Snapshot numerical leaves, including scalar flux and compiled indices."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    if type(value) in (float, complex):
        return {prefix: torch.tensor(value, dtype=torch.complex128 if type(value) is complex else torch.float64)}
    if isinstance(value, (np.ndarray, np.number, bool, int, float, complex)):
        return {prefix: torch.as_tensor(value).clone()}
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: tensor for name, item in value.items()
                for key, tensor in tensors(item, f"{prefix}/{name}").items()}
    if isinstance(value, (tuple, list)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in tensors(item, f"{prefix}/{index}").items()}
    return {}


def setup_tensors(solver) -> dict[str, torch.Tensor]:
    payload = {
        name: value for name, value in vars(solver).items()
        if name.startswith(("eps_", "sigma_e_", "sigma_m_", "mu_", "cex_", "cey_", "cez_",
                            "chx_", "chy_", "chz_", "inv_", "cpml_", "psi_"))
        and isinstance(value, torch.Tensor)
    }
    for name in ("_compiled_material_model", "_coefficient_uniformity", "_compiled_sources",
                 "_source_terms", "_electric_source_terms", "_magnetic_source_terms", "observers",
                 "_point_observer_groups", "_plane_observer_groups", "_observer_spectral_entries"):
        payload[name] = getattr(solver, name)
    return tensors(payload, "setup")


def name_has_nonlinear(name: str) -> bool:
    return name.endswith(("_kerr", "_tpa"))


def physics_tensors(solver, result) -> dict[str, torch.Tensor]:
    """Pin the private surface/wire state and the public port and loss outputs."""
    from witwin.maxwell.fdtd.checkpoint import (
        graph_state_tensors, iter_surface_impedance_state_specs, iter_thin_sheet_state_tensors,
        iter_breakdown_state_tensors,
    )

    state = dict(iter_surface_impedance_state_specs(solver))
    state.update(iter_thin_sheet_state_tensors(solver))
    if solver.dispersive_enabled:
        state.update(graph_state_tensors(solver))
    state.update(iter_breakdown_state_tensors(solver))
    if any(getattr(solver, flag, False) for flag in
           ("nonlinear_enabled", "modulation_enabled", "breakdown_enabled")):
        from witwin.maxwell.fdtd.runtime.stepping import _electric_decay_tensors, _electric_curl_tensors
        state.update({f"ce{axis}_{kind}": tensor for kind, values in
                      (("decay", _electric_decay_tensors(solver)), ("curl", _electric_curl_tensors(solver)))
                      for axis, tensor in zip("xyz", values)})
        for name in ("_modulation_step", "_modulation_time", "_nonlinear_peak"):
            value = getattr(solver, name, None)
            if value is not None:
                state[name] = value
    wire = getattr(solver, "_wire_runtime", None)
    if wire is not None:
        state.update(wire_current=wire.current, wire_charge=wire.charge)
    return tensors(state, "physics") | tensors({
        "ports": result.ports, "conductor_losses": result.conductor_losses,
        "circuits": result.circuits, "embedded_networks": result.embedded_networks,
        "breakdown": result.breakdown_data,
    }, "outputs")


def ade_tensors(solver):
    from witwin.maxwell.fdtd.checkpoint import (
        dispersive_state_name, iter_dispersive_state_specs, iter_magnetic_dispersive_state_specs,
    )

    return tensors({
        dispersive_state_name(component, model, index, name): entry[name]
        for specs in (iter_dispersive_state_specs, iter_magnetic_dispersive_state_specs)
        for component, model, index, names, entry in specs(solver) or ()
        for name in names
    }, "physics")


def partial_result(solver):
    """Export stopped-state observations using the ordinary runtime finalizers."""
    from witwin.maxwell.fdtd.runtime.stepping import (
        finalize_circuit_data, finalize_embedded_networks, finalize_port_data, finalize_wire_data,
    )
    from witwin.maxwell.fdtd.copper_foil import copper_loss_report

    if solver.dft_enabled:
        solver._sync_dft_legacy_state()
    if solver.observers_enabled:
        solver._sync_observer_legacy_state()
    monitors = solver.get_observer_results() if solver.observers_enabled else {}
    if getattr(solver, "time_observers_enabled", False):
        monitors = dict(monitors) | solver.get_time_observer_results()
    if getattr(solver, "breakdown_observers_enabled", False):
        monitors = dict(monitors) | solver.get_breakdown_observer_results()
    monitors = dict(monitors) | (finalize_wire_data(solver) or {})
    ports = finalize_port_data(solver) or {}
    return SimpleNamespace(
        fields={key.upper(): value for key, value in solver.get_frequency_solution(all_frequencies=True).items()
                if key in ("Ex", "Ey", "Ez")} if solver.dft_enabled else {},
        monitors=monitors, ports=ports, conductor_losses=copper_loss_report(solver),
        circuits=finalize_circuit_data(solver) or {},
        embedded_networks=finalize_embedded_networks(solver, ports) or {},
    )


def capture_case(args) -> None:
    bounded = json.loads(args.bounded_items.read_text()) if args.bounded_items else {}
    destination = args.output / args.arm / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    if args.gate == "g2":
        capture_g2(args, destination)
        return
    case = build_core_case(args.scene)
    steps = case.steps if case.steps is not None else args.steps
    snapshot_half = getattr(args, "snapshot_half", False)
    horizon = steps // 2 if snapshot_half else steps
    simulation = make_simulation(case, arm=args.arm, steps=args.steps, full_field_dft=True)
    from witwin.maxwell.fdtd.runtime import core_tables
    with patch.object(core_tables, "DELAYED_TABLE_BYTES", 0 if args.scene == "o2_delayed_inline" else core_tables.DELAYED_TABLE_BYTES):
        prepared = simulation.prepare()
    solver = prepared.solver
    if case.name in DISPERSIVE_SCENES:
        assert solver.dispersive_enabled, case.name
        if case.name == "magnetic_lorentz_slab":
            assert solver.magnetic_dispersive_enabled, case.name
        else:
            assert solver.electric_dispersive_enabled, case.name
    if case.name.startswith("w3_debye_"):
        # Electric ADE composed with live coefficients: both state machines must be active.
        assert solver.electric_dispersive_enabled, case.name
        assert solver.nonlinear_enabled is name_has_nonlinear(case.name), case.name
        assert solver.modulation_enabled is ("modulation" in case.name), case.name
    if case.name in ("copper_patch", "copper_board_60k"):
        assert solver._thin_sheets and solver._thin_sheets[0].copper is not None
    if case.name in W1_SCENES:
        feature = {
            "w1_copper_patch": "_copper_foils", "w1_thin_sheet_cavity": "_thin_sheets",
            "w1_lumped_port": "_port_runtimes", "w1_coax_port": "_port_runtimes",
            "w1_thin_wire": "_wire_runtime", "w1_cut_surface": "_cut_surface",
            "w1_sheet_feed": "_thin_sheets", "w1_sibc": "surface_impedance_enabled",
            "f3_coarse_11f": "_copper_foils", "f4_dipole_32": "_wire_runtime",
            "w1_circuit": "_circuit_runtimes", "w1_network": "_network_runtimes",
            "w1_terminal_port": "_port_runtimes", "w1_wave_port": "_electric_source_terms",
        }[case.name]
        assert getattr(solver, feature), (case.name, feature)
        if case.name == "w1_sheet_feed":
            assert any(sheet.feeds for sheet in solver._thin_sheets)
    if args.arm == "candidate":
        solver._core_runtime.chunk_size = args.chunk_size
    if case.initial_fields:
        for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
            field = getattr(solver, name)
            field.copy_(((torch.arange(field.numel(), device=field.device).reshape(field.shape) % 17) - 8)
                        * (2.0 ** (-20-index)))
    if args.gate == "g4":
        # Sampling indices/weights are normally resolved at solve entry. Build
        # those tables explicitly without advancing any fields for the setup gate.
        solver._prepare_observers(case.frequencies, "none", steps)
        solver.enable_dft(case.frequencies, window_type="none", end_step=steps)
        solver.build_dft_step_tables(steps)
        payload = setup_tensors(solver)
        rows = (solver._core_runtime.sources.rows if args.arm == "candidate" else source_table_reference(
            solver, max(1, steps), budget=0 if args.scene == "o2_delayed_inline" else 64*1024*1024))
        payload.update(tensors(rows, "tables/source"))
        payload.update(tensors((solver._dft_weighted_cos_table, solver._dft_weighted_sin_table), "tables/dft"))
    else:
        plots = []
        from witwin.maxwell.fdtd.runtime import stepping
        solver.verbose = case.boundaries
        solver.progress_update_interval = 11
        solver.plot_interval = 13
        def fail_capture(_solver, step, count):
            step()
            raise RuntimeError("Forced core whole-step capture failure after state mutation.")
        from contextlib import nullcontext
        failure = (patch.object(solver._core_runtime, "capture_chunk", side_effect=fail_capture)
                   if case.capture_failure and args.arm == "candidate" else nullcontext())
        # The observer engine is dropped after the run; record whether solve entry built one.
        from witwin.maxwell.fdtd import observers as observer_module
        built_engine = []
        make_engine = observer_module.make_single_observer_engine
        def record_engine(_solver, **kwargs):
            run = make_engine(_solver, **kwargs)
            built_engine.append(_solver._single_observer_engine is not None)
            return run
        with failure, patch.object(stepping, "visualize_slice", side_effect=lambda _solver, n: plots.append(n)), \
                patch.object(observer_module, "make_single_observer_engine", side_effect=record_engine):
            if steps == 0:
                # Stop before the first step, preserving the ordinary no-export
                # contract for configured monitors with no recorded samples.
                solver.solve(time_steps=1, stop_step=0, dft_frequency=case.frequencies,
                             dft_window="none", use_cuda_graph=True)
                result = None
            elif snapshot_half:
                # Preserve the full-horizon tables and normalization; stop a fresh
                # solve at half rather than changing the configured run length.
                simulation._execute_fdtd_solve(solver, stop_step=horizon)
                result = partial_result(solver)
            else:
                result = prepared.run()
        payload = tensors({name: getattr(prepared.solver, name)
                           for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
        if result is not None:
            payload.update(tensors(result.fields, "phasors"))
            payload.update(tensors(result.monitors, "monitors"))
            if case.name in W1_SCENES + W2_SCENES + W3_SCENES + DISPERSIVE_SCENES:
                payload.update(physics_tensors(solver, result))
            if case.name == "breakdown_slab" and not snapshot_half:
                assert (solver.breakdown_runtime["trigger_step"] >= 0).any().item()
                assert solver.breakdown_runtime["total_dissipated"].item() > 0
            if getattr(args, "half_horizon", False):
                payload.update(ade_tensors(solver))
        payload.update(tensors(solver._dft_batched_fields, "dft_accumulators"))
        payload.update(tensors({"shutoff": solver._shutoff_triggered, "shutoff_step": solver._shutoff_step,
            "plots": plots, "dft_step": solver._dft_step if solver.dft_enabled else None,
            "observer_entries": solver._observer_spectral_entries,
            "port_samples": [(r.sample_index, r.accumulator._sample_count,
                              r.drive_accumulator._sample_count if r.drive_accumulator is not None else None)
                             for r in getattr(solver, "_port_runtimes", ())]}, "state"))
        if solver.tfsf_enabled and "auxiliary_grid" in solver._tfsf_state:
            aux = solver._tfsf_state["auxiliary_grid"]
            payload.update(tensors({"electric": aux.electric, "magnetic": aux.magnetic, "step": aux.time_step}, "auxiliary"))
        if args.arm == "candidate":
            from witwin.maxwell.fdtd.runtime.core_path import core_path_step_loop_blockers
            active = getattr(solver, "_core_step_graph_active", False)
            if case.name in W2_SCENES + W3_SCENES + DISPERSIVE_SCENES:
                assert active is (horizon > 0), case.name
            assert active is (horizon > 0 and not case.capture_failure and not core_path_step_loop_blockers(solver)
                              and not solver.time_observers_enabled
                              and (not solver.observers_enabled or all(built_engine)
                                   or observer_module.uses_native_point_observers(solver))), (case.name, active, built_engine)
            if active:
                assert int(solver._core_runtime.step.item()) == (solver._shutoff_step + 1 if solver._shutoff_triggered else horizon)
        if case.shutoff and not snapshot_half:
            assert solver._shutoff_triggered and solver._shutoff_step + 1 < steps
    if case.name in DISPERSIVE_SCENES and not snapshot_half:
        from witwin.maxwell.fdtd.checkpoint import (
            capture_checkpoint_state, graph_state_tensors, restore_checkpoint_state,
        )
        from witwin.maxwell.fdtd.resume import _configuration_fingerprint

        checkpoint = capture_checkpoint_state(solver, step=steps)
        path = destination / "physics-checkpoint.pt"
        torch.save(checkpoint, path)
        live = graph_state_tensors(solver)
        ade_names = checkpoint.schema.dispersive_state_names + checkpoint.schema.magnetic_dispersive_state_names
        for name in ade_names:
            live[name].zero_()
        restore_checkpoint_state(solver, torch.load(path, weights_only=False))
        for name in ade_names:
            assert torch.equal(live[name], checkpoint.tensors[name]), name
        payload["resume/fingerprint"] = torch.tensor(
            list(bytes.fromhex(_configuration_fingerprint(solver))), dtype=torch.uint8)
    torch.save(payload, destination / ("tensors_half.pt" if snapshot_half else "tensors.pt"))
    (destination / ("capture_half.json" if snapshot_half else "capture.json")).write_text(json.dumps({
        "gate": args.gate, "arm": args.arm, "steps": steps,
        "process_id": os.getpid(),
        "core_runtime_active": hasattr(prepared.solver, "_core_runtime"),
        "whole_step_capture_active": getattr(solver, "_core_step_graph_active", False),
        "whole_step_capture_k": getattr(solver, "_core_step_graph_k", None),
        "requested_chunk_size": args.chunk_size,
        "whole_step_chunks": getattr(solver, "_core_step_graph_chunks", []),
        "launches_per_step": (2 + sum(map(len, solver._core_runtime.sources.groups.values()))
            + (4 if solver._core_runtime.sources.auxiliary is not None else 0)
            + int(solver.dft_enabled) + 1
            + (1 + len(solver._core_runtime.observer_groups) if hasattr(solver._core_runtime, "observer_groups") else 0)
            if getattr(solver, "_core_step_graph_active", False) and case.name not in W1_SCENES + W2_SCENES + W3_SCENES else None),
        "tensor_names": sorted(payload), "scene": case.plan(), "bounded_items": bounded,
        **({"horizon_steps": horizon} if getattr(args, "half_horizon", False) else {}),
    }, indent=2) + "\n")


def source_patch_gather_reference(field_shape, patch_shape, offsets, *, device, wrap=0):
    """Enumerate clipped entries and periodic images independently in C order."""
    contributions = []
    for entry, local in enumerate(np.ndindex(tuple(patch_shape))):
        coords = tuple(local[axis] + offsets[axis] for axis in range(3))
        if any(coord < 0 or coord >= size for coord, size in zip(coords, field_shape)):
            continue
        boundary = any(wrap & (1 << axis) and coord in (0, size - 1)
                       for axis, (coord, size) in enumerate(zip(coords, field_shape)))
        for subset in range(8):
            if subset & ~wrap:
                continue
            target = list(coords)
            for axis, size in enumerate(field_shape):
                if subset & (1 << axis):
                    if coords[axis] not in (0, size - 1):
                        break
                    target[axis] = size - 1 if coords[axis] == 0 else 0
            else:
                flat = (target[0] * field_shape[1] + target[1]) * field_shape[2] + target[2]
                contributions.append((0, flat, entry, subset | (int(boundary) << 3)))
    # A patch has component code zero; Python sorting retains entry/subset order.
    contributions.sort(key=lambda item: item[:2])
    starts, targets, entries = [], [], []
    for index, (component, target, entry, flags) in enumerate(contributions):
        if not targets or targets[-1] != (component, target):
            starts.append(index)
            targets.append((component, target))
        entries.append((entry, flags))
    starts.append(len(entries))
    return (torch.tensor(starts, dtype=torch.int64, device=device),
            torch.tensor(targets, dtype=torch.int64, device=device).reshape(-1, 2),
            torch.tensor(entries, dtype=torch.int64, device=device).reshape(-1, 2))


def source_table_reference(solver, steps, *, budget=64*1024*1024):
    """Independent scalar oracle plus the unchanged delayed-source CUDA evaluator."""
    from witwin.maxwell.sources import evaluate_source_time
    from witwin.maxwell.fdtd.excitation.temporal import _resolve_cw_signal
    def upload(values):
        return torch.tensor(np.asarray(values, dtype=np.float32), device=solver.device)
    rows = {"time_h": upload([n*solver.dt for n in range(steps)]),
            "time_e": upload([n*solver.dt+0.5*float(solver.dt) for n in range(steps)])}
    # Additive electric currents inject at (n+1/2)dt since the source-time-label change.
    groups = [("ordinary", solver._source_terms, True, False),
              ("magnetic", solver._magnetic_source_terms, False, True),
              ("electric", solver._electric_source_terms, True, True)]
    aux = solver._tfsf_state.get("auxiliary_grid") if solver.tfsf_enabled else None
    if aux is not None:
        rows["auxiliary/signals"] = upload([evaluate_source_time(aux.source_time, (aux.time_step+n+1)*aux.dt) for n in range(steps)])
        if aux.incident_reference is not None:
            rows["auxiliary/boundary_signals"] = upload([
                [evaluate_source_time(aux.source_time, (aux.time_step+n+k+1)*aux.dt) for k in range(aux.source_index)]
                for n in range(steps)])
    elif solver.tfsf_enabled:
        groups += [("tfsf_magnetic", solver._tfsf_state["magnetic_terms"], False, True),
                   ("tfsf_electric", solver._tfsf_state["electric_terms"], True, True)]
    for name, terms, half, generic in groups:
        for ordinal, term in enumerate(terms):
            source = term.get("source_time") or solver._source_time
            times = [n*solver.dt+(0.5*float(solver.dt) if half else 0.0) for n in range(steps)]
            if term["cw_cos_patch"] is not None:
                omega = solver.source_omega if term.get("omega") is None else term["omega"]
                signals = upload([_resolve_cw_signal(omega, source, t) for t in times]).reshape(steps, 2)
            elif term["delay_patch"] is not None:
                bits, indices = torch.unique(term["delay_patch"].view(torch.int32), return_inverse=True)
                delays = bits.view(torch.float32)
                size = steps*delays.numel()*4
                if size > budget:
                    signals = torch.empty(0, device=solver.device)
                else:
                    budget -= size
                    rows[f"{name}/{ordinal}/unique_delays"] = delays
                    rows[f"{name}/{ordinal}/delay_indices"] = indices.reshape(-1)
                    field = torch.zeros((delays.numel(), 1, 1), device=solver.device)
                    ones = torch.ones_like(field)
                    delay_patch = delays.reshape_as(field)
                    signals = torch.empty((steps, delays.numel()), device=solver.device)
                    for n, time_value in enumerate(times):
                        field.zero_()
                        solver.fdtd_module.addTimeShiftedSourcePatch3D(field=field, sourcePatch=ones,
                            delayPatch=delay_patch, activationDelayPatch=delay_patch,
                            offsetI=0, offsetJ=0, offsetK=0, timeKind=int(source["kind_code"]), time=time_value,
                            frequency=float(source["frequency"]), fwidth=float(source["fwidth"]),
                            amplitude=float(source["amplitude"]), phase=float(source["phase"]), delay=float(source["delay"]),
                            pedestal=float(source.get("pedestal", 0)), pedestalFwidth=float(source.get("pedestal_fwidth", 0)),
                            causalGate=0).launchRaw()
                        signals[n].copy_(field.reshape(-1))
            else:
                phase = float(term["phase_real"]) if not generic else 1.0
                if generic and solver.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
                    phase = float(term["phase_real"])
                signals = upload([evaluate_source_time(source, t)*phase for t in times])
                if solver.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
                    component = "xyz".index(term["field_name"][-1].lower())
                    wrap = sum(1 << axis for axis, name_axis in enumerate("xyz")
                               if axis != component and solver.scene.boundary.axis_kind(name_axis) == "periodic")
                    if wrap:
                        gather = source_patch_gather_reference(
                            getattr(solver, term["field_name"]).shape, term["patch"].shape,
                            term["offsets"], device=solver.device, wrap=wrap)
                        for key, tensor in zip(("gather_rows", "gather_targets", "gather_entries"), gather):
                            rows[f"{name}/{ordinal}/{key}"] = tensor
            rows[f"{name}/{ordinal}/signals"] = signals
    return rows


SCORED = (
    "field_l2", "field_shape_l2", "field_linf", "field_corr", "flux_error",
    "field_runtime_change", "flux_runtime_change", "per_frequency", "scalar_metrics",
)


def metric_row(row) -> dict:
    scores = {name: getattr(row, name) for name in SCORED}
    # Match the existing report's strict field/flux thresholds and independent
    # convergence exclusions. Preserve the convergence flags as part of status.
    return {"scores": scores, "status": {
        "field_converged": row.field_converged, "flux_converged": row.flux_converged,
        "field_l2_pass": None if row.field_converged is False else row.field_l2 < 0.1,
        "field_linf_pass": None if row.field_converged is False else row.field_linf < 0.1,
        "field_corr_pass": None if row.field_converged is False else row.field_corr > 0.99,
        "flux_pass": None if row.flux_converged is False or row.flux_error is None
        else row.flux_error < 0.05,
    }}


def capture_g2(args, destination: Path) -> None:
    from benchmark import paths, report, runner

    os.environ["WITWIN_BENCHMARK_NO_CLOUD"] = "1"
    original_fdtd = mw.Simulation.fdtd
    original_run = runner._run_maxwell
    captured = {}
    core_runtime_active = []
    capture_flags = []

    def fdtd(*positional, **keywords):
        simulation = original_fdtd(*positional, **keywords, **o1_candidate_simulation_options(args.arm))
        original_prepare = simulation.prepare

        def prepare(*positional, **keywords):
            prepared = original_prepare(*positional, **keywords)
            core_runtime_active.append(hasattr(prepared.solver, "_core_runtime"))
            if args.arm == "candidate":
                prepared.solver._core_runtime.chunk_size = args.chunk_size
            original_prepared_run = prepared.run
            def prepared_run(*positional, **keywords):
                # The benchmark probes dt with a one-step preparation, then
                # resolves its horizon. Freeze a fresh preparation at that horizon
                # so each solver uploads its source rows exactly once.
                if prepared.solver._time_plan["total_steps"] != simulation.config.run_time.time_steps:
                    prepared.solver = original_prepare().solver
                    if args.arm == "candidate":
                        prepared.solver._core_runtime.chunk_size = args.chunk_size
                result = original_prepared_run(*positional, **keywords)
                capture_flags.append({"active": getattr(prepared.solver, "_core_step_graph_active", False),
                                      "k": getattr(prepared.solver, "_core_step_graph_k", None)})
                return result
            prepared.run = prepared_run
            return prepared

        simulation.prepare = prepare
        return simulation

    def run(*positional, **keywords):
        output = original_run(*positional, **keywords)
        result = output[0]
        index = str(len(captured))
        captured[index] = tensors(result.fields, "phasors") | tensors(result.monitors, "monitors")
        return output

    # Redirect every generated report/plot; cached references remain read-only.
    with patch.object(paths, "CACHE_DIR", REFERENCE_CACHE), \
         patch.object(paths, "PLOTS_DIR", destination / "plots"), \
         patch.object(paths, "RESULTS_MD", destination / "results.md"), \
         patch.object(report, "RESULTS_MD", destination / "results.md"), \
         patch.object(runner, "_run_maxwell", run), \
         patch.object(mw.Simulation, "fdtd", side_effect=fdtd):
        row, = runner.run_benchmarks([args.scene])
    (destination / "metrics.json").write_text(json.dumps(metric_row(row), indent=2) + "\n")
    (destination / "capture.json").write_text(json.dumps({
        "gate": args.gate, "arm": args.arm, "core_runtime_active": core_runtime_active,
        "whole_step_capture": capture_flags,
        "requested_chunk_size": args.chunk_size,
    }, indent=2) + "\n")
    torch.save(tensors(captured, "runs"), destination / "tensors.pt")


def differences(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    assert reference.shape == candidate.shape
    assert reference.dtype == candidate.dtype
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    a = reference.to(torch.complex128 if reference.is_complex() else torch.float64)
    b = candidate.to(a.dtype)
    delta = (a - b).abs()
    relative = torch.where(delta == 0, 0.0, delta / a.abs())
    return {"equal": torch.equal(reference, candidate),
            "max_abs": delta.max().item() if delta.numel() else 0.0,
            "max_rel": relative.max().item() if delta.numel() else 0.0}


def compare(args) -> None:
    mode = getattr(args, "equivalence", "bitwise")
    if mode == "rule2" and args.gate not in ("g1", "g4"):
        raise ValueError("Rule 2 is supported only for G1 and G4")
    names = args.scenes or (G2_SCENES if args.gate == "g2" else GATE_SCENES)
    bounded = json.loads((args.output / "bounded-items.json").read_text()) if args.gate == "g1-prime" else {}
    if args.bounded_items:
        assert json.loads(args.bounded_items.read_text()) == bounded
    seen = set()
    results = {}
    passed = True
    for name in names:
        metadata_by_arm = {}
        for arm in ("baseline", "candidate"):
            metadata = json.loads((args.output / arm / name / "capture.json").read_text())
            assert metadata["gate"] == args.gate
            active = metadata["core_runtime_active"]
            activations = active if args.gate == "g2" else [active]
            assert activations and all(value is (arm == "candidate") for value in activations), (name, arm, active)
            metadata_by_arm[arm] = metadata
        if args.gate == "g2":
            baseline = args.baseline_json or args.output / "baseline.json"
            before = json.loads(baseline.read_text())[name]
            after = json.loads((args.output / "candidate" / name / "metrics.json").read_text())
            left, right = tensors(before["scores"], "scores"), tensors(after["scores"], "scores")
            left.update(torch.load(args.output / "baseline" / name / "tensors.pt", weights_only=True, map_location="cpu"))
            right.update(torch.load(args.output / "candidate" / name / "tensors.pt", weights_only=True, map_location="cpu"))
            assert before["status"] == after["status"], name
            # Check the complete score schema, including nonnumeric leaves.
            assert score_schema(before["scores"]) == score_schema(after["scores"]), name
        else:
            for arm in ("baseline", "candidate"):
                assert metadata_by_arm[arm]["bounded_items"] == bounded
            assert metadata_by_arm["baseline"]["steps"] == metadata_by_arm["candidate"]["steps"]
            left = torch.load(args.output / "baseline" / name / "tensors.pt", weights_only=True, map_location="cpu")
            right = torch.load(args.output / "candidate" / name / "tensors.pt", weights_only=True, map_location="cpu")
        assert left.keys() == right.keys() and left, name
        if mode == "rule2":
            horizons = {}
            if args.gate == "g1":
                for arm in ("baseline", "candidate"):
                    directory = args.output / arm / name
                    half_metadata = json.loads((directory / "capture_half.json").read_text())
                    full_metadata = metadata_by_arm[arm]
                    assert half_metadata["gate"] == "g1" and half_metadata["arm"] == arm
                    assert half_metadata["steps"] == full_metadata["steps"]
                    assert half_metadata["horizon_steps"] == full_metadata["steps"] // 2
                    assert half_metadata["core_runtime_active"] is (arm == "candidate")
                    horizons[arm] = torch.load(directory / "tensors_half.pt", weights_only=True, map_location="cpu")
            receipts = rule2_rows(left, right, level="setup" if args.gate == "g4" else "run",
                                  ref_half=horizons.get("baseline"), new_half=horizons.get("candidate"))
            for key in sorted(receipts):
                row = receipts[key]
                ok = row["tier"] != "T2"
                results[f"{name}/{key}"] = dict(row, equal=row["tier"] == "T0", passed=ok)
                passed &= ok
                print(f"{name}/{key}: tier={row['tier']} abs={row['max_abs']:.6g} "
                      f"bound={row['bound']:.6g} pass={ok}")
            continue
        for key in sorted(left):
            item = f"{name}/{key}"
            diff = differences(left[key], right[key])
            if item in bounded:
                entry = bounded[item]
                assert entry["reason"].strip()
                tolerance = {"fields": 1e-6, "phasors": 1e-5, "flux": 1e-5}[entry["kind"]]
                ok = diff["max_rel"] <= tolerance
                diff.update(reason=entry["reason"], tolerance=tolerance)
                seen.add(item)
            else:
                ok = diff["equal"]
            diff["passed"] = ok
            results[item] = diff
            passed &= ok
            print(f"{item}: equal={diff['equal']} abs={diff['max_abs']:.6g} rel={diff['max_rel']:.6g} pass={ok}")
    assert seen == bounded.keys(), "Every pre-named item must be compared"
    receipt = {"passed": passed, "items": results}
    if mode == "rule2":
        receipt["summary"] = tier_summary(results)
    (args.output / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


def determinism(args):
    """Compare complete snapshot manifests from independent capture directories."""
    directories = [Path(directory).resolve() for directory in args.dirs]
    if len(directories) < 3 or len(set(directories)) != len(directories):
        raise ValueError("Determinism requires at least 3 distinct independent capture directories")
    arm = args.arm
    roots = [directory / arm for directory in directories]
    manifests = [{path.relative_to(root) for path in root.rglob("*.pt")} for root in roots]
    assert manifests[0] and all(manifest == manifests[0] for manifest in manifests), "Missing snapshot files"
    assert any(path.name == "tensors.pt" for path in manifests[0]), "Missing full snapshots"
    assert all(path.with_name("tensors.pt") in manifests[0] for path in manifests[0]), "Missing full snapshots"
    results = {}
    for filename in sorted(manifests[0]):
        metadata_name = {"tensors.pt": "capture.json", "tensors_half.pt": "capture_half.json"}.get(filename.name)
        if metadata_name is not None:
            metadata = [json.loads((root / filename.with_name(metadata_name)).read_text()) for root in roots]
            processes = [record.get("process_id") for record in metadata]
            if any(process is not None for process in processes):
                assert all(type(process) is int and process > 0 for process in processes), "Missing capture process IDs"
                assert len(set(processes)) == len(processes), "Captures must come from independent fresh processes"
        snapshots = [torch.load(root / filename, weights_only=True, map_location="cpu") for root in roots]
        reference = snapshots[0]
        assert reference and all(value.keys() == reference.keys() for value in snapshots), (filename, "Missing tensors")
        for key in sorted(reference):
            a = reference[key]
            equal = all(a.shape == value[key].shape and a.dtype == value[key].dtype
                        and torch.equal(a, value[key]) for value in snapshots[1:])
            results[f"{filename}/{key}"] = {"equal": equal, "passed": equal}
    receipt = {"passed": all(row["passed"] for row in results.values()), "items": results,
               "arm": arm, "dirs": [str(directory) for directory in directories]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "determinism.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise SystemExit(1)
    print(f"DETERMINISM: {len(directories)} captures, {len(results)} tensors, passed")
    return receipt


def score_schema(value):
    if isinstance(value, dict):
        return {key: score_schema(item) for key, item in value.items()}
    if isinstance(value, list):
        return [score_schema(item) for item in value]
    return "number" if type(value) in (int, float) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "compare", "worker", "determinism"))
    parser.add_argument("--gate", choices=("g1", "g1-prime", "g2", "g4"), default="g1")
    parser.add_argument("--scenes", nargs="+", choices=GATE_SCENES)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--bounded-items", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--equivalence", choices=("bitwise", "rule2"), default="bitwise")
    parser.add_argument("--half-horizon", action="store_true")
    parser.add_argument("--snapshot-half", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dirs", nargs="+", type=Path)
    parser.add_argument("--scene", choices=GATE_SCENES)
    parser.add_argument("--chunk-size", type=int, choices=(1, 16, 32), default=32)
    args = parser.parse_args()
    args.arm = args.arm or ("candidate" if args.action == "determinism" else "baseline")
    if args.action == "determinism":
        if not args.dirs:
            parser.error("determinism requires --dirs with at least 3 capture directories")
        determinism(args)
        return
    if (args.half_horizon or args.snapshot_half) and args.gate != "g1":
        parser.error("half-horizon snapshots are supported only for G1")
    names = args.scenes or (G2_SCENES if args.gate == "g2" else GATE_SCENES)
    if args.gate == "g2":
        assert set(names) <= set(G2_SCENES)
    if args.dry_run:
        for name in names:
            print(json.dumps(build_core_case(name).plan()))
        print(json.dumps({"gate": args.gate, "action": args.action, "fresh_interpreters": True,
                          "arms": ["baseline", "candidate"], "steps": args.steps,
                          "comparison": "separate compare invocation", "output": str(args.output),
                          "candidate_options": o1_candidate_simulation_options("candidate")}))
    elif args.action == "worker":
        capture_case(args)
    elif args.action == "compare":
        compare(args)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        policy_args = []
        if args.gate == "g1-prime":
            bounded = json.loads(args.bounded_items.read_text()) if args.bounded_items else {}
            policy = args.output / "bounded-items.json"
            policy.write_text(json.dumps(bounded, indent=2) + "\n")
            policy_args = ["--bounded-items", str(policy)]
        for name in names:
            for arm in ("baseline", "candidate"):
                for half in ((False, True) if args.half_horizon else (False,)):
                    subprocess.run([
                        sys.executable, "-m", "benchmark.core_path.gates", "worker",
                        "--gate", args.gate, "--scene", name, "--arm", arm,
                        "--steps", str(args.steps), "--output", str(args.output),
                        "--chunk-size", str(args.chunk_size),
                        *(["--half-horizon"] if args.half_horizon else []),
                        *(["--snapshot-half"] if half else []),
                        *policy_args,
                    ], check=True, env=os.environ | THREAD_ENV | {
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                    })
        if args.gate == "g2":
            baseline = {name: json.loads((args.output / "baseline" / name / "metrics.json").read_text())
                        for name in names}
            (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")


if __name__ == "__main__":
    main()

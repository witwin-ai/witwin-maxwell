"""Immutable exact source rows and direct spectral routes for core graph replay."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ...sources import evaluate_source_time
from ..excitation.gather import build_patch_gather
from ..excitation.temporal import _resolve_cw_signal, injection_time


DELAYED_TABLE_BYTES = 64 * 1024 * 1024


def source_tables_supported(solver):
    sources = [source["source_time"] for source in solver._compiled_sources]
    if any(source.get("fn") is not None for source in sources):
        return False
    terms = solver._source_terms + solver._magnetic_source_terms + solver._electric_source_terms
    if any(term["delay_patch"] is not None and
           (term.get("source_time") or solver._source_time)["kind"] == "custom" for term in terms):
        return False
    if getattr(solver, "complex_fields_enabled", False) and solver.tfsf_enabled and any(
            term.get("delay_patch") is not None and (term.get("source_time") or solver._source_time)["kind"] == "custom"
            for key in ("magnetic_terms", "electric_terms") for term in solver._tfsf_state.get(key, ())):
        return False
    # Oblique auxiliary providers have not qualified for core graph replay.
    return not solver.tfsf_enabled or solver._tfsf_state["provider"] != "plane_wave_aux"


def _bloch_source_launch(solver, extension, step, term, injection_kind, steps):
    """One Bloch scatter launch with the eager per-step signal row on the device."""
    from ..cuda.backend import _patch_source_gather
    from ..excitation.temporal import _bloch_patch_common_kwargs, _resolve_term_source

    source, omega = _resolve_term_source(term, solver._source_time, solver.source_omega)
    times = [injection_time(injection_kind, n * solver.dt, solver.dt) for n in range(steps)]
    kwargs = _bloch_patch_common_kwargs(solver, term["field_name"], term)
    axis = kwargs["axisCode"]
    empty = torch.empty(0, device=solver.device)
    delays = activation = clock = empty
    waveform = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if term["cw_cos_patch"] is not None:
        kind, patch, second = 1, term["cw_cos_patch"], term["cw_sin_patch"]
        rows = [_resolve_cw_signal(omega, source, t) for t in times]
    elif term["delay_patch"] is not None:
        # temporal._launch_bloch_time_shifted_patch: the envelope at t - delay per cell, evaluated in
        # the float32 order of evaluate_time_shifted_waveform_torch from the Python double operands.
        kind, patch, second, rows = 2, term["patch"], term["patch"], None
        delays, activation = term["delay_patch"], term.get("activation_delay_patch")
        activation = empty if activation is None else activation
        clock = torch.as_tensor(np.asarray(times, dtype=np.float32), device=solver.device)
        frequency = float(source["frequency"])
        two_pi = 2.0 * np.pi
        waveform = [int(source["kind_code"]), float(source["amplitude"]), two_pi * frequency,
                    float(source.get("phase", 0.0)), float(source.get("delay", 0.0)),
                    max(two_pi * float(source.get("fwidth", 0.0)), 1.0e-30), float(source.get("pedestal", 0.0)),
                    two_pi * float(source.get("pedestal_fwidth", 0.0)), np.pi * frequency]
    else:
        kind, patch, second = 0, term["patch"], term["patch"]
        phase_real, phase_imag = float(term["phase_real"]), float(term["phase_imag"])
        rows = []
        for t in times:
            signal = float(evaluate_source_time(source, t))
            signal_sin = float(_resolve_cw_signal(omega, source, t)[1]) if source["kind"] == "cw" else 0.0
            rows.append((signal * phase_real + signal_sin * phase_imag,
                         signal * phase_imag - signal_sin * phase_real))
    signals = (clock if rows is None else
               torch.as_tensor(np.asarray(rows, dtype=np.float32).reshape(steps, 2), device=solver.device))
    real = getattr(solver, term["field_name"])
    gather = _patch_source_gather(real, patch, term["offsets"], axis, kwargs["wrapAxisA"], kwargs["wrapAxisB"],
                                  kwargs["sourceGather"])
    launch = extension.CoreStepLaunch(
        7, [step, real, getattr(solver, term["field_name"] + "_imag"), patch, second, signals, *gather,
            delays, activation, clock],
        [*term["offsets"], 1 if axis == 0 else 0, 1 if axis == 2 else 2, kind, waveform[0],
         int(activation.numel() > 0)],
        [kwargs["phaseCosA"], kwargs["phaseSinA"], kwargs["phaseCosB"], kwargs["phaseSinB"], *waveform[1:]])
    return launch, signals, gather


def compile_source_tables(solver, extension, step, steps):
    """Stage scalar double expressions once; evaluate unique delayed rows on CUDA."""
    def upload(values):
        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=solver.device)

    times = (upload([injection_time("magnetic_current", n * solver.dt, solver.dt) for n in range(steps)]),
             upload([injection_time("electric_current", n * solver.dt, solver.dt) for n in range(steps)]))
    empty = torch.empty(0, device=solver.device)
    empty_index = torch.empty(0, device=solver.device, dtype=torch.int64)
    groups = {}
    rows = {"time_h": times[0], "time_e": times[1]}
    budget = DELAYED_TABLE_BYTES

    bloch = bool(getattr(solver, "complex_fields_enabled", False))

    def compile_terms(name, terms, injection_kind, generic):
        nonlocal budget
        launches = []
        for ordinal, term in enumerate(terms):
            if bloch:
                launch, signals, gather = _bloch_source_launch(solver, extension, step, term, injection_kind, steps)
                rows[f"{name}/{ordinal}/signals"] = signals
                for key, tensor in zip(("gather_rows", "gather_targets", "gather_entries"), gather):
                    rows[f"{name}/{ordinal}/{key}"] = tensor
                launches.append(launch)
                continue
            source = term.get("source_time") or solver._source_time
            omega = solver.source_omega if term.get("omega") is None else term["omega"]
            values = [float(source.get(key, 0.0)) for key in
                      ("frequency", "fwidth", "amplitude", "phase", "delay", "pedestal", "pedestal_fwidth")]
            time_values = [injection_time(injection_kind, n * solver.dt, solver.dt)
                           for n in range(steps)]
            patch, second = term["patch"], empty
            delays = activation = empty
            delay_indices = empty_index
            tabulated = False
            if term["cw_cos_patch"] is not None:
                kind = 1
                patch, second = term["cw_cos_patch"], term["cw_sin_patch"]
                signals = upload([_resolve_cw_signal(omega, source, t) for t in time_values]).reshape(steps, 2)
            elif term["delay_patch"] is not None:
                kind = 2
                delays = term["delay_patch"]
                activation = term.get("activation_delay_patch")
                activation = delays if activation is None else activation
                bits, inverse = torch.unique(delays.view(torch.int32), return_inverse=True)
                unique_delays = bits.view(torch.float32)
                size = steps * unique_delays.numel() * 4
                tabulated = size <= budget
                if tabulated:
                    budget -= size
                    delay_indices = inverse.reshape(-1)
                    signals = torch.empty((steps, unique_delays.numel()), device=solver.device)
                else:
                    signals = empty
            else:
                kind = 0
                phase = float(term["phase_real"]) if not generic else 1.0
                if generic and solver.scene.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
                    phase = float(term["phase_real"])
                signals = upload([float(evaluate_source_time(source, t)) * phase for t in time_values])
            wrap = 0
            if kind == 0 and solver.scene.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
                component = "xyz".index(term["field_name"][-1].lower())
                wrap = sum(1 << a for a, axis in enumerate("xyz")
                           if a != component and solver.scene.scene.boundary.axis_kind(axis) == "periodic")
            tensors = [step, getattr(solver, term["field_name"]), patch, second, signals,
                       rows["time_h" if injection_kind == "magnetic_current" else "time_e"],
                       delays, activation, delay_indices]
            if wrap:
                gather = build_patch_gather(getattr(solver, term["field_name"]).shape,
                                            patch.shape, term["offsets"], device=solver.device, wrap=wrap)
                tensors.extend(gather)
                for key, tensor in zip(("gather_rows", "gather_targets", "gather_entries"), gather):
                    rows[f"{name}/{ordinal}/{key}"] = tensor
            integers = [*term["offsets"], kind, int(source["kind_code"]) if kind == 2 else 0,
                        wrap, int(term.get("activation_delay_patch") is not None), int(tabulated)]
            launch = extension.CoreStepLaunch(0, tensors, integers, values)
            if kind == 2 and tabulated:
                launch.compile_delayed(unique_delays, signals)
                rows[f"{name}/{ordinal}/delay_indices"] = delay_indices
                rows[f"{name}/{ordinal}/unique_delays"] = unique_delays
            rows[f"{name}/{ordinal}/signals"] = signals
            launches.append(launch)
        groups[name] = launches

    compile_terms("ordinary", solver._source_terms, "electric_current", False)
    compile_terms("magnetic", solver._magnetic_source_terms, "magnetic_current", True)
    compile_terms("electric", solver._electric_source_terms, "equivalent_electric_current", True)
    auxiliary = None
    aux = solver._tfsf_state.get("auxiliary_grid") if solver.tfsf_enabled else None
    if aux is not None and solver._tfsf_state.get("provider") == "plane_wave_fixed_k":
        # The complex fixed-k line reads its device drive rows by its own clock (fixed_k.py).
        auxiliary = SimpleNamespace(launch=aux.advance_electric)
    elif aux is not None:
        # Row n of the boundary table reads steps n..n+source_index-1: evaluate each
        # step once and gather, so every entry is the same scalar as before.
        width = aux.source_index if aux.incident_reference is not None else 1
        values = np.asarray([aux.source_value(aux.time_step+n) for n in range(steps + max(width - 1, 0))],
                            dtype=np.float32)
        signals = torch.as_tensor(values[:steps], device=solver.device)
        rows["auxiliary/signals"] = signals
        if aux.incident_reference is not None:
            window = np.arange(steps)[:, None] + np.arange(width)[None, :]
            rows["auxiliary/boundary_signals"] = torch.as_tensor(values[window], device=solver.device)
        auxiliary = extension.CoreStepLaunch(1, [step, aux.electric, aux.magnetic,
            aux.electric_decay, aux.electric_curl, signals], [aux.source_index], [])
    elif solver.tfsf_enabled:
        compile_terms("tfsf_magnetic", solver._tfsf_state["magnetic_terms"], "magnetic_current", True)
        compile_terms("tfsf_electric", solver._tfsf_state["electric_terms"], "equivalent_electric_current", True)
    return SimpleNamespace(groups=groups, auxiliary=auxiliary, rows=rows, steps=steps)


def compile_spectral_tables(solver, extension, step):
    """Reuse the existing exact phase recurrence tables without row gathers.

    Complex (Bloch split) fields add a second launch: the imaginary field into the
    auxiliary accumulators with the same rows, after the real one (accumulate_dft_gpu).
    """
    if not solver.dft_enabled:
        return ()
    from .spectral import dft_families

    parts = (("", "real", "imag"), ("_imag", "aux_real", "aux_imag"))
    launches = []
    for suffix, real, imag in parts[:2 if getattr(solver, "complex_fields_enabled", False) else 1]:
        # One launch per recorded family (E, then H when requested), as accumulate_dft_gpu.
        for family in dft_families(solver):
            tensors = [step, solver._dft_weighted_cos_table, solver._dft_weighted_sin_table]
            for component in family:
                buffers = solver._dft_batched_fields.get(component)
                tensors.extend([getattr(solver, component + suffix), buffers[real], buffers[imag]] if buffers is not None
                               else [solver._dft_empty_field, solver._dft_empty_accum, solver._dft_empty_accum])
            launches.append(extension.CoreStepLaunch(2, tensors, [], []))
    return tuple(launches)


def auxiliary_boundary_fold_plan(aux, batch):
    """Prove read/write separation using the patch kernel's fp32 coordinates.

    Interpolation uses a rounded reciprocal multiply, not a division. Include
    both fetched endpoints even when the interpolation weight is zero.
    The magnetic line reads every E endpoint, including the written ghosts,
    so correction separation alone never permits a same-grid boundary fold.
    """
    lower = upper = None
    if batch is not None:
        interpolated = "sample_indices_per_coeff" not in batch
        if interpolated:
            positions = batch["sample_positions"].detach()
            coord = (positions - float(np.float32(aux.s_min))) * float(np.float32(1.0 / aux.ds))
            indices = coord.clamp(0, aux.electric.numel() - 1).floor().to(torch.int64)
        else:
            indices = batch["sample_indices_per_coeff"].detach()
        if indices.numel():
            # Reduce on the owning device; only the two bounds cross to host.
            lower, upper = map(int, torch.stack(torch.aminmax(indices)).cpu().tolist())
            if interpolated:
                upper = min(upper + 1, aux.electric.numel() - 1)
    ghosts = range(1, aux.source_index) if aux.incident_reference is not None else range(0)
    correction_clear = (not ghosts or lower is None or upper < ghosts.start or lower >= ghosts.stop)
    # H[i] reads E[i] and E[i+1] for every i in [0, len(H)).
    line_clear = not ghosts or ghosts.start > aux.magnetic.numel()
    return SimpleNamespace(sample_range=(lower, upper), boundary_limit=aux.source_index,
                           correction_clear=correction_clear, line_clear=line_clear,
                           folded=False)


def compile_auxiliary_fusion(solver, extension, step, sources):
    """Prepare disjoint correction/line grids without changing provider admission."""
    solver._core_tfsf_aux_fused = False
    solver._core_tfsf_aux_boundary_folded = False
    if not getattr(solver, "tfsf_enabled", False) or sources.auxiliary is None:
        return None
    state = solver._tfsf_state
    aux = state["auxiliary_grid"]
    # The boundary kernel implements only the two-cell incident stencil.
    if (getattr(solver, "_core_tfsf_aux_fusion_disabled", False)
            or (aux.incident_reference is not None and aux.source_index != 2)
            or state["provider"] not in {"plane_wave_ref_x_ez", "plane_wave_axis_aligned"}):
        return None
    # A missing batch with nonempty terms would change the historical route.
    if any(state.get(key + "_terms") and state.get(key + "_batch") is None
           for key in ("magnetic", "electric")):
        return None
    for key, kind in (("magnetic", "electric"), ("electric", "magnetic")):
        assert all(term["sample_kind"] == kind for term in state[key + "_terms"]), \
            "Fused correction must read the opposite auxiliary field"
    plan = auxiliary_boundary_fold_plan(aux, state.get("magnetic_batch"))
    solver._core_tfsf_aux_fold_plan = plan
    empty = aux.electric.new_empty(0)
    empty_indices = torch.empty(0, dtype=torch.int32, device=aux.electric.device)
    signals = sources.rows["auxiliary/signals"]
    boundary_signals = sources.rows.get("auxiliary/boundary_signals", empty)
    common = [step, aux.electric, aux.magnetic, aux.electric_decay, aux.electric_curl,
              aux.magnetic_decay, aux.magnetic_curl, signals, boundary_signals, aux._incident_boundary_h]
    boundary = (extension.CoreStepLaunch(4, common, [aux.source_index], [])
                if aux.incident_reference is not None else None)

    def correction(key, operation, names, incident):
        # Identity is part of the race proof, rather than just a matching shape.
        assert incident is (aux.electric if operation == 5 else aux.magnetic)
        batch = state.get(key + "_batch")
        reference = batch is None or "sample_indices_per_coeff" in batch
        if batch is None:
            gather = [torch.zeros(1, dtype=torch.int64, device=aux.electric.device),
                      torch.empty((0, 2), dtype=torch.int64, device=aux.electric.device),
                      torch.empty((0, 2), dtype=torch.int64, device=aux.electric.device)]
            coeff, samples = empty, empty_indices
        else:
            gather = batch["source_gather"]
            coeff = batch["coeff_data"]
            samples = batch["sample_indices_per_coeff" if reference else "sample_positions"]
        origin = aux.s_min if operation == 5 else aux.s_min + .5 * aux.ds
        return extension.CoreStepLaunch(operation,
            [*common, *(getattr(solver, name) for name in names), coeff, samples, *gather],
            [aux.source_index, int(reference), int(aux.incident_reference is not None)],
            [origin, 1.0 / aux.ds])

    magnetic = correction("magnetic", 5, ("Hx", "Hy", "Hz"), aux.electric)
    electric = correction("electric", 6, ("Ex", "Ey", "Ez"), aux.magnetic)
    solver._core_tfsf_aux_fused = True
    return SimpleNamespace(boundary=boundary, magnetic=magnetic, electric=electric, fold_plan=plan)


def compile_monitor_routes(solver):
    """Keep the grouped engine's weight recurrence and gather kernels unchanged."""
    return solver._single_observer_engine


def _plane_point_indices(group, field, device):
    """Enumerate a plane group's cells in its accumulator order as point indices."""
    axis, index = int(group["axis_code"]), int(group["plane_index"])
    sizes = [size for a, size in enumerate(field.shape) if a != axis]
    first, second = torch.meshgrid(torch.arange(sizes[0]), torch.arange(sizes[1]), indexing="ij")
    planes = [first.reshape(-1), second.reshape(-1)]
    planes.insert(axis, torch.full_like(planes[0], index))
    return [coordinate.to(device=device, dtype=torch.int32) for coordinate in planes]


class IndexedObserverTables:
    """Exact eager observer weights and metadata, indexed by the core device step.

    Every point and plane group of the eager ``accumulate_observers`` loop gets one
    indexed point launch per local frequency (plane cells enumerated in accumulator
    order), and complex (Bloch split) fields a second launch on the imaginary field
    into the auxiliary accumulators, in the eager order.
    """

    def __init__(self, solver, step, time_steps, start_step=0):
        from copy import copy
        from ..observers import observer_step_weights
        from .spectral import observer_stagger_weights
        from ..cuda.backend import get_compiled_extension

        staged = copy(solver)
        staged._observer_spectral_entries = [entry.copy() for entry in solver._observer_spectral_entries]
        self.metadata_keys = ("phase_cos", "phase_sin", "window_normalization", "sample_count",
                              "source_dft_real", "source_dft_imag")
        self.metadata = np.empty((time_steps + 1, len(staged._observer_spectral_entries), 6), dtype=np.float64)

        def save_metadata(n):
            self.metadata[n] = [[entry[key] for key in self.metadata_keys]
                                for entry in staged._observer_spectral_entries]

        groups = list(solver._point_observer_groups.values()) + list(solver._plane_observer_groups.values())
        rows = [np.zeros((len(group["global_freq_indices"]), time_steps, 3), dtype=np.float64) for group in groups]
        # A resumed run starts from the restored entries; rows before start_step are never read.
        save_metadata(start_step)
        for n in range(start_step, time_steps):
            for index, entry, cosine, sine in observer_step_weights(staged, n):
                for group, table in zip(groups, rows):
                    local = group["freq_local_lookup"].get(index)
                    if local is not None:
                        c, s = observer_stagger_weights(entry, group["field_name"], cosine, sine)
                        table[local, n] = (1, c, s)
            save_metadata(n + 1)
        self.native = get_compiled_extension()
        self.routes = []
        state = []
        parts = (("", "real", "imag"), ("_imag", "aux_real", "aux_imag"))
        for group, table in zip(groups, rows):
            field = getattr(solver, group["field_name"])
            if "point_i" in group:
                points = [group["point_" + label] for label in "ijk"]
            elif "sample_points" in group:
                points = [group["sample_" + label].to(torch.int32) for label in "ijk"]
            else:
                points = _plane_point_indices(group, field, solver.device)
            weights = torch.as_tensor(table, dtype=group["real"].dtype, device=solver.device)
            for local in range(len(group["global_freq_indices"])):
                for suffix, real, imag in parts:
                    if real in group:
                        self.routes.append((getattr(solver, group["field_name"] + suffix), *points,
                                            group[real][local].reshape(-1), group[imag][local].reshape(-1),
                                            weights[local], step))
            state.extend(group[name] for _, real, imag in parts for name in (real, imag) if name in group)
        self.state_tensors = tuple(state)

    def launch(self):
        for args in self.routes:
            self.native.accumulate_point_observers_indexed(*args)

    def finalize(self, solver, steps_run):
        for entry, completed in zip(solver._observer_spectral_entries, self.metadata[steps_run]):
            entry.update(zip(self.metadata_keys, completed))
            entry["sample_count"] = int(entry["sample_count"])
        solver._sync_observer_primary_state()

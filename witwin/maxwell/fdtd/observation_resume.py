"""Snapshot observation accumulators without changing captured tensor identities."""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import numpy as np
import torch


_DFT_ARRAYS = (
    "_dft_phase_cos", "_dft_phase_sin", "_dft_phase_step_cos_values",
    "_dft_phase_step_sin_values", "_dft_start_steps", "_dft_end_steps",
    "_dft_window_normalization_values", "_dft_sample_count_values",
    "_dft_source_dft_real_values", "_dft_source_dft_imag_values",
)
_PARTS = ("real", "imag", "aux_real", "aux_imag")


def observation_configuration(solver):
    return {
        "dft_enabled": bool(getattr(solver, "dft_enabled", False)),
        "dft_window": getattr(solver, "dft_window_type", None),
        "dft_components": tuple(getattr(solver, "_dft_batched_fields", {})) if solver.dft_enabled else (),
        "dft_frequencies": tuple(entry["frequency"] for entry in getattr(solver, "_dft_entries", ())),
        "normalize_source": bool(getattr(solver, "_normalize_source", False)),
        "observer_window": getattr(solver, "observer_window_type", None),
    }


def _clone(value):
    if isinstance(value, Enum):
        return _clone(value.value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    if value is None or isinstance(value, (int, float, complex, bool, str)):
        return value
    raise TypeError(f"Unsupported observation checkpoint value {type(value).__name__}.")


def _groups(solver):
    result = {}
    if getattr(solver, "dft_enabled", False):
        for component, buffers in solver._dft_batched_fields.items():
            result[("dft", component)] = {part: buffers[part] for part in _PARTS}
    for family in ("point", "plane"):
        for key, group in getattr(solver, f"_{family}_observer_groups", {}).items():
            result[(family, key)] = {part: group[part] for part in _PARTS if part in group}
    return result


def capture_observations(solver):
    arrays = {name: getattr(solver, name) for name in _DFT_ARRAYS
              if getattr(solver, "dft_enabled", False) and getattr(solver, name, None) is not None}
    if solver.dft_enabled and isinstance(getattr(solver, "_dft_step", None), torch.Tensor):
        arrays["_dft_step"] = solver._dft_step
    time = []
    for record in getattr(solver, "time_observers", ()):
        count = record["next_index"]
        buffers = record["buffers"] if record["kind"] == "field_time" else {"flux": record["buffer"]}
        time.append({"count": count, "buffers": {name: value[:count] for name, value in buffers.items()}})
    return _clone({"configuration": observation_configuration(solver), "arrays": arrays, "groups": _groups(solver),
                   "entries": getattr(solver, "_observer_spectral_entries", []), "time": time})


def preflight_observations(solver, payload, step, copy_ops, host_updates):
    """Validate every observation before allowing any checkpoint mutation."""
    from .resume import _require_tensor_compatible

    current = capture_observations(solver)
    if not isinstance(payload, Mapping) or set(payload) != set(current):
        raise ValueError("FDTD resume observation state has an incompatible layout.")
    if payload["configuration"] != current["configuration"]:
        raise ValueError("FDTD resume observation configuration changed.")
    for category in ("arrays", "groups"):
        if not isinstance(payload[category], Mapping) or set(payload[category]) != set(current[category]):
            raise ValueError(f"FDTD resume observation {category} layout changed.")
    for name, saved in payload["arrays"].items():
        target = getattr(solver, name)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        _require_tensor_compatible(f"observations.{name}", saved, target)
        # Only the real precomputed-table path advances this device counter.
        # Host/complex paths can retain an unused counter from an earlier run.
        table_counter_active = (
            name == "_dft_step"
            and bool(getattr(solver, "dft_enabled", False))
            and bool(getattr(solver, "_dft_entries", ()))
            and not bool(getattr(solver, "complex_fields_enabled", False))
            and isinstance(getattr(solver, "_dft_weighted_cos_table", None), torch.Tensor)
            and isinstance(getattr(solver, "_dft_weighted_sin_table", None), torch.Tensor)
        )
        if table_counter_active and (
            saved.dtype != torch.int64 or saved.shape != (1,) or int(saved.item()) != step
        ):
            raise ValueError("FDTD resume observation DFT counter disagrees with the execution step.")
        copy_ops.append((target, saved))
    for key, buffers in _groups(solver).items():
        saved = payload["groups"][key]
        if not isinstance(saved, Mapping) or set(saved) != set(buffers):
            raise ValueError(f"FDTD resume observation group {key!r} changed.")
        for part, target in buffers.items():
            source = saved[part]
            if target is None:
                if source is not None:
                    raise ValueError("FDTD resume complex observation layout changed.")
            else:
                _require_tensor_compatible(f"observations.{key}.{part}", source, target)
                copy_ops.append((target, source))
    entries = getattr(solver, "_observer_spectral_entries", [])
    saved_entries = payload["entries"]
    if not isinstance(saved_entries, (tuple, list)) or len(saved_entries) != len(entries):
        raise ValueError("FDTD resume observer spectral layout changed.")
    for entry, saved in zip(entries, saved_entries):
        if not isinstance(saved, Mapping) or set(saved) != set(entry):
            raise ValueError("FDTD resume observer spectral entry changed.")
        for key, target in entry.items():
            value = saved[key]
            if isinstance(target, torch.Tensor):
                _require_tensor_compatible(f"observer.{key}", value, target)
                copy_ops.append((target, value))
            elif key in ("observer_spectral_origin", "source_reference_label", "source_reference",
                         "source_reference_label_offset", "source_reference_factor"):
                if value != target:
                    raise ValueError(f"FDTD resume observer {key} reference changed.")
            elif value is None or isinstance(value, (bool, int, float)):
                if key == "sample_count" and (not isinstance(value, int) or not 0 <= value <= step):
                    raise ValueError("FDTD resume observer sample count is invalid.")
                host_updates.append((entry, key, value))
            else:
                raise ValueError(f"FDTD resume observer {key} has invalid scalar state.")
    engine = getattr(solver, "_single_observer_engine", None)
    if engine is not None and engine.state is not None:
        names = ("phase_cos", "phase_sin", "phase_step_cos", "phase_step_sin", "h_offset_cos", "h_offset_sin",
                 "window_normalization", "sample_count", "source_dft_real", "source_dft_imag")
        source = torch.tensor([[[entry[name] for name in names] for entry in saved_entries]], dtype=engine.state.dtype)
        _require_tensor_compatible("observer.engine.state", source, engine.state)
        copy_ops.append((engine.state, source))
        copy_ops.append((engine.step, torch.full_like(engine.step, step)))
    time = getattr(solver, "time_observers", ())
    if not isinstance(payload["time"], (tuple, list)) or len(payload["time"]) != len(time):
        raise ValueError("FDTD resume time observer layout changed.")
    for record, saved in zip(time, payload["time"]):
        expected_count = sum(n < step for n in record["sample_steps"])
        if not isinstance(saved, Mapping) or saved.get("count") != expected_count:
            raise ValueError("FDTD resume time observer sample count is invalid.")
        buffers = record["buffers"] if record["kind"] == "field_time" else {"flux": record["buffer"]}
        if not isinstance(saved.get("buffers"), Mapping) or set(saved["buffers"]) != set(buffers):
            raise ValueError("FDTD resume time observer buffers changed.")
        for name, buffer in buffers.items():
            target = buffer[:expected_count]
            source = saved["buffers"][name]
            _require_tensor_compatible(f"time_observer.{name}", source, target)
            copy_ops.append((target, source))
        host_updates.append((record, "next_index", expected_count))

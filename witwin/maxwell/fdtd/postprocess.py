import numpy as np
import torch

from .coords import centered_cell_coords
from .runtime.representation import spectral_phasor_factor, spectral_scale


def _phasor_dtype(fields):
    return torch.promote_types(next(iter(fields.values()))["real"].dtype, torch.complex64)


def _tracks_grad(fields):
    """Adjoint seeds finalise leaf accumulators under autograd; ``out=`` writes cannot record that graph."""
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for payload in fields.values() for tensor in payload.values())


def _finalize_phasor(payload, scale, divisor, out):
    """``(real + 1j*imag) * scale [/ divisor]``, written into ``out`` when given.

    Bloch accumulators combine first (``combine_complex_spectral_components``). With
    ``out`` the quadrature product, the sum, the scale and the division run in place,
    in the out-of-place operation order (so the values are bitwise the same), and
    finalisation allocates no full-grid complex temporaries.
    """
    real, imag = payload["real"], payload["imag"]
    if payload["aux_real"] is not None and payload["aux_imag"] is not None:
        real, imag = real - payload["aux_imag"], imag + payload["aux_real"]
    if out is None:
        value = (real + 1j * imag) * scale
        return value if divisor is None else value / divisor
    torch.mul(imag, 1j, out=out)
    torch.add(real, out, out=out)
    out.mul_(scale)
    if divisor is not None:
        out.div_(divisor)
    return out


def _build_frequency_solution_from_fields(fields, *, frequency, sample_count, window_normalization, source_spectrum=None, phasor_factor=2.0, integration_dt=None, reference_frequency=None, out=None):
    """Finalise one DFT entry; ``out`` maps components to preallocated complex buffers (one frequency each)."""
    if sample_count == 0:
        raise ValueError("No DFT samples recorded. Run solve() with dft_frequency first.")

    scale = spectral_scale(window_normalization if window_normalization > 0 else sample_count,
                           phasor_factor=phasor_factor, integration_dt=integration_dt,
                           reference_frequency=reference_frequency, frequency=frequency)
    divisor = None
    if source_spectrum is not None:
        abs_s = np.abs(source_spectrum)
        floor = max(1e-10 * abs_s, 1e-30) if np.isscalar(abs_s) else 1e-30
        divisor = source_spectrum if abs_s > floor else floor
    if out is None and not _tracks_grad(fields):
        dtype = _phasor_dtype(fields)
        out = {name: torch.empty(payload["real"].shape, dtype=dtype, device=payload["real"].device)
               for name, payload in fields.items()}
    return {name: _finalize_phasor(payload, scale, divisor, None if out is None else out[name])
            for name, payload in fields.items()}


def _legacy_frequency_solution(solver):
    fields = {
        "Ex": {
            "real": solver.dft_Ex_real,
            "imag": solver.dft_Ex_imag,
            "aux_real": solver.dft_Ex_aux_real,
            "aux_imag": solver.dft_Ex_aux_imag,
        },
        "Ey": {
            "real": solver.dft_Ey_real,
            "imag": solver.dft_Ey_imag,
            "aux_real": solver.dft_Ey_aux_real,
            "aux_imag": solver.dft_Ey_aux_imag,
        },
        "Ez": {
            "real": solver.dft_Ez_real,
            "imag": solver.dft_Ez_imag,
            "aux_real": solver.dft_Ez_aux_real,
            "aux_imag": solver.dft_Ez_aux_imag,
        },
    }
    return _build_frequency_solution_from_fields(
        fields,
        sample_count=solver.dft_sample_count,
        window_normalization=solver.dft_window_normalization,
        phasor_factor=spectral_phasor_factor(solver),
        integration_dt=solver._spectral_integration_dt,
        reference_frequency=solver._spectral_reference_frequency,
        frequency=solver.dft_frequency,
    )


def _entry_source_spectrum(entry, scale):
    """Compute complex source spectrum from a DFT entry's accumulated source DFT."""
    sr = entry.get("source_dft_real", 0.0)
    si = entry.get("source_dft_imag", 0.0)
    from .runtime.spectral import convert_source_spectrum
    return convert_source_spectrum(entry, (sr + 1j * si) * scale)


def _solver_source_spectrum(solver, entry):
    """Return source spectrum for an entry if normalize_source is active, else None."""
    if not getattr(solver, '_normalize_source', False):
        return None
    if getattr(solver, '_normalization_source_time', None) is None:
        return None
    wn = entry.get("window_normalization", 0.0)
    sc = entry.get("sample_count", 0)
    if sc == 0:
        return None
    scale = spectral_scale(wn if wn > 0 else sc,
                           phasor_factor=entry.get("phasor_factor", spectral_phasor_factor(solver)),
                           integration_dt=solver._spectral_integration_dt,
                           reference_frequency=solver._spectral_reference_frequency, frequency=entry["frequency"])
    return _entry_source_spectrum(entry, scale)


def _resolve_dft_entry(solver, *, frequency=None, freq_index=None):
    if frequency is not None and freq_index is not None:
        raise ValueError("Pass either frequency or freq_index, not both.")

    entries = getattr(solver, "_dft_entries", ())
    if not entries:
        if frequency is not None or freq_index is not None:
            raise ValueError("Legacy DFT state does not support selecting by frequency or freq_index.")
        return None

    if freq_index is not None:
        index = int(freq_index)
        if index < 0 or index >= len(entries):
            raise IndexError(f"freq_index {index} is out of range for {len(entries)} DFT entries.")
        return entries[index]

    if frequency is not None:
        target = float(frequency)
        for entry in entries:
            if abs(float(entry["frequency"]) - target) <= max(abs(target), 1.0) * 1e-12:
                return entry
        raise ValueError(f"DFT frequency {target} was not accumulated.")

    return entries[0]


def get_frequency_solution(solver, *, frequency=None, freq_index=None, all_frequencies=False):
    if all_frequencies and (frequency is not None or freq_index is not None):
        raise ValueError("all_frequencies cannot be combined with frequency or freq_index.")

    entries = getattr(solver, "_dft_entries", ())
    if all_frequencies and entries:
        if len(entries) == 1:
            entry = entries[0]
            result = _build_frequency_solution_from_fields(
                entry["fields"],
                sample_count=entry["sample_count"],
                window_normalization=entry["window_normalization"],
                source_spectrum=_solver_source_spectrum(solver, entry),
                phasor_factor=entry.get("phasor_factor", spectral_phasor_factor(solver)),
                integration_dt=solver._spectral_integration_dt,
                reference_frequency=solver._spectral_reference_frequency,
                frequency=entry["frequency"],
            )
            if getattr(solver, "verbose", False):
                print(
                    f"DFT complete, accumulated {entry['sample_count']} samples, "
                    f"window: {solver.dft_window_type}"
                )
            return result

        # One preallocated (F, ...) buffer per component; each entry is finalised into its row.
        # Under autograd the rows are built out of place and stacked.
        tracked = _tracks_grad(entries[0]["fields"])
        if not tracked:
            dtype = _phasor_dtype(entries[0]["fields"])
            result = {component: torch.empty((len(entries), *payload["real"].shape), dtype=dtype,
                                             device=payload["real"].device)
                      for component, payload in entries[0]["fields"].items()}
        solutions = [
            _build_frequency_solution_from_fields(
                entry["fields"],
                sample_count=entry["sample_count"],
                window_normalization=entry["window_normalization"],
                source_spectrum=_solver_source_spectrum(solver, entry),
                phasor_factor=entry.get("phasor_factor", spectral_phasor_factor(solver)),
                integration_dt=solver._spectral_integration_dt,
                reference_frequency=solver._spectral_reference_frequency,
                frequency=entry["frequency"],
                out=None if tracked else {component: buffer[index] for component, buffer in result.items()},
            )
            for index, entry in enumerate(entries)
        ]
        if tracked:
            result = {component: torch.stack([solution[component] for solution in solutions], dim=0)
                      for component in entries[0]["fields"]}
        result["frequencies"] = tuple(float(entry["frequency"]) for entry in entries)
        if getattr(solver, "verbose", False):
            print(f"DFT complete, accumulated {len(entries)} frequency solutions, window: {solver.dft_window_type}")
        return result

    entry = _resolve_dft_entry(solver, frequency=frequency, freq_index=freq_index)
    if entry is None:
        result = _legacy_frequency_solution(solver)
        sample_count = solver.dft_sample_count
    else:
        result = _build_frequency_solution_from_fields(
            entry["fields"],
            sample_count=entry["sample_count"],
            window_normalization=entry["window_normalization"],
            source_spectrum=_solver_source_spectrum(solver, entry),
            phasor_factor=entry.get("phasor_factor", spectral_phasor_factor(solver)),
            integration_dt=solver._spectral_integration_dt,
            reference_frequency=solver._spectral_reference_frequency,
            frequency=entry["frequency"],
        )
        sample_count = entry["sample_count"]

    window_type = getattr(solver, "dft_window_type", "none")
    if getattr(solver, "verbose", False):
        print(f"DFT complete, accumulated {sample_count} samples, window: {window_type}")
    return result


def get_material_permittivity(solver):
    if solver.material_eps_r is not None:
        return solver.material_eps_r
    return solver.scene.permittivity


def get_centered_permittivity(solver):
    eps = get_material_permittivity(solver)
    return 0.125 * (
        eps[:-1, :-1, :-1] + eps[1:, :-1, :-1] +
        eps[:-1, 1:, :-1] + eps[1:, 1:, :-1] +
        eps[:-1, :-1, 1:] + eps[1:, :-1, 1:] +
        eps[:-1, 1:, 1:] + eps[1:, 1:, 1:]
    )


def interpolate_yee_to_center(solver, freq_solution):
    ex = freq_solution["Ex"]
    ey = freq_solution["Ey"]
    ez = freq_solution["Ez"]
    ex_int = 0.5 * (ex[:, :-1, :] + ex[:, 1:, :])
    ex_int = 0.5 * (ex_int[:, :, :-1] + ex_int[:, :, 1:])
    ey_int = 0.5 * (ey[:-1, :, :] + ey[1:, :, :])
    ey_int = 0.5 * (ey_int[:, :, :-1] + ey_int[:, :, 1:])
    ez_int = 0.5 * (ez[:-1, :, :] + ez[1:, :, :])
    ez_int = 0.5 * (ez_int[:, :-1, :] + ez_int[:, 1:, :])
    if isinstance(ex_int, torch.Tensor):
        field_data = torch.sqrt(torch.abs(ex_int) ** 2 + torch.abs(ey_int) ** 2 + torch.abs(ez_int) ** 2)
        field_data = field_data.detach().cpu().numpy()
    else:
        field_data = np.sqrt(np.abs(ex_int) ** 2 + np.abs(ey_int) ** 2 + np.abs(ez_int) ** 2)

    x_coords, y_coords, z_coords = centered_cell_coords(solver.scene, device=solver.scene.device)
    return field_data, x_coords, y_coords, z_coords


_TFSF_PATCH_KEYS = ("coeff_patch", "patch", "delay_patch", "cw_cos_patch", "cw_sin_patch")


def tfsf_run_record(solver):
    """The TFSF surface as the solver snapped it, or ``None`` without a TFSF source.

    ``node_indices`` are the ``(first, last)`` primal node indices per axis of the
    total-field region on the computational grid (absorber layers included, the
    index space of ``Result.fields`` coordinates). ``h_corrections`` lists every H
    block the H update adds an incident correction to: component, first index and
    block shape on that component's Yee grid.
    """
    state = getattr(solver, "_tfsf_state", None)
    if not getattr(solver, "tfsf_enabled", False) or state is None:
        return None
    corrections = []
    for term in state["magnetic_terms"]:
        patch = next(term[key] for key in _TFSF_PATCH_KEYS if term.get(key) is not None)
        corrections.append({"component": term["field_name"],
                            "start": tuple(int(v) for v in term["offsets"]),
                            "shape": tuple(int(v) for v in patch.shape[-3:])})
    return {"mode": state.get("mode", "box"), "axis": state.get("axis"),
            "node_indices": tuple((int(lo), int(hi)) for lo, hi in zip(state["lower"], state["upper"])),
            "h_corrections": tuple(corrections)}


def last_step_field_ratio(solver):
    """Per DFT frequency, the end-of-record term of the volume E phasor relative to the phasor.

    ``||E^N|| / (2 sin(omega dt / 2) ||sum_n w_n E^n exp(+i omega t_n)||)`` over the
    whole computational grid, with ``E^N`` the field after the last step run. The DFT
    scale and source normalization cancel. ``None`` without volume DFT accumulators.
    """
    entries = getattr(solver, "_dft_entries", None)
    fields = [getattr(solver, name, None) for name in ("Ex", "Ey", "Ez")]
    if not entries or any(not isinstance(field, torch.Tensor) for field in fields):
        return None
    if getattr(solver, "complex_fields_enabled", False):
        fields += [getattr(solver, name + "_imag") for name in ("Ex", "Ey", "Ez")]
    squares = [torch.linalg.vector_norm(field, dtype=torch.float64).square() for field in fields]
    last = torch.stack(squares).sum()
    sums = []
    for entry in entries:
        parts = []
        for name, payload in entry["fields"].items():
            if name[0] != "E":
                continue  # an opt-in H volume DFT is not part of the E record term
            real, imag = payload["real"], payload["imag"]
            if payload["aux_real"] is not None and payload["aux_imag"] is not None:
                real, imag = real - payload["aux_imag"], imag + payload["aux_real"]
            parts += [torch.linalg.vector_norm(real, dtype=torch.float64).square(),
                      torch.linalg.vector_norm(imag, dtype=torch.float64).square()]
        sums.append(torch.stack(parts).sum())
    norms = torch.stack(sums).sqrt().cpu().tolist()
    last = float(last.sqrt())
    return tuple(
        float("inf") if norm == 0.0 else last / (2.0 * abs(np.sin(np.pi * float(entry["frequency"]) * solver.dt)) * norm)
        for entry, norm in zip(entries, norms)
    )

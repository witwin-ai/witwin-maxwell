import numpy as np
import torch

from .boundary import combine_complex_spectral_components
from .coords import centered_cell_coords


def _build_frequency_solution_from_fields(fields, *, sample_count, window_normalization, source_spectrum=None):
    if sample_count == 0:
        raise ValueError("No DFT samples recorded. Run solve() with dft_frequency first.")

    if window_normalization > 0:
        scale = 2.0 / window_normalization
    else:
        scale = 2.0 / sample_count

    result = {}
    for component_name in ("Ex", "Ey", "Ez"):
        payload = fields[component_name]
        result[component_name] = combine_complex_spectral_components(
            payload["real"],
            payload["imag"],
            payload["aux_real"],
            payload["aux_imag"],
        ) * scale
    if source_spectrum is not None:
        abs_s = np.abs(source_spectrum)
        floor = max(1e-10 * abs_s, 1e-30) if np.isscalar(abs_s) else 1e-30
        safe = source_spectrum if abs_s > floor else floor
        for comp in ("Ex", "Ey", "Ez"):
            result[comp] = result[comp] / safe
    return result


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
    )


def _entry_source_spectrum(entry, scale):
    """Compute complex source spectrum from a DFT entry's accumulated source DFT."""
    sr = entry.get("source_dft_real", 0.0)
    si = entry.get("source_dft_imag", 0.0)
    return (sr + 1j * si) * scale


def _solver_source_spectrum(solver, entry):
    """Return source spectrum for an entry if normalize_source is active, else None."""
    if not getattr(solver, '_normalize_source', False):
        return None
    if getattr(solver, '_normalization_source_time', None) is None:
        return None
    wn = entry.get("window_normalization", 0.0)
    sc = entry.get("sample_count", 0)
    if wn > 0:
        scale = 2.0 / wn
    elif sc > 0:
        scale = 2.0 / sc
    else:
        return None
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
            )
            if getattr(solver, "verbose", False):
                print(
                    f"DFT complete, accumulated {entry['sample_count']} samples, "
                    f"window: {solver.dft_window_type}"
                )
            return result

        solutions = [
            _build_frequency_solution_from_fields(
                entry["fields"],
                sample_count=entry["sample_count"],
                window_normalization=entry["window_normalization"],
                source_spectrum=_solver_source_spectrum(solver, entry),
            )
            for entry in entries
        ]
        result = {
            "Ex": torch.stack([solution["Ex"] for solution in solutions], dim=0),
            "Ey": torch.stack([solution["Ey"] for solution in solutions], dim=0),
            "Ez": torch.stack([solution["Ez"] for solution in solutions], dim=0),
            "frequencies": tuple(float(entry["frequency"]) for entry in entries),
        }
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

    x_coords, y_coords, z_coords = centered_cell_coords(solver.scene)
    return field_data, x_coords, y_coords, z_coords

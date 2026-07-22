from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Sequence

import numpy as np
import torch

from ..sar import PowerNormalization, SARAveraging, SARResult
from .sar import compute_tissue_statistics

# Multi-source SAR combination (plan Phase 3).
#
# Same-frequency COHERENT sources must be summed as complex fields BEFORE the
# loss is formed (loss is quadratic in the field, so phase matters): the combiner
# sums the runs' complex electric spectra and performs a single SAR reduction.
#
# INCOHERENT sources add in the power domain: their point SAR / absorbed-power
# fields sum directly. This is only physical when every operand shares the same
# grid, tissue model, frequency set, field convention and power normalization, so
# the combiner validates that metadata and fails closed on any mismatch (it never
# silently combines results from different grids or normalizations).


def _require_results(results: Sequence, name: str):
    resolved = list(results)
    if len(resolved) < 2:
        raise ValueError(f"{name} requires at least two operands to combine.")
    return resolved


def _monitor_frequencies(result, monitor: str):
    from ..result import _find_scene_monitor

    public_monitor = _find_scene_monitor(result.scene, monitor)
    frequencies = (
        tuple(result.frequencies)
        if public_monitor.frequencies is None
        else tuple(public_monitor.frequencies)
    )
    return public_monitor, frequencies


def _prepared_grid_hash(result) -> str:
    """Hash a run's node-grid coordinates so same-shaped but differently spaced
    grids are distinguished (a bare field-shape check cannot tell them apart)."""
    scene = result.prepared_scene
    hasher = hashlib.sha256()
    for name in ("x_nodes64", "y_nodes64", "z_nodes64"):
        hasher.update(np.ascontiguousarray(getattr(scene, name), dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _stacked_electric_fields(result, frequencies):
    """Stack per-frequency complex Ex/Ey/Ez the way Result.power_loss consumes them."""
    fields = {}
    for component in ("Ex", "Ey", "Ez"):
        fields[component] = torch.stack(
            [result.tensor(component, frequency=frequency) for frequency in frequencies],
            dim=0,
        )
    return fields


def combine_coherent_sar(
    results: Sequence,
    *,
    monitor: str,
    weights: Sequence[complex] | None = None,
    averaging: SARAveraging | None = None,
    normalization: PowerNormalization | None = None,
) -> SARResult:
    """Coherently combine same-frequency source runs, then reduce once to SAR.

    Sums the complex electric spectra of every ``Result`` (optionally with complex
    ``weights`` to set relative amplitude/phase) and performs a single SAR
    reduction on the summed field, so interference is captured exactly. Every run
    must share the same monitor, grid and frequency set. Differentiable in the
    per-run fields.
    """
    resolved = _require_results(results, "combine_coherent_sar")
    reference, frequencies = _monitor_frequencies(resolved[0], monitor)
    freq_ref = torch.as_tensor(frequencies, dtype=torch.float64)
    grid_ref = _prepared_grid_hash(resolved[0])

    if weights is None:
        weights = [1.0] * len(resolved)
    if len(weights) != len(resolved):
        raise ValueError("weights must match the number of results.")

    summed = None
    for result, weight in zip(resolved, weights):
        if _prepared_grid_hash(result) != grid_ref:
            raise ValueError(
                "combine_coherent_sar requires an identical grid (node-coordinate "
                "hash mismatch); two runs with the same field shape but different "
                "spacing cannot be combined."
            )
        _, other_frequencies = _monitor_frequencies(result, monitor)
        other = torch.as_tensor(other_frequencies, dtype=torch.float64)
        if other.shape != freq_ref.shape or not bool(
            torch.all((other - freq_ref).abs() <= 1e-6 * freq_ref.abs().clamp(min=1.0))
        ):
            raise ValueError(
                "combine_coherent_sar requires identical monitor frequencies across runs."
            )
        fields = _stacked_electric_fields(result, frequencies)
        if summed is None:
            summed = {name: weight * tensor for name, tensor in fields.items()}
        else:
            for name, tensor in fields.items():
                if tensor.shape != summed[name].shape:
                    raise ValueError(
                        "combine_coherent_sar requires identical field grids across runs; "
                        f"component {name} shape {tuple(tensor.shape)} != "
                        f"{tuple(summed[name].shape)}."
                    )
                summed[name] = summed[name] + weight * tensor

    sar = resolved[0].sar(
        monitor,
        averaging=averaging,
        normalization=normalization,
        electric_fields=summed,
    )
    combined_provenance = dict(sar.provenance)
    combined_provenance["combination"] = {
        "mode": "coherent",
        "operands": len(resolved),
        "weights": [complex(weight) for weight in weights],
    }
    return _with_provenance(sar, combined_provenance)


def _with_provenance(sar: SARResult, provenance: dict) -> SARResult:
    return SARResult(
        point=sar.point,
        statistics=sar.statistics,
        normalization=sar.normalization,
        provenance=MappingProxyType(provenance),
        frequencies=sar.frequencies,
        coordinates=sar.coordinates,
        valid=sar.valid,
        occupancy=sar.occupancy,
        rho_cell=sar.rho_cell,
        cell_volume=sar.cell_volume,
        tissue_id=sar.tissue_id,
        tissue_names=sar.tissue_names,
        absorbed_power_density=sar.absorbed_power_density,
        averaged=sar.averaged,
        peaks=sar.peaks,
    )


def _check_metadata_match(reference: SARResult, other: SARResult):
    ref_prov = reference.provenance
    other_prov = other.provenance
    if ref_prov.get("grid_hash") != other_prov.get("grid_hash"):
        raise ValueError(
            "combine_incoherent_sar requires an identical grid (grid_hash mismatch)."
        )
    if ref_prov.get("field_convention") != other_prov.get("field_convention"):
        raise ValueError("combine_incoherent_sar requires an identical field convention.")
    if reference.normalization.payload() != other.normalization.payload():
        raise ValueError(
            "combine_incoherent_sar requires an identical power normalization on every operand."
        )
    if reference.frequencies.shape != other.frequencies.shape or not bool(
        torch.all(
            (reference.frequencies.to(torch.float64) - other.frequencies.to(torch.float64)).abs()
            <= 1e-6 * reference.frequencies.to(torch.float64).abs().clamp(min=1.0)
        )
    ):
        raise ValueError("combine_incoherent_sar requires identical frequencies.")
    if set(reference.absorbed_power_density) != set(other.absorbed_power_density):
        raise ValueError("combine_incoherent_sar requires identical loss channels.")
    if not torch.equal(reference.tissue_id, other.tissue_id):
        raise ValueError("combine_incoherent_sar requires an identical tissue map.")
    if not torch.equal(reference.valid, other.valid):
        raise ValueError("combine_incoherent_sar requires identical validity masks.")
    if reference.rho_cell.shape != other.rho_cell.shape or not torch.equal(
        reference.rho_cell, other.rho_cell
    ):
        raise ValueError(
            "combine_incoherent_sar requires an identical mass-density model."
        )


def combine_incoherent_sar(
    sar_results: Sequence[SARResult],
    *,
    averaging: SARAveraging | None = None,
) -> SARResult:
    """Incoherently combine SAR results by summing absorbed power in the power domain.

    Every operand must share grid, tissue model, frequencies, field convention and
    power normalization (validated; fails closed otherwise). The combined
    absorbed-power density is the per-channel sum, point SAR is re-formed as
    ``power / rho`` and per-tissue statistics are recomputed. Pass ``averaging`` to
    recompute cubical-prefix-v1 mass-averaged SAR and peaks on the combined power.
    Differentiable in the operands' fields.
    """
    resolved = _require_results(sar_results, "combine_incoherent_sar")
    for item in resolved:
        if not isinstance(item, SARResult):
            raise TypeError("combine_incoherent_sar operands must be SARResult instances.")
    reference = resolved[0]
    for other in resolved[1:]:
        _check_metadata_match(reference, other)

    valid = reference.valid
    rho_cell = reference.rho_cell
    cell_volume = reference.cell_volume
    safe_rho = torch.where(valid, rho_cell, torch.ones_like(rho_cell))
    nan = torch.full((), float("nan"), device=rho_cell.device, dtype=torch.float32)

    channels = tuple(
        channel for channel in reference.absorbed_power_density if channel != "total"
    )
    combined_density = {}
    point = {}
    total_q = None
    for channel in channels:
        summed = sum(item.absorbed_power_density[channel] for item in resolved)
        combined_density[channel] = summed
        point[channel] = torch.where(valid[None], summed / safe_rho[None], nan)
        total_q = summed if total_q is None else total_q + summed
    combined_density["total"] = total_q
    point["total"] = torch.where(valid[None], total_q / safe_rho[None], nan)

    total_power = total_q * cell_volume[None]
    statistics = compute_tissue_statistics(
        tissue_id=reference.tissue_id,
        valid=valid,
        rho_cell=rho_cell,
        cell_volume=cell_volume,
        total_power=total_power,
        total_sar=point["total"],
        tissue_names=reference.tissue_names,
    )

    averaged = {}
    peaks = {}
    if averaging is not None:
        from .sar_averaging import compute_mass_averaged_sar

        cell_sizes = reference.provenance.get("cell_sizes_dual")
        if cell_sizes is None:
            raise ValueError(
                "combine_incoherent_sar cannot recompute averaging: the operand SARResult "
                "carries no cell-size provenance (recompute it from a fresh Result.sar)."
            )
        averaged, peaks = compute_mass_averaged_sar(
            averaging,
            power_total=combined_density["total"],
            rho_cell=rho_cell,
            cell_volume=cell_volume,
            occupancy=reference.occupancy,
            valid=valid,
            coordinates=reference.coordinates,
            cell_sizes=tuple(cell_sizes),
            frequencies=reference.frequencies,
        )

    provenance = dict(reference.provenance)
    provenance["averaging_profile"] = None if averaging is None else averaging.payload()
    provenance["combination"] = {"mode": "incoherent", "operands": len(resolved)}

    return SARResult(
        point=MappingProxyType(point),
        statistics=MappingProxyType(statistics),
        normalization=reference.normalization,
        provenance=MappingProxyType(provenance),
        frequencies=reference.frequencies,
        coordinates=reference.coordinates,
        valid=valid,
        occupancy=reference.occupancy,
        rho_cell=rho_cell,
        cell_volume=cell_volume,
        tissue_id=reference.tissue_id,
        tissue_names=reference.tissue_names,
        absorbed_power_density=MappingProxyType(combined_density),
        averaged=MappingProxyType(averaged),
        peaks=MappingProxyType(peaks),
    )


__all__ = ["combine_coherent_sar", "combine_incoherent_sar"]

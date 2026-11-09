"""Static gather plans for native adjoint accumulation, built on the tensor device."""

from __future__ import annotations

import torch


def _sample_csr(samples, cells, weights, sample_count):
    # Stable sorting keeps flattened cell order (lower before upper for a tie).
    order = torch.argsort(samples, stable=True)
    counts = torch.bincount(samples, minlength=sample_count)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return offsets, cells[order].contiguous(), weights[order].contiguous()


def build_tfsf_sample_plan(term, *, sample_count, origin, ds):
    """CSR rows are auxiliary samples; entries are cells in ascending C order."""
    coeff = term["coeff_patch"]
    cells = torch.arange(coeff.numel(), device=coeff.device)
    if "sample_positions" in term:
        # Match the native float32 coordinate arithmetic, including reciprocal
        # rounding before multiplication and the clamped upper endpoint.
        positions = term["sample_positions"].reshape(-1)
        inv_ds = torch.tensor(1.0 / ds if ds > 0.0 else 0.0,
                              dtype=torch.float32, device=coeff.device)
        coord = ((positions - float(origin)) * inv_ds if ds > 0.0
                 else torch.zeros_like(positions))
        coord = coord.clamp(0, sample_count - 1)
        lower = coord.to(torch.long)
        upper = (lower + 1).clamp(max=sample_count - 1)
        frac = coord - lower.to(coord.dtype)
        keep = torch.stack((torch.ones_like(lower, dtype=torch.bool), upper != lower), dim=1).flatten()
        samples = torch.stack((lower, upper), dim=1).flatten()[keep]
        cells = cells[:, None].expand(-1, 2).reshape(-1)[keep]
        weights = torch.stack((1.0 - frac, frac), dim=1).flatten()[keep]
    else:
        weights = torch.ones(coeff.numel(), device=coeff.device, dtype=torch.float32)
        if term["scalar_sample_index"] is not None:
            samples = torch.full_like(cells, int(term["scalar_sample_index"]))
        else:
            from ..excitation.tfsf_specs import reference_sample_axis_code

            axis = reference_sample_axis_code(term)
            shape = [1, 1, 1]
            shape[axis] = coeff.shape[axis]
            samples = term["sample_indices"].to(device=coeff.device, dtype=torch.long)
            samples = samples.reshape(shape).expand(coeff.shape).reshape(-1)
    return _sample_csr(samples, cells, weights, sample_count)


def prepare_tfsf_sample_plans(solver):
    """Prepare once before reverse stepping; no plan work or host copies per step."""
    state = getattr(solver, "_tfsf_state", None)
    aux = (state or {}).get("auxiliary_grid")
    if aux is None:
        return
    ds = float(aux.ds)
    for key, samples, origin in (
        ("electric_terms", aux.magnetic, float(aux.s_min + 0.5 * ds)),
        ("magnetic_terms", aux.electric, float(aux.s_min)),
    ):
        for term in state.get(key, ()):
            if "adjoint_sample_plan" not in term:
                term["adjoint_sample_plan"] = build_tfsf_sample_plan(
                    term, sample_count=samples.numel(), origin=origin, ds=ds)


def build_point_cell_plan(point_i, point_j, point_k, field_shape):
    """Unique cells with CSR rows of original point indices in ascending order."""
    cells = (point_i * field_shape[1] + point_j) * field_shape[2] + point_k
    order = torch.argsort(cells, stable=True)
    unique, counts = torch.unique_consecutive(cells[order], return_counts=True)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return unique.contiguous(), offsets, order.contiguous()

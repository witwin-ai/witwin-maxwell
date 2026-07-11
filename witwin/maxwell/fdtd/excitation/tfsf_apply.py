from __future__ import annotations

import torch

from .tfsf_specs import reference_sample_axis_code
from .temporal import apply_generic_source_terms


_REFERENCE_PROVIDERS = {"plane_wave_ref_x_ez", "plane_wave_axis_aligned"}
_AUXILIARY_PROVIDERS = {"plane_wave_ref_x_ez", "plane_wave_axis_aligned", "plane_wave_aux"}

_MAGNETIC_FIELD_NAMES = ("Hx", "Hy", "Hz")
_ELECTRIC_FIELD_NAMES = ("Ex", "Ey", "Ez")


def _field_names_for_terms(term_key):
    return _MAGNETIC_FIELD_NAMES if term_key == "magnetic_terms" else _ELECTRIC_FIELD_NAMES


def _apply_batched_aux_terms(solver, batch, *, sample_kind, field_names):
    if batch is None:
        return

    aux = solver._tfsf_state["auxiliary_grid"]
    sample_origin = float(aux.s_min if sample_kind == "electric" else aux.s_min + 0.5 * aux.ds)
    sample_field = aux.electric if sample_kind == "electric" else aux.magnetic
    solver.fdtd_module.addBatchedInterpolatedSourcePatches3D(
        fieldX=getattr(solver, field_names[0]),
        fieldY=getattr(solver, field_names[1]),
        fieldZ=getattr(solver, field_names[2]),
        coeffData=batch["coeff_data"],
        incidentField=sample_field,
        samplePositions=batch["sample_positions"],
        termStarts=batch["term_starts"],
        termShapes=batch["term_shapes"],
        termOffsets=batch["term_offsets"],
        fieldCodes=batch["field_codes"],
        fieldCodesPerCoeff=batch["field_codes_per_coeff"],
        fieldOffsets=batch["field_offsets"],
        origin=sample_origin,
        ds=float(aux.ds),
    ).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=batch["grid"],
    )


def _apply_batched_reference_terms(solver, batch, *, sample_kind, field_names):
    if batch is None:
        return

    aux = solver._tfsf_state["auxiliary_grid"]
    sample_field = aux.electric if sample_kind == "electric" else aux.magnetic
    solver.fdtd_module.addBatchedReferenceSourcePatches3D(
        fieldX=getattr(solver, field_names[0]),
        fieldY=getattr(solver, field_names[1]),
        fieldZ=getattr(solver, field_names[2]),
        coeffData=batch["coeff_data"],
        incidentField=sample_field,
        termStarts=batch["term_starts"],
        termShapes=batch["term_shapes"],
        termOffsets=batch["term_offsets"],
        fieldCodes=batch["field_codes"],
        sampleAxisCodes=batch["sample_axis_codes"],
        sampleIndexStarts=batch["sample_index_starts"],
        sampleIndices=batch["sample_indices"],
        fieldCodesPerCoeff=batch["field_codes_per_coeff"],
        fieldOffsets=batch["field_offsets"],
        sampleIndicesPerCoeff=batch["sample_indices_per_coeff"],
    ).launchRaw(
        blockSize=solver.kernel_block_size,
        gridSize=batch["grid"],
    )


def _apply_aux_terms(solver, terms, *, sample_kind):
    if not terms:
        return
    aux = solver._tfsf_state["auxiliary_grid"]
    sample_origin = float(aux.s_min if sample_kind == "electric" else aux.s_min + 0.5 * aux.ds)
    sample_field = aux.electric if sample_kind == "electric" else aux.magnetic
    for term in terms:
        offset_i, offset_j, offset_k = term["offsets"]
        solver.fdtd_module.addInterpolatedSourcePatch3D(
            field=getattr(solver, term["field_name"]),
            coeffPatch=term["coeff_patch"],
            incidentField=sample_field,
            samplePositions=term["sample_positions"],
            origin=sample_origin,
            ds=float(aux.ds),
            offsetI=int(offset_i),
            offsetJ=int(offset_j),
            offsetK=int(offset_k),
            scale=float(term["component_scale"]),
        ).launchRaw(
            blockSize=solver.kernel_block_size,
            gridSize=term["grid"],
        )


def _apply_reference_terms(solver, terms):
    if not terms:
        return

    aux = solver._tfsf_state["auxiliary_grid"]
    for term in terms:
        sample_field = aux.electric if term["sample_kind"] == "electric" else aux.magnetic
        if term["scalar_sample_index"] is not None:
            offset_i, offset_j, offset_k = term["offsets"]
            solver.fdtd_module.addScaledSliceSourcePatch3D(
                field=getattr(solver, term["field_name"]),
                sourcePatch=term["coeff_patch"],
                incidentField=sample_field,
                sampleIndex=int(term["scalar_sample_index"]),
                offsetI=int(offset_i),
                offsetJ=int(offset_j),
                offsetK=int(offset_k),
                scale=float(term["component_scale"]),
            ).launchRaw(
                blockSize=solver.kernel_block_size,
                gridSize=term["grid"],
            )
            continue
        sample_indices_i32 = term.get("_sample_indices_i32")
        if (
            sample_indices_i32 is None
            or not torch.is_tensor(sample_indices_i32)
            or sample_indices_i32.device != sample_field.device
        ):
            sample_indices_i32 = term["sample_indices"].to(device=sample_field.device, dtype=torch.int32)
            term["_sample_indices_i32"] = sample_indices_i32
        sample_axis_code = int(term.get("_sample_axis_code", reference_sample_axis_code(term)))
        term["_sample_axis_code"] = sample_axis_code
        offset_i, offset_j, offset_k = term["offsets"]
        solver.fdtd_module.addScaledLineSourcePatch3D(
            field=getattr(solver, term["field_name"]),
            coeffPatch=term["coeff_patch"],
            incidentField=sample_field,
            sampleIndices=sample_indices_i32,
            sampleAxisCode=sample_axis_code,
            offsetI=int(offset_i),
            offsetJ=int(offset_j),
            offsetK=int(offset_k),
            scale=float(term["component_scale"]),
        ).launchRaw(
            blockSize=solver.kernel_block_size,
            gridSize=term["grid"],
        )


def _apply_tfsf_terms(solver, *, term_key, sample_kind, time_value):
    if not getattr(solver, "tfsf_enabled", False):
        return

    provider = solver._tfsf_state.get("provider")
    terms = solver._tfsf_state[term_key]
    batch = solver._tfsf_state.get("magnetic_batch" if term_key == "magnetic_terms" else "electric_batch")
    field_names = _field_names_for_terms(term_key)
    if provider in _REFERENCE_PROVIDERS:
        if batch is not None:
            _apply_batched_reference_terms(
                solver,
                batch,
                sample_kind=sample_kind,
                field_names=field_names,
            )
            return
        _apply_reference_terms(solver, terms)
        return
    if provider == "plane_wave_aux":
        if batch is not None:
            _apply_batched_aux_terms(
                solver,
                batch,
                sample_kind=sample_kind,
                field_names=field_names,
            )
            return
        _apply_aux_terms(solver, terms, sample_kind=sample_kind)
        return
    apply_generic_source_terms(
        solver,
        terms,
        source_time=solver._source_time,
        omega=solver.source_omega,
        time_value=time_value,
        clamp_pec=False,
    )


# TFSF providers whose in-block correction both (1) reads the incident field from
# the device-resident auxiliary 1D grid -- carrying no per-step host input -- and
# (2) accumulates through the integer-indexed *reference* injection kernel, whose
# warp-aggregated writes are collision-free for the axis-aligned face layout and
# therefore bit-reproducible. The oblique auxiliary-line provider
# (``plane_wave_aux``) is deliberately excluded: it accumulates through the batched
# *interpolated* kernel, whose cross-warp atomic adds at the overlapping oblique
# faces are not bit-reproducible run to run, so a captured graph could not be
# proven bit-identical to the eager path.
_CAPTURABLE_TFSF_PROVIDERS = frozenset(_REFERENCE_PROVIDERS)


def tfsf_incident_is_gpu_driven(solver) -> bool:
    """True when an active TFSF source couples into the 3D field-update block only
    by *reading* the device-resident auxiliary 1D incident grid through the
    bit-reproducible reference injection kernel, so the block can be captured into a
    CUDA graph and replayed bit-for-bit.

    The reference providers apply their in-block H correction with the batched /
    per-term reference kernels, which read ``aux.electric`` / ``aux.magnetic``
    (persistent device tensors) at fixed integer sample indices and carry no
    per-step host input. The 1D source waveform is evaluated on the host only in
    ``advance_tfsf_auxiliary_electric`` (the E-side auxiliary advance), which runs
    eagerly *after* the field-update block, outside the captured region. So for
    these providers the block is a fixed, deterministic kernel sequence.

    The oblique auxiliary-line provider (``plane_wave_aux``) is excluded because its
    batched interpolated injection kernel accumulates through non-deterministic
    cross-warp atomic adds (bit-identical capture parity is impossible), and every
    remaining provider evaluates the waveform on the host each step via
    ``apply_generic_source_terms`` and injects it as a kernel launch scalar that a
    captured graph would freeze. All of these stay on the eager path.
    """
    state = getattr(solver, "_tfsf_state", None)
    if not state:
        return False
    return state.get("provider") in _CAPTURABLE_TFSF_PROVIDERS


def apply_tfsf_h_correction(solver, time_value):
    _apply_tfsf_terms(
        solver,
        term_key="magnetic_terms",
        sample_kind="electric",
        time_value=time_value,
    )


def apply_tfsf_e_correction(solver, time_value):
    _apply_tfsf_terms(
        solver,
        term_key="electric_terms",
        sample_kind="magnetic",
        time_value=time_value + 0.5 * float(solver.dt),
    )


def _advance_auxiliary_grid(solver, *, magnetic):
    if not getattr(solver, "tfsf_enabled", False):
        return None
    if solver._tfsf_state.get("provider") not in _AUXILIARY_PROVIDERS:
        return None
    aux = solver._tfsf_state["auxiliary_grid"]
    if magnetic:
        aux.advance_magnetic()
    else:
        aux.advance_electric()
    return None


def advance_tfsf_auxiliary_magnetic(solver):
    return _advance_auxiliary_grid(solver, magnetic=True)


def advance_tfsf_auxiliary_electric(solver):
    return _advance_auxiliary_grid(solver, magnetic=False)

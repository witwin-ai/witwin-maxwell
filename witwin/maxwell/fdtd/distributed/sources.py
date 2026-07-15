from __future__ import annotations

import torch

from ...fdtd_parallel import FDTDShardLayout


_CELL_COMPONENTS = frozenset(("Ex", "Hy", "Hz"))
_PAYLOAD_KEYS = (
    "patch",
    "delay_patch",
    "activation_delay_patch",
    "cw_cos_patch",
    "cw_sin_patch",
)
_TERM_LIST_NAMES = (
    "_source_terms",
    "_magnetic_source_terms",
    "_electric_source_terms",
)


def _first_payload(term: dict) -> torch.Tensor:
    for key in _PAYLOAD_KEYS:
        value = term.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError("A compiled FDTD source term has no tensor payload.")


def _crop_term_to_owned_x(term: dict, layout: FDTDShardLayout) -> dict | None:
    """Crop a shard-local source term to component values owned by this rank.

    Local scenes intentionally contain one low ghost cell on every nonzero rank.
    Their ordinary source compiler may therefore generate terms on receive ghosts or
    on the high electric ghost plane.  This final ownership crop makes all source
    samples single-writer while retaining the existing source compiler and kernels.
    """

    field_name = str(term["field_name"])
    owned = (
        layout.storage_cell_owned
        if field_name in _CELL_COMPONENTS
        else layout.storage_node_owned
    )
    payload = _first_payload(term)
    offset_x = int(term["offsets"][0])
    term_stop_x = offset_x + int(payload.shape[0])
    owned_begin = max(offset_x, int(owned.start))
    owned_end = min(term_stop_x, int(owned.stop))
    if owned_end <= owned_begin:
        return None

    local_x = slice(owned_begin - offset_x, owned_end - offset_x)
    cropped = dict(term)
    cropped["offsets"] = (owned_begin, *tuple(term["offsets"])[1:])
    for key in _PAYLOAD_KEYS:
        value = term.get(key)
        if isinstance(value, torch.Tensor):
            cropped[key] = value[local_x].contiguous()
    return cropped


def crop_solver_source_terms_to_owned_x(solver, layout: FDTDShardLayout) -> None:
    """Apply the immutable Yee ownership contract to every compiled source list."""

    for attribute in _TERM_LIST_NAMES:
        terms = tuple(getattr(solver, attribute, ()) or ())
        cropped = []
        for term in terms:
            owned_term = _crop_term_to_owned_x(term, layout)
            if owned_term is not None:
                cropped.append(owned_term)
        setattr(solver, attribute, cropped)


__all__ = ["crop_solver_source_terms_to_owned_x"]

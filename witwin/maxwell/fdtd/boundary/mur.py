"""Mur boundary state and its ordered discrete transpose."""
from __future__ import annotations


def mur_state_name(index, history):
    return f"mur_{index}_{history}"


def iter_mur_state_tensors(solver):
    for index, entry in enumerate(getattr(solver, "_mur_state", ())):
        for history in ("prev_boundary", "prev_adjacent"):
            yield mur_state_name(index, history), entry[history]


def replay_mur_boundaries(solver, fields, previous_state):
    """Apply the exact face order, including history copies at shared corners."""
    entries = getattr(solver, "_mur_state", ())
    if not entries:
        return fields
    output = dict(fields)
    for name in {entry["field"] for entry in entries}:
        output[name] = output[name].clone()
    for index, entry in enumerate(entries):
        field = output[entry["field"]]
        adjacent = field.select(entry["axis"], entry["adjacent_index"]).clone()
        boundary_name = mur_state_name(index, "prev_boundary")
        adjacent_name = mur_state_name(index, "prev_adjacent")
        boundary = previous_state[adjacent_name] + entry["coef"] * (adjacent - previous_state[boundary_name])
        field.select(entry["axis"], entry["boundary_index"]).copy_(boundary)
        output[boundary_name] = boundary
        output[adjacent_name] = adjacent
    return output


def pullback_mur_boundaries(solver, adjoint):
    """Transpose boundary writes before the Maxwell/source reverse step.

    Face histories capture values before later faces overwrite shared edges.
    Reverse the face order and combine field/history cotangents before clearing
    the overwritten plane. All operations remain on the field device.
    """
    entries = getattr(solver, "_mur_state", ())
    if not entries:
        return adjoint
    output = dict(adjoint)
    for name in {entry["field"] for entry in entries}:
        output[name] = output[name].clone()
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        field = output[entry["field"]]
        boundary_name = mur_state_name(index, "prev_boundary")
        adjacent_name = mur_state_name(index, "prev_adjacent")
        boundary_plane = field.select(entry["axis"], entry["boundary_index"])
        combined = boundary_plane + output[boundary_name]
        adjacent_gradient = entry["coef"] * combined + output[adjacent_name]
        boundary_plane.zero_()
        field.select(entry["axis"], entry["adjacent_index"]).add_(adjacent_gradient)
        output[boundary_name] = -entry["coef"] * combined
        output[adjacent_name] = combined
    return output

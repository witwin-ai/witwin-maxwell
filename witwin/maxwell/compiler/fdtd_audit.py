"""On-demand preparation diagnostics from actual compiled FDTD state."""

from __future__ import annotations

import math

import numpy as np


def compile_fdtd_audit(solver):
    """Return JSON-safe coordinates, constraints and cost estimates, without stepping."""
    from ..fdtd.runtime.materials import _pec_edge_open_fractions

    scene = solver.scene
    if not all(hasattr(scene, f"{axis}_nodes64") for axis in "xyz"):
        raise NotImplementedError("Preparation diagnostics require a single-device FDTD grid.")
    grids = {}
    for axis in "xyz":
        nodes = np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64)
        widths = np.diff(nodes)
        index = int(np.argmin(widths))
        grids[axis] = {"nodes": len(nodes), "minimum_spacing_m": float(widths[index]),
                       "maximum_spacing_m": float(widths.max()),
                       "maximum_adjacent_ratio": float(np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:]).max()) if len(widths) > 1 else 1.,
                       "limiting_interval_m": [float(nodes[index]), float(nodes[index + 1])],
                       "bounds_m": [float(nodes[0]), float(nodes[-1])]}
    inverse_square = {axis: grids[axis]["minimum_spacing_m"] ** -2 for axis in "xyz"}
    total = sum(inverse_square.values())
    for axis in "xyz":
        grids[axis]["vacuum_cfl_inverse_square_fraction"] = inverse_square[axis] / total
    layers = []
    from witwin.core import Box
    for structure in scene.structures:
        geometry = structure.geometry
        if not isinstance(geometry, Box) or bool(getattr(structure.material, "is_pec", False)) or bool(getattr(structure.material, "is_pmc", False)):
            continue
        rotation = geometry.rotation.detach().cpu().numpy()
        if not np.array_equal(np.abs(rotation), (1., 0., 0., 0.)):
            continue
        position = geometry.position.detach().cpu().double().numpy()
        size = geometry.size.detach().cpu().double().numpy()
        axis_index = int(np.argmin(size))
        if size[axis_index] <= 0:
            continue
        axis = "xyz"[axis_index]
        nodes = np.asarray(getattr(scene, axis + "_nodes64"))
        widths = np.diff(nodes)
        lo, hi = position[axis_index] - size[axis_index] / 2, position[axis_index] + size[axis_index] / 2
        overlaps = np.maximum(0., np.minimum(nodes[1:], hi) - np.maximum(nodes[:-1], lo))
        layers.append({"name": structure.name, "axis": axis, "physical_thickness_m": float(size[axis_index]),
                       "overlapping_intervals": int((overlaps > 0).sum()),
                       "effective_intervals": float((overlaps / widths).sum()),
                       "bounds_m": [float(lo), float(hi)]})
    fractions = _pec_edge_open_fractions(solver)
    constraints = {} if fractions is None else {
        name: {"closed_edges": int((value == 0).sum().item()),
               "partially_open_edges": int(((value > 0) & (value < 1)).sum().item())}
        for name, value in fractions.items()}
    ports = []
    for runtime in getattr(solver, "_port_runtimes", ()):
        geometry = runtime.geometry
        indices = getattr(geometry, "voltage_indices", None)
        weights = getattr(geometry, "voltage_weights", None)
        ports.append({"name": geometry.port_name,
                      "kind": "wire" if getattr(geometry, "is_wire_bound", False) else "yee",
                      "voltage_component": getattr(geometry, "voltage_component", None),
                      "voltage_indices": None if indices is None else indices.detach().cpu().tolist(),
                      "voltage_weights_m": None if weights is None else weights.detach().cpu().tolist()})
    model = getattr(solver, "_compiled_material_model", {})
    return {"schema": "fdtd-preparation-audit/1", "status": "prepared_execution_not_qualified",
            "grid": grids, "thin_dimension_resolution": layers, "dt_s": float(solver.dt),
            "estimated_steps_for_12ns": math.ceil(12e-9 / float(solver.dt)),
            "time_step": dict(getattr(solver, "_time_step_metadata", {})),
            "pec_constraints": constraints,
            "planar_pec": model.get("pec_sheet_audit"), "ports": ports,
            "conductive": bool(getattr(solver, "conductive_enabled", False)),
            "electric_dispersive": bool(getattr(solver, "electric_dispersive_enabled", False)),
            "qualification": "No convergence, matching or physical accuracy claim follows from preparation."}

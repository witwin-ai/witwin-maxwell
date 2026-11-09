"""GPU mass-orthogonal native electric constraints for true PEC cut surfaces."""
from __future__ import annotations

import torch

from ..compiler.cut_surface import compile_integrated_traces
from ..compiler.power_loss import _component_geometry
from ..geometry.surface_mesh import SurfaceMesh


class ElectricTraceProjection:
    """Rank-revealed positive-metric projection with persistent GPU work buffers."""

    def __init__(self, matrix, weight):
        if weight.ndim != 1 or not bool(torch.isfinite(weight).all() & (weight > 0).all()):
            raise ValueError("PEC electric trace metric must be finite and positive.")
        self.matrix = matrix.coalesce()
        self.sqrt_weight = weight.sqrt()
        weighted = self.matrix.to_dense()/self.sqrt_weight[None]
        _u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
        self.rank_tolerance = max(weighted.shape)*torch.finfo(weighted.dtype).eps*singular[0]
        keep = singular > self.rank_tolerance
        self.rank = int(keep.sum())
        if self.rank == 0:
            raise ValueError("PEC surface has no independent electric trace constraints.")
        # Orthonormal row-space evaluation avoids squaring an ill-conditioned
        # Gram spectrum in every step. Only declared supported electric edges
        # are gathered/scattered; the explicit row/memory cap bounds this basis.
        self.basis = vh[keep].contiguous()
        self.coordinates = weight.new_empty(self.rank)
        self.correction = weight.new_empty(len(weight))
        self.weighted_field = weight.new_empty(len(weight))

    def apply(self, electric):
        torch.mul(electric, self.sqrt_weight, out=self.weighted_field)
        torch.mv(self.basis, self.weighted_field, out=self.coordinates)
        torch.mv(self.basis.T, self.coordinates, out=self.correction)
        self.weighted_field.sub_(self.correction)
        torch.div(self.weighted_field, self.sqrt_weight, out=electric)
        return electric


def _source_support_overlap(solver, groups):
    tfsf = getattr(solver, "_tfsf_state", None) or {}
    if tfsf:
        axes = ("xyz".index(tfsf["axis"]),) if tfsf.get("mode") == "slab" else range(3)
        for group in groups.values():
            if any(bool(((group["indices"][:, axis] <= tfsf["lower"][axis]) |
                         (group["indices"][:, axis] >= tfsf["upper"][axis])).any()) for axis in axes):
                raise NotImplementedError("PEC cut-surface support must lie strictly inside TFSF faces; joint source/constraint elimination is required.")
    for term in (*getattr(solver, "_source_terms", ()), *getattr(solver, "_electric_source_terms", ()),
                 *tfsf.get("electric_terms", ())):
        name = term["field_name"].removesuffix("_imag")
        if name not in groups:
            continue
        indices = groups[name]["indices"]
        for key in ("patch", "cw_cos_patch", "cw_sin_patch", "coeff_patch"):
            patch = term.get(key)
            if patch is None:
                continue
            local = indices-torch.tensor(term["offsets"], device=indices.device)
            inside = ((local >= 0) & (local < torch.tensor(patch.shape, device=indices.device))).all(-1)
            if bool(inside.any()) and bool((patch[tuple(local[inside].T)] != 0).any()):
                raise NotImplementedError("An electric source overlaps the PEC cut-surface interpolation support; joint source/constraint elimination is required.")
    supports = {name: group["linear_indices"] for name, group in groups.items()}
    for port in getattr(solver, "_port_runtimes", ()):
        if port.field_name not in supports:
            continue
        geometry = port.geometry
        shape = getattr(solver, port.field_name).shape
        indices = geometry.voltage_indices
        flat = (indices[:, 0]*shape[1]+indices[:, 1])*shape[2]+indices[:, 2]
        if bool(torch.isin(flat, supports[port.field_name]).any()):
            raise NotImplementedError("A port overlaps the PEC cut-surface interpolation support; joint port/constraint elimination is required.")
    for runtime, name in getattr(solver, "_lumped_element_runtimes", ()):
        if name in supports and bool(torch.isin(runtime.linear_indices, supports[name]).any()):
            raise NotImplementedError("A lumped element overlaps the PEC cut-surface interpolation support.")


def initialize_cut_surfaces(solver):
    solver._cut_surface = None
    from ..compiler.thin_sheet import owns_thin_sheet
    structures = [structure for structure in solver.scene.structures
                  if getattr(structure, "enabled", True) and isinstance(structure.geometry, SurfaceMesh)
                  and not owns_thin_sheet(structure)]
    if not structures:
        return
    if solver.device.type != "cuda":
        raise ValueError("Native cut-surface FDTD requires a CUDA device.")
    from ..mom.mesh import _has_intersections
    compiled = []
    nodes = tuple(torch.as_tensor(getattr(solver.scene, axis+"_nodes64"), device=solver.device, dtype=torch.float64) for axis in "xyz")
    for structure in structures:
        if not bool(getattr(structure.material, "is_pec", False)):
            raise NotImplementedError("SurfaceMesh requires PEC; finite copper needs independent two-sided cut traces.")
        geometry = structure.geometry
        vertices, triangles = geometry.to_mesh()
        vertices, triangles = vertices.to(solver.device), triangles.to(solver.device)
        if _has_intersections(vertices[triangles]):
            raise ValueError("SurfaceMesh contains self-intersections or nonconforming contacts.")
        if geometry.planar_pec(solver.scene) is not None:
            continue
        compiled.append(compile_integrated_traces(vertices, triangles, nodes))
    if not compiled:
        return
    for i, entry in enumerate(compiled):
        if any(bool(torch.isin(entry["cells"], other["cells"]).any()) for other in compiled[:i]):
            raise ValueError("Separate PEC surfaces share a Yee cut cell; refine the physical grid.")
    if any(bool(getattr(solver, name, False)) for name in ("complex_fields_enabled", "full_aniso_enabled",
            "nonlinear_enabled", "modulation_enabled", "gyromagnetic_enabled", "breakdown_enabled",
            "dispersive_enabled", "magnetic_dispersive_enabled")):
        raise NotImplementedError("PEC cut-surface projection currently requires real isotropic linear fields and disjoint soft sources.")
    if getattr(solver, "_wire_runtime", None) is not None:
        raise NotImplementedError("PEC cut-surface wire contact requires a joint connectivity discretization.")
    rows, columns, values, offset = [], [], [], 0
    total_dofs = sum(getattr(solver, name).numel() for name in ("Ex", "Ey", "Ez"))
    for entry in compiled:
        matrix = entry["matrix"]
        rows.append(matrix.indices()[0]+offset)
        columns.append(entry["support"][matrix.indices()[1]])
        values.append(matrix.values())
        offset += matrix.shape[0]
    if offset > 4096:
        raise NotImplementedError("Combined cut-surface constraints exceed the explicit 4096 row cap.")
    global_matrix = torch.sparse_coo_tensor(torch.stack((torch.cat(rows), torch.cat(columns))), torch.cat(values),
                                           (offset, total_dofs)).coalesce()
    support, inverse = torch.unique(global_matrix.indices()[1], sorted=True, return_inverse=True)
    if offset*len(support)*8 > 512*1024**2:
        raise NotImplementedError("Combined cut-surface rank-reveal matrix exceeds the explicit 512 MiB cap.")
    matrix = torch.sparse_coo_tensor(torch.stack((global_matrix.indices()[0], inverse)), global_matrix.values(),
                                     (offset, len(support))).coalesce()
    weights, groups, field_offset = [], {}, 0
    for name in ("Ex", "Ey", "Ez"):
        field = getattr(solver, name)
        positions = torch.nonzero((support >= field_offset) & (support < field_offset+field.numel())).flatten()
        linear = support[positions]-field_offset
        indices = torch.stack((linear//(field.shape[1]*field.shape[2]),
                               linear//field.shape[2] % field.shape[1], linear % field.shape[2]), -1)
        if len(positions):
            # Absorber and fixed outer-face electric values are not part of an
            # interior PEC constraint's positive physical mass.
            for axis, kind in enumerate("xyz"):
                if bool(((indices[:, axis] == 0) | (indices[:, axis] == field.shape[axis]-1)).any()):
                    raise NotImplementedError("PEC cut-surface interpolation support touches a domain boundary; move it inside the physical grid.")
            if solver.uses_cpml:
                for axis, coordinate in enumerate("xyz"):
                    profile = getattr(solver, "cpml_c_e_"+coordinate)
                    if bool((profile[indices[:, axis]] != 0).any()):
                        raise NotImplementedError("PEC cut-surface interpolation support intersects a CPML layer.")
            _, widths = _component_geometry(solver.scene, name)
            volume = torch.ones(len(linear), dtype=torch.float64, device=solver.device)
            for axis in range(3):
                volume *= torch.as_tensor(widths[axis], device=solver.device)[indices[:, axis]]
            epsilon = getattr(solver, "eps_"+name).flatten()[linear].double()
            sigma = getattr(solver, "sigma_e_"+name).flatten()[linear].double()
            curl = getattr(solver, "c"+name.lower()+"_curl").flatten()[linear].double()
            expected = solver.dt/(epsilon+sigma*solver.dt/2)
            if not bool(torch.allclose(curl, expected, rtol=2e-6, atol=0)):
                raise NotImplementedError("PEC cut-surface interpolation support overlaps PEC, PML or another suppressed electric update.")
            weights.append((epsilon+sigma*solver.dt/2)*volume)
            groups[name] = dict(indices=indices, positions=positions, linear_indices=linear,
                                buffer=torch.empty(len(linear), device=solver.device, dtype=field.dtype),
                                double_buffer=torch.empty(len(linear), device=solver.device, dtype=torch.float64))
        field_offset += field.numel()
    _source_support_overlap(solver, groups)
    for entry in getattr(solver, "_copper_foils", ()):
        name = entry["electric_name"]
        if name in groups:
            shape = getattr(solver, name).shape
            indices = entry["indices"]
            flat = (indices[:, 0]*shape[1]+indices[:, 1])*shape[2]+indices[:, 2]
            if bool(torch.isin(flat, groups[name]["linear_indices"]).any()):
                raise NotImplementedError("PEC cut-surface interpolation overlaps a two-sided copper trace.")
    weight = torch.cat(weights)
    solver._cut_surface = dict(projection=ElectricTraceProjection(matrix, weight), groups=groups,
        electric=torch.empty_like(weight), physical_area=sum(entry["area"] for entry in compiled),
        constraint_rows=offset, surface_count=len(compiled), support_count=len(support))


def apply_cut_surface_constraints(solver):
    runtime = getattr(solver, "_cut_surface", None)
    if runtime is None:
        return
    electric = runtime["electric"]
    for name, group in runtime["groups"].items():
        torch.index_select(getattr(solver, name).reshape(-1), 0, group["linear_indices"], out=group["buffer"])
        group["double_buffer"].copy_(group["buffer"])
        electric.index_copy_(0, group["positions"], group["double_buffer"])
    runtime["projection"].apply(electric)
    for name, group in runtime["groups"].items():
        field = getattr(solver, name)
        torch.index_select(electric, 0, group["positions"], out=group["double_buffer"])
        group["buffer"].copy_(group["double_buffer"])
        field.reshape(-1).index_copy_(0, group["linear_indices"], group["buffer"])

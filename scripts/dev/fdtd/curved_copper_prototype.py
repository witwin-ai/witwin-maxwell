"""Validation-only failed unsplit-field curved copper prototype.

This module never enables public SurfaceMesh-Cu preparation. Explicit diagnostic
contexts temporarily attach the rejected method to a native blank bulk scene.
Plane and spherical-shell failures are acceptance evidence, not solver support.
"""
from contextlib import contextmanager
import hashlib
from unittest.mock import patch
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.curved_copper import compile_copper_yee_response
from witwin.maxwell.compiler.cut_surface import compile_integrated_traces
from witwin.maxwell.copper_foil import CopperFoil
from witwin.maxwell.geometry.surface_mesh import SurfaceMesh
from witwin.maxwell.fdtd.curved_copper import JointCopperMortar


@contextmanager
def prototype_hooks():
    """Temporarily exercise rejected coupling without changing production hooks."""
    from witwin.maxwell.fdtd.runtime import stepping
    from witwin.maxwell.fdtd import checkpoint, resume
    begin, electric, advance = stepping.begin_copper_step, stepping.update_electric_fields, stepping.advance_copper_electric
    states, fingerprint = checkpoint.iter_surface_impedance_state_specs, resume._configuration_fingerprint
    def begin_hook(solver):
        begin(solver)
        begin_curved_copper_step(solver)
    def electric_hook(solver, *args, **kwargs):
        correct_curved_copper_magnetic(solver)
        return electric(solver, *args, **kwargs)
    def advance_hook(solver):
        advance(solver)
        advance_curved_copper_electric(solver)
    def state_hook(solver):
        yield from states(solver)
        yield from iter_curved_copper_state(solver)
    def fingerprint_hook(solver):
        hasher = hashlib.sha256(fingerprint(solver).encode())
        for index, entry in enumerate(getattr(solver, "_curved_copper", ())):
            structure = entry.get("structure")
            if structure is None:
                continue
            for name in ("vertices", "triangles", "position", "rotation", "physical_thickness"):
                resume._hash_value(hasher, f"prototype[{index}].geometry.{name}", getattr(structure.geometry, name))
            for name in ("thickness", "conductivity", "frequency_range", "mode_count"):
                resume._hash_value(hasher, f"prototype[{index}].material.{name}", getattr(structure.material, name))
        return hasher.hexdigest()
    with patch.object(stepping,"begin_copper_step",begin_hook), patch.object(stepping,"update_electric_fields",electric_hook), \
         patch.object(stepping,"advance_copper_electric",advance_hook), patch.object(checkpoint,"iter_surface_impedance_state_specs",state_hook), \
         patch.object(resume,"_configuration_fingerprint",fingerprint_hook):
        yield


class _PreparedPrototype:
    def __init__(self, prepared, scene):
        self._prepared, self._scene = prepared, scene
        self.solver = prepared.solver

    def run(self, **kwargs):
        result = self._prepared.run(**kwargs)
        # Region classification in this private diagnostic uses the intended
        # physical shell; the native bulk preparation intentionally excludes it.
        result.scene = self._scene
        return result

    def run_until(self, step):
        return self._prepared.run_until(step)


class PrototypeSimulation:
    """Private validation adapter, deliberately outside the public solver API."""
    def __init__(self, scene, **kwargs):
        self.scene, self.kwargs = scene, kwargs

    def prepare(self):
        bulk = self.scene.clone(structures=tuple(structure for structure in self.scene.structures
            if not (isinstance(structure.geometry,SurfaceMesh) and isinstance(structure.material,CopperFoil))))
        # The hooks extend the existing runner after preparation; the core path would
        # ignore them and an automatic-setup resume rebuild would drop them.
        prepared = mw.Simulation.fdtd(bulk,fdtd_core_path=False,accelerated_setup=False,**self.kwargs).prepare()
        initialize_curved_copper(prepared.solver,self.scene.structures)
        return _PreparedPrototype(prepared,self.scene)

    def run(self):
        return self.prepare().run()

def _support_groups(solver, support, *, magnetic):
    groups, masses, conductivities, offset = {}, [], [], 0
    from witwin.maxwell.compiler.power_loss import _component_geometry
    for component, name in enumerate(("Hx", "Hy", "Hz") if magnetic else ("Ex", "Ey", "Ez")):
        field = getattr(solver, name)
        positions = torch.nonzero((support >= offset) & (support < offset+field.numel())).flatten()
        linear = support[positions]-offset
        indices = torch.stack((linear//(field.shape[1]*field.shape[2]),
            linear//field.shape[2]%field.shape[1], linear%field.shape[2]), -1)
        offset += field.numel()
        if not len(linear):
            continue
        for axis in range(3):
            if bool(((indices[:, axis] == 0) | (indices[:, axis] == field.shape[axis]-1)).any()):
                raise NotImplementedError("Curved copper interpolation support must be strictly inside the domain.")
            if solver.uses_cpml:
                profile = getattr(solver, f"cpml_c_{'h' if magnetic else 'e'}_"+"xyz"[axis])
                if bool((profile[indices[:, axis]] != 0).any()):
                    raise NotImplementedError("Curved copper interpolation support intersects CPML.")
        if magnetic:
            widths = tuple(getattr(solver.scene, coordinate+("_dual64" if axis == component else "_primal64"))
                           for axis, coordinate in enumerate(("dx", "dy", "dz")))
        else:
            _, widths = _component_geometry(solver.scene, name)
        volume = torch.ones(len(linear), dtype=torch.float64, device=solver.device)
        for axis in range(3):
            volume *= torch.as_tensor(widths[axis], device=solver.device)[indices[:, axis]]
        material = getattr(solver, ("mu_" if magnetic else "eps_")+name).flatten()[linear].double()
        sigma = getattr(solver, ("sigma_m_" if magnetic else "sigma_e_")+name).flatten()[linear].double()
        expected = solver.dt/(material+sigma*solver.dt/2)
        actual = getattr(solver, "c"+name.lower()+"_curl").flatten()[linear].double()
        if not bool(torch.allclose(actual, expected, rtol=2e-6, atol=0)):
            raise NotImplementedError("Curved copper support intersects PEC or another suppressed native update.")
        masses.append(material*volume)
        conductivities.append(sigma*volume)
        groups[name] = dict(positions=positions, linear_indices=linear, indices=indices,
            buffer=torch.empty(len(linear), dtype=field.dtype, device=solver.device),
            double_buffer=torch.empty(len(linear), dtype=torch.float64, device=solver.device))
    return groups, torch.cat(masses), torch.cat(conductivities)


def _gather(solver, groups, vector):
    for name, group in groups.items():
        torch.index_select(getattr(solver, name).flatten(), 0, group["linear_indices"], out=group["buffer"])
        group["double_buffer"].copy_(group["buffer"])
        vector.index_copy_(0, group["positions"], group["double_buffer"])


def _scatter(solver, groups, vector):
    for name, group in groups.items():
        torch.index_select(vector, 0, group["positions"], out=group["double_buffer"])
        group["buffer"].copy_(group["double_buffer"])
        getattr(solver, name).flatten().index_copy_(0, group["linear_indices"], group["buffer"])


def initialize_curved_copper(solver, structures):
    solver._curved_copper = []
    structures = [structure for structure in structures if getattr(structure, "enabled", True)
                  and isinstance(structure.geometry, SurfaceMesh) and isinstance(structure.material, CopperFoil)]
    if not structures:
        return
    if solver.device.type != "cuda":
        raise ValueError("Curved copper requires native CUDA fields.")
    for name in ("complex_fields_enabled", "full_aniso_enabled", "nonlinear_enabled", "modulation_enabled",
                 "gyromagnetic_enabled", "breakdown_enabled", "dispersive_enabled", "magnetic_dispersive_enabled"):
        if getattr(solver, name, False):
            raise NotImplementedError(f"Joint curved copper does not yet include {name}.")
    if getattr(solver, "_wire_runtime", None) is not None:
        raise NotImplementedError("Curved copper wire contact requires joint current connectivity.")
    from witwin.maxwell.mom.mesh import _has_intersections
    from witwin.maxwell.fdtd.cut_surface import _source_support_overlap
    nodes = tuple(torch.as_tensor(getattr(solver.scene, axis+"_nodes64"), device=solver.device, dtype=torch.float64) for axis in "xyz")
    for structure in structures:
        vertices, triangles = structure.geometry.to_mesh()
        vertices, triangles = vertices.to(solver.device), triangles.to(solver.device)
        if _has_intersections(vertices[triangles]):
            raise ValueError("Curved copper contains self-intersections or nonconforming contacts.")
        integrated = compile_integrated_traces(vertices, triangles, nodes, include_magnetic=True)
        if 2*integrated["constraint_count"] > 4096:
            raise NotImplementedError("The coupled curved copper response exceeds the explicit 4096 row cap.")
        response = compile_copper_yee_response(integrated, nodes)
        e_groups, me, se = _support_groups(solver, response["electric_support"], magnetic=False)
        h_groups, mh, sh = _support_groups(solver, response["magnetic_support"], magnetic=True)
        _source_support_overlap(solver, e_groups)
        previous_groups = [entry["electric_groups"] for entry in solver._curved_copper]
        pec = getattr(solver, "_cut_surface", None)
        if pec is not None:
            previous_groups.append(pec["groups"])
        for groups in previous_groups:
            if any(name in groups and bool(torch.isin(group["linear_indices"], groups[name]["linear_indices"]).any())
                   for name, group in e_groups.items()):
                raise NotImplementedError("Nearby curved interfaces require a joint multi-surface mortar block.")
        for planar in getattr(solver, "_copper_foils", ()):
            name = planar["electric_name"]
            if name in e_groups:
                shape = getattr(solver, name).shape
                index = planar["indices"]
                flat = (index[:, 0]*shape[1]+index[:, 1])*shape[2]+index[:, 2]
                if bool(torch.isin(flat, e_groups[name]["linear_indices"]).any()):
                    raise NotImplementedError("Planar and curved copper contact requires a joint connectivity solve.")
        runtime = JointCopperMortar(me, mh, se, sh, response["curl_transpose"],
            response["electric_trace"].to_dense(), response["magnetic_trace"].to_dense(),
            integrated["surface_mass"], structure.material, solver.dt)
        solver._curved_copper.append(dict(name=str(structure.name), runtime=runtime, structure=structure,
            electric_groups=e_groups, magnetic_groups=h_groups, integrated=integrated,
            electric=torch.empty_like(me), magnetic=torch.empty_like(mh)))


def begin_curved_copper_step(solver):
    for entry in getattr(solver, "_curved_copper", ()):
        _gather(solver, entry["electric_groups"], entry["electric"])
        _gather(solver, entry["magnetic_groups"], entry["magnetic"])
        entry["runtime"].begin(entry["electric"], entry["magnetic"])


def correct_curved_copper_magnetic(solver):
    for entry in getattr(solver, "_curved_copper", ()):
        _gather(solver, entry["magnetic_groups"], entry["magnetic"])
        entry["runtime"].correct_magnetic_predictor(entry["magnetic"])
        _scatter(solver, entry["magnetic_groups"], entry["magnetic"])


def advance_curved_copper_electric(solver):
    for entry in getattr(solver, "_curved_copper", ()):
        _gather(solver, entry["electric_groups"], entry["electric"])
        _gather(solver, entry["magnetic_groups"], entry["magnetic"])
        entry["runtime"].advance(entry["electric"], entry["magnetic"])
        _scatter(solver, entry["electric_groups"], entry["electric"])
        _scatter(solver, entry["magnetic_groups"], entry["magnetic"])


def iter_curved_copper_state(solver):
    for index, entry in enumerate(getattr(solver, "_curved_copper", ())):
        for name in ("ze", "zm", "cumulative_loss", "last_electric_effort", "last_magnetic_effort",
                     "last_electric_current", "last_magnetic_current"):
            yield f"curved_copper_{index}_{name}", getattr(entry["runtime"], name)


@torch.no_grad()
def curved_copper_loss_report(solver):
    result = {}
    for entry in getattr(solver, "_curved_copper", ()):
        runtime = entry["runtime"]
        _gather(solver, entry["magnetic_groups"], entry["magnetic"])
        trace = runtime.gh@entry["magnetic"]
        tail = runtime.tail*(trace@torch.cholesky_solve(trace[:, None], runtime.surface_cholesky)[:, 0])/2
        result[entry["name"]] = dict(dissipated_energy=float(runtime.cumulative_loss),
            stored_energy=float(runtime.modal_storage()+tail),
            electric_modal_energy=float(torch.einsum("ni,ij,nj->", runtime.ze, runtime.mass, runtime.ze)/2),
            magnetic_modal_energy=float(torch.einsum("ni,ij,nj->", runtime.zm, runtime.mass, runtime.zm)/2),
            magnetic_tail_energy=float(tail), frequency_range=runtime.channel.frequency_range,
            certified_relative_model_error_bound=runtime.channel.relative_error_bound,
            surface_current_dof_count=len(runtime.mass), interface_kind="curved_two_mortar")
    return result

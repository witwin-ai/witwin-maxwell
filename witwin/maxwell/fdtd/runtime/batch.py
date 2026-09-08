"""Persistent native field storage for cooperative FDTD execution."""
from __future__ import annotations

import torch

from ..cuda.backend import get_compiled_extension
from .stepping import cpml_layout_params

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
COEFFICIENTS = tuple(f"c{f.lower()}_{kind}" for f in FIELDS for kind in ("decay", "curl"))
SPACING = tuple(f"inv_d{a}_{kind}" for kind in ("e", "h") for a in "xyz")
CPML_PROFILES = tuple(f"cpml_{p}_{kind}_{a}" for p in ("inv_kappa", "b", "c")
                      for kind in ("e", "h") for a in "xyz")
MATERIAL_ARRAYS = ("epsilon_r", "mu_r", "material_eps_r", "material_mu_r") + tuple(
    f"{prefix}_{component}" for prefix, components in (
        ("eps", FIELDS[:3]), ("sigma_e", FIELDS[:3]),
        ("mu", FIELDS[3:]), ("sigma_m", FIELDS[3:])) for component in components)


def _share_material_tree(value, base, aliases):
    if value is base:
        return value
    if isinstance(value, torch.Tensor) and isinstance(base, torch.Tensor):
        if id(value) in aliases:
            return aliases[id(value)]
        if (not value.requires_grad and not base.requires_grad
                and value.dtype == base.dtype and value.device == base.device
                and value.shape == base.shape and torch.equal(value, base)):
            aliases[id(value)] = base
            return base
    elif isinstance(value, dict) and isinstance(base, dict):
        for name in value.keys() & base.keys():
            value[name] = _share_material_tree(value[name], base[name], aliases)
    return value


def _share_static_materials(solvers):
    """Deduplicate proven equal read-only material arrays, never evolving state."""
    reference = solvers[0]
    for solver in solvers[1:]:
        aliases = {}
        _share_material_tree(solver._compiled_material_model, reference._compiled_material_model, aliases)
        # Direct runtime aliases must follow the replacements inside the model;
        # the PreparedScene cache already owns that same model dictionary.
        for name in MATERIAL_ARRAYS:
            setattr(solver, name, _share_material_tree(getattr(solver, name), getattr(reference, name), aliases))


def _psi_layout(solver, name):
    if solver._cpml_memory_mode == "dense":
        axis = "xyz".index(name[-1])
        return (getattr(solver, name).shape[axis], 0, 0)
    return cpml_layout_params(solver, name)


def validate_native_solver(solver):
    """Fail before advancing when the shared field block omits active physics."""
    unsupported = (
        "complex_fields_enabled", "dispersive_enabled", "electric_dispersive_enabled",
        "magnetic_dispersive_enabled", "nonlinear_enabled", "modulation_enabled",
        "full_aniso_enabled", "gyromagnetic_enabled", "breakdown_enabled", "tfsf_enabled",
    )
    for name in unsupported:
        if getattr(solver, name, False):
            raise NotImplementedError(f"Native batch field execution does not yet support {name}.")
    for name in ("_magnetic_source_terms", "_electric_source_terms", "_wire_runtime",
                 "_surface_impedance", "_circuit_runtimes", "_network_runtimes", "_mur_state"):
        if getattr(solver, name, None):
            raise NotImplementedError(f"Native batch field execution does not yet support {name}.")
    for name in FIELDS:
        value = getattr(solver, name)
        if value.dtype != torch.float32 or value.ndim != 3 or min(value.shape) < 2:
            raise NotImplementedError("Native batch requires real float32 three-dimensional Yee fields.")
    for axis in "xyz":
        for face in ("low", "high"):
            if getattr(solver, f"boundary_{axis}_{face}_code") not in (0, 1, 2, 4, 5):
                raise NotImplementedError("Native batch supports none, CPML, periodic, PEC and PMC boundaries.")


class NativeFieldSlot:
    """One capacity-B allocation, reusable across compatible compiled cohorts.

    Coefficients share storage only after compiled value equality across lanes.
    A later cohort can promote a coefficient to dense lane storage. Grid and
    CPML profiles share storage only after compiled tensor equality checks.
    """

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffers = {}
        self.initial = {}
        self.shared = {}
        self.solvers = ()
        self.active = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.native = get_compiled_extension()
        self._signature = None

    def bind(self, solvers):
        if not 0 < len(solvers) <= self.capacity:
            raise ValueError("A field slot requires between one and capacity compiled lanes.")
        for solver in solvers:
            validate_native_solver(solver)
        first = solvers[0]
        psi_names = tuple(sorted(name for name, value in vars(first).items()
                                 if name.startswith("psi_") and isinstance(value, torch.Tensor)))
        shared_names = SPACING + (CPML_PROFILES if first.uses_cpml else ())
        signatures = []
        for solver in solvers:
            signatures.append((
                float(solver.dt), bool(solver.uses_cpml),
                tuple((name, tuple(getattr(solver, name).shape)) for name in FIELDS + psi_names),
                tuple(getattr(solver, f"boundary_{a}_{f}_code") for a in "xyz" for f in ("low", "high")),
                tuple((name, _psi_layout(solver, name)) for name in psi_names),
            ))
        if any(value != signatures[0] for value in signatures[1:]):
            raise ValueError("Compiled batch lanes differ in dt, Yee shapes, boundaries or CPML layout.")
        # A changed group needs a different allocation; never reinterpret old storage.
        if self._signature is not None and signatures[0] != self._signature:
            raise ValueError("Compiled cohort does not fit this persistent field slot.")
        for name in shared_names:
            reference = getattr(first, name)
            if any(not torch.equal(reference, getattr(s, name)) for s in solvers[1:]):
                raise ValueError(f"Compiled batch lanes differ in {name}.")
            if name in self.shared:
                if not torch.equal(self.shared[name], reference):
                    raise ValueError(f"Compiled cohort changed the shared {name} values.")
            else:
                self.shared[name] = reference
            for solver in solvers:
                setattr(solver, name, self.shared[name])
        self._signature = signatures[0]
        self.initial.clear()
        self.state_names = FIELDS + psi_names
        for name in self.state_names + COEFFICIENTS:
            values = [getattr(solver, name) for solver in solvers]
            shared_coefficient = name in COEFFICIENTS and (
                name not in self.buffers or self.buffers[name].ndim == 3) and all(
                    torch.equal(values[0], value) for value in values[1:])
            shape = tuple(values[0].shape) if shared_coefficient else (self.capacity, *values[0].shape)
            if name not in self.buffers:
                self.buffers[name] = torch.empty(shape, dtype=values[0].dtype, device=self.device)
            target = self.buffers[name]
            if tuple(target.shape) != shape:
                if name in COEFFICIENTS and target.ndim == 3 and not shared_coefficient:
                    # A later cohort may invalidate a sharing proof. Promote
                    # once to per-lane storage; never demote or alias states.
                    target = torch.empty(shape, dtype=values[0].dtype, device=self.device)
                    self.buffers[name] = target
                else:
                    raise ValueError(f"Compiled cohort changed {name} shape.")
            if shared_coefficient:
                target.copy_(values[0])
                for solver in solvers:
                    setattr(solver, name, target)
                continue
            # Copy before rebinding, including when reloading a previous cohort.
            for lane, value in enumerate(values):
                target[lane].copy_(value)
            if len(solvers) < self.capacity:
                target[len(solvers):].zero_()
            if name in self.state_names and bool(torch.count_nonzero(target)):
                self.initial[name] = target.clone()
            for lane, solver in enumerate(solvers):
                setattr(solver, name, target[lane])
        self.solvers = tuple(solvers)
        _share_static_materials(self.solvers)
        self.active.zero_()
        self.active[:len(solvers)].fill_(1)
        self._bind_operations()

    def reset(self):
        for name in self.state_names:
            if name in self.initial:
                self.buffers[name].copy_(self.initial[name])
            else:
                self.buffers[name].zero_()
        self.active.zero_()
        self.active[:len(self.solvers)].fill_(1)

    def advance(self):
        for operation, args in self._operations:
            operation(*args)

    def _bind_operations(self):
        self._operations = []
        reference = self.solvers[0]
        for kind in ("h", "e"):
            for component, axis in enumerate("xyz"):
                others = [a for a in "xyz" if a != axis]
                field_name = kind.upper() + axis
                source_kind = "E" if kind == "h" else "H"
                axis_a = "y" if component == 0 else "x"
                axis_b = "y" if component == 2 else "z"
                args = [self.buffers[field_name],
                        self.buffers[source_kind + others[0]], self.buffers[source_kind + others[1]],
                        self.buffers[f"c{kind}{axis}_decay"], self.buffers[f"c{kind}{axis}_curl"],
                        self.shared[f"inv_d{axis_a}_{kind}"], self.shared[f"inv_d{axis_b}_{kind}"],
                        self.active, component]
                if reference.uses_cpml:
                    psi_a, psi_b = (f"psi_{kind}{axis}_{a}" for a in (axis_a, axis_b))
                    args.extend((self.buffers[psi_a], self.buffers[psi_b]))
                    args.extend(self.shared[f"cpml_{p}_{kind}_{a}"]
                                for a in (axis_a, axis_b) for p in ("inv_kappa", "b", "c"))
                    args.extend(_psi_layout(reference, psi_a))
                    args.extend(_psi_layout(reference, psi_b))
                    operation = (self.native.update_magnetic_cpml_batch if kind == "h"
                                 else self.native.update_electric_cpml_batch)
                else:
                    operation = self.native.update_magnetic_batch if kind == "h" else self.native.update_electric_batch
                if kind == "e":
                    args.extend(getattr(reference, f"boundary_{a}_{f}_code")
                                for a in (axis_a, axis_b) for f in ("low", "high"))
                if reference.uses_cpml:
                    args.append(reference._cpml_memory_mode == "dense")
                self._operations.append((operation, tuple(args)))

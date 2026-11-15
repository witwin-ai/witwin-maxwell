"""Persistent native field storage for cooperative FDTD execution."""
from __future__ import annotations

import torch

from ..cuda.backend import get_compiled_extension
from .stepping import cpml_layout_params
from ...errors import UnsupportedFeatureError

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
    if getattr(solver.scene.scene,"coordinates",None) is not None:
        raise UnsupportedFeatureError("Mapped native batching requires coordinate-aware cohort and output qualification.",
                                      feature="coordinates", method="fdtd")
    unsupported = (
        "complex_fields_enabled", "nonlinear_enabled", "modulation_enabled",
        "full_aniso_enabled", "gyromagnetic_enabled", "breakdown_enabled", "tfsf_enabled",
    )
    for name in unsupported:
        if getattr(solver, name, False):
            raise UnsupportedFeatureError(f"Native batch field execution does not yet support {name}.",
                                          feature="guard:1419545a", method="fdtd")
    for name in ("_magnetic_source_terms", "_electric_source_terms", "_wire_runtime",
                 "_surface_impedance", "_mur_state", "_abc_higdon_state", "_thin_sheets"):
        if getattr(solver, name, None):
            raise UnsupportedFeatureError(f"Native batch field execution does not yet support {name}.",
                                          feature="guard:c0d7cc25", method="fdtd")
    for name in FIELDS:
        value = getattr(solver, name)
        if value.dtype != torch.float32 or value.ndim != 3 or min(value.shape) < 2:
            raise UnsupportedFeatureError("Native batch requires real float32 three-dimensional Yee fields.",
                                          feature="guard:5aecb3bf", method="fdtd")
    for axis in "xyz":
        for face in ("low", "high"):
            if getattr(solver, f"boundary_{axis}_{face}_code") not in (0, 1, 2, 4, 5):
                raise UnsupportedFeatureError("Native batch supports none, CPML, periodic, PEC and PMC boundaries.",
                                              feature="guard:c93c7a37", method="fdtd")


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
        field_rebindings = {}
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
                if name in FIELDS:
                    field_rebindings[id(values[lane])] = target[lane]
                setattr(solver, name, target[lane])
        for solver in solvers:
            for runtime in getattr(solver, "_network_runtimes", ()):
                runtime.electric_fields = tuple(field_rebindings.get(id(value), value) for value in runtime.electric_fields)
                for group in runtime.terminal_groups:
                    group.electric_field = field_rebindings.get(id(group.electric_field), group.electric_field)
        self.solvers = tuple(solvers)
        from ..copper_foil import iter_copper_state
        self._foil_lanes = tuple((solver, self.active[lane]) for lane, solver in enumerate(solvers)
                                 if getattr(solver, "_copper_foils", ()))
        self._foil_initial = tuple((tensor, tensor.clone())
            for solver, _activity in self._foil_lanes for _name, tensor in iter_copper_state(solver))
        from .batch_materials import NativeMaterialLane
        self._material_lanes = tuple(NativeMaterialLane(solver, self.active[lane])
            for lane, solver in enumerate(solvers) if solver.dispersive_enabled or solver.magnetic_dispersive_enabled)
        from ..checkpoint import iter_lumped_state_specs, iter_circuit_state_specs, iter_network_state_specs, iter_network_delay_state_specs
        self._rf_initial = tuple((tensor, tensor.clone()) for solver in solvers for tensor in (
            *(getattr(runtime, name) for _key, runtime, name, *_rest in iter_lumped_state_specs(solver)),
            *(tensor for iterator in (iter_circuit_state_specs, iter_network_state_specs, iter_network_delay_state_specs)
              for _name, tensor in iterator(solver)),
            *(getattr(runtime, name) for runtime in getattr(solver, "_network_runtimes", ())
              for name in ("absorbed_energy", "generated_energy", "port_energy")),
            *(tensor for runtime in getattr(solver, "_port_runtimes", ())
              for tensor in (runtime.electric_time, runtime.magnetic_time, runtime.drive_buffer))))
        _share_static_materials(self.solvers)
        self.active.zero_()
        self.active[:len(solvers)].fill_(1)
        self._bind_operations()

    def reset(self):
        for lane in getattr(self, "_material_lanes", ()):
            for tensor, initial in lane.initial:
                tensor.copy_(initial)
        for tensor, initial in getattr(self, "_rf_initial", ()):
            tensor.copy_(initial)
        for tensor, initial in getattr(self, "_foil_initial", ()):
            tensor.copy_(initial)
        for name in self.state_names:
            if name in self.initial:
                self.buffers[name].copy_(self.initial[name])
            else:
                self.buffers[name].zero_()
        self.active.zero_()
        self.active[:len(self.solvers)].fill_(1)

    def advance(self, lanes=None):
        """One Yee step of the bound lanes; ``lanes`` lists the running ones (default all).

        Retired lanes keep ``active`` at zero for the activity-masked pieces and get no
        fused Yee launch, so their fields stay frozen at their final step.
        """
        from ..copper_foil import begin_copper_step, correct_copper_magnetic, advance_copper_electric, accumulate_copper_spectra
        cores = self._core if lanes is None else tuple(self._core[lane] for lane in lanes)
        for solver, _activity in self._foil_lanes:
            begin_copper_step(solver)
        for lane in self._material_lanes:
            lane.advance_magnetic()
        for core in cores:
            core.magnetic()
        for solver, activity in self._foil_lanes:
            correct_copper_magnetic(solver, activity=activity)
        for lane in self._material_lanes:
            lane.correct_magnetic()
            lane.advance_electric()
        for core in cores:
            core.electric()
        for solver, activity in self._foil_lanes:
            advance_copper_electric(solver, activity=activity)
            accumulate_copper_spectra(solver, activity=activity)

    def _bind_operations(self):
        """Bind one core runtime's fused Yee H/E launches to each lane's slot storage.

        Lanes advance in a host loop: two launches per lane and step instead of one
        lane-strided launch, with the same bytes per cell. Lanes that share a coefficient
        bank share its compressed coefficient tables.
        """
        from .core_path import CorePathRuntime

        tables = {}
        self._core = tuple(CorePathRuntime(solver, coefficient_tables=tables) for solver in self.solvers)
        for solver in self.solvers:
            # A solo core runtime from an ordinary preparation still points at the replaced storage.
            solver.__dict__.pop("_core_runtime", None)

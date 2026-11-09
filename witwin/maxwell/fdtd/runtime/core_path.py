"""Admission and prepared launches for the fused Yee update."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ...compiler.accel.dispatch import requires_grad
from ..boundary.common import BOUNDARY_PEC


STRIP = 32
REUSE_BYTES_PER_CELL = 36
L2_THRESHOLD_NUMERATOR = 2
L2_THRESHOLD_DENOMINATOR = 3


def yee_block_strip(ny: int, nz: int, l2_bytes: int) -> int:
    """Choose the core Yee launch's strip width in j-blocks."""
    # Strip planes when their reuse footprint exceeds two thirds of device L2.
    # Calibrated on A6000 (sm_86, 6 MiB) only; sm_89/sm_90/sm_100 remain
    # unvalidated and each needs a slab spot check.
    threshold = (L2_THRESHOLD_NUMERATOR * l2_bytes) // L2_THRESHOLD_DENOMINATOR
    return STRIP if REUSE_BYTES_PER_CELL * ny * nz > threshold else 0


def _device_l2_bytes(device) -> int | None:
    """Device-reported L2 size; None off CUDA so host-side construction never initializes CUDA."""
    device = torch.device(device)
    return torch.cuda.get_device_properties(device).L2_cache_size if device.type == "cuda" else None


def core_path_blockers(solver) -> list[str]:
    from ..boundary.runtime import _effective_face_kinds, closed_transverse_pml

    scene = solver.scene
    # Scene symmetry lowers its plane to a PEC/PMC face; the core kernels have no PMC face.
    face_kinds = {kind for pair in _effective_face_kinds(scene) for kind in pair}
    blockers = []
    if (closed_transverse_pml(scene)
            and getattr(solver, "active_absorber_type", "none") in ("cpml", "stablepml")):
        blockers.append("closed-transverse CPML (passivity guard)")
    for attr, label in (
        ("gyromagnetic_enabled", "gyromagnetic media"),
        ("full_aniso_enabled", "full or off-diagonal anisotropy"),
        ("magnetically_conductive_enabled", "magnetic conductivity"),
    ):
        if getattr(solver, attr, False):
            blockers.append(label)
    if scene.coordinates is not None:
        blockers.append("coordinate charts")
    if scene.boundary.uses_kind("bloch") or getattr(solver, "complex_fields_enabled", False):
        blockers.append("Bloch/complex fields")
    for kind in ("mur", "mode_abc", "pmc"):
        if scene.boundary.uses_kind(kind) or kind in face_kinds:
            blockers.append(kind.upper() + " boundaries")
    if any(torch.any(getattr(solver, "mu_" + component) != solver.mu0).item()
           for component in ("Hx", "Hy", "Hz") if hasattr(solver, "mu_" + component)):
        blockers.append("magnetic permeability")
    if any(getattr(solver, name).ndim != 3 for name in ("Ex", "Ey", "Ez") if hasattr(solver, name)):
        blockers.append("Batch lanes")
    if hasattr(solver, "shards"):
        blockers.append("multi-GPU/distributed execution")
    if hasattr(solver, "Ex") and solver.Ex.dtype != torch.float32:
        blockers.append("non-float32 fields")
    if torch.device(scene.device).type != "cuda":
        blockers.append("non-CUDA execution")
    if requires_grad(scene):
        blockers.append("gradient-requiring scene tensors")
    from ...sources import (
        PointDipole, PlaneWave, GaussianBeam, AstigmaticGaussianBeam, ModeSource,
        CustomFieldSource, CustomCurrentSource, UniformCurrentSource,
    )
    from ...monitors import (
        PointMonitor, PlaneMonitor, FluxMonitor, FinitePlaneMonitor, ClosedSurfaceMonitor,
        ModeMonitor, DiffractionMonitor, PermittivityMonitor, FieldTimeMonitor, FluxTimeMonitor, BreakdownMonitor,
    )
    # All admitted sources compile to ordinary, electric/magnetic surface or TFSF terms.
    if any(not isinstance(source, (PointDipole, PlaneWave, GaussianBeam, AstigmaticGaussianBeam,
                                   ModeSource, CustomFieldSource, CustomCurrentSource, UniformCurrentSource))
           for source in scene.sources):
        blockers.append("unsupported source type")
    # Spectral planes share grouped observers; material monitors are evaluated at export.
    # Time monitors retain the per-step observer tail through the run_chunks guard.
    if any(not isinstance(monitor, (PointMonitor, PlaneMonitor, FluxMonitor, FinitePlaneMonitor,
                                    ClosedSurfaceMonitor, ModeMonitor, DiffractionMonitor,
                                    PermittivityMonitor, FieldTimeMonitor, FluxTimeMonitor, BreakdownMonitor))
           for monitor in scene.monitors):
        blockers.append("unsupported monitor type")
    return list(dict.fromkeys(blockers))


def core_path_has_separate_physics(solver) -> bool:
    """Pieces that need stepping's ordered field and port blocks around Yee."""
    return (any(getattr(solver.scene, name, ()) for name in
                ("ports", "lumped_elements", "thin_wires", "circuits", "networks"))
            or any(getattr(solver, name, ()) for name in
                   ("_port_runtimes", "_lumped_element_runtimes", "_coaxial_element_runtimes",
                    "_wire_runtime", "_circuit_runtimes", "_network_runtimes"))
            or any(getattr(solver, name, False) for name in
                   ("_copper_foils", "_thin_sheets", "_cut_surface", "surface_impedance_enabled")))


def core_path_step_loop_blockers(solver) -> list[str]:
    """W2 capture exclusions; the fused Yee pair still runs in the per-step loop."""
    blockers = []
    if getattr(solver.scene, "circuits", ()) or getattr(solver, "_circuit_runtimes", ()):
        blockers.append("circuits")
    if any(getattr(runtime, "circuit_port", None) is not None
           for runtime in getattr(solver, "_port_runtimes", ())):
        blockers.append("circuit-bound ports")
    if getattr(solver, "surface_impedance_enabled", False):
        blockers.append("surface impedance")
    if getattr(solver, "_cut_surface", None):
        blockers.append("cut-surface constraints")
    if getattr(getattr(solver, "_wire_runtime", None), "monitor_state", ()):
        blockers.append("host-driven wire monitors")
    if core_path_has_separate_physics(solver):
        for active, label in (
            (getattr(solver, "tfsf_enabled", False), "TFSF with separate physics"),
            (bool(getattr(solver, "_magnetic_source_terms", ())), "magnetic sources with separate physics"),
            (getattr(solver, "modulation_enabled", False), "modulation with separate physics"),
        ):
            if active:
                blockers.append(label)
    return blockers


def _validate_separate_step(solver):
    # The field block receives a fixed time; ADE has no live time input.
    assert not any(solver.scene.boundary.uses_kind(kind) for kind in ("mur", "mode_abc")), \
        "Core separate-physics graph requires no Mur or mode-ABC boundaries."
    assert not getattr(solver, "tfsf_enabled", False), "Core separate-physics graph requires no TFSF."
    assert not getattr(solver, "_magnetic_source_terms", ()), \
        "Core separate-physics graph requires no magnetic surface sources."
    assert not getattr(solver, "modulation_enabled", False), \
        "Core separate-physics graph requires no modulation."


def _term_may_write_pec(solver, term) -> bool:
    """Intersect a compiled patch's destination indices with tangential PEC faces.

    Only metadata is read: even a zero-valued patch is a writer. Unknown layouts
    and periodic image gathers retain enforcement conservatively.
    """
    name = term.get("field_name")
    if name in ("Hx", "Hy", "Hz"):
        return False
    if name not in ("Ex", "Ey", "Ez"):
        return True
    if solver.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
        return True
    patch = term.get("cw_cos_patch")
    if patch is None:
        patch = term.get("patch")
    offsets = term.get("offsets")
    field = getattr(solver, name, None)
    if patch is None or field is None or offsets is None or patch.ndim != 3 or len(offsets) != 3:
        return True
    # patch_kernel clips each destination coordinate to the component's shape.
    bounds = [(max(0, int(start)), min(size, int(start) + length))
              for start, length, size in zip(offsets, patch.shape, field.shape)]
    if any(low >= high for low, high in bounds):
        return False
    for axis, (low, high) in enumerate(bounds):
        if axis == "xyz".index(name[-1].lower()):
            continue
        for side, index in (("low", 0), ("high", field.shape[axis] - 1)):
            if (getattr(solver, f"boundary_{'xyz'[axis]}_{side}_code") == BOUNDARY_PEC
                    and low <= index < high):
                return True
    return False


def pec_clamp_plan(solver, sources) -> tuple[bool, bool]:
    """Choose the two historical clamp sites once, before graph capture.

    Core component<Electric> already stores +0 on the entire tangential PEC
    face union, before curl/psi updates. A later clamp is redundant only while
    no intervening writer can change that union. Complex fields retain both
    historical sites; their imaginary enforcement belongs to stepping.
    """
    first = bool(sources.groups.get("ordinary"))
    second = not getattr(solver, "tfsf_enabled", False)
    if getattr(solver, "complex_fields_enabled", False):
        return first, second
    # Missing capability metadata cannot prove that correction is a no-op.
    dispersive = getattr(solver, "electric_dispersive_enabled", True)
    separate = core_path_has_separate_physics(solver)
    # P1: the original first clamp dominates the second, with no E writer between.
    port_phase = any(getattr(solver, name, ()) for name in (
        "_port_runtimes", "_lumped_element_runtimes", "_coaxial_element_runtimes", "_network_runtimes"))
    if first and not dispersive and not port_phase:
        second = False
    if not getattr(solver, "has_pec_faces", False) or separate:
        return first, second
    term_groups = [getattr(solver, name, None) for name in ("_source_terms", "_electric_source_terms")]
    if getattr(solver, "tfsf_enabled", False):
        # Auxiliary/reference TFSF terms use other index layouts; retain their clamp.
        if sources.auxiliary is not None:
            return first, second
        term_groups.append(getattr(solver, "_tfsf_state", {}).get("electric_terms"))
    if any(terms is None for terms in term_groups):
        return first, second
    if any(_term_may_write_pec(solver, term) for terms in term_groups for term in terms):
        return first, second
    # P2: the core E stores are still intact at the first site, and at the
    # second site too if electric ADE correction does nothing. Never add a site
    # (in particular TFSF without ordinary sources has no enforcement).
    return False, second and dispersive


class _FDTDCoreFallback(RuntimeError):
    """Abort an automatic attempt so its owner can rebuild the existing path."""

    def __init__(self, reason, *, setup_failed=False):
        super().__init__(reason)
        self.setup_failed = setup_failed


def core_path_request_blockers(simulation, scene=None, solver=None):
    config = simulation.config
    scene = simulation.scene if scene is None else scene
    # Before preparation, an explicit cpml/stablepml absorber already decides the guard
    # blocker; without it automatic admission builds a core solver only to rebuild it.
    blockers = core_path_blockers(
        SimpleNamespace(scene=scene, active_absorber_type=config.absorber.value) if solver is None else solver)
    if config.parallel is not None:
        blockers.append("multi-GPU/distributed execution")
    if hasattr(simulation, "_batch_source_scales"):
        blockers.append("Batch lanes")
    if simulation.has_trainable_parameters or config.adjoint_checkpoint_stride is not None:
        blockers.append("adjoint/gradient-requiring execution")
    if config.field_dtype != torch.float32:
        blockers.append("non-float32 fields")
    if config.backend != "native":
        blockers.append("non-native backend")
    return list(dict.fromkeys(blockers))


def validate_core_path_request(simulation, scene=None, solver=None):
    blockers = core_path_request_blockers(simulation, scene, solver)
    if blockers:
        raise ValueError("fdtd_core_path=True is blocked by: " + "; ".join(dict.fromkeys(blockers)))


def core_path_decision(simulation, scene=None, solver=None, *, initialize=True, resume_from=None):
    """Resolve auto admission with the same blockers as explicit core requests."""
    if simulation.config.fdtd_core_path is False:
        return "existing", None
    if resume_from is not None:
        return "existing", "resume_from"
    if not initialize:
        return "existing", "non-initializing prepare (initialize=False)"
    if simulation.material_cache is not None:
        return "existing", "material_cache override"
    blockers = core_path_request_blockers(simulation, scene, solver)
    return ("existing", blockers[0]) if blockers else ("core", None)


def _compress_coefficients(solver, family):
    names = [f"c{family}{axis}_{kind}" for axis in "xyz" for kind in ("decay", "curl")]
    scalars = [solver._coefficient_uniformity[name] for name in names]
    uniform = all(value is not None for value in scalars)
    shape = (solver.Nx, solver.Ny, solver.Nz)
    if uniform:
        return (torch.empty(0, device=solver.device, dtype=torch.uint16),
                torch.empty(0, device=solver.device, dtype=torch.float32), scalars, True)
    from ...compiler.accel.dispatch import UNCHANGED
    from ...compiler.accel.tables import compress

    compressed = compress([getattr(solver, name) for name in names], shape)
    if compressed is not UNCHANGED:
        return *compressed, [0.0] * 6, False
    padded = []
    for name in names:
        tensor = getattr(solver, name)
        padding = tuple(size - actual for size, actual in zip(shape, tensor.shape))
        padded.append(F.pad(tensor, (0, padding[2], 0, padding[1], 0, padding[0])))
    tuples = torch.stack(padded, dim=-1).reshape(-1, 6)
    # Deduplicate float32 bit patterns so signed zeros retain their exact values.
    table_bits, inverse = torch.unique(tuples.view(torch.int32), dim=0, return_inverse=True)
    table = table_bits.view(torch.float32)
    dtype = torch.uint32 if table.shape[0] > 65535 else torch.uint16
    return inverse.to(dtype).reshape(shape), table, [0.0] * 6, False


def _separate_graph_state_tensors(solver):
    """Capture scratch and diagnostics beyond graph_state_tensors' physical state."""
    state = []

    def add(owner, names):
        for name in names:
            tensor = owner.get(name) if isinstance(owner, dict) else getattr(owner, name, None)
            if isinstance(tensor, torch.Tensor):
                state.append(tensor)

    def branch(runtime):
        if runtime is None:
            return
        add(runtime, [name for name in vars(runtime) if name.startswith("last_")] + [
            "inductor_current", "capacitor_voltage", "edge_buffer", "correction_buffer",
            "fast_state_scratch", "part_voltage",
        ])
        for part in getattr(runtime, "parts", ()):
            add(part, ("edge_buffer", "correction_buffer"))
        add(getattr(runtime, "electric_history", None), ("edge_buffer", "voltage_buffer"))

    for entry in getattr(solver, "_copper_foils", ()):
        add(entry, ("old_e",))
        add(entry.get("fused_electric", {}), ("partials", "counter"))
    for sheet in getattr(solver, "_thin_sheets", ()):
        # Physical sheet/feed state is already enumerated by sheet.state_tensors().
        add(sheet, ("h_old", "h_new", "e_cols", "e_old", "e_new", "h_rows"))
        fused = (sheet.copper or {}).get("fused", {})
        add(fused, ("history", "rhs", "coupled", "midpoint", "partials", "counter"))
        add(fused.get("chord_solve", {}), ("residual", "direction", "temporary"))
        for feed in sheet.feeds:
            add(feed["binding"], ("column_term",))

    add(solver, ("_port_observer_step",))
    for runtime in getattr(solver, "_port_runtimes", ()):
        add(runtime, ("drive_buffer", "_drive_row", "electric_time", "magnetic_time",
                      "observer_current_buffer", "observer_window_buffer",
                      "observer_voltage_kernel_buffer", "observer_current_kernel_buffer"))
        for accumulator in (runtime.accumulator, runtime.drive_accumulator):
            add(accumulator, ("_voltage_sum", "_current_sum", "_window_weight_sum",
                              "_voltage_term", "_current_term"))
        branch(runtime.lumped)
        add(runtime.wire_provider, ("coordinate",))
        buffers = runtime.substep_buffers or {}
        add(buffers, tuple(buffers))
    for runtime, _ in getattr(solver, "_lumped_element_runtimes", ()):
        branch(runtime)
    for runtime in getattr(solver, "_coaxial_element_runtimes", ()):
        branch(runtime)
    for runtime in getattr(solver, "_network_runtimes", ()):
        # Same mutable buffers as make_network_runner, including energy diagnostics.
        add(runtime, ("state", "next_state", "state_drive", "free_voltage", "raw_free_voltage",
                      "carried_voltage", "network_voltage", "voltage_after", "branch_current",
                      "output_buffer", "direct_drive", "zero_rhs", "zero_solution", "zero_reflected",
                      "core_incident", "core_reflected", "scene_incident", "scene_reflected",
                      "solve_workspace", "solve_scalar", "net_power", "power_buffer", "port_energy",
                      "absorbed_increment", "generated_increment", "absorbed_energy", "generated_energy"))
        for group in runtime.terminal_groups:
            add(group, ("edge_buffer", "correction_buffer"))
        delay = runtime.delay_runtime
        if delay is not None:
            add(delay, ("forward_ring", "reverse_ring", "forward_previous_input", "forward_previous_output",
                        "reverse_previous_input", "reverse_previous_output", "forward_integer_sample",
                        "forward_fractional_sample", "forward_temp", "reverse_integer_sample",
                        "reverse_fractional_sample", "reverse_temp", "read_positions", "read_indices",
                        "write_indices", "cursor"))
    return state


def _live_graph_state_tensors(solver):
    """Restore coefficient storage and stress accumulators after capture."""
    state = []
    if any(getattr(solver, flag, False) for flag in
           ("nonlinear_enabled", "modulation_enabled", "breakdown_enabled")):
        from .stepping import _electric_decay_tensors, _electric_curl_tensors
        state.extend(_electric_decay_tensors(solver))
        state.extend(_electric_curl_tensors(solver))
    for record in getattr(solver, "breakdown_observers", ()):
        accumulator = record["accumulator"]
        state.extend(value for value in vars(accumulator).values() if isinstance(value, torch.Tensor))
    return state


class CorePathRuntime:
    def __init__(self, solver):
        from ..cuda.core.build import build_extension
        from .stepping import cpml_layout_params, _electric_decay_tensors, _electric_curl_tensors

        extension = build_extension()
        self.extension = extension
        self.chunk_size = 32
        self.step = torch.zeros(1, dtype=torch.int64, device=solver.device)
        self.sources = None
        self.point_observers = None
        self.graphs = {}
        self.coefficients = {}
        self.launches = {}
        self.yee_block_strip = {}
        l2_bytes = _device_l2_bytes(solver.device)
        for family in ("h", "e"):
            electric = family == "e"
            live = electric and any(getattr(solver, flag, False) for flag in
                                    ("nonlinear_enabled", "modulation_enabled", "breakdown_enabled"))
            modulated = electric and getattr(solver, "modulation_enabled", False)
            if live:
                indices = torch.empty(0, device=solver.device, dtype=torch.uint16)
                table = torch.empty(0, device=solver.device, dtype=torch.float32)
                scalars, uniform = [0.0] * 6, False
            else:
                indices, table, scalars, uniform = _compress_coefficients(solver, family)
            self.coefficients[family] = (indices, table)
            opposite = "H" if electric else "E"
            tensors = [getattr(solver, family.upper() + axis) for axis in "xyz"]
            tensors += [getattr(solver, opposite + axis) for axis in "xyz"]
            tensors += [getattr(solver, f"inv_d{axis}_{family}") for axis in "xyz"]
            tensors += [indices, table]
            integers = [solver.Nx, solver.Ny, solver.Nz]
            integers += [getattr(solver, f"boundary_{axis}_{side}_code")
                         for axis in "xyz" for side in ("low", "high")]
            cpml = 0
            if solver.uses_cpml:
                cpml = 1 if solver._cpml_memory_mode == "dense" else 2
                tensors += [getattr(solver, f"cpml_{profile}_{family}_{axis}")
                            for axis in "xyz" for profile in ("inv_kappa", "b", "c")]
                for component, axes in (("x", "yz"), ("y", "xz"), ("z", "xy")):
                    for axis in axes:
                        attr = f"psi_{family}{component}_{axis}"
                        tensors.append(getattr(solver, attr))
                        integers.extend(cpml_layout_params(solver, attr) if cpml == 2 else (0, 0, 0))
            if live:
                tensors += [tensor for pair in zip(_electric_decay_tensors(solver),
                                                   _electric_curl_tensors(solver)) for tensor in pair]
            if modulated:
                tensors += [getattr(solver, f"mod_{kind}_E{axis}")
                            for axis in "xyz" for kind in ("cos", "sin", "omega")]
                tensors.append(solver._modulation_time)
            strip = 0 if l2_bytes is None else yee_block_strip(integers[1], integers[2], l2_bytes)
            self.yee_block_strip["electric" if electric else "magnetic"] = strip
            integers.append(strip)
            self.launches[family] = extension.CoreLaunch(
                tensors, integers, scalars, electric, cpml, uniform, indices.dtype == torch.uint32,
                live, modulated,
            )

    def magnetic(self):
        self.launches["h"].launch()

    def electric(self):
        self.launches["e"].launch()

    def prepare_step(self, solver, *, use_gpu_dft, time_steps):
        from .core_tables import (
            compile_spectral_tables, compile_monitor_routes, NativePointObserverTables,
            compile_auxiliary_fusion,
        )
        from ..observers import uses_native_point_observers
        from .stepping import enforce_pec_boundaries, apply_mur_boundaries, _field_update_block, _port_step_block
        from ..ports import prepare_port_drive_tables
        from ..cut_surface import apply_cut_surface_constraints
        from .breakdown import advance_breakdown_state
        from ..excitation.tfsf_apply import (
            apply_tfsf_h_correction, apply_tfsf_e_correction, advance_tfsf_auxiliary_magnetic,
        )

        separate = core_path_has_separate_physics(solver)
        if separate:
            _validate_separate_step(solver)
        spectral = compile_spectral_tables(solver, self.extension, self.step) if use_gpu_dft else None
        engine = compile_monitor_routes(solver)
        self.point_observers = (NativePointObserverTables(solver, self.step, time_steps)
                                if uses_native_point_observers(solver) and engine is None else None)
        self.observer_groups = engine.groups if engine is not None else []
        self.observer_gather = None
        if engine is not None and engine.state is not None and engine.groups:
            if not all(args[3] is engine.weights and args[4] is engine.slot.active
                       for args in engine.groups):
                raise RuntimeError("Core observer groups must share weights and active tensors.")
            self.observer_gather = self.extension.CoreObserverGather(
                *([args[index] for args in engine.groups] for index in (0, 1, 2, 5, 6, 7)),
                engine.weights, engine.slot.active,
            )
        gather = self.observer_gather
        commit = self.extension.CoreStepLaunch(3, [self.step], [], [])
        sources = self.sources
        self.auxiliary_fusion = compile_auxiliary_fusion(solver, self.extension, self.step, sources)
        auxiliary_fusion = self.auxiliary_fusion
        clamp_after_sources, clamp_after_dispersion = pec_clamp_plan(solver, sources)
        self.pec_clamp_plan = (clamp_after_sources, clamp_after_dispersion)

        def inject(name):
            for launch in sources.groups.get(name, ()):
                launch.launch()

        def ordinary_step():
            if getattr(solver, "modulation_enabled", False):
                solver.fdtd_module.advanceModulationTime3D(
                    ModulationTime=solver._modulation_time,
                    ModulationStep=solver._modulation_step, dt=solver.dt,
                ).launchRaw()
            solver._advance_magnetic_dispersive_state()
            self.magnetic()
            if solver.tfsf_enabled:
                if auxiliary_fusion is not None:
                    if auxiliary_fusion.boundary is not None:
                        auxiliary_fusion.boundary.launch()
                    auxiliary_fusion.magnetic.launch()
                elif sources.auxiliary is not None:
                    values = sources.rows.get("auxiliary/boundary_signals")
                    if values is not None:
                        solver._tfsf_state["auxiliary_grid"].prepare_incident_boundary(
                            values.index_select(0, self.step).reshape(-1))
                    apply_tfsf_h_correction(solver, 0.0)
                    advance_tfsf_auxiliary_magnetic(solver)
                else:
                    inject("tfsf_magnetic")
            inject("magnetic")
            solver._apply_magnetic_dispersive_corrections()
            solver._advance_dispersive_state()
            if getattr(solver, "nonlinear_enabled", False):
                solver._update_nonlinear_electric_coefficients()
            self.electric()
            if solver.tfsf_enabled:
                if auxiliary_fusion is not None:
                    auxiliary_fusion.electric.launch()
                elif sources.auxiliary is not None:
                    apply_tfsf_e_correction(solver, 0.0)
                    sources.auxiliary.launch()
                else:
                    inject("tfsf_electric")
            inject("electric")
            inject("ordinary")
            if clamp_after_sources:
                enforce_pec_boundaries(solver)
            solver._apply_dispersive_corrections()
            if clamp_after_dispersion:
                enforce_pec_boundaries(solver)
            apply_mur_boundaries(solver)
            apply_cut_surface_constraints(solver)
            if getattr(solver, "breakdown_enabled", False):
                advance_breakdown_state(solver, self.step)
            if spectral is not None:
                spectral.launch()

        if separate:
            runtimes = tuple(getattr(solver, "_port_runtimes", ()))
            counter = None
            if runtimes:
                prepare_port_drive_tables(solver, time_steps)
                counter = solver._port_observer_step
                counter.fill_(runtimes[0].sample_index)
                solver._port_observer_graph_active = True
            assert runtimes or not any(sheet.feeds for sheet in getattr(solver, "_thin_sheets", ())), \
                "Fed sheets require prepared port runtimes."

            def tail():
                solver._apply_dispersive_corrections()
                if clamp_after_dispersion:
                    enforce_pec_boundaries(solver)
                apply_mur_boundaries(solver)
                apply_cut_surface_constraints(solver)
                if getattr(solver, "breakdown_enabled", False):
                    advance_breakdown_state(solver, self.step)
                if spectral is not None:
                    spectral.launch()

            # Standalone passive elements also need the port phase, even without V/I observers.
            port_phase = bool(runtimes or getattr(solver, "_lumped_element_runtimes", ())
                              or getattr(solver, "_coaxial_element_runtimes", ())
                              or getattr(solver, "_network_runtimes", ()))

            def physics_step():
                _field_update_block(solver, 0.0)
                inject("electric")
                inject("ordinary")
                if clamp_after_sources:
                    enforce_pec_boundaries(solver)
                if port_phase:
                    _port_step_block(solver, counter, post_source=tail)
                    if counter is not None:
                        counter.add_(1)
                else:
                    tail()

        def step():
            if separate:
                physics_step()
            else:
                ordinary_step()
            if engine is not None and engine.state is not None:
                engine.slot.native.batch_observer_weights(engine.state, engine.schedule, engine.signals,
                    engine.slot.active, self.step, engine.weights)
                if gather is not None:
                    gather.launch()
            if self.point_observers is not None:
                self.point_observers.launch()
            if getattr(solver, "breakdown_observers_enabled", False):
                solver.accumulate_breakdown_observers(self.step)
            commit.launch()

        return step, engine

    def capture_chunk(self, solver, step, count):
        from ..cuda.runtime.graph import CudaGraphRunner

        def chunk():
            for _ in range(count):
                step()

        return CudaGraphRunner(device=solver.device, enabled=True, warmup_steps=0).capture(chunk)

    def run_chunks(self, solver, *, time_steps, start_step, end_step, use_cuda_graph,
                   use_gpu_dft, cooperative, cooperative_driver, cooperative_observer,
                   resume_from, shutoff, shutoff_check_interval, shutoff_min_step,
                   enable_plot, pbar):
        """Return completed steps; dispersive capture failures restore state and raise."""
        from ..checkpoint import graph_state_tensors
        from ..cuda.runtime.graph import capture_suspended
        from ..observers import uses_native_point_observers
        from .stepping import _electric_field_energy, visualize_slice

        solver._core_step_graph_active = False
        solver._core_step_graph_k = None
        solver._core_step_graph_chunks = []
        # Unqualified pieces and time observers retain stepping's per-step tail.
        if core_path_step_loop_blockers(solver) or getattr(solver, "time_observers_enabled", False):
            return None
        if (not use_cuda_graph or capture_suspended() or cooperative or cooperative_driver or cooperative_observer is not None
                or resume_from is not None or start_step != 0 or self.sources is None
                or time_steps > self.sources.steps
                or (solver.observers_enabled and solver._single_observer_engine is None
                    and not uses_native_point_observers(solver))
                or (solver.dft_enabled and not use_gpu_dft)):
            return None
        if self.chunk_size not in (1, 16, 32):
            raise ValueError("Core graph chunk_size must be 1, 16 or 32.")
        if end_step == start_step:
            return 0
        schedule = []
        current = start_step
        while current < end_step:
            boundary = end_step
            if shutoff > 0:
                boundary = min(boundary, (current // shutoff_check_interval + 1) * shutoff_check_interval)
            if pbar is not None:
                interval = solver.progress_update_interval
                boundary = min(boundary, (current // interval + 1) * interval)
            if enable_plot:
                boundary = min(boundary, ((current + solver.plot_interval - 1) // solver.plot_interval)
                               * solver.plot_interval + 1)
            count = min(self.chunk_size, boundary - current)
            schedule.append((current, count))
            current += count

        port_graph_active = getattr(solver, "_port_observer_graph_active", False)
        step, engine = self.prepare_step(solver, use_gpu_dft=use_gpu_dft, time_steps=time_steps)
        state = list(graph_state_tensors(solver).values()) + [self.step]
        state.extend(_live_graph_state_tensors(solver))
        if core_path_has_separate_physics(solver):
            state.extend(_separate_graph_state_tensors(solver))
        if use_gpu_dft:
            state.append(solver._dft_step)
            for buffers in solver._dft_batched_fields.values():
                state.extend((buffers["real"], buffers["imag"]))
        point_observers = self.point_observers
        if point_observers is not None:
            state.extend(point_observers.state_tensors)
        if engine is not None:
            state.extend(tensor for tensor in (engine.step, engine.state, engine.weights) if tensor is not None)
            for args in engine.groups:
                state.extend((args[5], args[6]))
        saved = [(tensor, tensor.clone()) for tensor in {id(t): t for t in state}.values()]
        # Port accumulation also updates a host diagnostic during warmup/recording.
        counts = [(accumulator, accumulator._sample_count)
                  for runtime in getattr(solver, "_port_runtimes", ())
                  for accumulator in (runtime.accumulator, runtime.drive_accumulator)
                  if accumulator is not None]

        def restore():
            for target, value in saved:
                target.copy_(value)
            for accumulator, count in counts:
                accumulator._sample_count = count

        try:
            # Warm exactly one valid row and restore even a driven, nonzero state.
            step()
            torch.cuda.synchronize(solver.device)
            restore()
            for count in sorted({1, *(count for _, count in schedule)}):
                self.graphs[count] = self.capture_chunk(solver, step, count)
                restore()
        except Exception as error:
            restore()
            solver._port_observer_graph_active = port_graph_active
            self.graphs.clear()
            if getattr(solver, "_core_path_auto", False):
                raise _FDTDCoreFallback(f"{type(error).__name__}: {error}") from error
            if getattr(solver, "dispersive_enabled", False):
                raise
            return None
        finally:
            saved.clear()
        if use_gpu_dft:
            solver._dft_step = self.step
        if engine is not None:
            engine.step = self.step
        solver._core_step_graph_active = True
        solver._core_step_graph_k = self.chunk_size
        solver._cuda_graph_active = True
        aux = solver._tfsf_state.get("auxiliary_grid") if solver.tfsf_enabled else None
        for current, count in schedule:
            self.graphs[count]()
            solver._core_step_graph_chunks.append(count)
            steps_run = current + count
            n = steps_run - 1
            if aux is not None:
                aux.time_step += count
            if pbar is not None:
                pbar.update(count)
            if shutoff > 0 and steps_run % shutoff_check_interval == 0:
                e_energy = _electric_field_energy(solver)
                solver._shutoff_peak = torch.maximum(solver._shutoff_peak, e_energy)
                should_stop = (n >= shutoff_min_step and
                    bool(((solver._shutoff_peak > 0.0) & (e_energy < shutoff * solver._shutoff_peak)).item()))
                if should_stop:
                    solver._shutoff_triggered = True
                    solver._shutoff_step = n
                    break
            if pbar is not None:
                if ((steps_run % solver.progress_update_interval == 0 or n == end_step - 1)
                        and solver.dft_enabled and solver.dft_start_step is not None):
                    pbar.set_postfix({"status": "transient"} if n < solver.dft_start_step else
                                     {"status": "DFT accumulation", "samples": solver.dft_sample_count})
            if enable_plot and n % solver.plot_interval == 0:
                visualize_slice(solver, n)
        if point_observers is not None:
            point_observers.finalize(solver, steps_run)
        if pbar is not None:
            pbar.close()
        return steps_run


def activate_core_path(solver, simulation):
    if simulation.config.fdtd_core_path == "auto":
        _, reason = core_path_decision(simulation, solver=solver)
        if reason is not None:
            raise _FDTDCoreFallback(reason)
    else:
        validate_core_path_request(simulation, solver=solver)
    solver._core_runtime = CorePathRuntime(solver)
    from .core_tables import compile_source_tables, source_tables_supported

    if source_tables_supported(solver):
        runtime = solver._core_runtime
        runtime.sources = compile_source_tables(solver, runtime.extension, runtime.step,
                                                solver._time_plan["total_steps"])

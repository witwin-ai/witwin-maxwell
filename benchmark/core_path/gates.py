"""Fresh-process G1/G1-prime/G2/G4 capture and separate CPU comparison.

The candidate arm is the default FDTD execution of the working tree. The baseline
arm is the frozen existing-path capture in the evidence store
(``tests/support/frozen_oracles.py``): ``<store>/gates/<gate>`` for bitwise
comparisons and ``<store>/gates/g1-rule2`` (full and half horizon) for Rule 2 G1.
``--baseline`` points a comparison at another baseline directory.

Examples::

    python -m benchmark.core_path.gates capture --dry-run
    python -m benchmark.core_path.gates capture --gate g1 --output <evidence-root>/core-cuda-gates/g1
    python -m benchmark.core_path.gates compare --gate g1 --output <evidence-root>/core-cuda-gates/g1

G1-prime requires a JSON mapping of exact tensor names to {"kind": "fields"
or "phasors" or "flux", "reason": "pre-declared numerical change"}. All other
items remain bit-identity checks. O0 declares no bounded exceptions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import witwin.maxwell as mw

# The frozen store holds 0.6 result layouts; read results through the test API dialect.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "support"))
import api_dialect  # noqa: E402

from .scenes import (GATE_SCENES, G2_SCENES, W1_SCENES, W2_SCENES, W3_SCENES, DISPERSIVE_SCENES,
                     ANISO_SCENES, FERRITE_SCENES, FP64_SCENES, BLOCH_SCENES, CHART_SCENES,
                     CACHE_SCENES, MONITOR_SCENES, BLOCH_SEPARATE_SCENES, build_core_case)
from witwin.maxwell.scene import place_scene

# Scenes whose private surface/wire/ADE state and port, loss and breakdown outputs are pinned too.
PHYSICS_SCENES = (W1_SCENES + W2_SCENES + W3_SCENES + DISPERSIVE_SCENES + FERRITE_SCENES
                  + FP64_SCENES + CHART_SCENES + MONITOR_SCENES + BLOCH_SEPARATE_SCENES
                  + ("mur_lumped_port", "mu_slab", "bloch_debye_tfsf"))
# TFSF provider of each complex-field gate scene (fixed-k line, discrete CW slab, zero-k pulse).
BLOCH_PROVIDERS = {"bloch_oblique": "plane_wave_fixed_k", "bloch_oblique_te": "plane_wave_fixed_k",
                   "bloch_debye_tfsf": "plane_wave_fixed_k", "bloch_cw_tfsf": "plane_wave_grating_slab_cw",
                   "bloch_pulse_k0": "plane_wave_grating_slab_cw", "bloch_tfsf_resistor": "plane_wave_fixed_k"}


OUTPUT = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "core-cuda-gates"
# Cached external references live only in the main checkout (untracked); G2 reads them there.


def _reference_cache():
    # Reference caches are untracked and live in the main checkout, shared by its worktrees.
    if "WITWIN_BENCHMARK_CACHE" in os.environ:
        return Path(os.environ["WITWIN_BENCHMARK_CACHE"])
    root = Path(__file__).resolve().parents[2]
    common = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True)
    main = (root / common.stdout.strip()).resolve().parent if common.returncode == 0 else root
    return main / "benchmark" / "cache"


REFERENCE_CACHE = _reference_cache()
THREAD_ENV = {f"{name}_NUM_THREADS": "8" for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR")}
# A forced whole-step capture failure raises after restoring the solver state (W7 row
# 16c); the harness asserts that, then runs a fresh preparation of the same simulation
# without the pre-run field fill. Its run payloads must therefore equal the uninterrupted
# run of the identical scene.
REBUILT_BASELINES = {"o2_capture_failure": "o2_tfsf_pulse"}


def discrete_leaf(name, value):
    """Names describing discrete decisions remain exact even in floating storage."""
    return (not (value.is_floating_point() or value.is_complex())
            or any(token in name.lower() for token in
                   ("indices", "index", "mask", "owner", "count", "starts", "targets")))


def run_families(names):
    """Keep field, spectral and individual monitor scales independent."""
    family = {}
    for name in names:
        parts = name.split("/")
        if parts[0] in ("fields", "phasors", "dft_accumulators") and len(parts) > 1:
            if re.fullmatch(r"[EH][XYZ]", parts[1].upper()):
                family[name] = f"{parts[0]}/{parts[1][0].upper()}"
        elif parts[0] == "monitors" and len(parts) > 2:
            component = any(re.fullmatch(r"[EH][XYZ]", part.upper())
                            and parts[index + 1:] in ([], ["data"])
                            for index, part in enumerate(parts[2:], start=2))
            if component or (len(parts) == 3 and parts[2] in ("data", "field")):
                family[name] = f"monitors/{parts[1]}"
        elif parts[0] in ("physics", "outputs"):
            # Physics records get no cross-record structural-zero allowance.
            family[name] = name
    return family


def rule2_rows(reference, candidate, *, level, ref_half=None, new_half=None):
    """Collect shared-comparator receipts, including rejected T2 rows.

    Suppress only its final assertion so every rejection can be written to the
    gate receipt. All classification, bounds and growth checks remain shared.
    """
    from tests.support import numerical_equivalence as equivalence

    assert reference.keys() == candidate.keys() and reference
    if level == "run":
        assert ref_half is not None and new_half is not None, "Rule 2 G1 needs half-horizon captures"
        assert ref_half.keys() == new_half.keys() == reference.keys()
    rows = {}
    with patch.object(equivalence, "_assert_report"):
        for discrete in (False, True):
            keys = [key for key in reference if discrete_leaf(key, reference[key]) is discrete]
            if not keys:
                continue
            left = {key: reference[key] for key in keys}
            right = {key: candidate[key] for key in keys}
            if level == "run":
                report = equivalence.assert_run_level_equivalent(
                    {key: ref_half[key] for key in keys}, {key: new_half[key] for key in keys},
                    left, right, discrete=discrete, family=run_families(keys))
            else:
                report = equivalence.assert_numerically_equivalent(
                    left, right, level=level, discrete=discrete)
            for row in report["tensors"]:
                row.setdefault("structural_zero", False)
                rows[row["name"]] = row
    return rows


def tier_summary(rows):
    return {tier: sum(row["tier"] == tier for row in rows.values()) for tier in ("T0", "T1", "T2")}


def frozen_baseline(gate: str, *, rule2: bool = False) -> Path:
    """Verified frozen existing-path baseline directory for ``gate``."""
    from tests.support import frozen_oracles

    # G1-prime differs from G1 only in its comparison policy.
    name = "g1-rule2" if rule2 and gate == "g1" else "g1" if gate == "g1-prime" else gate
    root = frozen_oracles.store_root() / "gates" / name
    for key, entry in frozen_oracles.manifest()["payloads"].items():
        if key.startswith(f"gates/{name}/"):
            frozen_oracles.path(key)
    return root


def make_simulation(case, *, steps: int, full_field_dft: bool, material_cache=None):
    # Float64 cases stay inside the FP64 forward admission: sparse float64 spectra, no full-field DFT.
    fp64 = case.field_dtype == torch.float64
    precision = {"field_dtype": torch.float64} if fp64 else {}
    return mw.Simulation.fdtd(
        place_scene(case.scene, "cuda"), frequencies=case.frequencies,
        excitations=case.excitations,
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=max(1, case.steps if case.steps is not None else steps)),
            full_field_dft=full_field_dft and not fp64, **precision,
            spectral_sampler=mw.fdtd.SpectralSampler(window="none", normalize_source=False, **(
                {"observer_dtype": torch.float64} if fp64 else {})),
            cpml=mw.fdtd.CPMLConfig(memory_mode=case.cpml_memory, kappa_max=case.cpml_kappa_max), cuda_graph=True,
            shutoff=case.shutoff, shutoff_check_interval=7 if case.shutoff else 100,
            enable_plot=case.boundaries,
            **({"absorber": case.absorber} if case.absorber is not None else {}),
            material_cache=material_cache,
        ),
    )


# Process-counter identity attributes of a public witwin-core material (PhysicalMaterial and its
# subclasses) reached inside a compiled model: witwin-core assigns material_id from a process-wide
# counter at construction, so it depends on what the process built before (a round-tripped or
# reordered scene gets another value) and is a label, not setup data. G4 records such a material by
# its other dataclass fields (the benchmark/core_path/fingerprint.py rule) and drops these exact
# attributes on both arms; the capture lists the keys it left out. ``version`` is a constructor
# field, not a counter, and stays.
MATERIAL_COUNTER_ATTRIBUTES = ("material_id",)


def tensors(value, prefix: str, *, material_counters: list | None = None) -> dict[str, torch.Tensor]:
    """Snapshot numerical leaves, including scalar flux and compiled indices.

    With ``material_counters`` (the G4 setup capture), a witwin-core material is recorded without its
    MATERIAL_COUNTER_ATTRIBUTES and the left-out keys are appended to that list.
    """
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    if type(value) in (float, complex):
        return {prefix: torch.tensor(value, dtype=torch.complex128 if type(value) is complex else torch.float64)}
    if isinstance(value, (np.ndarray, np.number, bool, int, float, complex)):
        return {prefix: torch.as_tensor(value).clone()}
    counters = material_counters
    if is_dataclass(value) and not isinstance(value, type):
        skip = ()
        if counters is not None:
            from witwin.core import PhysicalMaterial

            if isinstance(value, PhysicalMaterial):
                skip = MATERIAL_COUNTER_ATTRIBUTES
                counters.extend(f"{prefix}/{name}" for name in skip)
        value = {field.name: getattr(value, field.name) for field in fields(value) if field.name not in skip}
    if isinstance(value, Mapping):
        return {key: tensor for name, item in value.items()
                for key, tensor in tensors(item, f"{prefix}/{name}", material_counters=counters).items()}
    if isinstance(value, (tuple, list)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in tensors(item, f"{prefix}/{index}", material_counters=counters).items()}
    return {}


# Per-step scratch of the nonlinear E update: build_nonlinear (fdtd/runtime/materials.py) allocates
# these with torch.empty_like, and every step's nonlinear coefficient kernel rewrites them before the
# E update reads them. At setup they hold uninitialized caching-allocator memory, whose content
# depends on allocation history, so G4 neither captures nor compares them. The exact names are
# listed so no real setup tensor can be hidden;
# tests/fdtd/core_path/test_nonlinear_scratch_poison.py proves they are written before being read.
SETUP_SCRATCH_ATTRIBUTES = tuple(f"ce{axis}_{kind}_dynamic" for kind in ("curl", "decay") for axis in "xyz")
SETUP_SCRATCH_KEYS = frozenset(f"setup/{name}" for name in SETUP_SCRATCH_ATTRIBUTES)


def drop_setup_scratch(payload: dict) -> tuple[dict, list[str]]:
    """``payload`` without the G4 setup scratch keys, and the sorted keys that were dropped."""
    dropped = sorted(SETUP_SCRATCH_KEYS & payload.keys())
    return {key: value for key, value in payload.items() if key not in SETUP_SCRATCH_KEYS}, dropped


def setup_scratch_attributes(solver) -> list[str]:
    """The scratch attributes this solver allocated (the nonlinear ones are None otherwise)."""
    return [name for name in SETUP_SCRATCH_ATTRIBUTES if isinstance(getattr(solver, name, None), torch.Tensor)]


def setup_tensors(solver, *, material_counters: list | None = None) -> dict[str, torch.Tensor]:
    payload = {
        name: value for name, value in vars(solver).items()
        if name.startswith(("eps_", "sigma_e_", "sigma_m_", "mu_", "cex_", "cey_", "cez_",
                            "chx_", "chy_", "chz_", "inv_", "cpml_", "psi_"))
        and isinstance(value, torch.Tensor) and name not in SETUP_SCRATCH_ATTRIBUTES
    }
    for name in ("_compiled_material_model", "_coefficient_uniformity", "_compiled_sources",
                 "_source_terms", "_electric_source_terms", "_magnetic_source_terms", "observers",
                 "_point_observer_groups", "_plane_observer_groups", "_observer_spectral_entries"):
        payload[name] = getattr(solver, name)
    return tensors(payload, "setup", material_counters=material_counters)


def drop_material_counters(payload: dict, keys) -> tuple[dict, list[str]]:
    """``payload`` without the material counter ``keys`` a G4 capture listed, and the keys it held."""
    keys = frozenset(keys)
    if any(key.rsplit("/", 1)[-1] not in MATERIAL_COUNTER_ATTRIBUTES for key in keys):
        raise ValueError(f"not material counter keys: {sorted(keys)}")
    return {key: value for key, value in payload.items() if key not in keys}, sorted(keys & payload.keys())


def name_has_nonlinear(name: str) -> bool:
    return name.endswith(("_kerr", "_tpa"))


def physics_tensors(solver, result) -> dict[str, torch.Tensor]:
    """Pin the private surface/wire state and the public port and loss outputs."""
    from witwin.maxwell.fdtd.checkpoint import (
        graph_state_tensors, iter_surface_impedance_state_specs, iter_thin_sheet_state_tensors,
        iter_breakdown_state_tensors, iter_gyromagnetic_state_specs,
    )

    state = dict(iter_surface_impedance_state_specs(solver))
    state.update(iter_gyromagnetic_state_specs(solver))
    state.update(iter_thin_sheet_state_tensors(solver))
    if solver.dispersive_enabled:
        state.update(graph_state_tensors(solver))
    state.update(iter_breakdown_state_tensors(solver))
    if any(getattr(solver, flag, False) for flag in
           ("nonlinear_enabled", "modulation_enabled", "breakdown_enabled")):
        from witwin.maxwell.fdtd.runtime.stepping import _electric_decay_tensors, _electric_curl_tensors
        state.update({f"ce{axis}_{kind}": tensor for kind, values in
                      (("decay", _electric_decay_tensors(solver)), ("curl", _electric_curl_tensors(solver)))
                      for axis, tensor in zip("xyz", values)})
        for name in ("_modulation_step", "_modulation_time", "_nonlinear_peak"):
            value = getattr(solver, name, None)
            if value is not None:
                state[name] = value
    wire = getattr(solver, "_wire_runtime", None)
    if wire is not None:
        state.update(wire_current=wire.current, wire_charge=wire.charge)
    if isinstance(result, SimpleNamespace):
        # partial_result (half-horizon snapshots) carries the finalized outputs directly.
        outputs = {"conductor_losses": result.conductor_losses, "circuits": result.circuits,
                   "embedded_networks": result.embedded_networks, "breakdown": result.breakdown_data}
    else:
        outputs = {"conductor_losses": mw.rf.conductor_losses(result), "circuits": mw.circuits.circuits(result),
                   "embedded_networks": mw.rf.embedded_networks(result),
                   "breakdown": mw.breakdown.breakdown_data(result)}
    return tensors(state, "physics") | tensors({"ports": result.ports, **outputs}, "outputs")


def ade_tensors(solver):
    from witwin.maxwell.fdtd.checkpoint import (
        dispersive_state_name, iter_dispersive_state_specs, iter_magnetic_dispersive_state_specs,
    )

    return tensors({
        dispersive_state_name(component, model, index, name): entry[name]
        for specs in (iter_dispersive_state_specs, iter_magnetic_dispersive_state_specs)
        for component, model, index, names, entry in specs(solver) or ()
        for name in names
    }, "physics")


def result_tensors(result) -> dict[str, torch.Tensor]:
    """Phasor fields and monitor payloads in the frozen-store (0.6) layout."""
    if isinstance(result, SimpleNamespace):
        fields = result.fields
        monitors = {name: api_dialect.legacy_payload(payload) for name, payload in result.monitors.items()}
    else:
        # Without a full-field DFT (float64 cases) the stored fields are the last-step time
        # snapshot, already pinned as fields/E*; phasors exist only with frequency-domain fields.
        if result._fields_are_time_snapshot():
            fields = {}
        elif result.scene.coordinates is not None:
            # Mapped results expose their stored covariant arrays only through the explicit chart frame.
            chart = mw.coordinates.chart_fields(result)
            fields = {name.upper(): chart.tensor(label) for label, name in chart._NATIVE.items()
                      if label in chart.components}
        else:
            fields = api_dialect.fields(result)
        monitors = api_dialect.monitors(result)
    return tensors(fields, "phasors") | tensors(monitors, "monitors")


def export_tensors(result) -> dict[str, torch.Tensor]:
    """Export-time monitor reductions: power-loss channels, component stress and typed dipole emission."""
    exports = {}
    for monitor in result.scene.monitors:
        if isinstance(monitor, mw.esd.ComponentStressMonitor):
            exports[monitor.name] = mw.esd.component_stress(result, monitor.name)
        elif isinstance(monitor, (mw.monitors.PowerLossMonitor, mw.monitors.DipoleEmissionMonitor)):
            exports[monitor.name] = api_dialect.monitor(result, monitor.name)
    return tensors(exports, "exports")


def partial_result(solver):
    """Export stopped-state observations using the ordinary runtime finalizers."""
    from witwin.maxwell.fdtd.runtime.stepping import (
        finalize_circuit_data, finalize_embedded_networks, finalize_port_data, finalize_wire_data,
    )
    from witwin.maxwell.fdtd.copper_foil import copper_loss_report
    from witwin.maxwell.fdtd.runtime.breakdown import finalize_breakdown_data

    if solver.dft_enabled:
        solver._sync_dft_legacy_state()
    if solver.observers_enabled:
        solver._sync_observer_legacy_state()
    monitors = solver.get_observer_results() if solver.observers_enabled else {}
    if getattr(solver, "time_observers_enabled", False):
        monitors = dict(monitors) | solver.get_time_observer_results()
    if getattr(solver, "breakdown_observers_enabled", False):
        monitors = dict(monitors) | solver.get_breakdown_observer_results()
    monitors = dict(monitors) | (finalize_wire_data(solver) or {})
    ports = finalize_port_data(solver) or {}
    return SimpleNamespace(
        fields={key.upper(): value for key, value in solver.get_frequency_solution(all_frequencies=True).items()
                if key in ("Ex", "Ey", "Ez")} if solver.dft_enabled else {},
        monitors=monitors, ports=ports, conductor_losses=copper_loss_report(solver),
        circuits=finalize_circuit_data(solver) or {},
        embedded_networks=finalize_embedded_networks(solver, ports) or {},
        breakdown_data=finalize_breakdown_data(solver) if getattr(solver, "breakdown_enabled", False) else None,
    )


def capture_case(args) -> None:
    bounded = json.loads(args.bounded_items.read_text()) if args.bounded_items else {}
    destination = args.output / "candidate" / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    if args.gate == "g2":
        capture_g2(args, destination)
        return
    case = build_core_case(args.scene)
    steps = case.steps if case.steps is not None else args.steps
    snapshot_half = getattr(args, "snapshot_half", False)
    horizon = steps // 2 if snapshot_half else steps
    cache = None
    if case.material_cache is not None:
        cache = mw.fdtd.FDTDMaterialCache()
        if case.material_cache != "miss":
            # Bind the warm-up scene first (the same scene for a hit); its solver is discarded.
            warmup = case if case.cache_warmup is None else replace(case, scene=case.cache_warmup)
            make_simulation(warmup, steps=args.steps, full_field_dft=True, material_cache=cache).prepare()
    simulation = make_simulation(case, steps=args.steps, full_field_dft=True, material_cache=cache)
    from witwin.maxwell.fdtd.runtime import core_tables
    with patch.object(core_tables, "DELAYED_TABLE_BYTES", 0 if args.scene == "o2_delayed_inline" else core_tables.DELAYED_TABLE_BYTES):
        prepared = simulation.prepare()
    solver = prepared.solver
    if cache is not None:
        assert solver._material_cache_info["outcome"] == case.material_cache, (args.scene, solver._material_cache_info)
        assert solver._core_runtime is not None, args.scene
    if case.name in DISPERSIVE_SCENES:
        assert solver.dispersive_enabled, case.name
        if case.name == "magnetic_lorentz_slab":
            assert solver.magnetic_dispersive_enabled, case.name
        else:
            assert solver.electric_dispersive_enabled, case.name
    if case.name.startswith("w3_debye_"):
        # Electric ADE composed with live coefficients: both state machines must be active.
        assert solver.electric_dispersive_enabled, case.name
        assert solver.nonlinear_enabled is name_has_nonlinear(case.name), case.name
        assert solver.modulation_enabled is ("modulation" in case.name), case.name
    if case.name in ("copper_patch", "copper_board_60k"):
        assert solver._thin_sheets and solver._thin_sheets[0].copper is not None
    if case.name in ANISO_SCENES:
        assert solver.full_aniso_enabled, case.name
        assert (solver._aniso_cond_current is not None) is (case.name == "aniso_conductive_dipole"), case.name
        assert (solver._aniso_disp_current is not None) is (case.name == "aniso_debye_periodic"), case.name
        assert solver.uses_cpml is (case.name != "aniso_debye_periodic"), case.name
        assert solver._full_aniso_cpml_overlap is (case.name == "full_tensor_slab"), case.name
    if case.name in FP64_SCENES:
        assert solver.Ex.dtype == torch.float64, case.name
    if case.name in BLOCH_SCENES:
        assert solver.complex_fields_enabled and solver.uses_cpml is (case.name != "bloch_pec_z"), case.name
        assert (solver._tfsf_state or {}).get("provider") == BLOCH_PROVIDERS.get(case.name), case.name
    if case.name in CHART_SCENES:
        assert "coordinate_frame" in solver._compiled_material_model, case.name
        assert solver.Ex.dtype == case.field_dtype, case.name
        assert bool(solver._copper_foils) is ("copper" in case.name), case.name
    if case.name in MONITOR_SCENES:
        # Each gate scene reaches the runtime piece of its monitor type (time and wire observers: after the run).
        if case.name == "power_loss_copper":
            assert solver._copper_foils and solver.electric_dispersive_enabled, case.name
        if case.name == "power_loss_chart":
            assert "coordinate_frame" in solver._compiled_material_model, case.name
        if case.name == "dipole_emission":
            assert any(observer.get("monitor_type") == "dipole_emission" for observer in solver.observers), case.name
    if case.name in BLOCH_SEPARATE_SCENES:
        # Complex fields beside the separate-physics runtime the scene names.
        feature = {"bloch_lumped_elements": "_lumped_element_runtimes", "bloch_loaded_ports": "_port_runtimes",
                   "bloch_copper_foil": "_copper_foils", "bloch_copper_resistor": "_lumped_element_runtimes",
                   "bloch_tfsf_resistor": "_lumped_element_runtimes"}[case.name]
        assert solver.complex_fields_enabled and getattr(solver, feature), case.name
        assert bool(solver._copper_foils) is ("copper" in case.name), case.name
        assert (solver._tfsf_state or {}).get("provider") == BLOCH_PROVIDERS.get(case.name), case.name
    if case.name in FERRITE_SCENES:
        assert solver.gyromagnetic_enabled, case.name
        general = case.name in ("ferrite_oblique_tfsf", "ferrite_mixed_bias")
        assert solver._gyromagnetic_state["general"] is general, case.name
    if case.name in W1_SCENES:
        feature = {
            "w1_copper_patch": "_copper_foils", "w1_thin_sheet_cavity": "_thin_sheets",
            "w1_lumped_port": "_port_runtimes", "w1_coax_port": "_port_runtimes",
            "w1_thin_wire": "_wire_runtime", "w1_cut_surface": "_cut_surface",
            "w1_sheet_feed": "_thin_sheets", "w1_sibc": "surface_impedance_enabled",
            "f3_coarse_11f": "_copper_foils", "f4_dipole_32": "_wire_runtime",
            "w1_circuit": "_circuit_runtimes", "w1_network": "_network_runtimes",
            "w1_terminal_port": "_port_runtimes", "w1_wave_port": "_electric_source_terms",
        }[case.name]
        assert getattr(solver, feature), (case.name, feature)
        if case.name == "w1_sheet_feed":
            assert any(sheet.feeds for sheet in solver._thin_sheets)
    solver._core_runtime.chunk_size = args.chunk_size
    if case.initial_fields:
        for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
            field = getattr(solver, name)
            field.copy_(((torch.arange(field.numel(), device=field.device).reshape(field.shape) % 17) - 8)
                        * (2.0 ** (-20-index)))
    if args.gate == "g4":
        # Sampling indices/weights are normally resolved at solve entry. Build
        # those tables explicitly without advancing any fields for the setup gate.
        solver._prepare_observers(case.frequencies, "none", steps)
        fp32 = case.field_dtype == torch.float32
        if fp32:
            solver.enable_dft(case.frequencies, window_type="none", end_step=steps)
            solver.build_dft_step_tables(steps)
        material_counters = []
        payload = setup_tensors(solver, material_counters=material_counters)
        excluded_setup_scratch = sorted(f"setup/{name}" for name in setup_scratch_attributes(solver))
        assert not SETUP_SCRATCH_KEYS & payload.keys(), case.name
        if solver._core_runtime.sources is not None:
            payload.update(tensors(solver._core_runtime.sources.rows, "tables/source"))
        if fp32:
            payload.update(tensors((solver._dft_weighted_cos_table, solver._dft_weighted_sin_table), "tables/dft"))
    else:
        plots = []
        from witwin.maxwell.fdtd.runtime import stepping
        solver.verbose = case.boundaries
        solver.progress_update_interval = 11
        solver.plot_interval = 13
        def fail_capture(_solver, step, count):
            step()
            raise RuntimeError("Forced core whole-step capture failure after state mutation.")
        from contextlib import nullcontext
        if case.capture_failure and horizon > 0:
            before = {name: getattr(solver, name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
            try:
                with patch.object(solver._core_runtime, "capture_chunk", side_effect=fail_capture):
                    if snapshot_half:
                        simulation._execute_fdtd_solve(solver, stop_step=horizon)
                    else:
                        prepared.run()
            except RuntimeError as error:
                assert str(error) == "Forced core whole-step capture failure after state mutation.", case.name
            else:
                raise AssertionError(f"{case.name}: the forced capture failure did not raise")
            assert all(torch.equal(getattr(solver, name), value) for name, value in before.items()), case.name
            assert not solver._core_step_graph_active, case.name
            prepared = simulation.prepare()
            solver = prepared.solver
            solver._core_runtime.chunk_size = args.chunk_size
            solver.verbose = case.boundaries
            solver.progress_update_interval = 11
            solver.plot_interval = 13
        failure = nullcontext()
        # The observer engine is dropped after the run; record whether solve entry built one.
        from witwin.maxwell.fdtd import observers as observer_module
        built_engine = []
        make_engine = observer_module.make_single_observer_engine
        def record_engine(_solver, **kwargs):
            run = make_engine(_solver, **kwargs)
            built_engine.append(_solver._single_observer_engine is not None)
            return run
        with failure, patch.object(stepping, "visualize_slice", side_effect=lambda _solver, n: plots.append(n)), \
                patch.object(observer_module, "make_single_observer_engine", side_effect=record_engine):
            if steps == 0:
                # Stop before the first step, preserving the ordinary no-export
                # contract for configured monitors with no recorded samples.
                solver.solve(time_steps=1, stop_step=0, dft_frequency=case.frequencies,
                             dft_window="none", use_cuda_graph=True)
                result = None
            elif snapshot_half:
                # Preserve the full-horizon tables and normalization; stop a fresh
                # solve at half rather than changing the configured run length.
                simulation._execute_fdtd_solve(solver, stop_step=horizon)
                result = partial_result(prepared.solver)
            else:
                result = prepared.run()
        payload = tensors({name + suffix: getattr(prepared.solver, name + suffix)
                           for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                           for suffix in ("", "_imag") if hasattr(prepared.solver, name + suffix)}, "fields")
        if result is not None:
            payload.update(result_tensors(result))
            if case.name in PHYSICS_SCENES:
                payload.update(physics_tensors(solver, result))
            if case.name in MONITOR_SCENES + BLOCH_SEPARATE_SCENES and not isinstance(result, SimpleNamespace):
                payload.update(export_tensors(result))
            if case.name in BLOCH_SEPARATE_SCENES:
                # The complete physical state (both banks of every lumped branch and copper foil).
                from witwin.maxwell.fdtd.checkpoint import graph_state_tensors
                payload.update(tensors(graph_state_tensors(solver), "state/graph"))
        if case.name in ("component_stress_port", "time_surface_aux_debye"):
            assert solver.time_observers_enabled, case.name
        if case.name == "wire_monitor":
            assert solver._wire_runtime.monitor_state, case.name
            if case.name == "breakdown_slab" and not snapshot_half:
                assert (solver.breakdown_runtime["trigger_step"] >= 0).any().item()
                assert solver.breakdown_runtime["total_dissipated"].item() > 0
            if getattr(args, "half_horizon", False):
                payload.update(ade_tensors(solver))
        payload.update(tensors(solver._dft_batched_fields, "dft_accumulators"))
        payload.update(tensors({"shutoff": solver._shutoff_triggered, "shutoff_step": solver._shutoff_step,
            # The pre-row-13 complex DFT was host driven (no device counter); its accumulators are pinned.
            "plots": plots, "dft_step": solver._dft_step if solver.dft_enabled and not solver.complex_fields_enabled
            else None,
            "observer_entries": solver._observer_spectral_entries,
            "port_samples": [(r.sample_index, r.accumulator._sample_count,
                              r.drive_accumulator._sample_count if r.drive_accumulator is not None else None)
                             for r in getattr(solver, "_port_runtimes", ())]}, "state"))
        # Guarded closed-transverse CPML scenes also pin the sampled guard state.
        guard = getattr(solver, "_passivity_state", None)
        if guard is not None:
            payload.update(tensors({key: guard[key] for key in (
                "history", "peak", "window_max", "sample_count", "window_count", "start_step")}, "passivity"))
        if solver.tfsf_enabled and "auxiliary_grid" in solver._tfsf_state:
            aux = solver._tfsf_state["auxiliary_grid"]
            if hasattr(aux, "time_step"):
                payload.update(tensors({"electric": aux.electric, "magnetic": aux.magnetic, "step": aux.time_step},
                                       "auxiliary"))
            else:
                # Complex fixed-k line: its full state and the step its clock reached.
                payload.update(tensors({name: tensor for name, tensor in aux.state_tensors().items() if name != "clock"}
                                       | {"step": int(aux.clock.reshape(-1)[0])}, "auxiliary"))
        from witwin.maxwell.fdtd.runtime.core_path import core_path_step_loop_blockers
        from witwin.maxwell.fdtd.runtime.core_tables import source_tables_supported
        active = getattr(solver, "_core_step_graph_active", False)
        if case.name in (W2_SCENES + W3_SCENES + DISPERSIVE_SCENES + ANISO_SCENES + FERRITE_SCENES + FP64_SCENES
                         + CHART_SCENES):
            assert active is (horizon > 0), case.name
        assert active is (horizon > 0 and not core_path_step_loop_blockers(solver)
                          and source_tables_supported(solver)
                          and not solver.time_observers_enabled
                          and (not solver.observers_enabled or all(built_engine)
                               or observer_module.uses_indexed_observers(solver))), (case.name, active, built_engine)
        if active:
            assert int(solver._core_runtime.step.item()) == (solver._shutoff_step + 1 if solver._shutoff_triggered else horizon)
        if case.shutoff and not snapshot_half:
            assert solver._shutoff_triggered and solver._shutoff_step + 1 < steps
    if case.name in DISPERSIVE_SCENES and not snapshot_half:
        from witwin.maxwell.fdtd.checkpoint import (
            capture_checkpoint_state, graph_state_tensors, restore_checkpoint_state,
        )
        from witwin.maxwell.fdtd.resume import _configuration_fingerprint

        checkpoint = capture_checkpoint_state(solver, step=steps)
        path = destination / "physics-checkpoint.pt"
        torch.save(checkpoint, path)
        live = graph_state_tensors(solver)
        ade_names = checkpoint.schema.dispersive_state_names + checkpoint.schema.magnetic_dispersive_state_names
        for name in ade_names:
            live[name].zero_()
        restore_checkpoint_state(solver, torch.load(path, weights_only=False))
        for name in ade_names:
            assert torch.equal(live[name], checkpoint.tensors[name]), name
        payload["resume/fingerprint"] = torch.tensor(
            list(bytes.fromhex(_configuration_fingerprint(solver))), dtype=torch.uint8)
    torch.save(payload, destination / ("tensors_half.pt" if snapshot_half else "tensors.pt"))
    (destination / ("capture_half.json" if snapshot_half else "capture.json")).write_text(json.dumps({
        "gate": args.gate, "arm": "candidate", "steps": steps,
        "process_id": os.getpid(),
        "core_runtime_active": hasattr(prepared.solver, "_core_runtime"),
        "whole_step_capture_active": getattr(solver, "_core_step_graph_active", False),
        "whole_step_capture_k": getattr(solver, "_core_step_graph_k", None),
        "requested_chunk_size": args.chunk_size,
        "whole_step_chunks": getattr(solver, "_core_step_graph_chunks", []),
        "launches_per_step": (2 + sum(map(len, solver._core_runtime.sources.groups.values()))
            + (4 if solver._core_runtime.sources.auxiliary is not None else 0)
            + int(solver.dft_enabled) + 1
            + (1 + len(solver._core_runtime.observer_groups) if hasattr(solver._core_runtime, "observer_groups") else 0)
            if getattr(solver, "_core_step_graph_active", False) and case.name not in W1_SCENES + W2_SCENES + W3_SCENES else None),
        "tensor_names": sorted(payload), "scene": case.plan(), "bounded_items": bounded,
        **({"material_cache": solver._material_cache_info} if cache is not None else {}),
        **({"horizon_steps": horizon} if getattr(args, "half_horizon", False) else {}),
        **({"excluded_setup_scratch": excluded_setup_scratch} if args.gate == "g4" else {}),
        **({"excluded_material_counters": sorted(material_counters)}
           if args.gate == "g4" and material_counters else {}),
    }, indent=2) + "\n")


def source_patch_gather_reference(field_shape, patch_shape, offsets, *, device, wrap=0):
    """Enumerate clipped entries and periodic images independently in C order."""
    contributions = []
    for entry, local in enumerate(np.ndindex(tuple(patch_shape))):
        coords = tuple(local[axis] + offsets[axis] for axis in range(3))
        if any(coord < 0 or coord >= size for coord, size in zip(coords, field_shape)):
            continue
        boundary = any(wrap & (1 << axis) and coord in (0, size - 1)
                       for axis, (coord, size) in enumerate(zip(coords, field_shape)))
        for subset in range(8):
            if subset & ~wrap:
                continue
            target = list(coords)
            for axis, size in enumerate(field_shape):
                if subset & (1 << axis):
                    if coords[axis] not in (0, size - 1):
                        break
                    target[axis] = size - 1 if coords[axis] == 0 else 0
            else:
                flat = (target[0] * field_shape[1] + target[1]) * field_shape[2] + target[2]
                contributions.append((0, flat, entry, subset | (int(boundary) << 3)))
    # A patch has component code zero; Python sorting retains entry/subset order.
    contributions.sort(key=lambda item: item[:2])
    starts, targets, entries = [], [], []
    for index, (component, target, entry, flags) in enumerate(contributions):
        if not targets or targets[-1] != (component, target):
            starts.append(index)
            targets.append((component, target))
        entries.append((entry, flags))
    starts.append(len(entries))
    return (torch.tensor(starts, dtype=torch.int64, device=device),
            torch.tensor(targets, dtype=torch.int64, device=device).reshape(-1, 2),
            torch.tensor(entries, dtype=torch.int64, device=device).reshape(-1, 2))


def source_table_reference(solver, steps, *, budget=64*1024*1024):
    """Independent scalar oracle plus the unchanged delayed-source CUDA evaluator."""
    from witwin.maxwell.sources import evaluate_source_time
    from witwin.maxwell.fdtd.excitation.temporal import _resolve_cw_signal
    def upload(values):
        return torch.tensor(np.asarray(values, dtype=np.float32), device=solver.device)
    rows = {"time_h": upload([n*solver.dt for n in range(steps)]),
            "time_e": upload([n*solver.dt+0.5*float(solver.dt) for n in range(steps)])}
    # Additive electric currents inject at (n+1/2)dt since the source-time-label change.
    groups = [("ordinary", solver._source_terms, True, False),
              ("magnetic", solver._magnetic_source_terms, False, True),
              ("electric", solver._electric_source_terms, True, True)]
    aux = solver._tfsf_state.get("auxiliary_grid") if solver.tfsf_enabled else None
    if aux is not None and solver._tfsf_state["provider"] == "plane_wave_fixed_k":
        pass  # The fixed-k line stages its own drive rows (prepare_tables at solve entry).
    elif aux is not None:
        rows["auxiliary/signals"] = upload([evaluate_source_time(aux.source_time, (aux.time_step+n+1)*aux.dt) for n in range(steps)])
        if aux.incident_reference is not None:
            rows["auxiliary/boundary_signals"] = upload([
                [evaluate_source_time(aux.source_time, (aux.time_step+n+k+1)*aux.dt) for k in range(aux.source_index)]
                for n in range(steps)])
    elif solver.tfsf_enabled:
        groups += [("tfsf_magnetic", solver._tfsf_state["magnetic_terms"], False, True),
                   ("tfsf_electric", solver._tfsf_state["electric_terms"], True, True)]
    bloch = solver.scene.scene.boundary.uses_kind("bloch")
    for name, terms, half, generic in groups:
        for ordinal, term in enumerate(terms):
            source = term.get("source_time") or solver._source_time
            times = [n*solver.dt+(0.5*float(solver.dt) if half else 0.0) for n in range(steps)]
            if bloch:
                # Complex Bloch scatter: per-step (real, imaginary) signal rows and the wrap-image gather.
                omega = float(solver.source_omega if term.get("omega") is None else term["omega"])
                if term["cw_cos_patch"] is not None:
                    values, patch = [_resolve_cw_signal(omega, source, t) for t in times], term["cw_cos_patch"]
                elif term["delay_patch"] is not None:
                    values, patch = None, term["patch"]  # the device envelope reads the step's time
                else:
                    pr, pi, patch = float(term["phase_real"]), float(term["phase_imag"]), term["patch"]
                    values = []
                    for t in times:
                        signal = float(evaluate_source_time(source, t))
                        quadrature = float(_resolve_cw_signal(omega, source, t)[1]) if source["kind"] == "cw" else 0.0
                        values.append((signal*pr + quadrature*pi, signal*pi - quadrature*pr))
                rows[f"{name}/{ordinal}/signals"] = upload(times) if values is None else upload(values).reshape(steps, 2)
                component = "xyz".index(term["field_name"][-1].lower())
                axes = [axis for axis in range(3) if axis != component]
                flags = term.get("periodic_wrap_axes", tuple(
                    int(solver.scene.scene.boundary.axis_kind("xyz"[axis]) in ("bloch", "periodic")) for axis in axes))
                gather = source_patch_gather_reference(
                    getattr(solver, term["field_name"]).shape, patch.shape, term["offsets"], device=solver.device,
                    wrap=sum(1 << axis for axis, flag in zip(axes, flags) if flag))
                for key, tensor in zip(("gather_rows", "gather_targets", "gather_entries"), gather):
                    rows[f"{name}/{ordinal}/{key}"] = tensor
                continue
            if term["cw_cos_patch"] is not None:
                omega = solver.source_omega if term.get("omega") is None else term["omega"]
                signals = upload([_resolve_cw_signal(omega, source, t) for t in times]).reshape(steps, 2)
            elif term["delay_patch"] is not None:
                bits, indices = torch.unique(term["delay_patch"].view(torch.int32), return_inverse=True)
                delays = bits.view(torch.float32)
                size = steps*delays.numel()*4
                if size > budget:
                    signals = torch.empty(0, device=solver.device)
                else:
                    budget -= size
                    rows[f"{name}/{ordinal}/unique_delays"] = delays
                    rows[f"{name}/{ordinal}/delay_indices"] = indices.reshape(-1)
                    field = torch.zeros((delays.numel(), 1, 1), device=solver.device)
                    ones = torch.ones_like(field)
                    delay_patch = delays.reshape_as(field)
                    signals = torch.empty((steps, delays.numel()), device=solver.device)
                    for n, time_value in enumerate(times):
                        field.zero_()
                        solver.fdtd_module.addTimeShiftedSourcePatch3D(field=field, sourcePatch=ones,
                            delayPatch=delay_patch, activationDelayPatch=delay_patch,
                            offsetI=0, offsetJ=0, offsetK=0, timeKind=int(source["kind_code"]), time=time_value,
                            frequency=float(source["frequency"]), fwidth=float(source["fwidth"]),
                            amplitude=float(source["amplitude"]), phase=float(source["phase"]), delay=float(source["delay"]),
                            pedestal=float(source.get("pedestal", 0)), pedestalFwidth=float(source.get("pedestal_fwidth", 0)),
                            causalGate=0).launchRaw()
                        signals[n].copy_(field.reshape(-1))
            else:
                phase = float(term["phase_real"]) if not generic else 1.0
                if generic and solver.scene.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
                    phase = float(term["phase_real"])
                signals = upload([evaluate_source_time(source, t)*phase for t in times])
                if solver.scene.scene.boundary.uses_kind("periodic") and not term.get("periodic_compiled", False):
                    component = "xyz".index(term["field_name"][-1].lower())
                    wrap = sum(1 << axis for axis, name_axis in enumerate("xyz")
                               if axis != component and solver.scene.scene.boundary.axis_kind(name_axis) == "periodic")
                    if wrap:
                        gather = source_patch_gather_reference(
                            getattr(solver, term["field_name"]).shape, term["patch"].shape,
                            term["offsets"], device=solver.device, wrap=wrap)
                        for key, tensor in zip(("gather_rows", "gather_targets", "gather_entries"), gather):
                            rows[f"{name}/{ordinal}/{key}"] = tensor
            rows[f"{name}/{ordinal}/signals"] = signals
    return rows


SCORED = (
    "field_l2", "field_shape_l2", "field_linf", "field_corr", "flux_error",
    "field_runtime_change", "flux_runtime_change", "per_frequency", "scalar_metrics",
)


def metric_row(row) -> dict:
    scores = {name: getattr(row, name) for name in SCORED}
    # Match the existing report's strict field/flux thresholds and independent
    # convergence exclusions. Preserve the convergence flags as part of status.
    return {"scores": scores, "status": {
        "field_converged": row.field_converged, "flux_converged": row.flux_converged,
        "field_l2_pass": None if row.field_converged is False else row.field_l2 < 0.1,
        "field_linf_pass": None if row.field_converged is False else row.field_linf < 0.1,
        "field_corr_pass": None if row.field_converged is False else row.field_corr > 0.99,
        "flux_pass": None if row.flux_converged is False or row.flux_error is None
        else row.flux_error < 0.05,
    }}


def capture_g2(args, destination: Path) -> None:
    from benchmark import paths, report, runner

    os.environ["WITWIN_BENCHMARK_NO_CLOUD"] = "1"
    original_fdtd = mw.Simulation.fdtd
    original_run = runner._run_maxwell
    captured = {}
    core_runtime_active = []
    capture_flags = []

    def fdtd(*positional, **keywords):
        simulation = original_fdtd(*positional, **keywords)
        original_prepare = simulation.prepare

        def prepare(*positional, **keywords):
            prepared = original_prepare(*positional, **keywords)
            core_runtime_active.append(hasattr(prepared.solver, "_core_runtime"))
            prepared.solver._core_runtime.chunk_size = args.chunk_size
            original_prepared_run = prepared.run
            def prepared_run(*positional, **keywords):
                # The benchmark probes dt with a one-step preparation, then
                # resolves its horizon. Freeze a fresh preparation at that horizon
                # so each solver uploads its source rows exactly once.
                if prepared.solver._time_plan["total_steps"] != simulation.config.run_time.time_steps:
                    prepared.solver = original_prepare().solver
                    prepared.solver._core_runtime.chunk_size = args.chunk_size
                result = original_prepared_run(*positional, **keywords)
                capture_flags.append({"active": getattr(prepared.solver, "_core_step_graph_active", False),
                                      "k": getattr(prepared.solver, "_core_step_graph_k", None)})
                return result
            prepared.run = prepared_run
            return prepared

        simulation.prepare = prepare
        return simulation

    def run(*positional, **keywords):
        output = original_run(*positional, **keywords)
        result = output[0]
        index = str(len(captured))
        captured[index] = result_tensors(result)
        return output

    # Redirect every generated report/plot; cached references remain read-only.
    with patch.object(paths, "CACHE_DIR", REFERENCE_CACHE), \
         patch.object(paths, "PLOTS_DIR", destination / "plots"), \
         patch.object(paths, "RESULTS_MD", destination / "results.md"), \
         patch.object(report, "RESULTS_MD", destination / "results.md"), \
         patch.object(runner, "_run_maxwell", run), \
         patch.object(mw.Simulation, "fdtd", side_effect=fdtd):
        row, = runner.run_benchmarks([args.scene])
    (destination / "metrics.json").write_text(json.dumps(metric_row(row), indent=2) + "\n")
    (destination / "capture.json").write_text(json.dumps({
        "gate": args.gate, "arm": "candidate", "core_runtime_active": core_runtime_active,
        "whole_step_capture": capture_flags,
        "requested_chunk_size": args.chunk_size,
    }, indent=2) + "\n")
    torch.save(tensors(captured, "runs"), destination / "tensors.pt")


def differences(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    assert reference.shape == candidate.shape
    assert reference.dtype == candidate.dtype
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    a = reference.to(torch.complex128 if reference.is_complex() else torch.float64)
    b = candidate.to(a.dtype)
    delta = (a - b).abs()
    relative = torch.where(delta == 0, 0.0, delta / a.abs())
    return {"equal": torch.equal(reference, candidate),
            "max_abs": delta.max().item() if delta.numel() else 0.0,
            "max_rel": relative.max().item() if delta.numel() else 0.0}


def compare(args) -> None:
    mode = getattr(args, "equivalence", "bitwise")
    if mode == "rule2" and args.gate not in ("g1", "g4"):
        raise ValueError("Rule 2 is supported only for G1 and G4")
    names = args.scenes or (G2_SCENES if args.gate == "g2" else GATE_SCENES)
    bounded = json.loads((args.output / "bounded-items.json").read_text()) if args.gate == "g1-prime" else {}
    if args.bounded_items:
        assert json.loads(args.bounded_items.read_text()) == bounded
    seen = set()
    results = {}
    excluded = {}
    material_excluded = {}
    passed = True
    baseline_root = getattr(args, "baseline", None) or frozen_baseline(args.gate, rule2=mode == "rule2")
    roots = {"baseline": baseline_root, "candidate": args.output / "candidate"}
    for name in names:
        rebuilt = args.gate in ("g1", "g1-prime") and name in REBUILT_BASELINES
        # A cache scene pins the payloads of its base scene.
        scene = REBUILT_BASELINES[name] if rebuilt else CACHE_SCENES.get(name, name)
        if getattr(args, "baseline", None) is None:
            from tests.support import frozen_oracles
            baseline_directory = frozen_oracles.scene_directory(baseline_root.name, scene)
        else:
            baseline_directory = roots["baseline"] / scene
        directories = {"baseline": baseline_directory,
                       "candidate": roots["candidate"] / name}
        metadata_by_arm = {}
        for arm in ("baseline", "candidate"):
            metadata = json.loads((directories[arm] / "capture.json").read_text())
            assert metadata["gate"] == (args.gate if arm == "candidate" or args.gate != "g1-prime" else "g1")
            active = metadata["core_runtime_active"]
            activations = active if args.gate == "g2" else [active]
            assert activations and all(value is (arm == "candidate") for value in activations), (name, arm, active)
            metadata_by_arm[arm] = metadata
        if args.gate == "g2":
            before = json.loads((roots["baseline"] / name / "metrics.json").read_text())
            after = json.loads((roots["candidate"] / name / "metrics.json").read_text())
            left, right = tensors(before["scores"], "scores"), tensors(after["scores"], "scores")
            left.update(torch.load(roots["baseline"] / name / "tensors.pt", weights_only=True, map_location="cpu"))
            right.update(torch.load(roots["candidate"] / name / "tensors.pt", weights_only=True, map_location="cpu"))
            assert before["status"] == after["status"], name
            # Check the complete score schema, including nonnumeric leaves.
            assert score_schema(before["scores"]) == score_schema(after["scores"]), name
        else:
            assert metadata_by_arm["baseline"]["bounded_items"] == {}
            assert metadata_by_arm["candidate"]["bounded_items"] == bounded
            assert metadata_by_arm["baseline"]["steps"] == metadata_by_arm["candidate"]["steps"]
            left = torch.load(directories["baseline"] / "tensors.pt", weights_only=True, map_location="cpu")
            right = torch.load(directories["candidate"] / "tensors.pt", weights_only=True, map_location="cpu")
        if args.gate == "g4":
            # Uninitialized scratch at setup (SETUP_SCRATCH_KEYS): ignored on both arms, also when
            # an older capture (the frozen store) still holds it.
            (left, dropped_left), (right, dropped_right) = drop_setup_scratch(left), drop_setup_scratch(right)
            if dropped_left or dropped_right:
                excluded[name] = {"baseline": dropped_left, "candidate": dropped_right}
            # Material counter labels (MATERIAL_COUNTER_ATTRIBUTES) the candidate capture left out:
            # dropped on both arms, also from an older capture (the frozen store) that holds them.
            counter_keys = metadata_by_arm["candidate"].get("excluded_material_counters", [])
            if counter_keys:
                (left, held_left), (right, held_right) = (drop_material_counters(left, counter_keys),
                                                          drop_material_counters(right, counter_keys))
                material_excluded[name] = {"listed": counter_keys, "baseline": held_left, "candidate": held_right}
        assert left.keys() == right.keys() and left, name
        if mode == "rule2":
            horizons = {}
            if args.gate == "g1":
                for arm in ("baseline", "candidate"):
                    directory = directories[arm]
                    half_metadata = json.loads((directory / "capture_half.json").read_text())
                    full_metadata = metadata_by_arm[arm]
                    assert half_metadata["gate"] == "g1" and half_metadata["arm"] == arm
                    assert half_metadata["steps"] == full_metadata["steps"]
                    assert half_metadata["horizon_steps"] == full_metadata["steps"] // 2
                    assert half_metadata["core_runtime_active"] is (arm == "candidate")
                    horizons[arm] = torch.load(directory / "tensors_half.pt", weights_only=True, map_location="cpu")
            receipts = rule2_rows(left, right, level="setup" if args.gate == "g4" else "run",
                                  ref_half=horizons.get("baseline"), new_half=horizons.get("candidate"))
            for key in sorted(receipts):
                row = receipts[key]
                ok = row["tier"] != "T2"
                results[f"{name}/{key}"] = dict(row, equal=row["tier"] == "T0", passed=ok)
                passed &= ok
                print(f"{name}/{key}: tier={row['tier']} abs={row['max_abs']:.6g} "
                      f"bound={row['bound']:.6g} pass={ok}")
            continue
        for key in sorted(left):
            item = f"{name}/{key}"
            diff = differences(left[key], right[key])
            if item in bounded:
                entry = bounded[item]
                assert entry["reason"].strip()
                tolerance = {"fields": 1e-6, "phasors": 1e-5, "flux": 1e-5}[entry["kind"]]
                ok = diff["max_rel"] <= tolerance
                diff.update(reason=entry["reason"], tolerance=tolerance)
                seen.add(item)
            else:
                ok = diff["equal"]
            diff["passed"] = ok
            results[item] = diff
            passed &= ok
            print(f"{item}: equal={diff['equal']} abs={diff['max_abs']:.6g} rel={diff['max_rel']:.6g} pass={ok}")
    assert seen == bounded.keys(), "Every pre-named item must be compared"
    receipt = {"passed": passed, "items": results}
    if excluded:
        # Only present when scratch was dropped, so other receipts stay byte-identical.
        receipt["excluded_setup_scratch"] = excluded
    if material_excluded:
        receipt["excluded_material_counters"] = material_excluded
    if mode == "rule2":
        receipt["summary"] = tier_summary(results)
    (args.output / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


def _load_snapshot(path: Path) -> dict:
    """Load a capture file as a flat name -> tensor mapping, safely (weights_only=True).

    tensors*.pt are plain dicts. physics-checkpoint.pt (dispersive scenes) holds an
    FDTDCheckpointState: its tensors are compared, and its step and schema enter as tensors so a
    drift in either fails the exact comparison too.
    """
    from witwin.maxwell.fdtd.checkpoint import FDTDCheckpointSchema, FDTDCheckpointState

    with torch.serialization.safe_globals([FDTDCheckpointState, FDTDCheckpointSchema]):
        value = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(value, FDTDCheckpointState):
        schema = json.dumps(asdict(value.schema), sort_keys=True).encode()
        return {**{f"tensors/{name}": tensor for name, tensor in value.tensors.items()},
                "step": torch.tensor(value.step, dtype=torch.int64),
                "schema": torch.tensor(list(schema), dtype=torch.uint8)}
    return value


def determinism(args):
    """Compare complete snapshot manifests from independent capture directories."""
    directories = [Path(directory).resolve() for directory in args.dirs]
    if len(directories) < 3 or len(set(directories)) != len(directories):
        raise ValueError("Determinism requires at least 3 distinct independent capture directories")
    arm = args.arm
    roots = [directory / arm for directory in directories]
    manifests = [{path.relative_to(root) for path in root.rglob("*.pt")} for root in roots]
    assert manifests[0] and all(manifest == manifests[0] for manifest in manifests), "Missing snapshot files"
    assert any(path.name == "tensors.pt" for path in manifests[0]), "Missing full snapshots"
    assert all(path.with_name("tensors.pt") in manifests[0] for path in manifests[0]), "Missing full snapshots"
    results = {}
    for filename in sorted(manifests[0]):
        metadata_name = {"tensors.pt": "capture.json", "tensors_half.pt": "capture_half.json"}.get(filename.name)
        metadata = None
        if metadata_name is not None:
            metadata = [json.loads((root / filename.with_name(metadata_name)).read_text()) for root in roots]
            processes = [record.get("process_id") for record in metadata]
            if any(process is not None for process in processes):
                assert all(type(process) is int and process > 0 for process in processes), "Missing capture process IDs"
                assert len(set(processes)) == len(processes), "Captures must come from independent fresh processes"
        snapshots = [_load_snapshot(root / filename) for root in roots]
        if metadata is not None and metadata[0].get("gate") == "g4":
            snapshots = [drop_setup_scratch(snapshot)[0] for snapshot in snapshots]
        reference = snapshots[0]
        assert reference and all(value.keys() == reference.keys() for value in snapshots), (filename, "Missing tensors")
        for key in sorted(reference):
            a = reference[key]
            equal = all(a.shape == value[key].shape and a.dtype == value[key].dtype
                        and torch.equal(a, value[key]) for value in snapshots[1:])
            results[f"{filename}/{key}"] = {"equal": equal, "passed": equal}
    receipt = {"passed": all(row["passed"] for row in results.values()), "items": results,
               "arm": arm, "dirs": [str(directory) for directory in directories]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "determinism.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise SystemExit(1)
    print(f"DETERMINISM: {len(directories)} captures, {len(results)} tensors, passed")
    return receipt


def score_schema(value):
    if isinstance(value, dict):
        return {key: score_schema(item) for key, item in value.items()}
    if isinstance(value, list):
        return [score_schema(item) for item in value]
    return "number" if type(value) in (int, float) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "compare", "worker", "determinism"))
    parser.add_argument("--gate", choices=("g1", "g1-prime", "g2", "g4"), default="g1")
    parser.add_argument("--scenes", nargs="+", choices=GATE_SCENES)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, help="baseline capture directory (default: the frozen store)")
    parser.add_argument("--bounded-items", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm", choices=("baseline", "candidate"), help="determinism capture arm")
    parser.add_argument("--equivalence", choices=("bitwise", "rule2"), default="bitwise")
    parser.add_argument("--half-horizon", action="store_true")
    parser.add_argument("--snapshot-half", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dirs", nargs="+", type=Path)
    parser.add_argument("--scene", choices=GATE_SCENES)
    parser.add_argument("--chunk-size", type=int, choices=(1, 16, 32), default=32)
    args = parser.parse_args()
    args.arm = args.arm or "candidate"
    if args.action == "determinism":
        if not args.dirs:
            parser.error("determinism requires --dirs with at least 3 capture directories")
        determinism(args)
        return
    if (args.half_horizon or args.snapshot_half) and args.gate != "g1":
        parser.error("half-horizon snapshots are supported only for G1")
    names = args.scenes or (G2_SCENES if args.gate == "g2" else GATE_SCENES)
    if args.gate == "g2":
        assert set(names) <= set(G2_SCENES)
    if args.dry_run:
        for name in names:
            print(json.dumps(build_core_case(name).plan()))
        print(json.dumps({"gate": args.gate, "action": args.action, "fresh_interpreters": True,
                          "arms": ["candidate"], "baseline": "frozen existing-path store", "steps": args.steps,
                          "comparison": "separate compare invocation", "output": str(args.output)}))
    elif args.action == "worker":
        capture_case(args)
    elif args.action == "compare":
        compare(args)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        policy_args = []
        if args.gate == "g1-prime":
            bounded = json.loads(args.bounded_items.read_text()) if args.bounded_items else {}
            policy = args.output / "bounded-items.json"
            policy.write_text(json.dumps(bounded, indent=2) + "\n")
            policy_args = ["--bounded-items", str(policy)]
        for name in names:
            for half in ((False, True) if args.half_horizon else (False,)):
                subprocess.run([
                    sys.executable, "-m", "benchmark.core_path.gates", "worker",
                    "--gate", args.gate, "--scene", name,
                    "--steps", str(args.steps), "--output", str(args.output),
                    "--chunk-size", str(args.chunk_size),
                    *(["--half-horizon"] if args.half_horizon else []),
                    *(["--snapshot-half"] if half else []),
                    *policy_args,
                ], check=True, env=os.environ | THREAD_ENV | {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                })


if __name__ == "__main__":
    main()

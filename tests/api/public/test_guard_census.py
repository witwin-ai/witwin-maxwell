"""Guard-convergence gate for the P5 functional-completeness plan.

Walks every ``raise NotImplementedError`` in ``witwin/`` via AST, subtracts the
committed contract-guard exclusion list (docs/reference/fdtd-capability-guard-census.md),
and fails when the remaining *capability* guard count exceeds the committed
budget. Each P5 phase lowers ``CAPABILITY_GUARD_BUDGET`` in the same commit
that removes guards, so the metric cannot silently regress or be gamed by
reclassifying capability guards as contracts without touching this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "witwin"

# This exact snapshot was reconciled on 2026-07-17 by integrating the array
# basis / active-S feature series (plan 06, phases 0-1) on top of the thin-wire
# integration baseline of 129. The 129 baseline was the 2026-07-15 base of 108
# plus the SPICE/MNA circuit co-simulation series (+5), the Touchstone
# network-embedding series (+6, giving 119), and the thin-wire subgrid conductor
# series (plan 07, phases 0-3, +10, giving 129). The array-basis series adds one
# capability guard (postprocess/array.py: "Array basis extraction is FDTD-only.")
# -- a genuine capability gap, since non-FDTD backends do not retain the
# full-wave PortSweep columns array_basis() consumes -- giving 129 + 1 = 130.
# The seventh network-embedding guard, simulation.py "Embedded network feedback
# is defined only for the time-domain FDTD update", is a permanent architectural
# boundary and is listed in CONTRACT_GUARDS below, not counted here. Lower it
# when a capability guard is implemented; do not raise it without updating the
# census.
#
# 2026-07-17 (plan 07 Phase 4, multi-GPU wire forward slice): 130 -> 132 (net +2),
# folded in on top of the array-basis 130. The distributed thin-wire FORWARD is
# now implemented, so the single blanket NotImplementedError "Multi-GPU ThinWire
# requires distributed fragment/state ownership..." in fdtd/distributed/solver.py
# is removed (-1). Three narrower capability gaps replace it (+3): a distributed
# thin-wire + CPML boundary, a distributed thin-wire + Mur boundary, and a
# thin-wire scene mixed with an embedded network or lumped circuit, all fail
# closed in DistributedFDTD._validate_distributed_wire_support. The Mur guard is
# the fail-closed integration decision folded into this merge (Mur + wire on the
# split is undocumented and unverified). Finite-conductor loss and distributed
# wire reverse/gradient remain separate slices; lower this budget when those
# guards are implemented.
#
# 2026-07-17 (plan 06 Phase 4, array scene-gradient slice): 132 -> 133 (+1).
# ArrayBasisData.scene_gradient_vjp in witwin/maxwell/array.py fails closed:
# "Scene-parameter gradients through the array basis require the aggregated
# per-column adjoint envelope". The retained-column basis stores detached
# embedded-pattern tensors, so scene/material/geometry backprop through the
# basis is a genuine capability gap (weight gradients through combine() are
# fully supported). It becomes a public promise only after the plan 06 Phase 4
# exit gate lands the aggregated per-column adjoint on the plan 02 Phase 7
# distributed result-aggregation contract; lower this budget then.
#
# 2026-07-17 (plan 07 Phase 4, finite-conductor wire series-impedance slice):
# 133 -> 134 (+1). compiler/thin_wire.py now raises NotImplementedError when a
# ThinWire carries a finite (non-PEC) conductor: the analytic series-impedance
# model and passive ADE fit are implemented (wire_impedance.py), but the lossy
# current recurrence is not yet wired into the FDTD runtime, so the compiler
# fails closed instead of silently dropping the loss. It is a genuine capability
# gap (Material compilers 11 -> 12); lower this budget when the lossy recurrence
# lands in the runtime.
#
# 2026-07-17 (plan 05 nonlinear-device fail-closed hardening): 134 -> 137 (+3).
# The nonlinear device family (diode / behavioral I-V / voltage-dependent
# capacitor) is admitted by Circuit.add and structurally validated by
# compile_circuit_graph, but the executable runtimes carry no Newton loop, so
# three genuine capability gaps now fail closed instead of silently dropping the
# device: (1) compiler/circuits.py reject_nonlinear_devices -- any nonlinear
# device in the linear MNA / coupled / FDTD Norton-companion path (reached from
# compile_mna_system, compile_coupled_mna_system, and scene.compile_circuits);
# (2) compiler/nonlinear_devices.py -- a diode with nonzero series_resistance,
# whose ohmic branch is not assembled into the ideal-Shockley conduction law;
# (3) compiler/nonlinear_devices.py newton_solve -- a diode with nonzero
# junction_capacitance entering the conduction-only DC solve that never
# differentiates the stored charge q(v). All three are Time-domain ports and
# lumped elements gaps (13 -> 16); lower this budget as the nonlinear
# device-runtime slices wire each capability in.
# 2026-07-17 (plan 08 Phase 0, gyromagnetic ferrite slices 0a/0b/1a): 134 -> 136
# (+2 capability). The GyromagneticFerrite material type lands with two fail-closed
# capability guards: compiler/materials.py rejects a ferrite structure ("its
# non-reciprocal off-diagonal permeability is produced by a local linearized-LLG
# magnetization ADE that the material compiler does not yet lower") because
# compiling it as a plain Material would silently discard the gyrotropy, and
# media.py PerturbationMedium rejects a GyromagneticFerrite base (its scalar-eps
# perturbation would silently discard the gyromagnetic state). Both disappear when
# the compiler/runtime slices (1b/1c) land; lower this budget then. The two scalar
# frequency-evaluation guards on GyromagneticFerrite (relative_permeability and
# evaluate_at_frequency) are permanent contracts -- a scalar/diagonal sample cannot
# represent an off-diagonal Polder tensor -- and are listed in CONTRACT_GUARDS.
# Merged union 2026-07-18: 134 + 3 (plan 05) + 2 (plan 08) = 139.
# 2026-07-17 (surface-impedance Phase 0 adapter fail-close): 134 -> 135 (+1).
# adapters/tidy3d.py::_convert_material now raises NotImplementedError for a generic
# SurfaceImpedanceMedium instead of letting it fall through to the non-dispersive
# td.Medium(permittivity=1.0) path and silently export as vacuum. Its physics is a
# broadband causal rational tangential surface impedance with no finite bulk-
# permittivity equivalent, and the reference backend has no surface-impedance mapping
# wired for a rational model yet (the narrowband good-conductor LossyMetalMedium keeps
# its dedicated lossy-metal surface path). It is a genuine capability gap (External
# interoperability adapter 18 -> 19); lower this budget when the surface-impedance
# adapter mapping is designed.
# 2026-07-18 (plan 05 nonlinear-device N1, standalone transient + DC continuation):
# 140 -> 139 (-1). The N0 conduction-only Newton solve fail-closed on a nonzero
# diode junction_capacitance ("... the stored charge q(v) is never differentiated
# into the system ...", compiler/nonlinear_devices.py). N1 implements the trapezoidal/
# backward-Euler stored-charge companion (NonlinearMNASystem charge integration,
# run_nonlinear_transient), so the transient path now consumes q(v) and the DC
# operating point correctly treats the capacitance as an open circuit; the guard is
# removed. The diode series_resistance guard stays: the internal-node ohmic branch is
# not implemented this round.
# 2026-07-18 (plan 08 slices 1b/1c, gyromagnetic ferrite compiler + forward):
# 140 -> 141 (net +1). The compiler/materials.py ferrite reject is REMOVED (-1):
# a GyromagneticFerrite now compiles as its diagonal background (eps_r /
# mu_infinity / sigma_e) and its non-reciprocal off-diagonal permeability is
# produced by the compiled magnetization-ADE layout plus the FDTD gyromagnetic
# forward hooks (fdtd/runtime/gyromagnetic.py), so it no longer silently drops the
# gyrotropy. Two narrower forward-runtime capability gaps replace it (+2), both in
# fdtd/runtime/gyromagnetic.py: (1) a general (non-axis-aligned) bias, or a scene
# mixing bias axes, fails closed because slice 1c advances only the axis-aligned
# z/x/y fast path (the arbitrary-bias local-frame rotation + 4-point Yee
# collocation is Phase 2); (2) a Bloch-periodic ferrite run fails closed because
# the real-valued magnetization-ADE correction would break the complex Bloch phase
# (the magnetic mirror of the existing nonlinear/full-aniso/modulation + Bloch
# guards). Both are genuine capability gaps; lower this budget as the arbitrary-
# bias kernel (Phase 2) and a complex-field ferrite correction land. The
# PerturbationMedium-wraps-a-ferrite reject stays (a scalar-eps perturbation cannot
# represent the gyromagnetic state).
# 2026-07-18 (plan 08 slice 1c fail-closed hardening): 141 -> 144 (net +3). Lifting
# the compiler-level ferrite reject exposed silently-wrong paths in every non-FDTD-
# forward consumer of a ferrite scene; each now fails closed with a narrower guard:
# (1) fdfd/solver.py rejects a GyromagneticFerrite in _ensure_material_components --
# the frequency-domain solver ingests only the diagonal background permeability and
# would silently drop the off-diagonal Polder gyrotropy; (2)
# fdtd/distributed/shard_engine.py rejects a gyromagnetic_enabled local solver -- the
# shard phases never run the magnetization-ADE hooks and the shard-local layout has
# no rank-seam handling (contract boundary 8, rejected until Phase 4); (3)
# fdtd/runtime/gyromagnetic.py splits the former single axis-aligned guard into a
# general-bias reject and a mixed-bias-direction reject, closing the sign-mixed hole
# (a scene mixing +z and -z ferrites previously passed the axis-only guard and
# silently inverted the non-reciprocity of the -bias region). The adjoint reject
# (contract boundary 9) is added as a return-string branch in
# _unsupported_adjoint_medium, reusing the existing generic raise, so it adds no new
# guard node. The checkpoint/resume schema gains the gyromagnetic magnetization state
# names (no new guard). All three new guards are genuine capability gaps; lower this
# budget as distributed ferrite (Phase 4), the FDFD gyromagnetic ingest, and the
# arbitrary-bias kernel (Phase 2) land.
# Merged union 2026-07-18: 140 - 1 (plan 05 N1) + 4 (plan 08 1b/1c+hardening) = 143.
# 2026-07-19 (plan 12 electrostatics Phase 0+1, scalar Laplace/Poisson slice):
# 144 -> 152 (+8 capability). The new cell-centred finite-volume electrostatic
# solver lands with eight fail-closed capability guards, all covering features
# that are genuinely out of this stage's scope and would otherwise be silently
# mishandled. Six are in compiler/electrostatic.py: (1) a grid-extending Scene
# boundary (PML/periodic) is rejected because electrostatics owns its own
# ElectrostaticBoundarySpec and must not solve on a PML-padded grid; (2) a
# PEC-material structure is rejected (a conductor must be an ElectrostaticTerminal
# equipotential, not a zero-permittivity dielectric); (3) a dispersive material is
# rejected because it exposes no zero-frequency permittivity and the solver refuses
# to guess a DC limit; (4) a DiagonalTensor3/Tensor3x3 anisotropic permittivity is
# rejected (scalar operator only; tensor eps is Phase 4); (5) a per-cell tensor
# permittivity sample is rejected; (6) a complex permittivity sample is rejected as
# not a valid DC static value. Two are in electrostatic/runtime.py: (7) a floating
# conductor with prescribed charge is rejected (the linear-superposition solve is
# Phase 2); (8) a pure-Neumann problem with no fixed potential is rejected as
# gauge-singular (charge-compatibility + gauge fix is Phase 2). Lower this budget as
# tensor-eps/open-boundary (Phase 4), floating-charge superposition, and pure-Neumann
# gauge handling (Phase 2) land.
CAPABILITY_GUARD_BUDGET = 152

# (posix path relative to the repo root, distinctive message substring).
# Keep in sync with docs/reference/fdtd-capability-guard-census.md.
CONTRACT_GUARDS = (
    ("witwin/maxwell/circuit_devices.py", "Transistor device BJT is gated behind the independent Phase 5"),
    ("witwin/maxwell/circuit_devices.py", "Transistor device MOSFET is gated behind the independent Phase 5"),
    ("witwin/maxwell/media.py", "Nonlinear Material frequency evaluation is not defined without a field amplitude"),
    ("witwin/maxwell/media.py", "relative_permittivity() currently supports isotropic Material only"),
    ("witwin/maxwell/media.py", "relative_permittivity() is not defined for nonlinear Material"),
    ("witwin/maxwell/media.py", "relative_permittivity() is not defined for spatially-varying custom dispersive poles"),
    ("witwin/maxwell/media.py", "relative_permeability() currently supports isotropic Material only"),
    ("witwin/maxwell/media.py", "relative_permeability() is not defined for nonlinear Material"),
    ("witwin/maxwell/media.py", "relative_permeability() is not defined for spatially-varying custom dispersive poles"),
    ("witwin/maxwell/media.py", "relative_permeability() is not defined for a GyromagneticFerrite"),
    ("witwin/maxwell/media.py", "evaluate_at_frequency() is not defined for a GyromagneticFerrite"),
    ("witwin/maxwell/media.py", "PerturbationMedium frequency evaluation is spatially varying"),
    ("witwin/maxwell/media.py", "relative_permittivity() is not defined for PerturbationMedium"),
    ("witwin/maxwell/scene.py", "SceneModule subclasses must implement to_scene()"),
    ("witwin/maxwell/geometry/polyslab.py", "ComplexPolySlab does not support mesh export"),
    ("witwin/maxwell/fdtd/adjoint/bridge.py", "FDTD backward requires an input that contributes"),
    ("witwin/maxwell/fdtd/adjoint/bridge.py", "FDTD adjoint requires complex field state for Bloch faces"),
    (
        "witwin/maxwell/fdtd/adjoint/bridge.py",
        "Differentiable thin-wire FDTD requires a fixed Maxwell time step",
    ),
    ("witwin/maxwell/fdfd/adjoint/bridge.py", "FDFD backward currently supports trainable scene inputs that contribute"),
    ("witwin/maxwell/simulation.py", "Embedded network feedback is defined only for the time-domain FDTD update"),
    ("witwin/maxwell/adapters/tidy3d.py", "Tidy3D export for magnetic dispersive Material"),
    ("witwin/maxwell/adapters/tidy3d.py", "Tidy3D export currently assumes mu_r = 1"),
    ("witwin/maxwell/fdtd/excitation/tfsf_state.py", "TFSF slab mode is required for Bloch-boundary TFSF injection"),
    ("witwin/maxwell/postprocess/stratton_chu.py", "requires at least two exterior material samples"),
    ("witwin/maxwell/postprocess/stratton_chu.py", "must have at least one material cell outside the surface"),
    ("witwin/maxwell/postprocess/stratton_chu.py", "must expose tangential_bounds metadata"),
)


def _literal_fragments(node: ast.expr) -> str:
    """Concatenated string-literal fragments of a raise message expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value
            for value in node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    if isinstance(node, ast.BinOp):
        return _literal_fragments(node.left) + _literal_fragments(node.right)
    return ""


def _raised_not_implemented(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        name = None
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        elif isinstance(exc, ast.Name):
            name = exc.id
        if name != "NotImplementedError":
            continue
        message = ""
        if isinstance(exc, ast.Call) and exc.args:
            message = _literal_fragments(exc.args[0])
        yield node.lineno, message


def collect_guards() -> list[tuple[str, int, str]]:
    guards = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for lineno, message in _raised_not_implemented(tree):
            guards.append((rel, lineno, message))
    return guards


def _is_contract(rel: str, message: str) -> bool:
    return any(rel == file and key in message for file, key in CONTRACT_GUARDS)


def test_capability_guard_budget():
    guards = collect_guards()
    capability = [(rel, lineno, msg) for rel, lineno, msg in guards if not _is_contract(rel, msg)]
    assert len(capability) <= CAPABILITY_GUARD_BUDGET, (
        f"{len(capability)} capability guards exceed the committed budget of "
        f"{CAPABILITY_GUARD_BUDGET}. New NotImplementedError guards need either an "
            "implementation or a contract-guard entry in docs/reference/fdtd-capability-guard-census.md:\n"
        + "\n".join(f"  {rel}:{lineno}  {msg[:90]}" for rel, lineno, msg in capability)
    )


def test_contract_guard_list_is_accurate():
    """Every exclusion-list entry must still match a real guard (no stale entries)."""
    guards = collect_guards()
    for file, key in CONTRACT_GUARDS:
        assert any(rel == file and key in msg for rel, _, msg in guards), (
            f"Stale contract-guard entry: {file!r} has no NotImplementedError containing {key!r}. "
            "Update CONTRACT_GUARDS and docs/reference/fdtd-capability-guard-census.md."
        )


# --- P5.5 phrase gate --------------------------------------------------------
# Plan P5.5 criterion: no public *forward-path* NotImplementedError may say a
# guard is merely deferred ("not implemented yet" / "not supported yet" / "in
# v1"); every such message must state the physical or mathematical reason the
# case is unsupported. The gated forward path is media.py, compiler/,
# fdtd/runtime/, fdtd/boundary/, scene.py, and simulation.py -- i.e. everything
# except the modules a LATER phase owns for rewording, enumerated below.

BANNED_DEFERRAL_PHRASES = ("not implemented yet", "not supported yet", "in v1")

# Modules whose NotImplementedError wording is owned by a later phase; each
# entry names the owning phase and is matched as a path prefix. These are the
# ONLY files allowed to keep a deferral phrase after P5.5.
PHRASE_GATE_ALLOWLIST = (
    # adapters/tidy3d.py was retired from the allowlist in P5.6: every remaining
    # export guard states a physical or architectural reason (Tidy3D has no such
    # construct, or the per-cell profile is not resolvable at material-conversion
    # time) and none uses a deferral phrase.
    # fdfd/ stays allowlisted: FDFD static parity (Kerr / Tensor3x3 / magnetic /
    # in-domain PEC) and FDFD nonuniform grids are deferred by user decision
    # (2026-07-11), so these guards remain as honest user-deferral statements
    # rather than being closed.
    ("witwin/maxwell/fdfd/", "P5.6 FDFD static parity deferred by user decision (2026-07-11)"),
    ("witwin/maxwell/postprocess/", "P5.9 postprocess generality reword"),
    ("witwin/maxwell/fdtd/adjoint/", "adjoint deferred-branch reword (P5.7+)"),
)


def _phrase_gate_allowlisted(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix, _ in PHRASE_GATE_ALLOWLIST)


def test_no_deferral_phrase_in_public_forward_path():
    guards = collect_guards()
    violations = []
    for rel, lineno, msg in guards:
        low = msg.lower()
        if not any(phrase in low for phrase in BANNED_DEFERRAL_PHRASES):
            continue
        if _phrase_gate_allowlisted(rel):
            continue
        violations.append((rel, lineno, msg))
    assert not violations, (
        "Public forward-path NotImplementedError messages must state a physical or "
        "mathematical reason, never a deferral phrase "
        f"{BANNED_DEFERRAL_PHRASES}. Reword these or, if a later phase owns the module, "
        "add it to PHRASE_GATE_ALLOWLIST with the owning phase:\n"
        + "\n".join(f"  {rel}:{lineno}  {msg[:90]}" for rel, lineno, msg in violations)
    )

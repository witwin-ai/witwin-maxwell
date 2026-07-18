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
CAPABILITY_GUARD_BUDGET = 137

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

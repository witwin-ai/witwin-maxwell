"""Guard-convergence gate for the P5 functional-completeness plan.

Walks every ``raise NotImplementedError`` in ``witwin/`` via AST, subtracts the
committed contract-guard exclusion list (docs/dev/fdtd_gap_05_guard_census.md),
and fails when the remaining *capability* guard count exceeds the committed
budget. Each P5 phase lowers ``CAPABILITY_GUARD_BUDGET`` in the same commit
that removes guards, so the metric cannot silently regress or be gamed by
reclassifying capability guards as contracts without touching this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "witwin"

# Lowered by each P5 phase in the commit that removes guards. Never raised
# without a corresponding update to docs/dev/fdtd_gap_05_guard_census.md.
# P5.6 (cross-solver parity) re-measured this at 63: the Tidy3D export commits
# (PEC/PECMedium, sigma_e+dispersive, Kerr/TPA, anisotropy, Medium2D/Graphene/
# LossyMetal, modulation, custom poles/PerturbationMedium, geometry/source/
# monitor kinds, nonuniform grids) lifted adapters/tidy3d.py capability guards
# from 13 -> 10, taking the AST capability count 66 -> 63. FDFD static parity
# and FDFD nonuniform grids are deferred by user decision (2026-07-11), so no
# fdfd/ guard was removed; the fdfd/solver.py messages were reworded to honest
# user-deferral statements instead.
# P5.7-P5.9 (final reconciliation, measured on merged master) re-measured this at
# 58. P5.7 (modal/port maturity) took fdtd/excitation/modes.py 7 -> 3: two dead
# differentiable node caps deleted, the full-vector complex-epsilon guard deleted
# (complex planes fall through to the scalar complex solve), and the sparse
# request-count guard reclassified NotImplementedError -> ValueError; the surviving
# three (Bloch transverse boundary, magnetic mu_r scalar plane, differentiable
# non-Hermitian lossy adjoint) are physics-worded. P5.9 (postprocess generality)
# took postprocess/scattering_parameters.py 1 -> 0 (broadband incident_power="auto"
# replaced its last NotImplementedError with actionable ValueErrors). P5.8
# (CUDA-graph coverage) added and removed no NotImplementedError: its capture
# coverage was lifted by narrowing boolean/predicate declines in
# fdtd/runtime/stepping.py, not by deleting raise statements. Net 63 -> 58.
CAPABILITY_GUARD_BUDGET = 58

# (posix path relative to the repo root, distinctive message substring).
# Keep in sync with the table in docs/dev/fdtd_gap_05_guard_census.md.
CONTRACT_GUARDS = (
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
    ("witwin/maxwell/fdtd/adjoint/bridge.py", "FDTD backward currently supports trainable scene inputs that contribute"),
    ("witwin/maxwell/fdtd/adjoint/bridge.py", "FDTD adjoint requires complex field state for Bloch faces"),
    ("witwin/maxwell/fdfd/adjoint/bridge.py", "FDFD backward currently supports trainable scene inputs that contribute"),
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
        "implementation or a contract-guard entry in docs/dev/fdtd_gap_05_guard_census.md:\n"
        + "\n".join(f"  {rel}:{lineno}  {msg[:90]}" for rel, lineno, msg in capability)
    )


def test_contract_guard_list_is_accurate():
    """Every exclusion-list entry must still match a real guard (no stale entries)."""
    guards = collect_guards()
    for file, key in CONTRACT_GUARDS:
        assert any(rel == file and key in msg for rel, _, msg in guards), (
            f"Stale contract-guard entry: {file!r} has no NotImplementedError containing {key!r}. "
            "Update CONTRACT_GUARDS and docs/dev/fdtd_gap_05_guard_census.md."
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

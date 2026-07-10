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
CAPABILITY_GUARD_BUDGET = 87

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

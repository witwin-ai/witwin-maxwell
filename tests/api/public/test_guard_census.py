"""Guard-convergence gate for the P5 functional-completeness plan.

Walks every ``raise NotImplementedError``/``UnsupportedFeatureError`` and explicitly inventoried safety refusal
in ``witwin/`` via AST, subtracts the
unchanged contract-guard exclusion list, and matches every remaining capability
site to its exact reviewed manifest. New predicates/messages/scopes and retired
guards require explicit review evidence; unrelated additions and removals cannot
cancel each other behind a stale scalar budget.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "witwin"

# Capability debt is an exact reviewed inventory, anchored to immutable baseline
# 10d3513 (300 capability sites, 29 unchanged contract exclusions). A new or
# changed predicate/message/scope requires a reviewed manifest entry. Retired
# entries retain positive implementation evidence; no scalar ceiling can hide a
# new guard behind an unrelated removal.
MANIFEST_PATH = PACKAGE_ROOT.parent / "docs/reference/fdtd-capability-guards.json"


def _assert_test_reference(reference):
    parts=reference.split('::')
    path=PACKAGE_ROOT.parent/parts[0]
    assert path.is_file(),reference
    if len(parts)>1:
        function=parts[-1].split('[',1)[0]
        tree=ast.parse(path.read_text())
        assert any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==function
                   for node in ast.walk(tree)),reference

# (posix path relative to the repo root, distinctive message substring).
# Keep in sync with docs/reference/fdtd-capability-guard-census.md.
CONTRACT_GUARDS = (
    # A quotient by an absent or ambiguous incident waveform is undefined.
    ("witwin/maxwell/fdtd/runtime/timeplan.py", "normalize_source requires an incident source spectrum"),
    ("witwin/maxwell/fdtd/runtime/timeplan.py", "normalize_source has no unique incident spectrum for circuit-driven fields"),
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


def _raised_not_implemented(tree: ast.AST, rel=""):
    from scripts.dev.fdtd.guard_census import SAFETY_GUARDS, TRACKED_EXCEPTIONS

    class Inventory(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.guards = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_FunctionDef = visit_ClassDef
        visit_AsyncFunctionDef = visit_ClassDef

        def visit_Raise(self, node):
            exc = node.exc
            name = exc.func.id if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) else getattr(exc, "id", None)
            safety = (rel, ".".join(self.scope), name) in SAFETY_GUARDS
            if name not in TRACKED_EXCEPTIONS and not safety:
                return
            message = _literal_fragments(exc.args[0]) if isinstance(exc, ast.Call) and exc.args else ""
            self.guards.append((node.lineno, message))

    visitor = Inventory()
    visitor.visit(tree)
    yield from visitor.guards


def collect_guards(package_root: Path | None = None) -> list[tuple[str, int, str]]:
    root = PACKAGE_ROOT if package_root is None else package_root
    guards = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for lineno, message in _raised_not_implemented(tree, rel):
            guards.append((rel, lineno, message))
    return guards


def _is_contract(rel: str, message: str) -> bool:
    return any(rel == file and key in message for file, key in CONTRACT_GUARDS)


@pytest.fixture
def inherited_debt_baseline():
    from scripts.dev.fdtd.guard_census import BASELINE
    if not BASELINE.exists():
        pytest.skip('repository evidence not in this checkout: '+str(BASELINE.relative_to(PACKAGE_ROOT.parent)))


@pytest.mark.usefixtures('inherited_debt_baseline')
def test_reviewed_capability_manifest_exactly_matches_source():
    import hashlib
    import json
    from collections import Counter
    from scripts.dev.fdtd.guard_census import (
        collect_detailed_guards, is_contract, inherited_guards,
        BASELINE_SHA256, BASELINE_DETAILED_SHA256,
    )

    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["schema"] == "fdtd-capability-guard-manifest/1"
    assert manifest["snapshot_status"] == "reviewed", "Finalize the guard review after the physical source freeze."
    baseline = manifest["baseline"]
    assert baseline["commit"] == "10d351310b36e9681867347e772a21977e234bdf"
    assert (baseline["capabilities"], baseline["contracts"]) == (300, 29)
    assert baseline["sha256"] == BASELINE_SHA256
    assert baseline["detailed_sha256"] == BASELINE_DETAILED_SHA256
    baseline_path = PACKAGE_ROOT.parent / baseline["path"]
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline["sha256"]
    actual = [guard for guard in collect_detailed_guards() if not is_contract(guard, CONTRACT_GUARDS)]
    expected = manifest["active"]
    assert Counter(guard["id"] for guard in actual) == Counter(guard["id"] for guard in expected), (
        "An unreviewed capability guard was added, changed or removed. Generate a separate "
        "review proposal with scripts/dev/fdtd/guard_census.py; retain capability debt and "
        "record negative boundaries or positive retirement evidence."
    )
    actual_by_id = {guard["id"]:guard for guard in actual}
    inherited_ids = {guard["id"] for guard in inherited_guards()}
    retired_ids = {guard["id"] for guard in manifest["retired"]}
    assert inherited_ids.issubset(set(actual_by_id) | retired_ids)
    for entry in expected:
        for key in ("path", "scope", "expression", "conditions", "message"):
            assert entry[key] == actual_by_id[entry["id"]][key], (entry["id"], key)
        assert entry["review_status"] == "reviewed"
        assert entry["origin"] in ("inherited_debt", "feature_boundary", "modified_boundary")
        assert (entry["origin"] == "inherited_debt") == (entry["id"] in inherited_ids)
        assert entry["unsupported_boundary"] and entry["rationale"] and entry["exit_acceptance"]
        if entry["origin"] != "inherited_debt":
            assert entry["negative_tests"], entry["id"]
            for test in entry["negative_tests"]:
                _assert_test_reference(test)
            assert entry["negative_control_status"] in ('coverage_pending','matching_negative_assertion','raising_site_observed')
    for entry in manifest["retired"]:
        assert entry["retirement_review"] == "reviewed" and entry["retirement_reason"]
        assert entry["positive_tests"]
        for test in entry["positive_tests"]:
            _assert_test_reference(test)


def test_migrated_raises_keep_their_reviews_through_the_receipt():
    """Every raise moved to UnsupportedFeatureError carries its review to the new identity."""
    import json
    from scripts.dev.fdtd.guard_census import collect_detailed_guards, migrated_identities
    manifest = json.loads(MANIFEST_PATH.read_text())
    active = {entry["id"] for entry in manifest["active"]}
    present = {guard["id"]: guard for guard in collect_detailed_guards()}
    for old, new in migrated_identities().items():
        assert old not in present and old not in active, old
        guard = present.get(new["id"])
        if guard is not None:
            assert guard["expression"] == new["expression"]
            assert guard["expression"].startswith("Call(func=Name(id='UnsupportedFeatureError'")
            assert new["id"] in active or _is_contract(guard["path"], guard["message"]), new["id"]


def test_detailed_guard_collector_preserves_the_complete_exception_inventory():
    from collections import Counter
    from scripts.dev.fdtd.guard_census import collect_detailed_guards
    detailed = collect_detailed_guards()
    assert Counter((guard["path"], guard["line"], guard["message"]) for guard in detailed) == Counter(collect_guards())
    assert len({guard["id"] for guard in detailed}) == len(detailed)


def test_guard_identity_ignores_lines_but_reviews_changed_predicates_scopes_and_messages(tmp_path):
    from scripts.dev.fdtd.guard_census import collect_detailed_guards
    package=tmp_path/'witwin'
    package.mkdir()
    source=package/'boundary.py'
    original='def physical_boundary(x):\n    if x.unsupported:\n        raise NotImplementedError("joint response is missing")\n'
    def identity(text):
        source.write_text(text)
        return collect_detailed_guards(package)[0]['id']
    baseline=identity(original)
    assert identity('\n\n'+original)==baseline
    assert identity(original.replace('x.unsupported','x.other_boundary'))!=baseline
    assert identity(original.replace('physical_boundary','different_scope'))!=baseline
    assert identity(original.replace('joint response is missing','joint trace response is missing'))!=baseline


def test_guard_identity_tracks_dynamic_loop_capability_sets(tmp_path):
    from scripts.dev.fdtd.guard_census import collect_detailed_guards
    package=tmp_path/'witwin'
    package.mkdir()
    source=package/'boundary.py'
    text='def check(x):\n    for name in ("anisotropy",):\n        if getattr(x, name):\n            raise NotImplementedError(f"Missing joint response for {name}")\n'
    source.write_text(text)
    before=collect_detailed_guards(package)[0]['id']
    source.write_text(text.replace('(\"anisotropy\",)','(\"anisotropy\", \"ade\")'))
    assert collect_detailed_guards(package)[0]['id']!=before


def test_proposal_command_cannot_overwrite_the_reviewed_manifest():
    import subprocess
    import sys
    before=MANIFEST_PATH.read_bytes()
    result=subprocess.run([sys.executable,str(PACKAGE_ROOT.parent/'scripts/dev/fdtd/guard_census.py'),
                           '--proposal',str(MANIFEST_PATH)],capture_output=True,text=True)
    assert result.returncode==2 and 'approval requires an explicit review edit' in result.stderr
    assert MANIFEST_PATH.read_bytes()==before


def test_contract_guard_list_is_accurate():
    """Every exclusion-list entry must still match a real guard (no stale entries)."""
    guards = collect_guards()
    for file, key in CONTRACT_GUARDS:
        assert any(rel == file and key in msg for rel, _, msg in guards), (
            f"Stale contract-guard entry: {file!r} has no tracked refusal containing {key!r}. "
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
    # fdfd/ stays allowlisted for the nonlinear and full off-diagonal tensor
    # extensions. Nonuniform grids, PEC and diagonal magnetic backgrounds
    # are implemented and no longer use these deferral guards.
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


def test_p7b_rekeyed_guards_keep_their_body_and_normalized_predicates():
    """Phase 7b re-keyed guards only because their enclosing predicates read prepared.scene."""
    import json
    from scripts.dev.fdtd.guard_census import P7B_RECEIPT, collect_detailed_guards, p7b_guard_hashes
    receipt = json.loads(P7B_RECEIPT.read_text())
    present = {guard["id"]: guard for guard in collect_detailed_guards()}
    assert set(receipt["review"]) <= set(receipt["guard_id_map"])
    for old, entry in receipt["guard_id_map"].items():
        guard = present[entry["id"]]
        assert old not in present
        assert guard["conditions"] == entry["conditions"]
        assert p7b_guard_hashes(guard) == (entry["body_sha256"], entry["normalized_conditions_sha256"])
        if old not in receipt["review"]:
            assert entry["old_normalized_conditions_sha256"] == entry["normalized_conditions_sha256"], old


def test_result_strict_reads_rekeyed_guards_keep_their_body_and_review():
    """fix/result-strict-scene-reads re-keyed guards only because result.py reads Scene fields strictly."""
    import hashlib
    import json
    from scripts.dev.fdtd.guard_census import collect_detailed_guards
    receipt = json.loads((PACKAGE_ROOT.parent / "scripts/dev/fdtd/result_strict_guard_receipt.json").read_text())
    manifest = {entry["id"]: entry for entry in json.loads(MANIFEST_PATH.read_text())["active"]}
    present = {guard["id"]: guard for guard in collect_detailed_guards()}
    assert receipt["review"] == []
    for old, entry in receipt["guard_id_map"].items():
        guard = present[entry["id"]]
        assert old not in present and old not in manifest
        assert guard["conditions"] == entry["conditions"]
        body = json.dumps([guard["path"], guard["scope"], guard["expression"], guard["message"]]).encode()
        assert hashlib.sha256(body).hexdigest() == entry["body_sha256"]
        assert entry["old_normalized_conditions_sha256"] == entry["normalized_conditions_sha256"], old
        assert manifest[entry["id"]]["review_status"] == "reviewed"


def test_strict_reads_2_rekeyed_guards_keep_their_body_and_review():
    """fix/strict-scene-reads-2 re-keyed guards only because simulation.py reads Scene fields strictly."""
    import hashlib
    import json
    from scripts.dev.fdtd.guard_census import collect_detailed_guards
    receipt = json.loads((PACKAGE_ROOT.parent / "scripts/dev/fdtd/strict_reads_2_guard_receipt.json").read_text())
    manifest = {entry["id"]: entry for entry in json.loads(MANIFEST_PATH.read_text())["active"]}
    present = {guard["id"]: guard for guard in collect_detailed_guards()}
    assert receipt["review"] == []
    for old, entry in receipt["guard_id_map"].items():
        guard = present[entry["id"]]
        assert old not in present and old not in manifest
        assert guard["conditions"] == entry["conditions"]
        body = json.dumps([guard["path"], guard["scope"], guard["expression"], guard["message"]]).encode()
        assert hashlib.sha256(body).hexdigest() == entry["body_sha256"]
        assert entry["old_normalized_conditions_sha256"] == entry["normalized_conditions_sha256"], old
        if old in receipt["manifest"]:
            assert manifest[entry["id"]]["review_status"] == "reviewed"
        else:
            assert any(guard["path"] == path and fragment in guard["message"] for path, fragment in CONTRACT_GUARDS)

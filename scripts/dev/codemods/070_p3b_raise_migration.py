"""Migrate capability refusals from ``NotImplementedError`` to ``UnsupportedFeatureError``.

0.7.0 phase 3b. Every ``raise NotImplementedError(...)`` in ``witwin/maxwell`` is
classified in ``070_p3b_raise_sites.csv`` (one row per site, keyed by the census
identity the raise had before migration):

* ``capability``: a refusal of a declared feature. It becomes
  ``UnsupportedFeatureError(message, feature=..., method=..., alternative=...)``.
* ``abstract``: a hook that subclasses must override; left alone.
* ``unreachable``: an internal invariant that no input can reach; left alone.
* ``pinned``: inside a file or scope that the consumer-metadata bridge pins byte for
  byte (``scripts/dev/fdtd/consumer_metadata/derive_consumer_bridge.py``); left alone
  until that bridge is re-baselined.

``feature`` is a capability-registry key (``witwin/maxwell/capability/keys.py``)
when the refusal names that declared feature, otherwise ``guard:<8 hex>``, the
first eight hex digits of the census identity of the unmigrated raise. ``method``
is a method name (``fdtd``, ``fdfd``, ``mom``, ``mom.layered``, ``electrostatic``,
``capacitance``), empty for a method-independent refusal, or ``=<expression>``
for a method only known at run time. ``alternative`` is optional text.

Commands (run from any directory with the maxwell environment):

    classify [--scope S]   add unclassified sites to the CSV as ``review=proposed``
    apply [--scope S]      rewrite every reviewed capability site still raising
                           NotImplementedError, then record the census identity
                           changes in scripts/dev/fdtd/raise_migration_receipt.json
    check [--scope S]      exit 1 while a site is unclassified, proposed or unmigrated

``--scope`` is ``package`` (default: witwin/maxwell without mom/), ``mom``
(witwin/maxwell/mom/ only) or ``all``. Every command is idempotent: migrated
sites no longer raise NotImplementedError and are skipped, so the script can be
re-run after a rebase. Afterwards generate a census proposal with
``scripts/dev/fdtd/guard_census.py``; reviews follow the receipt.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.dev.fdtd.consumer_metadata.derive_consumer_bridge import PROTECTED_FILES, PROTECTED_SCOPES  # noqa: E402
from scripts.dev.fdtd.guard_census import RAISE_RECEIPT, collect_detailed_guards  # noqa: E402

SITES = Path(__file__).with_name("070_p3b_raise_sites.csv")
PACKAGE = "witwin/maxwell/"
MOM = "witwin/maxwell/mom/"
NEW = "UnsupportedFeatureError"
COLUMNS = ("census_id", "path", "line", "scope", "kind", "feature", "method", "alternative", "review", "status",
           "message")
KINDS = ("capability", "abstract", "unreachable", "pinned")
METHODS = ("fdtd", "fdfd", "mom", "mom.layered", "electrostatic", "capacitance")
LINE_LIMIT = 120

# First match wins. Paths are prefixes or regular expressions over the repository path.
METHOD_RULES = (
    (r"witwin/maxwell/mom/layered/", "mom.layered"),
    (r"witwin/maxwell/mom/", "mom"),
    (r"witwin/maxwell/fdfd/", "fdfd"),
    (r"witwin/maxwell/(electrostatic/|compiler/electrostatic\.py)", "electrostatic"),
    (r"witwin/maxwell/(fdtd|execution)/", "fdtd"),
    (r"witwin/maxwell/compiler/(thin_sheet|planar_pec|cut_surface|thin_wire|coordinates|power_loss|fdtd_audit|"
     r"sources|circuits)\.py", "fdtd"),
    (r"witwin/maxwell/(simulation|material_cache|network_sweep|waveport_sweep|terminal_waveport)\.py", "fdtd"),
)
# (path pattern or None, message pattern or None, key); the key None means a guard key.
FEATURE_RULES = (
    (None, r"breakdown observers", "monitor.BreakdownMonitor"),
    (None, r"[Bb]reakdown", "material.breakdown"),
    (None, r"PerturbationMedium|perturbation grid", "material.perturbation"),
    (r"witwin/maxwell/scene\.py", r"MaterialRegion", None),
    (r"witwin/maxwell/(fdtd/adjoint/mapped_admission|compiler/coordinates|coordinates)\.py", None, "coordinates"),
    (None, r"TerminalWavePort", "port.TerminalWavePort"),
    (None, r"CoaxialLumped|[Cc]oaxial lumped", "port.CoaxialLumpedPort"),
    (None, r"\bWavePort", "port.WavePort"),
    (None, r"Wire-bound port", "port.wire_binding"),
    (None, r"TerminalPort", "port.TerminalPort"),
    (None, r"ModeSource|ModePort", "source.ModeSource"),
    (None, r"[Ss]heet[- ]gap feed", "geometry.surface_mesh"),
    (None, r"ThinWire|[Tt]hin[- ]wire", "thin_wire"),
    (None, r"[Ee]mbedded[- ]network|RationalModel|Network-bound", "network"),
    (None, r"[Cc]ircuit|\bMNA\b", "circuit"),
    (None, r"GyromagneticFerrite|[Gg]yromagnetic|ferrite", "material.gyromagnetic"),
    (None, r"[Mm]apped|BendCoordinates", "coordinates"),
    (None, r"^SurfaceMesh", "geometry.surface_mesh"),
    (r"witwin/maxwell/(fdtd/(copper_foil|plane_product|distributed/copper)\.py)", None, "material.copper_foil"),
    (None, r"CopperFoil|[Cc]opper|\bfoil", "material.copper_foil"),
    (None, r"[Ss]urface[- ]impedance|SurfaceImpedance|SIBC", "material.surface_impedance"),
    (r"witwin/maxwell/(fdtd|compiler)/(thin_sheet|cut_surface)\.py|witwin/maxwell/geometry/surface_mesh\.py",
     None, "geometry.surface_mesh"),
    (None, r"SurfaceMesh|[Tt]hin[- ]sheet|cut-surface", "geometry.surface_mesh"),
    (r"witwin/maxwell/(compiler|geometry)/planar_pec\.py", None, "geometry.planar_pec"),
    (None, r"PlanarPEC", "geometry.planar_pec"),
    (None, r"InternalAbsorber", "internal_absorber"),
    (None, r"ModeABC", "boundary.mode_abc"),
    (None, r"phase[-_ ]screen|UV assignment", "structure.optical_assignment"),
    (None, r"[Tt]ime-modulat|[Mm]odulation", "material.modulation"),
    (None, r"[Nn]onlinear|Kerr|chi2|chi3", "material.nonlinear"),
    (None, r"off-diagonal|Tensor3x3|anisotrop", "material.full_tensor"),
    (None, r"\bPMC\b", "material.pmc"),
    (None, r"2D sheet|Medium2D", "material.medium2d"),
    (r"witwin/maxwell/compiler/power_loss\.py", None, "monitor.PowerLossMonitor"),
    (None, r"PowerLossMonitor", "monitor.PowerLossMonitor"),
    (r"witwin/maxwell/fdtd/time_flux\.py", None, "monitor.FluxTimeMonitor"),
    (None, r"Closed-surface Huygens", "monitor.ClosedSurfaceMonitor"),
    (None, r"AuxFieldTimeMonitor", "monitor.AuxFieldTimeMonitor"),
    (None, r"KSpaceProjectionMonitor", "monitor.KSpaceProjectionMonitor"),
    (None, r"Gaussian-overlap", "monitor.GaussianOverlapMonitor"),
    (None, r"PlaneWave", "source.PlaneWave"),
    (None, r"TFSF", "source.injection_tfsf"),
    (None, r"PortSweep|^Port |Port  ", "port"),
    (None, r"[Ll]umped (element|object)|R/L/C|[Ll]umped", "lumped_element"),
    (None, r"\bBloch\b", "boundary.bloch"),
    (None, r"[Pp]eriodic", "boundary.periodic"),
    (None, r"\bMur\b", "boundary.mur"),
    (None, r"[Ss]ymmetry", "symmetry"),
    (None, r"homogeneous background|physical Maxwell mode", "background.homogeneous"),
)
ABSTRACT_MESSAGE = re.compile(r"subclass|must implement|abstract", re.IGNORECASE)
UNREACHABLE_MESSAGE = re.compile(r"unreachable|should not happen|internal error", re.IGNORECASE)


def registry_keys() -> frozenset[str]:
    from witwin.maxwell.capability.keys import MOM_CONFIG_KEYS, SCENE_KEYS

    return frozenset(SCENE_KEYS + MOM_CONFIG_KEYS)


def in_scope(path: str, scope: str) -> bool:
    if not path.startswith(PACKAGE):
        return False
    mom = path.startswith(MOM)
    return scope == "all" or mom == (scope == "mom")


def raised_name(expression: str) -> str | None:
    match = re.match(r"(?:Call\(func=)?Name\(id='(\w+)'", expression)
    return None if match is None else match.group(1)


def pinned(path: str, scope: str) -> bool:
    """Whether the consumer-metadata bridge pins the bytes of this file or scope."""
    module = path.removeprefix(PACKAGE)
    return module in PROTECTED_FILES or any(
        scope == name or scope.startswith(name + ".") for name in PROTECTED_SCOPES.get(module, ()))


def guard_key(census_id: str) -> str:
    return "guard:" + census_id[:8]


def read_sites() -> dict[str, dict]:
    if not SITES.exists():
        return {}
    with SITES.open(newline="") as handle:
        return {row["census_id"]: row for row in csv.DictReader(handle)}


def write_sites(rows: dict[str, dict]) -> None:
    with SITES.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda row: (row["path"], int(row["line"]), row["census_id"])):
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def read_receipt() -> dict:
    if RAISE_RECEIPT.exists():
        return json.loads(RAISE_RECEIPT.read_text())
    return {"schema": "raise-migration-receipt/1",
            "description": "Census identity of each raise migrated by scripts/dev/codemods/070_p3b_raise_migration.py "
                           "before (key) and after (value) the migration.",
            "guard_id_map": {}}


def enclosing_function(tree: ast.Module, line: int):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= node.end_lineno:
            if best is None or node.lineno >= best.lineno:
                best = node
    return best


def propose(guard: dict, tree: ast.Module) -> dict:
    path, message = guard["path"], guard["message"]
    function = enclosing_function(tree, guard["line"])
    body = [] if function is None else [
        statement for statement in function.body
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))]
    only_statement = len(body) == 1 and isinstance(body[0], ast.Raise) and body[0].lineno == guard["line"]
    if pinned(path, guard["scope"]):
        kind = "pinned"
    elif not message or (only_statement and ABSTRACT_MESSAGE.search(message)):
        kind = "abstract" if only_statement or not guard["expression"].startswith("Call(") else "capability"
    elif UNREACHABLE_MESSAGE.search(message):
        kind = "unreachable"
    else:
        kind = "capability"
    feature = method = ""
    if kind == "capability":
        method = next((value for pattern, value in METHOD_RULES if re.match(pattern, path)), "")
        key = next((key for path_pattern, message_pattern, key in FEATURE_RULES
                    if (path_pattern is None or re.match(path_pattern, path))
                    and (message_pattern is None or re.search(message_pattern, message))), None)
        feature = key or guard_key(guard["id"])
    return dict(census_id=guard["id"], path=path, line=str(guard["line"]), scope=guard["scope"], kind=kind,
                feature=feature, method=method, alternative="", review="proposed", status="pending",
                message=message[:200])


def validate(row: dict, keys: frozenset[str]) -> None:
    where = f"{row['path']}:{row['line']} ({row['census_id'][:8]})"
    if row["kind"] not in KINDS:
        raise SystemExit(f"{where}: kind must be one of {KINDS}, got {row['kind']!r}")
    if (row["kind"] == "pinned") != pinned(row["path"], row["scope"]):
        raise SystemExit(f"{where}: kind pinned must match the consumer-bridge pins")
    if row["kind"] != "capability":
        return
    feature, method = row["feature"], row["method"]
    if not (feature in keys or re.fullmatch(r"guard:[0-9a-f]{8,}", feature)):
        raise SystemExit(f"{where}: feature {feature!r} is neither a registry key nor guard:<hex>")
    if method and not method.startswith("=") and method not in METHODS:
        raise SystemExit(f"{where}: method {method!r} is not one of {METHODS}")


def nie_sites(guards, scope: str) -> list[dict]:
    return [guard for guard in guards
            if in_scope(guard["path"], scope) and raised_name(guard["expression"]) == "NotImplementedError"]


def classify(scope: str) -> dict[str, dict]:
    rows = read_sites()
    guards = collect_detailed_guards()
    trees = {}
    current = set()
    for guard in nie_sites(guards, scope):
        current.add(guard["id"])
        row = rows.get(guard["id"])
        if row is None:
            tree = trees.setdefault(guard["path"], ast.parse((ROOT / guard["path"]).read_text(encoding="utf-8-sig")))
            rows[guard["id"]] = propose(guard, tree)
        else:
            row.update(path=guard["path"], line=str(guard["line"]), scope=guard["scope"], message=guard["message"][:200],
                       status="kept" if row["kind"] != "capability" else "pending")
    migrated = {old: new["id"] for old, new in read_receipt()["guard_id_map"].items()}
    present = {guard["id"]: guard for guard in guards}
    for census_id, row in rows.items():
        if census_id in current or not in_scope(row["path"], scope):
            continue
        new_id = migrated.get(census_id)
        if new_id in present:
            row.update(status="migrated", line=str(present[new_id]["line"]))
        else:
            row["status"] = "stale"
    write_sites(rows)
    return rows


def _byte_offsets(text: str) -> list[int]:
    offsets, total = [0], 0
    for line in text.splitlines(keepends=True):
        total += len(line.encode())
        offsets.append(total)
    return offsets


def _indent(text: str) -> str:
    return text[:len(text) - len(text.lstrip())]


def _keywords(row: dict) -> list[str]:
    method = row["method"]
    method_source = "None" if not method else method[1:] if method.startswith("=") else repr(method)
    words = [f"feature={row['feature']!r}", f"method={method_source}"]
    if row["alternative"]:
        words.append(f"alternative={row['alternative']!r}")
    return [word.replace("'", '"') if "'" not in word[word.index("=") + 2:-1] else word for word in words]


def _rewrite_call(source: bytes, offsets, lines: list[str], call: ast.Call, row: dict) -> list[tuple[int, int, bytes]]:
    """Byte edits that rename the call and append the keyword arguments."""
    def position(line, column):
        return offsets[line - 1] + column

    if not call.args:
        raise SystemExit(f"{row['path']}:{row['line']}: a capability refusal needs a message argument")
    edits = [(position(call.func.lineno, call.func.col_offset), position(call.func.end_lineno, call.func.end_col_offset),
              NEW.encode())]
    last = (call.args + call.keywords)[-1]
    first = call.args[0]
    tail_start = position(last.end_lineno, last.end_col_offset)
    close = position(call.end_lineno, call.end_col_offset) - 1
    between = source[tail_start:close].decode()
    trailing_comma = between.lstrip().startswith(",")
    if trailing_comma:
        tail_start += between.index(",") + 1
    words = _keywords(row)
    single_line = call.lineno == call.end_lineno
    raise_line = lines[call.lineno - 1].rstrip("\n")
    inline = ", " + ", ".join(words)
    if single_line and len(raise_line) + len(inline) - len("NotImplementedError") + len(NEW) <= LINE_LIMIT:
        return edits + [(tail_start, tail_start, inline.encode())]
    first_line = lines[first.lineno - 1]
    if first.lineno == call.lineno:
        prefix = first_line.encode()[:first.col_offset].decode()
        indent = " " * (len(prefix) - len("NotImplementedError") + len(NEW))
    else:
        indent = _indent(first_line)
    groups, current = [], ""
    for word in words:
        candidate = f"{current}, {word}" if current else word
        if current and len(indent) + len(candidate) + 2 > LINE_LIMIT:
            groups.append(current)
            current = word
        else:
            current = candidate
    groups.append(current)
    body = ",\n".join(indent + group for group in groups)
    text = ("\n" + body + ",") if trailing_comma else (",\n" + body)
    return edits + [(tail_start, tail_start, text.encode())]


def _import_edit(text: str, tree: ast.Module, path: str) -> tuple[int, int, bytes] | None:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "errors" and node.level:
            if any(alias.name == NEW for alias in node.names):
                return None
    depth = len(Path(path).parent.relative_to("witwin/maxwell").parts)
    statement = f"from {'.' * (depth + 1)}errors import {NEW}\n"
    offsets = _byte_offsets(text)
    body = tree.body[1:] if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(
        tree.body[0].value, ast.Constant) else tree.body
    run = []
    for node in body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            break
        run.append(node)
    if not run:
        anchor = tree.body[0].end_lineno if body is not tree.body else 0
        return offsets[anchor], offsets[anchor], ("\n" if anchor else "").encode() + statement.encode()
    relative = [node for node in run if isinstance(node, ast.ImportFrom) and node.level]
    anchor = (relative or run)[-1].end_lineno
    # A first package-relative import opens its own import group.
    return offsets[anchor], offsets[anchor], (statement if relative else "\n" + statement).encode()


def _expected_expression(expression: str, row: dict) -> str:
    """The census expression of a migrated raise: renamed, with the keyword arguments appended."""
    if not expression.endswith("keywords=[])"):
        raise SystemExit(f"{row['path']}:{row['line']}: NotImplementedError with keyword arguments")
    call = ast.parse("f(" + ", ".join(_keywords(row)) + ")", mode="eval").body
    keywords = ", ".join(ast.dump(keyword, include_attributes=False) for keyword in call.keywords)
    renamed = expression.replace("Name(id='NotImplementedError'", f"Name(id='{NEW}'", 1)
    return renamed[:-len("keywords=[])")] + f"keywords=[{keywords}])"


def _tracked(guards, path):
    return [guard for guard in guards if guard["path"] == path]


def apply(scope: str) -> None:
    rows = classify(scope)
    keys = registry_keys()
    open_rows = [row for row in rows.values() if in_scope(row["path"], scope) and row["review"] != "reviewed"
                 and row["status"] != "stale"]
    if open_rows:
        raise SystemExit(f"{len(open_rows)} site(s) in scope are not reviewed; edit {SITES.name} first:\n"
                         + "\n".join(f"  {row['path']}:{row['line']} {row['kind']} {row['feature']}"
                                     for row in open_rows[:40]))
    for row in rows.values():
        validate(row, keys)
    before = collect_detailed_guards()
    targets = {}
    for guard in nie_sites(before, scope):
        row = rows[guard["id"]]
        if row["kind"] == "capability":
            targets.setdefault(guard["path"], []).append((guard, row))
    for path, sites in targets.items():
        file = ROOT / path
        text = file.read_text(encoding="utf-8")
        tree = ast.parse(text)
        raises = {node.lineno: node for node in ast.walk(tree) if isinstance(node, ast.Raise)}
        source, offsets, lines = text.encode(), _byte_offsets(text), text.splitlines(keepends=True)
        edits = []
        for guard, row in sites:
            edits += _rewrite_call(source, offsets, lines, raises[guard["line"]].exc, row)
        import_edit = _import_edit(text, tree, path)
        if import_edit is not None:
            edits.append(import_edit)
        for start, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
            source = source[:start] + replacement + source[end:]
        file.write_text(source.decode(), encoding="utf-8")
    after = collect_detailed_guards()
    receipt = read_receipt()
    for path, sites in targets.items():
        old, new = _tracked(before, path), _tracked(after, path)
        if len(old) != len(new):
            raise SystemExit(f"{path}: the census site count changed during the rewrite")
        migrated = {guard["id"]: row for guard, row in sites}
        for previous, current in zip(old, new):
            for field in ("scope", "message", "conditions"):
                if previous[field] != current[field]:
                    raise SystemExit(f"{path}:{current['line']}: {field} changed during the rewrite")
            row = migrated.get(previous["id"])
            if row is None:
                if previous["expression"] != current["expression"]:
                    raise SystemExit(f"{path}:{current['line']}: an unrelated raise changed")
                continue
            if current["expression"] != _expected_expression(previous["expression"], row):
                raise SystemExit(f"{path}:{current['line']}: unexpected rewrite {current['expression'][:120]}")
            receipt["guard_id_map"][previous["id"]] = {"id": current["id"], "expression": current["expression"]}
            row.update(status="migrated", line=str(current["line"]))
    unmigrated = {guard["id"] for guard in after}
    receipt["guard_id_map"] = {old: new for old, new in sorted(receipt["guard_id_map"].items())
                               if old not in unmigrated}
    RAISE_RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    write_sites(rows)
    print(json.dumps({"files": len(targets), "migrated": sum(len(sites) for sites in targets.values())}))


def check(scope: str) -> int:
    rows = classify(scope)
    counts = {}
    for row in rows.values():
        if in_scope(row["path"], scope):
            label = f"{row['kind']}/{row['review']}/{row['status']}"
            counts[label] = counts.get(label, 0) + 1
    print(json.dumps(dict(sorted(counts.items())), indent=1))
    return int(any(row["review"] != "reviewed" or (row["kind"] == "capability" and row["status"] == "pending")
                   for row in rows.values() if in_scope(row["path"], scope) and row["status"] != "stale"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("classify", "apply", "check"))
    parser.add_argument("--scope", choices=("package", "mom", "all"), default="package")
    args = parser.parse_args()
    if args.command == "classify":
        rows = classify(args.scope)
        proposed = sum(row["review"] == "proposed" for row in rows.values() if in_scope(row["path"], args.scope))
        print(json.dumps({"rows": len(rows), "proposed_in_scope": proposed}))
    elif args.command == "apply":
        apply(args.scope)
    else:
        raise SystemExit(check(args.scope))


if __name__ == "__main__":
    main()

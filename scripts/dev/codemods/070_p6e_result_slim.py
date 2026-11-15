"""Rewrite callers of the slimmed 0.7.0 ``Result`` (phase 6e).

``Result`` keeps its core members; domain accessors became namespace functions that take
the Result, ``select``/``tensor`` merged into ``at``/``field``, ``solver_stats`` into
``stats()``, and volume-field reads return ``FieldData``. The script rewrites call sites
mechanically (``ast`` positions, layout kept):

* moved accessors: ``r.antenna(...)`` -> ``mw.antenna.antenna(r, ...)``, ``r.circuits`` ->
  ``mw.circuits.circuits(r)``, ... (``MOVED``); ``Result.load_sharded(...)`` ->
  ``mw.execution.load_sharded(...)``; ``r.power_loss(name)`` -> ``r.monitor(name)`` and, with
  explicit density inputs, ``mw.postprocess.power_loss(r, name, ...)``;
* ``r.solver_stats`` -> ``r.stats()``; ``r.select(...)`` -> ``r.at(...)`` (keyword-only calls,
  so ``torch.Tensor.select(dim, index)`` is never touched);
* ``r.tensor(name, ...)`` -> ``r.field(name, ...).values``, ``r.field("Ex")`` -> ``r.field("Ex").values``,
  ``r.E.x`` -> ``r.E.x.values``: ``.isel(f=0)`` is inserted where a single-frequency result is read
  without a frequency selection (``field`` keeps a length-1 ``f`` axis), and reads of a run that kept
  only its last-step fields become ``r.time_snapshot["EX"].values``. ``r.field("E")["EX"]`` gets
  ``.values`` after the subscript;
* ``ModeAmplitudeData`` selections ``.sel(direction="+")`` / ``Direction.PLUS`` -> ``.sel(propagation=...)``.

Which result a volume-field read sees (method, frequency count, last-step snapshot, selection)
comes from ``070_p6e_field_sites.csv``, recorded by running the tests that read fields on the 0.6.x
layout (``sites`` command, from the trace of ``<evidence-root>/api070-20261108/p6e/plug/p6etrace.py``).
Rows are keyed by file and the read's source text (a text read under different conditions is ``mixed``
and reported). An untraced read (slow tests,
scripts, docs) is rewritten without ``.isel(f=0)`` and reported for review.

Commands (run with the maxwell environment)::

    sites TRACE_DIR...    write 070_p6e_field_sites.csv from trace JSONL files
    classify [PATH ...]   per-decision counts, then the review list
    apply [PATH ...]      rewrite in place (idempotent)
    check [PATH ...]      exit 1 while an old-form Result read remains

Without paths the scope is every tracked ``.py`` file under ``tests``, ``benchmark``, ``scripts`` and
``examples`` and the Python fences of ``docs/reference``, ``docs/memory``, ``docs/README.md``,
``README.md`` and ``FEATURE_LIST.md``. Excluded: ``tests/mom/`` and ``benchmark/mom_*`` (MoM-owned; RFIC follow-up, explicit paths still migrate them),
historical records, the codemods, the BITID probes and ``tests/support/api_dialect.py``,
the consumer-metadata bridge, ``benchmark/sie_single_buffer_equivalence_v3.py``, the p6e tests
of the new Result (``tests/api/public/test_result_slim.py``) and any file whose
sha256 a tracked benchmark, script or test file pins (reported as ``skip``).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SITES_CSV = Path(__file__).with_name("070_p6e_field_sites.csv")
SCOPE = ("tests", "benchmark", "scripts", "examples", "docs/reference", "docs/memory", "docs/README.md",
         "README.md", "FEATURE_LIST.md")
EXCLUDE = re.compile(
    r"^(tests/mom/|benchmark/mom_|docs/(assessments|archive|plans|releases)/|docs/reference/release-notes-"
    r"|CHANGELOG\.md$|scripts/dev/codemods/|tests/support/api_dialect\.py$"
    r"|tests/fdtd/_bit_identity_probe\.py$|tests/fdtd/test_raw_phase_bit_identity\.py$"
    r"|tests/boundaries/absorbers/_boundary_state_probe\.py$|tests/monitors/in_simulation/_bit_identity_runner\.py$"
    r"|tests/materials/surface_impedance/smooth_surface_probe\.py$|tests/support/rf_port_bit_identity_scene\.py$"
    r"|tests/support/rf_port_bit_identity_terminal_wave_scene\.py$|tests/support/port_step_graph_bit_identity_scene\.py$"
    r"|scripts/dev/fdtd/consumer_metadata/|benchmark/core_path/gates\.py$"
    r"|benchmark/sie_single_buffer_equivalence_v3\.py$"
    # Written against the 0.7.0 Result.
    r"|tests/api/public/test_result_slim\.py$)")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)

# Moved Result members -> (namespace, called-as-method). Properties became one-argument functions.
MOVED = {
    "antenna": ("antenna", True), "array_basis": ("antenna", True),
    "breakdown": ("breakdown", True), "breakdown_names": ("breakdown", True),
    "breakdown_data": ("breakdown", False), "breakdown_events": ("breakdown", False),
    "component_stress": ("esd", True), "component_stress_names": ("esd", True),
    "esd_waveform": ("esd", True), "esd_waveform_names": ("esd", True),
    "esd_generator": ("esd", True), "esd_generator_names": ("esd", True),
    "capacitance": ("electrostatic", False), "electrostatic": ("electrostatic", False),
    "electrostatic_prebias": ("electrostatic", False),
    "chart_fields": ("coordinates", False),
    "sar": ("sar", True), "incident_power_density": ("sar", True),
    "circuit": ("circuits", True), "circuits": ("circuits", False),
    "embedded_network": ("rf", True), "embedded_networks": ("rf", False),
    "conductor_losses": ("rf", False), "conductor_loss_columns": ("rf", False),
    "save_sharded": ("execution", True),
    "is_sharded": ("execution", False), "sharded_manifest": ("execution", False), "shard_paths": ("execution", False),
}
# Receivers that are never a Result: modules, the dialect, scenes, Simulation entry points, self.
NOT_RESULT = re.compile(r"^(_?mw(\..*)?|witwin\.maxwell(\..*)?|api|api_dialect|self|cls|scene|.*\.logical_scene"
                        r"|.*scene\(.*\)|.*scene\.clone\(\)|manifest|.*Simulation.*|.*\.(peak|combine)\(.*\)|peaks?(\[.*\])?"
                        r"|peak_\w+|beams?|.*_beam|basis\..*|typed|scaled|repeated|combined|sar(_\w+)?|bank|cap\d*"
                        r"|.*devices.*|load|parameters|runtime|value|left|right|es)$", re.DOTALL)
# Ambiguous property names: rewritten only on result-like receivers.
STRICT = {"capacitance", "electrostatic", "circuits"}
RESULT_LIKE = re.compile(r"^(\w*result\w*|res|r|dc|loaded|restored|resumed|reference|actual|expected|.*\.run\(.*\))$",
                         re.DOTALL)
FIELD_COMPONENTS = {f"{a}{b}" for a in "EH" for b in "xyz"}
SELECTORS = {"frequency", "freq_index"}
DIRECTION_LABELS = {"+", "-"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked(paths=()):
    args = ["git", "-C", str(ROOT), "ls-files", "--", *(paths or SCOPE)]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout.split()
    return [p for p in out if p.endswith((".py", ".md"))]


def _pinned_hashes():
    pinned = set()
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", "benchmark", "scripts", "tests"],
                           capture_output=True, text=True, check=True).stdout.split()
    pattern = re.compile(r"\b[0-9a-f]{64}\b")
    for name in files:
        try:
            pinned.update(pattern.findall((ROOT / name).read_text(errors="ignore")))
        except (IsADirectoryError, FileNotFoundError):
            pass
    pinned.discard(EMPTY_SHA256)
    return pinned


# --- trace -> sites -------------------------------------------------------------------
def _span_text(lines, pos):
    l0, l1, c0, c1 = pos
    if None in (l0, l1, c0, c1):
        return None
    chunk = [line.encode() for line in lines[l0 - 1:l1]]
    if l0 == l1:
        return chunk[0][c0:c1].decode()
    return b"\n".join([chunk[0][c0:], *chunk[1:-1], chunk[-1][:c1]]).decode()


def build_sites(trace_dirs):
    rows = defaultdict(lambda: {"snapshot": set(), "multi": set(), "selected": set(), "method": set(), "api": set()})
    cache = {}
    for directory in trace_dirs:
        for path in sorted(Path(directory).glob("*.jsonl")):
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if "site" not in rec or not rec["site"].get("pos"):
                    continue
                file = rec["site"]["file"]
                if file not in cache:
                    source = ROOT / file
                    cache[file] = source.read_text().splitlines() if source.exists() else None
                if cache[file] is None:
                    continue
                text = _span_text(cache[file], rec["site"]["pos"])
                if text is None:
                    continue
                selection = rec.get("selection") or {}
                selected = any(rec["kwargs"].get(key) is not None for key in SELECTORS) or any(
                    selection.get(key) is not None for key in SELECTORS)
                row = rows[(file, " ".join(text.split()))]
                row["snapshot"].add(bool(rec["snapshot"]))
                row["multi"].add(rec["nfreq"] > 1)
                row["selected"].add(selected)
                row["method"].add(rec["method"])
                row["api"].add(rec["api"])
    with SITES_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "text", "api", "method", "snapshot", "single_unselected"))
        for (file, text), row in sorted(rows.items()):
            single = {not multi and not sel for multi in row["multi"] for sel in row["selected"]}
            writer.writerow((file, text, "|".join(sorted(row["api"])), "|".join(sorted(map(str, row["method"]))),
                             _flag(row["snapshot"]), _flag(single)))
    print(f"wrote {len(rows)} sites to {SITES_CSV.relative_to(ROOT)}")


def _flag(values):
    return "mixed" if len(values) > 1 else ("1" if True in values else "0")


def load_sites():
    sites = {}
    if SITES_CSV.exists():
        with SITES_CSV.open() as handle:
            for row in csv.DictReader(handle):
                sites[(row["file"], row["text"])] = row
    return sites


# --- rewriting ------------------------------------------------------------------------
@dataclass
class Edit:
    start: int
    end: int
    text: str
    decision: str


class Source:
    def __init__(self, text):
        self.text = text
        self.data = text.encode()
        self.offsets = [0]
        for line in self.data.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))

    def offset(self, line, col):
        return self.offsets[line - 1] + col

    def span(self, node):
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)

    def get(self, node):
        start, end = self.span(node)
        return self.data[start:end].decode()


def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _alias(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "witwin.maxwell":
                    return alias.asname or "witwin.maxwell"
    return None


def _call_args(src, call):
    """Source text of the call's arguments (between the parentheses)."""
    whole = src.get(call)
    func = src.get(call.func)
    inner = whole[len(func):].strip()
    assert inner.startswith("(") and inner.endswith(")"), whole
    return inner[1:-1].strip()


def _is_direction_label(node):
    if isinstance(node, ast.Constant) and node.value in DIRECTION_LABELS:
        return True
    return isinstance(node, ast.Attribute) and node.attr in {"PLUS", "MINUS"} and "Direction" in ast.unparse(node)


def plan(path, text, sites, rel):
    """Edits for one Python source; returns (edits, review)."""
    tree = ast.parse(text)
    src = Source(text)
    parents = _parents(tree)
    mw = _alias(tree) or "mw"
    edits, review = [], []

    def site_row(node):
        return sites.get((rel, " ".join(src.get(node).split())))

    def migrated(node):
        # A read already followed by .values / .isel / .sel / .coords / .attrs is typed (idempotence).
        outer = parents.get(node)
        return isinstance(outer, ast.Attribute) and outer.attr in {"values", "isel", "sel", "coords", "attrs", "dims",
                                                                     "units", "to", "sizes"}

    for node in ast.walk(tree):
        # ModeAmplitudeData.sel(direction="+") -> sel(propagation="+")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"sel", "isel"}:
            for keyword in node.keywords:
                if keyword.arg == "direction" and _is_direction_label(keyword.value):
                    start = src.offset(keyword.lineno, keyword.col_offset)
                    edits.append(Edit(start, start + len("direction"), "propagation", "propagation"))
        if not isinstance(node, ast.Attribute):
            continue
        receiver = src.get(node.value)
        parent = parents.get(node)
        called = isinstance(parent, ast.Call) and parent.func is node
        start, end = src.span(node)
        attr = node.attr
        # Result.load_sharded -> mw.execution.load_sharded
        if attr == "load_sharded" and receiver in {"Result", f"{mw}.Result", "mw.Result"} and called:
            edits.append(Edit(start, end, f"{mw}.execution.load_sharded", "load_sharded"))
            continue
        if NOT_RESULT.match(receiver) and not re.search(r"\.run\([^()]*\)$", receiver):
            continue
        if attr == "solver_stats" and not called and isinstance(node.ctx, ast.Load):
            edits.append(Edit(start, end, f"{receiver}.stats()", "solver_stats"))
        elif attr == "select" and called and not parent.args and parent.keywords and all(
                keyword.arg in {"frequency", "freq_index", "expand_symmetry"} for keyword in parent.keywords):
            name_start = src.offset(node.end_lineno, node.end_col_offset) - len("select")
            edits.append(Edit(name_start, name_start + len("select"), "at", "select"))
        elif attr == "power_loss" and called:
            call_start, call_end = src.span(parent)
            args = _call_args(src, parent)
            if parent.keywords or len(parent.args) != 1:
                edits.append(Edit(call_start, call_end, f"{mw}.postprocess.power_loss({receiver}, {args})",
                                  "power_loss_explicit"))
            else:
                edits.append(Edit(call_start, call_end, f"{receiver}.monitor({args})", "power_loss_monitor"))
        elif attr in MOVED and isinstance(node.ctx, ast.Load):
            namespace, method = MOVED[attr]
            if attr in STRICT and not RESULT_LIKE.match(receiver):
                continue
            if method != called:
                continue
            if called:
                call_start, call_end = src.span(parent)
                args = _call_args(src, parent)
                joined = f"{receiver}, {args}" if args else receiver
                edits.append(Edit(call_start, call_end, f"{mw}.{namespace}.{attr}({joined})", f"moved:{attr}"))
            else:
                edits.append(Edit(start, end, f"{mw}.{namespace}.{attr}({receiver})", f"moved:{attr}"))
        elif attr in {"tensor", "field"} and called and receiver not in {"torch", "np", "numpy", "th"} \
                and not receiver.endswith("chart_fields") and "chart_fields(" not in receiver \
                and not receiver.startswith(f"{mw}.coordinates.chart_fields"):
            call = parent
            if attr == "field" and migrated(call):
                continue
            row = site_row(call)
            name = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "name"), None)
            literal = name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None
            family = literal is not None and literal.upper() in {"E", "H"} if attr == "field" else False
            if attr == "field" and not call.args and not call.keywords:
                family = True
            if row is None:
                if attr == "tensor" or literal in FIELD_COMPONENTS | {c.upper() for c in FIELD_COMPONENTS} or family:
                    review.append((call.lineno, f"untraced {attr}: {src.get(call)}"))
                    _field_edit(edits, src, call, attr, receiver, None, family, parents)
                continue
            _field_edit(edits, src, call, attr, receiver, row, family, parents, review=review, literal=literal)
        elif attr in {"x", "y", "z"} and isinstance(node.value, ast.Attribute) and node.value.attr in {"E", "H"} \
                and isinstance(node.ctx, ast.Load):
            if migrated(node):
                continue
            row = site_row(node)
            if row is None:
                review.append((node.lineno, f"untraced accessor read: {src.get(node)}"))
                continue
            if row["snapshot"] == "mixed" or row["single_unselected"] == "mixed":
                review.append((node.lineno, f"mixed accessor read: {src.get(node)}"))
                continue
            if row["snapshot"] == "1":
                base = src.get(node.value.value)
                if isinstance(node.value.value, ast.Call):
                    review.append((node.lineno, f"snapshot selection read: {src.get(node)}"))
                    continue
                edits.append(Edit(start, end, f'{base}.time_snapshot["{node.value.attr}{attr.upper()}"].values',
                                  "accessor_snapshot"))
            else:
                suffix = ".isel(f=0).values" if row["single_unselected"] == "1" else ".values"
                edits.append(Edit(end, end, suffix, "accessor"))
    return edits, review


def _field_edit(edits, src, call, attr, receiver, row, family, parents, review=None, literal=None):
    call_start, call_end = src.span(call)
    if row is not None and (row["snapshot"] == "mixed" or row["single_unselected"] == "mixed"):
        review.append((call.lineno, f"mixed {attr}: {src.get(call)}"))
        return
    snapshot = row is not None and row["snapshot"] == "1"
    single = row is not None and row["single_unselected"] == "1"
    suffix = ".isel(f=0).values" if single else ".values"
    if snapshot:
        if literal is None or call.keywords or family:
            review.append((call.lineno, f"snapshot {attr}: {src.get(call)}"))
            return
        edits.append(Edit(call_start, call_end, f'{receiver}.time_snapshot["{literal.upper()}"].values',
                          f"{attr}_snapshot"))
        return
    if attr == "tensor":
        name_start = src.offset(call.func.end_lineno, call.func.end_col_offset) - len("tensor")
        edits.append(Edit(name_start, name_start + len("tensor"), "field", "tensor"))
        edits.append(Edit(call_end, call_end, suffix, "tensor_values"))
        return
    if family:
        parent = parents.get(call)
        if isinstance(parent, ast.Subscript) and parent.value is call:
            _, sub_end = src.span(parent)
            edits.append(Edit(sub_end, sub_end, suffix, "field_family_item"))
        elif review is not None:
            review.append((call.lineno, f"field family read: {src.get(call)}"))
        return
    edits.append(Edit(call_end, call_end, suffix, "field_values"))


def _already(src_text, edits):
    """Drop edits that would re-apply (idempotence): an insertion whose text already follows."""
    kept = []
    data = src_text.encode()
    for edit in edits:
        if edit.start == edit.end and data[edit.end:edit.end + len(edit.text.encode())].decode(errors="ignore") == edit.text:
            continue
        kept.append(edit)
    return kept


def _apply(text, edits):
    data = text.encode()
    # Non-overlapping, applied back to front; an edit inside another one waits for the next pass.
    chosen, last_end = [], -1
    for edit in sorted(edits, key=lambda e: (e.start, -e.end)):
        if edit.start < last_end:
            continue
        chosen.append(edit)
        last_end = max(last_end, edit.end)
    for edit in sorted(chosen, key=lambda e: (e.start, e.end), reverse=True):
        data = data[:edit.start] + edit.text.encode() + data[edit.end:]
    return data.decode(), chosen, len(chosen) < len(edits)


def process(path: Path, sites, *, write):
    rel = str(path.relative_to(ROOT))
    text = path.read_text()
    counts, review = Counter(), []
    if path.suffix == ".md":
        new = text
        for match in reversed(list(FENCE.finditer(text))):
            if match["lang"] not in ("python", "py"):
                continue
            body = match["body"]
            indent = match["indent"]
            dedented = "\n".join(line[len(indent):] if line.startswith(indent) else line for line in body.split("\n"))
            try:
                out, c, r = _rewrite(dedented, sites, rel, ensure_import=False)
            except SyntaxError:
                continue
            counts.update(c)
            review.extend(r)
            out = "\n".join(indent + line if line else line for line in out.split("\n"))
            new = new[:match.start("body")] + out + new[match.end("body"):]
    else:
        new, counts, review = _rewrite(text, sites, rel)
    if write and new != text:
        path.write_text(new)
    return counts, review, new != text


def _bound_names(tree):
    """Module-level names (imports and assignments): a function-local ``import ... as mw`` does not
    bind ``mw`` for the other functions of the module."""
    names = set()
    for node in tree.body:
        for sub in ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else ():
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                names.update((alias.asname or alias.name).split(".")[0] for alias in sub.names)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                names.add(sub.id)
    return names


NAMESPACE_CALL = re.compile(r"\bmw\.(" + "|".join(sorted({ns for ns, _ in MOVED.values()} | {"postprocess"}))
                            + r")\.[a-z_]+\(")


def _ensure_import(text):
    """Add ``import witwin.maxwell as mw`` where a rewrite introduced ``mw.<namespace>.<function>(`` unbound."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False
    uses = any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
               and isinstance(node.value.value, ast.Name) and node.value.value.id == "mw"
               for node in ast.walk(tree))
    if not uses or "mw" in _bound_names(tree) or not NAMESPACE_CALL.search(text):
        return text, False
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    lines = text.split("\n")
    if imports:
        at = imports[-1].end_lineno
    else:
        at = 0
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
            at = tree.body[0].end_lineno
    lines.insert(at, "import witwin.maxwell as mw")
    return "\n".join(lines), True


def _rewrite(text, sites, rel, *, ensure_import=True):
    counts, review = Counter(), []
    for _ in range(6):
        edits, found_review = plan(None, text, sites, rel)
        edits = _already(text, edits)
        if not review:
            review = found_review
        if not edits:
            break
        text, chosen, more = _apply(text, edits)
        counts.update(edit.decision for edit in chosen)
        if not more:
            break
    text, added = _ensure_import(text) if ensure_import else (text, False)
    if added:
        counts["import_mw"] += 1
    return text, counts, review


def files(paths):
    pinned = _pinned_hashes()
    for name in _tracked(paths):
        # tests/mom/ and benchmark/mom_* (MoM-owned) are out of the default scope only; an explicit path migrates them.
        if EXCLUDE.match(name) and not (paths and name.startswith(("tests/mom/", "benchmark/mom_"))):
            continue
        path = ROOT / name
        if _sha256(path) in pinned:
            yield path, "skip"
        else:
            yield path, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sites", "classify", "apply", "check"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "sites":
        build_sites(args.paths)
        return 0
    sites = load_sites()
    total, changed, reviews, skipped = Counter(), [], [], []
    for path, status in files(args.paths):
        rel = path.relative_to(ROOT)
        if status == "skip":
            skipped.append(rel)
            continue
        try:
            counts, review, differs = process(path, sites, write=args.command == "apply")
        except (SyntaxError, UnicodeDecodeError):
            continue
        total.update(counts)
        reviews.extend(f"{rel}:{line}: {message}" for line, message in review)
        if differs:
            changed.append(rel)
    for rel in skipped:
        print(f"skip (sha256 pinned): {rel}")
    if args.command == "check":
        # Old-form reads that would still be rewritten; review items are hand-checked sites.
        for rel in changed:
            print(f"old-form Result read: {rel}")
        print(f"check: {len(changed)} files with old-form reads")
        return 1 if changed else 0
    for decision, count in sorted(total.items()):
        print(f"{decision:24s} {count}")
    print(f"files {'rewritten' if args.command == 'apply' else 'to rewrite'}: {len(changed)}")
    for line in reviews:
        print(f"review {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

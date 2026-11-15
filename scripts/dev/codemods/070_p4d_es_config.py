"""Move the options of ``Simulation.electrostatic``/``Simulation.capacitance`` into ``config=``.

0.7.0 phase 4d (PLAN section 6 Q6). Both factories return a ``Simulation`` and take
one configuration entry::

    Simulation.electrostatic(scene, *, config=ElectrostaticConfig(boundary=..., solver=...))
    Simulation.capacitance(scene, *, config=CapacitanceConfig(terminals=..., reference=...,
                                                              boundary=..., solver=...,
                                                              truncation_estimate=...))

The script rewrites call sites mechanically (``ast`` positions, layout kept):

* every option keyword moves into ``config=<prefix>ElectrostaticConfig(...)`` /
  ``config=<prefix>CapacitanceConfig(...)``, where the prefix is the one the call spells
  ``Simulation`` with (``mw.``, or a bare name that is then added to the
  ``from witwin.maxwell import ...`` that imports ``Simulation``);
* ``boundary=None`` and ``solver=None`` are dropped (the old factories replaced ``None``
  by the default; the config fields take a value);
* ``<factory>(...).prepare()`` becomes ``<factory>(...).prepare().solver``: ``prepare()``
  now returns a ``PreparedSimulation`` whose ``solver`` is the compiled electrostatic block.

Rewritten and reported for review: a maybe-``None`` ``boundary=``/``solver=`` value. Reported
for review, not rewritten: ``**options`` in a factory call, ``prepare()`` or runner attributes (``terminals``,
``reference``, ``boundary``, ``solver``, ``truncation_estimate``) read through a name bound
to a factory call (now ``simulation.config.<field>`` and ``prepare().solver``), and the
removed ``ElectrostaticSimulation``/``CapacitanceSimulation`` names. Markdown files are
rewritten inside fenced ``python`` blocks that parse.

Commands (run from any directory with the maxwell environment)::

    classify [PATH ...]   per-site table: rewrite / review / clean, plus review reasons
    apply [PATH ...]      rewrite in place (idempotent)
    check [PATH ...]      exit 1 while an old-form call or a removed runner name remains

Without paths the scope is every tracked ``.py`` and ``.md`` file under ``tests``,
``benchmark``, ``scripts``, ``examples``, ``experiments``, ``docs/reference``,
``docs/memory``, ``docs/README.md``, ``README.md`` and ``FEATURE_LIST.md``. Excluded:
``witwin/maxwell/mom/`` and ``tests/mom/`` (MoM line), historical records
(``docs/assessments``, ``docs/archive``, ``docs/plans``, ``docs/releases``,
``docs/reference/release-notes-*``, ``CHANGELOG.md``) and the 0.7.0 codemods.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

MOVED = {"electrostatic": frozenset({"boundary", "solver"}),
         "capacitance": frozenset({"terminals", "reference", "boundary", "solver", "truncation_estimate"})}
CONFIG = {"electrostatic": "ElectrostaticConfig", "capacitance": "CapacitanceConfig"}
KEPT = frozenset({"scene", "config"})
# Fields whose ``None`` the old factories replaced by the default.
NONE_DEFAULTED = frozenset({"boundary", "solver"})
RUNNER_ATTRIBUTES = frozenset({"terminals", "reference", "boundary", "solver", "truncation_estimate", "prepare",
                               "check", "scene"})
REMOVED_NAMES = ("ElectrostaticSimulation", "CapacitanceSimulation")
MAXWELL_MODULES = ("witwin.maxwell", "witwin.maxwell.simulation")

SCOPE = ("tests", "benchmark", "scripts", "examples", "experiments", "docs/reference", "docs/memory",
         "docs/README.md", "README.md", "FEATURE_LIST.md")
EXCLUDE = re.compile(r"^(witwin/maxwell/mom/|tests/mom/|docs/(assessments|archive|plans|releases)/|"
                     r"docs/reference/release-notes-|CHANGELOG\.md$|scripts/dev/codemods/070_)")
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)
OLD_TEXT = re.compile(r"Simulation\.(electrostatic|capacitance)\(|\b(ElectrostaticSimulation|CapacitanceSimulation)\b")


@dataclass
class Site:
    path: str
    line: int
    kind: str          # electrostatic | capacitance | prepare | runner | imports
    status: str        # rewrite | review | clean | old
    reasons: list[str] = field(default_factory=list)


class Source:
    """Text with ``ast`` (line, UTF-8 byte column) -> character offset conversion."""

    def __init__(self, text: str):
        self.text = text
        self.starts = [0]
        for line in text.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.lines = text.splitlines(keepends=True)

    def offset(self, line: int, col: int) -> int:
        raw = self.lines[line - 1].encode("utf-8")[:col]
        return self.starts[line - 1] + len(raw.decode("utf-8"))

    def span(self, node) -> tuple[int, int]:
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)


def apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    edits = sorted(edits)
    for (s0, e0, _), (s1, _e1, _) in zip(edits, edits[1:]):
        if s1 < e0:
            raise RuntimeError(f"overlapping edits at {s0}..{e0} and {s1}")
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def factory_kind(func) -> str | None:
    if not isinstance(func, ast.Attribute) or func.attr not in MOVED:
        return None
    value = func.value
    if (isinstance(value, ast.Name) and value.id == "Simulation") or (
            isinstance(value, ast.Attribute) and value.attr == "Simulation"):
        return func.attr
    return None


def _is_factory_call(node) -> bool:
    return isinstance(node, ast.Call) and factory_kind(node.func) is not None


def config_prefix(source: Source, func: ast.Attribute) -> str:
    value = func.value
    if isinstance(value, ast.Name):
        return ""
    start, end = source.span(value.value)
    return source.text[start:end] + "."


def _maybe_none(value) -> bool:
    if isinstance(value, ast.Call):
        return False
    if isinstance(value, ast.Constant):
        return value.value is None
    if isinstance(value, ast.IfExp):
        return _maybe_none(value.body) or _maybe_none(value.orelse)
    return True


def _reindent(text: str, extra: str) -> str:
    lines = text.split("\n")
    return "\n".join([lines[0], *[(extra + line) if line.strip() else line for line in lines[1:]]])


# --- pass 1: options -> config= ---------------------------------------------------------
def plan_call(source: Source, call, kind: str, path: str) -> tuple[list[tuple[int, int, str]], Site]:
    site = Site(path, call.lineno, kind, "clean")
    moved_names = MOVED[kind]
    items = sorted([*call.args, *call.keywords], key=lambda node: (node.lineno, node.col_offset))
    spans = {id(item): source.span(item) for item in items}
    index = {id(item): position for position, item in enumerate(items)}
    if len(call.args) > 1:
        site.reasons.append("extra positional argument")
    for keyword in call.keywords:
        if keyword.arg is None:
            site.reasons.append("**mapping in a factory call")
        elif keyword.arg not in moved_names and keyword.arg not in KEPT:
            site.reasons.append(f"unknown keyword {keyword.arg}=")
        elif keyword.arg in NONE_DEFAULTED and _maybe_none(keyword.value) and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None):
            site.reasons.append(f"{keyword.arg}= may be None (the config field is not optional)")
    if any(keyword.arg == "config" for keyword in call.keywords):
        if any(keyword.arg in moved_names for keyword in call.keywords):
            site.reasons.append("config= next to moved options")
        site.status = "review" if site.reasons else "clean"
        return [], site
    dropped = [kw for kw in call.keywords if kw.arg in NONE_DEFAULTED
               and isinstance(kw.value, ast.Constant) and kw.value.value is None]
    moved = [kw for kw in call.keywords if kw.arg in moved_names and kw not in dropped]
    if any(not reason.endswith("(the config field is not optional)") for reason in site.reasons):
        site.status = "review"
        return [], site
    if not moved and not dropped:
        return [], site
    site.status = "review" if site.reasons else "rewrite"

    def cut_with_separator(item):
        position = index[id(item)]
        start, end = spans[id(item)]
        if position + 1 < len(items):
            return start, spans[id(items[position + 1])][0]
        if position > 0:
            return spans[id(items[position - 1])][1], end
        return start, end

    if not moved:
        return [(*cut_with_separator(kw), "") for kw in dropped], site
    edits: list[tuple[int, int, str]] = []
    first, last = index[id(moved[0])], index[id(moved[-1])]
    segment_start, segment_end = spans[id(items[first])][0], spans[id(items[last])][1]
    inner = items[first:last + 1]
    for item in dropped:
        if item not in inner:
            edits.append((*cut_with_separator(item), ""))
    body = source.text[segment_start:segment_end]
    for item in sorted((item for item in inner if item not in moved), key=lambda item: spans[id(item)],
                       reverse=True):
        # A dropped ``None`` inside the moved run: cut it with the separator that follows it.
        start = spans[id(item)][0]
        end = spans[id(items[index[id(item)] + 1])][0]
        body = body[:start - segment_start] + body[end - segment_start:]
    line_start = source.text.rfind("\n", 0, segment_start) + 1
    leading = source.text[line_start:segment_start]
    name = config_prefix(source, call.func) + CONFIG[kind]
    if "\n" in body and not leading.strip():
        body = leading + "    " + _reindent(body, "    ").rstrip().rstrip(",")
        text = f"config={name}(\n{body},\n{leading})"
    elif "\n" in body:
        # The options start after ``scene, `` and continue on the following lines.
        indent = leading[:len(leading) - len(leading.lstrip())] + "    "
        text = f"config={name}(\n{indent}" + _reindent(body, "    ").rstrip().rstrip(",") + ")"
    else:
        text = f"config={name}({body})"
    edits.append((segment_start, segment_end, text))
    return edits, site


def pass_config(source: Source, tree, path: str) -> tuple[list, list[Site]]:
    edits, sites = [], []
    for call in sorted((node for node in ast.walk(tree) if _is_factory_call(node)), key=source.span):
        call_edits, site = plan_call(source, call, factory_kind(call.func), path)
        sites.append(site)
        edits.extend(call_edits)
    return edits, sites


# --- pass 2: <factory>(...).prepare() -> .prepare().solver -------------------------------
def _parents(tree) -> dict[int, ast.AST]:
    return {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def pass_prepare(source: Source, tree) -> list:
    parents = _parents(tree)
    edits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "prepare"
                and _is_factory_call(node.func.value) and not node.args and not node.keywords):
            parent = parents.get(id(node))
            if isinstance(parent, ast.Attribute) and parent.attr == "solver":
                continue
            _, end = source.span(node)
            edits.append((end, end, ".solver"))
    return edits


# --- pass 3: imports for bare config names ----------------------------------------------
def _imports(tree) -> tuple[dict[str, list[ast.ImportFrom]], set[str]]:
    names, bound = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
                if node.module in MAXWELL_MODULES and alias.asname is None:
                    names.setdefault(alias.name, []).append(node)
    return names, bound


def pass_imports(source: Source, tree) -> list:
    names, bound = _imports(tree)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    edits = []
    for wanted in CONFIG.values():
        if wanted in used and wanted not in bound:
            for statement in names.get("Simulation", []):
                alias = next(alias for alias in statement.names if alias.name == "Simulation")
                _, end = source.span(alias)
                edits.append((end, end, f", {wanted}"))
    return edits


# --- review: names bound to a factory call, removed runner names ------------------------
def review_sites(tree, path: str) -> list[Site]:
    bound = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign) and _is_factory_call(node.value)
             for target in node.targets if isinstance(target, ast.Name)}
    sites = [Site(path, node.lineno, "runner", "review", [f"{node.value.id}.{node.attr} on a static Simulation"])
             for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id in bound and node.attr in RUNNER_ATTRIBUTES]
    sites += [Site(path, node.lineno, "runner", "review", [f"removed name {name}"])
              for node in ast.walk(tree) for name in _removed_names(node)]
    names, bound_names = _imports(tree)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    missing = sorted(name for name in CONFIG.values() if name in used and name not in bound_names)
    if missing:
        sites.append(Site(path, 1, "imports", "review", [f"unresolved name {name}" for name in missing]))
    return sites


def _removed_names(node) -> list[str]:
    if isinstance(node, ast.Name) and node.id in REMOVED_NAMES:
        return [node.id]
    if isinstance(node, ast.Attribute) and node.attr in REMOVED_NAMES:
        return [node.attr]
    if isinstance(node, ast.ImportFrom):
        return [alias.name for alias in node.names if alias.name in REMOVED_NAMES]
    return []


def rewrite_python(text: str, path: str) -> tuple[str, list[Site]]:
    source = Source(text)
    edits, sites = pass_config(source, ast.parse(text), path)
    text = apply_edits(text, edits)
    for step in (pass_prepare, pass_imports):
        source = Source(text)
        text = apply_edits(text, step(source, ast.parse(text)))
    sites.extend(review_sites(ast.parse(text), path))
    return text, sites


def leftovers(text: str, path: str) -> list[Site]:
    """Old-form sites (``check``)."""
    tree = ast.parse(text)
    out = []
    for node in ast.walk(tree):
        if _is_factory_call(node):
            kind = factory_kind(node.func)
            bad = [kw.arg or "**" + ast.unparse(kw.value) for kw in node.keywords
                   if kw.arg is None or kw.arg in MOVED[kind]]
            if bad:
                out.append(Site(path, node.lineno, kind, "old", [", ".join(bad)]))
        for name in _removed_names(node):
            out.append(Site(path, node.lineno, "runner", "old", [name]))
    return out


def _dedent(body: str) -> tuple[str, str]:
    lines = [line for line in body.splitlines() if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in lines), default=0)
    prefix = lines[0][:indent] if lines else ""
    return "".join(line[indent:] if line.strip() else line for line in body.splitlines(keepends=True)), prefix


def _fenced(text: str):
    for match in FENCE.finditer(text):
        if match.group("lang").lower() in ("python", "py", "") and OLD_TEXT.search(match.group("body")):
            yield match, text.count("\n", 0, match.start("body")) + 1


def rewrite_markdown(text: str, path: str) -> tuple[str, list[Site]]:
    sites, edits = [], []
    for match, line in _fenced(text):
        code, prefix = _dedent(match.group("body"))
        try:
            new, block_sites = rewrite_python(code, path)
        except SyntaxError:
            sites.append(Site(path, line, "markdown", "review", ["fenced block does not parse"]))
            continue
        for site in block_sites:
            site.line += line - 1
        sites.extend(block_sites)
        if new != code:
            new = "".join(prefix + item if item.strip() else item for item in new.splitlines(keepends=True))
            edits.append((match.start("body"), match.end("body"), new))
    return apply_edits(text, edits), sites


def markdown_leftovers(text: str, path: str) -> list[Site]:
    out = []
    for match, line in _fenced(text):
        try:
            found = leftovers(_dedent(match.group("body"))[0], path)
        except SyntaxError:
            continue
        for site in found:
            site.line += line - 1
        out.extend(found)
    return out


def targets(paths: list[str]) -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "--", *(paths or SCOPE)], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.split()
    return [ROOT / item for item in listed if item.endswith((".py", ".md")) and not EXCLUDE.search(item)]


def run(command: str, paths: list[str]) -> int:
    sites, changed, old = [], 0, []
    for file in targets(paths):
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not OLD_TEXT.search(text):
            continue
        rel = str(file.relative_to(ROOT))
        markdown = file.suffix == ".md"
        try:
            new, found = (rewrite_markdown if markdown else rewrite_python)(text, rel)
        except SyntaxError as error:
            sites.append(Site(rel, error.lineno or 1, "parse", "review", [str(error.msg)]))
            continue
        sites.extend(found)
        if command == "apply" and new != text:
            file.write_text(new)
            changed += 1
        if command == "check":
            old.extend((markdown_leftovers if markdown else leftovers)(text, rel))
    if command == "classify":
        counts: dict[tuple[str, str], int] = {}
        for site in sites:
            counts[(site.kind, site.status)] = counts.get((site.kind, site.status), 0) + 1
        for (kind, status), count in sorted(counts.items()):
            print(f"{kind:13s} {status:8s} {count}")
    review = [site for site in sites if site.status == "review"]
    for site in review:
        print(f"review {site.path}:{site.line} [{site.kind}] {'; '.join(site.reasons)}")
    if command == "apply":
        print(f"rewrote {changed} file(s); {len(review)} site(s) to review")
    if command == "check":
        for site in old:
            print(f"old {site.path}:{site.line} [{site.kind}] {'; '.join(site.reasons)}")
        print(f"{len(old)} old-form site(s)")
        return 1 if old else 0
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("classify", "apply", "check"))
    parser.add_argument("paths", nargs="*", help="repository-relative paths (default: the p4d scope)")
    args = parser.parse_args(argv)
    return run(args.command, args.paths)


if __name__ == "__main__":
    sys.exit(main())

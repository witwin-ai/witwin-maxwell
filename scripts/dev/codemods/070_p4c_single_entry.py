"""Move per-run options of ``Simulation.fdtd``/``Simulation.fdfd`` into ``config=``.

0.7.0 phase 4c. The factories take one configuration entry::

    Simulation.fdtd(scene, *, frequencies, excitations=None, config=FDTDConfig(...))
    Simulation.fdfd(scene, *, frequencies, config=FDFDConfig(...))

The script rewrites call sites mechanically (``ast`` positions, layout kept):

* every option keyword that is an ``FDTDConfig``/``FDFDConfig`` field moves into
  ``config=<prefix>FDTDConfig(...)`` / ``config=<prefix>FDFDConfig(...)``, where the
  prefix is the one the call spells ``Simulation`` with (``mw.``, or a bare name that
  is then added to the ``from witwin.maxwell import ...`` that imports ``Simulation``);
* ``frequency=`` becomes ``frequencies=`` (Q1: one entry, a float or a sequence);
* ``run_time=None``, ``spectral_sampler=None``, ``cpml=None`` and ``solver=None`` are
  dropped (the old factories replaced ``None`` by the config default, the configs do not);
* ``**options`` in a factory call moves into the config call (flagged for review: the
  mapping must hold only config fields);
* ``GMRES`` becomes ``FDFDSolverConfig`` (Q8) where it is imported from ``witwin.maxwell``
  or spelled ``<module alias>.GMRES``.

Markdown files are rewritten inside fenced ``python`` blocks that parse; the others are
reported. ``Simulation.mom(scene, *, frequencies, config=MoMConfig(...))`` has no options to
move; only its ``frequency=`` becomes ``frequencies=`` (Q1 for MoM, applied by the MoM line).

Commands (run from any directory with the maxwell environment)::

    classify [PATH ...]   per-site table: rewrite / review / clean, plus review reasons
    apply [PATH ...]      rewrite in place (idempotent; re-run after a rebase)
    check [PATH ...]      exit 1 while an old-form call or GMRES spelling remains

Without paths the scope is every tracked ``.py`` and ``.md`` file under ``tests``,
``benchmark``, ``scripts``, ``examples``, ``experiments``, ``docs/reference``,
``docs/memory``, ``docs/README.md``, ``README.md`` and ``FEATURE_LIST.md``. Excluded:
``witwin/maxwell/mom/`` (MoM package internals), historical records
(``docs/assessments``, ``docs/archive``, ``docs/plans``, ``docs/releases``,
``docs/reference/release-notes-*``, ``CHANGELOG.md``), this script and
``tests/support/api_dialect.py`` (it translates the 0.6 keyword set itself).
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

FDTD_FIELDS = frozenset({
    "run_time", "spectral_sampler", "reference_frequency", "enable_plot", "full_field_dft", "field_dtype",
    "absorber", "cpml", "adjoint_checkpoint_stride", "shutoff", "shutoff_check_interval", "cuda_graph",
    "parallel", "initial_condition", "material_cache",
})
FDFD_FIELDS = frozenset({"solver", "enable_plot", "verbose", "local_response", "observations"})
KEPT = {"fdtd": frozenset({"scene", "frequencies", "frequency", "excitations", "config"}),
        "fdfd": frozenset({"scene", "frequencies", "frequency", "config"}),
        "mom": frozenset({"scene", "frequencies", "frequency", "config"})}
MOVED = {"fdtd": FDTD_FIELDS, "fdfd": FDFD_FIELDS, "mom": frozenset()}
CONFIG = {"fdtd": "FDTDConfig", "fdfd": "FDFDConfig", "mom": "MoMConfig"}
# Fields whose ``None`` the old factories replaced by the config default.
NONE_DEFAULTED = {"fdtd": frozenset({"run_time", "spectral_sampler", "cpml"}), "fdfd": frozenset({"solver"}),
                  "mom": frozenset()}
MAXWELL_MODULES = ("witwin.maxwell", "witwin.maxwell.simulation")
OLD_SOLVER, NEW_SOLVER = "GMRES", "FDFDSolverConfig"

# Reviewed ``**mapping`` arguments that already hold the new entry keywords (frequencies=,
# excitations=, config=); they stay on the factory call. Every other ``**mapping`` moves
# into the config call and is listed for review. Keyed by path and the mapping's source.
ENTRY_MAPPINGS = frozenset({
    ("benchmark/datagen/profile.py", 'sample.fdtd_kwargs | {\'config\': config}'),
    ("benchmark/datagen/profile.py", "control_kwargs"),
    ("benchmark/datagen/repro/copper_dipole_stability.py", "kw"),
    ("benchmark/datagen/repro/helix_accepted_power.py", "kwargs"),
    ("benchmark/datagen/scheduler.py", "sample.fdtd_kwargs"),
    ("benchmark/label_qualification/random_convergence.py", "sample.fdtd_kwargs"),
    ("scripts/dev/fdtd/subpixel_convergence.py", "sim_kw"),
    ("scripts/dev/fdtd/thinwire_finite_dipole_repro.py", "kwargs"),
    ("tests/rf/network/test_network_runtime.py", "run_args"),
})

SCOPE = ("tests", "benchmark", "scripts", "examples", "experiments", "docs/reference", "docs/memory",
         "docs/README.md", "README.md", "FEATURE_LIST.md")
EXCLUDE = re.compile(r"^(witwin/maxwell/mom/|docs/(assessments|archive|plans|releases)/|"
                     r"docs/reference/release-notes-|CHANGELOG\.md$|scripts/dev/codemods/070_p4c_single_entry\.py$|"
                     r"tests/support/api_dialect\.py$|benchmark/sie_single_buffer_equivalence_v3\.py$)")
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)


@dataclass
class Site:
    path: str
    line: int
    kind: str          # fdtd | fdfd | mom | reference | markdown
    status: str        # rewrite | review | clean
    reasons: list[str] = field(default_factory=list)


# --- source positions -------------------------------------------------------------------
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


# --- name resolution --------------------------------------------------------------------
def maxwell_aliases(tree) -> tuple[set[str], dict[str, list[ast.ImportFrom]], set[str]]:
    """(module aliases of witwin.maxwell, imported name -> ImportFrom nodes, every bound import name)."""
    modules, names, bound = set(), {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "witwin.maxwell" and alias.asname:
                    modules.add(alias.asname)
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
                if node.module == "witwin" and alias.name == "maxwell":
                    modules.add(alias.asname or "maxwell")
                if node.module in MAXWELL_MODULES and alias.asname is None:
                    names.setdefault(alias.name, []).append(node)
    return modules, names, bound


def factory_kind(func) -> str | None:
    if not isinstance(func, ast.Attribute) or func.attr not in MOVED:
        return None
    value = func.value
    if isinstance(value, ast.Name) and value.id == "Simulation":
        return func.attr
    if isinstance(value, ast.Attribute) and value.attr == "Simulation":
        return func.attr
    return None


def config_prefix(source: Source, func: ast.Attribute) -> str:
    value = func.value
    if isinstance(value, ast.Name):
        return ""
    start, end = source.span(value.value)
    return source.text[start:end] + "."


# --- pass 1: move options into config= -------------------------------------------------
def _maybe_none(value) -> bool:
    if isinstance(value, ast.Call):
        return False
    if isinstance(value, ast.Constant):
        return value.value is None
    if isinstance(value, ast.IfExp):
        return _maybe_none(value.body) or _maybe_none(value.orelse)
    return True


def _items(call) -> list:
    items = [*call.args, *call.keywords]
    return sorted(items, key=lambda node: (node.lineno, node.col_offset))


def _reindent(text: str, extra: str) -> str:
    lines = text.split("\n")
    return "\n".join([lines[0], *[(extra + line) if line.strip() else line for line in lines[1:]]])


def plan_call(source: Source, call, kind: str, path: str) -> tuple[list[tuple[int, int, str]], Site]:
    site = Site(path, call.lineno, kind, "clean")
    moved_names, kept_names = MOVED[kind], KEPT[kind]
    items = _items(call)
    spans = {id(item): source.span(item) for item in items}
    for item in call.args[1:]:
        site.reasons.append("extra positional argument")
    entry = [kw for kw in call.keywords if kw.arg is None and (path, ast.unparse(kw.value)) in ENTRY_MAPPINGS]
    for keyword in call.keywords:
        if keyword.arg is None and keyword in entry:
            continue
        if keyword.arg is None:
            site.reasons.append("**mapping moved into config (must hold config fields only)")
        elif keyword.arg not in moved_names and keyword.arg not in kept_names:
            site.reasons.append(f"unknown keyword {keyword.arg}=")
        elif keyword.arg in NONE_DEFAULTED[kind] and _maybe_none(keyword.value) and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None):
            site.reasons.append(f"{keyword.arg}= may be None (the config field is not optional)")
    if any(keyword.arg == "config" for keyword in call.keywords):
        if any(keyword.arg in moved_names or (keyword.arg is None and keyword not in entry)
               for keyword in call.keywords):
            site.reasons.append("config= next to moved options")
            site.status = "review"
        return [], site

    dropped = [kw for kw in call.keywords if kw.arg in NONE_DEFAULTED[kind]
               and isinstance(kw.value, ast.Constant) and kw.value.value is None]
    moved = [kw for kw in call.keywords
             if (kw.arg in moved_names or (kw.arg is None and kw not in entry)) and kw not in dropped]
    if not moved and not dropped:
        return [], site
    site.status = "review" if site.reasons else "rewrite"
    edits: list[tuple[int, int, str]] = []
    index = {id(item): position for position, item in enumerate(items)}
    call_start, call_end = source.span(call)

    def cut_with_separator(item):
        """Span of ``item`` plus the separator after it (or before it, for the last argument)."""
        position = index[id(item)]
        start, end = spans[id(item)]
        if position + 1 < len(items):
            return start, spans[id(items[position + 1])][0]
        if position > 0:
            return spans[id(items[position - 1])][1], end
        return start, end

    if not moved:
        return [(*cut_with_separator(kw), "") for kw in dropped], site

    first, last = index[id(moved[0])], index[id(moved[-1])]
    segment_start, segment_end = spans[id(items[first])][0], spans[id(items[last])][1]
    inner = [items[position] for position in range(first, last + 1)]
    lifted = [item for item in inner if item not in moved]
    for item in dropped:
        if item not in inner:
            edits.append((*cut_with_separator(item), ""))
    cuts = []
    for item in lifted:
        start, _ = spans[id(item)]
        cuts.append((start, spans[id(items[index[id(item)] + 1])][0]))
    body = source.text[segment_start:segment_end]
    for start, end in sorted(cuts, reverse=True):
        body = body[:start - segment_start] + body[end - segment_start:]
    lifted_text = [source.text[slice(*spans[id(item)])] for item in lifted if item not in dropped]

    line_start = source.text.rfind("\n", 0, segment_start) + 1
    leading = source.text[line_start:segment_start]
    multiline = "\n" in body and not leading.strip()
    strings = any(isinstance(node, ast.Constant) and isinstance(node.value, str) and node.lineno != node.end_lineno
                  for kw in moved for node in ast.walk(kw.value))
    name = config_prefix(source, call.func) + CONFIG[kind]
    if multiline and not strings:
        indent = leading
        body = indent + "    " + _reindent(body, "    ").rstrip().rstrip(",")
        text = f"config={name}(\n{body},\n{indent})"
        text = "".join(f"{item},\n{indent}" for item in lifted_text) + text
    else:
        text = "".join(f"{item}, " for item in lifted_text) + f"config={name}({body})"
    edits.append((segment_start, segment_end, text))
    del call_start, call_end
    return edits, site


def pass_config(source: Source, tree, path: str) -> tuple[list, list[Site]]:
    edits, sites, taken = [], [], []
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and factory_kind(node.func)]
    calls.sort(key=lambda node: source.span(node))
    for call in calls:
        start, end = source.span(call)
        if any(s <= start and end <= e for s, e in taken):
            continue  # nested factory call: the next fixpoint round handles it
        call_edits, site = plan_call(source, call, factory_kind(call.func), path)
        sites.append(site)
        if call_edits:
            taken.append((start, end))
            edits.extend(call_edits)
    return edits, sites


# --- pass 2: frequency= -> frequencies= -------------------------------------------------
def pass_frequency(source: Source, tree) -> list:
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and factory_kind(node.func):
            for keyword in node.keywords:
                if keyword.arg == "frequency":
                    start = source.offset(keyword.lineno, keyword.col_offset)
                    edits.append((start, start + len("frequency"), "frequencies"))
    return edits


# --- pass 3: GMRES -> FDFDSolverConfig --------------------------------------------------
def pass_solver(source: Source, tree) -> list:
    modules, names, _ = maxwell_aliases(tree)
    edits = []
    imported = OLD_SOLVER in names
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == OLD_SOLVER and isinstance(node.value, ast.Name) \
                and node.value.id in modules | {"mw"}:
            _, end = source.span(node)
            edits.append((end - len(OLD_SOLVER), end, NEW_SOLVER))
        elif imported and isinstance(node, ast.Name) and node.id == OLD_SOLVER:
            edits.append((*source.span(node), NEW_SOLVER))
    for statement in names.get(OLD_SOLVER, ()):
        for alias in statement.names:
            if alias.name == OLD_SOLVER:
                edits.append((*source.span(alias), NEW_SOLVER))
    return edits


# --- pass 4: imports for bare config names ----------------------------------------------
def pass_imports(source: Source, tree) -> list:
    _, names, bound = maxwell_aliases(tree)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    edits = []
    for wanted in ("FDTDConfig", "FDFDConfig"):
        if wanted not in used or wanted in bound:
            continue
        statements = names.get("Simulation", [])
        if not statements:
            continue
        for statement in statements:
            alias = next(alias for alias in statement.names if alias.name == "Simulation")
            _, end = source.span(alias)
            edits.append((end, end, f", {wanted}"))
    return edits


def unresolved_config_names(tree) -> list[str]:
    _, _, bound = maxwell_aliases(tree)
    defined = bound | {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return sorted(name for name in ("FDTDConfig", "FDFDConfig", NEW_SOLVER) if name in used and name not in defined)


# --- drivers ----------------------------------------------------------------------------
def rewrite_python(text: str, path: str) -> tuple[str, list[Site]]:
    sites: list[Site] = []
    for _ in range(8):
        source = Source(text)
        tree = ast.parse(text)
        edits, round_sites = pass_config(source, tree, path)
        if not sites:
            sites = round_sites
        if not edits:
            break
        text = apply_edits(text, edits)
    for step in (pass_frequency, pass_solver, pass_imports):
        source = Source(text)
        text = apply_edits(text, step(source, ast.parse(text)))
    tree = ast.parse(text)
    sites.extend(references(Source(text), tree, path))
    missing = unresolved_config_names(tree)
    if missing:
        sites.append(Site(path, 1, "imports", "review", [f"unresolved name {name}" for name in missing]))
    return text, sites


def references(source: Source, tree, path: str) -> list[Site]:
    funcs = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return [Site(path, node.lineno, "reference", "review", ["factory used as a value; check forwarded keywords"])
            for node in ast.walk(tree) if isinstance(node, ast.Attribute) and id(node) not in funcs
            and factory_kind(node)]


def leftovers(text: str, path: str) -> list[Site]:
    """Old-form sites after a rewrite (``check``)."""
    tree = ast.parse(text)
    source = Source(text)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and factory_kind(node.func):
            kind = factory_kind(node.func)
            bad = [kw.arg or "**" + ast.unparse(kw.value) for kw in node.keywords if kw.arg == "frequency"
                   or kw.arg in MOVED[kind]
                   or (kw.arg is None and (path, ast.unparse(kw.value)) not in ENTRY_MAPPINGS)]
            if bad:
                out.append(Site(path, node.lineno, kind, "old", [", ".join(bad)]))
    _, names, _ = maxwell_aliases(tree)
    if OLD_SOLVER in names or any(isinstance(node, ast.Attribute) and node.attr == OLD_SOLVER
                                  and isinstance(node.value, ast.Name) for node in ast.walk(tree)):
        out.append(Site(path, 1, "solver", "old", [OLD_SOLVER]))
    del source
    return out


def _dedent(body: str) -> tuple[str, str]:
    lines = [line for line in body.splitlines() if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in lines), default=0)
    prefix = lines[0][:indent] if lines else ""
    return "".join(line[indent:] if line.strip() else line for line in body.splitlines(keepends=True)), prefix


OLD_TEXT = re.compile(r"Simulation\.(fdtd|fdfd|mom)\(|\bGMRES\b")


def rewrite_markdown(text: str, path: str) -> tuple[str, list[Site]]:
    sites, edits = [], []
    for match in FENCE.finditer(text):
        if match.group("lang").lower() not in ("python", "py", ""):
            continue
        body = match.group("body")
        if not OLD_TEXT.search(body):
            continue
        code, prefix = _dedent(body)
        line = text.count("\n", 0, match.start("body")) + 1
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
    for match in FENCE.finditer(text):
        code, _ = _dedent(match.group("body"))
        line = text.count("\n", 0, match.start("body")) + 1
        try:
            found = leftovers(code, path)
        except SyntaxError:
            continue
        for site in found:
            site.line += line - 1
        out.extend(found)
    return out


def targets(paths: list[str]) -> list[Path]:
    roots = paths or list(SCOPE)
    listed = subprocess.run(["git", "ls-files", "--", *roots], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.split()
    return [ROOT / item for item in listed if item.endswith((".py", ".md")) and not EXCLUDE.search(item)]


def run(command: str, paths: list[str]) -> int:
    sites, changed, old = [], 0, []
    for file in targets(paths):
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not (OLD_TEXT.search(text) or "Config(" in text):
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
            print(f"{kind:10s} {status:8s} {count}")
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
    parser.add_argument("paths", nargs="*", help="repository-relative paths (default: the p4c scope)")
    args = parser.parse_args(argv)
    return run(args.command, args.paths)


if __name__ == "__main__":
    sys.exit(main())

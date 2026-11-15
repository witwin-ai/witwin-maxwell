"""Rewrite callers of the ``Scene``/``PreparedScene`` compile methods (0.7.0 phase 7b).

The compile methods are internal functions of ``witwin.maxwell.compiler`` that take the prepared scene
first: ``prepared.compile_materials()`` is ``compile_materials(prepared)``. ``PreparedScene`` is no longer a
``Scene``; its declarations are ``prepared.scene``. The script rewrites call sites mechanically (``ast``
positions, layout kept):

* ``prepared``: a receiver that is a prepared scene: ``X.compile_m(args)`` -> ``compile_m(X, args)``
  (``subpixel_samples=`` becomes ``subpixel=``). ``compile_array_monitors`` takes the declarations:
  ``compile_array_monitors(X.scene, args)`` (with ``device=X.device`` unless given);
* ``scene``: a receiver that is a declarative ``Scene``: ``X.compile_m(device=D, args)`` ->
  ``compile_m(prepare_scene(X, device=D), args)``, which is what the removed method did;
  ``compile_array_monitors`` and ``compile_electrostatics`` take the ``Scene`` itself:
  ``compile_m(X, args)``.

The receiver kind: the materials family (``compile_materials``, ``compile_material_components``,
``compile_relative_materials``, ``compile_material_tensors``, ``compile_mass_density``,
``compile_gyromagnetic_materials``) only existed on ``PreparedScene``. Otherwise a receiver whose text names
a prepared scene (``prepared``, ``*.solver.scene``, ``prepare_scene(...)``) or a name bound in the same
scope to ``prepare_scene(...)``/``PreparedScene(...)``/``*.solver.scene``/``*.prepared_scene`` is prepared,
anything else is a ``Scene``. A function parameter receiver is reported as ``review``.

The new names are imported from ``witwin.maxwell.compiler`` (``compile_electrostatics`` from
``witwin.maxwell.compiler.electrostatic``) after the module's last top-level import; package modules under
``witwin/`` are rewritten with ``--package`` and get their (relative, function-local) imports by hand.

Commands (run with the maxwell environment)::

    classify [PATH ...]   per-decision counts, then the review list
    apply [PATH ...]      rewrite in place (idempotent)
    check [PATH ...]      exit 1 while a rewritable site remains

Without paths the scope is every tracked ``.py`` file under ``tests``, ``benchmark``, ``scripts``,
``examples`` and ``experiments`` and the Python fences of ``docs/reference``, ``docs/memory``,
``docs/README.md``, ``README.md`` and ``FEATURE_LIST.md``. Excluded: historical records, the codemods, the
consumer-metadata bridge, ``tests/support/api_dialect.py`` (it keeps the 0.6 call for the BITID baseline
trees) and any file whose sha256 a tracked benchmark, script or test file pins (reported as ``skip``).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCOPE = ("tests", "benchmark", "scripts", "examples", "experiments", "docs/reference", "docs/memory", "docs/README.md",
         "README.md", "FEATURE_LIST.md")
EXCLUDE = re.compile(
    r"^(docs/(assessments|archive|plans|releases)/|docs/reference/release-notes-|CHANGELOG\.md$|scripts/dev/codemods/"
    r"|scripts/dev/fdtd/consumer_metadata/|tests/support/api_dialect\.py$)")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)

PREPARED_ONLY = {"compile_materials", "compile_material_components", "compile_relative_materials",
                 "compile_material_tensors", "compile_mass_density", "compile_gyromagnetic_materials"}
SCENE_ARGUMENT = {"compile_array_monitors", "compile_electrostatics"}
METHODS = PREPARED_ONLY | SCENE_ARGUMENT | {
    "compile_ports", "compile_thin_wires", "compile_wire_monitors", "compile_waveports", "compile_lumped_elements",
    "compile_circuits", "compile_networks"}
RENAMED = {"subpixel_samples": "subpixel"}
MODULE = {name: "witwin.maxwell.compiler" for name in METHODS}
MODULE["compile_electrostatics"] = "witwin.maxwell.compiler.electrostatic"
PREPARED_TEXT = re.compile(r"prepare|(^|\.)solver\.scene$|(^|\.)prepared_scene$")
PREPARED_CALLS = ("prepare_scene", "PreparedScene", "_prepared_scene", "_prepare")


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
        except (IsADirectoryError, FileNotFoundError, UnicodeDecodeError):
            pass
    pinned.discard(EMPTY_SHA256)
    return pinned


class Source:
    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def index(self, lineno, col):
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode()[:col].decode(errors="ignore"))

    def span(self, node):
        return self.index(node.lineno, node.col_offset), self.index(node.end_lineno, node.end_col_offset)

    def seg(self, node):
        start, end = self.span(node)
        return self.text[start:end]


def _scopes(tree):
    return [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))]


def _walk_scope(scope):
    """Nodes of ``scope`` without descending into nested function scopes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _call_name(node):
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _module_aliases(tree):
    """Names bound to imported modules (``import x.y as m``, ``from pkg import module``)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if not alias.name.startswith("compile_") and alias.name[:1].islower():
                    names.add(alias.asname or alias.name)
    return names


class Rewriter:
    def __init__(self, text, rel, package):
        self.src = Source(text)
        self.tree = ast.parse(text)
        self.rel = rel
        self.package = package
        self.edits = []
        self.counts = Counter()
        self.review = []
        self.needed = set()
        self.needs_prepare = False

    def kinds(self):
        """``(id(scope), name) -> "prepared" | "scene" | "param"`` for names receivers may use."""
        kinds = {}
        for scope in _scopes(self.tree):
            if not isinstance(scope, ast.Module):
                args = scope.args
                for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                    kinds[(id(scope), arg.arg)] = "param"
            for node in _walk_scope(scope):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                    prepared = (_call_name(value) in PREPARED_CALLS
                                or (isinstance(value, ast.Attribute) and PREPARED_TEXT.search(self.src.seg(value)))
                                or (isinstance(value, ast.Call) and PREPARED_TEXT.search(self.src.seg(value.func))))
                    for target in targets:
                        if isinstance(target, ast.Name):
                            key = (id(scope), target.id)
                            if prepared:
                                kinds[key] = "prepared"
                            elif kinds.get(key) != "prepared":
                                kinds[key] = "scene"
        return kinds

    def receiver_kind(self, method, receiver, scope_chain, kinds):
        if method in PREPARED_ONLY:
            return "prepared"
        text = self.src.seg(receiver)
        if PREPARED_TEXT.search(text):
            return "prepared"
        if isinstance(receiver, ast.Name):
            for scope in scope_chain:
                kind = kinds.get((id(scope), receiver.id))
                if kind is not None:
                    return kind
        return "scene"

    def run(self):
        aliases = _module_aliases(self.tree)
        kinds = self.kinds()
        parents = {}
        for scope in _scopes(self.tree):
            for node in _walk_scope(scope):
                parents.setdefault(id(node), scope)
        scope_parent = {}
        for scope in _scopes(self.tree):
            for node in _walk_scope(scope):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    scope_parent[id(node)] = scope
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in METHODS]
        # Rewrite innermost calls last so outer spans are computed on the original text.
        for call in sorted(calls, key=lambda n: self.src.span(n)[0]):
            receiver = call.func.value
            if isinstance(receiver, ast.Name) and receiver.id in aliases:
                continue
            if self.src.seg(receiver).startswith(("witwin.", "mw.compiler")):
                continue
            chain, scope = [], parents.get(id(call), self.tree)
            while scope is not None:
                chain.append(scope)
                scope = scope_parent.get(id(scope))
            kind = self.receiver_kind(call.func.attr, receiver, chain, kinds)
            if kind == "param":
                self.review.append((call.lineno, f"{call.func.attr} on parameter {self.src.seg(receiver)!r}: "
                                                 "classified as Scene, check the callers"))
                kind = "scene"
            self.rewrite(call, kind)
        return self.result()

    def rewrite(self, call, kind):
        """Replace ``X.compile_m(`` by ``compile_m(<first>, `` and edit keywords in place, keeping the layout."""
        src = self.src
        method = call.func.attr
        receiver = src.seg(call.func.value)
        call_start, call_end = src.span(call)
        open_paren = src.text.index("(", src.span(call.func)[1])
        inner = src.text[open_paren + 1:call_end - 1]
        base = open_paren + 1
        # Keyword edits as (start, end, text) relative to ``inner``.
        edits = []
        device = None
        for kw in call.keywords:
            if kw.arg is None:
                continue
            kw_start = src.index(kw.lineno, kw.col_offset) - base
            if kind == "scene" and kw.arg == "device" and method not in SCENE_ARGUMENT:
                device = src.seg(kw.value)
                kw_end = src.index(kw.end_lineno, kw.end_col_offset) - base
                before = inner[:kw_start]
                comma = before.rstrip()
                if comma.endswith(","):
                    kw_start = len(comma) - 1
                else:
                    after = inner[kw_end:]
                    stripped = after.lstrip()
                    if stripped.startswith(","):
                        kw_end += len(after) - len(stripped) + 1
                        kw_end += len(inner[kw_end:]) - len(inner[kw_end:].lstrip(" "))
                edits.append((kw_start, kw_end, ""))
            elif method in PREPARED_ONLY and kw.arg in RENAMED:
                edits.append((kw_start, kw_start + len(kw.arg), RENAMED[kw.arg]))
        for start, end, text in sorted(edits, key=lambda e: -e[0]):
            inner = inner[:start] + text + inner[end:]
        if kind == "prepared" and method == "compile_array_monitors":
            first = f"{receiver}.scene"
            if not any(kw.arg == "device" for kw in call.keywords):
                self.review.append((call.lineno, "compile_array_monitors on a prepared scene: add device=X.device"))
            self.counts["prepared_array"] += 1
        elif kind == "prepared" or method in SCENE_ARGUMENT:
            first = receiver
            self.counts[kind] += 1
        else:
            first = f"prepare_scene({receiver}{'' if device is None else f', device={device}'})"
            self.needs_prepare = True
            self.counts["scene_prepare"] += 1
        if not inner.strip():
            text = f"{method}({first})"
        elif inner[:1] == "\n" or inner.lstrip(" ")[:1] == "\n":
            text = f"{method}({first},{inner})"
        else:
            text = f"{method}({first}, {inner})"
        self.edits.append((call_start, call_end, text))
        self.needed.add(method)

    def result(self):
        text = self.src.text
        # Nested sites (a compile call inside another's arguments) apply innermost first.
        applied = []
        for start, end, new in sorted(self.edits, key=lambda e: (e[0], -e[1])):
            if any(s <= start and end <= e for s, e, _ in applied):
                self.review.append((text[:start].count("\n") + 1, "nested compile call: rerun apply"))
                continue
            applied.append((start, end, new))
        for start, end, new in sorted(applied, key=lambda e: -e[0]):
            text = text[:start] + new + text[end:]
        if applied and not self.package:
            text = self.add_imports(text)
        return text

    def add_imports(self, text):
        tree = ast.parse(text)
        have = {(node.module, alias.asname or alias.name) for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) for alias in node.names}
        bound = {name for _, name in have}
        lines = []
        by_module = {}
        for name in sorted(self.needed):
            if name not in bound:
                by_module.setdefault(MODULE[name], []).append(name)
        for module, names in sorted(by_module.items()):
            lines.append(f"from {module} import {', '.join(names)}\n")
        if self.needs_prepare and "prepare_scene" not in bound:
            lines.append("from witwin.maxwell.scene import prepare_scene\n")
        if not lines:
            return text
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        source = Source(text)
        at = source.starts[imports[-1].end_lineno] if imports else 0
        self.counts["import_lines"] += len(lines)
        return text[:at] + "".join(lines) + text[at:]


def _rewrite(text, rel, package):
    rewriter = Rewriter(text, rel, package)
    out = rewriter.run()
    if out != text:
        ast.parse(out)
    return out, rewriter.counts, rewriter.review


def process(path: Path, *, write, package=False):
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
                out, c, r = _rewrite(dedented, rel, package)
            except SyntaxError:
                continue
            counts.update(c)
            review.extend(r)
            out = "\n".join(indent + line if line else line for line in out.split("\n"))
            new = new[:match.start("body")] + out + new[match.end("body"):]
    else:
        new, counts, review = _rewrite(text, rel, package)
    if write and new != text:
        path.write_text(new)
    return counts, review, new != text


def files(paths):
    pinned = _pinned_hashes()
    for name in _tracked(paths):
        if EXCLUDE.match(name):
            continue
        path = ROOT / name
        if _sha256(path) in pinned:
            yield path, "skip"
        else:
            yield path, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("classify", "apply", "check"))
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--package", action="store_true", help="package modules: no import insertion")
    args = parser.parse_args(argv)
    total, changed, reviews, skipped, broken = Counter(), [], [], [], []
    for path, status in files(args.paths):
        rel = path.relative_to(ROOT)
        if status == "skip":
            skipped.append(rel)
            continue
        try:
            counts, review, differs = process(path, write=args.command == "apply",
                                              package=args.package or str(rel).startswith("witwin/"))
        except UnicodeDecodeError:
            continue
        except SyntaxError as error:
            broken.append(f"{rel}: {error}")
            continue
        total.update(counts)
        reviews.extend(f"{rel}:{line}: {message}" for line, message in review)
        if differs:
            changed.append(rel)
    for rel in skipped:
        print(f"skip (sha256 pinned): {rel}")
    for line in broken:
        print(f"unparsable: {line}")
    if args.command == "check":
        for rel in changed:
            print(f"compile method call: {rel}")
        print(f"check: {len(changed)} files with compile method calls")
        return 1 if changed else 0
    for decision, count in sorted(total.items()):
        print(f"{decision:24s} {count}")
    print(f"files {'rewritten' if args.command == 'apply' else 'to rewrite'}: {len(changed)}")
    for line in reviews:
        print(f"review {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

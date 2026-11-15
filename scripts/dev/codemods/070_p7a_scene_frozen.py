"""Rewrite callers of the immutable 0.7.0 ``Scene`` (phase 7a).

``Scene`` is a frozen dataclass with tuple collections: the ``add_*`` methods and ``clone`` are gone,
``scene.updated(**fields)`` returns a new scene, ``device``/``verbose``/``lazy_meshgrid`` left the
constructor (the solver config owns placement) and ``subpixel_samples=`` is ``subpixel=``. The
script rewrites call sites mechanically (``ast`` positions, layout kept where possible):

* ``fold``: ``X = Scene(...)`` followed by ``X.add_*(a)`` / ``X.<field>.append(a)`` / ``.extend(it)``
  statements on the same ``X`` folds the additions into the constructor keywords, in order. Plain
  statements between them that neither read ``X`` nor rebind a name the constructor reads stay before
  the constructor, which moves down to the last folded addition;
* ``chain``: ``Scene(...).add_*(a)...`` expressions fold into the constructor;
* ``updated``: every other addition becomes ``X = X.updated(field=(*X.field, a))`` (consecutive additions
  on one ``X`` share one call; a loop whose only statement adds its target becomes one call with
  ``*iterable``); used as an expression, ``X.add_*(a)`` becomes ``X.updated(...)``;
* ``ctor``: ``Scene(device=, verbose=, lazy_meshgrid=)`` keywords are dropped and ``subpixel_samples=``
  becomes ``subpixel=``; ``import_pcb_scene(..., device=)`` and ``scene_from_gds(..., device=)`` drop
  ``device=``;
* ``clone``: ``<scene>.clone(**fields)`` becomes ``<scene>.updated(**fields)`` (without the removed
  keywords) and ``<scene>.clone()`` the scene itself, for receivers named like a scene.

Sites it cannot decide are reported as ``review`` (an addition to a function parameter, which the caller
no longer sees; an addition to a free or global name; a dropped non-CUDA ``device=``; a field
assignment ``X.<field> = v``; reads of removed members) and hand-edited.

Commands (run with the maxwell environment)::

    classify [PATH ...]   per-decision counts, then the review list
    apply [PATH ...]      rewrite in place (idempotent)
    check [PATH ...]      exit 1 while a rewritable site remains

Without paths the scope is every tracked ``.py`` file under ``tests``, ``benchmark``, ``scripts`` and
``examples`` and the Python fences of ``docs/reference``, ``docs/memory``, ``docs/README.md``,
``README.md`` and ``FEATURE_LIST.md``. Excluded: ``tests/mom/`` and ``benchmark/mom_*`` (MoM-owned; an
explicit path migrates them), historical records, the codemods, the BITID probes and
``tests/support/api_dialect.py``, the consumer-metadata bridge and any file whose sha256 a tracked
benchmark, script or test file pins (reported as ``skip``).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCOPE = ("tests", "benchmark", "scripts", "examples", "experiments", "docs/reference", "docs/memory", "docs/README.md",
         "README.md", "FEATURE_LIST.md")
EXCLUDE = re.compile(
    r"^(tests/mom/|benchmark/mom_|docs/(assessments|archive|plans|releases)/|docs/reference/release-notes-"
    r"|CHANGELOG\.md$|scripts/dev/codemods/|tests/support/api_dialect\.py$"
    r"|tests/fdtd/_bit_identity_probe\.py$|tests/fdtd/test_raw_phase_bit_identity\.py$"
    r"|tests/boundaries/absorbers/_boundary_state_probe\.py$|tests/monitors/in_simulation/_bit_identity_runner\.py$"
    r"|tests/materials/surface_impedance/smooth_surface_probe\.py$|tests/support/rf_port_bit_identity_scene\.py$"
    r"|tests/support/rf_port_bit_identity_terminal_wave_scene\.py$|tests/support/port_step_graph_bit_identity_scene\.py$"
    r"|scripts/dev/fdtd/consumer_metadata/|benchmark/sie_single_buffer_equivalence_v3\.py$)")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)

ADD = {
    "add_structure": "structures", "add_thin_wire": "thin_wires", "add_source": "sources",
    "add_monitor": "monitors", "add_port": "ports", "add_network": "networks",
    "add_lumped_element": "lumped_elements", "add_circuit": "circuits",
    "add_material_region": "material_regions", "add_internal_absorber": "internal_absorbers",
    "add_electrostatic_terminal": "electrostatic_terminals", "add_charge_density": "charge_densities",
}
FIELDS = set(ADD.values())
SCENE_FIELDS = FIELDS | {"domain", "grid", "boundary", "background", "subpixel", "symmetry", "coordinates",
                         "metadata"}
DROPPED = ("device", "verbose", "lazy_meshgrid")
RENAMED = {"subpixel_samples": "subpixel"}
DEVICE_ONLY_CALLS = ("import_pcb_scene", "scene_from_gds")
SCENE_NAME = re.compile(r"(^|[._])\w*scene\w*$", re.IGNORECASE)
DEVICE_COMPILES = {"compile_ports", "compile_thin_wires", "compile_wire_monitors", "compile_waveports",
                   "compile_lumped_elements", "compile_networks", "compile_electrostatics"}
FACTORY_CONFIGS = {"fdtd": "FDTDConfig", "fdfd": "FDFDConfig", "mom": "mom.MoMConfig",
                   "electrostatic": "electrostatic.ElectrostaticConfig",
                   "capacitance": "electrostatic.CapacitanceConfig"}
REMOVED_READS = ("device", "verbose", "lazy_meshgrid", "subpixel_samples", "bc_type", "boundary_type",
                 "domain_size", "physical_positions")
WIDTH = 110


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


# --- source positions ---------------------------------------------------------------------
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

    def indent(self, lineno):
        line = self.lines[lineno - 1]
        return line[:len(line) - len(line.lstrip())]

    def after(self, node, limit):
        """End of ``node`` including the parentheses that group it, before the ``)`` at ``limit``."""
        end = self.index(node.end_lineno, node.end_col_offset)
        m = re.match(r"(\s*\))*", self.text[end:limit])
        closes = [i for i, ch in enumerate(m.group()) if ch == ")"]
        return end + (closes[-1] + 1 if closes else 0)

    def column(self, node):
        start = self.index(node.lineno, node.col_offset)
        return start - self.starts[node.lineno - 1]


@dataclass
class Edit:
    start: int
    end: int
    text: str


def _reindent(text, old_column, new_column):
    """Shift the continuation lines of an expression moved from ``old_column`` to ``new_column``."""
    lines = text.split("\n")
    if len(lines) == 1:
        return text
    shift = new_column - old_column
    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        # Keep relative indentation; a line indented less than the expression start stays at
        # least one level inside the new position.
        indent = len(line) - len(line.lstrip(" "))
        if indent < old_column:
            # A line outdented past the expression start: closing brackets align with the
            # expression, other lines sit one level inside it.
            floor = new_column if line.lstrip(" ")[:1] in ")]}" else new_column + 4
        else:
            floor = new_column
        out.append(" " * max(indent + shift, floor) + line.lstrip(" "))
    return "\n".join(out)


def _is_scene_ctor(node):
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "Scene") or (
        isinstance(func, ast.Attribute) and func.attr == "Scene")


def _target_key(node):
    """A dotted name for Name/Attribute chains (``scene``, ``self.scene``), else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _target_key(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _addition(node, scene_key=None):
    """(receiver, field, element texts are ast nodes, starred) for one addition call, else None.

    ``X.add_*(a)`` and ``X.<field>.append(a)`` add one element; ``X.<field>.extend(it)`` adds ``*it``.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    func = node.func
    if func.attr in ADD and len(node.args) == 1 and not node.keywords and not isinstance(node.args[0], ast.Starred):
        return func.value, ADD[func.attr], node.args[0], False
    if (func.attr in ("append", "extend") and isinstance(func.value, ast.Attribute)
            and func.value.attr in FIELDS and len(node.args) == 1 and not node.keywords
            and _target_key(func.value.value) is not None
            and (_target_key(func.value.value) == scene_key or SCENE_NAME.search(_target_key(func.value.value)))):
        return func.value.value, func.value.attr, node.args[0], func.attr == "extend"
    return None


def _names(node, ctx=ast.Load):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _stored(stmt):
    """Names a statement binds or mutates through an attribute/subscript store."""
    names = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names |= _names(node.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _walk_scope(scope):
    """``ast.walk`` of one function or module body, without entering nested functions or classes."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            todo.extend(ast.iter_child_nodes(node))


def _references(node, key):
    root = key.split(".")[0]
    return root in _names(node) or root in _names(node, ast.Store)


# --- formatting ---------------------------------------------------------------------------
_ATOMS = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.List, ast.Tuple, ast.ListComp,
          ast.GeneratorExp, ast.Set, ast.Dict)


def _star(node, text):
    """``*text`` with parentheses when ``node`` is not an atom (``*(a or b)``)."""
    return f"*{text}" if isinstance(node, _ATOMS) else f"*({text})"


def _elements(src, items, column):
    """Element texts of an addition list [(node, starred)], reindented to ``column``."""
    out = []
    for node, starred in items:
        text = _reindent(src.seg(node), src.column(node), column + (1 if starred else 0))
        out.append(_star(node, text) if starred else text)
    return out


def _tuple_text(prefix, elements, column, start=0):
    """``(prefix, e1, e2)`` inline when it fits from ``start``, else one element per line at
    ``column + 4`` (the elements are already indented for that layout) and ``)`` at ``column``."""
    parts = ([prefix] if prefix else []) + elements
    inline = "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
    if "\n" not in inline and max(start, column) + len(inline) <= WIDTH:
        return inline
    inner = " " * (column + 4)
    return "(\n" + "".join(f"{inner}{part},\n" for part in parts) + " " * column + ")"


# --- the rewriter -------------------------------------------------------------------------
class Rewriter:
    def __init__(self, text, rel):
        self.src = Source(text)
        self.rel = rel
        self.tree = ast.parse(text)
        self.edits: list[Edit] = []
        self.counts = Counter()
        self.review = []
        self.consumed: set[int] = set()
        self.device_calls: set[int] = set()
        self.needs_place = False

    # Edits ---------------------------------------------------------------------------------
    def add_edit(self, start, end, text):
        self.edits.append(Edit(start, end, text))

    def result(self):
        if self.needs_place:
            self.import_place()
        text = self.src.text
        for edit in sorted(self.edits, key=lambda e: (e.start, e.end), reverse=True):
            text = text[:edit.start] + edit.text + text[edit.end:]
        return text

    # Constructor surgery -------------------------------------------------------------------
    def ctor_text(self, call, additions, column):
        """The constructor source with removed keywords dropped, renames applied and ``additions``
        ({field: [(node, starred)]}) appended to its collection keywords."""
        src = self.src
        start, end = src.span(call)
        local = []  # (start, end, text) relative edits within the call text
        keywords = {kw.arg: kw for kw in call.keywords if kw.arg is not None}
        for kw in call.keywords:
            if kw.arg in DROPPED:
                kw_start = src.index(kw.lineno, kw.col_offset)
                kw_end = src.index(kw.end_lineno, kw.end_col_offset)
                # Remove the keyword with its separating comma.
                after = src.text[kw_end:end]
                m = re.match(r"\s*,[ \t]*\n?[ \t]*", after)
                if m and not after[m.end():].lstrip().startswith(")"):
                    kw_end += m.end()
                else:
                    before = src.text[start:kw_start]
                    m = re.search(r",\s*$", before)
                    if m:
                        kw_start = start + m.start()
                    elif re.match(r"\s*,", after):
                        kw_end += re.match(r"\s*,", after).end()
                local.append((kw_start, kw_end, ""))
                self.counts[f"ctor_drop_{kw.arg}"] += 1
                if kw.arg == "device" and not (isinstance(kw.value, ast.Constant) and kw.value.value in ("cuda", None)):
                    self.review.append((kw.lineno, f"Scene(device={src.seg(kw.value)}) dropped: set config/prepare device"))
            elif kw.arg in RENAMED:
                kw_start = src.index(kw.lineno, kw.col_offset)
                local.append((kw_start, kw_start + len(kw.arg), RENAMED[kw.arg]))
                self.counts["ctor_rename_subpixel"] += 1
        call_column = src.column(call)
        multiline = call.lineno != call.end_lineno
        appended = []
        for field, items in additions.items():
            if field in keywords:
                value = keywords[field].value
                v_start, v_end = src.span(value)
                if isinstance(value, (ast.List, ast.Tuple)):
                    close = v_end - 1
                    if not value.elts:
                        new = _elements(src, items, src.column(value) + 1)
                        one = isinstance(value, ast.Tuple) and len(new) == 1
                        local.append((v_start + 1, close, ", ".join(new) + ("," if one else "")))
                        continue
                    last_end = src.index(value.elts[-1].end_lineno, value.elts[-1].end_col_offset)
                    tail = src.text[last_end:close]
                    comma = "," in tail
                    insert_at = last_end + (tail.index(",") + 1 if comma else 0)
                    if value.lineno != value.end_lineno:
                        elem_column = src.column(value.elts[-1])
                        new = _elements(src, items, elem_column)
                        text = ("" if comma else ",") + ",".join(f"\n{' ' * elem_column}{e}" for e in new)
                        local.append((insert_at, insert_at, text + ("," if comma else "")))
                    else:
                        new = _elements(src, items, src.column(value) + 1)
                        local.append((insert_at, insert_at, (" " if comma else ", ") + ", ".join(new)))
                else:
                    base = len(src.indent(value.lineno))
                    new = _elements(src, items, base + 4)
                    old = src.seg(value)
                    star = old if isinstance(value, ast.Starred) else "*" + (
                        old if isinstance(value, (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.List,
                                                  ast.Tuple)) else f"({old})")
                    local.append((v_start, v_end, _tuple_text(star, new, base, src.column(value))))
            else:
                appended.append((field, items))
        if appended:
            # Insert before the closing parenthesis.
            close = end - 1
            kept = [kw for kw in call.keywords if kw.arg not in DROPPED]
            last = max([*call.args, *[kw.value for kw in kept]], key=lambda n: (n.end_lineno, n.end_col_offset),
                       default=None)
            # Insert right after the last kept argument: a dropped keyword's removal starts there.
            if multiline and last is not None:
                kw_column = src.column(kept[0]) if kept else src.column(call.args[0])
                insert_at = src.after(last, end - 1)
                text = ""
                for field, items in appended:
                    elements = _elements(src, items, kw_column + 4)
                    text += (f",\n{' ' * kw_column}{field}="
                             f"{_tuple_text('', elements, kw_column, kw_column + len(field) + 1)}")
                local.append((insert_at, insert_at, text))
            else:
                pieces = []
                base = len(src.indent(call.lineno))
                for field, items in appended:
                    elements = _elements(src, items, base + 4)
                    pieces.append(f"{field}={_tuple_text('', elements, base, column + len(src.seg(call)))}")
                if last is not None:
                    insert_at, lead = src.after(last, end - 1), ", "
                else:
                    insert_at, lead = end - 1, ""
                local.append((insert_at, insert_at, lead + ", ".join(pieces)))
        text = src.text[start:end]
        for l_start, l_end, l_text in sorted(local, key=lambda e: (e[0], e[1]), reverse=True):
            text = text[:l_start - start] + l_text + text[l_end - start:]
        return text

    # Passes ------------------------------------------------------------------------------
    def run(self):
        self.visit_devices()
        self.visit_blocks(self.tree, scope=None)
        self.visit_expressions()
        self.visit_misc()
        self.visit_rebinding_hazards()
        return self.result()

    def visit_rebinding_hazards(self):
        """List edits whose rebinding cannot reach the caller: loop targets and tuple-target writes."""
        def mutates(node, name):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr in ADD
                        and (_target_key(inner.func.value) == f"{name}.scene"
                             or _target_key(inner.func.value) == name and SCENE_NAME.search(name))):
                    return True
                if isinstance(inner, ast.Assign) and any(
                        isinstance(t, ast.Attribute) and t.attr in SCENE_FIELDS and _target_key(t.value) in (name, f"{name}.scene")
                        for target in inner.targets for t in ast.walk(target)):
                    return True
            return False
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name) and any(mutates(statement, target.id) for statement in node.body):
                        self.review.append((node.lineno, f"loop target {target.id!r} is rebuilt: write it back to the sequence"))
            elif isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Tuple) and any(
                        isinstance(item, ast.Attribute) and item.attr in SCENE_FIELDS and _target_key(item.value)
                        and SCENE_NAME.search(_target_key(item.value)) and not re.search(r"prepared|solver", _target_key(item.value), re.I)
                        for item in target.elts) for target in node.targets):
                self.review.append((node.lineno, "tuple-target scene field write: use one updated(...) call"))

    def visit_blocks(self, node, scope):
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                self.block(block, scope)
        for handler in getattr(node, "handlers", ()):
            self.block(handler.body, scope)
        for case in getattr(node, "cases", ()):
            self.block(case.body, scope)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                inner = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else scope
                self.visit_blocks(child, inner)

    def _own_lines(self, block, index):
        """True when statement ``index`` shares no line with a neighbour."""
        stmt = block[index]
        if index > 0 and block[index - 1].end_lineno >= stmt.lineno:
            return False
        if index + 1 < len(block) and block[index + 1].lineno <= stmt.end_lineno:
            return False
        line = self.src.lines[stmt.lineno - 1]
        return line[:len(line) - len(line.lstrip())] == line[:self.src.column(stmt)]

    def _stmt_addition(self, stmt, scene_key=None):
        if isinstance(stmt, ast.Expr):
            found = _addition(stmt.value, scene_key)
            if found is not None and _target_key(found[0]) is not None:
                return found
        return None

    def block(self, block, scope):
        i = 0
        while i < len(block):
            stmt = block[i]
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _target_key(stmt.targets[0])
                    and _is_scene_ctor(stmt.value) and self._own_lines(block, i)):
                i = self.fold(block, i, scope)
                continue
            found = self._stmt_addition(stmt)
            if found is not None and id(stmt) not in self.consumed:
                i = self.fallback_run(block, i, scope)
                continue
            if isinstance(stmt, ast.For) and not stmt.orelse and len(stmt.body) == 1:
                inner = self._stmt_addition(stmt.body[0])
                if inner is not None and not _references(stmt.iter, _target_key(inner[0])) and not inner[3]:
                    key = _target_key(inner[0])
                    if not _references(stmt.target, key):
                        self.loop_fold(stmt, inner, scope)
            i += 1

    def fold(self, block, i, scope):
        ctor_stmt = block[i]
        key = _target_key(ctor_stmt.targets[0])
        ctor = ctor_stmt.value
        ctor_loads = _names(ctor)
        additions: dict[str, list] = {}
        folded, moved = [], []
        seen_loads = set(ctor_loads)
        j = i + 1
        while j < len(block):
            stmt = block[j]
            found = self._stmt_addition(stmt, key)
            if found is not None and _target_key(found[0]) == key and not _references(found[2], key) \
                    and self._own_lines(block, j):
                receiver, field, element, starred = found
                additions.setdefault(field, []).append((element, starred))
                seen_loads |= _names(element)
                folded.append(stmt)
                j += 1
                continue
            simple = isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Import,
                                       ast.ImportFrom, ast.Pass))
            if (simple and not _references(stmt, key) and not (_stored(stmt) & seen_loads)
                    and self._own_lines(block, j)
                    and not (isinstance(stmt, ast.Expr) and _addition(stmt.value) is not None)):
                moved.append(stmt)
                j += 1
                continue
            break
        while moved and (not folded or moved[-1].lineno > folded[-1].lineno):
            moved.pop()
        if not folded:
            self.ctor_only(ctor_stmt)
            return i + 1
        last = folded[-1]
        src = self.src
        indent = src.indent(ctor_stmt.lineno)
        target_text = src.seg(ctor_stmt.targets[0])
        prefix_len = src.column(ctor) - src.column(ctor_stmt)
        new_ctor = self.ctor_text(ctor, additions, src.column(ctor))
        new_ctor = _reindent(new_ctor, src.column(ctor), len(indent) + prefix_len)
        statement = f"{src.text[src.index(ctor_stmt.lineno, ctor_stmt.col_offset):src.index(ctor.lineno, ctor.col_offset)]}{new_ctor}"
        del target_text
        # Delete the constructor and folded statements (whole lines); write the constructor in place
        # of the last folded statement, after any statements it moved past.
        for stmt in [ctor_stmt, *folded[:-1]]:
            self.delete_lines(stmt)
        l_start = src.starts[last.lineno - 1]
        l_end = src.starts[last.end_lineno]
        newline = "\n" if src.lines[last.end_lineno - 1].endswith("\n") else ""
        trailing = self.trailing_comment(last)
        self.add_edit(l_start, l_end, indent + statement + trailing + newline)
        for stmt in folded:
            self.consumed.add(id(stmt))
        self.consumed.add(id(ctor_stmt))
        self.consumed.add(id(ctor))
        self.counts["fold_additions"] += len(folded)
        self.counts["fold_scenes"] += 1
        if moved:
            self.counts["fold_moved_past"] += 1
        return block.index(last) + 1

    def trailing_comment(self, stmt):
        line = self.src.lines[stmt.end_lineno - 1]
        rest = line[len(line.encode()[:stmt.end_col_offset].decode(errors="ignore")):].rstrip("\n")
        return rest if rest.strip().startswith("#") else ""

    def delete_lines(self, stmt):
        src = self.src
        self.add_edit(src.starts[stmt.lineno - 1], src.starts[stmt.end_lineno], "")

    def ctor_only(self, ctor_stmt):
        ctor = ctor_stmt.value
        if any(kw.arg in DROPPED or kw.arg in RENAMED for kw in ctor.keywords):
            start, end = self.src.span(ctor)
            self.add_edit(start, end, self.ctor_text(ctor, {}, self.src.column(ctor)))
        self.consumed.add(id(ctor))

    def _param_or_free(self, key, scope, stmt):
        root = key.split(".")[0]
        if "." in key:
            return None
        if scope is None:
            return None
        args = scope.args
        params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        if args.vararg:
            params.add(args.vararg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)
        if root in params:
            return "parameter"
        assigned = set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
        if root not in assigned:
            return "free"
        return None

    def fallback_run(self, block, i, scope):
        stmt = block[i]
        key = _target_key(self._stmt_addition(stmt)[0])
        run = [stmt]
        j = i + 1
        while j < len(block):
            found = self._stmt_addition(block[j])
            if found is None or _target_key(found[0]) != key or _references(found[2], key) or not self._own_lines(block, j):
                break
            run.append(block[j])
            j += 1
        if not all(self._own_lines(block, block.index(s)) for s in run):
            run = [stmt]
            j = i + 1
        additions: dict[str, list] = {}
        for s in run:
            _, field, element, starred = self._stmt_addition(s)
            additions.setdefault(field, []).append((element, starred))
        receiver = self._stmt_addition(stmt)[0]
        receiver_text = self.src.seg(receiver)
        column = self.src.column(stmt)
        call = self.updated_text(receiver_text, additions, column, column + 2 * len(receiver_text) + 3)
        text = f"{receiver_text} = {call}"
        start = self.src.index(stmt.lineno, stmt.col_offset)
        end = self.src.index(run[-1].end_lineno, run[-1].end_col_offset)
        self.add_edit(start, end, text)
        for s in run:
            self.consumed.add(id(s))
        self.counts["updated_additions"] += len(run)
        self.counts["updated_statements"] += 1
        kind = self._param_or_free(key, scope, stmt)
        if kind is not None:
            self.review.append((stmt.lineno, f"addition to {kind} {key!r}: rebinding is not seen by the caller"))
        return j

    def updated_text(self, receiver_text, additions, column, start):
        """``receiver.updated(field=(*receiver.field, ...))``; multi-line tuples close at ``column``."""
        parts = []
        for field, items in additions.items():
            elements = _elements(self.src, items, column + 4)
            parts.append(f"{field}={_tuple_text(f'*{receiver_text}.{field}', elements, column, start + len(field))}")
        return f"{receiver_text}.updated({', '.join(parts)})"

    def loop_fold(self, loop, inner, scope):
        receiver, field, element, _ = inner
        key = _target_key(receiver)
        src = self.src
        receiver_text = src.seg(receiver)
        target = src.seg(loop.target)
        iterable = src.seg(loop.iter)
        column = src.column(loop)
        if isinstance(element, ast.Name) and isinstance(loop.target, ast.Name) and element.id == loop.target.id:
            spread = _star(loop.iter, iterable)
        else:
            body = _reindent(src.seg(element), src.column(element), column + 6)
            spread = f"*({body} for {target} in {iterable})"
        head = f"{receiver_text} = {receiver_text}.updated({field}="
        text = head + _tuple_text(f"*{receiver_text}.{field}", [spread], column, column + len(head)) + ")"
        start = src.index(loop.lineno, loop.col_offset)
        end = src.index(loop.end_lineno, loop.end_col_offset)
        self.add_edit(start, end, text)
        self.consumed.add(id(loop.body[0]))
        self.counts["loop_additions"] += 1
        kind = self._param_or_free(key, scope, loop)
        if kind is not None:
            self.review.append((loop.lineno, f"addition to {kind} {key!r}: rebinding is not seen by the caller"))

    def visit_expressions(self):
        """Additions used as values (chains on a constructor, ``return scene.add_x(a)``)."""
        consumed_spans = [(e.start, e.end) for e in self.edits]

        def inside(node):
            start, end = self.src.span(node)
            return any(s <= start and end <= e for s, e in consumed_spans)

        handled = set()
        parents = {}
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        for node in ast.walk(self.tree):
            if id(node) in handled or not isinstance(node, ast.Call):
                continue
            found = _addition(node)
            if found is None or inside(node):
                continue
            chain = []
            base = node
            while True:
                step = _addition(base)
                if step is None:
                    break
                chain.append(step)
                handled.add(id(base))
                base = step[0]
            chain.reverse()
            additions: dict[str, list] = {}
            for _, field, element, starred in chain:
                additions.setdefault(field, []).append((element, starred))
            start, end = self.src.span(node)
            if _is_scene_ctor(base):
                self.add_edit(start, end, self.ctor_text(base, additions, self.src.column(base)))
                self.consumed.add(id(base))
                self.counts["chain_additions"] += len(chain)
            elif _target_key(base) is not None:
                self.add_edit(start, end, self.updated_text(self.src.seg(base), additions,
                                                            len(self.src.indent(node.lineno)), self.src.column(node)))
                self.counts["expression_additions"] += len(chain)
                self.review.append((node.lineno, "addition used as a value: check the caller uses the new scene"))
            else:
                stmt = parents.get(id(node))
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                        and stmt.value is node):
                    # ``X = make().add_a(a)`` -> ``X = make()`` then ``X = X.updated(...)``.
                    name = stmt.targets[0].id
                    column = self.src.column(stmt)
                    indent = " " * column
                    text = (f"{name} = {self.src.seg(base)}\n{indent}{name} = "
                            f"{self.updated_text(name, additions, column, column + 2 * len(name) + 3)}")
                    s_start, s_end = self.src.span(stmt)
                    self.add_edit(s_start, s_end, text)
                    consumed_spans.append((s_start, s_end))
                    self.counts["computed_additions"] += len(chain)
                else:
                    self.review.append((node.lineno, "addition on a computed receiver: hand edit"))

    def prepared_names(self):
        """Names bound to ``prepare_scene(...)``/``PreparedScene(...)``/``SimpleNamespace(...)`` (mutable, left alone), per scope."""
        names = set()
        for scope in [self.tree, *(n for n in ast.walk(self.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]:
            for node in _walk_scope(scope):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and getattr(node.value.func, "id", getattr(node.value.func, "attr", None))
                        in ("prepare_scene", "PreparedScene", "SimpleNamespace")):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.add((id(scope), target.id))
        return names

    def visit_misc(self):
        consumed_spans = [(e.start, e.end) for e in self.edits]
        prepared = self.prepared_names()
        scope_of = {}
        for scope in [self.tree, *(n for n in ast.walk(self.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]:
            for node in _walk_scope(scope):
                scope_of[id(node)] = id(scope)

        def inside(node):
            start, end = self.src.span(node)
            return any(s <= start and end <= e for s, e in consumed_spans)

        # A field write under ``pytest.raises(...)`` asserts immutability; it stays a write.
        expected_failures = {id(inner) for node in ast.walk(self.tree) if isinstance(node, ast.With)
                             and any("raises" in ast.unparse(item.context_expr) for item in node.items)
                             for statement in node.body for inner in ast.walk(statement)}

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and _is_scene_ctor(node) and id(node) not in self.consumed and not inside(node):
                if any(kw.arg in DROPPED or kw.arg in RENAMED for kw in node.keywords):
                    start, end = self.src.span(node)
                    self.add_edit(start, end, self.ctor_text(node, {}, self.src.column(node)))
                    consumed_spans.append((start, end))
            elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                if name in DEVICE_ONLY_CALLS and not inside(node):
                    for kw in node.keywords:
                        if kw.arg == "device":
                            self.drop_keyword(node, kw)
                            self.counts[f"{name}_device"] += 1
                elif (name == "clone" and isinstance(node.func, ast.Attribute) and not inside(node) and not node.args
                      and (node.keywords and all(kw.arg in SCENE_FIELDS | set(DROPPED) | set(RENAMED)
                                                 for kw in node.keywords)
                           or not node.keywords and _target_key(node.func.value)
                           and SCENE_NAME.search(_target_key(node.func.value))
                           and not re.search(r"prepared|tensor|field|grad|density", _target_key(node.func.value), re.I))):
                    self.rewrite_clone(node)
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and not inside(node)
                  and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr in SCENE_FIELDS
                  and isinstance(node.targets[0].value, ast.Name)
                  and SCENE_NAME.search(node.targets[0].value.id)
                  and not re.search(r"prepared|solver", node.targets[0].value.id, re.I)
                  and (scope_of.get(id(node)), node.targets[0].value.id) not in prepared
                  and id(node) not in expected_failures):
                # ``X.field = v`` on a local scene -> ``X = X.updated(field=v)``.
                name, field = node.targets[0].value.id, node.targets[0].attr
                start, end = self.src.span(node)
                self.add_edit(start, end, f"{name} = {name}.updated({field}={self.src.seg(node.value)})")
                self.counts["field_assignment"] += 1
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and not inside(node)
                  and isinstance(node.targets[0], ast.Subscript) and not isinstance(node.targets[0].slice, ast.Slice)
                  and isinstance(node.targets[0].value, ast.Attribute) and node.targets[0].value.attr in FIELDS
                  and isinstance(node.targets[0].value.value, ast.Name)
                  and SCENE_NAME.search(node.targets[0].value.value.id)
                  and not re.search(r"prepared|solver", node.targets[0].value.value.id, re.I)):
                # ``X.field[i] = v`` -> ``X = X.updated(field=(*X.field[:i], v, *X.field[i + 1:]))``.
                target = node.targets[0]
                name, field, index = target.value.value.id, target.value.attr, self.src.seg(target.slice)
                after = f"{int(index) + 1}" if re.fullmatch(r"\d+", index) else f"{index} + 1"
                start, end = self.src.span(node)
                head = "" if index == "0" else f"*{name}.{field}[:{index}], "
                self.add_edit(start, end, f"{name} = {name}.updated({field}=({head}"
                                          f"{self.src.seg(node.value)}, *{name}.{field}[{after}:]))")
                self.counts["item_assignment"] += 1
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], (ast.Attribute, ast.Subscript))):
                target = node.targets[0]
                attr = target if isinstance(target, ast.Attribute) else target.value
                if (isinstance(attr, ast.Attribute) and attr.attr in SCENE_FIELDS | set(DROPPED)
                        and _target_key(attr.value) and SCENE_NAME.search(_target_key(attr.value))
                        and not re.search(r"prepared|solver", _target_key(attr.value), re.I)):
                    self.review.append((node.lineno, f"assignment to {self.src.seg(target)}: use updated()"))
            elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in REMOVED_READS
                  and _target_key(node.value) and SCENE_NAME.search(_target_key(node.value))
                  and not re.search(r"prepared|solver|result", _target_key(node.value), re.I)):
                self.review.append((node.lineno, f"read of removed Scene member {self.src.seg(node)}"))

    def visit_devices(self):
        """Carry a dropped non-CUDA ``Scene(device=D)`` to the preparation calls on that scene in the
        same function: ``prepare_scene(X)`` and ``X.compile_*()`` gain ``device=D``."""
        scopes = [self.tree] + [n for n in ast.walk(self.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for scope in scopes:
            devices = {}
            for node in _walk_scope(scope):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and _is_scene_ctor(node.value):
                    key = _target_key(node.targets[0])
                    device = next((kw.value for kw in node.value.keywords if kw.arg == "device"), None)
                    if key and device is not None and not (isinstance(device, ast.Constant)
                                                          and device.value in ("cuda", None)):
                        devices[key] = self.src.seg(device)
            for node in _walk_scope(scope):
                if not isinstance(node, ast.Call) or any(kw.arg == "device" for kw in node.keywords):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                first = node.args[0] if node.args else None
                cloned = (isinstance(first, ast.Call) and isinstance(first.func, ast.Attribute)
                          and first.func.attr == "clone"
                          and next((kw.value for kw in first.keywords if kw.arg == "device"), None))
                if name in ("prepare_scene", "PreparedScene") and cloned:
                    device = self.src.seg(cloned)
                elif name in ("prepare_scene", "PreparedScene") and node.args and _target_key(node.args[0]) in devices:
                    device = devices[_target_key(node.args[0])]
                elif name in DEVICE_COMPILES and isinstance(func, ast.Attribute) and _target_key(func.value) in devices:
                    device = devices[_target_key(func.value)]
                elif (name in FACTORY_CONFIGS and isinstance(func, ast.Attribute)
                      and (_target_key(func.value) or "").endswith("Simulation")
                      and node.args and _target_key(node.args[0]) in devices):
                    self.simulation_device(node, devices[_target_key(node.args[0])])
                    continue
                else:
                    continue
                if id(node) in self.device_calls:
                    continue
                self.device_calls.add(id(node))
                close = self.src.span(node)[1] - 1
                last = max([*node.args, *[kw.value for kw in node.keywords]],
                           key=lambda n: (n.end_lineno, n.end_col_offset), default=None)
                at = close if last is None else self.src.after(last, close)
                self.add_edit(at, at, ("" if last is None else ", ") + f"device={device}")
                self.counts["device_propagated"] += 1

    def simulation_device(self, call, device):
        """``Simulation.<method>(X, ...)`` for a scene built with ``device=D``: the config gets ``device=D``."""
        if id(call) in self.device_calls:
            return
        self.device_calls.add(id(call))
        config = next((kw.value for kw in call.keywords if kw.arg == "config"), None)
        if config is None:
            prefix = _target_key(call.func.value).rpartition(".")[0]
            if not prefix:
                self.review.append((call.lineno, f"Simulation without config: add device={device}"))
                return
            close = self.src.span(call)[1] - 1
            last = max([*call.args, *[kw.value for kw in call.keywords]], key=lambda n: (n.end_lineno, n.end_col_offset))
            at = self.src.after(last, close)
            self.add_edit(at, at, f", config={prefix}.{FACTORY_CONFIGS[call.func.attr]}(device={device})")
        elif (isinstance(config, ast.Call) and getattr(config.func, "attr", getattr(config.func, "id", "")).endswith("Config")
              and not any(kw.arg == "device" for kw in config.keywords)):
            close = self.src.span(config)[1] - 1
            last = max([*config.args, *[kw.value for kw in config.keywords]],
                       key=lambda n: (n.end_lineno, n.end_col_offset), default=None)
            at = close if last is None else self.src.after(last, close)
            self.add_edit(at, at, ("" if last is None else ", ") + f"device={device}")
        else:
            self.review.append((call.lineno, f"Simulation config is not a literal: add device={device}"))
            return
        self.counts["device_to_config"] += 1

    def drop_keyword(self, call, kw):
        src = self.src
        start = src.index(kw.lineno, kw.col_offset)
        end = src.index(kw.end_lineno, kw.end_col_offset)
        before = src.text[src.span(call)[0]:start]
        m = re.search(r",\s*$", before)
        if m:
            start = src.span(call)[0] + m.start()
        else:
            after = src.text[end:src.span(call)[1]]
            m2 = re.match(r"\s*,\s*", after)
            if m2:
                end += m2.end()
        self.add_edit(start, end, "")

    def rewrite_clone(self, node):
        """``X.clone(**fields)`` -> ``X.updated(**fields)``; ``clone(device=D)`` placed tensors on D,
        which ``place_scene(X, D)`` keeps."""
        src = self.src
        receiver = src.seg(node.func.value)
        kept = [kw for kw in node.keywords if kw.arg not in DROPPED]
        device = next((kw.value for kw in node.keywords if kw.arg == "device"), None)
        start, end = src.span(node)
        text = receiver if not kept else (
            f"{receiver}.updated({', '.join(f'{RENAMED.get(kw.arg, kw.arg)}={src.seg(kw.value)}' for kw in kept)})")
        if device is not None:
            text = f"place_scene({text}, {src.seg(device)})"
            self.needs_place = True
            self.counts["clone_place"] += 1
        self.add_edit(start, end, text)
        self.counts["clone"] += 1

    def import_place(self):
        """Add ``from witwin.maxwell.scene import place_scene`` after the module's last top-level import."""
        if any(isinstance(n, ast.ImportFrom) and n.module == "witwin.maxwell.scene"
               and any(a.name == "place_scene" for a in n.names) for n in self.tree.body):
            return
        imports = [n for n in self.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        at = self.src.starts[imports[-1].end_lineno] if imports else 0
        self.add_edit(at, at, "from witwin.maxwell.scene import place_scene\n")


def _rewrite(text, rel):
    rewriter = Rewriter(text, rel)
    out = rewriter.run()
    if out != text:
        ast.parse(out)
    return out, rewriter.counts, rewriter.review


def process(path: Path, *, write):
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
                out, c, r = _rewrite(dedented, rel)
            except SyntaxError:
                continue
            counts.update(c)
            review.extend(r)
            out = "\n".join(indent + line if line else line for line in out.split("\n"))
            new = new[:match.start("body")] + out + new[match.end("body"):]
    else:
        new, counts, review = _rewrite(text, rel)
    if write and new != text:
        path.write_text(new)
    return counts, review, new != text


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
    parser.add_argument("command", choices=("classify", "apply", "check"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    total, changed, reviews, skipped, broken = Counter(), [], [], [], []
    for path, status in files(args.paths):
        rel = path.relative_to(ROOT)
        if status == "skip":
            skipped.append(rel)
            continue
        try:
            counts, review, differs = process(path, write=args.command == "apply")
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
            print(f"old-form Scene use: {rel}")
        print(f"check: {len(changed)} files with old-form Scene uses")
        return 1 if changed else 0
    for decision, count in sorted(total.items()):
        print(f"{decision:24s} {count}")
    print(f"files {'rewritten' if args.command == 'apply' else 'to rewrite'}: {len(changed)}")
    for line in reviews:
        print(f"review {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

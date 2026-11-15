"""Rewrite FDTD monitor reads from the 0.6 payload dicts to the 0.7.0 typed results.

0.7.0 phase 6b. ``Result.monitor(name)`` returns typed data (``FieldData``, ``FluxData``,
``ModeAmplitudeData``, ...) instead of the solver's payload dict, frequency-domain values keep a
leading ``f`` dimension (length 1 for a single-frequency monitor) unless a frequency is selected,
and the payloads (``Result.raw_monitor``) lose their single-component ``"data"``/``"field"``
aliases. The script rewrites reads mechanically (``ast`` positions, layout kept):

* ``r.monitor("p")["Ez"]`` / ``["data"]`` / ``["components"]["Ez"]`` -> ``r.monitor("p").sel(component="Ez").values``;
  ``["flux"]`` -> ``.values``; ``["t"]`` -> ``.coords["t"]``; mode and Gaussian-overlap keys
  (``amplitude_forward``, ``power_fraction_forward``, ``effective_index``, ...) -> ``.isel(mode=0).sel(propagation="+")``
  values or ``attrs``; the same for a variable assigned from ``r.monitor(...)`` and read only by subscripts
  in the same function, and for ``r.monitors["p"][...]``;
* a single-frequency monitor read without a frequency selection gets ``freq_index=0`` so values keep
  their 0.6 shape (no ``f`` axis);
* FDTD ``r.fields["EX"]`` -> ``r.fields["EX"].values`` (``.isel(f=0).values`` for one frequency), or
  ``r.time_snapshot["EX"].values`` where the run kept its last-step fields (single frequency without
  ``full_field_dft``);
* reads the typed data cannot express (payload metadata such as ``cell_widths``, ``faces``,
  ``native_flux_heads``, native components of an aligned plane, or a payload passed on whole) move to
  ``r.raw_monitor(...)``, which returns the same payload ``monitor`` returned in 0.6 for those monitor
  kinds, with ``"data"``/``"field"`` replaced by the component key.

Which monitor a call reads, its frequency count and components come from ``070_p6b_monitor_sites.csv``,
recorded by running the test suite on the 0.6 layout (``sites`` command, from the trace of
``<evidence-root>/api070-20261108/p6b/plug/p6btrace.py``). Rows are keyed by file, enclosing function
and the call's source text, so they survive line shifts. A call without a row (untraced code: slow
tests, scripts, docs) uses the monitor declared under that literal name in the same file and is
rewritten without ``freq_index``. Calls of the mode-like kinds that need a hand edit are reported.

Commands (run with the maxwell environment)::

    sites TRACE_DIR       write 070_p6b_monitor_sites.csv from trace JSONL files
    classify [PATH ...]   per-decision counts, then the review list
    apply [PATH ...]      rewrite in place (idempotent; re-run after a rebase)
    check [PATH ...]      exit 1 while a 0.6 payload read of Result.monitor(...) remains

Without paths the scope is every tracked ``.py`` and ``.md`` file under ``tests``, ``benchmark``,
``scripts``, ``examples``, ``docs/reference``, ``docs/memory``, ``docs/README.md``, ``README.md`` and
``FEATURE_LIST.md``. Excluded: ``tests/mom/`` (MoM results are typed by phase 6d), historical
records, the codemods, the BITID probes and ``tests/support/api_dialect.py`` (they read 0.6 layouts
through the dialect), the consumer-metadata bridge, ``benchmark/sie_single_buffer_equivalence_v3.py``
and any file whose sha256 a tracked benchmark, script or test file pins (reported as ``skip``).
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
SITES_CSV = Path(__file__).with_name("070_p6b_monitor_sites.csv")
SCOPE = ("tests", "benchmark", "scripts", "examples", "docs/reference", "docs/memory", "docs/README.md",
         "README.md", "FEATURE_LIST.md")
EXCLUDE = re.compile(
    r"^(tests/mom/|docs/(assessments|archive|plans|releases)/|docs/reference/release-notes-"
    r"|CHANGELOG\.md$|scripts/dev/codemods/|tests/support/api_dialect\.py$"
    r"|tests/fdtd/_bit_identity_probe\.py$|tests/fdtd/test_raw_phase_bit_identity\.py$"
    r"|tests/boundaries/absorbers/_boundary_state_probe\.py$|tests/monitors/in_simulation/_bit_identity_runner\.py$"
    r"|tests/materials/surface_impedance/smooth_surface_probe\.py$|tests/support/rf_port_bit_identity_scene\.py$"
    r"|tests/support/rf_port_bit_identity_terminal_wave_scene\.py$|tests/support/port_step_graph_bit_identity_scene\.py$"
    r"|scripts/dev/fdtd/consumer_metadata/|benchmark/core_path/gates\.py$|tests/monitors/test_fdtd_typed_monitors\.py$"
    r"|benchmark/sie_single_buffer_equivalence_v3\.py$"
    # Migrated by hand: they pass typed monitor data on whole, which reads like a 0.6 payload hand-off.
    r"|tests/fdtd/test_time_consumer_consistency\.py$|tests/monitors/in_simulation/test_in_simulation_monitors_gpu\.py$"
    r"|tests/postprocess/resonance/test_resonance_fdtd_cavity\.py$)")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

# Monitor class -> typed kind when no trace row exists (static fallback).
CLASS_KINDS = {
    "PointMonitor": "point", "PlaneMonitor": "plane", "FinitePlaneMonitor": "plane", "FluxMonitor": "flux",
    "IncidentPowerDensityMonitor": "flux", "ClosedSurfaceMonitor": "closed", "ModeMonitor": "mode",
    "GaussianOverlapMonitor": "gaussian", "AstigmaticGaussianOverlapMonitor": "gaussian",
    "DipoleEmissionMonitor": "dipole", "DiffractionMonitor": "diffraction", "FieldTimeMonitor": "time",
    "AuxFieldTimeMonitor": "time", "FluxTimeMonitor": "flux_time", "SurfaceFieldTimeMonitor": "raw",
    "PermittivityMonitor": "raw", "MediumMonitor": "raw",
    "AngleProjectionMonitor": "unchanged", "CartesianProjectionMonitor": "unchanged",
    "KSpaceProjectionMonitor": "unchanged", "WireMonitor": "unchanged", "PowerLossMonitor": "unchanged",
}
# Kinds whose 0.6 payload had no frequency axis for one frequency; 0.7.0 keeps an f axis of length 1.
F_DROPPING = {"point", "plane", "plane_native", "flux", "closed", "mode", "dipole", "diffraction"}
# Kinds for which raw_monitor returns what monitor returned in 0.6.
RAW_EQUIVALENT = {"point", "plane", "plane_native", "flux", "closed", "time", "time_native", "flux_time", "raw"}
MODE_KEYS = {
    "amplitude_forward": '.isel(mode=0).sel(propagation="+").values',
    "amplitude_backward": '.isel(mode=0).sel(propagation="-").values',
    "power_forward": '.isel(mode=0).sel(propagation="+").attrs["power"]',
    "power_backward": '.isel(mode=0).sel(propagation="-").attrs["power"]',
    "power_fraction_forward": '.isel(mode=0).sel(propagation="+").attrs["power_fraction"]',
    "power_fraction_backward": '.isel(mode=0).sel(propagation="-").attrs["power_fraction"]',
}
# Keys that only resolved (non-raw) 0.6 monitor results carry: mode/Gaussian overlaps, diffraction, emission.
_RESOLVED_KEYS = frozenset({
    "amplitude_forward", "amplitude_backward", "power_forward", "power_backward", "power_fraction_forward",
    "power_fraction_backward", "effective_index", "beta", "beta_complex", "mode_power", "total_power", "mode_index",
    "orders", "propagating_power", "num_propagating", "incident_power", "reciprocal_vectors", "periods",
    "power_delivered", "current_spectrum", "purcell_factor", "plane", "candidate_diagnostics"})
MODE_ATTRS = ("effective_index", "beta", "beta_complex", "mode_power", "total_power", "mode_index",
              "polarization_axis", "background_index")


@dataclass
class Info:
    kind: str                 # point | plane | plane_native | flux | closed | mode | gaussian | dipole | diffraction
                              # | time | time_native | flux_time | raw | unchanged | mixed
    nfreq: str                # "1", "n", "mixed" or "" (untraced)
    fields: tuple[str, ...]
    traced: bool


@dataclass
class Site:
    path: str
    line: int
    decision: str             # typed | raw | review | skip
    detail: str = ""


# --- trace -> sites table -----------------------------------------------------------------
def trace_kind(record) -> str:
    value, cls = record["value"], record.get("monitor_class")
    if value["type"] != "dict":
        return "unchanged"
    kind = value.get("kind")
    fields = value.get("fields") or ()
    keys = set(value.get("keys") or ())
    if kind == "mode" or "amplitude_forward" in keys and cls == "ModeMonitor":
        return "mode"
    if cls in ("GaussianOverlapMonitor", "AstigmaticGaussianOverlapMonitor"):
        return "gaussian"
    if kind == "diffraction":
        return "diffraction"
    if kind == "dipole_emission" or value.get("monitor_type") == "dipole_emission":
        return "dipole"
    if kind == "closed_surface":
        return "closed"
    if kind in ("field_time", "aux_field_time"):
        return "time"
    if kind == "flux_time":
        return "flux_time"
    if kind in ("surface_field_time", "permittivity", "medium"):
        return "raw"
    if kind == "point" or "field_indices" in keys:
        return "point"
    if "flux" in keys and cls in ("FluxMonitor", "IncidentPowerDensityMonitor") or "compute_flux" in keys and \
            "flux" in keys:
        return "flux"
    if len(fields) > 1 and not all(field in keys for field in fields):
        return "plane_native"
    return "plane"


def build_sites(trace_dir: Path) -> None:
    rows = {}
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "site" not in record or record.get("api") not in ("monitor", "sel.monitor", "fields") \
                    or record.get("method") != "fdtd":
                continue
            site = record["site"]
            if not site["file"].startswith(("tests/", "benchmark/", "scripts/", "examples/")):
                continue
            text = call_text(ROOT / site["file"], site["pos"])
            if text is None:
                continue
            if record["api"] == "fields":
                snapshot = record.get("full_field_dft") is False and record["nfreq_result"] == 1
                kind = "fields_snapshot" if snapshot else "fields_phasor"
                nfreq, fields = ("1" if record["nfreq_result"] == 1 else "n"), ()
            else:
                kind = trace_kind(record)
                selected = bool(set(record["kwargs"]) & {"frequency", "freq_index"}) or record["api"] == "sel.monitor"
                nfreq = record["value"].get("nfreq")
                nfreq = "sel" if selected else "1" if nfreq == 1 else "n" if nfreq else ""
                fields = tuple(record["value"].get("fields") or ())
            key = (site["file"], site["func"], text)
            row = rows.setdefault(key, {"kinds": set(), "nfreq": set(), "fields": set()})
            row["kinds"].add(kind)
            row["nfreq"].add(nfreq)
            row["fields"].add(fields)
    with SITES_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "function", "call", "kind", "nfreq", "fields"))
        for (path, function, text), row in sorted(rows.items()):
            kind = row["kinds"].pop() if len(row["kinds"]) == 1 else "mixed"
            nfreq = row["nfreq"].pop() if len(row["nfreq"]) == 1 else "mixed"
            fields = row["fields"].pop() if len(row["fields"]) == 1 else ()
            writer.writerow((path, function, text, kind, nfreq, " ".join(fields)))
    print(f"wrote {len(rows)} site(s) to {SITES_CSV.relative_to(ROOT)}")


def call_text(path: Path, pos) -> str | None:
    if not pos or not path.exists():
        return None
    lineno, end_lineno, col, end_col = pos
    lines = path.read_text().splitlines()
    if lineno is None or end_lineno is None or end_lineno > len(lines):
        return None
    chunk = lines[lineno - 1:end_lineno]
    chunk[-1] = chunk[-1][:end_col]
    chunk[0] = chunk[0][col:]
    return normalize("\n".join(chunk))


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_sites() -> dict[tuple[str, str, str], Info]:
    if not SITES_CSV.exists():
        return {}
    with SITES_CSV.open(newline="") as handle:
        return {(row["path"], row["function"], row["call"]): Info(row["kind"], row["nfreq"], tuple(row["fields"].split()),
                                                                     True)
                for row in csv.DictReader(handle)}


# --- python rewriting ---------------------------------------------------------------------
class Source:
    def __init__(self, text: str):
        self.text = text
        self.starts = [0]
        for line in text.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line.encode("utf-8")))
        self.raw = text.encode("utf-8")

    def offset(self, line: int, col: int) -> int:
        return len(self.raw[:self.starts[line - 1] + col].decode("utf-8"))

    def span(self, node) -> tuple[int, int]:
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)

    def segment(self, node) -> str:
        start, end = self.span(node)
        return self.text[start:end]


def apply_edits(text: str, edits) -> str:
    edits = sorted(set(edits))
    for (s0, e0, _), (s1, _e1, _) in zip(edits, edits[1:]):
        if s1 < e0:
            raise RuntimeError(f"overlapping edits at {s0}..{e0} and {s1}")
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def _constant(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)) else None


class Expr(str):
    """A computed subscript key (its source text), admitted only as the component after ``"components"``."""


def _key_chain(node, parents, source=None):
    """Subscript keys applied to ``node`` and the outermost subscript node."""
    keys, outer = [], node
    while True:
        parent = parents.get(outer)
        if isinstance(parent, ast.Subscript) and parent.value is outer:
            key = _constant(parent.slice)
            if key is None and keys and keys[-1] == "components" and source is not None:
                key = Expr(source.segment(parent.slice))
            if key is not None:
                keys.append(key)
                outer = parent
                continue
        return keys, outer


DEFAULT_FIELDS = {"PointMonitor": ("Ez",), "PlaneMonitor": ("Ez",), "FinitePlaneMonitor": ("Ez",),
                  "AuxFieldTimeMonitor": ("Px", "Py", "Pz")}


def declared_monitors(tree) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Monitor name literal -> (class, recorded components) for the monitors constructed in a file."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            cls = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if cls == "box" and isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
                cls = func.value.attr
            if cls not in CLASS_KINDS:
                continue
            literal = _constant(node.args[0]) if node.args else None
            fields = DEFAULT_FIELDS.get(cls, ())
            for keyword in node.keywords:
                if keyword.arg == "name":
                    literal = _constant(keyword.value)
                if keyword.arg in ("fields", "components") and isinstance(keyword.value, (ast.Tuple, ast.List)):
                    fields = tuple(_constant(item) for item in keyword.value.elts)
                    fields = fields if all(isinstance(item, str) for item in fields) else ()
                elif keyword.arg in ("fields", "components"):
                    fields = ()
            if isinstance(literal, str):
                names.setdefault(literal, set()).add((cls, fields))
    return {name: declared.pop() for name, declared in names.items() if len(declared) == 1}


def typed_suffix(info: Info, keys) -> tuple[str, int] | None:
    """(suffix, number of keys consumed) of the typed expression for a 0.6 key chain, or None."""
    kind, first = info.kind, info.fields[0] if info.fields else None
    if not keys:
        return None
    head = keys[0]
    if kind in ("point", "plane", "time") and keys[:1] == ["components"] and len(keys) > 1 and isinstance(keys[1], Expr):
        if kind == "plane":
            if len(info.fields) == 1 and keys[2:3] == ["data"]:
                return f'.sel(component={keys[1]}).values', 3
            return None
        return f'.sel(component={keys[1]}).values', 2
    if kind in ("plane_native", "time_native") and keys[:1] == ["components"] and len(keys) > 1 and isinstance(keys[1], Expr):
        consumed = 3 if kind == "plane_native" and keys[2:3] == ["data"] else 2 if kind == "time_native" else 0
        return (f'[{keys[1]}].values', consumed) if consumed else None
    if kind in ("point", "plane", "time"):
        if head in COMPONENTS and (not info.fields or head in info.fields):
            if kind == "plane" and len(info.fields) > 1 and keys[1:2] == ["data"]:
                return None
            return f'.sel(component="{head}").values', 1
        if head in ("data", "field"):
            # The 0.6 alias existed only for one recorded component.
            return (f'.sel(component="{first}").values' if len(info.fields) == 1 else '.isel(component=0).values'), 1
        if head == "components" and len(keys) > 1 and keys[1] in COMPONENTS:
            if kind == "plane":
                if len(info.fields) == 1 and keys[2:3] == ["data"]:
                    return f'.sel(component="{keys[1]}").values', 3
                return None
            return f'.sel(component="{keys[1]}").values', 2
        if head == "t" and kind == "time":
            return '.coords["t"]', 1
        return None
    if kind == "plane_native":
        if head == "components" and len(keys) > 2 and keys[1] in COMPONENTS and keys[2] == "data":
            return f'["{keys[1]}"].values', 3
        return None
    if kind == "time_native":
        if head == "components" and len(keys) > 1 and keys[1] in COMPONENTS:
            return f'["{keys[1]}"].values', 2
        return None
    if kind in ("flux", "closed", "flux_time"):
        if head in ("flux", "power"):
            return ".values", 1
        if head == "t" and kind == "flux_time":
            return '.coords["t"]', 1
        return None
    if kind == "dipole":
        if head == "power_delivered":
            return ".values", 1
        if head == "current_spectrum":
            return '.attrs["current_spectrum"]', 1
        return None
    if kind in ("mode", "gaussian"):
        if head in MODE_KEYS:
            return MODE_KEYS[head], 1
        if head in MODE_ATTRS:
            return f'.attrs["{head}"]', 1
        return None
    return None


def raw_suffix(info: Info, keys) -> tuple[str, int] | None:
    """The 0.6 alias reads of a raw payload, rewritten to the component key."""
    if keys and keys[0] in ("data", "field"):
        if len(info.fields) != 1:
            return None
        component = info.fields[0]
        return (f'["components"]["{component}"]' if info.kind in ("time", "time_native") else f'["{component}"]'), 1
    return None


def rewrite_python(text: str, path: str, sites_table) -> tuple[str, list[Site]]:
    tree = ast.parse(text)
    source = Source(text)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for child in ast.walk(node):
                functions.setdefault(child, node)
    qualnames = _qualnames(tree)
    declared = declared_monitors(tree)
    edits, sites = [], []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "fields" and isinstance(node.ctx, ast.Load):
            _rewrite_fields(node, path, qualnames.get(functions.get(node), "<module>"), sites_table, source, parents,
                            edits, sites)
            continue
        if _is_payload_access(node):
            uses = _uses(node, parents, functions.get(node), source)
            if uses and any(use[0] == "keys" and use[2][0] in ("data", "field") for use in uses):
                sites.append(Site(path, node.lineno, "review", "alias read of a raw payload (\"data\"/\"field\" removed)"))
            continue
        call, name_node = _monitor_call(node)
        if call is None:
            continue
        if isinstance(call, ast.Call) and any(isinstance(arg, ast.Starred) for arg in call.args) or \
                isinstance(call, ast.Call) and any(keyword.arg is None for keyword in call.keywords):
            continue
        scope = functions.get(call)
        function = qualnames.get(scope, "<module>")
        info = _info(path, function, source.segment(call), sites_table, declared, name_node)
        line = call.lineno
        if info is None:
            # Unknown monitor: payload-key reads of a non-modal kind read what raw_monitor returns.
            uses = _uses(call, parents, scope, source)
            reads = [use for use in uses or () if use[0] == "keys"]
            if not reads and uses is not None:
                continue
            heads = {use[2][0] for use in reads}
            if uses is not None and reads and heads <= {"data", "field"} and all(use[0] == "keys" for use in uses):
                # Only the single-component alias is read: the stacked FieldData of any field monitor.
                for _use, outer, keys in reads:
                    base = _base(outer, len(keys))
                    edits.append((source.span(base)[1], source.span(_up(base, 1, parents))[1], '.isel(component=0).values'))
                sites.append(Site(path, line, "typed", "unresolved alias read"))
            elif uses is not None and reads and not heads & _RESOLVED_KEYS and not heads & {"data", "field"} \
                    and all(use[0] != "typed" for use in uses):
                edits.append(_raw_edit(call, source))
                sites.append(Site(path, line, "raw", f"unresolved: reads {_describe(uses)}"))
            else:
                sites.append(Site(path, line, "review", f"unresolved monitor: reads {_describe(uses)}"))
            continue
        if info.kind == "unchanged":
            continue
        if info.kind == "mixed" or info.nfreq == "mixed":
            sites.append(Site(path, line, "review", f"{info.kind}/{info.nfreq}: one call reads several monitor kinds"))
            continue
        uses = _uses(call, parents, scope, source)
        reads = [use for use in uses if use[0] == "keys"] if uses is not None else []
        if uses is not None and not reads and all(use[0] == "typed" for use in uses):
            continue  # already reads the typed API (or nothing)
        selected = isinstance(call, ast.Call) and _selected(call)
        add_index = info.nfreq == "1" and not selected and info.kind in F_DROPPING
        suffixes = [typed_suffix(info, keys) for _use, _outer, keys in reads]
        if uses is not None and all(use[0] != "whole" for use in uses) and all(suffixes):
            for (_use, outer, keys), (suffix, consumed) in zip(reads, suffixes):
                base = _base(outer, len(keys))
                edits.append((source.span(base)[1], source.span(_up(base, consumed, parents))[1], suffix))
            if isinstance(call, ast.Subscript):
                edits.append(_monitors_edit(call, source, add_index))
            elif add_index:
                edits.append(_freq_index_edit(call, source))
            sites.append(Site(path, line, "typed", f"{info.kind}{' +freq_index' if add_index else ''}"
                                                   f"{'' if info.traced else ' untraced'}"))
            continue
        if info.kind not in RAW_EQUIVALENT:
            sites.append(Site(path, line, "review", f"{info.kind}: reads {_describe(uses)}"))
            continue
        aliases = [(outer, keys, raw_suffix(info, keys)) for _use, outer, keys in reads]
        if any(keys[0] in ("data", "field") and suffix is None for _o, keys, suffix in aliases):
            sites.append(Site(path, line, "review", f"{info.kind}: alias read of a multi-component payload"))
            continue
        for outer, keys, suffix in aliases:
            if suffix is not None:
                base = _base(outer, len(keys))
                edits.append((source.span(base)[1], source.span(_up(base, 1, parents))[1], suffix[0]))
        edits.append(_raw_edit(call, source))
        sites.append(Site(path, line, "raw", f"{info.kind}: reads {_describe(uses)}"))
    return apply_edits(text, edits), sites


def _rewrite_fields(node, path, function, table, source, parents, edits, sites):
    """FDTD ``r.fields["EX"]``: phasors become ``FieldData`` (``f`` first), last-step fields ``r.time_snapshot``."""
    info = _info(path, function, source.segment(node), table, {}, None)
    if info is None or not info.kind.startswith("fields_"):
        return
    keys, outer = _key_chain(node, parents)
    if len(keys) != 1:
        sites.append(Site(path, node.lineno, "review", f"{info.kind}: Result.fields used whole"))
        return
    after = parents.get(outer)
    if isinstance(after, ast.Attribute) and after.attr in ("values", "isel", "sel", "coords", "attrs", "dims", "units"):
        return  # already reads the FieldData
    suffix = ".isel(f=0).values" if info.kind == "fields_phasor" and info.nfreq == "1" else ".values"
    if info.kind == "fields_snapshot":
        attr_end = source.offset(node.end_lineno, node.end_col_offset)
        edits.append((attr_end - len("fields"), attr_end, "time_snapshot"))
    edits.append((source.span(outer)[1], source.span(outer)[1], suffix))
    sites.append(Site(path, node.lineno, "typed", info.kind))


def _qualnames(tree) -> dict:
    names = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                names[child] = name
                visit(child, f"{name}.<locals>." if not isinstance(child, ast.ClassDef) else f"{name}.")
            elif isinstance(child, ast.Lambda):
                names[child] = f"{prefix}<lambda>"
                visit(child, f"{prefix}<lambda>.<locals>.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return names


def _is_payload_access(node) -> bool:
    """``x.raw_monitor(...)`` or ``x._monitors[...]``: payload dicts that lost their aliases."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "raw_monitor":
        return True
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "_monitors"


def _monitor_call(node):
    """``(node, name argument)`` for ``x.monitor(name, ...)`` or ``x.monitors[name]``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "monitor" \
            and node.args:
        return node, node.args[0]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "monitors" \
            and not (isinstance(node.value.value, ast.Name) and node.value.value.id == "scene"):
        return node, node.slice
    return None, None


def _info(path, function, call_text, table, declared, name_node) -> Info | None:
    key = (path, function, normalize(call_text))
    if key in table:
        return table[key]
    # Generator expressions and comprehensions run in their own code object.
    for (row_path, row_function, row_call), info in table.items():
        if row_path == path and row_call == key[2] and row_function.startswith(function):
            return info
    literal = _constant(name_node)
    if not isinstance(literal, str) or literal not in declared:
        return None
    cls, fields = declared[literal]
    kind = CLASS_KINDS[cls]
    if kind == "plane" and len(fields) > 1 and not set(fields) <= _ALIGNED_FIELDS:
        kind = "plane_native"
    return Info(kind, "", fields, False)


def _selected(call) -> bool:
    if any(keyword.arg in ("frequency", "freq_index") for keyword in call.keywords):
        return True
    owner = call.func.value
    return isinstance(owner, ast.Call) and isinstance(owner.func, ast.Attribute) and owner.func.attr in ("at", "select")


# Fields a plane monitor aligns on its cell centres for any normal (statically: never the normal H).
_ALIGNED_FIELDS = frozenset({"Ex", "Ey", "Ez"})
_DICT_METHODS = frozenset({"get", "keys", "items", "copy", "pop", "update", "setdefault"})


def _uses(call, parents, scope, source):
    """Every read of the call value as ``(use, outer node, keys)``, or None when the value escapes analysis.

    ``use`` is ``"keys"`` (a chain of constant subscripts), ``"typed"`` (an attribute of the typed API such as
    ``.sel``/``.values``) or ``"whole"`` (a dict method, a membership test, or the value passed on).
    """
    owner = parents.get(call)
    if not isinstance(owner, ast.Assign):
        return [_use(call, parents, source)]
    if len(owner.targets) != 1 or not isinstance(owner.targets[0], ast.Name) or scope is None:
        return None
    name = owner.targets[0].id
    stores = [node for node in ast.walk(scope)
              if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)]
    if len(stores) != 1:
        return None
    return [_use(node, parents, source) for node in ast.walk(scope)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)]


def _use(node, parents, source):
    keys, outer = _key_chain(node, parents, source)
    if keys:
        return ("keys", outer, keys)
    parent = parents.get(node)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return ("whole", node, [])  # a computed key
    if isinstance(parent, ast.Attribute):
        called = isinstance(parents.get(parent), ast.Call) and parents[parent].func is parent
        dict_api = parent.attr in _DICT_METHODS or (parent.attr == "values" and called)
        return ("whole" if dict_api else "typed", node, [])
    if isinstance(parent, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops) \
            and node is not parent.left:
        return ("whole", node, [])
    if isinstance(parent, ast.Expr):
        return ("typed", node, [])
    return ("whole", node, [])


def _describe(uses) -> str:
    if uses is None:
        return "?"
    return ",".join(sorted({"/".join(map(str, keys)) or f"<{use}>" for use, _o, keys in uses}))


def _base(outer, depth):
    """The node a chain of ``depth`` subscripts ending at ``outer`` is applied to."""
    for _ in range(depth):
        outer = outer.value
    return outer


def _up(node, count, parents):
    for _ in range(count):
        node = parents[node]
    return node


def _freq_index_edit(call, source):
    end = source.span(call)[1] - 1
    while source.text[end - 1] in " \n\t":
        end -= 1
    separator = "" if source.text[end - 1] in "(," else ", "
    return (end, end, f"{separator}freq_index=0")


def _monitors_edit(node, source, add_index):
    # r.monitors["p"] -> r.monitor("p")
    owner_start, owner_end = source.span(node.value)
    name = source.segment(node.slice)
    index = ", freq_index=0" if add_index else ""
    return (owner_start, source.span(node)[1], source.text[owner_start:owner_end - 1] + f"({name}{index})")


def _raw_edit(call, source):
    if isinstance(call, ast.Subscript):
        owner_start, owner_end = source.span(call.value)
        return (owner_start, source.span(call)[1],
                source.text[owner_start:owner_end - len("monitors")] + f"raw_monitor({source.segment(call.slice)})")
    attr_end = source.offset(call.func.end_lineno, call.func.end_col_offset)
    return (attr_end - len("monitor"), attr_end, "raw_monitor")


# --- markdown ---------------------------------------------------------------------------
def _dedent(body: str) -> tuple[str, str]:
    lines = [line for line in body.splitlines() if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in lines), default=0)
    prefix = lines[0][:indent] if lines else ""
    return "".join(line[indent:] if line.strip() else line for line in body.splitlines(keepends=True)), prefix


def rewrite_markdown(text: str, path: str, table) -> tuple[str, list[Site]]:
    sites, edits = [], []
    for match in FENCE.finditer(text):
        if match.group("lang").lower() not in ("python", "py", ""):
            continue
        code, prefix = _dedent(match.group("body"))
        line = text.count("\n", 0, match.start("body")) + 1
        try:
            new, block_sites = rewrite_python(code, path, table)
        except SyntaxError:
            if ".monitor(" in code:
                sites.append(Site(path, line, "review", "fenced block does not parse"))
            continue
        for item in block_sites:
            item.line += line - 1
        sites.extend(block_sites)
        if new != code:
            new = "".join(prefix + item if item.strip() else item for item in new.splitlines(keepends=True))
            edits.append((match.start("body"), match.end("body"), new))
    return apply_edits(text, edits), sites


# --- driver -----------------------------------------------------------------------------
def targets(paths: list[str]) -> list[Path]:
    inside, outside = [], []
    for item in paths or SCOPE:
        resolved = (ROOT / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        (inside if resolved.is_relative_to(ROOT) else outside).append(resolved)
    files = []
    if inside:
        listed = subprocess.run(["git", "ls-files", "--", *(str(p.relative_to(ROOT)) for p in inside)], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.split()
        files.extend(ROOT / item for item in listed if item.endswith((".py", ".md")) and not EXCLUDE.search(item))
    for root in outside:
        files.extend(sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.suffix in (".py", ".md")))
    return files


def pinned_hashes() -> set[str]:
    """sha256 digests that tracked benchmark, script and test files pin (a file with one of them stays as is)."""
    listed = subprocess.run(["git", "grep", "-ohE", "[0-9a-f]{64}", "--", "benchmark", "scripts", "tests"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
    return set(listed) - {EMPTY_SHA256}


def run(command: str, paths: list[str]) -> int:
    table = load_sites()
    pinned = pinned_hashes()
    sites, changed = [], 0
    for file in targets(paths):
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ".monitor" not in text and ".fields" not in text:
            continue
        rel = str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)
        if hashlib.sha256(file.read_bytes()).hexdigest() in pinned:
            sites.append(Site(rel, 1, "skip", "bytes pinned by sha256 elsewhere"))
            continue
        try:
            new, found = (rewrite_markdown if file.suffix == ".md" else rewrite_python)(text, rel, table)
        except SyntaxError as error:
            sites.append(Site(rel, error.lineno or 1, "review", f"does not parse: {error.msg}"))
            continue
        sites.extend(found)
        if command == "apply" and new != text:
            file.write_text(new, encoding="utf-8")
            changed += 1
    counts = Counter(site.decision for site in sites)
    if command == "classify":
        for decision, count in sorted(counts.items()):
            print(f"{decision:8s} {count}")
        details = Counter((site.decision, site.detail.split(":")[0]) for site in sites)
        for (decision, detail), count in sorted(details.items()):
            print(f"  {decision:8s} {detail:40s} {count}")
    review = [site for site in sites if site.decision == "review"]
    for site in review:
        print(f"review {site.path}:{site.line} {site.detail}")
    if command == "apply":
        print(f"rewrote {changed} file(s): {counts.get('typed', 0)} typed, {counts.get('raw', 0)} raw; "
              f"{len(review)} site(s) to review")
    if command == "check":
        left = [site for site in sites if site.decision in ("typed", "raw")]
        for site in left:
            print(f"old {site.path}:{site.line} {site.detail}")
        print(f"{len(left)} old-form site(s)")
        return 1 if left else 0
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("sites", "classify", "apply", "check"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "sites":
        build_sites(Path(args.paths[0]))
        return 0
    return run(args.command, args.paths)


if __name__ == "__main__":
    sys.exit(main())

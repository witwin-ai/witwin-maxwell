"""Generate the public-API inventory from the live namespaces and their stability markers.

    python scripts/dev/public_api.py          # check: print a diff and exit 1 when a file is stale
    python scripts/dev/public_api.py --write  # regenerate both files in place

``docs/reference/public-api.csv`` lists every public object: the top level ``mw`` and
every namespace ``__all__`` (``mw.rf``, ``mw.rf.rational``, ...), one row per object with
its kind, its stability (``stable`` or ``experimental``) and the first
release that exported its name (``since``, read from the release tags; names no release
exported carry the release in development). The FEATURE_LIST "API Stability" table is
generated between ``<!-- public-api-stability -->`` markers from the same markers.
Regenerate after adding, moving, removing or re-marking a public object.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import enum
import io
import re
import subprocess
import sys
import types
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import witwin.maxwell as mw  # noqa: E402
from witwin.maxwell._stability import EXPERIMENTAL, FEATURES, feature_of, stability  # noqa: E402

CSV_PATH = "docs/reference/public-api.csv"
FEATURE_LIST = "FEATURE_LIST.md"
BLOCK = "public-api-stability"
HEADER = ("namespace", "name", "kind", "stability", "since")
TAG_PREFIX = "witwin-maxwell-v"


def kind(obj) -> str:
    if isinstance(obj, type):
        if issubclass(obj, enum.Enum):
            return "enum"
        if issubclass(obj, Warning):
            return "warning"
        if issubclass(obj, BaseException):
            return "exception"
        return "class"
    if typing.get_origin(obj) is not None:
        return "type-alias"
    return "function" if callable(obj) else "constant"


def public_objects() -> list[tuple[str, str, object]]:
    """``(namespace, name, object)`` for the top level and every namespace ``__all__``."""
    found = [("mw", name, getattr(mw, name)) for name in mw.__all__]

    def collect(namespace: str, module: types.ModuleType) -> None:
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                collect(f"{namespace}.{name}", value)
            else:
                found.append((namespace, name, value))

    for name in mw._NAMESPACES:
        collect(f"mw.{name}", getattr(mw, name))
    return found


def rows() -> list[tuple[str, str, str, str]]:
    """``(namespace, name, kind, stability)`` of every public object, sorted by namespace and name."""
    result = []
    for namespace, name, value in public_objects():
        result.append((namespace, name, kind(value), stability(value)))
    return sorted(result, key=lambda row: (row[0] != "mw", row[0], row[1]))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _exported_names(source: str) -> set[str]:
    """String entries of ``__all__ = [...]``, ``__all__ += [...]`` and ``__all__.extend([...])``."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        value = None
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                value = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "extend"
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "__all__" and node.args):
            value = node.args[0]
        if value is not None:
            names |= {item.value for item in ast.walk(value)
                      if isinstance(item, ast.Constant) and isinstance(item.value, str)}
    return names


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def since() -> dict[str, str]:
    """First release whose top level or ``postprocess``/``invdes`` subpackage exported each name."""
    tags = sorted((tag[len(TAG_PREFIX):] for tag in _git("tag", "--list", f"{TAG_PREFIX}*").split()),
                  key=_version)
    first: dict[str, str] = {}
    for version in tags:
        for path in ("witwin/maxwell/__init__.py", "witwin/maxwell/postprocess/__init__.py",
                     "witwin/maxwell/invdes/__init__.py"):
            try:
                source = _git("show", f"{TAG_PREFIX}{version}:{path}")
            except subprocess.CalledProcessError:
                continue
            for name in _exported_names(source):
                first.setdefault(name, version)
    return first


def development_release() -> str:
    """The release in development: the package version, or the next minor once that is tagged."""
    version = _version(mw.__version__)
    if _git("tag", "--list", f"{TAG_PREFIX}{mw.__version__}").strip():
        version = (version[0], version[1] + 1, 0)
    return ".".join(map(str, version))


def render_csv(table, first: dict[str, str], default: str) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for row in table:
        writer.writerow((*row, first.get(row[1], default)))
    return buffer.getvalue()


def experimental_entry_points() -> list[tuple[str, str]]:
    """``(Simulation.<factory>, feature)`` for every experimental solver entry point."""
    entries = []
    for name in sorted(vars(mw.Simulation)):
        member = getattr(mw.Simulation, name)
        if callable(member) and stability(member) == EXPERIMENTAL:
            entries.append((f"Simulation.{name}", feature_of(member)))
    return entries


def render_stability_table() -> str:
    members: dict[str, list[str]] = {feature: [] for feature in FEATURES}
    for entry, feature in experimental_entry_points():
        members[feature].append(f"`{entry}`")
    for namespace, name, value in public_objects():
        if stability(value) == EXPERIMENTAL:
            members[feature_of(value)].append(f"`{namespace}.{name}`")
    lines = ["| Feature | Covers | Experimental objects |", "| --- | --- | --- |"]
    lines += [f"| `{feature}` | {description} | {', '.join(members[feature])} |"
              for feature, description in FEATURES.items()]
    return "\n".join(lines)


def render_feature_list(text: str) -> str:
    start, end = f"<!-- {BLOCK} -->", f"<!-- /{BLOCK} -->"
    pattern = re.compile(re.escape(start) + r"\n.*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"{FEATURE_LIST} has no {start} block")
    return pattern.sub(lambda _: f"{start}\n{render_stability_table()}\n{end}", text)


def render() -> dict[str, str]:
    return {
        CSV_PATH: render_csv(rows(), since(), development_release()),
        FEATURE_LIST: render_feature_list((ROOT / FEATURE_LIST).read_text()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="regenerate the files in place")
    args = parser.parse_args(argv)
    stale = 0
    for path, text in render().items():
        target = ROOT / path
        current = target.read_text() if target.exists() else ""
        if current == text:
            continue
        if args.write:
            target.write_text(text)
            print(f"wrote {path}")
        else:
            stale += 1
            sys.stdout.writelines(difflib.unified_diff(
                current.splitlines(True), text.splitlines(True), f"{path} (committed)", f"{path} (generated)"))
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())

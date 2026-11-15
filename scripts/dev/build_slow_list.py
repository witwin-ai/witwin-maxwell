"""Build tests/slow_tests.txt from measured per-node timings.

usage: build_slow_list.py OUT --collect COLLECT.txt [--threshold 60] [--base tests/slow_tests.txt]
                          LABEL=SOURCE... [--extra NODE=SECONDS:LABEL ...]

A SOURCE is either a timings JSON ({node: {"median": s, "count": n, ...}}) or JUnit XML: a
file, or a directory searched recursively (an integration batch output, CPU -n 24 runs, GPU
slot runs). Batch light-file group XML (group-*.xml) is skipped because each grouped unit
also has its own unit-*.xml. JUnit node ids are rebuilt from classname/name and kept only
when they exist in COLLECT.txt (``pytest --collect-only -q --run-slow --gpu`` of the tree the
list is for, without ``--run-mom-milestone``: milestone originals run only in the milestone
tier and stay off the slow list); skipped cases are ignored and durations are JUnit totals
(setup + call + teardown).

A node is slow when its median in any source exceeds the threshold; its comment keeps the
largest qualifying median and that source's label. ``--base`` keeps every entry of an
existing list (re-labelled only when a source measures a larger median), so a GPU source
such as an integration batch can be added without the receipts the list was first built
from; a base entry is dropped when its file was collected but no collected node matches it
(renamed or removed tests, milestone originals). ``--extra`` lists node-id prefixes (a file or a function covering its parameters);
entries under a prefix are folded into it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

HEADER = """# Slow tier: tests measured over 60 s on CPU or GPU (median of retained JUnit receipts, a
# CPU -n 24 run or a GPU slot run on a shared host). Each line is a pytest node id, or a
# node-id prefix (a file, or a test function covering all of its parameters), optionally
# followed by " # comment".
# tests/support/slow_tier.py adds the slow marker; directory-level runs deselect these
# unless --run-slow is given. Keep the list sorted; tests/support/test_slow_tier.py checks it.
"""
_COMMENT = re.compile(r"^(?P<seconds>\d+(?:\.\d+)?) s, (?P<label>.+)$")


def candidates(classname, name):
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        yield "::".join(["/".join(parts[:cut]) + ".py", *parts[cut:], name])


def junit_timings(source, nodes):
    """Per-node median of non-skipped JUnit durations under source; also the unmatched case count."""
    files = sorted(p for p in source.rglob("*.xml") if not p.name.startswith("group-")) if source.is_dir() else [source]
    seen, unmatched = {}, 0
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for case in root.iter("testcase"):
            classname, name = case.get("classname", ""), case.get("name", "")
            if not classname or case.find("skipped") is not None:
                continue
            nodeid = next((c for c in candidates(classname, name) if c in nodes), None)
            if nodeid is None:
                unmatched += 1
                continue
            seen.setdefault(nodeid, []).append(float(case.get("time") or 0))
    return {node: dict(median=statistics.median(values), count=len(values)) for node, values in seen.items()}, unmatched


def read_base(path):
    """Entries of an existing list: node -> (seconds, label); entries without a parsable comment keep 0 s."""
    entries = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        node, _, comment = line.partition(" #")
        match = _COMMENT.match(comment.strip())
        entries[node.strip()] = (float(match["seconds"]), match["label"]) if match else (0.0, comment.strip())
    return entries


def stale_entries(base, nodes):
    """Base entries whose file was collected but which match no collected node."""
    files = {node.split("::")[0] for node in nodes}
    return sorted(entry for entry in base if entry.split("::")[0] in files and entry not in nodes and
                  not any(node.startswith((entry + "::", entry + "[")) for node in nodes))


def build(sources, threshold, base=None, extras=()):
    """sources: [(label, {node: {"median", "count"}})]; returns sorted list lines."""
    best = dict(base or {})
    for label, timings in sources:
        for node, row in timings.items():
            seconds = row["median"]
            if seconds > threshold and seconds > best.get(node, (0.0, ""))[0]:
                best[node] = (seconds, f"{label} median of {row['count']}")
    prefixes = []
    for node, seconds, label in extras:
        best[node] = (seconds, label)
        prefixes.append(node)
    best = {node: row for node, row in best.items()
            if node in prefixes or not any(node.startswith((p + "::", p + "[")) for p in prefixes)}
    return [f"{node}  # {seconds:.0f} s, {label}" if label else node for node, (seconds, label) in sorted(best.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("sources", nargs="+", metavar="LABEL=SOURCE")
    parser.add_argument("--collect", type=Path, required=True, help="collect-only node listing of the target tree")
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--base", type=Path, help="existing slow list whose entries are kept")
    parser.add_argument("--extra", action="append", default=[], metavar="NODE=SECONDS:LABEL")
    args = parser.parse_args(argv)
    nodes = {line.strip() for line in args.collect.read_text().splitlines() if "::" in line}
    sources = []
    for spec in args.sources:
        label, path = spec.split("=", 1)
        path = Path(path)
        if path.suffix == ".json":
            timings = {node: row for node, row in json.loads(path.read_text()).items() if node in nodes}
            unmatched = None
        else:
            timings, unmatched = junit_timings(path, nodes)
        sources.append((label, timings))
        print(f"{label}: {len(timings)} timed nodes" + (f", {unmatched} cases not in the collection" if unmatched else ""))
    extras = []
    for spec in args.extra:
        node, rest = spec.rsplit("=", 1)
        seconds, label = rest.split(":", 1)
        extras.append((node, float(seconds), label))
    base = read_base(args.base) if args.base else None
    if base is not None:
        for entry in stale_entries(base, nodes):
            print(f"dropped stale base entry: {entry}")
            del base[entry]
    lines = build(sources, args.threshold, base, extras)
    args.out.write_text(HEADER + "\n".join(lines) + "\n")
    print(len(lines), "entries", f"({len(lines) - len(base)} new)" if base is not None else "")


if __name__ == "__main__":
    main()

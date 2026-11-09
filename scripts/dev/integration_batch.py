"""Plan, execute, report and bisect commit-bound integration batches.

The suite manifest is data only. Planning collects tests but never runs them.
CUDA_VISIBLE_DEVICES is inherited unchanged by every child.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import errno
import fcntl
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

PYTHON = sys.executable
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
MANIFEST = SCRIPT.with_name("integration_suites.toml")
STAGES = ("resident-bitid", "fdtd-baselines", "pcb-44", "mom-public-smoke", "thin-sheet-112", "benchmark",
          "two-device-slot", "two-device-full")
# Keep extraction and normalization identical to
# <workspace>/wt/cloud-p8/wcloud/staging.py.
_DOC_PATH = re.compile(rb"docs/[A-Za-z0-9_./-]*")
_DOC_JOIN = re.compile(rb'''["']docs["'](?:\s*[/,]\s*["'][A-Za-z0-9_./-]+["'])+''')
_DOC_PART = re.compile(rb'''["']([A-Za-z0-9_./-]+)["']''')

# Root keys accessed without defaults by planning, execution and pin checks.
_REQUIRED_MANIFEST_KEYS = frozenset({
    "base_commit", "components", "known_failures", "pin_files", "pytest_seconds_per_test", "suites",
})
# Include descriptive metadata documented by the manifest and its fixtures.
_MANIFEST_TABLE_KEYS = {
    "merge_declarations": frozenset({"branch", "commit", "pytest", "benchmark", "note"}),
    "known_failures": frozenset({"selector", "failure_contains", "reason", "breaking_commit", "owner"}),
    "expected_movers": frozenset({
        "commit", "selector", "row", "pins", "expected_change", "failure_contains",
        "tier", "max_rel_diff", "intended", "discrete",
    }),
    "suites": frozenset({
        "id", "kind", "status", "classification", "selectors", "argv", "estimated_seconds",
        "expected_tests", "tier", "requires_gpus", "two_device_only", "component", "report_format",
        "environment", "artifacts", "metric_limits", "scenario_metric_limits", "scenario_investigation",
        "scenario_quarantine", "owner", "evidence",
    }),
}


def load_manifest(path=None):
    """Load TOML and reject misplaced root keys and unknown array-table keys."""
    path = MANIFEST if path is None else Path(path)
    text = path.read_text()
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # Python 3.11 reports coordinates only in the exception message.
        location = re.search(r"\(at line (\d+), column (\d+)\)$", str(exc))
        lineno = getattr(exc, "lineno", int(location[1]) if location else text.count("\n") + 1)
        colno = getattr(exc, "colno", int(location[2]) if location else len(text.rsplit("\n", 1)[-1]) + 1)
        table = "top level"
        identities = []
        for number, line in enumerate(text.splitlines()[:lineno], 1):
            header = re.match(r"^\s*\[\[(merge_declarations|known_failures|expected_movers|suites)\]\]\s*(?:#.*)?$", line)
            if header:
                table = f"[[{header[1]}]] at line {number}"
                identities = []
            elif table != "top level" and re.match(r"^\s*(branch|commit|id)\s*=", line):
                identities.append(line.strip())
        context = table + (f" ({', '.join(identities)})" if identities else "")
        raise ValueError(f"{path}: line {lineno}, column {colno}, {context}: {exc}") from exc

    def entry_context(table, entry, index):
        key = next((key for key in ("branch", "commit", "id") if key in entry), next(iter(entry), None))
        identity = f"{key}={entry[key]}" if key is not None else f"entry={index}"
        return f"[[{table}]] {identity}"

    missing = []
    for key in sorted(_REQUIRED_MANIFEST_KEYS - manifest.keys()):
        locations = [entry_context(table, entry, index)
                     for table, entries in manifest.items() if isinstance(entries, list)
                     for index, entry in enumerate(entries, 1) if isinstance(entry, dict) and key in entry]
        context = "; found inside " + ", ".join(locations) if locations else ""
        missing.append(f"missing required top-level key {key!r}{context}")
    if missing:
        raise ValueError(f"{path}: " + "; ".join(missing))

    for table, allowed in _MANIFEST_TABLE_KEYS.items():
        for index, entry in enumerate(manifest.get(table, []), 1):
            unknown = entry.keys() - allowed
            if unknown:
                raise ValueError(f"{path}: {entry_context(table, entry, index)}: "
                                 f"unknown keys: {', '.join(sorted(unknown))}")
    return manifest


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def matches(node, selector):
    return (node == selector or node.startswith(selector.rstrip("/") + "/")
            or node.startswith(selector + "::") or node.startswith(selector + "["))


def balance(units, shards):
    """Deterministic longest-processing-time assignment; shard indices start at zero."""
    if shards < 1:
        raise ValueError("shards must be positive")
    lanes = [[] for _ in range(shards)]
    loads = [0.0] * shards
    for unit in sorted(units, key=lambda u: (-u["seconds"], u["id"])):
        lane = min(range(shards), key=lambda k: (loads[k], k))
        lanes[lane].append(unit)
        loads[lane] += unit["seconds"]
    return lanes, loads


def select_units(units, names):
    indexed = {u["id"]: u for u in units}
    return [indexed[name] for name in dict.fromkeys(names)]


def unit_order(unit):
    stage = min([i for i, name in enumerate(STAGES) if name in unit["suites"]] + [len(STAGES)])
    return stage, -unit["seconds"], unit["id"]


def unit_phase(unit):
    return 1 if unit_order(unit)[0] < 3 else 2


def stop_after(phase, status, policy):
    return status not in ("PASS", "INFRA") and (phase == 1 if policy is None else policy)


def docs_paths(text):
    """Best-effort ASCII literal paths/joins, matching the cloud stager."""
    paths = set(_DOC_PATH.findall(text))
    paths.update(b"/".join(_DOC_PART.findall(m.group())) for m in _DOC_JOIN.finditer(text))
    return {path.decode("ascii").rstrip("/.") for path in paths}


def missing_unit_docs(source, unit, excluded, text_mentions):
    paths = set()
    if unit["kind"] == "pytest":
        for path in {selector.split("::")[0] for selector in unit["selectors"]}:
            paths.update(docs_paths((source / path).read_bytes()))
    paths.difference_update(text_mentions)
    return sorted(path for path in paths if not (source / path).exists() or
                  any(matches(entry["path"], path) for entry in excluded))


def unit_lock(unit_id, default, overrides):
    lock = default
    for pattern, path in overrides:
        if fnmatchcase(unit_id, pattern):
            lock = path
    return Path(lock).resolve() if lock else None


def parse_window(value):
    def local(value):
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=ZoneInfo("America/Los_Angeles")) if parsed.tzinfo is None else parsed
    start, end = value.split("/")
    return local(start), local(end)


def deferred_until(now, seconds, windows):
    return next((end for start, end in sorted(windows)
                 if now < end and now + timedelta(seconds=1.5 * seconds) >= start), None)


@contextmanager
def unit_slot(lock, seconds, windows, on_defer):
    with (lock.open("a") if lock else nullcontext()) as handle:
        while True:
            until = deferred_until(datetime.now(timezone.utc), seconds, windows)
            if until is not None:
                on_defer(until)
                while (remaining := (until - datetime.now(timezone.utc)).total_seconds()) > 0:
                    time.sleep(min(60, remaining))
                continue
            if handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(1)
                    continue
                # Waiting for another lane must not consume the reserved execution margin.
                if deferred_until(datetime.now(timezone.utc), seconds, windows) is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    continue
            break
        try:
            yield
        finally:
            if handle:
                fcntl.flock(handle, fcntl.LOCK_UN)


def missing_shards(plans):
    """Numbered shards absent from the report. Claiming lanes are labelled, not numbered;
    their coverage is checked through missing units instead."""
    if all(p.get("claimed") for p in plans):
        return []
    return sorted({str(k) for k in range(plans[0]["shards"])} - {str(p["shard"]) for p in plans})


def failure_category(nodeid, data, details=""):
    for entry in data["known_failures"]:
        # A known-open row covers only its own failure signature; any other failure in that node is NEW.
        if matches(nodeid, entry["selector"]) and all(text in details for text in entry.get("failure_contains", [])):
            return "KNOWN", entry
    if details.startswith("Command exit "):
        # A crashed command produced no metrics to move; it is never an expected mover.
        return "NEW", None
    for entry in data.get("expected_movers", []):
        if mover_tier(entry) != "T2":
            continue
        if (("selector" in entry and matches(nodeid, entry["selector"]) and
             all(text in details for text in entry.get("failure_contains", []))) or
                ("row" in entry and nodeid.startswith("benchmark:") and
                 nodeid.removeprefix("benchmark:").split("[", 1)[0] == entry["row"])):
            return "EXPECTED", entry
    return "NEW", None


def known_only(result, data):
    failures = [c for c in result["cases"] if c["status"] == "FAIL"]
    return bool(failures) and all(failure_category(c["nodeid"], data, c.get("details", "") + c.get("assertion_message", ""))[0] == "KNOWN"
                                  for c in failures)


def two_device_reason(item):
    if "cuda_p2p_devices" in item.fixturenames:
        return "fixture cuda_p2p_devices"
    if item.get_closest_marker("nccl") is not None:
        return "marker nccl"
    for mark in item.iter_markers("skipif"):
        reason = mark.kwargs.get("reason", "")
        if re.search(r"\b(two|2)\b.*\b(cuda|gpu|devices)\b", reason, re.IGNORECASE):
            return f"skipif: {reason}"
    if hasattr(item, "callspec"):
        devices = item.callspec.params.get("devices", ())
        if isinstance(devices, (tuple, list)) and len({str(d) for d in devices}) >= 2 and all(str(d).startswith("cuda") for d in devices):
            return f"devices parameter: {devices}"
    return None


def require_two_gpu(enabled):
    if enabled:
        devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
        assert len(devices) == 2 and len(set(devices)) == 2 and all(devices), "--two-gpu requires two distinct CUDA_VISIBLE_DEVICES entries"


def suite_matches(node, suite, two_device):
    return (any(matches(node, selector) for selector in suite["selectors"])
            and (not suite.get("two_device_only", False) or node in two_device))


def progress(path, unit, event, passed, failed, elapsed):
    utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    line = (f"{utc} phase={unit_phase(unit)} unit={unit['id']} {event} "
            f"pass={passed} fail={failed} elapsed={elapsed:.3f}s")
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def test_ownership(files, components):
    owners = {path: [name for name, component in components.items()
                     if any(matches(path, s.split("::")[0]) for s in component["tests"])] for path in files}
    invalid = [f"{path}: {names or 'UNOWNED'}" for path, names in owners.items() if len(names) != 1]
    if invalid:
        raise RuntimeError("Test files must have exactly one component owner:\n" + "\n".join(invalid))
    return {path: names[0] for path, names in owners.items()}


def touched_components(paths, components):
    touched = set()
    for path in paths:
        if matches(path, "witwin/core") or "witwin-core" in Path(path).parts or path == "pyproject.toml":
            return list(components)
        source_matches = {name for name, c in components.items() if any(fnmatchcase(path, p) for p in c["sources"])}
        if path.startswith("witwin/maxwell/") and not source_matches:
            source_matches.add("shared")
        touched.update(source_matches)
        touched.update(name for name, c in components.items() if any(matches(path, s.split("::")[0]) for s in c["tests"]))
    return list(components) if "shared" in touched else [name for name in components if name in touched]


def provision_benchmark_cache(source):
    main = (ROOT / git(ROOT, "rev-parse", "--git-common-dir")).resolve().parent
    tracked = set(git(source, "ls-files", "--", "benchmark/cache").splitlines())
    for path in (main / "benchmark/cache").rglob("*"):
        relative = path.relative_to(main)
        if not path.is_file() or relative.as_posix() in tracked:
            continue
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            os.link(path, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(path, target)


def benchmark_cache_manifest(cache):
    entries = []
    for path in (p for p in cache.rglob("*") if p.is_file()):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        entries.append((path.relative_to(cache).as_posix(), path.stat().st_size, digest))
    entries.sort()
    return dict(benchmark_cache_sha256=hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest(),
                benchmark_cache_entries=len(entries), benchmark_cache_bytes=sum(e[1] for e in entries))


def checkout(sha, *, fresh=False):
    parent = ROOT / ".worktrees"
    parent.mkdir(exist_ok=True)
    with (parent / "ib-worktrees.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = Path(tempfile.mkdtemp(prefix=f"ib-{sha[:8]}-", dir=parent)) if fresh else parent / f"ib-{sha[:8]}"
        if not (path / ".git").exists():
            subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(path), sha], check=True)
        if git(path, "rev-parse", "HEAD") != sha or git(path, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)benchmark/RESULTS.md"):
            raise RuntimeError(f"Worktree must be clean at {sha}: {path}")
        provision_benchmark_cache(path)
        return path


def environment(source, extra=None):
    # The batch tree comes first so it shadows any installed package; an operator PYTHONPATH (for example a
    # sitecustomize memory cap inherited by spawned workers) stays on the path after it.
    path = os.pathsep.join(p for p in (str(source), os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, "PYTHONPATH": path, **(extra or {})}


def source_tree(args):
    if args.changed_files and not args.in_place:
        raise ValueError("--changed-files requires --in-place")
    if args.in_place:
        source = Path.cwd()
        # The uploaded tree has no .git; wcloud's provenance file is the only commit evidence.
        head = json.loads((source / ".wcloud-source.json").read_text())["git_head"]
        if not head.startswith(args.commit):
            raise ValueError(f"--commit {args.commit} does not match .wcloud-source.json git_head {head}")
        return head, source
    sha = git(ROOT, "rev-parse", f"{args.commit}^{{commit}}")
    return sha, checkout(sha)


def scenario_names(source):
    code = "import json; from benchmark.scenes import SCENARIOS; print(json.dumps(sorted(SCENARIOS)))"
    return json.loads(subprocess.check_output([PYTHON, "-c", code], cwd=source, env=environment(source), text=True))


def validate_scenario_metric_limits(suite, names, movers):
    for name, limits in suite.get("scenario_metric_limits", {}).items():
        if name not in names:
            raise ValueError(f"Unknown benchmark scenario: {name}")
        for key, value in limits.items():
            if key not in ("field_l2", "field_linf", "field_corr_min", "flux_error"):
                raise ValueError(f"Unknown benchmark metric: {key}")
            if not math.isfinite(value):
                raise ValueError(f"Non-finite benchmark limit: {name}.{key}")
            if key != "field_corr_min" and value <= 0:
                raise ValueError(f"Benchmark error limit must be positive: {name}.{key}")
            if key == "field_corr_min" and not 0 < value < 1:
                raise ValueError(f"Benchmark correlation limit must be in (0, 1): {name}.{key}")
    covered = {entry["row"] for entry in movers if "row" in entry}
    for name, reason in suite.get("scenario_investigation", {}).items():
        if name not in names:
            raise ValueError(f"Unknown benchmark scenario: {name}")
        if name in covered:
            raise ValueError(f"Investigation benchmark scenario is covered by a row mover: {name}")
        if name in suite.get("scenario_metric_limits", {}):
            raise ValueError(f"Investigation benchmark scenario has metric limits: {name}")
    for name, reason in suite.get("scenario_quarantine", {}).items():
        if name not in names:
            raise ValueError(f"Unknown benchmark scenario: {name}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Quarantined benchmark scenario needs a reason: {name}")
        if name in suite.get("scenario_metric_limits", {}):
            raise ValueError(f"Quarantined benchmark scenario has metric limits: {name}")


def stale_caches(source, names, directory):
    if not names:
        return []
    output = directory / "stale-caches.json"
    subprocess.run([PYTHON, str(SCRIPT), "_check-caches", str(output), *names],
                   cwd=source, env=environment(source, {"CUDA_VISIBLE_DEVICES": ""}), check=True)
    return json.loads(output.read_text())


def check_caches_main(args):
    import h5py

    sys.path.insert(0, str(Path.cwd()))
    from benchmark.paths import scenario_cache_path
    from benchmark.runner import _benchmark_cache_key, _resolve_source_normalization
    from benchmark.scenes import SCENARIOS, build_scene

    trust = os.environ.get("WITWIN_BENCHMARK_TRUST_CACHE") == "1"
    stale = []
    for name in args.names:
        scenario = SCENARIOS[name]
        if scenario.reference_solver != "tidy3d":
            continue
        scene = build_scene(name)
        expected = _benchmark_cache_key(
            scene, scenario.frequencies, scenario.run_time_factor,
            normalize_source=_resolve_source_normalization(scene, scenario.normalize_source))
        path = scenario_cache_path(name)
        found = None
        if path.is_file():
            with h5py.File(path, "r") as handle:
                found = handle.attrs.get("cache_key")
                if isinstance(found, bytes):
                    found = found.decode()
        if not path.is_file() or (not trust and found != expected):
            stale.append(dict(name=name, found=found, expected=expected))
    write_json(Path(args.output), stale)
    return 0


def benchmark_units(source, suite, names):
    if not names:
        return []
    timings = {}
    for line in (source / "benchmark/RESULTS.md").read_text().splitlines():
        row = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(row) == 18 and row[0] in names and row[11] != "-":
            timings[row[0]] = float(row[11])
    return [dict(suite, id=f"benchmark:{name}", argv=[*suite["argv"], name],
                 suites={suite["id"]: []}, nodes=[f"benchmark:{name}"], component=None, always_on=True,
                 seconds=timings.get(name, suite["estimated_seconds"] / len(names)),
                 evidence="RESULTS.md original-run timing" if name in timings else "Guess: benchmark total / scenario count")
            for name in names]


def collect(source, selectors, directory):
    result = directory / "collection.json"
    with (directory / "collection.log").open("w") as log:
        child = subprocess.run([PYTHON, str(SCRIPT), "_collect", str(result), *selectors],
                               cwd=source, env=environment(source), stdout=log, stderr=subprocess.STDOUT)
    data = json.loads(result.read_text())
    data["exit_code"] = child.returncode
    return data


def collect_main(args):
    import pytest

    sys.path.insert(0, str(Path.cwd()))
    data = {"nodes": [], "errors": [], "two_device": {}}

    class Collector:
        def pytest_collection_finish(self, session):
            data["nodes"] = [item.nodeid for item in session.items]
            data["two_device"] = {item.nodeid: reason for item in session.items if (reason := two_device_reason(item))}

        def pytest_collectreport(self, report):
            if report.failed:
                data["errors"].append(str(report.longrepr))

    code = pytest.main([*args.selectors, "--gpu", "--collect-only", "-q", "-o", "addopts="], plugins=[Collector()])
    from benchmark.scenes import SCENARIOS

    data["benchmark_scenarios"] = sorted(SCENARIOS)
    write_json(Path(args.output), data)
    return int(code)


def uncovered_bit_identity_files(source, manifest):
    """Bit-identity test files on disk that are not pinned and run by a BITID suite."""
    bitid = [sel for suite in manifest["suites"] if suite.get("classification") == "BITID" for sel in suite["selectors"]]
    files = sorted(p.relative_to(source).as_posix() for p in Path(source).glob("tests/**/test_*bit_identity*.py"))
    return [f for f in files if f not in manifest["pin_files"] or not any(matches(f, sel) for sel in bitid)]


def broad_movers(manifest):
    """Selector movers that would absorb every failure in a file or directory.

    A pytest mover must name test nodes ("file.py::test" or a parametrised node) or carry a
    failure_contains signature; file- or directory-wide movers masked real failures in batch 4.
    """
    return [e["selector"] for e in manifest.get("expected_movers", [])
            if "selector" in e and "::" not in e["selector"] and not e.get("failure_contains")]


def mover_tier(mover):
    """Intended or discrete movers require T2 even below numerical bounds."""
    return "T2" if mover.get("intended") or mover.get("discrete") else mover.get("tier", "T2")


def declared_pytest_movers(manifest):
    """Validate declaration syntax and return normalised pytest movers."""
    movers = []
    for entry in manifest.get("merge_declarations", []):
        if ("commit" in entry) == ("branch" in entry):
            raise ValueError(f"merge_declarations entry needs exactly one of commit or branch: {entry}")
        for key in ("pytest", "benchmark"):
            if entry[key] != "none" and not (isinstance(entry[key], list) and entry[key]):
                raise ValueError(f"merge_declarations {entry.get('commit') or entry['branch']}: {key} must be 'none' or a non-empty list")
        if isinstance(entry["pytest"], list):
            for index, mover in enumerate(entry["pytest"]):
                label = f"merge_declarations {entry.get('commit') or entry['branch']} pytest[{index}] {mover}"
                if isinstance(mover, str) and mover:
                    movers.append(dict(node=mover, tier="T2", approval="legacy declaration"))
                    continue
                if not isinstance(mover, dict) or not isinstance(mover.get("node"), str) or not mover["node"]:
                    raise ValueError(f"{label}: needs a node string")
                if mover.get("tier") not in ("T0", "T1", "T2"):
                    raise ValueError(f"{label}: tier must be T0, T1 or T2")
                for flag in ("intended", "discrete"):
                    if flag in mover and type(mover[flag]) is not bool:
                        raise ValueError(f"{label}: {flag} must be a boolean")
                tier = mover_tier(mover)
                if tier == "T1":
                    value = mover.get("max_rel_diff")
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{label}: T1 requires a finite nonnegative max_rel_diff")
                elif tier == "T2" and (not isinstance(mover.get("approval"), str) or not mover["approval"].strip()):
                    raise ValueError(f"{label}: T2 requires approval (including intended/discrete movers)")
                movers.append(dict(mover, tier=tier))
    return movers


def mover_declaration_errors(movers, declared):
    """Require each measured/expected mover to be disclosed at its effective tier."""
    errors = []
    for mover in movers:
        node = mover.get("nodeid", mover.get("selector"))
        if node is None:  # Benchmark row annotations retain their separate policy.
            continue
        tier = mover_tier(mover)
        if tier == "T0":
            continue
        entries = [entry for entry in declared if matches(node, entry["node"])]
        if tier == "T1":
            value = mover.get("max_rel", mover.get("max_rel_diff"))
            if not any(entry["tier"] == "T1" and type(value) in (int, float)
                       and 0 <= value <= entry["max_rel_diff"] for entry in entries):
                errors.append(f"{node}: T1 must be listed in merge_declarations with max_rel_diff "
                              f"covering the measured value {value}")
        elif not any(entry["tier"] == "T2" for entry in entries):
            errors.append(f"{node}: undeclared T2 mover requires a T2 merge_declarations entry with approval")
    return errors


def classify_movers(receipts, declared):
    rows = []
    for receipt in receipts:
        row = dict(receipt)
        node = row.get("nodeid", row.get("selector"))
        for entry in declared:
            if node is not None and matches(node, entry["node"]):
                for flag in ("intended", "discrete"):
                    if entry.get(flag):
                        row[flag] = True
        row["tier"] = mover_tier(row)
        rows.append(row)
    return rows


def undeclared_merges(source, sha, since, manifest):
    """First-parent solver/benchmark changes without a merge declaration."""
    declared_pytest_movers(manifest)
    declared = manifest.get("merge_declarations", [])
    missing = []
    for commit in git(source, "rev-list", "--first-parent", f"{since}..{sha}").splitlines():
        paths = git(source, "diff", "--name-only", f"{commit}^1", commit).splitlines()
        subject = git(source, "log", "-1", "--format=%s", commit).strip()
        # An open branch cannot know its merge commit: it declares `branch` instead, matched exactly against
        # the merge subject "Merge <branch>" followed by ":", " (" or the end of the subject.
        if any(p.startswith(("witwin/", "benchmark/")) for p in paths) and not any(
                commit.startswith(e["commit"]) if "commit" in e else
                subject == f"Merge {e['branch']}" or subject.startswith((f"Merge {e['branch']}:", f"Merge {e['branch']} (")) for e in declared):
            missing.append(commit[:8] + " " + subject[:80])
    return missing


def build_plan(source, sha, manifest, directory, since=None, tier="integration", two_gpu=False, exclude_components=(), changed_files=None):
    declared_movers = declared_pytest_movers(manifest)
    expected_movers = classify_movers(manifest.get("expected_movers", []), declared_movers)
    mover_errors = mover_declaration_errors(expected_movers, declared_movers)
    if manifest.get("require_merge_declarations", True) and mover_errors:
        raise ValueError("\n".join(mover_errors))
    components = manifest["components"]
    unknown = set(exclude_components) - components.keys()
    if unknown:
        raise ValueError(f"Unknown components: {', '.join(sorted(unknown))}")
    excluded = [name for name in components if name in exclude_components]
    files = sorted(p.relative_to(source).as_posix() for p in (source / "tests").rglob("*.py")
                   if p.name.startswith("test_") or p.name.endswith("_test.py"))
    owners = test_ownership(files, components)
    changed = Path(changed_files).read_text().splitlines() if changed_files is not None else None
    if since is not None:
        changed = git(source, "diff", "--name-only", f"{since}..{sha}").splitlines()
        broad = broad_movers(manifest)
        if broad:
            raise ValueError("Movers must name test nodes or carry failure_contains; file-wide selectors: " + ", ".join(broad))
        if manifest.get("require_merge_declarations"):
            missing = undeclared_merges(source, sha, since, manifest)
            if missing:
                raise ValueError("Merges without a [[merge_declarations]] entry (pytest/benchmark movers or 'none'):\n" + "\n".join(missing))
    selected = list(components) if changed is None else touched_components(changed, components)
    selected = [name for name in selected if name not in excluded]
    uncovered = uncovered_bit_identity_files(source, manifest)
    if uncovered:
        raise ValueError("Bit-identity test files must be in pin_files and a BITID suite: " + ", ".join(uncovered))
    declared = [s for s in manifest["suites"] if s.get("tier", "integration") != "milestone" or tier == "milestone"]
    suites = [dict(id=f"pytest-{name}", component=name, kind="pytest", status="READY", classification="MIXED",
                   selectors=components[name]["tests"]) for name in selected] + declared
    enabled = [s for s in suites if s.get("requires_gpus", 1) == 1 or two_gpu]
    cache = source / "benchmark/cache"
    cache_manifest = benchmark_cache_manifest(cache)
    # Work units are files, including the full files containing always-on node selectors.
    selectors = sorted({s.split("::")[0] for suite in suites if suite["status"] == "READY" and suite["kind"] == "pytest"
                        for s in suite["selectors"]})
    selectors = [s for s in selectors if not any(s != other and matches(s, other) for other in selectors)]
    collection = collect(source, selectors, directory)
    if collection["exit_code"] or collection["errors"]:
        raise RuntimeError("Collection failed. Set CUDA_VISIBLE_DEVICES to the lane GPU; see "
                           f"{directory / 'collection.log'}\n" + "\n".join(collection["errors"]))
    for suite in manifest["suites"]:
        if suite.get("report_format") == "benchmark":
            validate_scenario_metric_limits(suite, collection["benchmark_scenarios"], manifest.get("expected_movers", []))
    two_device = collection["two_device"]
    milestone_nodes = {node for node in collection["nodes"] if tier == "integration" and
                       any(s.get("tier") == "milestone" and s["kind"] == "pytest" and
                           suite_matches(node, s, two_device) for s in manifest["suites"])}

    def selected_by(node, suite):
        return (suite_matches(node, suite, two_device) and
                not (suite.get("component") and node in milestone_nodes))

    partial_files = {node.split("::")[0] for node in collection["nodes"] if node not in milestone_nodes}
    milestone_only = sorted({node if node.split("::")[0] in partial_files else node.split("::")[0]
                             for node in milestone_nodes})
    blocked = {node: reason for node, reason in two_device.items()
               if not (two_gpu and any(suite_matches(node, s, two_device) for s in enabled if s.get("requires_gpus") == 2))}
    by_file = defaultdict(list)
    for node in collection["nodes"]:
        if node not in blocked and any(selected_by(node, s) for s in enabled if s["kind"] == "pytest"):
            by_file[node.split("::")[0]].append(node)
    units = []
    counts = {}
    for suite in suites:
        if suite["status"] == "READY" and suite["kind"] == "pytest":
            counts[suite["id"]] = sum(selected_by(n, suite) for n in collection["nodes"])
            if counts[suite["id"]] == 0 and not suite.get("component"):
                raise RuntimeError(f"Suite selected no tests: {suite['id']}")
        else:
            counts[suite["id"]] = len(collection["benchmark_scenarios"]) if suite.get("report_format") == "benchmark" else suite["expected_tests"]
    for path, nodes in sorted(by_file.items()):
        members = {s["id"]: [n for n in nodes if selected_by(n, s)]
                   for s in enabled if s["status"] == "READY" and s["kind"] == "pytest"}
        members = {key: value for key, value in members.items() if value}
        seconds = max([manifest["pytest_seconds_per_test"] * len(nodes)] +
                      [s["estimated_seconds"] * len(members[s["id"]]) / counts[s["id"]]
                       for s in declared if s["id"] in members])
        text = (source / path).read_text()
        classification = "MIXED" if any(k in text for k in ("torch.equal", "rtol=0", "atol=0", "sha256", "hexdigest")) else "TOLERANCE"
        if any(s["classification"] == "BITID" and len(members.get(s["id"], [])) == len(nodes) for s in suites):
            classification = "BITID"
        selectors = [path] if set(nodes) == {n for n in collection["nodes"] if n.startswith(path + "::")} else nodes
        units.append(dict(id=f"pytest:{path}", kind="pytest", selectors=selectors, nodes=nodes,
                          suites=members, component=owners[path], classification=classification,
                          require_baselines=os.environ.get("WITWIN_REQUIRE_BASELINES") == "1" and
                          any(s["classification"] == "BITID" and s["id"] in members for s in suites),
                          always_on=any(s["id"] in members for s in declared),
                          seconds=manifest.get("unit_seconds", {}).get(path, seconds)))
    for suite in enabled:
        if suite["status"] == "READY" and suite["kind"] == "command":
            if suite.get("report_format") == "benchmark":
                units.extend(benchmark_units(source, suite, collection["benchmark_scenarios"]))
            else:
                units.append(dict(suite, suites={suite["id"]: []}, seconds=suite["estimated_seconds"],
                                  component=None, always_on=True, nodes=[]))
    stale = stale_caches(source, [u["id"].removeprefix("benchmark:") for u in units
                                  if u.get("report_format") == "benchmark"
                                  and u["id"].removeprefix("benchmark:") not in u.get("scenario_quarantine", {})], directory)
    for unit in units:
        if any(unit["id"] == f"benchmark:{entry['name']}" for entry in stale):
            unit.update(status="INFRA", reason="stale reference cache")
    return dict(commit=sha, manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), **cache_manifest,
                stale_caches=stale, trust_cache=os.environ.get("WITWIN_BENCHMARK_TRUST_CACHE") == "1",
                suites=suites, counts=counts, units=units, collection=collection, since=since,
                changed_paths=changed, components=list(components), selected_components=selected, tier=tier, two_gpu=two_gpu,
                excluded_components=excluded, milestone_only=milestone_only,
                disabled_suites=[s["id"] for s in suites if s not in enabled],
                known_failures=manifest["known_failures"], expected_movers=expected_movers,
                merge_declarations=manifest.get("merge_declarations", []),
                not_run=[dict(nodeid=n, reason=reason, status="NOT RUN (needs 2 GPUs)",
                              suites=[s["id"] for s in suites if s["kind"] == "pytest" and
                                      selected_by(n, s)])
                         for n, reason in blocked.items()])


def pin_values(source, *, effective=False):
    values = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not re.search(r"BASE|_FIELD_REF|_FINGERPRINT_REF", name):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "get":
            value = ast.Constant(os.environ.get(value.args[0].value, value.args[1].value)) if effective else value.args[1]
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and (effective or re.fullmatch(r"[0-9a-f]{7,64}", value.value)):
            values[name] = value.value
    return values


def pin_changes(sha, manifest):
    changes = []
    base = manifest["base_commit"]
    for path in manifest["pin_files"]:
        # A pin file added after the base has no previous pins.
        exists = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{base}:{path}"], capture_output=True).returncode == 0
        previous = pin_values(git(ROOT, "show", f"{base}:{path}")) if exists else {}
        for commit in git(ROOT, "rev-list", "--first-parent", "--reverse", f"{base}..{sha}", "--", path).splitlines():
            current = pin_values(git(ROOT, "show", f"{commit}:{path}"))
            for name in sorted(previous.keys() | current.keys()):
                if previous.get(name) != current.get(name):
                    changes.append(dict(file=path, pin=name, old=previous.get(name), new=current.get(name), causing_commit=commit))
            previous = current
    return changes


def junit_cases(path, require_baselines=False):
    cases = []
    for item in ET.parse(path).getroot().iter("testcase"):
        failure = item.find("failure")
        error = item.find("error")
        skipped = item.find("skipped")
        # pytest records an expected failure as <skipped type="pytest.xfail">; a strict XPASS
        # arrives as <failure>. XFAIL is pass-equivalent, real skips stay SKIP.
        xfail = skipped is not None and skipped.get("type") == "pytest.xfail"
        status = ("FAIL" if failure is not None or error is not None else "XFAIL" if xfail
                  else "SKIP" if skipped is not None else "PASS")
        details = [ET.tostring(x, encoding="unicode") for x in (failure, error, skipped) if x is not None]
        if require_baselines and status == "SKIP":
            status = "FAIL"
            details.append("Missing baseline: BITID skips fail when WITWIN_REQUIRE_BASELINES=1")
        classname = item.get("classname", "").split(".")
        module_end = next((i + 1 for i, part in enumerate(classname) if part.startswith("test_")), len(classname))
        nodeid = "/".join(classname[:module_end]) + ".py"
        nodeid += "::" + "::".join([*classname[module_end:], item.get("name", "")])
        cases.append(dict(nodeid=nodeid, status=status, details="\n".join(details), seconds=float(item.get("time", "0")),
                          assertion_message="\n".join(x.get("message", "") for x in (failure, error) if x is not None),
                          stdout_tail="\n".join(x.text or "" for x in item.findall("system-out"))[-8000:],
                          stderr_tail="\n".join(x.text or "" for x in item.findall("system-err"))[-8000:]))
    return cases


def benchmark_cases(path, unit):
    """Apply the existing metric guide targets, supplied explicitly by the manifest."""
    rows = [[c.strip() for c in line.split("|")[1:-1]] for line in path.read_text().splitlines()]
    cases = []
    for row in rows:
        if not row or f"benchmark:{row[0]}" not in unit["nodes"]:
            continue
        if len(row) == 18:
            metrics = ((4, "field_l2", False), (6, "field_linf", False), (7, "field_corr_min", True), (9, "flux_error", False))
            nodeid, seconds = f"benchmark:{row[0]}", float(row[11])
        elif len(row) == 7:
            metrics = ((2, "field_l2", False), (4, "field_linf", False), (5, "field_corr_min", True))
            nodeid, seconds = f"benchmark:{row[0]}[{row[1]} Hz]", 0
        else:
            continue
        quarantine = unit.get("scenario_quarantine", {}).get(row[0])
        if quarantine is not None:
            # A quarantined reference cannot qualify the scenario either way.
            cases.append(dict(nodeid=nodeid, status="SKIP", details=f"Quarantined reference: {quarantine}",
                              seconds=seconds))
            continue
        overrides = unit.get("scenario_metric_limits", {}).get(row[0], {})
        limits = {**unit["metric_limits"], **overrides}
        unconverged = any("unconverged" in row[i] for i, _, _ in metrics)
        failed = False
        details = []
        for i, key, lower in metrics:
            if row[i] == "-" or "unconverged" in row[i]:
                continue
            value = float(row[i])
            detail = f"{key}: {value}; target {'>' if lower else '<'} {limits[key]}"
            if key in overrides:
                detail += f" (scenario override; global {unit['metric_limits'][key]})"
            details.append(detail)
            if not (value > limits[key] if lower else value < limits[key]):
                failed = True
        if unconverged and not failed:
            details.append("Unconverged metrics are not passes")
        cases.append(dict(nodeid=nodeid, status="FAIL" if failed else "SKIP" if unconverged else "PASS",
                          details="; ".join(details), seconds=seconds))
    return cases


def execute(source, sha, unit, directory, lock):
    name = "unit-" + hashlib.sha256(unit["id"].encode()).hexdigest()[:16]
    xml = directory / f"{name}.xml"
    log = directory / f"{name}.log"
    command = [PYTHON]
    if unit["kind"] == "pytest":
        command += ["-m", "pytest", *unit["selectors"], "--gpu", "-ra", "-o", "addopts=",
                    "-o", "junit_logging=all", "-o", "junit_log_passing_tests=false",
                    f"--junitxml={xml}", f"--basetemp={directory / (name + '-tmp')}"]
    else:
        command += unit["argv"]
    benchmark = unit.get("report_format") == "benchmark"
    results_path = directory / f"{name}-RESULTS.md"
    tier_log = directory / f"{name}-pin-tiers.jsonl"
    tier_log.write_text("")
    extra_env = dict(unit.get("environment", {}))
    extra_env["WITWIN_PIN_TIER_LOG"] = str(tier_log.resolve())
    if benchmark:
        extra_env["WITWIN_BENCHMARK_RESULTS_PATH"] = str(results_path)
    # A quarantined scenario cannot be qualified (or may not run at all): do not execute it,
    # report it as SKIP with its reason so the unit stays INCOMPLETE and visible.
    quarantine = unit.get("scenario_quarantine", {}).get(unit["id"].removeprefix("benchmark:")) if benchmark else None
    started = time.monotonic()
    with log.open("w") as handle:
        if quarantine is None:
            child = subprocess.run(command, cwd=source, env=environment(source, extra_env),
                                   stdout=handle, stderr=subprocess.STDOUT)
        else:
            handle.write(f"Not executed: quarantined reference: {quarantine}\n")
            child = subprocess.CompletedProcess(command, 0)
    elapsed = time.monotonic() - started
    if quarantine is not None:
        cases = [dict(nodeid=unit["id"], status="SKIP", details=f"Quarantined reference: {quarantine}", seconds=0)]
    if unit["kind"] == "command":
        if quarantine is None:
            cases = benchmark_cases(directory / f"{name}-RESULTS.md", unit) if benchmark and (directory / f"{name}-RESULTS.md").exists() else []
        if quarantine is None and (not benchmark or child.returncode):
            cases.append(dict(nodeid=unit["id"], status="FAIL" if child.returncode else "PASS", seconds=elapsed,
                              details=f"Command exit {child.returncode}; see {log}"))
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite", name=unit["id"], tests=str(len(cases)),
                              failures=str(sum(c["status"] == "FAIL" for c in cases)))
        for outcome in cases:
            case = ET.SubElement(suite, "testcase", classname="command", name=outcome["nodeid"], time=str(outcome["seconds"]))
            if outcome["status"] != "PASS":
                ET.SubElement(case, "failure" if outcome["status"] == "FAIL" else "skipped").text = outcome["details"]
        ET.ElementTree(root).write(xml, encoding="unicode")
        for artifact in unit.get("artifacts", []):
            if not benchmark and (source / artifact).exists():
                shutil.copy2(source / artifact, directory / f"{name}-{Path(artifact).name}")
    else:
        if not xml.exists():
            root = ET.Element("testsuites")
            suite = ET.SubElement(root, "testsuite", name=unit["id"], tests="1", errors="1")
            case = ET.SubElement(suite, "testcase", classname="infrastructure", name=unit["id"])
            ET.SubElement(case, "error").text = f"Child exit {child.returncode} without JUnit; see {log}"
            ET.ElementTree(root).write(xml, encoding="unicode")
        cases = junit_cases(xml, unit.get("require_baselines", False))
    observed = {c["nodeid"] for c in cases}
    missing = sorted(set(unit.get("nodes", [])) - observed)
    status = "FAIL" if child.returncode or any(c["status"] == "FAIL" for c in cases) else "INCOMPLETE" if not cases or missing or any(c["status"] == "SKIP" for c in cases) else "PASS"
    result = dict(unit=unit, commit=sha, status=status, returncode=child.returncode, seconds=elapsed,
                  lock=str(lock) if lock else None,
                  cases=cases, missing_nodes=missing, log=str(log), junit=str(xml), command=command,
                  pin_tier_log=str(tier_log),
                  pin_tiers=[json.loads(line) for line in tier_log.read_text().splitlines()],
                  bitid_values={"old": None, "new": None, "causing_commit": None,
                                "exposure": "See original assertions and retained *-tmp payloads. Equality-only gates expose labels/booleans; add value dumps at the failing assertion in the selected test/probe."})
    write_json(directory / f"{name}.json", result)
    return result


def plan_main(args):
    require_two_gpu(args.two_gpu)
    sha, source = source_tree(args)
    with tempfile.TemporaryDirectory(prefix="ib-plan-") as temporary:
        plan = build_plan(source, sha, load_manifest(), Path(temporary), args.since, args.tier, args.two_gpu, args.exclude_component, args.changed_files)
    print("Component | Full pytest selected | GPU-h (including always-on overlap)")
    for name in plan["components"]:
        hours = sum(u["seconds"] for u in plan["units"] if u["component"] == name) / 3600
        selection = "EXCLUDED by operator" if name in plan["excluded_components"] else str(name in plan["selected_components"])
        print(f"{name} | {selection} | {hours:.3f}")
    always_hours = sum(u["seconds"] for u in plan["units"] if u["always_on"]) / 3600
    print(f"Always-on suites, deduplicated: {always_hours:.3f} GPU-h\n")
    for path in plan["milestone_only"]:
        print(f"milestone-only: {path}")
    for item in plan["not_run"]:
        print(f"NOT RUN (needs 2 GPUs): {item['nodeid']} [{item['reason']}]")
    if plan["trust_cache"]:
        print("WITWIN_BENCHMARK_TRUST_CACHE=1: cache-key comparison skipped (missing files still INFRA).")
    print("stale_caches: " + json.dumps(plan["stale_caches"]))
    for suite in plan["suites"]:
        seconds = sum(u["seconds"] for u in plan["units"] if suite["id"] in u["suites"])
        estimate = f"~{seconds / 60:.1f} min scheduled" if suite["status"] == "READY" else "time unknown"
        status = "NOT RUN (needs 2 GPUs)" if suite["id"] in plan["disabled_suites"] else suite["status"]
        print(f"{suite['id']}: {status} {suite['classification']}; {plan['counts'][suite['id']]} tests/cases; {estimate}.")
    lanes, loads = balance(plan["units"], args.shards)
    for k, lane in enumerate(lanes):
        print(f"\nShard {k}/{args.shards}: {len(lane)} units, ~{loads[k] / 3600:.2f} h")
        for unit in sorted(lane, key=unit_order):
            print(f"  phase={unit_phase(unit)} {unit['seconds']:.1f}s {unit['id']}"
                  + (f" INFRA ({unit['reason']})" if unit.get("status") == "INFRA" else "")
                  + (f" [{unit['evidence']}]" if unit.get("report_format") == "benchmark" else ""))
    print(f"\nUnique scheduled work: ~{sum(loads) / 3600:.2f} h; estimated wall ~{max(loads) / 3600:.2f} h (unresolved suites excluded).")
    for error in plan["collection"]["errors"]:
        print(error, file=sys.stderr)
    return int(plan["collection"]["exit_code"] != 0)


def claim_unit(claim_dir, unit_id):
    """Atomically claim a unit for this process (O_EXCL); False when another process has it."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(unit_id.encode()).hexdigest()[:16]
    try:
        os.close(os.open(claim_dir / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    (claim_dir / name).write_text(f"{unit_id}\n{os.getpid()}\n")
    return True


def wait_for_free_memory(min_free_mib):
    """Wait until the first visible GPU reports at least min_free_mib MiB free (nvidia-smi)."""
    if not min_free_mib:
        return
    index = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    while True:
        free = int(subprocess.check_output(["nvidia-smi", "-i", index, "--query-gpu=memory.free",
                                            "--format=csv,noheader,nounits"], text=True).strip())
        if free >= min_free_mib:
            return
        time.sleep(30)


def run_main(args):
    require_two_gpu(args.two_gpu)
    k, n = map(int, args.shard.split("/"))
    if not 0 <= k < n:
        raise ValueError("shard must be K/N with 0 <= K < N")
    sha, source = source_tree(args)
    label = args.label or str(k)
    directory = Path(args.out).resolve() / f"shard-{label}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest()
    plan = build_plan(source, sha, manifest, directory, args.since, args.tier, args.two_gpu, args.exclude_component, args.changed_files)
    lanes, _ = balance(plan["units"], n)
    # --units-from replaces the balanced lane with an explicit list (work stealing after a
    # stopped shard); every listed unit must exist in this commit's plan.
    assigned = ([line.strip() for line in Path(args.units_from).read_text().splitlines() if line.strip()]
                if args.units_from else [u["id"] for u in sorted(lanes[k], key=unit_order)])
    unknown = sorted(set(assigned) - {u["id"] for u in plan["units"]})
    if unknown:
        raise ValueError(f"--units-from lists units absent from the plan: {unknown[:5]}")
    plan.update(shard=label, shards=n, claimed=bool(args.claim_dir), assigned=sorted(assigned, key=lambda uid: unit_order(next(u for u in plan["units"] if u["id"] == uid))),
                stop_on_fail=args.stop_on_fail, pin_changes=None if args.in_place else pin_changes(sha, manifest),
                no_start_after=args.no_start_after,
                effective_pins={p: pin_values((source / p).read_text() if args.in_place else git(ROOT, "show", f"{sha}:{p}"),
                                              effective=True) for p in manifest["pin_files"]})
    write_json(directory / "plan.json", plan)
    overrides = [value.split("=", 1) for value in args.lock_for]
    windows = [parse_window(value) for values in args.no_start_after for value in values]
    progress_path = directory.parent / "progress.log"
    docs_manifest = os.environ.get("WITWIN_TEST_DOCS_MANIFEST") if args.in_place else None
    excluded_docs = json.loads(Path(docs_manifest).read_text())["excluded"] if docs_manifest else []
    results = []
    passed = failed = 0
    for unit in select_units(plan["units"], plan["assigned"]):
        started = time.monotonic()
        lock = unit_lock(unit["id"], args.lock, overrides)
        def on_defer(until):
            progress(progress_path, unit, f"DEFER until={until.isoformat()}", passed, failed, time.monotonic() - started)
        missing_docs = missing_unit_docs(source, unit, excluded_docs, manifest.get("docs_text_mentions", [])) if args.in_place else []
        reason = unit.get("reason") if unit.get("status") == "INFRA" else None
        if missing_docs:
            reason = "docs path not staged for cloud: " + ", ".join(missing_docs)
        with unit_slot(lock, unit["seconds"], windows, on_defer):
            if args.claim_dir and not claim_unit(Path(args.claim_dir), unit["id"]):
                continue
            if reason is None:
                wait_for_free_memory(args.min_free_mib)
            progress(progress_path, unit, "START", passed, failed, 0)
            if reason is not None:
                result = dict(unit=unit, commit=sha, status="INFRA", returncode=None, cases=[],
                              reason=reason)
                name = "unit-" + hashlib.sha256(unit["id"].encode()).hexdigest()[:16]
                write_json(directory / f"{name}.json", result)
            else:
                result = execute(source, sha, unit, directory, lock)
        results.append(result)
        passed += sum(c["status"] == "PASS" for c in result["cases"])
        failed += sum(c["status"] == "FAIL" for c in result["cases"])
        progress(progress_path, unit, f"END status={result['status']}", passed, failed, time.monotonic() - started)
        if stop_after(unit_phase(unit), result["status"], args.stop_on_fail) and not (unit_phase(unit) == 1 and known_only(result, manifest)):
            break
    return int(bool(plan["not_run"] or plan["disabled_suites"]) or any(r["status"] != "PASS" for r in results) or any(s["status"] != "READY" for s in plan["suites"]))


def pin_tier_summary(records, suites, declared=()):
    """Count BITID test nodes by worst tensor tier; exact fast paths emit no receipt.

    Passing nodes without receipts retain T0. Skips have no measured tier, and
    failures retain T2. Tensor-level T1 receipts are listed separately as movers.
    """
    bitid = {suite["id"] for suite in suites if suite["classification"] == "BITID"}
    summary = {}
    for result in records:
        nodes = {node for suite, members in result["unit"]["suites"].items() if suite in bitid for node in members}
        if not nodes:
            continue
        receipts = classify_movers([row for row in result.get("pin_tiers", []) if row["nodeid"] in nodes], declared)
        counts = dict(T0=0, T1=0, T2=0)
        for case in result["cases"]:
            if case["nodeid"] in nodes and case["status"] in ("PASS", "FAIL"):
                tier = "T2" if case["status"] == "FAIL" else max(
                    (row["tier"] for row in receipts if row["nodeid"] == case["nodeid"]), default="T0")
                counts[tier] += 1
        summary[result["unit"]["id"]] = dict(counts=counts,
                                              t1_movers=[row for row in receipts if row["tier"] == "T1"],
                                              movers=[row for row in receipts if row["tier"] != "T0"])
    return summary


def report_main(args):
    directories = sorted({p.parent.resolve() for d in args.directories
                          for pattern in ("shard-*/plan.json", "plan.json") for p in Path(d).glob(pattern)})
    plans = [json.loads((d / "plan.json").read_text()) for d in directories]
    first = plans[0]
    if len({(p["commit"], p["manifest_sha256"], p["shards"], p["since"], p["tier"], p["two_gpu"], tuple(p["excluded_components"]),
             None if p["changed_paths"] is None else tuple(p["changed_paths"])) for p in plans}) != 1:
        raise ValueError("Shard outputs must have the same commit, manifest, selection and GPU mode")
    if len({str(p["shard"]) for p in plans}) != len(plans):
        raise ValueError("Duplicate shard outputs")
    cache_keys = ("benchmark_cache_sha256", "benchmark_cache_entries", "benchmark_cache_bytes")
    if len({tuple(p[key] for key in cache_keys) for p in plans}) != 1:
        raise ValueError("Shards have different benchmark cache manifests")
    if any(p["units"] != first["units"] or p["collection"] != first["collection"] or p["effective_pins"] != first["effective_pins"] for p in plans):
        raise ValueError("Shards collected different inventories")
    records = sorted((json.loads(p.read_text()) for d in directories for p in d.glob("unit-*.json")),
                     key=lambda r: r["unit"]["id"])
    bisects = [json.loads(p.read_text()) for d in args.directories for p in Path(d).rglob("bisect.json")]
    declared_movers = declared_pytest_movers(first)
    mover_errors = []
    actual = {r["unit"]["id"]: r for r in records}
    if len(actual) != len(records):
        raise ValueError("Duplicate unit outputs")
    for result in records:
        if result["commit"] != first["commit"]:
            raise ValueError("Unit output belongs to another commit")
        if result["status"] == "INFRA":
            continue
        if result["unit"]["kind"] == "pytest":
            result["cases"] = junit_cases(Path(result["junit"]), result["unit"].get("require_baselines", False))
        result["pin_tiers"] = classify_movers(result.get("pin_tiers", []), declared_movers)
        mover_errors.extend(mover_declaration_errors(result["pin_tiers"], declared_movers))
        t2_nodes = {row["nodeid"] for row in result["pin_tiers"] if row["tier"] == "T2"}
        for case in result["cases"]:
            if case["nodeid"] in t2_nodes and case["status"] == "PASS":
                case.update(status="FAIL", details="Numerical equivalence T2: intended or discrete pin mover")
                result["status"] = "FAIL"
            if case["status"] == "FAIL":
                case["category"], case["annotation"] = failure_category(case["nodeid"], first, case["details"] + case["assertion_message"] if result["unit"]["kind"] == "pytest" else case["details"])
        for bisect in bisects:
            if result["unit"]["id"] in bisect["units"] and bisect["bad"] == first["commit"]:
                result["bitid_values"]["causing_commit"] = bisect["causing_commit"]
    missing = sorted({u["id"] for u in first["units"]} - actual.keys())
    suites = []
    for suite in first["suites"]:
        members = [u for u in first["units"] if suite["id"] in u["suites"]]
        outcomes = [actual[u["id"]] for u in members if u["id"] in actual]
        failures = [c for r in outcomes for c in r["cases"] if c["status"] == "FAIL" and
                    (r["unit"]["kind"] == "command" or c["nodeid"] in r["unit"]["suites"][suite["id"]])]
        blocked = any(suite["id"] in item["suites"] for item in first["not_run"])
        status = "NOT RUN (needs 2 GPUs)" if suite["id"] in first["disabled_suites"] else "UNRESOLVED" if suite["status"] != "READY" else "FAIL" if failures or any(r["returncode"] for r in outcomes) else "INFRA" if any(r["status"] == "INFRA" for r in outcomes) else "INCOMPLETE" if blocked or first["collection"]["exit_code"] or len(outcomes) != len(members) or any(r["status"] != "PASS" for r in outcomes) else "PASS"
        suites.append(dict(id=suite["id"], status=status, failures=failures))
    suites += [dict(id=f"pytest-{name}", status="EXCLUDED by operator", failures=[]) for name in first["excluded_components"]]
    report = dict(commit=first["commit"], since=first["since"], changed_paths=first["changed_paths"],
                  **{key: first[key] for key in cache_keys},
                  selected_components=first["selected_components"], tier=first["tier"], two_gpu=first["two_gpu"],
                  excluded_components=first["excluded_components"],
                  suites=suites, missing_units=missing, not_run=first["not_run"],
                  failing_scenarios=[r["unit"]["id"].removeprefix("benchmark:") for r in records
                                     if r["unit"].get("report_format") == "benchmark" and r["status"] == "FAIL"],
                  infra_units=[r for r in records if r["status"] == "INFRA"],
                  failures={category: [c for r in records for c in r["cases"] if c["status"] == "FAIL" and c["category"] == category]
                            for category in ("NEW", "KNOWN", "EXPECTED")},
                  missing_shards=missing_shards(plans),
                  collection_errors=[e for p in plans for e in p["collection"]["errors"]],
                  pin_changes=first["pin_changes"], effective_pins=first["effective_pins"], units=records, bisects=bisects,
                  bitid_tiers=pin_tier_summary(records, first["suites"], declared_movers),
                  mover_declaration_errors=mover_errors,
                  benchmark_note="Benchmark status applies the manifest's existing field/flux guide targets; unconverged or missing cases are incomplete. Scalar errors are retained in RESULTS.md without a new scalar gate.")
    target = Path(args.directories[0]).resolve()
    phase_one_infra = any(unit_phase(r["unit"]) == 1 for r in report["infra_units"])
    lines = [f"# Integration batch {report['commit']}" + (" — phase 1 incomplete (INFRA)" if phase_one_infra else ""), "",
             f"benchmark_cache_sha256: `{report['benchmark_cache_sha256']}`; "
             f"entries: {report['benchmark_cache_entries']}; bytes: {report['benchmark_cache_bytes']}.",
             "", "| Suite | Status |", "| --- | --- |"]
    lines += [f"| {s['id']} | {s['status']} |" for s in suites]
    lines += ["", "BITID tiers (test nodes, worst tensor tier; skips unmeasured):", "",
              "| Unit | T0 | T1 | T2 |", "| --- | --- | --- | --- |"]
    lines += [f"| {unit} | {row['counts']['T0']} | {row['counts']['T1']} | {row['counts']['T2']} |"
              for unit, row in report["bitid_tiers"].items()]
    lines += ["", "T1 tensor movers (auto-accepted):"]
    lines += [f"- `{unit}` / `{mover['nodeid']}` / `{mover['name']}`: max relative diff "
              f"`{mover['max_rel']:.17g}` ({mover['level']})."
              for unit, row in report["bitid_tiers"].items() for mover in row["t1_movers"]]
    lines += ["", "Mover declaration errors:", *mover_errors]
    lines += ["", "Failing benchmark scenarios: " + ", ".join(report["failing_scenarios"])]
    lines += ["", f"Diff base: `{report['since']}`; full pytest components: {', '.join(report['selected_components'])}."]
    lines += ["", report["benchmark_note"], "", f"Missing shards: {report['missing_shards']}", f"Missing units: {missing}", "", "Pin definition changes (manifest base to head):"]
    lines += [f"- `{p['file']}:{p['pin']}`: `{p['old']}` → `{p['new']}`; causing commit `{p['causing_commit']}`." for p in report["pin_changes"] or []]
    if report["pin_changes"] is None:
        lines += ["Unavailable in in-place mode (no git history)."]
    lines += ["", "Effective test pins (including environment overrides):"]
    lines += [f"- `{path}:{name}` = `{value}`" for path, pins in report["effective_pins"].items() for name, value in pins.items()]
    lines += ["", "## NOT RUN (needs 2 GPUs)", ""]
    lines += [f"- `{item['nodeid']}`: {item['reason']}" for item in report["not_run"]]
    lines += ["", "## INFRA", ""]
    lines += [f"- `{r['unit']['id']}`: {r['reason']}" for r in report["infra_units"]]
    for category in ("NEW", "KNOWN", "EXPECTED"):
        lines += ["", f"## {category}", ""]
        for result, failure in ((r, c) for r in records for c in r["cases"] if c["status"] == "FAIL" and c["category"] == category):
            lines += ["", f"Failing node: `{failure['nodeid']}`", "", "```text", failure["details"], "```"]
            if failure["annotation"] is not None:
                lines += ["", json.dumps(failure["annotation"], ensure_ascii=False)]
            if result["unit"]["classification"] in ("BITID", "MIXED"):
                for label, value in (("Assertion message", failure["assertion_message"]),
                                     ("JUnit stdout tail", failure["stdout_tail"]), ("JUnit stderr tail", failure["stderr_tail"])):
                    lines += ["", label + ":", "", "```text", value or "(not exposed in JUnit)", "```"]
                lines += ["", result["bitid_values"]["exposure"], f"Value dump location: `{result['unit'].get('selectors', [])}`; payloads beside `{result['log']}`. Causing commit: `{result['bitid_values']['causing_commit']}` (null means pending bisect)."]
    write_json(target / "integration-report.json", report)
    (target / "integration-report.md").write_text("\n".join(lines) + "\n")
    print(target / "integration-report.md")
    return int(bool(mover_errors or missing or report["missing_shards"] or report["collection_errors"] or report["not_run"]) or any(s["status"] != "PASS" for s in suites))


def probe_main(args):
    sha = git(Path.cwd(), "rev-parse", "HEAD")
    source = checkout(sha, fresh=True)
    directory = Path(args.out) / sha
    directory.mkdir(parents=True)
    manifest = load_manifest()
    for suite in manifest["suites"]:
        if suite.get("report_format") == "benchmark":
            validate_scenario_metric_limits(suite, scenario_names(source), manifest.get("expected_movers", []))
    units = []
    for name in args.units:
        if name.startswith("pytest:"):
            selector = name.removeprefix("pytest:")
            units.append(dict(id=name, kind="pytest", selectors=[selector], nodes=[], suites={}, classification="MIXED"))
        elif name.startswith("benchmark:"):
            suite = next(s for s in manifest["suites"] if s.get("report_format") == "benchmark")
            units.extend(benchmark_units(source, suite, [name.removeprefix("benchmark:")]))
        else:
            suite = next(s for s in manifest["suites"] if s["id"] == name and s["kind"] == "command")
            units.append(dict(suite, suites={name: []}, nodes=[f"benchmark:{n}" for n in scenario_names(source)]
                              if suite.get("report_format") == "benchmark" else []))
    try:
        results = [execute(source, sha, unit, directory, None) for unit in units]
        verdict = 125 if any(r["returncode"] not in (0, 1) or r["status"] == "INCOMPLETE" or
                             any(c["nodeid"].startswith("infrastructure/") for c in r["cases"]) for r in results) else int(any(r["status"] != "PASS" for r in results))
        write_json(directory / "verdict.json", {"exit_code": verdict})
        return verdict
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(source)], check=True)


def bisect_worker_main(args):
    # A probe exception must abort bisection, never label an unexecuted commit bad.
    sha = git(Path.cwd(), "rev-parse", "HEAD")
    child = subprocess.run([PYTHON, str(SCRIPT), "_probe", "--out", args.out, "--units", *args.units])
    verdict = Path(args.out) / sha / "verdict.json"
    return child.returncode if verdict.exists() else 128


def bisect_main(args):
    bad = git(ROOT, "rev-parse", f"{args.bad}^{{commit}}")
    control = checkout(bad, fresh=True)
    output = ROOT / ".worktrees" / f"ib-bisect-{bad[:8]}-{time.time_ns()}"
    output.mkdir()
    try:
        subprocess.run(["git", "-C", str(control), "bisect", "start", args.bad, args.good], check=True)
        command = ["git", "-C", str(control), "bisect", "run", PYTHON, str(SCRIPT), "_bisect-worker", "--out", str(output), "--units", *args.units]
        with (output / "bisect.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        (output / "bisect-history.txt").write_text(git(control, "bisect", "log") + "\n")
        log = (output / "bisect.log").read_text()
        culprit = re.search(r"([0-9a-f]{40}) is the first bad commit", log)
        write_json(output / "bisect.json", dict(good=args.good, bad=bad, units=args.units,
                   causing_commit=culprit.group(1) if culprit else None, returncode=result.returncode))
        print(log)
        print(output)
        return result.returncode
    finally:
        subprocess.run(["git", "-C", str(control), "bisect", "reset"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(control)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    plan = subs.add_parser("plan")
    plan.add_argument("--commit", required=True)
    plan.add_argument("--shards", type=int, default=1)
    selection = plan.add_mutually_exclusive_group()
    selection.add_argument("--since", help="base commit for component-scoped pytest selection")
    selection.add_argument("--in-place", action="store_true", help="use cwd without git or worktrees")
    plan.add_argument("--changed-files", type=Path, help="newline list of changed paths; requires --in-place")
    plan.add_argument("--exclude-component", action="append", default=[], metavar="NAME",
                      help="exclude a full-pytest component; always-on suites remain selected")
    plan.add_argument("--tier", choices=("integration", "milestone"), default="integration")
    plan.add_argument("--two-gpu", action="store_true", help="admit the selected two-device suites")
    run = subs.add_parser("run")
    run.add_argument("--commit", required=True)
    run.add_argument("--shard", required=True, help="zero-based K/N")
    run.add_argument("--out", required=True)
    run.add_argument("--lock")
    run.add_argument("--units-from", help="file of unit ids replacing the balanced shard lane")
    run.add_argument("--label", help="output directory label (shard-<label>); defaults to K")
    run.add_argument("--claim-dir", help="shared directory of O_EXCL unit claims; claimed units are skipped")
    run.add_argument("--min-free-mib", type=int, default=0, help="wait for this much free GPU memory before each unit")
    run.add_argument("--lock-for", action="append", default=[], metavar="PATTERN=PATH",
                     help="override the per-unit lock by unit-ID glob; last matching pattern wins")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--since", help="same base commit used by plan and the other shards")
    selection.add_argument("--in-place", action="store_true", help="use cwd without git or worktrees")
    run.add_argument("--changed-files", type=Path, help="newline list of changed paths; requires --in-place")
    run.add_argument("--exclude-component", action="append", default=[], metavar="NAME",
                     help="exclude a full-pytest component; always-on suites remain selected")
    run.add_argument("--tier", choices=("integration", "milestone"), default="integration")
    run.add_argument("--two-gpu", action="store_true", help="requires two CUDA_VISIBLE_DEVICES entries")
    run.add_argument("--no-start-after", action="append", nargs="+", default=[], metavar="START/END",
                     help="local Pacific exclusion windows; repeat the flag or supply several windows")
    policy = run.add_mutually_exclusive_group()
    policy.add_argument("--stop-on-fail", dest="stop_on_fail", action="store_const", const=True,
                        help="stop on any non-pass in either phase (default: stop only in phase 1)")
    policy.add_argument("--keep-going", dest="stop_on_fail", action="store_const", const=False,
                        help="continue after non-passing units in either phase")
    run.set_defaults(stop_on_fail=None)
    report = subs.add_parser("report")
    report.add_argument("directories", nargs="+")
    bisect = subs.add_parser("bisect")
    bisect.add_argument("--good", required=True)
    bisect.add_argument("--bad", required=True)
    bisect.add_argument("--units", nargs="+", required=True, help="plan unit IDs (pytest:path, benchmark:name or command suite ID)")
    collection = subs.add_parser("_collect", help=argparse.SUPPRESS)
    collection.add_argument("output")
    collection.add_argument("selectors", nargs="+")
    caches = subs.add_parser("_check-caches", help=argparse.SUPPRESS)
    caches.add_argument("output")
    caches.add_argument("names", nargs="+")
    probe = subs.add_parser("_probe", help=argparse.SUPPRESS)
    probe.add_argument("--out", required=True)
    probe.add_argument("--units", nargs="+", required=True)
    worker = subs.add_parser("_bisect-worker", help=argparse.SUPPRESS)
    worker.add_argument("--out", required=True)
    worker.add_argument("--units", nargs="+", required=True)
    args = parser.parse_args()
    return {"plan": plan_main, "run": run_main, "report": report_main,
            "bisect": bisect_main, "_collect": collect_main, "_check-caches": check_caches_main, "_probe": probe_main,
            "_bisect-worker": bisect_worker_main}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())

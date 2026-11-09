"""CPU coverage for integration work selection and deterministic balancing."""
import importlib.util
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from types import SimpleNamespace

import pytest


def test_benchmark_cache_provisioning_and_manifest(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    main = tmp_path / "main"
    source = tmp_path / "source"
    cache = main / "benchmark/cache"
    cache.mkdir(parents=True)
    (cache / "a").mkdir()
    (cache / "a/b").write_bytes(b"nested")
    (cache / "a.json").write_bytes(b"reference")
    (cache / "mom").mkdir()
    (cache / "mom/tracked.json").write_bytes(b"main")
    target = source / "benchmark/cache"
    (target / "mom").mkdir(parents=True)
    tracked = target / "mom/tracked.json"
    tracked.write_bytes(b"commit")
    monkeypatch.setattr(runner, "ROOT", main)
    monkeypatch.setattr(runner, "git", lambda root, *args:
                        ".git" if args[0] == "rev-parse" else "benchmark/cache/mom/tracked.json")
    runner.provision_benchmark_cache(source)
    runner.provision_benchmark_cache(source)
    assert (target / "a.json").samefile(cache / "a.json")
    assert tracked.read_bytes() == b"commit"
    with monkeypatch.context() as patch:
        def cross_device(*args):
            raise OSError(errno.EXDEV, "cross-device link")
        patch.setattr(runner.os, "link", cross_device)
        runner.provision_benchmark_cache(source)
        assert (target / "a.json").read_bytes() == b"reference"
        assert not (target / "a.json").samefile(cache / "a.json")
        def denied(*args):
            raise PermissionError(errno.EACCES, "denied")
        patch.setattr(runner.os, "link", denied)
        with pytest.raises(PermissionError):
            runner.provision_benchmark_cache(source)
    runner.provision_benchmark_cache(source)
    entries = [[name, len(data), hashlib.sha256(data).hexdigest()] for name, data in
               (("a.json", b"reference"), ("a/b", b"nested"), ("mom/tracked.json", b"commit"))]
    expected = dict(benchmark_cache_sha256=hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest(),
                    benchmark_cache_entries=3, benchmark_cache_bytes=21)
    assert runner.benchmark_cache_manifest(target) == expected
    (target / "a.json").unlink()
    (target / "a.json").write_bytes(b"different")
    assert runner.benchmark_cache_manifest(target)["benchmark_cache_sha256"] != expected["benchmark_cache_sha256"]
    (target / "a.json").unlink()
    runner.provision_benchmark_cache(source)

    suite = dict(id="benchmark", kind="command", status="READY", classification="TOLERANCE",
                 report_format="benchmark", estimated_seconds=1)
    manifest = dict(components={}, suites=[suite], known_failures=[], expected_movers=[])
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=[], exit_code=0, errors=[], two_device={}, benchmark_scenarios=[]))
    plan = runner.build_plan(source, "fixture", manifest, tmp_path)
    assert all(plan[key] == value for key, value in expected.items())
    runner.build_plan(tmp_path / "missing", "fixture", manifest, tmp_path)
    manifest["suites"] = []
    plan = runner.build_plan(source, "fixture", manifest, tmp_path)
    plan.update(shards=2, since=None, tier="integration", two_gpu=False, effective_pins={}, pin_changes=[])
    for shard in ("1b", "1c", 0, 1):
        directory = tmp_path / "results" / f"shard-{shard}"
        directory.mkdir(parents=True)
        runner.write_json(directory / "plan.json", dict(plan, shard=shard))
    nested = directory / "unit-x-tmp/nested/plan.json"
    nested.parent.mkdir(parents=True)
    runner.write_json(nested, {})
    args = SimpleNamespace(directories=[str(tmp_path / "results")])
    assert runner.report_main(args) == 0
    report = json.loads((tmp_path / "results/integration-report.json").read_text())
    assert all(report[key] == value for key, value in expected.items())
    assert expected["benchmark_cache_sha256"] in (tmp_path / "results/integration-report.md").read_text()
    assert runner.report_main(SimpleNamespace(directories=[str(directory)])) == 1
    for key in expected:
        runner.write_json(directory / "plan.json", dict(plan, shard=1, **{key: "different"}))
        with pytest.raises(ValueError, match="different benchmark cache manifests"):
            runner.report_main(args)


def test_scenario_units_and_cache_precheck(monkeypatch, tmp_path):
    import h5py
    import sys

    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    names = ["match", "mismatch", "missing", "unkeyed", "local"]
    cache = tmp_path / "benchmark/cache"
    cache.mkdir(parents=True)
    for name in ("match", "mismatch", "unkeyed"):
        with h5py.File(cache / f"{name}.h5", "w") as handle:
            if name != "unkeyed":
                handle.attrs["cache_key"] = "expected" if name == "match" else "old"
    scenarios = {name: SimpleNamespace(reference_solver="fdtd" if name == "local" else "tidy3d",
                 frequencies=(2.0,), run_time_factor=3.0, normalize_source=None) for name in names}
    keys = []
    def key(scene, frequencies, run_time_factor, *, normalize_source):
        keys.append((scene, frequencies, run_time_factor, normalize_source))
        return "expected"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "benchmark.paths", SimpleNamespace(scenario_cache_path=lambda n: cache / f"{n}.h5"))
        patch.setitem(sys.modules, "benchmark.scenes", SimpleNamespace(SCENARIOS=scenarios, build_scene=lambda n: n))
        patch.setitem(sys.modules, "benchmark.runner", SimpleNamespace(_benchmark_cache_key=key,
                      _resolve_source_normalization=lambda scene, requested: False))
        patch.delenv("WITWIN_BENCHMARK_TRUST_CACHE", raising=False)
        args = SimpleNamespace(names=names, output=str(tmp_path / "keys.json"))
        runner.check_caches_main(args)
        strict = json.loads(Path(args.output).read_text())
        patch.setenv("WITWIN_BENCHMARK_TRUST_CACHE", "1")
        runner.check_caches_main(args)
        trusted = json.loads(Path(args.output).read_text())
    expected = [dict(name=n, found="old" if n == "mismatch" else None, expected="expected")
                for n in ("mismatch", "missing", "unkeyed")]
    assert (strict, trusted, keys) == (expected, expected[1:2],
        [(n, (2.0,), 3.0, False) for n in names[:-1]] * 2)

    row = ["match", *("-" for _ in range(17))]
    row[11] = "7"
    (tmp_path / "benchmark/RESULTS.md").write_text("| " + " | ".join(row) + " |\n")
    suite = dict(id="benchmark", kind="command", status="READY", argv=["-m", "benchmark"],
                 report_format="benchmark", estimated_seconds=50, classification="TOLERANCE",
                 metric_limits=dict(field_l2=.1, field_linf=.1, field_corr_min=.99, flux_error=.05),
                 scenario_metric_limits={"match": dict(field_linf=.4781)})
    manifest = dict(components={}, suites=[suite], known_failures=[], expected_movers=[], pin_files=[],
                    base_commit="fixture", pytest_seconds_per_test=1)
    monkeypatch.setattr(runner, "collect", lambda *a: dict(nodes=[], two_device={}, exit_code=0,
                        errors=[], benchmark_scenarios=names))
    monkeypatch.setattr(runner, "stale_caches", lambda *a: strict)
    monkeypatch.setattr(runner, "source_tree", lambda a: ("fixture", tmp_path))
    monkeypatch.setattr(runner.tomllib, "loads", lambda text: manifest)
    commands = []
    def child(command, *, env, **kwargs):
        commands.append(command)
        row[0] = command[-1]
        row[4] = row[6] = "0.2" if row[0] == "match" else "0.05"
        Path(env["WITWIN_BENCHMARK_RESULTS_PATH"]).write_text("| " + " | ".join(row) + " |\n")
        return SimpleNamespace(returncode=int(command[-1] == "match"))
    monkeypatch.setattr(runner.subprocess, "run", child)
    args = SimpleNamespace(commit="fixture", shard="0/1", out=str(tmp_path / "out"),
                           since=None, tier="integration", two_gpu=False, exclude_component=[], changed_files=None,
                           in_place=True, lock=None, lock_for=[], label="0", units_from=None, claim_dir=None,
                           min_free_mib=0, stop_on_fail=None, no_start_after=[])
    status = runner.run_main(args)
    plan = json.loads((tmp_path / "out/shard-0/plan.json").read_text())
    report_status = runner.report_main(SimpleNamespace(directories=[args.out]))
    report = json.loads((tmp_path / "out/integration-report.json").read_text())
    assert (status, report_status, [c[1:] for c in commands],
            [(u["id"], u["seconds"], u["status"]) for u in plan["units"]],
            [(s["id"], s["status"]) for s in report["suites"]],
            [r["unit"]["id"] for r in report["infra_units"]],
            "Failing benchmark scenarios: match" in (tmp_path / "out/integration-report.md").read_text()) == (
        1, 1, [["-m", "benchmark", n] for n in ("local", "match")],
        [(f"benchmark:{n}", 7 if n == "match" else 10,
          "INFRA" if n in names[1:4] else "READY") for n in names],
        [("benchmark", "FAIL")], [f"benchmark:{n}" for n in names[1:4]], True)
    markdown = (tmp_path / "out/integration-report.md").read_text()
    assert "field_l2: 0.2; target < 0.1" in markdown
    assert "field_linf: 0.2; target < 0.4781 (scenario override; global 0.1)" in markdown

    worklist = tmp_path / "units.txt"
    worklist.write_text("benchmark:local\n")
    args.units_from, args.claim_dir, args.lock = str(worklist), str(tmp_path / "claims"), str(tmp_path / "lane.lock")
    args.min_free_mib = 100
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    free = iter(("0", "100"))
    queries = []
    sleeps = []
    events = []
    claim = runner.claim_unit
    def take_claim(directory, unit):
        events.append("claim")
        return claim(directory, unit)
    def query(command, **kwargs):
        queries.append(command)
        events.append("memory")
        return next(free)
    monkeypatch.setattr(runner, "claim_unit", take_claim)
    monkeypatch.setattr(runner.subprocess, "check_output", query)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    monkeypatch.setattr(runner, "progress", lambda path, unit, event, *a: events.append(event))
    statuses = []
    for label in ("1b", "1c"):
        args.label = label
        statuses.append(runner.run_main(args))
    assert (statuses, json.loads((tmp_path / "out/shard-1b/plan.json").read_text())["assigned"],
            list((tmp_path / "out/shard-1c").glob("unit-*.json")), queries, sleeps, events) == (
        [0, 0], ["benchmark:local"], [],
        [["nvidia-smi", "-i", "GPU-first", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]] * 2,
        [30], ["claim", "memory", "memory", "START", "END status=PASS", "claim"])
    # Claiming lanes are labelled, not numbered: their shards are not reported missing.
    units = [dict(id="a"), dict(id="b")]
    claimed = [dict(shard=label, shards=2, units=units, claimed=True) for label in ("g0", "g1")]
    assert runner.missing_shards(claimed) == []
    assert runner.missing_shards([dict(shard=0, shards=2, units=units, claimed=False)]) == ["1"]
    assert json.loads((tmp_path / "out/shard-1b/plan.json").read_text())["claimed"] is True


def test_benchmark_scenario_metric_limits_per_row(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    results = tmp_path / "RESULTS.md"
    rows = []
    for name in ("override", "global"):
        aggregate = [name, *("-" for _ in range(17))]
        aggregate[4], aggregate[6], aggregate[7], aggregate[9], aggregate[11] = ".2", ".2", "1", ".01", "7"
        rows.extend([aggregate, [name, "1e9", ".05", "-", ".2", "1", "-"]])
    results.write_text("\n".join("| " + " | ".join(row) + " |" for row in rows))
    unit = dict(nodes=["benchmark:override", "benchmark:global"],
                metric_limits=dict(field_l2=.1, field_linf=.1, field_corr_min=.99, flux_error=.05),
                scenario_metric_limits={"override": dict(field_linf=.4781)})
    cases = runner.benchmark_cases(results, unit)
    assert [(case["nodeid"], case["status"]) for case in cases] == [
        ("benchmark:override", "FAIL"), ("benchmark:override[1e9 Hz]", "PASS"),
        ("benchmark:global", "FAIL"), ("benchmark:global[1e9 Hz]", "FAIL")]
    assert "field_l2: 0.2; target < 0.1" in cases[0]["details"]
    for case in cases[:2]:
        assert "field_linf: 0.2; target < 0.4781 (scenario override; global 0.1)" in case["details"]
    for case in cases[2:]:
        assert "field_linf: 0.2; target < 0.1" in case["details"]
        assert "scenario override" not in case["details"]


@pytest.mark.parametrize("overrides, message", [
    ({"unknown": {"field_linf": .2}}, "Unknown benchmark scenario"),
    ({"known": {"unknown": .2}}, "Unknown benchmark metric"),
    ({"known": {"field_linf": float("nan")}}, "Non-finite benchmark limit"),
    ({"known": {"flux_error": float("inf")}}, "Non-finite benchmark limit"),
    ({"known": {"field_l2": 0}}, "error limit must be positive"),
    ({"known": {"flux_error": -.1}}, "error limit must be positive"),
    ({"known": {"field_corr_min": 0}}, "correlation limit must be in"),
    ({"known": {"field_corr_min": 1}}, "correlation limit must be in"),
])
def test_benchmark_scenario_metric_limits_validation(monkeypatch, tmp_path, overrides, message):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = dict(components={}, suites=[dict(report_format="benchmark", kind="command", status="READY",
                                               scenario_metric_limits=overrides)])
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=[], exit_code=0, errors=[],
                                                           benchmark_scenarios=["known"]))
    with pytest.raises(ValueError, match=message):
        runner.build_plan(tmp_path, "fixture", manifest, tmp_path)


def test_selection_and_balanced_shards(monkeypatch, tmp_path, capsys):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    build_plan = runner.build_plan
    units = [dict(id=name, seconds=seconds) for name, seconds in (("a", 9), ("b", 8), ("c", 4), ("d", 3))]
    lanes, loads = runner.balance(list(reversed(units)), 2)
    assert [[u["id"] for u in lane] for lane in lanes] == [["a", "d"], ["b", "c"]]
    assert loads == [12, 12]
    assert runner.select_units(units, ["d", "b", "d"]) == [units[3], units[1]]
    assert runner.balance(units, 6)[1] == [9, 8, 4, 3, 0, 0]
    assert runner.matches("tests/a/test_b.py::TestC::test_d[x]", "tests/a")
    assert runner.matches("tests/a/test_b.py::test_d[x]", "tests/a/test_b.py::test_d")
    assert not runner.matches("tests/ab/test_b.py::test_d", "tests/a")
    with pytest.raises(KeyError):
        runner.select_units(units, ["missing"])

    # The live manifest may declare no known failures; inject one so the known-only path stays covered.
    fixture = ('[[known_failures]]\nselector = "tests/sample/test_known.py"\nreason = "fixture"\n'
               'breaking_commit = "unknown"\nowner = "test"\n\n')
    text = runner.MANIFEST.read_text()
    text = (text.replace("known_failures = []\n", fixture) if "known_failures = []\n" in text
            else text.replace("[[known_failures]]\n", fixture + "[[known_failures]]\n", 1))
    # Merge declarations are exercised separately; this synthetic history has none.
    text = text.replace("require_merge_declarations = true", "require_merge_declarations = false")
    # Mover annotations change every batch; inject the two shapes these assertions exercise.
    text += (
        '\n[[expected_movers]]\ncommit = "fixture"\nrow = "directivity_two_dipoles"\n'
        'expected_change = "fixture row mover"\n'
        '\n[[expected_movers]]\ncommit = "fixture"\nselector = "tests"\n'
        'failure_contains = ["pytest.raises(NotImplementedError", '
        '"ValueError: normalize_source requires a single incident source spectrum"]\n'
        'pins = []\nexpected_change = "fixture exception mover"\n'
    )
    fixture_manifest = tmp_path / "integration_suites.toml"
    fixture_manifest.write_text(text)
    monkeypatch.setattr(runner, "MANIFEST", fixture_manifest)
    manifest = tomllib.loads(text)
    # Exercise the injected reference-only policy independently of live batches
    # that explicitly declare metric-policy changes for these same rows.
    manifest["expected_movers"] = [entry for entry in manifest.get("expected_movers", []) if entry["commit"] == "fixture"]
    components = manifest["components"]
    files = [p.relative_to(runner.ROOT).as_posix() for p in (runner.ROOT / "tests").rglob("*.py")
             if p.name.startswith("test_") or p.name.endswith("_test.py")]
    owners = runner.test_ownership(files, components)
    assert len(owners) == len(files)
    assert owners["tests/materials/compiler/test_local_material_cache.py"] == "compiler"
    with pytest.raises(RuntimeError, match="tests/unowned/test_new.py: UNOWNED"):
        runner.test_ownership(["tests/unowned/test_new.py"], components)
    with pytest.raises(RuntimeError, match="exactly one"):
        runner.test_ownership(["tests/rf/test_new.py"], {"a": components["rf"], "b": components["rf"]})
    assert runner.touched_components(["witwin/maxwell/fdtd/runtime/stepping.py", "witwin/maxwell/compiler/structures.py"], components) == ["fdtd", "compiler"]
    assert runner.touched_components(["witwin/maxwell/fdtd/ports.py", "witwin/maxwell/compiler/ports.py"], components) == ["fdtd", "compiler", "rf"]
    assert runner.touched_components(["tests/rf/network/test_network_conservation.py"], components) == ["rf"]
    assert runner.touched_components(["docs/plans/integration_batch.md"], components) == []
    for changed in ("witwin/maxwell/scene.py", "witwin/maxwell/new_module.py", "witwin/core/geometry.py", "witwin-core", "pyproject.toml"):
        assert runner.touched_components([changed], components) == list(components)
    monkeypatch.setattr(runner, "collect", lambda *args: dict(exit_code=2, errors=["collection sentinel"]))
    with pytest.raises(RuntimeError, match="collection sentinel"):
        runner.build_plan(runner.ROOT, "HEAD", manifest, runner.ROOT)

    schedule = [dict(id=name, seconds=seconds, suites=dict.fromkeys(suites)) for name, seconds, suites in (
        ("long", 100, ["pytest-fdtd"]), ("short", 1, ["pytest-compiler"]),
        ("benchmark", 50, ["benchmark"]), ("thin", 10, ["thin-sheet-112"]),
        ("smoke", 2, ["mom-public-smoke"]), ("pcb", 5, ["pcb-44"]),
        ("baseline", 8, ["fdtd-baselines"]), ("resident", 3, ["resident-bitid", "pcb-44"]))]
    order = ["resident", "baseline", "pcb", "smoke", "thin", "benchmark", "long", "short"]
    assert [u["id"] for u in sorted(schedule, key=runner.unit_order)] == order
    lane = tmp_path / "lane.lock"
    exclusive = tmp_path / "exclusive.lock"
    assert runner.unit_lock("benchmark", lane, [("bench*", lane), ("benchmark", exclusive)]) == exclusive
    assert runner.unit_lock("resident", lane, [("benchmark", exclusive)]) == lane
    assert runner.unit_lock("resident", None, []) is None
    monkeypatch.setattr(runner, "git", lambda root, *args: "fixture" if args[0] == "rev-parse" else "")
    monkeypatch.setattr(runner, "checkout", lambda sha: tmp_path)
    monkeypatch.setattr(runner, "build_plan", lambda *args: dict(units=schedule, not_run=[], disabled_suites=[], suites=[]))
    for name, policy, failing, expected in (("default", None, "resident", order[:1]),
                                          ("continue", False, "resident", order),
                                          ("phase2", None, "smoke", order),
                                          ("stop", True, "smoke", order[:4]),
                                          ("known", None, "resident", order)):
        calls = []

        def execute(source, sha, unit, directory, lock):
            calls.append((unit["id"], lock))
            status = "FAIL" if unit["id"] == failing else "PASS"
            nodeid = manifest["known_failures"][0]["selector"] + "::test_pin" if name == "known" else unit["id"]
            return dict(status=status, seconds=1, cases=[dict(status=status, nodeid=nodeid)])

        monkeypatch.setattr(runner, "execute", execute)
        args = SimpleNamespace(commit="HEAD", shard="0/1", since=None, out=str(tmp_path / name),
                               in_place=False, changed_files=None,
                               lock=str(lane), lock_for=[f"benchmark={exclusive}"], stop_on_fail=policy, no_start_after=[],
                               tier="integration", two_gpu=False, exclude_component=[], label=None, units_from=None, claim_dir=None, min_free_mib=0)
        assert runner.run_main(args) == 1
        assert [unit for unit, _ in calls] == expected
        assert all(lock == (exclusive if unit == "benchmark" else lane) for unit, lock in calls)
        progress = (tmp_path / name / "progress.log").read_text().splitlines()
        assert len(progress) == 2 * len(expected)
        assert "phase=1 unit=resident START pass=0 fail=0 elapsed=0.000s" in progress[0]
        assert "END status=FAIL" in progress[2 * expected.index(failing) + 1]
        assert all(line.split()[0].endswith("Z") for line in progress)

    known = manifest["known_failures"][0]["selector"] + "::test_pin"
    assert runner.known_only(dict(cases=[dict(status="FAIL", nodeid=known)]), manifest)
    assert not runner.known_only(dict(cases=[dict(status="FAIL", nodeid=known), dict(status="FAIL", nodeid="new")]), manifest)
    assert runner.failure_category("benchmark:directivity_two_dipoles[1e9 Hz]", manifest)[0] == "EXPECTED"
    # A scenario command crash is never absorbed by a row mover.
    assert runner.failure_category("benchmark:directivity_two_dipoles", manifest, "Command exit 1; see x.log") == ("NEW", None)
    for protected in ("benchmark:modulated_slab", "benchmark:modulated_slab_phase", "tests/fdtd/test_bit_identity_baseline.py::test_source_fields_match_pre_waveform_baseline", "tests/boundaries/absorbers/test_boundary_bit_identity.py::test_pin"):
        assert runner.failure_category(protected, manifest) == ("NEW", None)
    item = SimpleNamespace(fixturenames=["cuda_p2p_devices"])
    assert runner.two_device_reason(item) == "fixture cuda_p2p_devices"
    item = SimpleNamespace(fixturenames=[], get_closest_marker=lambda name: True)
    assert runner.two_device_reason(item) == "marker nccl"
    item = SimpleNamespace(fixturenames=[], get_closest_marker=lambda name: None, iter_markers=lambda name: (),
                           callspec=SimpleNamespace(params={"devices": ("cuda:0", "cuda:0")}))
    assert runner.two_device_reason(item) is None
    item.callspec.params["devices"] = ("cuda:0", "cuda:1")
    assert runner.two_device_reason(item).startswith("devices parameter:")

    window = runner.parse_window("2026-10-31T10:00/2026-10-31T12:15")
    assert window[0].astimezone(runner.timezone.utc).hour == 17
    before = window[0] - runner.timedelta(seconds=100)
    assert runner.deferred_until(before, 60, [window]) is None
    assert runner.deferred_until(before, 70, [window]) == window[1]
    assert runner.deferred_until(window[1], 60, [window]) is None
    clock = [before]
    locked = [False]
    acquisitions = []
    sleeps = []
    deferrals = []
    real_flock = runner.fcntl.flock

    class Clock(runner.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz)

    def flock(handle, operation):
        real_flock(handle, operation)
        locked[0] = operation != runner.fcntl.LOCK_UN
        if locked[0]:
            acquisitions.append(clock[0])
            if len(acquisitions) == 1:
                clock[0] += runner.timedelta(seconds=20)

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += runner.timedelta(seconds=seconds)

    def defer(until):
        assert not locked[0]
        deferrals.append(until)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "datetime", Clock)
        patch.setattr(runner.time, "sleep", sleep)
        patch.setattr(runner.fcntl, "flock", flock)
        with runner.unit_slot(lane, 60, [window], defer):
            assert locked[0] and clock[0] == window[1]
    assert deferrals == [window[1]] and len(acquisitions) == 2
    assert max(sleeps) <= 60 and not locked[0]

    source = tmp_path / "selection"
    tests = source / "tests/sample"
    tests.mkdir(parents=True)
    (tests / "test_mixed.py").write_text("")
    (tests / "test_two.py").write_text("")
    single = "tests/sample/test_mixed.py::test_one"
    blocked = ["tests/sample/test_mixed.py::test_two", "tests/sample/test_two.py::test_two"]
    data = dict(manifest, components={"sample": dict(sources=[], tests=["tests/sample"])}, suites=[])
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=[single, *blocked], exit_code=0, errors=[],
                                                           two_device=dict.fromkeys(blocked, "fixture cuda_p2p_devices")))
    plan = build_plan(source, "fixture", data, tmp_path)
    assert plan["counts"]["pytest-sample"] == 3 and len(plan["not_run"]) == 2
    assert len(plan["units"]) == 1 and plan["units"][0]["selectors"] == [single]

    slot = dict(id="two-device-slot", status="READY", kind="pytest", requires_gpus=2,
                selectors=["tests/sample/test_mixed.py"], classification="TOLERANCE", estimated_seconds=60)
    full = dict(id="two-device-full", status="READY", kind="pytest", tier="milestone", requires_gpus=2,
                selectors=["tests"], two_device_only=True, classification="MIXED", estimated_seconds=120)
    data["suites"] = [slot, full]
    one = build_plan(source, "fixture", data, tmp_path)
    assert one["disabled_suites"] == ["two-device-slot"]
    assert "two-device-full" not in one["counts"] and len(one["not_run"]) == 2
    two = build_plan(source, "fixture", data, tmp_path, two_gpu=True)
    assert [n["nodeid"] for n in two["not_run"]] == blocked[1:]
    assert two["disabled_suites"] == [] and two["counts"]["two-device-slot"] == 2
    milestone = build_plan(source, "fixture", data, tmp_path, tier="milestone", two_gpu=True)
    assert milestone["not_run"] == [] and milestone["counts"]["two-device-full"] == 2
    with monkeypatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "1")
        with pytest.raises(AssertionError, match="two distinct"):
            runner.require_two_gpu(True)
        patch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
        with pytest.raises(AssertionError, match="two distinct"):
            runner.require_two_gpu(True)
        patch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        runner.require_two_gpu(True)
    guard = "tests/sample/test_exception_guard.py::test_source_normalization_uses_common_spectrum"
    details = "with pytest.raises(NotImplementedError):\nValueError: normalize_source requires a single incident source spectrum"
    assert runner.failure_category(guard, manifest, details)[0] == "EXPECTED"
    assert runner.failure_category(guard, manifest, details.replace("ValueError:", "RuntimeError:"))[0] == "NEW"
    assert runner.failure_category(guard, manifest, "ValueError: unexpected numerical result")[0] == "NEW"

    mom = source / "tests/mom/test_smoke.py"
    mom.parent.mkdir()
    mom.write_text("")
    smoke = "tests/mom/test_smoke.py::test_public"
    other = "tests/mom/test_smoke.py::test_other"
    data["components"]["mom"] = dict(sources=["witwin/maxwell/mom/*"], tests=["tests/mom"])
    data["suites"] = [dict(id="mom-public-smoke", status="READY", kind="pytest", selectors=[smoke],
                           classification="TOLERANCE", estimated_seconds=10)]
    with monkeypatch.context() as patch:
        patch.setattr(runner, "build_plan", build_plan)
        patch.setattr(runner, "collect", lambda *args: dict(nodes=[single, smoke, other], exit_code=0,
                                                          errors=[], two_device={}, benchmark_scenarios=[]))
        patch.setattr(runner, "git", lambda root, *args: "fixture" if args[0] == "rev-parse" else "witwin/maxwell/scene.py")
        for since in (None, "base"):
            excluded = build_plan(source, "fixture", data, tmp_path, since=since, exclude_components=["mom", "mom"])
            assert excluded["selected_components"] == ["sample"] and excluded["excluded_components"] == ["mom"]
            assert "pytest-mom" not in excluded["counts"] and excluded["counts"]["mom-public-smoke"] == 1
            assert [u["selectors"] for u in excluded["units"] if u["component"] == "mom"] == [[smoke]]
        both = build_plan(source, "fixture", data, tmp_path, exclude_components=["mom", "sample"])
        assert both["selected_components"] == [] and len(both["units"]) == 1
        with pytest.raises(ValueError, match="Unknown components: missing"):
            build_plan(source, "fixture", data, tmp_path, exclude_components=["missing"])
        patch.setattr(runner, "checkout", lambda sha: source)
        patch.setattr(runner, "MANIFEST", tmp_path / "manifest.toml")
        runner.MANIFEST.write_text("")
        patch.setattr(runner.tomllib, "loads", lambda text: data)
        assert runner.plan_main(SimpleNamespace(commit="fixture", since="base", tier="integration",
                                               in_place=False, changed_files=None,
                                               two_gpu=False, exclude_component=["mom"], shards=1)) == 0
        assert "mom | EXCLUDED by operator |" in capsys.readouterr().out
        excluded.update(shard=0, shards=1, pin_changes=[], effective_pins={})
        output = tmp_path / "excluded-report"
        output.mkdir()
        runner.write_json(output / "plan.json", excluded)
        for k, unit in enumerate(excluded["units"]):
            runner.write_json(output / f"unit-{k}.json", dict(unit=unit, commit="fixture", junit="unused",
                                                           returncode=0, status="PASS", cases=[]))
        patch.setattr(runner, "junit_cases", lambda *args: [])
        assert runner.report_main(SimpleNamespace(directories=[str(output)])) == 1
        report = json.loads((output / "integration-report.json").read_text())
        statuses = {s["id"]: s["status"] for s in report["suites"]}
        assert statuses["pytest-mom"] == "EXCLUDED by operator" and statuses["mom-public-smoke"] == "PASS"
        assert report["excluded_components"] == ["mom"]
        assert "| pytest-mom | EXCLUDED by operator |" in (output / "integration-report.md").read_text()


@pytest.mark.parametrize("node_only", [False, True])
def test_milestone_selections_are_excluded_from_component_units(monkeypatch, tmp_path, capsys, node_only):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    files = ["tests/sample/test_regular.py", "tests/sample/test_long.py"]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    nodes = [f"{name}::test_{case}" for name in files for case in ("one", "two")]
    selector = nodes[2] if node_only else files[1]
    manifest = dict(pytest_seconds_per_test=1, known_failures=[], expected_movers=[],
                    components={"sample": dict(sources=[], tests=["tests/sample"])},
                    suites=[dict(id="long-horizon", tier="milestone", kind="pytest", status="READY",
                                 selectors=[selector], classification="TOLERANCE", estimated_seconds=10)])
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=nodes, exit_code=0, errors=[], two_device={}))
    assert runner.test_ownership(files, manifest["components"]) == dict.fromkeys(files, "sample")
    integration = runner.build_plan(tmp_path, "fixture", manifest, tmp_path)
    excluded = {nodes[2]} if node_only else set(nodes[2:])
    assert {n for u in integration["units"] for n in u["nodes"]} == set(nodes) - excluded
    assert integration["counts"]["pytest-sample"] == len(nodes) - len(excluded)
    assert integration["milestone_only"] == [selector]
    milestone = runner.build_plan(tmp_path, "fixture", manifest, tmp_path, tier="milestone")
    assert len(milestone["units"]) == 2
    assert sorted(n for u in milestone["units"] for n in u["nodes"]) == sorted(nodes)
    long_unit = next(u for u in milestone["units"] if u["id"] == f"pytest:{files[1]}")
    assert long_unit["component"] == "sample"
    assert set(long_unit["suites"]["long-horizon"]) == excluded
    assert milestone["milestone_only"] == []
    monkeypatch.setattr(runner, "source_tree", lambda args: ("fixture", tmp_path))
    monkeypatch.setattr(runner, "build_plan", lambda *args: integration)
    assert runner.plan_main(SimpleNamespace(two_gpu=False, since=None, tier="integration",
                                           exclude_component=[], changed_files=None, shards=1)) == 0
    assert f"milestone-only: {selector}" in capsys.readouterr().out


def test_in_place_missing_docs_are_infra(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.docs_paths(b'"docs/a.json"; root / "docs" / "b" / "c.md"; '
                            b'os.path.join(root, "docs", "d", "e.json"); docs/a.json. docs/b/') == {
                                 "docs/a.json", "docs/b/c.md", "docs/d/e.json", "docs/b"}
    assert runner.missing_unit_docs(runner.ROOT, dict(kind="pytest", selectors=[
        "tests/materials/compiler/test_subpixel_spec.py"]), [],
        tomllib.loads(runner.MANIFEST.read_text())["docs_text_mentions"]) == []
    monkeypatch.chdir(tmp_path)
    tests = tmp_path / "tests/sample"
    tests.mkdir(parents=True)
    (tests / "test_missing.py").write_text('PATH = "docs/missing.json"\n')
    (tests / "test_excluded.py").write_text('PATH = root / "docs" / "blocked"\n')
    (tests / "test_staged.py").write_text('"""Receipt: docs/staged.md."""\n'
                                         'PATH = join(root, "docs", "staged.md")\n')
    docs = tmp_path / "docs"
    (docs / "blocked").mkdir(parents=True)
    (docs / "blocked/file.pdf").write_text("excluded even if present")
    (docs / "staged.md").write_text("staged")
    docs_manifest = tmp_path / "docs-manifest.json"
    runner.write_json(docs_manifest, dict(files=[], excluded=[dict(path="docs/blocked/file.pdf",
                      tests=["tests/sample/test_excluded.py"])], unresolved=["docs/missing.json"]))
    monkeypatch.setenv("WITWIN_TEST_DOCS_MANIFEST", str(docs_manifest))
    manifest = tmp_path / "suites.toml"
    manifest.write_text('''base_commit = "base"
pytest_seconds_per_test = 1
pin_files = []
known_failures = []
expected_movers = []
[components.sample]
sources = []
tests = ["tests/sample"]
[[suites]]
id = "resident-bitid"
status = "READY"
kind = "pytest"
selectors = ["tests/sample"]
classification = "BITID"
estimated_seconds = 1
''')
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    nodes = [f"tests/sample/test_{name}.py::test_case" for name in ("missing", "excluded", "staged")]
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=nodes, exit_code=0, errors=[], two_device={}))
    sha = "a" * 40
    runner.write_json(tmp_path / ".wcloud-source.json", dict(git_head=sha))
    calls = []

    def child(command, **kwargs):
        selector = command[3]
        calls.append(selector)
        xml = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        xml.write_text(f'<testsuite><testcase classname="tests.sample.{Path(selector).stem}" '
                       'name="test_case"/></testsuite>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    monkeypatch.setattr(runner, "git", lambda *args: pytest.fail("in-place called git"))
    for policy in (None, True):
        calls.clear()
        args = SimpleNamespace(commit=sha, in_place=True, changed_files=None, since=None,
                               tier="integration", two_gpu=False, exclude_component=[], shard="0/1",
                               out=str(tmp_path / str(policy)), lock=None, lock_for=[],
                               stop_on_fail=policy, no_start_after=[],
                               label=None, units_from=None, claim_dir=None, min_free_mib=0)
        assert runner.run_main(args) == 1
        assert calls == ["tests/sample/test_staged.py"]
        assert runner.report_main(SimpleNamespace(directories=[args.out])) == 1
        report = json.loads((Path(args.out) / "integration-report.json").read_text())
        assert len(report["infra_units"]) == 2 and report["missing_units"] == []
        assert all(r["cases"] == [] and r["reason"].startswith("docs path not staged for cloud: ")
                   for r in report["infra_units"])
        assert all(not failures for failures in report["failures"].values())
        assert all(s["status"] == "INFRA" for s in report["suites"])
        markdown = (Path(args.out) / "integration-report.md").read_text()
        assert "phase 1 incomplete (INFRA)" in markdown.splitlines()[0]
        assert "## INFRA" in markdown and "docs/missing.json" in markdown and "docs/blocked" in markdown
        assert "END status=PASS pass=1 fail=0" in (Path(args.out) / "progress.log").read_text()
    calls.clear()
    monkeypatch.setattr(runner, "git", lambda *args: sha)
    monkeypatch.setattr(runner, "checkout", lambda *args: tmp_path)
    args.in_place = False
    args.out = str(tmp_path / "local")
    assert runner.run_main(args) == 0
    assert len(calls) == 3


def test_in_place_no_git_and_required_baselines(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "git", lambda *args: pytest.fail("in-place called git"))
    monkeypatch.setattr(runner, "checkout", lambda *args: pytest.fail("in-place created a worktree"))
    test_file = tmp_path / "tests/sample/test_case.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("BASE_REF = 'abcdef12'\n")
    changed = tmp_path / "changed-files.txt"
    changed.write_text("tests/sample/test_case.py\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text('''base_commit = "fixture"
pytest_seconds_per_test = 1
pin_files = ["tests/sample/test_case.py"]
known_failures = []
expected_movers = []
[components.sample]
sources = []
tests = ["tests/sample"]
[components.other]
sources = []
tests = ["tests/other"]
[[suites]]
id = "resident-bitid"
status = "READY"
kind = "pytest"
selectors = ["tests/sample/test_case.py::test_pin"]
classification = "BITID"
estimated_seconds = 1
''')
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    nodes = ["tests/sample/test_case.py::test_pin", "tests/sample/test_case.py::test_other"]
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=nodes, exit_code=0, errors=[], two_device={}))
    monkeypatch.setenv("WITWIN_BASELINE_TREES", "/uploaded/baseline trees")
    monkeypatch.setenv("WITWIN_REQUIRE_BASELINES", "1")

    def child(command, *, cwd, env, **kwargs):
        assert cwd == tmp_path
        assert env["WITWIN_BASELINE_TREES"] == "/uploaded/baseline trees"
        assert env["WITWIN_REQUIRE_BASELINES"] == "1"
        xml = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        xml.write_text('<testsuite><testcase classname="tests.sample.test_case" name="test_pin">'
                       '<skipped message="baseline source missing"/></testcase>'
                       '<testcase classname="tests.sample.test_case" name="test_other"/></testsuite>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    args = SimpleNamespace(commit="a" * 40, in_place=True, changed_files=changed, since=None,
                           tier="integration", two_gpu=False, exclude_component=[], shards=1,
                           shard="0/1", out=str(tmp_path / "results"), lock=None, lock_for=[],
                           stop_on_fail=None, no_start_after=[],
                           label=None, units_from=None, claim_dir=None, min_free_mib=0)
    with pytest.raises(FileNotFoundError):
        runner.source_tree(args)
    provenance = tmp_path / ".wcloud-source.json"
    provenance.write_text(json.dumps({"git_head": "b" * 40}))
    with pytest.raises(ValueError, match="does not match.*git_head"):
        runner.source_tree(args)
    provenance.write_text(json.dumps({"git_head": args.commit}))
    assert runner.plan_main(args) == 0
    assert runner.run_main(args) == 1
    plan = json.loads((tmp_path / "results/shard-0/plan.json").read_text())
    assert plan["commit"] == args.commit and plan["selected_components"] == ["sample"]
    assert plan["changed_paths"] == ["tests/sample/test_case.py"]
    assert plan["pin_changes"] is None
    assert plan["effective_pins"] == {"tests/sample/test_case.py": {"BASE_REF": "abcdef12"}}
    unit = next((tmp_path / "results/shard-0").glob("unit-*.json"))
    result = json.loads(unit.read_text())
    assert result["status"] == "FAIL" and result["cases"][0]["status"] == "FAIL"
    assert "Missing baseline" in result["cases"][0]["details"]
    assert runner.junit_cases(Path(result["junit"]))[0]["status"] == "SKIP"
    monkeypatch.delenv("WITWIN_REQUIRE_BASELINES")
    assert runner.report_main(SimpleNamespace(directories=[args.out])) == 1
    report = json.loads((tmp_path / "results/integration-report.json").read_text())
    assert all(s["status"] == "FAIL" for s in report["suites"])
    assert "Missing baseline" in report["failures"]["NEW"][0]["details"]
    for action in ("run", "plan"):
        monkeypatch.setattr(runner.sys, "argv", [str(path), action, "--commit", args.commit,
                                               "--in-place", "--since", "base"] +
                            (["--shard", "0/1", "--out", args.out] if action == "run" else []))
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 2


def test_junit_strict_xfail_is_pass_equivalent_and_real_skips_stay_skip(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.a.test_x" name="known"><skipped type="pytest.xfail">r</skipped></testcase>'
        '<testcase classname="tests.a.test_x" name="gone"><skipped type="pytest.skip">r</skipped></testcase></testsuite>')
    assert [c["status"] for c in runner.junit_cases(report)] == ["XFAIL", "SKIP"]
    assert [c["status"] for c in runner.junit_cases(report, require_baselines=True)] == ["XFAIL", "FAIL"]


def test_benchmark_scenario_quarantine_reports_skip_with_reason(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    aggregate = ["held", *("-" for _ in range(17))]
    aggregate[4], aggregate[6], aggregate[7], aggregate[9], aggregate[11] = ".01", ".01", "1", ".01", "7"
    results = tmp_path / "RESULTS.md"
    results.write_text("\n".join("| " + " | ".join(row) + " |"
                                 for row in (aggregate, ["held", "1e9", ".01", "-", ".01", "1", "-"])))
    unit = dict(nodes=["benchmark:held"], metric_limits=dict(field_l2=.1, field_linf=.1, field_corr_min=.99, flux_error=.05),
                scenario_quarantine={"held": "reference is not stationary"})
    cases = runner.benchmark_cases(results, unit)
    assert [(case["nodeid"], case["status"]) for case in cases] == [
        ("benchmark:held", "SKIP"), ("benchmark:held[1e9 Hz]", "SKIP")]
    assert all(case["details"] == "Quarantined reference: reference is not stationary" for case in cases)
    for quarantine, message in (({"unknown": "why"}, "Unknown benchmark scenario"),
                                ({"held": " "}, "needs a reason"),
                                ({"held": "why"}, "has metric limits")):
        suite = dict(scenario_quarantine=quarantine,
                     scenario_metric_limits={"held": dict(field_l2=.2)} if message == "has metric limits" else {})
        with pytest.raises(ValueError, match=message):
            runner.validate_scenario_metric_limits(suite, ["held"], [])


def test_quarantined_scenario_is_not_executed_and_stays_incomplete(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    marker = tmp_path / "ran"
    unit = dict(id="benchmark:held", kind="command", report_format="benchmark", nodes=["benchmark:held"],
                argv=["-c", f"open({str(marker)!r}, 'w'); raise SystemExit(3)"],
                metric_limits=dict(field_l2=.1, field_linf=.1, field_corr_min=.99, flux_error=.05),
                scenario_quarantine={"held": "refused until r2"})
    result = runner.execute(tmp_path, "sha", unit, tmp_path, None)
    assert not marker.exists()
    assert (result["status"], result["returncode"], [(c["nodeid"], c["status"], c["details"]) for c in result["cases"]]) == (
        "INCOMPLETE", 0, [("benchmark:held", "SKIP", "Quarantined reference: refused until r2")])
    # Without the quarantine the same crash is a FAIL command case.
    unit.pop("scenario_quarantine")
    result = runner.execute(tmp_path, "sha", unit, tmp_path, None)
    assert marker.exists() and result["status"] == "FAIL" and result["cases"][-1]["details"].startswith("Command exit 3")


def test_live_manifest_keeps_investigation_rows_out_of_movers():
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = tomllib.loads(runner.MANIFEST.read_text())
    suite = next(s for s in manifest["suites"] if s.get("report_format") == "benchmark")
    investigation = suite["scenario_investigation"]
    assert {"modulated_slab", "modulated_slab_phase"} <= set(investigation)
    # The live manifest: no row mover and no metric override may cover a class (c) row.
    assert not investigation.keys() & {e["row"] for e in manifest.get("expected_movers", []) if "row" in e}
    assert not investigation.keys() & suite.get("scenario_metric_limits", {}).keys()
    names = sorted(investigation)
    runner.validate_scenario_metric_limits(dict(scenario_investigation=investigation), names, manifest.get("expected_movers", []))
    with pytest.raises(ValueError, match="covered by a row mover: kerr_slab"):
        runner.validate_scenario_metric_limits(dict(scenario_investigation=investigation), names,
                                               [dict(commit="x", row="kerr_slab")])
    with pytest.raises(ValueError, match="has metric limits: kerr_slab"):
        runner.validate_scenario_metric_limits(dict(scenario_investigation=investigation,
                                                    scenario_metric_limits={"kerr_slab": dict(field_l2=.2)}), names, [])


def test_every_bit_identity_file_is_pinned_and_in_a_bitid_suite(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = tomllib.loads(runner.MANIFEST.read_text())
    assert runner.uncovered_bit_identity_files(runner.ROOT, manifest) == []
    extra = tmp_path / "tests/new/test_extra_bit_identity.py"
    extra.parent.mkdir(parents=True)
    extra.write_text("")
    assert runner.uncovered_bit_identity_files(tmp_path, manifest) == ["tests/new/test_extra_bit_identity.py"]



def test_known_open_rows_cover_only_their_signature():
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = dict(known_failures=[dict(selector="tests/a.py::test_x", failure_contains=["Tensor-likes are not equal"])])
    assert runner.failure_category("tests/a.py::test_x[p0]", data, "AssertionError: Tensor-likes are not equal!")[0] == "KNOWN"
    assert runner.failure_category("tests/a.py::test_x[p0]", data, "RuntimeError: CUDA error") == ("NEW", None)
    failing = dict(cases=[dict(status="FAIL", nodeid="tests/a.py::test_x", details="Tensor-likes are not equal", assertion_message="")])
    assert runner.known_only(failing, data)
    failing["cases"][0]["details"] = "other"
    assert not runner.known_only(failing, data)


def test_merges_without_declarations_are_listed(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import subprocess
    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "README").write_text("0"); git("add", "-A"); git("commit", "-qm", "base"); base = git("rev-parse", "HEAD")
    (tmp_path / "witwin").mkdir(); (tmp_path / "witwin/a.py").write_text("1"); git("add", "-A"); git("commit", "-qm", "solver change")
    solver = git("rev-parse", "HEAD")
    (tmp_path / "docs").mkdir(); (tmp_path / "docs/x.md").write_text("2"); git("add", "-A"); git("commit", "-qm", "docs only")
    head = git("rev-parse", "HEAD")
    assert [m.split()[0] for m in runner.undeclared_merges(tmp_path, head, base, {})] == [solver[:8]]
    declared = dict(merge_declarations=[dict(commit=solver[:8], pytest="none", benchmark="none")])
    assert runner.undeclared_merges(tmp_path, head, base, declared) == []
    git("checkout", "-qb", "feature", base); (tmp_path / "witwin/b.py").parent.mkdir(exist_ok=True); (tmp_path / "witwin/b.py").write_text("3")
    git("add", "-A"); git("commit", "-qm", "branch work"); git("checkout", "-q", "-"); git("merge", "-q", "--no-ff", "feature", "-m", "Merge owner/feature: work")
    merged = git("rev-parse", "HEAD")
    assert [m.split()[0] for m in runner.undeclared_merges(tmp_path, merged, base, declared)] == [merged[:8]]
    by_branch = dict(merge_declarations=[*declared["merge_declarations"], dict(branch="owner/feature", pytest="none", benchmark="none")])
    assert runner.undeclared_merges(tmp_path, merged, base, by_branch) == []
    collision = dict(merge_declarations=[*declared["merge_declarations"], dict(branch="owner/feat", pytest="none", benchmark="none")])
    assert [m.split()[0] for m in runner.undeclared_merges(tmp_path, merged, base, collision)] == [merged[:8]]
    git("checkout", "-qb", "tagged", base); (tmp_path / "witwin").mkdir(exist_ok=True); (tmp_path / "witwin/c.py").write_text("4")
    git("add", "-A"); git("commit", "-qm", "tagged work"); git("checkout", "-q", "-")
    git("merge", "-q", "--no-ff", "tagged", "-m", "Merge owner/tagged (F1-a): work")
    tagged = git("rev-parse", "HEAD")
    with_tagged = dict(merge_declarations=[*by_branch["merge_declarations"], dict(branch="owner/tagged", pytest="none", benchmark="none")])
    assert runner.undeclared_merges(tmp_path, tagged, base, with_tagged) == []
    prefix = dict(merge_declarations=[*by_branch["merge_declarations"], dict(branch="owner/tag", pytest="none", benchmark="none")])
    assert [m.split()[0] for m in runner.undeclared_merges(tmp_path, tagged, base, prefix)] == [tagged[:8]]
    for entry in (dict(commit=solver[:8], branch="owner/feature", pytest="none", benchmark="none"), dict(pytest="none", benchmark="none")):
        with pytest.raises(ValueError, match="exactly one of commit or branch"):
            runner.undeclared_merges(tmp_path, merged, base, dict(merge_declarations=[entry]))
    with pytest.raises(ValueError, match="must be 'none' or a non-empty list"):
        runner.undeclared_merges(tmp_path, head, base, dict(merge_declarations=[dict(commit=solver[:8], pytest=[], benchmark="none")]))


def test_pin_file_added_after_base_has_no_previous_pins(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import subprocess
    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "README").write_text("0"); git("add", "-A"); git("commit", "-qm", "base"); base = git("rev-parse", "HEAD")
    (tmp_path / "test_new_bit_identity.py").write_text('BASE = "abc1234"\n'); git("add", "-A"); git("commit", "-qm", "add pin")
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest = dict(base_commit=base, pin_files=["test_new_bit_identity.py"])
    assert runner.pin_changes(head, manifest) == [dict(file="test_new_bit_identity.py", pin="BASE", old=None,
                                                       new="abc1234", causing_commit=head)]


@pytest.mark.parametrize("mover", [
    "tests/a.py::test_field",
    dict(node="tests/a.py::test_field", tier="T0"),
    dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=1e-6),
    dict(node="tests/a.py::test_field", tier="T2", approval="user 2026-11-03"),
    dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=1e-6, intended=True, approval="user 2026-11-03"),
    dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=1e-6, discrete=True, approval="user 2026-11-03"),
])
def test_tiered_merge_declarations_valid(monkeypatch, tmp_path, mover):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "git", lambda *args: "")
    entry = dict(branch="infra/example", pytest=[mover], benchmark="none")
    assert runner.undeclared_merges(tmp_path, "head", "base", dict(merge_declarations=[entry])) == []


@pytest.mark.parametrize("mover,message", [
    (dict(node="tests/a.py::test_field", tier="T1"), "max_rel_diff"),
    (dict(node="tests/a.py::test_field", tier="T1", max_rel_diff="small"), "max_rel_diff"),
    (dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=-1e-6), "max_rel_diff"),
    (dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=float("inf")), "max_rel_diff"),
    (dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=1e-9, intended=True), "T2 requires approval"),
    (dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=1e-9, discrete=True), "T2 requires approval"),
    (dict(node="tests/a.py::test_field", tier="T1", max_rel_diff=1e-9, intended="yes"), "boolean"),
    (dict(node="tests/a.py::test_field", tier="T2"), "approval"),
    (dict(node="tests/a.py::test_field", tier="T2", approval=" "), "approval"),
    (dict(node="tests/a.py::test_field", tier="T3"), "tier"),
    (dict(tier="T1", max_rel_diff=1e-6), "node string"),
    ("", "node string"),
])
def test_tiered_merge_declarations_invalid(monkeypatch, tmp_path, mover, message):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "git", lambda *args: "")
    entry = dict(branch="infra/example", pytest=[mover], benchmark="none")
    with pytest.raises(ValueError, match=rf"merge_declarations infra/example pytest\[0\].*{message}"):
        runner.undeclared_merges(tmp_path, "head", "base", dict(merge_declarations=[entry]))


@pytest.mark.parametrize("tier,rc,flag,listed", [
    ("T1", 0, None, True), ("T2", 1, None, True),
    ("T1", 0, "intended", True), ("T1", 0, "discrete", True),
    ("T1", 0, None, False), ("T2", 1, None, False),
])
def test_pin_tier_shards_execute_and_report(monkeypatch, tmp_path, tier, rc, flag, listed):
    import torch
    from tests.support.numerical_equivalence import record_pin_comparison

    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    units = []
    for index in range(2):
        nodes = [f"tests/sample/test_pin{index}.py::test_exact", f"tests/sample/test_pin{index}.py::test_mover"]
        units.append(dict(id=f"pytest:pin{index}", kind="pytest", selectors=[f"tests/sample/test_pin{index}.py"],
                          nodes=nodes, suites={"resident-bitid": nodes}, classification="BITID"))
    plan = dict(commit="sha", since=None, manifest_sha256="manifest", shards=2, tier="integration", two_gpu=False,
                excluded_components=[], changed_paths=None, selected_components=[], not_run=[], disabled_suites=[],
                units=units, collection=dict(exit_code=0, errors=[]), effective_pins={}, pin_changes=[],
                benchmark_cache_sha256="cache", benchmark_cache_entries=0, benchmark_cache_bytes=0,
                suites=[dict(id="resident-bitid", status="READY", classification="BITID")],
                known_failures=[], expected_movers=[])
    effective_tier = "T2" if flag else tier
    effective_rc = int(effective_tier == "T2")
    declarations = [dict(node=unit["nodes"][1], tier=tier,
                         **(dict(max_rel_diff=1e-6) if tier == "T1" else dict(approval="fixture approval")))
                    for unit in units]
    if flag:
        for declaration in declarations:
            declaration.update({flag: True}, approval="fixture approval")
    plan["merge_declarations"] = [dict(branch="infra/fixture", benchmark="none", pytest=declarations)] if listed else []
    logs = []

    def child(command, *, env, **kwargs):
        index = int(Path(command[3]).stem[-1])
        log = Path(env["WITWIN_PIN_TIER_LOG"])
        logs.append(log)
        assert log.read_text() == ""  # A retry cannot inherit the previous attempt's receipts.
        with monkeypatch.context() as patch:
            patch.setenv("WITWIN_PIN_TIER_LOG", str(log))
            try:
                record_pin_comparison(torch.ones(1), torch.ones(1) + (1e-6 if tier == "T1" else 0.1),
                                      nodeid=units[index]["nodes"][1], name="field", level="run")
            except AssertionError as exc:
                assert tier == "T2"
                message = str(exc)
        failure = f'<failure message="{message}">{message}</failure>' if rc else ""
        xml = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        xml.write_text(f'<testsuite><testcase classname="tests.sample.test_pin{index}" name="test_exact"/>'
                       f'<testcase classname="tests.sample.test_pin{index}" name="test_mover">{failure}</testcase></testsuite>')
        return SimpleNamespace(returncode=rc)

    monkeypatch.setattr(runner.subprocess, "run", child)
    for index, unit in enumerate(units):
        directory = tmp_path / f"shard-{index}"
        directory.mkdir()
        runner.write_json(directory / "plan.json", dict(plan, shard=index))
        for _ in range(2):
            result = runner.execute(tmp_path, "sha", unit, directory, None)
            assert result["returncode"] == rc
            assert result["status"] == ("PASS" if rc == 0 else "FAIL")
            assert len(result["pin_tiers"]) == 1 and result["pin_tiers"][0]["tier"] == tier
    assert len(set(logs)) == 2
    assert runner.report_main(SimpleNamespace(directories=[str(tmp_path)])) == int(bool(effective_rc or not listed))
    report = json.loads((tmp_path / "integration-report.json").read_text())
    assert len(report["failures"]["NEW"]) == 2 * effective_rc
    assert report["suites"][0]["status"] == ("PASS" if effective_rc == 0 else "FAIL")
    for unit in units:
        row = report["bitid_tiers"][unit["id"]]
        assert row["counts"] == dict(T0=1, T1=1 - effective_rc, T2=effective_rc)
        assert len(row["t1_movers"]) == 1 - effective_rc
        if not effective_rc:
            assert row["t1_movers"][0]["max_rel"] == pytest.approx(1e-6, abs=1e-7)
        assert row["movers"][0]["name"] == "field"
        assert row["movers"][0]["tier"] == effective_tier
        assert row["movers"][0]["shape"] == [1]
    assert len(report["mover_declaration_errors"]) == (0 if listed else 2)
    if not listed:
        assert all(effective_tier in error for error in report["mover_declaration_errors"])
        if tier == "T1":
            assert all("must be listed" in error and "max_rel_diff" in error for error in report["mover_declaration_errors"])
    markdown = (tmp_path / "integration-report.md").read_text()
    assert "| Unit | T0 | T1 | T2 |" in markdown
    if not effective_rc:
        assert "max relative diff" in markdown and "test_mover" in markdown and "field" in markdown


def test_multibisect_classifies_truncated_and_timed_out_probes_as_inconclusive():
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_multibisect.py"
    spec = importlib.util.spec_from_file_location("integration_multibisect", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.classify(0, "........ [100%]\n8 passed in 231.72s (0:03:51)\n") == "PASS"
    assert tool.classify(1, "F [100%]\n=== FAILURES ===\n... \n=========== 1 failed in 53.47s ===========\n") == "FAIL"
    assert tool.classify(124, "..") == "INCONCLUSIVE"            # timeout, truncated log
    assert tool.classify(1, "FF") == "INCONCLUSIVE"              # failed tests but no summary line
    assert tool.classify(5, "no tests ran in 0.00s\n") == "INCONCLUSIVE"   # missing selector at this commit
    assert tool.classify(2, "=== 1 error in 0.50s ===\n") == "INCONCLUSIVE"  # collection/usage error



def test_file_wide_movers_are_rejected_and_live_movers_are_node_level():
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.broad_movers(tomllib.loads(runner.MANIFEST.read_text())) == []
    movers = dict(expected_movers=[dict(commit="c", selector="tests/a.py"),
                                   dict(commit="c", selector="tests/a.py::test_x"),
                                   dict(commit="c", selector="tests/b.py", failure_contains=["ValueError: x"]),
                                   dict(commit="c", row="planewave_vacuum")])
    assert runner.broad_movers(movers) == ["tests/a.py"]



def test_environment_keeps_operator_pythonpath_after_the_batch_tree(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setenv("PYTHONPATH", "/hooks")
    assert runner.environment(Path("/tree"))["PYTHONPATH"] == os.pathsep.join(["/tree", "/hooks"])
    monkeypatch.delenv("PYTHONPATH")
    assert runner.environment(Path("/tree"))["PYTHONPATH"] == "/tree"


@pytest.fixture
def tier_runner():
    path = Path(__file__).resolve().parents[3] / "scripts/dev/integration_batch.py"
    spec = importlib.util.spec_from_file_location("integration_batch", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_real_manifest_loads(tier_runner):
    manifest = tier_runner.load_manifest()
    assert manifest["suites"]
    assert dict(branch="infra/suites-toml-parse-error", pytest="none", benchmark="none") in manifest["merge_declarations"]
    assert dict(branch="infra/suites-manifest-structure", pytest="none", benchmark="none") in manifest["merge_declarations"]


@pytest.fixture
def manifest_root():
    return ('base_commit = "fixture"\npin_files = []\npytest_seconds_per_test = 1.0\n'
            'known_failures = []\nsuites = []\ncomponents = {}\n')


def test_manifest_header_declaration_swallows_root_keys(tier_runner, tmp_path):
    text, count = re.subn(r'^require_merge_declarations =',
                          '[[merge_declarations]]\nbranch = "x/y"\npytest = "none"\n'
                          'benchmark = "none"\nrequire_merge_declarations =',
                          tier_runner.MANIFEST.read_text(), count=1, flags=re.MULTILINE)
    assert count == 1
    path = tmp_path / "integration_suites.toml"
    path.write_text(text)
    with pytest.raises(ValueError) as caught:
        tier_runner.load_manifest(path)
    message = str(caught.value)
    assert str(path) in message and "pin_files" in message
    assert "found inside [[merge_declarations]] branch=x/y" in message


@pytest.mark.parametrize("key", [
    "base_commit", "components", "known_failures", "pin_files", "pytest_seconds_per_test", "suites",
])
def test_manifest_requires_root_keys(tier_runner, tmp_path, manifest_root, key):
    text = re.sub(rf'^{key} = .*\n', '', manifest_root, flags=re.MULTILINE)
    path = tmp_path / "integration_suites.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=rf"missing required top-level key '{key}'"):
        tier_runner.load_manifest(path)


@pytest.mark.parametrize("table,identity", [
    ("merge_declarations", 'branch = "x/y"'),
    ("merge_declarations", 'commit = "abc123"'),
    ("known_failures", 'selector = "tests/a.py::test_known"'),
    ("expected_movers", 'commit = "abc123"'),
    ("suites", 'id = "smoke"'),
])
def test_manifest_unknown_entry_keys(tier_runner, tmp_path, manifest_root, table, identity):
    text = re.sub(rf'^{table} = .*\n', '', manifest_root, flags=re.MULTILINE)
    path = tmp_path / "integration_suites.toml"
    path.write_text(text + f'[[{table}]]\n{identity}\nstray_key = true\nanother_stray = 1\n')
    with pytest.raises(ValueError) as caught:
        tier_runner.load_manifest(path)
    message = str(caught.value)
    assert str(path) in message and f"[[{table}]]" in message
    key, value = identity.split(" = ")
    value = value.strip('"')
    assert f"{key}={value}" in message
    assert "stray_key" in message and "another_stray" in message


def test_manifest_defaulted_root_keys_are_optional(tier_runner, tmp_path, manifest_root):
    path = tmp_path / "integration_suites.toml"
    path.write_text(manifest_root)
    assert tier_runner.load_manifest(path) == tomllib.loads(manifest_root)


def test_manifest_unescaped_quote_identifies_merge_declaration(tier_runner, tmp_path):
    text, count = re.subn(r'^\[\[merge_declarations\]\]$',
                          '[[merge_declarations]]\nbranch = "x/y"\nnote = "unescaped "quote""\n',
                          tier_runner.MANIFEST.read_text(), count=1, flags=re.MULTILINE)
    assert count == 1
    path = tmp_path / "integration_suites.toml"
    path.write_text(text)
    with pytest.raises(ValueError) as caught:
        tier_runner.load_manifest(path)
    message = str(caught.value)
    assert str(path) in message
    assert "merge_declarations" in message and "x/y" in message
    assert isinstance(caught.value.__cause__, tomllib.TOMLDecodeError)


@pytest.mark.parametrize("table,identity", [
    ("merge_declarations", 'branch = "x/y"'),
    ("merge_declarations", 'commit = "abc123"'),
    ("known_failures", 'id = "known-regression"'),
    ("expected_movers", 'commit = "abc123"'),
    ("suites", 'id = "smoke"'),
    ("suites", ""),
])
def test_manifest_error_location_and_nearest_table(tier_runner, tmp_path, table, identity):
    path = tmp_path / "integration_suites.toml"
    path.write_text(f'[[merge_declarations]]\nbranch = "earlier/entry"\n'
                    f'# [[known_failures]]\n[[{table}]]\n{identity}\nbroken = @\nid = "later"\n')
    with pytest.raises(ValueError) as caught:
        tier_runner.load_manifest(path)
    message = str(caught.value)
    assert f"{path}: line 6, column 10" in message
    assert f"[[{table}]] at line 4" in message
    assert identity in message
    assert "earlier/entry" not in message and 'id = "later"' not in message


@pytest.mark.parametrize("text,location", [('broken = @', "line 1, column 10"),
                                           ('[[suites]]\nid = "smoke"\nbroken = [', "line 3, column 11")])
def test_manifest_error_without_table_or_at_eof(tier_runner, tmp_path, text, location):
    path = tmp_path / "integration_suites.toml"
    path.write_text(text)
    with pytest.raises(ValueError) as caught:
        tier_runner.load_manifest(path)
    assert f"{path}: {location}" in str(caught.value)
    assert ("top level" if text.startswith("broken") else 'id = "smoke"') in str(caught.value)


@pytest.mark.parametrize("tier,intended,discrete,effective", [
    ("T0", False, False, "T0"),
    ("T1", False, False, "T1"),
    ("T2", False, False, "T2"),
    ("T1", True, False, "T2"),
    ("T1", False, True, "T2"),
])
def test_plan_requires_each_mover_declaration(tier_runner, monkeypatch, tmp_path, tier, intended, discrete, effective):
    runner = tier_runner
    node = "tests/a.py::test_field"
    mover = dict(selector=node, tier=tier, max_rel_diff=1e-7, intended=intended, discrete=discrete)
    manifest = dict(components={}, suites=[], known_failures=[], expected_movers=[mover], pin_files=[])
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=[], exit_code=0, errors=[], two_device={}, benchmark_scenarios=[]))
    if effective != "T0":
        message = "T1.*must be listed.*max_rel_diff" if effective == "T1" else "undeclared T2"
        with pytest.raises(ValueError, match=message):
            runner.build_plan(tmp_path, "head", manifest, tmp_path)
        # A declaration for a different node cannot approve this mover.
        manifest["merge_declarations"] = [dict(branch="infra/fixture", pytest=["tests/a.py::test_other"], benchmark="none")]
        with pytest.raises(ValueError, match=message):
            runner.build_plan(tmp_path, "head", manifest, tmp_path)
    declaration = dict(node=node, tier=effective)
    if effective == "T1":
        declaration["max_rel_diff"] = 1e-7
    elif effective == "T2":
        declaration["approval"] = "fixture user approval"
    manifest["merge_declarations"] = [dict(branch="infra/fixture", pytest=[declaration], benchmark="none")]
    plan = runner.build_plan(tmp_path, "head", manifest, tmp_path)
    assert plan["expected_movers"][0]["tier"] == effective
    assert plan["merge_declarations"] == manifest["merge_declarations"]


def test_plan_rejects_underdeclared_tier_and_magnitude(tier_runner, monkeypatch, tmp_path):
    runner = tier_runner
    node = "tests/a.py::test_field"
    declaration = dict(node=node, tier="T1", max_rel_diff=1e-7)
    mover = dict(selector=node, tier="T1", max_rel_diff=2e-7)
    manifest = dict(components={}, merge_declarations=[dict(branch="infra/fixture", pytest=[declaration], benchmark="none")],
                    expected_movers=[mover])
    with pytest.raises(ValueError, match="T1.*max_rel_diff"):
        runner.build_plan(tmp_path, "head", manifest, tmp_path)
    mover.update(tier="T2")
    with pytest.raises(ValueError, match="undeclared T2"):
        runner.build_plan(tmp_path, "head", manifest, tmp_path)


def test_t1_annotation_does_not_absorb_unrelated_failure(tier_runner):
    data = dict(known_failures=[], expected_movers=[dict(selector="tests/a.py::test_field", tier="T1", max_rel_diff=1e-7)])
    assert tier_runner.failure_category("tests/a.py::test_field", data, "unrelated failure") == ("NEW", None)


@pytest.mark.parametrize("flag", ["intended", "discrete"])
def test_plan_classifies_from_declaration_flags(tier_runner, monkeypatch, tmp_path, flag):
    runner = tier_runner
    node = "tests/a.py::test_field"
    declaration = dict(node=node, tier="T1", max_rel_diff=1e-9, **{flag: True}, approval="fixture approval")
    manifest = dict(components={}, suites=[], known_failures=[], pin_files=[],
                    expected_movers=[dict(selector=node, tier="T1", max_rel_diff=1e-9)],
                    merge_declarations=[dict(branch="infra/fixture", pytest=[declaration], benchmark="none")])
    monkeypatch.setattr(runner, "collect", lambda *args: dict(nodes=[], exit_code=0, errors=[], two_device={}, benchmark_scenarios=[]))
    plan = runner.build_plan(tmp_path, "head", manifest, tmp_path)
    assert plan["expected_movers"][0]["tier"] == "T2"

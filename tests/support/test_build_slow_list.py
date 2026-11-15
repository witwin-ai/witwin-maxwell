"""The slow-list builder merges an existing list with JUnit (for example GPU batch) timings."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _builder():
    spec = importlib.util.spec_from_file_location("build_slow_list", ROOT / "scripts/dev/build_slow_list.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_junit_adds_nodes_to_an_existing_list(tmp_path):
    builder = _builder()
    batch = tmp_path / "batch/shard-g0"
    batch.mkdir(parents=True)
    (batch / "unit-1.xml").write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.fdtd.test_gpu" name="test_heavy[a]" time="95.0"/>'
        '<testcase classname="tests.fdtd.test_gpu" name="test_light" time="2.0"/>'
        '<testcase classname="tests.fdtd.test_gpu" name="test_skipped" time="300"><skipped/></testcase>'
        '<testcase classname="tests.renamed.test_old" name="test_gone" time="500"/>'
        '<testcase classname="tests.fdtd.test_cpu.TestKeep" name="test_listed" time="300"/>'
        '</testsuite></testsuites>')
    # Group XML duplicates the per-unit XML of grouped light files and is skipped.
    (batch / "group-1.xml").write_text(
        '<testsuites><testsuite><testcase classname="tests.fdtd.test_gpu" name="test_light" time="900"/>'
        '</testsuite></testsuites>')
    collect = tmp_path / "collect.txt"
    collect.write_text("\n".join(["tests/fdtd/test_gpu.py::test_heavy[a]", "tests/fdtd/test_gpu.py::test_light",
                                  "tests/fdtd/test_gpu.py::test_skipped", "tests/fdtd/test_cpu.py::TestKeep::test_listed",
                                  "tests/fdtd/test_cpu.py::test_old_entry", "", "5 tests collected"]))
    base = tmp_path / "base.txt"
    # A renamed test (its file still collects) is dropped; an entry in a file this CPU collection
    # could not import (CUDA at import) is kept.
    base.write_text(builder.HEADER + "tests/fdtd/test_cpu.py::TestKeep::test_listed  # 70 s, hist median of 2\n"
                    "tests/fdtd/test_cpu.py::test_old_entry  # 80 s, cpu-n24 median of 1\n"
                    "tests/fdtd/test_cpu.py::test_renamed_away  # 90 s, hist median of 1\n"
                    "tests/rf/network/test_cuda_import.py::test_kept  # 70 s, gpu0 median of 1\n")
    out = tmp_path / "slow_tests.txt"
    builder.main([str(out), f"gpu-b5={tmp_path / 'batch'}", "--collect", str(collect), "--base", str(base)])
    lines = [line for line in out.read_text().splitlines() if not line.startswith("#")]
    assert out.read_text().startswith(builder.HEADER)
    assert lines == ["tests/fdtd/test_cpu.py::TestKeep::test_listed  # 300 s, gpu-b5 median of 1",
                     "tests/fdtd/test_cpu.py::test_old_entry  # 80 s, cpu-n24 median of 1",
                     "tests/fdtd/test_gpu.py::test_heavy[a]  # 95 s, gpu-b5 median of 1",
                     "tests/rf/network/test_cuda_import.py::test_kept  # 70 s, gpu0 median of 1"]
    assert lines == sorted(lines)
    timings, unmatched = builder.junit_timings(tmp_path / "batch", {"tests/fdtd/test_gpu.py::test_light"})
    assert timings == {"tests/fdtd/test_gpu.py::test_light": dict(median=2.0, count=1)} and unmatched == 3


def test_extra_prefix_folds_its_nodes():
    builder = _builder()
    sources = [("cpu", {"tests/a/test_x.py::test_y[1]": dict(median=90, count=1),
                        "tests/a/test_x.py::test_z": dict(median=61, count=3)})]
    assert builder.build(sources, 60, extras=[("tests/a/test_x.py::test_y", 200, "manual")]) == [
        "tests/a/test_x.py::test_y  # 200 s, manual", "tests/a/test_x.py::test_z  # 61 s, cpu median of 3"]

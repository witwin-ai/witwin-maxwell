from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.support import benchmark_rf_no_feature as benchmark
from tests.support.benchmark_rf_no_feature import (
    _git_metadata,
    _run_compare,
    _sha256_file,
    _tree_sha256,
    _verify_baseline_snapshot,
    _worktree_content_metadata,
)


ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _make_git_repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Phase Four Test")
    _git(root, "config", "user.email", "phase4@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    (root / "tracked.py").write_bytes(b"VALUE = 1\n")
    (root / "nested").mkdir()
    (root / "nested" / "data.txt").write_bytes(b"baseline\n")
    _git(root, "add", "tracked.py", "nested/data.txt")
    _git(root, "commit", "--quiet", "-m", "baseline")
    return _git(root, "rev-parse", "HEAD")


def _archive_snapshot(repository: Path, snapshot: Path, commit: str) -> Path:
    archive = snapshot.with_suffix(".zip")
    _git(repository, "archive", "--format=zip", f"--output={archive}", commit)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(snapshot)
    return archive


def _compare_args(baseline_root: Path, candidate_root: Path, commit: str) -> argparse.Namespace:
    return argparse.Namespace(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        baseline_commit=commit,
        baseline_archive=None,
        python=Path(sys.executable),
        rounds=1,
        warmup=1,
        repeats=3,
        steps=1,
        grid_cells=12,
        max_regression_pct=1.0,
    )


def test_tree_digest_changes_with_content_and_path(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    first.write_text("alpha", encoding="utf-8")
    original = _tree_sha256(tmp_path)

    first.write_text("beta", encoding="utf-8")
    changed_content = _tree_sha256(tmp_path)
    assert changed_content["sha256"] != original["sha256"]

    first.rename(tmp_path / "renamed.txt")
    changed_path = _tree_sha256(tmp_path)
    assert changed_path["sha256"] != changed_content["sha256"]


def test_file_digest_records_resolved_path_and_size(tmp_path: Path) -> None:
    archive = tmp_path / "baseline.zip"
    archive.write_bytes(b"immutable archive")

    metadata = _sha256_file(archive)

    assert metadata["resolved_path"] == str(archive.resolve())
    assert metadata["bytes"] == len(b"immutable archive")
    assert len(metadata["sha256"]) == 64


def test_nested_snapshot_does_not_inherit_parent_checkout_identity(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _make_git_repo(repository)
    snapshot = repository / "build" / "snapshot"
    snapshot.mkdir(parents=True)

    metadata = _git_metadata(snapshot)

    if metadata["repository_root"] is not None:
        assert metadata["repository_root"] != metadata["resolved_root"]
    assert metadata["is_checkout_root"] is False
    assert metadata["commit"] is None
    assert metadata["dirty_tracked_files"] is None


def test_extracted_snapshot_is_bound_to_git_tree_contents(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit = _make_git_repo(repository)
    snapshot = tmp_path / "snapshot"
    archive = _archive_snapshot(repository, snapshot, commit)

    metadata = _verify_baseline_snapshot(repository, snapshot, commit, archive)

    assert metadata["verified"] is True
    assert metadata["file_count"] == 2


@pytest.mark.parametrize("mutation", ("missing", "extra", "cache_extra", "changed"))
def test_extracted_snapshot_rejects_tree_mismatch(tmp_path: Path, mutation: str) -> None:
    repository = tmp_path / "repository"
    commit = _make_git_repo(repository)
    snapshot = tmp_path / "snapshot"
    archive = _archive_snapshot(repository, snapshot, commit)
    if mutation == "missing":
        (snapshot / "tracked.py").unlink()
    elif mutation == "extra":
        (snapshot / "extra.py").write_bytes(b"EXTRA = True\n")
    elif mutation == "cache_extra":
        (snapshot / "__pycache__").mkdir()
        (snapshot / "__pycache__" / "extra.pyc").write_bytes(b"generated")
    else:
        (snapshot / "tracked.py").write_bytes(b"VALUE = 2\n")

    with pytest.raises(ValueError, match="does not match baseline_commit"):
        _verify_baseline_snapshot(repository, snapshot, commit, archive)


def test_baseline_rejects_git_checkout_even_when_clean(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit = _make_git_repo(repository)
    archive = tmp_path / "baseline.zip"
    _git(repository, "archive", "--format=zip", f"--output={archive}", commit)

    with pytest.raises(ValueError, match="must be a non-Git directory"):
        _verify_baseline_snapshot(repository, repository, commit, archive)


def test_worktree_digest_detects_repeated_tracked_and_untracked_edits(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _make_git_repo(repository)
    tracked = repository / "tracked.py"
    untracked = repository / "new_source.py"
    tracked.write_bytes(b"VALUE = 2\n")
    untracked.write_bytes(b"VALUE = 'first'\n")
    before = _worktree_content_metadata(repository)

    tracked.write_bytes(b"VALUE = 3\n")
    after_tracked_edit = _worktree_content_metadata(repository)
    assert (
        after_tracked_edit["content_manifest"]["sha256"]
        != before["content_manifest"]["sha256"]
    )

    untracked.write_bytes(b"VALUE = 'second'\n")
    after_untracked_edit = _worktree_content_metadata(repository)
    assert (
        after_untracked_edit["content_manifest"]["sha256"]
        != after_tracked_edit["content_manifest"]["sha256"]
    )


def test_digest_helpers_fail_closed_for_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        _sha256_file(tmp_path / "missing.zip")
    with pytest.raises(ValueError, match="not a directory"):
        _tree_sha256(tmp_path / "missing-tree")


def test_compare_rejects_same_resolved_root(tmp_path: Path) -> None:
    args = _compare_args(tmp_path, tmp_path / ".", "HEAD")

    with pytest.raises(ValueError, match="must be different"):
        _run_compare(args)


def test_compare_rejects_unresolvable_baseline_commit(tmp_path: Path) -> None:
    (tmp_path / "snapshot.txt").write_text("snapshot", encoding="utf-8")
    args = _compare_args(tmp_path, ROOT, "0" * 40)

    with pytest.raises(ValueError, match="is not available"):
        _run_compare(args)


def test_compare_rejects_baseline_tree_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    commit = _make_git_repo(repository)
    snapshot_root = tmp_path / "snapshot"
    archive = _archive_snapshot(repository, snapshot_root, commit)
    snapshot = snapshot_root / "tracked.py"
    args = _compare_args(snapshot_root, repository, commit)
    args.baseline_archive = archive
    invocation_count = 0

    def fake_invoke_sampler(**kwargs):
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 1:
            snapshot.write_text("after", encoding="utf-8")
        label = kwargs["label"]
        return {
            "label": label,
            "median_ms": 1.0,
            "mad_ms": 0.0,
            "samples_ms": [1.0, 1.0, 1.0],
            "git": {},
            "environment": {},
        }

    monkeypatch.setattr(benchmark, "_invoke_sampler", fake_invoke_sampler)

    with pytest.raises(RuntimeError, match="baseline tree changed"):
        _run_compare(args)

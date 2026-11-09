"""Resolve pinned baseline package trees locally or from staged archives."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _missing(rev: str):
    message = f"baseline {rev!r} is not available"
    if os.environ.get("WITWIN_REQUIRE_BASELINES") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def baseline_tree(rev: str, tmp_path: Path) -> Path:
    """Copy or archive a baseline into a new PYTHONPATH root under tmp_path."""
    dest = tmp_path / "baseline"
    staged = os.environ.get("WITWIN_BASELINE_TREES")
    if staged is not None:
        root = Path(staged)
        manifest = json.loads((root / "manifest.json").read_text())
        matches = {
            entry["sha"] for entry in manifest
            if entry["project"] == "witwin-maxwell"
            and (entry["sha"].startswith(rev) or entry["rev"] == rev)
        }
        if not matches:
            pytest.fail(f"baseline {rev!r} is not available in staged trees", pytrace=False)
        if len(matches) != 1:
            pytest.fail(f"ambiguous baseline {rev!r}", pytrace=False)
        shutil.copytree(root / "witwin-maxwell" / matches.pop() / "witwin", dest / "witwin")
    else:
        if shutil.which("git") is None:
            _missing(rev)
        probe = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--verify", f"{rev}^{{commit}}"],
            capture_output=True, text=True,
        )
        if probe.returncode:
            _missing(rev)
        archive = subprocess.run(
            ["git", "-C", str(_ROOT), "archive", probe.stdout.strip(), "witwin"],
            capture_output=True, check=True,
        )
        dest.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(dest, filter="data")
    if not (dest / "witwin").is_dir():
        _missing(rev)
    return dest

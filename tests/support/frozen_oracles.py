"""Frozen existing-path oracles for the CUDA A/B tests and the core-path gates.

The per-component ("existing") FDTD path is not selectable through public
options. Tests and gates that compared the default path against it load payloads
captured once from that path instead. The payloads live outside git in the
evidence store ``$WITWIN_EVIDENCE_ROOT/w7-20261108/frozen-oracles`` (repository
``evidence/`` when the variable is unset). The store's ``manifest.json`` records
the capture commit, environment and a sha256 for every payload; the manifest
itself is pinned here by sha256, so a changed or substituted payload fails
instead of silently moving a baseline. Scenes admitted later are captured on
their own base commit into pinned addenda (``addenda`` in the pin): each addendum
manifest adds new payload keys and never replaces one, except a pin marked
``"supersedes": true``. That re-pin addendum, captured the same way after a
user-approved behaviour change, replaces existing keys only, and only complete
scene directories; the replaced files stay in the store untouched. Without the
store the tests skip with :data:`MISSING_REASON`.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

STORE_ID = "w7-20261108/frozen-oracles"
# Small in-git pin of the store: capture commit, payload count and the manifest digest.
PIN = json.loads(Path(__file__).with_name("frozen_oracles_manifest.json").read_text())
MISSING_REASON = ("frozen existing-path oracle store is not available: set WITWIN_EVIDENCE_ROOT to the "
                  f"evidence root that contains {STORE_ID}/manifest.json")


def store_root() -> Path:
    return Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / STORE_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def manifest() -> dict:
    path = store_root() / "manifest.json"
    if not path.is_file():
        pytest.skip(MISSING_REASON)
    assert _sha256(path) == PIN["manifest_sha256"], f"{path} does not match the pinned frozen-oracle manifest"
    data = json.loads(path.read_text())
    assert data["commit"] == PIN["commit"] and len(data["payloads"]) == PIN["payloads"], path
    payloads = dict(data["payloads"])
    for pin in PIN["addenda"]:
        addendum_path = store_root() / pin["manifest"]
        assert addendum_path.is_file(), f"frozen-oracle addendum {addendum_path} is missing"
        assert _sha256(addendum_path) == pin["manifest_sha256"], f"{addendum_path} does not match its pin"
        addendum = json.loads(addendum_path.read_text())
        assert addendum["commit"] == pin["commit"] and len(addendum["payloads"]) == pin["payloads"], addendum_path
        keys = addendum["payloads"].keys()
        if pin.get("supersedes", False):
            assert keys <= payloads.keys(), (addendum_path, sorted(keys - payloads.keys())[:5])
            for directory in {key.rsplit("/", 1)[0] for key in keys if key.startswith("gates/")}:
                existing = {key for key in payloads if key.rsplit("/", 1)[0] == directory}
                assert existing <= keys, (addendum_path, "partial scene", directory)
        else:
            assert not payloads.keys() & keys, addendum_path
        payloads.update(addendum["payloads"])
    return data | {"payloads": payloads}


@lru_cache(maxsize=None)
def path(key: str) -> Path:
    """Return the verified payload file for ``key``."""
    entry = manifest()["payloads"].get(key)
    assert entry is not None, f"no frozen payload {key!r} in {STORE_ID}"
    target = store_root() / entry["file"]
    assert target.is_file(), f"frozen payload {key!r} is missing: {target}"
    assert _sha256(target) == entry["sha256"], f"frozen payload {key!r} changed: {target}"
    return target


def scene_directory(gate: str, scene: str) -> Path:
    """Verified directory holding the frozen ``gates/<gate>/<scene>`` payloads."""
    keys = [key for key in manifest()["payloads"] if key.startswith(f"gates/{gate}/{scene}/")]
    assert keys, f"no frozen payloads for gates/{gate}/{scene}"
    directories = {path(key).parent for key in keys}
    assert len(directories) == 1, (gate, scene, sorted(directories))
    return directories.pop()


def load(key: str) -> dict:
    """Load the frozen existing-path payload ``key`` as a flat name -> CPU tensor mapping."""
    return torch.load(path(key), weights_only=True, map_location="cpu")


def assert_equal(actual: dict, expected: dict, context="") -> None:
    """Bitwise payload comparison: identical names, dtypes, shapes and values."""
    assert actual.keys() == expected.keys(), (context, sorted(actual.keys() ^ expected.keys())[:20])
    for name in expected:
        value = actual[name].detach().cpu()
        assert value.dtype == expected[name].dtype and torch.equal(value, expected[name]), (context, name)

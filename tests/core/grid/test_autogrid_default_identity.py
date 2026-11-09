"""Existing grid specs stay byte-identical to a pinned baseline tree.

Opt-in corner / layer-bounds / gap refinement must not move a single node of
any pre-existing ``GridSpec`` (uniform, anisotropic, custom, auto with any
existing option, MeshOverrideStructure, ``LayerRefinementSpec(min_cells)``,
autogrid plus explicit nodes). ``grid_identity_dump.py`` runs in two fresh
interpreters, one importing ``witwin.maxwell`` from ``git archive <base>`` (or its staged tree) and
one from this tree, and every compiled node array is compared exactly.

``WITWIN_GRID_IDENTITY_BASE`` overrides the baseline commit. The default pins
the last commit before opt-in refinement; move it only for a deliberate,
reviewed grid change.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from tests.support.baseline_tree_env import probe_environment
from tests.support.pinned_baselines import baseline_tree as _baseline_tree

_BASE = os.environ.get("WITWIN_GRID_IDENTITY_BASE", "d37de300")
_ROOT = Path(__file__).resolve().parents[3]
_DUMP = Path(__file__).with_name("grid_identity_dump.py")


def _run_dump(tree: Path, out: Path) -> None:
    env = probe_environment(tree, baseline=tree != _ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ""
    probe = subprocess.run(
        [sys.executable, "-c", "import witwin.maxwell as m; print(m.__file__)"],
        env=env, capture_output=True, text=True, check=True, cwd=out.parent,
    )
    assert Path(probe.stdout.strip()).resolve().is_relative_to(tree.resolve())
    subprocess.run([sys.executable, str(_DUMP), str(out)], env=env, check=True, capture_output=True, cwd=out.parent)


def test_existing_grid_specs_are_byte_identical_to_baseline(tmp_path):
    base_tree = _baseline_tree(_BASE, tmp_path)

    _run_dump(base_tree, tmp_path / "base_out")
    _run_dump(_ROOT, tmp_path / "head_out")

    names = sorted(path.stem for path in (tmp_path / "base_out").glob("*.npz"))
    assert len(names) >= 9
    for name in names:
        base = np.load(tmp_path / "base_out" / f"{name}.npz")
        head = np.load(tmp_path / "head_out" / f"{name}.npz")
        assert sorted(base.files) == sorted(head.files)
        for key in base.files:
            assert np.array_equal(base[key], head[key]), f"{name}.{key} moved"
        base_t = torch.load(tmp_path / "base_out" / f"{name}.pt")
        head_t = torch.load(tmp_path / "head_out" / f"{name}.pt")
        for axis in "xyz":
            assert torch.equal(base_t[axis], head_t[axis]), f"{name}.{axis} tensor moved"

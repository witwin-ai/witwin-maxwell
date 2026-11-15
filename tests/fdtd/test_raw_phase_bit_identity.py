"""B0 final fields and port payloads stay bitwise equal to observer-origin (A)."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.support.baseline_tree_env import probe_environment
from tests.support.pinned_baselines import baseline_tree


ROOT = Path(__file__).resolve().parents[2]
# Master immediately before the (ii) merge: contains wall-E (soft-sheet wall rows), route (i) and the
# replay hotfix but not (ii), so B0 checks that (ii) alone leaves non-TFSF sources and ports bitwise.
# Re-pinned 2026-11-04 (earlier: f7ae748e, aed46e9f, d52ce15d, d397c4c8, d7b7cdf3).
BASE = "b27cac68"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="BITID requires native CUDA")


# The probe module carries the API dialect (tests/support/api_dialect.py) as probe["api"].
SOURCE_PROBE = r'''
import runpy
import sys
import numpy as np
import torch
probe = runpy.run_path(sys.argv[1])
api = probe["api"]
mw = api.mw
configs = {key: value for key, value in probe["FIELD_CONFIGS"].items() if "tfsf" not in key}
def current(kind):
    scene = probe["_box_scene"](mw.BoundarySpec.pml(num_layers=6))
    waveform = probe["_field_pulse"]()
    if kind == "uniform":
        source = mw.UniformCurrentSource(size=(.1,) * 3, source_time=waveform)
    else:
        names = {"J": ("Jz",), "M": ("Mz",), "JM": ("Jz", "Mz")}[kind]
        source = mw.CustomCurrentSource(mw.CurrentDataset(
            ((-.05, .05),) * 3, {name: np.ones((2, 2, 2)) for name in names}), source_time=waveform)
    # An observer gives the solve an output; the compared payload is the full solver field state.
    return api.add(scene, sources=[source], monitors=[mw.PointMonitor("probe", (0.05, 0.0, 0.05), fields=("Ez",))])
for kind in ("uniform", "J", "M", "JM"):
    configs[kind] = lambda kind=kind: current(kind)
def custom_field():
    scene = probe["_box_scene"](mw.BoundarySpec.pml(num_layers=6))
    return api.add(scene, sources=[mw.CustomFieldSource(mw.FieldDataset(
        ((0.,), (-.1, .1), (-.1, .1)),
        {"Ey": np.ones((1, 2, 2)), "Hz": np.ones((1, 2, 2)) / 376.730313668}),
        source_time=probe["_field_pulse"]())],
        monitors=[mw.PointMonitor("probe", (0.05, 0.0, 0.05), fields=("Ey",))])
configs["custom_field"] = custom_field
probe["fields"].__globals__["FIELD_CONFIGS"] = configs
probe["fields"](sys.argv[2])
'''


@pytest.mark.parametrize("probe", ["sources", "ports"])
def test_b0_payloads_are_bitwise_equal_to_a(tmp_path, probe):
    (tmp_path / "tree").mkdir()
    baseline = baseline_tree(BASE, tmp_path / "tree")
    payloads = []
    for label, tree in (("a", baseline), ("ii", ROOT)):
        output = tmp_path / f"{label}.pt"
        if probe == "sources":
            command = [sys.executable, "-c", SOURCE_PROBE,
                str(ROOT / "tests/fdtd/_bit_identity_probe.py"), str(output)]
        else:
            command = [sys.executable, str(ROOT / "tests/support/port_step_graph_bit_identity_scene.py"),
                str(output), "lumped1", "lumped2_one_driven", "terminal_direct", "coax", "wire_gap"]
        subprocess.run(command, cwd=tmp_path, env=probe_environment(tree, baseline=label == "a"), check=True)
        payload = torch.load(output, weights_only=False)
        if probe == "sources":
            assert Path(payload.pop("module_file")).resolve().is_relative_to(tree)
            payload = payload["fields"]
            for fields in payload.values():
                assert any(torch.count_nonzero(field) > 0 for field in fields.values())
        else:
            assert Path(payload.pop("__module")).resolve().is_relative_to(tree)
        payloads.append(payload)
    if probe == "ports":
        from tests.support import port_step_graph_bit_identity_scene as scene
        # Core-path fixtures keep every end-state key; their per-step histories are
        # skipped: scene.HISTORY_SKIP_REASON (user-approved, 2026-11-09).
        base, branch, base_flags, flags, core = scene.comparable_payloads(*payloads)
        for fixture in sorted(core):
            assert torch.equal(flags[f"{fixture}/__steps_run"], base_flags[f"{fixture}/__steps_run"]), fixture
        payloads = [base, branch]

    def identical(a, b, path=""):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for key in a:
                identical(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, torch.Tensor):
            assert torch.equal(a, b), path
        else:
            assert a == b, path
    identical(*payloads)

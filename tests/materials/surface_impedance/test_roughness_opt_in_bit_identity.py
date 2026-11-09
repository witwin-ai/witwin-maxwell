"""Opt-in roughness leaves every smooth surface path bit-identical to a pinned baseline tree.

The same probe (``smooth_surface_probe.py``) runs in two fresh interpreters: one importing
the package from a ``git archive`` of the pinned baseline revision (``_BASELINE_REV``) or its staged tree,
one importing the current tree. It dumps the smooth ``LossyMetalMedium`` analytic helpers,
the public and compiler-layer rational surface fits, the CopperFoil diffusion realization,
the compiled surface layouts (Box and staircased), and -- on CUDA -- every prepared FDTD
solver tensor, the surface write descriptors (order-0 resistance and Z-form ADE matrices),
the CopperFoil runtime state and the fields after a short eager run. Eight fitted entries
are compared through their responses, with equal order and passivity flags; every other
entry must be exactly equal (``torch.equal``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

import torch

import witwin.maxwell as mw

from tests.support.baseline_tree_env import probe_environment
from tests.support.pinned_baselines import baseline_tree as _baseline_tree

# The earlier pin 0a66e11e predates the periodic point-dipole wrap fix, which changed this probe's source on purpose.
# Re-baselined at f52aa33f for the additive-J (n+1/2) injection timing; proven equal to 075491bc plus only
# that hunk (<evidence-root>/coverage-e-timing-proof/results/added_gates_gpu/). Copper stays 1b242d6d.
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: f52aa33f.
_BASELINE_REV = os.environ.get("WITWIN_ROUGHNESS_BASELINE_REV", "f36cdaf1")
# Copper realizations and fields include compression on the batch-2 base.
# Re-pinned 2026-11-04 to f36cdaf1 for the observer time origin (A) (physical Yee labels: raw E/H spectra and port V/I rotate by exp(+i omega dt), derived invariants hold to FP32 rounding, all else bitwise; user-approved T2; per-key proof docs/assessments/observer-origin-repin-cpu-2026-11-03/, <evidence-root>/coverage-observer-origin/repin-proof-final/). Previous: 1b242d6d.
_COPPER_BASELINE_REV = "f36cdaf1"
_REPO = Path(__file__).resolve().parents[3]
_PROBE = Path(__file__).resolve().with_name("smooth_surface_probe.py")

# Baseline vector-fitting pole placement is non-unique and not reproducible across
# processes. The physically meaningful invariant for these entries is the response.
_FITTED_KEYS = {
    "rational.model.direct", "rational.model.residues", "rational.model.z",
    "rational.zfit.A", "rational.zfit.B", "rational.zfit.C", "rational.zfit.D", "rational.zfit.poles",
}


def _fit_realizations(dump):
    model = mw.RationalModel(
        poles=dump["rational.model.poles"], residues=dump["rational.model.residues"],
        direct=dump["rational.model.direct"], representation="Y",
    )
    A, B, C, D = (dump[f"rational.zfit.{key}"] for key in "ABCD")
    identity = torch.eye(A.shape[0], dtype=A.dtype)
    left = A + identity
    solved_B = torch.linalg.solve(left, B)
    # Undo the probe's bilinear discretization (dt=2e-12) before evaluating with
    # the shared continuous-time convention s=-i*2*pi*f.
    dt = 2.0e-12
    zfit = mw.StateSpaceNetwork(
        A=(2.0 / dt) * torch.linalg.solve(left, A - identity),
        B=(2.0 / dt) * solved_B,
        C=2.0 * torch.linalg.solve(left.T, C.T).T,
        D=D - C @ solved_B, representation="Z",
    )
    return model, zfit


def _assert_response_close(name, baseline, current):
    maximum = float(((current - baseline).abs() / baseline.abs()).max())
    assert maximum <= 1.0e-9, f"{name} max relative response difference: {maximum:.17g}"
    return maximum


def _compare_fitted_responses(baseline, current):
    frequencies = torch.logspace(math.log10(0.5e9), math.log10(5.0e9), 1025, dtype=torch.float64)
    base_model, base_zfit = _fit_realizations(baseline)
    current_model, current_zfit = _fit_realizations(current)
    assert base_model.order == current_model.order
    assert base_zfit.state_count == current_zfit.state_count
    assert base_zfit.state_count == baseline["rational.zfit.poles"].numel()
    assert current_zfit.state_count == current["rational.zfit.poles"].numel()
    differences = {}
    for name, base_fit, current_fit in (
        ("model", base_model, current_model), ("zfit", base_zfit, current_zfit),
    ):
        base_flags = base_fit.check_passivity(frequencies)
        current_flags = current_fit.check_passivity(frequencies)
        assert (base_flags.passive, base_flags.certified) == (
            current_flags.passive, current_flags.certified,
        ), name
        base_response, current_response = base_fit.evaluate(frequencies), current_fit.evaluate(frequencies)
        if name == "model":
            base_response, current_response = base_response.reciprocal(), current_response.reciprocal()
        differences[name] = _assert_response_close(name, base_response, current_response)
    # This is model.surface_impedance(freqs), a fitted output on the probe's
    # 31-point grid, rather than the analytic samples supplied to the fitter.
    differences["model.z"] = _assert_response_close(
        "model.z", baseline["rational.model.z"], current["rational.model.z"],
    )
    return differences


def _run_probe(package_root: Path, output: Path):
    env = probe_environment(package_root, baseline=package_root != _REPO)
    completed = subprocess.run(
        [sys.executable, str(_PROBE), str(output)], cwd=str(package_root), env=env,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    dump = torch.load(output)
    module = bytes(dump.pop("__module__").tolist()).decode()
    assert Path(module).resolve().is_relative_to(package_root.resolve()), module
    return dump


def test_smooth_surface_paths_are_bit_identical_to_pre_roughness_revision(tmp_path, copper_torch_kernel):
    baseline_root = _baseline_tree(_BASELINE_REV, tmp_path)

    baseline = _run_probe(baseline_root, tmp_path / "baseline.pt")
    current = _run_probe(_REPO, tmp_path / "current.pt")
    copper_parent = tmp_path / "copper"
    copper_parent.mkdir()
    copper_root = _baseline_tree(_COPPER_BASELINE_REV, copper_parent)
    copper = _run_probe(copper_root, tmp_path / "copper_baseline.pt")
    baseline.update({key: value for key, value in copper.items()
                     if key.startswith(("foil.response.", "prepared.foil.", "run.foil."))})

    # The CopperFoil free-edge model adds runtime state entries; every baseline entry must still match.
    added = set(current) - set(baseline)
    assert set(baseline) <= set(current)
    assert {key.rsplit(".", 1)[1] for key in added} <= {"admittance_scale", "effective", "trace_conductance"}, added
    assert len(current) > 100
    if torch.cuda.is_available():
        assert any(key.startswith("prepared.box_lossy._surface_impedance") for key in current)
        assert any(key.startswith("prepared.box_rational._surface_impedance") for key in current)
        assert any(key.startswith("run.foil.") for key in current)
    mismatched = [
        key for key in sorted(baseline)
        if baseline[key].dtype != current[key].dtype
        or baseline[key].shape != current[key].shape
        or (key not in _FITTED_KEYS and not torch.equal(baseline[key], current[key]))
    ]
    assert not mismatched, f"{len(mismatched)} entries differ: {mismatched[:20]}"
    differences = _compare_fitted_responses(baseline, current)
    print(f"Max fitted-response relative differences: {differences}")

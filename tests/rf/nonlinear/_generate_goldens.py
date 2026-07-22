"""Freeze the oversampled rectifier reference goldens.

Run once (and whenever the frozen topology/parameters in ``_rectifier_fixtures``
change) to regenerate ``goldens/rectifier_goldens.pt``:

    conda run -n maxwell python tests/rf/nonlinear/_generate_goldens.py

The reference is a *differently-integrated* solution of the same circuit: a
backward-Euler run at ``REFERENCE_SUBDIVISION`` x finer time step than the graded
coarse run, linearly interpolated back onto the coarse grid. The graded gate runs
the coarse solver with the trapezoidal companion and asserts normalized-RMS
agreement < 1%, so a match cross-validates the transient integrator (TR vs a finer
BE reference), the charge companion, and the Newton solve against a finer solution
rather than the graded run's own arithmetic.

Known limitation: the reference is NOT code-independent. It is produced by the
same ``run_nonlinear_transient``/``newton_solve``/device-model code under test;
only the step size and the TR-vs-BE companion differ. A shared systematic error in
q(v), conduction, or terminal mapping would cancel across both runs and pass this
gate. The code-independent discriminators are the analytic Shockley DC gate
(< 1e-5) and the analytic RC-decay gate (< 1e-4) in ``test_nonlinear_transient``;
this golden gate adds integration-convergence coverage on top of those. Provenance
(parameters, dt, integration, torch/commit) is stored alongside the arrays.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

import _rectifier_fixtures as fx
from witwin.maxwell.compiler.nonlinear_devices import run_nonlinear_transient

GOLDEN_PATH = Path(__file__).resolve().parent / "goldens" / "rectifier_goldens.pt"


def _reference_output(builder) -> torch.Tensor:
    system, source_injection, output_index = builder()
    fine_times = fx.reference_times()
    result = run_nonlinear_transient(
        system,
        fine_times,
        source_injection,
        _reference_config(),
        integration="backward_euler",
    )
    fine_output = result.node_voltages[:, output_index]
    coarse = fx.coarse_times()
    # Linear interpolation of the fine reference onto the coarse sample grid.
    index = torch.searchsorted(fine_times, coarse).clamp(1, fine_times.numel() - 1)
    t0 = fine_times[index - 1]
    t1 = fine_times[index]
    w = (coarse - t0) / (t1 - t0)
    return fine_output[index - 1] * (1.0 - w) + fine_output[index] * w


def _reference_config():
    import witwin.maxwell as mw

    return mw.NonlinearSolveConfig(max_iterations=40)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"


def main() -> None:
    torch.manual_seed(0)
    half = _reference_output(fx.build_half_wave_system)
    full = _reference_output(fx.build_full_wave_system)
    payload = {
        "times": fx.coarse_times(),
        "half_wave_output": half,
        "full_wave_output": full,
        "provenance": {
            "integration": "backward_euler",
            "reference_subdivision": fx.REFERENCE_SUBDIVISION,
            "steps_per_period": fx.STEPS_PER_PERIOD,
            "periods": fx.PERIODS,
            "frequency": fx.FREQUENCY,
            "amplitude": fx.AMPLITUDE,
            "source_resistance": fx.SOURCE_RESISTANCE,
            "load_resistance": fx.LOAD_RESISTANCE,
            "smoothing_capacitance": fx.SMOOTHING_CAPACITANCE,
            "saturation_current": fx.SATURATION_CURRENT,
            "ideality": fx.IDEALITY,
            "torch_version": torch.__version__,
            "git_commit": _git_commit(),
        },
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, GOLDEN_PATH)
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()

"""Falsification driver for the F1b independent cross-check.

Characterizes the EM one-port and builds the independent scipy prediction from an
UNPERTURBED coupled run, then runs the coupled test scene again with the MNA
field-port companion conductance scaled by a few percent and shows the coupled
port voltage departs from the (unperturbed) independent prediction beyond the
port-voltage gate tolerance -- while the baseline run passes.

Reproduces the red/green transition of
``test_crosscheck_rejects_perturbed_mna_companion``.

Env (see acceptance doc):
    export CUDA_HOME=.../nvidia/cu13; export PATH="$CUDA_HOME/bin:$PATH"
    export PYTHONPATH=<worktree>; export CUDA_VISIBLE_DEVICES=0
    conda run -n maxwell --no-capture-output python \
        docs/assessments/f1-cosim-e2-probes/falsify_mna_companion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "tests" / "rf" / "circuits"))

import test_circuit_independent_crosscheck as T  # noqa: E402
from witwin.maxwell.fdtd.circuits import CircuitPortRuntime  # noqa: E402


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required."

    char_result = T._run(
        T._circuit(T._R_CHAR, T._drive(**T._CHAR_DRIVE).waveform()), steps=T._CHAR_STEPS
    )
    model = T.fit_admittance(T.measure_admittance(char_result))

    # Baseline (unperturbed) coupled run + independent prediction.
    base_result = T._run(T._circuit(T._R_TEST, T._drive(**T._TEST_DRIVE).waveform()), steps=T._TEST_STEPS)
    base = T._coupled_port_trace(base_result, T._R_TEST)
    predicted = T._predict_port_voltage(model, T._drive(**T._TEST_DRIVE), T._R_TEST, base.times)
    base_rel = float(np.abs(predicted.voltage - base.voltage).max() / base.peak_voltage)

    # Perturb the MNA field-port companion conductance and re-run.
    original = CircuitPortRuntime.conductance
    try:
        CircuitPortRuntime.conductance = lambda self, integration: T._FALSIFY_SCALE * original(self, integration)
        pert_result = T._run(T._circuit(T._R_TEST, T._drive(**T._TEST_DRIVE).waveform()), steps=T._TEST_STEPS)
    finally:
        CircuitPortRuntime.conductance = original
    pert = T._coupled_port_trace(pert_result, T._R_TEST)
    pert_rel = float(np.abs(predicted.voltage - pert.voltage).max() / pert.peak_voltage)

    tol = T._VOLTAGE_TOL
    print(f"companion scale           : {T._FALSIFY_SCALE}")
    print(f"port-voltage tol          : {tol:.1e}")
    print(f"baseline    rel (expect < tol) : {base_rel:.3e}  -> {'GREEN' if base_rel < tol else 'RED'}")
    print(f"perturbed   rel (expect > tol) : {pert_rel:.3e}  -> {'RED' if pert_rel > tol else 'GREEN'}")
    ok = base_rel < tol < pert_rel
    print(f"falsification demonstrated: {ok}")


if __name__ == "__main__":
    main()

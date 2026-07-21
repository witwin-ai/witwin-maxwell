"""Reproduce the F1b independent cross-check margins from the test fixtures.

Runs the same characterization + coupled test + independent scipy prediction the
gates use and prints the observed relative errors against the pre-registered
tolerances, so the acceptance-doc numbers cannot drift from the gate thresholds.

Env (see acceptance doc):
    export CUDA_HOME=.../nvidia/cu13; export PATH="$CUDA_HOME/bin:$PATH"
    export PYTHONPATH=<worktree>; export CUDA_VISIBLE_DEVICES=0
    conda run -n maxwell --no-capture-output python \
        docs/assessments/f1-cosim-e2-probes/crosscheck_margins_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "tests" / "rf" / "circuits"))

import test_circuit_independent_crosscheck as T  # noqa: E402


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required."

    char_result = T._run(
        T._circuit(T._R_CHAR, T._drive(**T._CHAR_DRIVE).waveform()), steps=T._CHAR_STEPS
    )
    admittance = T.measure_admittance(char_result)
    model = T.fit_admittance(admittance)
    fit = model.evaluate(torch.tensor(T._BAND, dtype=torch.float64)).detach().cpu().numpy().reshape(-1)
    fit_rel = float(np.abs(fit - admittance).max() / np.abs(admittance).max())

    test_result = T._run(T._circuit(T._R_TEST, T._drive(**T._TEST_DRIVE).waveform()), steps=T._TEST_STEPS)
    coupled = T._coupled_port_trace(test_result, T._R_TEST)
    predicted = T._predict_port_voltage(model, T._drive(**T._TEST_DRIVE), T._R_TEST, coupled.times)

    v_rel = float(np.abs(predicted.voltage - coupled.voltage).max() / coupled.peak_voltage)
    i_rel = float(np.abs(predicted.current - coupled.current).max() / coupled.peak_current)

    print(f"fit order                 : {T._FIT_ORDER}")
    print(f"fit rel error   (tol {T._FIT_TOL:.1e}) : {fit_rel:.3e}")
    print(f"port-voltage rel(tol {T._VOLTAGE_TOL:.1e}) : {v_rel:.3e}")
    print(f"port-current rel(tol {T._CURRENT_TOL:.1e}) : {i_rel:.3e}")
    print(f"peak |v_port|             : {coupled.peak_voltage:.4e} V")
    print(f"peak |i_port|             : {coupled.peak_current:.4e} A")
    print(f"poles (rad/s)             : {model.poles.detach().cpu().numpy()}")


if __name__ == "__main__":
    main()

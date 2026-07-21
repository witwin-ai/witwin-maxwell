"""Reprint the observed energy-balance margins asserted by the F1a suite.

Reuses the exact test fixtures so the printed numbers cannot drift from the
committed gate thresholds. Run from the worktree root with the standard env
(conda ``maxwell``, ``CUDA_HOME`` set, ``PYTHONPATH`` = worktree):

    CUDA_VISIBLE_DEVICES=0 python \
      docs/assessments/f1-cosim-e2-probes/conservation_margins_probe.py

The asserted tolerances live in ``tests/rf/circuits/test_circuit_conservation.py``
(``_CONSERVATION_TOL`` = 5e-3 of throughput, ``_LINK_TOL`` = 2e-2 of peak field
energy). The margins below are observations on the current host; the gates assert
the thresholds, not these values.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("tests", "rf", "circuits"))

import test_circuit_conservation as suite  # noqa: E402


def _report(name, circuit, resistor_names, source_names):
    record = suite._run_coupled_balance(
        circuit, resistor_names=resistor_names, source_names=source_names
    )
    throughput = record.throughput
    conservation = float(np.abs(record.conservation_residual()).max())
    d_field = record.u_field - record.u_field[0]
    link = float(np.abs(d_field - record.cum_field_change).max())
    print(f"[{name}]")
    print(f"  throughput (source/dissipation) = {throughput:.3e} J")
    print(f"  peak field energy               = {record.peak_field:.3e} J")
    print(f"  circuit stored (peak)           = {record.u_circuit.max():.3e} J")
    print(f"  conservation residual           = {conservation:.3e} J"
          f"  ({conservation / throughput:.3e} of throughput; tol {suite._CONSERVATION_TOL:.0e})")
    print(f"  field-link residual             = {link:.3e} J"
          f"  ({link / record.peak_field:.3e} of peak field; tol {suite._LINK_TOL:.0e})")


def main():
    _report("a resistive load", suite._scenario_a_circuit(), ("R1",), ("V1",))
    _report("b series RLC", suite._scenario_b_circuit(), ("R1",), ("V1",))
    _report("c VCVS network", suite._scenario_c_circuit(), ("R1", "R2", "R3"), ("V1", "E1"))


if __name__ == "__main__":
    main()

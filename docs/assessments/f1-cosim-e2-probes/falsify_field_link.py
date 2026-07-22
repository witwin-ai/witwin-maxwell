"""Falsification driver for the F1a genuine field-link gate.

Corrupts the coupling operator (over-scatters the port injection into the field
by 5% beyond the recorded current) and shows the field-link check goes RED, then
restores and shows it green. Run from the worktree root with the standard env:

    CUDA_VISIBLE_DEVICES=0 python \
      docs/assessments/f1-cosim-e2-probes/falsify_field_link.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join("tests", "rf", "circuits"))

import test_circuit_conservation as suite  # noqa: E402
from witwin.maxwell.fdtd.circuits import EMCircuitRuntime  # noqa: E402


_STEPS = 800


def _run():
    """Return (field_link_ok, conservation_residual/throughput) at a fixed length.

    The conservation ratio is always computed (never aborted on the field-link
    assert) so the broken case reports a reproducible number rather than raising.
    """

    record = suite._run_coupled_balance(
        suite._scenario_a_circuit(), resistor_names=("R1",), source_names=("V1",), steps=_STEPS
    )
    ratio = float(np.abs(record.conservation_residual()).max() / record.throughput)
    try:
        suite._assert_field_link(record)
        link_ok = True
    except AssertionError:
        link_ok = False
    return link_ok, ratio


def main():
    link_ok, ratio = _run()
    print(f"baseline ({_STEPS} steps): field-link {'PASS' if link_ok else 'RED'}; "
          f"conservation residual/throughput = {ratio:.3e}")

    original = EMCircuitRuntime._apply_field_current

    def broken(self, port, current, voltage, *, scatter_field=True):
        original(self, port, current, voltage, scatter_field=scatter_field)
        if scatter_field:
            field_tensor = getattr(port.solver, port.field_name)
            torch.mul(port.field.injection, current, out=port.field.correction_buffer)
            field_tensor.view(-1).index_add_(
                0, port.field.linear_indices, port.field.correction_buffer, alpha=-0.05
            )

    EMCircuitRuntime._apply_field_current = broken
    try:
        link_ok, ratio = _run()
        status = "RED as expected" if not link_ok else "did NOT fail (unexpected)"
        print(f"broken (injection over-scatter by 5%): field-link {status}; "
              f"conservation residual/throughput = {ratio:.3e}")
    finally:
        EMCircuitRuntime._apply_field_current = original

    link_ok, ratio = _run()
    print(f"restored ({_STEPS} steps): field-link {'PASS' if link_ok else 'RED'}; "
          f"conservation residual/throughput = {ratio:.3e}")


if __name__ == "__main__":
    main()

"""Emitter for the observed margins quoted in the E4a cascade acceptance doc.

The acceptance doc (``docs/assessments/e4a-network-cascade-acceptance-2026-07-19.md``,
"Pre-registered tolerances and observed margins") cites concrete observed
residuals. This script is the committed emitter for those numbers: it drives the
same reference/embedded paths and conservation scenarios used by the tests and
prints the observed margins against their asserted tolerances.

Run (from the worktree root, env exports per the acceptance doc):

    CUDA_VISIBLE_DEVICES=1 python docs/assessments/e4-network-e2-probes/cascade_margins_probe.py

It reuses the test modules' fixtures so the probe cannot drift from the gates:
the same bare-device sweep, Touchstone round-trip, cascade algebra, embedding,
and conservation scenarios. Correctness only -- no wall-clock timing.
"""

from __future__ import annotations

import os
import tempfile

import torch

import tests.rf.network.test_network_cascade_crosscheck as cc
import tests.rf.network.test_network_conservation as cons


def _cross_check_margins() -> None:
    print("== Cross-check |dS| (tolerance 1e-5) ==")
    bare = cc._measure_bare_three_port()
    for tag, factory, order in (
        ("lossy_attenuator", cc._lossy_model, 1),
        ("reactive", cc._reactive_model, 2),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"{tag}.s2p")
            cc._write_touchstone(factory(), path)
            raw = cc._raw_samples_at_measure(path)
            reference = bare.cascade(raw, port_map={"d1": "n0", "d2": "n1"})
            reference_s11 = reference.s[:, 0, 0]
            network_effect = torch.max(
                torch.abs(bare.s[:, 0, 0] - reference_s11)
            ).item()
            embedded_s11 = cc._embedded_input_reflection(path, order)
            residual = torch.max(torch.abs(embedded_s11 - reference_s11)).item()
        print(
            f"  {tag:18s} residual |dS| = {residual:.3e}   "
            f"network effect = {network_effect:.3e}"
        )


def _power_balance_margins() -> None:
    print("== Power-balance |P1-P2|/|P1| (tolerance 1e-3) ==")
    for scenario, factory in cons.SCENARIOS.items():
        layout, block, model, _has_state = factory()
        diagnostics = cons._run(layout, block, cons.BASE_STEPS)
        voltage = diagnostics.voltage
        current = diagnostics.current
        admittance = model.evaluate(
            torch.tensor(cons.MEASURE_FREQUENCIES, dtype=torch.float64, device=cons.DEVICE)
        )
        solved_power = torch.sum(0.5 * torch.real(voltage * torch.conj(current)), dim=0)
        model_current = torch.einsum("fij,jf->if", admittance, voltage)
        model_power = torch.sum(0.5 * torch.real(torch.conj(voltage) * model_current), dim=0)
        balance_error = torch.max(
            torch.abs(solved_power - model_power) / torch.abs(solved_power)
        ).item()
        generated = diagnostics.metadata["generated_energy"]
        absorbed = diagnostics.metadata["absorbed_energy"]
        rel_generated = generated / absorbed if absorbed else float("nan")
        print(
            f"  {scenario:10s} balance error = {balance_error:.3e}   "
            f"generated/absorbed = {rel_generated:.3e}"
        )


def _reactive_state_ring_down() -> None:
    print("== Reactive state-norm ring-down T -> 4T ==")
    layout, block, _model, _has_state = cons._reactive_scenario()
    short = cons._run(layout, block, cons.BASE_STEPS)
    long = cons._run(layout, block, 4 * cons.BASE_STEPS)
    print(
        f"  state_norm(T)  = {short.state_norm.item():.3e}   "
        f"state_norm(4T) = {long.state_norm.item():.3e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The cascade-margins probe requires CUDA.")
    _cross_check_margins()
    _power_balance_margins()
    _reactive_state_ring_down()


if __name__ == "__main__":
    main()

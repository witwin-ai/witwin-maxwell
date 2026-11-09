"""Finite-record identities and opt-in measured full-scene acceptance gates.

Set TFSF_RECORD_GATE_REPORT to the combined replacement report. Original
decay, few-ulp and steady-window checks remain report diagnostics only.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.dev.tfsf_record_gates import (
    BASE, CWS, FLOOR_K, PULSES, SCHEMA_VERSION, advance_spectrum, dft, endpoint_prediction,
    floor_gate, floor_rule, harmonic_parts, physics_gate, predicted_reference, replacement_report, transfer,
)


@pytest.mark.parametrize("label", (0., .5, 1.))
def test_endpoint_identity_preserves_global_labels_and_rectangular_scale(label):
    rng = np.random.default_rng(147)
    values = rng.normal(size=(513, 2)) + 1j * rng.normal(size=(513, 2))
    theta = .057
    expected = np.sum(values[1:] * np.exp(1j * theta * (np.arange(512) + label))[:, None], axis=0)
    # Both pulse dt*f and CW 2/N scale the whole endpoint expression.
    for scale in (4.7619e-12 * 2e9, 2 / 512):
        np.testing.assert_allclose(scale * advance_spectrum(values, theta, label=label),
                                   scale * expected, rtol=2e-14, atol=2e-14)


def test_initial_state_defect_is_needed_for_a_causal_started_recurrence():
    theta = .059
    n = np.arange(514)
    drive = np.exp(-((n - 170) / 60)**2) * np.cos(theta * n)
    old, new = [0.], [.03]
    for k in range(513):
        old.append(.96 * old[-1] + drive[k])
        new.append(.96 * new[-1] + drive[k + 1])
    y_a = np.column_stack([old[1:], 2 * np.asarray(old[1:])])
    y_b = np.column_stack([new[1:], 2 * np.asarray(new[1:])])
    a = dict(y=y_a, r=drive[:513], theta=theta, beta=1., provider="plane_wave_axis_aligned")
    b = dict(y=y_b, r=drive[1:514], theta=theta, beta=1., provider=a["provider"])
    defect = (.03 - old[1]) * .96**np.arange(1, 513)
    defect = np.column_stack([defect, 2 * defect])
    predicted = endpoint_prediction(a, b, defect=defect)
    np.testing.assert_allclose(predicted, transfer(b, stop=512), rtol=2e-14)
    assert np.max(np.abs(endpoint_prediction(a, b) - transfer(b, stop=512))) > 1e-6


def test_analytic_cw_reference_changes_labels_without_advancing_samples():
    theta = .061
    samples = np.cos(theta * np.arange(513))
    a = dict(r=samples, beta=1., theta=theta, provider="analytic_profile")
    b = dict(beta=0.)
    expected = np.sum(samples[:512] * np.exp(1j * theta * np.arange(512)))
    np.testing.assert_allclose(predicted_reference(a, b), expected, rtol=2e-14)
    assert abs(advance_spectrum(samples, theta) / np.exp(1j * theta) - expected) > .1


@pytest.mark.parametrize("offset", (0., .5))
def test_causal_cw_mask_needs_a_forcing_defect_at_each_yee_stage(offset):
    theta = .059
    old, new, defect = 0., 0., 0.
    # A's phase is -theta. B removes that phase, but keeps the activation time.
    old = .97 * old + (offset >= 8.4) * np.cos(theta * offset - theta)
    defect = -old
    for n in range(32):
        time, future = n + offset, n + 1 + offset
        carrier = np.cos(theta * time)
        old = .97 * old + (future >= 8.4) * np.cos(theta * future - theta)
        new = .97 * new + (time >= 8.4) * carrier
        defect = .97 * defect + ((time >= 8.4) - int(future >= 8.4)) * carrier
        assert abs(new - old - defect) < 1e-14


def test_steady_cw_counter_rotating_term_cancels_on_integer_periods():
    theta = 2 * np.pi / 105
    n = np.arange(8193)
    positive = np.array([.7 + .2j, .04 - .01j])
    negative = np.conj(positive)
    startup = np.exp(-n / 25)[:, None] * np.array([.13, .02])
    y = (np.exp(-1j * theta * n)[:, None] * positive
         + np.exp(1j * theta * n)[:, None] * negative + startup)
    row = dict(y=y, theta=theta)
    pos, neg, transient, fitted = harmonic_parts(row)
    np.testing.assert_allclose(fitted, np.stack([positive, negative]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(pos + neg + transient, y[:513], rtol=1e-14, atol=1e-14)
    assert np.max(np.abs(dft(y, theta, start=6090, stop=8190) / 2100
                            - np.exp(1j * theta) * positive)) < 1e-14
    assert abs(dft(np.exp(1j * theta * n), theta, stop=512)) > 1.


@pytest.fixture(scope="module")
def measured_report():
    path = os.environ.get("TFSF_RECORD_GATE_REPORT")
    if path is None:
        pytest.skip("set TFSF_RECORD_GATE_REPORT to enforce measured replacement gates")
    report = json.loads(Path(path).read_text())
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["baseline"] == BASE
    assert report["smoke"] is False, "smoke captures do not qualify the full gates"
    assert report["constants"] == dict(physics_rtol=1e-12, endpoint_rtol=5e-9, floor_k=3)
    assert set(report["diagnostics"]) == {"float64", "float32"}
    return report


def assert_gate(gate):
    assert gate["passes"], gate
    assert gate["value"] <= gate["bound"], gate
    assert gate["margin"] == pytest.approx(gate["bound"] - gate["value"])


@pytest.mark.parametrize("case", PULSES)
def test_measured_pulse_physics_gate(measured_report, case):
    gates = [gate for gate in measured_report["gates"]["a_prime"] if gate["case"] == case]
    assert {tuple(gate["window"]) for gate in gates} == {(0, n) for n in (512, 2048, 8192)}
    assert len(gates) == 6
    for gate in gates:
        assert_gate(gate)


@pytest.mark.parametrize("case", PULSES)
def test_measured_pulse_endpoint_gate(measured_report, case):
    gates = [gate for gate in measured_report["gates"]["a1_endpoint"] if gate["case"] == case]
    assert len(gates) == 2
    for gate in gates:
        assert gate["window"] == [0, 8192]
        assert gate["relative_bound"] == 5e-9
        assert_gate(gate)


@pytest.mark.parametrize("case", (*CWS, "grating_cw"))
def test_measured_cw_physics_gate(measured_report, case):
    if case == "grating_cw" and case not in measured_report["cases"]:
        pytest.skip("grating CW was not captured in either precision")
    gates = [gate for gate in measured_report["gates"]["b_prime"] if gate["case"] == case]
    assert len(gates) == 2
    for gate in gates:
        assert set(gate["normalized_contributions"]) == {"positive", "counter_rotating", "startup"}
        assert_gate(gate)


def test_measured_production_floor_gates(measured_report):
    gates = measured_report["gates"]["a_double_prime"]
    assert {gate["case"] for gate in gates} == set(measured_report["cases"])
    for gate in gates:
        assert gate["k"] == 3
        assert gate["bound"] == 3 * (gate["floor_a"] + gate["floor_ii"])
        assert_gate(gate)


@pytest.mark.parametrize("group", ("pulse", "cw", "endpoint"))
@pytest.mark.parametrize("passes", (True, False))
def test_fixed_physics_bounds(group, passes):
    bound = 5e-9 if group == "endpoint" else 1e-12
    observed = 2j
    predicted = observed + bound * (1 if passes else 4)
    gate = physics_gate(observed, predicted, endpoint_only=group == "endpoint")
    assert gate["passes"] is passes
    assert gate["bound"] == bound * abs(observed)
    assert (gate["margin"] >= 0) is passes


@pytest.mark.parametrize("error,passes", ((5., True), (6., True), (7., False)))
def test_production_floor_rule_has_fixed_k(error, passes):
    gate = floor_rule(error, .5, 1.5)
    assert gate["k"] == FLOOR_K == 3
    assert gate["bound"] == 6.
    assert gate["passes"] is passes


def test_production_floors_are_measured_per_tree_with_complex_values():
    gate = floor_gate(2 + 3j, 4 + 5j, 1 + 3j, 4 + 3j)
    assert gate["floor_a"] == 1.
    assert gate["floor_ii"] == 2.
    assert gate["value"] == abs(1 - 2j)
    assert_gate(gate)
    assert floor_rule(0., 0., 0.)["passes"]
    assert not floor_rule(1e-30, 0., 0.)["passes"]


def synthetic_leg(dtype, *, mismatch=False):
    key = "axis_pulse/record"
    record = dict(y=np.ones((513, 2), dtype=dtype), r=np.ones(513),
                  theta=.057, dt=1e-12, beta=1., provider="plane_wave_axis_aligned", fields=("Ex", "Hy"))
    changed = dict(record, y=record["y"] * (1.01 if mismatch else 1.))
    leg = dict(a=dict(module_file="A", records={key: record}),
               ii=dict(module_file="ii", records={key: changed}),
               defects=dict(records={"axis_pulse": dict(y=np.zeros((512, 2)))}),
               diagnostics=dict(baseline=BASE, field_dtype=np.dtype(dtype).name, device="cpu", smoke=True,
                                diagnostics_pass=False, pulses={"old_decay_passes": False}))
    for tree, row in (("a", record), ("ii", changed)):
        leg[f"{tree}_observers"] = dict(module_file=leg[tree]["module_file"],
            records={key: {n: transfer(row, stop=n) for n in (512, 513)}})
    return leg


@pytest.mark.parametrize("mismatch", (False, True))
def test_combined_cli_check_exit_code(tmp_path, mismatch):
    for dtype in ("float64", "float32"):
        directory = tmp_path / dtype
        directory.mkdir()
        leg = synthetic_leg(dtype, mismatch=mismatch)
        (directory / "report.json").write_text(json.dumps(leg.pop("diagnostics")))
        for name, filename in (("a", "a.pt"), ("ii", "ii.pt"), ("defects", "defects.pt"),
                               ("a_observers", "a-observers.pt"), ("ii_observers", "ii-observers.pt")):
            torch.save(leg[name], directory / filename)
    result = subprocess.run([sys.executable, "-m", "scripts.dev.tfsf_record_gates", "--replacement-gates",
        "--analyze-only", "--smoke", "--check", "--fp64-out", str(tmp_path / "float64"),
        "--fp32-out", str(tmp_path / "float32"), "--out", str(tmp_path / "combined")],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert result.returncode == int(mismatch), result.stdout + result.stderr
    report = json.loads((tmp_path / "combined/report.json").read_text())
    assert report["passes"] is not mismatch
    assert report["diagnostics"]["float64"]["diagnostics_pass"] is False


def test_combined_report_rejects_missing_precision_case():
    fp64, fp32 = synthetic_leg("float64"), synthetic_leg("float32")
    fp32["ii"]["records"] = {}
    with pytest.raises(ValueError, match="case/variant mismatch"):
        replacement_report(fp64, fp32, smoke=True)

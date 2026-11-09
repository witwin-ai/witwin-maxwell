"""Independent checks of the offline incident-line qualification measurements.

These validate the measuring instruments; production acceptance is the fixed
pole/time-domain/reflection gate in the sweep command, which can exit nonzero.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.dev.tfsf_aux_line_termination import (
    CONFIGURATIONS, Configuration, appended_line, entry_history, prepare_line,
    outgoing_control, pulse_tail, reflection, sampled_region_lossless, slowest_pole,
    snapped_cell_count, source_free_update_operator,
)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def baseline():
    return prepare_line(Configuration())


@pytest.mark.parametrize("x,expected", [
    (36., 36), (36. + 1e-10, 36), (36. - 1e-10, 36),
    (36. + 1e-6, 37), (.1, 1), (1e-10, 0),
])
def test_requested_extent_snap(x, expected):
    assert snapped_cell_count(x) == expected


def test_pole_reproduces_original_diagnosis(baseline):
    repeats = [slowest_pole(baseline[0]) for _ in range(3)]
    assert repeats[0] == repeats[1] == repeats[2]
    pole = repeats[0]
    assert pole["frequency_Hz"] == pytest.approx(3.806429917351778e9, rel=1e-11)
    assert pole["per_period_modulus"] == pytest.approx(.9962279520673138, abs=2e-10)


@pytest.mark.parametrize("config", [CONFIGURATIONS[i] for i in (0, 1, 2, 3, 4, 5, 6, 7)],
                         ids=lambda c: c.name)
def test_appended_candidate_has_lossless_face_samples_and_unchanged_interior(config):
    original, state = prepare_line(config)
    candidate = appended_line(original, 20, 2, 1e-8)
    assert sampled_region_lossless(original, state)
    assert sampled_region_lossless(candidate, state)
    # Exact production coefficient comparison, in both supported coefficient dtypes.
    for dtype in (torch.float32, torch.float64):
        for kind in ("electric", "magnetic"):
            old_decay = getattr(original, kind + "_decay").to(dtype)
            old_curl = getattr(original, kind + "_curl").to(dtype)
            for suffix, old in (("_decay", old_decay), ("_curl", old_curl)):
                new = getattr(candidate, kind + suffix).to(dtype)[:len(old)]
                assert torch.equal(new[old_decay == 1], old[old_decay == 1])
    assert torch.all(candidate.electric_decay[:candidate.sampled_nodes] == 1)
    assert torch.all(candidate.magnetic_decay[:candidate.sampled_nodes - 1] == 1)


def test_reduced_operator_matches_actual_basis_probe(baseline):
    line = appended_line(baseline[0], 20, 2, 1e-8)
    line.source_time = dict(line.source_time, amplitude=0.)
    j = line.source_index
    ne = len(line.electric) - j - 2
    nh = len(line.magnetic) - j
    measured = np.empty((ne + nh, ne + nh))
    for column in range(ne + nh):
        line.electric.zero_()
        line.magnetic.zero_()
        line.time_step = 0
        if column < ne:
            line.electric[j + 1 + column] = 1.
        else:
            line.magnetic[j + column - ne] = 1 / line.impedance
        line.advance()
        measured[:, column] = np.r_[line.electric[j + 1:-1].numpy(),
                                   line.magnetic[j:].numpy() * line.impedance]
    np.testing.assert_allclose(source_free_update_operator(line), measured, atol=3e-16, rtol=3e-15)


def test_numpy_pulse_measurement_matches_actual_line_updates(baseline):
    line = appended_line(baseline[0], 20, 2, 1e-8)
    expected = entry_history(line, 3000)
    actual = []
    for _ in expected:
        line.advance()
        actual.append(max(abs(line.electric[line.source_index + 1].item()),
                          abs(line.magnetic[line.source_index].item() * line.impedance)))
    np.testing.assert_allclose(actual, expected, atol=4e-14, rtol=3e-12)


def test_harmonic_reflection_matches_direct_driven_line(baseline):
    line = appended_line(baseline[0], 20, 2, 1e-8)
    line.source_time = dict(kind="cw", frequency=2e9, amplitude=1., phase=0.)
    samples = []
    times = []
    for step in range(24000):
        line.advance()
        if step >= 16000:
            samples.append(line.electric[line.source_index + 1:line.source_index + 3].numpy().copy())
            times.append(line.electric_time)
    theta = 2 * np.pi * 2e9 * np.array(times)
    temporal = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    fit = np.linalg.lstsq(temporal, samples, rcond=None)[0]
    phasor = fit[0] + 1j * fit[1]
    kds = 2 * np.arcsin(np.sin(np.pi * 2e9 * line.dt) * line.ds / (line.wave_speed * line.dt))
    basis = np.exp(1j * np.outer([1, 2], [kds, -kds]))
    forward, backward = np.linalg.solve(basis, phasor)
    assert abs(backward / forward) == pytest.approx(reflection(line, 2e9), abs=2e-6)


@pytest.mark.parametrize("h", [.025, .0125])
def test_candidate_layout_and_poles_are_padding_independent(h):
    records = []
    for half in (.15, .3):
        for layers in (4, 8, 12):
            line, _ = prepare_line(replace(Configuration(), h=h, half=half, layers=layers))
            candidate = appended_line(line, 20, 2, 1e-8)
            records.append((len(candidate.electric), slowest_pole(candidate)))
    assert len({record[0] for record in records}) == 1
    # The contract preserves ds bitwise; domain construction changes its last
    # ulps, so derived physical pole frequencies share double-precision accuracy.
    for _, pole in records[1:]:
        assert pole["frequency_Hz"] == pytest.approx(records[0][1]["frequency_Hz"], rel=1e-12)
        assert pole["per_period_modulus"] == pytest.approx(records[0][1]["per_period_modulus"], abs=2e-11)


def test_original_line_fails_the_literal_fixed_deadline(baseline):
    line, _ = baseline
    tail = pulse_tail(line, slowest_pole(line))
    assert tail["relative_bound"] == pytest.approx(1.032382731180714e-6)
    assert tail["entry_tail"] == pytest.approx(.0016958249502784, rel=1e-9)
    assert not tail["time_domain_gate"]
    assert not tail["pole_gate"]


def test_distant_lossless_end_retains_the_coarse_pulse_wake(baseline):
    line = baseline[0]
    candidate = appended_line(line, 320, 4, 1e-16)
    row = dict(cells=320, order=4, target=1e-16,
               tail=pulse_tail(candidate, slowest_pole(candidate)))
    control = outgoing_control(line, row)
    assert control["candidate_entry_tail"] > 20 * control["relative_bound"]
    assert control["lossless_reference_entry_tail"] == pytest.approx(
        control["candidate_entry_tail"], abs=1e-10)
    assert control["reference_roundtrip_ns"] > 10 * control["deadline_ns"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("config", CONFIGURATIONS[:5], ids=lambda c: c.name)
def test_full_sample_instrument_matches_actual_physical_line(config, dtype):
    from scripts.dev.tfsf_aux_line_termination import sampled_history, sampled_indices

    line, state = prepare_line(config, dtype)
    line = appended_line(line, 40, 3, 1e-8)
    expected = sampled_history(line, state, 1600)
    ei, hi = sampled_indices(line, state)
    actual = []
    for _ in expected:
        line.prepare_incident_boundary()
        before = line.electric[ei].numpy().copy()
        line.advance_magnetic()
        h = line.magnetic[hi].numpy().copy() * line.impedance
        line.advance_electric()
        actual.append(np.r_[before, line.electric[ei].numpy(), h])
    np.testing.assert_array_equal(actual, expected)

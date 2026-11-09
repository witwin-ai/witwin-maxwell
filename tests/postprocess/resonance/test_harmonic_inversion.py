import math

import pytest
import torch

from witwin.maxwell.postprocess import ResonanceModes, find_resonances, harmonic_inversion
from witwin.maxwell.postprocess.resonance import _fdm_matrices

DT = 1e-11
NUM_SAMPLES = 4000


def _times(num_samples=NUM_SAMPLES, dt=DT):
    return torch.arange(num_samples, dtype=torch.float64) * dt


def _real_signal(modes, t):
    """Sum of ``A exp(-gamma t) cos(2 pi f t - phi)`` terms."""
    return sum(
        amplitude * torch.exp(-decay * t) * torch.cos(2.0 * math.pi * frequency * t - phase)
        for frequency, decay, amplitude, phase in modes
    )


def test_closed_form_matrices_match_direct_double_sum():
    torch.manual_seed(0)
    num_samples = 41
    half_length = (num_samples - 3) // 2
    signal = torch.randn(num_samples, dtype=torch.complex128)
    phi = torch.linspace(0.3, 0.9, 6, dtype=torch.float64)
    matrices, projection = _fdm_matrices(signal, phi, half_length)

    n = torch.arange(half_length + 1, dtype=torch.float64)
    powers = torch.exp(1j * phi[:, None] * n[None, :])
    for p, matrix in enumerate(matrices):
        hankel = torch.stack([signal[i + p : i + p + half_length + 1] for i in range(half_length + 1)])
        direct = powers @ hankel @ powers.T
        torch.testing.assert_close(matrix, direct, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(projection, powers @ signal[: half_length + 1], rtol=1e-12, atol=1e-12)


def test_recovers_close_high_q_and_low_q_modes_from_real_signal():
    modes = [
        (1.00e9, 1e7, 1.0, 0.3),  # Q ~ 314
        (1.02e9, 2e6, 0.5, -1.0),  # Q ~ 1602, closer than the 1/(M dt) = 50 MHz resolution
        (1.50e9, 3e8, 0.8, 2.0),  # low Q ~ 15.7
        (3.00e9, 1e5, 0.2, 0.0),  # very high Q ~ 9.4e4
    ]
    t = _times()
    result = harmonic_inversion(_real_signal(modes, t), DT, (0.5e9, 4e9))

    assert isinstance(result, ResonanceModes)
    assert len(result) == len(modes)
    assert torch.all(result.frequency[1:] > result.frequency[:-1])
    for k, (frequency, decay, amplitude, phase) in enumerate(modes):
        assert float(result.frequency[k]) == pytest.approx(frequency, rel=1e-9)
        assert float(result.decay_rate[k]) == pytest.approx(decay, rel=1e-6)
        assert float(result.quality_factor[k]) == pytest.approx(math.pi * frequency / decay, rel=1e-6)
        # A real cos(2 pi f t - phi) contributes (A / 2) exp(i phi) at +f.
        assert float(result.magnitude[k]) == pytest.approx(0.5 * amplitude, rel=1e-6)
        assert float(result.phase[k]) == pytest.approx(phase, abs=1e-6)
        assert float(result.error[k]) < 1e-9
    torch.testing.assert_close(result.reconstruct(t, real=True), _real_signal(modes, t), rtol=0, atol=1e-6)


def test_complex_signal_resolves_signed_frequencies():
    t = _times()
    signal = 2.0 * torch.exp(-1j * 2.0 * math.pi * 1.2e9 * t - 5e6 * t + 0.4j)
    signal = signal + 0.7 * torch.exp(1j * 2.0 * math.pi * 0.8e9 * t - 2e7 * t)

    result = harmonic_inversion(signal, DT, (-1.5e9, 1.5e9))

    assert len(result) == 2
    assert float(result.frequency[0]) == pytest.approx(-0.8e9, rel=1e-9)
    assert float(result.frequency[1]) == pytest.approx(1.2e9, rel=1e-9)
    assert float(result.decay_rate[1]) == pytest.approx(5e6, rel=1e-6)
    assert complex(result.amplitude[1]) == pytest.approx(2.0 * complex(math.cos(0.4), math.sin(0.4)), rel=1e-7)
    assert float(result.magnitude[0]) == pytest.approx(0.7, rel=1e-7)
    torch.testing.assert_close(result.reconstruct(t), signal, rtol=0, atol=1e-7)

    # Only the positive member of a real signal's conjugate pair lies in a positive window;
    # the same real signal with a symmetric window yields both members with conjugate amplitudes.
    real = _real_signal([(1.0e9, 1e7, 1.0, 0.5)], t)
    pair = harmonic_inversion(real, DT, (-2e9, 2e9))
    assert len(pair) == 2
    assert float(pair.frequency[0]) == pytest.approx(-1.0e9, rel=1e-9)
    assert complex(pair.amplitude[0]) == pytest.approx(complex(pair.amplitude[1]).conjugate(), rel=1e-7)


def test_noisy_signal_keeps_physical_modes_with_amplitude_threshold():
    modes = [(1.00e9, 2e7, 1.0, 0.3), (1.40e9, 5e7, 0.6, -0.5), (2.50e9, 1e8, 0.8, 1.0)]
    t = _times()
    generator = torch.Generator().manual_seed(1234)
    noise = 1e-3 * torch.randn(NUM_SAMPLES, dtype=torch.float64, generator=generator)
    signal = _real_signal(modes, t) + noise

    raw = harmonic_inversion(signal, DT, (0.5e9, 3e9), max_error=1.0)
    assert len(raw) > len(modes)  # noise produces spurious low-amplitude modes

    # Either the amplitude threshold or a strict error threshold isolates the physical modes.
    for options in ({"min_relative_amplitude": 0.05, "max_error": 1.0}, {"max_error": 1e-5}):
        result = harmonic_inversion(signal, DT, (0.5e9, 3e9), **options)
        assert len(result) == len(modes), options
        for k, (frequency, decay, amplitude, _) in enumerate(modes):
            assert float(result.frequency[k]) == pytest.approx(frequency, rel=1e-5)
            q_exact = math.pi * frequency / decay
            assert float(result.quality_factor[k]) == pytest.approx(q_exact, rel=1e-2)
            assert float(result.magnitude[k]) == pytest.approx(0.5 * amplitude, rel=1e-2)


def test_window_edges_select_modes_inside_the_window():
    modes = [(0.9e9, 1e7, 1.0, 0.0), (1.2e9, 1e7, 1.0, 0.0), (1.6e9, 1e7, 1.0, 0.0)]
    signal = _real_signal(modes, _times())

    centered = harmonic_inversion(signal, DT, (1.0e9, 1.4e9))
    assert len(centered) == 1
    assert float(centered.frequency[0]) == pytest.approx(1.2e9, rel=1e-8)

    # A mode just inside an edge is still extracted accurately despite the
    # strong neighbour just outside the other edge.
    edge = harmonic_inversion(signal, DT, (1.19e9, 1.59e9))
    assert len(edge) == 1
    assert float(edge.frequency[0]) == pytest.approx(1.2e9, rel=1e-6)
    assert float(edge.decay_rate[0]) == pytest.approx(1e7, rel=1e-2)

    # A window between modes reports nothing rather than out-of-window eigenvalues.
    empty = harmonic_inversion(signal, DT, (1.3e9, 1.5e9))
    assert len(empty) == 0
    assert empty.frequency.shape == (0,)


def test_output_is_deterministic_and_device_preserving():
    signal = _real_signal([(1.0e9, 1e7, 1.0, 0.3), (1.05e9, 3e7, 0.4, 1.0)], _times())
    first = harmonic_inversion(signal, DT, (0.5e9, 1.5e9))
    second = harmonic_inversion(signal.clone(), DT, (0.5e9, 1.5e9))
    for name in ("frequency", "decay_rate", "quality_factor", "amplitude", "error"):
        assert torch.equal(getattr(first, name), getattr(second, name))
        assert getattr(first, name).device == signal.device
    assert first.amplitude.dtype == torch.complex128
    assert first.frequency.dtype == torch.float64


def test_amplitudes_refer_to_t0():
    modes = [(1.0e9, 2e7, 1.0, 0.3)]
    t = _times()
    t0 = 7e-9
    shifted = _real_signal(modes, t + t0)
    result = harmonic_inversion(shifted, DT, (0.5e9, 1.5e9), t0=t0)
    expected = 0.5 * math.exp(-2e7 * t0) * complex(math.cos(0.3 - 2 * math.pi * 1e9 * t0), math.sin(0.3 - 2 * math.pi * 1e9 * t0))
    assert complex(result.amplitude[0]) == pytest.approx(expected, rel=1e-7)
    torch.testing.assert_close(result.reconstruct(t + t0, real=True), shifted, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frequency_window": (2e9, 1e9)}, "fmin < fmax"),
        ({"frequency_window": (1e9, 6e10)}, "Nyquist"),
        ({"dt": 0.0}, "dt must be positive"),
    ],
)
def test_rejects_invalid_arguments(kwargs, message):
    arguments = {"signal": torch.ones(100), "dt": DT, "frequency_window": (1e9, 2e9)}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        harmonic_inversion(**arguments)


def _field_time_payload(buffers, t):
    first = next(iter(buffers.values()))
    payload = {
        "kind": "field_time",
        "differentiable": False,
        "name": "probe",
        "t": t,
        "components": dict(buffers),
        "fields": tuple(buffers),
        "start": 0,
        "stop": None,
        "interval": 1,
    }
    if len(buffers) == 1:
        payload["data"] = first
        payload["field"] = first
    return payload


def test_find_resonances_on_point_payload_discards_transient():
    t = _times()
    ringdown = _real_signal([(1.1e9, 1e7, 1.0, 0.2)], t)
    start_time = 5e-9
    # A strong broadband "source-on" transient before start_time.
    transient = torch.where(t < start_time, 50.0 * torch.sin(2 * math.pi * 1.25e9 * t), torch.zeros_like(t))
    payload = _field_time_payload({"Ez": ringdown + transient}, t)

    polluted = find_resonances(payload, (0.8e9, 1.5e9), max_error=1.0)
    clean = find_resonances(payload, (0.8e9, 1.5e9), max_error=1.0, start_time=start_time)

    assert len(clean) == 1
    assert clean.t0 == pytest.approx(start_time, abs=0.5 * DT)
    assert float(clean.frequency[0]) == pytest.approx(1.1e9, rel=1e-9)
    assert float(clean.decay_rate[0]) == pytest.approx(1e7, rel=1e-6)
    # Fitting through the source-on transient adds broadband spurious modes,
    # which the default error threshold then rejects.
    assert len(polluted) > 1
    default = find_resonances(payload, (0.8e9, 1.5e9))
    assert len(default) == 1
    assert float(default.frequency[0]) == pytest.approx(1.1e9, rel=1e-9)


def test_find_resonances_spatial_selection_on_plane_payload():
    t = _times()
    # Two modes with odd/even spatial profiles across a 4 x 3 plane.
    odd_profile = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float64)[:, None].expand(4, 3)
    even_profile = torch.ones((4, 3), dtype=torch.float64)
    odd = _real_signal([(1.0e9, 1e7, 1.0, 0.0)], t)
    even = _real_signal([(1.3e9, 2e7, 1.0, 0.0)], t)
    field = odd[:, None, None] * odd_profile + even[:, None, None] * even_profile
    payload = _field_time_payload({"Ex": field, "Hy": 0.1 * field}, t)

    window = (0.8e9, 1.5e9)
    summed = find_resonances(payload, window, component="ex", reduce="sum")
    assert len(summed) == 1  # the odd mode cancels in the plane sum
    assert float(summed.frequency[0]) == pytest.approx(1.3e9, rel=1e-9)
    assert float(summed.magnitude[0]) == pytest.approx(0.5 * 12, rel=1e-6)

    mean = find_resonances(payload, window, component="Ex", reduce="mean")
    assert float(mean.magnitude[0]) == pytest.approx(0.5, rel=1e-6)

    point = find_resonances(payload, window, component="Ex", sample=(3, 1))
    assert len(point) == 2
    torch.testing.assert_close(point.frequency, torch.tensor([1.0e9, 1.3e9], dtype=torch.float64), rtol=1e-9, atol=0)

    weights = odd_profile
    weighted = find_resonances(payload, window, component="Hy", weights=weights)
    assert len(weighted) == 1
    assert float(weighted.frequency[0]) == pytest.approx(1.0e9, rel=1e-9)

    with pytest.raises(ValueError, match="component is required"):
        find_resonances(payload, window, reduce="sum")
    with pytest.raises(ValueError, match="need one of"):
        find_resonances(payload, window, component="Ex")
    with pytest.raises(ValueError, match="at most one"):
        find_resonances(payload, window, component="Ex", reduce="sum", sample=(0, 0))
    with pytest.raises(KeyError):
        find_resonances(payload, window, component="Ez", reduce="sum")
    with pytest.raises(ValueError, match="FieldTimeMonitor payload"):
        find_resonances({"kind": "flux_time"}, window)

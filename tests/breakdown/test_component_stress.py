"""Golden and float64-parity tests for component rating / stress reduction.

Capability level under test: stress-only. The reduction maps recorded port
V(t)/I(t) into P = V I, cumulative integral P dt, and an exceedance summary
versus a declared ComponentRating envelope.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.breakdown_stress import ComponentRating, ComponentStressData
from witwin.maxwell.result import Result


# ---------------------------------------------------------------------------
# ComponentRating validation.
# ---------------------------------------------------------------------------


def test_rating_requires_at_least_one_limit():
    with pytest.raises(ValueError):
        ComponentRating()


def test_rating_rejects_nonpositive_limits():
    with pytest.raises(ValueError):
        ComponentRating(voltage=-1.0)
    with pytest.raises(ValueError):
        ComponentRating(energy=0.0)


def test_rating_records_model_string():
    rating = ComponentRating(voltage=100.0, model="tvs-diode-rev2")
    assert rating.model == "tvs-diode-rev2"
    assert rating.as_dict()["current"] is None


# ---------------------------------------------------------------------------
# Golden reduction.
# ---------------------------------------------------------------------------


def test_power_and_energy_golden():
    t = torch.linspace(0.0, 4.0, 5, dtype=torch.float64)
    v = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0], dtype=torch.float64)
    i = torch.tensor([0.0, 2.0, 4.0, 2.0, 0.0], dtype=torch.float64)
    rating = ComponentRating(voltage=1.5, current=10.0, energy=10.0, model="demo")
    data = ComponentStressData.from_time_series(t, v, i, rating, name="c", port_name="p")

    power = (v * i).numpy()
    assert np.allclose(data.power.numpy(), power)
    assert data.peak_power == pytest.approx(8.0)
    assert data.peak_voltage == pytest.approx(2.0)
    assert data.peak_current == pytest.approx(4.0)
    # Trapezoidal integral of P over t.
    expected_energy = np.trapezoid(power, t.numpy())
    assert data.total_energy == pytest.approx(expected_energy)
    assert float(data.cumulative_energy[-1]) == pytest.approx(expected_energy)


def test_exceedance_summary_flags():
    t = torch.linspace(0.0, 1e-8, 11, dtype=torch.float64)
    v = torch.full((11,), 5.0, dtype=torch.float64)
    i = torch.full((11,), 2.0, dtype=torch.float64)
    rating = ComponentRating(voltage=3.0, current=5.0, energy=1e-9, model="demo")
    data = ComponentStressData.from_time_series(t, v, i, rating, name="c", port_name="p")
    assert data.exceedance["voltage"]["exceeded"] is True
    assert data.exceedance["voltage"]["margin"] == pytest.approx(2.0)
    assert data.exceedance["current"]["exceeded"] is False
    # total energy = 10 W * 1e-8 s = 1e-7 J > 1e-9 J rating.
    assert data.exceedance["energy"]["exceeded"] is True
    assert data.any_exceeded is True


def test_disabled_rating_channel_reports_no_exceedance():
    t = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
    v = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    i = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    rating = ComponentRating(voltage=0.5)  # only a voltage limit
    data = ComponentStressData.from_time_series(t, v, i, rating, name="c", port_name="p")
    assert data.exceedance["current"]["rated"] is None
    assert data.exceedance["current"]["exceeded"] is False


def test_requires_increasing_time():
    t = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        ComponentStressData.from_time_series(
            t, torch.zeros(3), torch.zeros(3), ComponentRating(voltage=1.0)
        )


# ---------------------------------------------------------------------------
# Device float32 vs CPU float64 parity.
# ---------------------------------------------------------------------------


def _reference_float64(t, v, i):
    tt = np.asarray(t, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    ii = np.asarray(i, dtype=np.float64)
    power = vv * ii
    energy = np.trapezoid(power, tt)
    return float(power.max()), float(energy), float(np.abs(vv).max()), float(np.abs(ii).max())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_reduction_matches_float64_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    rng = np.random.default_rng(0)
    n = 257
    t_np = np.sort(rng.uniform(0.0, 1e-7, n))
    t_np[0] = 0.0
    t_np = np.unique(t_np)
    n = t_np.size
    v_np = rng.uniform(-20.0, 20.0, n)
    i_np = rng.uniform(-3.0, 3.0, n)

    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    i = torch.tensor(i_np, dtype=torch.float32, device=device)
    rating = ComponentRating(voltage=15.0, current=2.5, energy=1e-9, model="demo")
    data = ComponentStressData.from_time_series(t, v, i, rating)

    ref_peak_power, ref_energy, ref_peak_v, ref_peak_i = _reference_float64(t_np, v_np, i_np)
    scale = max(abs(ref_peak_power), 1.0)
    assert data.peak_power == pytest.approx(ref_peak_power, rel=1e-4, abs=1e-6 * scale)
    assert data.total_energy == pytest.approx(ref_energy, rel=1e-3, abs=1e-9 * abs(ref_energy) + 1e-15)
    assert data.peak_voltage == pytest.approx(ref_peak_v, rel=1e-5)
    assert data.peak_current == pytest.approx(ref_peak_i, rel=1e-5)


# ---------------------------------------------------------------------------
# Result.component_stress time-axis validation.
# ---------------------------------------------------------------------------


def _stress_result(v_time, i_time):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.5),
        device="cpu",
    )
    scene.add_monitor(
        mw.ComponentStressMonitor(
            "part",
            port="feed",
            rating=ComponentRating(voltage=100.0),
            voltage_series="vprobe",
            current_series="iprobe",
        )
    )
    n = v_time.numel()
    monitors = {
        "vprobe": {"t": v_time, "field": torch.ones(n, dtype=torch.float64)},
        "iprobe": {"t": i_time, "field": torch.full((n,), 2.0, dtype=torch.float64)},
    }
    return Result(method="fdtd", scene=scene, frequency=1e9, monitors=monitors)


def test_component_stress_reduces_on_shared_time_axis():
    t = torch.linspace(0.0, 1e-8, 6, dtype=torch.float64)
    result = _stress_result(t, t.clone())
    data = result.component_stress("part")
    # P = V*I = 1*2 = 2 everywhere on a shared axis.
    assert float(data.power[0]) == pytest.approx(2.0)


def test_component_stress_rejects_mismatched_time_axes():
    # Same sample count but different time axes must fail closed, not silently
    # adopt the voltage axis.
    v_time = torch.linspace(0.0, 1e-8, 6, dtype=torch.float64)
    i_time = torch.linspace(0.0, 2e-8, 6, dtype=torch.float64)
    result = _stress_result(v_time, i_time)
    with pytest.raises(ValueError, match="different time axes"):
        result.component_stress("part")


def test_falsify_rectangle_energy_would_break_trapezoid_parity():
    # A non-uniform ramp where the trapezoidal and left-rectangle integrals differ,
    # documenting that the energy contract is specifically trapezoidal.
    t = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    v = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    i = torch.ones(3, dtype=torch.float64)
    data = ComponentStressData.from_time_series(t, v, i, ComponentRating(voltage=10.0))
    trapezoid = np.trapezoid((v * i).numpy(), t.numpy())
    left_rectangle = float(((v * i)[:-1] * (t[1:] - t[:-1])).sum())
    assert data.total_energy == pytest.approx(trapezoid)
    assert trapezoid != pytest.approx(left_rectangle)

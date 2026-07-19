"""Construction and validation of the public BreakdownMonitor / stress monitors.

Capability level under test: stress-only declarative monitor objects.
"""

from __future__ import annotations

import pytest

import witwin.maxwell as mw
from witwin.maxwell.breakdown_stress import ComponentRating


def test_breakdown_monitor_requires_critical_field():
    with pytest.raises(ValueError):
        mw.BreakdownMonitor("m", position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))


def test_breakdown_monitor_validates_geometry_and_quantities():
    with pytest.raises(ValueError):
        mw.BreakdownMonitor("m", position=(0, 0, 0), size=(0.0, 1.0, 1.0), critical_field=1e6)
    with pytest.raises(ValueError):
        mw.BreakdownMonitor(
            "m", position=(0, 0, 0), size=(1, 1, 1), critical_field=1e6, quantities=("bogus",)
        )
    with pytest.raises(ValueError):
        # 'damage' requires a damage_exponent.
        mw.BreakdownMonitor(
            "m", position=(0, 0, 0), size=(1, 1, 1), critical_field=1e6,
            quantities=("electric_field", "damage"),
        )


def test_breakdown_monitor_stores_bounds_and_thresholds():
    m = mw.BreakdownMonitor(
        "insulator",
        position=(0.0, 0.0, 0.0),
        size=(2.0, 4.0, 6.0),
        critical_field=2.2e7,
        minimum_duration=2e-9,
        damage_exponent=3.0,
        quantities=("electric_field", "exposure", "damage"),
    )
    assert m.bounds == ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
    assert m.critical_field == pytest.approx(2.2e7)
    assert m.minimum_duration == pytest.approx(2e-9)
    assert m.damage_exponent == pytest.approx(3.0)
    assert m.kind == "breakdown"


def test_breakdown_monitor_accepts_region_with_bounds():
    class _Region:
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    m = mw.BreakdownMonitor("m", _Region(), critical_field=1e6)
    assert m.size == (2.0, 2.0, 2.0)
    assert m.position == (0.0, 0.0, 0.0)


def test_breakdown_monitor_rejects_region_and_explicit_box():
    class _Region:
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    with pytest.raises(ValueError):
        mw.BreakdownMonitor("m", _Region(), position=(0, 0, 0), size=(1, 1, 1), critical_field=1e6)


def test_component_stress_monitor_binds_rating_and_series():
    rating = ComponentRating(voltage=100.0, energy=1e-6, model="tvs")
    monitor = mw.ComponentStressMonitor(
        "diode",
        port="feed",
        rating=rating,
        voltage_series="vprobe",
        current_series="iprobe",
    )
    assert monitor.port == "feed"
    assert monitor.voltage_series == "vprobe"
    assert monitor.current_series == "iprobe"
    assert monitor.kind == "component_stress"


def test_component_stress_monitor_requires_rating_type():
    with pytest.raises(TypeError):
        mw.ComponentStressMonitor(
            "diode", port="feed", rating="not-a-rating",
            voltage_series="v", current_series="i",
        )


def test_monitors_attach_to_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.2),
        device="cpu",
    )
    scene.add_monitor(
        mw.BreakdownMonitor("stress", position=(0, 0, 0), size=(1, 1, 1), critical_field=1e6)
    )
    assert any(getattr(m, "name", None) == "stress" for m in scene.monitors)

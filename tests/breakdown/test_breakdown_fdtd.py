"""End-to-end FDTD integration for BreakdownMonitor and ComponentStressMonitor.

Capability level under test: stress-only. A BreakdownMonitor accumulates on
device during a standard FDTD run without perturbing the field solve, and a
ComponentStressMonitor reduces recorded port time series into a rating envelope
check. Requires CUDA for the native FDTD runtime.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.breakdown_stress import BreakdownStressData, ComponentRating


def _base_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.03),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.UniformCurrentSource(
            size=(0.06, 0.06, 0.06),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=5e8),
            center=(0.0, 0.0, 0.0),
        )
    )
    return scene


def _run(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=120),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_breakdown_monitor_produces_device_stress_maps_and_provenance():
    scene = _base_scene()
    scene.add_monitor(
        mw.BreakdownMonitor(
            "stress",
            position=(0.0, 0.0, 0.0),
            size=(0.12, 0.12, 0.12),
            critical_field=1e-3,
            minimum_duration=2e-11,
            damage_exponent=2.0,
            quantities=("electric_field", "exposure", "dissipated_energy", "damage"),
        )
    )
    result = _run(scene)

    assert result.breakdown_names() == ("stress",)
    data = result.breakdown("stress")
    assert isinstance(data, BreakdownStressData)
    assert data.capability_level == "stress-only"
    assert data.peak_field > 0.0
    assert data.exceedance_duration > 0.0
    # Per-cell maps stay on the GPU.
    assert data.max_field_map.device.type == "cuda"
    assert data.exceedance_time_map.device.type == "cuda"
    assert data.damage_map is not None
    # Provenance records thresholds, colocation, capability, model version.
    prov = data.provenance
    assert prov["critical_field"] == pytest.approx(1e-3)
    assert prov["capability_level"] == "stress-only"
    assert "energy-consistent" in prov["colocation"]
    assert prov["model_version"]
    assert prov["quantities"] == ("electric_field", "exposure", "dissipated_energy", "damage")
    # Peak index is inside the region map.
    assert len(data.peak_index) == 3
    for axis, extent in zip(data.peak_index, data.max_field_map.shape):
        assert 0 <= axis < extent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_breakdown_monitor_does_not_perturb_fields():
    # Zero-impact guard: the same scene with and without a BreakdownMonitor must
    # produce bitwise-identical fields (non-feedback observer, no per-step effect).
    baseline = _run(_base_scene())
    scene = _base_scene()
    scene.add_monitor(
        mw.BreakdownMonitor(
            "stress", position=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.12), critical_field=1e-3
        )
    )
    monitored = _run(scene)

    assert baseline.breakdown_names() == ()
    b, m = baseline.fields, monitored.fields
    assert set(b) == set(m)
    for key in b:
        assert np.array_equal(
            b[key].detach().cpu().numpy(), m[key].detach().cpu().numpy()
        ), f"BreakdownMonitor perturbed field component {key}"


def _pec_box(name, position, size):
    return mw.Structure(
        geometry=mw.Box(position=position, size=size),
        material=mw.Material.pec(),
        name=name,
    )


def _terminal_stress_scene():
    # Aligned two-terminal geometry (footprint edges + reference plane on the Yee
    # half-grid for a 0.02 m uniform grid) so the bound TerminalPort compiles.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
    )
    scene.add_structure(_pec_box("signal", (0.0, 0.0, 0.06), (0.10, 0.10, 0.04)))
    scene.add_structure(_pec_box("ground", (0.0, 0.0, -0.06), (0.10, 0.10, 0.04)))
    scene.add_port(
        mw.TerminalPort(
            "feed",
            mw.TerminalRef("signal"),
            mw.TerminalRef("ground"),
            mw.AxisPath("z"),
            0.01,
        )
    )
    scene.add_source(
        mw.UniformCurrentSource(
            size=(0.06, 0.06, 0.04),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=5e8),
            center=(0.0, 0.0, 0.0),
        )
    )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_component_stress_reduction_matches_float64_reference_on_run():
    scene = _terminal_stress_scene()
    # Two point time series used as V(t) and I(t) proxies for the reduction; the
    # ComponentStressMonitor binds a real scene port ("feed").
    scene.add_monitor(
        mw.FieldTimeMonitor("vprobe", components=("Ez",), position=(0.0, 0.0, 0.0), interval=1)
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("iprobe", components=("Ez",), position=(0.02, 0.0, 0.0), interval=1)
    )
    rating = ComponentRating(voltage=1e-2, current=1e-2, energy=1e-6, model="demo-rev1")
    scene.add_monitor(
        mw.ComponentStressMonitor(
            "part", port="feed", rating=rating, voltage_series="vprobe", current_series="iprobe"
        )
    )
    result = _run(scene)

    data = result.component_stress("part")
    assert data.name == "part"
    assert data.port_name == "feed"
    assert data.capability_level == "stress-only"

    # Float64 reference recomputation from the same recorded samples.
    t = result.monitor("vprobe")["t"].detach().cpu().numpy().astype(np.float64)
    v = result.monitor("vprobe")["field"].detach().cpu().numpy().reshape(len(t), -1)[:, 0].astype(np.float64)
    i = result.monitor("iprobe")["field"].detach().cpu().numpy().reshape(len(t), -1)[:, 0].astype(np.float64)
    power = v * i
    ref_peak_power = float(power.max())
    ref_energy = float(np.trapezoid(power, t))
    ref_peak_v = float(np.abs(v).max())
    ref_peak_i = float(np.abs(i).max())

    scale = max(abs(ref_peak_power), 1e-30)
    assert data.peak_power == pytest.approx(ref_peak_power, rel=1e-4, abs=1e-5 * scale)
    assert data.total_energy == pytest.approx(ref_energy, rel=1e-3, abs=1e-6 * abs(ref_energy) + 1e-30)
    assert data.peak_voltage == pytest.approx(ref_peak_v, rel=1e-5)
    assert data.peak_current == pytest.approx(ref_peak_i, rel=1e-5)
    assert set(data.exceedance) == {"voltage", "current", "energy", "pulse_width"}

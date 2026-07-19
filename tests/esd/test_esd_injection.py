"""Terminal injection and result plumbing for ESD current sources.

Capability level under test: stress-only ideal current injection into a terminal
port, plus provenance / port V-I plumbing on the run result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


def _pec_box(name, position, size):
    return mw.Structure(
        geometry=mw.Box(position=position, size=size),
        material=mw.Material.pec(),
        name=name,
    )


def _terminal_scene(*, device="cpu", boundary=None):
    return mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none() if boundary is None else boundary,
        structures=(
            _pec_box("signal", (0.5, 0.5, 0.8), (0.5, 0.5, 0.2)),
            _pec_box("ground", (0.5, 0.5, 0.2), (0.5, 0.5, 0.2)),
        ),
        ports=(
            mw.TerminalPort(
                "feed",
                mw.TerminalRef("signal"),
                mw.TerminalRef("ground"),
                mw.AxisPath("z"),
                0.45,
            ),
        ),
        device=device,
    )


# ---------------------------------------------------------------------------
# Construction and validation.
# ---------------------------------------------------------------------------


def test_current_source_validates_inputs():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    with pytest.raises(ValueError):
        mw.ESDCurrentSource("", port="feed", waveform=waveform)
    with pytest.raises(ValueError):
        mw.ESDCurrentSource("gun", port="", waveform=waveform)
    with pytest.raises(TypeError):
        mw.ESDCurrentSource("gun", port="feed", waveform="not-a-waveform")
    with pytest.raises(ValueError):
        mw.ESDCurrentSource("gun", port="feed", waveform=waveform, direction="up")


def test_current_source_requires_existing_terminal_port():
    scene = _terminal_scene()
    esd = mw.ESDCurrentSource("gun", port="missing", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0))
    with pytest.raises(ValueError):
        esd.injection_geometry(scene)


def test_current_source_rejects_non_terminal_port():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.2),
        ports=(mw.ModePort("m", position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.0)),),
        device="cpu",
    )
    esd = mw.ESDCurrentSource("gun", port="m", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0))
    with pytest.raises(ValueError):
        esd.injection_geometry(scene)


# ---------------------------------------------------------------------------
# Geometry resolution and lowering.
# ---------------------------------------------------------------------------


def test_injection_geometry_matches_port_gap():
    scene = _terminal_scene()
    esd = mw.ESDCurrentSource("gun", port="feed", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0))
    geometry = esd.injection_geometry(scene)
    assert geometry["axis"] == "z"
    assert geometry["footprint_area"] == pytest.approx(0.25)
    assert geometry["gap"] == pytest.approx(0.4)
    assert geometry["polarization_sign"] == pytest.approx(1.0)
    assert geometry["center"][2] == pytest.approx(0.5)


def test_direction_flips_polarization_sign():
    scene = _terminal_scene()
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    positive = mw.ESDCurrentSource("gun", port="feed", waveform=waveform, direction="+")
    negative = mw.ESDCurrentSource("gun", port="feed", waveform=waveform, direction="-")
    assert positive.injection_geometry(scene)["polarization_sign"] == pytest.approx(1.0)
    assert negative.injection_geometry(scene)["polarization_sign"] == pytest.approx(-1.0)


def test_resolve_lowers_to_uniform_current_source_over_gap():
    scene = _terminal_scene()
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    esd = mw.ESDCurrentSource("gun", port="feed", waveform=waveform)
    scene.add_source(esd)

    resolved = scene.resolved_sources()
    assert len(resolved) == 1
    lowered = resolved[0]
    assert isinstance(lowered, mw.UniformCurrentSource)
    assert lowered.kind == "uniform_current"
    assert lowered.name == "gun::current"
    assert lowered.polarization == pytest.approx((0.0, 0.0, 1.0))

    # The tabulated current density integrates to the target port current: the
    # source-time amplitudes are the target current divided by the footprint area.
    source_time = lowered.source_time
    area = esd.injection_geometry(scene)["footprint_area"]
    table_times = torch.as_tensor(source_time.times, dtype=torch.float64)
    table_values = torch.as_tensor(source_time.amplitudes, dtype=torch.float64)
    target = waveform.current(table_times)
    assert torch.allclose(table_values * area, target, rtol=1e-9, atol=1e-9)


def test_resolved_sources_compile_for_fdtd():
    from witwin.maxwell.compiler.sources import compile_fdtd_sources

    scene = _terminal_scene()
    scene.add_source(mw.ESDCurrentSource("gun", port="feed", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0)))
    compiled = compile_fdtd_sources(scene, default_frequency=1e9)
    assert len(compiled) == 1
    assert compiled[0]["kind"] == "uniform_current"
    assert compiled[0]["source_time"]["kind"] == "custom"


def test_scene_without_esd_is_unaffected():
    scene = _terminal_scene()
    scene.add_source(
        mw.UniformCurrentSource(
            size=(0.2, 0.2, 0.2),
            polarization="Ez",
            source_time=mw.CW(frequency=1e9),
            center=(0.5, 0.5, 0.5),
            name="plain",
        )
    )
    resolved = scene.resolved_sources()
    assert len(resolved) == 1
    assert resolved[0].name == "plain"
    assert resolved[0] is scene.sources[0]


# ---------------------------------------------------------------------------
# End-to-end FDTD run and result plumbing (CUDA).
# ---------------------------------------------------------------------------


def _to_numpy(value):
    tensor = torch.as_tensor(value)
    return tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Target vs measured port record exposure (CPU, no FDTD run).
# ---------------------------------------------------------------------------


def _esd_result(*, ports=None):
    from witwin.maxwell.result import Result

    scene = _terminal_scene()
    scene.add_source(
        mw.ESDCurrentSource("gun", port="feed", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0))
    )
    # Provide run dt / time_steps so the charge-conserving projection is built.
    metadata = {"dt": 2e-10, "time_steps": 200}
    return Result(method="fdtd", scene=scene, frequency=1e9, metadata=metadata, ports=ports)


def test_esd_waveform_exposes_target_and_no_measured_for_ideal_injection():
    # Ideal-current injection lowers to a volumetric source and runs no
    # terminal-port recorder, so the measured record is None by design; the
    # injected current on the run grid is the resampled (target) projection.
    record = _esd_result().esd_waveform("gun")
    assert record.resampled is not None
    assert record.target_currents is not None
    assert record.measured is None


def test_esd_waveform_exposes_measured_port_record_when_recorded():
    # When the run recorded terminal-port V/I (RF PortData) for the bound port,
    # esd_waveform surfaces it as the measured record for a target-vs-measured
    # comparison.
    from witwin.maxwell.network import PortData

    port = PortData(
        port_name="feed",
        frequencies=torch.tensor([1e9]),
        voltage=torch.tensor([1.0 + 0.0j]),
        current=torch.tensor([0.02 + 0.0j]),
        z0=50.0,
    )
    record = _esd_result(ports={"feed": port}).esd_waveform("gun")
    assert record.measured is port


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_esd_terminal_injection_drives_causal_transient_and_records_provenance():
    waveform = mw.ESDWaveform.iec_61000_4_2(8000.0)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
    )
    # Terminal footprints and the reference plane must land on the Yee half-grid
    # (edges at +-0.05, reference plane at +0.01 for a 0.02 m uniform grid).
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
    scene.add_source(mw.ESDCurrentSource("gun", port="feed", waveform=waveform))
    scene.add_monitor(
        mw.FieldTimeMonitor("vprobe", components=("Ez",), position=(0.0, 0.0, 0.0), interval=1)
    )

    # ~60 ns of run time resolves the ESD pulse tail on the FDTD grid.
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=1600),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()

    # Provenance and diagnostics are exposed on the result.
    record = result.esd_waveform("gun")
    assert record.port_name == "feed"
    assert record.provenance["waveform"]["standard_revision"] == "ed2-contact"
    assert record.provenance["capability_level"] == "stress-only"
    assert record.provenance["injection"] == "ideal_current"
    analytic = waveform.diagnostics()
    assert record.diagnostics.peak_current == pytest.approx(analytic.peak_current, rel=1e-9)

    # The charge-conserving projection the solver injects reproduces the target
    # charge (the "injected current tracks the target" contract on the run grid).
    assert record.resampled is not None
    assert record.resampled.charge_ratio == pytest.approx(1.0, rel=1e-3)

    # The gap voltage is the time integral of the injected (ideal) current: an
    # ideal current source feeding the capacitive gap gives V(t) = Q(t)/C, so the
    # measured field's time derivative tracks the target current, dV/dt ~ I(t).
    payload = result.monitor("vprobe")
    times = _to_numpy(payload["t"]).reshape(-1)
    field = _to_numpy(payload["field"]).reshape(len(times), -1)[:, 0]
    magnitude = np.abs(field)

    # Causal onset and monotone charge accumulation toward a plateau.
    assert field[0] == 0.0
    assert magnitude.max() > 0.0
    assert magnitude[-1] == pytest.approx(magnitude.max(), rel=1e-6)

    # dV/dt tracks the target current: its peak aligns with the ESD current peak
    # and its early-window profile correlates strongly with the target waveform.
    slope = np.gradient(field, times)
    analytic = waveform.diagnostics()
    slope_peak_time = times[int(np.abs(slope).argmax())]
    assert abs(slope_peak_time - analytic.peak_time) < 3.0e-9

    window = slice(0, 300)
    target = _to_numpy(waveform.current(torch.as_tensor(times[window], dtype=torch.float64)))
    measured = slope[window]
    correlation = np.corrcoef(
        measured / np.abs(measured).max(),
        target / np.abs(target).max(),
    )[0, 1]
    # Sign-agnostic tracking tolerance (documented): |rho| >= 0.9 over the rise
    # and early tail. The absolute-amplitude gap-capacitance calibration is not
    # asserted here (Phase-1 stress-only scope).
    assert abs(correlation) >= 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_plain_scene_result_has_no_esd_waveforms():
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
            source_time=mw.CW(frequency=1e9),
            center=(0.0, 0.0, 0.0),
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=40),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).run()
    assert result.esd_waveform_names() == ()
    with pytest.raises(KeyError):
        result.esd_waveform("gun")

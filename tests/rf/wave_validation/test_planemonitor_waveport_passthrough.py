"""User-declared PlaneMonitors must ride through WavePort / PortSweep Results.

Gate taxonomy (S0.3): **wave-level** (field-level plumbing + physics).

Open item closed here (``docs/reference/rf-wave-validation-2026-07-18.md`` §5):
PlaneMonitors were silently dropped from WavePort / PortSweep Results, so a normal
run could not expose the injected/propagated transverse field. The drop point was
the WavePort Result assembly (``witwin/maxwell/waveport_sweep.py``), which built
the final Result with ``monitors={}`` even though every column run already
computed the user monitors. ``compact_array_column_result`` likewise retained only
closed-surface payloads. Both now thread the user-declared monitors through (the
internal per-port ModeMonitors, tagged ``WAVEPORT_MONITOR_PREFIX``, stay hidden).

What binds:

* the monitor is PRESENT on a WavePort excitation Result (no ``KeyError``) and
  carries non-trivial field data;
* the field is PHYSICALLY CORRECT for the coax TEM mode -- the transverse ``|Ey|``
  is concentrated in the dielectric annulus and is exactly zero inside the PEC
  inner conductor and outside the PEC shield. Stale / zeroed / garbage data cannot
  satisfy this, so it is an independent correctness anchor (no reference solver);
* the SAME transverse mode profile is produced by an independent PLAIN-FDTD run of
  the same coax launched by a bare ``ModeSource`` (magnitude-profile correlation)
  -- the "correct vs a plain-FDTD run of the same scene" check the deliverable
  requires;
* a PortSweep drives one channel per column, so the monitor rides through both the
  flat top-level Result (first drive channel) AND per drive column in
  ``array_run_data.column_results``;
* falsification: reverting the passthrough (monitors dropped again) makes
  ``result.monitor(...)`` raise ``KeyError`` -- the test detects the regression.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.rf.coax_thru import (
    INNER_RADIUS,
    OUTER_RADIUS,
    PORT_X,
    coax_thru_scene,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level FDTD passthrough gate requires CUDA"
)

_DX = 0.01
_FREQ = 1.0e9
_MONITOR = "guide_plane"


def _scene_with_probe():
    scene = coax_thru_scene(dx=_DX, device="cuda")
    scene.add_monitor(mw.PlaneMonitor(_MONITOR, axis="x", position=0.0, fields=("Ey",)))
    return scene


def _radial_masks(payload, ey_shape):
    y = torch.as_tensor(payload["y"]).cpu().flatten()
    z = torch.as_tensor(payload["z"]).cpu().flatten()
    ny, nz = ey_shape
    yy, zz = torch.meshgrid(y[:ny], z[:nz], indexing="ij")
    radius = torch.sqrt(yy * yy + zz * zz)
    annulus = (radius > INNER_RADIUS) & (radius < OUTER_RADIUS)
    inner_pec = radius < 0.7 * INNER_RADIUS
    outer_pec = radius > 1.1 * OUTER_RADIUS
    return annulus, inner_pec, outer_pec


def _waveport_excitation_result():
    return mw.Simulation.fdtd(
        _scene_with_probe(),
        frequencies=(_FREQ,),
        excitations=mw.PortExcitation("left"),
        run_time=mw.TimeConfig(time_steps=1500),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()


def _plain_modesource_result():
    scene = coax_thru_scene(dx=_DX, device="cuda")
    launch = mw.ModeSource(
        position=(-PORT_X - _DX, 0.0, 0.0),
        size=(0.0, 0.40, 0.40),
        mode_index=0,
        direction="+",
        polarization="Ey",
        source_time=mw.CW(frequency=_FREQ, amplitude=1.0),
        name="tem_launch",
    )
    scene = scene.clone(
        ports=(),
        sources=(launch,),
        monitors=(mw.PlaneMonitor(_MONITOR, axis="x", position=0.0, fields=("Ey",)),),
    )
    return mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQ,),
        run_time=mw.TimeConfig(time_steps=1500),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()


@pytest.fixture(scope="module")
def waveport_result():
    return _waveport_excitation_result()


@pytest.fixture(scope="module")
def plain_result():
    return _plain_modesource_result()


def test_planemonitor_present_on_waveport_excitation(waveport_result):
    """The user PlaneMonitor is not dropped from a direct WavePort excitation."""

    assert _MONITOR in waveport_result.monitors
    payload = waveport_result.monitor(_MONITOR)
    assert "Ey" in payload
    ey = payload["Ey"].cpu()
    assert ey.ndim == 2
    assert float(ey.abs().max()) > 1.0, "monitor rode through but carries no field"


def test_planemonitor_field_is_physical_coax_tem(waveport_result):
    """The ridden-through field is the injected TEM mode: annulus-concentrated,
    exactly zero inside the PEC conductors (independent correctness anchor)."""

    payload = waveport_result.monitor(_MONITOR)
    ey = payload["Ey"].cpu()
    annulus, inner_pec, outer_pec = _radial_masks(payload, tuple(ey.shape))
    mag = ey.abs()
    annulus_mean = float(mag[annulus].mean())
    assert annulus_mean > 1.0, f"no TEM field in the coax annulus: {annulus_mean:.3e}"
    # PEC interior/exterior must be field-free (a correct FDTD run cannot leak here).
    assert float(mag[inner_pec].max()) < 1.0e-3 * annulus_mean
    assert float(mag[outer_pec].max()) < 1.0e-3 * annulus_mean


def test_planemonitor_matches_plain_modesource_run(waveport_result, plain_result):
    """Correct vs a plain-FDTD run of the same scene: an independent bare
    ModeSource launch reproduces the same transverse |Ey| mode profile."""

    payload_wp = waveport_result.monitor(_MONITOR)
    payload_ms = plain_result.monitor(_MONITOR)
    ey_wp = payload_wp["Ey"].cpu()
    ey_ms = payload_ms["Ey"].cpu()
    assert ey_wp.shape == ey_ms.shape

    annulus, _inner, _outer = _radial_masks(payload_wp, tuple(ey_wp.shape))
    # The plain run must itself be a valid TEM launch (annulus-concentrated).
    ms_mag = ey_ms.abs()
    assert float(ms_mag[annulus].mean()) > 1.0

    a = ey_wp.abs()[annulus].to(torch.float64)
    b = ey_ms.abs()[annulus].to(torch.float64)
    magnitude_corr = float((a * b).sum() / (a.norm() * b.norm()))
    # Both launches are the same coax TEM mode; the normalized magnitude-profile
    # correlation over the annulus is high (measured ~0.85 at this tier; the two
    # differ only in absolute amplitude normalization and residual standing-wave
    # ripple, not in mode shape). A loose floor keeps the gate from drifting.
    assert magnitude_corr > 0.75, f"plain-FDTD mode profile disagrees: {magnitude_corr:.3f}"


def test_planemonitor_rides_through_portsweep():
    """A PortSweep exposes the user monitor at top level (first drive) and per
    drive column in array_run_data.column_results."""

    result = mw.Simulation.fdtd(
        _scene_with_probe(),
        frequencies=(_FREQ,),
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig(time_steps=1200),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    # Flat top-level: present and tagged with the drive channel it belongs to.
    assert _MONITOR in result.monitors
    assert result._metadata.get("user_monitor_drive_channel") == "left::TEM0"
    top = result.monitor(_MONITOR)
    assert float(top["Ey"].cpu().abs().max()) > 1.0

    # Per drive column: one payload per driven channel (no silent drop).
    columns = result._array_run_data.column_results
    assert len(columns) == 2
    for column in columns:
        assert _MONITOR in column[0].monitors


def test_planemonitor_passthrough_falsification(monkeypatch):
    """Reverting the passthrough (drop the user monitors again) must make the
    monitor unavailable -- proving the gate detects the original regression."""

    import witwin.maxwell.waveport_sweep as wp

    monkeypatch.setattr(wp, "_user_monitor_payloads", lambda result, scene: {})

    result = mw.Simulation.fdtd(
        _scene_with_probe(),
        frequencies=(_FREQ,),
        excitations=mw.PortExcitation("left"),
        run_time=mw.TimeConfig(time_steps=1200),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()

    assert _MONITOR not in result.monitors
    with pytest.raises(KeyError):
        result.monitor(_MONITOR)

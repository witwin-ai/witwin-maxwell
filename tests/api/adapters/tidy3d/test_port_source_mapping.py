"""RF port / lumped-feed excitation mapping through the reference adapter.

The four owner-authorized reference scenes (``rf/coax_thru``,
``rf/lumped_open_short_match``, ``antenna/half_wave_dipole``, ``antenna/patch``)
are port-driven: they carry a ``WavePort`` TEM aperture or a ``LumpedPort``
delta-gap feed but no ``source_time``. Before this mapping they exported with
``sources == 0`` and fail-closed at the reference-generation runnable gate.

These tests pin the documented adapter drive convention implemented in
``_convert_ports_for_reference``:

* the first declared ``WavePort`` maps to a reference modal launch
  (``ModeSource``) of its fundamental mode; every ``WavePort`` additionally maps
  to a receiving ``ModeMonitor`` at its aperture;
* a ``LumpedPort`` delta-gap feed (wire-bound or coordinate-bound) maps to its
  equivalent current injection: a ``UniformCurrentSource`` electric filament
  spanning the feed gap along the voltage-path axis.

They run against the REAL tidy3d package (offline export only -- no cloud call),
so the assertions check the genuine constructed source/monitor objects.
"""

from __future__ import annotations

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")

from benchmark.scenes.antenna.half_wave_dipole import half_wave_dipole_scene
from benchmark.scenes.antenna.patch import patch_antenna_scene
from benchmark.scenes.rf.coax_thru import coax_thru_scene
from benchmark.scenes.rf.lumped_open_short_match import (
    coax_sol_scene,
    default_frequencies as sol_frequencies,
)


# --------------------------------------------------------------------------- #
# WavePort -> ModeSource (drive, port 0) + ModeMonitor (every port).          #
# --------------------------------------------------------------------------- #
def test_coax_thru_wave_ports_export_drive_and_receive_monitors():
    scene = coax_thru_scene(dx=0.01, device="cpu")
    sim = scene.to_tidy3d(frequencies=(0.6e9, 1.0e9, 1.6e9))

    # Exactly one modal drive (the first declared port, "left").
    sources = [s for s in sim.sources if isinstance(s, td.ModeSource)]
    assert len(sources) == 1
    drive = sources[0]
    assert drive.name == "left::drive"
    assert drive.direction == "+"  # "left" port launches in +x
    # The aperture plane is normal to x (zero size along the propagation axis).
    assert drive.size[0] == 0.0

    # Every wave port is a receiving mode monitor.
    monitors = [m for m in sim.monitors if isinstance(m, td.ModeMonitor)]
    assert {m.name for m in monitors} == {"left", "right"}
    for monitor in monitors:
        assert monitor.size[0] == 0.0


def test_lumped_open_short_match_single_wave_port_is_runnable():
    scene = coax_sol_scene("matched", dx=0.01, device="cpu")
    sim = scene.to_tidy3d(frequencies=sol_frequencies())

    sources = [s for s in sim.sources if isinstance(s, td.ModeSource)]
    monitors = [m for m in sim.monitors if isinstance(m, td.ModeMonitor)]
    assert len(sources) == 1
    assert len(monitors) == 1
    # Runnable gate: the previously source-less export now carries a drive.
    assert len(sim.sources) >= 1


# --------------------------------------------------------------------------- #
# LumpedPort -> equivalent current injection (UniformCurrentSource filament).  #
# --------------------------------------------------------------------------- #
def test_dipole_wire_gap_feed_exports_z_current_filament():
    scene = half_wave_dipole_scene(design_frequency=3.0e9, device="cpu")
    sim = scene.to_tidy3d(frequencies=scene.monitors[0].frequencies)

    sources = [s for s in sim.sources if isinstance(s, td.UniformCurrentSource)]
    assert len(sources) == 1
    feed = sources[0]
    assert feed.name == "feed::drive"
    # Feed axis is z (the wire-gap voltage path); transverse extents are zero.
    assert feed.polarization == "Ez"
    assert feed.size[0] == 0.0 and feed.size[1] == 0.0
    assert feed.size[2] > 0.0
    # Center sits on the gap midpoint at the origin (symmetric dipole).
    assert feed.center[0] == pytest.approx(0.0, abs=1e-6)
    assert feed.center[1] == pytest.approx(0.0, abs=1e-6)
    assert feed.center[2] == pytest.approx(0.0, abs=1e-6)
    # The NF2FF box still lowers to its six face field monitors.
    field_monitors = [m for m in sim.monitors if isinstance(m, td.FieldMonitor)]
    assert len(field_monitors) == 6


def test_patch_probe_feed_exports_z_current_filament_at_feed_inset():
    scene = patch_antenna_scene(
        frequencies=tuple(f * 1e9 for f in (4.4, 4.8, 5.2, 5.6, 6.0)),
        device="cpu",
    )
    sim = scene.to_tidy3d(frequencies=scene.monitors[0].frequencies)

    sources = [s for s in sim.sources if isinstance(s, td.UniformCurrentSource)]
    assert len(sources) == 1
    feed = sources[0]
    assert feed.polarization == "Ez"
    assert feed.size[0] == 0.0 and feed.size[1] == 0.0
    assert feed.size[2] > 0.0
    # Feed is offset from the patch center along -x (probe inset), not at x = 0.
    assert feed.center[0] < 0.0
    field_monitors = [m for m in sim.monitors if isinstance(m, td.FieldMonitor)]
    assert len(field_monitors) == 6


# --------------------------------------------------------------------------- #
# Convention guards.                                                          #
# --------------------------------------------------------------------------- #
def test_current_filament_orientation_follows_voltage_axis():
    """The filament axis is the negative->positive voltage-path axis, not fixed."""
    from witwin.maxwell.adapters.tidy3d import _lumped_port_current_source

    scene = patch_antenna_scene(
        frequencies=(5.0e9,), device="cpu"
    )
    feed_port = scene.ports[0]
    source = _lumped_port_current_source(feed_port, scene, td, 1.0e6, td.GaussianPulse(freq0=5e9, fwidth=1e9))
    # Patch voltage path is along z; the nonzero size axis must be z (index 2).
    nonzero_axis = int(np.argmax([abs(v) for v in source.size]))
    assert nonzero_axis == 2
    assert source.polarization == "Ez"


def test_port_export_requires_frequencies():
    scene = coax_thru_scene(dx=0.01, device="cpu")
    with pytest.raises(ValueError, match="frequencies"):
        scene.to_tidy3d(frequencies=None)

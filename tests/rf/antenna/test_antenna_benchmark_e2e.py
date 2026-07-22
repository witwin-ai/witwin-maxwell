"""End-to-end antenna benchmark gates through the real ``Result.antenna`` path.

These tests drive the ``benchmark/scenes/antenna`` scene builders through a real
FDTD ``Scene -> Simulation -> Result`` run and consume the near-field-to-far-field
transform via :meth:`Result.antenna` WITHOUT monkeypatching the surface currents
or the far field (contrast ``test_result_antenna.py``, which unit-tests the
reduction kernel with synthetic surfaces). The driven feed :class:`PortData` and
the ``ClosedSurfaceMonitor`` both come from the time-stepped solver.

Falsification records live in
``docs/assessments/e2-rf-scenes-acceptance-2026-07-19.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from benchmark.scenes.antenna.half_wave_dipole import (
    analytic_directivity_dbi,
    default_frequencies,
    half_wave_dipole_scene,
)
from benchmark.scenes.antenna.patch import patch_antenna_scene


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD antenna benchmarks require CUDA"
)


def _sin_squared_correlation(directivity: torch.Tensor, theta: torch.Tensor) -> float:
    reference = torch.sin(theta).square()
    pattern = directivity / directivity.max()
    reference = reference / reference.max()
    numerator = torch.sum(pattern * reference)
    denominator = torch.sqrt(torch.sum(pattern.square()) * torch.sum(reference.square()))
    return float(numerator / denominator)


def _run_dipole(design_frequency: float = 3.0e9):
    frequencies = default_frequencies(design_frequency)
    scene = half_wave_dipole_scene(
        design_frequency=design_frequency,
        frequencies=frequencies,
        device="cuda",
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=list(frequencies),
        excitations=mw.PortExcitation(
            "feed",
            source_time=mw.GaussianPulse(frequency=design_frequency, fwidth=1.5e9),
        ),
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    prepared = simulation.prepare()
    steps = math.ceil(12.0e-9 / float(prepared.solver.dt))
    simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    result = prepared.run()
    return result, frequencies


def test_half_wave_dipole_end_to_end_radiation_and_impedance():
    design_frequency = 3.0e9
    result, frequencies = _run_dipole(design_frequency)
    design_index = frequencies.index(design_frequency)

    port = result.port("feed")
    resistance = port.z_in.real

    # Radiation-resistance class: the real part sweeps THROUGH the thin-dipole
    # ~73 Ohm class within the band (min below, max above), with at least one
    # sampled frequency inside the 60-90 Ohm band. The input reactance carries a
    # large positive delta-gap feed offset (documented) and is deliberately NOT
    # gated here.
    assert float(resistance.min()) < 73.0 < float(resistance.max())
    in_band = (resistance >= 60.0) & (resistance <= 90.0)
    assert bool(torch.any(in_band))

    data = result.antenna(
        surface="radiation",
        driven_port="feed",
        theta_points=181,
        phi_points=8,
        radius=10.0,
    )
    assert isinstance(data, mw.AntennaData)
    # The real path built genuine surface currents (six NF2FF faces per frequency).
    assert data.surface_currents is not None
    assert len(data.surface_currents[design_index].surfaces) == 6

    theta = data.theta[:, 0]
    e_plane = data.directivity[design_index, :, 0]
    correlation = _sin_squared_correlation(e_plane, theta)
    assert correlation >= 0.99

    directivity_dbi = float(data.directivity_db.amax(dim=(-2, -1))[design_index])
    assert 1.9 <= directivity_dbi <= 2.4
    # Bracket the analytic thin half-wave dipole directivity (2.15 dBi).
    assert abs(directivity_dbi - analytic_directivity_dbi()) <= 0.3

    p_rad = data.p_rad[design_index]
    p_accepted = data.p_accepted[design_index]
    closure = float(torch.abs(p_rad - p_accepted) / torch.abs(p_accepted))
    assert closure < 0.08

    assert torch.all(torch.isfinite(data.realized_gain[design_index]))


# --------------------------------------------------------------------------- #
# Patch antenna: real Result.antenna pipeline over a grounded dielectric slab. #
# --------------------------------------------------------------------------- #
# The probe-fed patch exercises the NF2FF transform over a finite grounded
# substrate (the harder homogeneous-exterior case). The pipeline gate below is a
# genuine PASS; the matched-broadside TM010 physics is a DOCUMENTED GAP recorded
# as a strict xfail (it cannot silently xpass) and deferred to stage E2c.
_PATCH_FREQUENCIES = tuple(f * 1e9 for f in (4.4, 4.8, 5.2, 5.6, 6.0))


@pytest.fixture(scope="module")
def patch_antenna_run():
    scene = patch_antenna_scene(frequencies=_PATCH_FREQUENCIES, device="cuda")
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=list(_PATCH_FREQUENCIES),
        excitations=mw.PortExcitation(
            "feed",
            source_time=mw.GaussianPulse(frequency=5.2e9, fwidth=2.2e9),
        ),
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    prepared = simulation.prepare()
    steps = math.ceil(16.0e-9 / float(prepared.solver.dt))
    simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    result = prepared.run()
    data = result.antenna(
        surface="radiation",
        driven_port="feed",
        theta_points=91,
        phi_points=73,
        radius=10.0,
    )
    return result, data


def test_patch_antenna_result_antenna_pipeline_is_valid(patch_antenna_run):
    result, data = patch_antenna_run

    assert isinstance(data, mw.AntennaData)
    assert data.frequencies.numel() == len(_PATCH_FREQUENCIES)
    # NF2FF over the finite grounded slab: six air-exterior faces per frequency.
    assert data.surface_currents is not None
    for currents in data.surface_currents:
        assert len(currents.surfaces) == 6

    # Every spectral sample yields finite, physical engineering data.
    assert torch.all(torch.isfinite(data.realized_gain))
    assert torch.all(torch.isfinite(data.directivity))
    assert torch.all(data.p_rad > 0.0)
    assert torch.all(data.p_accepted > 0.0)
    assert bool(torch.any(data.directivity_db.amax(dim=(-2, -1)) > 0.0))

    # Radiated power closes against accepted power at the best-coupled sample.
    closure = torch.abs(data.p_rad - data.p_accepted) / torch.abs(data.p_accepted)
    assert float(closure.min()) < 0.05

    # The driven feed PortData rode through the run.
    assert result.port("feed").z_in.numel() == len(_PATCH_FREQUENCIES)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Probe-fed patch does not resonate: F2b diagnosis (docs/assessments/"
        "f2-rf-trio-acceptance-2026-07-21.md) shows Re(Zin) < 4 Ohm with no resonance "
        "peak across 2-8 GHz and a purely capacitive reactance -- the lumped feed couples "
        "capacitively, never galvanically exciting the TM010 cavity mode; the cavity "
        "resonance (~3.39 GHz) also sits below the driven band and the finite ground is "
        "~0.07 lambda. The galvanic via added in F2b cut the feed reactance ~5x but a "
        "matched broadside D >= 5 dBi needs a wire-bound clean-gap probe feed + larger "
        "ground + on-resonance drive (multi-run antenna co-design, deferred)."
    ),
)
def test_patch_antenna_matched_broadside_gate(patch_antenna_run):
    result, data = patch_antenna_run
    theta = data.theta[:, 0]
    broadside_index = int(torch.argmin(torch.abs(theta)))
    broadside_dbi = data.directivity_db[:, broadside_index, :].amax(dim=-1)
    reflection = result.port("feed").reflection_coefficient.abs()
    matched = 20.0 * torch.log10(reflection)
    # Physical patch gate (documented target): a broadside D >= 5 dBi lobe at a
    # frequency where the feed is matched better than -10 dB.
    assert bool(
        torch.any((broadside_dbi >= 5.0) & (matched <= -10.0))
    )

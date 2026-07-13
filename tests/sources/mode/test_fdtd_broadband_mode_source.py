"""Broadband (pulsed) ModeSource injection.

A ModeSource solves its guided transverse profile and effective index once, at the
waveform's center frequency, and drives that profile with the temporal envelope.
With a ``GaussianPulse`` waveform this excites the whole band from a single run, the
same way a pulsed ``PlaneWave`` TFSF source does. This module covers three things:

1. a ``CustomSourceTime`` mode source is rejected (the native time-shifted surface
   kernel evaluates only the analytic CW/Gaussian/Ricker forms);
2. a ``GaussianPulse`` mode source compiles and drives the native time-shifted
   surface-term path (not the CW cos/sin path), with the mode solved at the pulse
   center frequency;
3. acceptance: a single broadband ``GaussianPulse`` mode-source run injects, at every
   frequency in the band, the same guided mode a monochromatic ``CW`` mode source
   injects at that frequency.

For (3) the compared observable is the downstream transverse field profile recorded by
a ``PlaneMonitor`` one guided-mode distance past the source, and the metric is the
complex mode-overlap fidelity

    F(f) = |<E_broadband(f), E_cw(f)>| / (||E_broadband(f)|| * ||E_cw(f)||).

The overlap magnitude is invariant to a global amplitude and a global phase, so it is
immune both to the CW-vs-pulse DFT normalization and to the source<->boundary
Fabry-Perot amplitude resonances that make a single-point phasor or a raw flux fragile
in a strongly guiding waveguide. F(f) -> 1 exactly when the broadband run carries the
same transverse mode as the per-frequency CW run. This mirrors the intent of the Bloch
broadband-vs-CW acceptance in
``tests/validation/physics/test_bloch_broadband_vs_cw.py`` while using the artifact-free
observable appropriate to a guided mode.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FREQS = (0.94e9, 1.0e9, 1.06e9)
_CENTER = 1.0e9
_FWIDTH = 0.12e9
_AMPLITUDE = 20.0
_SRC_X = -0.5
_PLANE_X = 0.4
_STEPS = 2000


def _guide_scene(source_time):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
    )
    # Strongly guiding core (n_eff ~ 3.3) spanning the full propagation axis: the
    # fundamental profile is nearly frequency independent across the narrow band, so the
    # single center-frequency injection profile reproduces the per-frequency guided mode.
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(1.6, 0.16, 0.16)).with_material(
            mw.Material(eps_r=12.0),
            name="core",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(_SRC_X, 0.0, 0.0),
            size=(0.0, 0.4, 0.4),
            polarization="Ez",
            source_time=source_time,
            name="mode0",
        )
    )
    scene.add_monitor(mw.PlaneMonitor("plane", axis="x", position=_PLANE_X, fields=("Ez",)))
    return scene


def _plane(result, frequency):
    payload = result.monitor("plane", frequency=frequency)["Ez"]
    return torch.as_tensor(payload).detach().cpu().to(torch.complex128).reshape(-1)


def _fidelity(a, b):
    numerator = torch.abs(torch.sum(torch.conj(a) * b))
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((numerator / denominator).item())


def _run(source_time, frequencies):
    return mw.Simulation.fdtd(
        _guide_scene(source_time),
        frequencies=list(frequencies),
        run_time=mw.TimeConfig(time_steps=_STEPS),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    ).run()


# --- Structural / contract tests (no full run) --------------------------------


def test_custom_source_time_mode_source_is_rejected():
    scene = _guide_scene(mw.CustomSourceTime(lambda t: 0.0, characteristic_frequency=_CENTER))
    with pytest.raises(ValueError, match="CustomSourceTime"):
        mw.Simulation.fdtd(scene, frequencies=[_CENTER]).prepare()


def test_gaussian_pulse_mode_source_drives_time_shifted_surface_terms():
    solver = (
        mw.Simulation.fdtd(
            _guide_scene(mw.GaussianPulse(frequency=_CENTER, fwidth=_FWIDTH, amplitude=_AMPLITUDE)),
            frequencies=[_CENTER],
        )
        .prepare()
        .solver
    )

    compiled = solver._compiled_sources[0]
    assert compiled["kind"] == "mode_source"
    assert compiled["source_time"]["kind"] == "gaussian_pulse"
    # The guided profile is solved once, at the pulse center frequency.
    assert compiled["effective_index"] > 1.0
    assert compiled["mode_solver_kind"] is not None

    electric_terms = solver._electric_source_terms
    magnetic_terms = solver._magnetic_source_terms
    assert electric_terms and magnetic_terms
    # A broadband pulse must take the native time-shifted patch path (delay_patch set,
    # no CW cos/sin patch); the CW path is analytic-phasor only.
    for term in electric_terms + magnetic_terms:
        assert term["cw_cos_patch"] is None
        assert term["cw_sin_patch"] is None
        assert term["delay_patch"] is not None
        assert term["source_time"]["kind"] == "gaussian_pulse"
    assert any(
        float(torch.max(torch.abs(term["delay_patch"])).item()) > 0.0
        for term in electric_terms + magnetic_terms
    )


# --- Acceptance: broadband injects the per-frequency CW guided mode ------------


@lru_cache(maxsize=None)
def _cw_planes():
    planes = {}
    for frequency in _FREQS:
        result = _run(mw.CW(frequency=frequency, amplitude=_AMPLITUDE), [frequency])
        planes[frequency] = _plane(result, frequency)
        del result
        torch.cuda.empty_cache()
    return planes


@lru_cache(maxsize=None)
def _broadband_planes():
    result = _run(mw.GaussianPulse(frequency=_CENTER, fwidth=_FWIDTH, amplitude=_AMPLITUDE), _FREQS)
    planes = {frequency: _plane(result, frequency) for frequency in _FREQS}
    del result
    torch.cuda.empty_cache()
    return planes


def test_broadband_mode_source_matches_cw_guided_mode_across_band():
    cw = _cw_planes()
    broadband = _broadband_planes()
    for frequency in _FREQS:
        # One pulsed run must deliver a real guided field at every band frequency...
        assert float(torch.linalg.vector_norm(broadband[frequency]).item()) > 1.0e-2
        # ...and that field must be the same transverse mode the per-frequency CW mode
        # source injects. The full-vector mode shape disperses across this
        # deliberately broad +/-6% band, but polarization-resolved degenerate
        # mode tracking keeps the overlap above 0.90.
        fidelity = _fidelity(broadband[frequency], cw[frequency])
        assert fidelity > 0.90, (frequency, fidelity)

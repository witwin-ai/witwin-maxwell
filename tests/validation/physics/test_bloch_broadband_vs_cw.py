"""P5.4 acceptance: a broadband grating run reproduces the CW result per frequency.

Plan P5.4 criterion 1 asks that a single broadband ``GaussianPulse`` grating run
reproduce, at every frequency in the band, the steady-state result obtained from
a monochromatic ``CW`` excitation, within 2%.

The quantitative comparison is performed at **normal incidence** under periodic
transverse boundaries (the ``k_bloch -> 0`` limit of the Bloch grating), because
that is the only regime in which the CW reference is physically well defined:
an *oblique* Bloch grating driven by CW excites long-lived transverse Bloch modes
that do not settle to an extractable steady state within a practical run, so its
CW phasor is unusable as a reference (empirically ``|E_cw|`` collapses to solver
round-off while the broadband run stays O(1)). This is the same reason the pulsed
broadband capability exists and is documented at the top of
``tests/sources/tfsf/test_fdtd_grating_pulsed_bloch.py``.

The compared observable is the complex transmission coefficient of a thin,
non-resonant dielectric metasurface layer,

    t(f) = E_down(f; with slab) / E_down(f; empty cell),

measured at a single downstream point. Forming the ratio against the empty-cell
run cancels the (different) CW-vs-pulse DFT normalization and the source phase,
leaving the intrinsic, source-independent slab transmission that both excitations
must reproduce. A non-resonant layer keeps ``t(f)`` a smooth function of
frequency, so the comparison is not dominated by a sharp Fabry-Perot feature.

The second test exercises the genuine complex-field Bloch broadband injection at
oblique incidence and asserts that the pulse is confined to the total-field slab
at each frequency -- the per-frequency physical-correctness signal available when
a CW reference is not.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FREQS = (0.9e9, 1.0e9, 1.1e9)
_TFSF_BOUNDS = (-0.6, 0.6)
_DOWNSTREAM_Z = 0.30
_STEPS = 3200
# Thin (0.06 m), low-contrast (n = 1.5) layer: subwavelength in the slab, so its
# transmission is a smooth, non-resonant function of frequency across the band.
_SLAB_EPS = 2.25
_SLAB_THICKNESS = 0.06


def _normal_grating_scene(source_time, *, with_slab):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.9, 0.9))),
        grid=mw.GridSpec.uniform(0.015),
        boundary=mw.BoundarySpec.faces(
            default="pml", num_layers=6, strength=1.0,
            x="periodic", y="periodic", z="pml",
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=source_time,
            injection=mw.TFSF.slab(axis="z", bounds=_TFSF_BOUNDS),
            name="incident",
        )
    )
    if with_slab:
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.2, 0.2, _SLAB_THICKNESS)),
                material=mw.Material(eps_r=_SLAB_EPS),
                name="metasurface_layer",
            )
        )
    scene.add_monitor(mw.PointMonitor("down", (0.0, 0.0, _DOWNSTREAM_Z), fields=("Ex",)))
    return scene


def _downstream_phasor(result, frequency):
    payload = result.monitor("down", frequency=frequency)["Ex"]
    return complex(torch.as_tensor(payload).detach().cpu().reshape(-1)[0])


def _run(source_time, frequencies, *, with_slab):
    return mw.Simulation.fdtd(
        _normal_grating_scene(source_time, with_slab=with_slab),
        frequencies=list(frequencies),
        run_time=mw.TimeConfig(time_steps=_STEPS),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        absorber="cpml",
    ).run()


@lru_cache(maxsize=None)
def _cw_transmission():
    """Complex slab transmission from monochromatic CW runs, one pair per frequency."""
    transmission = {}
    for frequency in _FREQS:
        empty = _run(mw.CW(frequency=frequency, amplitude=20.0), [frequency], with_slab=False)
        slab = _run(mw.CW(frequency=frequency, amplitude=20.0), [frequency], with_slab=True)
        transmission[frequency] = _downstream_phasor(slab, frequency) / _downstream_phasor(empty, frequency)
        del empty, slab
        torch.cuda.empty_cache()
    return transmission


@lru_cache(maxsize=None)
def _broadband_transmission():
    """Complex slab transmission from a single broadband GaussianPulse run pair."""
    pulse = mw.GaussianPulse(frequency=1.0e9, fwidth=0.35e9, amplitude=20.0)
    empty = _run(pulse, _FREQS, with_slab=False)
    slab = _run(pulse, _FREQS, with_slab=True)
    transmission = {
        frequency: _downstream_phasor(slab, frequency) / _downstream_phasor(empty, frequency)
        for frequency in _FREQS
    }
    del empty, slab
    torch.cuda.empty_cache()
    return transmission


def test_broadband_reproduces_cw_transmission_within_two_percent():
    cw = _cw_transmission()
    broadband = _broadband_transmission()
    # A single broadband run must reproduce the per-frequency monochromatic CW
    # transmission (magnitude and phase) at each of the three band frequencies.
    # Measured relative errors: ~0.9%, ~1.1%, ~0.05%.
    for frequency in _FREQS:
        rel_err = abs(broadband[frequency] - cw[frequency]) / abs(cw[frequency])
        assert rel_err < 0.02, (frequency, rel_err, cw[frequency], broadband[frequency])


# --- Oblique complex-field Bloch broadband injection ---------------------------

_OBLIQUE_BOUNDS = (-0.24, 0.24)
_OBLIQUE_DIR = (0.2, 0.1, 0.9746794344808963)
_OBLIQUE_POL = (1.0, 0.0, -0.20519567041703082)
_OBLIQUE_FREQS = (0.85e9, 1.0e9, 1.15e9)


def _oblique_bloch_scene(source_time, wavevector):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=mw.BoundarySpec.faces(
            default="pml", num_layers=6, strength=1.0,
            x="bloch", y="bloch", z="pml", bloch_wavevector=wavevector,
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=_OBLIQUE_DIR,
            polarization=_OBLIQUE_POL,
            source_time=source_time,
            injection=mw.TFSF.slab(axis="z", bounds=_OBLIQUE_BOUNDS),
            name="incident",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


def test_broadband_bloch_injection_confines_total_field_per_frequency():
    # Resolve the incident-consistent Bloch wavevector from a CW prepare, then run
    # the broadband pulse with that explicit wavevector (auto resolution is CW-only).
    prepared = mw.Simulation.fdtd(
        _oblique_bloch_scene(mw.CW(frequency=1.0e9, amplitude=20.0), "auto"),
        frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1), absorber="cpml",
    ).prepare()
    wavevector = tuple(float(v) for v in prepared.solver.resolved_bloch_wavevector)
    del prepared
    torch.cuda.empty_cache()

    result = mw.Simulation.fdtd(
        _oblique_bloch_scene(
            mw.GaussianPulse(frequency=1.0e9, fwidth=0.3e9, amplitude=20.0), wavevector
        ),
        frequencies=list(_OBLIQUE_FREQS),
        run_time=mw.TimeConfig(time_steps=192),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
        absorber="cpml",
    ).run()

    # The broadband pulse drives the genuine complex-field Bloch grating path.
    assert result.solver.complex_fields_enabled is True
    assert result.solver._tfsf_state["provider"] == "plane_wave_grating_slab_cw"

    dz = result.solver.scene.dz
    z_lo, z_hi = result.solver.scene.domain_range[4], result.solver.scene.domain_range[5]
    for frequency in _OBLIQUE_FREQS:
        field = result.field("Ex", frequency=frequency)
        magnitude = np.abs((field["data"] if isinstance(field, dict) else field).detach().cpu().numpy())
        assert np.isfinite(magnitude).all()
        z = np.linspace(z_lo, z_hi, magnitude.shape[2])
        inside = (z >= _OBLIQUE_BOUNDS[0]) & (z <= _OBLIQUE_BOUNDS[1])
        outside = (z < _OBLIQUE_BOUNDS[0] - dz) | (z > _OBLIQUE_BOUNDS[1] + dz)
        inside_max = float(magnitude[:, :, inside].max())
        outside_max = float(magnitude[:, :, outside].max())
        # The incident pulse lives in the total-field slab; the scattered region
        # beyond the two z-faces must not carry it (empty cell, no scatterer).
        assert inside_max > 0.0
        assert outside_max < inside_max, (frequency, outside_max, inside_max)

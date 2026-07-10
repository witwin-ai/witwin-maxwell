"""Forward FDTD validation for the ``nl-dispersive-same-material`` combination.

Second-harmonic generation (SHG) needs *both* an instantaneous ``chi2``
nonlinearity *and* dispersion in the same material: the nonlinearity generates
the ``2w`` polarization, and the dispersion sets ``n(w)`` vs ``n(2w)`` and hence
the phase mismatch ``dk = (2w/c) * (n(2w) - n(w))`` that governs the
coherent-buildup ``sinc(dk*L/2)`` factor. The construction guard that forbade a
single ``Material`` from carrying nonlinearity and poles is lifted, and the
runtime subtracts the ADE polarization current against the same field-dependent
effective permittivity the nonlinear displacement-current term uses, keeping the
two responses mutually consistent.

The undepleted-pump plane-wave result for the second-harmonic field at the slab
exit is

    E(2w, L) ~ (w * chi2 / (2 * n(2w) * c)) * E(w)^2 * L * sinc(dk*L/2),

so with fixed pump and ``chi2`` the second-harmonic amplitude scales as
``|sin(dk*L/2)|``: it grows linearly with ``L`` when phase matched (``dk -> 0``)
and forms Maker fringes (a peak near the coherence length ``L_c = pi/dk`` and a
null near ``2*L_c``) when mismatched. The tests below check exactly this
contrast between a phase-matched (dispersionless) slab and a deliberately
mismatched (dispersive) slab, with the mismatch length scale ``L_c`` and the
``sinc`` factor computed analytically from the material poles.

The absolute conversion efficiency is not asserted to 5%: a small FDTD scene
carries frequency-dependent numerical dispersion (the resolved ``2w`` index is a
few percent above the continuum value, which drifts the fringe positions) and a
``2w`` impedance/transmission prefactor between the dispersive and dispersionless
slabs. The phase-matching *ratio* physics (linear-in-L phase-matched growth vs
``sinc``-suppressed mismatched fringes) is robust and is what is validated here.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_C0 = 299_792_458.0
_FREQUENCY = 5.0e8
_OMEGA = 2.0 * np.pi * _FREQUENCY

# Lossless-ish Lorentz pole placed above 2w so both w and 2w sit in the
# normal-dispersion tail (n increases with frequency, low loss) and dk is large
# enough that the material dispersion dominates the grid's numerical dispersion.
_EPS_INF = 3.0
_DELTA_EPS = 2.0
_POLE_FREQUENCY = 1.25e9
_POLE_GAMMA = 1.0e6
_CHI2 = 1.0e-6


def _lorentz_permittivity(freq):
    """Analytic relative permittivity of the single Lorentz pole material."""
    omega = 2.0 * np.pi * freq
    omega0 = 2.0 * np.pi * _POLE_FREQUENCY
    gamma = 2.0 * np.pi * _POLE_GAMMA
    return _EPS_INF + _DELTA_EPS * omega0 * omega0 / (omega0 * omega0 - omega * omega - 1j * gamma * omega)


def _phase_matching():
    n_w = float(np.sqrt(_lorentz_permittivity(_FREQUENCY).real))
    n_2w = float(np.sqrt(_lorentz_permittivity(2.0 * _FREQUENCY).real))
    dk = (2.0 * _OMEGA / _C0) * (n_2w - n_w)
    coherence_length = np.pi / abs(dk)
    return n_w, n_2w, dk, coherence_length


def _dispersive_material(chi2):
    kwargs = dict(
        eps_r=_EPS_INF,
        lorentz_poles=(mw.LorentzPole(delta_eps=_DELTA_EPS, resonance_frequency=_POLE_FREQUENCY, gamma=_POLE_GAMMA),),
    )
    if chi2 != 0.0:
        kwargs["nonlinearity"] = mw.NonlinearSusceptibility(chi2=chi2)
    return mw.Material(**kwargs)


def _matched_material(n_w, chi2):
    # Dispersionless slab whose (frequency-independent) index equals the
    # dispersive slab's index at w, so the pump coupling matches and only the
    # 2w phase matching differs (dk = 0 here).
    return mw.Material(eps_r=n_w * n_w, nonlinearity=mw.NonlinearSusceptibility(chi2=chi2))


def _plane_complex_mean(payload):
    data = payload["data"] if isinstance(payload, dict) else payload
    field = data.detach().cpu().numpy()
    crop = field[3:-3, 3:-3]
    if crop.size == 0:
        crop = field
    return complex(crop.mean())


def _second_harmonic_amplitude(material, thickness):
    """|E(2w)| at a monitor behind a slab of the given material and thickness."""
    entrance = -0.45
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.15, 0.15), (-0.15, 0.15))),
        grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                # A pulsed pump keeps the turn-on smooth and its own spectral
                # content at 2f negligible (checked by the chi2=0 background run).
                source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=1.0e8, amplitude=100.0),
                name="pw",
            )
        ],
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(entrance + thickness / 2.0, 0.0, 0.0), size=(thickness, 0.6, 0.6)),
            material=material,
        )
    )
    scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.6, fields=("Ez",)))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_FREQUENCY, 2.0 * _FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
        spectral_sampler=mw.SpectralSampler(window="hanning", normalize_source=False),
        full_field_dft=False,
    ).run()
    return abs(_plane_complex_mean(result.monitor("post", freq_index=1)["components"]["Ez"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_nonlinear_dispersive_same_material_compiles_and_runs():
    """A single Material carrying chi2 and electric poles composes at runtime.

    Both the nonlinear coefficient channel and the electric ADE subsystem must be
    active on the same structure (the general nonlinear kernel is forced by the
    chi2 channel), which is the state the phase-matching runs rely on.
    """
    n_w, _, _, coherence_length = _phase_matching()
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.15, 0.15), (-0.15, 0.15))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=1.0e8, amplitude=50.0),
                name="pw",
            )
        ],
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(coherence_length, 0.6, 0.6)),
            material=_dispersive_material(_CHI2),
        )
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[_FREQUENCY, 2.0 * _FREQUENCY]).prepare().solver
    assert solver.nonlinear_enabled
    assert solver.nonlinear_general_enabled
    assert solver.electric_dispersive_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_chi2_shg_dispersive_phase_matching():
    """Phase-matched (dk=0) SHG grows with L; mismatched (dk!=0) SHG makes Maker fringes.

    All length scales and the sinc factor are derived from the analytic material
    dispersion. The dispersive slab is driven at the coherence length ``L_c``
    (first SHG maximum, sinc(pi/2)), ``1.5*L_c`` (falling), and ``2*L_c`` (the
    first null, sinc(pi)=0); the dispersionless slab (same index at w) is the
    phase-matched reference that keeps growing with L.
    """
    n_w, n_2w, dk, coherence_length = _phase_matching()
    assert n_2w > n_w  # normal dispersion between w and 2w

    disp_peak = _second_harmonic_amplitude(_dispersive_material(_CHI2), coherence_length)
    disp_mid = _second_harmonic_amplitude(_dispersive_material(_CHI2), 1.5 * coherence_length)
    disp_null = _second_harmonic_amplitude(_dispersive_material(_CHI2), 2.0 * coherence_length)
    background = _second_harmonic_amplitude(_dispersive_material(0.0), coherence_length)
    matched_peak = _second_harmonic_amplitude(_matched_material(n_w, _CHI2), coherence_length)
    matched_null = _second_harmonic_amplitude(_matched_material(n_w, _CHI2), 2.0 * coherence_length)

    # The pump's own spectral leakage at 2f (chi2 = 0) must be negligible.
    assert background < 0.1 * disp_peak

    # Phase-matched reference (dk = 0): the second harmonic grows ~linearly with
    # L, so doubling the length from L_c to 2*L_c must nearly double it
    # (analytic factor 2; pump depletion / numerical loss keep it below 2).
    assert 1.6 < matched_null / matched_peak < 2.2

    # Deliberately mismatched (dk != 0): the second harmonic is NON-monotonic in
    # L, falling from the coherence-length peak toward the 2*L_c null -- the
    # signature of phase mismatch, impossible under phase matching.
    assert disp_peak > disp_mid > disp_null
    assert disp_null < 0.4 * disp_peak

    # sinc^2 ratio, mismatched / phase-matched at the same length. At the null
    # (dk*L/2 = pi) sinc(pi) = 0, so the dispersive slab is strongly suppressed
    # relative to the phase-matched slab.
    assert disp_null / matched_null < 0.2

    # At the coherence length (dk*L/2 = pi/2) the analytic amplitude ratio is
    # sinc(pi/2) = 2/pi ~ 0.637; the measured ratio brackets it (the slack
    # absorbs the 2w impedance/transmission prefactor between the two slabs).
    sinc_half = abs(np.sinc((dk * coherence_length / 2.0) / np.pi))  # np.sinc(x) = sin(pi x)/(pi x)
    assert sinc_half == pytest.approx(2.0 / np.pi, rel=1.0e-6)
    assert 0.45 < disp_peak / matched_peak < 0.80

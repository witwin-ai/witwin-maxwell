import warnings

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_C0 = 299_792_458.0


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plane_field_mean(data):
    if isinstance(data, dict):
        data = data["data"]
    field = np.abs(_to_cpu_numpy(data))
    crop = field[3:-3, 3:-3]
    if crop.size == 0:
        crop = field
    return float(crop.mean())


def _build_plane_wave_scene(frequency, *, amplitude=1.0):
    return mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=frequency, amplitude=amplitude),
                name="pw",
            )
        ],
    )


def _run_fdtd(scene, frequency):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()


# ---------------------------------------------------------------------------
# Validator behavior (no GPU required).
# ---------------------------------------------------------------------------


def test_lorentz_gain_pole_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_gain=True"):
        mw.LorentzPole(delta_eps=-1.0, resonance_frequency=1.0e9, gamma=1.0e8)

    with pytest.raises(ValueError, match="allow_gain=True"):
        mw.Material.lorentz(
            eps_inf=1.0,
            delta_eps=-1.0,
            resonance_frequency=1.0e9,
            gamma=1.0e8,
        )


def test_lorentz_gain_pole_warns_and_constructs_with_opt_in():
    with pytest.warns(UserWarning, match="gain"):
        pole = mw.LorentzPole(
            delta_eps=-0.5,
            resonance_frequency=1.0e9,
            gamma=1.0e8,
            allow_gain=True,
        )
    assert pole.delta_eps == pytest.approx(-0.5)
    assert pole.allow_gain is True

    with pytest.warns(UserWarning, match="Courant"):
        material = mw.Material.lorentz(
            eps_inf=1.0,
            delta_eps=-0.5,
            resonance_frequency=1.0e9,
            gamma=1.0e8,
            allow_gain=True,
        )
    assert material.is_electric_dispersive
    assert material.lorentz_poles[0].delta_eps == pytest.approx(-0.5)


def test_positive_lorentz_pole_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pole = mw.LorentzPole(delta_eps=1.2, resonance_frequency=1.0e9, gamma=1.0e8)
    assert pole.allow_gain is False
    assert pole.delta_eps == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Physics: single-pass amplification vs analytic exp(g * L).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_linear_gain_slab_amplifies_vs_analytic_single_pass():
    frequency = 1.0e9  # operate exactly at the resonance (design) frequency
    slab_thickness = 0.24

    with pytest.warns(UserWarning, match="gain"):
        gain_material = mw.Material.lorentz(
            eps_inf=1.0,
            delta_eps=-0.007,  # negative oscillator strength -> linear gain
            resonance_frequency=frequency,
            gamma=1.0e8,
            allow_gain=True,
        )

    vacuum_scene = _build_plane_wave_scene(frequency)
    vacuum_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.4, fields=("Ez",)))

    gain_scene = _build_plane_wave_scene(frequency)
    gain_scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(slab_thickness, 0.8, 0.8)),
            material=gain_material,
        )
    )
    gain_scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.4, fields=("Ez",)))

    vacuum_result = _run_fdtd(vacuum_scene, frequency)
    gain_result = _run_fdtd(gain_scene, frequency)

    vacuum_post = _plane_field_mean(vacuum_result.monitor("post")["data"])
    gain_post = _plane_field_mean(gain_result.monitor("post")["data"])
    measured_amplitude_gain = gain_post / max(vacuum_post, 1e-12)

    # Analytic single-pass amplitude gain exp(|Im(n)| * k0 * L) at the design
    # frequency, from the material's own frequency-domain permittivity.
    eps = gain_material.relative_permittivity(frequency)
    refractive_index = np.sqrt(eps)
    if refractive_index.real < 0.0:
        refractive_index = -refractive_index
    k0 = 2.0 * np.pi * frequency / _C0
    expected_amplitude_gain = float(np.exp(abs(refractive_index.imag) * k0 * slab_thickness))

    # The slab must amplify (ratio > 1), and match the small-gain exponential to
    # within a few percent.
    assert expected_amplitude_gain > 1.0
    assert measured_amplitude_gain > 1.02
    assert measured_amplitude_gain == pytest.approx(expected_amplitude_gain, rel=0.06)

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.fdtd.excitation.injection import (
    _ideal_axis_weights,
    _normalized_point_dipole_profile,
)
from witwin.maxwell.sources import POINT_DIPOLE_REFERENCE_WIDTH


def _make_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    )


def test_point_dipole_compiles_default_cw_from_simulation_frequency():
    scene = _make_scene()
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", width=0.05))

    compiled = compile_fdtd_sources(scene, default_frequency=2.5e9)

    assert len(compiled) == 1
    assert compiled[0]["kind"] == "point_dipole"
    assert compiled[0]["profile"] == "gaussian"
    assert compiled[0]["source_time"]["kind"] == "cw"
    assert compiled[0]["source_time"]["frequency"] == pytest.approx(2.5e9)
    assert compiled[0]["source_time"]["amplitude"] == pytest.approx(1.0)


def test_point_dipole_supports_ideal_profile_switch():
    source = mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", profile="IDEAL")

    assert source.profile == "ideal"


def test_point_dipole_rejects_unknown_profile():
    with pytest.raises(ValueError):
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", profile="delta")


def test_gaussian_pulse_is_delayed_and_broadband():
    pulse = mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9, amplitude=2.0)

    assert abs(pulse.evaluate(0.0)) < 1e-5
    assert pulse.evaluate(pulse.delay) == pytest.approx(2.0, rel=1e-6)
    assert pulse.characteristic_frequency > pulse.frequency


def test_ricker_wavelet_has_zero_phase_and_positive_band_limit():
    wavelet = mw.RickerWavelet(frequency=1.0e9, amplitude=3.0)

    assert wavelet.phase == 0.0
    assert wavelet.characteristic_frequency > wavelet.frequency
    assert abs(wavelet.evaluate(wavelet.delay)) == pytest.approx(3.0, rel=1e-6)


def test_plane_wave_requires_transverse_polarization():
    with pytest.raises(ValueError):
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9),
        )


def test_gaussian_beam_normalizes_direction_and_polarization():
    source = mw.GaussianBeam(
        direction=(2.0, 0.0, 0.0),
        polarization=(0.0, 0.0, 5.0),
        beam_waist=0.2,
        focus=(0.0, 0.0, 0.0),
        source_time=mw.CW(frequency=1.0e9, phase=math.pi / 4.0),
    )

    assert source.direction == (1.0, 0.0, 0.0)
    assert source.polarization == (0.0, 0.0, 1.0)
    assert source.source_time.phase == pytest.approx(math.pi / 4.0)


def test_tfsf_injection_compiles_with_bounds():
    scene = _make_scene()
    scene.add_source(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1.0e9),
            injection=mw.TFSF(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        )
    )

    compiled = compile_fdtd_sources(scene, default_frequency=1.0e9)

    assert len(compiled) == 1
    assert compiled[0]["injection"]["kind"] == "tfsf"
    assert compiled[0]["injection"]["mode"] == "box"
    assert compiled[0]["injection"]["bounds"] == ((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))


def test_tfsf_slab_compiles_public_metadata():
    source = mw.TFSF.slab(axis="z", bounds=(-0.2, 0.3))

    assert source.kind == "tfsf"
    assert source.mode == "slab"
    assert source.axis == "z"
    assert source.axis_bounds == (-0.2, 0.3)
    assert source.bounds is None


def test_tfsf_slab_rejects_invalid_axis_and_bounds():
    with pytest.raises(ValueError, match="axis"):
        mw.TFSF.slab(axis="q", bounds=(-0.2, 0.2))
    with pytest.raises(ValueError, match="end > start"):
        mw.TFSF.slab(axis="z", bounds=(0.2, -0.2))


def test_tfsf_slab_injection_compiles_axis_metadata():
    scene = _make_scene()
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.2, 0.3)),
        )
    )

    compiled = compile_fdtd_sources(scene, default_frequency=1.0e9)

    assert compiled[0]["injection"] == {
        "kind": "tfsf",
        "mode": "slab",
        "axis": "z",
        "axis_bounds": (-0.2, 0.3),
    }


def test_fdtd_sources_compile_to_list_in_scene_order():
    scene = _make_scene()
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez"))
    scene.add_source(
        mw.PlaneWave(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 1.0, 0.0),
            source_time=mw.CW(frequency=1.0e9),
        )
    )

    compiled = compile_fdtd_sources(scene, default_frequency=1.0e9)

    assert [item["kind"] for item in compiled] == ["point_dipole", "plane_wave"]


def test_mode_source_resolves_source_plane_and_tangential_polarization():
    source = mw.ModeSource(
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.6, 0.4),
        polarization="Ez",
        source_time=mw.CW(frequency=1.0e9),
    )

    assert source.kind == "mode_source"
    assert source.normal_axis == "x"
    assert source.polarization_axis == "z"
    assert source.polarization == (0.0, 0.0, 1.0)


def test_mode_source_compiles_default_cw_from_simulation_frequency():
    scene = _make_scene()
    scene.add_source(
        mw.ModeSource(
            position=(0.0, 0.0, 0.0),
            size=(0.0, 0.6, 0.6),
            polarization="Ez",
        )
    )

    compiled = compile_fdtd_sources(scene, default_frequency=2.5e9)

    assert len(compiled) == 1
    assert compiled[0]["kind"] == "mode_source"
    assert compiled[0]["normal_axis"] == "x"
    assert compiled[0]["direction_vector"] == (1.0, 0.0, 0.0)
    assert compiled[0]["source_time"]["kind"] == "cw"
    assert compiled[0]["source_time"]["frequency"] == pytest.approx(2.5e9)


def test_mode_source_rejects_non_tangential_polarization():
    with pytest.raises(ValueError):
        mw.ModeSource(
            size=(0.0, 0.6, 0.6),
            polarization="Ex",
            source_time=mw.CW(frequency=1.0e9),
        )


def test_point_dipole_gaussian_profile_integrates_to_unit_current_moment():
    width = 0.005
    support = 3.0 * max(width, 0.5 * POINT_DIPOLE_REFERENCE_WIDTH)
    coords = np.arange(-support, support + 1e-12, 0.01)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    dist_sq = torch.tensor(xx**2 + yy**2 + zz**2, dtype=torch.float32)

    control_volumes = torch.full_like(dist_sq, 0.01**3)
    normalized = _normalized_point_dipole_profile(dist_sq, width, control_volumes)

    assert float(torch.sum(normalized * control_volumes).item()) == pytest.approx(1.0, rel=1e-6)


def test_ideal_point_dipole_linear_weights_preserve_position_and_mass():
    coords = torch.tensor((-0.02, 0.0, 0.03), dtype=torch.float64)
    indices, weights = _ideal_axis_weights(coords, 0.01)

    assert sum(weights) == pytest.approx(1.0)
    assert sum(float(coords[index]) * weight for index, weight in zip(indices, weights)) == pytest.approx(0.01)

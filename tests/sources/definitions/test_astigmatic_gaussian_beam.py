import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources


# ---------------------------------------------------------------------------
# Tier A: construction and compilation (CPU, no solver)
# ---------------------------------------------------------------------------


def _cpu_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    )


def test_astigmatic_gaussian_beam_stores_per_axis_waist_and_focus():
    source = mw.AstigmaticGaussianBeam(
        direction=(2.0, 0.0, 0.0),
        polarization=(0.0, 0.0, 5.0),
        beam_waist=(0.3, 0.18),
        focus=(0.1, 0.0, 0.0),
        focus_u=0.05,
        focus_v=-0.05,
        source_time=mw.CW(frequency=1.0e9),
    )

    assert source.kind == "astigmatic_gaussian_beam"
    assert source.direction == (1.0, 0.0, 0.0)
    assert source.polarization == (0.0, 0.0, 1.0)
    assert source.beam_waist == (0.3, 0.18)
    assert source.focus == (0.1, 0.0, 0.0)
    assert source.focus_u == pytest.approx(0.05)
    assert source.focus_v == pytest.approx(-0.05)


def test_astigmatic_gaussian_beam_rejects_bad_waist_and_polarization():
    with pytest.raises(ValueError):
        mw.AstigmaticGaussianBeam(beam_waist=(0.2,))
    with pytest.raises(ValueError):
        mw.AstigmaticGaussianBeam(beam_waist=(0.2, -0.1))
    with pytest.raises(ValueError):
        mw.AstigmaticGaussianBeam(
            direction=(1.0, 0.0, 0.0),
            polarization=(1.0, 0.0, 0.0),
        )


def test_astigmatic_gaussian_beam_compiles_per_axis_fields():
    scene = _cpu_scene()
    scene.add_source(
        mw.AstigmaticGaussianBeam(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            beam_waist=(0.3, 0.18),
            focus=(0.0, 0.0, 0.0),
            focus_u=0.04,
            focus_v=-0.02,
            source_time=mw.CW(frequency=2.5e9),
        )
    )

    compiled = compile_fdtd_sources(scene, default_frequency=2.5e9)

    assert len(compiled) == 1
    entry = compiled[0]
    assert entry["kind"] == "astigmatic_gaussian_beam"
    assert entry["beam_waist_u"] == pytest.approx(0.3)
    assert entry["beam_waist_v"] == pytest.approx(0.18)
    assert entry["focus"] == (0.0, 0.0, 0.0)
    assert entry["focus_u"] == pytest.approx(0.04)
    assert entry["focus_v"] == pytest.approx(-0.02)
    assert entry["injection"]["kind"] == "soft"
    assert entry["source_time"]["kind"] == "cw"
    assert entry["source_time"]["frequency"] == pytest.approx(2.5e9)


# ---------------------------------------------------------------------------
# Tier B: real FDTD physics (CUDA only)
# ---------------------------------------------------------------------------


def _build_scene(source):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(source)
    return scene


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _one_over_e_half_width(profile, spacing):
    """1/e half-width of a 1D bell profile using linear interpolation of the crossing."""
    profile = np.asarray(profile, dtype=float)
    peak_index = int(np.argmax(profile))
    threshold = profile[peak_index] / math.e

    def crossing(step):
        i = peak_index
        while 0 <= i + step < len(profile) and profile[i + step] > threshold:
            i += step
        j = i + step
        if not 0 <= j < len(profile):
            return None
        denom = profile[i] - profile[j]
        frac = 0.0 if denom == 0.0 else (profile[i] - threshold) / denom
        return abs((i + frac * step) - peak_index)

    widths = [w for w in (crossing(-1), crossing(+1)) if w is not None]
    assert widths, "profile does not cross the 1/e level inside the domain"
    return spacing * float(np.mean(widths))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_astigmatic_beam_reproduces_per_axis_waists_at_focus():
    dx = 0.08
    waist_u = 0.36  # along polarization (z axis on an x-plane monitor)
    waist_v = 0.24  # along binormal (y axis on an x-plane monitor)
    scene = _build_scene(
        mw.AstigmaticGaussianBeam(
            direction=(1.0, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            beam_waist=(waist_u, waist_v),
            focus=(0.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=120.0),
            name="beam",
        )
    )
    scene.add_monitor(mw.PlaneMonitor("focus_ez", axis="x", position=0.0, fields=("Ez",)))

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()

    field = np.abs(_to_cpu_numpy(result.monitor("focus_ez")["data"]))
    field = np.squeeze(field)
    assert field.ndim == 2

    peak_row, peak_col = np.unravel_index(int(np.argmax(field)), field.shape)
    # Axis 0 is y (binormal -> waist_v), axis 1 is z (polarization -> waist_u).
    measured_v = _one_over_e_half_width(field[:, peak_col], dx)
    measured_u = _one_over_e_half_width(field[peak_row, :], dx)

    assert measured_u == pytest.approx(waist_u, rel=0.15)
    assert measured_v == pytest.approx(waist_v, rel=0.15)
    # The beam must be genuinely astigmatic on the grid.
    assert measured_u > measured_v * 1.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_isotropic_astigmatic_beam_matches_gaussian_beam():
    waist = 0.25
    common = dict(
        direction=(1.0, 0.0, 0.0),
        polarization=(0.0, 0.0, 1.0),
        focus=(0.0, 0.0, 0.0),
        source_time=mw.CW(frequency=1.0e9, amplitude=120.0),
    )

    def run(source):
        scene = _build_scene(source)
        scene.add_monitor(mw.PlaneMonitor("plane_ez", axis="x", position=0.0, fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[1.0e9],
            run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        ).run()
        return _to_cpu_numpy(result.monitor("plane_ez")["data"])

    reference = run(mw.GaussianBeam(beam_waist=waist, name="beam", **common))
    astigmatic = run(
        mw.AstigmaticGaussianBeam(
            beam_waist=(waist, waist),
            focus_u=0.0,
            focus_v=0.0,
            name="beam",
            **common,
        )
    )

    rel_error = np.linalg.norm(astigmatic - reference) / max(np.linalg.norm(reference), 1e-30)
    assert rel_error < 0.02

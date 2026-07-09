from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

_FREQ = 2.0e9


def _dielectric_dipole_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.04)),
            material=mw.Material(eps_r=6.0),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.02, 0.0, -0.02),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=0.7e9),
        )
    )
    return scene


def _run(scene, *, cuda_graph):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=300),
        cuda_graph=cuda_graph,
    ).run()
    view = result.at(frequency=_FREQ)
    fields = {a: np.asarray(getattr(view.E, a).detach().cpu()) for a in ("x", "y", "z")}
    return fields, result.stats()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_cuda_graph_matches_non_graph_bitexact():
    ref, ref_stats = _run(_dielectric_dipole_scene(), cuda_graph=False)
    got, got_stats = _run(_dielectric_dipole_scene(), cuda_graph=True)
    assert ref_stats["cuda_graph_active"] is False
    assert got_stats["cuda_graph_active"] is True  # standard real-field path -> captured
    assert got_stats["tail_graph_active"] is True  # GPU-driven DFT + tail captured too
    # Graph replay (field update + GPU-driven running DFT) reproduces the host
    # path exactly.
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_cuda_graph_multifreq_dft_bitexact():
    freqs = (1.6e9, 2.0e9, 2.4e9)

    def run(cuda_graph):
        result = mw.Simulation.fdtd(
            _dielectric_dipole_scene(),
            frequencies=freqs,
            full_field_dft=True,
            run_time=mw.TimeConfig(time_steps=300),
            cuda_graph=cuda_graph,
        ).run()
        return {
            f: {a: np.asarray(getattr(result.at(frequency=f).E, a).detach().cpu()) for a in ("x", "y", "z")}
            for f in freqs
        }, result.stats()

    ref, _ = run(False)
    got, got_stats = run(True)
    assert got_stats["tail_graph_active"] is True
    for f in freqs:
        for a in ("x", "y", "z"):
            assert np.array_equal(ref[f][a], got[f][a]), (f, a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
def test_cuda_graph_falls_back_for_plane_wave_source():
    # A soft plane wave injects a magnetic surface source inside the update block,
    # which carries per-step host input, so graph capture must decline and the run
    # must still match the non-graph result exactly.
    def scene():
        s = mw.Scene(
            domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.12, 0.12))),
            grid=mw.GridSpec.uniform(0.008),
            boundary=mw.BoundarySpec.pml(num_layers=8),
            device="cuda",
        )
        s.add_source(
            mw.PlaneWave(
                direction=(0.0, 0.0, 1.0),
                polarization=(1.0, 0.0, 0.0),
                source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=0.7e9),
            )
        )
        return s

    ref, ref_stats = _run(scene(), cuda_graph=False)
    got, got_stats = _run(scene(), cuda_graph=True)
    assert got_stats["cuda_graph_active"] is False  # declined, not captured
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw


def _grating_boundary(*, bloch_wavevector=(math.pi, 0.5 * math.pi, 0.0)):
    return mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector=bloch_wavevector,
    )


def _grating_scene(*, injection):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=_grating_boundary(),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=injection,
            name="grating_tfsf",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_tfsf_currently_rejected_until_implemented():
    scene = _grating_scene(
        injection=mw.TFSF(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.24, 0.24)))
    )
    simulation = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1))
    with pytest.raises(NotImplementedError, match="mixed Bloch boundaries"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_tfsf_slab_runtime_reports_unsupported_until_state_exists():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="slab_tfsf",
        )
    )
    simulation = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1))
    with pytest.raises(NotImplementedError, match="TFSF slab"):
        simulation.prepare()

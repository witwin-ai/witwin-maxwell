from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

_FREQ = 2.0e9


def _pure_bloch_scene():
    # All three faces Bloch-periodic: the solver marches the real field plus a
    # second imaginary split field, coupled by a fixed per-axis wrap phase. No
    # absorbing axis, so the standard (non-CPML) complex Bloch update runs.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.bloch((math.pi / 1.2, math.pi / 2.4, math.pi / 3.6)),
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


def _mixed_bloch_cpml_scene():
    # Two Bloch axes (x, y) + one CPML axis (z): exercises the complex Bloch CPML
    # correction path, whose imaginary psi memory (psi_*_imag) is captured through
    # the ``psi`` snapshot prefix.
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.12, 0.12))),
        grid=mw.GridSpec.uniform(0.008),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=8,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(math.pi / 1.2, math.pi / 2.4, 0.0),
        ),
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


def _run(scene_factory, *, cuda_graph, absorber=None):
    kwargs = dict(
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=300),
        cuda_graph=cuda_graph,
    )
    if absorber is not None:
        kwargs["absorber"] = absorber
    result = mw.Simulation.fdtd(scene_factory(), **kwargs).run()
    view = result.at(frequency=_FREQ)
    fields = {a: np.asarray(getattr(view.E, a).detach().cpu()) for a in ("x", "y", "z")}
    return fields, result.stats()


# The complex split-field Bloch path (Bloch-periodic scene, optionally with a
# single CPML axis) is now captured into the field-update CUDA graph. Capture must
# reproduce the eager host path bit-for-bit: the block marches the real and
# imaginary split fields through the same curl/decay kernels with only a fixed
# wrap phase, carrying no per-step host input, and the zero field (real and
# imaginary) is a fixed point through warmup/capture. The complex running DFT keeps
# the host tail, so the field-update block captures while the tail stays inline.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")
@pytest.mark.parametrize(
    ("scene_factory", "absorber"),
    [(_pure_bloch_scene, None), (_mixed_bloch_cpml_scene, "cpml")],
    ids=["pure_bloch", "mixed_bloch_cpml"],
)
def test_cuda_graph_complex_bloch_matches_non_graph_bitexact(scene_factory, absorber):
    ref, ref_stats = _run(scene_factory, cuda_graph=False, absorber=absorber)
    got, got_stats = _run(scene_factory, cuda_graph=True, absorber=absorber)
    assert ref_stats["cuda_graph_active"] is False
    # Complex Bloch field-update block is captured.
    assert got_stats["cuda_graph_active"] is True
    # The complex running DFT declines the GPU weight table, so the post-source
    # tail stays inline (matching the eager path) and is not graph-captured.
    assert got_stats["tail_graph_active"] is False
    for component, field in ref.items():
        assert np.array_equal(field, got[component]), component

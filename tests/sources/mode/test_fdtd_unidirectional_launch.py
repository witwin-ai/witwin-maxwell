"""Unidirectional mode launch and sparse vector-mode solver robustness.

The rectangular guide (0.20 x 0.30, eps_r = 4) is uniform along ``x`` and the
grid resolves the guided wavelength (dx = 0.0125 -> ~6.1 cells per lambda_g at
2 GHz), so the launched fundamental Ez mode must travel one way: the upstream
plane sees only residual backward radiation, and two downstream planes differ
by the numerical propagation phase ``exp(+j beta L)`` (the running DFT uses the
``exp(+j omega t)`` convention, so forward propagation advances phase as
``+beta x``).

The 0.5 x 0.5 aperture at this resolution has 39 x 39 interior unknowns, above
``_FULL_VECTOR_DENSE_LIMIT``, so these tests also cover the sparse (ARPACK)
full-vector path end to end. Before the ARPACK robustness fixes (global-phase
alignment, degenerate-group realification, duplicate-tolerant grouping) that
path raised ``RuntimeError: ... materially complex field profile`` for exactly
this configuration.
"""

from __future__ import annotations

import cmath
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import (
    _discrete_mode_profile_power,
    solve_mode_source_profile,
)
from witwin.maxwell.postprocess import compute_mode_overlap
from witwin.maxwell.scene import prepare_scene

FREQ = 2.0e9
DX = 0.0125
HALF = 0.32


def _guide_scene(device: str) -> mw.Scene:
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-HALF, HALF),) * 3),
        grid=mw.GridSpec.uniform(DX),
        boundary=mw.BoundarySpec.pml(num_layers=12),
        subpixel_samples=mw.SubpixelSpec(samples=3, averaging="polarized"),
        device=device,
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0, 0, 0), size=(4 * HALF, 0.20, 0.30)),
            material=mw.Material(eps_r=4.0),
            name="waveguide",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(-0.20, 0, 0),
            size=(0, 0.5, 0.5),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=FREQ, fwidth=0.5e9),
            name="mode",
        )
    )
    return scene


def _cpu_mode_context(scene: mw.Scene):
    prepared = prepare_scene(scene.clone(device="cpu"))
    return SimpleNamespace(
        scene=prepared,
        Ex=torch.empty((1,), device=prepared.device, dtype=torch.float32),
        c=299792458.0,
        boundary_kind=prepared.boundary.kind,
        _compiled_material_model=prepared.compile_materials(),
    )


def test_sparse_vector_mode_solve_returns_real_profile_and_stable_neff():
    scene = _guide_scene("cpu")
    context = _cpu_mode_context(scene)
    compiled = _compile_mode_source(scene.sources[0], default_frequency=FREQ)

    effective_indices = []
    for plane_position in (-0.20, 0.05, 0.15):
        positioned = dict(compiled)
        position = list(positioned["position"])
        position[0] = plane_position
        positioned["position"] = tuple(position)
        mode_data = solve_mode_source_profile(context, positioned)
        assert mode_data["mode_solver_kind"] == "vector_sparse"
        expected_shape = (mode_data["coords_u"].numel(), mode_data["coords_v"].numel())
        for name in ("Ez", "Hy"):
            profile = mode_data["component_profiles"][name]
            assert not torch.is_complex(profile)
            assert tuple(profile.shape) == expected_shape
        power = _discrete_mode_profile_power(
            mode_data["component_profiles"],
            coords_u=mode_data["coords_u"],
            coords_v=mode_data["coords_v"],
            normal_axis=mode_data["normal_axis"],
        )
        assert float(torch.abs(power).item()) == pytest.approx(1.0, rel=1e-6)
        effective_indices.append(float(mode_data["effective_index"]))

    for n_eff in effective_indices:
        assert 1.90 < n_eff < 2.00
    spread = max(effective_indices) - min(effective_indices)
    assert spread < 1e-6 * effective_indices[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_mode_source_launch_is_unidirectional_and_propagates_with_grid_beta():
    scene = _guide_scene("cuda")
    for name, position in (("up", -0.26), ("dn1", 0.05), ("dn2", 0.15)):
        scene.add_monitor(
            mw.ModeMonitor(
                name,
                position=(position, 0, 0),
                size=(0, 0.5, 0.5),
                polarization="Ez",
                direction="+",
                frequencies=(FREQ,),
            )
        )

    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=[FREQ],
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
        full_field_dft=False,
    )
    prepared = simulation.prepare()
    c0 = 299792458.0
    steps = int(np.ceil(30 * (4 * HALF) / c0 / float(prepared.solver.dt)))
    simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    result = simulation.run()

    overlaps = {
        name: compute_mode_overlap(result, name, direction="+")
        for name in ("up", "dn1", "dn2")
    }
    amplitudes = {
        name: (
            complex(overlap["amplitude_forward"].cpu().item()),
            complex(overlap["amplitude_backward"].cpu().item()),
        )
        for name, overlap in overlaps.items()
    }

    forward_dn1, backward_dn1 = amplitudes["dn1"]
    forward_dn2, _ = amplitudes["dn2"]
    forward_up, backward_up = amplitudes["up"]

    # (a) One-way launch: the downstream forward amplitude carries the pulse
    # (source-normalized), the backward contamination stays below 15%, and the
    # upstream plane holds only residual backward radiation.
    assert abs(forward_dn1) > 0.7
    assert abs(backward_dn1) / abs(forward_dn1) < 0.15
    assert abs(backward_up) / abs(forward_dn1) < 0.05
    assert abs(forward_up) < abs(backward_up)

    # (b) Plane-to-plane transfer is a pure propagation phase: flat magnitude
    # and a phase advance of +beta*L up to the O((beta*dx)^2/6) grid-dispersion
    # correction (~0.25 rad here; 0.35 rad bound).
    transfer = forward_dn2 / forward_dn1
    assert abs(abs(transfer) - 1.0) < 0.10
    beta = float(overlaps["dn1"]["beta"])
    length = 0.10
    phase_error = cmath.phase(transfer * cmath.exp(-1j * beta * length))
    phase_error = (phase_error + math.pi) % (2.0 * math.pi) - math.pi
    assert abs(phase_error) < 0.35

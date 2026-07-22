"""Zero-impact gate for the staircased surface-impedance (SIBC) machinery.

A scene with NO surface-impedance metal must produce bitwise-identical six-field
Yee state whether the SIBC compile hook (``compile_surface_impedance_layout``)
runs normally or is monkeypatched out to the empty layout it returns for a
SIBC-free scene: the surface-impedance path is fully gated behind a non-empty
layout and perturbs nothing when unused. This mirrors the breakdown track's
zero-impact/parity gate (``tests/breakdown/test_breakdown_parity.py``).

The companion control test proves the monkeypatch is load-bearing -- with a real
``LossyMetalMedium`` structure present, removing the compile hook changes the
fields -- so the zero-impact assertion is not vacuous.

The scene uses a fixed step count, a full PML box, and ``window="none"`` (the
breakdown-suite convention) so the single-GPU FDTD forward is bitwise
deterministic run-to-run; the strictest observable is the raw last-step six-field
Yee state on the solver.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.compiler.materials as material_compiler
from witwin.core import Box
from witwin.maxwell.compiler.materials import CompiledSurfaceImpedanceLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD SIBC requires CUDA"
)

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _last_step_fields(result):
    solver = result.solver
    return {name: getattr(solver, name).clone() for name in _FIELDS}


def _scene(*, metal: bool):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1e6),
        device="cuda",
    )
    if metal:
        # A mid-domain lossy-metal box: its exposed faces (not flush against the
        # PML) staircase into the surface-impedance layout, so the SIBC path is
        # actually engaged.
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.12, 0.24, 0.24)),
                material=mw.LossyMetalMedium(conductivity=50.0),
            )
        )
    scene.add_source(
        mw.PointDipole(
            position=(-0.2, 0.0, 0.0),
            width=0.04,
            polarization=(0.0, 0.0, 1.0),
            source_time=mw.CW(frequency=1e9, amplitude=1.0),
        )
    )
    return scene


def _run(scene):
    return _last_step_fields(
        mw.Simulation.fdtd(
            scene,
            frequencies=[1e9],
            run_time=mw.TimeConfig(time_steps=80),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=True,
        ).run()
    )


def _empty_layout(_scene):
    """Stand-in for ``compile_surface_impedance_layout`` with the hook removed.

    Returns exactly what the real hook returns for a SIBC-free scene, so the
    downstream material compile and runtime configuration see no surface layout.
    """

    return CompiledSurfaceImpedanceLayout(metals=(), faces=(), total_area=0.0)


def test_sibc_free_scene_bitwise_identical_with_compile_hook_removed(monkeypatch):
    """A SIBC-free scene is bit-for-bit identical with the SIBC machinery present
    vs. a monkeypatched-out compile hook: the surface path is a true no-op."""
    present = _run(_scene(metal=False))

    monkeypatch.setattr(
        material_compiler, "compile_surface_impedance_layout", _empty_layout
    )
    removed = _run(_scene(metal=False))

    for name in _FIELDS:
        assert torch.equal(present[name], removed[name]), (
            f"{name} diverged on a SIBC-free scene: "
            f"max|delta|={(present[name] - removed[name]).abs().max().item():.3e}"
        )


def test_removing_compile_hook_changes_a_real_sibc_scene(monkeypatch):
    """Control (teeth): with a real ``LossyMetalMedium`` present, removing the
    compile hook DOES change the fields, so the zero-impact gate is not vacuous."""
    present = _run(_scene(metal=True))

    monkeypatch.setattr(
        material_compiler, "compile_surface_impedance_layout", _empty_layout
    )
    removed = _run(_scene(metal=True))

    diverged = any(
        not torch.equal(present[name], removed[name]) for name in _FIELDS
    )
    assert diverged, (
        "removing the SIBC compile hook must change a scene that carries a metal"
    )

from __future__ import annotations

import os

import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native solver parity tests.")
requires_slang_oracle = pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_SLANG_PARITY") != "1",
    reason="Set WITWIN_RUN_SLANG_PARITY=1 to run Slang-vs-CUDA full-solve parity.",
)
requires_extension_build = pytest.mark.skipif(
    not bool(os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD")),
    reason="native CUDA extension build is opt-in for solver parity tests",
)


def _point_dipole_scene(*, boundary: mw.BoundarySpec | None = None) -> mw.Scene:
    return (
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
            grid=mw.GridSpec.uniform(0.25),
            boundary=boundary or mw.BoundarySpec(kind="none"),
            device="cuda",
        )
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.CW(frequency=1.0e9),
            )
        )
    )


def _run(scene: mw.Scene, monkeypatch: pytest.MonkeyPatch, *, backend: str):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", backend)
    if backend == "cuda":
        monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    else:
        monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)

    result = mw.Simulation.fdtd(
        scene.clone(device="cuda"),
        frequency=1.0e9,
        run_time=mw.TimeConfig(time_steps=4),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).run()
    torch.cuda.synchronize()
    return {
        "Ex": result.E.x.detach().clone(),
        "Ey": result.E.y.detach().clone(),
        "Ez": result.E.z.detach().clone(),
    }


@requires_slang_oracle
@requires_extension_build
@pytest.mark.parametrize(
    "scene",
    [
        _point_dipole_scene(),
        _point_dipole_scene(boundary=mw.BoundarySpec(kind="pml", num_layers=1)),
    ],
)
def test_cuda_full_solve_matches_slang_reference(monkeypatch, scene):
    expected = _run(scene, monkeypatch, backend="slang")
    actual = _run(scene, monkeypatch, backend="cuda")

    for name, expected_field in expected.items():
        torch.testing.assert_close(actual[name], expected_field, rtol=5.0e-5, atol=5.0e-6)

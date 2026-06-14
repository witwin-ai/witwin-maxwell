from __future__ import annotations

import os

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.cuda.runtime.parity import LONG_SOLVE_ATOL, LONG_SOLVE_RTOL


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for native solver performance tests.",
)
requires_slang_oracle = pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_SLANG_PARITY") != "1",
    reason="Set WITWIN_RUN_SLANG_PARITY=1 to run Slang-vs-CUDA performance parity.",
)
requires_extension_build = pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA field kernels.",
)
requires_performance = pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_PERFORMANCE") != "1",
    reason="Set WITWIN_RUN_CUDA_PERFORMANCE=1 to run timing-sensitive CUDA-vs-Slang checks.",
)


_FREQUENCY = 1.0e9
_TIME_STEPS = 64
_MAX_CUDA_TO_SLANG_RATIO = 1.10
_ABSOLUTE_TIMING_SLACK_S = 1.0e-3


def _point_dipole_scene(boundary: mw.BoundarySpec) -> mw.Scene:
    return (
        mw.Scene(
            domain=mw.Domain(bounds=((-0.75, 0.75), (-0.75, 0.75), (-0.75, 0.75))),
            grid=mw.GridSpec.uniform(0.15),
            boundary=boundary,
            device="cuda",
        )
        .add_source(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.CW(frequency=_FREQUENCY),
            )
        )
    )


def _run(scene: mw.Scene, monkeypatch: pytest.MonkeyPatch, *, backend: str, time_steps: int):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", backend)
    if backend == "cuda":
        monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    else:
        monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)

    result = mw.Simulation.fdtd(
        scene.clone(device="cuda"),
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).run()
    torch.cuda.synchronize()
    return result


def _best_elapsed(scene: mw.Scene, monkeypatch: pytest.MonkeyPatch, *, backend: str) -> tuple[float, mw.Result]:
    results = [_run(scene, monkeypatch, backend=backend, time_steps=_TIME_STEPS) for _ in range(2)]
    best = min(results, key=lambda result: result.stats()["elapsed_s"])
    return float(best.stats()["elapsed_s"]), best


@requires_slang_oracle
@requires_extension_build
@requires_performance
@pytest.mark.parametrize(
    ("name", "scene"),
    [
        ("standard", _point_dipole_scene(mw.BoundarySpec.none())),
        ("pml", _point_dipole_scene(mw.BoundarySpec.pml(num_layers=2, strength=1.0))),
    ],
)
def test_native_cuda_full_solve_matches_slang_and_is_not_slower(monkeypatch, name, scene):
    _run(scene, monkeypatch, backend="slang", time_steps=2)
    _run(scene, monkeypatch, backend="cuda", time_steps=2)

    slang_elapsed_s, expected = _best_elapsed(scene, monkeypatch, backend="slang")
    cuda_elapsed_s, actual = _best_elapsed(scene, monkeypatch, backend="cuda")

    for component in ("x", "y", "z"):
        torch.testing.assert_close(
            getattr(actual.E, component),
            getattr(expected.E, component),
            rtol=LONG_SOLVE_RTOL,
            atol=LONG_SOLVE_ATOL,
            msg=f"{name} E.{component}",
        )

    allowed_elapsed_s = max(
        slang_elapsed_s * _MAX_CUDA_TO_SLANG_RATIO,
        slang_elapsed_s + _ABSOLUTE_TIMING_SLACK_S,
    )
    assert cuda_elapsed_s <= allowed_elapsed_s, (
        f"{name} native CUDA solve is slower than Slang: "
        f"cuda={cuda_elapsed_s:.6f}s, slang={slang_elapsed_s:.6f}s, "
        f"allowed={allowed_elapsed_s:.6f}s"
    )

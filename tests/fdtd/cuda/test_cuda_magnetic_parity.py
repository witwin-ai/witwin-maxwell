from __future__ import annotations

import os

import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests.")


def _standard_scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.3),
        boundary=mw.BoundarySpec(kind="none"),
        device="cuda",
    )


def _prepared_solver(monkeypatch, backend: str, scene: mw.Scene):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", backend)
    return mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver


def _seed_standard_state(solver):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).copy_(torch.randn(getattr(solver, name).shape, device="cuda", generator=generator))
    for name in ("cex_decay", "cey_decay", "cez_decay", "chx_decay", "chy_decay", "chz_decay"):
        getattr(solver, name).copy_(0.91 + 0.08 * torch.rand(getattr(solver, name).shape, device="cuda", generator=generator))
    for name in ("cex_curl", "cey_curl", "cez_curl", "chx_curl", "chy_curl", "chz_curl"):
        getattr(solver, name).copy_(1.0e-4 * torch.rand(getattr(solver, name).shape, device="cuda", generator=generator))


def _standard_python_reference(solver):
    ex = solver.Ex.clone()
    ey = solver.Ey.clone()
    ez = solver.Ez.clone()
    hx = solver.Hx.clone()
    hy = solver.Hy.clone()
    hz = solver.Hz.clone()

    hx_next = hx * solver.chx_decay - solver.chx_curl * (
        (ez[:, 1:, :] - ez[:, :-1, :]) * solver.inv_dy
        - (ey[:, :, 1:] - ey[:, :, :-1]) * solver.inv_dz
    )
    hy_next = hy * solver.chy_decay - solver.chy_curl * (
        (ex[:, :, 1:] - ex[:, :, :-1]) * solver.inv_dz
        - (ez[1:, :, :] - ez[:-1, :, :]) * solver.inv_dx
    )
    hz_next = hz * solver.chz_decay - solver.chz_curl * (
        (ey[1:, :, :] - ey[:-1, :, :]) * solver.inv_dx
        - (ex[:, 1:, :] - ex[:, :-1, :]) * solver.inv_dy
    )

    ex_next = ex.clone()
    ey_next = ey.clone()
    ez_next = ez.clone()
    ex_next[:, 1:-1, 1:-1] = ex[:, 1:-1, 1:-1] * solver.cex_decay[:, 1:-1, 1:-1] + solver.cex_curl[:, 1:-1, 1:-1] * (
        (hz_next[:, 1:, 1:-1] - hz_next[:, :-1, 1:-1]) * solver.inv_dy
        - (hy_next[:, 1:-1, 1:] - hy_next[:, 1:-1, :-1]) * solver.inv_dz
    )
    ey_next[1:-1, :, 1:-1] = ey[1:-1, :, 1:-1] * solver.cey_decay[1:-1, :, 1:-1] + solver.cey_curl[1:-1, :, 1:-1] * (
        (hx_next[1:-1, :, 1:] - hx_next[1:-1, :, :-1]) * solver.inv_dz
        - (hz_next[1:, :, 1:-1] - hz_next[:-1, :, 1:-1]) * solver.inv_dx
    )
    ez_next[1:-1, 1:-1, :] = ez[1:-1, 1:-1, :] * solver.cez_decay[1:-1, 1:-1, :] + solver.cez_curl[1:-1, 1:-1, :] * (
        (hy_next[1:, 1:-1, :] - hy_next[:-1, 1:-1, :]) * solver.inv_dx
        - (hx_next[1:-1, 1:, :] - hx_next[1:-1, :-1, :]) * solver.inv_dy
    )
    return {
        "Ex": ex_next,
        "Ey": ey_next,
        "Ez": ez_next,
        "Hx": hx_next,
        "Hy": hy_next,
        "Hz": hz_next,
    }


def test_native_cuda_standard_eh_one_step_matches_python_reference(monkeypatch):
    scene = _standard_scene()
    native_solver = _prepared_solver(monkeypatch, "cuda", scene)
    _seed_standard_state(native_solver)
    expected = _standard_python_reference(native_solver)

    native_solver._update_magnetic_fields(
        native_solver.Hx,
        native_solver.Hy,
        native_solver.Hz,
        native_solver.Ex,
        native_solver.Ey,
        native_solver.Ez,
    )
    native_solver._update_electric_fields(
        native_solver.Ex,
        native_solver.Ey,
        native_solver.Ez,
        native_solver.Hx,
        native_solver.Hy,
        native_solver.Hz,
    )
    torch.cuda.synchronize()

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(
            getattr(native_solver, name),
            expected[name],
            rtol=2.0e-6,
            atol=2.0e-7,
        )


def test_compiled_cuda_extension_standard_eh_one_step_matches_python_reference(monkeypatch):
    if os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1":
        pytest.skip("Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA field kernels.")
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    scene = _standard_scene()
    native_solver = _prepared_solver(monkeypatch, "cuda", scene)
    _seed_standard_state(native_solver)
    expected = _standard_python_reference(native_solver)

    native_solver._update_magnetic_fields(
        native_solver.Hx,
        native_solver.Hy,
        native_solver.Hz,
        native_solver.Ex,
        native_solver.Ey,
        native_solver.Ez,
    )
    native_solver._update_electric_fields(
        native_solver.Ex,
        native_solver.Ey,
        native_solver.Ez,
        native_solver.Hx,
        native_solver.Hy,
        native_solver.Hz,
    )
    torch.cuda.synchronize()

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(
            getattr(native_solver, name),
            expected[name],
            rtol=2.0e-6,
            atol=2.0e-7,
        )

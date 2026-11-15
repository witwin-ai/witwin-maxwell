from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests.")


# Strongly graded, per-axis-distinct node coordinates spanning (-0.6, 0.6).
_GRADED_X = np.array([-0.6, -0.38, -0.2, 0.15, 0.6], dtype=np.float64)
_GRADED_Y = np.array([-0.6, -0.45, -0.05, 0.24, 0.6], dtype=np.float64)
_GRADED_Z = np.array([-0.6, -0.3, 0.02, 0.27, 0.6], dtype=np.float64)


def _standard_scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.3),
        boundary=mw.BoundarySpec(kind="none"),
    )


def _graded_scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.custom(_GRADED_X, _GRADED_Y, _GRADED_Z),
        boundary=mw.BoundarySpec(kind="none"),
    )


def _independent_inv_spacing(nodes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute (inv_dual, inv_primal) from the user node coordinates.

    Derived with numpy, independently of the solver's inv_d*_e/_h tensors, so
    the reference below pins the kernels' per-element indexing rather than
    echoing solver state. Non-periodic boundary dual entries per the design:
    dual[0] = primal[0], dual[-1] = primal[-2].
    """
    primal = np.diff(np.asarray(nodes, dtype=np.float64))
    dual = np.empty(len(nodes), dtype=np.float64)
    dual[1:-1] = 0.5 * (primal[:-1] + primal[1:])
    dual[0] = primal[0]
    dual[-1] = primal[-2]

    def to_device(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(1.0 / values).to(torch.float32).to("cuda")

    return to_device(dual), to_device(primal)


def _prepared_solver(scene: mw.Scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=1.0e9,
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=1),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
        ),
    ).prepare().solver


def _seed_fields(solver):
    # The core path reads the coefficient tables built at preparation, so only the
    # fields are reseeded (W7 row 16e deleted the per-component kernels).
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).copy_(torch.randn(getattr(solver, name).shape, device="cuda", generator=generator))


def _core_step(solver):
    solver._core_runtime.magnetic()
    solver._core_runtime.electric()


def _standard_python_reference(solver, spacing=None):
    ex = solver.Ex.clone()
    ey = solver.Ey.clone()
    ez = solver.Ez.clone()
    hx = solver.Hx.clone()
    hy = solver.Hy.clone()
    hz = solver.Hz.clone()

    # Per-axis spacing arrays: `_h` (primal) scales forward diffs of E in the H
    # updates, `_e` (dual) scales backward diffs of H in the E updates.
    if spacing is None:
        spacing = {
            name: getattr(solver, name)
            for name in ("inv_dx_e", "inv_dy_e", "inv_dz_e", "inv_dx_h", "inv_dy_h", "inv_dz_h")
        }
    inv_dx_h = spacing["inv_dx_h"].view(-1, 1, 1)
    inv_dy_h = spacing["inv_dy_h"].view(1, -1, 1)
    inv_dz_h = spacing["inv_dz_h"].view(1, 1, -1)
    inv_dx_e = spacing["inv_dx_e"][1:-1].view(-1, 1, 1)
    inv_dy_e = spacing["inv_dy_e"][1:-1].view(1, -1, 1)
    inv_dz_e = spacing["inv_dz_e"][1:-1].view(1, 1, -1)

    hx_next = hx * solver.chx_decay - solver.chx_curl * (
        (ez[:, 1:, :] - ez[:, :-1, :]) * inv_dy_h
        - (ey[:, :, 1:] - ey[:, :, :-1]) * inv_dz_h
    )
    hy_next = hy * solver.chy_decay - solver.chy_curl * (
        (ex[:, :, 1:] - ex[:, :, :-1]) * inv_dz_h
        - (ez[1:, :, :] - ez[:-1, :, :]) * inv_dx_h
    )
    hz_next = hz * solver.chz_decay - solver.chz_curl * (
        (ey[1:, :, :] - ey[:-1, :, :]) * inv_dx_h
        - (ex[:, 1:, :] - ex[:, :-1, :]) * inv_dy_h
    )

    ex_next = ex.clone()
    ey_next = ey.clone()
    ez_next = ez.clone()
    ex_next[:, 1:-1, 1:-1] = ex[:, 1:-1, 1:-1] * solver.cex_decay[:, 1:-1, 1:-1] + solver.cex_curl[:, 1:-1, 1:-1] * (
        (hz_next[:, 1:, 1:-1] - hz_next[:, :-1, 1:-1]) * inv_dy_e
        - (hy_next[:, 1:-1, 1:] - hy_next[:, 1:-1, :-1]) * inv_dz_e
    )
    ey_next[1:-1, :, 1:-1] = ey[1:-1, :, 1:-1] * solver.cey_decay[1:-1, :, 1:-1] + solver.cey_curl[1:-1, :, 1:-1] * (
        (hx_next[1:-1, :, 1:] - hx_next[1:-1, :, :-1]) * inv_dz_e
        - (hz_next[1:, :, 1:-1] - hz_next[:-1, :, 1:-1]) * inv_dx_e
    )
    ez_next[1:-1, 1:-1, :] = ez[1:-1, 1:-1, :] * solver.cez_decay[1:-1, 1:-1, :] + solver.cez_curl[1:-1, 1:-1, :] * (
        (hy_next[1:, 1:-1, :] - hy_next[:-1, 1:-1, :]) * inv_dx_e
        - (hx_next[1:-1, 1:, :] - hx_next[1:-1, :-1, :]) * inv_dy_e
    )
    return {
        "Ex": ex_next,
        "Ey": ey_next,
        "Ez": ez_next,
        "Hx": hx_next,
        "Hy": hy_next,
        "Hz": hz_next,
    }


def test_core_standard_eh_one_step_matches_python_reference_with_prepared_coefficients():
    # The core path reads the coefficient tables built at preparation, so only
    # the fields are reseeded here.
    native_solver = _prepared_solver(_standard_scene())
    assert native_solver._core_runtime is not None
    _seed_fields(native_solver)
    expected = _standard_python_reference(native_solver)
    _core_step(native_solver)
    torch.cuda.synchronize()

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(getattr(native_solver, name), expected[name], rtol=2.0e-6, atol=2.0e-7)


def test_core_standard_eh_one_step_matches_independent_reference_on_graded_grid():
    # Graded per-axis-distinct custom grid; the reference spacing arrays are
    # recomputed from the raw node coordinates so a per-element indexing
    # regression in the forward kernels (or in the solver's inv_d*_e/_h
    # construction) fails this test instead of cancelling out.
    scene = _graded_scene()
    native_solver = _prepared_solver(scene)
    _seed_fields(native_solver)
    inv_dx_e, inv_dx_h = _independent_inv_spacing(_GRADED_X)
    inv_dy_e, inv_dy_h = _independent_inv_spacing(_GRADED_Y)
    inv_dz_e, inv_dz_h = _independent_inv_spacing(_GRADED_Z)
    expected = _standard_python_reference(
        native_solver,
        spacing={
            "inv_dx_e": inv_dx_e,
            "inv_dy_e": inv_dy_e,
            "inv_dz_e": inv_dz_e,
            "inv_dx_h": inv_dx_h,
            "inv_dy_h": inv_dy_h,
            "inv_dz_h": inv_dz_h,
        },
    )

    _core_step(native_solver)
    torch.cuda.synchronize()

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        torch.testing.assert_close(
            getattr(native_solver, name),
            expected[name],
            rtol=2.0e-6,
            atol=2.0e-7,
        )

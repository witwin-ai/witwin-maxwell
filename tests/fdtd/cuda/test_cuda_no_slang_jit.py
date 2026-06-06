from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests.")


def _minimal_cuda_scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec(kind="none"),
        device="cuda",
    )


def test_cuda_backend_fdtd_solve_does_not_call_slang_load_module(monkeypatch):
    import slangtorch

    def fail_load_module(*args, **kwargs):
        raise AssertionError("native CUDA backend must not call slangtorch.loadModule()")

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    monkeypatch.setattr(slangtorch, "loadModule", fail_load_module)

    sim = mw.Simulation.fdtd(
        _minimal_cuda_scene(),
        frequency=1.0e9,
        run_time=mw.TimeConfig(time_steps=2),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    )

    result = sim.run()

    assert result.method == "fdtd"
    assert result.E.x.is_cuda
    assert result.stats()["time_steps"] == 2

from __future__ import annotations

import math
import os

import pytest
import torch

from tests.fdtd.cuda._parity_backend import backend


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests."),
    pytest.mark.usefixtures("native_cuda_extension"),
]


def _seeded(shape: tuple[int, int, int], generator: torch.Generator, *, scale: float = 1.0) -> torch.Tensor:
    return scale * torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)


def _coeff(shape: tuple[int, int, int], generator: torch.Generator) -> torch.Tensor:
    return 0.85 + 0.1 * torch.rand(shape, device="cuda", dtype=torch.float32, generator=generator)


def _spacing(length: int, value: float) -> torch.Tensor:
    # Graded per-element values (including distinct wrap entries at [0] and
    # [-1]) so a kernel indexing regression cannot hide behind constant arrays.
    return torch.linspace(value, 1.4 * value, length, device="cuda", dtype=torch.float32)


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA Bloch electric kernels.",
)
def test_compiled_cuda_bloch_electric_updates_match_torch_dispatcher(monkeypatch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2521)

    ex_shape = (3, 4, 5)
    ey_shape = (4, 3, 5)
    ez_shape = (4, 4, 4)
    hx_shape = (4, 3, 4)
    hy_shape = (3, 4, 4)
    hz_shape = (3, 3, 5)
    ex_real, ex_imag = _seeded(ex_shape, generator), _seeded(ex_shape, generator)
    ey_real, ey_imag = _seeded(ey_shape, generator), _seeded(ey_shape, generator)
    ez_real, ez_imag = _seeded(ez_shape, generator), _seeded(ez_shape, generator)
    hx_real, hx_imag = _seeded(hx_shape, generator), _seeded(hx_shape, generator)
    hy_real, hy_imag = _seeded(hy_shape, generator), _seeded(hy_shape, generator)
    hz_real, hz_imag = _seeded(hz_shape, generator), _seeded(hz_shape, generator)
    ex_decay, ey_decay, ez_decay = _coeff(ex_shape, generator), _coeff(ey_shape, generator), _coeff(ez_shape, generator)
    ex_curl, ey_curl, ez_curl = _coeff(ex_shape, generator) * 1.0e-3, _coeff(ey_shape, generator) * 1.0e-3, _coeff(ez_shape, generator) * 1.0e-3
    phase_x = (math.cos(0.23), math.sin(0.23))
    phase_y = (math.cos(-0.37), math.sin(-0.37))
    phase_z = (math.cos(0.61), math.sin(0.61))
    inv_dx, inv_dy, inv_dz = _spacing(4, 1.5), _spacing(4, 0.75), _spacing(5, 1.25)

    expected = {
        "ex_real": ex_real.clone(),
        "ex_imag": ex_imag.clone(),
        "ey_real": ey_real.clone(),
        "ey_imag": ey_imag.clone(),
        "ez_real": ez_real.clone(),
        "ez_imag": ez_imag.clone(),
    }
    actual = {name: value.clone() for name, value in expected.items()}

    def launch(values):
        backend._electric_ex_bloch(
            ExReal=values["ex_real"],
            ExImag=values["ex_imag"],
            HyReal=hy_real,
            HyImag=hy_imag,
            HzReal=hz_real,
            HzImag=hz_imag,
            ExDecay=ex_decay,
            ExCurl=ex_curl,
            phaseCosY=phase_y[0],
            phaseSinY=phase_y[1],
            phaseCosZ=phase_z[0],
            phaseSinZ=phase_z[1],
            invDy=inv_dy,
            invDz=inv_dz,
        )
        backend._electric_ey_bloch(
            EyReal=values["ey_real"],
            EyImag=values["ey_imag"],
            HxReal=hx_real,
            HxImag=hx_imag,
            HzReal=hz_real,
            HzImag=hz_imag,
            EyDecay=ey_decay,
            EyCurl=ey_curl,
            phaseCosX=phase_x[0],
            phaseSinX=phase_x[1],
            phaseCosZ=phase_z[0],
            phaseSinZ=phase_z[1],
            invDx=inv_dx,
            invDz=inv_dz,
        )
        backend._electric_ez_bloch(
            EzReal=values["ez_real"],
            EzImag=values["ez_imag"],
            HxReal=hx_real,
            HxImag=hx_imag,
            HyReal=hy_real,
            HyImag=hy_imag,
            EzDecay=ez_decay,
            EzCurl=ez_curl,
            phaseCosX=phase_x[0],
            phaseSinX=phase_x[1],
            phaseCosY=phase_y[0],
            phaseSinY=phase_y[1],
            invDx=inv_dx,
            invDy=inv_dy,
        )

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    launch(expected)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    launch(actual)
    torch.cuda.synchronize()

    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=2.0e-6, atol=2.0e-7)


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA boundary kernels.",
)
def test_compiled_cuda_pec_boundary_clamp_matches_torch_dispatcher(monkeypatch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3031)
    expected = _seeded((5, 4, 3), generator)
    actual = expected.clone()

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._clamp_pec_boundary(field=expected, axisA=0, axisB=2)

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._clamp_pec_boundary(field=actual, axisA=0, axisB=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

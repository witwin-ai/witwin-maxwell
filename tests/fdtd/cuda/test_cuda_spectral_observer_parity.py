from __future__ import annotations

import os

import pytest
import torch

from tests.fdtd.cuda._parity_backend import backend


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests.")


def _rand(shape, generator):
    return torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA spectral kernels.",
)
def test_compiled_cuda_extension_batched_dft_matches_torch_dispatcher(monkeypatch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(404)
    ex = _rand((2, 3, 4), generator)
    ey = _rand((3, 2, 4), generator)
    ez = _rand((3, 3, 3), generator)
    weighted_cos = torch.tensor([0.25, -0.5, 0.75], device="cuda", dtype=torch.float32)
    weighted_sin = torch.tensor([-0.125, 0.375, 0.625], device="cuda", dtype=torch.float32)

    reference = [torch.zeros((3, *field.shape), device="cuda", dtype=torch.float32) for field in (ex, ex, ey, ey, ez, ez)]
    actual = [tensor.clone() for tensor in reference]

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._accumulate_dft_batched(
        Ex=ex,
        Ey=ey,
        Ez=ez,
        ExRealAccum=reference[0],
        ExImagAccum=reference[1],
        EyRealAccum=reference[2],
        EyImagAccum=reference[3],
        EzRealAccum=reference[4],
        EzImagAccum=reference[5],
        weightedCos=weighted_cos,
        weightedSin=weighted_sin,
    )

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._accumulate_dft_batched(
        Ex=ex,
        Ey=ey,
        Ez=ez,
        ExRealAccum=actual[0],
        ExImagAccum=actual[1],
        EyRealAccum=actual[2],
        EyImagAccum=actual[3],
        EzRealAccum=actual[4],
        EzImagAccum=actual[5],
        weightedCos=weighted_cos,
        weightedSin=weighted_sin,
    )
    torch.cuda.synchronize()

    for actual_tensor, reference_tensor in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=2.0e-6, atol=2.0e-7)


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA spectral kernels.",
)
def test_compiled_cuda_extension_large_batched_dft_split_path_matches_torch_dispatcher(monkeypatch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(405)
    shape = (128, 128, 64)
    frequency_count = 4
    ex = _rand(shape, generator)
    ey = _rand(shape, generator)
    ez = _rand(shape, generator)
    weighted_cos = torch.linspace(-0.5, 0.75, frequency_count, device="cuda", dtype=torch.float32)
    weighted_sin = torch.linspace(0.25, -0.625, frequency_count, device="cuda", dtype=torch.float32)

    reference = [
        torch.zeros((frequency_count, *field.shape), device="cuda", dtype=torch.float32)
        for field in (ex, ex, ey, ey, ez, ez)
    ]
    actual = [tensor.clone() for tensor in reference]

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._accumulate_dft_batched(
        Ex=ex,
        Ey=ey,
        Ez=ez,
        ExRealAccum=reference[0],
        ExImagAccum=reference[1],
        EyRealAccum=reference[2],
        EyImagAccum=reference[3],
        EzRealAccum=reference[4],
        EzImagAccum=reference[5],
        weightedCos=weighted_cos,
        weightedSin=weighted_sin,
    )

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._accumulate_dft_batched(
        Ex=ex,
        Ey=ey,
        Ez=ez,
        ExRealAccum=actual[0],
        ExImagAccum=actual[1],
        EyRealAccum=actual[2],
        EyImagAccum=actual[3],
        EzRealAccum=actual[4],
        EzImagAccum=actual[5],
        weightedCos=weighted_cos,
        weightedSin=weighted_sin,
    )
    torch.cuda.synchronize()

    for actual_tensor, reference_tensor in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=2.0e-6, atol=2.0e-7)


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA observer kernels.",
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.long])
def test_compiled_cuda_extension_point_and_plane_observers_match_torch_dispatcher(monkeypatch, index_dtype):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(406)
    field = _rand((4, 5, 6), generator)
    point_i = torch.tensor([0, 1, 3], device="cuda", dtype=index_dtype)
    point_j = torch.tensor([2, 4, 1], device="cuda", dtype=index_dtype)
    point_k = torch.tensor([5, 0, 3], device="cuda", dtype=index_dtype)

    reference_point_real = torch.zeros(3, device="cuda")
    reference_point_imag = torch.zeros(3, device="cuda")
    actual_point_real = torch.zeros_like(reference_point_real)
    actual_point_imag = torch.zeros_like(reference_point_imag)
    reference_plane_real = torch.zeros((4, 6), device="cuda")
    reference_plane_imag = torch.zeros((4, 6), device="cuda")
    actual_plane_real = torch.zeros_like(reference_plane_real)
    actual_plane_imag = torch.zeros_like(reference_plane_imag)

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._accumulate_point_observers(
        field=field,
        pointI=point_i,
        pointJ=point_j,
        pointK=point_k,
        realAccum=reference_point_real,
        imagAccum=reference_point_imag,
        weightedCos=0.5,
        weightedSin=-0.25,
    )
    backend._accumulate_plane_observer(
        field=field,
        planeRealAccum=reference_plane_real,
        planeImagAccum=reference_plane_imag,
        axisCode=1,
        planeIndex=2,
        weightedCos=-0.75,
        weightedSin=0.125,
    )

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._accumulate_point_observers(
        field=field,
        pointI=point_i,
        pointJ=point_j,
        pointK=point_k,
        realAccum=actual_point_real,
        imagAccum=actual_point_imag,
        weightedCos=0.5,
        weightedSin=-0.25,
    )
    backend._accumulate_plane_observer(
        field=field,
        planeRealAccum=actual_plane_real,
        planeImagAccum=actual_plane_imag,
        axisCode=1,
        planeIndex=2,
        weightedCos=-0.75,
        weightedSin=0.125,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_point_real, reference_point_real, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_point_imag, reference_point_imag, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_plane_real, reference_plane_real, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_plane_imag, reference_plane_imag, rtol=2.0e-6, atol=2.0e-7)

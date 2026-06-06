from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests.")


def test_debug_linear_indices_match_contiguous_row_major_layout():
    from witwin.maxwell.fdtd.cuda import backend

    shape = (3, 4, 5)
    linear, i_index, j_index, k_index = backend.debug_linear_indices(shape, device="cuda")
    expected = torch.arange(3 * 4 * 5, device="cuda", dtype=torch.int64).reshape(shape)

    torch.cuda.synchronize()

    torch.testing.assert_close(linear, expected)
    torch.testing.assert_close(i_index, expected // (shape[1] * shape[2]))
    torch.testing.assert_close(j_index, (expected // shape[2]) % shape[1])
    torch.testing.assert_close(k_index, expected % shape[2])


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
def test_extension_debug_linear_indices_match_contiguous_row_major_layout():
    from witwin.maxwell.fdtd.cuda import backend

    shape = (3, 4, 5)
    linear, i_index, j_index, k_index = backend.debug_linear_indices(shape, device="cuda", use_extension=True)
    expected = torch.arange(3 * 4 * 5, device="cuda", dtype=torch.int64).reshape(shape)

    torch.cuda.synchronize()

    torch.testing.assert_close(linear, expected)
    torch.testing.assert_close(i_index, expected // (shape[1] * shape[2]))
    torch.testing.assert_close(j_index, (expected // shape[2]) % shape[1])
    torch.testing.assert_close(k_index, expected % shape[2])

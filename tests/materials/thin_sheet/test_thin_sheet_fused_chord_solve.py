"""Orchestration control for the cooperative chord kernels against the Torch sheet oracle."""
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.cuda import backend

from tests.oracles import thin_sheet_torch

from .test_thin_sheet_fused_copies import _prepare
from .test_thin_sheet_feeds import _simulation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fed", [False, True])
def test_chord_schedule_with_kernel_doubles(fed, monkeypatch):
    """Exercise real preparation/scalars and both sweeps with Torch stand-ins for the chord kernels."""
    solver = _simulation(tilted=True, steps=1).prepare().solver if fed else _prepare()
    sheet = solver._thin_sheets[0]
    calls = []

    def initialize(rhs, y, residual):
        calls.append("initialize")
        y.zero_()
        residual.copy_(rhs)

    def product(rows, columns, values, x):
        matrix = torch.sparse_csr_tensor(rows, columns, values, size=(x.numel(), x.numel()))
        return torch.mv(matrix, x)

    def matvec(rows, columns, values, lanes, residual, temporary):
        calls.append("matvec")
        temporary.copy_(product(rows, columns, values, residual))

    def direction(rows, columns, values, lanes, temporary, d, alpha, beta, setup):
        calls.append("setup" if setup else "direction")
        value = beta * product(rows, columns, values, temporary)
        if not setup:
            value = alpha * d + value
        d.copy_(value)

    def residual(rows, columns, values, lanes, d, y, r):
        calls.append("residual")
        value = product(rows, columns, values, d)
        y.add_(d)
        r.sub_(value)

    extension = SimpleNamespace(thin_sheet_copper_rhs=None, thin_sheet_chord_forcing=None,
                                thin_sheet_copper_state=None, thin_sheet_chord_initialize=initialize,
                                thin_sheet_chord_matvec=matvec,
                                thin_sheet_chord_direction=direction, thin_sheet_chord_residual=residual)
    monkeypatch.setattr(backend, "get_compiled_extension", lambda: extension)
    sheet._init_copper_fused()
    cu = sheet.copper
    if fed:
        assert int((cu["R"].crow_indices()[1:] - cu["R"].crow_indices()[:-1]).max()) > 2
    degree = cu["chebyshev"][2]
    sweep = ["matvec", "setup"] + ["residual", "matvec", "direction"] * (degree - 1) + ["residual"]
    generator = torch.Generator().manual_seed(314)
    results = []
    for _ in range(2):
        rhs = torch.randn(2 * cu["count"], dtype=torch.float64, generator=generator).to(sheet.device)
        saved_rhs = rhs.clone()
        expected = thin_sheet_torch.chord_solve(sheet, rhs)
        # Setup must overwrite stale scratch without reading the previous direction.
        for name in ("residual", "direction", "temporary"):
            cu["fused"]["chord_solve"][name].fill_(float("nan"))
        calls.clear()
        actual = sheet.chord_solve(rhs)
        assert calls == ["initialize"] + 2 * sweep
        assert torch.equal(rhs, saved_rhs)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=0.)
        results.append((actual, actual.clone()))
    for actual, saved in results:
        assert torch.equal(actual, saved)

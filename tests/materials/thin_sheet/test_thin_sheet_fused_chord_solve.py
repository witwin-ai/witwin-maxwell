"""Dispatch and CPU orchestration controls for cooperative chord kernels."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd import thin_sheet
from witwin.maxwell.fdtd.cuda import backend
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block

from .test_thin_sheet_fused_copies import _prepare
from .test_thin_sheet_feeds import _simulation


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kernel", [None, "fused", "torch"], ids=["default", "fused", "torch"])
def test_chord_solve_dispatch(device, kernel, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    solver = _prepare(device=device, kernel=kernel)
    calls = Counter()
    for name in ("_chord_solve_fused", "_chord_solve_torch"):
        original = getattr(thin_sheet, name)

        def counted(*args, _name=name, _original=original):
            calls[_name] += 1
            return _original(*args)

        monkeypatch.setattr(thin_sheet, name, counted)
    _field_update_block(solver, 0.)
    fused = device == "cuda" and kernel != "torch"
    assert calls["_chord_solve_fused"] == int(fused)
    assert calls["_chord_solve_torch"] == int(not fused)
    print(dict(dispatch=device, selector=kernel or "default", **calls), flush=True)


@pytest.mark.parametrize("fed", [False, True])
def test_chord_schedule_with_cpu_kernel_doubles(fed, monkeypatch):
    """Exercise real preparation/scalars and both sweeps without loading CUDA."""
    solver = (_simulation(device="cpu", tilted=True, steps=1).prepare().solver if fed
              else _prepare(device="cpu", kernel="torch"))
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
        rhs = torch.randn(2 * cu["count"], dtype=torch.float64, generator=generator)
        saved_rhs = rhs.clone()
        expected = thin_sheet._chord_solve_torch(sheet, rhs)
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

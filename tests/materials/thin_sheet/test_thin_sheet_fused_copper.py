"""Elementwise, accumulator and dispatch gates for native copper updates."""
import os
from unittest.mock import patch

import pytest
import torch

from witwin.maxwell.fdtd import thin_sheet
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, _port_step_block
from witwin.maxwell.fdtd.ports import prepare_port_spectral_accumulators, prepare_port_drive_tables

from .test_thin_sheet_fused_copies import FIELDS, SELECTOR, _prepare, _seed_identical
from . import test_thin_sheet_feeds as feeds


def elementwise_relative(a, b):
    if not a.numel():
        return 0.
    scale = torch.maximum(a.abs(), b.abs())
    difference = (a - b).abs()
    return float(torch.where(scale == 0, 0., difference / scale).max())


def scaled_elementwise(a, b):
    """max |a - b| / max |b|: elementwise, but scaled by the tensor (user-approved E1 for copper state)."""
    if not a.numel():
        return 0.
    scale = torch.maximum(a.abs().max(), b.abs().max())
    return float(torch.where(scale == 0, 0., (a - b).abs().max() / scale))


def normwise_relative(a, b):
    if not a.numel():
        return 0.
    difference = torch.linalg.vector_norm((a-b).reshape(-1))
    scale = torch.maximum(torch.linalg.vector_norm(a.reshape(-1)), torch.linalg.vector_norm(b.reshape(-1)))
    return float(torch.where(scale == 0, 0., difference / scale))


def max_ulp(a, b):
    # Ordered integer representations make neighboring floats one unit apart.
    def ordered(value):
        bits = value.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(bits < 0, -2147483648 - bits, bits)
    return int((ordered(a)-ordered(b)).abs().max()) if a.numel() else 0


def compare_step(a, b):
    sheet, reference = a._thin_sheets[0], b._thin_sheets[0]
    metrics = {"chord_e": elementwise_relative(sheet.chord_e, reference.chord_e),
               # Per-element relative is ill-posed on cancellation residuals of the modal state
               # (docs/assessments/thin-sheet-fused-copper-step2-diagnosis-2026-11-01.md).
               "copper_state": scaled_elementwise(sheet.copper["state"], reference.copper["state"]),
               "pair_e": elementwise_relative(sheet.e_new[sheet.copper["pair_position"]],
                                               reference.e_new[reference.copper["pair_position"]]),
               "private_e": elementwise_relative(sheet.e_new, reference.e_new),
               "private_h": elementwise_relative(sheet.h_new, reference.h_new),
               "loss": normwise_relative(sheet.loss, reference.loss),
               "field_max_ulp": max(max_ulp(getattr(a, name), getattr(b, name)) for name in FIELDS)}
    for feed, other in zip(sheet.feeds, reference.feeds):
        for name in ("drive", "voltage", "current", "column_voltage", "column_sum"):
            metrics["feed_"+name] = normwise_relative(getattr(feed["binding"], name),
                                                     getattr(other["binding"], name))
    return metrics


def assert_step_budget(metrics):
    for name, value in metrics.items():
        assert value <= (1 if name == "field_max_ulp" else 1e-13), (name, value, metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fed,tilted", [(False, True), (True, True), (True, False)])
def test_one_copper_step_elementwise_and_accumulators(fed, tilted):
    if fed:
        solvers = []
        for kernel in ("fused", "torch"):
            with patch.dict(os.environ, {SELECTOR: kernel}):
                solvers.append(feeds._simulation(device="cuda", tilted=tilted, steps=1).prepare().solver)
        fused, reference = solvers
    else:
        fused, reference = _prepare(), _prepare(kernel="torch")
    _seed_identical(fused, reference)
    # Loss starts at zero so a prior accumulator cannot mask this step's error.
    for solver in (fused, reference):
        sheet = solver._thin_sheets[0]
        sheet.loss.zero_()
        sheet.begin(solver)
        sheet.magnetic(solver)
        sheet.electric(solver)
    metrics = compare_step(fused, reference)
    print(dict(E1="copper", fed=fed, tilted=tilted, chords=fused._thin_sheets[0].copper["count"],
               pairs=fused._thin_sheets[0].copper["pair_position"].shape[0], **metrics), flush=True)
    assert_step_budget(metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_one_fed_step_spectral_accumulators():
    solvers = []
    for kernel in ("fused", "torch"):
        with patch.dict(os.environ, {SELECTOR: kernel}):
            solver = feeds._simulation(device="cuda", tilted=True, steps=20).prepare().solver
        prepare_port_spectral_accumulators(solver, 20, "none")
        prepare_port_drive_tables(solver, 20)
        solvers.append(solver)
    _seed_identical(*solvers)
    for solver in solvers:
        solver._thin_sheets[0].loss.zero_()
        solver._thin_sheets[0].feeds[0]["binding"].column_sum.zero_()
        solver._port_observer_step.fill_(10)
        _field_update_block(solver, 10*solver.dt)
        _port_step_block(solver, solver._port_observer_step, post_source=lambda: None)
    metrics = compare_step(*solvers)
    assert bool(solvers[0]._thin_sheets[0].feeds[0]["binding"].column_sum.abs().any())
    print(dict(E1b="fed_spectra", **metrics), flush=True)
    assert_step_budget(metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_copper_repeat_run_is_bitwise_deterministic():
    first, second = _prepare(kernel="fused"), _prepare(kernel="fused")
    _seed_identical(first, second)
    for solver in (first, second):
        assert "fused" in solver._thin_sheets[0].copper
        for step in range(20):
            _field_update_block(solver, step * solver.dt)
    for name in FIELDS:
        a, b = getattr(first, name), getattr(second, name)
        assert a.dtype == b.dtype == torch.float32
        assert torch.equal(a, b), name
    left, right = first._thin_sheets[0], second._thin_sheets[0]
    for name, a, b in (("chord_e", left.chord_e, right.chord_e),
                       ("state", left.copper["state"], right.copper["state"])):
        assert a.dtype == b.dtype == torch.float64
        assert torch.equal(a, b), name


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kernel", [None, "fused", "torch"], ids=["default", "fused", "torch"])
def test_copper_dispatch(device, kernel, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    solver = _prepare(device=device, kernel=kernel)
    calls = {name: 0 for part in ("_copper_rhs", "_chord_forcing", "_copper_state")
             for name in (part+"_fused", part+"_torch")}
    for name in calls:
        original = getattr(thin_sheet, name)

        def counted(*args, _name=name, _original=original):
            calls[_name] += 1
            return _original(*args)

        monkeypatch.setattr(thin_sheet, name, counted)
    _field_update_block(solver, 0.)
    fused = device == "cuda" and kernel != "torch"
    for name, count in calls.items():
        assert count == int(name.endswith("fused") == fused)
    print(dict(dispatch=device, selector=kernel or "default", **calls), flush=True)

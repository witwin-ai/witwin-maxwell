"""Bitwise index-copy, one-step and graph gates for fused sheet plans.

The reference arm is the frozen Torch thin-sheet oracle (tests/oracles/thin_sheet_torch.py).
"""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import graph_state_tensors
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, _make_field_update_runner

from tests.oracles import thin_sheet_torch

from . import test_thin_sheet_cuda as scenes

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _prepare(kind="copper"):
    scene = scenes._cavity(kind, "cuda")
    return mw.Simulation.fdtd(scene, frequencies=[2e9], config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=1))).prepare().solver


@pytest.fixture(scope="module")
def copy_solver():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return _prepare()


def _bits_equal(a, b):
    integer = torch.int32 if a.dtype == torch.float32 else torch.int64
    assert a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(integer), b.contiguous().view(integer))


def _special_values(device):
    return torch.tensor([0., -0., 1., -1., .1, 1 + 2**-24, 1 + 3 * 2**-24,
                         2**-149, -2**-149, 2**-150, -2**-150, 2**-1074,
                         float("inf"), -float("inf"), float("nan")],
                        dtype=torch.float64, device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["patch_h", "patch_e", "h_cols", "e_rows", "native", "extra", "empty"])
def test_gather_scatter_matches_torch_bits(copy_solver, monkeypatch, dtype, name):
    solver = copy_solver
    sheet = solver._thin_sheets[0]
    if name in ("native", "extra", "empty"):
        slots = sheet.complex.e_extra
        subset = (np.flatnonzero(slots < 0) if name == "native" else
                  np.flatnonzero(slots >= 0) if name == "extra" else np.empty(0, dtype=np.int64))
        plan = sheet._subset_plan(solver, "e", subset)
    else:
        plan = getattr(sheet, "plan_" + name)
    special = _special_values(sheet.device)
    generator = torch.Generator(device=sheet.device).manual_seed(73)
    for field_name in plan["fused"]["names"]:
        field = getattr(solver, field_name)
        monkeypatch.setattr(solver, field_name, torch.randn(field.shape, dtype=dtype,
                                                         device=sheet.device, generator=generator))
    for field_name, _, offsets in plan["native"]:
        n = min(len(offsets), len(special))
        getattr(solver, field_name).view(-1)[offsets[:n]] = special[:n].to(dtype)
        assert offsets.unique().numel() == offsets.numel()
    extra = torch.randn(sheet.e_extra.shape if plan["fused"]["names"][0] == "Ex" else sheet.h_extra.shape,
                        dtype=torch.float64, device=sheet.device, generator=generator)
    n = min(len(plan["extra_slots"]), len(special))
    extra[plan["extra_slots"][:n]] = special[:n]
    assert plan["extra_slots"].unique().numel() == plan["extra_slots"].numel()
    expected = thin_sheet_torch.gather(solver, plan, extra, torch.empty(plan["size"], dtype=torch.float64,
                                                                      device=sheet.device))
    actual = sheet.gather(solver, plan, extra)
    _bits_equal(actual, expected)
    out = torch.empty_like(actual)
    assert sheet.gather(solver, plan, extra, out) is out
    _bits_equal(out, expected)

    values = torch.randn(plan["size"], dtype=torch.float64, device=sheet.device, generator=generator)
    n = min(values.numel(), special.numel())
    values[:n] = special[:n]
    saved_fields = {field_name: getattr(solver, field_name).clone() for field_name in plan["fused"]["names"]}
    saved_extra = extra.clone()
    thin_sheet_torch.scatter(solver, plan, extra, values)
    expected_fields = {field_name: getattr(solver, field_name).clone() for field_name in saved_fields}
    expected_extra = extra.clone()
    for field_name, field in saved_fields.items():
        getattr(solver, field_name).copy_(field)
    extra.copy_(saved_extra)
    sheet.scatter(solver, plan, extra, values)
    for field_name in saved_fields:
        _bits_equal(getattr(solver, field_name), expected_fields[field_name])
    _bits_equal(extra, expected_extra)


def _seed_identical(a, b):
    generator = torch.Generator(device=a.device).manual_seed(91)
    left, right = graph_state_tensors(a), graph_state_tensors(b)
    assert left.keys() == right.keys()
    for name, tensor in left.items():
        value = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator) * 1e-4
        tensor.copy_(value)
        right[name].copy_(value)


def _assert_step_equal(a, b):
    left, right = graph_state_tensors(a), graph_state_tensors(b)
    for name in left:
        _bits_equal(left[name], right[name])
    for name in ("e_new", "h_new", "e_old", "h_old", "e_cols", "h_rows"):
        _bits_equal(getattr(a._thin_sheets[0], name), getattr(b._thin_sheets[0], name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["pec", "copper"])
def test_one_field_step_matches_torch(kind, monkeypatch):
    fused, reference = _prepare(kind), _prepare(kind)
    _seed_identical(fused, reference)
    _field_update_block(fused, 0.)
    with monkeypatch.context() as patch:
        thin_sheet_torch.install(patch)
        _field_update_block(reference, 0.)
    if kind == "pec":
        _assert_step_equal(fused, reference)
        print(dict(E1=kind, fp64_difference=0, fp32_max_ulp=0), flush=True)
    else:
        from .test_thin_sheet_fused_copper import compare_step, assert_step_budget
        metrics = compare_step(fused, reference)
        print(dict(E1=kind, **metrics), flush=True)
        assert_step_budget(metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["pec", "copper"])
def test_fused_graph_matches_eager_from_nonzero_state(kind):
    graph, eager = _prepare(kind), _prepare(kind)
    _seed_identical(graph, eager)
    run = _make_field_update_runner(graph, True)
    assert graph._cuda_graph_active
    for _ in range(5):
        run(0.)
        _field_update_block(eager, 0.)
        _assert_step_equal(graph, eager)
    print(dict(E3=kind, graph_eager_bit_difference=0, steps=5), flush=True)

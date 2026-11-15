"""NaN-poison probe: the nonlinear E coefficient scratch is written before it is read.

build_nonlinear allocates ``c{e}{x,y,z}_{curl,decay}_dynamic`` with ``torch.empty_like``; each
step's nonlinear coefficient kernel rewrites them before the E update reads them. G4 therefore
excludes them from its setup captures (``benchmark.core_path.gates.SETUP_SCRATCH_KEYS``). This
probe fills them with NaN after prepare, then checks that one step overwrites every element and
that a K=32 whole-step-graph run is torch.equal to an unpoisoned run, with no NaN anywhere.
"""
from dataclasses import replace

import pytest
import torch

from benchmark.core_path.gates import api_dialect, make_simulation, setup_scratch_attributes, tensors
from benchmark.core_path.scenes import build_core_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Core path requires CUDA.")

STEPS, CHUNK = 40, 32
# Kerr only (curl scratch) and the general kernel under modulation and Debye ADE (curl + decay).
SCENES = {"kerr_slab": 3, "w3_debye_modulation_tpa": 6}


def _prepare(name, *, poison):
    case = replace(build_core_case(name), steps=STEPS)
    prepared = make_simulation(case, steps=STEPS, full_field_dft=True).prepare()
    solver = prepared.solver
    solver._core_runtime.chunk_size = CHUNK
    scratch = {attribute: getattr(solver, attribute) for attribute in setup_scratch_attributes(solver)}
    if poison:
        for value in scratch.values():
            value.fill_(float("nan"))
    return prepared, scratch


@pytest.mark.parametrize("name,count", SCENES.items())
def test_first_step_overwrites_every_scratch_element(name, count):
    prepared, scratch = _prepare(name, poison=True)
    assert len(scratch) == count
    assert all(torch.isnan(value).all() for value in scratch.values())
    prepared.run_until(1)
    solver = prepared.solver
    assert hasattr(solver, "_core_runtime"), name
    for attribute, value in scratch.items():
        # The poisoned storage itself, still bound to the solver, holds no NaN after one step.
        assert getattr(solver, attribute) is value, attribute
        assert torch.isfinite(value).all(), (name, attribute, int(torch.isnan(value).sum()))


def _outputs(name, *, poison):
    prepared, scratch = _prepare(name, poison=poison)
    result = prepared.run()
    solver = prepared.solver
    assert solver._core_step_graph_active and solver._core_step_graph_k == CHUNK, name
    assert solver.nonlinear_enabled, name
    payload = tensors({field: getattr(solver, field) for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
    payload.update(tensors(scratch, "scratch"))
    payload.update(tensors(solver._dft_batched_fields, "dft_accumulators"))
    payload.update(tensors(api_dialect.fields(result), "phasors"))
    payload.update(tensors(api_dialect.monitors(result), "monitors"))
    return payload


@pytest.mark.parametrize("name", SCENES)
def test_poisoned_scratch_leaves_the_run_bitwise(name):
    clean = _outputs(name, poison=False)
    poisoned = _outputs(name, poison=True)
    assert clean.keys() == poisoned.keys()
    assert any(key.startswith("phasors/") for key in clean)
    assert torch.count_nonzero(clean["fields/Ez"]) or torch.count_nonzero(clean["fields/Ex"])
    for key, value in poisoned.items():
        if value.is_floating_point() or value.is_complex():
            assert not torch.isnan(value).any(), key
        assert value.dtype == clean[key].dtype and torch.equal(value, clean[key]), key

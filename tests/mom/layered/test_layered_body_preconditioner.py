"""D2 gates: body-aware preconditioner pattern for closed conductor bodies.

- The widened pattern contains the assembly's near list and adds only pairs of
  the same body (face owner); sheet layouts keep the near pattern.
- On the closed F-A model the block GMRES steps drop while the action is
  unchanged: admittances with and without it agree within the two GMRES bounds.
"""

import dataclasses
import gc
import json
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import body_preconditioner as bp
from witwin.maxwell.mom.layered.matrix_free import build_near_structure
from tests.mom.layered.test_layered_lowfreq import loop
from tests.support.rfic_fixtures import fixture_fa


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


@pytest.fixture(autouse=True)
def budget(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    record_property("seconds", seconds)
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())
    assert torch.cuda.max_memory_allocated() < 6 * 2**30


def closed(backend="hmatrix"):
    fixture = fixture_fa(substrate=False)
    layered = dataclasses.replace(fixture.config.layered, conductor_model="closed", operator_backend=backend)
    return fixture.scene, dataclasses.replace(fixture.config, layered=layered)


def test_body_pairs_extend_near_within_bodies(record_property):
    scene, config = closed()
    prepared = mw.Simulation.mom(scene, frequencies=(1e9,), config=config).prepare().solver.prepared
    structure = build_near_structure(prepared)
    pairs = bp.body_pairs(prepared, structure)
    faces = len(prepared.layout.mesh.faces)
    keys = pairs[:, 0] * faces + pairs[:, 1]
    near = structure.pairs[:, 0] * faces + structure.pairs[:, 1]
    assert bool((keys[1:] > keys[:-1]).all())
    extra = ~torch.isin(keys, near)
    assert bool(torch.isin(near, keys).all()) and int(extra.sum()) > 0
    owners = prepared.layout.face_owners
    assert bool((owners[pairs[extra, 0]] == owners[pairs[extra, 1]]).all() & (owners[pairs[extra, 0]] >= 0).all())
    record_property("pattern", json.dumps(dict(near=len(near), wide=len(keys), ratio=bp.WIDE_RATIO)))
    sheet = loop((1e9,))
    assert bp.body_pairs(sheet, build_near_structure(sheet)) is None


def test_closed_steps_drop_with_admittance_unchanged(monkeypatch, record_property):
    # Only M^-1 changes: both solves stop on the 1e-5 admittance bound of the same
    # exact system, so their admittances agree within twice that bound.
    frequencies = (1e10,)
    scene, config = closed()
    wide = mw.Simulation.mom(scene, frequencies=frequencies, config=config).run().network
    monkeypatch.setattr(bp, "body_preconditioner_structure", lambda prepared, structure: None)
    near = mw.Simulation.mom(scene, frequencies=frequencies, config=config).run().network
    difference = (torch.linalg.matrix_norm(wide.to_y() - near.to_y()) / torch.linalg.matrix_norm(near.to_y())).tolist()
    record_property("closed", json.dumps(dict(
        steps_wide=wide.metadata["block_gmres_steps"], steps_near=near.metadata["block_gmres_steps"],
        factor_nnz_wide=wide.metadata["preconditioner_factor_nnz"],
        factor_nnz_near=near.metadata["preconditioner_factor_nnz"], difference=difference,
        unknowns=wide.metadata["unknowns"])))
    assert wide.metadata["block_gmres_steps"][0] < near.metadata["block_gmres_steps"][0]
    assert max(difference) <= 2e-5

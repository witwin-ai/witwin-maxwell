"""Fail-closed guards for unsupported breakdown combinations."""

from __future__ import annotations

import types

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.breakdown_data import (
    advance_breakdown_state,
    finalize_breakdown_data,
)

from tests.breakdown_data._common import build_breakdown_scene, prepare_solver, set_uniform_ez

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD breakdown requires CUDA"
)


# --- descriptor construction guards (no CUDA needed, but grouped here) ---------


def test_unsupported_model_rejected_at_construction():
    with pytest.raises(NotImplementedError, match="field_duration"):
        mw.DielectricBreakdown(
            critical_field=1e6, post_breakdown_conductivity=1.0, model="instantaneous"
        )


def test_unsupported_state_rejected_at_construction():
    with pytest.raises(NotImplementedError, match="latching"):
        mw.DielectricBreakdown(
            critical_field=1e6, post_breakdown_conductivity=1.0, state="recovering"
        )


def test_reserved_fields_rejected_at_construction():
    with pytest.raises(NotImplementedError, match="recovery"):
        mw.DielectricBreakdown(
            critical_field=1e6, post_breakdown_conductivity=1.0, recovery=object()
        )
    with pytest.raises(NotImplementedError, match="damage_parameters"):
        mw.DielectricBreakdown(
            critical_field=1e6,
            post_breakdown_conductivity=1.0,
            damage_parameters=object(),
        )


# --- trainable reject -----------------------------------------------------------


class _TrainableBreakdownScene(mw.SceneModule):
    """A trainable density region alongside a breakdown structure."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros((1, 1, 1), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.04),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="breakdown_box",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                material=mw.Material(
                    eps_r=3.0,
                    breakdown=mw.DielectricBreakdown(
                        critical_field=1e6, post_breakdown_conductivity=5.0
                    ),
                ),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, -0.06), size=(0.12, 0.12, 0.12)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.12),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=1.0),
            )
        )
        return scene


def test_trainable_breakdown_rejected():
    model = _TrainableBreakdownScene().cuda()
    simulation = mw.Simulation.fdtd(
        model, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=8)
    )
    with pytest.raises(NotImplementedError, match="Trainable scenes cannot enable"):
        simulation.prepare()


# --- multi-GPU reject -----------------------------------------------------------


def test_multi_gpu_breakdown_rejected_by_static_guard():
    """The distributed static-capability guard rejects breakdown scenes.

    Two physical GPUs are not available in the correctness environment, so the
    guard is exercised directly on the breakdown scene rather than through a real
    distributed allocation. The guard runs at the tail of ``DistributedFDTD``
    construction and depends only on the logical scene.
    """
    from witwin.maxwell.fdtd.distributed.solver import DistributedFDTD

    scene = build_breakdown_scene(critical_field=1e6, minimum_duration=0.0)
    fake = types.SimpleNamespace(_nccl=False, logical_scene=scene)
    with pytest.raises(ValueError, match="does not yet support deterministic dielectric breakdown"):
        DistributedFDTD._validate_static_capabilities(fake)


# --- event-buffer overflow hard error ------------------------------------------


def test_event_buffer_overflow_hard_errors():
    """Shrinking the capacity below the trigger count is a hard error, not a drop."""
    solver = prepare_solver(build_breakdown_scene(critical_field=1.0, minimum_duration=0.0))
    rt = solver.breakdown_runtime
    assert rt["node_count"] > 1
    rt["capacity"] = 1  # force overflow: all capable cells will trigger

    set_uniform_ez(solver, 100.0)
    advance_breakdown_state(solver, 0)  # min_dur = 0 -> every capable cell triggers

    with pytest.raises(RuntimeError, match="event buffer overflow"):
        finalize_breakdown_data(solver)

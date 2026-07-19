"""Compiler last-writer-wins: a later non-breakdown structure that overwrites a
breakdown region must strip the breakdown capability from the cells it claims,
exactly like the material compiler. Otherwise an overwritten cell keeps a phantom
breakdown descriptor and could trigger where the winning material is inert.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.breakdown import compile_breakdown_layout
from witwin.maxwell.fdtd.runtime.breakdown import advance_breakdown_state
from witwin.maxwell.scene import prepare_scene

from tests.breakdown._common import prepare_solver, set_uniform_ez


def _layout(scene):
    """Compile the breakdown layout on the grid-materialized (prepared) scene."""
    return compile_breakdown_layout(prepare_scene(scene))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD breakdown requires CUDA"
)

_H = 0.02
_HALF = 0.3


def _base_scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-_HALF, _HALF),) * 3),
        grid=mw.GridSpec.uniform(_H),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1e6),
        device="cuda",
    )


def _breakdown_material(critical_field: float = 1.0):
    return mw.Material(
        eps_r=3.0,
        sigma_e=0.0,
        breakdown=mw.DielectricBreakdown(
            critical_field=critical_field,
            minimum_duration=0.0,
            post_breakdown_conductivity=5.0,
            default_ramp_steps=10,
        ),
    )


def _add_breakdown_box(scene, *, size, position=(0.0, 0.0, 0.0), priority=0, critical_field=1.0):
    scene.add_structure(
        mw.Structure(
            name="breakdown",
            geometry=mw.Box(position=position, size=size),
            material=_breakdown_material(critical_field),
            priority=priority,
        )
    )


def _add_plain_box(scene, *, size, position=(0.0, 0.0, 0.0), priority=1):
    scene.add_structure(
        mw.Structure(
            name="plain_override",
            geometry=mw.Box(position=position, size=size),
            material=mw.Material(eps_r=3.0, sigma_e=0.0),
            priority=priority,
        )
    )


def test_later_nonbreakdown_structure_clears_overlapped_cells():
    """A higher-priority plain box strips breakdown capability where it overlaps.

    Uses only order-independent set invariants (subset, strict shrink, cleared
    params) so the assertion never depends on exact cell/box-edge colocation.
    """
    # Reference: breakdown box alone.
    ref = _base_scene()
    _add_breakdown_box(ref, size=(0.12, 0.12, 0.12))
    ref_mask = _layout(ref).node_mask.clone()
    assert int(ref_mask.sum().item()) > 0

    # Same breakdown box, then a plain box (higher priority) over its right half.
    scene = _base_scene()
    _add_breakdown_box(scene, size=(0.12, 0.12, 0.12), priority=0)
    _add_plain_box(scene, size=(0.30, 0.30, 0.30), position=(0.15, 0.0, 0.0), priority=1)
    layout = _layout(scene)
    mask = layout.node_mask

    # Only removal: every surviving capable cell was capable in the reference.
    assert not bool((mask & ~ref_mask).any().item()), "override must not add cells"
    # The override actually removed some capability but not all (partial overlap).
    assert 0 < int(mask.sum().item()) < int(ref_mask.sum().item())

    removed = ref_mask & ~mask
    assert int(removed.sum().item()) > 0
    # Parameters are zeroed / material id reset on the cells the plain box claimed.
    assert float(layout.critical_field[removed].abs().sum().item()) == 0.0
    assert float(layout.post_conductivity[removed].abs().sum().item()) == 0.0
    assert int(layout.material_id[removed].max().item()) == -1
    # Survivors keep their descriptor values.
    survivors = mask
    assert float(layout.critical_field[survivors].min().item()) == 1.0


def test_fully_overwritten_breakdown_box_never_triggers():
    """A breakdown box fully covered by a later plain box triggers nothing."""
    scene = _base_scene()
    _add_breakdown_box(scene, size=(0.10, 0.10, 0.10), priority=0, critical_field=1.0)
    # Plain box strictly larger, higher priority -> owns every breakdown cell.
    _add_plain_box(scene, size=(0.16, 0.16, 0.16), priority=1)

    layout = _layout(scene)
    assert int(layout.node_mask.sum().item()) == 0

    solver = prepare_solver(scene)
    # Fully overwritten -> no runtime machinery at all (empty capable set).
    assert solver.breakdown_runtime is None
    assert not solver.breakdown_enabled


def test_partial_overwrite_only_survivors_trigger():
    """Drive a supra-critical field: only the surviving breakdown cells fire, and
    their count matches the (shrunk) capable mask -- the overwritten cells are
    absent from the runtime entirely, so they cannot trigger."""
    ref = _base_scene()
    _add_breakdown_box(ref, size=(0.12, 0.12, 0.12))
    ref_count = int(_layout(ref).node_mask.sum().item())

    scene = _base_scene()
    _add_breakdown_box(scene, size=(0.12, 0.12, 0.12), priority=0, critical_field=1.0)
    _add_plain_box(scene, size=(0.30, 0.30, 0.30), position=(0.15, 0.0, 0.0), priority=1)

    solver = prepare_solver(scene)
    rt = solver.breakdown_runtime
    assert rt is not None and 0 < rt["node_count"] < ref_count

    set_uniform_ez(solver, 100.0)  # far above critical=1.0
    for n in range(3):
        advance_breakdown_state(solver, n)

    triggered = int((rt["trigger_step"] >= 0).sum().item())
    # Exactly the surviving capable cells fire; no phantom from the overwritten half.
    assert triggered == rt["node_count"]

"""Staircase PEC must short a node-consistent Yee edge set.

The staircase policy shorts tangential E on an edge iff both endpoint nodes are
conductor nodes (inside or on the surface of the PEC union). The former rule
thresholded the two-node average of the smoothed occupancy, which also shorted
edges straddling a curved or off-grid surface: a zero-E edge then hung off an
exterior node whose other edges (tangential and longitudinal) stayed free. On a
round coax those one-cell slots slowed the TEM wave to ~0.93c; on a union of
node-centred boxes to ~0.83c. ``tests/validation/physics/test_pec_curved_coax_tem.py``
carries the field-level runs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.materials import _pec_signed_distance
from witwin.maxwell.fdtd.runtime.materials import _pec_edge_open_fractions, average_node_to_component
from witwin.maxwell.scene import prepare_scene

H = 0.25e-3
HALF = 3.0e-3
LZ = 1.0e-3
_ENDPOINTS = {
    "Ex": ((slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
    "Ey": ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
    "Ez": ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
}


def _node_centred_boxes(radius):
    """One h x h box centred on every node inside the disc (faces on half-cell planes)."""
    n = int(math.ceil(radius / H))
    return [
        mw.Box(position=(i * H, j * H, 0.5 * LZ), size=(H, H, 4 * LZ))
        for i in range(-n, n + 1)
        for j in range(-n, n + 1)
        if (i * H) ** 2 + (j * H) ** 2 <= radius**2
    ]


def _slab_boxes(radius):
    """Abutting one-cell-high node-aligned slabs approximating a disc."""
    boxes = []
    n = int(math.ceil(radius / H))
    for j in range(-n, n):
        yc = (j + 0.5) * H
        half = math.floor(math.sqrt(max(radius**2 - yc**2, 0.0)) / H) * H
        if half > 0:
            boxes.append(mw.Box(position=(0.0, yc, 0.5 * LZ), size=(2 * half, H, 4 * LZ)))
    return boxes


def _scene(geometries, pec_mode="staircase"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-HALF, HALF), (-HALF, HALF), (0.0, LZ))),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.pec(),
        subpixel_samples=mw.SubpixelSpec(pec=pec_mode, averaging="arithmetic", samples=1),
        device="cpu",
    )
    for index, geometry in enumerate(geometries):
        scene.add_structure(mw.Structure(geometry, mw.Material.pec(), name=f"pec{index}"))
    return prepare_scene(scene)


def _round_shield(radius=2.3e-3, vertices=96):
    frame = [(-HALF - H, -HALF - H), (HALF + H, -HALF - H), (HALF + H, HALF + H), (-HALF - H, HALF + H)]
    hole = [(radius * math.cos(2 * math.pi * k / vertices), radius * math.sin(2 * math.pi * k / vertices))
            for k in range(vertices)]
    return mw.ComplexPolySlab([frame, hole], bounds=(-LZ, 2 * LZ), axis="z")


def _open_fractions(prepared):
    return _pec_edge_open_fractions(SimpleNamespace(_compiled_material_model=prepared.compile_materials()))


def _former_average_rule(prepared, component):
    occupancy = prepared.compile_materials()["pec_occupancy"]
    return 1.0 - (average_node_to_component(None, occupancy, component) >= 0.5).to(occupancy.dtype)


CURVED_CASES = {
    "cylinder_in_round_shield": lambda: [_round_shield(), mw.Cylinder(position=(0, 0, 0.5 * LZ), radius=1.0e-3,
                                                                      height=4 * LZ, axis="z")],
    "node_centred_box_union": lambda: _node_centred_boxes(1.0e-3),
    "sphere": lambda: [mw.Sphere(position=(0.0, 0.0, 0.5 * LZ), radius=1.1e-3)],
}


@pytest.mark.parametrize("case", sorted(CURVED_CASES))
def test_staircase_shorts_exactly_the_edges_between_conductor_nodes(case):
    prepared = _scene(CURVED_CASES[case]())
    opened = _open_fractions(prepared)
    conductor = _pec_signed_distance(prepared) <= 1.0e-5 * H
    former_inconsistent = 0
    for component, (low, high) in _ENDPOINTS.items():
        both = conductor[low] & conductor[high]
        shorted = opened[component] == 0.0
        assert torch.equal(shorted, both), component
        assert bool(torch.all((opened[component] == 0.0) | (opened[component] == 1.0)))
        former_inconsistent += int(((_former_average_rule(prepared, component) == 0.0) & ~both).sum())
    # The former two-node-average rule shorted many edges with an exterior endpoint here.
    assert former_inconsistent > 20


def test_every_conductor_node_has_all_its_conductor_edges_shorted():
    """No exterior 'tooth' nodes: an endpoint of a shorted edge is a conductor node."""
    prepared = _scene(CURVED_CASES["cylinder_in_round_shield"]())
    opened = _open_fractions(prepared)
    conductor = _pec_signed_distance(prepared) <= 1.0e-5 * H
    touched = torch.zeros_like(conductor)
    for component, (low, high) in _ENDPOINTS.items():
        shorted = opened[component] == 0.0
        touched[low] |= shorted
        touched[high] |= shorted
    assert not bool((touched & ~conductor).any())


def test_abutting_box_union_shorts_like_one_box():
    """Two node-aligned boxes sharing a face behave as one conductor, bit for bit."""
    union = _scene([
        mw.Box(position=(-0.5e-3, 0.0, 0.5 * LZ), size=(1.0e-3, 1.5e-3, 4 * LZ)),
        mw.Box(position=(0.5e-3, 0.0, 0.5 * LZ), size=(1.0e-3, 1.5e-3, 4 * LZ)),
    ])
    single = _scene([mw.Box(position=(0.0, 0.0, 0.5 * LZ), size=(2.0e-3, 1.5e-3, 4 * LZ))])
    opened_union, opened_single = _open_fractions(union), _open_fractions(single)
    for component in ("Ex", "Ey", "Ez"):
        assert torch.equal(opened_union[component], opened_single[component]), component
    # The seam plane x = 0 is interior: its tangential edges are shorted.
    seam = int(round(HALF / H))
    assert bool(torch.all(opened_union["Ey"][seam, 9:14, :] == 0.0))
    assert bool(torch.all(opened_union["Ez"][seam, 9:16, :] == 0.0))


@pytest.mark.parametrize("geometries", [
    lambda: [mw.Box(position=(0.25e-3, -0.5e-3, 0.5 * LZ), size=(1.5e-3, 1.0e-3, 0.5e-3))],
    lambda: _slab_boxes(1.0e-3),
    lambda: [mw.Box(position=(0.0, 0.0, 0.5 * LZ), size=(2.0e-3, 2.0e-3, 4 * LZ)),
             mw.ComplexPolySlab([[(-HALF - H, -HALF - H), (HALF + H, -HALF - H), (HALF + H, HALF + H),
                                  (-HALF - H, HALF + H)],
                                 [(-1.75e-3, -1.75e-3), (1.75e-3, -1.75e-3), (1.75e-3, 1.75e-3),
                                  (-1.75e-3, 1.75e-3)]], bounds=(-LZ, 2 * LZ), axis="z")],
], ids=["single_box", "abutting_slab_stair", "square_coax"])
def test_grid_aligned_conductors_keep_the_former_edge_set_bitwise(geometries):
    prepared = _scene(geometries())
    opened = _open_fractions(prepared)
    for component in ("Ex", "Ey", "Ez"):
        former = _former_average_rule(prepared, component)
        assert opened[component].dtype == former.dtype
        assert torch.equal(opened[component], former), component

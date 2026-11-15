"""G4 records a witwin-core material inside a compiled model without its process-counter id.

``material_id`` comes from witwin-core's process-wide counter, so the same scene built after other
materials (or rebuilt from a scene dictionary) carries another value; the compiled surface-impedance
layout of w1_sibc holds the public material. CPU only: the layout compiles on the prepared CPU scene.
"""

import witwin.maxwell as mw

from benchmark.core_path.gates import MATERIAL_COUNTER_ATTRIBUTES, drop_material_counters, tensors
from benchmark.core_path.scenes import build_core_case
from witwin.maxwell.compiler.materials import compile_surface_impedance_layout
from witwin.maxwell.scene import prepare_scene


def _layout_snapshot(scene, counters):
    layout = compile_surface_impedance_layout(prepare_scene(scene, device="cpu"))
    return tensors({"_compiled_material_model": {"surface_impedance": layout}}, "setup", material_counters=counters)


def test_w1_sibc_snapshot_does_not_depend_on_build_order():
    first = build_core_case("w1_sibc").scene
    [mw.Material(eps_r=2.0) for _ in range(7)]  # advance the process-wide material counter
    second = build_core_case("w1_sibc").scene
    assert first.structures[0].material.material_id != second.structures[0].material.material_id
    counters_first, counters_second = [], []
    left, right = _layout_snapshot(first, counters_first), _layout_snapshot(second, counters_second)
    key = "setup/_compiled_material_model/surface_impedance/metals/0/material/material_id"
    assert counters_first == counters_second == [key]
    assert left.keys() == right.keys() and key not in left
    assert all(left[name].dtype == right[name].dtype and left[name].equal(right[name]) for name in left)
    # The capture without the rule (G1 and older G4 captures) still holds the label; compare drops it.
    full = _layout_snapshot(first, None)
    assert key in full and drop_material_counters(full, counters_first) == (left, [key])
    assert MATERIAL_COUNTER_ATTRIBUTES == ("material_id",)

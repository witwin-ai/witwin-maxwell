"""The gate scenes, datagen scenes and benchmark antennas round-trip exactly through ``witwin.scene/2``.

Exact means every fingerprint part of ``benchmark/core_path/fingerprint.py`` is equal after
``from_dict(json(to_dict(scene)))``: the prepared grid, the geometry/structure/source/monitor/circuit
declarations and the remaining scene attributes (class, every attribute, tensors by dtype and bytes),
and the document is a fixed point. CPU only. The census over the whole benchmark registry is
``<evidence-root>/api070-p8a/scripts/roundtrip_corpus.py`` (``WITWIN_EVIDENCE_ROOT``).
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import witwin.maxwell as mw

from benchmark.core_path.fingerprint import case_fingerprint, canonical
from benchmark.core_path.scenes import GATE_SCENES, build_core_case
from benchmark.datagen.scenes import scene_builders
from benchmark.scenes.antenna import half_wave_dipole_scene, patch_antenna_scene, printed_patch_scene
from benchmark.scenes.antenna.printed_dipole import printed_dipole_scene


def _restore(scene, *, sort_keys=False):
    text = json.dumps(scene.to_dict(), allow_nan=False, sort_keys=sort_keys)
    restored = mw.Scene.from_dict(json.loads(text))
    if not sort_keys:
        assert json.dumps(restored.to_dict(), allow_nan=False) == text
    return restored


@pytest.mark.parametrize("name", GATE_SCENES)
def test_gate_scene_round_trips_to_the_same_fingerprint(name):
    case = build_core_case(name)
    restored = _restore(case.scene)
    assert case_fingerprint(dataclasses.replace(case, scene=restored)) == case_fingerprint(case)
    # A store that sorts JSON object keys (EMFC records) reads back the same scene.
    assert canonical(dict(vars(_restore(case.scene, sort_keys=True)))) == canonical(dict(vars(case.scene)))


def _other_scenes():
    cases = [pytest.param(lambda b=b: b.build().scene, id=f"datagen-{b.family}-{b.name}") for b in scene_builders()]
    for factory in (half_wave_dipole_scene, patch_antenna_scene, printed_patch_scene, printed_dipole_scene):
        cases.append(pytest.param(factory, id=f"antenna-{factory.__name__}"))
    return cases


@pytest.mark.parametrize("build", _other_scenes())
def test_benchmark_scene_round_trips_exactly(build):
    scene = build()
    restored = _restore(scene)
    assert canonical(dict(vars(restored))) == canonical(dict(vars(scene)))

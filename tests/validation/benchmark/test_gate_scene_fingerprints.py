"""Gate scenes stay on the definitions their G1/G4 frozen-store payloads were captured on.

Benchmark re-specs reach gate scenes through the benchmark registry. validation/bench-references
moved four of them silently: full_tensor_slab went from 69^3 to 270^3 nodes and ran out of memory
under the 8 GiB cap. These CPU-only checks fail before such a change merges. A gate scene that
moves needs a pin in benchmark/core_path/pinned_scenes.py, or its own gate evidence and a
regenerated benchmark/core_path/gate_scene_fingerprints.json.
"""

import json

import pytest

from benchmark.core_path import scenes as gate_scenes
from benchmark.core_path.fingerprint import FINGERPRINTS, case_fingerprint
from benchmark.core_path.pinned_scenes import PINNED_SCENES

RECORDED = json.loads(FINGERPRINTS.read_text())


def test_recorded_fingerprints_cover_every_gate_scene():
    assert sorted(RECORDED) == sorted(gate_scenes.GATE_SCENES)


@pytest.mark.parametrize("name", gate_scenes.GATE_SCENES)
def test_gate_scene_matches_recorded_fingerprint(name):
    current = case_fingerprint(gate_scenes.build_core_case(name))
    moved = sorted(part for part in set(current) | set(RECORDED[name]) if current.get(part) != RECORDED[name].get(part))
    assert not moved, (
        f"gate scene {name!r} moved ({', '.join(moved)}; grid {RECORDED[name]['grid_shape']} -> "
        f"{current['grid_shape']}): pin it in benchmark/core_path/pinned_scenes.py or regenerate "
        "gate_scene_fingerprints.json with gate evidence")


@pytest.mark.parametrize("name", sorted(PINNED_SCENES))
def test_pinned_scene_does_not_read_the_benchmark_registry(name, monkeypatch):
    assert name in gate_scenes.GATE_SCENES

    def registry(*_args, **_kwargs):
        raise AssertionError(f"pinned gate scene {name!r} read the benchmark registry")

    class Registry(dict):
        def __getitem__(self, key):
            registry()

    monkeypatch.setattr(gate_scenes, "build_benchmark_scene", registry)
    monkeypatch.setattr(gate_scenes, "SCENARIOS", Registry(gate_scenes.SCENARIOS))
    assert case_fingerprint(gate_scenes.build_core_case(name)) == RECORDED[name]


def test_fingerprint_names_the_moved_part():
    import math
    from dataclasses import replace

    import witwin.maxwell as mw

    case = gate_scenes.build_core_case("dipole_vacuum")
    dipole, = case.scene.sources
    # The bench-references re-spec of the registry row: the same pulse on a sine carrier.
    carrier = mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", profile="ideal",
                             source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9, phase=math.pi / 2))
    moved = case_fingerprint(replace(case, scene=case.scene.updated(sources=(carrier,))))
    assert {part for part in moved if moved[part] != RECORDED["dipole_vacuum"][part]} == {"sources"}
    refined = case_fingerprint(replace(case, scene=case.scene.updated(grid=mw.GridSpec.uniform(0.0125))))
    assert {part for part in refined if refined[part] != RECORDED["dipole_vacuum"][part]} == {
        "geometry", "grid", "grid_shape"}
    assert case_fingerprint(case) == RECORDED["dipole_vacuum"] and dipole.source_time.phase == 0.0

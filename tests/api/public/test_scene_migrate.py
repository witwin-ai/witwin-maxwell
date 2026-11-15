"""``witwin.scene/1`` dictionaries are upgraded to ``/2`` by ``mw.io.migrate_scene_dict`` (plan phase 8b, Q20).

The fixtures in ``data/scene_v1`` were written by the last ``/1`` writer (master 3c2631231) from the builders
in ``scene_v1_fixtures.py``; ``legacy_fp64_vacuum_port.json`` predates tensor dtypes (bare-list geometry,
master 8cdc1804d). The corpus-wide check (230 gate/benchmark/datagen documents) is in the p8b evidence.
"""

import json
import logging
from pathlib import Path

import pytest
import torch
import witwin.maxwell as mw

from benchmark.core_path.fingerprint import canonical
from benchmark.core_path.scenes import build_core_case
from tests.api.public import scene_v1_fixtures as fixtures
from witwin.core.geometry.base import _rotation_quaternion

DATA = Path(__file__).with_name("data") / "scene_v1"
LOGGER = "witwin.maxwell.io.migrate"


def _v1(name):
    return json.loads((DATA / f"{name}.json").read_text())


def _messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


@pytest.mark.parametrize("name", sorted(fixtures.EXACT))
def test_lossless_v1_documents_rebuild_the_same_scene(name, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    document = _v1(name)
    migrated = mw.io.migrate_scene_dict(document)
    assert migrated["schema"] == "witwin.scene/2" and document["schema"] == "witwin.scene/1"
    restored = mw.Scene.from_dict(json.loads(json.dumps(migrated, allow_nan=False)))
    expected = fixtures.EXACT[name]()
    assert canonical(dict(vars(restored))) == canonical(dict(vars(expected)))
    # The migrated document is the /2 document of the same scene, identity numbers included.
    assert restored.to_dict() == expected.to_dict()
    messages = _messages(caplog)
    assert messages[0] == "scene dictionary: migrating witwin.scene/1 -> witwin.scene/2"
    assert any("structure identities were process-wide counter values" in message for message in messages)


def test_from_dict_reads_v1_through_the_migrator(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    restored = mw.Scene.from_dict(_v1("pcb"))
    assert canonical(dict(vars(restored))) == canonical(dict(vars(fixtures.pcb())))
    assert "scene dictionary: migrating witwin.scene/1 -> witwin.scene/2" in _messages(caplog)


def test_v1_arrays_table_is_read_unchanged():
    document = json.loads((DATA / "large_grid.arrays.json").read_text())
    arrays = torch.load(DATA / "large_grid.arrays.pt", weights_only=True)
    migrated = mw.io.migrate_scene_dict(document)
    assert migrated["grid"]["x_coords"] == {"array": "grid/x_coords", "ndarray": True}
    restored = mw.Scene.from_dict(migrated, arrays=arrays)
    assert canonical(dict(vars(restored))) == canonical(dict(vars(fixtures.large_grid())))


def test_mixed_boundaries_keep_their_face_kinds(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    for name in ("wire", "mode_abc", "periodic"):
        restored = mw.Scene.from_dict(_v1(name))
        expected = getattr(fixtures, name)()
        assert restored.boundary.face_kinds == expected.boundary.face_kinds
        assert canonical(restored.boundary) == canonical(expected.boundary)
    assert any("a mixed BoundarySpec stored kind 'none'; its authored base kind is taken as 'pml'" in message
               for message in _messages(caplog))


def test_euler_tensor_rotation_comes_back_as_its_quaternion(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    restored = mw.Scene.from_dict(_v1("euler_rotation"))
    expected = fixtures.euler_rotation()
    rebuilt, original = restored.structures[0].geometry, expected.structures[0].geometry
    assert original.rotation.shape == (3,) and rebuilt.rotation.shape == (4,)
    assert torch.equal(rebuilt.rotation, _rotation_quaternion(original.rotation))
    assert torch.equal(rebuilt.position, original.position) and torch.equal(rebuilt.size, original.size)
    assert any("structures/0/geometry/rotation: witwin.scene/1 stored rotations as quaternions" in message
               for message in _messages(caplog))


def test_pre_dtype_bare_list_document_reads_float64_values(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    restored = mw.Scene.from_dict(_v1("legacy_fp64_vacuum_port"))
    expected = build_core_case("fp64_vacuum_port").scene
    assert canonical(dict(vars(restored))) == canonical(dict(vars(expected)))
    assert any("written before tensor dtypes" in message for message in _messages(caplog))


def test_v2_documents_pass_through_and_unknown_schemas_are_refused():
    document = fixtures.pcb().to_dict()
    migrated = mw.io.migrate_scene_dict(document)
    assert migrated == document and migrated is not document
    with pytest.raises(mw.SchemaError, match="cannot migrate scene schema 'witwin.scene/0'"):
        mw.io.migrate_scene_dict({**document, "schema": "witwin.scene/0"})
    with pytest.raises(mw.SchemaError, match="unsupported scene schema 'witwin.scene/9'"):
        mw.Scene.from_dict({**document, "schema": "witwin.scene/9"})
    v1 = _v1("dielectrics")
    v1["sources"][0]["type"] = "UnknownSource"
    with pytest.raises(mw.SchemaError, match="sources/0: unknown witwin.scene/1 type 'UnknownSource'"):
        mw.io.migrate_scene_dict(v1)
    assert _v1("dielectrics")["sources"][0]["type"] == "PointDipole"  # the input is not modified

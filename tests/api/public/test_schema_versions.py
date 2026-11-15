"""Versioned payload schemas stay committed, current and versioned (plan phase 8c).

Policy (user decision 2026-11-13): an additive schema change keeps the version and bumps the revision;
a breaking change needs a major version bump and a migrator. Regenerate with
``python scripts/dev/scene_schema.py --write``.
"""

import json
from pathlib import Path
import tomllib

import pytest
import torch
import witwin.maxwell as mw

from tests.api.public import test_scene_dict as scenes
from tests.support.json_schema_lite import errors
from witwin.maxwell.io.schema import (
    MIGRATIONS, classify_schema_change, payloads, schema_file, schema_sha256, schema_text,
)

SCHEMAS = Path(mw.__file__).parent / "schemas"
VERSIONS = tomllib.loads((SCHEMAS / "VERSIONS.toml").read_text())
PAYLOADS = payloads()


def test_versions_table_lists_every_payload():
    assert list(VERSIONS) == list(PAYLOADS)


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_every_schema_change_is_recorded(name):
    version, generate = PAYLOADS[name]
    recorded = VERSIONS[name]
    assert type(recorded["revision"]) is int and recorded["revision"] >= 0
    assert recorded["version"] == version, (
        f"{name}: the code writes version {version}, VERSIONS.toml records {recorded['version']}; "
        "run python scripts/dev/scene_schema.py --write (a major bump needs a migrator in MIGRATIONS)")
    if generate is None:
        assert recorded["schema"] == "" and recorded["sha256"] == ""
        return
    schema = generate()
    if schema_sha256(schema) != recorded["sha256"]:
        kind, reasons = classify_schema_change(json.loads((SCHEMAS / recorded["schema"]).read_text()), schema)
        pytest.fail(f"{name} version {version}: unrecorded {kind} schema change ({'; '.join(reasons[:4])}); "
                    + ("run python scripts/dev/scene_schema.py --write to bump the revision" if kind == "additive"
                       else "bump the major version, add a migrator to MIGRATIONS, then run --write"))
    assert recorded["schema"] == schema_file(name, version)
    assert (SCHEMAS / recorded["schema"]).read_text() == schema_text(schema), "committed schema is stale"


def test_current_major_versions_have_their_migrators():
    assert set(MIGRATIONS["witwin.scene"]) == {"1"}
    assert mw.io.migrate_scene_dict.__module__ + ".migrate_scene_dict" == "witwin.maxwell.io.migrate.migrate_scene_dict"


_BASE = {"$defs": {"Box": {"type": "object", "properties": {"type": {"const": "Box"}, "size": {"$ref": "#/$defs/v"}},
                           "required": ["type"], "additionalProperties": False, "description": "a box"},
                   "kind": {"enum": ["a", "b"]}, "value": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/Box"}]}},
         "type": "object", "properties": {"schema": {"const": "x/2"}}, "required": ["schema"]}


def _changed(edit):
    import copy

    schema = copy.deepcopy(_BASE)
    edit(schema)
    return classify_schema_change(_BASE, schema)


@pytest.mark.parametrize("edit,kind", [
    pytest.param(lambda s: None, "same", id="identical"),
    pytest.param(lambda s: s["$defs"].update(Sphere={"type": "object"}), "additive", id="new-definition"),
    pytest.param(lambda s: s["$defs"]["Box"]["properties"].update(rotation={"$ref": "#/$defs/v"}), "additive",
                 id="new-optional-property"),
    pytest.param(lambda s: s["$defs"]["kind"]["enum"].append("c"), "additive", id="new-enum-member"),
    pytest.param(lambda s: s["$defs"]["value"]["anyOf"].append({"$ref": "#/$defs/Sphere"}), "additive",
                 id="new-alternative"),
    pytest.param(lambda s: s["$defs"]["Box"].update(description="a rotated box"), "additive", id="new-description"),
    pytest.param(lambda s: s["$defs"].pop("kind"), "breaking", id="removed-definition"),
    pytest.param(lambda s: s["$defs"]["Box"]["properties"].pop("size"), "breaking", id="removed-property"),
    pytest.param(lambda s: (s["$defs"]["Box"]["properties"].update(position={}),
                            s["$defs"]["Box"]["required"].append("position")), "breaking",
                 id="new-required-property"),
    pytest.param(lambda s: s["$defs"]["Box"]["required"].append("size"), "breaking", id="newly-required"),
    pytest.param(lambda s: s["$defs"]["kind"]["enum"].remove("b"), "breaking", id="removed-enum-member"),
    pytest.param(lambda s: s["$defs"]["value"]["anyOf"].pop(), "breaking", id="removed-alternative"),
    pytest.param(lambda s: s["properties"]["schema"].update(const="x/3"), "breaking", id="changed-const"),
    pytest.param(lambda s: s["$defs"]["Box"].update(additionalProperties=True), "breaking",
                 id="changed-keyword"),
    pytest.param(lambda s: s["$defs"]["Box"]["properties"].update(extent=s["$defs"]["Box"]["properties"].pop("size")),
                 "breaking", id="renamed-property"),
])
def test_schema_changes_are_classified(edit, kind):
    assert _changed(edit)[0] == kind


def test_versions_table_applies_the_policy(monkeypatch):
    import copy

    from witwin.maxwell.io import schema as module

    def run(recorded, generated, version="2", migrations=None):
        monkeypatch.setattr(module, "payloads", lambda: {"x": (version, lambda: generated)})
        monkeypatch.setattr(module, "MIGRATIONS", migrations or {})
        return module.versions_table(recorded, lambda file: _BASE)

    entry = {"version": "2", "revision": 3, "schema": "x-2.json", "sha256": module.schema_sha256(_BASE)}
    table, files, problems = run({"x": entry}, _BASE)
    assert table["x"]["revision"] == 3 and not problems and list(files) == ["x-2.json"]
    grown = copy.deepcopy(_BASE)
    grown["$defs"]["Sphere"] = {"type": "object"}
    table, _, problems = run({"x": entry}, grown)
    assert table["x"]["revision"] == 4 and table["x"]["version"] == "2" and not problems
    shrunk = copy.deepcopy(_BASE)
    shrunk["$defs"].pop("kind")
    _, _, problems = run({"x": entry}, shrunk)
    assert problems and "breaking schema change" in problems[0]
    _, _, problems = run({"x": entry}, shrunk, version="3")
    assert problems and "needs a migrator entry" in problems[0]
    table, files, problems = run({"x": entry}, shrunk, version="3", migrations={"x": {"2": "migrate"}})
    assert not problems and table["x"] == {"version": "3", "revision": 0, "schema": "x-3.json",
                                           "sha256": module.schema_sha256(shrunk)}


def test_scene_json_schema_is_the_committed_schema():
    assert mw.io.scene_json_schema() == json.loads((SCHEMAS / "witwin.scene-2.json").read_text())
    with pytest.raises(mw.SchemaError, match="no JSON Schema for witwin.scene/1"):
        mw.io.scene_json_schema(1)


def _documents():
    cases = [pytest.param(factory, id=factory.__name__) for factory in
             (scenes._scene_a, scenes._scene_b, scenes._scene_c, scenes._scene_d, scenes._scene_e, scenes._scene_f)]

    def shared():
        eps = torch.tensor(2.5, requires_grad=True)
        return mw.Scene(structures=[mw.Structure(mw.Box(), mw.Material(eps_r=eps)),
                                    mw.Structure(mw.Sphere(), mw.Material(eps_r=eps, sigma_e=.1))])

    def columns():
        return mw.Scene(domain=mw.Domain(((-1., 1.),) * 3), grid=mw.GridSpec.uniform(.1),
                        monitors=[mw.PointMonitor(f"p{index:03d}", (0.01 * index, 0., 0.)) for index in range(80)])

    cases += [pytest.param(shared, id="shared-tensors"), pytest.param(columns, id="records")]
    return cases


@pytest.mark.parametrize("factory", _documents())
def test_documents_validate_against_the_committed_schema(factory):
    schema = mw.io.scene_json_schema()
    scene = factory()
    for arrays in (None, {}):
        document = json.loads(json.dumps(scene.to_dict(arrays=arrays), allow_nan=False))
        assert errors(document, schema) == []


def test_schema_rejects_unknown_types_and_fields():
    schema = mw.io.scene_json_schema()
    document = json.loads(json.dumps(scenes._scene_a().to_dict()))
    document["sources"][0]["type"] = "UnknownSource"
    assert errors(document, schema)
    document = json.loads(json.dumps(scenes._scene_a().to_dict()))
    document["sources"][0]["extra"] = 1
    assert errors(document, schema)

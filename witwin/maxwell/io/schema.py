"""JSON Schemas of the versioned payloads (plan phase 8c).

``scene_schema()`` derives the JSON Schema 2020-12 of ``witwin.scene/2`` from the type registry of
:mod:`witwin.maxwell.io.scene_dict`: one ``$defs`` entry per registered type with its constructor
fields (described by their annotations), plus the value encodings. The dataclass payloads (port,
wire and network data, the resume checkpoint, the distributed result manifest) get a field schema
of their dataclass. ``payloads()`` lists every versioned payload; ``witwin/maxwell/schemas/VERSIONS.toml``
records each version, revision and the sha256 of its generated schema.

Versioning policy (user decision 2026-11-13): a purely additive schema change (new ``$defs`` entries, new
optional properties, new enum members or ``anyOf`` alternatives, changed descriptions) keeps the version
and bumps the revision; a breaking change (anything removed, renamed, newly required or changed in
meaning) needs a new major version and a migrator entry in ``MIGRATIONS``. ``classify_schema_change``
decides; ``versions_table`` applies the policy (``python scripts/dev/scene_schema.py --write``) and
``tests/api/public/test_schema_versions.py`` fails on any change VERSIONS.toml does not record.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, is_dataclass
import hashlib
from importlib import resources
from inspect import Parameter, signature
import json

JSON_SCHEMA = "https://json-schema.org/draft/2020-12/schema"
_DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64",
           "complex64", "complex128", "uint16", "uint32", "uint64")


def _ref(name):
    return {"$ref": f"#/$defs/{name}"}


def _tagged(tag, properties, required=()):
    return {"type": "object", "properties": {"type": {"const": tag}, **properties},
            "required": ["type", *required], "additionalProperties": False}


def _annotation(cls, name):
    if is_dataclass(cls):
        for field in dataclass_fields(cls):
            if field.name == name:
                return field.type if isinstance(field.type, str) else getattr(field.type, "__name__", str(field.type))
    parameter = signature(cls).parameters.get(name)
    if parameter is None or parameter.annotation is Parameter.empty:
        return None
    annotation = parameter.annotation
    return annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))


def scene_schema() -> dict:
    """The JSON Schema of ``witwin.scene/2`` generated from the registered scene types."""
    from .scene_dict import ARRAY_THRESHOLD, COLUMNAR_THRESHOLD, SCHEMA, _codecs, _scene_fields

    codecs = sorted(_codecs().values(), key=lambda codec: codec.tag)
    float_record = _tagged("Float", {"value": {"enum": ["inf", "-inf", "nan"]}}, ("value",))
    number = {"anyOf": [{"type": "number"}, _ref("Float")]}
    nested = {"anyOf": [{"type": "array", "items": {"$ref": "#/$defs/elements"}}, {"type": "number"},
                        {"type": "boolean"}, _ref("Float")]}
    defs = {
        "Float": float_record,
        "Complex": _tagged("Complex", {"real": number, "imag": number}, ("real", "imag")),
        "elements": nested,
        "Tensor": _tagged("Tensor", {
            "dtype": {"enum": list(_DTYPES)}, "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "values": {"$ref": "#/$defs/elements"}, "requires_grad": {"const": True}}, ("dtype", "shape", "values")),
        "NDArray": _tagged("NDArray", {
            "dtype": {"type": "string"}, "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "values": {"$ref": "#/$defs/elements"}}, ("dtype", "shape", "values")),
        "ArrayRef": {"type": "object", "description": f"a tensor of more than {ARRAY_THRESHOLD} elements in the "
                                                      "caller's arrays table, named by its document path",
                     "properties": {"array": {"type": "string"}, "ndarray": {"const": True},
                                    "requires_grad": {"const": True}},
                     "required": ["array"], "additionalProperties": False},
        "DType": _tagged("DType", {"value": {"enum": list(_DTYPES)}}, ("value",)),
        "List": _tagged("List", {"items": {"type": "array", "items": _ref("value")}}, ("items",)),
        "Mapping": _tagged("Mapping", {"items": {"type": "array", "items": {
            "type": "array", "prefixItems": [_ref("value"), _ref("value")], "items": False, "minItems": 2}}},
            ("items",)),
        "MaterialRef": _tagged("MaterialRef", {"key": {"type": "string", "pattern": "^m[0-9]+$"}}, ("key",)),
        "TensorRef": _tagged("TensorRef", {"key": {"type": "string", "pattern": "^t[0-9]+$"}}, ("key",)),
        "Records": {**_tagged("Records", {
            "of": {"enum": [codec.tag for codec in codecs]}, "count": {"type": "integer",
                                                                          "minimum": COLUMNAR_THRESHOLD},
            "columns": {"type": "object", "additionalProperties": _ref("column")}}, ("of", "count", "columns")),
            "description": f"a run of {COLUMNAR_THRESHOLD} or more same-type records, one column per field"},
        "column": {"anyOf": [
            {"type": "object", "properties": {"constant": _ref("value")}, "required": ["constant"],
             "additionalProperties": False},
            {"type": "object", "properties": {"prefix": {"type": "string"}, "digits": {"type": "integer"},
                                              "start": {"type": "integer"}},
             "required": ["prefix", "digits", "start"], "additionalProperties": False},
            {"type": "object", "properties": {"array": {"type": "string"}}, "required": ["array"],
             "additionalProperties": False},
            {"type": "object", "properties": {"values": {"type": "array", "items": _ref("value")}},
             "required": ["values"], "additionalProperties": False}]},
        "record": {"anyOf": [_ref(codec.tag) for codec in codecs]},
        "value": {"anyOf": [
            {"type": "null"}, {"type": "boolean"}, {"type": "number"}, {"type": "string"},
            {"type": "array", "items": _ref("value")},
            *[_ref(name) for name in ("Float", "Complex", "Tensor", "NDArray", "ArrayRef", "DType", "List",
                                      "Mapping", "MaterialRef", "TensorRef", "Records")],
            _ref("record")]},
    }
    for codec in codecs:
        properties = {}
        for name in codec.fields:
            entry = dict(_ref("value"))
            annotation = _annotation(codec.cls, name)
            if annotation:
                entry["description"] = annotation
            properties[name] = entry
        defs[codec.tag] = {**_tagged(codec.tag, properties),
                           "description": f"{codec.cls.__module__}.{codec.cls.__qualname__}"}
    scene_fields = {name: _ref("value") for name in _scene_fields()}
    return {
        "$schema": JSON_SCHEMA, "$id": "witwin.scene-2.json", "title": SCHEMA,
        "description": "WiTwin Maxwell scene dictionary (Scene.to_dict / Scene.from_dict). Lengths in meters, "
                       "frequencies in hertz.",
        "type": "object",
        "properties": {
            "schema": {"const": SCHEMA}, "package_version": {"type": ["string", "null"]},
            "unit": {"const": {"length": "m", "frequency": "Hz"}},
            "materials": {"type": "object", "propertyNames": {"pattern": "^m[0-9]+$"},
                          "additionalProperties": _ref("record")},
            "tensors": {"type": "object", "propertyNames": {"pattern": "^t[0-9]+$"},
                        "additionalProperties": {"anyOf": [_ref("Tensor"), _ref("ArrayRef")]}},
            **scene_fields,
        },
        "required": ["schema"], "additionalProperties": False, "$defs": defs,
    }


def dataclass_schema(cls, title) -> dict:
    """A field schema of a dataclass payload: its fields with their annotations, all required."""
    return {
        "$schema": JSON_SCHEMA, "title": title, "description": f"{cls.__module__}.{cls.__qualname__}",
        "type": "object",
        "properties": {field.name: {"description": field.type if isinstance(field.type, str) else str(field.type)}
                       for field in dataclass_fields(cls)},
        "required": [field.name for field in dataclass_fields(cls)],
    }


def _payloads():
    from .. import network, result, thin_wire
    from ..fdtd import resume
    from ..fdtd.distributed import persistence
    from .scene_dict import SCHEMA

    return {
        "witwin.scene": (SCHEMA.rsplit("/", 1)[1], scene_schema),
        "result-snapshot": (str(result.RESULT_SNAPSHOT_SCHEMA_VERSION), None),
        "port-data": (str(network.PortData.schema_version),
                      lambda: dataclass_schema(network.PortData, "PortData")),
        "network-data": (str(network.PERSISTENCE_SCHEMA_VERSION),
                         lambda: dataclass_schema(network.NetworkData, "NetworkData")),
        "wire-data": (str(thin_wire.WireData.schema_version),
                      lambda: dataclass_schema(thin_wire.WireData, "WireData")),
        "fdtd-resume": (str(resume._RESUME_SCHEMA_VERSION),
                        lambda: dataclass_schema(resume.FDTDResumeCheckpoint, "FDTDResumeCheckpoint")),
        "distributed-result": (str(persistence._FORMAT_VERSION),
                               lambda: dataclass_schema(persistence.DistributedResultManifest,
                                                        "DistributedResultManifest")),
        "pcb-import": (_pcb_import_version(), None),
    }


def _pcb_import_version():
    """The version of the ``pcb-import/N`` record the PCB adapter writes (a literal in its source)."""
    import inspect
    import re

    from ..adapters import pcb

    versions = set(re.findall(r'"pcb-import/(\d+)"', inspect.getsource(pcb)))
    if len(versions) != 1:
        raise RuntimeError(f"expected one pcb-import version in {pcb.__name__}, found {sorted(versions)}")
    return versions.pop()


def payloads() -> dict[str, tuple[str, object]]:
    """``name -> (version, schema generator or None)`` of every versioned payload."""
    return _payloads()


def schema_text(schema: dict) -> str:
    """The committed form of a schema: indented, key-sorted JSON with a trailing newline."""
    return json.dumps(schema, indent=1, sort_keys=True) + "\n"


def schema_sha256(schema: dict) -> str:
    return hashlib.sha256(schema_text(schema).encode()).hexdigest()


def schema_file(name: str, version: str) -> str:
    return f"{name}-{version}.json"


def scene_json_schema(version: int = 2) -> dict:
    """The committed JSON Schema (2020-12) of ``witwin.scene/<version>`` scene dictionaries."""
    if version != 2:
        from ..errors import SchemaError

        raise SchemaError(f"no JSON Schema for witwin.scene/{version}; this version publishes witwin.scene/2.")
    text = resources.files("witwin.maxwell.schemas").joinpath(schema_file("witwin.scene", "2")).read_text()
    return json.loads(text)


#: Older major versions each payload can still read, mapped to the migrator that upgrades them.
MIGRATIONS = {"witwin.scene": {"1": "witwin.maxwell.io.migrate_scene_dict"}}
_METADATA_KEYS = frozenset(("description", "title"))
_GROWING_LISTS = frozenset(("enum", "anyOf"))


def classify_schema_change(old: dict, new: dict) -> tuple[str, list[str]]:
    """``("same" | "additive" | "breaking", reasons)`` of the change from schema ``old`` to ``new``.

    Additive: new ``$defs`` entries, new properties that are not required, new ``enum`` members and new
    ``anyOf`` alternatives (a new type adds both), and changed descriptions or titles. Everything else is
    breaking: a removed definition, property, enum member or alternative, a newly required or no longer
    required property, any other added, removed or changed keyword or value.
    """
    additive, breaking = [], []

    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                where = f"{path}/{key}"
                if key not in b:
                    (additive if key in _METADATA_KEYS else breaking).append(f"{where}: removed")
                elif key not in a:
                    (additive if key in _METADATA_KEYS else breaking).append(f"{where}: added")
                elif key in _METADATA_KEYS:
                    if a[key] != b[key]:
                        additive.append(f"{where}: changed text")
                elif key in ("$defs", "properties"):
                    required = set(b.get("required", ())) if key == "properties" else set()
                    for name in sorted(set(a[key]) | set(b[key])):
                        item = f"{where}/{name}"
                        if name not in b[key]:
                            breaking.append(f"{item}: removed")
                        elif name not in a[key]:
                            (breaking if name in required else additive).append(
                                f"{item}: added{' (required)' if name in required else ''}")
                        else:
                            compare(a[key][name], b[key][name], item)
                elif key == "required":
                    old_required, new_required = set(a[key]), set(b[key])
                    newly = sorted(new_required - old_required - (set(b.get("properties", {}))
                                                                  - set(a.get("properties", {}))))
                    if newly:
                        breaking.append(f"{where}: newly required {newly}")
                    if old_required - new_required:
                        breaking.append(f"{where}: no longer required {sorted(old_required - new_required)}")
                elif key in _GROWING_LISTS and isinstance(a[key], list) and isinstance(b[key], list):
                    old_items = {json.dumps(item, sort_keys=True) for item in a[key]}
                    new_items = {json.dumps(item, sort_keys=True) for item in b[key]}
                    if old_items - new_items:
                        breaking.append(f"{where}: lost {len(old_items - new_items)} member(s)")
                    if new_items - old_items:
                        additive.append(f"{where}: gained {len(new_items - old_items)} member(s)")
                else:
                    compare(a[key], b[key], where)
        elif a != b:
            breaking.append(f"{path}: {json.dumps(a, sort_keys=True)[:60]} -> {json.dumps(b, sort_keys=True)[:60]}")

    compare(old, new, "#")
    if breaking:
        return "breaking", breaking
    return ("additive", additive) if additive else ("same", [])


def versions_table(recorded: dict, committed_schema) -> tuple[dict, dict[str, str], list[str]]:
    """Apply the versioning policy to the generated schemas.

    ``recorded`` is the parsed VERSIONS.toml, ``committed_schema(file)`` the committed schema of that
    file name (or ``None``). Returns the new table (payload -> version, revision, schema, sha256), the
    schema files to write and the problems that block an update (a breaking change without a major
    version bump, or a major bump without a migrator entry).
    """
    table, files, problems = {}, {}, []
    for name, (version, generate) in payloads().items():
        schema = None if generate is None else generate()
        file = "" if schema is None else schema_file(name, version)
        sha = "" if schema is None else schema_sha256(schema)
        previous = recorded.get(name)
        revision = 0
        if previous is not None and previous["version"] != version:
            if previous["version"] not in MIGRATIONS.get(name, {}):
                problems.append(f"{name}: version {previous['version']} -> {version} needs a migrator entry in "
                                "witwin.maxwell.io.schema.MIGRATIONS")
        elif previous is not None:
            revision = int(previous.get("revision", 0))
            if sha != previous["sha256"]:
                old = committed_schema(previous["schema"]) if previous["schema"] else None
                kind, reasons = ("breaking", ["no committed schema to compare"]) if (old is None) != (schema is None) \
                    else classify_schema_change(old or {}, schema or {})
                if kind == "breaking":
                    problems.append(f"{name} version {version}: breaking schema change ({'; '.join(reasons[:4])}); "
                                    "bump the major version and add a migrator entry")
                else:
                    revision += 1
        table[name] = {"version": version, "revision": revision, "schema": file, "sha256": sha}
        if schema is not None:
            files[file] = schema_text(schema)
    return table, files, problems

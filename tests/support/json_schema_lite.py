"""A small JSON Schema 2020-12 validator for the keywords the committed witwin schemas use.

Supported: $ref (local "#/$defs/..."), anyOf, const, enum, type, properties, required,
additionalProperties, propertyNames, pattern, items, prefixItems, minItems, minimum. Any other keyword
in a schema raises, so the committed schemas cannot silently outgrow this validator.
"""

import re

_KNOWN = {"$schema", "$id", "$defs", "$ref", "title", "description", "anyOf", "const", "enum", "type", "properties",
          "required", "additionalProperties", "propertyNames", "pattern", "items", "prefixItems", "minItems",
          "minimum"}
_TYPES = {"object": dict, "array": list, "string": str, "null": type(None), "boolean": bool}


def _is_type(value, name):
    if name == "integer":
        return type(value) is int
    if name == "number":
        return type(value) in (int, float)
    return type(value) is _TYPES[name]


def errors(instance, schema, root=None, path="$"):
    """Every validation error of ``instance`` as ``"path: message"`` (empty when valid)."""
    root = schema if root is None else root
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    unknown = set(schema) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported schema keywords {sorted(unknown)}")
    found = []
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        found += errors(instance, target, root, path)
    if "anyOf" in schema and not any(not errors(instance, option, root, path) for option in schema["anyOf"]):
        found.append(f"{path}: matches no alternative")
    if "const" in schema and instance != schema["const"]:
        found.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        found.append(f"{path}: {instance!r} not in enum")
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, name) for name in names):
            return found + [f"{path}: not of type {names}"]
    if isinstance(instance, dict):
        for name in schema.get("required", ()):
            if name not in instance:
                found.append(f"{path}: missing {name!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if "propertyNames" in schema:
                found += errors(key, schema["propertyNames"], root, f"{path}.{key}<name>")
            if key in properties:
                found += errors(value, properties[key], root, f"{path}.{key}")
            elif "additionalProperties" in schema:
                found += errors(value, schema["additionalProperties"], root, f"{path}.{key}")
    if isinstance(instance, list):
        prefix = schema.get("prefixItems", [])
        if len(instance) < schema.get("minItems", 0):
            found.append(f"{path}: fewer than {schema['minItems']} items")
        for index, item in enumerate(instance):
            if index < len(prefix):
                found += errors(item, prefix[index], root, f"{path}[{index}]")
            elif "items" in schema:
                found += errors(item, schema["items"], root, f"{path}[{index}]")
    if isinstance(instance, str) and "pattern" in schema and not re.search(schema["pattern"], instance):
        found.append(f"{path}: {instance!r} does not match {schema['pattern']}")
    if type(instance) in (int, float) and "minimum" in schema and instance < schema["minimum"]:
        found.append(f"{path}: below {schema['minimum']}")
    return found

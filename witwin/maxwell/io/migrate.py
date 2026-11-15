"""Upgrade older scene dictionaries to ``witwin.scene/2``.

A migrator reads data; it keeps no old API alive (plan Q20). Every upgrade step and every value the old
document did not store (and the migrator therefore assumed) is logged at INFO on the
``witwin.maxwell.io.migrate`` logger, with the document path it concerns.

``witwin.scene/1`` (0.6, written by ``Scene.to_dict`` until 0.7.0 phase 8) differs from ``/2`` in:

- tags: core geometries were tagged by ``kind`` (``"box"``), maxwell sheets by ``kind`` (``"poly_slab"``),
  complex numbers by ``"complex"``; tensors had no ``shape``; mappings and metadata were plain JSON
  objects and lists;
- materials: ``Structure.material``/``ThinWire.conductor`` held a bare material key, the background was
  ``{"material": key}``; material records carried an extra ``kind`` and ``LossyMetalMedium`` /
  ``CopperFoil`` an extra ``roughness: "none"`` (the ``/1`` writer refused a rough medium, so ``"none"``
  always means smooth);
- ``BoundarySpec`` faces were written resolved (all six) and a mixed spec's kind as ``"none"``: a face
  equal to what its axis override or the base kind implies becomes an absent override again; the base
  of a mixed spec is the most common kind of the faces no axis override covers (a tie prefers ``"pml"``,
  then ``"none"``), which resolves to the same face kinds whatever the authored base was;
- geometry rotations were written as quaternions: an Euler-angle tensor rotation comes back as its
  quaternion (the same rotation, a different stored form);
- structure identities were process-wide counter values: they are numbered per scene, and a structure
  whose ``material_id`` is shared exactly by the structures of its material inherits it;
- ``GridSpec`` custom coordinates were plain lists (NumPy float64 arrays in the scene);
- documents written before tensor dtypes were kept stored geometry fields as bare lists: a value float32
  cannot represent exactly is read as float64, any other one as before (float32);
- the Scene collections ``/1`` refused are absent and become empty.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math

import numpy as np

from ..errors import SchemaError
from .scene_dict import SCHEMA

logger = logging.getLogger(__name__)

SCHEMA_V1 = "witwin.scene/1"

_V1_GEOMETRY_KINDS = {
    "box": "Box", "sphere": "Sphere", "cylinder": "Cylinder", "ellipsoid": "Ellipsoid", "cone": "Cone",
    "pyramid": "Pyramid", "prism": "Prism", "extruded_polygon": "ExtrudedPolygon", "torus": "Torus",
    "hollow_box": "HollowBox", "mesh": "Mesh", "union": "Union", "intersection": "Intersection",
    "difference": "Difference", "poly_slab": "PolySlab", "planar_pec": "PlanarPEC", "surface_mesh": "SurfaceMesh",
}
_V1_TYPES = frozenset((
    "Domain", "GridSpec", "BoundarySpec", "MurABC", "ModeABC", "MeshOverrideStructure", "LayerRefinementSpec",
    "GridRefinement", "CornerFinderSpec", "Material", "CopperFoil", "LossyMetalMedium", "DebyePole", "DrudePole",
    "LorentzPole", "Structure", "ThinWire", "WireConductor", "WireEnd", "PointDipole", "PlaneWave", "CW",
    "GaussianPulse", "RickerWavelet", "TFSF", "LumpedPort", "TerminalPort", "TerminalRef", "AxisPath",
    "WireNodeRef", "WirePortBinding", "Resistor", "Capacitor", "Inductor", "SeriesRLC", "ParallelRLC",
    "PointMonitor", "PlaneMonitor", "FinitePlaneMonitor", "FluxMonitor", "ClosedSurfaceMonitor",
    "AngleProjectionMonitor", "CartesianProjectionMonitor", "KSpaceProjectionMonitor", "PermittivityMonitor",
    "SubpixelSpec",
))
_V1_SCENE_FIELDS = ("domain", "grid", "boundary", "structures", "thin_wires", "sources", "ports",
                    "lumped_elements", "monitors", "subpixel")
_V2_SCENE_FIELDS = ("domain", "grid", "boundary", "background", "structures", "thin_wires", "sources", "monitors",
                    "ports", "networks", "lumped_elements", "circuits", "material_regions",
                    "electrostatic_terminals", "charge_densities", "internal_absorbers", "subpixel", "symmetry",
                    "coordinates", "metadata")
_IDENTITIES = ("structure_id", "assignment_id", "surface_id")
_FACES = tuple(f"{axis}_{side}" for axis in "xyz" for side in ("low", "high"))
_IDENTITY_QUATERNION = [1.0, 0.0, 0.0, 0.0]


def migrate_scene_dict(d: Mapping) -> dict:
    """Return ``d`` as a ``witwin.scene/2`` dictionary (``d`` itself is not modified).

    A ``/2`` document is returned as a shallow copy. A ``witwin.scene/1`` document is upgraded; every
    step and every assumed value is logged at INFO on this module's logger.
    Arrays tables are unchanged: ``/1`` array references keep their names.
    """
    if not isinstance(d, Mapping):
        raise SchemaError(f"a scene document is a mapping, not {type(d).__name__}.")
    schema = d.get("schema")
    if schema == SCHEMA:
        return dict(d)
    if schema != SCHEMA_V1:
        raise SchemaError(f"cannot migrate scene schema {schema!r}; known: {SCHEMA_V1!r} -> {SCHEMA!r}.")
    logger.info("scene dictionary: migrating %s -> %s", SCHEMA_V1, SCHEMA)
    return _V1(d).document()


def _note(path, message):
    logger.info("%s: %s", path, message)


def _float(value):
    if math.isfinite(value):
        return value
    return {"type": "Float", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}


def _shape(values):
    shape = []
    while isinstance(values, list):
        shape.append(len(values))
        if not values:
            break
        values = values[0]
    return shape


def _numeric(value):
    if isinstance(value, list):
        return all(_numeric(item) for item in value)
    return type(value) in (int, float)


def _ragged(value):
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        return False
    return len({len(item) if isinstance(item, list) else -1 for item in value}) > 1 or any(map(_ragged, value))


def _flat(values):
    if isinstance(values, list):
        for item in values:
            yield from _flat(item)
    else:
        yield values


def _float32_exact(value):
    return all(float(np.float32(item)) == float(item) for item in _flat(value))


class _V1:
    def __init__(self, document):
        self.source = document
        self.material_keys = set(document.get("materials", {}))
        self.identities = {}
        self.inherited_material_ids = self._inherited_material_ids(document)

    # -- document ---------------------------------------------------------------------------------
    def document(self):
        source = self.source
        unknown = sorted(set(source) - {"schema", "unit", "background", "materials", "metadata", *_V1_SCENE_FIELDS})
        if unknown:
            raise SchemaError(f"unknown {SCHEMA_V1} document key(s) {unknown}.")
        if source.get("unit", {"length": "m", "frequency": "Hz"}) != {"length": "m", "frequency": "Hz"}:
            raise SchemaError(f"unsupported units {source['unit']!r}.")
        materials = {key: self.record(record, f"materials/{key}")
                     for key, record in source.get("materials", {}).items()}
        background = source.get("background", {"material": None})
        result = {"schema": SCHEMA, "package_version": None, "unit": {"length": "m", "frequency": "Hz"},
                  "materials": materials}
        _note("package_version", "not stored by witwin.scene/1; written as null")
        for name in _V2_SCENE_FIELDS:
            if name == "background":
                result[name] = None if background.get("material") is None else self.material_ref(
                    background["material"], "background")
            elif name == "metadata":
                result[name] = self.metadata(source.get("metadata"), "metadata")
            elif name in _V1_SCENE_FIELDS:
                result[name] = self.value(source[name], name) if name in source else self.missing(name)
            elif name == "symmetry":
                result[name] = [None, None, None]
            elif name == "coordinates":
                result[name] = None
            else:
                result[name] = []
        _note("<document>", "networks, circuits, material_regions, electrostatic_terminals, charge_densities, "
                            "internal_absorbers, symmetry and coordinates were refused by witwin.scene/1: "
                            "written empty")
        return result

    def missing(self, name):
        raise SchemaError(f"{name}: missing from the {SCHEMA_V1} document.")

    # -- values -----------------------------------------------------------------------------------
    def value(self, value, path):
        if isinstance(value, float):
            return _float(value)
        if isinstance(value, list):
            return [self.value(item, f"{path}/{index}") for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        if isinstance(value.get("array"), str):
            return dict(value)
        tag = value.get("type")
        if tag is None:
            return {"type": "Mapping", "items": [[key, self.value(item, f"{path}/{key}")]
                                                 for key, item in value.items()]}
        if tag == "Tensor":
            return self.tensor(value, path)
        if tag == "complex":
            return {"type": "Complex", "real": _float(value["real"]), "imag": _float(value["imag"])}
        if tag in _V1_GEOMETRY_KINDS or tag in _V1_TYPES:
            return self.record(value, path)
        raise SchemaError(f"{path}: unknown {SCHEMA_V1} type {tag!r}.")

    def tensor(self, record, path):
        values = record["values"]
        complex_ = record["dtype"].startswith("complex")

        def element(item):
            if isinstance(item, list):
                return [element(entry) for entry in item]
            if isinstance(item, dict):  # a complex element
                return [_float(item["real"]), _float(item["imag"])]
            if complex_:
                return [_float(float(item)), 0.0]
            return _float(item) if isinstance(item, float) else item

        converted = element(values)
        shape = _shape(converted)
        if complex_:
            shape = shape[:-1]
        return {"type": "Tensor", "dtype": record["dtype"], "shape": shape, "values": converted}

    def legacy_tensor(self, value, path):
        """A bare numeric geometry value of a pre-dtype document: float64 when float32 cannot hold it."""
        if _ragged(value):
            return [self.legacy_tensor(item, f"{path}/{index}") for index, item in enumerate(value)]
        if isinstance(value, bool) or not isinstance(value, (int, float, list)) or not _numeric(value):
            return self.value(value, path)
        if _float32_exact(value):
            return self.value(value, path)
        _note(path, "bare list of a document written before tensor dtypes: float32 cannot hold it, read as float64")
        return {"type": "Tensor", "dtype": "float64", "shape": _shape(value), "values": value}

    def metadata(self, value, path):
        """Plain JSON metadata: objects become mappings, arrays lists (as the /1 reader passed them)."""
        if isinstance(value, dict):
            return {"type": "Mapping", "items": [[key, self.metadata(item, f"{path}/{key}")]
                                                 for key, item in value.items()]}
        if isinstance(value, list):
            _note(path, "metadata arrays were plain JSON lists in witwin.scene/1; read as lists")
            return {"type": "List", "items": [self.metadata(item, f"{path}/{index}")
                                              for index, item in enumerate(value)]}
        if isinstance(value, float):
            return _float(value)
        return value

    def material_ref(self, key, path):
        if key not in self.material_keys:
            raise SchemaError(f"{path}: unknown material key {key!r}.")
        return {"type": "MaterialRef", "key": key}

    def identity(self, kind, value):
        numbers = self.identities.setdefault(kind, {})
        return numbers.setdefault(int(value), len(numbers))

    @staticmethod
    def _inherited_material_ids(document):
        """Material keys whose structures all carry one material_id that no other material's structures use."""
        by_key = {}
        for structure in document.get("structures", ()):
            if isinstance(structure, dict) and structure.get("material_id") is not None:
                by_key.setdefault(structure.get("material"), set()).add(structure["material_id"])
        owners = {}
        for key, ids in by_key.items():
            for value in ids:
                owners.setdefault(value, set()).add(key)
        return {key for key, ids in by_key.items() if len(ids) == 1 and len(owners[next(iter(ids))]) == 1}

    # -- records ----------------------------------------------------------------------------------
    def record(self, record, path):
        tag = record["type"]
        geometry = tag in _V1_GEOMETRY_KINDS
        name = _V1_GEOMETRY_KINDS.get(tag, tag)
        legacy = geometry and not any(isinstance(item, dict) and item.get("type") == "Tensor"
                                      for item in record.values())
        result = {"type": name}
        for key, item in record.items():
            if key == "type":
                continue
            field_path = f"{path}/{key}"
            if key == "kind" and name in ("Material", "CopperFoil", "LossyMetalMedium", "GridSpec", "TFSF"):
                continue  # derived labels the /1 writer added beside the constructor fields
            if key == "roughness" and name == "CopperFoil":
                continue
            if key == "roughness" and name == "LossyMetalMedium":
                if item != "none":
                    raise SchemaError(f"{field_path}: unexpected {SCHEMA_V1} roughness {item!r}.")
                # The /1 writer refused rough media, so its "none" always stands for a smooth medium.
                result[key] = None
            elif key == "metadata":
                result[key] = self.metadata(item, field_path)
            elif (name, key) in (("Structure", "material"), ("ThinWire", "conductor")):
                result[key] = self.material_ref(item, field_path)
            elif name == "Structure" and key in _IDENTITIES and item is not None:
                result[key] = self.identity(key, item)
            elif name == "Structure" and key == "material_id" and item is not None:
                if record.get("material") in self.inherited_material_ids:
                    result[key] = None
                else:
                    result[key] = self.identity("material_id", item)
            elif name == "GridSpec" and key.endswith("_coords") and item is not None:
                result[key] = self.coordinates(item, field_path)
            elif name == "ThinWire" and key == "endpoints" and item == []:
                result[key] = None
            elif geometry and legacy:
                result[key] = self.legacy_tensor(item, field_path)
            else:
                result[key] = self.value(item, field_path)
            if geometry and key == "rotation" and self._rotation_values(item) not in (None, _IDENTITY_QUATERNION):
                _note(field_path, "witwin.scene/1 stored rotations as quaternions; an Euler-tensor rotation "
                                  "comes back as its quaternion")
        if name == "Structure":
            _note(path, "structure identities were process-wide counter values; numbered per scene")
        if name == "BoundarySpec":
            self.boundary_faces(result, path)
        if name == "Mesh":
            self.mesh_host_arrays(result, path)
        return result

    @staticmethod
    def _rotation_values(item):
        if isinstance(item, dict) and item.get("type") == "Tensor":
            return item["values"]
        return item if isinstance(item, list) else None

    def coordinates(self, item, path):
        if isinstance(item, dict) and isinstance(item.get("array"), str):
            return {"array": item["array"], "ndarray": True}
        values = [_float(float(value)) for value in item]
        return {"type": "NDArray", "dtype": "float64", "shape": [len(values)], "values": values}

    def boundary_faces(self, result, path):
        """Resolved /1 faces back to overrides: a face equal to what its axis override or the base implies is absent."""
        base = result.get("kind", "none")
        mixed = base == "none" and any(result.get(face) not in (None, "none") for face in _FACES)
        if mixed:
            base = self.mixed_base(result)
            result["kind"] = base
        for face in _FACES:
            axis, side = face.split("_")
            override = result.get(axis)
            implied = base if override is None else override[0 if side == "low" else 1]
            if result.get(face) == implied:
                result[face] = None
        if mixed:
            _note(path, f"a mixed BoundarySpec stored kind 'none'; its authored base kind is taken as {base!r} "
                        "and the other faces as overrides (the same face kinds either way)")

    @staticmethod
    def mixed_base(result):
        """The most common kind of the faces no axis override covers; a tie prefers 'pml', then 'none'."""
        counts = {}
        for face in _FACES:
            if result.get(face.split("_")[0]) is None and result.get(face) is not None:
                counts[result[face]] = counts.get(result[face], 0) + 1
        if not counts:
            return "none"
        best = max(counts.values())
        tied = [kind for kind, count in counts.items() if count == best]
        return tied[0] if len(tied) == 1 else "pml" if "pml" in tied else "none"

    def mesh_host_arrays(self, result, path):
        """The /1 reader rebuilt float32 meshes with topology diagnostics from host arrays; keep that path."""
        vertices = result.get("vertices")
        dtype = vertices.get("dtype") if isinstance(vertices, dict) else "float32"
        if not result.get("topology_diagnostics", True) or dtype != "float32":
            return
        for key in ("vertices", "faces"):
            item = result[key]
            if isinstance(item, dict) and isinstance(item.get("array"), str):
                result[key] = {"array": item["array"], "ndarray": True}
            elif isinstance(item, dict) and item.get("type") == "Tensor":
                result[key] = {**item, "type": "NDArray"}
            else:
                values = item
                result[key] = {"type": "NDArray", "dtype": "float32" if key == "vertices" else "int64",
                               "shape": _shape(values), "values": values}
        _note(path, "float32 mesh with topology diagnostics: rebuilt from host arrays, as the witwin.scene/1 reader did")

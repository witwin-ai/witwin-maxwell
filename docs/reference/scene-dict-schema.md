# Scene dictionaries: `witwin.scene/2`

`Scene.to_dict(arrays=None)` writes a scene as a strict-JSON document; `Scene.from_dict(d, arrays=None)` rebuilds
it through the public constructors. Implementation: `witwin/maxwell/io/scene_dict.py` (the type registry
`_codecs()` lists every serialized type with its fields). Superseded design: `docs/archive/plans/scene_dict.md`
(`witwin.scene/1`).

## Contract

- **Exact.** `from_dict(to_dict(scene))` rebuilds the same object graph: the same classes, attributes and tensor
  bytes and dtypes (the structural fingerprint of `benchmark/core_path/fingerprint.py`), so the prepared grid and
  every compiled table are equal. The rebuilt document equals the original one.
- **Complete or refused.** Every `Scene` field is written. A value outside the schema raises `mw.SchemaError`
  naming its document path (`sources/0/source_time/...: callable <lambda> is not serializable ...`). Nothing is
  dropped.
- **Strict JSON.** String keys, lists, strings, booleans, integers, finite floats and `null` only;
  `json.dumps(doc, allow_nan=False)` always works. Key order carries no meaning: mappings are item lists, so a
  store that sorts object keys reads back the same scene.
- **Deterministic.** The same scene always gives the same document and the same array names (document paths),
  in a fresh process and after building unrelated scenes. Structure identity integers (`structure_id`,
  `assignment_id`, `surface_id`, `material_id`) come from process-wide counters; a document numbers each kind
  per scene in document order (equal identities stay equal), and a structure that inherits its material's
  `material_id` writes `null`, so the rebuilt structure inherits it again.
- **Device-free.** Tensors come back on the CPU; the solver config places them.

## Top level

```json
{"schema": "witwin.scene/2", "package_version": "0.6.0", "unit": {"length": "m", "frequency": "Hz"},
 "materials": {"m0": {"type": "Material", ...}}, "tensors": {"t0": {"type": "Tensor", ...}},
 "domain": {...}, "grid": {...}, "boundary": {...}, "background": null, "structures": [...],
 "thin_wires": [...], "sources": [...], "monitors": [...], "ports": [...], "networks": [...],
 "lumped_elements": [...], "circuits": [...], "material_regions": [...], "electrostatic_terminals": [...],
 "charge_densities": [...], "internal_absorbers": [...], "subpixel": {...}, "symmetry": [null, null, null],
 "coordinates": null, "metadata": {"type": "Mapping", "items": []}}
```

Every `Scene` dataclass field is a key; `tensors` is present only when a tensor is shared. Unknown keys are
refused. The JSON Schema (2020-12) of this document is committed as `witwin/maxwell/schemas/witwin.scene-2.json`
and returned by `mw.io.scene_json_schema()`.

## Values

| Python value | Document |
|---|---|
| `None`, `bool`, `int`, `str`, finite `float` | the JSON value |
| non-finite `float` | `{"type": "Float", "value": "inf" \| "-inf" \| "nan"}` |
| `complex` | `{"type": "Complex", "real": ..., "imag": ...}` |
| `tuple` | JSON list (read back as a tuple) |
| `list` | `{"type": "List", "items": [...]}` |
| mapping (any key type, metadata included) | `{"type": "Mapping", "items": [[key, value], ...]}` |
| `Enum` member | its value; the constructor parses it back |
| `torch.dtype` | `{"type": "DType", "value": "float64"}` |
| `torch.Tensor` | `{"type": "Tensor", "dtype": "float32", "shape": [3], "values": [...]}` (complex values as `[re, im]` pairs, non-finite elements as `Float` records); with an arrays table and more than 4096 elements `{"array": "<document path>"}` and the tensor bit-exact in the table |
| trainable tensor | the same plus `"requires_grad": true`; read back as a fresh leaf |
| a tensor object reached more than once (by identity) | written once in `tensors` and referenced by `{"type": "TensorRef", "key": "t<k>"}`; reads back as one shared tensor (one shared leaf when trainable), so an optimizer sees one parameter and gradients from every use accumulate in it |
| `numpy.ndarray` (numeric) | `{"type": "NDArray", "dtype", "shape", "values"}`, or `{"array": "<path>", "ndarray": true}` |
| material (`witwin.core.PhysicalMaterial` subclass) or `WireConductor` | `{"type": "MaterialRef", "key": "m<k>"}`; the record is stored once in `materials`, so shared objects stay shared |
| registered scene object | `{"type": "<ClassName>", "<constructor field>": <value>, ...}` |

A run of 64 or more consecutive same-type records in a Scene collection (for example 16,384 point monitors) is
one block `{"type": "Records", "of": "<ClassName>", "count": N, "columns": {<field>: <column>}}`, a column
being `{"constant": value}`, `{"prefix": "point_", "digits": 5, "start": 0}` (consecutive zero-padded names),
`{"array": "<collection>/<index>/columns/<field>"}` (equal-length float lists as an (N, k) float64 table array,
when an arrays table is given) or `{"values": [...]}`; it reads back as the N records.

The key `array` with a string value appears only in array references. A record's keys are `type` and its
constructor fields; missing fields take the constructor default, unknown ones are refused.

## Objects

A record holds the constructor parameters of its class (`inspect.signature`, `device` excluded), read from the
attributes of the same name, and is rebuilt by calling the class with them. Where the stored state differs from
the constructor parameters, the registry names the reader or builder:

- geometry `rotation` is written as the quaternion; a hard boolean writes its smoothness as a number; a host
  float32 `Mesh` is rebuilt from host arrays; `ComplexPolySlab` writes its `loops`;
- `BoundarySpec.kind` is the authored base kind (a mixed spec takes it from a face no override covers);
- `TFSF` writes `mode`, `bounds`, `axis`, `axis_bounds` and rebuilds a slab through `TFSF.slab`;
- `TerminalPort.current_surface` is the declared surface; `ThinWire.endpoints` is `null` for a closed loop;
  `BreakdownMonitor.region` is resolved into `position`/`size`; ESD sources write `port` from `port_name`;
  `ESDWaveform` rebuilds through `ESDWaveform.iec_61000_4_2`; `MeasuredWaveform` writes its user provenance;
- `PerturbationMedium` writes the base material it copied (refused when pole sensitivities lowered the base poles);
- `RationalSurfaceImpedance` writes its `RationalModel` and the certificate settings that reproduce its passivity
  report; `TouchstoneNetwork` writes its `NetworkData` inline;
- `Circuit` writes `name`, `parameters`, `config`, `metadata`, `nodes` (in creation order), `generated_nodes`,
  `devices`, `bindings` (`[port, positive node, negative node]`) and `initial_conditions`
  (`[node, value, constraint]`) and is rebuilt by replaying the builder calls.

A constructor that renormalizes an already unit vector (a plane-wave direction) can move it by an ulp; the reader
restores a recorded float vector that the rebuilt object differs from by at most two ulps per component.

## Not serializable

- callables (a `CustomSourceTime` built from a function, ...);
- instances of unregistered classes, including user subclasses of registered ones;
- `SceneModule` (serialize `module.to_scene()`);
- a `PerturbationMedium` with nonzero pole sensitivities;
- NumPy object arrays and other values outside the table above.

## Versions

`witwin.scene/1` (0.6) documents are read through `mw.io.migrate_scene_dict` (`witwin/maxwell/io/migrate.py`),
which `Scene.from_dict` calls. The migrator logs every step and every assumed value at INFO on the
`witwin.maxwell.io.migrate` logger. What `/1` did not store:

- structure identities were process-wide counters: they are numbered per scene, and a structure whose
  `material_id` its material's structures share exactly inherits it again;
- `BoundarySpec` faces were written resolved with a mixed kind as `"none"`: a face equal to what its axis
  override or the base implies becomes an absent override, and a mixed spec's base is the most common kind of
  the faces no axis override covers (a tie prefers `"pml"`, then `"none"`); the face kinds are always the same;
- rotations were written as quaternions: an Euler-angle tensor rotation comes back as its quaternion;
- metadata was plain JSON: arrays come back as lists;
- documents written before tensor dtypes held geometry as bare lists: a value float32 cannot represent is read
  as float64, any other value as float32 (as the `/1` reader did);
- `/1` refused rough `LossyMetalMedium`s, so its `"roughness": "none"` always means smooth.

`witwin/maxwell/schemas/VERSIONS.toml` lists every versioned payload (scene, result snapshot, port, network and
wire data, the FDTD resume checkpoint, the distributed result manifest, `pcb-import`) with its version, revision
and the sha256 of its generated schema. Policy (user decision 2026-11-13):

- **Additive** changes keep the version (`witwin.scene/2` stays `/2`) and bump the revision: new `$defs` entries
  (a new scene type), new optional properties (a new constructor field with a default), new `enum` members and
  new `anyOf` alternatives (a new type is a new alternative of `record`), and changed descriptions or titles.
  Documents written before the change still read; a reader older than the revision may refuse the new parts.
- **Breaking** changes need a new major version (`witwin.scene/3`) and a migrator from the previous one, listed in
  `witwin.maxwell.io.schema.MIGRATIONS`: a removed or renamed definition or property, a removed enum member or
  alternative, a property that becomes or stops being required, and any other changed keyword or value.

`witwin.maxwell.io.schema.classify_schema_change(old, new)` decides. `python scripts/dev/scene_schema.py --write`
bumps the revision of an additive change, and refuses a breaking change until the version constant is bumped and
a migrator entry exists (the revision then restarts at 0). `tests/api/public/test_schema_versions.py` fails on
any schema change VERSIONS.toml does not record and names its class.

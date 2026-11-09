# `Scene.to_dict` / `Scene.from_dict` (`witwin.scene/1`)

Approved design (user, 2026-10-30): `<workspace>/instamax/docs/plans/mw_scene_tokenization.md`
§2, with the rules in §2 ("defaults written explicitly", "geometry and material independent",
"RIS states are scenes") and §7 item 2. Geometry dicts come from witwin-core
`witwin.core.geometry_to_dict / geometry_from_dict` (branch `scene-dict/geometry`, worktree
`<workspace>/wt/core-scene-dict`; use it via PYTHONPATH until it merges).

## Files

- New: `witwin/maxwell/scene_dict.py` (all serializers).
- `witwin/maxwell/scene.py`: add only two thin methods, `Scene.to_dict(arrays=None)` and the
  classmethod `Scene.from_dict(d, arrays=None, device="cuda")`, delegating to the module.
- New: `tests/api/public/test_scene_dict.py` (the acceptance tests below).
- `FEATURE_LIST.md`: one entry.
- Nothing else. The PCB line edits ports/media/sources heavily, so do not edit those files.

## Format

The top level is the §2 layout: schema "witwin.scene/1", unit, domain, grid, boundary, background,
materials (a table keyed by generated ids; identical material objects share one id), structures
(geometry dict, material id, name, priority, enabled, plus every other Structure constructor field
that is set), thin_wires, sources, ports, lumped_elements, monitors, subpixel, metadata.

- Every serialized class is serialized from its constructor parameters (`inspect.signature` /
  `dataclasses.fields`), reading the stored attributes, with defaults written explicitly. There are
  no hand-maintained key lists; nested objects (source_time, termination, TFSF, CopperFoil,
  poles, BoundarySpec overrides, GridSpec layer_refinement, SubpixelSpec) are handled the same way,
  recursively, with a `"type"` tag.
- Classes in scope:
  - Domain;
  - GridSpec: uniform, custom (axis coordinates; large arrays by reference) and auto (with
    layer_refinement and override_structures);
  - BoundarySpec, including per-face overrides, PML layers/strength and the Bloch wavevector;
  - background;
  - Material: every constructor field, including the Debye/Drude/Lorentz pole families, mu
    poles, sigma_m, and pec/pmc via their factory flags;
  - CopperFoil, including `edge_model`; roughness written explicitly as "none" when absent;
  - LossyMetalMedium, if `Structure` accepts it;
  - Structure;
  - mw geometries PolySlab, PlanarPEC, SurfaceMesh (in this repo, same signature-driven rule;
    `SurfaceMesh.sheet_model` will arrive with pcb/3d-thin-sheet and must then be picked up
    automatically);
  - ThinWire;
  - PointDipole;
  - PlaneWave, including TFSF injection;
  - source times CW, GaussianPulse, RickerWavelet;
  - LumpedPort, TerminalPort, and their terminations;
  - Resistor, Capacitor, Inductor, SeriesRLC, ParallelRLC;
  - monitors: PointMonitor, PlaneMonitor, FinitePlaneMonitor, FluxMonitor, ClosedSurfaceMonitor,
    the three projection monitors, PermittivityMonitor;
  - SubpixelSpec.
- Anything else present in the Scene raises `TypeError("not in witwin.scene/1: <what>")`. That
  includes non-empty networks, circuits, material_regions, electrostatic terminals, charge
  densities, internal_absorbers, ModePort/WavePort/ModeSource/GaussianBeam, CustomSourceTime,
  anisotropic/gyromagnetic media, a non-None symmetry or coordinates, trainable (requires_grad)
  tensors, and metadata that is not JSON-safe.
- JSON-safe: `json.dumps(scene.to_dict())` works. With `arrays` given, tensors over 4096
  elements go there by reference, the same convention as the core module.

## Acceptance tests (tests/api/public/test_scene_dict.py, nothing else)

1. `test_round_trip_compiles_identically`, parametrized over four scenes:
   - (a) the dielectric sphere + rotated lossy box + ideal dipole scene of
     instamax/tests/test_neural_simulation.py;
   - (b) a metahz-like PCB: two PolySlab/Box substrates; finite copper as zero-thickness Box and
     PolySlab traces with CopperFoil (the combination the compiler accepts today); a PlanarPEC
     sheet with a hole and a SurfaceMesh sheet, both with Material.pec() (the compiler rejects
     finite-copper SurfaceMesh/PlanarPEC until the PCB thin-sheet work lands); a LumpedPort with a
     SeriesRLC termination; a Resistor; a nonuniform custom grid; PML;
   - (c) a periodic cell: Bloch x/y, PML z, PlaneWave with TFSF at oblique incidence, a
     ParallelRLC load, a dispersive Lorentz/Drude material;
   - (d) a ThinWire dipole with PEC/PMC faces, a GaussianPulse source, a boolean geometry and
     every monitor type in scope.

   For each: `s2 = Scene.from_dict(json.loads(json.dumps(s.to_dict())))`, then
   - `s2.to_dict() == s.to_dict()`;
   - the compiled material/coefficient tensors are bit-identical (`torch.equal`). Use
     `compile_material_tensors` where the scene compiles there; otherwise use the tensors of
     the backend that accepts the scene (e.g. `Simulation.fdtd(...).prepare()`). Report which
     entry point each scene used. The dict round trip itself (`s2.to_dict() == s.to_dict()`)
     is required for every scene regardless;
   - the FDFD source vector of `Simulation.fdfd(s, frequency=f).prepare().solver` is
     bit-identical, for scenes the FDFD path accepts.
2. `test_every_constructor_parameter_is_serialized`: for every class in scope, the serialized keys
   cover all constructor parameters (minus device). A parameter added later must fail this test.
3. `test_out_of_scope_raises`: one parametrized test, each case raises TypeError (a ModePort, a
   MaterialRegion, a symmetry, an anisotropic medium).

## Rules

- No defensive programming: no speculative validation, fallbacks, try/except, or compat shims.
  Fail loudly.
- Minimal tests: only the three above. Do not edit other tests and do not run the full suite.
- One command per shell call, no `&&`. GPU tests:
  `env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<workspace>/wt/core-scene-dict:<workspace>/wt/maxwell-scene-dict python -m pytest tests/api/public/test_scene_dict.py`.
  Never GPU 0. Check free memory with nvidia-smi first.
- Commit on branch `scene-dict/scene` with
  as the repository owner. No AI attribution,
  no commercial product names in commits or comments. Do not push or merge.
- Report: files changed, test output, every class you had to special-case and why, and anything
  ambiguous.

Answer to the background question (reviewer, 2026-10-30): serialize the Scene constructor value literally. background=None is written as "background": {"material": null}, and from_dict passes background=None back. Consumers (tokenizers) interpret null as vacuum. No vacuum id and no extra flag.

## Implementation notes (2026-10-30)

The implementation uses constructor reflection for the scoped classes and delegates core geometry
dictionaries to the core worktree. Material ids preserve object identity, including a shared
background. Small tensors retain dtype information; arrays above 4096 elements use path-qualified
references when an array store is supplied. Custom-grid coordinates remain NumPy float64 arrays
on reconstruction, without a detour through CUDA. Scalar Material constructor fields, including
mass density, use the same recursive encoding; unsupported nested classes raise TypeError.

Class-specific handling:

- Scene: schema layout, material table, null background, subpixel constructor name, and excluded
  scene capabilities.
- Structure: material table reference; metadata remains literal JSON data.
- ThinWire: conductor table reference; empty closed-loop endpoints reconstruct with None because
  its constructor rejects an explicit endpoint sequence for a closed loop.
- Material: medium/PEC/PMC kind follows its constructor flags.
- CopperFoil: explicit copper_foil kind and roughness="none" (roughness is not a constructor field).
- LossyMetalMedium: explicit lossy_metal kind; absent roughness is written as "none" and passed
  back to the constructor as None.
- GridSpec: uniform/custom/auto kind; custom coordinate lists and references reconstruct as NumPy
  arrays rather than tensor-valued dataclass fields.
- BoundarySpec: mixed kinds reconstruct through a none default plus explicit resolved face kinds,
  since mixed is a derived summary that the constructor does not accept.
- TFSF: dataclass fields retain its slab representation, whose axis and interval are not exposed
  in its box constructor signature; slab dictionaries use TFSF.slab on reconstruction.
- TerminalPort: current_surface reads the original declaration, not the resolved runtime property.
- PolySlab, PlanarPEC, SurfaceMesh: local signature-driven geometry records use the core geometry
  kind/list conventions; rotation fields use quaternions. Future constructor parameters are read
  automatically.
- Tensor, NumPy ndarray, and complex values have JSON encodings; metadata mappings bypass object
  tag interpretation so user keys such as "type" retain their meaning.

Acceptance paths: all four scenes compare tensors from
prepare_scene(...).compile_material_tensors(frequency=FREQUENCY). Scene (a) additionally compares
the Yee source vector from Simulation.fdfd(...).prepare().solver. Scenes (b), (c), and (d) do not
use FDFD: respectively they contain a PlanarPEC sheet, Bloch faces, and a ThinWire with PMC faces
and projection monitors, which that runtime does not accept.

Validation is pending CUDA device access in the execution environment. The prescribed pytest
command collected nine cases; all nine failed during CUDA initialization before serialization or
compilation assertions. nvidia-smi reported GPU 1, but /dev/nvidia* was absent in this environment.
No CPU substitute or other test suite was run.

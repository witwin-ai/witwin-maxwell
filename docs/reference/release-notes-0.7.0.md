# WiTwin Maxwell 0.7.0

Previous published release: 0.6.0 (2026-11-08). Release date: 2026-11-14.

0.7.0 is an API release. It finishes the move to one public model, `Scene -> Simulation -> Result`, and removes
the old forms without aliases. Almost every script written for 0.6 needs edits; the codemods listed below do most
of them. The CHANGELOG (`## [0.7.0]`) has the full entry for every change, with old-to-new tables, and
FEATURE_LIST.md remains the capability inventory.

Numerically, FDTD results of 0.6.0 scenes are unchanged except for the user-approved changes listed under
[Numerical changes](#4-numerical-changes): symmetry scenes with a soft plane wave or Gaussian beam, thin-wire ports
bound to nodes, partially filled dispersive interfaces under polarized or Kottke averaging, short polygon edges,
resumed and multi-GPU corner cases, and deterministic least-squares fits. No admission blocker is left: every FDTD
scene the solver accepts runs on the CUDA core path, and the automatic fallback to the existing stepping path is
gone.

## 1. Highlights

- **One immutable scene.** `Scene` is a frozen dataclass. Build it with the constructor and derive variants with
  `scene.updated(...)`, which shares every unchanged object. `SceneModule.to_scene()` returns the same `Scene`, and
  its static parts are compiled once and reused across optimization steps.
- **One configuration entry per solver.** `Simulation.fdtd/fdfd/electrostatic/capacitance(scene, ..., config=...)`
  take one frozen config (`FDTDConfig`, `FDFDConfig`, `mw.electrostatic.ElectrostaticConfig`,
  `mw.electrostatic.CapacitanceConfig`, `mw.mom.MoMConfig`); the device lives on the config, not on the scene.
- **Nothing is silently dropped.** Every solver refuses, at `Simulation` construction, the declarations it cannot
  represent and the trainable inputs it cannot differentiate. Refusals come from one capability registry
  (`mw.capabilities(method)`) and raise one `UnsupportedFeatureError` listing every item with its alternative.
- **Domain exceptions.** `MaxwellError` with `UnsupportedFeatureError`, `SetupError`, `ConvergenceError`,
  `SchemaError` and `CapacityError`; each keeps its built-in base, so existing `except` clauses still match.
- **Physics-typed results.** Monitors, fields, far fields and S-parameters are `FieldData`, `FluxData`,
  `ModeAmplitudeData`, `SParameterData` or `FarFieldData` with named dimensions, coordinates, units and phasor
  conventions. `Result` keeps its core; domain accessors are namespace functions.
- **Namespaces and stability.** The top level holds the core model; every other public object has one home
  (`mw.rf`, `mw.media`, `mw.monitors`, `mw.fdtd`, `mw.mom`, `mw.io`, ...). Experimental objects are marked and warn
  once. `mw.__version__` and a `py.typed` marker are shipped.
- **CUDA core path everywhere.** Batch lanes, multi-GPU shards, differentiable forwards, resumed runs, float64,
  Bloch/complex fields, bent charts, anisotropic and gyromagnetic media, material caches and every monitor type
  run on the fused CUDA core path in whole-step graphs. Results are bitwise equal to the former existing-path
  runs except the approved changes below.
- **Scene dictionaries `witwin.scene/2`.** `Scene.to_dict()` covers every scene field and public scene type,
  keeps tensor dtypes, trainable flags and shared tensors, writes strict and byte-deterministic JSON, and reads
  0.6 (`/1`) documents through a logged migrator. JSON Schemas of every versioned payload are published.
- **Layered MoM for RFIC layouts** (experimental): H-matrix backend, polygon layouts and GDS import, surface
  admittance conductors, mm-wave Green tables, TRL de-embedding, device-parameter extraction and passive
  macromodels.

## 2. Breaking changes

Codemods (AST based, idempotent; run `apply <paths>`, then `check`) cover the mechanical rewrites:

| Codemod | Rewrites |
|---|---|
| `scripts/dev/codemods/070_p3b_raise_migration.py` | raise sites and `pytest.raises` types of the new exceptions |
| `scripts/dev/codemods/070_p4c_single_entry.py` | factory keywords into `config=FDTDConfig(...)` / `FDFDConfig(...)` |
| `scripts/dev/codemods/070_p4d_es_config.py` | electrostatic and capacitance keywords into their configs |
| `scripts/dev/codemods/070_p5b_namespaces.py` | top-level names into their namespaces |
| `scripts/dev/codemods/070_p6b_typed_monitors.py`, `070_p6e_result_slim.py` | monitor payload reads, removed `Result` members |
| `scripts/dev/codemods/070_p7a_scene_frozen.py` | `add_*`, `clone`, field assignment, `Scene(device=...)` |
| `scripts/dev/codemods/070_p7b_compiler_internal.py` | `compile_*` calls |

### Migration table

| 0.6 | 0.7.0 |
|---|---|
| `scene.add_structure(s)` (and the eleven other `add_*`) | `mw.Scene(..., structures=(s, ...))` or `scene = scene.updated(structures=(*scene.structures, s))` |
| `scene.clone(**overrides)` | `scene.updated(**fields)` |
| `scene.grid = g`, `scene.sources.clear()` | `scene = scene.updated(grid=g, sources=())` |
| `mw.Scene(device="cuda", verbose=..., lazy_meshgrid=...)` | `FDTDConfig(device=...)` (likewise `FDFDConfig`, `MoMConfig`, `ElectrostaticConfig`, `CapacitanceConfig`); `lazy_meshgrid` is internal |
| `Scene.from_dict(d, device="cuda")` | `Scene.from_dict(d)` (CPU tensors; the config places them) |
| `scene.bc_type`, `scene.dx`, `scene.pml_thickness`, ... | `scene.boundary.kind`, `scene.grid.dx`, `scene.boundary.num_layers`, ... |
| `scene.compile_ports(device=d)`, `prepared.compile_materials(...)` (all `compile_*`) | internal functions `witwin.maxwell.compiler.compile_ports(prepare_scene(scene, device=d))`, `compile_materials(prepared, ...)` |
| `prepared.structures` (`PreparedScene` was a `Scene` subclass) | `prepared.scene.structures` (`PreparedScene` is a composition) |
| `scene.to_<solver>(...)` (export to the external reference solver) | the adapter function `scene_to_<solver>(scene, frequencies=..., run_time=...)` of the adapter module `witwin.maxwell.adapters.<solver>` (the method only delegated to it) |
| `Simulation.fdtd(scene, frequencies=f, run_time=..., absorber=..., cpml_config={...}, ...)` | `Simulation.fdtd(scene, frequencies=f, config=mw.FDTDConfig(run_time=..., absorber=..., cpml=mw.fdtd.CPMLConfig(...)))` |
| `Simulation(..., reference_frequency=..., material_cache=...)` | `FDTDConfig(reference_frequency=..., material_cache=...)` |
| `Simulation.fdtd(..., fdtd_core_path=..., accelerated_setup=...)` | removed: the core path is the only path |
| `Simulation.fdtd(..., backend="cpu_reference")`, `WITWIN_MAXWELL_*_KERNEL` | removed: FDTD runs on CUDA only |
| `mw.fdtd.FDTDCoreFallbackWarning`; `result.stats()["execution_path"]`, `["fallback_reason"]` | removed: a failed core activation, capture or library load raises; a failed accelerated setup retries once with the PyTorch setup and warns (`mw.MaxwellWarning`) |
| `mw.GMRES` | `mw.fdfd.FDFDSolverConfig` |
| `Simulation.electrostatic(scene, boundary=..., solver=...)` (returned a result) | `Simulation.electrostatic(scene, config=mw.electrostatic.ElectrostaticConfig(...)).run()` |
| `config.run_time = ...` (configs were mutable) | `dataclasses.replace(config, run_time=...)` (frozen, re-validated) |
| `mw.CopperFoil`, `mw.WavePort`, `mw.Circuit`, ... (274 top-level names) | `mw.media.CopperFoil`, `mw.rf.WavePort`, `mw.circuits.Circuit`, ... (one namespace each, no alias) |
| `mw.run(sim)`, `mw.BJT`, `mw.MOSFET` | `sim.run()`; the placeholders are removed |
| `polarization="ez"`, `absorber="CPML"` (any case) | exact value strings or Enum members (`"Ez"`, `"cpml"`); anything else raises `SetupError` |
| `result.monitor(name)["Ez"]` (payload dict) | `result.monitor(name)` is `FieldData`/`FluxData`/`ModeAmplitudeData` with a leading `f` dimension |
| `result.solver_stats`, `result.tensor(...)`, MoM-only `Result` members | `result.stats()`, `result.field(...).values`, `mw.mom.<function>(result)` |
| `ModeAmplitudeData.sel(direction="+")` | `.sel(propagation="+")` |
| `Result(..., scene=prepared)`, `Result.load(path, scene=prepared)` with a `PreparedScene` or stand-in | `scene=prepared.scene` (or `scene=None`); any other object raises `AttributeError` |
| `Result.load(...)` of a 0.6 snapshot (schema 2) | refused (`SchemaError`); re-run and save (schema 3) |
| `"schema": "witwin.scene/1"` documents, `TypeError("not in witwin.scene/1: ...")` | `"witwin.scene/2"`; `/1` is read through `mw.io.migrate_scene_dict`; unsupported objects raise `mw.SchemaError` with their document path |

### Details by area

- **Immutable `Scene` (plan phase 7a).** Collections are tuples; assigning a field or mutating a collection raises.
  `updated()` re-validates only the replaced fields and the cross-object rules that read them. The legacy
  convenience properties are removed. `SceneModule` subclasses that build their static part once
  (`self._base = mw.Scene(...)`) and return `self._base.updated(material_regions=...)` share the static objects, and
  the material compiler reuses their compiled prefix (phase 7d).
- **Device on the config (Q18).** Preparation places the grid and compiled tensors on `config.device`; the scene's
  declarative tensors are not copied. A non-CUDA device is refused for the grid solvers.
  `mw.rf.PECFrame(length).structures(port, scene, device=...)` takes the device explicitly.
- **`compile_*` internal, `PreparedScene` a composition (phase 7b).** `witwin.maxwell.compiler` is not public. A
  `PreparedScene` holds the grid state, the compile caches and `prepared.scene` (the scene with the resolved grid,
  the PML extrusions and the coordinates on the device).
- **External export (phase 7c).** The scene method is removed; the adapter function it delegated to is the
  entry point, and the exported simulation is unchanged.
- **Configs (phase 4).** `FDTDConfig`, `FDFDConfig` and `SpectralSampler` are frozen; `FDTDConfig.cpml` is a
  `CPMLConfig` dataclass (a misspelled key is now an error instead of being ignored); the factories take one
  `config`. Invalid electrostatic values raise `SetupError`.
- **Correctness refusals (phase 2).** FDFD refuses lossy metals, surface impedance, copper foil, modulation, lumped
  elements, ports, internal absorbers, thin sheets, conformal PEC, phase screens and a CW frequency that differs from
  the solve frequency. Homogeneous MoM, layered MoM and the electrostatic solves refuse the scene collections and
  material fields they ignored. Scene constructors refuse trainable tensors for values they store as Python floats.
  FDTD refuses trainable inputs it cannot differentiate (and differentiates a trainable `MaterialRegion` geometry);
  FDFD routes every trainable run through the gradient bridge. `Structure.phase_screen`/`uv` and
  `MaterialRegion(symmetry=...)` are refused. FDTD with `Scene.symmetry` refuses projection monitors, a diffraction
  monitor across a symmetry axis and closed surfaces at an image plane (these results were silently wrong).
- **Capability registry (phase 3).** `mw.capabilities(method)`, `Simulation.check()`, the field-consumption manifest
  (`witwin.maxwell.capability.fields`) and the public key vocabulary. 461 non-MoM and 142 MoM raise sites raise
  `UnsupportedFeatureError` with `.feature`, `.method`, `.alternative` and `.items`.
- **Typed results (phase 6).** FDFD, FDTD and MoM results are physics-typed; values are the same tensors. Every
  frequency-domain value keeps a leading `f` dimension (length 1 for one frequency) unless a frequency is selected.
  `Result.save` writes snapshot schema 3; schema 2 is refused (Q20).
- **Namespaces, Enums, stability, version (phase 5).** 274 names leave the top level. String options are `str`
  Enums parsed once; other letter cases and aliases raise. Experimental objects emit one `ExperimentalWarning`
  (a `MaxwellWarning`) per object and process on first use from outside the package.
- **FDTD execution (W7).** The CPU reference profile and the kernel-selector environment variables are removed;
  reference implementations moved to the test suite (`tests/oracles/`). There is no fallback to the existing path:
  core failures raise (W7 row 16c, user decision 2026-11-12). Wave-port, terminal wave-port and modal-port sweeps
  refuse a configured `material_cache`, which they never applied.
- **Scene dictionaries (phase 8).** See [witwin.scene/2](#witwinscene2-and-schemas) under New features.

- **Existing FDTD runner deleted (W7 row 16e).** The per-component FDTD runner is removed from the package: its 37
  CUDA operators (per-component magnetic and electric Yee updates in standard, bounded, dense and compressed CPML and
  modulated forms, and the batched per-component lane updates) and their launchers, the bounded per-component and
  overlapped schedules of the multi-GPU solver, and the per-component loop of the differentiable forward. Every FDTD
  run already took the core path (rows 16c and 16d), so no public API and no result changes. The main FDTD extension
  ABI is `stable_abi_v17`; prebuilt binaries with an older marker are rejected and rebuilt. The Torch Yee reference
  that tests compare against lives in `tests/oracles/`.
- **Strict scene reads.** `Result.scene` is the run's `Scene` or `None` (a scene-less `Result.load(path,
  scene=None)`, unchanged). Any other object, such as a `PreparedScene` or a stand-in, raises `AttributeError` naming
  the field when the Result is built or loaded and on monitor, field, material, power-loss and ESD reads; before, it
  was silently read as "no chart", "no monitors", "no sources" or "no circuits". Pass the `Scene`
  (`prepared.scene` for a prepared one). `Simulation`, the adjoint input census and the multi-GPU wire guard read
  scene fields the same way.

## 3. New features

### Scene and serialization

- <a id="witwinscene2-and-schemas"></a>**`witwin.scene/2` scene dictionaries.** `Scene.to_dict(arrays=None)` /
  `Scene.from_dict(d, arrays=None)` cover every `Scene` field (networks, circuits, material regions with their density
  values, electrostatic terminals, charge densities, internal absorbers, symmetry, coordinates, layered backgrounds)
  and every public scene type (177 registered types), rebuilt through the public constructors. Tensors keep dtype and
  shape, inline or in a caller-owned arrays table (over 4096 elements, named by document path); trainable tensors
  record `requires_grad` and come back as fresh leaves; a tensor shared by identity is written once (`TensorRef`) and
  reloads as one shared leaf, so an optimizer sees one parameter. Runs of 64 or more same-type records (for example
  16,384 point monitors) are written as one columnar `Records` block. Documents are strict JSON, deterministic byte
  for byte across processes (structure identities are numbered per scene), and record `package_version`. Objects
  outside the schema (callables, user subclasses of scene types, a pole-perturbed `PerturbationMedium`) raise
  `mw.SchemaError` with the document path. Every serializable scene of the gate, benchmark and data-generation corpus
  round-trips to identical fingerprints and prepared tables, and validates against the published schema.
- **`mw.io.migrate_scene_dict(d)`** upgrades a 0.6 `witwin.scene/1` document; `Scene.from_dict` calls it and logs
  every assumed value at INFO.
- **JSON Schemas.** `mw.io.scene_json_schema()` returns the JSON Schema 2020-12 of `witwin.scene/2`, generated from
  the type registry; field schemas of `PortData`, `NetworkData`, `WireData`, the FDTD resume checkpoint and the
  distributed result manifest are published under `witwin/maxwell/schemas/` with `VERSIONS.toml`. Additive schema
  changes keep the version and bump a recorded revision; breaking changes need a new major version and a migrator.
- **SceneModule structural sharing** reused by the material compiler: static structures are compiled once per
  prepared scene.

### FDTD

- **Opt-in magnetic volume DFT:** `FDTDConfig(full_field_dft=True, volume_dft_components=("Ex", ..., "Hz"))` records
  the H phasors on their Yee positions (`Result.H`, `Result.field("Hx")`).
- **Run records for volume-field consumers:** `Result.stats()` gains `dft_window_requested`,
  `last_step_field_ratio` and related keys; `Result.release_solver()` also frees the compiled material cache
  (a 64^3 one-frequency result keeps 7.0 MB instead of 38 MB).
- **Core path coverage (W7 rows 4-16b):** PMC faces and PMC symmetry, magnetic materials, Mur and mode-ABC faces,
  internal absorbers, closed-transverse CPML, resume, gyromagnetic ferrites (CUDA magnetization kernels), full
  `Tensor3x3` anisotropy, float64 fields with CUDA graphs, Bloch/complex fields (also beside lumped elements, ports
  and copper foil), bent coordinate charts, material caches, differentiable forwards, batch lanes, multi-GPU shards
  and every monitor type.

### Results and data

- `witwin.maxwell.results`: `TensorData`, `FieldData`, `FluxData`, `ModeAmplitudeData`, `SParameterData`,
  `FarFieldData` (also over an arbitrary list of directions), `field_component_units(...)`;
  `NetworkData.s_parameters`.
- Diffraction and Purcell postprocessing return tensors (differentiable through trainable runs).

### Layered MoM (experimental)

- H-matrix backend (`operator_backend="hmatrix"`), body-aware near-field preconditioner.
- Polygon layouts: non-Manhattan `PlanarPEC` with holes, `Prism` and arbitrary-outline `ExtrudedPolygon` vias,
  vias through intermediate planes, GDS import (optional extra `witwin-maxwell[rfic]`).
- `conductor_model="surface_admittance"` and the default `conductor_model="auto"` (surface admittance for lossy
  conductors with t/delta >= 0.3 at the top frequency, sheet otherwise).
- Frequency-dependent layer dielectrics (`mw.media.WidebandDebye`, `mw.media.PolePermittivity`), edge ports using
  `LumpedPort.voltage_aperture`, vertical ports on closed bodies, mm-wave vertical Green tables up to 200 GHz.
- Device-parameter extraction (L, R, Q, M, k, self-resonance, pi/T models), two-line characterization, TRL
  de-embedding, passive macromodels (`mw.mom.passive_macromodel`), adaptive sweeps with passivity auto-enforcement.

### Other

- `mw.capabilities(method)`, the capability vocabulary and the field manifest; `Simulation.check()`.
- `mw.__version__`, PEP 561 `py.typed`, stability markers and `docs/reference/public-api.csv`.
- New experimental geometry `mw.geometry.ExtrudedPolygon`.
- Data generation: `benchmark/datagen/scheduler.py --format emfc` writes one EMFC record per task, with the scene
  document and stored H where the H-from-E derivation is not valid.
- Test tiers: plain `pytest` runs the fast tier; `--run-slow`, `--slow-budget`; the `test` extra installs
  `pytest-xdist`.

## 4. Numerical changes

Each change below moved a pinned result and was approved by the user as an intended behaviour change (tier T2 of
the numerical-equivalence standard). Every other pinned FDTD result is bitwise unchanged against 0.6.0 (G1/G4 gates
and the BITID pins).

| Change | Affects | Physics reason |
|---|---|---|
| `Scene.symmetry` reproduces the explicit mirrored domain | symmetry scenes with a soft `PlaneWave` (fields fall by `1/sqrt(2)` per folded axis), a `GaussianBeam` (phase), or domain-spanning plane flux (mirrored-domain power) | A folded scene stands for the kept part plus its images: the plane-wave aperture, the beam reference center and plane flux must be taken over the mirrored domain, not the folded one |
| Mode planes on a symmetry image plane are refused | `ModeSource`, `ModeMonitor`, mode ports and wave ports under `Scene.symmetry` | The folded transverse mode solve imposed a zero-field edge at the plane (0.6 % index error under PEC, a different mode under PMC) |
| Thin-wire ports bound to nodes read the chord field | `WirePortBinding.nodes(...)` ports | The node-potential difference alone missed the Yee field between the terminals; the port now converges under refinement and matches the qualified MoM reference |
| Dispersive interfaces under `polarized`, `polarized_tensor`, `kottke_smooth` | partially filled cells of Debye/Lorentz/Sellmeier media | Pole strengths were blended arithmetically while the static part followed the window rule; curved dispersive interfaces now converge (Sellmeier sphere extinction 3.3 % -> 1.2 % error against Mie) |
| Coax shield occupancy (`pec_occupancy`) | polygons with edges shorter than about `sqrt(eps32)` (a 96-gon coax shield) | witwin-core f7f08ca clamps the squared polygon edge length at `finfo.tiny` instead of `eps`, so short edges are measured exactly; no staircase cell flips |
| Deterministic least squares | rational fits (`fit_rational`, roughness and surface-impedance models), dispersion-fit NNLS | The default CPU driver (`gelsy`) was not run-to-run deterministic; explicit `gelsd` (CPU) and `gels` (CUDA) drivers make fits repeatable (pinned poles moved by 2.2e-14 relative) |
| CPML `memory_mode="auto"` depends only on the scene | CPML runs whose dense storage is at most 64 MiB | The mode used to follow the card's free memory, and dense and slab storage round differently; identical runs are now bitwise repeatable |
| Resumed runs with embedded networks or terminal ports | port and network V/I after `resume_from` | The resumed observer graph read the port DFT rows from step 0; resumed spectra now equal the uninterrupted run bitwise |
| Multi-GPU seam probe | a point monitor in a shard's last owned cell | The fused launch updates the ghost `Hx` node the bounded launches left at 0; equals the single-device run bitwise |
| Gradient runs sample the passivity guard | differentiable closed-transverse CPML runs | The differentiable forward runs on the core path like an ordinary run |
| Opt-in H volume DFT | only runs that request H components | New feature; E-only runs are unchanged |
| `Result.stats()["dft_window"]` reports the recorded window | pulsed runs | Pulsed spectra are always recorded without a window; the requested value moved to `dft_window_requested` (numbers unchanged) |
| Layered MoM rounds 1-7 | layered MoM networks | Surface-admittance conductors, compressed body blocks with a deterministic sparse LU, Schur preconditioning, streamed graded mm-wave calls, `conductor_model="auto"`, passivity auto-enforcement of adaptive sweeps, derived `StoppingRule.rtol`; each round approved with its mover list (MoM results within the stated bounds) |

The benchmark suite re-specified scenes (scored frequencies at 10 or more cells per wavelength, longer ring-down
windows, zero-DC dipole carriers) and regenerated the external references of 29 rows (`validation/bench-respec`,
`validation/bench-references`); these change benchmark definitions and scores, not library numerics.
Gate scenes are pinned to their recorded definitions and fingerprints.

The harmonic MoM snapshot fix (see Fixed bugs) was approved as T2 only because it changes a file protected by the
consumer-metadata bridge; its numerics are bitwise unchanged (T0), and its consumer re-receipt passed.

## 5. Fixed bugs

- Thin-wire port impedance did not converge under refinement (0.6.0 known limitation); fixed (above).
- Resumed existing-path runs with embedded-network or terminal ports reported wrong port spectra (up to 89 % of the
  peak).
- Silently wrong symmetry results (plane-wave normalization, beam reference, plane flux, closed surfaces and
  projections at image planes) are corrected or refused.
- Scene dictionaries lost float64 geometry precision and int32 mesh faces, could not rebuild some port apertures,
  raised `AttributeError` on `ComplexPolySlab`, silently wrote a callable `CustomSourceTime` without its function, and
  wrote process-dependent structure ids; all fixed in `witwin.scene/2`.
- FDTD: a trainable `MaterialRegion` geometry ran forward-only; CPML runs were not bitwise repeatable when the card was
  busy.
- FDFD: multi-frequency and local-response results lost `Simulation.metadata`; a trainable `sigma_e` at 0 had zero
  gradient; multi-frequency trainable runs failed or returned detached fields.
- Differentiable batch lanes returned no embedded networks and detached port/circuit records.
- Electrostatics and capacitance detached a trainable `eps_r`.
- Layered MoM refused accurate vertical and horizontal Green tables over a PEC ground.
- Wave-port and terminal wave-port `prepare().run()` returned the internal prepared scene as `result.scene`.
- Coupled harmonic MoM snapshot archives could not be loaded with `torch.load(weights_only=True)`: the snapshot stored
  MoM option Enum members, which the safe unpickler refuses. They are written as plain strings again; values and
  restored configurations are unchanged, and archives written in between still load through `Result.load`.

## 6. Known limitations and follow-ups

### Library

- **Mode planes at symmetry image planes are refused.** PEC/PMC-aware transverse mode operators are a follow-up;
  until then keep mode planes more than half a cell from every image plane or simulate the full domain.
- **Fixed-k TFSF absolute phase.** Phase stays relative to the incident-line reference (0.6.0 limitation); the
  absolute-phase branch met its phase gates but not the long-run cutoff gate and is deferred.
- **Architecture coverage.** The core path, the bitwise census and every performance figure are qualified on A6000
  (sm_86) only. Other architectures run the same kernels and may differ at rounding level; there is no longer a
  non-core path to fall back to.
- **`BoundarySpec.strength`** keeps two meanings (multiplier and absolute sigma scale); documented only, a single
  meaning would change CPML numerics (Q13). The shared conductivity sign convention is unchanged (Q21).
- **Trainable nonlinear coefficients** (chi2, chi3, TPA) are refused (Q22).
- **0.6 result snapshots** (schema 2) cannot be loaded (Q20).
- Remaining 0.6.0 limitations (curved and tilted copper edge response, oblique TE under fixed k, fixed-k source
  coverage, benchmark residuals against the external reference solver) are unchanged; see FEATURE_LIST.md
  "Known Limitations".

### Test qualification

- **MoM GPU tier.** The pre-existing MoM test failures found before release were fixed on the test side (no library
  or pinned value changed) or skip with a named reason; the weights-only snapshot failures were a library payload bug,
  fixed (see Fixed bugs). The expensive MoM nodes moved to a milestone tier (`mom_milestone` marker, deselected unless
  `--run-mom-milestone`, 20 nodes in 12 files, about 65,000 A6000-seconds), and two of them have smaller regression
  twins in the default tier that reach the same code branches with the same assertions.
- **MoM milestone tier: partly run on 0.7.0 code.**
  - Passed on 0.7.0 code (master 25e8a298a), 9 nodes: `test_ewald_refinement` (original case);
    `test_harmonic_dual_streamed` (root, tangent and nonlinear gradients); `test_magnetic_volume_physical_control[1]`;
    `test_periodic_hierarchical_public` `[1-mixed-hmatrix-2]` and `[2-mixed-h2-2]`; `test_streamed_dual_public`
    (root and archive); `test_tensor_cell_response` (kernel batch 8192); `test_tensor_compressed_preparation`;
    `test_tensor_volume_harmonic_bundle`.
  - Not run on 0.7.0 code, 11 nodes: the 10 directional milestone nodes (`test_directional_sparse_system`,
    `test_directional_dag_system`, `test_directional_automatic_interior`), last passed on 2026-11-12 at 189fff6af;
    and `test_tensor_harmonic_magnetic_recovery[compressed]`, last run locally on the tree before the MoM test
    tiering.
- **GSP radial-order bridge tests need an external artifact.** The 13 `test_fmm_radial_order_bridge` tests read a
  bundle named by `GSP_RADIAL_ORDER_BRIDGE`, produced by
  `scripts/dev/fdtd/regression_shards/build_module_inputs.py --radial-bridge` for the current source. Without it they
  skip with that reason; a skip is not a qualification, and they count as not qualified for 0.7.0 (user decision).
- **Other artifact-gated MoM tests.** 11 tests that compare against the separate `maxwell_bench` package skip when it
  is not installed, and 7 single-buffer assembly tests skip unless a baseline checkout is named by
  `WITWIN_SIE_COMPACT_RIESZ_BASELINE` (the rebuild recipe is in the test file).
- **Large FDFD test memory.** `tests/fdfd/test_symmetric_formulation.py::test_interior_matches_large_domain_reference`
  peaks at 17.7 GiB of GPU memory; run it on a device with more than 18 GiB free.

## 7. Supported platforms

- Python `>=3.10,<3.15`; PyTorch `>=2.10` (wheel compatibility tested with torch 2.10 to 2.13 on CUDA 12.6/12.8
  indexes); NumPy, SciPy, Matplotlib, tqdm, IPython.
- witwin-core `>=0.4.7,<0.5`. 0.7.0 relies on the witwin-core changes released in 0.4.7: `ExtrudedPolygon`,
  geometry dictionaries that keep tensor dtypes, the scale-relative convex polygon inside tolerance and the
  `finfo.tiny` polygon edge guard. witwin-core 0.4.7 is published before 0.7.0.
- Linux x86_64 and Windows x64 wheels with prebuilt binaries for the main FDTD extension (`stable_abi_v17`), the
  core-path extension (`core_abi_v15`) and the accelerated-setup extension (`setup_abi_v7`), built against the
  LibTorch Stable ABI. CUDA SASS for sm_70, 75, 80, 86, 87, 89, 90, 100, 101, 120 and PTX for sm_120. A
  machine without a CUDA toolkit runs the prebuilt binaries.
- The experimental layered MoM extension (`mom_layered_abi_v1`) is not prebuilt: it is compiled on first use with
  `torch.utils.cpp_extension` and needs a CUDA toolkit on the host.
- FDTD and FDFD need a CUDA GPU (`config.device` must be CUDA). Validated on NVIDIA RTX A6000 (sm_86).
- Optional extras: `rfic` (`shapely>=2.1`, `gdstk>=1.0.1`) for layered MoM polygon layouts and GDS import; `test`
  (`pytest`, `pytest-xdist`); `xarray`; the CUDA direct-solver extras `direct` / `direct-cu13`.

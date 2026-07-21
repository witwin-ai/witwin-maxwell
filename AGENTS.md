# Repository Agent Guide

This document defines the working rules for coding agents in this repository. Keep `AGENTS.md` and `CLAUDE.md` identical.

## Core Rules

- Use the `maxwell` conda environment for all Python commands, tests, and scripts.
- Keep the codebase clean. Do not preserve legacy code, compatibility shims, or backward-support paths unless explicitly requested.
- Write all code comments and commit messages in English.
- Keep third-party commercial product, service, and solver names out of commit messages, branch names, PR titles, general development documents, comments, and feature descriptions. Describe the behavior with neutral terms such as `external reference solver`, `reference backend`, or `third-party adapter` so commercial brands do not spread through the repository or GitHub history.
- Use an exact commercial product name only where technically required for an interoperability API, adapter module, executable command, dependency declaration, legal attribution, or a narrowly scoped adapter test/document. Do not repeat that name in unrelated implementation, validation, or numerical-method changes unless the user explicitly requests it.
- Maintain `FEATURE_LIST.md` as the user-facing feature inventory. Every new user-visible feature, public API capability, or meaningful workflow addition must update that file in the same change.
- Store active development documents by type under `docs/plans/`, `docs/memory/`, `docs/assessments/`, and `docs/reference/`. Keep solver and feature areas together in each type directory; use descriptive filenames instead of extra `fdtd`, `fdfd`, or similar directory layers. Move completed or superseded development documents to the matching type under `docs/archive/`. Keep `FEATURE_LIST.md` at the repository root as the user-facing feature inventory.
- Prioritize correctness and efficiency over convenience.
- Core computation is GPU-first. Minimize GPU-CPU transfers and do not introduce new CPU fallback paths unless explicitly requested.
- PyTorch-native integration is a hard requirement. New differentiable or optimization-facing APIs must be designed as native PyTorch workflows, not thin wrappers around non-PyTorch side paths.
- The stable public architecture is `Scene + Simulation + Result`. New user-facing code must follow this model.
- `Scene` must support both direct declarative construction (`mw.Scene(...)`) and a PyTorch-native module-style scene definition that ultimately materializes into the same public `Scene` object model.
- Do not reintroduce old public convenience paths such as `Scene.set_*`, `Scene.with_*`, `mw.FDFD`, or `mw.FDTD`.
- Preserve the compile-layer naming convention: scene-facing helpers use `compile_*` names such as `Scene.compile_materials(...)`, `Scene.compile_relative_materials(...)`, and `Scene.compile_material_tensors(...)`.
- Shared core geometry constructors default to `device=None`; `Scene(...)` owns device placement and defaults to CUDA.

## Project Overview

This repository is a PyTorch-native differentiable full-wave electromagnetic simulation framework for Maxwell's equations with two internal solver runtimes:

- `FDFD`: frequency-domain finite-difference solver using sparse linear algebra.
- `FDTD`: time-domain Yee-grid solver with native CUDA kernels, CPML support, and differentiable adjoint workflows.

The public API is declarative:

- `Scene`: physical problem definition
- `Simulation`: execution configuration and backend selection
- `Result`: fields, monitors, metadata, plotting, and persistence

For differentiable and optimization workflows, the public API must also remain PyTorch-native:

- trainable scene definitions may be implemented as `torch.nn.Module` via `SceneModule`
- module-style scenes must compile or materialize to the same `Scene` public contract
- solver entry remains `Scene -> Simulation -> Result`, even when the `Scene` is produced by a PyTorch module

Supporting first-class Maxwell scene objects include:

- `Domain`
- `GridSpec`
- `BoundarySpec`
- `BoundaryKind`
- `MaterialRegion`
- `Structure`
- `Material`
- `PointDipole`
- `PlaneWave`
- `GaussianBeam`
- `ModeSource`
- `PointMonitor`
- `PlaneMonitor`
- `FluxMonitor`
- `ModeMonitor`
- `ModePort`

The current codebase is organized as a Python package under `witwin/maxwell/`, with tests under `tests/`. Shared geometry and structure primitives are re-exported through `witwin.maxwell` but originate from the shared `witwin.core` dependency.

## Repository Layout

- `witwin/maxwell/__init__.py`: public package exports.
- `witwin/maxwell/scene.py`: declarative scene model, domain/grid/boundary handling, `MaterialRegion`, and `SceneModule`.
- `witwin/maxwell/media.py`: public Maxwell material models, dispersion poles, and diagonal anisotropy descriptors.
- `witwin/maxwell/sources.py`: public source definitions and source-time waveforms.
- `witwin/maxwell/monitors.py`: point, plane, flux, and modal monitors.
- `witwin/maxwell/ports.py`: first-class modal port definitions.
- `witwin/maxwell/simulation.py`: public simulation entrypoint, runtime preparation, and backend dispatch.
- `witwin/maxwell/result.py`: unified simulation result container.
- `witwin/maxwell/compiler/`: scene compilers for materials, sources, ports, and monitors.
- `witwin/maxwell/fdfd/`: internal FDFD runtime and postprocessing helpers.
- `witwin/maxwell/fdtd/`: internal FDTD runtime, CPML logic, source injection, observers, and adjoint support.
- `witwin/maxwell/postprocess/`: near/far-field, directivity, RCS, scattering-parameter, and modal postprocessing.
- `witwin/maxwell/adapters/`: interoperability adapters such as Tidy3D export.
- `witwin/maxwell/visualization/`: shared visualization utilities.
- `docs/plans/`: active implementation plans and functional roadmaps.
- `docs/memory/`: current implementation notes and durable engineering context.
- `docs/assessments/`: current audits and capability assessments.
- `docs/reference/`: maintained technical reference material.
- `docs/archive/`: completed or superseded plans, assessments, design records, and status snapshots.
- `tests/`: pytest-based coverage for public API behavior, numerics, gradients, and validation workflows.

## Working Expectations

- Prefer package-style imports from `witwin.maxwell`, typically `import witwin.maxwell as mw`.
- New public examples, tests, and scripts should look like `Scene -> Simulation -> Result`, not direct backend construction.
- Build scenes with `Domain`, `GridSpec`, `BoundarySpec`, `Structure`, `Material`, source objects, monitor objects, `ModePort`, and `MaterialRegion` as appropriate.
- Use the declarative `BoundaryKind` / `BoundarySpec` vocabulary consistently; do not introduce new raw boundary-mode strings outside validated APIs.
- Use `Scene.add_structure(...)`, `Scene.add_source(...)`, `Scene.add_monitor(...)`, `Scene.add_port(...)`, and `Scene.add_material_region(...)` for scene assembly.
- For PyTorch-native scene definitions, prefer `SceneModule` subclasses that expose `to_scene()` and return a standard `Scene`; do not create a second incompatible simulation entrypoint.
- When behavior changes in a way a user would notice, update `FEATURE_LIST.md` to keep the documented feature inventory current.
- If backend internals must be inspected, prefer `Simulation.prepare()` and then inspect `prepared.solver` rather than importing or constructing backend classes through the public path.
- Keep new implementations aligned with the current declarative 3D API and compiler/runtime split.
- If a design choice would add extra abstraction, data copies, or compatibility branches, prefer the simpler and faster path.
- When changing numerics, preserve Yee-grid consistency, PML behavior, Bloch/periodic boundary semantics, and GPU execution flow.
- Avoid adding duplicate implementations in parallel files. Update the primary module instead.

## Running Code

Activate the environment first:

```bash
conda activate maxwell
```

Common commands:

```bash
python -m pytest tests
python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py
python -m pytest tests/core/scene/test_scene.py tests/materials/compiler/test_material_compiler.py
python -m pytest tests/boundaries/cpml/test_fdtd_cpml.py tests/monitors/observers/test_fdtd_observers.py
python -m pytest tests/gradients/test_fdtd_adjoint_bridge.py
python -m pytest tests/validation/benchmark/test_benchmark_system.py
python -m benchmark
python -m benchmark dipole_vacuum
python -m benchmark planewave_vacuum
```

Benchmark / validation workflow:

- Use `python -m benchmark` as the unified Maxwell-vs-Tidy3D validation entrypoint.
- Scenario definitions live under `benchmark/scenes/`, grouped by family under directories such as `dipole/` and `planewave/`.
- Tidy3D reference caches are stored under `benchmark/cache/`.
- Generated comparison plots are stored under `benchmark/plots/`.
- Benchmark metrics are aggregated into `benchmark/RESULTS.md` and should be updated by rerunning the benchmark command after benchmark-related changes.

If you add or modify Python code, prefer validating with targeted pytest coverage first, then broader tests if needed.

## Windows / Codex Execution Notes

- In this environment, `rg` may exist but fail with `Access is denied`. If that happens, switch immediately to PowerShell-native search with `Get-ChildItem` + `Select-String` instead of retrying `rg`.
- `conda run -n maxwell` may fail to pass stdin through to `python -` reliably in this Codex/PowerShell setup. Prefer `conda run -n maxwell python -c ...` for short snippets, or invoke the environment interpreter directly with `C:\Users\Asixa\miniconda3\envs\maxwell\python.exe`.
- Windows command-line length limits are easy to hit with `python -c` and large inline scripts. For long scripts, pass the code through an environment variable such as `PYCODE` and execute `python -c "import os; exec(os.environ['PYCODE'])"`, or use the environment interpreter with stdin if that path is reliable.
- Large single-shot `apply_patch` updates can also hit Windows path/command length limits. If a full-file replacement fails for size reasons, rewrite it in smaller patch chunks instead of retrying the same oversized patch.
- When checking whether `AGENTS.md` and `CLAUDE.md` match, do not use bare `fc` in PowerShell because it resolves to `Format-Custom`. Call `fc.exe` explicitly if you want the file-compare tool.

## Dependencies

Primary dependencies include:

- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm
- CuPy
- witwin

Optional dependencies:

- `tidy3d` for adapter export and benchmark cross-validation workflows

An NVIDIA GPU with CUDA is expected for core solver workflows.

# Documentation

The documentation tree is organized by document purpose. Solver and feature names stay in filenames rather than creating another directory layer.

## Active documentation

- [`plans/`](plans/): implementation plans and functional roadmaps that still have open work.
- [`memory/`](memory/): current implementation notes and durable engineering context.
- [`assessments/`](assessments/): current audits, gap analyses, and capability evaluations.
- [`reference/`](reference/): maintained technical facts, capability matrices, and code-linked inventories, including the [FDTD multi-GPU joint-solve guide](reference/fdtd-multi-gpu-joint-solve.md) and [capability-guard census](reference/fdtd-capability-guard-census.md).
- [`releases/`](releases/): release notes.
- [`validation/`](validation/): stable release-validation records.

The active functional roadmap is indexed in [`plans/next-functional-2026-07/README.md`](plans/next-functional-2026-07/README.md).

## Archive

[`archive/`](archive/) contains completed or superseded plans, historical assessments, implementation snapshots, and design records. Archived material preserves engineering history but is not the current source of truth.

## Placement rules

1. Choose the directory from the document's purpose, not its solver backend.
2. Use a descriptive filename such as `fdtd-cuda-graph-option-b.md`; do not add solver-specific directory layers.
3. Move a document to `archive/` when its work is complete, its implementation model no longer exists, or a newer document supersedes its facts.
4. Keep repository-wide user documents such as `README.md`, `CHANGELOG.md`, and `FEATURE_LIST.md` at the repository root.
5. Keep executable development utilities under `scripts/`, never under `docs/`.

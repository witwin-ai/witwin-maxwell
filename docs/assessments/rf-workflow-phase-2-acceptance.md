# RF Workflow Phase 2 Acceptance

Status: accepted for the single-device scope

Date: 2026-07-15

Evidence level: E2

## Scope

Phase 2 delivers geometry-backed `TerminalPort`, deterministic `PortSweep`
execution, complete complex `NetworkData`, S/Z/Y transforms, reference-impedance
renormalization, reference-plane shifting, mixed-mode conversion, Result snapshot
integration, and Touchstone 1.0/2.0 export.

All sweep columns execute sequentially on one CUDA device. Task-level and spatial
multi-device scheduling are explicitly excluded by user direction and are not
implemented, tested, or claimed by this acceptance record.

## Independent gates

- Terminal contract: named PEC `Box` terminals resolve to the same compiled Yee
  geometry as an equivalent explicit `LumpedPort`; missing, duplicate, rotated,
  non-PEC, out-of-domain, non-overlapping, off-grid, and invalid-reference-plane
  declarations fail during scene preparation or compilation.
- Sweep contract: the manifest fixes scene port order and one independent run per
  input column. Every inactive RF port receives an internal matched termination.
  Missing, non-finite, weak-spectrum, wrong-frequency, wrong-device, wrong-dtype,
  wrong-reference-impedance, and convention-mismatched columns are rejected
  instead of being marked valid.
- Discrete V/I consistency: coupled port voltage and branch current are both
  sampled at the implicit midpoint and assigned the same stagger time. The CUDA
  two-port fixture limits the residual inactive incident wave to `1e-5` of the
  outgoing wave.
- Network algebra: well-conditioned S/Z/Y round trips meet `1e-5` relative error;
  singular and unsafe solves report their frequency indices. Renormalization,
  reference-plane shifts, and mixed-mode conversion retain the autograd graph.
- Physical diagnostics: the CUDA two-port fixture meets
  `max|S12-S21| < 0.02` and maximum singular value `<= 1.02`. The canonical
  lossless-through matrix has power imbalance below `2%` for every input column.
- Persistence and interchange: `NetworkData` preserves explicit port ordering,
  valid-column state, conventions, and safe metadata. Touchstone export covers
  one-, two-, and three-port RI/MA/DB data, frequency units, standard two-port
  ordering, version-specific reference impedances, and detached CPU-only I/O.

## Validation commands

All Python commands use the repository `witwin2` environment and repository-local
pytest temporary directories.

```text
python -m pytest tests/rf
python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/core/scene/test_scene.py
ruff check <Phase 2 Python files and RF tests>
git diff --check
```

Acceptance result: `162 passed` for the complete RF suite and `63 passed` for
the public API, simulation smoke, and scene regression set. Ruff and
`git diff --check` both passed.

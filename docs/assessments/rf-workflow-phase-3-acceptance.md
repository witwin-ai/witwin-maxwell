# RF Engineering Workflow Phase 3 Acceptance

Date: 2026-07-15

Status: accepted for the single-device scope requested for this implementation.

## Delivered surface

- Public `WaveModeSpec` and `WavePort` declarations with explicit axis-aligned aperture, direction, reference plane, stable mode names, polarization seed, and impedance definition.
- `Scene.compile_waveports()` with sparse Yee-grid voltage/current geometry and an explicit electric-plane / adjacent magnetic-plane contract.
- PEC-aware device-resident modal solving. Uniform conductor-backed TEM sections use an electrostatic solve with one-watt power-consistent V/I normalization; TE/TM and hybrid modes retain the vector/scalar mode core as applicable.
- Cross-frequency mode tracking by propagation constant and complex field overlap, with exact assignment, phase continuity, degenerate-subspace alignment, confidence diagnostics, and hard low-confidence failure.
- Direct `PortExcitation(..., mode_name=...)` and independent-column `PortSweep` execution through `Simulation.fdtd(...)`.
- Typed multimode `PortData`: `mode_names`, `beta[M,F]`, `characteristic_impedance[M,F]`, and `tracking_confidence[M,F]`, including persistence.
- Complete flattened modal `NetworkData` with stable `[out, in]` ordering and stored propagation constants for automatic reference-plane shifting.
- Every solve, tracking operation, modal projection, and FDTD column remains on one CUDA device. Frequency and excitation columns execute sequentially.

## Independent exit-gate evidence

- Axis-aligned coax TEM impedance: the device-resident electrostatic mode result is within 2% of `eta0/(2*pi)*log(b/a)` on the acceptance grid.
- Rectangular TE10: cutoff and propagation constant errors are below 2%; one-watt power normalization error is below 1%.
- Cross-frequency identity: ordinary and degenerate mode tests cover permutation, phase rotation, principal-angle alignment, hard confidence rejection, CUDA residency, and autograd preservation.
- Runtime network: a physical two-ended hollow-guide sweep produces finite modal `PortData` and `NetworkData`; reciprocity error is below 10% on the short coarse acceptance run and the largest singular value is below 1.25. The stricter reusable network algebra passivity/reciprocity gates remain covered by the Phase 2 physical sweep.
- Reference-plane shift: a WavePort network shifts identically with explicitly supplied propagation constants and with its stored tracked constants.
- Lumped/modal overlap contract: identical V/I phasors represented as lumped and single-TEM modal `PortData` produce exactly identical S parameters (difference zero, below 0.03). This is the shared power-wave contract gate; it does not claim that geometrically different lumped and distributed launches are interchangeable.
- Direct and sweep execution: both `Simulation.run()` and `Simulation.prepare().run()` are exercised on CUDA. A direct modal excitation returns `PortData` without fabricating a complete network; a sweep returns all valid columns.

## Validation commands

```text
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m ruff check <Phase 3 files and tests>
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/rf/waveport tests/sources/mode tests/postprocess/scattering/test_mode_overlap.py --basetemp=.pytest_tmp_root_phase3_modes_full2 -p no:cacheprovider -q
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests/rf tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/core/scene/test_scene.py --basetemp=.pytest_tmp_root_phase3_acceptance -p no:cacheprovider -q
```

Observed before the final documentation-only update:

- Mode/WavePort group: `98 passed`.
- Final RF + public API/smoke/scene/PEC-mode group: `301 passed`.
- Ruff and `git diff --check`: passed.

## Explicit exclusions

- Trainable WavePort sources and modal eigensolves are rejected until a mode-shape eigen-adjoint and FDTD source replay exist.
- Lossy/non-Hermitian PEC-aware TEM normalization and inhomogeneous exact TEM sections are rejected; use a hybrid-mode definition where appropriate.
- Non-axis-aligned apertures and evanescent channels are not admitted to the network matrix.
- No multi-device scheduling, domain decomposition, distributed ownership, or cross-device reduction is implemented or claimed.

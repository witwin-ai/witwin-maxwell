# Array workflow Phase 2 acceptance (codebook, scan, max-hold)

Status: functional gates accepted on branch `codex/array-phases-2-4`.
Date: 2026-07-17
Scope: one device. Task-level multi-GPU was removed from this plan's scope on
2026-07-16 and is not claimed here. The frozen 96^3 / 4096-step qualification run
and the phase-1 threshold requalification remain deferred to an exclusive timing
window (OPEN).

## Delivered contract

- `BeamCodebook` names a set of `[beam, port]` or `[beam, frequency, port]`
  incident power-wave vectors over one basis and holds no solver state.
- `BeamCodebook.from_scan_angles(...)` builds progressive-phase steering weights
  `a_{b,n} = amplitude_n * exp(+j k (r_n . d_b))` from explicit element positions
  (no geometry is inferred) and a scan-angle list, producing `[B, F, N]` weights.
- `ArrayBasisData.combine(codebook)` evaluates every beam's active network
  quantities, far field, realized gain, and EIRP in one batched call and carries
  the beam names and codebook metadata on the result.
- `BeamData.max_hold(metric=...)` reduces the beam axis to a per-direction
  envelope; `torch.amax` carries a subgradient through the winning beam while the
  `torch.argmax` winning-beam index is detached and non-differentiable.
- `ArrayBasisData.cache_key` is the content fingerprint: weight-invariant across
  `combine`, but sensitive to any embedded-pattern / physical-content change.

## HARD CONTRACT: zero additional field-solver steps

`tests/rf/array/test_array_codebook.py::test_combining_any_number_of_beams_executes_zero_fdtd_steps`
installs a step-summing fingerprint over the FDTD time-stepping entry
(`witwin.maxwell.fdtd.solver.FDTD.solve`, monkeypatched to accumulate
`time_steps`) and combines 64-, 256-, and 1024-beam codebooks plus a
`max_hold("realized_gain")` envelope.

- Measured executed FDTD steps during combination for B in {64, 256, 1024}: **0**.

Falsification (2026-07-17): the counting wrapper is not vacuous — invoking the
wrapped step function with `time_steps=40` increments the fingerprint to 40, and
a real FDTD `PortSweep`/dipole run drives it positive. Only `combine`/`max_hold`
leave it at zero.

## Command

```bash
conda run -n maxwell --no-capture-output python -m pytest tests/rf/array/test_array_codebook.py -q
# 12 passed
```

Covered: codebook-vs-per-beam parity, frequency-flat broadcast, shape/name
validation, zero-step contract (64/256/1024), max-hold envelope/argmax
consistency and masked-metric refusal, scan-angle progressive-phase construction
and validation, weight-invariant cache key with content-change invalidation, and
the fail-closed scene-gradient guard.

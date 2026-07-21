# G2a acceptance — passive lossy-wire current recurrence (B2) + ohmic_loss

Track: `g2-lossy-wire`. Stage: **G2a** (B2 recurrence + real ohmic_loss + gates + falsifications).
Worktree: `.worktrees/wg2-lossy-wire` (branch `fable/lossy-wire`), base master `b89a75c`.
Env: `maxwell`; `CUDA_VISIBLE_DEVICES=0`. Date: 2026-07-21.

## Delivered

1. **Passive lossy-current ADE companion** (`witwin/maxwell/fdtd/wire_lossy.py`, new).
   A finite round conductor's per-unit-length internal series impedance
   `Z'(w) = R_dc + Z_excess(w)` (compiler/wire_impedance.py, unchanged) folds into
   the segment current update as an energy-consistent trapezoidal (Tustin)
   companion:
   - leapfrog current stays at half steps; every loss term uses the integer-step
     trapezoidal current `I_bar = (I^{n+1/2}+I^{n-1/2})/2`;
   - explicit update `(L/dt + G/2) I^+ = (L/dt - G/2) I^- + (emf + V_tail - V_head - Cs x)`,
     `G = R0 + length*Dd`, ADE state `x^+ = Ad x + Bd I_bar`;
   - PEC limit `G=0`, no ADE, recovers the bitwise lossless leapfrog.
   Derivation and passivity discussion in the module docstring.

2. **Adaptive stability certificate.** The improper skin-effect impedance makes the
   shared rational fit carry a large out-of-band direct term, so the joint
   leapfrog/ADE realization is not positive-real at every order. The build fits the
   **highest** order in `[6, 13]` whose exact combined `[I; x]` transition spectral
   radius is `< 1 - 1e-6` on every sharing segment **and** whose in-band AC
   resistance is strictly positive (physical), else fails closed with a documented
   positive-real-realization message.

3. **Runtime consumption** (`witwin/maxwell/fdtd/wire.py`). `initialize_wire_runtime`
   builds the companion with the final CFL-adjusted dt and a band derived from the
   monitored frequencies (fail-closed with no frequency). `sample_and_update_wire`
   routes a network with any finite segment through a torch lossy step (leapfrog
   ordering preserved); pure-PEC scenes never reach it (CUDA path untouched).

4. **Real `ohmic_loss` monitor** (`finalize_wire_data`). Emits
   `0.5 Re(Z'(f)) length |I(f)|^2` per monitored segment/frequency (previously
   zeros). PEC segments report exactly zero. `Re(Z')` uses the fitted model's AC
   resistance clamped non-negative (physical).

5. **Compiler** (`witwin/maxwell/compiler/thin_wire.py`). Finite conductors compile
   (deferral guard removed) and carry `metadata["conductor"]` (kinds / conductivity /
   permeability per wire) + `metadata["has_finite_conductor"]`; topology and the
   PEC CUDA path are unchanged.

6. **Fail-closed** lossy reverse/adjoint replay (`wire.py::replay_wire_state`) and
   checkpoint/resume (`checkpoint.py`) until B3 (ADE-state transpose + checkpoint
   schema).

## Test inventory (env `maxwell`, GPU 0)

Commands (prefix each with the env exports from the brief):
```
conda run -n maxwell --no-capture-output python -m pytest \
  tests/fdtd/thin_wire/test_wire_lossy_recurrence.py \
  tests/fdtd/thin_wire/test_thin_wire_lossy_forward.py \
  tests/fdtd/thin_wire/test_wire_finite_conductivity.py \
  tests/fdtd/thin_wire/test_thin_wire_compiler.py \
  tests/fdtd/thin_wire/test_thin_wire_forward.py \
  tests/fdtd/thin_wire/test_thin_wire_reference.py \
  tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q
```

- `test_wire_lossy_recurrence.py` — **9 passed** (companion float64 oracles):
  - `test_analytic_ac_resistance_sweep` — Gate (a). Realized internal resistance
    (time-stepped ADE extraction) vs analytic scaled-Bessel across the band,
    pre-registered **< 8%** (compile-layer relative-max class; B1 fit noise).
  - `test_realized_matches_fitted_model` — recurrence realizes the fitted model to
    `< 2e-3` (recurrence correctness, fit-independent).
  - `test_energy_closure_single_tone` — Gate (b). Cycle-averaged input power ==
    companion dissipation (`rel 3e-3`); reported `0.5 Re(Z') len |I|^2` matches
    that dissipation (`rel 8%`).
  - `test_dc_resistance_exact` — Gate (c). `R0 == R_dc*length` to `rel 1e-12`.
  - `test_companion_is_passive_and_positive` / `test_unforced_non_growing_from_physical_state`
    / `test_long_run_stability_no_growth` — Gate (d). `spectral_radius < 1`,
    bounded 50k-step forced run, non-growing 60k-step unforced release.
  - `test_pec_conductor_builds_no_model`, `test_resolve_band_requires_frequencies`.
- `test_thin_wire_lossy_forward.py` — **7 passed** (end-to-end CUDA):
  - `test_pec_wire_bitwise_parity` — Gate (e). PEC scene: `lossy_model is None`,
    two runs byte-identical wire current.
  - `test_pec_parity_falsification`, `test_lossy_current_differs_from_pec` —
    finite conductor takes the lossy path and changes the answer.
  - `test_lossy_wire_runs_and_emits_ohmic_loss` — stable run, finite/positive
    ohmic_loss, `spectral_radius < 1`.
  - `test_pec_ohmic_loss_is_zero` — Gate (f)-adjacent: PEC dissipation exactly 0.
  - `test_lossy_reverse_replay_fails_closed`, `test_lossy_checkpoint_fails_closed`.
- `test_wire_finite_conductivity.py`, `test_thin_wire_compiler.py` — updated the two
  obsolete deferral tests to assert the new supported behavior (compile succeeds +
  conductor metadata). Full files green.
- `test_thin_wire_forward.py` (PEC), `test_thin_wire_reference.py`,
  `test_public_api.py`, `test_simulation_smoke.py` — green (no regression).
- `test_guard_census.py` — green at budget **176** (see below).

Robustness: the companion suite and the forward suite were each rerun 3-5x to
confirm the nondeterministic shared fitter (B1) stays within the pre-registered
tolerances.

## Falsifications (recorded; `scratch/falsify.py`, not committed)

- **F1 — ADE consumption is load-bearing (Gate a).** Zeroing the ADE history term
  `Cs x` (drop the excess-voltage feedback) blows the analytic-AC error at 1.5 GHz
  from `0.0225` to `1205` (>> 0.08) → RED. Restored → green.
- **F2 — DC exactness (Gate c).** Perturbing `R0` by +1% gives DC rel error `1e-2`
  (>> `1e-12`) → RED.
- **F3 — stability certificate (Gate d).** Accepting order 16 without the spectral
  certificate yields combined spectral radius `1.000075 > 1` (would grow over a
  long run) → RED; the adaptive build rejects it and selects a certified order.
- **PEC parity (Gate e)** is falsified in-suite: `test_pec_parity_falsification`
  and `test_lossy_current_differs_from_pec` show the finite path changes fields /
  current, proving the PEC parity is a real bypass, not a no-op.

## Guard-census reconciliation

`175 -> 176` (net +1). Removed: finite-conductor compile deferral (recurrence
implemented, "Material compilers" 12 -> 11). Added (both genuine capability gaps):
lossy reverse/adjoint replay (`wire.py::replay_wire_state`) and lossy
checkpoint/resume (`checkpoint.py`). The compiler kind check for malformed
conductors is a defensive `ValueError` (not counted). Doc + budget updated in the
same commit (`docs/reference/fdtd-capability-guard-census.md`,
`tests/api/public/test_guard_census.py`).

## Known gaps / deferred (G2b or beyond)

- **B3 conductivity adjoint** (reverse pass through the lossy recurrence, FD gates):
  deferred; lossy reverse replay fails closed.
- **Checkpoint/resume** of the ADE loss state: fails closed (schema is B3).
- **B4 distributed lossy reverse / multi-GPU lossy forward**: unchanged, fail-closed.
- **Gate (a) tolerance is 8%, not 2%** (pre-registered): the shared complex vector
  fit recovers `Re(Z')` only to ~1-6% and nondeterministically (blocker B1,
  recorded in `docs/memory/thin-wire-finite-conductivity-model-2026-07-17.md`), and
  the stability certificate caps the usable order. The analytic curve itself still
  meets the exact 2% gate at the compile layer; the recurrence tracks it within the
  fit envelope. Tightening needs a positive-real / real-part-protecting refinement
  of the shared fitter (out of this slice).
- **CUDA kernel not extended**: the lossy update is a torch path gated to lossy
  wires (brief-sanctioned alternative); the PEC CUDA kernel is untouched. No
  wall-clock timing claims (shared GPU).

## Commits

- `590aa74` feat(thin-wire): passive lossy-current ADE companion (B2 recurrence)
- `c1b5205` feat(thin-wire): consume lossy ADE recurrence in the FDTD runtime + real ohmic_loss

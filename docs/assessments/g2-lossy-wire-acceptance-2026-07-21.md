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

---

# G2b acceptance — conductivity adjoint (B3) + multi-GPU disposition

Stage: **G2b** (B3 conductivity adjoint + FD gates + multi-GPU disposition + docs/census/FEATURE_LIST).
Env `maxwell`, `CUDA_VISIBLE_DEVICES=0`. Date 2026-07-21. Builds on G2a (base as above).

## Design decision (binding)

The finite-conductor recurrence coefficients `(G, Cs, Ad, Bd, ade_output, R0)` are
derived from the shared rational vector fit, which is **nondeterministic (B1)** and
is not a differentiable/reproducible map of `sigma`. So the field-coupled current
sensitivity `dI/dsigma` through the multi-step recurrence cannot be certified by an
exact reverse replay and **stays fail closed** (per the brief's explicit
"fail closed on the affected configs rather than shipping a noisy gradient").

What **ships** is the *deterministic* conductivity adjoint of the dissipation
channel, computed in closed form from the exact scaled-Bessel internal impedance
(no fit involved):

    d Z'(omega)/d sigma = (i omega mu) / (4 pi sigma) . (1 - R^2),   R = I0(ma)/I1(ma), ma = sqrt(i omega mu sigma) . a

Derivation: differentiate `Z' = m/(2 pi a sigma) . I0(ma)/I1(ma)` with the Bessel
recurrences `I0'=I1`, `I1'=I0 - I1/z` and `d(ma)/d sigma = ma/(2 sigma)`; the
`z R'(z) - R` term telescopes to `z(1 - R^2)` and `m^2 = i omega mu sigma` cancels
the radius, leaving the one-line form above. Its DC limit is exactly
`d R_dc/d sigma = -1/(pi a^2 sigma^2)` (matches the closed-form DC resistance).

## Delivered

1. **Analytic conductivity gradient** (`compiler/wire_impedance.py`):
   `internal_impedance_conductivity_gradient(radius, sigma, mu, freqs)` -> complex
   `d Z'/d sigma`, `[F]` or `[F, S]`, reusing the same scaled-Bessel `ive` ratio.
2. **PyTorch-native autograd path** (`fdtd/wire_lossy.py`): a
   `torch.autograd.Function` and `analytic_ac_resistance(sigma_leaf, radius=, permeability=, frequencies=)`
   so a scalar conductivity leaf differentiates `Re(Z'(f; sigma))`; hence
   `0.5 . Re(Z') . length . |I|^2` has an exact conductivity gradient.
   `LossySegmentModel.conductivity_ac_resistance_gradient(freqs)` returns the
   per-segment `d Re(Z')/d sigma` from the built model.
3. **Sharpened fail-closed guard** (`fdtd/wire.py::replay_wire_state`): the message
   now names the conductivity-fit nondeterminism and points to
   `analytic_ac_resistance`. Checkpoint guard unchanged (ADE schema still B3+).
4. **Multi-GPU disposition** (`fdtd/distributed/solver.py::_validate_distributed_wire_support`):
   a finite-conductor wire on the distributed forward now fails closed. Verified
   defect it prevents: the distributed owner runtime (`fdtd/distributed/wire.py`)
   builds a `WireRuntime` with **no** `lossy_model`, so a lossy wire would silently
   run as PEC across shards. B4 distributed reverse stays fail closed.

## Test inventory (env `maxwell`, GPU 0)

- `tests/gradients/test_fdtd_thin_wire_conductivity_adjoint.py` — **7 passed** (new):
  closed-form `dZ'/dsigma` vs float64 central difference `< 1e-6` across a freq
  sweep and 3 (radius, sigma) cases; DC-limit `dR_dc/dsigma` exact; autograd of the
  AC resistance and of the full ohmic-loss objective vs central difference `< 1e-6`;
  the built model's per-segment gradient == the analytic closed form; scalar-input
  guard; PEC -> no lossy model.
- `tests/fdtd/thin_wire/test_thin_wire_lossy_forward.py`, `test_wire_lossy_recurrence.py`,
  `tests/fdtd/multi_gpu/test_wire_owner.py`, `tests/api/public/test_guard_census.py`,
  `test_public_api.py`, `test_simulation_smoke.py` — **58 passed, 3 skipped** (the
  3 skips are the two-GPU physical-parity tests; all reject/guard tests run on GPU 0).
  Includes the new `test_distributed_lossy_wire_is_rejected` and the sharpened
  `test_lossy_reverse_replay_fails_closed` (asserts the message names `conductivity`
  and `analytic_ac_resistance`).
- `tests/gradients/test_fdtd_thin_wire_adjoint.py`,
  `tests/fdtd/thin_wire/test_wire_finite_conductivity.py`, `test_thin_wire_compiler.py`,
  `test_thin_wire_forward.py`, `test_thin_wire_reference.py` — **119 passed** (PEC
  reverse path unaffected by the sharpened lossy guard; no regression).

Standalone closed-form cross-check (`scratch/verify_grad.py`, not committed):
`dZ'/dsigma` vs central difference `< 1e-6` over `a in {1,2} mm`,
`sigma in {5.8e7, 3.5e7, 1e6}`, `f in {1e3 .. 2e10}`; DC real part matches
`-1/(pi a^2 sigma^2)`.

## Falsifications (recorded; `scratch/`, not committed)

- **F1 — closed-form gradient (Gate a/DC).** Flip `(1 - R^2)` to `(1 + R^2)` in
  `internal_impedance_conductivity_gradient`: the central-difference gate and the
  DC-exactness gate go RED; restore -> GREEN.
- **F2 — autograd backward (Gate: autograd == FD).** Scale the backward gradient by
  2x: the AC-resistance-autograd and ohmic-loss-gradient gates go RED; restore ->
  GREEN.
- **F3 — multi-GPU lossy reject.** Disable the finite-conductor branch in
  `_validate_distributed_wire_support` (`and False`): `test_distributed_lossy_wire_is_rejected`
  goes RED (no raise); restore -> GREEN.

## Guard-census reconciliation

`176 -> 177` (net +1). No guard removed (the field-coupled `dI/dsigma` reverse and
the ADE checkpoint schema stay fail closed; the replay message is only sharpened).
Added (+1): the distributed lossy-wire forward reject
(`fdtd/distributed/solver.py`). Budget, `CONTRACT_GUARDS`-unrelated ledger comment,
and `docs/reference/fdtd-capability-guard-census.md` updated in the same change.

## Known gaps / deferred

- **Field-coupled conductivity adjoint `dI/dsigma`**: fail closed (nondeterministic
  shared fit -> non-differentiable, non-reproducible `sigma -> coeffs` map). A
  positive-real / reproducible fit refinement is the prerequisite.
- **ADE loss-state checkpoint/resume**: fail closed (schema unchanged).
- **B4 distributed lossy reverse**: fail closed. **Distributed lossy forward**: now
  fail closed (was a silent-PEC hole).
- Public `WireConductor.finite(conductivity=...)` takes a Python float, so there is
  no scene-level trainable-conductivity path yet; the shipped adjoint is the
  model-layer differentiable readout usable in a torch optimization loop.
- No wall-clock timing claims (shared GPU).

## Commits (G2b)

- see `git log` for the G2b commit hash(es) on `fable/lossy-wire`.

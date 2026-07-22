# Thin-wire Phase 4 — finite-conductivity model layer (2026-07-17)

Plan: `docs/plans/next-functional-2026-07/07-thin-wire-model.md` (sections 6.3, 9 Phase 4, 10.2).
Branch: `codex/wire-conductivity-reverse`. Worktree: `.worktrees/r4-wire`.

## Delivered (single-GPU, model + fit + analytic layer)

- `WireConductor.finite(conductivity, permeability=MU_0)` — public conductor law
  in `witwin/maxwell/thin_wire.py`, validated (positive real scalars; rejects
  bool/complex/nonpositive; `pec` rejects material params; unknown kinds
  rejected). `MU_0` and `_positive_real_scalar` added there.
- `witwin/maxwell/compiler/wire_impedance.py`:
  - `dc_resistance` — exact `1/(pi a^2 sigma)` (scalar or per-segment).
  - `internal_impedance` — analytic solid-round-wire skin-effect
    `Z'(omega) = m/(2 pi a sigma) . I0(ma)/I1(ma)`, `m = sqrt(j omega mu sigma)`,
    via `scipy.special.iv` (complex Bessel) in double, prepare-time only.
  - `surface_resistance`, `ohmic_loss_density` (`0.5 Re(Z')|I|^2`).
  - `SeriesImpedanceModel` + `fit_series_impedance` — passive rational ADE fit of
    the *excess* impedance `Z' - R_dc` (zero at DC). DC kept exact and applied
    separately. **Reuses the shared network rational stack** per the plan's
    no-duplicate-pole-fitting rule: `rational.fit_rational` (stable/passive vector
    fit), `RationalModel.to_state_space`, `StateSpaceNetwork.discretize`
    (bilinear/trapezoidal), and `RationalModel.check_passivity` (pole-aware
    positive-real certificate). No new pole fitter written.
- Exported the above from `witwin/maxwell/compiler/__init__.py`.
- Finite conductor compiled into an FDTD run now fails closed with a clear
  `NotImplementedError` pointing at the model API (`compiler/thin_wire.py`).
- Tests: `tests/fdtd/thin_wire/test_wire_finite_conductivity.py` (22): API
  validation, DC exact, analytic DC/HF-asymptote/monotone limits, scipy-reference
  match, ohmic-loss density, fit reuse/passivity/stability, fit magnitude
  accuracy, AC-tracks-analytic (interior), analytic 2% gate, per-segment/band
  rejection, compile deferral. Updated two existing tests for the new contract
  (`test_thin_wire_api.py`, `test_thin_wire_compiler.py`).
- Falsification recorded: 3% impedance perturbation reddens DC/scipy/2%-gate;
  20% fit-output perturbation reddens AC-tracks; restored green.

## Key numerical finding / blocker (B1) — AC-from-fit 2% is not a robust gate

The shared complex vector fitter recovers the AC resistance (`Re(Z')`) from the
passive fit to only ~1-2% at interior band points and is **nondeterministic run
to run** (threaded lstsq/eigvals last-bit variation) — a strict 2% automated gate
flips pass/fail (observed 2.48% then 1.4% on identical code/config, order 14).
Higher order (16-20) does not robustly cross under 2% and can over-condition.

Resolution taken: the **analytic** `internal_impedance` carries the exact 2%
analytic gate (it *is* the skin-effect curve). The fit is gated on what is robust
and deterministic: passivity certified, discrete pole radius < 1, magnitude
report `relative_rms <= 5%` / `relative_max <= 10%`, and AC-tracks-analytic at a
safe 5% interior margin. Design implication for the deferred recurrence: the
`ohmic_loss` monitor should report `0.5 Re(Z'_analytic(omega))|I|^2` (exact),
while the rational ADE only carries the time-domain broadband dynamics. Tightening
AC-from-simulation to a robust 2% needs a real-part-protecting / positive-real
weighting refinement on the shared fitter — out of this slice's scope.

Robust reference config: `order=16`, `band=(4e8, 3e9)`, `samples=240`,
`iterations=20`, relative weighting on; copper `sigma=5.8e7`, `a=5e-4`.

## Deferred with concrete blockers

- B2 — Lossy current recurrence (reference + CUDA) and `ohmic_loss` monitor
  output. Needs the energy-consistent leapfrog/state-space coupling schedule
  (plan 6.2) signed off, and a CUDA `update_wire_state` signature extension
  (series R + ADE state) that ripples into `extension.cpp`, the backend, and
  `fdtd/distributed/wire.py`. `wire.cu` incremental rebuild ~10s, so tractable
  once the discrete-ADE coupling is fixed. `finalize_wire_data` still emits
  `ohmic_loss` as zeros.
- B3 — Conductivity adjoint through the wire pullback (FD-gated) + checkpoint
  schema carrying ADE state (resume round-trip). Depends on B2.
- B4 — Part B: multi-GPU wire reverse parity. Give the Phase 7 bridge a
  wire-state channel (owner I/q + ADE checkpoint/replay; transposed EMF-gather /
  deposit reusing S1 transposed transports); 1-vs-2-GPU gradient parity on a
  radius/conductivity-trainable wire across the split. Depends on B2/B3.

## Guards

Added guards are all narrow validations of verified behavior: `WireConductor`
kind/param validation, `_positive_real_scalar`, `wire_impedance` input checks
(finite/positive radius, band ordering, per-segment-radius rejection,
non-finite impedance), and the finite-conductor FDTD-compile `NotImplementedError`
(a fail-closed deferral, tested). No speculative guards added.

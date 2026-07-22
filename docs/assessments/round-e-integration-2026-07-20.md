# Round-E integration acceptance — waveguide wave-level PASS + external reference

> Date: 2026-07-20 (integration on main `master`, HEAD after the round-E merges)
> Track: integration dev (main checkout)
> Depends on: E1a/E1b (`docs/assessments/e1-rf-modes-acceptance-2026-07-19.md`,
> `docs/assessments/e1-rf-mode-operator-acceptance-2026-07-19.md`) — the Yee-staggered
> transverse full-vector operator and its selector wiring; round-E monitor passthrough
> and the M3 external-reference generation wiring.

## 0. Scope

Un-BLOCK `rf/rectangular_waveguide` end to end now that the transverse mode operator
is fixed: turn the benchmark's BLOCKED record into a committed wave-level PASS, add a
committed wave-validation test, and generate + compare an external-reference-solver
cross-check for the waveguide scene. This closes the two round-4 open items that were
this scene's blockers (transverse operator; PlaneMonitor passthrough) and the
external-reference authorization item for the covered scenes.

## 1. Delivered

1. **`benchmark/rf_validation.py::run_rectangular_waveguide` rewritten to a committed
   PASS** on the coax_thru wave-level pattern: S from `B = S*A`, gated on extraction
   conditioning `cond(A)` + post-solve passivity, then `beta(omega)` from `arg(S21)/L`
   vs the analytic TE10 dispersion `beta = sqrt(k0^2 - (pi/a)^2)`. The stale
   NRW-de-embedding path and the `a_passive`-ratio validity gate were removed (dead
   `_nrw_beta` / `A_PASSIVE_RATIO_LIMIT` deleted; `a_passive` kept as a diagnostic). The
   mode-shape `sin(pi y/a)`-correlation check is retained as a fail-closed regression
   guard (< 0.9 -> BLOCKED). External-reference cross-check wired in when the cache is
   present.
2. **`tests/rf/wave_validation/test_waveguide_wave_level.py`** (new committed gate):
   terminated TE10 two-port is conditioned + passive + beta within 1%; plus a
   length-sensitivity falsification test.
3. **`benchmark/scenes/rf/rectangular_waveguide.py`**: added `sweep_frequencies()` and
   `rectangular_waveguide_reference_scene(...)` — a TE10 `ModeSource`-driven guide with
   two `ModeMonitor` planes for the external-reference cross-check.
4. **`benchmark/rf_tidy3d_references.py`**: registered `rf/rectangular_waveguide` as a
   real (runnable, `sources=1`) reference target; ran one authorized cloud job.
5. Docs: `docs/reference/rf-wave-validation-2026-07-18.md` §1 table / §1.2 / §4 / §5
   updated (BLOCKED -> PASS; open items resolved); `benchmark/RESULTS.md` regenerated.

## 2. Waveguide wave-level gate values (executed)

Command: `python -m benchmark rf rf/rectangular_waveguide` (and
`tests/rf/wave_validation/test_waveguide_wave_level.py`). Band 1.2 fc .. 2.2 fc,
11 frequencies, `fc = c/(2a) = 249.83 MHz`, port separation `L = 0.60 m`.

| dx | sin-corr | cond(A) | max sv (bandmax) | beta median rel err | a_passive (diag) |
|---|---|---|---|---|---|
| 0.05 | 1.0000 | 1.270 | 1.0010 | 0.41% | 0.244 |
| 0.025 | 1.0000 | 1.117 | 1.0008 | 0.07% | 0.112 |
| 0.02 | 1.0000 | 1.090 | 1.0007 | **0.05%** | 0.088 |

Finest tier (dx=0.02): `|S11|` in `[0.0002, 0.0083]`, `|S21| ~ 1.0`, reciprocity ~1e-4.
Pre-registered gate: `cond(A) <= 10`, max singular value `<= 1.05`, beta median rel
error `<= 1%` (1%-class from the coax_thru precedent, which gates its own `arg(S21)/L`
beta at 3%). The independent Yee numerical-dispersion floor at dx=0.02 is ~0.03%. All
three tiers pass; status = **pass**, gate class **wave-level**.

## 3. Falsification (executed)

* **Mode corruption / mode-index slip (EXECUTED, `docs/assessments/round-e-integration-probes/falsify_waveguide.py`).** The
  benchmark keeps a fail-closed guard: the injected TE10 `sin(pi y/a)`-correlation must be
  `>= 0.9`. (1) Forcing the runner's correlation to a legacy-checkerboard-class `0.55`
  flips `run_rectangular_waveguide()` to `status='blocked'` (observed) -- no spurious
  S-matrix reported. (2) On the REAL Yee-staggered operator the fundamental TE10 correlates
  `1.0000` with `sin(pi y/a)` while the NEXT transverse eigenmode (TE20-class, exactly what
  a `mode_index=1` slip would inject) correlates `0.0000` -- so a mode-index slip is caught
  by the guard. (3) Restoring the runner gives `status='pass'`. The operator-level golden
  gates (`tests/rf/wave_validation/test_transverse_operator.py`,
  `test_te10_mode_selection.py`) pin the clean spectrum.
* **Length sensitivity (committed test).**
  `test_waveguide_beta_gate_is_falsifiable_in_length`: with the true `L` the beta median
  rel error is `<= 1%`; perturbing `L` by +10% drives it `> 5%`, busting the gate -- the
  gate is not vacuously satisfiable. Executed green.
* **Load discrimination.** `test_matched_s11_wave_level.py` (green): a matched TE10
  termination reflects far less than a PEC short (`shorted |S11| > 0.85`,
  `> 2x` the matched value).

## 4. External-reference-solver cross-check (one authorized cloud run)

* **Authorization.** The external-solver (cloud) spend is owner-authorized for this
  program round: the owner's instruction for this task explicitly allows external-reference
  validation use ("auth is live; ONE cloud run, smallest honest grid"). This is not a
  self-authorized spend — it is executed under that owner instruction, which supersedes the
  round-4 "deferred pending owner cost authorization" note in
  `docs/reference/rf-wave-validation-2026-07-18.md` §4.
* Path: `python -m benchmark.rf_tidy3d_references` (M3 wiring). The waveguide reference
  is a TE10 `ModeSource`-driven guide (adapter maps `ModeSource`/`ModeMonitor` to the
  reference solver's native modal source/monitor), so it exports with `sources=1` and is
  genuinely runnable. The four port/lumped scenes stay `pending-generation` at the
  `sources=0` runnable gate (no credits spent).
* **Cloud task id: `fdve-3c2a2d95-4809-4dfb-98d6-1b6b5416c39a`**
* **Cost: 0.025 FlexCredits** (estimate == actual; budget ceiling 2.0). Smallest honest
  grid: exported at dx=0.05 (the reference solver auto-meshes).
* Cache: `benchmark/cache/rf/rectangular_waveguide.h5` (gitignored, per benchmark
  convention); marker `benchmark/cache/rf/rf__rectangular_waveguide.generated.json`.
* **Comparison (Maxwell vs reference).** The reference forward-mode-amplitude phase
  constant `beta_ref = |d arg(amp_fwd)|/L_ref` (normalization-independent) vs the analytic
  TE10 dispersion over the same 11 frequencies: **median 1.21%, max 2.74%**. `|S21|_ref`
  in `[0.956, 1.095]` (near-cutoff edge slightly above 1). This is an independent-solver
  confirmation of the same `beta(omega)` the FDTD two-port measures to 0.05%; the analytic
  dispersion remains the binding first-line reference. RESULTS.md M3 row added via the
  existing updater (`## RF / antenna external reference generation`).

## 5. Tests run

Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=0`, `CUDA_HOME` exported.

* `tests/rf/wave_validation/test_waveguide_wave_level.py` — new (see §6 counts).
* `tests/rf/wave_validation/` (full dir).
* `tests/rf/wave_validation/test_te10_mode_selection.py`,
  `test_transverse_operator.py` — operator golden gates (regression falsification base).
* `tests/api/public/test_guard_census.py` — capability-guard budget unchanged (no guards
  added/removed; no numeric masking path introduced).
* `tests/api/public/test_public_api.py`, `tests/api/public/test_simulation_smoke.py`.

## 6. Known gaps / deferred

* microstrip / differential_pair remain BLOCKED on interior-PEC (trace) masking of the
  staggered operator for production substrate+air scenes. Operator-level hybrid physics is
  validated (E1 half-filled-guide LSE mode, machine-precision vs the 1D SL reference), but
  the production compiler wiring + contour-snap fix are outstanding.
* Adapter port/lumped source mapping (to make coax / lumped / antenna scenes
  cloud-runnable) is a deferred adapter feature.
* RLC resonance wave-level gate is a strict xfail open gap.

## 7. Files changed

* `benchmark/rf_validation.py` (waveguide runner; dead-code removal; reference hook; RF
  intro).
* `benchmark/scenes/rf/rectangular_waveguide.py` (reference scene + sweep band).
* `benchmark/rf_tidy3d_references.py` (waveguide reference target; docstring/RESULTS
  intro).
* `tests/rf/wave_validation/test_waveguide_wave_level.py` (new committed gate).
* `tests/rf/wave_validation/test_rf_reference_generation.py` (split source-less vs
  runnable targets; added the runnable-waveguide export gate test).
* `docs/assessments/round-e-integration-probes/falsify_waveguide.py` (falsification probe).
* `docs/reference/rf-wave-validation-2026-07-18.md`, `benchmark/RESULTS.md`,
  `FEATURE_LIST.md`, this acceptance doc.

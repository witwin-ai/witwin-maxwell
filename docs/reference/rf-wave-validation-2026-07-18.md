# RF port wave-level validation (audit step S1)

> Date: 2026-07-18 (revised after the double-REJECT adversarial audit)
> Implements: `docs/assessments/next-functional-audit-2026-07-18.md` step S1
> Binding taxonomy: `docs/reference/gate-classification.md`
> Scope: RF port wave-level validation (P0 -- the single biggest plan-01 debt)

This is the maintained reference for the RF validation delivered under audit step
S1. It records the scenes, the binding analytic references, the gate taxonomy
re-labelling, and -- per the honest-exit mandate -- the gaps that remain, with
measured values rather than API-existence. Exact per-scene numbers live in the
machine-readable artifacts under
`docs/assessments/rf-wave-validation-2026-07-18/` and in
`benchmark/RESULTS.md`; this document does not restate drifting figures.

## 0. What changed and why

The first S1 round was rejected because it re-badged the 2D **mode eigensolve**
as "wave-level" evidence and shipped false "pass" claims. In particular the
retired round claimed coax "Z0 ~1.1%, beta exact (0.0%)" and waveguide
"beta/Z ~0.08%" -- but those numbers came from
`resolve_waveport_run_manifest(...)` (the modal solver), not from a time-stepped
FDTD S-matrix, and the coax "beta = k0 sqrt(eps mu)" is assigned *by construction*
in the TEM path (`witwin/maxwell/fdtd/excitation/modes.py:1879-1884`) -- an
`analytic-identity`, forbidden as gate evidence by the taxonomy.

This round makes the binding metric a genuine FDTD `Scene -> Simulation -> Result`
measurement wherever the two-port bench yields a usable S-matrix, and records the
honest outcome (including FAIL / BLOCKED) everywhere else. The modal Z0/beta are
kept only as `modal-eigensolve` supporting rows and never gate.

## 1. Scenes and honest status (S1.1)

Six scenes live under `benchmark/scenes/rf/` and are driven through
`python -m benchmark rf`. This is **not** six passing wave-level scenes:

| Scene | Binding metric | Class | Status |
|---|---|---|---|
| `rf/rectangular_waveguide` | FDTD S-matrix -> NRW-de-embedded `beta(omega)` vs analytic TE10 dispersion; passivity/reciprocity convergence | `wave-level` | see artifact (pass/gap vs the Yee floor) |
| `rf/coax_thru` | real FDTD two-port S-matrix | `wave-level` | **FAIL** (|S11| ~ 1, gross non-passivity; TEM WavePort does not match the round line) |
| `rf/microstrip_two_port` | -- | `wave-level` | **BLOCKED** (TEM categorically inapplicable to substrate+air) |
| `rf/differential_pair` | -- | `wave-level` | **BLOCKED** (same TEM inapplicability, coupled 4-port) |
| `rf/series_parallel_rlc` | FDTD load-port resonance peak vs analytic f0 | `wave-level` | **gap** (parasitic-dominated; peak does not track C) |
| `rf/lumped_open_short_match` | FDTD feed |S11| vs analytic Gamma | `wave-level` | **FAIL** (feed decoupled from load; identical Gamma for all loads) |

### 1.1 Rectangular waveguide (the one genuine wave-level FDTD scene)

- A real two-port `PortSweep` FDTD run is executed at three grid-commensurate
  tiers (dx in {0.05, 0.025, 0.02}, each dividing 0.1 so the a/b aperture edges
  land on Yee nodes).
- `beta(omega)` is extracted from the FDTD S-matrix by **Nicolson-Ross-Weir
  de-embedding** (symmetrized S11/S21), which removes the port-mismatch
  standing-wave ripple that contaminates the raw `arg(S21)/L`. Raw phase gives
  ~2% band-averaged ripple; NRW recovers the intrinsic beta.
- **Tolerance basis (not tuned-to-pass):** the 3D Yee numerical-dispersion floor
  at the run's dx and Courant dt, `|beta_numeric - beta_continuous|/beta` over
  the band, computed from the discrete dispersion relation (transverse
  second-difference eigenvalue `(2/dx^2)(1-cos(pi dx/a))`). The artifact records
  the floor and the measured median rel error; status is `pass` iff the measured
  value is within the floor, else `gap` with the residual attributed to port
  mismatch.
- **Conservation (wave-level):** max singular value and reciprocity of the real
  S-matrix are reported per tier and **converge toward the physical limits (1 and
  0) under refinement** -- this is the wave-level conservation evidence.
- **Supporting only:** a `modal-eigensolve` beta/Z_TE cross-check is recorded but
  never gates.

## 2. Gate taxonomy re-labelling (S1.2)

The S0.3 taxonomy uses five verbatim classes:
`analytic-identity | tautology | symmetric | postprocess-only | wave-level`,
plus `modal-eigensolve` for supporting rows. Status (`pass | gap | fail |
blocked | pending`) is a separate axis and is never folded into the class.

The retired plan-01 Phase-1/2 headline gates are re-classified and **lose
exit-gate status**:

| Retired plan-01 gate | Old (claimed) | New class | New wave-level gate |
|---|---|---|---|
| "matched load \|S11\| < -30 dB" (single implicit update, `V=Z0 I` at 1e-12) | E3 | `analytic-identity` | `tests/rf/wave_validation/test_matched_s11_wave_level.py` (propagating waveguide matched vs shorted, from fields) |
| "series/parallel RLC resonance < 2%" (trapezoidal-formula sweep, solver not run) | E3 | `analytic-identity` | RLC resonance recorded as an **open gap** (section 3) |
| "coax/microstrip reciprocity < 0.02" (mirror-symmetric fixture) | E3 | `symmetric` | `tests/rf/wave_validation/test_asymmetric_reciprocity_power_balance.py` (physically asymmetric ports, `S12==S21` is physics) |
| "power imbalance < 2%" (hand-written unitary matrix `assert`) | E3 | `tautology` | field-derived power balance in the same asymmetric-reciprocity test |
| coax "beta exact / beta measured 0.0%" | E3 | `analytic-identity` | **removed** -- beta = k0 sqrt(eps mu) is assigned by construction (modes.py:1879-1884), never a gate |
| coax/waveguide modal `Z0`/`beta` "pass" | E3 | `modal-eigensolve` | supporting rows only; FDTD S-matrix is the binding metric |

Every new wave-level gate is falsification-checked (perturb -> red -> restore);
the records are in the test docstrings and the scene artifacts.

## 3. Honest gaps and defects (measured, not papered over)

* **Coax two-port is a wave-level FAIL.** A real FDTD sweep reflects almost all
  incident power (|S11| ~ 1) with a max singular value well above 1: the TEM
  WavePort does not launch/absorb a matched TEM wave on the round coax at
  benchmark resolution, and the mirror-symmetric geometry makes reciprocity
  trivial. The half-grid contour snapping is now deterministic (B5,
  `benchmark/scenes/rf/coax_thru.py:snap_contour_half`) so tiers build, but that
  does not fix the matching. Open work: an impedance-matched coax feed.
* **microstrip / differential_pair are BLOCKED for the correct reason.**
  `WaveModeSpec('tem')` is categorically inapplicable to their inhomogeneous
  (substrate + air) cross-sections: the TEM electrostatic normalization requires
  a uniformly filled cross-section and raises `NotImplementedError`
  (`modes.py:1846-1849`). A hybrid full-vector mode solve is required. The
  earlier "snapping" attribution was wrong; a secondary contour-snapping error
  also exists but is not the primary blocker. `reference: pending-generation`.
* **Wave-level RLC resonance is an open gap.** The lumped two-port bench is
  parasitic-dominated: the load-port current peak barely tracks the circuit `C`
  (the C(1pF)->C(2pF) peak ratio is far from the ideal sqrt(2); exact numbers in
  the artifact). Analytic `f0` binds; the propagating-structure RLC gate is
  outstanding and encoded as a **strict** xfail
  (`tests/rf/wave_validation/test_rlc_resonance_wave_level.py`, `strict=True` so a
  silent xpass cannot close the gap).
* **lumped_open_short_match is a broken bench (FAIL), not a floor.** matched,
  short and open all read |Gamma| ~ 0.997 at the *same* phase: the feed sees
  near-total reflection independent of the load, i.e. the feed port is not
  coupled to the load. Two lumped ports two cells apart in a tiny PML box radiate
  into the boundary rather than forming a transmission path. A propagating feed
  line terminated by the load is required.

## 4. Reference-solver policy

Analytic transmission-line / waveguide solutions are the binding first-line
reference (audit section 3). External reference-solver cross-references are the
future primary cross-check for the covered port families; adapter-driven
generation is **not yet wired** (M3), so `python -m benchmark.rf_tidy3d_references`
only stamps `reference: pending-generation` markers under `benchmark/cache/rf/`
and never fabricates a numerical comparison. `series_parallel_rlc` is a
lumped-circuit resonance with an analytic-only reference.

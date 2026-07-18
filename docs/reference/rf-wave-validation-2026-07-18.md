# RF port wave-level validation (audit step S1)

> Date: 2026-07-18
> Implements: `docs/assessments/next-functional-audit-2026-07-18.md` step S1
> Scope: RF port wave-level validation (P0 -- the single biggest plan-01 debt)
> Evidence target: E2 (wave-level); honest-exit where the bench falls short

This document is the maintained reference for the RF wave-level validation
delivered under audit step S1. It records the scenes, the binding analytic
references, the gate taxonomy re-labelling, and -- per the audit's honest-exit
mandate -- the gaps that remain, with measured values rather than API-existence.

## 1. Scenes (S1.1)

Six scenes live under `benchmark/scenes/rf/` and are driven by real FDTD through
`python -m benchmark rf`:

| Scene | Realisation | Analytic reference | Status |
|---|---|---|---|
| `rf/coax_thru` | round coax, two TEM `WavePort`s | `Z0 = eta0/(2pi) ln(b/a)`, `beta = k0` | **pass** (Z0 ~1.1%, beta exact) |
| `rf/rectangular_waveguide` | hollow guide, two TE10 `WavePort`s | `fc = c/2a`, `beta = sqrt(k0^2-(pi/a)^2)`, `Z = eta0 k0/beta` | **pass** (beta/Z ~0.08%) |
| `rf/microstrip_two_port` | grounded substrate + strip, quasi-TEM `WavePort`s | Hammerstad-Jensen `Z0`/`eps_eff` | gap (modal contour snapping; analytic recorded) |
| `rf/series_parallel_rlc` | lumped feed + RLC-terminated port | `f0 = 1/(2pi sqrt(LC))`, `Q` | open gap (parasitic-dominated bench; see section 3) |
| `rf/lumped_open_short_match` | lumped feed + resistively terminated port | `Gamma = (R-Z0)/(R+Z0)` | gap (near-field coupling; measured floor recorded) |
| `rf/differential_pair` | coupled microstrip, four quasi-TEM ports | coupled-line even/odd mixed-mode | pending-generation (4-port mixed-mode) |

Each scene writes a machine-readable artifact (grid convergence + conservation /
passivity) under `docs/assessments/rf-wave-validation-2026-07-18/` and a row in
the `## RF wave-level validation` table of `benchmark/RESULTS.md`.

The two scenes with **exact** analytic references and proven WavePort mode-solve
paths (coax, rectangular waveguide) pass their plan-01 section-10 targets and
carry a grid study. The remaining scenes record measured-with-gap or
pending-generation honestly.

## 2. Gate taxonomy re-labelling (S1.2)

The S0.3 taxonomy uses five verbatim classes:
`analytic-identity | tautology | symmetric | postprocess-only | wave-level`.

The retired plan-01 Phase-1/2 headline gates are re-classified and **lose
exit-gate status**:

| Retired plan-01 gate | Old (claimed) | New class | New wave-level gate |
|---|---|---|---|
| "matched load \|S11\| < -30 dB" (single implicit update, `V=Z0 I` at 1e-12) | E3 | `analytic-identity` | `tests/rf/wave_validation/test_matched_s11_wave_level.py` (propagating waveguide matched vs shorted, from fields) |
| "series/parallel RLC resonance < 2%" (trapezoidal-formula sweep, solver not run) | E3 | `analytic-identity` | RLC resonance recorded as an **open gap** (section 3) |
| "coax/microstrip reciprocity < 0.02" (mirror-symmetric fixture) | E3 | `symmetric` | `tests/rf/wave_validation/test_asymmetric_reciprocity_power_balance.py` (asymmetric ports, `S12==S21` is physics) |
| "power imbalance < 2%" (hand-written unitary matrix `assert`) | E3 | `tautology` | field-derived power balance in the same asymmetric-reciprocity test |

The old formula/identity checks may remain as **fast contract tests** (e.g.
`test_series_rlc_companion_impedance_matches_analytic` in the wave_validation
suite is explicitly `analytic-identity` and non-gating), but they are no longer
the exit evidence. Every new wave-level gate is falsification-checked: detuning
the matched load to a short drives `|S11|` red, and injecting non-reciprocity /
gain into the measured S matrix drives the reciprocity / power-balance gates red.

## 3. Honest gaps

* **Wave-level RLC resonance is an open gap.** Three benches were attempted this
  session (details in `test_rlc_resonance_wave_level.py`): the lumped two-port
  bench is parasitic-dominated (the load-port current peak barely moves,
  7.90 GHz at C=1pF vs 7.74 GHz at C=2pF where the ideal ratio is `sqrt(2)`), the
  circuit-bound port imposes `V/I` (tautological), and a parallel-plate line did
  not guide a clean TEM wave at benchmark resolution. The analytic `f0` binds as
  the first-line reference; a propagating transmission-structure RLC gate is the
  outstanding work. This is recorded, not papered over with the coincidental
  parasitic peak.
* **Matched \|S11\| floor.** The propagating waveguide matched load reflects at
  ~-12 dB on the coarse grid, above the -30 dB plan-01 target; the gap is
  recorded. What binds is the wave-level, load-discriminating behaviour (matched
  << short), which the falsification confirms.
* **microstrip / differential_pair.** Quasi-TEM and coupled-line mixed-mode
  extraction did not fully resolve (contour half-grid snapping / 4-port coupled
  apertures). Scenes, analytic references, and the Tidy3D generation path are
  registered; status is gap / pending-generation.
* **Grid convergence.** Round-coax and TE10 apertures snap to the Yee half-grid
  only at specific float32-safe resolutions, and the TE10 tracker occasionally
  locks onto the free-space (k0) branch; the artifacts record the resolved tiers
  and flag the fallback tiers rather than reporting them as convergence data.

## 4. Reference-solver policy

Analytic transmission-line / waveguide solutions are the binding first-line
reference (audit section 3). Tidy3D cross-references for the covered port families
are generated through `python -m benchmark.rf_tidy3d_references`; offline it
stamps `reference: pending-generation` markers under `benchmark/cache/rf/` and the
analytic gate is not relaxed. `series_parallel_rlc` is a lumped-circuit resonance
with no Tidy3D cross-reference (analytic only).

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
in the TEM path (`witwin/maxwell/fdtd/excitation/modes.py:1978-1982`) -- an
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
| `rf/rectangular_waveguide` | terminated FDTD two-port; NRW-de-embedded `beta(omega)` vs analytic TE10 dispersion, gated on the `a_passive/a_driven` precondition | `wave-level` | see artifact (pass/gap vs the Yee floor) |
| `rf/coax_thru` | terminated FDTD two-port; clean TEM launch but `a_passive/a_driven ~ 1` (far-end re-entry) | `wave-level` | **gap** (S=b/a premise violated; thin-PML-vs-wavelength, NOT a port/line mismatch) |
| `rf/microstrip_two_port` | -- | `wave-level` | **BLOCKED** (TEM categorically inapplicable to substrate+air) |
| `rf/differential_pair` | -- | `wave-level` | **BLOCKED** (same TEM inapplicability, coupled 4-port) |
| `rf/series_parallel_rlc` | FDTD load-port resonance peak vs analytic f0 | `wave-level` | **gap** (parasitic-dominated; peak does not track C) |
| `rf/lumped_open_short_match` | FDTD feed |S11| vs analytic Gamma | `wave-level` | **FAIL** (feed decoupled from load; identical Gamma for all loads) |

### 1.1 Rectangular waveguide (the one genuine wave-level FDTD scene)

- **Terminated bench:** the PEC walls run the full x-domain, through the PML to
  the boundaries, and the **PML physical thickness is held fixed in metres across
  tiers** (`num_layers` scales with 1/dx: 0.2 m -> 4 / 8 / 10 layers at
  dx = 0.05 / 0.025 / 0.02). The first round left the walls short of the boundary
  with a fixed layer count, so the PML thickness shrank with dx and the open guide
  end drifted out of the absorber -- a dx-dependent standing-wave ripple that
  masqueraded as a refinement trend. That defect is fixed.
- A real two-port `PortSweep` FDTD run is executed at the three grid-commensurate
  tiers (dx in {0.05, 0.025, 0.02}, each dividing 0.1 so the a/b aperture edges
  land on Yee nodes).
- **S-extraction precondition (F2):** every S-derived quantity is gated on
  `a_passive/a_driven` (the `S = b/a` extraction assumes the passive port carries
  no incident wave). The measured ratio is recorded per tier; the exact numbers
  live in the artifact.
- `beta(omega)` is extracted from the FDTD S-matrix by **Nicolson-Ross-Weir
  de-embedding** (symmetrized S11/S21), which de-embeds the interface reflection
  that contaminates the raw `arg(S21)/L`. The effective reference-plane separation
  is re-fit from the `arg(S21)` phase slope and reconciled against the nominal
  port separation (exact value in the artifact).
- **Tolerance basis (not tuned-to-pass):** the 3D Yee numerical-dispersion floor
  at the run's dx and **the dt the runtime actually selects** (`min(period/30,
  Courant)`, F7b), `|beta_numeric - beta_continuous|/beta` over the band, computed
  from the discrete dispersion relation (transverse second-difference eigenvalue
  `(2/dx^2)(1-cos(pi dx/a))`). The artifact records the floor and the measured
  median rel error; status is `pass` iff the precondition holds and the measured
  value is within the floor, else `gap` with the measured residual disclosed.
- **Conservation (wave-level):** the `a_passive/a_driven` precondition, max
  singular value and reciprocity of the real S-matrix are reported per tier; the
  per-tier numbers and their behaviour under refinement are stated in the artifact
  and `benchmark/RESULTS.md`, not paraphrased here.
- **Supporting only:** a `modal-eigensolve` beta cross-check is recorded but never
  gates. (The guided TE10 mode selection was itself fixed this round -- see the
  spurious near-`k0` rejection, F5, `modes.py`.)

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
| coax "beta exact / beta measured 0.0%" | E3 | `analytic-identity` | **removed** -- beta = k0 sqrt(eps mu) is assigned by construction (modes.py:1978-1982), never a gate |
| coax/waveguide modal `Z0`/`beta` "pass" | E3 | `modal-eigensolve` | supporting rows only; FDTD S-matrix is the binding metric |

Every new wave-level gate is falsification-checked (perturb -> red -> restore);
the records are in the test docstrings and the scene artifacts.

## 3. Honest gaps and defects (measured, not papered over)

* **Coax two-port is a wave-level gap (terminated, clean launch).** The false
  "TEM WavePort does not match the round line / redesign the feed" attribution is
  **withdrawn**. The bench is now terminated -- the inner rod and outer shield run
  the full x-domain through the PML to the boundaries -- and the TEM launch is
  clean: `|arg(S21)|` tracks `k0*L` to a few percent (executed; exact number in
  the artifact). What still fails is the `S = b/a` extraction premise: the
  launched TEM wave reflects off the far termination and re-enters the passive
  port, so `a_passive/a_driven ~ 1` (the fully re-entrant limit). Root cause: the
  coax TEM wavelength (~0.3 m at 1 GHz) is large versus the PML that fits its
  compact ~0.4 m transverse cross-section -- `BoundarySpec.num_layers` is uniform
  across axes, so a thick enough x-PML would swallow the transverse cross-section
  (or force an intractable grid). Increasing PML layers 12 -> 20 and lengthening
  the run leave the standing wave unchanged (executed), confirming the far-end
  reflection is the limit, not launch matching. The half-grid contour snap is
  deterministic and its snap distance is persisted per tier
  (`benchmark/scenes/rf/coax_thru.py:snap_contour_half`). Per the F2 precondition
  no S-derived coax quantity is reported as a valid wave measurement; recorded as
  a gap with the measured residual. Open work: per-axis PML layer counts (thick x,
  thin transverse) or a lower single-mode band.
* **microstrip / differential_pair are BLOCKED, with the blockers in firing
  order.** The current-contour plane does not land on the Yee half-grid, so a
  contour-snap `ValueError` (`witwin/maxwell/compiler/waveports.py`) fires **first**
  and masks the mode solve. Underneath it, `WaveModeSpec('tem')` is categorically
  inapplicable to their inhomogeneous (substrate + air) cross-sections: the TEM
  electrostatic normalization requires a uniformly filled cross-section and raises
  `NotImplementedError` (`modes.py:1944-1946`). A hybrid full-vector mode solve is
  required. `reference: pending-generation`.
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

External reference-solver generation (S1.1) is **deferred pending owner
authorization of external-solver (cloud) runs**: this is a cost decision, so no
cloud generation was run this round and the `pending-generation` markers are kept
deliberately. Recorded as an S1.1 re-scope question for the owner.

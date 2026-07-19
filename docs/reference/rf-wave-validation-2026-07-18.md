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
| `rf/coax_thru` | terminated FDTD two-port; `beta` from `arg(S21)/L` vs `k0`, S via `B=S*A`, gated on extraction conditioning + passivity | `wave-level` | **pass** (a_passive/a_driven 0.17, \|S11\|<0.02, \|S21\|~1, max sv ~1, cond(A) ~1.2) |
| `rf/rectangular_waveguide` | terminated FDTD two-port; TE10 `beta(omega)` | `modal-eigensolve` | **BLOCKED** on the transverse mode-operator redesign (see 1.1 / open items) |
| `rf/microstrip_two_port` | -- | `wave-level` | **BLOCKED** (TEM categorically inapplicable to substrate+air) |
| `rf/differential_pair` | -- | `wave-level` | **BLOCKED** (same TEM inapplicability, coupled 4-port) |
| `rf/series_parallel_rlc` | FDTD load-port resonance peak vs analytic f0 | `wave-level` | **gap** (parasitic-dominated; peak does not track C) |
| `rf/lumped_open_short_match` | FDTD feed |S11| vs analytic Gamma | `wave-level` | **FAIL** (feed decoupled from load; identical Gamma for all loads) |

### PML-padding semantics (so "through the PML" claims are checkable)

The FDTD grid appends the absorbing-boundary nodes **outside** the declared domain
bounds: `scene._build_axis_grid64` extends each declared `[-DOMAIN, DOMAIN]` by
`num_layers * dx` on each side before building the primal/dual spacings. The
prepared solver grid therefore spans `+-(DOMAIN + num_layers*dx)`, and a structure
that reaches only `+-DOMAIN` ends **at the PML interface**, not through it. To
terminate a line in the absorber a conductor/wall must span `>= 2*(DOMAIN +
num_layers*dx)` (plus a margin). Any "the conductor runs through the PML" claim
must be verified against the **prepared** PEC occupancy along the axis (the grid
edges), never the scene-file length constant -- the round-2/3 benches set
`2*DOMAIN` and were NOT terminated despite the docstring saying so.

### 1.1 Coax two-port thru (the genuine wave-level PASS)

- **Terminated bench (round-4 root cause):** the inner rod and outer shield now
  span `2*(DOMAIN_X + num_layers*dx)` plus a margin, so they run **through** the
  computational PML to the padded grid edges (verified against the prepared PEC
  occupancy: at dx=0.01 the grid spans `+-0.18` and the rod PEC spans `+-0.18`).
  Rounds 2/3 set `2*DOMAIN_X`, ending the line at the PML interface (`+-0.12`) in
  an open stub; the launched TEM wave reflected off it and re-entered the passive
  port. Extending through the PML is by itself sufficient: `a_passive/a_driven`
  collapses `1.17 -> 0.17` (executed). The `1.17` re-entrant figure is the round-3
  config at dx=0.005; the round-4 shortened-conductor counterfactual measures a
  higher `a_passive/a_driven` bandmax `~1.478` at dx=0.01 -- the two counterfactual
  figures are labelled by config/tier so they are not read as contradictory.
- **S extraction (F3):** the network S is assembled by solving `B = S*A` across
  the drive columns, the correct extraction whenever the passive port carries any
  incident wave; the per-drive `b/a` ratio is a special case (A diagonal). cond(A)
  of the incident matrix is recorded per frequency (~1.2 -- near-orthonormal
  drives).
- **Gate (F5):** the wave-level precondition is now **extraction conditioning**
  (cond(A) small) plus **post-solve passivity** (max singular value <= 1 + slack),
  not the retired `a_passive/a_driven <= 0.5` validity gate; `a_passive/a_driven`
  is kept as a bench-quality diagnostic. Measured (finest tier): `a_passive` 0.17,
  |S11| < 0.02, |S21| ~ 1, max singular value ~1.0, reciprocity ~1e-3, `beta` from
  `arg(S21)/L` within 0.83% of `k0`. Exact per-tier numbers live in the artifact.
- **Reciprocity is symmetric-trivial** for this fixture: the coax is
  mirror-symmetric about x=0, so `S12 = S21` by construction. It is a sanity check,
  NOT independent conservation evidence -- the passivity singular value carries the
  conservation content.

### 1.2 Rectangular waveguide (BLOCKED on the transverse mode-operator)

- The withdrawn round-3 "shifted 3D-Yee TE10 onset" story was **false physics**:
  the discrete TE10 cutoff is `0.99752 fc`, BELOW the continuum, and the band
  propagates at every tier. The real defect was in mode SELECTION: the vector
  selector injected a **checkerboard-aliased** eigenvector that merely shares the
  TE10 eigenvalue (`sin(pi y/a)`-correlation `0.000`), whose odd profile couples to
  TE20 (cutoff exactly `2 fc`) and whose evanescent tunnelling reproduced the old
  measured |S21| to 3 significant figures.
- The selector is now hardened (F1): it rejects the `k0` transverse null branch by
  an **absolute** transverse-uniformity signature (the old squared-difference
  threshold scaled as `dx^2` and silently rejected legitimate fine-grid / high-f
  modes -- TE10 at 6 fc has `beta/k0 = 0.986`, executed). The checkerboard filter is
  **scoped to the graded (structure-enforcing) path only** and is NOT applied on the
  uniform-isotropic aperture (that path also serves free-space / open TE ports whose
  fundamental is legitimately plane-wave-like, so a generic checkerboard reject there
  discards valid modes). The **wall-peak gate is disabled** (`wall_peaked` is
  hard-coded `False`; `wall_peak_fraction` is retained only as a persisted diagnostic).
  On the hollow guide the selector therefore **returns the checkerboard-aliased
  candidate** with its `checkerboard_fraction` persisted; it is the **benchmark's
  `sin(pi y/a)`-correlation gate** (< 0.9, section 1.2 head) that refuses to use it.
  The `k0` null branch is still rejected structurally, and the selector **never
  substitutes** another mode for a genuinely absent requested index -- it raises.
- The transverse VECTOR operator itself, however, cannot yet produce a clean
  full-grid TE10 on a hollow metallic guide. Executed evidence: the centered
  uniform-isotropic branch composes a stride-two stencil that decouples the
  odd/even transverse sublattices, so the half-wave `sin(pi y/a)` lives on ONE
  sublattice with the other ~0; the best full-grid `sin`-correlation recoverable
  over the **entire** degenerate subspace at `beta_TE10` is only in the `0.51-0.59`
  range (independently measured: dx=0.05->0.548, 0.025->0.522, 0.02->0.592,
  0.01->0.509), and every candidate has `checkerboard_fraction > 0.35`. The
  alternative staggered branch
  couples the sublattices but has an asymmetric boundary (one wall Neumann, one
  Dirichlet) that shifts `beta` ~10% low. The waveguide is therefore recorded as
  **blocked** (status), pending the operator redesign in the open items.

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

* **Coax two-port is now a wave-level PASS.** Two false attributions are
  **withdrawn**: the round-2 "TEM WavePort does not match the round line" and the
  round-3 "TEM wavelength vs thin PML under a uniform num_layers API" stories, and
  the round-3 coax falsification note that extension through the PML was "necessary
  but not sufficient". Executed: extension through the computational PML IS
  sufficient. The only defect was that the conductors ended at the declared bounds
  (the PML interface); running them to the padded grid edges terminates the line
  (`a_passive/a_driven` 1.17 -> 0.17, the `1.17` re-entrant figure being the round-3
  config at dx=0.005). With `B=S*A` extraction the S-matrix is
  physical and passive: |S11| < 0.02, |S21| ~ 1, max singular value ~1.0, cond(A)
  ~1.2, `beta` from `arg(S21)/L` within 0.83% of `k0` (finest tier). No API change
  was needed. The half-grid contour snap is deterministic and its snap distance is
  persisted per tier (`benchmark/scenes/rf/coax_thru.py:snap_contour_half`).
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

## 5. Open items (round-4, filed not implemented)

* **Transverse mode-operator redesign (blocks the waveguide).** The full-vector
  transverse operator (`modes.py:_build_vector_operator_sparse`) cannot represent a
  clean full-grid guided mode on a hollow metallic aperture: the centered
  uniform-isotropic branch decouples the odd/even sublattices (checkerboard
  copies), and the staggered branch has an asymmetric metallic boundary that shifts
  `beta` ~10%. A symmetric-BC Yee-staggered transverse operator (or a mixed
  Dirichlet/Neumann scalar Helmholtz reduction for the homogeneous-guide case) is
  required so the selector's structural filters have a genuine `sin(pi y/a)`
  candidate to return. Until then `rf/rectangular_waveguide` and the matched/short
  |S11| gate are xfail/blocked. Evidence: `sin`-correlation cap in the `0.51-0.59`
  range over the full degenerate subspace (dx=0.05->0.548, 0.025->0.522,
  0.02->0.592, 0.01->0.509), all candidates `checkerboard_fraction > 0.35` (executed).
* **PlaneMonitors are silently dropped from WavePort / PortSweep Results.** This
  blocks field-level falsification of the wave benches (you cannot inspect the
  injected/propagated transverse field to confirm the mode shape from a normal
  run). Filed as an open item; not addressed this round.
* **External reference-solver generation remains deferred** pending owner cost
  authorization (section 4).

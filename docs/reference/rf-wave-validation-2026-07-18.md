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
| `rf/rectangular_waveguide` | terminated FDTD two-port; TE10 `beta(omega)` from `arg(S21)/L` vs analytic dispersion, S via `B=S*A`, gated on extraction conditioning + passivity | `wave-level` | **pass** (sin-corr 1.0000, cond(A) ~1.1, max sv ~1.001, beta 0.05% median vs 1% gate; external-reference cross-check 1.2%) — see 1.2 |
| `rf/microstrip_two_port` | terminated FDTD two-port; `beta` from `arg(S21)/L` vs Hammerstad `beta=k0 sqrt(eps_eff)`, S via `B=S*A`, gated on extraction conditioning + passivity | `wave-level` | **gap** (F2b: unblocked and RUNS -- cond(A) 1.23, max sv 1.09, a_passive 0.16, \|S21\| 0.77-0.89, \|S11\| 0.05-0.24; measured eps_eff ~1.86 vs Hammerstad 3.27 is a resolution-limited quasi-TEM under-loading, recorded not forced) -- see 1.3 |
| `rf/differential_pair` | terminated FDTD four-port; single-ended `B=S*A` + mixed-mode conversion, gated on extraction conditioning + passivity | `wave-level` | **gap** (F2b: unblocked and RUNS -- cond(A) 1.31, max sv 1.18, coupled 4-port with \|Sdd21\|!=\|Scc21\|; same resolution-limited impedance gap) -- see 1.3 |
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

### 1.2 Rectangular waveguide (wave-level PASS on the Yee-staggered operator)

**Status: PASS (round E, E1b).** The transverse mode-operator redesign filed as the
round-4 open item landed: the selector now solves the hollow guide on the genuinely
Yee-staggered transverse full-vector operator
(`modes.py:_build_yee_transverse_operator_sparse`), which keeps each transverse E
component on its own Yee location and imposes symmetric PEC walls, reproducing the
closed-form discrete eigenpairs of the guide to machine precision. The injected TE10
`Ez` is now a clean full-grid `sin(pi y/a)` (**correlation 1.0000** at dx=0.05 / 0.025 /
0.02 and at 6 fc), so the terminated two-port yields a physical, passive S-matrix and a
genuine wave-level `beta(omega)`.

- **Measured wave-level gate (executed, `python -m benchmark rf rf/rectangular_waveguide`;
  `tests/rf/wave_validation/test_waveguide_wave_level.py`):** the two-port S is assembled
  by solving `B = S*A` across the drive columns, gated (coax_thru precedent) on
  extraction conditioning `cond(A)` and post-solve passivity, then `beta` from
  `arg(S21)/L` vs the analytic dispersion `beta = sqrt(k0^2 - (pi/a)^2)` across the
  1.2 fc .. 2.2 fc band (11 frequencies, all propagating). Finest tier (dx=0.02):
  `cond(A) = 1.09`, max singular value `1.0007`, `|S11|` in `[0.0002, 0.0083]`,
  `|S21| ~ 1.0`, `a_passive/a_driven = 0.088` (diagnostic), and **beta median rel error
  `0.05%`** (per tier: dx=0.05 -> 0.41%, 0.025 -> 0.07%, 0.02 -> 0.05%) against a
  pre-registered **1%** gate (1%-class, coax bench gates its own `arg(S21)/L` beta at 3%;
  the independent Yee numerical-dispersion floor at dx=0.02 is ~0.03%). The a_passive
  ratio collapsing to ~0.09 (from the contaminated-mode ~0.4 floor) confirms the old
  floor was the checkerboard mode, not the termination.
- **External-reference-solver cross-check (one authorized cloud run, M3).** A TE10
  `ModeSource`-driven guide with two `ModeMonitor` planes exports through the adapter
  with `sources=1` (the adapter maps `ModeSource`/`ModeMonitor` to the reference solver's
  native modal source/monitor), so it is genuinely runnable. Cost 0.025 FlexCredits (task
  id in the acceptance doc). The reference forward-mode-amplitude phase constant
  `|d arg(amp_fwd)|/L` agrees with the analytic TE10 dispersion to **1.21% median / 2.74%
  max** over 11 frequencies (an independent solver at a coarse auto-mesh; the ~1% class
  is expected). The analytic dispersion remains the binding first-line reference; the
  external solver is a supporting cross-check.
- **Historical defect (withdrawn).** The withdrawn round-3 "shifted 3D-Yee TE10 onset"
  story was false physics (the discrete TE10 cutoff is `0.99752 fc`, BELOW the continuum,
  and the band propagates at every tier). The round-4 defect was that the centered
  uniform-isotropic transverse operator composed a stride-two stencil that decoupled the
  odd/even transverse sublattices, capping the full-grid `sin(pi y/a)` correlation at
  `0.51-0.59` (dx=0.05->0.548, 0.025->0.522, 0.02->0.592, 0.01->0.509) with every
  candidate `checkerboard_fraction > 0.35`. The Yee-staggered operator resolves this at
  the operator level (golden gates:
  `tests/rf/wave_validation/test_transverse_operator.py`,
  `test_te10_mode_selection.py`). The benchmark's `sin(pi y/a)`-correlation gate is kept
  as a fail-closed regression guard: if the sublattice-decoupling defect ever returns,
  the correlation drops below 0.9 and the scene records BLOCKED rather than reporting a
  spurious S-matrix.

### 1.3 Microstrip / differential pair (F2b: unblocked quasi-TEM wave-level, resolution gap)

**Status: gap (was BLOCKED).** Both scenes were BLOCKED on two stacked blockers: a
single-precision current-contour snap error that fired before the mode solve, and the
categorical inapplicability of `WaveModeSpec('tem')` to the inhomogeneous (substrate +
air) cross-section. F2b resolves both.

- **Routing.** The inhomogeneous interior-PEC quasi-TEM mode now routes through the
  quasi-static electrostatic line-mode engine (`modes.py:_solve_quasistatic_line_modes`,
  `eps_eff = C/C0`) inside `_assemble_vector_mode_data`: when the closed-form uniform-fill
  TEM solve (`_solve_pec_tem_mode_torch`) fails closed on an inhomogeneous fill, a
  non-magnetic cross-section falls through to the quasi-static engine
  (`mode_solver_kind = "quasistatic_line_torch"`). A uniform (air) coax still selects the
  closed-form electrostatic solve (`tem_electrostatic_torch`) -- the fallback is content
  dependent, not blanket (`tests/rf/wave_validation/test_microstrip_diffpair_wave_level.py`).
- **Scene rebuild (coax_thru precedent).** The measurement ports now sit near the origin
  (`+-PORT_X`), not at the line ends, so the current-contour planes stay on the Yee
  half-grid in single precision (a `+-0.15` contour rounds ~6.6e-9 > the ~5e-9 grid
  tolerance; `+-PORT_X` stays under it). The ground/substrate/strip run THROUGH the
  computational PML to the padded grid edges (`line_length`) so the launched quasi-TEM
  wave terminates. Both scenes use integer-cell node arrays (`GridSpec.custom(arange*dx)`)
  so every conductor face, port plane and contour lands on an exact Yee node/half-node
  (`GridSpec.uniform` overshoots the y-span by one cell for lengths whose `length/dx` is a
  hair above an integer).
- **Microstrip measured (executed, `python -m benchmark rf rf/microstrip_two_port`;
  `test_microstrip_diffpair_wave_level.py`):** the terminated two-port yields a
  well-conditioned (cond(A) 1.23), passive (max singular value 1.09) S-matrix with
  `a_passive/a_driven` 0.16, `|S21|` 0.77-0.89, `|S11|` 0.05-0.24 across 0.6-1.6 GHz. The
  quasi-TEM `beta` from `arg(S21)/L` gives a measured `eps_eff ~ 1.86` vs the Hammerstad
  3.27 (median beta rel error ~24%).
- **HONEST GAP (resolution, not extraction defect).** The absolute quasi-TEM `eps_eff` is
  resolution-limited: the thin high-eps substrate is only 4 cells at dx = 5 mm, so the
  discrete Laplace under-loads the field. The quasi-static engine converges toward
  Hammerstad with aperture resolution -- for this eps_r=4.4, W/h=1.5 geometry the
  standalone `eps_eff` rises 2.31 (h=4 cells) -> 2.58 (8) -> 2.77 (16) -> 2.90 (32) toward
  3.27 (a slow, edge-singularity-limited first-order convergence). This is recorded as a
  resolution gap; the S-matrix itself is passive and well conditioned, so it is NOT forced
  to pass. Dropping the substrate to vacuum collapses `eps_eff` to 1.0 (the substrate is
  load-bearing).
- **Differential pair measured (executed):** the single-ended four-port S is
  well-conditioned (cond(A) 1.31) and near-passive (max singular value 1.18); the
  mixed-mode conversion gives `|Sdd21| ~ 0.88` != `|Scc21| ~ 0.68` (even and odd modes at
  different velocities -- a genuine coupled line) with non-zero single-ended coupling
  `|S21|`. The mode-conversion `|Sdc21| ~ 0` is correct physics (the pair is
  mirror-symmetric, so differential and common modes do not convert). The absolute even/odd
  impedances carry the same resolution-limited under-loading, and the passivity singular
  value rides slightly above unity at this coarse aperture.
- **The legacy inhomogeneous diagonal-anisotropic operator is retained** for magnetic
  (mu != 1) apertures: the quasi-static line-mode fallback is guarded to non-magnetic
  cross-sections and re-raises the uniform-fill guard for a magnetic inhomogeneous line.

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
* **microstrip / differential_pair are UNBLOCKED (F2b) and RUN with a resolution
  gap.** Both former blockers are resolved (section 1.3): the contour-snap error was a
  single-precision truncation of large port coordinates (fixed by placing the ports near
  the origin, coax_thru precedent), and the TEM inapplicability is resolved by routing the
  inhomogeneous quasi-TEM mode to the quasi-static electrostatic line-mode engine
  (`eps_eff = C/C0`). The terminated FDTD two-/four-port now yields a well-conditioned,
  passive S-matrix. The remaining gap is the absolute quasi-TEM `eps_eff` accuracy vs
  Hammerstad (~24% low at dx = 5 mm), a documented first-order under-resolution of the thin
  substrate that converges with aperture resolution -- recorded, not forced to pass.
  `reference: generated` (F2c): the WavePort TEM aperture now maps through the adapter to a
  reference `ModeSource` launch + `ModeMonitor` per port, so the microstrip/diff-pair-class
  external cross-check is cloud-runnable (the two RF port caches `coax_thru` and
  `lumped_open_short_match` were generated; see section 4).
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
primary cross-check for the covered port families; adapter-driven generation is
now **wired and authorized** (M3, round E): `python -m benchmark.rf_tidy3d_references`
exports each target scene through `Scene.to_tidy3d`, gates it on being physically
runnable (>= 1 source), cost-estimates, and runs one cloud job per runnable scene,
writing an `.h5` cache plus a `.generated.json` record with the task id and cost. It
never fabricates a comparison: a source-less export records `pending-generation` with
the concrete reason. `series_parallel_rlc` is a lumped-circuit resonance with an
analytic-only reference.

External reference-solver generation is **authorized and executed** for all covered
scenes. `rf/rectangular_waveguide` was generated (a TE10 `ModeSource`-driven guide;
one cloud run, 0.025 FlexCredits, task id in the acceptance doc; beta cross-check
1.21% median, section 1.2). The four port/lumped-driven scenes (`coax_thru`,
`lumped_open_short_match`, `antenna/half_wave_dipole`, `antenna/patch`) are now
**mapped and generated (F2c)**: the adapter maps each `WavePort` TEM aperture to a
reference `ModeSource` drive (port 0) + a receiving `ModeMonitor` per port, and each
`LumpedPort` delta-gap feed to an equivalent `UniformCurrentSource` current injection,
with the NF2FF box lowered to six face field monitors. All four export with `sources=1`
and were cloud-generated (one run each, 0.025 FlexCredits each; task ids in the F2c
acceptance doc and `benchmark/RESULTS.md`). The analytic references remain binding
regardless.

## 5. Open items

### Resolved (round E)

* **Transverse mode-operator redesign — RESOLVED.** The symmetric-BC Yee-staggered
  transverse full-vector operator (`modes.py:_build_yee_transverse_operator_sparse`)
  landed and is wired into the selector. It keeps each transverse E component on its own
  Yee location, imposes symmetric PEC walls, and reproduces the closed-form discrete
  eigenpairs of the hollow guide to machine precision (golden gates:
  `tests/rf/wave_validation/test_transverse_operator.py` — full discrete spectrum,
  exact TE10/TE20/TM11 eigenpairs, second-order convergence, inhomogeneous half-filled
  LSE mode). The selected TE10 is a clean full-grid `sin(pi y/a)` (correlation 1.0000),
  so `rf/rectangular_waveguide` is a wave-level PASS (section 1.2) and the matched/short
  `|S11|` discrimination gate (`test_matched_s11_wave_level.py`) is green.
* **PlaneMonitors are dropped from WavePort / PortSweep Results — RESOLVED (round E).**
  Monitor passthrough for WavePort / PortSweep Results was added, so the
  injected/propagated transverse field is inspectable for field-level falsification
  (`tests/rf/wave_validation/test_planemonitor_waveport_passthrough.py`).
* **External reference-solver generation — AUTHORIZED and GENERATED for the covered
  scenes (round E).** The M3 adapter-driven generation path is wired (section 4). The
  waveguide reference was cloud-generated (one run, 0.025 FlexCredits); the four
  port/lumped scenes are now mapped and cloud-generated as well (F2c; section 4).

### Resolved (F2b)

* **microstrip / differential_pair unblocked -- quasi-TEM wave-level extraction.** The
  inhomogeneous interior-PEC quasi-TEM mode routes through the quasi-static electrostatic
  line-mode engine; the scenes were rebuilt on the coax_thru precedent (ports near the
  origin, conductors through the PML, integer-cell node arrays). Both now run a terminated
  FDTD two-/four-port with a well-conditioned, passive S-matrix; the absolute quasi-TEM
  `eps_eff` is a documented resolution gap (section 1.3). Gates:
  `tests/rf/wave_validation/test_microstrip_diffpair_wave_level.py`.

### Still open

* **microstrip / differential_pair are UNBLOCKED (F2b)** -- see the "Resolved (F2b)"
  block below. The residual item is the absolute quasi-TEM `eps_eff` accuracy (a
  resolution gap, section 1.3), not a blocker.
* **RLC resonance wave-level gate** remains a strict xfail open gap (section 3).

### Resolved (F2c)

* **Adapter port/lumped source mapping — RESOLVED (F2c).** `coax_thru`,
  `lumped_open_short_match`, and the antenna scenes are cloud-runnable and were
  generated: `WavePort` -> reference `ModeSource` + `ModeMonitor`, `LumpedPort` ->
  equivalent `UniformCurrentSource` current injection (adapter
  `_convert_ports_for_reference`; gates
  `tests/api/adapters/tidy3d/test_port_source_mapping.py`).

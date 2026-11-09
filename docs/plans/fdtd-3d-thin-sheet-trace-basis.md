# Conformal thin-sheet copper: chord trace basis, locally implicit (design)

Status: Tasks 1-2 implemented and CPU-qualified; Task 3 CUDA qualification pending.
Replaces the per-face reconstructed trace
(`u_side = W_side E`, Gaussian least squares) and the FSAI/Chebyshev copper
solve of `docs/plans/fdtd-3d-thin-sheet-runtime.md`. Branch
`pcb/3d-thin-sheet`. Runtime sweep, floor and consistency measurements in
section 4 use the CPU reference solver. The original diagnostic numbers come from
`scripts/dev/fdtd/thin_sheet_trace_basis_diagnostic.py` (CPU, complexes from the
real compiler, operators assembled as specified here, no solver runs) unless it
is marked as an estimate. Scenes: the 2 um tilted cavity plate (380 traces),
the cavity-spanning 35 um sheet (1619), a 28.6-degree rotated square prism
(quasi-2D, PMC z walls, L = 10 h / 30 h, 240 / 740 traces) and a plate with
the PCB board normal (0.04, 0.025, 1) (534 traces); h = 5 mm.

## Verdict on the hypothesis

Partly right, and incomplete as stated.

* Right: the trace must be its own degree of freedom. With one trace unknown
  pair per cut face (a "chord") the copper term is a 2x2 block per trace,
  exactly the foil's, and the locking disappears: in the PEC limit the current
  basis has 51 / 140 spurious sheet-localized modes below half the fundamental
  (L = 10 h / 30 h) and the chord basis has none (section 1).
* Wrong: "two tangential trace unknowns per cut cell" is the same row count as
  one per cut face, and non-overlapping or lumped fits cannot be linear-exact
  (section 2). Independent *reconstruction rows* are not achievable; an
  independent *unknown* is.
* Incomplete: chord unknowns sit on the sheet, a sub-cell distance from the
  Cartesian unknowns; treated explicitly they are a local half-cell refinement.
  Measured `lambda_max` is 7-11 x `lambda_Yee` with today's floors, and every
  mass floor that brings it to about 1.0-1.4 costs 1-9 % of the PEC-limit
  eigenfrequency (section 2). So the chords must be advanced **implicitly**.
  Done that way (section 5), the leapfrog bound is exactly the bound of
  today's conformal PEC operator, the implicit system is local (13-14
  non-zeros per row, block-Jacobi condition 1.00 for copper), and nothing
  else changes.

One thing cannot be had: the current copper path's leapfrog bound (1.041
`lambda_Yee` on the cavity plate) is lower than the PEC path's (1.423) only
because the overlapping reconstruction smooths the small cuts, the same
over-constraint that locks. Any locking-free copper inherits the PEC path's
bound. "Zero extra CFL restriction" holds relative to the PEC path; relative
to today's copper path dt shrinks by 0.973 (board normal), 0.855 (tilted
plate) and 0.785 (rotated prism). See open question 1.

## Decision

Every active cut face carries a chord (its cut polyline) with two E unknowns,
one per side; every cut cell carries a sheet face whose normal flux enters
only through the chords. Faraday reads the chord unknowns directly, Ampere is
the transpose, the Cartesian masses and magnetic floors stay exactly as today
(the PEC path is bit-identical). The chords are advanced with a locally
implicit, energy-conserving Newmark (beta = 1/4) treatment of every term that
involves them; the rest stays leapfrog. The per-step implicit system lives on
the chords only, couples a chord to its neighbours on the sheet (13-14
non-zeros per row), and is solved by a few block-Jacobi-preconditioned
Chebyshev steps. The copper two-port enters that system as the foil's 2x2
block. The reconstruction `W`, `S`, FSAI and the dense `K` reference are
deleted.

## 1. Why the current basis is non-local and locks

Non-local: rows of `W` are overlapping same-side linear fits, one per cut face
(41 samples per side on average), so `Z = W C^-1 W^T` is nearly singular: on
the spanning sheet 42 % of the Jacobi-scaled spectrum is below 1e-2 and 18 %
below 1e-3, and `S = I + R Z R` spans [1, 4.75e3] (quantiles 10 / 50 / 90 %:
1.05, 13.2, 1.94e3). Electrically thin resistive sheets are worse (`S` up to
1.67e5 for g = 0.5: the odd thickness mode of a thin two-port is very stiff),
so there is no "soft" regime in which the current basis is cheap. `S^-1` is
local iff `W C^-1/2` is well conditioned, i.e. iff every trace owns a degree
of freedom with its own mass.

Locking: in the copper PEC limit the current basis enforces `W E = 0` on the
Cartesian field; the PEC scheme enforces nothing (it only drops the cut
contribution from Faraday). Interior Hz-type modes of the rotated square prism,
`omega / (pi c / L)`, analytic 1, 1, 1.4142 (Yee dispersion alone gives about
0.9973 at 10 h, 0.9997 at 30 h for the first pair):

| basis, PEC limit | L = 10 h | L = 30 h | spurious modes below 0.5 (10 h / 30 h) | localized in-band modes |
| --- | --- | --- | --- | --- |
| current PEC path | 0.9905, 0.9928, 1.3989 | 0.9980, 0.9981, 1.4104 | 0 / 0 | 1 / 0 |
| current copper basis (exact restriction to `W E = 0`) | 0.9934, 0.9945, 1.3939 | 0.9992, 1.0017, 1.4119 | **51 / 140** | **15 / 100** |
| chord basis (this design; identical to the PEC path) | 0.9905, 0.9928, 1.3989 | 0.9980, 0.9981, 1.4104 | 0 / 0 | 1 / 0 |

(Two more prism positions: copper basis 47 and 134-148 spurious modes, PEC path
at most 3 localized modes. "Localized": participation ratio < 0.05.)

The extended modes of the constrained copper basis are fine; the defect is a
band of sheet-localized spurious modes, about 0.2 per trace below half the
fundamental, that grows with the trace count rather than converging away.
Their origin: static fields that the PEC scheme keeps at zero frequency
(gradients of a potential constant on the sheet) violate the overlapping
fits and are pushed to low non-zero frequencies. At large but finite `Y` they
are lightly damped and couple to the large normal field of TM incidence (the
mixed z-face chords of a tilted or curved sheet). This matches every
observation in the runtime plan: TM only, growing with `g` at fixed geometry
(0.090 / 0.120 at `g eta0` = 1e3 / 1e4), slightly better with a smaller
stencil.

## 2. Alternatives and measurements

* **Two tangential traces per cut cell.** Every pierce vertex of the sheet has
  degree 4 in the cut graph, so chords = 2 x vertices and cut cells = vertices
  (spanning sheet: 1619 cut faces, 840 pierced edges). Two rows per cell is
  one row per face: same count, same overlapping fits, same conditioning.
* **Non-overlapping or lumped fits.** A linear-exact fit needs at least 4
  same-side samples per Cartesian component it uses (8 for a chord mixing two
  axes); there are 1.2 same-side Cartesian unknowns within one cell of the
  sheet per chord and side. Disjoint linear-exact stencils would reach about
  3 cells out; constant-exact ones break the linear-field circulation
  exactness of the sub-faces (an O(1) curl error). And any reconstruction
  keeps the `W E = 0` constraint, hence the locking.
* **Coarser or low-rank trace spaces, impedance (Z-form) traces.** Fewer rows
  still need a global solve; a massless trace in impedance form puts the sheet
  admittance on H and couples all sheet faces.
* **Chord unknowns, explicit.** Measured `lambda_max / lambda_Yee` and the
  PEC-limit first mode of the prism (the PEC limit must use the same masses):

| chord masses and floors | plate | prism 10 h | PEC-limit (1,0), 10 h / 30 h |
| --- | --- | --- | --- |
| today's copper basis, for reference | 1.041 | 1.318 | 0.9934 / 0.9992 (+ spurious modes) |
| today's PEC path, for reference | 1.423 | 2.139 | 0.9905 / 0.9980 |
| FIT dual mass (`delta` = centroid distance), today's floors | 88.8 | 262 | 0.9960 / 0.9993 |
| same, `delta >= h/4` | 7.2 | 10.7 | 0.9970 / 0.9997 |
| same, chord counted in the stiffness floor (ratio 2) | 1.40 | 2.56 | 0.9130 / 0.9717 |
| same, iterative floors on the top leapfrog mode | 2.36 (not converged) | 1.315 | 0.9300 / 0.9953 (30 h: 2.23, not converged) |
| heavy chords (`delta >= h/2`, Cartesian volumes untouched), floor ratio 2 | 1.140 | 1.421 | 0.9612 / 0.9888 |
| same, floor ratio 1 | 0.986 | 1.151 | 0.8619 / 0.9504 |

  Removing the chord's FIT volume from the Cartesian pieces (the 1D FIT dual:
  a sheet 0.3 h above node k gives `E_x(k)` 0.65 h instead of 0.8 h) makes the
  PEC limit more accurate (0.9960 / 0.9993 against 0.9905 / 0.9980) but makes
  the Cartesian pieces light: its explicit part alone has `lambda_max` 2.67
  (plate) and 6.1-8.5 (prism). Rejected for now; see open question 3.
* **Chord unknowns, locally implicit (chosen).** The explicit part is exactly
  today's PEC operator (plate 1.423, prism 2.139 / 2.253, board normal 1.897;
  today's copper 1.041, 1.318 / 1.332, 1.796). The implicit chord system has
  13-14 non-zeros per row; its block-Jacobi condition number is in the table
  of section 5.

## 3. Unknowns

Terms: *active cut face* f: a Yee face split by the sheet inside the outline
(today's `active_face`, one per current trace). *Chord* c_f: its cut path
(`path` in `compile_thin_sheet`: two zero points, plus the fold point on fold
faces), oriented from `zeros[0]` to `zeros[-1]`, length `l_c`, strip width
`w_c` (today's `trace_width`), direction tensor
`tau_c = sum_i (l_i / l_c) t_i t_i^T` over its pieces. *Cut cell*: a Yee
primal cell whose chords (and on-sheet trace pairs on its edges) close exactly
one loop.

| unknown | where | count | storage |
| --- | --- | --- | --- |
| `c_{f,-}`, `c_{f,+}` (V/m, mean tangential E along the chord, per side) | each chord | 2 per active cut face | private float64 `chord_e[T, 2]`, plus `chord_e_prev[T, 2]` |
| chord curl accumulator `F_{f,s}` (A) | each chord side | 2 per chord | private float64 `chord_f[T, 2]` |
| sheet faces (P, -), (P, +) (independent normal fluxes) | each cut cell | 2 per cut cell (about T) | none: they enter only through the operator `A_cc` |
| on-sheet trace pairs (exact grid coincidence) | Yee edges on the sheet | unchanged | native slot (-), private (+), unchanged |

On-sheet trace pairs that are not on any sheet-face loop keep today's
treatment, which is the foil (explicit curl, 2x2 copper CN). An on-sheet trace
pair that lies on a sheet-face loop (a tilted plane passing exactly through a
grid edge) is treated as a chord with its clipped half-box volumes.

`u_{f,s} = l_c c_{f,s}`: the "reconstruction" is the identity, rows are
independent by construction.

Rim cells whose chords do not close a loop get no sheet face (O(h) rim, as
today). Chords on held PEC-wall faces are held at zero.

## 4. Faraday, Ampere, masses

Circulation matrix `K` (rows: H, columns: E, E-value units):

* Sub-face (f, s): today's own pieces unchanged, plus `o_{f,s} l_c` in the
  column of `c_{f,s}`; `o_{f,s} = +1` iff the chord orientation matches the
  sub-face's CCW traversal (`ordered` equals `path` in the compiler). Replaces
  the `path_integral` entries. Each sub-face has exactly one chord.
* Sheet face (P, s): entries `o_{P,c} l_c` in the columns of `c_{c,s}`
  only, including promoted on-sheet traces of that same side. Orientation is
  counter-clockwise about the + side normal at the loop's vertex mean.
  The two sides have independent magnetic fluxes: no side averaging enters
  these curl rows. The copper two-port is the only cross-side coupling.
  The rejected shared flux leaked thickness-independently at 3.8e-9; removing
  it gave 1.1e-10 at 35 um and restored the approximately 1/t shielding law.
* PEC: no chords, no sheet faces: every row is today's PEC row.

Ampere is `K^T` with diagonal masses, Lh = `ltilde / (mu A_eff)`:

* Cartesian unknowns, segments, sub-faces, regular faces: today's volumes and
  magnetic areas, unchanged (including the stiffness-ratio-2 sub-face floor
  computed on own pieces without the chord). The PEC complex is unchanged.
* Chord: `V_{c,s} = kappa l_c w_c max(d_{c,s}, h/16)`, `kappa = 0.5`
  (`CHORD_MASS_FACTOR` in the compiler, with no user option),
  `d_{c,s}` = |level function| at the centroid of sub-face (f, s) (polygon
  including the fold point). `l_c w_c d_{c,s}` is the 1D FIT dual of a node on
  the sheet; `kappa` makes it a small additive capacitance, because the chord
  needs no mass for stability (section 5) and every unit of chord mass is an
  extra sheet capacitance (additive bias measured at `kappa = 1`:
  0.19-0.43 cell of thickness per side). The sweep, independent chord-free
  controls and mass-consistency checks below select 0.5 for final qualification.
  Every absolute-error gate is retained; only the floor-qualified resistive
  reflection-order gate is amended.
* Per-side sheet face: both sides use `A_P` and
  `A_eff = max(A_P, a_min h^2)`, with triangle-fan loop area `A_P` and
  `a_min = 0.25`. Split the pair dual length
  `ltilde_P = sum_s mean_{c in dP} d_{c,s}` in proportion to the cut cell's
  clipped side volumes. Each side keeps at least one quarter of the pair
  volume, and the pair total is conserved; dual lengths use those same
  fractions. Cartesian masses stay unchanged (FIT removal remains deferred).
  Held PEC-limit chords leave both independent sheet fluxes constant.

The pair `(K, K^T)` with positive diagonal masses is skew-adjoint in
`1/2 E^T M_e E + 1/2 H^T M_h H` whatever the mass values; the time integrator
below keeps a modified version of that energy.

Mass-factor sweep at 98.5% of the PEC bound (V11: 30-degree TE,
`g eta0 = 0.5`; refinement: `g eta0 = 1`; V12: 35 um TM at lambda/20):

| kappa | V11 transmission error | V11 reflection error | refinement error ratio | V12 copper-PEC difference |
| --- | --- | --- | --- | --- |
| 0.1 | 0.0355913 | 0.0386927 | 0.507910 | 3.54094e-5 |
| 0.25 | 0.0296721 | 0.0328063 | 0.504051 | 3.54213e-5 |
| 0.5 | 0.0197971 | 0.0229830 | 0.492228 | 3.53842e-5 |
| 1.0 | 0.00117595 | 0.00337344 | 0.719355 | 3.53881e-5 |

Every factor preserves the held-chord prism modes and the board/plate/prism
explicit-bound ratios (all 1.0). At 1 GHz, the chord-only Newmark-minus-leapfrog
phase symbol is -2.41286e-4 rad/step for the oblique case, -2.42070e-4 /
-3.01828e-5 for its coarse/fine refinement pair, and -2.31960e-4 for V12 TM.
Those symbols depend on frequency and dt, so they are unchanged across the
mass sweep; the measured scattering phase changes with the mass.

Only 0.5 passes all swept V11 cases. Before calibration, its
`g eta0 = 4` reflection errors 0.00334608 / 0.00242178 gave order 0.466403,
missing the original order gate. Factor 1.0 reduces the oblique errors, but
its normal-incidence transmission error at `g eta0 = 0.5` is 0.0190998,
above the unchanged 0.01 gate. It is not selected.

Chord-free controls and lambda/80 evidence were measured on 2026-10-30,
at 1 GHz, TE, 30-degree incidence, with the same pulse, gate and timestep
as the corresponding tilted scene. Transmission refinement uses a sheet
half-length of 5 wavelengths; reflection uses 6. The native foil control
places the foil at x = 0 and rotates the source and observers to preserve
incidence, with one wavelength of tangential padding. Both that path and
the tilted PEC controls have zero chord unknowns. The native foil errors
include its own discretization and oblique-source errors, so they are not
interpreted as pure measurement noise.

| measurement | lambda/20 | lambda/40 | lambda/80 |
| --- | --- | --- | --- |
| kappa 1.0, tilted transmission, g eta0 = 1 | 0.00114827 | 0.000826013 | 0.000509926 |
| native aligned foil, same transmission setup | 0.0158788 | 0.0136545 | 0.0121712 |
| tilted PEC reflection, transmission geometry | 0.00491616 | 0.00116664 | 0.000760368 |
| kappa 1.0, tilted reflection, g eta0 = 4 | 0.00377585 | 0.00115905 | 0.000660703 |
| native aligned foil, same reflection setup | 0.0104220 | 0.0104800 | 0.0103173 |
| tilted PEC reflection, reflection geometry | 0.00604137 | 0.00130369 | 0.000724781 |
| selected kappa 0.5, tilted transmission | 0.0123835 | 0.00609551 | 0.00303324 |
| selected kappa 0.5, tilted reflection | 0.00334608 | 0.00242178 | 0.00144208 |

Use the tighter, geometry-matched tilted-PEC reflection errors as the
declared floor: 0.001303694437364873 at lambda/40 and
0.0007247811853102816 at lambda/80. The selected reflection errors are
1.85763 and 1.98967 times these independent floors. Only for `g eta0 = 4`,
admit an observed order below 1 when the finer error is at most twice the
declared floor at that resolution. The two selected orders are 0.466403
and 0.747920. PEC retains its original strict lambda/20 -> lambda/40 gate.
Transmission refinement remains strict: its ratios are 0.492228 and
0.497620. No absolute-error gate changes. The floor constants and their
provenance are in `test_thin_sheet_scattering.py`.

Mass sensitivity was measured with `g eta0 = 0.5`, TE, half-length 6
wavelengths: oblique incidence is 30 degrees; normal incidence keeps the
original physical sheet offset of 0.0045 m at every grid. Each mass factor
uses exactly the same timestep at each resolution. Spreads below are the
complex-coefficient differences, not differences of magnitudes.

| setup and spread | lambda/20 | lambda/40 | lambda/80 | order 20 -> 40 | order 40 -> 80 |
| --- | --- | --- | --- | --- | --- |
| oblique, abs(T(0.25) - T(1.0)) | 0.0296428 | 0.0155455 | 0.00796703 | 0.931184 | 0.964385 |
| oblique, abs(T(0.5) - T(1.0)) | 0.0197725 | 0.0103649 | 0.00531149 | 0.931785 | 0.964520 |
| normal, abs(T(0.25) - T(1.0)) | 0.0375902 | 0.0188481 | 0.00942069 | 0.995936 | 1.000514 |
| normal, abs(T(0.5) - T(1.0)) | 0.0250675 | 0.0125660 | 0.00628064 | 0.996298 | 1.000537 |

All four spreads shrink at both refinement steps, approximately at first
order. This agrees with the mass rule: a cut-face chord adds O(kappa h^3)
volume, with O(h^-2) chords per sheet area, so the additive capacitance per
area is O(kappa h) and vanishes under refinement at fixed kappa. Promoted
on-sheet pairs keep their original masses and add no kappa-dependent term.

## 5. Time integration, copper and the chord solve

Write the semi-discrete system in E: `M u'' + G u' + A u = 0`,
`A = K^T Lh K`, split by unknown sets X (Cartesian patch unknowns) and C
(chords): `A_xx`, `A_xc`, `A_cc`. `G` is the copper (per chord pair, area
`A_t = l_c w_c` times the foil's two-port, with its modal states) plus the side
media conductivity. The scheme is

```
M (u^{n+1} - 2 u^n + u^{n-1}) / dt^2 + G-terms + A_e u^n + A_i ubar^n = 0,
A_e = [[A_xx, 0], [0, 0]],   A_i = A - A_e,   ubar^n = (u^{n+1} + 2 u^n + u^{n-1}) / 4
```

i.e. leapfrog for everything except the terms that touch a chord, which get
the Newmark beta = 1/4 average. Energy: multiplying by
`(u^{n+1} - u^{n-1}) / 2` gives the conserved (lossless) quantity
`1/2 v^T (M - dt^2/4 A_e) v + 1/2 w^T A w`, `v = (u^{n+1} - u^n)/dt`,
`w = (u^{n+1} + u^n)/2`. It is positive iff `M_x - dt^2/4 A_xx > 0`, i.e.
`dt^2 lambda_max(M_x^-1 A_xx) < 4`: **the leapfrog bound of today's PEC
operator**, independent of the chord masses and of `A_i` (which may be
indefinite: `A` itself is PSD). Copper adds a PSD dissipation and the foil's
modal storage; passivity follows as for the foil. The computed Lanczos bound
and `thin_sheet_cfl` tightening stay, run on the operator with the chord
columns dropped (= the PEC complex).

Staggered realization per step (n -> n+1). `X_c` = Cartesian unknowns that
share a sub-face row with a chord; `L` = sub-face rows with a chord.

1. H stage, chord sub-face rows: predictor
   `H_L* = H_L^{n-1/2} - (Db / A_eff) (K_Lx E_X^n + K_Lc (2 c^n + c^{n-1}) / 4)`.
   All other H rows as today.
2. E stage, explicit patch rows as today with `H_L*`: gives `E_X*`.
3. Chord system for `c^{n+1}` (from the two relations below):
   ```
   F^{n+1/2} = F^{n-1/2} - dt [A_cx (E_X^{n+1} + 2 E_X^n + E_X^{n-1}) / 4 + A_cc (c^{n+1} + 2 c^n + c^{n-1}) / 4]
   M_c (c^{n+1} - c^n) / dt = F^{n+1/2} - A_t [Y_eff (c^{n+1} + c^n) / 2 + history^n] - Sigma_c (c^{n+1} + c^n) / 2
   E_X^{n+1} = E_X* - Q c^{n+1},   Q = (cb_X / V_X) K_Lx^T diag(ltilde Db / A_eff) K_Lc / 4   (= dt^2/4 M_x^-1 A_xc, lossless)
   ```
   giving `S c^{n+1} = r` with
   `S = M_c/dt + Sigma_c/2 + A_t Y_eff/2 + dt/4 A_cc - dt/4 A_cx Q` (SPD; it is
   `dt` times `M_c/dt^2 + G/(2 dt) + A_cc/4 - dt^2/16 A_cx M_x^-1 A_xc`).
   `Y_eff`, `history`, `decay`, `drive`, the state update and the heat are the
   foil's (`fdtd/copper_foil.py`), per trace, with `E = c`.
4. Solve for the increment `delta = c^{n+1} - (2 c^n - c^{n-1})`:
   `S delta = r - S (2 c^n - c^{n-1})`, with `k` Chebyshev steps on the
   block-Jacobi-preconditioned system (2x2 blocks, one per chord pair) and
   **no** Richardson step, so the error polynomial `s_k(mu)^2` lies in
   `[0, eps]` and the applied inverse satisfies `0 < P <= S^-1`. The computed
   `delta` is then the exact solution with `S` replaced by `P^-1 >= S`, i.e.
   with chord mass `M_c + dt (P^-1 - S) >= M_c`: still the scheme above,
   still energy-stable, for any degree. `k` is the smallest with
   `(b/a - 1) eps_k^2 <= 1e-12` on Lanczos bounds `[a, b]` widened by 2 %.
5. Update `F`, apply `E_X^{n+1} = E_X* - Q c^{n+1}` (scatter), correct
   `H_L^{n+1/2} = H_L* - (Db / A_eff) K_Lc c^{n+1} / 4`, update the copper
   state and heat, shift `c^{n-1} <- c^n <- c^{n+1}` and keep `E_{X_c}^n` for
   the next step's `E_X^{n-1}`.

Initial state: everything zero. Sheet-face fluxes are never stored: their
whole effect is the `A_cc` term (`F` accumulates it). Sub-face H stays a
stored unknown because the explicit Cartesian rows read it.

Measured block-Jacobi condition number of `S` (dt at the explicit bound,
`kappa = 0.1`, Cartesian masses unchanged, 13.2-14.4 non-zeros per row):

| sheet-face floor `a_min` | scene | 35 um Cu | g = 4 | g = 0.5 | `Y = 0` |
| --- | --- | --- | --- | --- | --- |
| none | plate / prism 10 h | 1.01 / 1.00 | 18.7 / 2.97 | 78 / 8.2 | 408 / 21.7 |
| 0.1 | plate / prism 10 h / prism 30 h | 1.00 | 1.97 / 2.10 / 2.06 | 5.43 / 4.59 / 4.57 | 11.7 / 10.3 / 11.7 |
| **0.25 (chosen)** | plate / prism 10 h / prism 30 h | 1.00 | 1.73 / 1.72 / 1.71 | 4.26 / 3.30 / 3.25 | 8.18 / 6.60 / 7.98 |

(Without a floor, tiny corner polygons couple their chords strongly. The floor
only touches sheet faces smaller than a quarter cell, an O(h) local change of
the normal-flux energy at the sheet, which is zero for PEC-like copper.
Without the `kappa` mass the `Y = 0` condition grows without bound: 408 at
`kappa = 0.1`, 3.8e3 at 0.01, 5.1e4 at 1e-6, plate, no floor.)

Copper is dominated by the 2x2 blocks (condition 1.00: `k` = 2-3). Resistive
sheets need `k` of about 15 and the `Y -> 0` limit about 20 for 1e-12.

## 6. Per-step cost

Per trace and step, float64, from the operator layout (estimate):

| | current | chord basis |
| --- | --- | --- |
| copper / chord operators read | about 190 KB: `S` (227 nnz/row) and FSAI (2 x 116 nnz/row) applied 15-17 times, Woodbury ends, `W`, `W^T` | `S` 14 nnz/row x 2 rows x 12 B = 0.34 KB per application, 2x2 blocks 32 B; `k + 1` applications plus `A_cc`, `A_cx`, `Q` once (about 0.7 KB): about 1.7 KB for copper (`k` = 3), about 6 KB for resistive (`k` = 15) |
| modal state (both bases) | `16 M` B: M = 257 (35 um, 0.5-4 GHz) 4.1 KB; 129 (0.3-2 GHz) 2.1 KB; 17 (2 um) 0.3 KB; 5 (resistive) 0.1 KB | same |
| implicit rows | whole reconstruction stencil (41 unknowns per side and trace) | chords only |
| private state | side unknowns | plus `c^{n-1}`, `F`, `E_{X_c}^{n-1}` (about 64 B) |

Expected step ratio (estimate; A6000 at about 600 GB/s effective, captured
graph, uncontended): 61k traces on 241 x 221 x 101 (plain Yee 2.6 ms) add
about 61k x (1.7 + 4.1 state + 1.4 patch rows) KB = 0.44 GB, about 0.7 ms of
traffic plus about 50 graph kernels (0.25 ms): ratio about 1.4. 6188 traces
on 149^3: about 1.1. Confidence: high that the copper term stops dominating
(more than 100x less operator traffic); moderate on the absolute ratio,
because the measured non-copper patch cost (5 ms at 6188 traces, 19 ms at
61k, eager, on a GPU shared with a job at 100 %) is 20-60x its traffic
estimate and was never measured uncontended; part of it scales with today's
copper stencil and disappears. Steps per simulated time: unchanged for the
board normal within 3 %, up to 1.27x more for strongly tilted sheets
(section "Verdict").

## 7. Limits

* **Axis-aligned (exact grid coincidence).** No cut faces: no chords, no sheet
  faces, nothing implicit. The on-sheet trace pairs are the foil edges; the
  2x2 CN must equal the foil's `inverse` / `rhs_matrix` to rounding. Test:
  extract both per-edge `[E, 2, 2]` tensors on the aligned scene of
  `test_axis_aligned_mesh_reproduces_existing_planar_paths`, match edges,
  relative difference <= 1e-14; the 260-step comparison keeps its 2e-5 gate.
  The foil files are not touched, so the foil path is bit-identical.
* **Nearly aligned sheets (PCB).** Board normal: explicit bound 1.897 vs
  today's copper 1.796 (dt x 0.973).
* **PEC limit.** As `Y -> infinity`, `S` is dominated by `A_t Y_eff / 2`,
  `c -> 0`, `Q c -> 0`, `A_xc cbar -> 0`: the scheme becomes today's PEC
  scheme exactly (same masses, same rows). Copper minus PEC is then the
  physical `O(Z_s / eta0)` difference (about 1e-4 for 35 um at 1 GHz).
* **Resistive sheets.** `S` stays SPD for every passive two-port including
  `sigma -> 0` (`A_cc` from the sub-face rows is positive definite on chords).
  The chord mass adds a sheet capacitance of about `kappa` times the FIT
  sliver: estimated `|T - T_a|` change about 0.005 for g = 0.5 at normal
  incidence and lambda/20 (gate 0.01), less at oblique incidence and with
  refinement.
* **Faceted sheets.** Fold faces carry one chord along the two-piece path
  (today's trace); sheet faces across a fold use the bent loop. Rejections
  unchanged.
* **Sheet outline / rim.** Chords only on active faces (midpoint outline test,
  unchanged); open rim loops get no sheet face. O(h) rim, unchanged in kind.
* **CPML proximity.** Unchanged rule. The patch shrinks (no 1.75 h
  reconstruction stencils), so it reaches the absorber less often.
* **Not precluded.** A free-edge impedance (`edge_model="auto"` on master) is
  a per-chord term in `G` of the rim chords; feeds are a modified `G` and a
  known right-hand side on the gap chords. Both keep `S` SPD and local.

## 8. Expected effect on accuracy

* Locking: removed by construction (no constraint on the Cartesian field; the
  copper PEC limit is the PEC scheme). Planar 30 degree TM `|R_Cu - R_PEC|`
  (35 um): from 0.0016 / 0.00094 (lambda/20 / 40) to below 1e-3 at both.
  Cylinder at lambda/40: copper vs PEC from 0.0145 (TE) / 0.0513 (TM) to below
  1e-3, copper TM series error from 0.053 to the PEC level (0.0042).
  Confidence: high for the copper -> PEC identity (structural) and the absence
  of spurious modes (measured); moderate-high for the cylinder numbers.
* PEC results are unchanged (same operator). Resistive sheets: the
  reconstruction error of today's traces is replaced by the `kappa` sliver
  capacitance and the Newmark averaging on chords (second order); expected
  within the existing gates, moderate confidence.

## 9. Validation plan (gates declared before implementation)

CPU unless marked CUDA; CUDA needs the GPU owner's grant
(`neural_maxwell_solver_roadmap.md`).

| ID | Check | Gate |
| --- | --- | --- |
| V1 | Foil path untouched | no diff in `compiler/copper_foil.py`, `fdtd/copper_foil.py`, `fdtd/copper_foil_ade.py` |
| V2 | PEC path untouched | PEC complex arrays (`K`, volumes, areas) bit-identical to `e2a93e4e`; PEC runtime tests unchanged |
| V3 | Axis-aligned operators | foil-matched 2x2 `inverse`, `rhs_matrix` <= 1e-14 relative; no chords, no sheet faces |
| V4 | Axis-aligned fields | existing 260-step test <= 2e-5 of peak (copper, PEC) |
| V5 | Exactness | linear fields: every sub-face and sheet-face circulation with chords set to the exact mean tangential field <= 1e-9 (plate, bent, vee, prism) |
| V6 | Structure | sub-face rows have exactly one chord; sheet-face rows read chords only; chord count = active faces; never both sides in a Cartesian row |
| V7 | Copper -> PEC | with `c` held at zero the explicit operator equals the PEC complex's bit for bit; interior prism modes (port of diagnostic part B) equal the PEC path's, no mode below 0.5 |
| V8 | Explicit bound | `lambda_max` of the explicit part equals the PEC complex's (same matrix) |
| V9 | Implicit solve | `S` SPD; block-Jacobi condition <= 1.05 (copper), <= 6 (g = 0.5), <= 12 (`Y = 0`) on the plate and prism scenes (measured 1.00 / 4.26 / 8.18); one step with `P` vs a dense solve: increment relative error <= 1e-10; `P <= S^-1` checked on 20 random vectors |
| V10 | Passivity, 20000 steps | tilted 2 um Cu cavity and bent board: modified energy non-increasing (1e-9 slack), heat > 0, physical energy bounded; PEC drift <= 1e-3 |
| V11 | Scattering vs analytic | absolute gates unchanged (normal 0.01, oblique 0.03, PEC 1e-6); transmission refinement ratio <= 0.6; only g eta0 = 4 reflection admits order >= 1 OR finer error <= 2 x its independently measured floor at lambda/40 and lambda/80 (section 4); PEC order gate unchanged |
| V12 | Copper vs PEC, planar | 30 deg TM and TE, 35 um: abs(R_Cu - R_PEC) <= 1e-3 at lambda/20 and lambda/40 |
| V13 | Copper vs PEC, cylinder (CUDA) | lambda/40: difference <= 1e-3 (TE, TM); copper series error <= 0.015 TE, 0.01 TM; remove the xfail and the regression gate |
| V14 | Cross-sheet floor | spanning 35/70 um Cu far/near <= model floor, ratio >= 1.6 (existing) |
| V15 | Cost (CUDA, uncontended, captured graph, `thin_sheet_overhead.py`) | `copper_patch` (6188 traces) ratio <= 1.5; `copper_board_60k` (61342) <= 2.0; chord operators <= 2 KB per trace excluding modal state |

Task 2 measurements at `CHORD_MASS_FACTOR = 0.5` (CPU reference).
Added dispersion below means the chord-only temporal-symbol difference
`2 [atan(pi f dt) - asin(pi f dt)]`, Newmark minus leapfrog in rad/step.
It is not a measured whole-scene frequency shift; at fixed frequency and dt
it is mass-independent. Held-chord modes have exactly zero added dispersion.
V9 and scattering use 1 GHz; V10 uses the 2 GHz source center.

| gate | measured result | added dispersion (rad/step) |
| --- | --- | --- |
| V2 / V7 structural | plate, bent, vee, prism: zero changed PEC/explicit/mass entries | 0 |
| V3 | inverse and rhs-matrix relative differences both 0 | 0 |
| V4 | copper 4.64257e-6; PEC 1.13695e-6, relative to peak | 0 |
| V7 prism modes | 10 h: 0.990510, 0.992773, 1.398908; 30 h: 0.998011, 0.998064, 1.410436, in units pi c/L; no mode below 0.5 | 0 |
| V8 | board / tilted plate / prism bound ratios 1 / 1 / 1; PEC bounds 6.99188 / 8.07336 / 6.58323 ps; harness uses 0.985 of the bound | 0 |
| V9 | block-Jacobi conditions plate/prism: Cu 1.000512/1.000526; g eta0=0.5 2.90397/2.45393; Y=0 4.86391/3.70182; max actual-step increment error 1.18970e-11; all SPD | plate -1.55903e-5; prism -8.45339e-6 |
| V10 | Cu plate/bent modified-energy ratios 0.999692671/0.999698534; heat 1.94029e-7/1.91901e-7 J; physical-energy ratios 0.999489755/0.999554911; PEC physical drifts -2.00259e-4/-1.53273e-4 | Cu plate -1.24664e-4; bent -9.30595e-5; PEC 0 |
| V11 absolute | max normal T error 0.00602274; oblique T/R 0.0197971/0.0229830; PEC transmission 2.55577e-8; dielectric-interface TE/TM errors 0.00365281/0.00637558 | normal -5.43988e-4; oblique TE/TM -2.41286e-4/-2.31960e-4; PEC 0 |
| V11 refinement | transmission ratios 0.492225/0.497620; g eta0=4 reflection errors 0.00334608/0.00242177/0.00144206, orders 0.466414/0.747928, inside the restricted floor exception; PEC reflection order 2.21227 | transmission lambda/20,40,80: -2.42070e-4, -3.01828e-5, -3.77165e-6; reflection: -2.41286e-4, -3.02150e-5, -3.76840e-6 |
| V12 | TE copper-PEC differences lambda/20,40: 2.34043e-4/8.52497e-5; TM: 3.53842e-5/3.55363e-5 | TE -2.41286e-4/-3.02150e-5; TM -2.31960e-4/-2.90533e-5 |

V9 checks 20 random vectors per scene/material. The smallest raw normalized
inverse-order defect is -1.08399e-11, inside the dense FP64 reference's
`10 condition(S) eps` uncertainty; the even polynomial has the required
inverse order analytically. Maximum random-vector inverse error is
1.46598e-11. Energy comparisons use steps 2000 and 20000 after the pulse.
The unchanged spanning-copper regression gives far/near
1.14777e-10 / 5.76144e-11 for 35/70 um (ratio 1.99216), below the model
floors 1.63739e-7 / 8.21922e-8. The PEC regression gives far = 0,
near = 79384.390625. No post-source-phase hook is added; the field-update
hooks suffice. Foil files, native kernels and CPU-reference semantics are
unchanged. All 48 named cases are covered by targeted passing blocks (15
operator, 5 aligned/regression, 4 energy, 24 scattering); the two normal
transmission cases reuse the earlier passing selected-factor run. Final
refinement reruns use four CPU threads and differ from the calibration
tables in section 4 by at most 3.1e-8 in absolute error. CUDA was unavailable
under the required GPU-0 command, so native CUDA is not qualified here.
V13 and V15 remain Task 3 CUDA qualification.

## 10. Codex task breakdown

All tasks: `maxwell` env; only the named tests plus the listed regressions;
nothing outside thin sheets changes (no edits outside the listed files); V1
and V2 hold after every task. No legacy switches: the reconstruction, FSAI,
Chebyshev-on-`S` and dense `K` are deleted in task 2, not kept behind flags.

**Task 1. Compiler: chords and sheet faces.**
Files: `witwin/maxwell/compiler/thin_sheet.py`,
`tests/materials/thin_sheet/test_thin_sheet_runtime.py`,
`tests/materials/thin_sheet/test_thin_sheet_faceted.py`.
Do not touch: `fdtd/*`, `compiler/copper_foil.py`.
Add to `ThinSheetComplex` (copper only): per chord the active face, the two
sub-face rows, `o_{f,s}`, `l_c`, `w_c`, `tau_c`, `d_{c,s}`, `V_{c,s}`; per
sheet face the loop (chord ids, orientations, on-sheet trace ids), `A_P`,
`A_eff`, `ltilde_P`; the chord columns of the sub-face rows and the sheet-face
rows as a separate sparse block `K_c`. Keep `trace_rows` and today's `K` in
this task so the runtime still runs. Acceptance: V2, V5, V6, and the V7
structural part (explicit operator with chords held equals the PEC complex);
regression: `test_compiled_complex_never_mixes_sides`,
`test_subface_circulations_are_exact_for_linear_fields`,
`test_faceted_circulations_are_exact_for_linear_fields`.

**Task 2. Runtime: locally implicit chords with the foil two-port.**
Files: `witwin/maxwell/fdtd/thin_sheet.py`, `compiler/thin_sheet.py` (delete
`fit`, `reconstruct`, `path_integral`, `trace_rows` and the reconstruction
entries of `K`), `tests/materials/thin_sheet/*` (delete
`test_thin_sheet_copper_solver.py` and the `COPPER_SOLVER` uses in
`test_thin_sheet_admission.py` / `test_thin_sheet_cuda.py`; add the V9 tests
and the V7 prism mode test). Do not touch: foil files, native kernels,
`fdtd/cpu_reference.py` semantics. Implement section 5 (steps 1-5), private
`chord_e`, `chord_e_prev`, `chord_f`, `E_{X_c}^{n-1}`, the foil's per-trace
copper arithmetic, the even-polynomial Chebyshev with 2x2 block-Jacobi;
delete `_fsai`, the dense `K`, `COPPER_SOLVER` and the `COPPER_*` constants
that no longer apply; extend `thin_sheet_energy` with the chord terms and add
the modified energy for V10. Acceptance: V3, V4, V7, V8, V9, V10, V11, V12;
regression: `test_cavity_spanning_tilted_copper_leaks_only_through_the_copper_model`,
`test_cavity_spanning_tilted_pec_sheet_is_watertight`.

**Task 3. CUDA parity, cylinder, cost, documentation.**
Files: `tests/materials/thin_sheet/test_thin_sheet_cuda.py`,
`test_thin_sheet_faceted.py` (cylinder gates),
`scripts/dev/fdtd/thin_sheet_overhead.py` (drop `--copper-solver`),
`docs/plans/fdtd-3d-thin-sheet-runtime.md` (replace the copper-solve, cost and
copper-vs-PEC sections), `FEATURE_LIST.md` only if a user-visible limit
changes. Acceptance: V13, V14, V15; CUDA graph replay bit-identical to eager;
one-step CUDA vs CPU float64 patch parity <= 1e-12; regression:
`test_cuda_matches_cpu_reference`,
`test_cuda_axis_aligned_mesh_reproduces_native_planar_paths`.

**Task 4 (only if V15 fails). Fused patch kernel.**
One CUDA kernel per stage (H rows, explicit E rows with the `Q` correction,
chord Chebyshev step) reading native fields and private buffers through
precomputed index lists, replacing the gather / CSR / scatter sequence.
Files: a new thin-sheet kernel under `fdtd/native_cuda`, `fdtd/thin_sheet.py`.
Acceptance: V15; fused vs unfused patch step bit-identical in float64;
regression: the task 3 CUDA tests.

## 11. Open questions for the user

1. dt: a locking-free copper sheet runs at the PEC path's leapfrog bound.
   Relative to today's copper path that is dt x 0.973 on the PCB board
   normal, x 0.855 on the 3D tilted plate, x 0.785 on the rotated prism.
   Accept, or first make the PEC small-cut bound itself tighter (a separate
   piece of work on the PEC masses that would benefit PEC and copper alike)?
2. Is the Newmark (beta = 1/4) treatment of chord-coupled terms acceptable as
   a second time integrator inside the leapfrog patch? It is second order and
   energy-stable, but it adds O(dt^2) dispersion to chord-carried modes that
   plain leapfrog does not have.
3. The FIT mass rule (move the chord's dual sliver out of the Cartesian
   pieces) improves the PEC limit (prism (1,0): 0.9905 -> 0.9960 at 10 h,
   0.9980 -> 0.9993 at 30 h) but stiffens the explicit operator (2.7-8.5
   `lambda_Yee`). It would apply to the PEC path too. Worth a follow-up with
   the PEC small-cut work of question 1?

**User decisions (2026-10-30).**
- Q1, dt at the PEC leapfrog bound: accepted.
- Q2, Newmark beta = 1/4 on the chord-coupled terms only: accepted. Gates V7-V13 must report the added dispersion explicitly.
- Q3, FIT mass rule: deferred (item D2 below).
- Codex tasks 1-3 run in order. Task 4 runs only if V15 fails.

## 12. Deferred follow-ups

Parked on purpose. Each entry states what the item is, why it waits, what resumes it, and what it is expected to buy.

**D1. Tighter PEC small-cut leapfrog bound.**
- *What:* raise the explicit stability bound of the conformal PEC sheet path, whose small-cut masses and floors currently set dt, without reintroducing locking.
- *Why deferred:* it is separate work on the PEC masses. The chord basis only needs to match the existing PEC bound.
- *Resume when:* task 3 has landed (chord copper runs at the PEC bound).
- *Payoff:* it recovers the dt lost to Q1 for PEC and copper alike. Today the factors are x0.973 (PCB board normal), x0.855 (tilted 3D plate) and x0.785 (rotated prism) relative to the old copper path; the full gain is up to about 1.27x on the prism.

**D2. FIT mass rule.**
- *What:* move each chord's dual sliver out of the Cartesian pieces into the chord mass.
- *Why deferred:* it stiffens the explicit operator to 2.7-8.5 `lambda_Yee` and changes the PEC path too. It is only worth doing with D1, which has to absorb that stiffness.
- *Resume when:* together with D1.
- *Payoff:* a more accurate PEC (and therefore copper) limit. The prism (1,0) mode goes from 0.9905 to 0.9960 at 10 h and from 0.9980 to 0.9993 at 30 h.

**D3. Fused thin-sheet patch kernel (task 4).**
- *What:* one CUDA kernel per stage (H rows, explicit E rows with the `Q` correction, chord Chebyshev step) in place of the roughly 60 gather/CSR/scatter launches per step.
- *Why deferred:* it is needed only if V15 fails.
- *Resume when:* V15 fails, or a later profile shows the non-copper patch work above about 30 % of the step.
- *Payoff:* the non-copper patch work was 5 ms (6188 traces) and 19 ms (61k traces) per step eager on a contended GPU. Fusing it is the remaining lever once the copper term is local.

**D3 status (2026-10-31): TRIGGERED, deferred.**
- V15 was measured uncontended on an exclusive A6000 (GPU 0). It fails:
  - copper_patch, 6188 traces: 2.97x with the original copper model, 2.26x with the P-9 compressed modal model (gate 1.5x);
  - copper_board_60k, 61342 traces: 6.46x original, 3.02x with P-9 (gate 2.0x).
- Operator storage passes at 731-739 B per trace (limit 2000). P-9 cuts the board's modal state from 126 MB to 12.3 MB.
- Contention therefore did not explain the earlier misses, and task 4 is required.
- Per the user's standing decision of 2026-10-30, D3 (with the P-9 copper-kernel fusion folded in) waits in the deferred CUDA-migration backlog until the main features are complete. No ABI version is reserved for it yet.
- Evidence: <evidence-root>/timing-hour-20261031/SUMMARY.md.

**D4. Preparation cost of large sheets.**
- *What:* replace the Python cut-complex compiler and the leapfrog Lanczos over the sheet bounding box.
- *Why deferred:* it is a prepare-time cost only and does not affect the per-step cost or accuracy.
- *Resume when:* boards above about 60k traces become routine, or prepare dominates a workflow.
- *Payoff:* about 263 s of the 514 s prepare at 61k traces is the compiler, and 1297 s at 122k traces. The chord basis also removes the FSAI and Lanczos work on `S`.

**D5. Modal-state footprint.**
- *What:* reduce the copper modal state (M = 257 states for 35 um over 0.5-4 GHz, 4.1 KB per trace), e.g. a narrower declared band or a compressed modal realization shared with the foil.
- *Why deferred:* it becomes the largest per-trace term only after task 2, and it belongs to the foil's diffusion realization, which the chord work must not change.
- *Resume when:* V15 is limited by modal-state traffic.
- *Payoff:* up to about 4 KB per trace per step at 35 um.

**D6. Feeds on sheets.**
- *What:* ports, lumped elements and sources on a conformal sheet, following the gap-region design in `fdtd-3d-thin-sheet-runtime.md` ("Next stage: feeds on sheets").
- *Why deferred:* it needs the chord basis first. The gap's resistive trace enters the chord 2x2 blocks as one more passive term.
- *Resume when:* task 3 has landed.
- *Payoff:* driven tilted and curved printed antennas, validated by power balance and rotation invariance against the axis-aligned CopperFoil/LumpedPort path.

**D7. Remaining unsupported combinations.**
- *What:* the items in the runtime plan's "Out of scope" list:
  - curved facets under one cell, folds over 90 degrees, triple junctions;
  - sheet in CPML, periodic/Bloch boundaries, mapped coordinates;
  - dispersive, nonlinear or anisotropic media in the patch;
  - gradients, native batch, multi-GPU, resume;
  - side-aware observers.
- *Why deferred:* each is a separate capability. All are rejected explicitly today.
- *Resume when:* a user workflow needs one.
- *Payoff:* broader coverage. No accuracy or cost change to the chord basis.

**D8. Pre-existing test failures, owned by task 1.**
- `test_trainable_surface_mesh_vertices_are_rejected`: `SurfaceMesh` now rejects trainable vertices at construction.
- `test_vanishing_fold_converges_to_the_planar_sheet[pec|copper]`: the change saturates near 1e-5 at fold 1e-6 -> 1e-7, so the 5x ratio fails at the smallest pair.
- Both fail on 7a1af23b and after the master merge, on CPU. Task 1 fixes them.

# FDTD conformal thin-sheet runtime (tilted and faceted PEC and finite copper)

Status: CPU reference stage, native CUDA execution and piecewise-planar
(faceted) sheets implemented on branch `pcb/3d-thin-sheet`. On CUDA the patch
runs as device-resident Torch operators next to the native Yee kernels, inside
the captured CUDA graph.

## Goal and scope of this stage

Embed planar zero-thickness conductor sheets at an arbitrary tilt in the Cartesian
Yee grid, made of PEC or of finite copper (`CopperFoil`, including the electrically
thin resistive limit sigma*t), without staircasing and without collapsing the two
sides into one field. Execution goes through the ordinary
`Scene -> Simulation -> Result` path and returns an ordinary `Result`.

Declaration:

* `SurfaceMesh`, consistently oriented and manifold, made of one or more planar
  facets (coplanar edge-connected triangles) joined at straight fold lines
  with a turning angle of at most 90 degrees: tilted plates, bent or folded
  boards, closed prisms (faceted cylinders).
* `material=CopperFoil(...)` selects the new runtime automatically (this was the
  previously rejected "SurfaceMesh finite copper" case).
* PEC selects it only with `SurfaceMesh(..., sheet_model="conformal_two_sided")`;
  plain PEC meshes keep the existing mass-orthogonal projection path.
* `Simulation.fdtd(...)` on a CUDA scene (native kernels), or
  `Simulation.fdtd(..., backend="cpu_reference")` on a `device="cpu"` scene.

## Why not the earlier research design

The 2026-10-28 research (cut-cochain algebra, mixed-clock algebra, stepper v1/v2)
built a fully consistent polyhedral complex with non-diagonal per-cell Hodge
matrices, independent normal fluxes on each side and a local implicit
Crank-Nicolson patch on an integer clock coupled to the native half-step clock.
Its algebra is sound, but tiny cuts forced compensated (double-double) arithmetic
and 90-digit oracles to keep the mixed-clock energy gates, and it never ran on
Scene-prepared native background fields or returned a `Result`. The rejected
unsplit mortar (one field per Yee edge, sheet conductance smeared by
interpolation) leaked about a thousand times the physical copper transmission.

This stage keeps the two research conclusions that matter physically (two
independent sides; the passive copper two-port couples conductor traces across the sides)
and uses diagonal masses. The Cartesian fields remain on the native leapfrog
clock; copper chords use a locally implicit Newmark update (beta = 1/4) with
the foil's passive two-port. The implemented discretization and acceptance
evidence are specified in [the chord trace-basis plan](fdtd-3d-thin-sheet-trace-basis.md),
sections 3-5 and 9. That plan supersedes the reconstructed-trace copper method.

## Discretization

Plane `n.x = d`. Every grid node is classified below (-1), on (0, |phi| <= 1e-9 h)
or above (+1). Inside a local box around the sheet:

1. **Split faces.** A Yee face with nodes of both signs whose cut segment midpoint
   lies in the sheet outline is replaced by two sub-faces (one per side), each
   with its own magnetic flux. The native slot stores the -1 side, a private
   float64 buffer the +1 side.
2. **Split edges.** A pierced Yee edge bounding a split face becomes two
   segments with their own line integrals (native slot -1, private +1).
3. **On-sheet edges/faces.** Edges whose both nodes lie on the plane and inside
   the outline are two-sided traces (the axis-aligned foil case, and exact grid
   coincidences of tilted planes). Traces participating in a closed sheet-face
   loop are promoted to chords. The two sides have independent normal fluxes.
4. **Faraday** on every (sub-)face is the exact oriented circulation of its
   boundary pieces. The cut segment contributes `integral E_t dl` of its own
   side: zero for PEC; for copper an independent chord field on each side,
   multiplied by the chord length. Closed cut-cell loops contribute independent
   per-side sheet-face fluxes through the chord stiffness operator.
5. **Ampere** uses the transpose of the same circulation matrix `K` with
   diagonal masses, `eps V_e dE_e/dt = sum_f K[f,e] ltilde_f H_f`.
   Electric masses partition every Yee dual box exactly between the sides
   (exact box/half-space clipping): a segment or regular edge keeps its own
   side's part, the part beyond the sheet moves to the nearest same-component
   unknown on that side, and on-sheet traces split their box (reproducing the
   existing foil half widths). Each segment keeps at least a quarter of its
   prism volume with the pair total conserved; held PEC-wall edges and boxes
   straddling the sheet outline donate nothing (the rim stays O(h)). This removed an O(h) capacitive position bias
   (offset normal-incidence reflection error 0.024 -> 0.009). Magnetic
   sub-faces keep dual length times their area (Dey-Mittra convention).
6. **Copper.** Each trace carries the existing passive diffusion two-port
   (`compile_copper_response`: D, B, rates incl. the zero-rate inductive state)
   acting on `(u_-, u_+)/l` with in-sheet width `w = dual_a * sqrt(1 - n_a^2)`
   for a cut segment in a face of normal axis `a`. These widths are the unique
   diagonal 2D Hodge weights for the three line families a plane cuts from the
   grid (`sum_a (1-n_a^2) t_a t_a^T = I_2`), so sheet power integrates
   `E_t . Y E_t` consistently. Exact grid-aligned trace pairs use the existing
   per-edge 2x2 foil update. Cut chords use the locally implicit solve below.

### Copper chord solve

Every active cut face owns two tangential electric unknowns, one per side.
Faraday reads these chords directly and Ampere uses its transpose. The copper
two-port is the only coupling between the sides. The Cartesian masses and
explicit curl are the PEC complex's; holding chords at zero reproduces that
operator exactly, removing the reconstruction's copper-to-PEC locking.

Chord mass is `0.5 l w max(d, h/16) eps`, with per-side centroid distance `d`.
Each closed cut-cell loop has two independent sheet-face fluxes. They enter
only through `Acc`; no sheet-face magnetic buffer is advanced. Their area
floor is `h^2/4`. These additive chord masses are an accuracy choice, not a
stability requirement; calibration and its restricted resistive-reflection
floor exception are recorded in the trace-basis plan, section 4.

Newmark beta = 1/4 treats every chord-coupled term implicitly. Its constant
SPD system is

`S = Mc/dt + G/2 + area Yeff/2 + dt/4 (Acc - Acx Q)`.

`Q` gives the Cartesian electric correction. Preparation builds sparse
`Acc`, `Acx`, `Q`, `S` and the sub-face magnetic correction. A 2x2 block-Jacobi
inverse square root preconditions `S`. Lanczos bounds widened by 2% select
the smallest Chebyshev degree with `(b/a - 1) eps_k^2 <= 1e-12`. Applying the
residual polynomial twice gives `0 < P <= S^-1`, equivalent to adding a
positive chord mass `dt (P^-1 - S)`. The modified energy therefore remains
passive under the PEC complex's explicit leapfrog bound. There is no global
reconstruction, dense reference solve, solver switch or approximate-inverse
factorization.

The foil's modal state, per-trace arithmetic and diffusion realization are
unchanged. At exact grid alignment there are no chords or sheet faces, and
the foil-matched 2x2 inverse and rhs matrix agree exactly in the Task 2 tests.
The chord integrator adds second-order temporal dispersion: at frequency `f`
and timestep `dt`, the chord-only Newmark-minus-leapfrog symbol is
`2 [atan(pi f dt) - asin(pi f dt)]` rad/step. This is not a whole-scene
frequency shift; held-chord PEC modes have zero added dispersion. CPU
physics, passivity and solve measurements are in section 9 of the trace-basis
plan; native CUDA qualification is recorded below.

### Faceted (piecewise-planar) sheets

The mesh is split into planar facets; shared mesh edges between facets are fold
lines (convex or concave, from the side of the + normal). A planar mesh is a
single facet and takes the single-plane path; a straight cut's trace width is
exactly its strip width (tested bit for bit).

* Level function and sides: every node takes the signed plane function of its
  nearest facet (distance to the closed facet polygon). Its zero set is the
  surface, continued beyond the outer rim by the boundary facets' planes, and
  its sign is consistent across folds (both facets agree in the wedge regions).
  Equidistant facets are broken toward the larger |plane value|: at an exact
  90-degree fold a node on one facet's plane beyond the fold is as far from
  that facet (in-plane) as from the other (off-plane) and is not on the sheet.
  The outline test counts a point beyond its facet as inside only when it is
  strictly nearer a fold than the rim (the fold line does not continue past the
  rim).
* Pierced edges: the crossing is the root on a candidate facet plane (the
  endpoints' nearest facets and their fold neighbours) that lies on the zero set;
  an edge crossing the sheet twice is rejected.
* Cut faces: a face whose two zero points lie on different facets is cut by the
  two-piece polyline through the point where their fold line pierces the face;
  the fold point is inserted into both sub-face polygons. Faces cut by
  non-adjacent facets (facets narrower than about a cell) are rejected.
* Masses: a dual box seeing several facets is clipped exactly by their wedge
  (intersection of the - half-spaces around convex folds, complement of the
  intersection of the + half-spaces around concave folds; convex polyhedron
  clipping). The facets are those of 27 box samples plus both facets of every
  fold segment that meets the box (exact slab test), so a fold crossing a box
  corner is not missed. Boxes seeing facets not joined by folds of one kind are
  rejected.
* Copper: one independent chord pair per cut face. A fold chord follows the
  two-piece polyline; its strip width is the length-weighted mean. Bent
  cut-cell loops supply the sheet-face stiffness on each side. Orphan-mass
  receivers cannot reach across a non-adjacent part of the sheet.

## Stability (CFL) treatment

`(K, K^T)` with positive diagonal masses is skew-adjoint in the electromagnetic
energy, so leapfrog is stable iff `dt^2 lambda_max(M_e^-1 K^T L_h K) <= 4`.
For copper this bound applies only to the explicit Cartesian operator, which
is exactly the PEC complex's. The locally implicit chord solve preserves
modified-energy passivity at that bound. Two measures keep the explicit
`lambda_max` near the Yee value:

* **Stiffness-matched sub-face mass.** A sub-face whose curl-curl diagonal
  `(ltilde/(mu A)) sum K^2/(eps V)` would exceed `kappa` times its parent Yee
  face's diagonal uses the area `A_eff` that meets it (kappa = 2; absolute floor
  0.02 of the face). The diagonal is measured on the sub-face's own edge pieces
  for PEC and copper alike. Implicit chords are excluded from this floor.
  Only genuinely small cuts are affected, with an O(h) error confined to them;
  there is no geometric snap or topology change.
* **Computed bound.** At preparation the largest eigenvalue of the local
  operator (whole compilation box, Lanczos) is computed and `dt` is tightened to
  0.985 of `2/sqrt(lambda_max)` when necessary (`limiting_factor =
  "thin_sheet_cfl"`); an explicit `TimeConfig.dt` above it is rejected.

## Integration in the compile -> runtime path

* `compiler/thin_sheet.py`: ownership (`owns_thin_sheet`), plane extraction,
  cut complex, masses, Cartesian and chord curl blocks, sheet-face loops.
  Owned structures are skipped
  by `compile_copper_faces`, `compile_planar_pec` and the PEC cut-surface
  projection, so each physical structure has exactly one owner.
* `fdtd/thin_sheet.py`:
  * `prepare_thin_sheet_time_step` runs in `init_field` right after the wire
    runtime and before update coefficients, so the tightened dt is final.
  * `initialize_thin_sheets` runs after all other initializers. It rebuilds
    the patch E coefficients from side-resolved media (below), rejects patch
    edges with PEC-suppressed native updates, rejects any patch node within one
    node of the absorber mask (`_absorbing_region_node_mask`; CPML does not
    alter the update coefficients, so a coefficient check cannot see it), and
    builds gather/scatter plans and sparse operators.
  * In `_field_update_block`: save patch H, native H update, patch H rows,
    native E update (and existing foils/SIBC), patch E rows (explicit) and the
    copper chord solve and its Cartesian E / sub-face H corrections.
    Exterior Yee unknowns never read a sub-face or segment.
  * Device execution: all patch operators (CSR circulation rows, gather and
    scatter plans, chord solve operators, private side and chord unknowns,
    modal copper state, heat accumulator) live on the solver device in float64;
    the fixed sequence of device operations is CUDA-graph
    capturable. The private state is registered with `graph_state_tensors`, so
    capture warmup is undone. Graph replay and eager execution are
    bit-identical.
* `fdtd/cpu_reference.py`, `fdtd/cpu_profile.py`: explicit CPU kernel profile
  (`backend="cpu_reference"`), parity-tested against the frozen Torch reference.

## Ownership and boundaries

* CPML: the patch must lie outside the absorber, checked on the absorber node
  mask grown by one node.
* Media: each side's permittivity/conductivity is read from same-side unknowns
  one to two cells from the plane and must be homogeneous there; every
  definite-side unknown within one cell receives its own side's medium (so a
  pierced segment on the + side of copper-on-FR4 carries the FR4 value, not an
  average). A deviation on an unknown whose dual box lies wholly on one side,
  or a near value outside the two side media, means a third material in the
  cut cells and is rejected.
* Sources (electric and magnetic) overlapping the patch, surface-impedance
  boundaries anywhere in the scene, and point monitors on split unknowns:
  the first two are rejected; monitors warn that they read the -1 side only.
* Domain walls: PEC walls hold their tangential E (fixed zero); PMC walls use
  the native half dual cell. Sheets may cross PEC/PMC walls (used by the
  quasi-2D validation scenes).
* Axis-aligned CopperFoil and PEC cut-surface constraints overlapping the
  patch, and any ports, lumped elements or wires in the scene are rejected.
* Separate sheets sharing a patch edge are rejected.

## Validation (CPU, see tests/materials/thin_sheet)

Current CPU qualification is recorded in [the trace-basis plan, section 9](fdtd-3d-thin-sheet-trace-basis.md#9-validation-plan-gates-declared-before-implementation)
(Task 2, `CHORD_MASS_FACTOR = 0.5`). It replaces the reconstructed-trace
measurements previously listed here. In particular:

| Check | Measured | Gate |
| --- | --- | --- |
| Copper plate / bent board modified energy, 2000 -> 20000 steps | ratios 0.999692671 / 0.999698534; heat 1.94029e-7 / 1.91901e-7 J | non-increasing, heat > 0 |
| PEC physical-energy drift, plate / bent board | -2.00259e-4 / -1.53273e-4 | <= 1e-3 |
| Normal transmission / oblique transmission and reflection, maximum absolute error | 0.00602274 / 0.0197971 / 0.0229830 | 0.01 / 0.03 / 0.03 |
| Transmission refinement ratios, lambda/20 -> /40 -> /80 | 0.492225 / 0.497620 | <= 0.6 |
| g eta0 = 4 reflection orders | 0.466414 / 0.747928; finer errors within twice the independent PEC floor | order >= 1 or the restricted floor exception in the trace-basis plan |
| 35 um Cu minus PEC reflection, TE, lambda/20 and /40 | 2.34043e-4 / 8.52497e-5 | <= 1e-3 |
| same, TM | 3.53842e-5 / 3.55363e-5 | <= 1e-3 |

The last two rows have chord-only Newmark-minus-leapfrog symbols (rad/step)
TE -2.41286e-4 / -3.02150e-5 and TM -2.31960e-4 / -2.90533e-5.
PEC has zero added chord dispersion. All V7-V12 dispersion measurements,
including resistive and energy scenes, are in the linked evidence table.

## Native CUDA execution (tests/materials/thin_sheet/test_thin_sheet_cuda.py)

Task 3 measurements (2026-10-30), GPU 0, captured native CUDA graph,
`CHORD_MASS_FACTOR = 0.5`. Only the named qualification tests were rerun.

| Check | Measured | Gate |
| --- | --- | --- |
| CUDA vs CPU reference, 400 steps, PEC series / fields | 4.07690e-6 / 6.59998e-7 | <= 1e-5 and <= 3x no-sheet baseline |
| same, 2 um Cu series / fields / heat | 4.95939e-6 / 4.79999e-7 / 2.61125e-7 | fields <= 1e-5 and <= 3x baseline; heat <= 1e-5 |
| No-sheet float32 baseline, series / fields | 4.51292e-6 / 2.98630e-7 | reference |
| One patch step, all private float64 state, PEC / Cu | 1.73216e-16 / 2.62776e-16 | <= 1e-12 |
| One patch step, float32 field rows, PEC / Cu | 0 / 0 | <= 2.4e-7 |
| Graph vs eager, 300 steps, series, fields, heat and all persistent private state | bit-identical (max difference 0) | identical |
| Axis-aligned mesh vs native foil / PEC, 260 steps | 4.23200e-6 / 1.38961e-6 | <= 2e-5 |
| V14, cavity-spanning 35 / 70 um Cu far/near | 1.147766e-10 / 5.761476e-11, reduction 1.992139 | <= model floors 1.637393e-7 / 8.219221e-8; reduction >= 1.6 |

The patch parity check seeds nonzero independent chord fields, previous-step
fields, chord forcing and diffusion state on both devices. It includes the
implicit rows directly at the original 1e-12 gate. The longer CUDA/CPU
comparison measures float32 Yee accumulation; neither the native kernels nor
the thin-sheet runtime needed a change for this qualification.

Rejected on CUDA as on the CPU profile: periodic/Bloch boundaries, native batch
(`validate_native_solver`), trainable copper parameters (forward-only),
checkpoint/resume, multi-GPU sharding, mapped coordinates, ports, wires,
lumped elements, sources on the patch.

### Per-step cost (scripts/dev/fdtd/thin_sheet_overhead.py)

Task 3 measured both required scenes on GPU 0 (RTX A6000), 2026-10-30.
The script no longer accepts `--copper-solver`; it measures the chord runtime.
Both solvers use the same grid and dt. After one public initialization step,
the script captures the native field-update block, warms up 20 steps, then
uses CUDA events for 200 replays per sample. Plain and sheet samples run back
to back, alternating order across three pairs. Preparation, capture, source
injection and monitor sampling are outside these field-step timings. Ratios
below are medians of the paired ratios; times are separate medians.

| Scenario | traces | grid | plain ms/step | sheet ms/step | paired ratio / V15 gate | chord bytes/trace | prepare s | peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copper_patch, 35 um, 24 x 20 mm | 6188 | 149^3 | 1.831644 | 5.480965 | 2.990644 / <= 1.5: missed | 738.535 | 75.532 | 1.538 |
| copper_board_60k, 35 um, 90 x 80 mm, normal (0.04, 0.025, 1) | 61342 | 241 x 221 x 101 | 3.647140 | 22.849509 | 6.265049 / <= 2.0: missed | 731.196 | 253.801 | 3.192 |

All samples used captured graphs for both paths. GPU load was recorded with
`nvidia-smi` immediately before and after every pair:

| Scene / pair | time interval (local) | plain ms | sheet ms | ratio | GPU utilization | concurrent compute jobs: PID / MiB |
| --- | --- | --- | --- | --- | --- | --- |
| patch / 1 | 11:36:44.901-11:36:46.448 | 1.831644 | 5.475486 | 2.989383 | 100% before/after | 1174634 / 3576; 1190517 / 3302 |
| patch / 2 | 11:36:46.523-11:36:48.067 | 1.832704 | 5.480965 | 2.990644 | 100% before/after | same |
| patch / 3 | 11:36:48.143-11:36:49.686 | 1.821983 | 5.483259 | 3.009501 | 100% before/after | same |
| board / 1 | 11:41:04.932-11:41:10.318 | 3.647140 | 22.849509 | 6.265049 | 100% before/after | 1174634 / 3576; 1190517 / 3214 |
| board / 2 | 11:41:10.396-11:41:15.784 | 3.647037 | 22.875791 | 6.272431 | 100% before/after | same |
| board / 3 | 11:41:15.858-11:41:21.239 | 3.656340 | 22.848604 | 6.249036 | 100% before/after | same |

Total device memory was 9111 MiB (patch) / 10585 MiB (board), including this
measurement process (2102 / 3664 MiB); temperature was 85-86 C. A preparation
sample showed the two concurrent jobs at 56% and 42% SM utilization. These
are contended measurements: the time gates are missed on this GPU state,
and the plan's **uncontended V15 remains unqualified**. No uncontended control
was available, so the misses cannot be attributed solely to contention. Task 4
was not started.

Chord operator storage passes the <= 2000 B/trace gate: 4,570,056 B / 6188
and 44,853,000 B / 61342. This counts `Acc`, `Acx`, `Q`, `S`, `h_chord`,
`R` (values, column indices and CSR row pointers), mass, conductance, area,
held masks and the effective 2x2 admittance. It is stored bytes, not repeated
per-step traffic. Modal state is separate: 12,722,528 / 126,119,152 B
(2056 B/trace, the declared 1-4 GHz band). Both scenes use Chebyshev degree 3;
block-Jacobi spectra are [0.999310, 1.000869] / [0.997519, 1.002408].

At the 2.4 GHz source center the timesteps are 0.796392709 / 0.686133707 ps,
and the added chord-only Newmark-minus-leapfrog symbols are -2.16501672e-7 /
-1.38454381e-7 rad/step. The cost estimates in the trace-basis plan, section 6,
are estimates; these measurements supersede them for this shared GPU state.

## Faceted sheets (tests/materials/thin_sheet/test_thin_sheet_faceted.py)

Tolerances declared before the runs. Scenes: an 8-facet bent board
(`SurfaceMesh.bent_rectangle`, 0.06 x 0.05 m, R = 0.04 m, facets 1.5 cells), a
spanning V sheet (one fold), a closed 12-facet prism spanning the cavity, all at
h = 5 mm; cylinder scattering at h = lambda/20, /40, /80 (1 GHz).

| Check | Measured | Gate |
| --- | --- | --- |
| Linear-field circulation, bent / vee / prism (copper, incl. fold faces) | 6.6e-14 / 6.3e-14 / 7.8e-14 | <= 1e-9 |
| Fold-wedge dual volume vs exact column integration | <= 1e-4 of box | <= 1e-4 |
| Planar trace widths vs exact strip widths | bit-identical | identical |
| L bracket at exactly 90 deg on grid lines: on-sheet edges off the surface (PEC, Cu) | 0 (was 178 phantom pairs) | 0 |
| Closed PEC box on grid lines (90 deg folds, three-facet corners): interior | exactly 0 (was rejected: patch reached its box) | == 0 |
| Fold crossing a dual-box corner, - side volume vs column integration | <= 1e-4 of box | <= 1e-4 |
| Spanning vee / prism rows read one side only (PEC, Cu) | yes | yes |
| 1e-4 rad fold vs planar sheet, PEC (600 steps) | 1.7e-3 | <= rigid 3 um shift: 8.7e-3 |
| Bent PEC board, energy drift 2000 -> 20000 steps (CUDA) | -1.5e-4 | <= 1e-3 |
| Closed PEC prism interior / spanning PEC vee far side | exactly 0 / exactly 0 | == 0 |
| 32-facet PEC cylinder (ka = pi) vs Bessel series, TE | 0.017 / 0.0034 / 0.0012 (order 1.9) | 0.10 / 0.04 / 0.015, order >= 1.5 |
| same, TM | 0.016 / 0.0042 / 0.0015 (order 1.7) | same |
| 35 um Cu cylinder, lambda/40: series error TE / TM (Task 3) | 0.00342931 / 0.00425045 | <= 0.015 / 0.01 |
| 35 um Cu vs PEC cylinder, lambda/40, TE / TM (Task 3) | 3.60883e-5 / 6.26972e-5 | <= 1e-3 |

Copper near the PEC limit now passes V13 with the chord basis. At lambda/40,
the matching PEC series errors are 0.00341987 (TE) and 0.00421793 (TM). The
copper cylinder target is an ordinary passing test; the expected failure and
the relaxed post-failure regression gate have been removed. No native kernel
or runtime change was required in Task 3. Earlier faceted structural and PEC
measurements above are retained; Task 3 reran the copper cylinder comparisons
and their lambda/40 PEC controls only.

At 1 GHz the TE / TM timesteps are 9.88635992317 / 9.73298578124 ps. Added
chord-only Newmark-minus-leapfrog dispersion is -2.99539619e-5 /
-2.85815978e-5 rad/step, respectively; the PEC controls have no chords and
zero added dispersion. The spatial and temporal error contributions are not
separated by this scattering measurement.

The scattered phasor is total minus incident run on the same grid and dt,
normalized by the incident phasor at a reference node; the polygon uses the
area-equivalent radius and an off-node centre; observers sit on exact sample
nodes. The remaining PEC error is dominated by grid dispersion (second order).

### Reflection diagnosis (the earlier "0.06 phase error")

Point monitors sample the nearest Yee node. The first harness snapped its front
and back observers independently and compared the front reflection with the
incident field at the snapped back point, which is not the mirror of the
snapped front point. The resulting phase error k * (mirror mismatch) reproduced
every measured case (0.06 at lambda/20, 0.023 or 0.092 at lambda/40 depending
on sheet length and node alignment) and changed sign with observer distance; an
independent review reached the same prediction. The harness now places
observers on exact sample positions (grid extents are multiples of h) and
interpolates the incident plane wave at the off-node mirror x with its measured
numerical phase step. No transpose, orientation or clock error was found; a
linear-field test confirms every sub-face circulation integrates linear fields
to 8e-14, and the small-cut stiffness setting does not change the result.

After the fix, PEC TE at 30 deg, 1 GHz, observer 0.5 wavelength in front:

| h | abs(R) | arg(-R) | implied shift | predicted Yee dispersion phase |
| --- | --- | --- | --- | --- |
| lambda/20 | 0.9963 | -0.0057 | -0.0105 h | -0.0057 |
| lambda/40 | 0.9988 | +0.0005 | +0.0017 h | -0.0014 |
| lambda/80 | 0.9997 | +0.0007 | +0.0049 h (0.018 mm) | -0.0004 |

(diagnostic runs with the intermediate code; the suite value at lambda/20 with
the final partition is 0.0060). The residual phase at lambda/20 equals the
discrete anisotropy between the incident (x) and reflected directions and
doubles when the observer distance doubles; the remaining sheet-position offset
is below 0.02 mm at every resolution.

## Sheet gap feeds (D6)

An existing `LumpedPort` with both terminals on one sheet becomes a resistive
sheet gap feed. Its feed axis must be Cartesian and lie in the supporting
facet; non-Cartesian feed directions are rejected. The sheet outline must cover
the full-width rectangular gap. The gap and one-cell longitudinal margins must
lie on one planar facet; folds farther along the arms are allowed. The gap must
span at least one cell, and widths below two cells warn.

Gap copper is replaced by a passive conductance and series drive in the chord
Newmark solve. Partitioned quadrature supplies one longitudinal series branch
per transverse column. The native port stepping, observations, `PortData` and
checkpoint path carry the feed state. Detached checkpoint/resume is admitted
for sheet scenes containing a sheet gap feed; no-feed scenes retain their
explicit checkpoint guard.

Port voltage is the conductance-weighted column voltage, and port current is
the sum of column currents. `PortData.metadata["sheet_gap"]` retains column
spectra, exact `field_accepted_power`, and redistribution `variance_power`.
F3 uses exact field-side accepted power; variance power is a diagnostic and is
not added again as a load. Manually accumulated pulse spectra use `dt*f` for
each bin when no explicit reference frequency is selected.

F1 narrow-strip equivalence and F4 modified-energy passivity are qualified.
The F2 antenna rotation gate fails on the prescribed tilted dipole, and F2b
fails impedance self-convergence at the common resonance and upper band edge.
Its lambda/60-to-lambda/80 D_max change passes at 0.01641 dB. F3 passes at
lambda/60: worst exact field-side power residuals are 0.02849% (tilted dipole),
0.001689% (tilted patch), and 0.015023% (curved dipole), below the unchanged 1%
gate. Passing planar scattering or power-balance gates does not qualify driven
antenna rotation invariance.
Measurements, unchanged acceptance thresholds, F2b curved self-convergence and
F3 power-balance results are recorded in
`docs/assessments/thin-sheet-feeds-validation-2026-11.md`.

## Out of scope for this stage

* Smoothly curved meshes whose facets are narrower than about a cell, folds
  sharper than 90 degrees, vertices where three or more facets meet inside a
  dual cell, multiple sheet parts per cut cell, sheet junctions/contacts, holes
  as true topology (only the outline test is used).
* Sheet intersecting CPML, periodic/Bloch boundaries, mapped coordinates.
* Ports other than local sheet gap feeds, reactive terminations, circuits or
  embedded networks on sheet feeds, wires, sources on the patch, TFSF.
* Dispersive, nonlinear, magnetic or anisotropic media in the patch.
* Gradients/adjoint, native batch, multi-GPU, checkpoint/resume without a sheet
  gap feed (rejected on both backends).
* Fused thin-sheet patch kernels and reduced modal-state traffic; the current
  patch is a sequence of device-resident gather, sparse and scatter operators.
  Measured cost and the V15 gates are recorded above.
* Side-aware observers: a monitor within one cell of the sheet reads the native
  slot (side -1) of split unknowns; place observers at least two cells away.
* Rim treatment beyond the face-midpoint outline test (rim is resolved to O(h)).
* Second-order accuracy is not guaranteed: masses and small-cut treatment are
  first-order local, although the measured reflection/transmission errors
  converge at order 2-3 in the tested cases.
* Guard census: capability and input-admission boundaries are recorded in
  `docs/reference/fdtd-capability-guards.json`. The D6 reconciliation includes
  five capability raises and nine configuration raises.

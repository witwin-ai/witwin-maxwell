# Conformal thin-sheet feeds (D6): lumped ports and sources on tilted or faceted sheets

Status: spec, 2026-10-31. Started by user decision 2026-10-31 ("D6也开始做", i.e. "start D6 too").
Builds on the chord trace basis (`fdtd-3d-thin-sheet-trace-basis.md`, merged e9fadab3) and on the original gap-feed sketch in `fdtd-3d-thin-sheet-runtime.md` ("Next stage: feeds on sheets").

Task 1 compiler metadata is implemented (2026-10-31): 27 CPU feed geometry /
admission cases and 21 unchanged structural cases pass, including all four
historical PEC operator hashes. Runtime feeds and port binding remain Task 2.

## Goal

A tilted or faceted printed antenna on a conformal `SurfaceMesh` sheet, PEC or CopperFoil, can be driven and measured like the axis-aligned CopperFoil + `LumpedPort` path. That covers S11, Z_in, accepted/radiated power, efficiency and pattern. Today any source or port overlapping a thin-sheet patch is rejected (`fdtd/thin_sheet.py`, "A source overlaps a conformal thin-sheet patch").

## Public API

No new public class. An existing `mw.LumpedPort` whose `positive` and `negative` terminals both lie on the same conformal sheet (within the sheet's roundoff, both on the sheet surface) becomes a **sheet gap feed**:

- The gap is the strip of sheet between the two terminals: length `g = |positive - negative|` along the in-sheet feed direction `d`, and width `w` equal to the local conductor width of the sheet outline across `d` at the gap midpoint.
  - The sheet must exist in the gap, i.e. the outline covers it. Like the axis-aligned path, the gap carries the feed instead of copper.
  - If the outline does not cover the gap, the port is rejected with a clear message.
- The port's existing `impedance` and `reference_impedance`, and `PortExcitation` / `source_time`, keep their current meaning. Port V/I, S-parameters, `available_power` and `excitation_threshold` come out of the existing `PortData` path unchanged.
- Out of scope, and still rejected explicitly:
  - wave/mode ports on sheets;
  - terminals on two different sheets;
  - circuits/MNA or nonlinear/modulated loads on a sheet;
  - feeds under complex (Bloch) fields;
  - adjoint;
  - feeds at an open end along `d`: the conductor must continue at least one
    cell beyond each terminal along `d`. Rejections name the positive or
    negative terminal at the rim. The transverse ends defining `w` coincide
    with the strip edges by construction; that contact is allowed;
  - a fold crossing the full-width gap or its one-cell margin along `d` on
    either side. This whole region must lie on one planar facet. Folds
    elsewhere on the arms are allowed (including the F2b cylinder).
- **Resolution admission**:
  - Reject a gap whose length `g` spans less than one chord (one cell along `d`), with a message naming `g`, the local cell size and the fix (refine, or lengthen the gap). Otherwise a sub-cell gap would silently feed a neighbouring chord pair.
  - Warn when the width `w` is resolved by fewer than 2 cells.

## Discretization

Within the chord basis, the gap's cut cells carry no copper two-port. Their chord pairs (and on-sheet trace pairs) instead become a **resistive feed sheet with a series source**, following the original gap-feed sketch:

- Sheet impedance `R_s = R * w / g` (R = port resistance). Series EMF per unit length `V(t)/g` along `d`.
- Each gap chord tangential to `d` gets the feed current `J = (E_t·d - V/g) * g / (R w)` per unit width. It is applied to the two-sided **average** of the chord pair, because the gap conductor is electrically thin and the feed couples both sides equally. Chords normal to `d` inside the gap carry no feed term and no copper (open).
- The feed term enters the chord 2x2 local implicit system as one more passive conductance block, `(w_c l_c / R_s) * [1 1; 1 1]/4` on the side pair, plus a known right-hand side from `V(t)`. So:
  - the chord Newmark solve, the block-Jacobi Chebyshev and the stability argument are unchanged (an extra PSD conductance);
  - CFL is unchanged.
- Port V is the reconstructed line integral of the chord-average tangential E along `d` across the gap. Port I is the total feed current through the gap (`sum` of `J * w_c`). These are the same two quantities the axis-aligned `LumpedPort` uses (line-integral V, branch current I), so field/circuit power duality holds.
- PEC sheets: the gap is a non-PEC window in the sheet, carrying the same resistive feed sheet. PEC chords elsewhere stay held at zero.

## Integration points

- `compiler/thin_sheet.py`: gap classification (gap cut cells, gap chord set with `d`, `w`, `g`, side volumes); the feed-term geometry.
  Task 1 records `ThinSheetComplex.feeds` only; existing copper/PEC arrays
  and the runtime port/source guards stay unchanged. Gap cells have a
  positive-area intersection with the rectangular gap (both incident cells
  at exact grid-face coincidence). Chords and on-sheet trace pairs use the
  existing midpoint outline convention: their midpoints lie between the
  terminal planes and within the transverse strip edges. For driven chords
  exactly on a terminal plane, include the lower Cartesian endpoint only,
  independent of port polarity, so a one-cell gap always owns one chord along
  `d`. Perpendicular chords on a terminal plane belong to the arm. Metadata
  marks perpendicular chords open and records signed projections along `d`
  for driven chords. The existing `LumpedPort` still requires a Cartesian
  `AxisPath`; the supporting facet
  must contain that axis. No new port geometry API is introduced here.
- `fdtd/thin_sheet.py`: feed conductance in the chord blocks; the RHS from the port drive; V and I extraction into the port runtime buffers; CPU reference mirror (`cpu_reference` backend).
- `fdtd/ports.py`: bind a sheet-gap `LumpedPort` to a thin-sheet feed runtime instead of the native edge-current runtime. The drive goes through `stepping._port_step_block` (graph path) like every other port. Lift the overlap rejection ONLY for a sheet gap feed.
- `checkpoint.py` / resume: feed state (any accumulated current) is captured and restored.
- No CUDA kernel or ABI change in D6. The feed is torch ops inside the existing captured thin-sheet patch. Kernels come with D3.

## Validation gates (declared before any comparison)

| Gate | Test | Pass |
| --- | --- | --- |
| F1 axis-aligned equivalence | sheet gap feed on an axis-aligned SurfaceMesh sheet vs the existing CopperFoil + LumpedPort path, same geometry, PEC and 35 um Cu | Z_in and S11 rel diff <= 2e-3 over the band; port V/I time traces rel L2 <= 2e-3 |
| F2 rotation invariance | half-wave printed dipole with arms and feed along x, on sheets tilted about x (normals (0, 0.025, 1) and (0, sin 30 deg, cos 30 deg)), vs the same dipole axis-aligned (F1 path), lambda/40 and lambda/60. The feed axis must be Cartesian (task-1 admission), so the tilt is about the dipole axis; non-Cartesian feed axes are out of D6 scope and rejected | Z_in rel diff <= 3% at lambda/40, decreasing at lambda/60; D_max within 0.1 dB; resonance shift <= 0.5% |
| F2b curved coverage | the same dipole on a faceted cylindrical sheet (axis y; feed on the top facet with d along x, arms wrapping), three grids (lambda/40, lambda/60, lambda/80) | self-convergence: abs(Z60 - Z40) > abs(Z80 - Z60) for Z_in (complex, at resonance and band edges); D_max change <= 0.1 dB between lambda/60 and lambda/80; F3 power balance on the lambda/60 case |
| F3 power balance | driven tilted dipole and patch, closed flux box | abs(P_acc - P_rad - P_Cu - P_load) / P_acc <= 1% |
| F4 passivity | loaded and unexcited feed (R > 0), 20000 steps | the scheme's modified discrete energy (leapfrog/Newmark invariant) non-increasing: max <= initial * (1 + 1e-12), with strict decay for R > 0 |
| F5 CPU/CUDA parity | `cpu_reference` vs CUDA, float64, one step and 200 steps | one step <= 1e-12; graph replay == eager bitwise |
| F6 no-feed identity | every existing thin-sheet test and pin with no sheet feed | bit-identical to master (PEC operator hashes, CUDA parity, leakage gates) |
| F7 resume | checkpoint mid-run and resume | bit-equal to uninterrupted, port V/I included |

F1's 2e-3 compares the chord basis against the axis-aligned Yee CopperFoil path. If the measured difference turns out to be a discretization difference rather than a bug, the gate is NOT loosened silently: report the measured number and the reason (for example, which edges the two paths treat differently) for a coordinator decision.

F2's reference is the existing axis-aligned path, which is already validated (CopperFoil free-edge model, line-accuracy and antenna convergence evidence). F2 therefore measures the conformal feed plus discretization error, not absolute antenna accuracy.

## Codex tasks (sequential, one worktree: `.worktrees/thin-sheet-feeds`, branch `pcb/thin-sheet-feeds`)

1. **Compiler**: gap classification and feed geometry, with admission and rejection messages. Gates: F6 structural (no-feed complexes unchanged); gap chord set and `w`/`g` unit tests on plate and faceted meshes.
2. **Runtime + port binding**: CPU reference first, then CUDA. Feed conductance and RHS, V/I extraction, `_port_step_block` drive, checkpoint. Gates: F4, F5, F6, F7, F1.
3. **Validation**: F2, F2b and F3, plus docs:
   - `fdtd-3d-thin-sheet-runtime.md`: replace "Next stage: feeds on sheets";
   - `pcb-fdtd-support-matrix.md`: the SurfaceMesh rows;
   - `FEATURE_LIST.md`: the user-visible capability.
Fast tier per task (batched validation policy). F2/F3 GPU runs on GPU 0 within shared-memory limits.

## ETA

About 2-3 Codex days: task 1 ~0.5 day, task 2 ~1-1.5 days, task 3 ~0.5-1 day. Plus ~4-6 GPU-h for F2/F3 on GPU 0.


## Gate notes from task 2 (2026-10-31)

- **F4 metric (corrected).** The physical sum E^2 + H^2 (plus copper modal and chord energy) is not an invariant of the scheme. In an EMPTY cavity, with no sheet and no port, it rises by 7.9e-3 of its initial value over 20000 steps, because leapfrog conserves a modified discrete energy instead. F4 therefore gates the modified energy: it stays <= 1.0x its initial value and decays to 0.122x with the R load.
- **Longitudinal lumping.** Each transverse column of gap chords is one series branch (G_c = w_c / (R w)) driven by the full source V, so the feed obeys the LumpedPort contract along d. F1-narrow (one-cell strip) passes 2e-3 at both dx: PEC Z_in <= 1.3e-7, Cu <= 1.6e-4.
- **Width behaviour.** Port V is the conductance-weighted column voltage, which gives the LumpedPort branch law I = G (V_bar - V_s). The term sum_c G_c |V_c - V_bar|^2 (P_var) is the power dissipated by transverse current redistribution. It is exactly zero for one column and reported as a diagnostic; F3 closes on the exact field-side power.

| dx | w / dx | Z_in diff vs native (PEC) | P_var / P_acc |
|---|---:|---:|---:|
| 5 mm | 1 | 1.3e-7 | 0 |
| 5 mm | 2 | 2.75e-3 | 0.90 % |
| 5 mm | 4 | 3.44e-3 | 0.98 % |
| 2.5 mm | 1 | 8.5e-8 | 0 |
| 2.5 mm | 2 | 1.86e-3 | 0.81 % |
| 2.5 mm | 4 | 1.97e-3 | 0.76 % |

  At fixed physical width (10 mm) the difference falls from 2.75e-3 to 1.97e-3 with refinement. So part of the wide-gap difference vs the single-edge native branch is converging discretization. The roughly 0.8-1 % P_var share is the genuine transverse model-form term.

## Deferred: D6b, non-Cartesian feed axes (user decision 2026-11-01)

D6b would feed oblique in-sheet current directions: general chord-basis weight vectors b_c,
with the LumpedPort axis check moved to compile time. It is DEFERRED. The adopted rim fix
(partial rim chords rejected, master 9e7b0933) gives the conformal sheet the native PlanarPEC
staircase, about 3% in-plane rotation spread at lambda/120. The planned rotated-current gate
(Z_in <= 3%, resonance <= 0.5%) is therefore capped by the sheet, not the feed. Resume on a real
oblique-feed use case, or once a sub-cell rim treatment exists. Estimate when resumed: 20-30
Codex h + 4-8 GPU-h.

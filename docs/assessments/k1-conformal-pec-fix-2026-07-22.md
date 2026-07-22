# K1 — Conformal PEC release blocker: root cause, fix, gates, re-measurement

Date: 2026-07-22
Tree: `master`, blocker opened at `0f1c03d`
Predecessors: `docs/assessments/results-regeneration-2026-07-22.md` (blocker evidence),
`docs/assessments/f4-subpixel-lever-acceptance-2026-07-21.md` (the gate that approved
the `pec="conformal"` benchmark default, whose geometry cluster contains no PEC scene).

---

## 1. The blocker as measured

Round F4 made `pec="conformal"` the benchmark-harness default in
`benchmark/scenes/_common.py`. Two PEC rows degraded badly; flipping the flag back
and forth reproduces both numbers exactly.

| scene | `staircase` | `conformal` (F4, pre-fix) | change |
|---|---|---|---|
| `pec_box` | field_l2 2.5124e-02, corr 0.9997 | 2.8278e-01, corr 0.9702 | **+1025%** |
| `rcs_pec_sphere` | 2.2131e-01, corr 0.9833 | 3.6376e-01, corr 0.9315 | +64% |
| `pec_sphere` | 3.4255e-01, corr 0.9403 | 2.6598e-01, corr 0.9644 | −22.4% |

`pec_box` is a flat, axis-aligned PEC slab. Its staircase representation is exact,
so a correct conformal treatment must reduce to staircase there.

---

## 2. Mechanism (proven)

`_apply_pec_edge_suppression` folds an open fraction into **both** the decay and the
curl coefficient of the electric update:

```
E_new = open * (decay * E_old + curl * (curl H - J)),   open = 1 - fill
```

That is a per-timestep multiplicative factor, i.e. an effective conductivity
`sigma_eff = eps * fill / dt` on that edge, *not* a geometric wall placement. The
fill must therefore be zero on any edge the conductor does not touch.

It was not. `_pec_edge_open_fractions` built the fill as the two-node average of
`_pec_occupancy`, the `tanh`-smoothed union PEC occupancy with width
`beta = 0.5 * cell`. `tanh` has no compact support, so the occupancy leaks several
cells into vacuum. Dump for the `pec_box` scene (`dx = 0.024615` after PML padding,
slab faces at `±0.05`, nearest node plane `±0.049231`), tangential `Ex` edges, which
sit exactly at node `z` planes so their fill *is* the node occupancy:

| z (edge) | signed distance | fill (conformal, pre-fix) | `open` | fill (staircase) |
|---|---|---|---|---|
| 0.03692 | −0.01308 | 0.983069 | 0.016931 | 1 |
| 0.04923 | +0.00115 | 0.530730 | 0.469270 | 1 |
| 0.06154 | +0.01154 | 0.021554 | 0.978446 | 0 |
| 0.08615 | +0.03615 | 0.000429 | 0.999571 | 0 |
| 0.11077 | +0.06077 | 0.000008 | 0.999992 | 0 |

Two distinct errors follow.

1. **The metal face is not a short.** The tangential edge lying on the PEC face gets
   `fill = 0.53`, hence `sigma_eff = eps0 * 0.53 / dt = 0.100 S/m` at
   `dt = 4.7e-11 s`, i.e. a sheet resistance `Rs = 1/(sigma*dz) = 405 ohm/sq` and a
   reflection coefficient `Gamma = -1/(1 + 2*Rs/eta0) = -0.32`. A PEC face must
   reflect −1. Under `staircase` that same edge thresholds to `fill = 1`,
   `open = 0`, an exact short.
2. **A lossy halo in vacuum.** Vacuum edges one to three cells outside the metal get
   `fill = 2.2e-2, 4.3e-4, 8e-6`, each a resistive sheet. Because `open` multiplies
   every step, even `fill = 1e-4` costs `0.9999^N`; over the benchmark's few
   thousand steps that is order-unity attenuation of the field next to the slab.

On the compiled masks, for a cube whose faces land exactly on node planes
(`dx = 0.02`), the pre-fix node-average put a nonzero fill on **5298 edges per
component** whose two endpoints are both strictly outside the conductor.

Neither error requires the surface to cut a cell — both fire on a perfectly aligned
face. That is why `pec_box` collapsed while the F4 acceptance cluster, which
contains no PEC material at all, never saw it.

---

## 3. Fix level chosen

**Level (a) at the library, plus level (b) for the harness default.**

### (a) Library — the conformal fill now has compact support

`witwin/maxwell/compiler/materials.py` gained
`_pec_signed_distance` / `_edge_line_coverage` / `_pec_edge_fill`, and the compiled
model carries `pec_edge_fill` (only in `conformal` mode; `None` otherwise, so
`staircase` is byte-identical to before). The fill on a Yee E edge is now the
**geometric coverage fraction of that edge**: the union PEC signed distance is
interpolated linearly between the edge's two endpoint nodes and the covered fraction
is the part of the segment with `sd <= 0`:

```
lower = min(sd0, sd1); upper = max(sd0, sd1); span = upper - lower
fill  = clamp(-lower / span, 0, 1)            when span > 0
fill  = 1 if upper <= 0 else 0                when span == 0  (edge parallel to the face)
```

Properties, all exercised by the gates below:

* exactly `0` when both endpoints are outside — no halo, at any distance;
* exactly `1` when both endpoints are inside, and when the edge lies *in* the surface
  (`sd == 0` along it), which is what keeps tangential E on a PEC face at zero;
* fractional only on edges the surface genuinely cuts;
* exact for a planar cut, second-order for a curved one (an SDF is eikonal);
* differentiable in the conductor geometry;
* spacing-exact on graded grids (no smoothing width involved).

`witwin/maxwell/fdtd/runtime/materials.py::_pec_edge_open_fractions` consumes
`pec_edge_fill` in `conformal` mode and keeps the thresholded node-average for
`staircase`. `_pec_occupancy` itself is unchanged — the mode solver, modal ports,
terminal-contact checks and material summaries still read the same node field.

**Capability change.** A flat wall parallel to the grid cuts no tangential edge, so
conformal now reproduces staircase there *exactly* and no longer tracks such a wall
sub-cell. That sub-cell tracking was only ever produced by the smearing that caused
this blocker. Doing it correctly requires the area-scaled (Dey–Mittra) magnetic
update (`1/A_open` on the cut H face, with a stability floor), which is a redesign
and is recorded as follow-up work.

### An alternative that was measured and rejected

Scaling only the **curl** coefficient by `open` (leaving `decay` alone) makes the cut
edge a lossless reduced-area integrator instead of a soft short. It is lossless but
measurably worse against the reference: `pec_sphere` 3.6044e-01 (vs 2.9499e-01 for
the soft short and 3.4255e-01 for staircase) and `rcs_pec_sphere` 2.4801e-01 (vs
2.3674e-01 / 2.2131e-01). Reverted; the suppression formulation is unchanged.

### (b) Benchmark harness default back to `staircase`

Even with compact support, the soft short is **lossy on cut edges** — `sigma_eff =
eps*fill/dt` is a real conductivity, and a PEC scatterer is lossless. Measured in a
closed PEC cavity holding an off-centre PEC sphere (energy retained after 6000 steps,
5200 of them source-free, relative to step 800):

| PEC mode | retained energy |
|---|---|
| `staircase` | 0.999989 |
| `conformal`, fixed | 0.450052 |
| `conformal`, pre-fix node-average | 0.124665 |

The fix removes 3.6× of the spurious absorption but cannot remove the rest without
the H-side redesign. Making a demonstrably lossy scheme the default for lossless
metal is not defensible, so `benchmark/scenes/_common.py` is back to
`pec="staircase"`, with per-scene opt-in available. No scene opts in today: the only
row conformal improves (`pec_sphere`) does so while absorbing energy it should not,
so that improvement is not attributable to better physics.

Nothing about this fix touches a tolerance or an expected value.

---

## 4. Gates (committed)

### `tests/materials/compiler/test_pec_conformal_alignment.py` (CPU, 9 tests, 2.6 s)

Asserts on the compiled masks — sharper and far cheaper than a field comparison.

| test | assertion | measured |
|---|---|---|
| `..._equals_staircase_for_grid_aligned_faces` | conformal edge fill is **bitwise** equal to the staircase mask for `Ex`/`Ey`/`Ez` on a cube whose faces land on node planes; zero fractional edges | equal, 0 fractional |
| `..._has_compact_support_around_an_aligned_conductor` | fill is exactly 0 on every edge whose two endpoints are outside; and the pre-fix node-average was nonzero on >1000 of them | max fill 0.0; 5298 halo edges per component |
| `..._is_fractional_only_on_edges_the_surface_cuts` | every fractional edge is genuinely cut (endpoint SDFs of opposite sign) | 194 fractional, 194 cut, 0 outside cut |
| `..._resolves_a_curved_surface_better_than_staircase` | conformal conductor volume error `< 0.5x` staircase and `< 1%` | conformal 0.74%, staircase 1.85% (r=0.11, dx=0.02) |
| `..._is_differentiable_in_the_conductor_geometry` | `d(sum fill)/d(radius)` finite and positive | passes |
| `test_staircase_mode_carries_no_conformal_edge_fill` | `pec_edge_fill is None` under `staircase` | passes |
| `test_non_pec_scene_has_no_edge_fill` | both artifacts `None` | passes |
| `..._open_fractions_agree_across_modes[staircase|conformal]` | runtime-visible fill identical in both modes on an aligned box | passes |

### `tests/validation/physics/test_pec_conformal.py` (CUDA, 4 tests, 29 s)

| test | assertion | measured |
|---|---|---|
| `..._reproduces_staircase_for_a_grid_parallel_wall` | PEC cavity resonance, wall at 0%, 20%, 50%, 80% across one cell: `|conformal − staircase| / staircase < 1e-8` | 1.8e-14 (aligned) … 1.8e-10 (mid-cell) |
| `..._pec_slab_reflection_is_identical_under_conformal` | plane wave off a node-aligned PEC slab: probe `Ex(t)` **bitwise equal** between modes | `np.array_equal` true over 1500 samples |
| `..._closed_pec_cavity_..._lossless_under_staircase` | staircase energy retention within 1e-3 of 1 | 0.999989 |
| `..._spurious_absorption_..._stays_bounded` | conformal retention `> 0.30` and `<= 1` | 0.450052 (pre-fix 0.124665) |

This file replaces the previous
`test_conformal_pec_wall_sweep_is_smoother_than_staircase`, which asserted the
sub-cell smearing of a flat, grid-parallel wall — the behaviour this fix removes.

---

## 5. Falsifications (break it, watch it go red, restore)

**F1 — revert the runtime to the node-average fill** (`_pec_edge_open_fractions`
ignores `pec_edge_fill`):

```
FAILED ..._reproduces_staircase_for_a_grid_parallel_wall
  AssertionError: (0.3, 499360631.5071745, 484302082.04543906, 0.03015566008134359)
  assert 0.03015566008134359 < 1e-08
FAILED ..._pec_slab_reflection_is_identical_under_conformal
  assert np.array_equal(...) is False
FAILED ..._spurious_absorption_..._stays_bounded
  assert 0.12466508088151958 > 0.3
3 failed, 1 passed in 12.29s
```

**F2 — drop the surface-coincident short** (`degenerate = (upper < 0)` instead of
`(upper <= 0)`, so an edge lying exactly in the conductor face is no longer covered):

```
FAILED ..._edge_fill_equals_staircase_for_grid_aligned_faces        (AssertionError: Ex)
FAILED ..._open_fractions_agree_across_modes[conformal]
2 failed, 7 passed in 2.59s
```

**F3 — restore the smoothed node-average as the edge fill** inside `_pec_edge_fill`:

```
FAILED ..._edge_fill_equals_staircase_for_grid_aligned_faces
FAILED ..._fill_has_compact_support_around_an_aligned_conductor
FAILED ..._fill_is_fractional_only_on_edges_the_surface_cuts
FAILED ..._fill_resolves_a_curved_surface_better_than_staircase
FAILED ..._open_fractions_agree_across_modes[conformal]
5 failed, 4 passed in 2.62s
```

Tree restored after each; both files green (`13 passed in 29.27s`).

---

## 6. Re-measured benchmark rows

Cached references only (`WITWIN_BENCHMARK_NO_CLOUD=1`, no cloud run; the reference
cache key does not include `subpixel_samples`, so every row reuses the same
reference). Only the three PEC-material rows in the main table are affected;
`metal_sphere` is a Drude medium and every other row is dielectric or vacuum.

| scene | shipped `0f1c03d` (F4 conformal) | conformal **after** the fix | **new default (`staircase`)** |
|---|---|---|---|
| `pec_box` | 2.8278e-01 / corr 0.9702 / flux 6.6033e-01 | 2.5124e-02 / 0.9997 / 5.0162e-03 | 2.5124e-02 / 0.9997 / 5.0162e-03 |
| `pec_sphere` | 2.6598e-01 / 0.9644 / 4.0632e-02 | 2.9499e-01 / 0.9556 / 3.0013e-02 | 3.4255e-01 / 0.9403 / 2.5317e-02 |
| `rcs_pec_sphere` | 3.6376e-01 / 0.9315 | 2.3674e-01 / 0.9749 | 2.2131e-01 / 0.9833 |

`rcs_pec_sphere` scalar RCS errors, same three configurations:

| observable | F4 conformal | fixed conformal | staircase (shipped table) |
|---|---|---|---|
| `rcs_forward` | 6.0912e-01 | 4.8724e-01 | 3.5857e-01 |
| `rcs_back` | 4.9446e-01 | 5.4341e-02 | 1.5669e-01 |
| `rcs_broadside_E` | 5.5305e-01 | 4.1440e-01 | 4.2281e-01 |
| `rcs_broadside_H` | 4.4704e-02 | 1.9508e-01 | 2.6060e-01 |

`benchmark/RESULTS.md` was updated in place for exactly these rows (main table,
per-frequency table, and the `rcs_pec_sphere` scalar block); every row now matches
the pre-F4 2026-07-18 values, i.e. the blocker's regression is fully retired rather
than merely reduced.

The fix is also a real improvement for anyone who does opt into conformal: at the
same default, conformal after the fix is 13.9% better than staircase on `pec_sphere`
(2.9499e-01 vs 3.4255e-01) and 35% better than the pre-fix conformal on
`rcs_pec_sphere`, and it no longer destroys `pec_box`.

---

## 7. Residual limitations

1. **Conformal PEC is lossy on cut edges.** `sigma_eff = eps * fill / dt` is a
   `dt`-dependent conductivity; a lossless PEC scatterer absorbs (measured: 55% of
   the cavity energy over 5200 source-free steps at `dx = 0.02`). The correct fix is
   the area-scaled Dey–Mittra magnetic update (`1/A_open` with a stability floor) or
   contour-path H-loop averaging — both already listed as future work in
   `FEATURE_LIST.md`. The gate bounds the loss at `> 0.30` retention so a regression
   toward the pre-fix behaviour fails.
2. **No sub-cell placement of a grid-parallel flat wall.** Conformal is now exactly
   staircase there. This is a deliberate capability removal (documented in
   `FEATURE_LIST.md`); the previous behaviour was the defect.
3. **Union of overlapping conductors** uses `min` over per-structure signed
   distances, which is exact in sign and slightly conservative in magnitude near a
   re-entrant junction of two PEC bodies; the coverage fraction there is
   first-order rather than second-order accurate.
4. The linear interpolation of the SDF along an edge is second-order for a curved
   surface, so accuracy remains bounded by SDF quality for primitives with kinked
   distance fields (faces/edges/corners) — the pre-existing caveat.

---

## 8. Files

* `witwin/maxwell/compiler/materials.py` — `_pec_periodic_shift_options`,
  `_pec_signed_distance`, `_edge_line_coverage`, `_pec_edge_fill`; both compile paths
  emit `model["pec_edge_fill"]`.
* `witwin/maxwell/fdtd/runtime/materials.py` — `_pec_edge_open_fractions` consumes it;
  docstrings on `_apply_pec_edge_suppression` state the conductivity interpretation.
* `tests/materials/compiler/test_pec_conformal_alignment.py` — new.
* `tests/validation/physics/test_pec_conformal.py` — rewritten around the corrected
  contract.
* `benchmark/scenes/_common.py` — default back to `pec="staircase"`.
* `benchmark/RESULTS.md` — three rows re-measured.
* `FEATURE_LIST.md`, `docs/reference/release-notes-0.4.0.md` — semantics, the default
  revert, and the residual limitations.

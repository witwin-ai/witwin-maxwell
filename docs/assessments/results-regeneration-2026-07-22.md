# `benchmark/RESULTS.md` regeneration — 2026-07-22

Main checkout, `master` at `92b5af5` (worktree otherwise clean). Host: 2× RTX A6000,
driver 595.71.05, conda `maxwell`. Every run pinned with
`numactl --cpunodebind=0 --membind=0` on `CUDA_VISIBLE_DEVICES=0` (GPU 1 carries the
desktop session).

**Why.** The committed table was generated 2026-07-18. That predates the round-F
edge-native per-Yee-component material sampling and the conformal-PEC benchmark
default (`f0737c5`), and the round-J kernel work (`f5ae24e`, `9a7332f`). 102 of its
103 data rows therefore did not describe the shipped code.

## What was run

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export WITWIN_BENCHMARK_NO_CLOUD=1
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  conda run -n maxwell --no-capture-output python -m benchmark --solver fdtd
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  conda run -n maxwell --no-capture-output python -m benchmark rf
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  conda run -n maxwell --no-capture-output python -m benchmark sar
```

`--solver fdtd` rather than a bare `python -m benchmark` because the four FDFD
scenarios sit at positions 49–52 of the registry and abort the process (see
*FDFD* below); splitting the selection keeps the remaining 102 from being lost to
that abort. `WITWIN_BENCHMARK_NO_CLOUD=1` makes a missing or stale reference a hard
error instead of a billable submission. **No cloud run was launched.**

### Cache validation: strict, zero trust-hook uses

`WITWIN_BENCHMARK_TRUST_CACHE` was **never set**. All 102 scenarios loaded their
external reference under the default fail-closed staleness guard and reported
`Reference cache = hit`; the log contains zero `cache invalid` lines.

This is worth recording because `f4-subpixel-lever-acceptance-2026-07-21.md` had to
score the geometry cluster under `WITWIN_BENCHMARK_TRUST_CACHE=1`, the key drift there
coming from post-generation bookkeeping. The `fix(benchmark): strip null material
fields from reference cache keys` commit (`abb052e`) closed that drift on master, so
the escape hatch is no longer needed for any scene in this table.

## Headline shifts (102 FDTD scenarios)

| Metric | Before (2026-07-18) | After (2026-07-22) |
|---|---:|---:|
| median `field_l2` | 9.8920e-02 | **8.0412e-02** (−18.7%) |
| mean `field_l2` | 1.8732e-01 | 1.5982e-01 |
| median `field_corr` | 0.9959 | **0.9972** |
| scenarios meeting `field_l2 < 1e-1` | 52 | **61** |
| scenarios below `field_corr > 0.99` | 39 | **28** |

Per-scenario movement on `field_l2` (relative tolerance 1e-4 counts as unchanged):

| | count |
|---|---:|
| improved | **35** |
| regressed | **11** |
| unchanged | **56** |

`field_corr`: 32 up, 7 down, 63 identical. `shape_l2`: 38 down, 12 up, 52 identical.

Largest improvements: `hollow_box_scatter` −91.5% (corr 0.8603 → 0.9990),
`material_region_slab` −77.4%, `polyslab_wg` −72.4%, `multi_dielectric` −66.3%
(corr 0.8332 → 0.9794), `polyslab_pentagon` −66.0%, `mode_source_higher_order`
−62.8%, `mesh_primitive_scatter` −55.8%, `anisotropic_slab` −55.3%,
`tfsf_dielectric_sphere` −34.4% (corr 0.7822 → 0.9101).

### Geometry cluster reproduces the F4 prediction exactly

All 16 scenes of the `grid_geometry` cluster land on the `field_l2` values recorded in
`docs/assessments/f4-geometry-cluster-after.json`, to the precision the table prints
(16/16), and the cluster median is **0.083642**, identical to the recorded
`median_field_l2_after`. The expected direction in
`f4-geometry-cluster-delta.json` (11 improved, 5 flat, 0 regressed, median −59.6%)
is confirmed under the standard entrypoint with strict cache validation.

### Stepping cost

Every one of the 102 scenarios got faster: median `ms/step` **−6.92%**, mean −15.86%,
range −1.20% to −46.71%, zero slower. Largest: `lorentz_resonator` 0.7996 → 0.4261
(−46.7%), `grating_diffraction` −46.2%, `high_eps_box` −45.5%, `multi_dielectric`
−45.2%. This is the round-J kernel work (uniform-coefficient scalar fast path,
compressed-CPML interior kernels, template folding) landing in the table. `ms/step` is
a diagnostic column with no pass/fail target and this run is not an exclusive-window
timing measurement — treat it as directional.

## Scenes that got materially WORSE

Two, and they share one cause. Both were falsified against the alternative rather
than assumed: rebuilding the identical scene with `SubpixelSpec(pec="staircase")`
instead of the shipped `pec="conformal"` reproduces the old 2026-07-18 number
**exactly** in both cases, and restoring `pec="conformal"` reproduces the new number
exactly.

| Scenario | 2026-07-18 | shipped default (`pec="conformal"`) | same scene, `pec="staircase"` |
|---|---:|---:|---:|
| `pec_box` | 2.5124e-02 (corr 0.9997) | **2.8278e-01 (corr 0.9702)** — +1025.5% | 2.5124e-02 (corr 0.9997) |
| `rcs_pec_sphere` | 2.2131e-01 (corr 0.9833) | **3.6376e-01 (corr 0.9315)** — +64.4% | 2.2131e-01 (corr 0.9833) |
| `pec_sphere` | 3.4255e-01 (corr 0.9403) | 2.6598e-01 (corr 0.9644) — −22.4% | 3.4255e-01 (corr 0.9403) |

So the conformal-PEC benchmark default introduced by `84d7dba` / `f0737c5` is
responsible for both regressions, and it is a **mixed** result on PEC scenes: it helps
`pec_sphere` (−22.4%) and hurts `rcs_pec_sphere` (+64.4%) and `pec_box` (+1025.5%).
`pec_box` is the worst case and the least defensible one — it is a flat,
**axis-aligned** PEC slab, i.e. exactly the geometry a staircase already represents
without error, and fractional-fill edge suppression can only move a boundary that was
already in the right place. `metal_sphere` (Drude, not PEC material) is bit-identical
under both settings, confirming the effect is specific to PEC material.

**This was never measured before shipping.** `f4-subpixel-lever-acceptance-2026-07-21.md`
§"Conformal-PEC scope note" states that the geometry cluster contains no PEC material,
so `pec="conformal"` was a no-op for the cluster the acceptance gate scored. The
default was therefore turned on for the benchmark harness without any PEC scene ever
being run against it. This regeneration is the first measurement, and it is net
negative: 2 of 3 PEC scenes worse, one catastrophically.

**Recommendation (not applied here — this change is a measurement, not a fix):** treat
the conformal-PEC benchmark default as a release blocker until either (a) it is
restricted to curved/oblique PEC and disabled for axis- and node-plane-aligned PEC
faces, or (b) the default is reverted to `pec="staircase"` for the benchmark harness
and the conformal path is opted into per scene. Fixing it inside this task would have
meant shipping a numerics change under cover of a table regeneration.

**Resolved 2026-07-22 (K1).** Root-caused and fixed; see
`docs/assessments/k1-conformal-pec-fix-2026-07-22.md`. The conformal PEC edge fill
was the node average of a `tanh`-smoothed occupancy, so it leaked fractional fill
(hence, because the open fraction multiplies the E update every step, an effective
conductivity `eps*fill/dt`) onto vacuum edges up to three cells outside the metal and
left the face edge itself a 405 ohm/sq sheet instead of a short. The fill is now the
per-edge geometric coverage fraction, which has compact support and reduces to the
staircase mask bit for bit on grid-parallel faces (option (a)). The residual — the
soft short is still lossy on genuinely cut edges — is bounded and documented, and the
harness default is back to `pec="staircase"` with per-scene opt-in (option (b)). All
three PEC rows in this table were re-measured against their existing cached references
and returned to their pre-F4 2026-07-18 values.

### The other nine regressions

All small and none crossing a metric target:

| Scenario | `field_l2` | change |
|---|---|---:|
| `bloch_oblique` | 8.6987e-02 → 9.0912e-02 | +4.51% |
| `dipole_dielectric_sphere` | 4.0239e-02 → 4.1982e-02 | +4.33% |
| `sphere_rcs` | 1.6704e-01 → 1.7096e-01 | +2.35% |
| `rcs_dielectric_box` | 1.7039e-01 → 1.7213e-01 | +1.02% |
| `sigma_e_drude_slab` | 2.0795e-01 → 2.0923e-01 | +0.62% |
| `mode_port_straight_wg` | 1.9777e+00 → 1.9854e+00 | +0.39% (corr *rose* 0.9921 → 0.9938) |
| `waveguide_s_matrix` | 4.3126e-02 → 4.3172e-02 | +0.11% |
| `mode_source_wg` | 4.4225e-02 → 4.4243e-02 | +0.04% |
| `kerr_slab_strong` | 1.3438e-01 → 1.3442e-01 | +0.03% |

`sphere_rcs` is bit-identical under `pec="staircase"` and `pec="conformal"`, so its
+2.35% comes from the edge-native material sampling, not from the PEC default. The
rest are second-order effects of the same sampling change and are the expected price
of the 35 improvements.

## FDFD: honestly represented, still deferred

The four FDFD scenarios (`fdfd_dielectric_slab`, `fdfd_drude_sphere`,
`fdfd_sigma_e_slab`, `fdfd_diag_aniso_slab`) fail on this host:

```
GPU solve failed: No module named 'nvmath'.
RuntimeError: FDFD solve did not produce any field data.
```

Per the standing user deferral, `nvmath` was **not** installed and no FDFD triage was
attempted. The failure aborts `run_benchmarks` on the first FDFD scenario, so those
rows cannot be produced.

The previously committed table simply had no FDFD rows at all — the family was
silently absent, and nothing in the file said so. That is fixed at the generator
rather than by hand: `benchmark/report.py` now emits a **`## Registered scenarios with
no measured row`** section listing every scenario present in `benchmark/runner.SCENARIOS`
that has no row in the tables above, with its solver and description, and stating
explicitly that such a row is *not* a pass, a fail, or a tolerance waiver but the
absence of evidence. The four FDFD scenarios appear there. Because it is generated,
the disclosure cannot rot out of the file on the next regeneration.

## Sections owned by other harnesses

`write_results_markdown` rebuilds `RESULTS.md` from the main-table rows, so the
sections written by the other harnesses are dropped whenever the field-vs-reference
runner writes. Their handling:

| Section | Handling |
|---|---|
| `## RF wave-level validation` | **re-run** (`python -m benchmark rf`), regenerated 2026-07-22 |
| `## Antenna wave-level validation` | **re-run** (same command), regenerated 2026-07-22 |
| `## SAR exposure validation` | **re-run** (`python -m benchmark sar`), regenerated 2026-07-22 |
| `## RF / antenna external reference generation` | **preserved verbatim** from 2026-07-21 |

The reference-generation section is cloud-submission bookkeeping produced by
`benchmark.rf_tidy3d_references`; re-running it is exactly the billable cloud action
this task forbids, so it is carried across unchanged and is the one section in the
file still dated 2026-07-21.

RF/antenna status changes versus the committed section (nothing that was passing
stopped passing):

| Scene | before | after |
|---|---|---|
| `rf/microstrip_two_port` | blocked | gap |
| `rf/differential_pair` | blocked | fail |
| all others (`coax_thru`, `rectangular_waveguide`, `lossy_waveguide_attenuation`, `series_parallel_rlc`, `lumped_open_short_match`, `antenna/half_wave_dipole`) | pass | pass |
| `antenna/patch` | gap | gap |

SAR: all four statuses identical (`one_gram_cube`, `uniform_lossy_cube`,
`layered_slab` pass; `antenna_near_phantom` blocked).

The RF and SAR harnesses were each executed twice during this session and returned
identical statuses both times, which is a free determinism check on those two
families.

## Reproduction notes

`benchmark/cache` is a symlink to the shared reference cache and was not written to.
The falsification probes flipped `benchmark/scenes/_common.py` to `pec="staircase"`,
recorded the numbers, then reverted with `git checkout` and re-ran the affected
scenarios so the committed table carries the shipped-default values; the restored
values reproduce the main run bitwise, which also demonstrates run-to-run determinism
for these scenarios.

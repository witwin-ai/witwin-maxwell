# Per-edge Kottke averaging for FDTD (spec; no code until approved)

Status: proposal, 2026-10-31. Owner: coverage line. Trigger: LQ-1b
(`instamax` branch `lq/discretization-ab`, `docs/assessments/lq1b-discretization-ab-2026-10-31.md`).
On the p16s mesh/cylinder scene, the change between successive levels at ppw 24→32 is (shell / inside):

| Arm | 24→32 change | Three-level order |
|---|---:|---:|
| instamax per-edge Kottke (FDFD at f_eff) | 1.5% / 1.7% | 1.0–1.4 |
| witwin FDTD `polarized`, samples=3 | 2.1% / 2.8% | 0.7–0.8 |
| witwin FDTD `polarized_tensor` | 2.8% / 4.1% | 1.18 (larger ppw16 error) |

With `polarized`, F1 field labels need about ppw 40 (shell) to 55 (inside) for 5% on this geometry.

## 0. Is the gap averaging, or FDTD vs FDFD?

It is averaging; temporal dispersion is already excluded:

1. The LQ-1b A arm solves at f_eff = (2/dt)·sin(ω·dt/2)/(2π) with the half-step phase removed, which is the
   correct comparison rule (`docs/reference/fdfd-fdtd-consistency.md`).
2. Given the same compiled edge ε, witwin FDFD at f_eff and same-grid witwin FDTD agree to 0.15% / 0.007%
   on p16s samples 0 / 168, after one global complex scale (cf0dc081 gate).
3. Given the same ε, witwin's FDFD operator and instamax's Yee FDFD agree to 7e-6–1e-5 (the InstaMax
   operator differential, 5 p16s samples).

So the arms differ only in the ε each puts on the Yee edges. witwin FDFD-A (cf0dc081) is therefore the
control: witwin FDFD-A vs instamax FDFD on the same scene isolates the averaging, at FDFD cost and with
no time stepping. Step 1 of the plan (§5) makes this explicit.

## 1. What instamax does differently

instamax `instamax/scene.py:265-316` (`rasterize_eps_subpixel`, commit 9623a75):

| Aspect | instamax per-edge Kottke | witwin `polarized` | witwin `polarized_tensor` |
|---|---|---|---|
| Window | a cube of side d centred on each Yee edge (`positions[c]`, l.291-293) | per-edge windows (`_edge_axis_offsets`, `compiler/materials.py:1037`) and node windows, `samples` per axis | same as `polarized` |
| Samples | `samples=4` per axis, 64 points, binary inside/outside (`_inside`, winding number for meshes, l.255-262); only edges with \|sdf\| < (√3/2)·d (l.294, l.302-311) | `samples=3` per axis (27); each sub-sample is a smooth coverage-ramp fill fraction from the SDF with a curvature shift (`_subsample_fill_fraction`, `compiler/subpixel.py:199`) | same |
| **Interface normal** | **in-window first moment** of the sampled ε: grad = Σ (ε_s − ⟨ε⟩)·offset_s (l.312-314), local to the edge's own window | **cross-window gradient**: `torch.gradient` of the sub-sample-averaged arithmetic ε between neighbouring edges/nodes (`_field_gradients`, `compiler/subpixel.py:101`; `_reconstruct_edge_polarized`, `compiler/materials.py:1087`), a central difference over about 2 cells | gradient of the node arithmetic field (`_normalized_field_gradients`, `compiler/subpixel.py:410`), interpolated to edges by the tensor runtime |
| **Blend** | **physical tensor diagonal**: ε_cc = n_c²·⟨1/ε⟩⁻¹ + (1 − n_c²)·⟨ε⟩ (l.270-273, l.315) | **inverse-tensor diagonal**: ε_cc = 1 / (n_c²·⟨1/ε⟩ + (1 − n_c²)/⟨ε⟩) (`_inverse_diagonal_blend`, `compiler/subpixel.py:372`) | full tensor ⟨ε⟩I + (⟨1/ε⟩⁻¹ − ⟨ε⟩)nnᵀ with off-diagonals (`compiler/subpixel.py:410-419`) |
| Off-diagonals | none | none | yes (FDTD only) |
| Material scope | complex ε treated together; later objects take priority | real ε, σ blended separately (arithmetic), dispersive pole weights arithmetic | same as `polarized` |
| Differentiable in geometry | no (binary point tests) | yes | yes |

The two substantive differences are the **normal source** and the **blend**. Window placement is already
per-edge in both codes, and witwin's sample fill is finer (fractional) than instamax's binary samples.

Why each might matter on p16s geometry:
- **Normal.** A central difference across neighbouring windows mixes interfaces when features are about
  1–2 cells wide (thin cylinder walls, mesh edges and corners). It is also biased near another body, and
  at a corner it gives an averaged direction. The in-window moment uses only the material inside the
  edge's own cube, so it stays local.
- **Blend.** For a diagonal-only update E_a = D_a/ε_aa, the literature choice for second order on smooth
  interfaces is the inverse diagonal (ε⁻¹)_aa. instamax uses the physical diagonal, which is always larger
  or equal (diag(A) ≥ 1/diag(A⁻¹) for SPD A) and differs by O(contrast·n_a²(1−n_a²)) at oblique faces.
  The Part B prototype found that the physical diagonal fed to FDFD was much worse on p16s sample 0
  (40% vs 18%). But that used witwin's gradient normal, so blend and normal were confounded. Which factor
  explains LQ-1b is the first thing to measure (§5 step 1).

## 2. Mapping into witwin (new opt-in mode, default unchanged)

New `SubpixelSpec(averaging="polarized_edge", samples>1)`; the name follows the existing `polarized*`
vocabulary. It is FDTD and FDFD, since it is diagonal and FDFD already consumes the edges (cf0dc081). Two
switches, fixed by the factorial in §5:

1. **In-window moment normal.**
   - During the existing multi-sample loop (`compile_material_model`, `compiler/materials.py` ~l.2700,
     and its edge-native counterpart `compile_edge_material_components`, ~l.1451), accumulate the first
     moment M_a = Σ_s offset_{s,a}·(ε_s − ⟨ε⟩) per edge component, next to the existing arithmetic and
     inverse accumulators.
   - The offsets are already known per sample (`_axis_sample_offsets`, `_edge_axis_offsets`).
   - Then n_a² = M_a² / |M|², replacing the `_field_gradients` weight in `_reconstruct_edge_polarized`.
   - This is differentiable, because the fills are differentiable.
   - It costs one extra accumulator per component and no extra SDF queries.
2. **Blend choice:** physical diagonal n²·H + (1 − n²)·A as an alternative to `_inverse_diagonal_blend`
   (`compiler/subpixel.py`, a 10-line helper).

Scope rules:
- Keep σ and the dispersive pole weights on their current (arithmetic) path, as in `polarized`. instamax
  blends complex ε jointly, which only matters for lossy constituents; it is out of scope here.
- Exclude anisotropic constituents locally, the same way `polarized_tensor` does (reuse its mask).
- Node components: FDTD reads edge components wherever `_edge_native_eligible` holds. Where it does not
  (sheets, surface-impedance metals, prescribed full tensors), the mode falls back to `polarized` behaviour
  for that model. Document that.
- `averaging="polarized"`, `"polarized_tensor"`, `"arithmetic"` and the default samples=1 stay
  bit-identical. Gate the new code on the mode string at the same points `polarized_tensor` uses.
- The PCB conductivity replay (`_PCBConductivityReplay`): σ is unchanged, so it can stay. Add the same
  fail-closed raise as `polarized_tensor` unless the replay gate shows it bitwise.

## 3. FDTD stability / CFL

- The result is diagonal and positive. Each ε_aa lies between the window's harmonic and arithmetic means,
  so it is ≥ min constituent ε ≥ 1 for dielectrics. The material-aware dt bound (`fdtd/runtime/timestep.py`)
  is therefore unchanged, because the vacuum/min-ε bound controls. There is no off-diagonal coupling,
  hence none of `polarized_tensor`'s 0.05–3.3% dt reduction and no tensor-update cost.
- The physical-diagonal variant gives ε_aa ≥ the inverse-diagonal value, which also cannot lower dt.
- Lossy/dispersive cells keep their current σ/pole handling, so the passivity and ADE stability arguments
  are unchanged.

## 4. Gates

Fast tier for the implementation:
- **torch.equal for the existing modes** vs master, fresh interpreters, every compiled FDTD tensor and
  prepared coefficient: default samples=1, arithmetic, `polarized`, `polarized_tensor`, FDFD homogeneous.
  Use the 20-case set from cf0dc081, including the p16s scenes and mesh edge cases. Plus the PCB replay
  gate (`pcb_prep_cache_compare materials`, metamaterial-solid + pcb-flexible).
- **Same-grid consistency**: FDFD with `polarized_edge` at f_eff vs same-grid FDTD with
  `polarized_edge`: ≤ 0.5% after one global complex scale, with phase ≈ ω·dt/2, on p16s 0/168 plus one
  LQ-1 sphere. This shows FDFD and FDTD consume the same new ε.

Accuracy (decide adoption; this is the LQ-3 recipe question, not a merge gate):
- **LQ-1b re-run** with a `polarized_edge` B arm: same scenes, ppw 16/24/32, same windows, reusing the A
  observations. The target is a 24→32 change ≤ instamax's (1.5% / 1.7%) on p16s and order ≥ 1.0. Spheres
  must not regress (the current B is order 1.9, about 1% limit gap).
- **LQ-1 Mie and Airy orders**: the sphere and slab subset at 16/20/30 cells/λ0, as for #3: order on curved
  interfaces ≥ `polarized`, with no slab regression.
- Optional, when available: the external-reference B2-L-p16s run, which gives absolute errors per arm.

## 5. Plan, ETA and cost

1. **Diagnosis first (0.5 Codex day, ~0.5 GPU-h; FDFD only, no FDTD needed per §0).**
   - On p16s sample 0 (cropped, scaled) at ppw 16/24/32, build four ε variants on the Yee edges:
     {gradient normal, in-window normal} × {inverse diagonal, physical diagonal}. Do this offline, without
     touching the library, by recomputing from witwin's per-sample fills.
   - Solve each with witwin FDFD (cf0dc081) and compare with instamax FDFD on the same grid, using the
     24→32 change.
   - Also diff the ε fields directly against instamax's `rasterize_eps_subpixel`, to locate where they
     differ (thin walls, corners).
   - This says which factor carries the LQ-1b gap. If neither does (e.g. binary vs fractional samples
     dominates), stop and report before implementing.
2. **Implement** the winning factor(s) as `polarized_edge`: 1 Codex day.
3. **Fast-tier gates** (bit-identity, consistency, PCB replay): 0.5 Codex day, ~1 GPU-h.
4. **LQ-1b re-run plus the LQ-1 subset**: 0.5 Codex day, ~4–6 GPU-h (p16s plus 3 sphere scenes × 3
   levels, Courant 0.5, finest 88³; LQ-1 spheres/slab × 3 levels).

Total: about **2.5 Codex days and 6–8 GPU-h** on GPU 1. Step 1 alone (0.5 day, 0.5 GPU-h) decides
whether the rest is worth doing.

Movers: none for existing modes. The new mode has only its own tests. A `polarized` re-pin happens only
if LQ-3 later makes the new mode the default recipe, as a separate decision.

## Step 1 result (2026-10-31; user-approved step 1 only)

Factorial run on p16s sample 0 (cropped and scaled), ppw 16/24/32, at the LQ-1b observation points.
- Every arm is solved with the instamax Yee FDFD at f_eff and differs only in edge eps.
- The W arm, solved with witwin FDFD, cross-checks at 3e-6.
- GPU 1 only, under 0.33 GPU-h.

Successive-level change, 16→24 ; 24→32, shell / inside (%):

| Normal / sampling | Physical diagonal | Inverse diagonal |
|---|---|---|
| in-window moment, binary 4³ | **A** 3.44/3.54 ; 1.48/1.75 | F1 3.55/4.17 ; 1.50/1.89 |
| neighbour gradient, binary 4³ | F2 3.52/3.84 ; 1.54/1.87 | F3 3.89/4.88 ; 1.77/2.31 |
| neighbour gradient, witwin fractional 3³ | — | **W** 4.03/5.23 ; 2.13/2.85 |

Field difference vs A, without a fitted scale, at ppw 16/24/32 (shell / inside, %):

| Arm | ppw 16 | ppw 24 | ppw 32 |
|---|---:|---:|---:|
| F1 | 2.9/3.5 | 1.8/2.2 | 1.3/1.6 |
| F2 | 1.8/2.1 | 1.0/1.2 | 0.6/0.8 |
| F3 | 5.4/6.6 | 3.4/4.1 | 2.3/2.8 |
| W | 7.0/8.8 | 4.8/6.0 | 3.2/3.9 |

Attribution of the 24→32 gap (W − A), with the normal × blend interaction split equally:

| Factor | Shell | Inside |
|---|---:|---:|
| normal | 32% | 32% |
| blend | 34% | 33% |
| sampling (W − F3) | 34% | 35% |

Each factor alone is small. Normal alone (F2) and blend alone (F1) each move the 24→32 change by
≤0.14 points. Together (F3) they account for about half of the gap; the interaction is superadditive. The
witwin sampling recipe (fractional 3³ with the curvature-shifted ramp, vs binary 4³) accounts for the
other half.

Where eps differs (ppw 16, 954 interface edges):
- W vs A differ by more than 5% on 43% of interface edges, and F3 vs A on 25%.
- The differences sit on curved interfaces. This crop has no sharp edges, thin walls or nearby second body.
- Moment vs gradient normals differ by a median of 12°, p90 25° and max 55°. 353 moment normals are zero.

**Verdict: no single factor explains a clear majority, so the step-1 stop rule applies.** Steps 2–4 are
not started. Any further work is a new decision; the user decides after the external-reference B2-L result
gives absolute errors per arm. Raw data and scripts are in <scratch>/witwin_cov_pek/ (report.txt,
results.json, diagnose.py, eps_ppw16.npz).

If the work is resumed, the factorial suggests the design target. A `polarized_edge` mode would have to change
BOTH the normal (in-window moment) AND the blend (physical diagonal) to recover about half the gap. The
other half needs a sampling study (sample count, binary vs fractional, the curvature shift) before any
design.

## Decision 2026-10-31: port all three ingredients (user, "移植")

Evidence: the external-reference absolute errors (lq/external-refs-b2 e9a86c9e,
`docs/assessments/external-reference-lq1b-absolute-error-2026-10-31.md`). On p16s sample 0, instamax A is
6.0/7.0 → 4.1/4.9 → 3.3/4.0% at ppw 16/24/32, against witwin `polarized` at 10.9/14.3 → 8.0/10.0 →
6.1/7.6%. That is about 1.8× lower error at every ppw. Smooth spheres are equal for both (2nd order).

Step 1 showed that the in-window normal plus the physical diagonal recover only about half of the gap,
and that the sampling recipe carries the rest. The approved mode therefore reproduces
`instamax/scene.py:265-316` exactly. The name is **`SubpixelSpec(averaging="kottke_edge", samples=n)`**,
FDTD and FDFD, opt-in, default unchanged.

For each Yee E-edge component c:
- **Window:** a cube of side d around the edge position. On graded axes, d is the local primal or dual
  width, matching `_edge_axis_offsets`.
- **Samples:** n³ points at offsets ((i + ½)/n − ½)·d, with instamax's n = 4.
- **Occupancy:** binary point occupancy per sample: the SDF sign, with core's winding/sign for meshes.
  Later structures take priority.
- **Band:** only edges with |sdf| < (√3/2)·d of some structure are supersampled. Other edges take the
  material at the edge point.
- **Normal:** the in-window first moment Σ(ε_s − ⟨ε⟩)·offset_s of the real permittivity. If the moment
  is zero, n² = 0, i.e. the arithmetic mean.
- **Blend:** the physical diagonal ε_cc = n_c²·⟨1/ε⟩⁻¹ + (1 − n_c²)·⟨ε⟩.

Scope:
- Scalar isotropic ε constituents, plus vacuum. Cells that contain an anisotropic constituent use the
  local diagonal fallback of `polarized`.
- σ_e and dispersive pole weights stay on their existing (arithmetic) path.
- μ uses the same rule with μ in place of ε.
- **Not differentiable in geometry.** The binary occupancy has no shape gradient, so trainable
  geometry with `kottke_edge` must raise (fail closed, with a census entry). Material-value gradients
  can stay.

CFL as in §3: the result is diagonal and lies between the harmonic and arithmetic window means, so the
dt bound is unchanged.

Gates:
- **Merge:**
  - every existing mode is torch.equal on the 20-case set (cf0dc081) plus the PCB replay gate;
  - FDFD at f_eff vs same-grid FDTD ≤ 0.5% after one global complex scale;
  - the ε-field diff vs instamax `rasterize_eps_subpixel` on p16s0 at ppw 16 is ≈ 0, reporting the max
    and median relative difference over interface edges. Any residual must be explained, e.g. by
    winding/SDF sign ties at the surface.
- **Accuracy:**
  - the B arm (witwin FDTD with `kottke_edge`) is scored against the B2-L external reference on p16s0
    with the existing data (no new credits);
  - its absolute error must be within 10% of instamax A's at ppw 16/24/32;
  - spheres must not regress (εr 4, 2nd order).

Plan: Codex in batch 3, about 2.5 Codex days plus 6–8 GPU-h on GPU 1 (not 19:00–21:00). Send a
downstream notice to instamax and the data plan when it lands.

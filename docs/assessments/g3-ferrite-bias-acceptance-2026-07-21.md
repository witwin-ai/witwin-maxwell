# G3 Ferrite general-bias forward — acceptance (2026-07-21)

Track G3 (plan 08 Wave C), branch `fable/ferrite-bias`. This document is the
living acceptance record; the **G3a** section below is complete. G3b extends it
(handedness/passivity/zero-impact gates, mixed-bias disposition, census/FEATURE
reconciliation).

Environment: conda env `maxwell`, `CUDA_VISIBLE_DEVICES=1`,
`PYTHONPATH=<worktree>`, `CUDA_HOME=.../nvidia/cu13`. All commands run from the
worktree root.

---

## G3a — general (non-axis-aligned) bias magnetization ADE

### Delivered

- **General-bias forward runtime** (`witwin/maxwell/fdtd/runtime/gyromagnetic.py`).
  The slice-1c axis-aligned fail-closed guard is lifted; an arbitrary uniform bias
  now runs through a general-bias path. Design:
  - The magnetization ADE is unchanged (same linearized-LLG implicit-midpoint
    Cayley update from the frozen contract). The general path is a pure coordinate
    rotation of that update: the RF drive is the projection `h_u = u·H`,
    `h_v = v·H` (columns `u`,`v` of the per-cell right-handed local frame
    `R = [u|v|w]`, `w = b̂`) gathered from all three lab `H` components; the
    back-reaction `dM = dm_u u + dm_v v` scatters onto all three.
  - **Derivation that the coupled 2×2 solve is frame-independent** (recorded in the
    module docstring): with `h^new = [u|v]^T H_new`, `H_new = H_tmp − (u dm_u + v dm_v)/mu_inf`,
    and `[u|v]^T[u|v] = I` (orthonormal), the local reduction is exactly
    `h^new = h_tmp − dm/mu_inf`, so the coupled system
    `(I + Γ/(2 mu_inf)) m_new = (Φ + Γ/(2 mu_inf)) m_old + (Γ/2)(h_pre + h_tmp)`
    is **identical** to the axis-aligned derivation. `B`,`C` are therefore the same
    matrices — no new integrator, no new coefficients.
  - **Collocation**: identity collocation `C = I` (each lab `H` component truncated
    to the shared `(Nx−1,Ny−1,Nz−1)` cell overlap) is reused from the axis-aligned
    slice per supervisor decision (brief §1). No 4-point gather is introduced.
  - Per-cell dense `u`/`v` projection fields (masked to zero on inactive cells) make
    a future mixed-bias relaxation a data change, not a code change.
  - Fast path (axis-aligned) is retained as an optimization (single signed
    component per transverse axis); the general path reduces to it bit-for-bit.
  - `build_gyromagnetic(..., force_general=True)` drives the general code path on an
    axis-aligned scene for the rotation-equivalence gate (changes no physics).
- **Fail-closed preserved**: mixed-bias-direction (sign-inclusive) and Bloch-periodic
  ferrite still raise `NotImplementedError`. The lifted guard is the single
  general-bias reject; the mixed-bias reject stays (message reworded, still matches
  `"single uniform bias direction"`).

### Files added / changed

- `witwin/maxwell/fdtd/runtime/gyromagnetic.py` — general-bias path, shared
  `_gather_drive` / `_scatter_correction`, `force_general`, derivation docstring.
- `tests/materials/ferrite/test_gyromagnetic_general_bias.py` — new gate suite.
- `tests/materials/ferrite/test_gyromagnetic_forward.py` — retired the
  `general_bias_fails_closed` expectation (now `general_bias_builds_via_general_path`);
  updated module docstring.

### Gates (test inventory)

Commands (env prefix omitted; see top):

```
python -m pytest tests/materials/ferrite/test_gyromagnetic_general_bias.py -q
python -m pytest tests/materials/ferrite/ -q            # 60 tests (mock+CUDA)
```

Mock (float64, deterministic) — `test_gyromagnetic_general_bias.py`:

| Gate | Test | Result |
| --- | --- | --- |
| Rotation equivalence (headline): general reduces to fast bit-for-bit, b∈{z,x,y}, coupled step + m | `test_general_reduces_to_fast_bitwise[2/0/1]` | max\|diff\| = **0.0** (bitwise) |
| Reduction gate has teeth | `test_general_path_perturbation_breaks_bitwise_reduction` | pass (corrupt `ux` → diverges) |
| Oblique vs oracle (headline): b=(1,1,1)/√3 CW vs discrete Polder oracle | `test_oblique_cw_matches_oracle` | chi_uu rel ≈ 1.197e-13 / chi_vu rel ≈ 1.9e-13 ≤ `reference_polder_rtol`=1e-5 |
| Oblique oracle gate has teeth (gyrotropy sign) | `test_oblique_cw_gyrotropy_sign_falsification` | pass (κ→−κ mismatches oracle) |
| Oblique passivity (α=0 energy non-growth) | `test_oblique_energy_non_growth_lossless` | \|E₁/E₀−1\| < 1e-9 over 2e5 steps |
| General path builds & engages | `test_oblique_bias_builds_and_engages_general_path` | pass (orthonormal u,v) |

CUDA (real FDTD field update):

| Gate | Test | Result |
| --- | --- | --- |
| Rotation equivalence on the production solver: forced-general = fast, b=z, 200 steps, all 6 fields | `test_cuda_general_matches_fast_bitwise` | bitwise-equal |
| Oblique forward stability + precession + general engaged | `test_cuda_oblique_forward_stable_and_precesses` | finite, m_u,m_v > 0 |

### Falsifications recorded

1. **Headline reduction + oblique-oracle (production-code break).** Deleted the
   `hu.addcmul_(state["uz"], hz)` term from `_gather_drive` (dropping the `uz`
   projection). `test_oblique_cw_matches_oracle` went red (obtained
   `1.876e5+6.87e3j` vs expected `2.251e5+8.24e3j`) and
   `test_general_reduces_to_fast_bitwise[1]` (b=y, where `u=ẑ` so `uz=1`) went red;
   b=z/x reductions stayed green because their `uz=0`. Restored → green.
2. **Reduction gate teeth (committed test).** `test_general_path_perturbation_breaks_bitwise_reduction`
   perturbs `state["ux"]` by 1e-6 and asserts the general step diverges from fast.
3. **Oblique-oracle gate teeth (committed test).** `test_oblique_cw_gyrotropy_sign_falsification`
   negates the skew off-diagonals of Φ/Γ (κ→−κ) and asserts `chi_vu` matches the
   sign-flipped oracle, not the original.

### Adjacent suites run

```
python -m pytest tests/materials/ferrite/ tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py -q
```

→ **131 passed**. Guard census passes (see gaps below).

### Known gaps / deferred (to G3b or later)

- **Census budget not lowered.** Removing the general-bias guard drops the
  capability-guard count by 1; `test_guard_census.py` (ceiling `≤ 175`) still passes.
  Per brief, census reconciliation (lowering the budget + census-doc entry) is a
  **G3b** deliverable; not done here. (Brief cited anchor budget 176; this worktree's
  committed budget is 175.)
- **FEATURE_LIST.md not updated** — G3b deliverable.
- **Mixed-bias** still fail-closed. The per-cell `u`/`v` field layout supports it
  cheaply; disposition (two-material test or reject) is a G3b decision.
- **Handedness (Faraday-direction) and driven-cavity passivity gates for oblique
  bias, PEC/no-ferrite zero-impact** — G3b.
- Bloch-periodic, FDFD ingest, multi-GPU, adjoint, PerturbationMedium-over-ferrite
  remain fail-closed (unchanged).
- No wall-clock timing measured (shared-GPU policy).

---

## G3b — handedness / passivity / zero-impact gates, mixed-bias support, census

### Delivered

- **Mixed-bias support (disposition: SUPPORT).** The "single uniform bias direction"
  fail-closed guard in `build_gyromagnetic` is **removed**. A mixed-bias scene
  (different bias axes, opposed signs on one axis such as a `+z`/`-z` latching
  circulator, or differing magnitudes/materials) now routes through the per-cell
  general path. The compiled layout already stores a per-cell bias, right-handed
  local frame `[u|v|w]`, and per-cell `Phi`/`Gamma`/`B`/`C`; the magnetization ADE is
  purely local (fields couple only via the ordinary reciprocal Yee curl), so a
  mixed-bias scene is the exact direct sum of independent per-cell passive blocks.
  Routing rule: the axis-aligned fast path is used **only** when the bias is uniform
  (sign included) AND axis-aligned; every other case (uniform oblique, or any mixed
  bias) uses the general per-cell path. No new kernel, no new coefficients.
- **Handedness / Faraday-direction gate** (lab-frame runtime extraction): reversing
  an oblique bias flips the gyrotropic (cross-polarized) lab response and leaves the
  co-polarized response unchanged.
- **Oblique driven-cavity passivity** (real CUDA solver): energy-envelope non-growth
  in a closed lossless PEC cavity.
- **Zero-impact** gates: a ferrite-free scene and a PEC-only scene leave the runtime
  disabled and every hook a bitwise no-op.
- **Polder-tensor spot check**: the compiled per-cell full permeability tensor equals
  the frozen lab-frame Polder tensor, and its antisymmetric (gyrotropic) part flips
  sign under bias reversal while the symmetric part does not.
- **Census reconciliation, FEATURE_LIST, this doc.**

### Files changed (G3b)

- `witwin/maxwell/fdtd/runtime/gyromagnetic.py` — removed the uniform-bias guard;
  route non-uniform (oblique/mixed) bias to the per-cell general path; docstrings.
- `tests/materials/ferrite/test_gyromagnetic_general_bias.py` — G3b gate suite
  (handedness, Polder spot-check, mixed-bias per-cell independence, zero-impact,
  CUDA oblique driven-cavity passivity).
- `tests/materials/ferrite/test_gyromagnetic_forward.py` — the two mixed-bias
  fail-closed tests become positive per-cell-frame build tests.
- `tests/api/public/test_guard_census.py` — `CAPABILITY_GUARD_BUDGET` `175 -> 173`.
- `docs/reference/fdtd-capability-guard-census.md` — general+mixed reconciliation.
- `FEATURE_LIST.md` — corrected the stale "fail closed" clause + new delimited
  subsection.

### Census reconciliation

- Measured capability guards: **173** (was 174 after G3a's general-bias removal;
  anchor 175). G3a removed the general-bias guard (budget left at ceiling 175,
  measured 174); G3b removes the mixed-bias-direction guard → measured **173**.
  `CAPABILITY_GUARD_BUDGET` lowered `175 -> 173` in this change, tracking both G3
  removals. Only the Bloch-periodic ferrite guard remains in
  `fdtd/runtime/gyromagnetic.py`. Verified: `test_guard_census.py` all pass; the
  AST counter reports `capability: 173`.

### Gates (test inventory) — G3b additions

Mock (float64), `test_gyromagnetic_general_bias.py`:

| Gate | Test | Result |
| --- | --- | --- |
| Handedness / Faraday direction: oblique bias reversal flips lab gyrotropy, co-pol unchanged | `test_oblique_bias_reversal_flips_lab_gyrotropy` | pass (`gyro_down ≈ -gyro_up` rel 1e-6, `|gyro|>1e-2`) |
| Polder spot check: compiled tensor == frozen Polder; antisymmetric part flips under reversal | `test_compiled_polder_tensor_gyrotropy_flips_under_reversal` | pass (rtol 1e-10) |
| Mixed-bias per-cell independence: combined == direct sum of standalone runs, BIT-FOR-BIT | `test_mixed_bias_per_cell_independence` | pass (opposed oblique regions, differing materials) |
| Zero-impact: ferrite-free scene disables, hooks bitwise no-op on random field | `test_no_ferrite_zero_impact_bitwise` | pass |
| Zero-impact: PEC-only scene disables | `test_pec_only_scene_zero_impact` | pass |

`test_gyromagnetic_forward.py` (mixed-bias now builds):

| Gate | Test | Result |
| --- | --- | --- |
| Mixed-sign +z/-z builds via general path; opposed per-cell `w` | `test_mixed_sign_bias_builds_via_general_path` | pass |
| Mixed-axis +z/+x builds via general path; per-region `w` | `test_mixed_axis_bias_builds_via_general_path` | pass |

CUDA (real FDTD):

| Gate | Test | Result |
| --- | --- | --- |
| Oblique driven-cavity passivity (α=0): energy-envelope non-growth over 12k steps | `test_cuda_oblique_driven_cavity_energy_non_growth_lossless` | pass (2nd/1st half peak ratio ≈ 1.00) |

### Falsifications recorded (G3b)

1. **Mixed-bias per-cell independence (production-code break).** In
   `_build_general_state`, replaced the per-cell `u`/`v` projection fields with the
   global cell-0 frame (`basis[:, comp, 0] * 0 + basis[0, comp, 0]`, same for `v`).
   `test_mixed_bias_per_cell_independence` went red: the combined scene's high-region
   magnetization no longer matched the standalone high run (the high cells used the
   low region's frame — values came out near-negated). Restored → green. (A `u`-only
   break did NOT falsify — the two regions' `u` columns coincide — confirming the
   test's sensitivity is carried by the full per-cell frame.)
2. **Handedness gate teeth (media-level monkeypatch, `scratch/falsify_handedness.py`).**
   Patched `compiler.gyromagnetic.gyromagnetic_local_basis` to strip the bias sign
   (`b -> |b|`), so `-b` builds the same frame as `+b`. The extracted
   `gyro_down = -1.428e-7 - 1.0003j` became **equal** to `gyro_up` (not negated), so
   the assertion `gyro_down ≈ -gyro_up` would fail. This proves the runtime must
   encode the bias sign in the local frame to reverse the Faraday direction.
3. **Oblique passivity metric.** Probed the oblique cavity energy trajectory: it
   OSCILLATES 1x↔3.66x with a **flat envelope** over 20k steps (first-half peak
   3.666, second-half peak 3.665, ratio 0.9998) — bounded/passive. The naive
   `eps0|E|²+mu0|H|²` proxy is not the conserved leapfrog invariant, so an oblique
   mode sloshes it within a bounded envelope; the gate therefore checks envelope
   non-growth, not peak-vs-initial. (The axis-aligned +z mode kept the proxy flat by
   coincidence.) The energy-injection teeth remain the axis-aligned
   `test_cuda_driven_cavity_energy_growth_detected`.

### Commands (G3b)

```
# env prefix per top of doc; CUDA_VISIBLE_DEVICES=1
python -m pytest tests/materials/ferrite/ -q                              # 107 passed
python -m pytest tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py -q  # 30 passed
```

### Mixed-bias disposition — honest assessment

Mixed-bias is **trivially correct and cheap** with the general path, so it is
supported (not rejected). Justification: (a) the magnetization ADE carries no spatial
coupling — each active cell advances independently from its own per-cell frame and
per-cell `Phi`/`Gamma`/`B`/`C`, which the layout already stores; (b) the gather
`h = [u|v]^T H` and scatter `H -= [u|v] dm / mu` are per-cell transposes, so each cell
is an independent passive block; (c) the opposed-bias (`+z`/`-z`) handedness is
carried by the right-handed local frame, verified. The former guard existed only to
stop the axis-aligned FAST path from applying a single global transverse sign to both
regions; the fast path is now restricted to uniform axis-aligned bias, closing that
hole structurally.

### Known gaps / deferred (unchanged from G3a)

- Bloch-periodic ferrite, FDFD ingest, multi-GPU, adjoint,
  PerturbationMedium-over-ferrite remain fail-closed.
- Identity collocation: the general/oblique path is accurate for smooth/uniform
  fields and passive (non-secularly-growing) in a closed cavity, but it is NOT the
  4-point-collocated Yee gather; a higher-order collocation is a later refinement.
- No wall-clock timing measured (shared-GPU policy).

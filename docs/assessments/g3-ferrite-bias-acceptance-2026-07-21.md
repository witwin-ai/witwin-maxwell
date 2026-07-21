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
| Oblique vs oracle (headline): b=(1,1,1)/√3 CW vs discrete Polder oracle | `test_oblique_cw_matches_oracle` | rel ≈ 1.6e-13 / 1.9e-13 ≤ `reference_polder_rtol`=1e-5 |
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

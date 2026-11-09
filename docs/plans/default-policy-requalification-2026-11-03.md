# Default material policy requalification for v0.6.0

User decision: 2026-11-03. The cylinder absorbed-power and one-cell Rayleigh
amplitude gates retain their qualified `SubpixelSpec(samples=1,
averaging="arithmetic")` policy. Requalifying both under the shipped
`SubpixelSpec(samples=4, averaging="kottke_edge")` policy is a v0.6.0 release
gate. This document defines the measurements, not their eventual acceptance.

The retained batch-4 evidence reports an 8.17% resolved cylinder tier gap
(0.6/0.5 mm, existing gate 6%) and a one-cell sphere maximum RCS ratio of
1.6066 (the arithmetic calibration was 0.35–0.45). Evidence:
`<evidence-root>/integration-batch-4-3dfbda86/` and
`docs/assessments/batch4-owned-failures-2026-11-03.md`.

## Study plan

- `scripts/dev/requalify_sibc_default.py` imports the cylinder scene and inward
  closed-box flux measurement from the pinned test. Resolved material is
  `Material(eps_r=1, sigma_e=30)` at 6 GHz, with spacings 0.6, 0.5, 0.42, 0.35,
  and 0.30 mm. SIBC uses the same conductivity at 1.2 and 0.8 mm. JSON records
  spacing, skin-depth resolution, absorbed power, elapsed time and GPU memory.
  Summaries report each adjacent relative change (fine-tier denominator), the
  coarsest available resolved pair below the requested 6% screen, Richardson
  orders and limits from every consecutive available triple, and each SIBC
  gap against the finest triple's extrapolated reference. Unequal spacing
  ratios are solved directly. Flat, oscillatory or nonconverging data produce
  no extrapolated reference; an earlier fit does not replace a failed finest
  triple. An isolated passing adjacent pair does not establish convergence.
- `scripts/dev/requalify_rayleigh_default.py` imports the shared TFSF fixture
  and its analytic normalization and pattern measurements. The sphere remains
  radius 1 cm, epsilon 4, at 1 GHz; spacings are 1, 0.5 and 0.25 cm (radius
  1, 2 and 4 cells). Domain bounds remain ±20 cm, TFSF bounds ±4 cm, Huygens
  box ±6 cm, with the original time horizon, window and angular grid. Each
  resolution runs its own empty-cell incident-amplitude reference, releasing
  that solver before the scattering solve. NF2FF angle batches are reduced
  to 128 for memory. JSON reports maximum-RCS/analytic ratio and all three
  original normalized-pattern errors (`rel_l2`, `phi0_rel`, `phi90_rel`).

Both preserve the fixtures' eight-cell CPML, so physical CPML thickness changes
with spacing. Geometry snapping, finite time, boundary reflections and finite
size relative to wavelength can remain in the measured trend. The analytic
Rayleigh expression is a small-sphere approximation; its exact continuum ratio
need not equal one at finite `ka` (about 0.21).

For Rayleigh, fit `ratio(h) = ratio_inf + C (h/h_max)^p` separately at fixed
orders 1 and 2. Three points leave one residual degree of freedom per fit;
a free-order three-parameter fit would interpolate exactly and cannot justify
a residual margin. The provisional one-cell band uses the second-order
prediction at 1 cm as center. Its margin is twice that fit's residual prediction
standard error, `2 sqrt(SSE/(n-2)) sqrt(1 + leverage)`, plus the difference
between the two orders' one-cell predictions. This is an empirical calibration
envelope, **not** a statistical confidence interval or an approved release
threshold. Report both continuum limits, residuals and observed Richardson
order. The qualitative "approaches one" diagnostic requires decreasing measured
distance from one and both fitted limits closer to one than the finest tier;
it imposes no numerical acceptance tolerance.

## GPU commands

Run from this worktree after checking local GPU ownership under the shared
workspace rules. The example uses physical GPU 0; select the available owned
device before execution. These commands do not provision cloud resources.

```bash
conda activate maxwell
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/default-requalification-mpl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -e

for tier in resolved-0.6 resolved-0.5 resolved-0.42 resolved-0.35 resolved-0.3 sibc-1.2 sibc-0.8; do
  python scripts/dev/requalify_sibc_default.py --tiers "$tier"
done
for tier in 1 2 4; do
  python scripts/dev/requalify_rayleigh_default.py --tiers "$tier"
done
```

Each loop iteration is a fresh process. To run just one tier:

```bash
python scripts/dev/requalify_sibc_default.py --tiers resolved-0.42
python scripts/dev/requalify_rayleigh_default.py --tiers 4
```

Default output roots are
`<evidence-root>/coverage-default-requalification/sibc/` and
`<evidence-root>/coverage-default-requalification/rayleigh/`.
Use `--out /absolute/path` to override. Each tier is atomically saved as
`<tier>.json`; `summary.json` includes all completed tiers in that directory,
including tiers outside the current `--tiers` selection. Rerunning the commands
resumes completed records. Configuration and study/fixture/solver source hashes
must match; use a fresh directory after source changes. Run tiers sequentially
when sharing an output directory. An interrupted or out-of-memory tier leaves
earlier completed JSON intact and writes no completed record for itself.

Production execution requires `device="cuda"`, native FDTD, seed zero and
deterministic PyTorch algorithms. There is no CPU fallback or backend CLI
override. The PyTorch allocator is capped at `--allocator-cap-gib` (default 4 GiB, or device capacity if less); the cap is recorded per tier and is not part of the resume configuration.
Reported peaks distinguish allocated and reserved PyTorch GPU bytes; CUDA
context and allocations outside that allocator are excluded. A scheduler's
total-process cap remains authoritative; an allocation failure must be reported
and must not trigger an automatic policy, grid or precision change. Measured 2026-11-04 on master 9999c3f4: every tier except the three finest
resolved cylinder tiers completes under 4 GiB (peak 2.84 GiB reserved at
resolved-0.5). Resolved 0.42, 0.35 and 0.3 run out of memory at 4 GiB in
material setup (Cylinder `signed_distance` over the subsample grid), not in
stepping; they ran once with a coordinator-approved one-off 12 GiB cap
(`--allocator-cap-gib 12`). The setup memory is a performance item, not a
policy change.

Summaries can be regenerated without a GPU:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/dev/requalify_sibc_default.py --summarize-only
CUDA_VISIBLE_DEVICES='' python scripts/dev/requalify_rayleigh_default.py --summarize-only
```

## Runtime budget and review

Planning estimates on an RTX A6000, excluding cold native-extension builds:
the measured original four cylinder runs took 228 s, consistent with
`1.3e-7 N^4` seconds for `N = ceil(90 mm / dx)`. Refinement increases both
volume and step count. Estimates are not measured results for these scripts.

| Cylinder tier | Estimated seconds |
| --- | ---: |
| Resolved 0.60 mm | 66 |
| Resolved 0.50 mm | 136 |
| Resolved 0.42 mm | 278 |
| Resolved 0.35 mm | 576 |
| Resolved 0.30 mm | 1053 |
| SIBC 1.20 mm | 4 |
| SIBC 0.80 mm | 21 |

Total cylinder budget is about 36 minutes. The measured 40³ Rayleigh case is
14 s; using the same fourth-power scaling gives 14, 224 and 3584 s for
40³/80³/160³, about 64 minutes total. Smaller NF2FF batches, setup costs and
memory pressure can increase those times; allow roughly two hours for both
studies, plus cold compilation. Record actual wall times rather than replacing
these estimates with assumptions.

After the GPU data are complete, review sign/positivity, order stability,
successive changes, extrapolation sensitivity, SIBC-reference gaps, all pattern
errors and both Rayleigh limits. Decide the default-policy reference grid pair,
SIBC tolerance, one-cell amplitude band, pattern tolerances and any additional
tiers or duration/boundary controls **after** that review. The existing 6%
adjacent-change screen is reported as requested; this plan adds no other
pass/fail thresholds and neither script grants release qualification.

## CPU checks

```bash
conda activate maxwell
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/default-requalification-mpl \
  python -m pytest -q tests/validation/physics/test_default_policy_requalification.py
```

These cover argument selection, synthetic regular/irregular convergence fits,
fit rejection and residual math, partial summaries, resume identity, shared
geometry/policy, and tiny explicit `backend="cpu_reference"` runs. Short smoke
runs exercise compilation/readers and do not measure converged physics. The
cylinder smoke uses a two-step unwindowed DFT. Rayleigh preparation succeeds,
but stepping explicitly rejects the absent CPU TFSF correction kernel
`addBatchedReferenceSourcePatches3D`; its full simulation and NF2FF pipeline
remain GPU-only. No CPU TFSF implementation or fallback is introduced.

CPU verification on 2026-11-03: **15 passed, 8 skipped** across this test module
and both original physics test files, using the `maxwell` environment with
`CUDA_VISIBLE_DEVICES=''`. All skips are GPU physics tests. Both script help
commands and empty-output `--summarize-only` commands also succeed. GPU data,
memory peaks, runtime measurements and release acceptance remain pending.

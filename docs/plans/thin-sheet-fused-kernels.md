# D3b: fused CUDA kernels for the conformal thin-sheet patch step

Status: phase 1 step 1 qualified, 2026-11-01; remaining fusion steps pending.
Owner: PCB line. User decisions: D3 split 2026-10-31 (the copper
half, P-9(c), merged as 04b4e66c); this thin-sheet half starts after the rim fix (master
9e7b0933), with the same formal-tolerance pattern as P-9(c), on ABI v6.

## Goal

The V15 timing gates (`scripts/dev/fdtd/thin_sheet_overhead.py`, uncontended GPU 0):
- copper_patch: sheet / plain <= 1.5x (2026-10-31, before the rim fix: 2.26x, 1.386 / 0.613 ms);
- copper_board_60k: <= 2.0x (3.02x, 3.018 / 1.000 ms).

The sheet overhead must fall from 0.77 to <= 0.31 ms/step (patch) and from 2.02 to <= 1.0
ms/step (board).

## Phase 0: baseline and attribution (first, about 1-2 h)

1. Re-measure V15 on master 9e7b0933 (post-rim-fix) as the D3b baseline, recording whether
   GPU 0 is contended.
2. Profile one steady captured step of both scenarios, attributing every kernel to its call
   site in fdtd/thin_sheet.py:
   - gather/scatter plans;
   - h_matrix / e_matrix mat-vecs;
   - the copper chord history/state/loss;
   - chord_solve (the Chebyshev loop: 2k iterations x one S + two R CSR mat-vecs);
   - the pair 2x2 solves;
   - feed loops;
   - other.
   Produce a bucket table (kernels, us). Report before fusing.

## Phase 1: fuse the largest buckets (by the profile, not by guess)

Approved execution order, step 1 only: gather/scatter fusion is implemented on
ABI v6, default fused on CUDA with the unchanged torch reference and CPU path.
E1 copies and one-step fields are bitwise equal; E3 passes; E4 covers 245 passing
cases and 3 existing F2 xfails. Captured counts are 193 kernels/step in both
scenarios (copy plans 77 -> 7). No timing was performed. See
`docs/assessments/thin-sheet-fused-gather-scatter-step1-2026-11-01.md`.
The remaining candidates are not started; this change stops after step 1.
Performance qualification waits for the coordinator's timing window.

Expected candidates:
- **chord_solve:** use cooperative CSR rows for both S and R, with kernel boundaries
  between residual, temporary and direction updates. Feed-column R blocks can span
  multiple chords, so no per-chord 2x2 assumption or precomputed R^2 is valid in general.
  Keep the polynomial and the degree k exactly as compiled.
- **gather/scatter plans:** one kernel per plan instead of per-component index ops.
- **copper chord history, state and loss update:** fused like P-9(c), FP64.
- **feed column loops:** vectorized once at prepare, not per-step Python. Feeds are rare, but
  the loops are host-bound.

All fused paths are FP64 where the torch path is FP64, and use the field dtype where torch uses
the field dtype (the P-9(c) fp32 lesson). They must be graph-capturable, with no host syncs.

## Reference and selection

The torch step stays unchanged as the reference and as the CPU path. The internal switch is
`WITWIN_MAXWELL_THIN_SHEET_KERNEL=fused|torch`, default fused on CUDA. No public API change.

## Gates (formal tolerance, as P-9(c))

- **E1, one step from an identical state:** FP64 chord_e, pair E and the fields before the cast,
  elementwise max relative <= 1e-13; fp32 fields <= 1 ulp. **Copper modal state (amended
  2026-11-01, user decision): scale-aware elementwise, max|fused - torch| <= 1e-13 * max|torch| per
  tensor**, because per-element relative error is ill-posed on cancellation residuals.
- **E1b:** accumulators (loss, feed column spectra) normwise relative <= 1e-13 per tensor.
- **E2 (user decision 2026-11-02, option A; it replaces the 2026-11-01 amendment):**
  - *Tight early check:* on the V15 patch, the fp32 fields of fused and torch are bitwise equal for
    the first 50 steps (`step2_scenes.py early`).
  - *Well-conditioned run:* the V15 patch driven by a smooth 1.5-cell Gaussian-profile dipole with a
    zero-mean Ricker pulse, sampled at 300/600/1000/2000 steps (`step2_scenes.py smooth`). Each
    component's difference is normalized by that component's PEAK norm over the samples:
    E <= 1e-6, H <= 1e-5, loss and feed <= 1e-6.
  - *Report-only:* the V15 patch, the board and the fed dipole at 2000 steps.
  - *Rationale (the fp32-field floor):* fields are stored in fp32, so once one fp32 rounding
    differs, any non-bitwise implementation follows a different fp32 trajectory. E levels off at
    about 5e-7 (about 8 fp32 ulps) and H, a curl, about 10x higher; FP64 sheet roundoff only decides
    when the first flip happens. Fixed bounds below that floor (H <= 1e-6, loss <= 1e-10) are
    unattainable. Final-norm metrics after a pulse decays, and one-cell ideal-dipole near fields,
    make the comparison ill-conditioned. A single noise-injected control realization varies about
    10x, so it is not a sound yardstick.
    Evidence: docs/assessments/thin-sheet-fused-copper-step2-diagnosis-2026-11-01.md.
- **E3:** graph == eager bitwise on the fused path.
- **E4:** the full tests/materials/thin_sheet suite on GPU passes on the fused default:
  - the CUDA-vs-CPU float64 parity tests (<= 1e-12) and the D6 F1/F3-F7 gates run on the fused
    path, which qualifies it;
  - only the existing torch.equal graph/eager pins that compare to stored torch outputs run
    under a torch-kernel fixture;
  - list each one; no re-pins.
- **E5:** V15 patch <= 1.5x and board <= 2.0x, measured UNCONTENDED on GPU 1 in a
  coordinator-granted window (the same device as the baseline), with kernels/step and the bucket
  table after fusion. Baseline (2026-11-01, GPU 1, master 3616ba32): patch 0.618 / 1.437 ms =
  2.32x; board 1.007 / 3.053 ms = 3.03x.
- **E6:** cuobjdump shows sm_86/89/90/100 SASS plus compute_100 PTX.
- **E7:** census, suite owner and FEATURE_LIST.

Dispatch is proven as in P-9(c): test-only counters show torch 0 calls and fused > 0.

## ETA (wall-clock)

| Stage | Time |
|---|---|
| Phase 0 (V15 baseline + attribution) | 1-2 h |
| Phase 1 implementation | 20-35 Codex h |
| Qualification (E1-E7) | 4-6 GPU-h |
| Review | 1-2 h |
| **Total** | **about 28-45 h** |

Main risk: the Chebyshev loop's iteration count makes it launch-bound even when fused. If
Phase 0 shows chord_solve dominating and per-iteration fusion is insufficient, the next lever
is a single persistent kernel for the whole Chebyshev sweep. That decision is taken after
Phase 0.

## Progress

- Step 1 (17584e7b): fused gather/scatter plans. E1 bitwise (0 ulp), E3 graph == eager, E4 238
  passed plus 3 F2 xfails, dispatch proven. Kernels/step 263 -> 193 (gather/scatter 77 -> 7).

### Step 2b: cooperative copper CSR rows (qualified 2026-11-02, see Measured results)

The coordinator approved changing the summation order within Acx/Acc rows.
Each row now uses eight lanes: lane `l` accumulates entries `start + l + 8*k`
with `__fma_rn`, then an explicit `__dadd_rn` shuffle tree reduces at offsets
4, 2, 1. Each affine element retains the existing `__dmul_rn` / `__dadd_rn`
operations before its product. Sub-warp masks contain exactly the eight row
lanes, including in the final partial block; empty rows contribute zero.
These sums have one writer per row and no atomics.

CPU-only reconstruction of the runtime's CSR operators from the V15 board
complex (60,987 chords, 121,974 rows per matrix) gives Acx row lengths 2--4
(mean 3.00) and Acc lengths 3--8 (mean 7.01; 111,516 rows have seven entries).
The patch (6,120 chords, 12,240 rows per matrix) has Acx lengths 2--4
(mean 3.00) and Acc lengths 3--9 (mean 7.22). Eight lanes cover every board
row and most patch rows in one iteration, with fewer idle lanes than sixteen.

Both row kernels launch 256 threads per block, 32 rows per block, with
`ceil(2 * chords / 32)` blocks. A new copper coupling prepass writes the
existing FP64 `coupled` buffer before the unchanged per-trace modal/RHS work.
The forcing kernel evaluates its Acx/Acc rows cooperatively and writes the
forcing directly. The state/loss kernel and tensor reference are unchanged.
There are no new allocations, host synchronizations or extension API changes.
Same-stream ordering preserves graph capture. Copper launches increase from
three to four per step: coupling, RHS/history, forcing, state/loss. For the
V15 patch and board the expected total is **132 kernels/step**, versus 131
in step 2; this is a source-derived count, pending GPU confirmation.

The added repeat-run test advances two explicitly fused preparations from
identical seeded states for 20 steps, then requires `torch.equal` for all six
fp32 fields, FP64 `chord_e` and FP64 copper modal state. Existing gates,
tolerances and expectations are unchanged. CPU dispatch checks pass (3 tests);
CUDA compilation, numerical gates, determinism, graph parity and timing have
not been run for step 2b.

GPU owner commands, from this worktree with the `maxwell` environment active
and the assigned device visible as CUDA device 0:

```bash
python -m scripts.dev.fdtd.d3b.step2_scenes early --out /tmp/d3b-step2b-early.json
python -m scripts.dev.fdtd.d3b.step2_scenes smooth --out /tmp/d3b-step2b-smooth.json
WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR=/tmp/pcb-d3b-build TORCH_EXTENSIONS_DIR=/tmp/pcb-d3b-build python -m pytest tests/materials/thin_sheet/test_thin_sheet_fused_copper.py --gpu -q -s
python -m scripts.dev.fdtd.d3b.step2_suite --json /tmp/d3b-step2b-E4.json
```

### Step 3-lite: cooperative Chebyshev chord solve (qualified 2026-11-02, see Measured results)

The revised approved design uses arbitrary CSR R and S, including feed columns.
CPU inspection of the existing tilted fed-copper scene found 137 chords, an
R shape of 274 x 274, 250 rows with two entries and 24 with four entries, and
48 entries crossing chord boundaries. R is per-chord 2x2 only in the unfed
construction. No R-squared matrix is formed.

Each nonfinal inner step now launches three kernels on the current stream:
K1 updates y and residual with S times the immutable current direction;
K2 writes temporary = R times the completed residual; K3 updates direction
from R times the completed temporary. The last step launches only K1.
Both outer passes start with two cooperative R passes, with the second
scaling by the host-precomputed inverse theta. The complete rho coefficient
sequence is also computed at preparation. All row products reuse step 2b's
eight-lane accumulation and fixed 4, 2, 1 shuffle tree. Direction products
and the residual/y updates use explicit rounded FP64 operations. Kernel
boundaries supply global ordering; there are no grid barriers or atomics.

A single initialization kernel zeroes y and copies rhs into the residual.
For degree k there are exactly **1 + 6k kernels per chord solve**: at k=3,
**19 kernels**, comprising one initialization, six K1, six K2 and six K3
(including the four setup passes). Each solve owns its returned y; the
residual, direction and temporary buffers are reused from preparation.
The torch solve's arithmetic is unchanged and selected through the existing
`WITWIN_MAXWELL_THIN_SHEET_KERNEL` switch. Empty chord solves retain the
existing clone behavior. Existing tests are unchanged.

The suite dispatch counters now include fused/torch chord solves, and the
captured-count proof checks each chord kernel against the prepared degree.
New CPU controls use kernel doubles to exercise actual preparation, feeds,
both outer passes, skipped final direction updates, stale scratch, rhs
preservation and independent returned storage. These controls do not qualify
the CUDA arithmetic. All 17 selected CPU orchestration/dispatch and extension
loading controls pass. An AST comparison confirms that the moved torch
reference body is unchanged. CUDA compilation and all GPU gates are pending.

GPU owner commands, with the maxwell environment active and the assigned GPU
visible as CUDA device 0:

```bash
python -m scripts.dev.fdtd.d3b.step2_scenes early --out /tmp/d3b-step3lite-early.json
python -m scripts.dev.fdtd.d3b.step2_scenes smooth --out /tmp/d3b-step3lite-smooth.json
python -m scripts.dev.fdtd.d3b.step2_scenes counts --scenario copper_patch --out /tmp/d3b-step3lite-patch-counts.json
python -m scripts.dev.fdtd.d3b.step2_scenes counts --scenario copper_board_60k --out /tmp/d3b-step3lite-board-counts.json
WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR=/tmp/pcb-d3b-build TORCH_EXTENSIONS_DIR=/tmp/pcb-d3b-build python -m pytest tests/materials/thin_sheet/test_thin_sheet_fused_copper.py tests/materials/thin_sheet/test_thin_sheet_fused_copies.py tests/materials/thin_sheet/test_thin_sheet_fused_chord_solve.py --gpu -q -s
python -m scripts.dev.fdtd.d3b.step2_suite --json /tmp/d3b-step3lite-E4.json
```

### Step 3-lite-b: per-matrix lanes and int32 CSR indices (2026-11-02)

The first 3-lite timing passed the patch gate but regressed the board: at a
fixed eight lanes per row, the cooperative kernels were slower per call than
the tensor sparse products on the large board matrices. Each fused CSR plan
(Acx, Acc, R, S, feed) now uses int32 row pointers and column indices, and a
lanes-per-row template L in {1, 2, 4, 8}. L is chosen once per matrix at
preparation as the smallest power of two >= ceil(mean nnz per row / 2),
clamped to [1, 8]. For the V15 patch and board this selects Acx 2, Acc 4,
R 1 and S 8. L=1 is the plain sequential fma loop; larger L keeps the fixed
shuffle tree. The algorithm, the kernel count and determinism are unchanged.
`scripts/dev/fdtd/d3b/chord_kernel_bench.py` times each chord kernel at every
L against `torch.mv`.

## Measured results (2026-11-02)

All gates ran on GPU 0 except timing. Timing (V15, 3 paired trials, median)
ran in exclusive uncontended GPU-1 windows. Artifacts are under
`<evidence-root>/d3b-*`.

| Head | Kernels/step | Patch ratio (overhead ms) | Board ratio (overhead ms) |
|---|---:|---:|---:|
| master baseline (Phase 0) | 263 | 2.32x (0.819) | 3.03x (2.046) |
| step 1+2 (56c7544d) | 131 | 1.61x (0.375) | 2.14x (1.149) |
| step 2b (01895029) | 132 | 1.60x (0.368) | 1.99x (0.996) |
| step 3-lite (5e19986e) | 69 | 1.39x (0.238) | 2.10x (1.107) |
| **step 3-lite-b (3553840e)** | **69** | **1.34x (0.210)** | **1.83x (0.830)** |

Gates: patch <= 1.5x and board <= 2.0x, both met at 3553840e. Step 2b,
3-lite and 3-lite-b each passed, on GPU 0:
- E1 (scale-aware), E3 graph == eager and the repeat-run determinism test;
- early-50 fp32 bitwise on all six components;
- smooth-Ricker E2, exactly 0.0 on fields through 1000 steps; at 2000 steps,
  E <= 1.8e-9 and H <= 1.4e-7, peak-normalized.
The step-2b E4 suite passed 256 tests plus 3 F2 xfails.

Optional future tune (coordinator: freeze at 3553840e, no new cycle). The
microbench shows that the rule's L=8 for S is not optimal. On the board, the
S residual takes 36.5 us/call at L=2 or 4 against 45.4 at L=8; on the patch,
4.2 against 5.4. A rule of pow2 >= ceil(mean/4) (S -> 4) would save about
50 us/step on the board. It changes summation order, so it needs the full
gate set and a GPU-1 timing window.


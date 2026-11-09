# P-9(c): fused CUDA kernel for the planar CopperFoil step

Status: implemented and qualified locally on sm_86, 2026-11-01; E4 retains 14
master-pre-existing failures. See the [qualification report](../assessments/copper-foil-fused-dtype-qualification.md).
Owner: PCB line. User decision 2026-10-31: D3 is split, and this
copper part starts now. Thin-sheet chord fusion waits for the conformal-sheet rim fix.

## Goal

f3_coarse_11f (copper + port, steady captured graph, `scripts/dev/fdtd/copper_step_floor_profile.py`):
from **0.891 ms/step to <= 0.35 ms/step** (target 0.25-0.35), with ~438 kernels per step today, of
which ~392 are torch copper ops.

## What is fused

`fdtd/copper_foil.py` electric step, per copper entry (today about 20 torch ops each):
- gather the low/high normal and tangential H;
- the transverse term and forcing;
- `history = ((B^T * (1+decay)/2) @ state) * admittance_scale`;
- `rhs = rhs_matrix . old_e + forcing - history`;
- `updated = inverse . rhs`, with the activity mask;
- write E (cast to the field dtype) and `right_e`;
- `next_state = decay * state + drive @ (updated + old_e)`;
- loss accumulation;
- `state <- next_state`.

The approved follow-up also fuses begin-step trace snapshots, magnetic
correction and side-spectrum accumulation in one companion CUDA kernel with
separate phase launches. Entry selection, precision and step ordering are the
same as for the electric kernel.
Companion gain/width products remain FP64; final H additions use the field
dtype and the reference indexed-add primitive at the same cast boundaries.

The fusion is one kernel per entry, with faces across blocks and the K = 47 modal states held in
registers or shared memory, all in FP64 (the declared precision; no precision change). It must be
CUDA-graph capturable: no host sync, and fixed launch shapes per prepared solver.

Scope split, by entry flags (not a fallback):
- the fused kernel covers real-field entries without the material trace ADE;
- `paired`/`phase` entries (complex Bloch) and `material_ade` entries keep the torch step. The
  census records this as a performance scope, not a capability raise.

CPU stays torch.

## Reference and VJP

The torch step stays as the reference implementation and as the adjoint/replay path
(`fdtd/adjoint/copper_foil.py` and checkpoint replay are unchanged). The selection is an
internal environment switch, `WITWIN_MAXWELL_COPPER_KERNEL=fused|torch`: the default on CUDA is
`fused`, and CPU is always `torch`. No public API changes.

## Equivalence gate: formal tolerance, not bitwise

Bitwise equality is not achievable: the K-term FP64 dot products and the 2x2 solves change
reduction order and FMA contraction. The gate is therefore:
- **one step**, from identical prepared state: FP64 `state`, `right_e` and updated E (before the
  cast) have max relative difference <= 1e-13 (47-term FP64 dot products give ~5e-15; this
  leaves a 20x margin). The fp32 written field differs by <= 1 ulp;
- **E1b companion spectra**, additionally: each `spectral_real`, `spectral_imag`,
  `spectral_step` and `spectral_norm` tensor has normwise relative difference
  `||fused - torch||_2 / ||torch||_2 <= 1e-13`; a zero reference norm requires
  exact equality. Elementwise relative differences are ill-posed for cancelling
  spectral sums: a residual can be `1e-27` in a tensor with maximum `3.5e-8`.
  E1 retains its original tensors and unchanged elementwise limit;
- **trajectory**, 2000 steps on f3_coarse_11f and on one G2 foil: the fp32 field relative L2
  difference is <= 1e-6 and the cumulative copper loss relative difference is <= 1e-10. A
  passive scheme (rho <= 1) bounds the growth;
- **physics**: the 18 G2 cases, the PCB declared-band foils, and F3 power balance all pass their
  existing tolerances with the fused kernel;
- **graph**: graph and eager are bitwise equal on the fused path (same kernels).

**No bitwise re-pins.** Existing `torch.equal` copper tests pin the reference implementation, so
they run with `WITWIN_MAXWELL_COPPER_KERNEL=torch` through a fixture, and their values do not
move. The fused path is qualified by the tolerance gate above. If the user prefers the fused
path pinned instead, that would need re-pins with approval.

## Build

- ABI bump: `STABLE_ABI_VERSION` goes from `stable_abi_v5r1` to `stable_abi_v6` (v6 is claimed for
  this line; v7 belongs to the data plan).
- `WITWIN_CUDA_GENCODE_ARCHES="8.6;8.9;9.0;10.0+PTX"` must produce sm_86/89/90/100 SASS plus
  compute_100 PTX. The gate is a `cuobjdump --list-elf --list-ptx` check on the built extension.
- Runtime is qualified locally on sm_86 only. A cloud sm_90/100 run would be a separate
  `wcloud plan`, with user approval per launch.

## Gates (fast tier, GPU 0)

- E1 one-step, E1b companion spectra and E2 trajectory equivalence, as above;
- E3 graph == eager;
- E4 the copper test files that import copper_foil (G2, power loss, regional power,
  curved-copper runtime, roughness opt-in identity) pass with fused as the default, and the
  bitwise ones pass under the torch fixture;
- E5 timing: f3_coarse_11f copper + port <= 0.35 ms/step on uncontended GPU 0, with
  kernels/step reported;
- E6 the cuobjdump arch check;
- E7 the guard census and integration_suites owner for any new test file, plus FEATURE_LIST.

## ETA (honest, wall-clock)

- Codex implementation (kernel, binding, dispatch, tests): 8-12 h;
- GPU qualification (E1-E6): ~2 h;
- review and fixes: 1-2 h.

That is about 12-16 h to merge-ready. Batch 3 or the next one.

## Qualification record (2026-11-01)

- **E1 scope (user-approved, applied as written):** FP64 `state`, `right_e` and updated E, elementwise
  max relative <= 1e-13, plus the fp32 field <= 1 ulp. Measured: 1.478e-14 and 0 ulp.
- **E1b (an additional gate, not a loosening):** spectral real/imag accumulators, clocks and
  normalization have per-tensor NORMWISE relative difference <= 1e-13 (measured 9.5e-18).
  Elementwise relative error is ill-posed for these tensors. The single motivating element was
  `spectral_imag` = -1.71e-27, a cancellation residual of +/-3.55e-25 terms against a tensor max of
  3.5e-8, with an elementwise relative difference of 1.07e-13 that traces to a 4.45e-16 `Ex`
  difference. E2's loss <= 1e-10 still covers the accumulated spectra.
- **fp32 semantics:** the fused magnetic correction performs its final H additions in the field
  dtype with torch's indexed-add primitive, matching the reference operation by operation. Doing
  them in FP64 and then casting gave subnormal-scale differences (a state element of 1e-45, one
  Ex element 51k ulp off).
- **Result:** E1-E3 and E5-E7 pass. E4 shows no new failures (14 master pre-existing in
  test_short_spectral_tail_execution.py). f3_coarse_11f fused runs at 0.293 ms/step clean, with
  54 kernels/step (torch: 0.92 ms/step, 438 kernels/step). The largest remaining bucket is the
  electric kernel at 126 us/step (one 128-thread block per entry), the next lever if needed.

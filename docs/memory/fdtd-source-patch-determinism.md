# Ordered CUDA source-patch accumulation

Branch: `fix/deterministic-source-patches`, based on `28bf2808`.
Status: CPU validation and offline `sm_86` compilation only; GPU gates pending.

## Cause and dispatch

An oblique Gaussian-pulse PlaneWave with direction `(0.6, 0, 0.8)` and `Ey`
polarization selects `plane_wave_aux` in `excitation/tfsf_state.py`.
`_apply_tfsf_terms` in `tfsf_apply.py` selects `_apply_batched_aux_terms`, then
`addBatchedInterpolatedSourcePatches3D`. The CUDA backend previously selected
the warp kernel for int32 offsets and the ordinary atomic kernel for int64
offsets. Separate TFSF faces can contribute to the same Yee field cell; warp
aggregation cannot order contributions from different warps or blocks.

A CPU-only preparation of a vacuum box with domain `[-0.96, 0.96]^3`, spacing
`0.08`, four CPML layers, TFSF bounds `[-0.32, 0.32]^3`, and that direction
produced eight electric patches: 576 contributions, 544 targets, 32 targets
with two contributions. Its four magnetic patches had 288 distinct targets.
The `(0, 0, 1)`/`Ey` and `(1, 0, 0)`/`Ez` controls select
`plane_wave_axis_aligned` and `plane_wave_ref_x_ez`, respectively. Each has four
patches and 288 distinct targets per half-step. Zero vector components remove
the extra correction terms, so these axis-aligned controls have no competing
writes. This confirms the routing cause without claiming a GPU reproduction.

Ordinary sources dispatch through `injection.py` and `temporal.py`; periodic
patches and complex Bloch patches can also have overlapping wrapped images.
BatchSourceEngine retains one patch ordinal per lane per launch, but images
of a patch can still overlap.

## Plan, execution, and numerical order

`excitation/gather.py` builds immutable CSR tensors at preparation:

- `rows`: int64 row starts, including the terminal sentinel;
- `targets`: int64 `(field component, flattened field offset)` per unique cell;
- `entries`: int64 `(original coefficient index, image flags)` per contribution.
  Bits 0–2 select wrapped axes; bit 3 marks an entry that formerly used atomics.

The contribution order is patch ordinal, then C-order entry index within that
patch, then increasing wrap-image subset. Stable sorts group targets without
changing that order. The existing component-specific image order is base,
axis A, axis B, both; A and B are increasing tangential axes. Core and batched
sources use increasing subsets of the three spatial axes. Lane bases stay in
target offsets, and images never cross lanes. Out-of-bounds patch entries are
clipped before sorting, retaining their original coefficient indices.

One CUDA thread owns each target, loads its current field, and adds each
already-evaluated contribution in order before storing once. It never sums
the deltas separately from the field. This is the same recurrence as
stream-ordered per-patch `+=` with the existing compiled coefficients and the
same per-entry rounding. Distinct target operations commute. Therefore a
collision-free old scene retains its arithmetic, while overlapping patches
now have a defined serial order. A former warp reduction over duplicate
targets may have a different association; those cases are potential movers.

Rounding was checked against offline master `sm_86` PTX:

- Batched reference/interpolated deltas retain rounded coefficient products;
  interpolation retains its original expression and contraction.
- Wrapped periodic specializations round the product even for interior
  entries. The no-wrap specialization retains its fused multiply-add.
- Bloch base deltas round both products, including interior cells. Phase
  products, sums, and the positive corner rotation preserve the former
  contraction explicitly with CUDA rounding intrinsics.
- Batched CW keeps its fused cosine/sine delta followed by a separate add;
  unwrapped ordinary batch updates retain their fused multiply-add.
- Former global-float atomic additions retain their flush-to-zero arithmetic
  through `add.rn.ftz.f32`; ordinary interior additions retain normal rounding.

The single-patch `add_interpolated_source_patch_kernel` is unchanged. Reference
and interpolated batched entrypoints share the gather kernel for both offset
dtypes; warp aggregation is removed. Periodic/Bloch launch names, separate CW
quadrature launches, source ordering, and capture/core admission are preserved.
`runtime/core_tables.py` builds the same clipped periodic plan for core sources;
`core/step.cu` uses a cell gather instead of wrapped scatter writes.

Plans are bounded to 8,000,000 candidate entry/image pairs before expansion.
Oversized plans raise at prepare; allocation failures propagate. There is no
atomic fallback. Normal prepared stepping reuses tensors without rebuilding or
host transfers. Batch source construction runs during cohort preparation,
before stepping. Ad-hoc backend calls construct their routes before invoking
the compiled operator. Stable extension ABI is `stable_abi_v9`; core ABI is
`core_abi_v5`, preventing reuse of binaries with the old operator/table layout.

## Potential expectation movement

Only cells with multiple contributions can change summation order. Candidate
pytest files with such cases or fixtures are:

- `tests/fdtd/cuda/test_cuda_sources_parity.py`: explicit duplicate-target
  reference/interpolated patches and periodic/Bloch patches covering both seams;
- `tests/fdtd/test_cuda_graph_tfsf.py`: the oblique pulsed preparation fixture
  has shared targets; its admission assertions remain unchanged;
- `tests/sources/tfsf/test_pulsed_incident_reference.py`: oblique preparation
  fixtures have shared targets; auxiliary-line-only expectations do not change.

The separate coverage branch supplies the new determinism assertions. No
duplicate determinism tests or baseline re-pins are introduced here. Existing
test adapters accept the prepared-route argument without changing reference
arithmetic. The CW phase fixture uses valid 3D patches, and the ABI expectation
advances with the changed internal operator signatures.

All source/field configurations in `tests/fdtd/_bit_identity_probe.py` were
prepared on CPU for route inspection, including the Bloch slab and periodic
dipole. Every affected gather had one contribution per target. The two
boundary probe configurations using those paths also use owned periodic
samples. No historical BITID expectation is predicted to move. Run all seven:

1. `tests/fdtd/test_bit_identity_baseline.py`
2. `tests/boundaries/absorbers/test_boundary_bit_identity.py`
3. `tests/monitors/in_simulation/test_existing_monitor_bit_identity.py`
4. `tests/rf/lumped/test_port_step_graph_bit_identity.py`
5. `tests/rf/lumped/test_rf_port_bit_identity.py`
6. `tests/rf/terminal_wave/test_terminal_wave_bit_identity.py`
7. `tests/materials/surface_impedance/test_roughness_opt_in_bit_identity.py`

The remaining BITID fixtures use ordinary per-patch sources or source-free
port probes. This prediction still needs GPU confirmation; any movement must
be reported without re-pinning. Scalar waveform evaluation, compiled source
inputs, and configuration fingerprints are unchanged.

## Remaining atomics outside this change

`kernels/adjoint.cu` has 36 atomic sites: auxiliary-line pullbacks, duplicate
sample accumulation, point-monitor adjoint seeding, and the anisotropic
transpose stencil. Multiple writers can make these adjoint fields or gradients
nondeterministic. They do not write the forward E/H fields.

`kernels/copper.cu:170` corrects the normal magnetic forward field. Its double
intermediates are cast to the field type, so the template instantiates float or
double atomic additions. `prepare_copper_electric` makes `normal_targets`
unique; each target's two sides execute in a single thread, and separate foil
launches are stream ordered. Thus this supported path touches forward fields
but has no competing target writers and is deterministic. A malformed plan
with duplicate targets could make it nondeterministic.

## Validation and GPU work

182 targeted CPU tests passed; 55 GPU-only cases were skipped. Separate CPU
checks cover 288 clipped/image configurations, repeated plan equality,
int64/component ordering, lane isolation, serial field-add order, bounded
failure, public CPU/source behavior, and existing phase/ABI contracts.
Both native extensions compile offline for `sm_86` with CUDA initialization
disabled. The coordinator must run three raw-field repeats for oblique TFSF,
periodic, Bloch and batched sources; all seven BITID files; periodic core G1
at K=32/16/1; and an oblique step-time comparison against master using the
median of five repeats. No GPU numerical or performance claim is made here.

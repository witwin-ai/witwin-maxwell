# Array workflow Phase 1 acceptance

Status: functional and numerical gates accepted; frozen performance qualification pending

Date: 2026-07-16

Scope: one CUDA device. Task-level multi-GPU execution was removed from scope by the
user and is not claimed here.

## Delivered contract

- `PortSweep` retains compact closed-surface results and measured incident waves for
  every physical basis column without retaining full-volume solver state.
- `Result.array_basis(...)` extracts measured-wave-normalized embedded element
  patterns without rerunning FDTD and preserves manifest port/channel order.
- `ArrayBasisData.combine(...)` evaluates active network quantities, complex far
  fields, realized gain, EIRP, and absolute radiated power from the Hermitian
  closed-surface operator `Q_rad`.
- Monitor phasors use the exact Yee E/H time schedules and exact primal-cell face
  quadrature. The adjoint implements the corresponding transposed schedules.
- Tensor dtype/device matching is exact except for the documented complex64 solver
  phasor to complex128 `NetworkData` promotion.
- Persistence reuses the safe `NetworkData` schema and `torch.load(weights_only=True)`.
  Saved `Result` objects intentionally omit ephemeral basis columns.

## Numerical acceptance

The canonical CUDA scene is a four-element, half-wavelength-spaced linear fed array at
1 GHz on an exact `96 x 96 x 96` cell grid with eight absorbing cells on every face,
4,096 time steps, and a `181 x 361` full-sphere angular grid. No global phase or
amplitude fit is applied.

| Comparison | Weighted complex L2 | Phase RMS | Incident power rel. | Reflected power rel. | Accepted power rel. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broadside basis vs direct multi-source FDTD | 1.766e-6 | 1.004e-4 deg | 4.183e-8 | 6.584e-8 | 3.103e-8 |
| Endfire basis vs direct multi-source FDTD | 2.219e-6 | 1.518e-4 deg | 8.520e-8 | 9.696e-9 | 1.452e-7 |

The maximum accepted-to-radiated physical power residual is `6.997e-4` (0.0700%),
below the 1% gate. The measured `Q_rad` is Hermitian to machine precision and its
minimum eigenvalue in the independent review was `2.47e-4`.

## Test evidence

- Full Phase 1 RF/FDTD regression: `257 passed`.
- Independent focused acceptance rerun: `33 passed`.
- Real CUDA full-wave coverage includes two- and four-element broadside/endfire
  comparisons, direct simultaneous lumped sources, complex port powers and far
  fields, and a two-frequency WavePort basis.
- Observer quadrature covers uniform, nonuniform, and single-cell tangential axes.
- Ruff and `git diff --check`: passed.
- Independent API/RF contract reviews: passed after dtype/device and content
  fingerprint findings were closed.
- Independent Phase 1 review: implementation passed; it required the frozen
  performance record and direct adjoint time-schedule tests before commit.

## Performance evidence

An exploratory two-sample, two-alternating-round run produced the following four raw
samples on the local NVIDIA GeForce RTX 5080 (16,303 MiB, driver 596.49, torch 2.10,
CUDA 12.8):

| Workflow | Raw CUDA-event seconds | Median |
| --- | --- | ---: |
| Four basis solves plus 16 combinations | 77.4324, 78.1716, 97.1647, 78.7779 | 78.4747 |
| Sixteen direct solves | 390.8792, 484.3927, 391.6239, 372.8254 | 391.2515 |
| Sixteen combinations only | 0.002893, 0.003471, 0.003694, 0.002170 | 0.003182 |

The exploratory basis/direct ratio is `0.2006` against the `<= 0.40` gate, and the
combine/one-solve ratio is `0.000130` against the `< 0.10` gate. This run is not the
frozen qualifying protocol. Qualification requires three warmups, five samples, and
four alternating order rounds; its complete raw record will be added before Phase 1
is committed.

## Commands

All commands use the `witwin2` environment and worktree-local CUDA/build caches.

```text
python -m pytest -q tests/fdtd/test_observer_quadrature.py tests/fdtd/test_observer_time_stagger.py tests/gradients/test_fdtd_rf_lumped_adjoint.py tests/rf/contracts/test_network_data_contract.py tests/rf/network tests/rf/array tests/rf/antenna tests/rf/lumped tests/rf/waveport
python -m benchmark.array_phase1 --output .cache/array_phase1_qualification.json
```

## Diagnostic history

One frozen-protocol attempt reached repeated direct solves and then reported an
asynchronous CUDA illegal-memory-access error. Shorter pre-fix 1x1 and 2x2 runs passed.
The benchmark now synchronizes each direct beam before its temporary `Result` can be
released. Runs affected by shared GPU contention or external timeout are retained only
as diagnostics and are not qualification evidence.

Post-fix qualification is currently blocked by the local GPU/driver execution state.
Three isolated retries, including a stage-logged diagnostic with no competing Python,
failed to complete the first canonical `_build_basis` within 30 minutes; the progress
log never advanced beyond scene construction. During the wait the process accumulated
only seconds of CPU time and the GPU remained mostly at 0%/P8 while retaining the CUDA
context. A separate synchronized CUDA tensor probe completed successfully in 2.8
seconds, so the failure is specific to the large solver workflow. No shared lock or
compiler process was present. A clean driver/host execution window is required before
the frozen 3/5/4 qualification can be rerun.

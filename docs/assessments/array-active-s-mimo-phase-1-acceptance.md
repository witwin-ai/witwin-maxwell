# Array workflow Phase 1 acceptance

Status: committed as `3223e0c` (checkpoint); functional and numerical gates accepted on
that commit. On the final integrated tree (`4b24b60`) the frozen qualification attempt of
2026-07-17 FAILED the physical-power-closure numerical gate (2.865% vs 1%), so the frozen
performance qualification is still not obtained and is now blocked by a numerical
regression rather than by host execution state. See "Frozen qualification attempt
2026-07-17" below.

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
below the 1% gate.

### Retracted: the `Q_rad` Hermiticity claim

An earlier revision of this document offered "the measured `Q_rad` is Hermitian to
machine precision" as evidence. That claim is **retracted as tautological**.
`postprocess/array.py` constructs the operator as `0.5 * (M + M.mH)`, which is the
correct extraction of the Hermitian (real-power) part of the complex Poynting operator
but also makes the Hermiticity residual identically zero by construction. Re-measured on
2026-07-16 the residual is exactly `0.000000e+00`, not a small machine-precision value —
the signature of an identity, not of a passing test. The check cannot fail and therefore
proves nothing about the solver.

The same revision reported a "minimum eigenvalue in the independent review of `2.47e-4`".
That number is real but **unrepresentative**: it is the two-element case only. The
property that carries physics is positive semidefiniteness, because `real(a^H Q_rad a)`
is radiated power and a negative eigenvalue means some excitation radiates negative
power.

Re-measured on the contract scene (`maxwell` env, one RTX A6000), holding grid and
NF2FF surface fixed and varying only the PML thickness:

| Case | PML layers | min eigenvalue | max eigenvalue | min/max |
| --- | ---: | ---: | ---: | ---: |
| Two-element | 2 | 2.469986e-04 | 5.082438e-03 | 4.859846e-02 |
| Four-element | 2 | -2.394505e-07 | 8.453165e-03 | -2.832672e-05 |
| Two-element | 4 | 2.304486e-04 | 5.467894e-03 | 4.214577e-02 |
| Four-element | 4 | 1.312841e-08 | 9.058985e-03 | 1.449214e-06 |

Previous diagnosis: the negative four-element eigenvalue at 2 PML layers was attributed
to quadrature/truncation error on a near-rank-deficient sub-wavelength operator, and gated
with a `-1e-3` relative floor. That diagnosis was **refuted** by refinement: the gate
metric (min/max) does not shrink under grid or time refinement, and the sign of the
smallest eigenvalue flips with PML thickness alone (2 layers: negative; 4 and 6 layers:
positive). The root cause is PML under-absorption: the `0.05 m` NF2FF box sits ~1 cell
inside the `0.06 m` interior, so at 2 layers reflected field contaminates the
closed-surface complex-Poynting integral and drives `Q_rad` indefinite.

Fix: the contract scene now uses 4 PML layers (`tests/rf/array/test_array_fullwave.py`),
which restores a genuinely positive-definite spectrum (four-element min/max `+1.449e-6`).
`radiated_power_psd_relative_floor` was tightened from `1e-3` to `1e-9` — an `eigvalsh`
roundoff band (~`1e-16 * max_eig`) rather than a floor loose enough to bless the
under-resolved 2-layer scene — and the gate now also requires `max_eig > 0`, in both
`test_array_fullwave.py` and the benchmark, so a total sign inversion cannot pass. The
Hermiticity guard in `ArrayBasisData.__post_init__` is retained: it is unreachable from
the internal path but still fail-closed for directly constructed or deserialized
operators, and `test_radiated_power_operator_rejects_non_hermitian_matrix` now covers it.
The frozen 96^3 benchmark scene already uses 8 absorbing cells per face and is unchanged.

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
- Independent Phase 1 review: implementation passed; it asked for the frozen
  performance record and direct adjoint time-schedule tests before commit. The
  implementation was committed as `3223e0c` without the frozen performance record, so
  that condition is outstanding against a commit that already exists rather than a
  pending one.

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
four alternating order rounds. That record does not exist: Phase 1 was committed as
`3223e0c` without it, so it is now owed against a landed commit. The numbers above are
exploratory and are not qualification evidence.

The recorded host (RTX 5080, driver 596.49, torch 2.10, CUDA 12.8) is **not** the
current qualification host, which has two RTX A6000 GPUs on torch 2.13/CUDA 13.0. The
timings above therefore cannot be reproduced as-is and must be re-measured on the host
that will issue the frozen record. `AcceptanceBudget.local_hardware` still names the
5080 and is stale for the same reason.

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

## Frozen qualification attempt 2026-07-17 (final tree `4b24b60`, A6000 host): FAIL

The frozen 3-warmup/5-sample/4-round qualification was rerun on a clean host to
re-anchor the performance contract: 2x NVIDIA RTX A6000 (driver 595.71.05, torch
2.13.0+cu130, CUDA 13.0), one quiet GPU, `numactl --cpunodebind=0 --membind=0`, governor
`performance`, no competing compute. The host execution problem described above did not
recur; the harness ran to the numerical gate in under a minute.

Outcome: FAIL at the physical-power-closure gate, which the harness asserts before the
timing loop. Measured `max_physical_power_residual = |P_accepted - P_rad| / P_incident =
0.028646 (2.865%)` against the 1% frozen gate; the recorded `3223e0c` value was `6.997e-4`
(0.07%). The timing loop was never reached, so no basis/direct or combine timing ratio was
measured and the performance contract remains unqualified.

All other numerical gates passed with wide margin on this host:

| Comparison | Weighted complex L2 | Phase RMS | Incident power rel. | Reflected power rel. | Accepted power rel. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broadside basis vs direct | 7.839e-7 | 4.689e-5 deg | 1.040e-7 | 9.952e-8 | 9.993e-8 |
| Endfire basis vs direct | 8.970e-7 | 6.518e-5 deg | 1.334e-8 | 7.031e-8 | 6.718e-8 |

`Q_rad` spectrum: min eig `0.442`, max eig `0.929`, min/max `0.476`, `max_eig > 0` (PSD).
Grid `96^3`, 8 PML cells/face, 4096 steps, `181x361` angular grid, 16 beams: all match.

Diagnosis: the basis-vs-direct comparisons are a same-solver linearity check and are
insensitive to absolute calibration, so their agreement (in fact tighter than the
`3223e0c` record) shows superposition is intact. The power-closure gate is the only one
that compares the network accepted power against the NF2FF-integrated radiated power, so
the 2.865% mismatch is an absolute NF2FF radiated-power calibration regression introduced
between `3223e0c` and `4b24b60` — the spectral observer colocation (`21be130`) and the
NF2FF quadrature clip (`3f13ff2`). The same-day external-reference refresh
(`benchmark/RESULTS.md`, `7932af3`) corroborates this independently and shows it is not
confined to the array: 242 accuracy cells worsened vs the ad3427f baseline (187 improved,
910 unchanged), dominated by a systematic flux (Poynting) regression on nearly every
flux-comparison scenario (worst `mixed_faces` `1.004e-2 -> 1.444e0`) while field
correlation stayed unchanged, plus far-field/RCS/directivity complex-error regressions.
This is a numerical regression to be resolved before the frozen performance qualification
can be re-attempted; it is not a host or timing artifact.

Evidence: `docs/assessments/array-active-s-mimo-phase-1-qualification.json`.

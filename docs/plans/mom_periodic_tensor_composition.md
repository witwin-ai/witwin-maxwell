# Periodic scalar exterior with tensor interiors

## Implementation and scope

Periodic dense assembly combines one scalar Ewald exterior across all body pairs
with body-local tensor or scalar primary-image interiors. It adds no interior
images and no duplicate incident carrier forcing. Regional dyadic kernels are
shared with field recovery. Schema 16 binds sampled media, lattice, spectral
policy, cache budgets and the combined representation. Restore rejects altered
identities without assembling a new operator; Bench preserves native current
bytes and both periodic/dyadic policies.

General tensor interior recovery checks E and vacuum-impedance-scaled H separately
for every query. Isotropic tensor samples retain the scalar norm. Tensor periodic
H/H2/FMM and harmonic balance remain rejected and remain full-goal obligations.
GSP-01, all original physical/dimensional/combination/complexity gates and final
four-repository acceptance remain OPEN. None of this stage grants production
qualification. Open/nested surfaces and the original excluded scope remain excluded.

## Completed targeted evidence

- /tmp/gsp-periodic-tensor-targeted-v1: Maxwell 33 passed, Bench 2 passed, no skips;
  recorded source hashes unchanged. This covers 1D/2D two-body scalar limits,
  nonzero Bloch translation, exact restoration/tamper rejection, exterior-only
  image ownership, manufactured E/H gates, propagated tensor failure and affected
  scalar periodic fields/FMM. The Bench candidates preserve declared reciprocal
  and gyrotropic tensors. 3D routing includes spectral-pole rejection; it does
  not establish 3D public physical qualification.
- /tmp/gsp-periodic-tensor-bench-host-v1.{log,xml}: five synthetic schema checks
  passed. They were not repeated after unrelated controls.
- Actual scalar-limit public saved roots pass schema 16 Bench write/read with
  byte-exact J/M and operator construction/direct solves forbidden:
  /tmp/gsp-periodic-tensor-bench-replay-1d-v1 manifest
  a830b4fea15c34b98871069d24da1057ae85187e3fc5c4f0fda0905bba472bea;
  /tmp/gsp-periodic-tensor-bench-replay-2d-v1 manifest
  a56f439facd8997c1e7c40dc9ededb4a0cbce480687fa84f6037543a06903c2f.
- The prior frozen merge regression remains valid and separate: Maxwell a01549b
  tests/mom 1093 passed/3 skipped; Bench c5b566d 1653 passed/36 skipped, all 720
  recorded hashes unchanged and independently audited. Receipt at
  /tmp/gsp11-merge-regression-v3 has SHA256
  f6a1f2f0de698213344002c308ea37bb8b741a72fc6f6997c01335f6429d75bc.
  It excludes this periodic WIP, external dependencies outside the hash scope,
  full Maxwell/four-repository qualification and physical gates. No duplicate
  full regression was run for this targeted composition stage.

## Genuine anisotropic controls and exact archives

The controls use Scene -> Simulation -> Result, a fixed four-face/six-edge
anisotropic tetrahedron, zero Bloch vector, unchanged material and spectral
budgets. Initial surface/singular order is 1; spectral rtol/atol are 1e-4/1e-9,
maximum radial/azimuthal orders 512/64. Original balanced residual must be below
1e-8 and retained spectral caches must fit the 1 GiB body budget.

Both controls are terminal measured, solver_converged and exact_restore true,
physical_qualification false. Both retain 699376704 cache bytes, peak allocated
849462784 and peak reserved 1054867456 bytes. They restore coefficients and
exterior scattered fields exactly; they do not establish independent accuracy.

| Dimension | Evidence directory | Balanced residual | Seconds | Result SHA256 |
| --- | --- | --- | --- | --- |
| 1D | /tmp/gsp-periodic-tensor-anisotropic-1d-v1 | 2.623758529190355e-15 | 355.076250352 | 3c5272d2598282faf2c0caad836172e72674376609a8e8ad04209c733d72e793 |
| 2D | /tmp/gsp-periodic-tensor-anisotropic-2d-v1 | 2.6839685241008263e-15 | 386.002095357 | bf41ce52828958c6b9e6279b2aa44756e36f98ff1a51d852c4a174ad3d21b329 |

The genuine saved roots also pass Bench exact archive replay, with no new solve:
/tmp/gsp-actual-periodic-tensor-bench-replay-1d-v1 and -2d-v1. Their manifests are
482822ba43eb239e4fdb4b3217b711d436698d1d98e73801222fb32e1a2059d3 and
70efa5e00afce1e725f9e209b7f036ce2f5cb1727a5b72ed09d5b5cbfdd29b84.
The reviewed harness /tmp/gsp_actual_periodic_tensor_bench_replay.py binds the
producer/source/result/helper digests and reconstructed scene, asserts the actual
import trees, forbids assembly and direct solves, and checks schema 16,
off-diagonal/nonisotropic permittivity and byte-exact J/M dtype/shape/data.
Source hashes remain unchanged; physical qualification remains false.
The scalar-limit replay harness is not suitable for these distinct scenes.

## Preserved interior and trace failures

/tmp/gsp-periodic-tensor-fields-1d-v1 is terminal FAILED, session 67909 exit 1,
elapsed 16194.904936773703 seconds. It reuses the exact 1D root and forbids root
construction/assembly. Both interior and face-interior trace fail at surface
order 8 with surface_quadrature_budget_exhausted; all order 2/4/8 checks fail.
Adaptive gates remain rtol 1e-4, atol 1e-9, separate E and eta0 H at both translated
points. Zero Bloch only tests unit-phase translation, not nonzero-phase physics.

| Case | Last E difference | E threshold | Last eta0 H difference | eta0 H threshold |
| --- | --- | --- | --- | --- |
| interior | 0.005369879223194581 | 7.567800602066272e-5 | 0.006573060614131444 | 9.79176460440684e-5 |
| interior_trace | 0.0007118982015596215 | 7.516612471447921e-5 | 0.0013869639951846816 | 0.00010174202131919294 |

Both original failure.pt files preserve their actual receipts. Independent
terminal evidence audit verified all 158 source hashes before/after/current,
producer/result binding and unchanged thresholds. The admitted algebraic root
remains distinct from failed field quadrature. physical_qualification,
ewald_truncation_verified and certified_error_bound all remain false.

## Separate field integration optimization

Read-only inspection found two evaluations of the same spatial Green nodes for
scattered and incident sheets. Mode caching does not share spatial evaluations
between these calls. The all-near four-face/two-query sequence at orders 1/2/4/8
can evaluate 4080 displacement nodes across both sheets; trace adds PV annulus
work. No duplicate nested surface adaptivity or forced per-order mode-cache
clearing was found. Runtime alone was not used as evidence of failure.

The shared-sheet integration change was developed in the separate clone
<scratch>/witwin-maxwell-tensor-field-fusion on branch gsp-tensor-field-fusion.
It combines linear current/divergence samples before kernel application while
preserving the separate raw/carrier jumps. Independent review found and fixed a
test-only frozen-dataclass monkeypatch error; no further confirmed blocker was
found. /tmp/gsp-tensor-fusion-targeted-v1 is terminal PASSED: 26 tests, no skips,
25.134076808 seconds including process overhead, unchanged source hashes.
Coverage includes six new scalar-limit/manufactured equivalence, gradients,
PV/jump and adaptive acceptance/rejection cases plus affected field/public paths.
It does not prove genuine tensor spectral accuracy, near-threshold cancellation
stability or wall-clock speedup. After composition commits Maxwell d5b4211 and Bench 4690caa, only the field
implementation and new test module were integrated. All 586 numerical/test
source hashes in the integrated tree match the validated clone exactly; the
unchanged tests were not rerun. The optimization receives a separate commit. Independent integration audit
verified all 586 hashes and the 26 passing XML cases, unchanged PV/jump/sign and
threshold behavior, and no additional numerical WIP; no confirmed blocker was
found. Genuine anisotropic equivalence remains a subsequent control.

## Remaining work

All sequential runners (field 67909, 2D 34645, fusion 59060, Bench 93114) are
terminal. Do not rerun their roots, tests or archives without a relevant change.
The independent composition stage audit found no confirmed code blocker and
verified targeted XML, current source hashes, both genuine producer admissions,
all archive array digests and byte-exact J/M against original results. This is an
engineering stage audit, not full-goal acceptance. Composition commits are
Maxwell d5b4211 and Bench 4690caa; review and commit the integrated fusion change
separately. Preserve all
physical failures and continue the original dimension/material/fast-backend and
nonlinear requirements. Full regression and independent final audits remain at
actual major merge/final acceptance points.

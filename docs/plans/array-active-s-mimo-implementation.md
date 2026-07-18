# Array, active-network, and MIMO implementation record

Status: active

Source plan: `next-functional-2026-07/06-array-active-s-mimo.md`

Architecture: FDTD-only, GPU-first, PyTorch-native, `Scene -> Simulation -> Result`

## AcceptanceBudget (frozen)

The following thresholds are frozen before full-wave implementation in the internal
`ARRAY_ACCEPTANCE_BUDGET` constant and mirrored here. Threshold changes
must retain the previous value, measured evidence, and a technical reason.

| Gate | Threshold |
| --- | --- |
| Pure torch analytic combination, float64/complex128 | `rtol <= 1e-6`, `atol <= 1e-10` |
| CUDA complex64 device parity | `rtol <= 2e-5`, `atol <= 1e-6` |
| Basis versus direct FDTD complex vector far field, converged Phase 1 scene | solid-angle-weighted relative L2 `<= 1e-4` |
| Basis versus direct FDTD phase, converged Phase 1 scene | wrapped RMS `<= 1e-2 deg` above 10% of peak field; no global phase alignment |
| Basis versus direct FDTD complex vector far field, coarse contract scene | solid-angle-weighted relative L2 `<= 5e-3` |
| Basis versus direct FDTD phase, coarse contract scene | wrapped RMS `<= 0.5 deg` above 10% of peak field; no global phase alignment |
| Basis versus direct port powers | incident/reflected/accepted relative error `<= 0.5%` |
| Passive physical power closure | `abs(Paccepted - Prad - Ploss) / Pincident <= 1%` |
| `Q_rad` positive semidefiniteness | `max_eig > 0` and `min_eig >= -1e-9 * max_eig`, reduced globally over the spectrum (contract and benchmark scenes are single-frequency, so global and per-frequency coincide) |
| Independent reference | realized-gain error `<= 0.25 dB`; ECC error `<= 0.02` |
| Active impedance where reference data exists | magnitude error `<= 5%`, phase error `<= 3 deg` |
| Weight and supported scene gradients | relative error `< 2%`, absolute floor `1e-8` |
| Inherited distributed diagnostic-field parity | max absolute error `<= 2e-6`, significant-field max relative error `<= 2e-5` |
| Domain-decomposition monitor/EEP aggregation | `rtol <= 5e-5`, `atol <= 5e-6`, exact public port order |

### Threshold change record, 2026-07-16

Basis-versus-direct is a linearity check on one solver, so it must agree to solver
precision, not to an engineering tolerance. The single `0.03` / `3 deg` pair was 4 orders
looser than the converged measurement and could not discriminate a real superposition
regression from noise. It was also applied to two scenes whose truncation errors differ
by three orders, so it is now split per scene.

| Gate | Old | New | Measured worst case | Reason |
| --- | ---: | ---: | ---: | --- |
| Converged-scene far-field L2 | 0.03 | 1e-4 | 2.219e-6 (recorded, endfire) | 45x margin over recorded evidence; 300x tighter |
| Converged-scene phase RMS | 3 deg | 1e-2 deg | 1.518e-4 deg (recorded, endfire) | 66x margin over recorded evidence; 300x tighter |
| Contract-scene far-field L2 | 0.03 | 5e-3 | 1.433e-4 (re-measured, four-element endfire, 4-layer PML) | 35x margin; the 192-step scene truncates a Gaussian pulse before it decays, so it cannot reach converged-scene precision |
| Contract-scene phase RMS | 3 deg | 0.5 deg | 6.776e-3 deg (re-measured, four-element endfire, 4-layer PML) | 74x margin, same reason |
| Port power relative error | 1% | 0.5% | 9.015e-6 (re-measured, two-element endfire accepted power, 4-layer PML) | large margin |
| Contract-scene PML | 2 layers | 4 layers | min/max eigenvalue ratio -2.833e-5 (2 layers) -> +1.449e-6 (4 layers) | 2-layer PML put the NF2FF box ~1 cell from the boundary; reflected field contaminated the closed-surface complex-Poynting operator and drove `Q_rad` indefinite. 4 layers restore a genuinely PSD spectrum |
| `Q_rad` PSD floor | none | -1e-9 relative (plus `max_eig > 0`) | +1.449e-6 min/max ratio (re-measured, four-element, 4-layer PML) | new gate; enforces PSD with only an eigvalsh roundoff band (~1e-16*max) instead of the earlier -1e-3 floor that blessed the under-absorbed 2-layer scene |

The converged-scene thresholds derive from the Phase 1 numbers recorded in the
acceptance document and were **not** independently re-measured here, because the
benchmark is a timing workload that is serialized separately. The contract-scene and
`Q_rad` numbers were re-measured directly on 2026-07-16 (`maxwell` env, one RTX A6000)
after raising the contract scene from 2 to 4 PML layers; the frozen 96^3 benchmark
scene already uses 8 absorbing cells per face and is unchanged.

Full-wave field comparisons use the same phase center and raw complex fields. Global
phase or amplitude fitting is prohibited for same-solver basis/direct comparisons.
Independent-reference alignment, when required by a different reference-plane
convention, must be recorded explicitly.

## Performance contract

The frozen qualification host is one NVIDIA RTX A6000 (48,541 MiB, driver 595.71.05,
PCI bus `17:00.0`), torch 2.13.0+cu130, and CUDA 13.0, re-anchored on 2026-07-18 from the
retired RTX 5080 by a passing frozen qualification (see the Phase 1 qualification
amendment below). The Phase 1 single-device
benchmark is a four-element linear fed array at 1 GHz, half-wavelength element
spacing, 50-ohm matched inactive terminations, exactly `96 x 96 x 96` cells including
eight absorbing cells per boundary face, 4,096 time steps, one frequency, and a
`181 x 361` angular grid. The 16 weights are
`a_n = exp(i*n*2*pi*m/16)/2`, `n=0..3`, `m=0..15`. Timings use CUDA events with
explicit synchronization, three warmups, five samples, and four alternating order
rounds.

- Four basis solves plus 16 combinations must take at most 40% of 16 direct solves.
- Combination time must be below 10% of one solve.
- Combining 64, 256, or 1,024 beams must execute zero additional FDTD steps.

## Approved scope adjustment

On 2026-07-16 the user explicitly removed task-level multi-GPU work from this
implementation. Phase 2 therefore retains codebook, scan, max-hold, metadata, and
basis-cache delivery, but omits the device-pool scheduler and 1/2/4-GPU scaling gate.
Phase 4 retains single-device weight and scene gradients plus the domain-decomposition
aggregation contract, but omits multi-GPU value/gradient parity. These omitted gates
are recorded as user-approved scope reductions, not as passing evidence.

The task-level scaling protocol that this section previously specified (four RTX A6000
GPUs, `E2 = T1 / (2*T2) >= 0.80`, `E4 = T1 / (4*T4) >= 0.70`, 16 round-robin basis
tasks) is removed by that adjustment, together with the `two_gpu_parallel_efficiency`,
`four_gpu_parallel_efficiency`, `scaling_hardware`, `task_s_rtol`, `task_s_atol`, and
`task_basis_count` budget fields that encoded it. `AcceptanceBudget` no longer defines
them and `test_acceptance_budget_carries_no_cancelled_task_level_multi_gpu_scope` pins
their absence, so the cancelled gate cannot silently return.

The plan of record, `next-functional-2026-07/06-array-active-s-mimo.md`, still carries
the pre-adjustment task-level multi-GPU scope in its section 8, section 9 Phase 2/Phase 4
exit gates, and section 10.5. This implementation record and that plan disagree; the
adjustment above is the newer decision, but the plan file itself has not been amended.

## Phase 0 contract

- Incident beam weights are Kurokawa power waves in `sqrt(W)` and use the existing
  `NetworkData` ordering `[frequency, output_port, input_port]`.
- Embedded patterns use complex `[F, N, T, P]` `E_theta/E_phi`; each column is
  normalized to measured `a_n = 1 sqrt(W)` with all other ports matched.
- Weight shapes are `[N]`, exact `[F, N]`, or `[B, F, N]`. Frequency interpolation and
  implicit device/dtype conversion are forbidden.
- An unexcited port returns `active_mask=False` and NaN active quantities. Complex
  reference impedances use the existing power-wave-to-voltage/current conversion.
- Full-sphere fields carry the phase center, its explicit/AABB provenance, frame,
  spherical polarization basis, observation radius, impedance, dtype, and device.
- Sweep compilation consumes the RF or modal run manifest for the exact ordered basis
  names, frequency values, matched-termination provenance, and a stable manifest
  fingerprint; it never infers modal channels from physical Scene port names.
- Array-basis compilation rejects independent sources, nonlinear materials, and
  time-modulated materials by reporting the offending object names.

Phase 0 evidence is maintained in `docs/assessments/array-active-s-mimo-phase-0-acceptance.md`.

## Phase 1 implementation

- `PortSweep` execution retains one compact result per physical input column (and per
  WavePort frequency) plus the measured diagonal incident wave. Compact columns share
  one prepared scene, omit full-volume solver fields and port/modal payloads, and keep
  only declared closed-surface monitor data.
- `Result.array_basis(...)` performs no solver rerun. It normalizes each embedded
  `E_theta/E_phi` column by measured `a_n(f)`, preserves manifest port/channel order,
  and produces a content fingerprint over the complete network, EEP fields, power
  operator, angular/frame contract, and normalization inputs.
- Absolute beam power uses a Hermitian closed-surface complex-Poynting operator
  `Q_rad`; combination evaluates `real(a^H Q_rad a)`. Raw complex far fields are never
  phase- or amplitude-fitted. Monitor DFTs sample E at `(n+1)dt` and H at
  `(n+1/2)dt`; the adjoint uses the exact transposed schedules.
- Monitor-derived equivalent currents carry outward normals and exact primal-cell
  area weights, including nonuniform and single-cell tangential axes. Standalone
  user-supplied planar currents retain trapezoidal quadrature by default.
- Derived solver phasors must remain on the `NetworkData` device. Exact dtype is kept,
  except the explicit precision policy permits complex64 solver phasors to be promoted
  to a complex128 `NetworkData`; downcasts and all implicit tensor device moves are
  rejected. Tensor angle/frame/center/radius configuration follows the same exact
  device contract.
- `ArrayBasisData.save/load` reuses the safe `NetworkData` serializer and
  `weights_only=True`; a saved `Result` intentionally omits in-memory sweep columns,
  so delayed reuse requires extracting and saving `ArrayBasisData` first.

Phase 1 evidence is maintained in
`docs/assessments/array-active-s-mimo-phase-1-acceptance.md`.

## Phase 1 qualification amendment, 2026-07-18 (host re-anchor: PASS)

The frozen `96^3` performance qualification (`benchmark/array_phase1.py`, 3 warmups,
5 samples, 4 alternating rounds) was rerun after the flux/observer convention was
corrected (commit `1cc4a71`: Yee-staggered E-plain/H-retard observer DFT and full-primal
NF2FF quadrature) and now **PASSES every gate**, so the performance contract is
re-anchored from the retired RTX 5080 to this A6000 host.

- Previous recorded host (`AcceptanceBudget.local_hardware`, old value): `NVIDIA GeForce
  RTX 5080 16303 MiB, driver 596.49, PCI 00000000:01:00.0`. That host is retired, ran
  torch 2.10 / CUDA 12.8, and cannot reproduce the timings.
- New recorded host (`AcceptanceBudget.local_hardware`, new value): `NVIDIA RTX A6000
  48541 MiB, driver 595.71.05, PCI 00000000:17:00.0, torch 2.13.0+cu130, CUDA 13.0 (one
  GPU, numactl node 0)`. Qualification host is 2x NVIDIA RTX A6000, run on one quiet GPU
  (`CUDA_VISIBLE_DEVICES=0`, `numactl --cpunodebind=0 --membind=0`, governor
  `performance`, no competing compute processes).
- Reason: the frozen-budget change rule permits re-anchoring the qualification host only
  by a passing frozen qualification. This run is that passing qualification.

Outcome: **PASS**. Both numerical and timing gates cleared.

Numerical: physical power closure `|P_accepted - P_rad| / P_incident = 6.971e-4 (0.0697%)`
against the 1% gate — restored to the recorded `3223e0c` value of `6.997e-4` (0.07%) after
the plain-plain-convention regression that measured `0.028646` (2.865%) on `4b24b60`.
Basis-vs-direct far-field weighted complex L2 `7.856e-7` (broadside) / `8.837e-7`
(endfire) against `1e-4`; phase RMS `4.605e-5 deg` / `6.524e-5 deg` against `1e-2 deg`;
port powers `~1e-7` / `~1e-8` against `5e-3`; `Q_rad` spectrum positive definite (min eig
`0.456`, max eig `0.955`, min/max `0.478`, `max_eig > 0`). Grid `96^3`, 8 PML cells/face,
4096 steps, `181x361` angular grid, 16 beams: all match the frozen contract.

Timing (CUDA-event medians, 3/5/4 protocol): four basis solves plus 16 combinations
`24.527 s`; 16 direct solves `119.361 s`; 16 combinations `2.561e-3 s`. Basis/direct ratio
`0.2055` against the `<= 0.40` gate; combine/one-solve ratio `3.434e-4` against the
`< 0.10` gate; combination executes zero additional FDTD steps (pure einsum). Per-sample
spread was tight (direct `118.8-120.2 s`, basis `24.3-24.7 s`), corroborating an
uncontended GPU throughout.

Interpretation: the closure regression that failed the 2026-07-17 attempt was localized
at the time to the NF2FF/observer flux machinery, not to superposition or to the host
(the basis-vs-direct linearity check passed then and passes now, and the external-reference
refresh showed a systematic flux/Poynting regression across many scenarios). Commit
`1cc4a71` corrected exactly that machinery, and the closure returned to its golden value,
confirming the diagnosis.

Evidence: `docs/assessments/array-active-s-mimo-phase-1-qualification.json`.

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
| Basis versus direct FDTD complex vector far field | solid-angle-weighted relative L2 `<= 0.03` |
| Basis versus direct FDTD phase | wrapped RMS `<= 3 deg` above 10% of peak field; no global phase alignment |
| Basis versus direct port powers | incident/reflected/accepted relative error `<= 1%` |
| Passive physical power closure | `abs(Paccepted - Prad - Ploss) / Pincident <= 1%` |
| Independent reference | realized-gain error `<= 0.25 dB`; ECC error `<= 0.02` |
| Active impedance where reference data exists | magnitude error `<= 5%`, phase error `<= 3 deg` |
| Weight and supported scene gradients | relative error `< 2%`, absolute floor `1e-8` |
| Inherited distributed diagnostic-field parity | max absolute error `<= 2e-6`, significant-field max relative error `<= 2e-5` |
| Task-level 1/2/4-GPU S parity | `rtol <= 2e-5`, `atol <= 1e-6`, exact public port order |
| Domain-decomposition monitor/EEP aggregation | `rtol <= 5e-5`, `atol <= 5e-6`, exact public port order |

Full-wave field comparisons use the same phase center and raw complex fields. Global
phase or amplitude fitting is prohibited for same-solver basis/direct comparisons.
Independent-reference alignment, when required by a different reference-plane
convention, must be recorded explicitly.

## Performance contract

The local qualification host has one NVIDIA GeForce RTX 5080 (16,303 MiB, driver
596.49, PCI bus `01:00.0`), torch 2.10, and CUDA 12.8. The Phase 1 single-device
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

Task-level scaling is qualified separately on one host containing four NVIDIA RTX A6000
48 GiB GPUs on PCIe Gen4 x16 with pairwise peer access enabled. `T1`, `T2`, and `T4`
use the same host, fixed clocks, 16 independent basis tasks, the exact Phase 1 grid and
4,096-step workload, and stable round-robin port indices; the two-GPU run uses devices
0-1 and the four-GPU run uses devices 0-3. Minimum parallel efficiency is
`E2 = T1 / (2*T2) >= 0.80` and `E4 = T1 / (4*T4) >= 0.70`. The local one-GPU host
cannot supply that evidence; simulated devices or mocked execution do not satisfy it.

## Approved scope adjustment

On 2026-07-16 the user explicitly removed task-level multi-GPU work from this
implementation. Phase 2 therefore retains codebook, scan, max-hold, metadata, and
basis-cache delivery, but omits the device-pool scheduler and 1/2/4-GPU scaling gate.
Phase 4 retains single-device weight and scene gradients plus the domain-decomposition
aggregation contract, but omits multi-GPU value/gradient parity. These omitted gates
are recorded as user-approved scope reductions, not as passing evidence.

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

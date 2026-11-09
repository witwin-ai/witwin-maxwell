# Fixed-k broadband reconciliation with master, 2026-11-04

The `bloch/fixed-k-broadband-tfsf-r3` branch at `803ab562` (base `f2ba002b`)
merges master `2e8308d2`. This reconciliation is CPU-only in the `maxwell`
environment with `CUDA_VISIBLE_DEVICES=`. GPU qualification remains a reviewer
task; the local, uncommitted `.r3-gpu-plan.md` lists the commands.

## Merge decisions

- The integration manifest retains all master declarations and movers. The
  branch declaration is inserted before the exact line-anchored table
  `[[merge_declarations]]` / `branch = "perf/core-cuda-source-dedup"`. It names
  all six branch test files and the census, integration and timing gates.
- The guard ledger retains master's active/retired reviews and the branch's
  narrower directional-source boundary: retire `af76d41c...:0`, admit broadband
  fixed-k slab plane waves and retain `84b8fe15...:0`. Both census notes survive.
  Refresh 16 moved source-line anchors without changing their identities or
  review evidence. The merged inventory has 553 active, 45 retired and 28
  contract-excluded sites (581 current exception sites).
- Preserve master's observer origin and ABI, real auxiliary-line physical
  entry/ghost/replay timing, CW face-current instants, electric wall support,
  source-replay boundary codes and old-checkpoint preflight checks. The replay
  implementation matches master exactly. The fixed-k checkpoint schema retains
  the branch's fields, CPML memory, clock and spectrum.
- Preserve the capturable TFSF whitelist as the two reference providers plus
  `plane_wave_fixed_k`. The fixed-k magnetic advance and complex slice additions
  remain device-driven; source evaluation stays in the eager electric hook.

## Timing findings and limited compatibility changes

Master's real incident line imposes entry E as `W((n+1)dt)`; CW faces evaluate
M at `n*dt` and J at `(n+1/2)dt`. Neither convention should be applied to the
fixed-k line by relabeling its soft current as a hard entry E. The fixed-k line
retains its existing `W(n*dt)` soft drive, discrete Maxwell update and internal
`n*dt` spectrum accumulator. Its entry plane contains the propagated line field,
rather than the imposed waveform. The full raw-exact change to the half-step
drive and analytic `g(omega)` is a separate follow-up.

The automatic merge had two compatibility gaps:

1. The provider-reference planner did not recognize `plane_wave_fixed_k`. Register
   its existing waveform sampling and normalization label as alpha=beta=0.
   Keep the source-reference offset at zero rather than inheriting the real
   auxiliary line's alpha=1. Shared normalization still follows the first
   declared source, including a leading point dipole's half-step reference.
2. Physical Yee observers label post-update E with `(n+1)dt`, while the line's
   retained accumulator labels that same E with `n*dt`. Rotate only the exported
   measured incident spectrum by `exp(+i*omega*dt)`. This aligns line-normalized
   Floquet coefficients with matched-run physical observer spectra, without
   altering the drive, field dynamics or checkpointed accumulator. A direct
   sample-by-sample Fourier check covers this conversion; a separate control
   pins integer-step soft-drive evaluation.

The unchanged-path GPU probe now compares against master `2e8308d2` instead of
`f2ba002b`. No historical BITID baseline, acceptance bound or tolerance is
changed. CPU preparation and line tests do not qualify native graph execution,
Airy accuracy or GPU bit identity.

## Final CPU gates

One combined pytest run returned **rc 0**, **295 passed**, **146 skipped** in
136.50 seconds. All Python commands used `maxwell`; CUDA was hidden, with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for the final run.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| `tests/api/public/test_guard_census.py` | 7 | 0 |
| `tests/validation/benchmark/test_integration_batch.py` | 82 | 0 |
| `tests/sources/tfsf/` | 168 | 134 |
| Other Bloch/Floquet files below | 2 | 10 |
| `tests/fdtd/test_source_time_labels.py` | 36 | 0 |
| `tests/fdtd/test_raw_phase_bit_identity.py` | 0 | 2 |
| Total | 295 | 146 |

`tests/boundaries/bloch/` does not exist. The additional files were
`tests/boundaries/boundary_physics/test_bloch_generalized_layouts.py`,
`test_bloch_pec_supercell.py` in the same directory,
`tests/postprocess/scattering/test_floquet.py`,
`tests/validation/physics/test_bloch_broadband_vs_cw.py` and
`tests/sources/point/test_fixed_k_broadband_point_source.py`. Both raw-phase BITID
cases require native CUDA and were collected and skipped. The two branch-added
test files are included in the TFSF directory gate.

The focused fixed-k line/source tests also returned rc 0 (**47 passed**), and
the census rerun after the source-line refresh returned rc 0 (**7 passed**).
Final run log and JUnit output are `/tmp/r3-cpu-final.log` and
`/tmp/r3-cpu-final.xml`; the census rerun log is `/tmp/r3-census-final.log`.

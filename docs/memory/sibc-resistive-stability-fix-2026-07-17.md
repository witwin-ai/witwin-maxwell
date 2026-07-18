# SIBC surface-impedance stability fix (resistive Leontovich)

Durable record for the surface-impedance boundary. Plan 09 (surface impedance /
metal roughness) builds directly on this runtime, so the stability characterization
here is a prerequisite.

## Symptom

Two forward-validation gates failed (present at the origin baseline `9131c89` and
every lineage point since):

- `tests/validation/physics/test_lossy_metal_sibc.py::test_sibc_reflection_matches_analytic_at_three_frequencies`
- `tests/validation/physics/test_lossy_metal_sibc.py::test_sibc_absorbs_more_than_pec`

Both failed on the finiteness guard `assert np.isfinite(ez).all()` ("SIBC field
diverged (non-finite)"). The third test in the file (single 1 GHz solve) passed.

## Root cause (evidence-backed)

The `LossyMetalMedium` SIBC overwrote the two tangential surface-E faces each step
with the full narrowband Leontovich impedance `Zs(omega0) = (1 + j) R`, discretized as

```
E_t = sign * ( R * H_now + (Ls/dt) * (H_now - H_prev) ),   Ls = R / omega0
```

The `(Ls/dt) * (H_now - H_prev)` inductive term is an explicit forward-difference
derivative feedback in an E-from-H overwrite. It is **non-passive**: it injects
energy at the surface every step, with a per-step growth rate that scales with
`R, Ls ~ sqrt(f)`.

Evidence:

- A 1D leapfrog model (`vacuum + resistive/RL surface`, PEC-terminated resonator)
  shows the same class of behavior: the RL form grows with an f-scaled rate while
  the resistive form stays bounded. CAVEAT (independent audit, 2026-07-17): the 1D
  rig's SIGN conventions did not replicate under re-derivation — an auditor's 1D
  model with the actual 3D stepper ordering found the physically-signed inductive
  term stable at normal incidence and the FLIPPED sign unstable, i.e. the 1D
  narrative here is likely sign-mirrored and setup-dependent. The 3D instability
  itself is reproduced empirically and unambiguously (re-adding the inductive term
  to the fixed 3D code brings the 2 GHz divergence back; the destabilizing channel
  is 3D-specific — oblique/transverse-mode or CPML coupling — not the pure
  normal-incidence 1D story). Plan 09 work must treat the 3D empirical
  characterization as the ground truth and NOT build on the 1D sign claim.
- Why 1 GHz survived but 2/3 GHz did not: the growth rate scales with `sqrt(f)`. At
  1 GHz the open scene's ambient radiation/PML loss exceeds the tiny surface gain
  (field saturates ~3e8, bounded); at >=2 GHz the surface gain exceeds ambient loss
  -> unbounded -> NaN. A PEC slab in the identical scene is stable at all f (bounded
  ~1e9 at 2 GHz), confirming the metal SIBC is the culprit, not the source drive.
- The reactance is negligible for the reflection magnitude the SIBC targets: for a
  good conductor `|Gamma|` from the full `(1 + j) R` versus the resistive `R` differs
  by `< 1.3e-3` across the validity domain (1 GHz 2.5e-5, 3 GHz 1.3e-4). The test's
  standing-wave readout depends only on `|Gamma|`, not phase.

## Fix

Drop the reactive term; implement the passive resistive Leontovich surface

```
E_t = sign * R * (n_hat x H),   R = sqrt(omega0*mu0/(2*sigma))
```

Removed the surface-inductance plumbing end to end (kernel + binding + backend +
runtime + resume checkpoint): the surface update is now stateless (no `h_prev`).

Touch points:

- `witwin/maxwell/fdtd/cuda/kernels/boundary.cu` (`sibc_surface_kernel`, `apply_sibc_surface_cuda`)
- `witwin/maxwell/fdtd/cuda/extension.cpp` (decl + op schema)
- `witwin/maxwell/fdtd/cuda/backend.py` (`_apply_sibc_surface`)
- `witwin/maxwell/fdtd/runtime/materials.py` (`_configure_sibc`)
- `witwin/maxwell/fdtd/runtime/stepping.py` (`apply_sibc_surface`, capture snapshot)
- `witwin/maxwell/fdtd/resume.py` (checkpoint capture/restore/hash)
- `witwin/maxwell/fdtd/adjoint/bridge.py` (stale comment)

The public `surface_impedance(omega)` helper (`media.py`) and the Tidy3D export path
are unchanged: they still expose the full complex `Zs = (1 + j) R`. The analytic test
gates (which compute the full complex `Zs`) are unmodified and pass.

## Stability domain (no prepare-time guard added)

Per-process real-solver sweep at 2 GHz (open scene, PML both x faces), resistive form:

| sigma | loss tangent | R/eta0 | finite | gamma vs analytic |
|------:|-------------:|-------:|:------:|:-----------------:|
| 50    | 449          | 0.033  | yes    | rel err 0.000 |
| 10    | 90           | 0.075  | yes    | 0.000 |
| 5     | 45           | 0.105  | yes    | 0.002 |
| 2     | 18           | 0.167  | yes    | 0.008 |
| 1     | 9            | 0.236  | yes    | 0.026 |
| 0.5   | 4.5          | 0.334  | yes    | 0.077 |
| 0.3   | 2.7          | 0.431  | yes    | 0.165 |

The resistive form is stable across the entire good-conductor validity domain and well
into the poor-conductor regime; accuracy degrades only where the Leontovich
approximation itself breaks down (low loss tangent), not from instability. No numerical
guard is warranted for open-region (radiation/PML-terminated) use, which is the SIBC's
purpose. (A lossless PEC-PEC cavity round-trips the surface repeatedly and shows a
marginal `|Gamma_num| > 1` above R/eta0 ~ 0.02, but that pathological closed geometry is
outside SIBC use and is not what the tests or real half-space terminations exercise.)
The test frequencies sit at loss tangent 300-900 (R/eta0 0.024-0.041), a large margin.

## Falsification

- Scale the fixed `surface_r` by 3: reflection gate goes red (rel err 0.428 at 1 GHz;
  divergence at 2 GHz).
- Re-add the inductive derivative term (pure-torch overwrite): the 2 GHz divergence
  returns (non-finite) -> red.

## Verification

- All 3 SIBC tests pass at original tolerances; target test reproducibly green (5/5).
- No regressions: `tests/validation/physics`, `tests/materials/sheet`,
  `tests/boundaries`, `tests/fdtd`, `tests/api/public`, `tests/api/adapters/tidy3d`,
  `tests/rf/network`, `tests/rf/antenna`, `tests/postprocess/rcs`,
  `tests/rf/circuits/test_fdtd_circuit_resume.py`,
  `tests/gradients/test_fdtd_adjoint_bridge.py`, `tests/fdtd/test_cuda_graph_capture_coverage.py`
  (updated `test_lossy_metal.py::test_lossy_metal_simulation_prepare_configures_sibc`,
  which asserted the removed `surface_l` descriptor field).

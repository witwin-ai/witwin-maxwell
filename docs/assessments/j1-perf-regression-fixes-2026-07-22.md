# J1 — perf/cleanup line regression fixes (2026-07-22)

Branch `fable/perf-regress`, based on `bf9c3aa`.

Master was last verified green at `16985a1` (3076 passed / 16 failed, all in the
user-deferred FDFD family). Three commits then landed on the perf/cleanup line:

- `9650439` refactor: phase A cleanup — dead code removal, CUDA-graph stepping
  made the public default, GPU DFT weight tables un-gated, shared helpers, new
  `constants.py`
- `8a1e5bd` perf(fdtd-adjoint): reuse reverse-step buffers, memoize constant
  curls, guard the RF grad path
- `64df75f` perf(fdtd-cuda): uniform-coefficient scalar fast path, compressed
  CPML interior kernels

followed by `2bd33bb` / `bf9c3aa` (behavior-preserving decomposition and kernel
template fold).

Three regressions were found beyond the deferred FDFD family. **All three
bisect to `9650439`.** `8a1e5bd` and `64df75f` are clean of these defects.

| # | Test | First-bad | Mechanism | Fix | Commit |
|---|------|-----------|-----------|-----|--------|
| 1 | `tests/sources/incident/test_soft_planewave_absolute_power.py::test_plane_wave_power_scale_is_derived_unit_power_factor` | `9650439` | `ETA_0` redefined as `MU_0 * C_0`; the truncated CODATA literals shift the impedance 3.0e-12 relative | restore the CODATA `ETA_0` literal | `365ac45` |
| 2 | `tests/fdtd/multi_gpu/test_ensemble_network_sweep.py::test_two_gpu_ensemble_sweep_matches_serial_with_provenance` | `9650439` | graph capture not bound to the solver device → silently empty graph → run stops integrating after warmup | bind the device + explicit capture stream; empty capture now raises | `494a836` |
| 3 | `tests/fdtd/multi_gpu/test_ensemble_run_many.py::test_run_many_matches_serial_and_respects_lease` (+ `test_run_many_end_to_end_orders_results`, `test_run_many_isolates_a_failing_task`) | `9650439` | process-global capture aborts concurrent worker threads with `cudaErrorStreamCaptureUnsupported` | executor suspends capture for any plan that can run >1 task at a time | `55c90e4` |

The "third regression" the truncated battery log hid is #2 above. A full
`tests/fdtd` run at `bf9c3aa` reported `4 failed, 485 passed, 52 skipped`: the
network sweep plus the three `test_ensemble_run_many.py` cases. The
`run_many` failures are a race (see #3), which is why a single battery run
reported only three regressions rather than four.

---

## Regression 1 — plane-wave absolute-power calibration

**First-bad commit:** `9650439`. Verified by bisect worktrees: at `f9c6bef` the
test passes (`1 passed`), at `9650439` it fails.

**Mechanism.** `9650439` introduced `witwin/maxwell/constants.py` and migrated
~22 files off their local `_ETA0 = 376.730313668` literals. While doing so it
redefined the shared value as `ETA_0 = MU_0 * C_0`, described in the module
docstring as "derived for internal consistency". But `MU_0` and `EPSILON_0` are
themselves CODATA decimal literals quoted to twelve significant digits, so
neither derivation reproduces the recommended impedance:

| expression | value | relative offset from CODATA |
|---|---|---|
| CODATA 2018 recommended `Z_0` | `376.730313668` | — |
| `MU_0 * C_0` | `376.7303136668535` | `-3.043e-12` |
| `1 / (EPSILON_0 * C_0)` | `376.7303136668698` | `-3.000e-12` |

That 3e-12 shift propagated into every absolute normalization consuming `ETA_0`:
the soft `PlaneWave` and `GaussianBeam` unit-power scales
(`fdtd/excitation/injection.py`), the TEM and vector-mode impedances
(`fdtd/excitation/modes/`), the far-field constants (`postprocess/antenna.py`)
and `array.py`'s default wave impedance. The convention that broke is purely the
choice of physical constant — no discretization convention changed.

**Quantified divergence.** The pinned power scale
`1/sqrt(A*cos(theta)/(2*eta0))` for a 0.8 x 0.5 m aperture at normal incidence:

- expected (CODATA `eta0`): `43.40105492197165`
- observed at `bf9c3aa`: `43.40105492190561`
- test tolerance: `rel=1e-12` (`± 4.3e-11` absolute)

**Fix.** `ETA_0` is the CODATA 2018 recommended literal again, matching the value
every call site used before the migration. The module docstring records why both
"internally consistent" derivations are rejected.

**Falsification.** With `ETA_0` reverted to `MU_0 * C_0` in a scratch edit:

```
>       assert ETA_0 == 376.730313668
E       assert 376.7303136668535 == 376.730313668
>       assert MU_0 * C_0 != ETA_0
E       assert (1.25663706212e-06 * 299792458.0) != 376.7303136668535
>       assert scale == pytest.approx(1.0 / math.sqrt(unit_power), rel=1e-12)
E       Obtained: 43.40105492190561
E       Expected: 43.40105492197165 ± 4.3e-11
3 failed in 2.16s
```

Restored: `3 passed in 2.12s`.

**Guard added.** `tests/core/constants/test_physical_constants.py` pins all four
vacuum constants by exact value and asserts explicitly that `ETA_0` is neither
truncated product, so the derivation cannot silently return.

---

## Regression 2 — CUDA-graph capture was not bound to the solver device

**First-bad commit:** `9650439`. At `f9c6bef` the network sweep passes
(`2 passed`); at `9650439` it fails with the empty-graph warning. The capture
helper was already device-blind before `9650439`; making `cuda_graph=True` the
public default is what made every non-default-GPU run reach it.

**Mechanism.** `torch.cuda.graph` opens its capture stream on whatever device is
current, and caches that stream on its class after the first capture in the
process. `CudaGraphRunner.capture` ran with no device context, so a solver whose
tensors live on `cuda:1` while the calling thread still has `cuda:0` current
recorded an **empty** graph — PyTorch says so itself
(`UserWarning: The CUDA Graph is empty. This usually means that the graph was
attempted to be captured on wrong device or stream.`). The warmup and capture
launches still executed eagerly on the tensors' own stream, so nothing raised:
the run silently stopped integrating after the warmup steps and returned
plausible-looking but wrong fields. The ensemble network sweep leases whichever
GPU is free per column and compares against the serial result, so it saw this
deterministically.

**Quantified divergence.** 40-step vacuum dipole, `mw.Simulation.fdtd(...)`,
graph vs eager, main thread with `cuda:0` current:

| device | eager peak &#124;Ez&#124; | graph peak &#124;Ez&#124; | bit-identical |
|---|---|---|---|
| `cuda:0` | `2.126512e+04` | `2.126512e+04` | yes |
| `cuda:1` (before fix) | `2.126512e+04` | `7.333388e+04` | no |
| `cuda:1` (after fix) | `2.126512e+04` | `2.126512e+04` | yes |

**Fix.** `witwin/maxwell/fdtd/cuda/runtime/graph.py`: `CudaGraphRunner` now takes
a mandatory `device`, runs warmup / capture / synchronize inside
`torch.cuda.device(device)`, and passes an explicit
`torch.cuda.Stream(device=device)` so it cannot inherit the class-cached stream
from an earlier capture on another device. All six construction sites
(`runtime/stepping.py` x3, `networks.py`, `ports.py`, `circuits.py`) pass the
solver's — or the circuit system's — device.
`witwin/maxwell/execution/executor.py` additionally binds each worker thread to
its leased device, since a fresh thread inherits the process default.

**Validity condition now enforced.** A capture that records no work raises
`RuntimeError` instead of returning a no-op replay, so the callers' existing
`except Exception` fallback degrades to eager stepping rather than freezing the
fields. This converts the whole class of silent-corruption failures into a loud,
correct fallback.

**Falsification.** Unbinding the device (dropping the `torch.cuda.device` guard,
the explicit stream and the empty-graph check) in a scratch edit:

```
>           assert torch.equal(graphed.fields[name], reference), (
E           AssertionError: EX diverges between graph and eager stepping on cuda:1
1 failed, 4 passed
```

Restored: `5 passed`.

**Guard added.** `tests/fdtd/multi_gpu/test_cuda_graph_device_binding.py::
test_graph_stepping_on_non_default_device_matches_eager` runs the solve on
`cuda:1` with `cuda:0` deliberately current and requires bit-identical fields
against eager stepping — the configuration a `cuda:0`-only test can never reach.
`test_runner_rejects_a_non_cuda_device` pins the constructor guard.

---

## Regression 3 — process-global capture aborted concurrent ensemble tasks

**First-bad commit:** `9650439`. At `f9c6bef`, `test_ensemble_run_many.py` is
`12 passed` on 3 of 3 repetitions; at `9650439` it is `12 passed`, then
`1 failed, 11 passed`, then `2 failed, 10 passed`.

**Mechanism.** CUDA-graph capture is process-global, not per-device. Only one
capture may be underway in a process at a time, and PyTorch captures in the
default `cudaStreamCaptureModeGlobal`, which makes an ordinary synchronizing
call — `Tensor.item()`, a device-to-host copy, an allocation — in *any* other
thread fail with `cudaErrorStreamCaptureUnsupported` for as long as the capture
window is open. So while the ensemble task leased on `cuda:0` captured its step
graph, an unrelated task on `cuda:1` aborted mid-prepare with
`CUDA error: operation not permitted when stream is capturing`. Observed raising
from three different places across repeated runs — `fdtd/runtime/materials.py`
`build_materials` and `_store_coefficient_uniformity`, and
`fdtd/excitation/injection.py` `_ideal_axis_weights` — i.e. wherever the victim
task's solver setup happened to touch the host next. Each aborted task lands in
the `ResultSequence` as a `DistributedFailure` with status `failed`, which is
what the test reports as "not all completed".

This is a genuine race, not a deterministic failure: with the fix reverted, 3 of
6 repeated runs of `test_ensemble_run_many.py` were red.

**Fix.** `witwin/maxwell/fdtd/cuda/runtime/graph.py` gains a reference-counted
process-wide `suspend_capture()` / `capture_suspended()` guard.
`execute_plan` wraps the whole plan in it whenever `workers > 1`. Plans limited
to one task at a time keep the graph default. Placing the guard in the executor
covers every consumer at once — `run_many`, the ensemble network sweep
(`Simulation._run_network_sweep_ensemble`) and the array-gradient plan
(`array_gradient.py`) — rather than requiring one opt-out per caller.
`CudaGraphRunner.capture` raises while suspended rather than silently returning
the eager closure, so the callers' `except Exception` fallback takes the eager
path *and* leaves their `*_graph_active` flags cleared.

**Falsification.** Replacing the guard with `nullcontext()` in a scratch edit:

```
FAILED .../test_cuda_graph_device_binding.py::test_concurrent_plans_suspend_capture_and_serial_plans_do_not
1 failed, 16 passed

# and, repeating the racy ensemble file:
run 1: 1 failed, 11 passed     run 4: 1 failed, 11 passed
run 2: 1 failed, 11 passed     run 5: 12 passed
run 3: 12 passed               run 6: 12 passed
```

Restored, the same 6 repetitions of both files: `17 passed` each time.

**Guard added.** `test_capture_stands_down_while_suspended`,
`test_suspend_capture_is_reference_counted`, and — the deterministic gate for
this defect, since the ensemble failure itself is racy —
`test_concurrent_plans_suspend_capture_and_serial_plans_do_not`, which drives
`execute_plan` with a probe task and asserts the guard is active for a
2-concurrency plan and inactive for a serial one.

---

## Verification

Test counts on the fixed tree (`55c90e4`):

- `tests/fdtd/multi_gpu` — **285 passed** (4 failed at `bf9c3aa`).
- `tests/sources tests/fdtd/multi_gpu tests/gradients tests/materials
  tests/monitors tests/core/constants tests/api/public/test_guard_census.py` —
  **3 failed, 1153 passed, 1 skipped, 2 xfailed, 1 xpassed**. The three failures
  are the deferred FDFD family
  (`tests/gradients/test_fdfd_adjoint.py::test_fdfd_gradient_matches_central_difference_for_density`,
  `...for_geometry_position`, `...test_fdfd_bridge_reports_solver_stats_and_converges`),
  all reporting `GPU solve failed: No module named 'nvmath'`. Nothing else failed.
- `tests/rf` (the circuit / embedded-network / port graph-capture call sites this
  change touched, including the circuit CUDA-graph replay gates) —
  **778 passed, 1 xfailed**.

Baseline for comparison: a full `tests/fdtd` run at `bf9c3aa` reported
`4 failed, 485 passed, 52 skipped` — the network sweep plus the three
`test_ensemble_run_many.py` cases, all now green.

FDFD was not touched, no FDFD test was run for triage, and `nvmath` was not
installed. No tolerance was relaxed, nothing was marked xfail or skipped, and no
assertion was weakened.

## For the supervisor

1. **Ensemble throughput trade.** Concurrent ensemble tasks now step eagerly.
   `9650439` measured graph stepping at +29% throughput at 96^3 and neutral at
   288^3, so a small-grid ensemble gives that up. Keeping graphs there would
   require concurrent capture from several threads, which PyTorch explicitly
   does not support ("graphs already have the general, explicitly documented
   restriction that only one capture may be underway at a time in the process",
   `torch/cuda/graphs.py`). A capture mutex plus
   `capture_error_mode="thread_local"` would in principle allow it; that mode
   only scopes the *error check*, and the allocator interaction across
   concurrent captures is not a supported configuration, so it was not taken.
   Serial plans (`max_concurrency == 1`) keep the graph default.
2. **`ETA_0` provenance.** The repo's convention is CODATA recommended literals
   per constant, not one constant derived from the others. This is now asserted
   rather than assumed. If the intent was ever full internal consistency, that
   is a deliberate physics-convention decision and it would have to move the
   pinned test values with it.
3. **Empty-capture hard error.** Any future capture that legitimately records no
   CUDA work will now take the eager fallback instead of installing a no-op
   replay. That is the safe direction, but it does mean a genuinely
   work-free closure silently loses its (useless) graph.

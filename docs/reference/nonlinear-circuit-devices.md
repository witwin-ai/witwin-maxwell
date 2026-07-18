# Nonlinear Circuit Devices

Maintained technical reference for plan 05 (nonlinear circuit devices). This
document is the durable home of the **frozen AcceptanceBudget** for the whole
plan. Source plan: `docs/plans/next-functional-2026-07/05-nonlinear-circuit-devices.md`.

Nonlinear circuit devices are *device-terminal* nonlinearities (diode, behavioral
I-V, voltage-dependent charge) that enter a `Circuit` through `Circuit.add` and
run through the standard `Scene -> Simulation -> Result` flow. They are distinct
from the nonlinear electromagnetic *material* constitutive path.

## Slice status

| Slice | Scope | State |
| --- | --- | --- |
| N0 | Device + Newton contract, standalone (pure torch): `circuit_devices.py`, `compiler/nonlinear_devices.py`, `Circuit.add` extension, graph type-set registration, transistor contract guards | landed |
| N1 | Standalone transient + DC continuation | landed |
| N2 | EM same-step coupling (forward), `DeviceData` | pending |
| N3 | Production convergence + resume + graph path | pending |
| N4 | Implicit adjoint + multi-GPU forward parity | pending |
| N5 | BJT / Level-1 MOSFET (independent go/no-go) | reserved, fails closed |

### N0 delivered

* Public devices `Diode`, `PiecewiseLinearIV`, `PolynomialIV`,
  `VoltageDependentCapacitor` and the `NonlinearSolveConfig` solve configuration
  (`witwin/maxwell/circuit_devices.py`), all satisfying the shared
  `_CircuitDeviceContract` (`.terminals` / `.parameters` / `.initial_condition` /
  `.kind`).
* `compile_nonlinear_devices` -> fixed-shape, per-model-signature
  `CompiledNonlinearDevice` batches with analytic conduction `i(v)` / `di/dv` and
  stored charge `q(v)` / `dq/dv` (`witwin/maxwell/compiler/nonlinear_devices.py`).
* `NonlinearMNASystem` dense node-voltage residual/Jacobian assembler and the
  `newton_solve` Newton-Raphson core with the **dual convergence gate**, stable
  `expm1` + `pnjlim` junction limiting, and backtracking line search.
* `Circuit.add` admits the nonlinear device types; the graph compiler registers
  the DC-conducting nonlinear devices (`Diode`, `PiecewiseLinearIV`,
  `PolynomialIV`) as DC-connecting types (`VoltageDependentCapacitor` stays
  charge-only / DC-open like `Capacitor`).
* Transistor public surfaces `BJT` / `MOSFET` fail closed with a contract guard
  until the independent Phase 5 gate; registered in the guard census.

### N0 deferred

* The **"0 per-iteration host syncs" profiler gate is CUDA-deferred**: the
  standalone eager Newton path uses per-iteration host reductions for its
  convergence check; the sync-free device-resident reduction is the fixed-
  iteration unrolled graph path delivered in N3, and the profiler assertion is
  measured in the exclusive GPU timing window, not this round.
* FDTD coupling, transient companions, DC source/gmin stepping corpus,
  `DeviceData`, adjoints, and benchmarks are later slices (N1-N4).

### N1 delivered

* Stored-charge transient companion on `NonlinearMNASystem`: the trapezoidal
  (`2/dt`) / backward-Euler (`1/dt`) charge companion `i_cap(v)` and its
  differential-capacitance Jacobian `dcap/dv` fold into the residual and linear
  step (`_charge_companion`, `advance_charge_state`), matching the linear reactive
  companion convention (`compiler/mna.py`, `lumped.py`). The charge history
  advances only on an accepted step, so a rejected/limited Newton iterate never
  corrupts the integrator memory. The diode `junction_capacitance` and
  `VoltageDependentCapacitor` `q(v)` are consumed here; the DC operating point
  treats the capacitance as an open circuit, so the N0 conduction-only
  junction-capacitance guard is removed (capability budget `140 -> 139`).
* `run_nonlinear_transient`: Newton-in-the-loop transient driver on a fixed or
  non-uniform time grid, with a backward-Euler first step (TR startup) that avoids
  the trapezoidal `i_cap^0 = 0` first-step artifact for prescribed initial
  conditions, per-step KCL diagnostics, and an optional `initial_state` for
  initial-condition runs. Under `failure='record_and_stop'` a non-convergent
  step (or DC start) truncates the trajectory: the unconverged iterate is neither
  committed as a sample nor advanced into the charge history, `converged=False`,
  and `stopped_step` records the failing step index (`0` for the DC start). The
  default `failure='raise'` path never returns an unconverged iterate.
* `solve_dc_operating_point`: DC operating point with a geometric gmin
  continuation ladder ending exactly at `gmin=0` (no residual artificial
  conductance) and a warm-started source path.
* PWL-kink breakpoints: a step in which a `PiecewiseLinearIV` device crosses a
  knot is re-solved with a local backward-Euler companion (the standalone analogue
  of `LinearMNASystem._local_backward_euler_steps`) so the trapezoidal reactive
  companion does not ring on the conduction-slope kink. The breakpoint step's KCL
  residual is measured and its charge history advanced under the same backward-Euler
  companion it was solved with; the requested companion is restored only after the
  step is committed.
* `multistart_dc` + `MultistartReport`: a seed sweep that deduplicates the
  converged DC operating points, reporting uniqueness vs multistability on an
  N-shaped negative-differential-resistance device.

### N1 gates (all green, CPU fp64)

* Diode DC I-V vs analytic Shockley over a `[-2 V, +2 V]` bias sweep: worst
  relative error `< 1e-5`.
* Half-wave and center-tapped full-wave rectifier steady-ripple output vs a frozen
  oversampled backward-Euler reference (`tests/rf/nonlinear/goldens/`, generated by
  `_generate_goldens.py` at 25x finer step): normalized RMS `< 1%`.
* Every converged transient step satisfies the KCL residual tolerance.
* gmin continuation converges a cold hard-forward start and leaves no residual
  gmin (true gmin-free residual within tolerance).
* Linear-`Q(V)` capacitor discharge reproduces the analytic RC decay
  (`< 1e-4` normalized RMS), pinning the charge companion to the reactive
  convention.
* Multistart reports `>= 2` distinct operating points on the N-shaped device and
  exactly one in the monostable bias; PWL knot crossing triggers a backward-Euler
  breakpoint step whose per-step KCL residual clears the residual tolerance (the
  breakpoint step is measured under its own BE companion).
* `record_and_stop` truncates the transient at the first non-convergent step (or
  at the DC start): `converged=False`, `stopped_step` set, and no unconverged
  iterate is committed to the trajectory or the charge history.

## Frozen AcceptanceBudget

Frozen at Phase 0. Every number below is fixed; altering any of them requires the
change rule at the end of this section. Every gate is falsification-checked (a
deliberately broken variant must fail the gate).

**Newton core (Phase 0)**
- `residual_atol=1e-10`, `residual_rtol=1e-7` (plan §3 API defaults; fp64 headroom on scaled KCL residual).
- `update_atol=1e-12`, `update_rtol=1e-9` — independent second gate so residual-only false convergence is impossible (§5.2).
- Analytic-root accuracy `‖x-x*‖/‖x*‖ <= 1e-10` (fp64): machine-limited (~2e-16) over ≤20 iters; 1e-10 is a non-gameable floor.
- Per-iteration host syncs `= 0` (profiler-asserted): reduction + active mask device-resident, matches existing `graph_eligible` no-sync path.
- `max_iterations=20` default; exceeding on a non-convergent fixture raises deterministically (no NaN, no silent return).

**Static I-V (Phase 1)**
- Diode I-V rel err `< 1e-5` vs analytic Shockley (§8.1).
- Half/full-wave rectifier normalized RMS `< 1%` vs frozen golden (§8.1).
- Converged-step KCL residual `< residual` tolerance (§8.1).
- Device i(v),q(v) first derivatives vs complex-step/finite-diff `< 1e-6` (NEW): the analytic Jacobian is the Newton foundation; local derivative error must be far below the 2% end-to-end budget or Newton+adjoint degrade silently.

**Charge / varactor / EM coupling (Phase 2)**
- Nonlinear-C charge conservation `< 1e-4` (§8.2).
- Varactor harmonic freq/amp `< 2%` vs oversampled (§8.2).
- RF rectenna field+circuit energy imbalance `< 3%` (§8.2).
- U1 linear-limit exactness pin `<= 2e-6` (NEW, adopts program cross-model pin): a deep-linear-biased diode ≡ native `Resistor` port coupling to 2e-6, and a linear-region Q(V)-cap ≡ native `Capacitor` to 2e-6. Falsification anchor proving the nonlinear step reduces exactly to the unified half-step interface — blocks one-step-delay/convention drift hiding inside Newton.

**Production convergence (Phase 3)**
- Convergence corpus `>= 99%` accepted, remainder explicit failures (§8.3).
- Checkpoint resume `rtol <= 1e-6` (§8.3), matching the U1 resume regression.
- Failure report localizes to time/step/node/device + residual + iteration-trajectory summary; a non-localizing failure fails the gate.

**Differentiable + multi-GPU (Phase 4)**
- Grad Is/ideality/PWL-slope/Q-coeff vs multi-step finite diff `< 2%` away from kink/bifurcation (§8.4).
- Single↔multi-GPU transient `rtol <= 2e-5` (§8.4, program forward-parity convention).
- 32-device step overhead `< 2x` linear MNA with profiler breakdown (§8.4). This timing gate's measurement is deferred to the exclusive timing window (no timing claims this round); number frozen now.

**Transistor (Phase 5, independent)** — DC/transient/charge each `< 2%`; per-param range/unit/derivative tests (§8.5).

**Cross-cutting** — every unsupported-surface `NotImplementedError` (transistors pre-Phase-5, multi-circuit, trainable-params+spatial-multiGPU) is a CONTRACT guard on `docs/reference/fdtd-capability-guard-census.md`, machine-checked by `tests/api/public/test_guard_census.py` (census-reconciliation convention).

### Change rule

A frozen number is changed only by an explicit, recorded decision that (1)
updates the number here, (2) updates the falsification-checked test that pins it,
and (3) records the justification (a physical/numerical reason, not convenience)
in the same change. Loosening a gate additionally requires the coordinator's
approval note in the commit. New gates may be added at a later phase without
touching earlier-phase numbers.

## Fail-closed boundaries (frozen)

1. Newton non-convergence within `max_iterations` -> raise / record-and-stop; never return an unconverged iterate; no CPU fallback / step-skip / dt change.
2. gmin/source continuation must end at requested value; residual gmin in final state -> error.
3. Non-finite device output inside an iterate -> error with device/node localization; stable `expm1` / limiting, no clamp-to-wrong.
4. Singular / below-pivot Jacobian -> error; near-singular / multistable / non-converged backward marks gradient invalid and raises — never finite-but-wrong.
5. PWL exact knot hit -> non-differentiable diagnostic under declared subgradient policy.
6. Q(V)-only: `C(V)` without a consistent single-valued `Q(V)` rejected at construction.
7. Arbitrary Python per-step callable rejected; compiled tensor expression / declared protocol only.
8. Multi-circuit with nonlinear devices: keep single-circuit guard until lifted.
9. Trainable nonlinear params + spatial multi-GPU rejected until distributed adjoint lands.
10. `BJT` / `MOSFET` raise a contract guard until Phase 5 passes; a parser recognising a card is not support.
11. Bound port cannot also be directly excited / in `PortSweep`.
12. Nonlinear device in a linear executable runtime (linear/coupled MNA, FDTD Norton companion) -> capability guard; the constant-conductance stamp cannot carry a nonlinear terminal law, so the device is rejected rather than silently dropped. Lifted when the device-runtime Newton integration lands.
13. Diode nonzero `series_resistance` -> compile fails closed; the ohmic series branch is not assembled into the ideal-Shockley `i(v)`. Lifted when the series branch lands.
14. Diode nonzero `junction_capacitance` in the conduction-only DC Newton solve -> fails closed; the solve never differentiates `q(v)`. Lifted when the charge-aware transient solve lands. (`charge()` itself remains exercised for the device-math surface.)

### N0 implementation deviations (recorded)

* **Boundary 4 singular-Jacobian gate.** The N0 Newton core (`_solve_linear`)
  fails closed on an exactly-singular / below-pivot Jacobian by wrapping
  `torch.linalg.solve` in `try/except RuntimeError` and re-raising as
  `NonlinearDeviceError`, and records an (ungated) `torch.linalg.cond` estimate;
  it does not reuse the linear MNA `_factor` SVD singular-value gate
  (`compiler/mna.py`). This is a deliberate Phase-0 deviation from the brief's
  boundary-4 parenthetical: exactly-singular raises, and near-singular /
  non-converged iterates are caught by the dual convergence gate plus the
  iteration cap (boundary 1), so no finite-but-wrong iterate is returned. The
  shared SVD pivot gate is adopted when the nonlinear solve is integrated into
  the MNA `_factor` path.
* **Census document name.** The brief referred to a census file
  `docs/reference/fdtd-gap-05-guard-census.md` that does not exist in this
  repository; the guards are registered in the maintained
  `docs/reference/fdtd-capability-guard-census.md` (the actual census consumed by
  `tests/api/public/test_guard_census.py`), following repo convention.
* **Per-iteration host syncs.** The eager Newton path performs per-iteration host
  reductions (residual `.item()` trajectory, `bool()` convergence branch, mask
  `.any()` gates) beyond the `torch.linalg.cond` estimate. The "0 per-iteration
  host syncs" gate needs the full N3 device-resident / fixed-iteration unrolled
  restructure and is CUDA-deferred, not achievable by moving `cond` off the hot
  path alone.

## Numerical notes

- **Dual convergence gate.** An iterate is accepted only when *both* the scaled
  KCL residual (`||F|| <= atol + rtol * max(|branch/source current|)`) and the
  Newton update (`||dx|| <= uatol + urtol * ||x||`) satisfy their own tolerances.
  A residual-only or update-only pass never counts as convergence, which
  forecloses the classic exponential-device false convergence where a tiny
  residual floor masks a still-moving solution.
- **Stable `expm1` + `pnjlim`.** The diode law uses `Is * expm1(v / (n Vt))` and
  a SPICE-style pn-junction voltage limit `pnjlim` around the critical voltage
  `vcrit = n Vt ln(n Vt / (sqrt(2) Is))`. Limiting bounds the per-iteration
  junction-voltage change so the exponential cannot overflow, and is inactive
  near the root, where Newton recovers quadratic convergence to the machine
  floor.
- **Thermal voltage** `Vt = k T / q` is frozen from the device `temperature` at
  compile time (CODATA 2018 constants).

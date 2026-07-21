# F1 co-simulation E2 evidence — acceptance (Plan 04, S3)

Track `f1-cosim-e2`, worktree `.worktrees/wf1-cosim-e2`, branch `fable/cosim-e2`,
base master `b3d3c77`. GPU `CUDA_VISIBLE_DEVICES=0`, conda `maxwell`,
`CUDA_HOME=.../nvidia/cu13`, `PYTHONPATH=<worktree>`.

This document is appended per stage. **Stage F1a** (coupled EM+circuit
conservation / energy-residual suite) is recorded below. Stage F1b (independent
offline circuit cross-check) is a later stage in this track and will append here.

## Stage F1a — delivered items

A multi-scenario coupled FDTD + MNA global energy-conservation suite:
`tests/rf/circuits/test_circuit_conservation.py`. Three strongly coupled
field/circuit scenarios are each driven from an in-circuit source and run in a
closed, non-absorbing (PEC) vacuum box, so the global balance closes with no
boundary outflow and no material loss:

    S_source(t) = dU_field(t) + dU_circuit(t) + D_circuit(t)

mapping onto the brief's balance (boundary outflow = 0 by the closed box,
material dissipation = 0 in vacuum):

| brief term            | measured channel | source of record |
| --------------------- | ---------------- | ---------------- |
| source injected       | `S_source`       | MNA source branch V*I (independent + controlled sources) |
| delta EM stored       | `dU_field`       | raw Yee E/H fields: `0.5 eps E^2 + 0.5 mu H(n-1/2).H(n+1/2)` |
| boundary outflow      | 0                | closed PEC box (validated, see cavity gate) |
| material dissipation  | 0                | vacuum |
| circuit dissipated    | `D_circuit`      | MNA resistor V*I, integrated |
| circuit stored        | `dU_circuit`     | MNA companion state `0.5 C V^2 + 0.5 L I^2` |

The three scenarios (brief-mandated types):

- **(a) resistive load on a driven port** — series source + 50 ohm resistor bound
  to a `LumpedPort` (the established MNA-coupling terminal port; `TerminalPort` in
  the public API is the wire-binding variant and is not the circuit-bound path).
- **(b) resonant series RLC via MNA** — source + R + L + C assembled from MNA
  primitives (`Resistor`/`Inductor`/`Capacitor`), **not** the native `SeriesRLC`
  termination.
- **(c) controlled-source network** — a `VoltageControlledVoltageSource` (VCVS)
  driving a resistive output stage; the VCVS delivered power is included in
  `S_source`.

The coupled runs are genuine coupled FDTD+MNA runs: the per-step order
(magnetic update, electric update, port/circuit coupling `apply_port_runtimes`,
PEC clamp) is exactly the solver's per-step order for a source-free,
non-dispersive, PEC-bounded vacuum scene; only per-step energy diagnostics are
added. This manual-loop-with-diagnostics pattern matches the existing
`tests/rf/circuits/test_fdtd_circuit_coupling.py` and
`tests/rf/lumped/test_fdtd_lumped_runtime.py` energy tests.

Sources are driven from a zero DC operating point (`SineWaveform` offset 0), so
the runtime's zero-initial-field DC-consistency guard is satisfied.

## Honest gate classes

- **Consistency class:** `S_source`, `D_circuit`, `dU_circuit`. The MNA solve
  enforces KCL/KVL (Tellegen) and the trapezoidal companion model, so the
  circuit-internal statement `S_source = D_circuit + dU_circuit + W_port`
  (W_port = port work) is algebraically forced by the coupling. The memoryless
  resistive scenario (a) is deliberately dissipation-dominated: its field-storage
  term is ~6e-4 of throughput, so the conservation gate there is largely a
  Tellegen-consistency statement (this is the same "memoryless V/I dissipation is
  consistency class" trap flagged in the E4a acceptance).
- **Genuine, two-sided:** `dU_field == -W_port`. The whole-domain
  electromagnetic energy computed from the **raw E/H fields** must equal the work
  the port injection did on the field, taken from the **MNA port V/I record**.
  These share no code path (Yee field update + dual-mesh energy metric vs the
  companion port stamp). This is carried by the dedicated `field-link` gate and
  is what makes the field-coupling term load-bearing and falsifiable.
- **Conservation gate:** the whole balance
  `dU_field + dU_circuit + D_circuit - S_source` closing simultaneously.

The closed-box (zero boundary outflow) assumption is validated by
`test_lossless_cavity_conserves_discrete_energy`: the same discrete-energy
functional is conserved to < 1e-5 (observed ~0, i.e. below print resolution)
over a 3000-step source-free run, establishing the box does not leak. On the
uniform grid the control-volume metric is a global constant factor, so absolute
metric correctness is instead established by the field-link gate (below), whose
`dU_field == -W_port` closure is in physical Joules.

## Test inventory and pass counts

Command (env exports as above):

```
python -m pytest tests/rf/circuits/test_circuit_conservation.py -q      # 6 passed (~29 s)
```

Nodes:

- `test_lossless_cavity_conserves_discrete_energy` — closed-box energy-functional
  closure (support gate).
- `test_resistive_load_conserves_coupled_energy` — scenario (a): conservation +
  field-link.
- `test_series_rlc_conserves_coupled_energy` — scenario (b): conservation +
  field-link + reactive storage > 0.
- `test_controlled_source_network_conserves_coupled_energy` — scenario (c):
  conservation + field-link.
- `test_conservation_gate_rejects_one_percent_channel_imbalance[dissipation_scale]`
  and `[source_scale]` — in-suite falsification: a 1% imbalance in a throughput
  channel is rejected (baseline passes, +1% fails).

Adjacent suites run (env exports as above):

```
python -m pytest tests/rf/circuits/test_circuit_conservation.py \
  tests/rf/circuits/test_fdtd_circuit_coupling.py \
  tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q                          # 46 passed (~39 s)
python -m pytest tests/rf/circuits/ -q                                  # 123 passed (~73 s)
```

## Pre-registered tolerances and observed margins

Tolerances were frozen before measurement and are asserted in the test module:
`_CONSERVATION_TOL = 5e-3` (of energy throughput), `_LINK_TOL = 2e-2` (of peak
field energy), `_STEPS = 2000`. The observed margins are reproduced by a
committed probe that reuses the test fixtures (so the printed numbers cannot
drift from the gate thresholds):

```
CUDA_VISIBLE_DEVICES=0 python \
  docs/assessments/f1-cosim-e2-probes/conservation_margins_probe.py
```

Observed on this host:

| scenario | throughput (J) | conservation residual / throughput (tol 5e-3) | field-link residual / peak field (tol 2e-2) |
| -------- | -------------: | -------------------------------------------: | ------------------------------------------: |
| (a) resistive load | 1.93e-10 | 1.20e-4 | 2.94e-3 |
| (b) series RLC     | 8.47e-11 | 1.36e-3 | 2.94e-3 |
| (c) VCVS network   | 4.74e-10 | 4.86e-5 | 2.94e-3 |

The conservation residual is a bounded half-step artifact (absolute residual is
constant in step count: scenario (b) `|R| = 1.16e-13 J` at both 1000 and 2000
steps, so the relative figure halves as throughput grows). The field-link
absolute residual is `3.6e-16 J` (float32 field-accumulation noise at the peak
field scale `1.2e-13 J`).

## Falsifications recorded

Each load-bearing gate was broken, observed red, restored, and re-verified green.

- **F1 — conservation gate, throughput channel (in-suite, permanent test).**
  `test_conservation_gate_rejects_one_percent_channel_imbalance` scales the
  `dissipation_scale` and `source_scale` channels by 1.01 and asserts the residual
  exceeds `5e-3 * throughput`. Baseline (scale 1.0) passes; +1% fails. This test
  is committed and green, so the falsification is standing evidence that the
  conservation assert is load-bearing on the source and dissipation channels.
  (The field and circuit-store channels are individually below the throughput
  floor and are covered by F2 below, not by a 1% global perturbation.)

- **F2 — field-link gate, injection operator (monkeypatch).** Wrapped
  `EMCircuitRuntime._apply_field_current` to scatter an extra 5% of the port
  injection into the field beyond the recorded current (breaking the field vs
  record symmetry). `test`-equivalent field-link check went **RED**
  (`dU_field` diverges from the port record); the same corruption drove the
  conservation residual to `609 x throughput` (over-injection destabilizes the
  coupled system). Restored -> both green. Driver:
  `docs/assessments/f1-cosim-e2-probes/falsify_field_link.py` (reproduces the
  red/green transition).

- **F3 — closed-box functional (control volume / closure).** The cavity gate is a
  support check; its physical content (zero boundary outflow, correct absolute
  energy scale) is exercised transitively by F2 — a wrong control-volume metric
  or a leaking boundary breaks the field-link closure `dU_field == -W_port` in
  Joules, which F2 shows is a live, falsifiable equality.

## Capability-guard census

Budget unchanged at **176** (`tests/api/public/test_guard_census.py` passes in
every run above). Stage F1a adds only a test module, a probe, and a probe driver;
no product code and no fail-closed FDTD capability guard was added or removed, so
no census reconciliation was required.

## Design notes / accepted deviations

- **In-circuit drive vs "driven port".** The brief phrases scenario (a) as a
  "driven `TerminalPort`". The port is driven — by an in-circuit source through
  the MNA coupling — which makes the source-injected energy directly measurable
  from the MNA branch record (`-V*I*dt`), giving a fully closed, GPU-native
  balance with no need to instrument an EM-side dipole. `LumpedPort` is the
  circuit-bound terminal port used by all existing MNA coupling tests.
- **Field term smallness is physical, not a defect.** The quasi-static port gap
  (0.01 m) is electrically small at 3 GHz, so near-field storage is a small
  fraction of throughput. The genuine field coupling is therefore validated by
  the tight, falsifiable field-link equality rather than by its weight in the
  global residual.
- **`dU_circuit` is consistency-class here.** An independent offline circuit
  solver cross-check of the circuit state (the E2-blocking item) is stage F1b.

## Known gaps / handoff to F1b

- No independent (offline `scipy.integrate.solve_ivp`) circuit cross-check yet;
  that is F1b's deliverable and lifts `dU_circuit`/`S_source`/`D_circuit` from
  consistency-class to independently cross-validated.
- No external-reference (third-party solver) lumped-load cross-check; recorded as
  pending per the brief's stretch item (only if F2's adapter lumped mapping lands
  first).
- No product-code change; no `FEATURE_LIST` capability change — the additive
  `FEATURE_LIST` note records this as validation evidence, not a new feature.

## Stage F1b — independent offline circuit cross-check (the E2-blocking item)

New test module: `tests/rf/circuits/test_circuit_independent_crosscheck.py`. This
predicts a coupled FDTD+MNA run's port voltage/current from a fully independent
path — a hand-derived equivalent-circuit ODE integrated by
`scipy.integrate.solve_ivp` — that shares no runtime code with the solver, closing
the E2 gap left open by F1a (which was consistency-class on the circuit channels).

### Structure and method

1. **EM one-port under test.** A small PML-terminated (open) vacuum box with a
   single `LumpedPort` (`feed`). The box is the "EM structure characterized as a
   port network"; the port is the field/circuit interface.
2. **Measure (passive characterization).** A broadband modulated-Gaussian source
   drives the port through a reference resistor. The port driving-point admittance
   `Y_em(f) = I_port(f)/V_port(f)` is read off the port DFT phasors
   (`result.port("feed").current / .voltage`) across `1.5–5.5 GHz` (25 points). The
   V/I ratio is intrinsic to the linear one-port (independent of the drive), so it
   is **data**, not a shared code path.
3. **Fit.** `Y_em(f)` is fit to an order-4 stable rational model with the shared
   vector fitter `fit_rational` (representation `Y`). Per the brief, `fit_rational`
   is reused **only** to fit the measured data — it is not the MNA runtime. Fit
   rel. error `2.41e-5` over the band.
4. **Derive the state equations by hand.** `_realize_admittance` builds a real
   modal state-space `(A, B, C, D, Cp)` for `Y_em(s)` directly from the fit
   poles/residues (2×2 controllable-companion block per conjugate pair; the
   proportional term `Cp` is the port shunt-capacitance). **No framework
   realization helper is called** — the realization is written in the test.
   `_predict_port_voltage` writes the series-loop KVL ODE by hand:

       dx/dt  = A x + B v_port
       dv/dt  = ( V1(t) − v_port − R ( C x + D v_port ) ) / ( R Cp )
       I_port = ( V1(t) − v_port ) / R

   (`Cp>0` regularizes the port node, giving a well-posed explicit ODE).
5. **Integrate independently.** `solve_ivp` (RK45, `rtol 1e-9`, `max_step 2·dt`)
   integrates the ODE for a **different** drive waveform and a **different** series
   resistance (`R_test = 30 Ω`, center `2.6 GHz`) than the characterization run
   (`R_char = 50 Ω`, center `3.5 GHz`). Because the characterization is a separate
   run with a different stimulus, the agreement is a genuine prediction, not a
   self-fit.

The two transient paths (FDTD+MNA vs scipy modal ODE) share no runtime code; only
the measured impedance data (via `fit_rational`) and the excitation waveform (a
shared input) cross between them.

### Honest gate classes

- **Load-bearing, two-path:** the port **voltage** `v_port(t)`. `v_port` is the
  electrically dominant quantity (the electrically small port is high-impedance, so
  most of the source voltage stands across it), which is why it is the tight gate.
- **Corroboration (cancellation-limited):** the port **current**
  `I_port(t) = (V1 − v_port)/R` is a small difference of near-equal terminal
  voltages (the port draws little current), so its relative precision floor is set
  by the ~`3e-4` difference between the independent analytic stimulus and the
  solver's internal source sampling — not a physics disagreement. It is gated at a
  correspondingly looser, honestly-derived tolerance.

### Pre-registered tolerances and observed margins

Tolerances are frozen in the module (`_FIT_TOL=1e-3`, `_VOLTAGE_TOL=5e-4`,
`_CURRENT_TOL=2e-2`, `_FIT_ORDER=4`, `_FALSIFY_SCALE=1.05`). Reproduced by a
committed probe that reuses the test fixtures:

```
CUDA_VISIBLE_DEVICES=0 python \
  docs/assessments/f1-cosim-e2-probes/crosscheck_margins_probe.py
```

| quantity                     | tol   | observed | headroom |
| ---------------------------- | ----: | -------: | -------: |
| rational fit rel. error      | 1e-3  | 2.41e-5  | ~41×     |
| port-voltage cross-check     | 5e-4  | 1.16e-5  | ~43×     |
| port-current corroboration   | 2e-2  | 7.02e-3  | ~2.8×    |

Context: `peak|v_port| = 5.48e-1 V`, `peak|i_port| = 1.41e-3 A` (the port draws
little current, hence the cancellation-limited current floor). Fit poles
`≈ -9.3e9 ± 2.4e10 j` and `-8.7e9 ± 5.1e10 j` rad/s (stable). The port voltage
rings up and fully decays (tail mean `< 1e-3 ·` peak) so the comparison spans the
whole pulse.

### Falsification recorded

- **F4 — port-voltage gate vs MNA field-port companion (in-suite + driver).**
  `test_crosscheck_rejects_perturbed_mna_companion` builds the independent
  prediction from the **unperturbed** characterization, then scales the MNA
  field-port **companion conductance** (`CircuitPortRuntime.conductance`) by
  `1.05` in a fresh coupled run and asserts the coupled port voltage departs from
  the prediction beyond `_VOLTAGE_TOL`. Baseline `1.16e-5` (GREEN), perturbed
  `4.10e-3` (RED) — a ~350× separation, comfortably straddling the `5e-4` gate.
  This is the brief-mandated "perturb an MNA companion coefficient → coupled trace
  departs → gate red" falsification. Standing in-suite; also reproduced by the
  driver:

  ```
  CUDA_VISIBLE_DEVICES=0 python \
    docs/assessments/f1-cosim-e2-probes/falsify_mna_companion.py
  ```

### Test inventory and pass counts

```
python -m pytest tests/rf/circuits/test_circuit_independent_crosscheck.py -q   # 4 passed (~20 s)
```

Nodes:

- `test_measured_admittance_fits_low_order_rational` — data-quality gate (stable
  order-4 rational fit of the measured `Y_em`).
- `test_independent_prediction_matches_coupled_port_voltage` — **headline gate**:
  independent scipy ODE reproduces the coupled port voltage (with non-vacuous +
  full-decay checks).
- `test_independent_prediction_matches_coupled_port_current` — current
  corroboration.
- `test_crosscheck_rejects_perturbed_mna_companion` — falsification (F4).

Adjacent suites (env exports as above):

```
python -m pytest \
  tests/rf/circuits/test_circuit_independent_crosscheck.py \
  tests/rf/circuits/test_circuit_conservation.py \
  tests/rf/circuits/test_fdtd_circuit_coupling.py \
  tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q                                  # 50 passed (~49 s)
```

### Capability-guard census

Budget unchanged at **176** (`tests/api/public/test_guard_census.py` passes in the
run above). Stage F1b adds only a test module and two probe scripts; no product
code and no fail-closed FDTD capability guard was added or removed.

### Known gaps / deferred

- **External-reference lumped-load cross-check (stretch).** Still pending: F2's
  adapter lumped mapping has not landed on the merge base (`git -C <main> log`
  head is `b3d3c77`), so the brief's stretch item is recorded as pending, not
  attempted.
- The port-current comparison is cancellation-limited (see gate classes) and is
  corroboration, not an independent gate; the load-bearing content is `v_port(t)`.

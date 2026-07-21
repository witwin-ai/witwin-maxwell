# F1 co-simulation E2 evidence — acceptance (Plan 04, S3)

Track `f1-cosim-e2`, worktree `.worktrees/wf1-cosim-e2`, branch `fable/cosim-e2`,
base master `b3d3c77`. GPU `CUDA_VISIBLE_DEVICES=0`, conda `maxwell`,
`CUDA_HOME=.../nvidia/cu13`, `PYTHONPATH=<worktree>`.

This document is appended per stage. **Stage F1a** (coupled EM+circuit
conservation / energy-residual suite) is recorded below. Stage F1b (independent
offline circuit cross-check) is a later stage in this track and will append here.

## Stage F1a — delivered items

A multi-scenario coupled FDTD + MNA global energy-conservation suite:
`tests/rf/circuits/test_circuit_conservation.py`. Three **distinct,
genuinely two-way-coupled** field/circuit scenarios are each driven from an
in-circuit source and run in a closed, non-absorbing (PEC) vacuum box, so the
global balance closes with no boundary outflow and no material loss:

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

**Two-way coupling (not a single degenerate EM case).** In every scenario the
port (`feed`) is bound to a node *behind* a series impedance, **not** to the ideal
source node. This is the decisive fix: if the port is bound directly across the
`VoltageSource` node, its terminal voltage is rigidly pinned to `V1(t)`, the field
cannot back-react, and all three circuits drive a bit-identical field trajectory
(one degenerate EM case run three times). Binding behind the series element lets
the field back-react (the port current develops a drop across the series
impedance), so the three networks present different source impedances to the port
and produce **distinct** field trajectories. This is enforced by
`test_scenarios_are_distinct_em_coupling_cases`, which asserts the pairwise
`max|u_field|` differences are non-trivial (observed 0.10 / 0.49 / 0.54 of peak
field; they would be exactly 0.0 in the pinned/degenerate topology).

The three scenarios (brief-mandated types):

- **(a) resistive load on a driven port** — a `VoltageSource` drives the port
  through a 50 ohm **series** resistor; the port is bound to the node behind that
  resistor. The port is bound via a `LumpedPort`: this is the established
  MNA-coupling path used by every existing circuit/field coupling test, and no
  existing test binds a circuit to a `TerminalPort`. (The circuit compiler,
  `witwin/maxwell/compiler/circuits.py`, does accept a `TerminalPort` name for
  binding as well; a `TerminalPort` would additionally require explicit PEC
  terminal structures, so `LumpedPort` is used here as the simplest established
  circuit-bound terminal — see accepted deviations below.)
- **(b) resonant series RLC via MNA** — source + R + L + C assembled from MNA
  primitives (`Resistor`/`Inductor`/`Capacitor`), **not** the native `SeriesRLC`
  termination, with the R–L–C in series between the source and the port node so
  the port sees the full series-RLC impedance and the reactive elements genuinely
  store the field-driven port current.
- **(c) controlled-source network** — a `VoltageControlledVoltageSource` (VCVS)
  output stage drives the port through a 50 ohm series resistor (`R3`); the VCVS
  delivered power is included in `S_source`.

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
  (W_port = port work) is algebraically forced by the coupling. (The circuit
  channels are lifted off consistency class for the resistive port coupling by the
  F1b independent cross-check; see the F1b handoff note.)
- **Genuine, two-sided:** `dU_field == -W_port`. The whole-domain
  electromagnetic energy computed from the **raw E/H fields** must equal the work
  the port injection did on the field, taken from the **MNA port V/I record**.
  These share no code path (Yee field update + dual-mesh energy metric vs the
  companion port stamp). This is carried by the dedicated `field-link` gate and
  is what makes the field-coupling term load-bearing and falsifiable.
- **Conservation gate:** the whole balance
  `dU_field + dU_circuit + D_circuit - S_source` closing simultaneously. Its
  residual is a bounded half-step-stagger artifact between the field-side and
  circuit-side port-work records (absolute size ~1.6–3.1e-14 J, **constant in step
  count**), so the relative residual falls as 1/steps as throughput accumulates.
  The suite therefore runs `_STEPS = 6000` so the residual is a small fraction of
  the (honest, coupling-scale) throughput. The port is now behind a series
  impedance, so throughput reflects genuine port coupling rather than a resistor
  slapped across the ideal source (the "memoryless V/I dissipation is consistency
  class" trap flagged in the E4a acceptance).

The closed-box (zero boundary outflow) assumption is validated by
`test_lossless_cavity_conserves_discrete_energy`: the same discrete-energy
functional is conserved to < 1e-5 — observed `max|E-mean|/mean = 0.0` to float64
resolution (the symmetric leapfrog functional is exactly conserved on the uniform
grid) — over a 3000-step source-free run, establishing the box does not leak. On the
uniform grid the control-volume metric is a global constant factor, so absolute
metric correctness is instead established by the field-link gate (below), whose
`dU_field == -W_port` closure is in physical Joules.

## Test inventory and pass counts

Command (env exports as above):

```
python -m pytest tests/rf/circuits/test_circuit_conservation.py -q      # 6 passed (~57 s)
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
- `test_scenarios_are_distinct_em_coupling_cases` — anti-regression guard: the
  three scenarios drive **distinct** field trajectories (guards against the
  port-pinned/degenerate topology; pairwise `max|u_field|` diffs must be > 1e-2
  of peak field).
- `test_conservation_gate_rejects_channel_imbalance` — in-suite falsification: a
  3% imbalance (`_FALSIFY_IMBALANCE`) in either the source or dissipation channel
  is rejected (baseline passes, +3% pushes the residual well above tolerance).

Adjacent suites run (env exports as above):

```
python -m pytest tests/rf/circuits/test_circuit_conservation.py \
  tests/rf/circuits/test_fdtd_circuit_coupling.py \
  tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q                          # 46 passed (~75 s)
python -m pytest tests/rf/circuits/ -q                                  # 127 passed (~137 s)
```

(The `tests/rf/circuits/` total includes the F1b independent cross-check suite;
the longer wall time reflects the longer `_STEPS = 6000` conservation runs.)

## Pre-registered tolerances and observed margins

Tolerances were frozen before measurement and are asserted in the test module:
`_CONSERVATION_TOL = 1.5e-2` (of energy throughput), `_LINK_TOL = 2e-2` (of peak
field energy), `_STEPS = 6000`, `_FALSIFY_IMBALANCE = 1.03`. The observed margins
are reproduced by a committed probe that reuses the test fixtures (so the printed
numbers cannot drift from the gate thresholds):

```
CUDA_VISIBLE_DEVICES=0 python \
  docs/assessments/f1-cosim-e2-probes/conservation_margins_probe.py
```

Observed on this host (`_STEPS = 6000`):

| scenario | throughput (J) | peak field (J) | conservation residual / throughput (tol 1.5e-2) | field-link residual / peak field (tol 2e-2) |
| -------- | -------------: | -------------: | ----------------------------------------------: | ------------------------------------------: |
| (a) resistive load | 7.71e-12 | 9.90e-14 | 2.04e-3 | 2.93e-3 |
| (b) series RLC     | 2.63e-12 | 8.95e-14 | 6.70e-3 | 2.86e-3 |
| (c) VCVS network   | 3.04e-10 | 1.94e-13 | 1.02e-4 | 2.93e-3 |

Throughput now reflects genuine port coupling (the port is behind a series
impedance), not a resistor across the ideal source; the scenarios are distinct
(different peak field energies and throughputs). The conservation residual is a
bounded half-step artifact — the **absolute** residual is constant in step count
((a) `1.58e-14 J`, (b) `1.76e-14 J`, (c) `3.09e-14 J`, unchanged at 2000 / 4000 /
8000 steps), so the relative figure falls as ~1/steps as throughput accumulates;
`_STEPS = 6000` gives ≥ 2.2× headroom for the tightest scenario (b). The field-link
absolute residual is ~`2.6–5.7e-16 J` (float32 field-accumulation noise at the
peak-field scale ~`1e-13 J`).

## Falsifications recorded

Each load-bearing gate was broken, observed red, restored, and re-verified green.

- **F1 — conservation gate, throughput channel (in-suite, permanent test).**
  `test_conservation_gate_rejects_channel_imbalance` scales the `dissipation_scale`
  and `source_scale` channels by `_FALSIFY_IMBALANCE = 1.03` and asserts the
  residual exceeds `1.5e-2 * throughput`. Baseline (scale 1.0) passes at
  `2.04e-3`; +3% pushes the residual to ~`3.1e-2` (well above the `1.5e-2` gate).
  This test is committed and green, so the falsification is standing evidence that
  the conservation assert is load-bearing on the source and dissipation channels.
  (A 3% imbalance is used, not 1%: the honest coupling-scale residual is ~0.2–0.7%
  of throughput, so a 1% perturbation is only marginally separated; 3% is cleanly
  caught. The field and circuit-store channels are covered by F2 below.)

- **F2 — field-link gate, injection operator (monkeypatch).** Wrapped
  `EMCircuitRuntime._apply_field_current` to scatter an extra 5% of the port
  injection into the field beyond the recorded current (breaking the field vs
  record symmetry). The field-link check went **RED** (`dU_field` diverges from
  the port record); the same corruption also grossly violates global conservation.
  Reproduced by the committed driver
  `docs/assessments/f1-cosim-e2-probes/falsify_field_link.py` at 800 steps, which
  always computes the conservation ratio (it no longer aborts on the field-link
  assert), printing:

  ```
  baseline (800 steps): field-link PASS; conservation residual/throughput = 1.453e-02
  broken (injection over-scatter by 5%): field-link RED as expected; conservation residual/throughput = 4.374e-01
  restored (800 steps): field-link PASS; conservation residual/throughput = 1.453e-02
  ```

  i.e. the broken injection drives the conservation residual from ~1.5% to ~44% of
  throughput at 800 steps (the exact factor is step-count dependent because the
  over-injection is unstable and grows with run length; the driver fixes the step
  count so the printed number is reproducible).

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

- **In-circuit drive vs "driven port"; `LumpedPort` vs `TerminalPort`.** The brief
  phrases scenario (a) as a "driven `TerminalPort`". The port is driven — by an
  in-circuit source through the MNA coupling — which makes the source-injected
  energy directly measurable from the MNA branch record (`-V*I*dt`), giving a fully
  closed, GPU-native balance with no need to instrument an EM-side dipole.
  `LumpedPort` (not `TerminalPort`) is used for the binding. The circuit compiler
  (`witwin/maxwell/compiler/circuits.py`) does accept **both** `LumpedPort` and
  `TerminalPort` names for circuit binding, so this is a deliberate choice, not a
  hard constraint: `LumpedPort` is the established MNA-coupling path exercised by
  every existing circuit/field coupling test, no existing test binds a circuit to a
  `TerminalPort`, and a `TerminalPort` would additionally require explicit PEC
  terminal structures. `LumpedPort` is therefore the simplest established
  circuit-bound terminal for this evidence.
- **Field term smallness is physical, not a defect.** The quasi-static port gap
  (0.01 m) is electrically small at 3 GHz, so near-field storage is a small
  fraction of throughput. The genuine field coupling is therefore validated by
  the tight, falsifiable field-link equality rather than by its weight in the
  global residual.
- **`dU_circuit` is consistency-class here.** An independent offline circuit
  solver cross-check of the circuit state (the E2-blocking item) is stage F1b.

## Known gaps / handoff to F1b

- No independent (offline `scipy.integrate.solve_ivp`) circuit cross-check yet;
  that is F1b's deliverable. F1b's cross-check circuit is source + series **R**
  only (no L/C), so it independently cross-validates `S_source` and `D_circuit`
  and the resistive port coupling, but it does **not** exercise the reactive
  companion storage — `dU_circuit` (the C/L stored-energy channel) therefore
  remains consistency-class plus whatever pre-existing MNA reactive-companion unit
  coverage exists, and is not lifted by F1b. (Scenario (b) here does drive genuine
  reactive storage, but that channel is Tellegen-forced within the coupled run.)
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
  by the small difference between the independent analytic stimulus and the
  solver's internal source sampling — not a physics disagreement. Its observed
  relative error is `7.02e-3` (reproduced by the committed probe below), looser
  than the port-voltage gate; it is gated at a correspondingly looser tolerance and
  is corroboration, not the headline gate.

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
  tests/api/public/test_simulation_smoke.py -q                                  # 50 passed (~89 s)
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

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

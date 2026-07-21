# H4 acceptance — circuit-driven ESD through the standard-network MNA topology

Track: `h4-esd-circuit`. Stage: **H4a** (circuit-ESD coupling + gates (a)-(d) +
falsifications). Worktree base: `18bc42a` (branch `fable/esd-circuit`). GPU:
`CUDA_VISIBLE_DEVICES=1`. Env: `maxwell`.

All commands below assume:

```bash
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wh4-esd-circuit
export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

## Delivered items

1. **ESD waveform -> circuit source resample.** `_WaveformBase.to_circuit_waveform(
   t_end, samples, scale)` (in `witwin/maxwell/esd.py`) resamples any
   `ESDWaveform`/`MeasuredWaveform` onto a `PiecewiseLinearWaveform` MNA source
   table `(t, scale * i(t))`; charge/impulse converges to the analytic value.
2. **`ESDVoltageSource`** (`witwin/maxwell/esd.py`): the source-impedance-network
   ESD excitation. `build_circuit()` assembles the standard 330 ohm / 150 pF
   generator `Circuit` (voltage source driven by `R_d * i_esd(t)` -> discharge
   resistor -> storage capacitor to ground, bound to the named scene port) and
   stamps generator provenance on `circuit.metadata['esd_generator']`. Constants
   `ESD_STANDARD_DISCHARGE_RESISTANCE` / `ESD_STANDARD_STORAGE_CAPACITANCE`.
   Exported from `witwin.maxwell`.
3. **Provenance ride-through.** `Result.esd_generator(name)` /
   `Result.esd_generator_names()` (in `witwin/maxwell/result.py`) surface the
   generator-network provenance (standard, revision, level voltage, model
   version, element values, `source_impedance_network` injection tag).
4. **Gates (a)-(d) + falsifications** (below).

Capability framing (docstrings + provenance + FEATURE_LIST): circuit
**approximation of the standard network**, NOT discharge-gun geometry,
calibration-target, or system certification.

## Files added / changed

- `witwin/maxwell/esd.py` (+`to_circuit_waveform`, `ESDVoltageSource`, constants).
- `witwin/maxwell/__init__.py` (export `ESDVoltageSource`).
- `witwin/maxwell/result.py` (`esd_generator`, `esd_generator_names`).
- `FEATURE_LIST.md` (additive subsection).
- `tests/esd/test_esd_circuit.py` (gate a + c + unit).
- `tests/esd/test_esd_circuit_energy.py` (gate b).
- `tests/esd/test_esd_circuit_e2e.py` (gate d + fail-closed blocker pin).
- `docs/assessments/h4-esd-circuit-acceptance-2026-07-21.md` (this doc).

## Gates and observed margins (reproducible)

Pre-registered tolerances are frozen in the test modules. Observed margins are
reproducible by rerunning the named nodes (values recorded on the reference host,
`CUDA_VISIBLE_DEVICES=1`).

### Gate (a) — RC-load analytic cross-check (F1b pattern; no shared runtime code)

`tests/esd/test_esd_circuit.py`. The ESD waveform drives the 330 ohm / 150 pF
network into a resistive load; the coupled FDTD+MNA port voltage/current is
reproduced by an independent scipy `solve_ivp` integration of the hand-derived
series-loop ODE, with the EM one-port measured, fit (`fit_rational`, data only),
and realized by hand into a modal state space.

- Rational fit rel. error: **1.9e-5** (gate `_FIT_TOL = 1.0e-3`).
- Port-voltage cross-check rel. error (headline): **7.8e-4** (gate
  `_VOLTAGE_TOL = 1.0e-2`).
- Source-current cross-check rel. error (corroboration): **9.3e-5** (gate
  `_CURRENT_TOL = 3.0e-2`).
- Context: peak port stress ~1328 V, peak source current ~33.5 A at the 8 kV
  contact level (consistent with the ~30 A standard first peak).

### Gate (b) — coupled global energy conservation (F1a pattern)

`tests/esd/test_esd_circuit_energy.py`. Closed PEC box, in-circuit ESD generator;
`S_source = dU_field + dU_circuit + D_circuit` with every channel from an
independent record (MNA branch record, MNA companion state, raw Yee E/H).

- Conservation residual / throughput: **1.34e-4** (gate `_CONSERVATION_TOL = 1.5e-2`).
- Field-link residual / peak field (`dU_field == -W_port`): **5.2e-4** (gate
  `_LINK_TOL = 2.0e-2`).

### Gate (c) — provenance ride-through

`tests/esd/test_esd_circuit.py::test_generator_provenance_rides_through_to_result`
and the e2e assertion. `Result.esd_generator(name)` returns the waveform standard
revision (`ed2-contact`), level voltage (8000.0), the 330 ohm / 150 pF element
values, and the `source_impedance_network` injection tag.

### Gate (d) — circuit-driven end-to-end

`tests/esd/test_esd_circuit_e2e.py::test_circuit_driven_prebias_esd_stress_end_to_end`.
Electrostatic pre-bias (DC -> Yee, Gauss residual within tolerance) + circuit-driven
ESD through the `TerminalPort` strong MNA coupling + non-feedback `BreakdownMonitor`
stress, in one FDTD run. Asserts pre-bias provenance, a genuine non-zero coupled
port trace, generator provenance ride-through, and a populated on-device stress
record.

## Falsifications recorded

- **Gate (a)** —
  `test_esd_circuit.py::test_crosscheck_rejects_perturbed_storage_capacitance`:
  the independent prediction uses the nominal 150 pF network; a fresh coupled run
  with the storage capacitance scaled by 1.06 shifts its port voltage to
  **4.5e-2** rel. error against the prediction (>> the 1.0e-2 gate; ~58x the
  unperturbed 7.8e-4 baseline). Restored to nominal -> green.
- **Gate (b)** —
  `test_esd_circuit_energy.py::test_conservation_gate_rejects_channel_imbalance`:
  scaling the source channel by 1.03 -> residual/throughput **3.0e-2**; scaling
  the dissipation channel by 1.03 -> **2.6e-2**; both exceed the 1.5e-2 gate,
  while the unperturbed balance is 1.34e-4.

## Known gaps / deferred

- **Design blocker (documented, fail-closed):** the strong FDTD+MNA port coupling
  does not support conductive media (`_validate_supported_field_coupling` in
  `witwin/maxwell/fdtd/ports.py` raises `NotImplementedError` "Lumped FDTD
  coupling in conductive media requires a conductance-aware port update
  coefficient"). A `DielectricBreakdown` material fundamentally introduces a
  (post-breakdown) conductivity, so the *dynamic conductive breakdown feedback*
  cannot ride the circuit-driven port path in the current runtime. Closest
  achievable behavior delivered: gate (d) pairs the circuit-driven port with a
  lossless dielectric + non-feedback stress monitor; the dynamic conductive
  breakdown feedback stays on the ideal-current-injection path (the companion
  `tests/esd/test_prebias.py::test_prebias_esd_breakdown_end_to_end`, unchanged).
  The fail-closed guard is pinned by
  `test_esd_circuit_e2e.py::test_circuit_port_coupling_fails_closed_in_breakdown_media`.
  A conductance-aware port update coefficient is the required future work.
- Stage **H4b** (`SmoothBreakdownRisk` surrogate + gradient/monotonicity gates)
  is not part of H4a.

## Test inventory (this stage)

Command:

```bash
python -m pytest tests/esd tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py -q
```

Result: **73 passed** (includes the pre-existing ESD/prebias suites plus the new
`test_esd_circuit*.py`). New nodes:

- `tests/esd/test_esd_circuit.py`: 9 passed (gate a headline+corroboration+fit,
  gate-a falsification, gate c, waveform-resample unit x2, topology, validation).
- `tests/esd/test_esd_circuit_energy.py`: 3 passed (gate b conservation, field
  link, gate-b falsification).
- `tests/esd/test_esd_circuit_e2e.py`: 2 passed (gate d, fail-closed blocker pin).

Adjacent suites (touched modules `esd.py`, `result.py`, `__init__.py`):

```bash
python -m pytest tests/rf/circuits/test_result_circuit_data.py \
  tests/rf/circuits/test_circuit_contract.py tests/breakdown/test_breakdown_monitor.py -q
```

Result: **31 passed**.

## Census

`CAPABILITY_GUARD_BUDGET = 175` at base `18bc42a` (verified;
`tests/api/public/test_guard_census.py` -> 3 passed). This stage adds **no**
`raise NotImplementedError` guards (it adds a capability and reuses an existing
fail-closed guard), so the budget is unchanged at 175.

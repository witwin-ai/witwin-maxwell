# H4 acceptance — circuit-driven ESD through the standard-network MNA topology

Track: `h4-esd-circuit`. Stage: **H4a** (circuit-ESD coupling + gates (a)-(d) +
falsifications). Worktree base: `589188e` (branch `fable/esd-circuit`). GPU:
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

- `tests/esd/test_esd_circuit.py`: 10 passed (gate a headline+corroboration+fit,
  gate-a falsification, gate-a EM-load-bearing companion [audit-minor, see below],
  gate c, waveform-resample unit x2, topology, validation).
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

`CAPABILITY_GUARD_BUDGET = 175` at base `589188e` (verified;
`tests/api/public/test_guard_census.py` -> 3 passed). This stage adds **no**
`raise NotImplementedError` guards (it adds a capability and reuses an existing
fail-closed guard), so the budget is unchanged at 175.

---

# H4b — SmoothBreakdownRisk differentiable surrogate (non-physical, non-regulatory)

Base for this stage: `bfa0d2e` (H4a). Delivered on branch `fable/esd-circuit`.

## Delivered

- `witwin/maxwell/breakdown_risk.py` (new module, typed SEPARATELY from
  `breakdown.py` and `breakdown_stress.py`): `SmoothBreakdownRisk` (config +
  differentiable `evaluate` / `evaluate_from_components`) and
  `SmoothBreakdownRiskData` (typed output). Risk =
  `reduce_cells(occupancy * sum_t sigmoid((|E|-Ecrit)/w) dt)`; reductions
  `sum` (soft over-stress dose, s) / `mean` / `softmax` (temperature-weighted
  worst cell). Optional `damage_exponent` adds a `soft_damage` map without
  changing the primary `risk`. Provenance carries `non_physical=True`,
  `non_regulatory=True`, the sigmoid-margin definition, model version
  `smooth-breakdown-risk-1`, and the verbose capability tag.
- `colocate_electric_magnitude` refactored to slice the trailing three (X,Y,Z)
  axes, so a leading time axis `(T,X,Y,Z)` rides through untouched — a single
  implementation now serves both the per-step observer and the surrogate's
  recorded `|E|(t)` series (no duplicate colocation). Backward compatible with
  the 3D observer call.
- Public exports: `mw.SmoothBreakdownRisk`, `mw.SmoothBreakdownRiskData`.
- `simulation.py` hard-breakdown trainable-rejection guard KEPT (not weakened);
  only its hint text updated to point at the now-available surrogate.

## Gates + observed margins (pre-registered tolerances)

All in `tests/breakdown/test_smooth_breakdown_risk.py`, CPU float64 (pure torch,
no CUDA kernels):

- **gradient (FD, small scene, float64)**: analytic backward vs central
  difference for a source-amplitude and a material-screening parameter threaded
  through a synthetic differentiable `|E|(t)` — both relative errors `< 1e-4`
  (gate 1e-4), opposite-sign sensitivities (source `> 0`, material `< 0`), and a
  `torch.nn.Parameter` reach check.
- **monotonicity**: `risk` strictly increases across source amplitudes
  `[0.8, 1.0, 1.2, 1.5, 2.0] MV/m`.
- **zero far below threshold**: peak field ~1e-2 of `Ecrit` (margin ~ -50
  widths) → `risk < 1e-18`, `peak_instant_risk < 1e-15`; and a far-vs-at
  contrast with `at > 1e6 * far`.
- **colocation reuse**: `evaluate_from_components` == manual colocation; batched
  colocation == per-step loop; uniform field → `sqrt(3)*val` exactly.
- **typing / non-physical tag**: `SmoothBreakdownRiskData`, capability level
  contains "non-physical" and "non-regulatory", provenance flags asserted.
- **reductions/diagnostics**: softmax bracketed `mean <= soft <= peak`;
  occupancy zeroing; damage map optional and non-invasive.
- **fail-closed validation**: bad `critical_field`/`width`/`reduction`/
  `temperature`/`damage_exponent` and bad `evaluate` inputs all raise.

## Falsifications recorded (headline gates)

- **A (gradient)**: `p = torch.sigmoid(margin.detach())` — both gradient tests go
  red with `RuntimeError: element 0 ... does not require grad`. Restored.
- **B/C (monotonicity + zero-far-below)**: `p = torch.sigmoid(-margin)` (sign
  flip) — `test_risk_monotone_in_source_amplitude`,
  `test_risk_grows_from_negligible_to_finite_across_threshold`, and
  `test_risk_vanishes_far_below_threshold` all go red. Restored.

## Test commands / counts

```bash
export CUDA_HOME=.../nvidia/cu13; export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=<worktree>; export CUDA_VISIBLE_DEVICES=1
python -m pytest tests/breakdown/test_smooth_breakdown_risk.py -q          # 17 passed
python -m pytest tests/breakdown tests/esd \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py \
  tests/api/public/test_guard_census.py -q                                 # 157 passed
```

## Census

`CAPABILITY_GUARD_BUDGET = 175` verified at this stage's base (`test_guard_census`
green). H4b adds **no** `raise NotImplementedError` guards (the surrogate's input
validation uses `ValueError`/`TypeError`, which the census does not count), and
the hard-breakdown trainable-rejection guard is retained. Budget unchanged at 175.

## Known gaps / must-know for the next agent

- The surrogate is a post-hoc differentiable functional (SAR `soft_peak`
  precedent): gradient flows through whatever produces `e_magnitude_series`. The
  production CUDA forward does NOT autograd-tape `FieldTimeMonitor` time-series
  buffers, so an end-to-end gradient through the real solver's recorded `|E|(t)`
  is out of scope here — it would require seeding time-domain observers in the
  FDTD adjoint bridge (a separate, larger slice). The gate validates the
  surrogate's differentiability on a small taped `|E|(t)` field, exactly as the
  SAR surrogate is gated on leaf field tensors.
- No `Result` convenience method was wired to a `FieldTimeMonitor` this stage
  (colocation from volume buffers is non-differentiable through the forward and
  would risk implying a trainable path that does not exist). The public entry is
  the functional `SmoothBreakdownRisk.evaluate[_from_components]`.

---

## Audit-minor cleanup (round-H, 2026-07-21)

Round-H audit minors on the H4 delivery. Env: `maxwell`, `CUDA_VISIBLE_DEVICES=1`.
`python -m pytest tests/esd -q -> 44 passed`.

### (a) Gate (a) is EM-insensitive on its own; added an EM-load-bearing companion

Gate (a) compares the coupled FDTD+MNA port voltage to an independent scipy ODE that
includes the measured EM one-port. But the standard network's 150 pF storage cap so
heavily shunts the ~0.13 pF field one-port (measured `Im(Y)/omega ~ -1.3e-13 F`,
`|Y| ~ 1e-3..6e-3 S` over 1.5-5.5 GHz) that **zeroing the EM one-port in the
prediction changes nothing** at the slow ESD timescale: on the standard bench the
with-EM prediction rel is `7.76e-4` and the zeroed-EM prediction rel is `6.90e-4`
(0.89x — the field coupling is immaterial), so gate (a) alone does not make EM
coupling load-bearing. Confirmed the auditor's finding directly.

Added `test_em_one_port_is_load_bearing`: a variant network (storage cap 1 pF into a
2000 ohm load — same characterized scene / same field one-port, just a fast
high-impedance drive network) where the field one-port materially shifts the port
voltage. Reusing the existing characterization fixture, with a fresh coupled run of
the variant network:

- with-EM prediction rel (headline): **2.0e-3** (gate `_EM_VOLTAGE_TOL = 6.0e-3`).
- zeroed-EM prediction rel: **2.5e-2** (gate `_EM_MATERIAL_TOL = 1.0e-2` — EM must
  MISS by more than this), improvement factor **~12.7x** (gate
  `_EM_IMPROVEMENT_FACTOR = 4.0x`).

So the field one-port is now load-bearing: only the full model reproduces the coupled
FDTD, and dropping it misses materially. Tolerances pre-registered from the reference
host with generous slack. Probe of candidate networks (all on `CUDA_VISIBLE_DEVICES=1`):
`cs=2pF/rload=1k -> 7.9x`, `cs=1pF/rload=1k -> 12.0x`, `cs=1pF/rload=2k -> 12.7x`,
`cs=0.5pF/rload=2k -> 19.8x`; chose `1pF/2k` (robust, rel_true well inside the fit's
valid band).

- **Falsification (committed, in-test)**: the `zero_em=True` branch **is** the
  falsification — it drops the field coupling (`I_field = 0`, `C_p = 0`) from the
  prediction and the reproduction against the same coupled run reddens to `2.5e-2`
  (`> _EM_VOLTAGE_TOL`, failing both assertion (2) and the improvement-factor
  assertion (3)). The with-EM branch stays green at `2.0e-3`.

### (b) Gate (b) falsification is post-hoc arithmetic; runtime sensitivity noted

Gate (b)'s committed falsification
(`test_conservation_gate_rejects_channel_imbalance`) scales the **recorded** balance
channels (`conservation_residual(dissipation_scale=1.03)` /
`source_scale=1.03`) — a post-hoc arithmetic perturbation of the already-integrated
energy record, not a re-run of the solver with a corrupted runtime. It proves the
balance residual is sensitive to a channel imbalance (a genuine and useful guard) but
does not by itself re-exercise the runtime. The auditor verified runtime sensitivity
separately (a real runtime change also breaks the balance).

A committed runtime falsification would require a **second full closed-box balance
run** (the `record` fixture is one ~2400-step FDTD run; a leaky-boundary or
corrupted-coupling variant is another of the same cost) for coverage the auditor has
already confirmed and the post-hoc gate already localizes. Judged **not cheap**
relative to that redundancy, so it is left out per the "if cheap" latitude; this note
records the honest disposition. (The new gate-(a) EM-load-bearing test above is itself
a fresh *runtime* falsification of the ESD field coupling — a real coupled run whose
result changes load-bearingly with the field one-port.)

### Census

No `raise NotImplementedError` guard added or removed by this cleanup. Budget
unchanged at **176** (current master lineage).

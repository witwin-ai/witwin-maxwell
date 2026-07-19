# Track E2 (rf-scenes) acceptance — 2026-07-19

Worktree: `.worktrees/we2-rf-scenes` (branch `fable/rf-scenes-validation`).
Env: `maxwell`; `CUDA_VISIBLE_DEVICES=1`. Reproduce every run with the common-brief
CUDA_HOME / PYTHONPATH pattern (PYTHONPATH = this worktree).

This document is appended stage-by-stage. Stage E2b (antenna benches) and E2c (M3
cache generation + RESULTS.md rows + cloud task ids) extend it below.

---

## Stage E2a — PlaneMonitor plumbing + open/short/match rebuild + RLC rebuild

Commits:
- `49ba300` feat(rf): thread user monitors through WavePort/PortSweep Results
- `923c85a` feat(rf): rebuild open/short/match and RLC benches on the coax line

### 1. PlaneMonitor passthrough (deliverable 1)

Root cause: both WavePort Result assemblies (`witwin/maxwell/waveport_sweep.py`
`run_waveport_sweep` / `run_waveport_excitation`) built the final `Result` with
`monitors={}`, and `compact_array_column_result` retained only closed-surface
payloads — so user-declared monitors were computed in every column run and then
discarded.

Fix:
- `WAVEPORT_MONITOR_PREFIX` marks the internal per-port ModeMonitors; a
  `_user_monitor_payloads(result, scene)` helper keeps only user monitors.
- A direct `PortExcitation` (single drive column) carries its user monitors
  through unchanged (identical to a plain FDTD run of the injected mode).
- A `PortSweep` exposes the first drive channel at the flat top level (recorded in
  `Result` metadata `user_monitor_drive_channel` / `user_monitor_frequency`) and
  every drive column in `array_run_data.column_results` (`compact_array_column_result`
  widened to retain all non-internal monitor payloads).

Evidence (coax `guide_plane` PlaneMonitor, dx=0.01, 1 GHz):
- WavePort excitation: annulus mean |Ey| = 75.7, PEC inner mean = 0.0, outside
  shield mean = 0.0 (field is the injected TEM mode, exactly zero in the PEC).
- Magnitude-profile correlation WavePort vs an independent plain `ModeSource`
  launch over the annulus = 0.853.
- PortSweep: top-level monitor present (`drive = left::TEM0`); 2 columns, each with
  the monitor in `column_results`.

Tests: `tests/rf/wave_validation/test_planemonitor_waveport_passthrough.py` — 5
passed. Falsification (embedded): monkeypatch `_user_monitor_payloads -> {}`, then
`result.monitor("guide_plane")` raises `KeyError` (recorded green in-test).

### 2. lumped_open_short_match rebuild (deliverable 2)

Rebuilt as `benchmark/scenes/rf/lumped_open_short_match.py::coax_sol_scene`
(standard in {matched, short, open}) on the proven coax line: TEM WavePort feed →
de-embedded load plane. matched = reflectionless coax-through-PML (presents Z0);
short = PEC plug; open = truncated inner rod (shield continues; operated below the
outer-guide TM01 cutoff 0.717 GHz so the open does not leak). PML raised to 12
layers so the sub-GHz matched load reaches −20 dB.

Measured (dx=0.01, 0.45 & 0.50 GHz):
- matched |Γ| max = 0.073 (−23 dB) ≤ 0.1 gate;
- short |Γ| min ≈ 0.998, open |Γ| min ≈ 0.998 (≥ 0.944 = −0.5 dB);
- open/short phase separation ≈ 113–120° (> 90°); short-referenced open Re ≈
  0.40–0.50 (+1 class); short is the −1 reference (documented SOL convention);
- open-end fringe extension ≈ 5.5 cm (the measured, documented departure of the
  open/short separation from an ideal 180°).

Phase-reference convention: the short defines the −1 load-plane reference; the open
then lands in the +1 class. Documented in the scene docstring.

Tests: `tests/rf/wave_validation/test_open_short_match_wave_level.py` — 6 passed.
Falsification (recorded, break→red→restore): forcing `coax_sol_scene` to always
build the `matched` standard turned short/open into ~0.07 reflection and collapsed
the discrimination — `test_short_and_open_totally_reflect`,
`test_short_open_discriminate_by_phase`, `test_three_standards_are_mutually_distinct`
all went red; restored → green. Plus an in-test falsification: feeding identical
Γ (the decoupled-bench signature) into the discrimination metric fails the gate.

### 3. series_parallel_rlc rebuild (deliverable 3)

Rebuilt as `benchmark/scenes/rf/series_parallel_rlc.py::series_rlc_scene`
(`parallel=` flag): the RLC is an IN-LINE two-terminal `LumpedPort` element in the
coax inner conductor (2-cell axial gap, voltage path along the line, current
contour encircling the rod) ahead of a matched through-PML continuation, so it
carries the full axial line current. The single-spoke SHUNT termination was tried
first and rejected (it barely coupled to the symmetric coax mode — the notch was
stuck near the TE11 artifact and ignored C, the same failure mode as the retired
bench); the in-line element tracks C cleanly.

Measured (dx=0.01, L=25 nH; C = 3.2 / 4.0 / 4.8 pF = nominal ±20%):
- series |S11| notch (interp): 0.4787 / 0.4355 / 0.4000 GHz; `f_res*sqrt(C)`
  spread = 0.0097 (~1% → tracks 1/√(LC)); ±20% ratios [1.099, 1.000, 0.919] vs
  analytic [1.118, 1.000, 0.913] (within 2%); absolute notch = 0.865 × ideal f0
  (the consistent, documented ~13% parasitic downshift from the rod-gap fringe C);
- parallel |S11| peak (interp): 0.3616 / 0.3465 / 0.3324 GHz — monotone in C
  (correct direction); the slope is diluted because the fringe C adds directly to
  the parallel C (documented).

Tests: `tests/rf/wave_validation/test_rlc_resonance_wave_level.py` — 6 passed (the
retired strict-xfail open-gap test is replaced by real passing tracking gates; the
fast companion-impedance analytic-identity contract test is retained, non-gating).
Falsification (recorded, break→red→restore): forcing the scene to ignore the
requested C made `f_res` C-independent — `test_series_resonance_tracks_inverse_sqrt_c`
and `test_series_resonance_moves_correctly_under_20pct_c` went red (ratio 1.0 vs
1.118, monotone broken); restored → green. Plus an in-test falsification: a
constant `f_res` fails the `f_res*sqrt(C)=const` gate.

### Benchmark runners

`benchmark/rf_validation.py::run_lumped_open_short_match` and `run_series_rlc` are
rewritten to drive the rebuilt scenes and record the discrimination / tracking
metrics (status pass). `benchmark/scenes/rf/__init__.py` now exports
`coax_sol_scene` (was `lumped_one_port_scene`). RESULTS.md rows + external-reference
caches are E2c.

### Test commands and results (E2a)

```
python -m pytest tests/rf/wave_validation/                       # 30 passed, 6 xfailed (pre-existing te10 blocked-operator xfails)
python -m pytest tests/rf/wave_validation/test_planemonitor_waveport_passthrough.py   # 5 passed
python -m pytest tests/rf/wave_validation/test_open_short_match_wave_level.py          # 6 passed
python -m pytest tests/rf/wave_validation/test_rlc_resonance_wave_level.py             # 6 passed
python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py tests/monitors/observers/test_fdtd_observers.py  # 38 passed
python -m pytest tests/rf/waveport/test_waveport_sweep.py tests/rf/array/test_array_contracts.py tests/rf/antenna/test_result_antenna.py  # 39 passed
python -m pytest tests/rf/array/test_array_fullwave.py tests/api/public/test_guard_census.py  # 8 passed
```

Guard-census budget unchanged (176); no fail-closed guard added, removed, or
weakened. No FDFD tests touched.

### Known gaps / notes for E2b, E2c

- The matched SOL standard is realized by the reflectionless coax-through-PML
  termination (presents Z0), not a lumped resistor — documented; this reuses the
  proven coax_thru termination.
- The RLC bench operates at the dx=0.01 tier only (the current-loop half-extent is
  snapped to that grid via `_snapped_loop_half`; other dx are validated by the
  snap assertion but not swept). The parallel tracking slope is parasitic-diluted
  (direction-only gate) — an honest limitation, not hidden.
- E2c cloud caches: brief authorizes `rf/coax_thru`, the rebuilt
  `rf/lumped_open_short_match`, `antenna/half_wave_dipole`, `antenna/patch`.

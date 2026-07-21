# F2b acceptance — production quasi-TEM benches + patch feed diagnosis

> Date: 2026-07-21
> Track: f2-rf-trio (branch `fable/rf-trio`)
> Stage F2b: wire the inhomogeneous interior-PEC quasi-TEM mode into the production
> WavePort path; unblock the `rf/microstrip_two_port` and `rf/differential_pair`
> wave-level benches; update the RF validation doc rows honestly; diagnose and attempt
> the patch antenna feed redesign; flip the strict xfail if the patch meets the gate.
> GPU: `CUDA_VISIBLE_DEVICES=1`. Builds on F2a (interior-PEC masking + quasi-static
> line-mode engine, commit `c873e71`).

## 0. Environment / reproduction

```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wf2-rf-trio
export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

## 1. Delivered

1. **Production routing of the inhomogeneous interior-PEC quasi-TEM mode**
   (`witwin/maxwell/fdtd/excitation/modes.py`). In `_assemble_vector_mode_data`, the
   `has_interior_pec and wave_family == "tem"` branch now tries the closed-form
   uniform-fill electrostatic solve (`_solve_pec_tem_mode_torch`) and, when it fails
   closed with `NotImplementedError` on an inhomogeneous (substrate + air) fill, falls
   through — for a **non-magnetic** cross-section — to the F2a quasi-static electrostatic
   line-mode engine (`_solve_quasistatic_line_modes`, `eps_eff = C/C0`,
   `mode_solver_kind = "quasistatic_line_torch"`). A magnetic inhomogeneous line re-raises
   (the legacy diagonal-anisotropic operator keeps serving `mu != 1` apertures). A new
   helper `_tem_signal_potentials(pec_occupancy, mode_index)` maps the isolated-conductor
   count to the drive: single-signal `[1]`; two-signal pair even `[1,1]` (mode_index 0) /
   odd `[1,-1]` (mode_index 1); more than two coupled signals raise a `ValueError`
   (routing boundary, not a new census-tracked `NotImplementedError`).

2. **`rf/microstrip_two_port` and `rf/differential_pair` scenes rebuilt and unblocked**
   (`benchmark/scenes/rf/microstrip_two_port.py`, `differential_pair.py`). Following the
   `coax_thru` precedent: measurement ports near the origin (`+-PORT_X`) so the
   single-precision current-contour planes stay on the Yee half-grid; ground/substrate/
   strips run through the computational PML (`line_length`) so the launched waves
   terminate; integer-cell node arrays (`GridSpec.custom(arange*dx)`) so every face and
   contour lands on an exact Yee node/half-node.

3. **`benchmark/rf_validation.py` runners rewritten** (`run_microstrip`,
   `run_differential_pair`): both now run the real terminated FDTD sweep, assemble the
   network S via `B = S*A`, gate on extraction conditioning + passivity (coax precedent),
   and record the measured quasi-TEM `eps_eff`/mixed-mode conversion with the honest
   resolution gap. Status is measured, not forced.

4. **RF validation doc updated honestly** (`docs/reference/rf-wave-validation-2026-07-18.md`):
   §1 table rows for microstrip/diff-pair moved from **BLOCKED** to **gap** with a new §1.3
   subsection; §3 and §5 blocker bullets updated; a "Resolved (F2b)" block added.

5. **Patch antenna feed: galvanic via + full diagnosis** (`benchmark/scenes/antenna/patch.py`).
   A PEC probe via now carries the feed current from the patch underside to a single-cell
   lumped-port gap above the ground plane. It cut the feed reactance ~5x but the patch
   still does not resonate; the strict-xfail matched-broadside gate is kept fail-closed
   (see §4) and its reason string updated to the F2b diagnosis.

## 2. Test inventory

| target | result |
|---|---|
| `tests/rf/wave_validation/test_microstrip_diffpair_wave_level.py` | **6 passed** |
| `tests/sources/mode tests/rf/wave_validation/test_interior_pec_operator.py tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py` | **89 passed, 1 xfailed** |
| `tests/rf/antenna/test_antenna_benchmark_e2e.py` | **2 passed, 1 xfailed** (patch matched-broadside strict xfail stays red) |

The one non-antenna xfail is the pre-existing bent-mode symmetry xfail (unrelated).
`CAPABILITY_GUARD_BUDGET = 176` unchanged — every new guard is a `ValueError`
(routing/capability boundary), no new `NotImplementedError`; the guard census suite is
green.

### Measured numbers (reproducible from the pytest nodes / `benchmark.rf_validation` runners)

- **Microstrip** (dx = 5 mm, 0.6–1.6 GHz, `run_microstrip`): mode routes to
  `quasistatic_line_torch`; terminated two-port cond(A) 1.23, max singular value 1.09,
  `a_passive/a_driven` 0.16, `|S21|` 0.77–0.89, `|S11|` 0.05–0.24. Measured quasi-TEM
  `eps_eff` (median from `arg(S21)/L`) ≈ **1.86** vs Hammerstad–Jensen **3.27**
  (beta median rel error ≈ 24%).
- **Differential pair** (dx = 5 mm, 0.6–1.2 GHz, `run_differential_pair`): all four ports
  route to `quasistatic_line_torch`; single-ended 4-port cond(A) 1.31, max singular value
  1.18. Mixed-mode `|Sdd21|` ≈ 0.88 != `|Scc21|` ≈ 0.68 (even/odd coupling), non-zero
  single-ended `|S21|`, `|Sdc21|` ≈ 0 (mirror-symmetry — correct physics).
- **Quasi-static `eps_eff` convergence toward Hammerstad** (committed pytest node
  `tests/rf/wave_validation/test_interior_pec_operator.py::test_microstrip_eps_eff_converges_toward_hammerstad_with_resolution`,
  eps_r=4.4 / W/h=1.5 on a large box so finite-box truncation is small): substrate 4/8/16
  cells → `eps_eff` 3.4276 / 3.3304 / 3.2661, converging monotonically to Hammerstad–Jensen
  **3.2646** (<1% at 16 cells). This isolates the substrate-resolution effect and shows the
  quasi-static engine itself reaches the closed form. **Attribution correction:** the ~24%
  gap of the *shielded bench* microstrip is not substrate resolution alone — the bench's
  small shielded aperture also under-loads the field (aperture-shielding); the earlier
  shielded-box ladder (2.31→2.58→2.77→2.90) mixed those two effects and its low values were
  specific to that aperture. The correct diagnosis of the bench gap is aperture-shielding +
  resolution together, and the extraction itself is not the defect.

## 3. Falsifications performed (perturb → red → restore → green)

1. **The inhomogeneous → quasi-static routing is load-bearing.** With
   `_solve_quasistatic_line_modes` monkeypatched to raise, the microstrip WavePort mode
   solve goes RED (raises) instead of returning `quasistatic_line_torch`; restored → green
   (`scratch/falsify_routing.py`: baseline `quasistatic_line_torch` → perturbed
   `RuntimeError` → restored `quasistatic_line_torch`).
2. **The routing is content-dependent, not blanket** (built-in regression contrast):
   `test_uniform_fill_interior_pec_tem_keeps_the_closed_form_electrostatic_solve` shows a
   uniform (air) coax still selects `tem_electrostatic_torch` with `n_eff = 1`, so the
   quasi-static fallback fires only for the inhomogeneous fill the closed-form path
   rejects. Were the routing unconditional, this test goes red.
3. **The substrate is load-bearing for the slow wave** (F2a-carried, re-checked here):
   dropping the microstrip substrate to vacuum collapses the quasi-static `eps_eff` to 1.0
   (`test_interior_pec_operator.py::test_microstrip_without_dielectric_has_unit_eps_eff`).
4. **The differential drive distinguishes even/odd**
   (`test_tem_signal_potentials_maps_conductor_count_to_drive`): a two-signal aperture with
   the even drive `[1,1]` and the odd drive `[1,-1]` are the only accepted mode indices;
   mode_index 1 on a single-signal aperture raises.

## 4. Patch antenna feed — diagnosis and deferred redesign (fail-closed)

The stage brief (decision #4) asked for a patch redesign to a matched broadside `TM010`
(`D >= 5` dBi, `|Gamma|` dip at resonance) and to flip the strict xfail. **The patch does
not meet the gate; the xfail is NOT flipped** (flipping it would fake success). The
diagnosis is thorough and recorded:

- **Off-resonance drive.** The cavity-model resonance of the shipped geometry
  (eps_r=2.2, h=3 mm, L=28 mm, W=34 mm) is **3.39 GHz** (`cavity_resonance`), but the
  historical test/monitor band is 4.4–6.0 GHz — the patch was driven ~1.5x above
  resonance.
- **No galvanic coupling (root cause).** The retired feed drove the whole substrate gap
  with no metal probe, coupling only capacitively: `Re(Zin) ~ 0.3` Ohm with a huge
  (~700 Ohm) reactance. Adding a galvanic PEC probe via + single-cell lumped gap cut the
  reactance ~5x, but a wide-band sweep (2–8 GHz, `scratch` driver) shows **`Re(Zin) < 4`
  Ohm with NO resonance peak anywhere and a purely capacitive reactance** — the lumped gap
  is still capacitively shorted by the adjacent PEC, so the TM010 cavity mode is never
  excited and the radiated pattern is a broadside-**null** monopole from the z-via (peak D
  ~1.3–3.4 dBi off broadside, broadside D negative).
- **Finite ground too small.** The ground/substrate extend only ~6 cells (~0.07 lambda at
  3.4 GHz) beyond the patch — too small for a clean broadside patch even if it resonated.
- **Redesign path (deferred, multi-run antenna co-design).** A wire-bound clean-gap probe
  feed (the dipole's `WirePortBinding.nodes` pattern, so the gap is not PEC-shorted), a
  larger finite ground, and an on-resonance drive. This needs several long NF2FF
  validation runs and is deferred; the strict-xfail
  `test_patch_antenna_matched_broadside_gate` remains the fail-closed guard (it cannot
  silently xpass), reason string updated to this diagnosis. The
  `test_patch_antenna_result_antenna_pipeline_is_valid` pipeline gate stays green with the
  via feed (verified).

## 5. Files added / changed

- `witwin/maxwell/fdtd/excitation/modes.py`: `_tem_signal_potentials`; inhomogeneous-TEM →
  quasi-static fallback in `_assemble_vector_mode_data`.
- `benchmark/scenes/rf/microstrip_two_port.py`, `benchmark/scenes/rf/differential_pair.py`:
  rebuilt (ports near origin, through-PML conductors, integer-cell custom grid, half-grid
  contours).
- `benchmark/rf_validation.py`: `run_microstrip`, `run_differential_pair` run the real
  sweep and record honest measured status.
- `benchmark/scenes/antenna/patch.py`: galvanic probe via + single-cell lumped gap;
  docstring updated to the F2b diagnosis.
- `tests/rf/wave_validation/test_microstrip_diffpair_wave_level.py` (new, 6 gates).
- `tests/rf/antenna/test_antenna_benchmark_e2e.py`: strict-xfail reason updated (still red).
- `docs/reference/rf-wave-validation-2026-07-18.md`: §1/§1.3/§3/§5 rows updated honestly.
- `FEATURE_LIST.md`: additive `f2b-quasistatic-benches` subsection.
- `docs/assessments/f2b-quasistatic-benches-acceptance-2026-07-21.md`: this document.

## 6. Known gaps / handoff to F2c

- **Absolute quasi-TEM `eps_eff` accuracy** for microstrip/diff-pair is resolution-limited
  (~24% low at dx = 5 mm; converges with aperture resolution). The benches are recorded as
  `gap`, not forced. A finer dx or a subpixel-aware capacitance would close it but costs a
  much larger 3D grid.
- **Guided (non-TEM) interior-PEC path is not wired into production** (decision #2 partial).
  Only the quasi-TEM half was routed: an inhomogeneous-with-interior-PEC quasi-TEM request
  falls through to the quasi-static engine, but the *guided* interior-PEC branch of
  `_assemble_vector_mode_data` (`elif has_interior_pec`) still routes to the legacy
  `_pec_vector_operator_torch`; the new `_solve_yee_transverse_pec_mode` is exercised only by
  the F2a operator tests, not by any production scene. F2a's handoff flagged this; it is
  recorded here explicitly. No production scene currently needs the masked guided path, and
  wiring it would also require resolving the surface-sample rasterization tradeoff noted in
  the F2a acceptance §3 (a guided interior-PEC mode with nonzero normal `E` at the conductor
  surface is affected by the threshold-0.5 masking). Deferred pending a scene that needs it.
- **Adapter port/lumped source mapping + 4 cloud caches + RESULTS rows** (decision #3) are
  F2c. The microstrip/diff-pair external cross-check remains `pending-generation`.
- **Patch matched-broadside redesign** (§4) is deferred as a multi-run antenna co-design;
  the strict xfail is the fail-closed guard. The external patch cache (F2c) can seed the
  redesign target.

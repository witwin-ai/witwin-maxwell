# E4a acceptance — cascade/termination helpers, raw-sample cross-check, passivity suite

Track: `e4-network-gaps` (Plan 03 E2-evidence gaps), stage E4a.
Worktree: `.worktrees/we4-network-gaps`, branch `fable/network-e2`, base master `663f9aa`.
GPU: `CUDA_VISIBLE_DEVICES=1` (correctness only; no timing measurements — out of scope this round).
Env: conda `maxwell`, `CUDA_HOME=.../nvidia/cu13`, `PYTHONPATH=<worktree>`.

## Delivered items (E4a scope: deliverables 1, 2, 3)

1. **Public cascade/termination helpers** (`witwin/maxwell/network.py`).
   - `NetworkData.cascade(other, port_map=...)`: general N-port star connection from
     first principles, `S'_EE = S_EE + S_EC · P · (I − S_CC · P)^-1 · S_CE`, batched over
     frequency, differentiable, no third-party dependency. Fails closed on complex/mismatched
     connected reference impedance, frequency-grid mismatch, duplicate connections, empty map,
     duplicate result port names, and fully-connected (zero external port) results.
   - `NetworkData.terminate(port, gamma=... | impedance=...)`: single-port closure,
     `S'_KK = S_KK + S_Kp · Γ · (1 − S_pp · Γ)^-1 · S_pK`; `impedance` maps via
     `Γ = (Z − conj(z0))/(Z + z0)`. Fails closed on ambiguous/omitted argument, singular
     `(1 − S_pp Γ)`, and incomplete excitation columns.
   - `NetworkData.renormalize(...)` already existed (checked; unchanged).
   - Module helpers added: `_resolve_single_port`, `_require_real_reference`,
     `_connect_s_matrix`.

2. **Independent raw-sample S-cascade cross-check**
   (`tests/rf/network/test_network_cascade_crosscheck.py`). Breaks the fit-model-class
   circularity: a bare three-port FDTD device S is measured by a port sweep, then a network's
   **raw Touchstone samples** (read directly with `from_touchstone`, NOT the rational fit) are
   connected across two device ports with `NetworkData.cascade`; the resulting input reflection
   is compared to the same network **embedded in the time domain** (rational fit + state-space
   stepping). **Code-path independence:** the reference path (raw sampled S + first-principles
   connection algebra in `network.py`) shares no code with the embedded path (rational fit in
   `rational.py` + state-space time stepping in `fdtd/networks.py`). Two networks: a lossy
   attenuator-like flat conductance and a reactive single-pole RC.

3. **Multi-scenario passivity/conservation suite**
   (`tests/rf/network/test_network_conservation.py`). Three embedded scenarios (lossy 2-port,
   reactive 2-port, 4-port), each real FDTD run gated on:
   - (a) **Terminal power-balance magnitude** — `P1 = Σ 0.5 Re(V·conj(I))` (field-solve voltage,
     network-solve current) vs `P2 = Σ 0.5 Re(V^H · Y(ω) · V)` (field-solve voltage, model
     admittance). **Gate class (honest):** *consistency* for the two memoryless scenarios and
     *genuine two-sided* for the reactive scenario. For a memoryless network the embedding enforces
     the terminal law `I = Y·V` exactly, so `P1 == P2` is an algebraic identity (it confirms the
     solve applied `D` and DFT'd consistently, not an independent conservation law). For the reactive
     state-space scenario `I(t)` carries a dynamic history, so `DFT(I) = Y(ω)·DFT(V)` only holds once
     the finite analysis window has captured the LTI response — a transient/window bug breaks it —
     making that scenario a real balance. The only field-side records are `V` and `I` (linked by the
     model), so a model-independent dissipation estimate would need field-energy monitors and is
     deliberately not added; the conservation content lives in (b)/(c). Also `Σ|a|² ≥ Σ|b|²` is a
     separate non-tautological **sign** check (`Re(V·conj(I)) ≥ 0`, net power into the network — holds
     for any model value, fails for a non-passive embedding).
   - (b) **Passivity** — `generated_energy ≤ 1e-6 · absorbed_energy` and running cumulative net
     energy `≥ 0` sampled at run lengths spanning pulse rise/peak/ring-down (both energies are
     time-domain accumulations of per-step port V/I).
   - (c) **Stability** — between T and 2T the accumulated net energy converges (no divergence),
     the dynamic state norm rings down under the PML, and all diagnostics stay finite.

Deliverables 4 (fixed-cost reduction), 5 (delay adjoint), 6 (WavePort disposition) are E4b scope
and NOT touched here.

## Files added / changed

- `witwin/maxwell/network.py` — cascade/terminate helpers + module algebra helpers (+242 lines).
- `tests/rf/network/test_network_cascade.py` — new, 16 analytic-identity + fail-closed unit gates
  (includes the post-audit `TypeError`-ordering negative gate).
- `tests/rf/network/test_network_cascade_crosscheck.py` — new, 2 raw-sample cross-check gates (CUDA).
- `tests/rf/network/test_network_conservation.py` — new, 7 passivity/conservation gates (CUDA).
- `FEATURE_LIST.md` — additive subsection `e4a-network-cascade`.
- `docs/assessments/e4a-network-cascade-acceptance-2026-07-19.md` — this doc.

## Test inventory and pass counts

Commands (env exports as above):

```
python -m pytest tests/rf/network/test_network_cascade.py -q            # 16 passed
python -m pytest tests/rf/network/test_network_cascade_crosscheck.py \
                 tests/rf/network/test_network_conservation.py -q       # 9 passed (44.8s)
python -m pytest tests/rf/network/ -q                                   # 182 passed (32.4s)
python -m pytest tests/rf/network/test_network_cascade.py \
  tests/rf/network/test_network_cascade_crosscheck.py \
  tests/rf/network/test_network_conservation.py \
  tests/rf/network/test_network_algebra.py \
  tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py -q                          # 63 passed (51.8s)
```

New tests: 26 total (16 cascade unit + 2 cross-check + 7 conservation + 1 op-stream gain_voltage-path
gate; the op-stream file also carries the pre-existing 2 composite-solve gates). The post-audit
additions are the cascade `TypeError`-ordering negative gate and the op-stream gain_voltage-path
no-regression gate.

## Pre-registered tolerances and observed margins

Emitter for the observed margins below (they are otherwise not reproducible from a committed
artifact): `docs/assessments/e4-network-e2-probes/cascade_margins_probe.py`. It reuses the test
fixtures (bare sweep, Touchstone round-trip, cascade algebra, embedding, conservation scenarios) so
the printed numbers cannot drift from the asserted gates. Run from the worktree root with the env
exports above:

```
CUDA_VISIBLE_DEVICES=1 python docs/assessments/e4-network-e2-probes/cascade_margins_probe.py
```

Observed margins below are the values printed by that probe on this GPU; the asserted tolerances are
the committed thresholds in the corresponding test nodes.

- **Cross-check** `|ΔS| < 1e-5` across the band. Observed residuals (probe): lossy ~4.6e-9, reactive
  ~5.7e-8 (≥170× margin). The connected network changes the input reflection by ~3.9e-4 (lossy) /
  ~4.8e-3 (reactive) — well above the in-test `network_effect > 10·tol` requirement, so the gate
  provably exercises the connection algebra. Bare device S is passive (`svd max ≤ 1.01`). Grid
  `dx=5mm`, PML 8 layers, 2000 steps, adjacent ports (±5mm) for strong coupling.
- **Power balance** `|P1−P2|/|P1| < 1e-3`. Observed (probe) ≤~1.1e-5 (max over scenarios; reactive is
  the largest, the two memoryless are ~1.6e-7).
- **Passivity** `generated_energy ≤ 1e-6·absorbed_energy` (probe: 0 for the memoryless scenarios,
  ~1.8e-10 relative for reactive), cumulative net `≥ −1e-15`.
- **Stability** `|net(2T)−net(T)| ≤ 1e-3·net(T)` (observed ~0, energy converges by T), reactive
  state norm rings down (probe) ~5.8e-6→~1.5e-6 over T→4T.

(Exact last-digit values shift slightly run-to-run on a shared GPU; the probe reprints them and the
committed thresholds — not the observed values — are what the gates assert.)

## Falsifications recorded

Each load-bearing gate was broken, observed red, restored, and re-verified green. No falsification
residue remains (`grep -rn FALSIFY witwin/` clean).

- **F1 — cascade connection sign.** In `_connect_s_matrix`, `s_ee + s_ec@P@solved` → `s_ee - …`.
  `test_cascade_through_matched_thru_returns_identity` (rel diff 2.0) and
  `test_cascade_of_two_attenuators_adds_attenuation_in_db` both FAILED. Restored → 15 passed.
- **F2 — termination scale sign.** In `terminate`, `scale = (reflection/denom)…` →
  `(-reflection/denom)…`. `test_terminate_two_port_matches_closed_form_input_reflection` FAILED
  (abs diff 0.213). Restored → 15 passed.
- **F3 — cross-check via cascade sign.** Same F1 break; the raw-sample cross-check
  `test_raw_sample_cascade_matches_embedded_run[lossy_attenuator-…]` FAILED (residual jumps to
  ~2·network_effect ≫ 1e-5). Restored.
- **F4 — embedded network current.** In `apply_network_runtime`, injected
  `runtime.branch_current.mul_(1.05)` after the solve. Terminal power balance
  `test_embedded_network_power_balance_and_passivity[lossy]` FAILED (5% P1/P2 mismatch ≫ 1e-3).
  Restored → clean.

## Post-audit fix falsifications (2026-07-19)

Supervisor-selected minor fixes after both audits passed; each new/tightened gate re-falsified.

- **F5 — cascade error-type ordering.** `NetworkData.cascade` now runs the `isinstance(other,
  NetworkData)` check before `other._require_complete(...)`, so a non-NetworkData `other` raises the
  intended `TypeError` rather than `AttributeError`. New gate
  `test_cascade_rejects_non_network_other_with_type_error` passes on the fix. Falsification: reverting
  the two lines (completeness checks first) turned the gate RED — with an incomplete `self` the call
  raised `RuntimeError: cascade requires complete excitation columns` instead of `TypeError`. Restored
  → green.

- **F6 — op-stream gain_voltage-path blindness.** The combined no-regression gate
  (`test_composite_solve_matches_legacy_lu_no_regression`) stays green under a 5% `gain_voltage`
  corruption because its roundoff bound is dominated by the ill-scaled `C@state` matvec. Added a
  dedicated path-specific gate `test_composite_solve_matches_legacy_lu_gain_voltage_path` (support
  helper `gain_voltage_path_equivalence`) that holds `state=0`, removing the `C` term from both the
  compared value and the bound. Falsification: corrupting the source
  (`gain_voltage = 1.05 * lu_solve(...)` in `fdtd/networks.py`) left the combined gate GREEN (passed)
  while the new gate went RED (residual/bound ratio 8.24e+03 ≫ 1). Restored → both green.

## Capability-guard census

Budget unchanged at **176** (`tests/api/public/test_guard_census.py` passes in every run above).
The new fail-closed `ValueError`/`RuntimeError` checks in `cascade`/`terminate` are input-validation
raises on a public algebra method, not FDTD capability-rejection guards tracked by the census; no
census-tracked guard was added or removed, so no reconciliation was required.

## Design notes / decisions

- **Cross-check bench is a lumped three-port (accepted deviation from the brief).** The brief's
  deliverable 2 suggested a coax two-port bench. This cross-check instead uses a lumped-port
  three-port device (`d0`/`d1`/`d2` `LumpedPort`s on a uniform grid), connecting the network across
  `d1`/`d2` and reading the input reflection at the free port `d0`. This is a deliberate, accepted
  substitution: the independence requirements the brief actually demands are met — the reference path
  (raw Touchstone samples read via `from_touchstone` + the first-principles `cascade` algebra in
  `network.py`) shares no code with the embedded path (rational fit in `rational.py` + state-space
  time stepping in `fdtd/networks.py`), two network classes are exercised (lossy flat conductance and
  reactive single-pole RC), and the connection changes S11 by ~2e-4 (≫ the 1e-5 tolerance, asserted
  in-test). The lumped three-port is cheaper to set up on the existing uniform-grid lumped-port
  infrastructure and gives the same code-path-independence guarantee a coax bench would.

- **Wave convention.** The connection relation `a_k = b_l` is exact for real reference impedances
  (traveling-wave S). Connected ports are required real; complex `z0` fails closed and must be
  renormalized first. Terminations accept complex load `Z` via the Kurokawa reflection form.
- **Cross-check strength.** Ports are placed adjacent (strong coupling) so the network's effect on
  S11 (~2e-4) sits well above the tolerance; the port sweep must be the *only* excitation (a stray
  interior source floods every port and corrupts the S extraction — the bare device S must be
  source-free). Network models are chosen near-exactly fittable so the residual isolates
  algebra+embedding consistency from fit error; the reference path's independence is in the code
  path (raw samples + offline algebra), not the model class.
- **Passivity "at all times".** The runtime exposes final time-domain `absorbed_energy` /
  `generated_energy` (per-step V/I accumulation), not the running minimum. The running cumulative is
  therefore sampled at several increasing run lengths (100/200/400/800 steps) rather than
  instrumenting the hot path — this avoids adding per-step ops that would conflict with E4b's
  hot-path reduction.

## Known gaps / handoff to E4b

- No runtime hot-path changes were made (deliverable 4 is E4b). The passivity running-integral is
  sampled via multiple runs rather than a new per-step accumulator, deliberately, to keep the E4b
  op-stream baseline clean.
- Delay adjoint (deliverable 5) and WavePort disposition (deliverable 6) untouched.
- `NetworkData.cascade`/`terminate` are single-device, differentiable public helpers; no distributed
  path was added.

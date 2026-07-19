# E4a acceptance — cascade/termination helpers, raw-sample cross-check, passivity suite

Track: `e4-network-gaps` (Plan 03 E2-evidence gaps), stage E4a.
Worktree: `.worktrees/we4-network-gaps`, branch `fable/network-e2`, base master `039cead`.
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
   - (a) **Terminal power balance** — `P1 = Σ 0.5 Re(V·conj(I))` (field-solve voltage, network-solve
     current) vs `P2 = Σ 0.5 Re(V^H · Y(ω) · V)` (field-solve voltage, model admittance). Independent
     computations; agreement is conservation, not identity. Also `Σ|a|² ≥ Σ|b|²` (passive sink).
   - (b) **Passivity** — `generated_energy ≤ 1e-6 · absorbed_energy` and running cumulative net
     energy `≥ 0` sampled at run lengths spanning pulse rise/peak/ring-down (both energies are
     time-domain accumulations of per-step port V/I).
   - (c) **Stability** — between T and 2T the accumulated net energy converges (no divergence),
     the dynamic state norm rings down under the PML, and all diagnostics stay finite.

Deliverables 4 (fixed-cost reduction), 5 (delay adjoint), 6 (WavePort disposition) are E4b scope
and NOT touched here.

## Files added / changed

- `witwin/maxwell/network.py` — cascade/terminate helpers + module algebra helpers (+242 lines).
- `tests/rf/network/test_network_cascade.py` — new, 15 analytic-identity + fail-closed unit gates.
- `tests/rf/network/test_network_cascade_crosscheck.py` — new, 2 raw-sample cross-check gates (CUDA).
- `tests/rf/network/test_network_conservation.py` — new, 7 passivity/conservation gates (CUDA).
- `FEATURE_LIST.md` — additive subsection `e4a-network-cascade`.
- `docs/assessments/e4a-network-cascade-acceptance-2026-07-19.md` — this doc.

## Test inventory and pass counts

Commands (env exports as above):

```
python -m pytest tests/rf/network/test_network_cascade.py -q            # 15 passed
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

New tests: 24 total (15 unit + 2 cross-check + 7 conservation).

## Pre-registered tolerances and observed margins

- **Cross-check** `|ΔS| < 1e-5` across the band. Observed residuals: lossy ~8e-9, reactive ~7e-8
  (≥140× margin). The connected network changes the input reflection by ~2e-4 (≥16× the tolerance),
  asserted in-test (`network_effect > 10·tol`) so the gate provably exercises the connection algebra.
  Bare device S is passive (`svd max ≤ 1.01`). Grid `dx=5mm`, PML 8 layers, 2000 steps, adjacent
  ports (±5mm) for strong coupling.
- **Power balance** `|P1−P2|/|P1| < 1e-3`. Observed ≤1.3e-5.
- **Passivity** `generated_energy ≤ 1e-6·absorbed_energy` (observed 0 to ~1e-25 relative ~1e-10),
  cumulative net `≥ −1e-15`.
- **Stability** `|net(2T)−net(T)| ≤ 1e-3·net(T)` (observed ~0, energy converges by T), reactive
  state norm rings down 4.6e-6→5.9e-7 over T→4T.

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

## Capability-guard census

Budget unchanged at **176** (`tests/api/public/test_guard_census.py` passes in every run above).
The new fail-closed `ValueError`/`RuntimeError` checks in `cascade`/`terminate` are input-validation
raises on a public algebra method, not FDTD capability-rejection guards tracked by the census; no
census-tracked guard was added or removed, so no reconciliation was required.

## Design notes / decisions

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

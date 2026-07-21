# E1b acceptance — Yee-staggered transverse operator wired into the selector

> Date: 2026-07-19
> Track: e1-rf-modes (branch `fable/rf-mode-operator`)
> Stage E1b: selector / WavePort integration, flip the six strict-xfail TE10 pins,
> inhomogeneous (hybrid) operator validation, retire the defective uniform branch
> from production. Builds on E1a
> (`docs/assessments/e1-rf-modes-acceptance-2026-07-19.md`, operator + golden
> spectrum gates).
>
> Re-audit fix round (2026-07-20): added the load-bearing selector-path pin
> `test_uniform_dielectric_fill_selector_path_carries_filled_beta` for the
> uniform-dielectric routing physics (previously unpinned); rewrote falsification
> record #3 truthfully (the prior record described a revert that could not fail the
> named test); corrected the retracted-spectrum parenthetical in §3 to the regenerated
> symmetrized numbers; and committed the probe scripts the doc cites under
> `docs/assessments/e1-rf-mode-operator-probes/`. No production-code change this round —
> `witwin/maxwell/fdtd/excitation/modes.py` is unchanged; the routing was already
> correct, only the evidence was.

## 1. Delivered

1. **Homogeneous non-magnetic full-vector modes solve on the Yee-staggered operator
   end to end.** A new solve+select+reconstruct path
   (`_solve_yee_transverse_vector_mode` / `_select_yee_transverse_mode_numpy` /
   `_yee_reconstruct_node_profiles` in `witwin/maxwell/fdtd/excitation/modes.py`)
   is wired into `_assemble_vector_mode_data`. It:
   - interpolates the aperture node permittivity onto the three staggered Yee
     component grids (`_yee_stagger_eps_from_nodes`),
   - builds `P et = beta**2 et` via the E1a builder,
   - solves it (dense `eigh` / sparse `eigsh` for the symmetric homogeneous case),
   - reconstructs the longitudinal `Ew` (Gauss law) and transverse `Hu, Hv` (discrete
     curl-E, real arithmetic) and interpolates every component onto the common
     aperture node grid,
   - selects the requested forward mode with the same hardened filters as the legacy
     selector (transverse-null-space `beta -> k0` rejection for guided requests,
     checkerboard/duplicate diagnostics on the structure-enforcing path, forward-power
     gate, requested-polarization family, degenerate-subspace SVD + polarization
     rotation, and a fail-closed raise when the requested family index is absent).
   The `WaveModeSpec` / `WavePort` / `ModeSource` public API is unchanged; there is no
   second entrypoint.
2. **The six strict-xfail TE10 selector pins are now passing real asserts**
   (`tests/rf/wave_validation/test_te10_mode_selection.py`): five `dx` tiers of
   `test_waveguide_te10_eigenvector_is_sin` and
   `test_waveguide_te10_high_frequency_returns_genuine_te10`. The former
   `..._is_checkerboard_contaminated_operator_blocker` (which asserted `corr < 0.9` as
   the defect fingerprint) is repurposed to `..._operator_returns_clean_sin` asserting
   `corr >= 0.99`. The module docstring is updated to record the fix.
3. **Inhomogeneous (hybrid) operator validation** — three new gates in
   `tests/rf/wave_validation/test_transverse_operator.py` exercise the per-component-eps
   path against a half-filled parallel-plate guide (`eps1` for `u < d`, `eps2` for
   `u > d`): machine-precision agreement with an independently assembled 1D discrete
   `LSE` Sturm-Liouville operator, the physical mode structure (uniform along the
   plates, `Ev`-polarized, concentrated in the high-permittivity half), and continuum
   convergence to the analytic transverse-resonance root. Scope of what these gate: the
   `LSE` mode is `v`-uniform, so `Gv @ Ev = 0` and the `eps_ww` divergence coupling and
   the cross blocks `P_uv/P_vu` vanish identically for it — the gates therefore pin the
   `eps_vv` placement and the `u`-Laplacian block, not the full inhomogeneous stencil.
   The complete operator (all three eps components, cross terms, `eps_ww` divergence) was
   independently checked against the discrete Yee Maxwell curl system to `~6e-14` in the
   audit; that broader check is not yet a committed pytest node.
4. **FEATURE_LIST.md** gains an additive `e1-rf-modes-b` subsection.

## 2. Grid / reconstruction (integration derivation)

Aperture node grid `(nu_nodes, nv_nodes)`, `nu_cells = nu_nodes - 1`. The eigenvector
`et = (Eu, Ev)` lives on the staggered grids (`Eu` on `u`-half / `v`-interior-node,
`Ev` on `u`-interior-node / `v`-half). With `Eu, Ev` real, `Ew` is 90° out of phase
(`Ew = i * ew`, `ew` real):

    ew_node = -(d/du(eps_u Eu) + d/dv(eps_v Ev)) / (beta * eps_w)        (Gauss)
    Hu = -(beta Ev + d/dv ew) / (k0 * eta0)      (on the Ev grid)
    Hv =  (beta Eu + d/du ew) / (k0 * eta0)       (on the Eu grid)

`omega mu0 = k0 * eta0`. The two `i` factors of the H-curls cancel `Ew`'s phase, so
`Hu, Hv` are real. For `TE10` (`ew = 0`) this reduces to `Hu = -beta Ev / (k0 eta0)`,
i.e. the exact analytic wave impedance `Z_TE10 = eta0 k0 / beta`, so the reconstructed
power normalization reproduces the analytic modal power. Interpolation to the node
grid: a component tangential to a wall is Dirichlet-zero on that wall; the half-grid
axis maps to nodes by the Yee average with a zeroth-order Neumann extrapolation at the
walls (`_yee_half_to_node_neumann`, `_yee_interior_to_node_dirichlet`). Longitudinal
`Ew`/`Hw` are stored as zero on the node grid (matching the legacy contract; they are
not consumed by the power integrator, tracking basis, or TFSF injection for the
non-magnetic families in scope).

## 3. Scope decision — routing and the retained legacy operator

The Yee-staggered operator is the production path for **homogeneous non-magnetic**
cross-sections — routed on `_is_uniform_isotropic_vector_plane(...)` true **and**
`mu = 1` everywhere. This covers the hollow metallic guide (the six pins), a
**uniformly dielectric-filled** guide (any uniform `eps_r` with `mu = 1`), and
free-space `WavePort` apertures. **Inhomogeneous** (dielectric-graded) and **magnetic**
(`mu_r != 1`, uniform or graded) cross-sections keep the legacy diagonal-anisotropic
operator (`_build_vector_operator_sparse`). This is a regime split, not a duplicate:

- **Uniform dielectric fill carries its real eps.** For a homogeneous cross-section the
  Yee operator with the aperture's actual per-component eps equals the vacuum operator
  plus a scalar `(eps_r - 1) * k0**2` identity shift — it stays exactly symmetric and
  every eigenvalue is its vacuum counterpart shifted by `(eps_r - 1) * k0**2`, so the
  fundamental sits at `beta**2 = eps_r * k0**2 - k~c**2`.
  `_solve_yee_transverse_vector_mode` therefore always assembles the operator from the
  real staggered eps; the `uniform` flag only selects the symmetric eigensolve and
  disables the structure-enforcing filters, it does not mean vacuum. EXECUTED through
  the selector entry `_solve_yee_transverse_vector_mode` on the `a = 1.0`, `b = 0.6`,
  `k0 = 10.0` hollow guide (`docs/assessments/e1-rf-mode-operator-probes/repro_uniform_eps.py`):
  `eps_r = 1.0 / 2.25 / 4.0` gives TE10 `beta**2 = 90.1445 / 215.1445 / 390.1445`,
  each equal to the discrete analytic `eps_r*k0**2 - k~x(1)**2` to `rel <= 4e-15` with
  `sin`-correlation `1.000000`; the vacuum-defect value the routing must NOT return is
  `90.1445`. Gated at the operator level by
  `test_uniform_dielectric_fill_shifts_spectrum_and_stays_symmetric` and, load-bearing,
  through the selector routing function by
  `test_uniform_dielectric_fill_selector_path_carries_filled_beta` (both in
  `tests/rf/wave_validation/test_transverse_operator.py`), and at the WavePort-manifest
  integration level by `test_waveguide_te10_beta_matches_analytic`
  (`tests/rf/wave_validation/test_te10_mode_selection.py`).
- **Magnetic (`mu != 1`)** is out of the E1 redesign scope by construction — the
  Yee-staggered derivation eliminates `Hz` assuming `mu = 1` (E1a §8). A **uniformly**
  magnetic aperture also classifies as uniform-isotropic, so the routing explicitly
  requires `mu = 1` and sends every magnetic cross-section (uniform or graded) to the
  legacy operator, which threads `mu` through the eliminated longitudinal fields
  (EXECUTED end to end through `solve_mode_source_profile`,
  `docs/assessments/e1-rf-mode-operator-probes/repro_uniform_mu_e2e.py`, on a uniform
  `eps_r = 4` guide: `mu_r = 1` routes to the Yee operator and returns `beta = 41.6322`,
  while `mu_r = 2` routes to the legacy operator and returns the `mu`-dependent
  `beta = 59.2794` — scaling with `sqrt(eps_r*mu_r)`, NOT collapsing onto the `mu = 1`
  value `41.6322`. Both dense paths carry the same `mode_solver_kind = vector_dense`
  label, so the routing is evidenced by the `mu`-dependent `beta`, not the label). This
  preserves `test_full_vector_mode_solver_supports_mu_not_unity`.
- **Inhomogeneous production integration is deferred, on verified grounds.** The
  inhomogeneous operator is real-valued in storage but **non-symmetric**; its spectrum,
  however, is **entirely real and bounded by the physical mode** — for the half-filled
  `LSE` gate a dense `numpy.linalg.eigvals` gives `max Re(beta**2) = 224.2234 /
  225.0614 / 225.5435` with `max|Im| = 0` at `nu = 24 / 48 / 96`, i.e. the physical
  `LSE` mode IS the spectral maximum and `eigs(which='LR')` returns the physical branch
  cleanly (EXECUTED,
  `docs/assessments/e1-rf-mode-operator-probes/verify_spurious_spectrum.py`). (An
  earlier revision of this doc claimed spurious `beta**2 = 785 / 2746 / 10600` maxima
  growing as `~1/dx^2` and attributed them to `eigvalsh` of the *symmetrized* operator;
  that specific triple does NOT reproduce from any probe and is withdrawn. The same
  probe does show the symmetrized surrogate `0.5*(P + P^T)` growing with `1/dx` —
  `eigvalsh` max `= 224.2234 / 381.6176 / 1219.2024` at `nu = 24 / 48 / 96` — but that
  is a symmetrization artifact, not the true spectrum, whose real maximum is the
  physical `225`-region mode above.) The genuine, verified reason to
  defer is **test-migration risk**: the existing inhomogeneous unit tests
  (`tests/sources/mode/test_mode_eigensolver_physics.py`) pin legacy-operator-specific
  invariants — candidate power-orthogonality to `1e-6`, `raw_indices` grouping, and
  three-grid mode-index ordering — that the node-grid reconstruction path does not
  reproduce to those tolerances. Rerouting the inhomogeneous non-magnetic path through
  the Yee operator (EXECUTED, temporary reroute) turns **3 of the 9**
  `test_mode_eigensolver_physics.py` tests RED
  (`test_higher_order_mode_converges_on_three_grids_without_index_changes`,
  `test_square_degenerate_subspace_rotates_to_stable_requested_polarizations`,
  `test_candidate_power_gram_and_discrete_divergence_are_orthogonal` — the last with an
  off-diagonal power-overlap of `7.5e-3` vs the `1e-6` gate). Migrating those pins is a
  supervisor-gated decision, so inhomogeneous stays on the legacy operator and the
  **operator-level** hybrid capability is exercised by the new half-filled gates
  instead.
- **Microstrip / differential-pair production hybrid gates are deferred** (fail-closed
  unchanged). Beyond the test-migration above, these carry an **interior PEC
  conductor** in the aperture (the signal strip/ground), which the Yee-staggered
  operator does not yet mask; the existing `NotImplementedError`
  (`_solve_pec_tem_mode_torch`, inhomogeneous TEM cross-section) remains the
  fail-closed behavior. The half-filled `LSE` gate validates the inhomogeneous
  per-component-eps physics that a future interior-PEC + spurious-filtered integration
  will build on.

No capability-guard census change: the integration adds no `NotImplementedError`
guard, and the microstrip TEM-inapplicability guard is unchanged
(`CAPABILITY_GUARD_BUDGET = 176`, `test_guard_census.py` green).

## 4. Test inventory (commands)

Environment:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/we1-rf-modes
export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

| target | result |
|---|---|
| `tests/rf/wave_validation/test_transverse_operator.py` | **13 passed** (8 E1a + 3 inhomogeneous + 1 operator-level uniform-fill regression + 1 selector-path uniform-fill pin) |
| `tests/rf/wave_validation/test_te10_mode_selection.py` | all pass incl. the six former xfails and 3 new `beta`-vs-analytic tiers |
| `tests/rf/wave_validation tests/rf/waveport tests/sources/mode tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py` | **176 passed, 2 xfailed** |

The two remaining xfails are pre-existing and unrelated: `test_wave_level_rlc_resonance_open_gap`
(S1.2 open gap) and `test_bent_effective_index_is_symmetric_in_radius_sign_for_a_symmetric_guide`
(the bent-mode conformal-eps path is inhomogeneous and stays on the legacy operator).

Measured TE10 acceptance (`sin`-correlation, EXECUTED via the WavePort manifest):
`dx = 0.05 / 0.025 / 0.02 / 0.0125 / 0.01 -> corr 1.00000` (all); high-frequency
`dx = 0.005, 6 fc -> corr 1.00000` (genuine TE10, not TE20). The `beta`-vs-analytic
`sqrt(k0^2 - (pi/a)^2)` acceptance is now a committed pytest node
(`test_waveguide_te10_beta_matches_analytic`, error `0.127% / 0.020% / 0.008%` at
`dx = 0.05 / 0.02 / 0.0125`, gated `<= 0.15% / 0.05% / 0.02%`). Operator-level uniform-fill
regression (`test_uniform_dielectric_fill_shifts_spectrum_and_stays_symmetric`): `eps_r = 4`
guide `beta**2` equals the vacuum value plus `(eps_r - 1) k0**2` to `1e-12`, operator
symmetric to `1e-9`. Selector-path uniform-fill pin
(`test_uniform_dielectric_fill_selector_path_carries_filled_beta`, load-bearing through
`_solve_yee_transverse_vector_mode`): `eps_r = 2.25` guide TE10 `beta**2 = 215.1445`
matches the discrete analytic to `rel <= 1e-10`, `Ey` `sin`-correlation `1.000000`, and
must not collapse onto the vacuum `90.1445` (see falsification #3). Inhomogeneous half-filled `LSE`: 2D operator vs 1D discrete reference agrees
to `rtol ~1e-14` (only the `nu = 48` tier is test-gated; the exact residual is
ARPACK-seed dependent at the `1e-15`–`1e-13` level, so the specific triple is not pinned
as an acceptance number); continuum transverse-resonance error `0.41% / 0.22% / 0.11%`
(first-order, material discontinuity on a node), fundamental analytic `beta = 15.0354`.

## 5. Falsifications performed (perturb → red → restore → green)

1. **Integration is load-bearing (headline TE10 gate).** Forcing the uniform guide back
   through the legacy operator (`if uniform_isotropic and nonmagnetic:` → `if False:`)
   reintroduced the
   sublattice-decoupling defect: `test_waveguide_te10_eigenvector_is_sin[0.02]` and
   `..._operator_returns_clean_sin` went **RED** with `corr 0.554` (vs the `>= 0.99`
   gate). Restored → green.
2. **Inhomogeneous per-component-eps assembly is load-bearing.** Dropping `eps_vv`
   (the component the `LSE` mode sees) from the operator build in
   `_half_filled_operator` (`eps_vv=eps_vv` → `eps_vv=None`) drove
   `test_inhomogeneous_operator_spectrum_matches_one_dimensional_lse_reference` **RED**
   (2D eigenvalue no longer matches the independent 1D reference). Restored → green.

3. **The uniform-fill routing carries the real eps (re-audit fix — REWRITTEN).**
   The perturbation executed is in the routing function itself: in
   `_solve_yee_transverse_vector_mode` (`witwin/maxwell/fdtd/excitation/modes.py`),
   `operator_eps = (eps_uu, eps_vv, eps_ww)` → `operator_eps = (None, None, None)`
   (the vacuum-physics defect: `None` per-component eps means the builder assumes
   `eps = 1`). This drove the new selector-path pin
   `tests/rf/wave_validation/test_transverse_operator.py::
   test_uniform_dielectric_fill_selector_path_carries_filled_beta` **RED**, observed:

       E  AssertionError: filled TE10 beta**2 90.1444803026 != discrete analytic 215.1444803026
       E  assert 124.99999999999964 <= (1e-10 * 215.1444803026296)
       1 failed in 2.27s

   i.e. the `eps_r = 2.25` guide `beta**2` collapsed from the filled `215.1445` onto the
   vacuum value `90.1445`. Restored → **green** (`1 passed`). The companion probe
   `docs/assessments/e1-rf-mode-operator-probes/repro_uniform_eps.py` shows the same
   collapse target across fills (`eps_r = 1 / 2.25 / 4` → filled `beta**2 =
   90.1445 / 215.1445 / 390.1445`, all vs the single vacuum `90.1445`).

   *Correction to the previous revision:* the earlier record #3 claimed this same
   `operator_eps → (None, None, None)` revert turned
   `test_uniform_dielectric_fill_shifts_spectrum_and_stays_symmetric` red. That was
   **false** — that test builds its operator by calling the raw
   `_build_yee_transverse_operator_sparse` directly (via the module-local
   `_uniform_filled_operator`), so it never executes `_solve_yee_transverse_vector_mode`
   and the routing revert cannot affect it. The uniform-dielectric *routing* physics was
   therefore unpinned until the selector-path test above was added; that test is the
   genuine load-bearing pin.
4. **Uniform magnetic routes to legacy.** Dropping the `nonmagnetic`
   conjunct from the routing (`_assemble_vector_mode_data`,
   `if uniform_isotropic and nonmagnetic:` → `if uniform_isotropic:`) sends a uniform
   `mu_r = 2` guide back into the `mu = 1` Yee operator, returning the `mu`-independent
   `beta = 41.6322` (identical to the `mu_r = 1` value); with the conjunct it routes to
   the legacy operator and returns the `mu`-dependent `beta = 59.2794`. EXECUTED end to
   end through `solve_mode_source_profile` via
   `docs/assessments/e1-rf-mode-operator-probes/repro_uniform_mu_e2e.py` (perturbed vs
   restored). Restored → green.

(All code perturbations were reverted immediately; the working tree is clean. Every
quoted number regenerates from a committed command — the pytest node above or one of
the probe scripts under `docs/assessments/e1-rf-mode-operator-probes/`
[`repro_uniform_eps.py`, `verify_spurious_spectrum.py`, `repro_uniform_mu_e2e.py`].)

## 6. Files added / changed

- `witwin/maxwell/fdtd/excitation/modes.py`: added `_yee_stagger_eps_from_nodes`,
  `_yee_half_to_node_neumann`, `_yee_interior_to_node_dirichlet`,
  `_yee_reconstruct_node_profiles`, `_select_yee_transverse_mode_numpy`,
  `_solve_yee_transverse_vector_mode`; stashed the staggered differences in the E1a
  builder's `meta`; routed the homogeneous non-magnetic branch of
  `_assemble_vector_mode_data` to the new solver. Post-audit: the Yee solver always
  assembles from the real staggered eps (uniform dielectric fill no longer collapses to
  vacuum), and the routing requires `mu = 1` (uniform magnetic goes to the legacy
  operator).
- `tests/rf/wave_validation/test_te10_mode_selection.py`: un-xfailed the six pins,
  repurposed the blocker regression, updated the docstring; added
  `test_waveguide_te10_beta_matches_analytic` (3 tiers) pinning the eigenvalue.
- `tests/rf/wave_validation/test_transverse_operator.py`: three inhomogeneous
  half-filled `LSE` gates, the operator-level
  `test_uniform_dielectric_fill_shifts_spectrum_and_stays_symmetric`, and (re-audit fix)
  the selector-path
  `test_uniform_dielectric_fill_selector_path_carries_filled_beta` that pins the
  uniform-dielectric routing physics through `_solve_yee_transverse_vector_mode`.
- `docs/assessments/e1-rf-mode-operator-probes/` (`git add -f`): committed probe scripts
  cited by this doc — `repro_uniform_eps.py` (selector-path uniform-fill repro),
  `verify_spurious_spectrum.py` (inhomogeneous-spectrum real/bounded verification and the
  symmetrized-surrogate growth), `repro_uniform_mu_e2e.py` (uniform-magnetic routing,
  end to end).
- `FEATURE_LIST.md`: additive `e1-rf-modes-b` subsection (unchanged this round; the
  uniform-dielectric-fill behavior it documents is now test-pinned).
- `docs/assessments/e1-rf-mode-operator-acceptance-2026-07-19.md`: this document.

## 7. Known gaps / handoff

- **Inhomogeneous / hybrid production integration is deferred** (see §3): needs a
  staggered-grid power-orthogonal reconstruction that reproduces the legacy candidate
  invariants, and migration of the **3** legacy-invariant
  `test_mode_eigensolver_physics.py` pins that a direct reroute turns red. (The
  inhomogeneous operator's spectrum is real and physical-mode-bounded; no
  spurious-high-`|beta|` filter is required — see the §3 retraction.)
- **Microstrip / differential-pair (interior-PEC) hybrid gates deferred**: additionally
  need interior-PEC masking on the staggered operator. Fail-closed behavior
  (`NotImplementedError` on an inhomogeneous TEM cross-section) is unchanged.
- **Bent (conformal-eps) mode source** rides the inhomogeneous legacy path; its
  order-`1e-6` mirror-symmetry xfail would likely close once the component-staggered
  operator serves the inhomogeneous path — a bonus for the deferred integration.
- **Legacy `_build_vector_operator_sparse` retained** as the inhomogeneous + magnetic
  regime operator (not a duplicate); the uniform non-magnetic production case no longer
  reaches its defective centered branch. A **uniform magnetic** aperture does still hit
  the legacy centered (uniform-isotropic) branch, which carries the correct eigenvalue
  but the known checkerboard eigenvector — acceptable within the out-of-scope magnetic
  regime, and superseded whenever the magnetic Yee derivation lands.

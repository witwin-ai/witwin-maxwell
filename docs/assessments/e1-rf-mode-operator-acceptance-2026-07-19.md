# E1b acceptance — Yee-staggered transverse operator wired into the selector

> Date: 2026-07-19
> Track: e1-rf-modes (branch `fable/rf-mode-operator`)
> Stage E1b: selector / WavePort integration, flip the six strict-xfail TE10 pins,
> inhomogeneous (hybrid) operator validation, retire the defective uniform branch
> from production. Builds on E1a
> (`docs/assessments/e1-rf-modes-acceptance-2026-07-19.md`, operator + golden
> spectrum gates).

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
   `tests/rf/wave_validation/test_transverse_operator.py` prove the per-component-eps
   path against a half-filled parallel-plate guide (`eps1` for `u < d`, `eps2` for
   `u > d`): machine-precision agreement with an independently assembled 1D discrete
   `LSE` Sturm-Liouville operator, the physical mode structure (uniform along the
   plates, `Ev`-polarized, concentrated in the high-permittivity half), and continuum
   convergence to the analytic transverse-resonance root.
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
cross-sections (`_is_uniform_isotropic_vector_plane` true): the hollow metallic guide
(the six pins) and free-space `WavePort` apertures. **Inhomogeneous** (dielectric-graded)
and **magnetic** (`mu_r != 1`) cross-sections keep the legacy diagonal-anisotropic
operator (`_build_vector_operator_sparse`). This is a regime split, not a duplicate:

- **Magnetic (`mu != 1`)** is out of the E1 redesign scope by construction — the
  Yee-staggered derivation eliminates `Hz` assuming `mu = 1` (E1a §8). Retaining the
  legacy operator preserves `test_full_vector_mode_solver_supports_mu_not_unity`.
- **Inhomogeneous production integration is deferred, with measured cause.** The
  inhomogeneous operator is real but non-symmetric and carries **spurious high-`|beta|`
  eigenvalues above the physical spectrum** (EXECUTED: for the half-filled guide the
  spectral maximum grows as `~1/dx^2` — `beta^2 = 785 / 2746 / 10600` at
  `nu = 24 / 48 / 96` — while the physical `LSE` mode sits at `beta^2 ≈ 225`). A
  `which='LA'` selection would pick a spurious mode, so a production inhomogeneous path
  needs a spurious-mode filter and a power-orthogonality reconstruction on the
  staggered (not node) grid. The existing inhomogeneous unit tests
  (`tests/sources/mode/test_mode_eigensolver_physics.py`) also pin legacy-specific
  invariants — candidate power-orthogonality to `1e-6`, `raw_indices` grouping, and
  three-grid mode-index ordering — that a node-grid reconstruction does not reproduce
  to those tolerances. Rerouting inhomogeneous to the new operator (EXECUTED) broke
  four of them; migrating those pins is a supervisor-gated decision, so inhomogeneous
  stays on the legacy operator and the **operator-level** hybrid capability is proven
  by the new half-filled gates instead.
- **Microstrip / differential-pair production hybrid gates are deferred** (fail-closed
  unchanged). Beyond the spurious-mode filtering above, these carry an **interior PEC
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
| `tests/rf/wave_validation/test_transverse_operator.py` | **11 passed** (8 E1a + 3 new inhomogeneous) |
| `tests/rf/wave_validation/test_te10_mode_selection.py` | all pass incl. the six former xfails |
| `tests/rf/wave_validation tests/rf/waveport tests/sources/mode tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py` | **171 passed, 2 xfailed** |

The two remaining xfails are pre-existing and unrelated: `test_wave_level_rlc_resonance_open_gap`
(S1.2 open gap) and `test_bent_effective_index_is_symmetric_in_radius_sign_for_a_symmetric_guide`
(the bent-mode conformal-eps path is inhomogeneous and stays on the legacy operator).

Measured TE10 acceptance (`sin`-correlation, EXECUTED via the WavePort manifest):
`dx = 0.05 / 0.025 / 0.02 / 0.0125 / 0.01 -> corr 1.00000` (all), `beta` matching
analytic `sqrt(k0^2 - (pi/a)^2)` to `< 0.15%`; high-frequency `dx = 0.005, 6 fc ->
corr 1.00000` (genuine TE10, not TE20). Inhomogeneous half-filled `LSE`: 2D operator vs
1D discrete reference `rtol 1.1e-14 / 3.2e-15 / 7.4e-13` at `nu = 24 / 48 / 96`;
continuum transverse-resonance error `0.41% / 0.22% / 0.11%` (first-order, material
discontinuity on a node), fundamental analytic `beta = 15.0354`.

## 5. Falsifications performed (perturb → red → restore → green)

1. **Integration is load-bearing (headline TE10 gate).** Forcing the uniform guide back
   through the legacy operator (`if uniform_isotropic:` → `if False:`) reintroduced the
   sublattice-decoupling defect: `test_waveguide_te10_eigenvector_is_sin[0.02]` and
   `..._operator_returns_clean_sin` went **RED** with `corr 0.554` (vs the `>= 0.99`
   gate). Restored → green.
2. **Inhomogeneous per-component-eps assembly is load-bearing.** Dropping `eps_vv`
   (the component the `LSE` mode sees) from the operator build in
   `_half_filled_operator` (`eps_vv=eps_vv` → `eps_vv=None`) drove
   `test_inhomogeneous_operator_spectrum_matches_one_dimensional_lse_reference` **RED**
   (2D eigenvalue no longer matches the independent 1D reference). Restored → green.

(Both were scratch edits reverted immediately; reproducible from the descriptions.)

## 6. Files added / changed

- `witwin/maxwell/fdtd/excitation/modes.py`: added `_yee_stagger_eps_from_nodes`,
  `_yee_half_to_node_neumann`, `_yee_interior_to_node_dirichlet`,
  `_yee_reconstruct_node_profiles`, `_select_yee_transverse_mode_numpy`,
  `_solve_yee_transverse_vector_mode`; stashed the staggered differences in the E1a
  builder's `meta`; routed the homogeneous non-magnetic branch of
  `_assemble_vector_mode_data` to the new solver.
- `tests/rf/wave_validation/test_te10_mode_selection.py`: un-xfailed the six pins,
  repurposed the blocker regression, updated the docstring.
- `tests/rf/wave_validation/test_transverse_operator.py`: three inhomogeneous
  half-filled `LSE` gates.
- `FEATURE_LIST.md`: additive `e1-rf-modes-b` subsection.
- `docs/assessments/e1-rf-mode-operator-acceptance-2026-07-19.md`: this document.

## 7. Known gaps / handoff

- **Inhomogeneous / hybrid production integration is deferred** (see §3): needs a
  spurious-mode filter, staggered-grid power-orthogonal reconstruction, and migration
  of the four legacy-invariant `test_mode_eigensolver_physics.py` pins.
- **Microstrip / differential-pair (interior-PEC) hybrid gates deferred**: additionally
  need interior-PEC masking on the staggered operator. Fail-closed behavior
  (`NotImplementedError` on an inhomogeneous TEM cross-section) is unchanged.
- **Bent (conformal-eps) mode source** rides the inhomogeneous legacy path; its
  order-`1e-6` mirror-symmetry xfail would likely close once the component-staggered
  operator serves the inhomogeneous path — a bonus for the deferred integration.
- **Legacy `_build_vector_operator_sparse` retained** as the inhomogeneous + magnetic
  regime operator (not a duplicate); the uniform-isotropic production case no longer
  reaches its defective centered branch.

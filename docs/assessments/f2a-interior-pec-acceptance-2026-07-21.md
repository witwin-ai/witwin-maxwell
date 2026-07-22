# F2a acceptance — interior-PEC masking on the staggered transverse operator

> Date: 2026-07-21
> Track: f2-rf-trio (branch `fable/rf-trio`)
> Stage F2a: interior-PEC masking on the Yee-staggered transverse mode operator, plus the
> companion quasi-static electrostatic line-mode engine that serves the TEM/quasi-TEM
> interior-PEC lines the curl-curl operator cannot represent. Operator-level gates and
> falsifications only; production scene routing and benches are F2b.
> GPU: `CUDA_VISIBLE_DEVICES=1`.

## 1. Delivered

1. **Interior-PEC masking on the staggered operator** (`witwin/maxwell/fdtd/excitation/modes.py`).
   `_build_yee_transverse_operator_sparse` gains `pec_eu` / `pec_ev` / `pec_ww` masks:
   - a transverse component sample (`Eu` on the `u`-half / `v`-node grid, `Ev` on the
     `u`-node / `v`-half grid) whose staggered location lies inside a conductor is
     eliminated (Dirichlet 0) with the SAME symmetric row/column removal as the outer
     walls — no penalty term, no asymmetry (the divergence coupling term is
     `-gu_big^T D gu_big` with `D` diagonal, so zeroing entries keeps it symmetric);
   - a longitudinal (`Ew`) node inside a conductor drops from the `eps_ww` divergence
     coupling (its `eps_ww^{-1}` diagonal entry is zeroed);
   - the stacked active-unknown mask is returned in `meta["pec_active"]` for the caller to
     reduce `P -> P[active][:, active]`.
   `_yee_stagger_pec_from_nodes` rasterizes node occupancy onto the three staggered grids
   with the SAME placement as `_yee_stagger_eps_from_nodes` (node→half arithmetic mean,
   interior-node selection). `_solve_yee_transverse_pec_mode` orchestrates the masked
   solve (build → reduce → eigensolve → scatter back → select → reconstruct), forwarding
   the interior-node PEC mask so the reconstructed `Ew` is zero inside conductors.
2. **Connectivity check per conductor-free region** (`_yee_pec_connectivity_check`,
   `_label_connected_components`). Reports the number of connected conductor-free
   (dielectric) regions and distinct interior conductors; fails closed on a degenerate
   pinch (a conductor-free node fully surrounded by conductor) and on fewer than two
   conductor-free interior nodes.
3. **Fail-closed TEM boundary**: `_solve_yee_transverse_pec_mode` raises for a
   `wave_family="tem"` request. The staggered curl-curl `beta**2` operator *does* carry the
   TEM branch in its spectrum, but the shipped occupancy rasterization (threshold 0.5)
   eliminates the conductor-surface straddling normal-`E` samples where the TEM energy
   concentrates, so the masked reduced operator exposes only guided modes (see §3);
   TEM/quasi-TEM lines route to the quasi-static engine instead. This is a `ValueError`
   (a routing/capability boundary, not a new `NotImplementedError`), so the capability-guard
   census budget is unchanged (176).
4. **Quasi-static electrostatic line-mode engine** (`_solve_quasistatic_line_modes`,
   `_quasistatic_laplace_energy`). Solves the variable-coefficient Laplace
   `div(eps grad phi) = 0` with face permittivities (the Yee dielectric average) as a
   sparse SPD system via the same SciPy sparse path the transverse mode operators already
   use, and returns `eps_eff = C / C0` (capacitance-ratio) and `beta = k0 sqrt(eps_eff)`.
   Grounds boundary-connected conductors, treats each isolated interior conductor as a
   signal conductor ordered by centroid, and drives them with caller-supplied potentials
   (`[1.0]` coax/microstrip, `[1,1]` even / `[1,-1]` odd for a pair). Profiles are the
   electrostatic `E = -grad phi` with `H` from the effective wave impedance
   `Z_eff = eta0 / sqrt(eps_eff)`; the returned tensors land on the input eps device.

## 2. Test inventory (commands)

Environment:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/xingyu/code/witwin/witwin-maxwell/.worktrees/wf2-rf-trio
export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest <targets> -q
```

| target | result |
|---|---|
| `tests/rf/wave_validation/test_interior_pec_operator.py` | **13 passed** |
| `tests/rf/wave_validation tests/sources/mode tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py tests/api/public/test_guard_census.py` | **151 passed, 1 xfailed** |

The one xfail is pre-existing and unrelated
(`test_bent_effective_index_is_symmetric_in_radius_sign_for_a_symmetric_guide`).
`CAPABILITY_GUARD_BUDGET = 176` unchanged (no new `NotImplementedError`).

Measured acceptance numbers (all reproducible from the pytest nodes above; probe scripts
under `scratch/` are NOT committed):

- **Coax (uniform, `eps_r = 2.25`, `k0 = 5`, 40x40)**: `eps_eff = 2.2500` (exact
  capacitance ratio), `beta = 7.5000 = k0 sqrt(eps_r)` to `rel <= 1e-6`, and equal to the
  legacy electrostatic `_solve_pec_tem_mode_torch` `beta` to `rel <= 1e-6`
  (`test_coax_quasistatic_tem_matches_analytic_and_legacy`).
- **Parallel-plate limit**: an isolated full-width interior plate over a ground wall in a
  uniform `eps_r = 3` fill returns `eps_eff = 3.0000` to `rel <= 1e-6`
  (`test_parallel_plate_limit_eps_eff_equals_fill`).
- **Microstrip (`eps_r = 4`, `W/h = 2`, box 1.2x2.0, 60x100)**: `eps_eff = 3.0920` vs
  Hammerstad–Jensen (1980) `3.0701`, `rel = 0.71%`; pre-registered gate `<= 3%`
  (`test_microstrip_quasistatic_eps_eff_matches_hammerstad_jensen`). Substrate replaced by
  vacuum gives `eps_eff = 1.0000` (`test_microstrip_without_dielectric_has_unit_eps_eff`).
- **Differential pair (`eps_r = 4`, two strips, 60x100)**: even `eps_eff = 3.2040`, odd
  `eps_eff = 2.9048` (distinct, `9.3%` apart, both in `(1, eps_r)`, even > odd as physical
  for coupled microstrips); potential parity is mirror-symmetric (even) / antisymmetric
  (odd) about the pair centreline to `< 1e-6` of the opposite-parity residual
  (`test_differential_pair_even_odd_modes_are_distinct_and_parity_classified`).
- **Masked operator, guided (analytic septum)**: a full-span PEC septum splits a 2x1
  guide into two 1x1 half-guides; masked `beta = 2.4788` vs the analytic half-guide
  `sqrt(k0^2 - pi^2) = 2.4760` (`rel = 0.11%`, gate `<= 0.5%`), far from the un-split
  full-guide `sqrt(k0^2 - (pi/2)^2) = 3.6787`; connectivity reports 2 dielectric regions,
  1 conductor region (`test_masked_operator_guided_septum_matches_half_guide_analytic`).
  The reconstructed transverse fields vanish on the conductor row to `<= 5%` of peak
  (`test_interior_pec_masked_profile_is_zero_inside_conductor`).

Formula variant: Hammerstad–Jensen effective permittivity as published in E. Hammerstad
and O. Jensen, "Accurate Models for Microstrip Computer-Aided Design", IEEE MTT-S 1980
(the `a(u)`, `b(eps_r)` form), coded in `_hammerstad_jensen_eps_eff`.

## 3. Why TEM routes to the quasi-static engine — the operator carries TEM, the shipped masking removes it (evidence)

> **Correction (2026-07-21, audit remediation).** An earlier draft of this section claimed
> the staggered curl-curl operator "structurally has no TEM branch" and that "there is no
> eigenvalue near 56.25." That claim is false and is retracted here. The operator **does**
> carry the gradient TEM branch in its spectrum; the TEM branch is absent from the shipped
> *masked* reduced operator because of the occupancy rasterization threshold (0.5), a
> masking choice, not a structural property. The fail-closed TEM routing to the quasi-static
> engine is unchanged and correct, but for the accurate reason stated below.

The stage brief's decision #1 asks the coax/microstrip/diff-pair **TEM-class** fundamentals
to come from the interior-PEC-masked staggered operator. The math: for a curl-free field
`Et = -grad(phi)` with `div(eps grad phi) = 0`, both the curl-curl term and the divergence-
coupling term of `P` vanish identically, leaving `P Et = eps k0**2 Et`. So the TEM branch is
an **exact eigenvalue `eps k0**2`** of the operator, discretely as well as in the continuum.

What actually removes it is the rasterization of the PEC occupancy onto the staggered grids.
The shipped `_yee_stagger_pec_from_nodes` with `_PEC_OCCUPANCY_THRESHOLD = 0.5` and `>=`
eliminates the conductor-surface *straddling* normal-`E` samples (occupancy exactly 0.5),
which physically carry the TEM surface charge and where the TEM field energy concentrates.
Killing those samples is what drops the TEM branch from the masked reduced operator.

Evidence (reproducible pytest node
`tests/rf/wave_validation/test_interior_pec_operator.py::test_masked_operator_tem_branch_is_a_masking_artifact`;
uses the identical committed operator builder `_build_yee_transverse_operator_sparse`):

- Square coax annulus `a = b = 1`, inner square PEC, `eps_r = 2.25`, `k0 = 5`
  (`eps k0**2 = 56.25`):
  - **Shipped threshold 0.5** (surface-straddling normal-`E` samples eliminated): the reduced
    operator's spectral maximum is `beta**2 = 30.366`, well below `56.25` — no TEM branch.
  - **Keep-straddle threshold 0.75** (only fully-interior transverse samples eliminated,
    `pec_ww` unchanged): the spectral maximum is `56.25000` `= eps k0**2` exactly, and the
    directly constructed discrete TEM gradient field `-grad(phi)` of the discrete harmonic
    potential is that eigenvector (Rayleigh `56.2500`, relative curl-curl residual `2.2e-13`).
- A hollow homogeneous guide (same box, no inner conductor) tops out at
  `beta**2 = 46.385 = eps k0**2 - (pi/a)**2` (TE10). This is **expected physics** and is *not*
  evidence about the coax: a simply-connected hollow guide has no TEM mode at all, so its
  spectral maximum is correctly TE10. It does not bear on whether the doubly-connected coax
  operator carries TEM.

Consequence for the shipped code: **no wrong numbers ship.** The masked operator is fail-
closed on TEM (raises and points at the quasi-static engine), and the guided interior-PEC
non-TEM branch is not routed through this operator in production (it still uses the legacy
`_pec_vector_operator_torch`; `_solve_yee_transverse_pec_mode` is exercised only by tests).
The quasi-static engine delivers the coax/microstrip/diff-pair physics with the analytic
agreement recorded in §2. The masked operator itself is validated on the guided regime it
does serve by the analytic septum gate (whose normal `E` is zero at the septum, so the septum
gate alone cannot see the surface-sample masking tradeoff).

**Open item for the supervisor (decision #1):** the original decision #1 asked for a coax
TEM cross-implementation gate *on the masked staggered operator* vs the legacy
`_solve_pec_tem_mode_torch`. That gate is achievable — the keep-straddle rasterization above
recovers the exact TEM eigenvalue — but switching the shipped elimination criterion to keep-
straddle carries a real tradeoff: a one-cell-thick conductor sheet (occupancy exactly 0.5 at
its straddling samples) would then be left entirely unmasked. Because this changes the
binding decision #1 gate design and trades off thin-sheet masking, the elimination-criterion
change and the re-execution of the coax cross-implementation gate are **deferred to the
supervisor** rather than made unilaterally here. The doc/code claims are corrected; the
behavior is unchanged and fail-closed.

## 4. Falsifications performed (perturb → red → restore → green)

Recorded via runtime monkeypatch (perturbations reverted immediately; committed tests all
green afterward). Baselines: septum `beta = 2.4788`, microstrip `eps_eff = 3.0920`.

1. **Interior-PEC transverse elimination is load-bearing.** Forcing
   `_build_yee_transverse_operator_sparse` to ignore `pec_eu`/`pec_ev` (no unknowns
   eliminated) drove the septum `beta` from `2.4788` to `4.0000` — no longer the half-guide
   `2.4760` — so `test_masked_operator_guided_septum_matches_half_guide_analytic` and
   `test_interior_pec_masked_profile_is_zero_inside_conductor` go RED. Restored → `2.4788`.
2. **The dielectric weighting in the quasi-static energy is load-bearing.** Forcing
   `_quasistatic_laplace_energy` to solve with vacuum permittivity collapsed the microstrip
   `eps_eff` from `3.0920` to `1.0000`, failing the `<= 3%` H–J gate — RED for
   `test_microstrip_...`, the coax analytic gate, the parallel-plate gate, and the
   diff-pair distinctness gate. Restored → `3.0920`.
3. **The differential drive sign is load-bearing.** Running the "odd" solve with the even
   potentials `[1, 1]` returned `eps_eff = 3.2040`, identical to the even mode, so the
   `even != odd` distinctness assertion in
   `test_differential_pair_even_odd_modes_are_distinct_and_parity_classified` goes RED. The
   correct odd drive `[1, -1]` gives `2.9048` (9.3% apart).

## 5. Files added / changed

- `witwin/maxwell/fdtd/excitation/modes.py`: interior-PEC masking on
  `_build_yee_transverse_operator_sparse` (`pec_eu`/`pec_ev`/`pec_ww`, active mask, dropped
  `eps_ww` coupling); `_yee_stagger_pec_from_nodes`, `_label_connected_components`,
  `_yee_pec_connectivity_check`, `_solve_yee_transverse_pec_mode` (with the fail-closed TEM
  boundary); `_yee_reconstruct_node_profiles` / `_select_yee_transverse_mode_numpy` accept
  an optional `pec_ww_mask`; `_quasistatic_laplace_energy` and
  `_solve_quasistatic_line_modes` (quasi-static electrostatic line-mode engine).
- `tests/rf/wave_validation/test_interior_pec_operator.py`: 13 operator-level gates.
- `FEATURE_LIST.md`: additive `f2a-interior-pec` subsection.
- `docs/assessments/f2a-interior-pec-acceptance-2026-07-21.md`: this document.

## 6. Known gaps / handoff to F2b

- **Production routing is F2b.** `_solve_yee_transverse_pec_mode` and
  `_solve_quasistatic_line_modes` are not yet wired into `_assemble_vector_mode_data` /
  `solve_mode_source_profile`. F2b routes inhomogeneous-with-interior-PEC cross-sections:
  TEM/quasi-TEM fundamentals → `_solve_quasistatic_line_modes`; guided (non-TEM) hybrid
  modes → `_solve_yee_transverse_pec_mode`. Note the existing production TEM path
  (`_assemble_vector_mode_data`, `has_interior_pec and wave_family=="tem"`) still calls the
  legacy `_solve_pec_tem_mode_torch`, which only handles UNIFORM fill; wiring the new
  quasi-static engine unblocks inhomogeneous microstrip/diff-pair there.
- **Quasi-static engine runs the Laplace solve on CPU via SciPy sparse**, matching the
  established mode-solver convention (all transverse eigensolves in `modes.py` are SciPy).
  This is one-time mode setup, not the FDTD hot loop; if a fully device-resident line-mode
  solve is later required, a torch CG on the 5-point stencil is the drop-in.
- **Profile power normalization for the quasi-static profiles** uses the effective wave
  impedance; F2b should confirm the injected-power normalization against the `coax_thru`
  precedent when wiring the benches.
- **The masked operator's guided regime** is validated against one analytic septum case; a
  broader hybrid interior-PEC validation (e.g. ridge/finline) can be added with F2b's
  benches if a reference is generated.

# Track H2 acceptance — SPD tensor permittivity electrostatics (Plan 12 Phase 4)

Stage **H2a**: SPD tensor-eps FVM operator with numerical symmetry / positive-
definiteness gates, rotated-frame manufactured-solution convergence, diagonal-
reduction parity, and anisotropic capacitance reciprocity, plus falsifications.

Base: master `18bc42a` (worktree `wh2-es-tensor`, branch `fable/es-tensor-open`).
Environment: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`, float64 oracles on GPU.

## Delivered items

1. **Full symmetric-positive-definite 3x3 tensor permittivity `epsilon(x)`** in the
   cell-centred finite-volume electrostatic operator. A `Structure` material with
   `epsilon_tensor=DiagonalTensor3(...)` or a full `epsilon_tensor=Tensor3x3(...)`
   now compiles and solves through `Simulation.electrostatic(...)` and
   `Simulation.capacitance(...)`.

2. **Symmetric discretization (derivation).** The operator is
   `A = A_diag + A_cross`:
   - `A_diag` is the existing conservative two-point harmonic-mean face flux, using
     the diagonal tensor entries `eps_xx / eps_yy / eps_zz` per axis. Unchanged for
     isotropic media.
   - `A_cross` carries the off-diagonal entries. It is defined as the gradient of a
     discrete quadratic energy
     `W_cross(phi) = sum_c eps0 V_c (eps_xy gx gy + eps_xz gx gz + eps_yz gy gz)`
     where `gx, gy, gz` are cell-centred central-difference gradients. Because
     `A_cross = grad_phi(W_cross)` is the gradient of a scalar quadratic form, it is
     **symmetric by construction** (`A_cross = Lx^T D_xy Ly + Ly^T D_xy Lx + ...`).
   - **Why the energy form, not the face-averaged-tangential-gradient flux:** the
     plan warns that the natural MPFA face-cross-flux is generally non-symmetric and
     forbids claiming support for a non-symmetric variant. The energy formulation
     guarantees exact symmetry, which is then verified numerically (below).
   - The cross operator has an identically-zero main diagonal (a cell couples to
     itself only through the diagonal face-flux terms), so the Jacobi preconditioner
     is unchanged and stays strictly positive.
   - Total discrete identity `field_energy(phi) = 0.5 phi^T A phi` holds including
     the cross terms, so the `0.5 integral(E.D) = 0.5 sum(V Q)` energy identity and
     the discrete Gauss law continue to close.

3. **Correct anisotropic postprocessing:** `ElectrostaticResultData.D` is the full
   tensor contraction `D_i = eps0 sum_j eps_ij E_j`; the result carries the compiled
   `epsilon_tensor` field.

4. **Compiler validation:** `_material_static_matrix` returns a validated real,
   symmetric, positive-definite 3x3 relative-permittivity matrix (asymmetric /
   indefinite tensors raise `ValueError` as physically invalid lossless
   permittivity). Scenes with only isotropic media keep `epsilon_tensor = None` and
   the byte-identical scalar path (so every existing isotropic analytic result is
   unchanged).

5. **Fail-closed differentiability disposition (H2a portion):** a trainable tensor
   permittivity, or a trainable free charge alongside a tensor dielectric, raises
   `NotImplementedError` (`_reject_trainable_tensor`) rather than silently detaching
   its gradient — the off-diagonal cross-flux has no reverse-mode VJP yet. (The full
   differentiability decision and the open-boundary / truncation study are stage
   H2b.)

## Files added / changed

- `witwin/maxwell/compiler/electrostatic.py`: new `TensorEpsilon` dataclass;
  `CompiledElectrostatics.epsilon_tensor` field; `_material_static_matrix`
  (replaces `_static_epsilon_scalar`, removing the anisotropic-tensor
  `NotImplementedError`); tensor-aware `_rasterize_epsilon` returning
  `(epsilon_r, epsilon_tensor)`.
- `witwin/maxwell/electrostatic/runtime.py`: tensor mode in `ElectrostaticOperator`
  (per-axis diagonal face flux + `_cross_energy` / `_apply_cross`); `field_energy`
  cross term; correct tensor `D`; `epsilon_tensor` on `ElectrostaticResultData`;
  `_reject_trainable_tensor`; `solve_fixed_potential` (routes isotropic ->
  implicit-diff wrapper, tensor -> plain reduced solve).
- `witwin/maxwell/electrostatic/capacitance.py`: tensor routing via
  `solve_fixed_potential`; `_reject_trainable_tensor`.
- `tests/electrostatic/test_tensor_eps.py`: new gate suite (13 tests -- the
  original 11, plus two audit-minor additions, see "Audit-minor cleanup" below).
- `tests/electrostatic/test_api.py`: repurposed the obsolete
  `test_anisotropic_tensor_permittivity_rejected` into
  `test_anisotropic_tensor_permittivity_supported`.
- `docs/reference/fdtd-capability-guard-census.md`,
  `tests/api/public/test_guard_census.py`: H2a census note (net 0, budget 175).
- `FEATURE_LIST.md`: additive `h2-es-tensor` subsection.

## Test inventory (all pass; float64, GPU 1)

Command:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export PYTHONPATH=<worktree>; export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest \
  tests/electrostatic/ tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py \
  tests/materials/compiler/test_material_compiler.py tests/core/scene/test_scene.py -q
```
Result: **183 passed**.

`tests/electrostatic/test_tensor_eps.py` (13 tests) headline gates:
- `test_operator_symmetry_and_positive_definite` — dense 216x216 operator (grounded
  6^3 box): asymmetry `< 1e-9`, min symmetric eigenvalue `> 0`.
- `test_operator_symmetry_random_vectors` — `<Ax,y> = <x,Ay>` to `< 1e-10` relative.
- `test_positive_energy_for_nonzero_field` — `field_energy > 0` for 5 random fields.
- `test_energy_identity_matches_quadratic_form` — `field_energy = 0.5 phi^T A phi`.
- `test_isotropic_material_stays_on_scalar_path` — scalar and isotropic
  `DiagonalTensor3(s,s,s)` keep `epsilon_tensor = None`.
- `test_diagonal_tensor_has_zero_cross_coupling` — diagonal tensor: `_apply_cross`
  is exactly zero; matches the isotropic operator to `< 1e-12`.
- `test_rotated_mms_second_order_convergence` — rotated SPD tensor
  `diag(1,2,3) @ 30deg/45deg`, smooth `sin^2` bump MMS, grids `24/36/54`, observed
  orders both `> 1.9` (gate `> 1.8`).
- `test_mms_gauss_closure` — discrete Gauss residual `< 1e-9` relative.
- `test_anisotropic_capacitance_reciprocity` — three asymmetric terminals in a
  rotated anisotropic block: `reciprocity_error < 1e-6`, positive diagonal /
  non-positive off-diagonal.
- `test_anisotropic_capacitance_energy_consistency` — `energy[j] = 0.5 C[j,j]`.
- `test_trainable_tensor_eps_fails_closed` — trainable free charge + tensor
  dielectric raises `NotImplementedError`.

## Falsifications recorded

1. **Break the cross-term symmetrization.** In `_apply_cross`, detach the second
   factor of each cross-energy product (`gy.detach()`, `gz.detach()`), so only one
   half `Lx^T D Ly` of each symmetric pair survives -> an asymmetric operator.
   Observed RED:
   - `test_operator_symmetry_and_positive_definite`: dense asymmetry `8.40e-2` (>>
     `1e-9`).
   - `test_operator_symmetry_random_vectors`: relative asymmetry ~`4e-2`.
   - `test_anisotropic_capacitance_reciprocity`: `reciprocity_error = 1.92e-5` (>
     `1e-6`).
   (`test_energy_identity` stays green under this break, correctly: a quadratic form
   `phi^T A phi` is insensitive to symmetrization — this is why the dense/random
   symmetry gates, not the energy identity, are the symmetrization discriminators.)
   Restored -> all green.

2. **Drop the cross term.** In `apply_full`, `pass` instead of adding
   `_apply_cross(phi)` (operator ignores the off-diagonals). Observed RED:
   - `test_rotated_mms_second_order_convergence`: observed orders collapse to ~`0.03`
     (the solve converges to the wrong operator's solution, so the MMS error stops
     converging). Restored -> all green.

   Note: `test_mms_gauss_closure` stays green under this break because Gauss closure
   is the residual of the *solved* system (self-consistency), not a check that the
   operator is the correct one — the MMS convergence gate is the discriminator for
   the cross term's presence.

## Test-quality fix found during falsification

`test_operator_symmetry_random_vectors` originally normalized by
`max(|lhs|, |rhs|, 1.0)`. Because the operator entries are `~eps0`-scale (`~1e-12`),
the bilinear form values are `~1e-9`, so the fixed `1.0` floor made the relative
tolerance meaningless (the test would pass under any asymmetry). Fixed to normalize
by `max(|lhs|, |rhs|)`; the falsification above confirms it now discriminates.
Likewise the anisotropic capacitance reciprocity gate was moved from a
mirror-symmetric two-terminal cell (which reports a symmetric matrix even with a
broken operator) to three asymmetrically placed terminals so reciprocity genuinely
probes operator symmetry.

## Capability-guard census

Net **0**, budget stays **175** (measured 175 == budget). Removed the
`compiler/electrostatic.py` anisotropic-tensor reject (`-1`); added the reachable
`electrostatic/runtime.py` `_reject_trainable_tensor` (`+1`). Documented in
`docs/reference/fdtd-capability-guard-census.md` and the
`tests/api/public/test_guard_census.py` header. `tests/api/public/test_guard_census.py`
passes.

## Known gaps / deferred to H2b (or later)

- **Trainable tensor-eps backward:** not implemented (fail-closed). The off-diagonal
  cross-flux VJP would extend the implicit-diff wrapper.
- **Dirichlet-wall cross-flux boundary layer:** the interior scheme is 2nd-order, but
  the cross-flux at a Dirichlet wall is 1st-order for a field with a strong tangential
  gradient at the wall (one-sided `torch.gradient` edge). Verified: the interior
  truncation is machine-zero and a smooth field with vanishing wall gradient
  (`sin^2` bump) converges at 2nd order, while a field with nonzero wall tangential
  gradient shows a 1st-order boundary layer that propagates. A consistent 2nd-order
  wall cross-flux is a candidate for H2b's controlled-truncation work.
- **Open boundary / domain-extension convergence study + `truncation_estimate`
  API:** stage H2b (a hard fail-closed on any `open` boundary spec is also H2b).
- **Per-cell (spatially-varying) tensor permittivity from a single Material** (a
  `torch.Tensor` `eps_r` with `numel != 1`) remains rejected (`NotImplementedError`,
  unchanged). Anisotropy is expressed per-structure via `DiagonalTensor3` /
  `Tensor3x3`; spatial variation comes from multiple structures.

---

## Stage H2b — open-boundary / domain-extension convergence + `truncation_estimate` API + differentiability disposition

Base for this stage: `70074e7` (the H2a commit on branch `fable/es-tensor-open`).
Environment: conda `maxwell`, `CUDA_VISIBLE_DEVICES=1`, float64 oracles on GPU.

### Delivered items

1. **`open` electrostatic boundary fail-close.** `ElectrostaticBoundarySpec` now
   rejects an `open` (infinite-domain) boundary kind on any face with a clear
   `NotImplementedError` (there is no exact radiation condition on the scalar
   potential at a finite Cartesian face; a boundary-element open boundary is a
   later phase). One reviewed capability guard, `electrostatic/api.py`
   `_normalize_bc_entry`.

2. **Opt-in `truncation_estimate` domain-extension API.** New public
   `TruncationEstimate(padding_cells=N)` config and `TruncationReport` result type
   (exported from `witwin.maxwell`). `Simulation.capacitance(scene, ...,
   truncation_estimate=...)` runs ONE additional enlarged grounded-box capacitance
   solve — never silently; only when the config is passed. The enlarged solve holds
   the interior cell grid fixed to floating-point round-off (the domain grows by
   exactly `padding_cells` cell widths per side per axis, so the recomputed cell
   centres reproduce the originals to ULP — max ~1.1e-16 absolute drift, measured
   base-vs-enlarged on the isolated-conductor scene; not literally "byte-identical"),
   so the reported change isolates the pure boundary-truncation effect. `result.capacitance.truncation_estimate` carries the
   base/enlarged matrices, their `delta`, `max_relative_delta` (relative sensitivity
   of C to the enclosure size), the effective base/enlarged enclosure sizes, and a
   1/L Richardson extrapolation `richardson_matrix` to the infinite-domain limit
   with `richardson_max_relative_shift`. `max_relative_delta` and the Richardson
   shift are also surfaced on `Result.solver_stats`.

3. **Two-axis domain-extension convergence study** (committed tests): self-capacitance
   of an isolated conductor vs enclosure size `L` at fixed grid (monotone decrease
   toward a stable Richardson `C_inf`; every finite-`L` capacitance exceeds `C_inf`;
   truncation error shrinks with `L`) AND vs grid at fixed `L` (Cauchy-convergent
   under refinement).

4. **Differentiability disposition — decided: fail-closed.** Plan 12 assigns
   electrostatic gradients to Phase 5; Phase 4 owns the forward SPD tensor-eps
   operator only. A trainable input under a tensor dielectric therefore fails closed
   on both the `Simulation.electrostatic(...)` (H2a `_reject_trainable_tensor`) and
   `Simulation.capacitance(...)` public paths (the capacitance extractor calls the
   same guard), rather than shipping an unverified/detached gradient. No new
   differentiability guard is added or removed; the isotropic implicit-diff backward
   is unchanged.

### Files added / changed (H2b)

- `witwin/maxwell/electrostatic/api.py`: `_normalize_bc_entry` `open`-boundary reject.
- `witwin/maxwell/electrostatic/capacitance.py`: `TruncationEstimate`,
  `TruncationReport`, `CapacitanceData.truncation_estimate`, `_enlarged_scene`,
  `CapacitanceSimulation._truncation_report` + opt-in wiring in `run()`.
- `witwin/maxwell/simulation.py`: `Simulation.capacitance(..., truncation_estimate=...)`.
- `witwin/maxwell/electrostatic/__init__.py`, `witwin/maxwell/__init__.py`: export
  `TruncationEstimate` / `TruncationReport`.
- `tests/electrostatic/test_open_boundary.py`: new gate suite (11 tests).
- `docs/reference/fdtd-capability-guard-census.md`,
  `tests/api/public/test_guard_census.py`: H2b census note, budget `175 -> 176`.
- `FEATURE_LIST.md`: additive H2b lines in the `h2-es-tensor` subsection.

### Test inventory (H2b; all pass; float64, GPU 1)

Command:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export PYTHONPATH=<worktree>; export CUDA_VISIBLE_DEVICES=1
conda run -n maxwell --no-capture-output python -m pytest \
  tests/electrostatic/ tests/api/public/test_guard_census.py \
  tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py \
  tests/materials/compiler/test_material_compiler.py tests/core/scene/test_scene.py -q
```
Result: **194 passed** (183 from H2a + 11 new H2b `test_open_boundary.py`).

`tests/electrostatic/test_open_boundary.py` (11 tests):
- `test_open_boundary_default_fails_closed`, `test_open_boundary_per_face_fails_closed`
  — the `open` boundary raises `NotImplementedError`.
- `test_domain_extension_convergence_vs_size` — C(L) strictly decreasing over
  L in {0.8, 1.2, 1.6, 2.0} (fixed h=0.05); consecutive 1/L Richardson `C_inf`
  agree to `< 5%`; `C_inf` below every finite-L C; truncation error shrinks with L.
- `test_grid_convergence_at_fixed_size` — C Cauchy-convergent under h in
  {0.1, 0.05, 0.025} at fixed L=1.0.
- `test_truncation_estimate_is_opt_in` — no report unless requested.
- `test_truncation_estimate_reports_domain_error` — report populated; enlarged
  domain grew by exactly `2*pad` cells/axis; `delta[0,0] < 0`; Richardson below
  enlarged; stats surfaced.
- `test_truncation_estimate_reduces_with_larger_padding` — larger extension moves
  C more (captures more residual truncation).
- `test_truncation_estimate_requires_dirichlet_enclosure`,
  `test_truncation_estimate_requires_uniform_grid`, `test_truncation_estimate_type_check`
  — misuse `ValueError`/`TypeError`.
- `test_capacitance_trainable_tensor_fails_closed` — public capacitance path rejects
  a trainable free charge under an anisotropic dielectric (`NotImplementedError`).

### Falsifications recorded (H2b)

1. **Remove the `open`-boundary special-case** (`electrostatic/api.py`: change the
   `if kind == "open":` trigger to an impossible kind). Observed RED:
   `test_open_boundary_default_fails_closed` and `test_open_boundary_per_face_fails_closed`
   both fail — an `open` face now falls through to the generic `_BC_KINDS` check and
   raises `ValueError` ("boundary kind must be one of ...") instead of the specific
   `NotImplementedError`. Restored -> green.

2. **Neutralize the enlargement** (`_enlarged_scene`: pad by zero cells so the
   enlarged domain equals the base). Observed RED:
   `test_truncation_estimate_reports_domain_error` fails — `enlarged_size == base_size`
   (1.0 vs the expected 1.8), `delta ~ 0`, and the 1/L Richardson divides by
   `f - 1 = 0` producing `nan`. Restored -> green. (Confirms the report genuinely
   reflects a real, non-trivial domain extension, not a self-comparison.)

### Capability-guard census (H2b)

Net **+1**, budget `175 -> 176` (measured 176 == budget; `test_guard_census.py`
passes). Added: `electrostatic/api.py` `_normalize_bc_entry` `open`-boundary reject.
The differentiability disposition is fail-closed (Phase 4 forward-only; gradients are
Phase 5), so the H2a `_reject_trainable_tensor` guard is retained unchanged and no
differentiability guard is added or removed. `truncation_estimate` misuse cases
(non-Dirichlet enclosure, non-uniform grid) raise `ValueError` and are not capability
guards.

### Known gaps / deferred (after H2b)

- **Tensor-eps differentiable backward** (Phase 5): the off-diagonal cross-flux VJP
  is still unimplemented; trainable tensor eps / free charge under a tensor
  dielectric fails closed. Decided disposition, not an oversight.
- **Exact open boundary** (boundary-element / infinite-domain): deferred to a later
  phase; the domain-extension `truncation_estimate` is the controlled-truncation
  substitute this round.
- **Dirichlet-wall cross-flux boundary layer** (carried from H2a): the interior
  cross-flux is 2nd-order, the one-sided wall cross-flux is 1st-order for a field
  with a strong tangential wall gradient. Now covered by a committed convergence gate
  (see Audit-minor cleanup (a)); a *second-order* wall cross-flux remains future work.
- The Richardson extrapolation assumes a 1/L leading truncation error (physically
  the monopole image term of a charged conductor in a grounded shell); it is an
  estimate, not a certified bound, and is documented as such on `TruncationReport`.

---

## Audit-minor cleanup (round-H, 2026-07-21)

Round-H audit minors on the H2 delivery. Env: `maxwell`, `CUDA_VISIBLE_DEVICES=1`,
float64 oracles on GPU. `python -m pytest tests/electrostatic -q -> 90 passed`.

### (a) Wall cross-flux now exercised with a nonzero tangential wall gradient

The rotated-MMS gate (`test_rotated_mms_second_order_convergence`) uses
`phi = prod sin^2(pi t)`, whose `f` and `f'` both vanish at the walls, so its
tangential (and normal) gradient is zero on every Dirichlet face — the wall
cross-flux was never exercised. Added
`test_wall_tangential_cross_flux_mms_converges`: a manufactured solution
`phi = prod sin(t + 0.5)` with nonzero value AND nonzero derivative at `t = 0, 1`,
so on every wall face the field has a genuine tangential gradient and the full
rotated tensor's off-diagonal cross-flux is active up to the boundary (walls pinned
to the exact nonzero values). Observed L2 errors at `n = 24/36/54`:
`1.789e-3 / 1.224e-3 / 8.297e-4`, orders `0.936 / 0.958` (first order — the
documented one-sided wall cross-flux + half-cell Dirichlet ghost are first order for
a field with a nonzero wall gradient, versus the interior second order the sin^2
gate sees). Gate: monotone AND `min order > 0.85`.
- **Falsification**: dropping the cross term (`_apply_cross -> 0`) makes it diverge —
  errors *rise* `4.865e-3 / 4.893e-3 / 4.912e-3`, orders `-0.014 / -0.010`,
  non-monotone → RED. Restored → green.

### (b) `_apply_cross` no longer rebuilds an autograd graph per PCG iteration

`ElectrostaticOperator._apply_cross` previously ran `torch.enable_grad()` +
`torch.autograd.grad(_cross_energy)` on every reduced-solve iteration. Replaced with
a direct precomputed stencil: the per-axis central-difference gradient operator is
materialized once at construction (`_gradient_matrix`, the exact linear map
`torch.gradient` applies), and `A_cross = Mx^T sx + My^T sy + Mz^T sz` is applied as
matrix products / transposes. This is bit-for-bit the autograd result (same
finite-difference map and its transpose), so the energy identity still closes
(`0.5 phi^T A_cross phi == _cross_energy`).
- **Equality gate** (committed): `test_apply_cross_matches_autograd_energy_gradient`
  — direct stencil vs `grad_phi(_cross_energy)` on random fields (nonzero tangential
  gradient at every wall): rel `4.9e-16` (gate `< 1e-12`). Energy identity rel `0.0`.
- **Falsification**: applying `M` instead of `M^T` in the transpose step (a wrong
  adjoint) drives the equality rel to `2.15` → RED.
- This resolves the H2a "per-iteration autograd" perf note; no accuracy change.

### (c) Test count corrected

The doc previously said `test_tensor_eps.py` had "12 tests"; the original suite had
**11**. With the two audit-minor additions above it is now **13**. (`test_open_boundary.py`
likewise grows from 11 to 13, see (e).)

### (d) "byte-identical" corrected to the ULP statement

The `truncation_estimate` enlargement recomputes the interior cell centres; they
reproduce the originals only to floating-point round-off, not byte-for-byte. Measured
base-vs-enlarged cell-centre drift on the isolated-conductor scene (pad 8, h 0.05):
**max 1.1102e-16** absolute. Docstrings (`TruncationEstimate`, `_enlarged_scene`) and
this doc now say "fixed to ULP (~1.1e-16)" instead of "byte-identical".

### (e) `truncation_estimate` structure-at-wall confound now fails closed

Enlarging the domain fills the new cells with background medium. If a dielectric
structure reaches the base domain wall, the enlarged solve replaces it with
background there, so `delta` would confound boundary truncation with a change of the
surrounding medium. `_structure_reaches_boundary` (checked in `_truncation_report`)
now **raises `ValueError`** when the compiled permittivity deviates from the vacuum
background on the outer cell shell — the documented fail-closed choice (matching the
other truncation misuse guards, which also raise). Terminals/conductors are
unaffected (the existing isolated-conductor convergence tests have no structures and
still run). Tests: `test_truncation_estimate_rejects_structure_touching_boundary`
(a full-span dielectric bar → raises), `test_truncation_estimate_allows_interior_structure`
(a dielectric with a background margin → runs).
- **Falsification**: forcing `_structure_reaches_boundary -> False` lets the
  wall-touching scene run and report a confounded `delta[0,0] = -1.20e-12` (no
  error); with the check restored it raises. RED without the guard.

### Census

No `raise NotImplementedError` guard added or removed by this cleanup (the confound
check raises `ValueError`, which the census does not count). Budget unchanged at
**176**.

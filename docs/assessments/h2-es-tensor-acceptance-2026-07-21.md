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
- `tests/electrostatic/test_tensor_eps.py`: new gate suite (12 tests).
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

`tests/electrostatic/test_tensor_eps.py` (12 tests) headline gates:
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

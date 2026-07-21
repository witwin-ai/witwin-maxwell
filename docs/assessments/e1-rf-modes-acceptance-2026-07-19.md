# E1a acceptance — Yee-staggered transverse full-vector mode operator

> Date: 2026-07-19
> Track: e1-rf-modes (branch `fable/rf-mode-operator`)
> Stage E1a: derive + implement the staggered operator (new builder alongside the
> old), exact-discrete golden gates + connectivity + convergence + TE20/TM11. The
> selector is NOT touched in this stage (that is E1b).
> Binding context: `docs/reference/rf-wave-validation-2026-07-18.md` §1.2 / §5.

## 1. The defect this replaces (measured, not re-litigated)

The legacy full-vector operator (`_build_vector_operator_sparse`) could not
represent a clean full-grid guided mode on a hollow metallic aperture:

- The centered uniform-isotropic branch reused ONE centered derivative for both
  eliminated curls, composing a **stride-two** stencil that decoupled the odd/even
  transverse sublattices. The half-wave `sin(pi u/a)` lived on ONE sublattice with
  the other ~0; the best full-grid `sin`-correlation over the whole degenerate
  subspace capped at 0.51–0.59 (dx=0.05->0.548, 0.025->0.522, 0.02->0.592,
  0.01->0.509), every candidate `checkerboard_fraction > 0.35`.
- The alternative staggered branch coupled the sublattices but imposed an
  asymmetric metallic boundary (one wall Neumann, one Dirichlet) → `beta` ~10% low.

Root cause: the operator did not keep each transverse field component at its own
Yee location.

## 2. Grid layout (verified against the repo's 3D Yee convention)

Propagation along the port normal `w`; transverse coords `(u, v)` with `Nu x Nv`
cells (wall-to-wall extents `a = Nu*du`, `b = Nv*dv`; walls at nodes `0` and `Nu`
/ `Nv`). The repo's 3D placement is `Ex@(i+1/2, j, k)`, `Ey@(i, j+1/2, k)`,
`Ez@(i, j, k+1/2)`, `Hx@(i, j+1/2, k+1/2)`, `Hy@(i+1/2, j, k+1/2)`,
`Hz@(i+1/2, j+1/2, k)` (from `witwin/maxwell/scene.py` `x_half/y_half/z_half` and
`modes.py:_field_component_axis_coords`). Its transverse restriction (the `w`
half-offset is analytic under `exp(-i beta w)`):

| field | u-index | v-index | grid |
|---|---|---|---|
| `Eu` | half `(i+1/2)` | node `(j)` | `Nu x (Nv-1)` interior (Dirichlet on v-walls) |
| `Ev` | node `(i)` | half `(j+1/2)` | `(Nu-1) x Nv` interior (Dirichlet on u-walls) |
| `Ew` | node `(i)` | node `(j)` | interior nodes `(Nu-1) x (Nv-1)` (Dirichlet on all walls) |
| `Hu` | node | half | |
| `Hv` | half | node | |
| `Hw` | half | half | |

PEC walls are **symmetric by construction**: on a `u`-wall the samples lying ON
the wall are `Ev`, `Ew` (tangential) → Dirichlet 0; `Eu` is half a cell inside →
no condition (its wall-side curl neighbour uses the Dirichlet `Ev`). Both opposite
walls get identical treatment.

## 3. Derivation (discrete equations → elimination → final stencil)

Time-harmonic, source-free, non-magnetic (`mu = 1`), `exp(-i beta w)` so
`d/dw -> -i beta`. `k0 = omega/c`. Discrete Yee first differences: `d+` forward
(node→half), `d-` backward (half→node), with `d- = -(d+)^T` (the PEC ghost
half-value outside a wall is 0). Curl `E = -i omega mu0 H`, curl `H = i omega eps0
eps E`.

**Longitudinal fields.** The `w`-components of the two curls, evaluated at their
Yee locations:

    Hw = -(d+_u Ev - d+_v Eu) / (i omega mu0)                         (curl E)_w
    d-_u Hv - d-_v Hu = i omega eps0 eps_w Ew                         (curl H)_w

Rather than invert the second (nonlocal), eliminate `Ew` with Gauss' law
`div(eps E) = 0`, whose `w`-derivative gives `-i beta eps_w Ew = -(d-_u(eps_u Eu) +
d-_v(eps_v Ev))`, i.e.

    Ew = (1 / (i beta eps_w)) ( d-_u(eps_u Eu) + d-_v(eps_v Ev) )     (GAUSS)

sampled at interior nodes.

**Transverse `E`-curls** give `Hu, Hv`:

    Hu = -(d+_v Ew + i beta Ev) / (i omega mu0)                       (curl E)_u
    Hv =  (i beta Eu + d+_u Ew) / (i omega mu0)                       (curl E)_v

**Transverse `H`-curls** (the remaining two equations) close the system:

    d-_v Hw + i beta Hv = i omega eps0 eps_u Eu                       (curl H)_u
   -i beta Hu - d-_u Hw = i omega eps0 eps_v Ev                       (curl H)_v

Substitute `Hu, Hv, Hw`, multiply by `i omega mu0`, use
`(i omega)^2 eps0 mu0 = -k0^2`. Two β² equations drop out. In `(curl H)_u`:

    d-_v d+_v Eu - d-_v d+_u Ev + i beta d+_u Ew + k0^2 eps_u Eu = beta^2 Eu

and in `(curl H)_v`:

    d-_u d+_u Ev - d-_u d+_v Eu + i beta d+_v Ew + k0^2 eps_v Ev = beta^2 Ev

Insert (GAUSS): `i beta d+_u Ew = d+_u [ eps_w^{-1} (d-_u(eps_u Eu) + d-_v(eps_v
Ev)) ]` — **the `i beta` cancels the `1/(i beta)`**, leaving a clean β² eigenproblem
with no β on the left. Final `2x2` operator `P` on `et = (Eu, Ev)`:

    P_uu = d-_v d+_v          + k0^2 eps_u + d+_u eps_w^{-1} d-_u eps_u
    P_vv = d-_u d+_u          + k0^2 eps_v + d+_v eps_w^{-1} d-_v eps_v
    P_uv = -d-_v d+_u                      + d+_u eps_w^{-1} d-_v eps_v
    P_vu = -d-_u d+_v                      + d+_v eps_w^{-1} d-_u eps_u

This is the Fallahkhair–Li–Murphy full-vectorial operator specialized to `mu = 1`,
now assembled directly from the Yee differences of the layout above.

**Discrete building blocks** (`_yee_half_to_node_first_difference`, spacing `dx`,
`Nu` cells): `Gu` is the `(Nu-1) x Nu` backward difference half→interior-node,
`Su = -Gu^T` the forward difference interior-node→half. Then the transverse
Laplacians are `Gu Su = tridiag(1,-2,1)/dx^2` (Dirichlet, interior nodes) and
`Su Gu = tridiag(1,-2,1)/dx^2` (Neumann, half grid, with a **constant null vector**
= the `TE_{m0}` transverse-uniform direction). `eps_w^{-1}` sits at interior nodes;
`eps_u`, `eps_v` at their component grids.

**Homogeneous reduction.** For constant `eps`, `eps_w^{-1} eps_u = 1` and the
cross terms cancel (`-d-_v d+_u + d+_u d-_v = 0` since orthogonal-axis differences
commute as Kronecker factors), so `P` block-diagonalizes into two scalar transverse
Helmholtz problems — the physically correct TE/TM decoupling, NOT a sublattice
defect. Each block is a standard 5-point graph → **connected**.

**Exact discrete eigenpairs.** `Ev = sin(m pi u_i/a) * cos(n pi v_{j+1/2}/b)` is an
exact eigenvector with

    beta^2 = k0^2 - k~(m, a)^2 - k~(n, b)^2 ,   k~(m, a) = (2/dx) sin(m pi dx / (2a))

(Dirichlet in `u`, Neumann in `v`); `Eu = cos(m pi u_{i+1/2}/a) * sin(n pi v_j/b)`
mirrors it. TE10 is `(m,n)=(1,0)` on the `Ev` block (`cos(0)=1`, transverse-uniform
in `v`, null `v`-Laplacian), giving `beta^2 = k0^2 - k~(1,a)^2` with an exact
`sin(pi u/a)` eigenvector.

## 4. Delivered items

- `witwin/maxwell/fdtd/excitation/modes.py`:
  - `_yee_half_to_node_first_difference(cells, spacing)` — staggered Yee first
    difference (half→interior node) with its `-transpose` forward partner.
  - `_yee_transverse_grids(...)` — Yee sample coordinates + block shapes.
  - `_yee_transverse_discrete_transverse_wavenumber(order, extent, spacing)` —
    the exact `k~` above.
  - `_build_yee_transverse_operator_sparse(...)` — assembles `P` (β² eigenproblem),
    per-component `eps` at the correct Yee locations, `mu = 1`, returns the sparse
    operator + metadata (block matrices, grids, shapes).
  - `_split_yee_transverse_eigenvector(vector, meta)` — reshape to `(Eu, Ev)`.
- The legacy `_build_vector_operator_sparse` and the selector are **unchanged**
  (E1b wires the new builder in and retires the old branch).

## 5. Test inventory (`tests/rf/wave_validation/test_transverse_operator.py`)

Run:
```
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export PYTHONPATH=<worktree>; export CUDA_VISIBLE_DEVICES=0
conda run -n maxwell --no-capture-output python -m pytest tests/rf/wave_validation/test_transverse_operator.py -q
```
Result: **8 passed** (float64 CPU oracle; no CUDA needed for these unit gates).

| test | gate | measured |
|---|---|---|
| `test_operator_is_symmetric_for_homogeneous_cross_section` | `P == P^T` | symmetric (atol 1e-9) |
| `test_full_discrete_spectrum_matches_closed_form` | every eigenvalue = closed-form β², rtol <= 1e-10 | max rtol **1.8e-14** (540 eigenvalues) |
| `test_te10_eigenpair_is_exact_sin` | β² rtol <= 1e-10; `Ev` `sin(pi u/a)` corr >= 0.9999; `checkerboard_fraction < 0.05`; `Eu` energy ~0 | β² rtol **3.9e-15**, corr **1.0000**, cb **0.0018** |
| `test_te20_eigenpair_is_exact_sin` | β² rtol <= 1e-10; `Ev` `sin(2 pi u/a)` corr >= 0.9999 | β² rtol **3.2e-15**, corr **1.0000** |
| `test_tm11_analytic_field_is_an_exact_eigenvector` | analytic `sin cos` `Ev` residual `\|Pv-β²v\|/\|Pv\|` <= 1e-10; eigenvalue present | residual **4.0e-15** |
| `test_homogeneous_blocks_are_connected_no_sublattice_decoupling` | each of `block_uu`, `block_vv` a single connected component | 1 and 1 |
| `test_te10_converges_second_order_to_continuum` | solved TE10 β → continuum, order >= 1.9 on 3 grids (16/32/64 cells) | orders **2.00**, **2.00** (err 1.67e-3 / 4.17e-4 / 1.04e-4) |
| `test_operator_rejects_degenerate_grid` | `nu_cells=1` raises `ValueError` | fail-closed |

## 6. Falsifications recorded (perturb → red → restore → green)

1. **Staggering removed (the legacy defect).** Monkeypatched
   `_yee_half_to_node_first_difference` to a centered **stride-two** stencil (reuse
   one centered derivative, as the retired operator did). `test_te10_eigenpair_is_exact_sin`
   went **RED** (β² no longer matches; the sin profile collapses). Restored → **green**.
2. **Connectivity gate discrimination.** A genuine stride-two second difference
   (offsets `+/-2`) yields **2** connected components (would fail the gate); the
   staggered `block_uu` yields **1**. Confirms the gate catches the exact
   odd/even-sublattice decoupling.
3. **Spectrum gate.** Injecting a `+1.0` diagonal defect into the assembled
   operator drove the spectrum max rtol to **2.4e-4** (>> the 1e-10 gate) →
   `test_full_discrete_spectrum_matches_closed_form` **RED**.

(Falsification scripts were run under `scratch/` and removed; they are reproducible
from the descriptions above.)

## 7. Adjacent suites

```
conda run -n maxwell --no-capture-output python -m pytest \
  tests/rf/wave_validation/test_transverse_operator.py \
  tests/rf/wave_validation/test_te10_mode_selection.py \
  tests/api/public/test_public_api.py \
  tests/api/public/test_simulation_smoke.py \
  tests/api/public/test_guard_census.py -q
```
Result: **40 passed, 6 xfailed**. The 6 xfails are the strict TE10 selector pins in
`test_te10_mode_selection.py` — they stay RED because E1a does not touch the
selector (E1b flips them). The guard census (`test_guard_census.py`, budget 176)
passes unchanged: E1a adds only new functions and no capability guards.

## 8. Known gaps / deferred to E1b

- The new builder is **not wired into the selector / WavePort path** yet; the 6
  strict-xfail TE10 pins remain RED by design.
- **Hybrid (inhomogeneous) validation deferred.** The operator supports per-component
  `eps` at the correct Yee locations, but the microstrip quasi-TEM `eps_eff` gate,
  the manufactured half-filled parallel-plate gate, and the differential-pair
  even/odd split are E1b gates (they also require the selector interpolating scene
  `eps` onto the staggered component grids). E1a validates the homogeneous golden
  spectrum only.
- **`mu = 1` assumption.** The derivation eliminates `Hz` assuming non-magnetic
  media (correct for the RF port families in scope). Magnetic cross-sections would
  need the `mu_w` term restored; not in scope.
- For a homogeneous guide the physical TM_mn / TE_mn modes are degenerate; the
  eigensolver returns an arbitrary rotation of that subspace. The TM11 gate is
  therefore rotation-independent (exact-eigenvector residual), and selector-level
  polarization disambiguation is an E1b concern.

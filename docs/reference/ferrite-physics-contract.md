# Magnetized Ferrite / Gyromagnetic Permeability — Phase-0 Numerical Contract

Status: Phase 0 frozen contract (plan 08, slices 0a/0b/1a).
Scope: DC-biased ferrite gyromagnetic (Polder) permeability as a linearized
Landau-Lifshitz-Gilbert (LLG) magnetization ADE advanced locally at each ferrite
cell. Time-domain (FDTD) only; FDFD is user-deferred.

Base commit: master `d26e83c`.

This document freezes the sign/unit conventions, the LLG->Polder derivation, the
implicit-midpoint discretization with its discrete-energy (passivity) proof, the
collocation operator with its exact transpose, and the `AcceptanceBudget`. Nothing
below may be silently changed; the change rule in the last section governs
loosening any tolerance.

The reference oracle that binds this contract to code is
`witwin/maxwell/fdtd/ferrite_reference.py` (a verification oracle, not a solver
and not a CPU fallback — nothing in the runtime may import it), and the public
material type is `witwin.maxwell.GyromagneticFerrite`.

---

## 1. Frozen sign and unit conventions

| Symbol | Meaning | Unit | Sign / value convention |
| --- | --- | --- | --- |
| `exp(-i*omega*t)` | time convention | -- | frozen; matches every existing pole `susceptibility_at_freq` (`media.py`) |
| `gamma` | gyromagnetic ratio magnitude | rad/(s*T) | `> 0`, default `1.760859e11` |
| `mu_0` | vacuum permeability | H/m | `4*pi*1e-7` |
| `Ms` | saturation magnetization | A/m | `> 0` |
| `H0` | static internal bias field | A/m | 3-vector, `!= 0`; user supplies the internal field (no magnetostatic solve) |
| `b` | bias unit vector | -- | `b = H0 / |H0|` |
| `alpha` | Gilbert damping | -- | `>= 0` (dimensionless) |
| `mu_infinity` | high-frequency background permeability | -- | `> 0`, default `1.0` |
| `omega_0` | Larmor precession frequency | rad/s | `omega_0 = gamma * mu_0 * |H0|` (uses the bias magnitude, `> 0`) |
| `omega_m` | magnetization frequency | rad/s | `omega_m = gamma * mu_0 * Ms` (`> 0`) |

All public API is SI. CGS datasheet quantities (`4*pi*Ms` in Gauss, bias in
Oersted) enter **only** through `GyromagneticFerrite.from_cgs(...)`, which records
the exact conversion factors in the material metadata. The CGS conversions are:

```text
mu_0 * Ms  [T] = (4*pi*Ms [Gauss]) * 1e-4          ->  Ms  [A/m] = (4*pi*Ms [Gauss]) * 1e-4 / mu_0
mu_0 * H0  [T] = (H0 [Oe])        * 1e-4           ->  H0  [A/m] = (H0 [Oe])        * 1e-4 / mu_0
```

so `1 Oe -> 1000/(4*pi) = 79.57747 A/m` and `1 Gauss of 4*pi*Ms -> 79.57747 A/m`
of `Ms`. Example: `4*pi*Ms = 1750 Gauss -> Ms = 139261 A/m`.

The gyromagnetic ratio is frequently quoted as `gamma/(2*pi)` in `GHz/T`
(`~28.0`) or `MHz/Oe` (`~2.80`); `from_cgs` accepts the SI `gamma` directly and
records any quoted form in metadata.

---

## 2. Linearized LLG -> Polder derivation

### 2.1 Continuous LLG

The Landau-Lifshitz-Gilbert equation for the magnetization `M` in the internal
field `H` is

```text
dM/dt = -gamma * mu_0 * (M x H) + (alpha / Ms) * (M x dM/dt).
```

Linearize about the saturated static state `M = Ms*b + m`, `H = H0*b + h`, with
the small RF parts `m`, `h` transverse to `b` (the longitudinal parts are
second order). Choose the local frame with `b = z_hat`; then `m = (m_x, m_y)`,
`h = (h_x, h_y)`. Keeping first-order terms:

```text
dm_x/dt = -omega_0 * m_y + omega_m * h_y - alpha * dm_y/dt
dm_y/dt = +omega_0 * m_x - omega_m * h_x + alpha * dm_x/dt
```

with `omega_0 = gamma*mu_0*H0` and `omega_m = gamma*mu_0*Ms`.

### 2.2 State-space form (used by the ADE and the reference oracle)

Collect the Gilbert cross-terms on the left:

```text
G * dm/dt = K * m + S * h
G = [[1, alpha], [-alpha, 1]]     (Gilbert mass)
K = omega_0 * [[0, -1], [1, 0]]   (precession, skew-symmetric)
S = omega_m * [[0, 1], [-1, 0]]   (RF drive)
```

so with `c = 1 / (1 + alpha^2)`:

```text
dm/dt = P*m + Q*h
P = G^-1 * K = c*omega_0 * [[-alpha, -1], [ 1, -alpha]]
Q = G^-1 * S = c*omega_m * [[ alpha,  1], [-1,  alpha]]
```

The induction is `B = mu_0 * (mu_infinity * h + m)`, i.e. the relative
permeability tensor is `mu_r = mu_infinity*I + chi` with `chi = m/h`.

### 2.3 Frequency-domain susceptibility

For `exp(-i*omega*t)`, `d/dt -> -i*omega`, so the transverse susceptibility is

```text
chi(omega) = (-i*omega*I - P)^-1 * Q.
```

Solving the 2x2 system gives the scalar Polder components (verified against the
state-space inverse to machine precision):

```text
W(omega)     = omega_0 - i*alpha*omega
D(omega)     = W^2 - omega^2
mu(omega)    = mu_infinity + omega_m * W / D          (diagonal transverse)
kappa(omega) = omega_m * omega / D                    (gyrotropic off-diagonal)
mu_parallel  = mu_infinity                            (along the bias)
```

In the lossless limit `alpha = 0` this reduces to the textbook Polder form
`mu = mu_infinity + omega_0*omega_m/(omega_0^2 - omega^2)`,
`kappa = omega*omega_m/(omega_0^2 - omega^2)`. For `exp(-i*omega*t)` the loss
enters as `omega_0 - i*alpha*omega` (the conjugate of the `exp(+i*omega*t)`
engineering form), giving a positive absorptive imaginary part near resonance,
consistent with every other dispersive pole in `media.py`.

### 2.4 Frozen lab-frame tensor form

With the bias unit vector `b` and the cross-product matrix

```text
[b]_x = [[0, -b_z, b_y], [b_z, 0, -b_x], [-b_y, b_x, 0]]      ([b]_x v = b x v),
```

the frozen covariant Polder tensor is

```text
mu_r(omega) = mu(omega)*(I - b b^T) + mu_parallel(omega)*(b b^T) + i*kappa(omega)*[b]_x.
```

For `b = z_hat` this is exactly

```text
mu_r = [[ mu,        -i*kappa,  0        ],
        [ i*kappa,    mu,       0        ],
        [ 0,          0,        mu_parallel]],
```

matching the plan's frozen Polder form (`mu_xy = -i*kappa`, `mu_yx = +i*kappa`).

### 2.5 Circular eigen-permeabilities

The transverse block has circular-polarization eigenvalues

```text
mu_+ = mu + kappa = mu_infinity + omega_m/(omega_0 - omega)   (lossless)
mu_- = mu - kappa = mu_infinity + omega_m/(omega_0 + omega)   (lossless)
```

so the resonant (gyromagnetic-resonance) circular polarization is `mu_+`, with
its pole at `omega = omega_0`; `mu_-` is non-resonant. The two circular waves
propagate at `k_pm = (omega/c)*sqrt(eps_r * mu_pm)`; their phase difference over a
length `L` is the Faraday rotation `theta = (Re(k_+) - Re(k_-))*L/2`.

Because the scalar `kappa` is magnitude-based (built from `|H0|`, `Ms`), the
gyrotropy handedness for a wave propagating along `k_hat` is carried by
`s = sign(b . k_hat)`: the along-propagation eigen-permeabilities are
`mu_pm = mu +/- s*kappa`. A bias reversal (`b -> -b`) flips `s`, swapping `mu_+`
and `mu_-` and flipping the sign of the Faraday rotation.

### 2.6 Bias-reversal (non-reciprocity) property

Because `omega_0` and `omega_m` are built from magnitudes (`|H0|`, `Ms`), a bias
reversal is purely `b -> -b`, which sends `[b]_x -> -[b]_x`. Therefore:

- `kappa` (the gyrotropic off-diagonal) flips sign,
- the diagonal `mu` and `mu_parallel` are unchanged,
- `mu_+` and `mu_-` swap, so the Faraday rotation angle `theta` flips sign.

This is the defining non-reciprocity and is an exact algebraic symmetry (gated at
`bias_reversal_symmetry`, not a fitted tolerance).

---

## 3. Implicit-midpoint discretization

### 3.1 Update

The magnetization ADE `dm/dt = P*m + Q*h` is advanced by the implicit midpoint
(trapezoidal) rule, with the RF field sampled at the half step to co-time with
the leapfrog magnetic update:

```text
(I - (dt/2)*P) * m^{n+1} = (I + (dt/2)*P) * m^n + dt*Q*h^{n+1/2}.
```

Precompute per cell (all frozen at compile time, no per-step allocation):

```text
Phi   = (I - (dt/2)*P)^-1 * (I + (dt/2)*P)   (Cayley propagator)
Gamma = (I - (dt/2)*P)^-1 * (dt*Q)           (drive)
m^{n+1} = Phi*m^n + Gamma*h^{n+1/2}.
```

### 3.2 Discrete CW response

Substituting `m^n = chi_d * h_0 * z^n`, `h^{n+1/2} = h_0 * z^{n+1/2}` with
`z = exp(-i*omega*dt)` gives the exact discrete transfer function

```text
chi_d(omega, dt) = dt * z^{1/2} * [(z - 1)*I - (dt/2)*(z + 1)*P]^-1 * Q,
```

which converges to `chi(omega)` as `dt -> 0` at second order (verified:
relative error `8.0e-4 / 2.0e-4 / 5.0e-5` at `dt = 4e-13 / 2e-13 / 1e-13 s`,
i.e. the expected `O(dt^2)` ratio of 4 per halving). The closed-form phasor
`m^n = chi_d h_0 z^n` satisfies the recurrence in section 3.1 to machine
precision (relative residual `~5e-18`); this is the pure-algebra identity that
`reference_polder_rtol` gates.

### 3.3 Unconditional stability (no dt shrink)

The symmetric part of `P` is

```text
(P + P^T)/2 = -(omega_0 * alpha / (1 + alpha^2)) * I  <= 0    for alpha >= 0,
```

so every eigenvalue of `P` has non-positive real part. The Cayley transform
`Phi = (I - (dt/2)P)^-1 (I + (dt/2)P)` maps the closed left half-plane into the
closed unit disk, so the discrete magnetization poles lie inside the unit circle
for every `dt > 0` and every `alpha >= 0`. The scheme is therefore
unconditionally stable and needs no dt shrink near resonance; the forbidden
explicit-Euler escape (dt shrinking to chase a lightly-damped pole) is excluded
by contract.

---

## 4. Discrete energy and passivity

Define the local precession energy density `E^n = (1/2) * (m^n)^T * m^n`, the
first-order small-angle Zeeman energy of the transverse magnetization.

- **`alpha = 0` (lossless).** `P = K` is skew-symmetric, so `Phi` is exactly
  orthogonal (`Phi^T Phi = I`). Hence `E^{n+1} = E^n` **exactly** — zero growth,
  not merely bounded (verified: energy ratio `1.0` to `2e-11` over `1e5` steps).
  Continuously, `d/dt((1/2)|m|^2) = m^T*K*m = 0` because `K` is skew.

- **`alpha > 0` (lossy).** Continuously,
  `d/dt((1/2)|m|^2) = m^T*P*m = -(omega_0*alpha/(1 + alpha^2)) |m|^2 <= 0`.
  Discretely, `Phi` is a strict contraction (Cayley transform of a matrix with a
  negative-definite symmetric part), so `E^{n+1} <= E^n` (monotone decay) for
  every `dt > 0`.

Passivity therefore holds for all `alpha >= 0` with no dt restriction. The
`passive_energy_residual <= 1%` gate uses the zero-growth (not merely bounded)
falsifiable form at `alpha = 0`.

---

## 5. Frozen collocation operator and exact transpose

On the Yee grid `H_x`, `H_y`, `H_z` live on three distinct edges. The
gyromagnetic correction at a target edge `e` (say an `H_x` edge) needs the two
partner components at `e`'s location. Freeze the gather `C` as the standard
4-point Yee arithmetic average of the nearest partner-component edge samples
surrounding `e` (each weight `1/4`). The precessional correction is deposited
back onto the partner edges with **exactly** `C^T` (the same four `1/4` weights).

Because deposit `= gather^T`, the discrete field<->ADE power-exchange operator
inherits the anti-Hermitian structure of `i*kappa*[b]_x`, so at `alpha = 0` the
correction injects **zero** net energy — this is what prevents the
staggered-collocation spurious source of Risk 2. This `C`/`C^T` pair is frozen
here and is reused verbatim by the Phase-3 reverse kernel (the transpose of a
`C^T` deposit is a `C` gather).

The single-cell / 1D reference oracle uses co-located components (`C = I`) to
isolate the precession/ADE physics; the collocation operator is exercised only
by the 3-D CUDA kernel (slice 1c) but is frozen now so the forward and reverse
kernels share one definition.

---

## 6. Frozen AcceptanceBudget

The budget below is frozen at Phase 0. It is mirrored verbatim in
`ferrite_reference.AcceptanceBudget`; the two must stay identical.

| Gate | Value | Justification |
| --- | --- | --- |
| `reference_polder_rtol` | `1e-5` | Torch reference vs. closed-form analytic Polder in-band away from resonance: a pure-algebra identity of the discretization in the CW limit, so it must hold at machine-adjacent tolerance (matches the thin-wire `rtol <= 1e-5` house standard). |
| `analytic_response_rel_err` | `<= 2%` | Full FDTD complex response (circular eigenmode `k_pm`, slab R/T) vs. analytic: finite grid/dt/run-length discretization error, same class as existing dispersive-material acceptance. |
| `analytic_phase_err` | `<= 3 deg` | Non-reciprocal phase / Faraday rotation is the primary physical observable; 3 deg bounds staircase + temporal-dispersion phase error while still resolving isolator directionality. |
| `passive_energy_residual` | `<= 1%` | With `alpha = 0` the discrete magnetic energy must not grow; the falsifiable form is zero-growth, not merely bounded. |
| `convergence_tiers` | `>= 3` grid AND `>= 3` dt AND `>= 3` run-length | Monotone convergence with a fitted order distinguishes a correct scheme from a coincidentally-close one; three tiers is the minimum for an order estimate. |
| `param_gradient_rel_err` | `< 2%` | Parameter gradients (`omega_0`, `omega_m`, damping, bias direction, `eps_r`, `sigma_e`) vs. high-precision central FD / complex-step; 2% covers FD truncation + step-size window (Phase 3). |
| `bias_reversal_symmetry` | exact sign flip within `reference_polder_rtol` | `kappa -> -kappa`, `S21 <-> S12` under bias reversal is the defining non-reciprocity; an exact algebraic symmetry, not a fitted tolerance. |
| `multigpu_parity` | inherit plan-02 parity (value/phase/loss/gradient) | Multi-GPU must be bit-comparable to single-GPU within plan-02's registered parity; ferrite adds no new numerics across shards (Phase 4). |
| `ferrite_free_perf_regression` | `< 1%` AND zero extra kernel launches / state tensors on ferrite-free scenes | The gyromagnetic path is fully gated behind `gyromagnetic_enabled`; verified by kernel-launch count, not wall time. |
| `high_Q_near_resonance` | pre-registered per-scene budget | Any scene within one linewidth of resonance registers its own looser tolerance + run-length before running; no silent tolerance widening. |

### Change rule (frozen at Phase 0)

Budget numbers may only be **loosened** by pre-registering a named scene with its
physical error budget and the reason (high-Q / near-resonance long-run).
Tightening is always allowed. No gate may be weakened to make a failing test
pass. Reclassifying a gate or altering the sign/unit conventions in sections 1-5
requires an explicit contract revision recorded in this document.

---

## 7. Fail-closed boundary list

Every path below must reject rather than mis-simulate (SI-only, gyrotropy carried
in the ADE state, never widened into `mu_tensor`):

1. `bias_field == 0` (or below a threshold) -- reject; zero bias has no gyrotropy and the local basis is degenerate.
2. `gilbert_damping < 0` -- reject (injects gain / breaks passivity). `saturation_magnetization <= 0`, `gyromagnetic_ratio <= 0` -- reject.
3. CGS / ambiguous units -- SI only; Oe/Gauss only through `from_cgs(...)`, conversion recorded in metadata.
4. Full off-diagonal `mu_tensor` / `sigma_e_tensor` still rejected (`media.py`) -- ferrite must not reach this path.
5. `orientation` matrix still rejected -- bias direction is a vector, not a crystal-frame rotation.
6. ~~Spatially-varying bias -- rejected in Phases 1-3 (uniform 3-vector only); enabled only in Phase 4.~~ **Superseded 2026-07-21 (plan 08 Wave C / G3b).** Mixed / per-material bias now ships: a scene combining multiple ferrite structures with different bias axes, opposed signs on one axis (e.g. a `+z`/`-z` latching circulator), or differing magnitudes/materials runs through the per-cell general-bias path (`fdtd/runtime/gyromagnetic.py`). Because the magnetization ADE is purely local (fields couple only through the ordinary reciprocal Yee curl), a mixed-bias scene is the exact direct sum of independent per-cell passive blocks, each precessing around its own `b̂` at the correct handedness; the compiled `CompiledGyromagneticLayout` already stores a per-cell bias unit vector and right-handed local frame, so a continuously spatially-varying bias field is a data change on the same path. Only Bloch-periodic ferrite remains fail-closed (the real magnetization-ADE correction cannot carry the complex Bloch phase); see `docs/assessments/g3-ferrite-bias-acceptance-2026-07-21.md`.
7. Conformal / subpixel gyrotropic mixing -- rejected; partial fill uses explicit staircase (no scalar averaging of an anti-symmetric tensor).
8. Multi-GPU ferrite -- rejected until Phase 4 lands the component-owner gather/scatter; trainable joint ferrite solve rejected until plan-02 Phase 7.
9. Adjoint of ferrite -- a `GYROMAGNETIC` capability gate rejects reverse until the local reverse kernel lands (Phase 3).
10. Explicit-Euler / dt-shrinking stability escape -- forbidden; the update is implicit midpoint (section 3.3).
11. Resonance / band mismatch -- warn (not silent) when the discrete pole is near the band edge; reject configurations where the discrete pole is unresolvable.

Slices 1b/1c have landed. The compiler lowers a ferrite to
`CompiledGyromagneticLayout` (`compiler/gyromagnetic.py`,
`Scene.compile_gyromagnetic_materials`) and its diagonal background compiles
through the ordinary material path; the FDTD forward advances the local
magnetization-ADE and applies `H -= dM/mu_infinity`
(`fdtd/runtime/gyromagnetic.py`, inserted at the magnetic-update slot in
`stepping.py`). Slice 1c covers the axis-aligned z/x/y fast path with identity
collocation (`C = I`), valid for the transversely-uniform-field regime; the
general-bias 4-point collocation and the arbitrary-bias local-frame rotation (and
a complex-field / Bloch ferrite correction) remain fail-closed until the later
slices. The magnetization state is co-located per transverse component on the
shared Yee overlap; the 4-point gather/deposit `C`/`C^T` of section 5 is exercised
only by the arbitrary-bias kernel.

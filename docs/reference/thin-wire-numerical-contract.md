# Thin-Wire Numerical Contract

Status: Phase 0 frozen contract with accepted Phase 2 topology/adjoint addendum
Scope: energy-paired subgrid current/charge network for Yee FDTD

## Orientation And Continuous Semi-Discrete Equations

For every oriented segment, the incidence matrix `B` contains `+1` at the tail
node and `-1` at the head node. `G` samples the oriented Yee electric line
integral. The same coefficients, with an exact transpose, deposit wire current
into Ampere's equation:

```text
M_e dE/dt = r_E - G^T I
dq/dt     = -B I
L dI/dt   = G E + B^T C^-1 q - R I - ADE_loss
```

`M_e` is the positive electric field mass, `L` is the positive segment self
inductance, and `C` is the positive nodal capacitance. With loss and the ordinary
Maxwell curl exchange removed, the quadratic energy is

```text
H = 1/2 E^T M_e E + 1/2 q^T C^-1 q + 1/2 I^T L I.
```

The two `G/G^T` terms cancel exactly in `dH/dt`, as do the two incidence terms.
Consequently `dH/dt = -I^T R I <= 0` for a passive series law. This transpose
identity is a compiler invariant, not a tolerance-based runtime approximation.

## Leapfrog Schedule And Discrete Energy

Electric field and node charge live at integer time `n`; current lives at
`n-1/2`. The lossless reference step is

```text
I(n+1/2) = I(n-1/2)
           + dt L^-1 [G E(n) + B^T C^-1 q(n)]
E(n+1)   = E(n) - dt M_e^-1 G^T I(n+1/2)
q(n+1)   = q(n) - dt B I(n+1/2).
```

This schedule exactly satisfies the node continuity equation. It conserves the
staggered invariant

```text
H_d(n) = 1/2 E(n)^T M_e E(n)
       + 1/2 q(n)^T C^-1 q(n)
       + 1/2 I(n+1/2)^T L I(n-1/2).
```

Define

```text
A = L^-1/2 [G M_e^-1 G^T + B^T C^-1 B] L^-1/2.
```

The cross-time current term is not positive by itself. Let
`y = [E, q]`, `W = diag(M_e, C^-1)`, and
`K = [G, B^T C^-1]`. On states satisfying the half-step current update, completing
the square gives

```text
H_d = 1/2 ||I(n+1/2) - dt L^-1 K y(n) / 2||_L^2
    + 1/2 y(n)^T [W - dt^2 K^T L^-1 K / 4] y(n).
```

Therefore the full invariant is positive definite under the strict reduced-system
condition `dt^2 lambda_max(A) < 4`; this statement does not assign positivity to
the cross-time term alone.

`A` omits the Maxwell curl because the Phase 0 reference isolates the
`E/q/I` exchange. It is not the complete production FDTD CFL. Phase 1
preparation must validate the combined Yee-curl plus wire operator (or a proven
conservative bound) before stepping. It may not silently clamp the physical
radius or replace it with a cell width.

For finite series resistance, the current update uses the trapezoidal local law

```text
(L + dt R/2) I(n+1/2)
  = (L - dt R/2) I(n-1/2) + dt drive(n),
```

which is passive for `R >= 0`. Rational surface-loss states must use the same
positive-real storage/dissipation accounting and are checkpointed with `I/q`.

## Physical Radius, Coupling Kernel, And Length Assembly

The physical radius is `a`. The coupling distance `d_avg` is the geometric-mean
distance from the wire axis to the support of the regularized current-deposition
kernel. It belongs to `G/G^T`; it is not a replacement conductor radius and it
is not an independently tunable public coefficient. For homogeneous `epsilon`
and `mu`, the only per-unit-length coefficients entering the auxiliary network
are

```text
L' = mu / (2 pi) log(d_avg / a),
C' = mu epsilon / L',
v  = 1 / sqrt(L' C') = 1 / sqrt(mu epsilon).
```

No separate analytical exterior `L/C` is added to the recurrence. The Yee field
receives and returns power through `G/G^T`; adding a second exterior proxy would
double count energy. The grid dependence of `d_avg` is deliberate kernel
renormalization, while the physical radius remains explicit in the logarithm.

For a segment of physical length `ell_s` and a node dual length `ell_n`, the
compiled coefficients are

```text
L_segment = L' ell_s,
C_node    = C' ell_n.
```

The matching row of `G` contains the oriented electric line-integral weights over
the same segment, so `G E` has units of voltage. This segment/dual-node assembly
is the unique production contract. The Phase 0 oracle accepts scalar per-unit
parameters only. Phase 1 may add per-segment radius, but then each node
capacitance must be assembled from its adjacent half-segment contributions; a
segment coefficient may never be paired elementwise with an unrelated node.

## Candidate Decision

Three candidates were evaluated against the energy, topology, and nonuniform-grid
requirements:

| Candidate | Decision | Reason |
| --- | --- | --- |
| Historical edge-radius correction (`0.230 delta` on a cubic grid) | comparator only | It is tied to a uniform edge model and does not define the required charge network or arbitrary-grid coupling. |
| Contour-path field-coefficient correction | not selected | It controls effective radius for its field stencil, but does not by itself provide reciprocal segment deposition and node continuity for the auxiliary network. |
| Kernel-matched auxiliary current/charge network | selected | Its `d_avg` is derived from the actual deposition support; exact adjoint interpolation gives the power identity, and compatible composite kernels give discrete charge conservation. |

The Phase 1 axis-aligned kernel is BS0 in the Yee-staggered direction and BS1 in
the two transverse node-aligned directions. The torch reference computes
`d_avg` from the BS1 x BS1 transverse support. The historical edge radius remains
only a numerical comparator; it is never substituted for `a`.

This choice follows the primary thin-wire formulation and charge-conserving
coupling analysis in
[Composite B-Spline Current Deposition and Interpolation Operators for Thin-Wire FDTD Simulations](https://arxiv.org/pdf/2605.21450),
especially its per-unit coefficients, adjoint power identity, and composite
kernel condition. The contour-path alternative and its grid-convergence tests
are documented in
[An Accurate 2-D Hard-Source Model for FDTD](https://doi.org/10.1109/7260.914307).

The initial production validity band is `a / delta_perp <= 0.2`, plus the strict
requirement `a < d_avg`. The compiler reports validity metadata and rejects a
nonpositive logarithmic term or indistinguishable parallel-wire stencils; it
never falls back to voxelization.

## Differentiation Boundary

Coefficients, physical radius, continuous coordinates within a fixed stencil,
material values, current, charge, and monitor postprocessing remain torch-native.
Graph topology, fragment count, owner assignment, snap decisions, and rational
model order are discrete preparation decisions. Coordinate gradients are valid
only while the compiled stencil remains unchanged.

Phase 2 differentiates physical radius and local isotropic host permittivity.
Centerline-coordinate differentiation remains disabled until the arbitrary-
direction conservative stencil is frozen. A differentiable run uses the fixed
Maxwell step selected before wire compilation; the automatic joint-CFL clamp is
not differentiated because whether it activates is a discrete preparation
branch.

## Phase 2 Graph, Checkpoint, And Exact Reverse

Named endpoints at the same snapped coordinate merge into one physical node.
The compiler emits a single global charge state, deterministic minimum-wire
ownership, per-wire sparse node membership, and one incidence row containing
all branch currents. A closed path merges its first and last occurrence without
creating an open endpoint. Internal revisits, undeclared crossings/touches, and
positive-length overlaps remain errors.

Checkpoint schema version 2 appends `wire_current` and `wire_charge` whenever a
wire runtime is present. Both are deep-cloned device tensors; EMF is derived
scratch and is recomputed from the frozen electric fields. No-wire checkpoints
carry no wire names or storage.

The reverse step is the exact transpose of the ordered Phase 2 forward map:

```text
sample E -> update I -> update q -> deposit I into E.
```

It propagates cotangents through deposition, continuity, node potential,
inductance, and line-integral sampling in reverse order. The coefficient
pullback returns gradients for segment `L`, node `C`, and the electric mass used
by reciprocal deposition; recompiling the same fixed graph maps these to radius
and host material tensors. Standard and CPML Maxwell reverses compose with this
wire transpose. RF/lumped composition remains guarded until explicit wire-port
binding defines shared state ownership.

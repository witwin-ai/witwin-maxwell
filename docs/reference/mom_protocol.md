# MoM numerical protocol

This is the implemented candidate protocol. Cross-project H0/H1/H2 review is
distinct from the local implementation and numerical tests.

```python
import torch
import witwin.maxwell as mw

scene = mw.Scene(
    structures=[mw.Structure(mw.Sphere(radius=0.1), mw.Material(eps_r=2.25))],
    sources=[mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0))],
)
simulation = mw.Simulation.mom(
    scene, frequency=3e8, config=mw.MoMConfig(subdivisions=2)
)
prepared = simulation.prepare()
handle = prepared.solver.handle
result = prepared.run()
points = torch.tensor([[0.2, 0.05, 0.03]], device="cuda", dtype=torch.float64)
E, H = result.query_fields(points, region_id="exterior", field_kind="total")
J, M = result.surface_currents.J, result.surface_currents.M
result.save("surface.pt")
restored = mw.Result.load("surface.pt", scene=scene)
```

The grid in Scene remains available for FDTD compilation; MoM preparation
compiles its own triangular surface. MoM does not allocate a fictitious Yee
field. `fields`, `field`, `tensor`, and volume prepared-scene access reject on
surface results. PointMonitor E/H and explicit physical queries are supported.
Plane/flux/modal monitors that are not compiled by MoM reject explicitly.
SceneModule must materialize the same Scene and follows the same validation.

## Geometry, materials and sources

Core scope is one closed, outward, connected, homogeneous nonmagnetic passive
dielectric in vacuum. Sphere, Ellipsoid and explicitly authored solid Mesh
geometry compile to the same RWG representation. Mesh recentering and thin-shell
semantics are not inferred. Existing none/PML truncation declarations are
compatible with an unbounded MoM exterior; periodic, reflecting, reduced-symmetry,
nonlocal and other unsupported physics fail before assembly.

Use metres, hertz, A/m for J coefficients, V/m for M coefficients and
`exp(-i omega t)`. Conductivity is added once. `k` has nonnegative imaginary
part; impedance is computed as `omega*mu/k` after branch selection. Shared CW
`A*cos(omega*t+phase)` maps to `A*exp(-i phase)`. The unknown is total Cauchy
trace `(n cross H, -n cross E)`, which is nonzero for a nonzero no-contrast
incident wave. Reference accuracy is never inferred from an algebraic residual.

## Fixed-q numerical operations

| Operation | Implemented entry |
|---|---|
| Compile and validate surface | `Simulation.mom(...).prepare()` |
| Inspect mesh/basis | `prepared.solver.handle.basis` |
| Validate a numerical mesh | `mom.mesh.validate_mesh(mesh, regions=None)` |
| Build weak source RHS | `mom.operator.build_rhs(basis, exterior, sources, order=...)` |
| Apply A / Hermitian A* | `mom.linalg.apply_A(handle, x)`, `apply_A_adjoint(handle, y)` |
| Apply test metric inverse | `mom.linalg.apply_test_riesz_inverse(handle, r)` |
| Native residual loss | `mom.linalg.residual_loss(handle, x, b)` |
| Reference solve | `mom.linalg.solve_reference(handle, b, stopping_rule)` |
| Arbitrary-initial-guess correction | `mom.linalg.correct(handle, b, x0, stopping_rule)` |
| Project native currents | `mom.mesh.project_current(source_basis, target_basis, x)` |
| Decode physical fields | `result.query_fields(points, region_id=..., field_kind=...)` |
| Radiation-zone amplitude | `result.far_field(unit_directions)` |

### Exact compact metric storage

`MoMConfig(riesz_storage='compact_rwg_v1')` opts into exact real RWG Gram
storage for the same impedance-balanced test metric
`R = diag(G / abs(eta)**2, G)`. The default `riesz_storage='dense'` retains
the existing full complex tensor. Compact storage retains real CUDA `G` and
its real Cholesky factor; complex residuals use real/imaginary CUDA solves.
This changes storage only, with no mass lumping, metric approximation or CPU
numerical fallback. Dense PMCHWT assembly and the existing CUDA solvers remain
unchanged. It is not a streamed or fast operator backend.

The opted-in handle's `test_riesz` is an explicit
`mom.linalg.CompactRWGRiesz(gram, impedance_squared)`, not a tensor or an
implicitly materialized block matrix. Use `apply_test_riesz_inverse` and
`residual_loss` for either representation. Generic tensor-based
`DenseOperator` callers retain their arbitrary positive-metric interface.
Compact operands must match the operator's CUDA device and real precision,
with a symmetric positive-definite Gram and a positive finite scalar weight.
Gram replacement/in-place changes and scalar-tensor changes invalidate the
metric cache; matrix changes independently invalidate solve caches. Native
coefficient gradients and supported Gram/scaling gradients are preserved.

The storage choice is explicit in prepared configuration, compact operator
metadata and native result persistence. Loading currents reconstructs the
physical decoder without allocating an operator matrix or metric. The
mathematical residual metric and its existing L2-proxy limitation do not change.
At `E` RWG edges in complex128, compact metric plus factor costs `16*E**2`
bytes instead of `128*E**2` for the full metric/factor. Dense direct persistent
storage is then at least `144*E**2` including A and LU; assembly, validation,
factorization and query workspaces require additional measured memory.
This formula does not qualify an unmeasured resolution or physical result.

For data handoffs that do not need an inverse metric, the prepared basis exposes
`mass_blocks()` as one exact `3 x 3` real Gram block per triangle. Both local
axes use that triangle's `face_edges` order. Shared global edges are deliberately
unassembled, so storage grows with the triangle count instead of `E**2`.
`mass_apply(coefficients)` applies the assembled global Gram action directly
from these blocks to real or complex CUDA tensors with the coefficient dimension
last. `mass_matrix()` uses the same blocks for dense assembly. These APIs change
only storage and application: quadrature order, RWG orientation, units (`m^2`),
and the surface-L2-proxy limitation remain unchanged.

Linear-algebra batches put the coefficient dimension last. Matrices are
complex64 or complex128 CUDA tensors. Core work has no numerical CPU fallback.
The residual metric is the impedance-balanced **surface L2 dual proxy**, not a
certified fractional Maxwell trace norm. Small SVD values are diagnostics.
Geometry/material differentiation is outside this fixed-q contract.

GMRES logs true unpreconditioned residuals, absolute floors, failure reason,
restarts and every operator action. Reused LU/preconditioners and optional
initial augmentation have explicit cost and storage logs. Row/column balancing
and an explicitly selected direct fallback preserve physical-coordinate
residual checks; failed primary attempts remain in fallback logs.

## Queries and persistence

Use region `exterior`/`interior` (or 0/1). Exterior kinds are `total`, `scattered`
and `incident`; interior uses `total`. A surface-face point requires
`trace_side="exterior"` or `"interior"` matching its region. Edge and vertex
point traces reject. Query quadrature choices and convergence logs are separate
from the surface equation's algebra tolerance. FDTD's shared query interpolates
each component at its actual Yee coordinates and preserves native DFT phase.
Incident/scattered FDTD queries require a matching empty-background Result with
the same source, sampling and normalization provenance; unsupported TFSF
sampling regions and interface-crossing stencils reject.

Surface persistence stores exact mesh/coefficient arrays, units, order,
configuration and a physical Scene signature. It omits dense matrices and live
autograd graphs. Reload requires the matching Scene and restores CUDA query
capability without re-solving. Different geometry/material/source signatures
or malformed coefficients reject. Array storage is independent of any optional
research dataset codec; an external codec roundtrip is a separate handoff.

## Validation commands

Use the `maxwell` environment and the available CUDA toolkit. In the present
host, `CUDA_HOME` points to the environment's `site-packages/nvidia/cu13`.

```text
python -m pytest tests/mom -q
python -m benchmark mom convergence --levels 1 2 3 4 --order 4
python -m benchmark mom pilot --subdivisions 2
python -m benchmark mom cross
python -m pytest tests --gpu -q
```

The pilot is a cost/algebra measurement, not physical label certification.
Sphere references use independent special-function solutions; nonspherical
and complete MoM/FDTD field/observable convergence remain separate checks.

# Thin-Wire Phase 3 Acceptance

Date: 2026-07-16
Status: accepted
Maturity: E2
Scope: arbitrary-direction coupling, nonuniform grids, fixed-stencil geometry adjoint, and single-device RF port binding

## Delivered

- Every user polyline span remains one physical circuit segment. Grid crossings
  compile into cell-local `WireFragment` coupling records without duplicating
  current/charge state. Segment inductance, capacitance, and conservative
  sampling rows aggregate the fragment contributions.
- Tensor-product conservative line integration supports arbitrary directions on
  uniform, custom, and automatic grids. Sampling and deposition retain the
  exact sorted `G` / `G^T` pairing. Interior paths remain legal under
  real-periodic boundaries; face-touching wraps, Bloch paths, and ambiguous
  topology cases fail closed.
- Continuous coordinates and radii remain PyTorch-native inside a fixed sparse
  stencil. Compile metadata identifies the fixed-stencil gradient boundary and
  records physical-segment and fragment semantics.
- `WireNodeRef` and `WirePortBinding` attach ordinary `LumpedPort` or
  `TerminalPort` declarations to two nodes or to a removed feed gap. Ports use
  the existing `Scene -> Simulation -> Result.port(...)` workflow rather than a
  wire-specific source or result architecture.
- Wire-bound active and passive circuits use the existing device-resident
  circuit state, checkpoint/replay, and reverse path. Pure resistance has an
  exact exponential correction; general RLC uses two midpoint substeps. RF
  observation preserves Yee staggering: the post-correction generalized
  voltage is sampled at electric time and branch current at magnetic time.

## Exit-Gate Evidence

- Compiler, CUDA forward, core adjoint, port runtime/adjoint, rotation, and
  public API coverage passed 88 focused tests. The matrix includes custom/auto grids,
  conservative fragment rows, periodic legality, topology invariance under grid
  refinement, node/gap binding, TerminalPort composition, checkpoint replay,
  profiler guards, and finite-difference gradients.
- A three-grid rotation study compared the same axis-aligned and oblique dipole
  at 32, 64, and 128 cells across the reference span. The rotation gate is
  enforced by `tests/fdtd/thin_wire/test_thin_wire_rotation_acceptance.py`.
  UNEVIDENCED: the impedance and far-field figures previously quoted here
  (`6.56e-4`, `1.42e-3`, `6.41e-2`, `7.54e-3`) have no registered artifact. Unlike
  Phase 2, which registered `thin-wire-phase-2-reverse-performance.json`, Phase 3
  registered no machine-readable result. The numbers must be regenerated into an
  artifact before they can be cited as evidence.
- A smooth full-pulse PEC dipole run closed accepted power against direct
  closed-surface Poynting flux below the registered 1% energy budget; PEC wire
  ohmic loss is exactly zero. The gate is enforced by
  `test_pec_wire_port_accepted_power_closes_against_radiated_surface_flux`.
  UNEVIDENCED: the specific power figures previously quoted here (`6.6332e-5 W`,
  `6.6377e-5 W`, `6.8e-4`) have no registered artifact.
- End-to-end source-amplitude, radius, and continuous oblique gap-coordinate
  gradients agree with central finite differences within the registered 2%
  relative budget at the finest step of each sweep, and coarser steps either also
  meet the budget or show measurable improvement under refinement. The radius
  sweep is truncation-dominated and only converges into budget at its finest
  step; the amplitude sweep sits at the float32 roundoff floor. Exact local port
  VJPs also cover gap weights, electric masses, node capacitance, resistance, and
  drive amplitude.
- CUDA profiler coverage found no `aten::item`, local-scalar extraction, or
  host/device copy in the wire-port stepping and observer path, nor in the wire
  reverse transpose. Runtime state remains proportional to physical
  segments/nodes plus sparse fragment entries, not to the full Yee volume.

## Independent Review

Independent numerical and module reviews checked the physical-segment versus
fragment split, conservative transpose, port voltage/current timing, replay and
reverse seeds, public API composition, and Phase 3 exit gates. Review findings
were corrected before acceptance and locked by focused regressions.

## Boundary Of This Acceptance

- Geometry gradients are valid only while the compiled fragment-to-grid stencil
  is fixed. Crossing a cell boundary is a discrete recompilation event.
- PEC has no material loss. Finite conductivity, passive broadband skin-effect
  state, ohmic-loss production/adjoint, and distributed fragment/state ownership
  belong to Phase 4.
- Single-device execution is accepted here. Multi-GPU value/energy/gradient
  parity depends on the distributed reverse and RF ownership infrastructure
  identified by the Phase 4 dependency contract.

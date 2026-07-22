# Thin-Wire Phase 3 Acceptance

Date: 2026-07-16
Status: reopened-for-evidence (2026-07-18 audit; see "Measured evidence grade" section at end)
Original status (archived): accepted
Maturity: PEC E2 / lossy E0 (measured; plan-level E3 claim not met — see end section)
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

## Measured evidence grade (2026-07-18 audit rollback)

Appended per `docs/assessments/next-functional-audit-2026-07-18.md` §1.5 and §4
(no-inflation rule). The Phase 3 record above is retained verbatim; this section
records the **plan-level measured grade and outstanding debt**. Where it
conflicts with the plan's E3 production claim, this section's grade governs.

- **Measured grade: PEC E2 / lossy E0** (not the plan-level claimed E3). The PEC
  straight / bent / branched / closed-loop paths (forward + adjoint + multi-GPU
  forward, energy-consistent recurrence) hold and are credible E2-grade
  evidence — Phase 3 (PEC, single device) is genuinely E2 in its own scope.
- **Finite conductivity is only a series-impedance model.** Analytic
  skin-effect (scaled-Bessel internal impedance) + passive rational ADE passes
  the 2% analytic gate, but this is a compile-time impedance layer; **the solver
  does not yet consume lossy current physics.**
- **All lossy runtime paths fail closed (lossy is E0).** Lossy current
  recurrence, the `ohmic_loss` monitor, conductivity adjoint and distributed
  wire reverse all reject with explicit errors; trainable wire under multi-GPU,
  distributed CPML/Mur, and wire-with-circuit/network mixing also fail closed.
- **Unevidenced numbers noted.** Some previously cited impedance / far-field /
  power numbers were honestly flagged UNEVIDENCED above (no registered
  artifact); confirmed here as not constituting E2 evidence until regenerated.
- **Evidence required to lift lossy from E0 (convergence route, audit S6; this
  is Wave C, unfrozen only after S1–S3 all pass):**
  1. lossy current recurrence (energy-consistent staggering) + `ohmic_loss` +
     conductivity adjoint landed and consumed by the solver (lossy E0 -> E2);
  2. independent reference: the external reference backend does not cover
     subgrid thin wires (audit §3, row 07), so place analytic skin-effect /
     transmission-line and tag `reference: future-xfdtd`;
  3. convergence report + a wire benchmark scenario entering RESULTS.
- Entry gate: this plan is Wave C solver consumption; the lossy runtime is not
  started before S6 unfreezes (see S0.2 freeze).

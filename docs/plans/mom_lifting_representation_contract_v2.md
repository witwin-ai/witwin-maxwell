# Lifted-current representation and migration decision

Status: implementation contract for the isolated candidate, not an enabled
production capability. The main runtime remains schema v1 and total RWG.
GSP-01 physical acceptance is still open. This decision records the necessary
amendment to the frozen generalized contract before any default switch.

## Authoritative physical state

Store one `scattered_coefficients` tensor y, with explicit J_scattered then
M_scattered order, together with the immutable analytic incident carrier for
that physical RHS. Its direction, polarization, complex amplitude, phase
origin, frequency, exterior medium, units and phasor convention participate
in its canonical identity. Preserve shared PlaneWave vectors and its existing
transverse tolerance; do not silently project the declared source.

Expose `J_scattered`, `M_scattered` and explicit-face
`evaluate_total_currents(points, face_indices=...)`. Total current is analytic
incident trace plus RWG y and generally has no exact coefficient-only total
RWG representation. Reject ambiguous old `.coefficients`, `.J`, and `.M`
access on lifted results with actionable replacement names. An optional
projection is a named derived diagnostic, never the authoritative stored state.

An algebraic RHS tensor without a declared carrier is not automatically a
physical result. Every column in physical multi-RHS storage must own its
carrier identity. Column selection, reordering and persistence must preserve
that association. Replacing an operator's mutable source list must not alter
already prepared physical observations. Current-vector PyTorch gradients remain
supported; source/material/geometry gradients are not implied by frozen records.

## Executable inner snapshot v2

The outer Result schema remains unchanged. The inner surface payload requires
schema 2, `payload_kind=physical_lifted_solution`, `executable=true`, basis,
current kind, coefficient order/units, exact y and dtype, full carrier payload
and SHA256, formulation/kernel/backend/RHS/decoder versions, all integration,
solver and query settings, Scene signature, sampled materials, and body table.

For the present single body, the table records its actual enabled Scene
structure index, region 1, name, vertices/faces, contiguous edge/face offsets,
and explicit disjoint J/M slices. Future multi-body, lattice, harmonic and
volume fields must be versioned before enabling those combinations. Validate
identity and structural metadata before production operator allocation; loading
a result needs geometry/decoder reconstruction, not matrix assembly.

Hash the supplied carrier payload itself, not only a recompiled expected
payload. Python numeric equality does not establish canonical byte identity:
booleans and floats can compare equal. Require supplied hash, expected hash,
and Scene-compiled identity to agree. Exact tensor persistence must preserve
coefficient bytes without projection, renormalization or relabeling.

## One explicit offline archive migration

Schema migration and formulation conversion are different operations. An
analytic incident trace is generally outside RWG, so subtracting its projection
from old total coefficients does not give an exact lifted conversion. Having
the full source declaration does not remove that representability obstruction.

Retain one offline v1-to-v2 archive migration. Validate the supplied matching
single-body Scene and frequency under the frozen v1 contract. Preserve exact
old coefficient bytes, original representation and decoder identities, sampled
media and body table; record the source payload hash and migration version.
Its output has `payload_kind=historical_total_rwg_archive`, `executable=false`.
The lifted runtime rejects it before assembly. Schema 2 alone does not imply
a supported executable formulation. No legacy numerical runtime branch is
introduced by archive migration.

If the historical payload lacks exact decoder/source version evidence, do not
infer it from whichever code is currently installed. Preserve unknown or
unverified identity explicitly with the original available evidence, or reject
migration when that missing identity is necessary to validate the archive.

Creating a new executable lifted result from the old Scene is an explicit new
solve with new result/evidence identity, optionally linked through `derived_from`.
It must not be called an exact coefficient migration or turn old failed evidence
into a passing result. A failed new solve cannot produce a successful lifted
artifact. Historical artifacts remain available through their retained frozen
environment and archive, not silent execution in the new runtime.

Bench must distinguish old total-RWG labels, historical archives and new
scattered labels. Historical archives are not new training targets. New solves
of the same physical Scene remain in the same dataset split group, even though
their representation and evidence identities differ.

## Stage-order amendment and enablement gate

The original contract postponed inner schema changes to GSP-03/GSP-10 and
made GSP-01 a prerequisite to numerical changes. The measured zero-contrast
discretization defect requires a candidate numerical repair during GSP-01.
Isolated candidate implementation and testing are allowed while GSP-01 stays
open. They do not complete it or authorize qualification claims for later stages.

When adopting this representation, explicitly replace the old production
implementation and inner loader; do not retain an implicit compatibility mode.
Before merging that switch, require the versioned public current/persistence
contracts, Bench label handling, source-bound regression and independent
nonzero weak, lossy, resonance, trace and power convergence gates. Main runtime
v1 behavior remains unchanged until that enabling commit. Fast-backend stages
are not automatically advanced by this decision; any resource-driven ordering
change needs its own measured obstruction and recorded dependency decision.

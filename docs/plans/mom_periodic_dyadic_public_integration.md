# Public periodic dyadic integration

Status: implementation contract, not implemented or physically qualified.
Reviewed against Maxwell `4b7461d` by an independent read-only agent.
The GSP objective and every acceptance threshold remain unchanged.

## Required public behavior

Keep `Scene -> Simulation -> Result` as the only public execution path.
Declare a homogeneous background material in Scene and compile it at the
simulation frequency. Compile explicit physical Maxwell plane modes into a
`TensorIncidentCarrier` bound to that exact sampled exterior. A mode declaration
must include physical q [1/m], E [V/m], H [A/m], origin [m], and frequency;
dimensionless modal eigenvalues must never be interpreted as physical q.
Validate both curl equations, passivity, precision and Bloch compatibility.

The background declaration must participate in Scene identity and persistence.
Every other backend must either implement it consistently or explicitly reject
it before numerical preparation. Existing material-grid background scaling is
not a substitute for this physical declaration. Existing vacuum PlaneWave
behavior remains a separately identified scalar case; a tensor mode must not
be coerced into its real direction/polarization representation.

## Coherent implementation boundary

Implement declaration, compilation, dense execution, field recovery and exact
Result restoration together before claiming a public dyadic solve capability.
The existing scalar-exterior Ewald path remains separately identified.

- `scene.py`, `sources.py`, public exports and scene signatures: background and
  physical mode declarations with explicit backend admission.
- `mom/physics.py`, `compile_surfaces` and `MoMRuntime`: sample the exterior
  once and bind all operators and carriers to that same medium.
- `mom/periodic_config.py` and `MoMConfig`: explicit dyadic kernel selection,
  spectral/primary/regular budgets and supported backend dispatch. The existing
  `dyadic_integration` option alone means tensor interiors, not tensor exterior.
- `periodic_dyadic_operator.py`: compose primary plus nonprimary exterior and
  local interiors; build the analytic lifted RHS without periodic image forcing.
- A corresponding decode handle and `SurfaceData` dispatch: use
  `periodic_dyadic_fields.py`, preserve total/scattered fields, local interior
  ownership, translated-cell phase and explicit trace sides/half jumps.
- `SurfaceData.snapshot/restore`, Result persistence and Bench lifted state:
  restore an executable decode state without assembling or solving again.

Dense integration is an implementation milestone. H/H2 and FMM require actual
dyadic construction, action, adjoint and field integration afterward; they do
not inherit the scalar-exterior backend qualification. Fixed surface order
must not be reported as converged adaptive field recovery. Unsupported power,
far-field or modal observables must reject before returning a misleading result.

## Three-dimensional spectral contract

A nonzero homogeneous Maxwell plane mode compatible with a real 3D Bloch
sector lies on the background Maxwell spectrum. It cannot be used as an
ordinary nonresonant forcing for that same resolvent. Adding strict passive
loss does not create a compatible nonzero real-Bloch propagating mode.

Keep the ordinary forced-resolvent pole rejection. Define projected/eigen/modal
quantities separately, including their forcing compatibility, null-space
freedom, normalization and tolerance. Existing finite projected modal values
are not a general scattering solution or a spectral-rank proof. This distinction
does not remove 3D periodic Green functions or anomaly semantics from the goal.

## Identity and acceptance obligations

Use a new snapshot schema; do not relabel scalar-exterior schemas 16/17.
Bind formulation, real kernel representation, backend, RHS and decoder versions;
sampled exterior/interior tensors and reciprocity; frequency and precision;
lattice/Bloch; carrier and regular-kernel payloads; primary, regular, surface,
query and solve budgets; geometry, body ordering, all-J/all-M ordering and raw
coefficients. Restoration must rebind owned state and retain numerical guards.
Saved caches or convergence receipts cannot grant future admission.

Targeted checks must cover public multi-body execution, nonzero tensor contrast,
actual kernel dispatch, source/Bloch rejection, contribution ownership, current
gradients, traces, translated fields, exact save/load and corrupted identities.
Use scalar degeneration as one independent comparison, not as proof of general
tensor physics. Preserve numerical failures and separate routing/algebra checks
from physical accuracy. Consolidate full regression, cross-repository acceptance
and independent audits at the stage merge and final acceptance.

The existing frozen GPU jobs remain unchanged: periodic merge regression v2 at
`bec7a76` and reciprocal full-tensor 1D source-plane v2 at `838e69e`. Their results
cannot cover this future public integration or the later persistence commits.

## Hierarchical public integration boundary (2026-09-28)

The selected nonprimary and complete body-pair primitives are committed as
`763825a` and `de7e310`. The global selected provider and explicit H/H2 residual
proxy pass targeted internal controls, including refined far compression.
These internal objects do not enable public backend admission. The coarse
60-by-60 control needs rank 40 at relative Frobenius tolerance 1e-8 and retains
exact leaves; it cannot be evidence of compression savings.

Public H/H2 integration must change the following as one coherent boundary:

1. Admit linear `hmatrix` and `h2` only with the checked public dyadic periodic
   configuration and complete hierarchical numerical settings. Keep HB, FMM,
   3D incident-pole and modal-observable obligations explicit. The lossy image
   reference used by internal algebra controls cannot become the public kernel.
2. Construct `PeriodicDyadicSystemBlocks` with the same sampled exterior,
   primary/interior kernel objects and checked regular kernel owned by the
   decode handle. Build H from selected blocks and H2 from that verified H
   partition. Never construct a dense global matrix to enable this dispatch.
3. Share analytic tensor RHS and field/trace decoding with the dense physical
   path. Preserve the carrier and local interior ownership when constructing
   or converting the hierarchical operator. Refresh must validate both physical
   bindings and hierarchical factor/tree/metric bindings.
4. Version the public operator identity for the hierarchical tensor exterior,
   including the explicit vacuum-impedance residual proxy policy, all H/H2
   budgets, regional integration/cache policy and regular-kernel payload.
   Internal scalar-named H metadata cannot serve as the physical identity.
5. Use a distinct hierarchical tensor-periodic snapshot schema, keeping the
   dense schema18 admission strict. Restore H/H2 numerical configuration before
   computing the expected representation and operator identity; reconstruct
   owned decode-only state without assembly, factor construction or solve.
   Extend Bench schema selection and exact operator/carrier validation together.
   A physical coefficient snapshot is not a resumable compression-factor cache.

Targeted integration must demonstrate public multi-body prepare/run for both
backends, actual retained far factors/couplings where compression is admissible,
original operator residual sampling and current gradients, shared decoded
fields, and exact native/Bench replay with construction/solves forbidden.
Reject swapped backend/metric/config/kernel identities, including jointly
rehashed inconsistent metadata. Separate manufactured routing/storage controls,
actual scalar degeneration, and nonzero full-tensor physics. Consolidate full
regression at the merge point; none of these additions closes existing physical,
complexity, resource, guard-budget or final cross-repository gates by itself.

## Public hierarchy implementation checkpoint (2026-09-28)

The boundary above is implemented in uncommitted development code: checked
linear H/H2 dispatch, shared physical decoder/RHS, sampled operator identity
with fixed residual proxy, native schema 19, and Bench exact replay. Six targeted
routing/storage checks pass in `/tmp/gsp-dyadic-hierarchical-public-v1`; four
H/H2 nodes use manufactured selected blocks and zero contrast, while two nodes
cover affected dense replay. Independent review reports no confirmed P1/P2.
This closes only those routing/storage contracts, not the numerical acceptance
obligations above. Actual nonzero-contrast scalar-limit H2 integration is running
in an isolated frozen tree with 1194 pinned source/test/config hashes and explicit
module-root assertions (`/tmp/gsp-dyadic-hierarchical-actual-v1`). Coarse exact
leaves are permitted for this control and cannot demonstrate compression.
The older full-tensor dense integration and periodic merge regression remain
live in their original frozen trees. No existing failure gate is relaxed.

The frozen actual H2 control subsequently terminated during preparation after
530.559 seconds: the source-plane budget passed, but six off-plane queries failed
the initial shells 8/16 check (about 6.94e5 to 7.01e5 versus 0.0025). All 1194
hashes remain unchanged. The adapter currently applies max_shells only to plane
estimation; bounded off-plane refinement requires follow-up. No solve, storage
or query acceptance was reached; retain the original failed receipt.

## Combined periodic and linear integration checkpoint (2026-09-28)

The isolated Maxwell integration tree is based on the committed linear public
path c25d017; the paired Bench tree is based on committed linear execution and
population reports 1648aa3. Shared compilation and schema18/19/20 dispatch are
being validated together. This engineering merge does not grant physical or
resource qualification. Targeted manufactured routing/storage controls must
remain identified as manufactured.

Bounded off-plane refinement is implemented and selected production Duffy nodes
pass the unchanged component budget with an independent scalar Ewald reference.
The subsequent actual H2 attempt still failed source-plane extrapolation: its
last consecutive differences were 7.27913 and 0.605085 against 0.01. Independent
Ewald height refinement indicates deeper levels are needed; a selected actual
cross-body diagnostic remains running with the unchanged tolerance. The older
full-tensor dense run also terminated during preparation with an off-plane
budget failure. Neither attempt reached solve, field or state acceptance.
All failure evidence remains in the execution checkpoint and root plan.

The frozen older periodic full regression remains live and does not validate
these combined sources. Do not launch a duplicate full suite while it is live.
Final current-source four-repository regression, general tensor/periodic/HB
physics, 32-by-8 and larger datasets, complexity and two independent final
audits remain required by the original plan.

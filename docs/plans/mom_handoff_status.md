# MoM completion and project handoff status

Status reviewed on 2026-09-12, after the local delivery archive was frozen.
Local MoM development, source-aware regression coverage, independent numerical
and algorithm audits, and archive verification are complete. The original
development goal and the cross-project research plan remain incomplete.

This versioned ledger records the current decisions in the workspace Maxwell,
master, Bench and EMFC plans. Those four workspace plans are outside this Git
repository. It does not claim that Bench, EMFC or Baselines work was implemented
or accepted during this documentation update.

## Completed local work

Checked rows apply only to their stated local scope. External handoff acceptance
is recorded separately below; optional and gated work is not silently omitted.

- [x] MW-01: SI/phasor/material/source contract, scope rejection and weak RHS.
- [x] MW-02: closed mesh, RWG orientation/divergence, L2 proxy metric,
  projection, remeshing and native structural payloads for H1 packaging.
- [x] MW-03: actual double-RWG PMCHWT, all five panel-pair classes, CUDA
  assembly, quadrature refinement and documented independent physical samples.
- [x] MW-04: actual A/A*, metric inverse, complex inner products, fixed-geometry
  current-vector PyTorch gradients and precision comparisons.
- [x] MW-05: arbitrary-initial-guess scalar GMRES, actual block GMRES,
  reusable LU, true residuals, failure/fallback policies and measured costs.
- [x] MW-06: regional total/scattered/interior E/H, one-sided traces,
  near-singular queries, complex far field/RCS and independent volume power.
- [x] MW-07 local scope: Scene/Simulation/Result integration, native exact
  persistence, CUDA execution, independent sphere/ellipsoid samples,
  three-level FDTD comparisons, three-level 32x8 resources and actual streamed
  forward/adjoint/solve feasibility.
- [x] MW-08 local scope: regular TE/TM, control-energy normalization, complete
  198-column response and cutoff/angular/spatial calibration.
- [x] MW-10 archive scope: source/environment/configuration, original failures
  and passes, Git history, offline replay and independent package verification.
- [x] Stage commits, affected baseline repairs, full primary regression plus
  affected/new-node reruns, and independent numerical/algorithm/artifact audits.
- [ ] H0/H1: common versioned schema and actual consumer contract acceptance.
- [ ] MW-07 H2/G0: specified FEM, representative coverage, formal population,
  precision/resource freeze and joint numerical acceptance with Bench.
- [ ] MW-08 H5: downstream model/baseline worst-source comparison, actual
  active-sampling rounds and joint report.
- [ ] MW-09: optional mixed Calderon branch after H2; not started.
- [ ] MW-10 extension: select one physical extension after G2, then implement
  and independently validate it; not selected or started.

Production fast operators are not delivered: the streamed prototype repeats
quadratic integration. Bench's actual scale/cost qualification determines the
next acceleration requirement. Scalar initial augmentation is not a full
iteration-by-iteration recycled method; the real block solver supplies the
current block/recycled alternative. Shape/material derivatives and the separate
extended-physics backlog remain outside the completed core scope.

The implemented scope is one closed homogeneous passive nonmagnetic dielectric
in vacuum. The metric is an explicitly named surface L2 proxy. No broad-domain
accuracy certificate, fractional trace-norm certificate or speedup is inferred.

## Handoffs and gates

| Handoff or gate | Current status | Next owner and acceptance requirement |
| --- | --- | --- |
| H0 | Producer protocol exists; common freeze pending | Bench MB-01 consolidates versioned schema; EMFC EF-01 and model consumers verify semantics |
| H1 | Producer structures available; consumer acceptance pending | MB-01 packages mesh/RWG/sign/metric/complex RHS fixtures; EF-02, MB-02 and ML-02 consume them |
| H2 / G0 | Not passed; local numerical delivery available | MB-03 specified nonspherical FEM and at least 10% representative Mie/FEM coverage; MB-04 formal 32x8 population, precision and resources; Maxwell jointly closes numerical findings |
| H3 | No qualified initial research dataset | After H2, MB-05 freezes groups, qualified labels, failures and hashes using EF-03 or a verified exact-array path |
| G1 | Not passed | After H2+H3, ML-05 real supervised models, structural controls and remeshing; Bench evaluation |
| H4 / G2 | Not passed | ML-04–07 and MB-06: strong traditional baselines, actual models, risk calibration, paired experiments and full equal-accuracy costs |
| H5 / G3 | Not passed; numerical modal component delivered | MB-07/08 and ML-08/09: worst-source/modal comparisons and four actual active-sampling rounds |
| G4 | Not passed; local source/evidence archive delivered | Select extension after G2; MW-10, MB-09 and ML-10 complete its independent evidence and the full research archive |

The final sampled sphere and ellipsoid measurements meet their documented
budgets. Auxiliary electric-dipole collocation and FDTD are not the required
FEM reference. Existing resource cases are not a label-qualified formal core
population. The reference-to-neural-error ratio must still be computed against
MB-04's frozen quantities, norms and budgets (at most one quarter, preferably
one tenth). The final mesh for the reported sphere accuracy is 7680 RWG edges;
1920 edges fail that same 0.5% physical target.

The modal 0.2154% error is a weighted far-response operator measurement. Only
25 of 198 individual relative column errors meet 0.5%; the added-band ratio
is 0.03441% under a 0.05% budget. These results do not certify every weak column,
the current-space worst-source error, or an infinite modal tail.

## Evidence and commits

The effective inventory contains **3877 nodes: 3873 ordinary passes, three
XFAIL and one non-strict XPASS**, with no unresolved failures or ordinary skips.
This is source-aware coverage from the complete primary run and later affected
and new-node runs, not a single full-repository rerun at final HEAD. The fresh
numerical/public subset passes 184 tests. All 797 final numerical/test source
hashes are independently verified; later changes are documentation only.

| Stage | Initial or additional implementation commits |
| --- | --- |
| MW-01 | `af00b7b` |
| MW-02 | `8c06652` |
| MW-03 | `3cc1ab9` |
| MW-04 | `47b3ea0` |
| MW-05 | `dd11b3a`, actual block `d7caceb`, block driver `dfe91a3` |
| MW-06 | `e42d68c` |
| MW-07 | `e79b5e5`, `ce0e25a`, streamed consumer/provider `c030a19` / `d7caceb`, streamed CLI `bd07610`, resource qualification `58b3ec3` |
| MW-08 | `f0328f0` |
| MW-10 archive | `7afafb1` |
| Final numerical audit / closure / digest | `d270696`, `b17cc43`, `1ffd10b` |

The staged history contains later fixes and explicitly recorded dependencies;
these short hashes do not assert every intermediate commit is a standalone
final acceptance checkout. Use the exact packaged source revision for replay.

The frozen archive is `benchmark/cache/sie/witwin-maxwell-local-acceptance-20260912.zip`:
118,363,104 bytes, 4017 payload members plus one manifest. Its source revision is
`b17cc430982c25c317797dc97c49258cdad7c6ee`; documentation commit `1ffd10b`
records the completed digest and independent audit.

```text
SHA256 da09d12dd115c825c8ce43a0ee985287bffdd1449798e7da6450c89de5f04915
```

See the [local acceptance closure](../assessments/sie_local_acceptance_closure.md),
[artifact identities and independent audit](../assessments/sie_delivery_artifacts.md),
[stage record](mom_delivery.md) and [reproduction guide](../reference/mom_reproduction.md).
This status update does not change that ZIP, its evidence directories or the
historical plan snapshots inside it. It does not mark the original goal complete.

## Recommended next project

**Make Maxwell-Bench the next primary project; develop EMFC EF-01/02 in parallel.**
If only one project can be advanced at a time, Bench closes the current H2/G0
critical path: independent references, representative coverage and budget
qualification. Storage alone cannot close those conditions.

1. MB-01/02: create Bench, consume the frozen package and
   [producer protocol](../reference/mom_protocol.md), freeze minimal H0 and
   package H1. Implement sample/source/query semantics, grouping, fixture versus
   reference states, hashes, explicit failures and restartable orchestration.
2. MB-03/04: reuse the independently archived Mie and cross-backend evidence
   within its supported scope, add the specified nonspherical FEM and required
   representative coverage, qualify the actual core 32x8 population, freeze
   precision and compute budgets, and jointly pass H2/G0. Reference preparation
   can run alongside schema and orchestration work.
3. In parallel, EF-01/02: inventory legal exact-record capabilities and verify
   complex128 J/M, mesh, signs and metadata roundtrips plus malformed-data
   rejection against H0/H1. Prefer existing exact paths; a dedicated RWG primary
   is optional. Existing Yee SurfaceData and Maxwell's own persistence are not
   proof of an accepted EMFC RWG roundtrip.
4. After H2, EF-03 plus MB-05: validate real qualified-label roundtrips and
   matching residual/physical queries, then deliver grouped and versioned H3
   data. A verified independent exact-array path remains a valid alternative.
5. Baselines ML-01/02/03/04 engineering and traditional adapters may proceed
   earlier. Actual ML-05/06 training waits for H2+H3; H4/G2, H5/G3 and the selected
   physical extension follow their original acceptance dependencies.

The workspace currently contains the Maxwell and EMFC repositories but no
Bench or Baselines project directories. No new implementation in those projects
was started by this status review, and EMFC GPU codec acceptance was not rerun.

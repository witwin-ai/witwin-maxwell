# Bench provider epoch v3 and consumer metadata requalification

Planning record, 2026-10-30. No GPU execution, collection campaign, bundle generation or test changes are authorized by this document alone. Review stops below precede future execution.
Work only in Maxwell `sie/provider-epoch-v3` and Bench `epoch/provider-v3`; Bench below means `<workspace>/wt/bench-epoch-v3`. This change adds only this plan; no public capability changes or FEATURE_LIST update.

## Stop A decisions (user, 2026-10-30)

- Single-device v3 excludes exactly the three two-device nodes listed in section 2; retain them as unqualified multi-device coverage. Replace the two-device requirement with reviewed single-device shards and exact aggregation.
- pFFT is source-hash coverage only, outside v3 numerical qualification. Its five dedicated test modules and the pFFT parameter of the near-preconditioner test stay in the full catalog, outside the admitted numerical catalog.
- Consumer metadata is outside this task. The PCB session owns the section 5 full rebaseline after v3 passes; do not modify its maintained scripts or public test.
- Up to eight GPU-hours on physical GPU 1 are approved in principle. **First round stops after host collection, self-review and cost audit; the user must review the concrete catalog and say resume before any GPU shard.** Record any unavailable/stale originals or budget overrun instead of excluding further tests.

## Revised dense-core/default scope (user, 2026-10-30; supersedes earlier execution scope)

- Select the 27 distinct modules of the executed v2 catalog (already including the public API file), plus exactly `test_fmm_config.py`, `test_fmm_scalar_near_config.py` and `test_right_preconditioner_hook.py`: 30 modules. Keep all three two-device nodes unqualified; only the cache test is in these selected modules.
- Named existing controls are `test_linalg.py::test_batched_rhs_restarts_and_zero_rhs_absolute_floor`, `test_linalg.py::test_budget_and_stagnation_never_report_success`, `test_fmm_config.py::test_stagnation_window_defaults_to_restart_length` and `test_fmm_config.py::test_hierarchical_inner_tolerance_requires_dual_budget_margin`.
- H/H2/direct-H2, general FMM, tensor, periodic, volume/harmonic/static, open surfaces, pFFT and near-field sparse LU retain source hashes and explicit family exclusions. The named default/right-hook controls do not qualify those backends generally. No radial-order bridge or PCB rebaseline dependency is admitted.
- `test_spatial_validation.py` is outside this scope and stays untouched. Freeze the installed core, its genuine distribution metadata and all file bytes below the execution root; collect and every run resolve that copy first through PYTHONPATH, record its actual version/source SHA, and never read live core again.
- Only physical GPU 1 lanes A/C are authorized; never acquire lane B or all locks. Each hold/process stays <=885 s/20 GB. Larger indivisible jobs stop. After successful recollection/review and a <=8-hour estimate, execute without another approval round; stop on any blocker and always before refinement at Stop B.

## Collection root-cause repair (user, 2026-10-30; supersedes the spatial test edit restriction)

- Defer spatial parametrization mesh creation to test execution, preserving all labels/cases, function bodies and tolerances; commit this test-only change separately. Keep spatial tests outside the admitted catalog, and leave the nested entrypoint control and every v1/v2 harness/control unchanged.
- Recollect and review the changed tests inventory, then execute the same 42 bounded A/C shards. Supply the CUDA 13 toolkit variables on each invocation that needs them. Audit/package only complete successful evidence, and stop at Stop B.

## Dense-v4 corrections and handoff (user; supersedes prior artifact admission)

- Change only the stale quadrature error regex to `Panel quadrature failed tolerance` in its own test-only commit. RuntimeError behavior, case geometry, budgets and tolerances stay unchanged.
- Exclude exactly `tests/mom/test_fmm_scalar_near_config.py::test_all_old_payload_and_construction_bytes_and_strict_scalar_dk_versions` as artifact-unavailable and unqualified. Keep its source hashed and the other node in that module admitted; do not supply or regenerate the missing original pair.
- Recollect/review into a fresh dense-v4 directory, provide exact rerun commands, and stop. The user reruns the entire revised scope from the main session; earlier passed shards never join it. Aggregate, audit and package only on a later resume.

## 1. Exact source boundary and numerical delta

BOUNDED_V2 is source `5093aa2032360ae353b0d034ee754230ce1da93e`, packaged by `96b3e7e32b9fbae06b69919ad83a354d3b3bde6e`; both have MoM tree `4c0a385deb36e366cd85e0086e6dedcc6be4af17`.
The proposed source is `0c364a40c3a6888f661aa9ea50d355b81b908a28`, MoM tree `6f739365f561f35b57d6380e928690666eba577e`. Later harness-only commits must preserve that numerical tree or restart review.
The exact net MoM delta is **204 files: nine modified, 195 added, none deleted**. The complete filename inventory is in section 7. There are **376 MoM-touching commits**, including merges; do not describe this as just the recent S-1/L-1 changes.
These read-only commands define every commit/file association without a selected-history approximation:

```bash
git log --reverse --format='%H %s' --name-status 5093aa2032360ae353b0d034ee754230ce1da93e..0c364a40c3a6888f661aa9ea50d355b81b908a28 -- witwin/maxwell/mom
git diff --name-status 5093aa2032360ae353b0d034ee754230ce1da93e 0c364a40c3a6888f661aa9ea50d355b81b908a28 -- witwin/maxwell/mom
```

The second command's UTF-8 output has SHA256 `0aa17d00f8bc59b5a9f80ad6488ae4bc0e911d8c759d667e2021b78c737e6ee0`. Full package coverage is larger: **369 paths under `witwin/maxwell` (98 modified, 271 added)**, including public orchestration, material/source compilation and native code. A MoM-only allowlist is insufficient for admission.
Important correction: the **executed v2 catalog** has 362 nodes in 27 modules, including combined-direct storage, matrix-free and compact-Riesz tests, but **no H/H2/FMM modules**. Those backends were added later. Reusing the directory scope at HEAD expands the campaign; old v2 results never qualify these additions.

| Effect on retained dense/combined paths or new H/FMM coverage | Commits and files under `witwin/maxwell/mom/` |
| --- | --- |
| Changes shared representation and execution; requires fresh dense results | `8c0a8a3f` compact mass blocks; `fcb8dd2a`, `009231c9` incident lifting, scattered-current results and migration; `1a44ad81` operator protocol/Krylov; `34659b76` coupled bodies; `07601da4` magnetic media. `runtime.py`, `operator.py`, `physics.py`, `mesh.py`, `modes.py`, `observables.py`, plus `carrier.py`, `lifting*.py`, `riesz.py`, `kernels.py`, `topology.py`, `migration.py`, snapshot modules. |
| Changes finite-precision matrix/RHS/field evaluation on existing closed bodies | `08b7c6fb` bounded pair classification; `19566c1a`, `8c222a68`, `cf27dd50`, `a024d114`, `e687319c` field batching, analytic static extraction, stable edge logs and finite convergence metrics. `operator.py`, `fields.py`, `quadrature.py`, `ordered_reduction.py`. Near fields may change beyond roundoff. |
| S-1b/performance work reaches dense and compressed closed surfaces | `f107c590`, `a74b1506`, `aa1ed997`, `0f771f45`, `2a267626`: share multi-medium quadrature, batch lifted RHS/fields, bound touching workspace and restore compressed RHS recording. `kernels.py`, `operator.py`, `carrier.py`, `fields.py`, `lifting.py`, `lifting_fields.py`, `runtime.py`; preserve mathematical operators, allow reduction-order differences. |
| S-1c/d quadrature changes numerical rules, not merely optional geometry | `02dc4b78`, `9e03c437`, `51c5c270`: gap-aware unresolved near pairs, anisotropic touching rules and empty-pair handling; selected H/H2/FMM near blocks now use the touching rule too. `operator.py`, `quadrature.py`, `lifting.py`, `nearfield.py`. Smooth accepted product pairs retain their rules; slivers/gaps can change materially. |
| Shared geometry preparation | `c9877af4` CUDA spatial pruning (`mesh.py`, `spatial.py`), then `54364704` edge/topology classification (`mesh.py`, `topology.py`, `open_surface.py`). Closed-surface exact predicates should retain decisions; setup time/memory changes. |
| H/H2/direct-H2 and FMM are new since v2 | H: `86ddae82`, `66c9af86`, `3effe537`, `27f3c646`; H2: `dc7773cb`, `5f02a0fb`; direct H2: `38ec0c57`; public FMM: `3b120585`, compact storage `23b9177c`/`955e6d12`, then shared/reciprocal/streamed/signed-axis/radial variants through `122f1e78`. Files in section 7's H/FMM groups. No historical H/FMM equivalence claim. |
| Shared iterative solver changes | `c62477a3` residual restart after progressed finite Arnoldi closure; `a4eb33c2` flexible right-preconditioner hook (`linalg.py`, `preconditioning.py`, `operator_protocol.py`, handles). Hook is opt-in; scalar GMRES closure behavior affects existing solves. `63e5c695` batches FMM near assembly (`fmm_regional.py`, `nearfield.py`, `packed_near.py`). |

**Default-changing merge `015cf41c` (implementation `9dafc627`) must be qualified explicitly:**

- (b) `fmm_config.py`: `table_storage='compact_shared_displacement_v1'`, `shift_storage='separable_v1'`, `near_storage='scalar_reciprocal_dk_v1'` replace dense/dense/full-matrix defaults. Configuration payloads now retain explicit storage/workspace fields. Expect different retained memory, configuration identity and roundoff; do not transplant old order/resource receipts.
- (c) `runtime.py`: H/H2/direct-H2/FMM require `stopping.rtol <= 0.1 * solve_dual_tolerance`. Incompatible configurations now reject at construction; this does not silently tighten their tolerance. Dense direct and pFFT are outside this particular condition.
- (d) `linalg.py`: omitted `stagnation_window` becomes `restart` (formerly 20). With default restart 40, iteration histories and convergence/failure outcomes can change; explicit windows stay explicit. Direct LU results should not change because of this setting.

Opt-in additions are outside the retained Bench closed scalar dense method, although shared integration/dispatch edits above still need review:

- Open sheets/EFIE/Calderon: `54364704`, `febeda59`, `a7e5fae5`, `12c0624c`, merge `e56b59fb`, empty chunks `d44f2fd8`; `open_surface.py`, `efie_open.py`, `efie_preconditioning.py`, `single_layer.py`, runtime dispatch. Primitive meshing `ff21faa3` adds `primitive_mesh.py`; it does not select open-sheet physics automatically.
- pFFT: `1d7378ce`, `4703e938`, `306e1827`, `67fc1888`, `7fa64a98`; prepare merge `9d762a1e` / `482f8e0d`; lifting merge `0c364a40` / `21e0ab03`. All seven `pfft_*.py`, plus dispatch and prepare/preconditioner batching. Explicit backend selection; no change to the Bench dense runner merely by importing these files.
- Near-field sparse LU: merge `36f650e4`, implementation `807c9bc3`, removal of the other variant `de3fe8e3`; `near_preconditioner.py`, `preconditioning.py`. Explicit right-preconditioner choice, no new default. Its factorization allocates outside Torch accounting.
- Tensor, periodic, volume, nonlinear/harmonic/DC paths: `8ae92c98`, `812c5678`, `1b910792`, `dc43f26c`, `bf432850`, `2453d50a`, `f73d8934`, `3d2e48c5` and subsequent commits in the exact log. Section 7 enumerates every file; selected scalar nonperiodic Bench cases do not exercise those capabilities. Their shared-file edits are not exempt from source review.

## 2. Harness v3 and qualification scope

Copy `benchmark/sie_combined_full_acceptance_v2.py` to `sie_combined_full_acceptance_v3.py`; keep v1/v2 and their bundles immutable. Use BOUNDED_V2 as the explicit predecessor, not the old three-file e95 allowlist.
`REVIEWED_DELTAS` must pin SHA256 of **all 369 net package additions/modifications at the frozen source**, including all 204 MoM files. Pin the exact predecessor and candidate inventories, including added paths; review the full package/native diff before freezing literals. There are no MoM deletions; enumerate any full-package deletions explicitly. Keep the binding helper's reviewed bytes if used only for import binding; its historical e95 equivalence checks are not v3 equivalence evidence.
Use new protocol/acceptance versions, `<scratch>/witwin-maxwell-provider-epoch-v3-<source-short>/` as a proposed future detached execution root and distinct collect/run output paths. Freeze actual core sources, distribution metadata, native sources/build and harness/test/helper inventories. Host inspection here found the pinned Torch tuple unchanged (`2.13.0+cu130`, CUDA build `13.0`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`), but installed core is **0.4.3**; do not reuse old core receipts.
Preserve collect -> independent catalog/source review -> actual run -> independent terminal/readback -> engineering and terminal reviews -> package/commit. Counts, runtime-file count, archive members and receipt SHA come from v3 evidence, never copied v2 constants. Preserve all 22 required direct-storage allocation witnesses unless review establishes a changed exact node contract.

Recommended case coverage is the existing `tests/mom` plus `tests/api/public/test_public_api.py` acceptance scope, with an explicitly reviewed **single-device qualification boundary**:

- Carry the v2 dense/direct cases: compact/full Riesz, eight RHS, repeated sealed restore, fields, current gradients, persistence, both precisions, existing sphere/ellipsoid/harmonic assembly controls. Keep original numerical tolerances and original oracle/provider identities.
- Admit existing current H/H2/direct-H2/FMM cases for original weak residuals, adjoints, current/field agreement, capacity/configuration and persistence. Use existing default/configuration and GMRES controls for (b)(c)(d), lifted-RHS batching and near/touching integration. Do not add a broad new test suite or refactor tests.
- Full collection contains **three known two-device nodes**: `test_numerical_audit.py::test_gauss_cache_resolves_implicit_cuda_device_before_lookup` and both dtype instances of `test_successor_robustness.py::test_actual_compact_dense_solver_matrix_and_cross_device`. They cannot pass with only physical GPU 1 visible. At review stop A approve their exact exclusion as unqualified multi-device coverage, or leave v3 blocked. Do not count skips as passes or simulate another device.
- Freeze the full collected catalog and the exact admitted single-device node list separately. Review any further artifact-dependent skips before launch; existing GSP original-data readers require pinned per-module input policies and helper closure. Never regenerate an oracle as an implicit fallback. Preserve excluded nodes/reasons in the receipt; require zero failures, skips, XFAIL/XPASS or deselections inside the admitted list.
- V3 needs a reviewed single-device/sharded protocol: replace the hardcoded `expected_device_count == 2`, and bind each invocation to the same source/core/catalog and the exact disjoint assigned nodes. Aggregate phase/JUnit/allocation evidence without gaps or duplicates. Update the v3 gate to verify this new scope explicitly; do not weaken historical v1/v2 verification. If review requires the old literal full-directory/no-exclusions protocol, the available hardware blocks execution.

Expected comparisons: dense direct true residual/restore invariants should pass unchanged, but lifted representations, quadrature and field integration rule out blanket bitwise equality with v2. Recompute mesh differences. H/FMM memory and Krylov histories can change under (b)(d); (c) may expose invalid test/consumer configurations that need explicit reviewed tolerances. No relaxed success thresholds.
**Keep pFFT outside Bench v3 numerical qualification and the 6/12/24 method.** Hash its source and retain any admitted existing small pFFT tests as engineering-only coverage; that is not a pFFT provider accuracy/resource receipt. Its separate acceptance drivers, approximate action/admission model and recent lifting changes deserve a later backend qualification. Do not spend this epoch's budget rerunning L-1 scale campaigns, open-sheet campaigns or sparse-LU comparisons.
Known quadrature limits remain explicit: successive near-order estimates can understate true error (~1.2–1.3e-5 for a 1e-5 target); mild creases can retain fixed-order touching error (~3.3e-4). See `mom_open_surface_thin_conductor.md` S-1c/d notes; its old statement that all hierarchical touching blocks lack anisotropic routing is superseded by `51c5c270`.

## 3. Single-A6000 budget and scheduling

All new numerical work uses physical **GPU 1 only**, `CUDA_VISIBLE_DEVICES=1`; its process-local logical ordinal 0 refers to physical GPU 1. Physical GPU 0 belongs to another session. No cloud jobs.
Historical measurements below are evidence, estimates are planning reservations; no new performance/memory measurement was made here. Decimal GB denotes 1e9 bytes.

| Stage | Historical observation | Single A6000 reservation / expected peak | Lock treatment |
| --- | --- | --- | --- |
| Source review, collect, catalog, metadata CPU qualification | v2 362 nodes; collection host-only | 10–30 min host per preparation; no GPU allocation | No GPU locks; hide CUDA for host commands |
| Retained small dense/direct acceptance | v2 entire invocation 148.78 s; device peaks 1,138,003,968 / 546,816 bytes | 3–10 min, provision 2–4 GB; two-device topology tests excluded explicitly | One lock per <=885 s shard |
| Added existing H/FMM and remaining admitted tests | No comparable v3 whole-catalog timing; original-artifact availability controls cost | Provisional 2–6 GPU hours total, shards aimed at 5–12 min and <=20 GB; heavy individual nodes need dedicated reservations | One lock for bounded shards; all three for any job >15 min or >20 GB; revise estimate after host catalog/cost audit before approval |
| Fresh core-18 n6, 1080 edges / 2160 unknowns | 23.0 s, 271,814,656 bytes | 1–3 min, 0.3–1 GB | One lock |
| Fresh core-18 n12, 4320 edges / 8640 unknowns | 178.1 s, 2,174,544,384 bytes | 4–10 min, 2.2–5 GB | One lock |
| Fresh core-18 n24, 17280 edges / 34560 unknowns | 2398.3 s, 26,585,339,392 bytes | 40–80 min, 27–35 GB reservation; unchanged measured-allocation cap **29 GB** remains binding | All three locks for one atomic task |
| Strict assessment, package and readback | v2 785 original members / 237 runtime files | 10–30 min host, evidence-size dependent | No GPU locks |

Budget **3–8 GPU hours** initially, conditional on reviewed catalog/costs; no automatic retry allocation. Serial scheduling replaces dual-device observations; v2's tiny second-device peak does not justify doubling its wall time as a performance model. The complete modern catalog could exceed this reservation; stop and reprice rather than silently run it overnight.
Try nonblocking `flock -n` on `<workspace>/wt/cuda1.lock`, then `cuda1b.lock`, then `cuda1c.lock`; hold one free lane for <=15 min and process usage <=20 GB. For larger/longer jobs take **all three nonblocking, all-or-nothing**, releasing partial acquisitions if any is busy. Never wait holding a partial set. Inspect free GPU-1 memory before launch; library sparse LU requires all three locks and free memory > estimated need +4 GB, irrespective of Torch caps. It is not needed by the retained dense method.
No subprocess may outlive its lock. Use <=885 s timeouts for lane jobs; use a separately approved dedicated n24 limit (e.g. 5400s), recording timeout as failure. Split between complete nodes/tasks, not halfway through a solver or between RHS with reused unrecorded state. The Bench runner's `.frequency-worker.lock` is not a CUDA lane lock; lease/run each level separately under the physical locks, retaining one immutable campaign provenance. Wait by PID/output file, never `pgrep -f`.
Every GPU Python invocation must have this form (lock owned by its parent for the full process lifetime):

```bash
env CUDA_VISIBLE_DEVICES=1 [OTHER=VARS] python ...
```

No bare Python, shell-variable prefix, chained GPU commands or physical GPU-0 probes. Record total device free/used memory as well as Torch allocated/reserved peaks, native workspace, RSS, sealed host backup bytes, transfers, restore and failed-phase costs. A 35 GB scheduling reservation does not raise the 29 GB acceptance cap.

## 4. Bench migration and fresh refinement

After v3 evidence is reviewed and committed, add `PROVIDER_V3 = Epoch(...)` in `maxwell_bench/provider_acceptance.py`, version `bench-native-provider-engineering-gate-v3`, acceptance `native-combined-engineering-acceptance-v3`, bundle `docs/assessments/combined-direct-full-v3`. Fill original detached root, SHA, counts, exact scientific deltas and admitted scope from final evidence. Register it in `EPOCHS`; retain COMBINED_V1/BOUNDED_V2 and historical default aliases unchanged.
Generate **`cases/further-frequency-refinement-v4-ready.json`**, not another v3 file: source and bundle commits, provider receipt, complete runtime/core identities, supersession reason and recalculated configuration/file hashes. Preserve geometry, sources, queries, 8 RHS, 6/12/24 meshes, order 4, integration controls, 0.005 mesh screen and 29e9-byte cap.
Move current execution in `further_frequency_pilot.py` to that plan and epoch; update `PLAN_PATH`, `PLAN_SHA256`, `PLAN_CONFIGURATION`, source dependencies and actual runtime capture. Resolve its `ROOT.parent/'witwin-maxwell'` assumptions through the existing `repository_paths.provider_root()`/explicit source roots: the two task worktrees are not siblings. Never accidentally import or write the main checkout. Retain frozen historical provenance readers.
Update the existing provider gate/runner tests (`tests/test_provider_acceptance.py`, `tests/test_further_frequency_pilot.py`) for the new live receipt while preserving historical cases. Any new scope/shard aggregation regression and metadata regression together are limited to **1–2 targeted new controls**; use existing acceptance tests otherwise. No full repository regression run.
`joint_bounded_evidence.py` hardcodes `5093aa2` and a two-file transition. Keep it and `tests/test_joint_bounded_evidence.py` as historical readers. A new v3 joint transition/consumer must consume its immutable predecessor view plus the exact v3 runtime and fresh run; it must not repoint the bounded constants. Preserve joint v8 and `portable_evidence.py`'s historical plan digests; add a separately versioned current evidence export/report only when requested. Update live roadmap references (Bench `docs/plans/workspace/{maxwell_master,maxwell_bench,witwin_maxwell}_development_plan.md`) after actual results, and record the new boundary in `g0lite_provisional_qualification.md` without relabelling its frozen provisional epoch.
Run all three levels freshly into a new campaign, proposed `runs/further-frequency-provider-v3`; nominal historical sum **2599.4 s = 43.3 min**, reserve 45–95 min plus host I/O. Never mix old currents/meshes into the new convergence sequence. Preserve original 5/10/20 and prior 6/12/24 failures/successes as predecessor evidence only.
The n24 memory blocker is **closed at BOUNDED_V2**, not merely projected: Bench `docs/assessments/bounded-epoch-stage-v1/ACCEPTANCE.md` terminal table and `docs/assessments/joint-bounded-progress-v8/README.md` report 26,585,339,392 <29,000,000,000 bytes, residual `3.740901108909912e-10`, final mesh difference `0.004517474374781277`. Diagnosis is **in Bench**, `docs/assessments/n24-memory-diagnosis-v1/README.md`: old peak 38,245,576,192; whole-matrix checks/Gram transients caused it, not LU duplication. Use that directory's `provider_bundle_readback_v2.py` as the packaging/readback pattern, adapted to actual v3 paths/counts.
The raw `runs/further-frequency-formal-combined-v2` directory is not present in this Bench worktree; recover its exact source-bound evidence through the configured evidence root before claiming a reopened chain. The committed terminal documents supply the historical estimates here, not a newly verified raw run.
Strict assessment must reopen new arrays, storage restoration, cost and provenance at the v3 epoch. Passing engineering/refinement establishes neither the full 32x8 population at v3 nor external-reference qualification. That broader migration is a separate reviewed campaign; **H2/G0 stays not passed**.

## 5. Consumer metadata: rebaseline without rewriting history

`tests/api/public/test_consumer_metadata.py` reads `consumer-preparation-final.provenance.json` under the 2026-10-28 archive, then selects that record's **baseline**, not its output. The real anchor is `/tmp/pcb-current-order-reuse-candidate-20261027.json` (SHA `06ffa5e98aa86c901775089da7db1f64d80e01cd352ef90a5abf6cd142722dbc`) and matching provenance (SHA `8489064c62f265f7312ec4a62b30ceb8843e54576a9612039034341ba42c4aeb`). Its source is `/tmp/pcb-production-d6804b8-core373af-20261027`, commit `d6804b804e1826461c42eca579575f091b7c267a`.
That preserved **consumer** has 187 MoM files. At the proposed source, **18 are modified and 17 added**. `compatibility()` correctly refuses before derivation; a new overall package hash cannot authorize a numerical change. The raw-witness provider is a third, older epoch, already different from that consumer in 23 MoM files. These are not the Bench BOUNDED_V2 comparison.
Git provenance: `ce966c73486c6858d5ddb5d110b97901c7ac220d`, 2026-10-28 06:11:56 -0700, **author and committer Xingyu Chen**, introduced the archive, maintained derive/closure scripts and public tests. `18a77ee9` (06:30:05, same human identity) added final launch qualification evidence. Git establishes ownership; it does not identify an unrecorded tool/operator behind the archived runs.
The archive README and `derive_consumer_bridge.py` prescribe host-only derivation only across unchanged protected MoM/files/AST/core scopes, with actual imports and genuine wheel metadata pinned. They explicitly do **not** implement numerical requalification or automatic baseline advancement. The historical consumer loader itself is byte-identical to today's `scripts/dev/measure_mom_fmm_scaling.py`; preserving its contract is necessary but insufficient.

Minimal correct **current-consumer requalification**, after explicit source review:

1. Keep the old archive and all 455 historical pins (including the 24 witnesses), raw records, nested original selections/provider hashes and old costs immutable. A Bench v3 PASS is relevant source evidence, not replacement radial-series/resource qualification.
2. Establish a **new independently reviewed consumer baseline**, source-frozen with the accepted current MoM tree, protected shared scopes, actual core **0.4.3** inventory and distribution METADATA. This is an explicit rebaseline/bootstrap operation, not `derive(old_baseline)` with the refusal removed. Review its changed numerical source and closed-loader selection contract; retain the old-to-new delta and evidence links. If this cannot be established, live derivation must continue to fail.
3. Write new bridge/provenance and a new dated archive, e.g. `docs/assessments/fdtd-consumer-metadata-bridge-2026-10-30/`, using fresh execution paths. Supply the baseline schema actually read by `derive`: `source_root`, matching `current_package_before/after`, `actual_core`, pinned inputs, output hash and false native-action/production flags. Do not reuse the newer derivation report as the old baseline format; they differ. Rebuild module inputs/helper closure against the same source and preserve relative helper pins.
4. Change the test's baseline reference to that reviewed archive; keep strict MoM/protected-scope comparisons and the changed-numerical-file rejection. Assertions must reflect the new baseline: do not require outer header hashes to differ when source is identical, nonempty shared-file changes when none occurred, or exactly 455 *total* pins after adding lineage. Keep the original 455/24 lineage separately and bind actual new baseline pin counts. Retain false provider/production qualification and unchanged original records/selections.
5. Run **only `tests/api/public/test_consumer_metadata.py`** for this repair with CUDA hidden, preserving its CPU subprocess checks, JUnit, source/core/input before/after identities and no-CUDA-initialization record in the new archive. No FDTD field/reference campaign or broad regression follows from a metadata repair.

If only historical regression coverage is desired instead, the smaller valid test-only repair is to run positive derivation against the frozen compatible consumer and assert that live HEAD is rejected. That does **not** requalify current metadata; choose it explicitly at review stop A rather than silently changing the success contract. Do not skip/xfail the current failure or patch old MoM hashes to make it green. This task makes neither repair.

## 6. Execution steps and review stops

1. **Now:** commit only this plan on `sie/provider-epoch-v3`, human author/committer Xingyu Chen. No GPU runs, no bundles, no test edits.
2. Prepare v3 harness and host inventories on the designated branches; pin the exact source/native/core delta, resolve worktree paths, original artifacts and module-local policies. Collect only after that implementation is separately authorized. Inspect node costs and the full/admitted catalogs without CUDA initialization.
3. **Stop A — before any GPU:** user reviews delta/defaults, single-device exclusions/sharding, pFFT exclusion, consumer rebaseline versus historical-test scope, catalog SHA, available original inputs, per-job time/memory reservations and total budget. Unavailable originals or an insistence on dual-device acceptance blocks execution; report it, never substitute.
4. Freeze the reviewed implementation commit and future detached execution root; independent catalog review binds exact bytes. Run serial bounded acceptance shards under GPU-1 locks, preserving failures and per-node phase/allocation evidence. Stop on a failure or budget overrun; a source fix requires a new collection/review, not edited evidence.
5. Audit and package v3 with runtime-source-map, original-evidence.zip/member inventory, exact readback, two review documents and receipt. Commit only on the authorized branch; no merge/push. All old absolute-path bundles remain untouched.
6. Prepare Bench Epoch/ready-plan v4/runner changes and targeted existing gate/runner acceptance, plus the reviewed new consumer archive/test repair. **Stop B — before refinement GPU:** user reviews concrete provider bundle/readback, Bench source/plan hashes, host acceptance and n24 all-three-lock reservation.
7. Execute fresh n6, n12, then n24 as complete tasks at one frozen provider/core/Bench epoch. Review lower-level costs/results before taking the dedicated n24 reservation. Stop if the 29 GB cap, numerical criteria or original provenance fail; preserve cost and request a newly bounded follow-up.
8. Independently assess/publish only the new scoped evidence; CPU-qualify the metadata repair, commit staged artifacts on their current branches, and report exact commits/files, measured costs, acceptance results and unresolved qualification. **Stop C:** user reviews the terminal evidence before any expanded population/backend campaign. No full regression, cloud, merge or push.

## 7. Complete net MoM file inventory

All names below are relative to `witwin/maxwell/mom/`. The nine modified files are `fields.py linalg.py mesh.py modes.py observables.py operator.py physics.py quadrature.py runtime.py`; every other name is added. Families locate review work; optional modules can share infrastructure with covered paths, as section 1 explains. The exact log above supplies every intervening commit, including commits whose intermediate edits were later superseded.

Shared closed-surface paths (18):

```text
carrier.py fields.py kernels.py lifting.py lifting_fields.py linalg.py mesh.py modes.py observables.py operator.py operator_protocol.py
ordered_reduction.py physics.py quadrature.py riesz.py runtime.py spatial.py topology.py
```

H, H2, direct H2 and original near blocks (28):

```text
cluster.py direct_capacity.py direct_config.py direct_operator.py direct_regional.py direct_system.py h2.py h2_basis.py h2_capacity.py
h2_operator.py hierarchical_blocks.py hierarchical_runtime.py hmatrix.py interpolated_far.py interpolation.py interpolation_error.py
interpolation_kernel.py interpolation_order.py lowrank.py nearfield.py nested_h2_snapshot.py nested_moments.py operator_error.py
packed_near.py rwg_interpolation_error.py rwg_moments.py rwg_refinement.py scalar_moment_action.py
```

FMM, including optional directional/periodic/storage variants (32):

```text
fmm_capacity.py fmm_compact.py fmm_config.py fmm_directional_fft.py fmm_directional_moments.py fmm_directional_rwg.py
fmm_directional_shift.py fmm_error.py fmm_ewald_fft.py fmm_ewald_real.py fmm_ewald_spectral.py fmm_ewald_translation.py
fmm_expansion.py fmm_fft.py fmm_fft_coefficients.py fmm_moments.py fmm_operator.py fmm_order.py fmm_periodic.py
fmm_periodic_capacity.py fmm_periodic_error.py fmm_periodic_operator.py fmm_periodic_order.py fmm_periodic_regional.py
fmm_periodic_system.py fmm_regional.py fmm_rwg_error.py fmm_shift.py fmm_storage.py fmm_system.py fmm_translation.py
scalar_near_channels.py
```

Explicit preconditioners (3):

```text
block_preconditioner.py near_preconditioner.py preconditioning.py
```

Open surfaces and primitive meshing (4):

```text
efie_open.py efie_preconditioning.py open_surface.py primitive_mesh.py
```

pFFT (7):

```text
pfft_config.py pfft_coupling.py pfft_efie.py pfft_grid.py pfft_lifting.py pfft_operator.py pfft_system.py
```

Periodic and Ewald support (17):

```text
ewald_nearfield.py ewald_refinement.py periodic.py periodic_config.py periodic_cylindrical.py periodic_diffraction.py
periodic_dyadic.py periodic_dyadic_fields.py periodic_dyadic_nearfield.py periodic_dyadic_operator.py periodic_fields.py
periodic_hierarchical_runtime.py periodic_limits.py periodic_nearfield.py periodic_operator.py periodic_runtime.py periodic_spectral.py
```

Tensor support (16):

```text
anisotropic_fields.py anisotropic_lifting.py anisotropic_operator.py complex_static_tensor_green.py dyadic_contrast.py dyadic_green.py
matched_spd_green.py spectral_linalg.py tensor_carrier.py tensor_harmonic_bundle.py tensor_harmonic_preparation.py
tensor_harmonic_runtime.py tensor_harmonic_snapshot.py tensor_medium.py tensor_nearfield.py tensor_volume.py
```

Volume/harmonic/static, snapshots and additional physical helpers (79):

```text
action_harmonic_balance.py background_bank_snapshot.py compressed_background.py compressed_harmonic_snapshot.py
compressed_volume_preparation.py conductive_static.py coupled_field_refinement.py coupled_original_check.py
coupled_response_snapshot.py coupled_volume_action.py dipole.py dual_action_snapshot.py dual_current_action.py dual_current_material.py
dual_current_receipt.py dual_current_response.py dual_preparation_snapshot.py dual_static_current.py dual_static_snapshot.py
fixed_tangent_inverse.py harmonic_balance.py harmonic_config.py harmonic_convergence.py harmonic_current.py
harmonic_current_snapshot.py harmonic_cutoff_receipt.py harmonic_dual_bridge.py harmonic_dual_current.py harmonic_fields.py
harmonic_jacobian_action.py harmonic_material.py harmonic_mixed.py harmonic_mixed_snapshot.py harmonic_observables.py
harmonic_preparation.py harmonic_runtime.py harmonic_snapshot.py harmonic_state.py implicit_surface_solve.py incident_work.py
interior_volume_hmatrix.py linear_runtime.py linear_volume_state.py magnetostatic.py migration.py nonlinear.py
physical_surface_snapshot.py single_layer.py static_coupling.py static_current.py static_current_projector.py static_floating.py
static_magnetic.py static_magnetic_snapshot.py static_magnetic_volume.py static_mixed.py static_projector_snapshot.py
static_tensor_floating.py static_tensor_green.py static_tensor_interface.py static_tensor_material.py static_tensor_polarization.py
static_tensor_preparation.py static_tensor_snapshot.py static_tensor_volume.py static_volume_curl.py streamed_dual_state_snapshot.py
surface_h_snapshot.py surface_volume.py surface_volume_hmatrix.py vie_operator.py volume_basis.py volume_h2.py volume_h2_snapshot.py
volume_h_snapshot.py volume_hmatrix.py volume_mesh.py volume_response_matrix.py volume_source_refinement.py
```

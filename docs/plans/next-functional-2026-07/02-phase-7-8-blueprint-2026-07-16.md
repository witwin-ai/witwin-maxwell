I now have everything needed. Composing the final audit + blueprint.

# PART A — Audit of plan 02 Phases 0–6 as landed on origin/master

All paths below are under `/home/xingyu/code/witwin/witwin-maxwell/.worktrees/integration/` unless absolute. Test evidence: `tests/fdtd/multi_gpu` run today on both A6000s (maxwell env, CUDA_HOME export pattern): **114 passed in 15.18 s** — matches the tracked acceptance record ("109 → 114" in `docs/reference/fdtd-multi-gpu-joint-solve.md:281-284`).

## A.1 Phase-by-phase exit-gate verdicts

**Phase 0 (contracts & baselines, E0) — PARTIALLY MET (joint-solve half only).**
- `FDTDParallelConfig` frozen and structurally validated: `witwin/maxwell/fdtd_parallel.py:83-119`. Partition/ownership for all six Yee components complete and unambiguous: `fdtd_parallel.py:245-680`; exhaustive pure-Python tests `tests/fdtd/multi_gpu/test_partition.py` (13 tests, incl. uneven partitions, terminal-node ownership, single-owner interface, frozen dataclasses).
- Baseline record (hardware, topo, per-workload one-GPU ms/step, warmup protocol) lives in `docs/reference/fdtd-multi-gpu-joint-solve.md:236-309` with the `bench_joint.py` harness (`scripts/dev/fdtd/multi_gpu/bench_joint.py`).
- **NOT delivered:** the ensemble half of the Phase-0 contract. `MultiGPUExecution`, `run_many`, `DevicePool`, `ExecutionPlan`, `ResultSequence`, `ExecutionRecord`, `DistributedFailure` do not exist anywhere in `witwin/` or `tests/` (grep returns only `DistributedResultManifest` in `fdtd/distributed/persistence.py:353`, which is the sharded save manifest, not the ensemble layer). Memory estimation exists only as joint-solve DFT/gather capacity preflight (`fdtd/distributed/capacity.py`).

**Phase 1 (Ensemble MVP, E2) — NOT LANDED.** No executor, no ordered `run_many`, no device lease, no ensemble tests. Nothing to audit.

**Phase 2 (RF-aware ensemble, E2) — NOT LANDED.** `NetworkRunManifest` exists from plan 01 (`witwin/maxwell/network_sweep.py:15`) but executes serially through `PreparedNetworkSweep`; there is no multi-GPU executor for it to plug into.

**Phase 3 (partition + P2P microbenchmark, E0) — MET within the declared 2-GPU hardware boundary.**
- Topology preflight: `fdtd/distributed/transport.py:82-103` (indexed CUDA devices, bidirectional peer per neighbor, no host fallback) plus result-device peer checks and homogeneity in `fdtd/distributed/solver.py:474-498`.
- Preallocated plane exchange on dedicated streams/events: `transport.py:105-144`; persistent magnetic receive halos allocated at prepare (`solver.py:557-559`).
- Tests: `test_transport.py` — tagged asymmetric halo round-trip, allocator-growth-zero check (a Python-level proxy; **Nsight proof of no host staging was not produced** — honestly declared at `fdtd-multi-gpu-joint-solve.md:304-309`). 3/4-GPU halo round-trip is covered only at partition-metadata level; the host has 2 GPUs. P2P bandwidth 52.65–52.69 GB/s both directions recorded in the acceptance doc (not re-measured today; timing excluded).

**Phase 4 (standard real-field forward MVP, E1) — MET for 1↔2 GPU; the 3-GPU leg of the gate is structural only.**
- Bounded kernels: registered ops `fdtd/cuda/backend.py:3467-3488`; implementations in `fdtd/cuda/kernels/electric.cu` / `magnetic.cu`. Critically, the full-domain legacy wrappers delegate to the bounded implementations with the full x range (e.g. `update_electric_ey_standard_cuda` → `..._bounded_cuda(0, x_extent, 0, x_extent)`, electric.cu:2035-2051), so single- and multi-GPU paths cannot drift. Internal-face physics is suppressed via `effective_x_low/high_mode = BOUNDARY_NONE` computed from `global_x_offset`/`global_x_extent` (electric.cu:1985-1989).
- Eager `DistributedFDTD` with overlapped and serialized-safe schedules: `fdtd/distributed/solver.py:658-764`; CUDA Graph explicitly rejected (`solver.py:983-984`).
- Parity: `test_solver_acceptance.py` (uniform + uneven nonuniform grid, source on interface), `test_numerical_matrix.py` (zero-field, Gaussian impulse six fields, six-component FieldTimeMonitor, multi-frequency full-field DFT, magnetic Lorentz ADE crossing the interface, early shutoff with identical step counts). Assertions implement exactly the plan §7.2 gates (`max_abs <= 2e-6`, masked `max_rel <= 2e-5`, monitors `rtol 5e-5 / atol 5e-6`) against an independently constructed single-GPU `FDTD` reference, with nonzero-reference assertions — not tautological, no loosened tolerances found.
- Memory distribution + positive speedup: recorded in the acceptance table (strong 1.97385×, memory factor 1.01058 for 257³) — recorded evidence, not re-run (timing excluded by instruction).

**Phase 5 (forward engineering preview, E2) — PARTIALLY LANDED: a guarded subset with genuine E2 evidence; several enumerated items exist only as fail-fast guards.**
Delivered: (1) shard-aware compile via per-shard local scene construction (`solver.py:160-198` — local `GridSpec.custom` slices, internal faces set to `"none"`, no global-tensor-then-slice); (2) CPML/PEC/PMC/Mur global-face ownership incl. slab CPML memory accounting (`test_capacity_cpml_stats.py`); y-periodic/y-symmetry parity (`test_review_regressions.py:159-207`); (3) plane/flux/mode monitor tiling+stitching (`fdtd/distributed/monitor_merge.py`, 5 numerical cases in `test_monitor_numerical.py`, ownership tests in `test_monitor_merge_ownership.py`); (4) sharded DFT + manifest persistence (`fdtd/distributed/persistence.py`, `test_sharded_persistence.py`, `test_sharded_export.py`); (7) electric/magnetic ADE shard-local with interface-crossing parity.
Not delivered (fail-fast instead): (5) plane/mode/TFSF/beam sources — only `PointDipole(profile="ideal")` + `UniformCurrentSource` (`solver.py:462-472`); (6) periodic/Bloch ring on x (`solver.py:375-382`); (8) RF ports (`solver.py:391`); (9) NetworkBlock/circuit reduce-scatter; (10) aniso/nonlinear/SIBC (guards at `solver.py:440-444, 544-552`; modulation forward accepted serialized-only without acceptance claim). This matches the support matrix in the reference doc; the doc does not over-claim.

**Phase 6 (forward production hardening, E3) — NOT LANDED, and correctly not claimed.** No Nsight Systems/Compute artifacts (host lacks them — declared at `fdtd-multi-gpu-joint-solve.md:304-309`), no break-even estimator/single-GPU auto-fallback (the below-break-even 129×65×65 case is documented but nothing selects one GPU), no per-shard compute CUDA Graph. Status remains "engineering preview" (`FEATURE_LIST.md:244-246` mirrors the evidence boundary). Note the plan's own fallback ("keep engineering preview and publish profiler evidence") is only half-satisfied: preview status yes, profiler evidence absent.

**Phases 7–8 — NOT STARTED**, confirmed by `simulation.py:779-784` (blanket trainable rejection), `solver.py:330-333` (NCCL raise), and the doc's support matrix rows for adjoint/NCCL.

**Discipline findings:** no tautologies (independent single-GPU references, nonzero-signal assertions), no cherry-picked tolerances (test constants equal plan constants), numbers in the acceptance record are labeled with hardware and honestly bounded; the one test-count claim (114) reproduced today. Main audit deltas vs the plan: **the entire ensemble track (Phases 0-ensemble-half, 1, 2) is absent from master**, and the Phase 3/4 "2/3/4-GPU" legs are qualified only structurally beyond two devices.

## A.2 Distributed runtime catalog (file:line map)

- **Public config / partition metadata:** `witwin/maxwell/fdtd_parallel.py` — `FDTDParallelConfig` :83; `FDTDHaloRegion` :122; `FDTDComponentLayout` :143 (owned/allocation slices, `global_origin`, receive/send regions built at :491-598); `FDTDShardLayout` :245 (`storage_cell_owned`/`storage_node_owned`, `physical_faces`); `FDTDPartitionPlan` :320 (balanced physical-cell intervals :468-477, PML cells pinned to outer shards :627-635, terminal node to last shard :634-635).
- **Shard layout semantics:** rank>0 shards carry one low ghost cell (storage slices start at 1, `solver.py:603-618`); rank<last shards carry one high electric ghost node. Ownership: `Ex/Hy/Hz` by cell interval, `Ey/Ez/Hx` by low node interval + terminal node on last shard.
- **Halo exchange (`fdtd/distributed/transport.py`):** `HaloTransport` ABC :34-52 (preflight/exchange_electric/exchange_magnetic/teardown); `CudaP2PHaloTransport` :55. Electric: right shard's first owned node plane `Ey/Ez[storage_node_owned.start]` → left shard's ghost `Ey/Ez[storage_node_owned.stop]` (:105-123). Magnetic: left shard's last owned cell plane `Hy/Hz[storage_cell_owned.stop-1]` → right shard's ghost views `halo_hy_low/halo_hz_low = solver.Hy[0]/Hz[0]` (:125-144). Event ordering: producer `*_ready` recorded on compute streams, copies on destination's high-priority communication stream, `*_received` gates the boundary-strip kernel (`solver.py:671, 701`).
- **Time step (`fdtd/distributed/solver.py:766-818`):** per step: modulation clock + magnetic-ADE advance → `exchange_electric` → H update (overlapped :658-674 interior/boundary split, or serialized :711-730 full-tensor native incl. CPML) → magnetic sources/ADE corrections → `exchange_magnetic` → E update (:676-709 or :732-764) → aniso/SIBC-guard, electric surface sources, point sources, dispersive corrections, PEC/Mur, DFT/observer accumulation. Overlap active only for non-CPML/non-complex/non-modulated (`_overlap_active` :648).
- **Guards:** `_validate_static_capabilities` `solver.py:373-472` rejects Bloch, x-periodic, x-symmetry, **MaterialRegion densities (:385-389)**, ports (:391), closed-surface/diffraction/flux-time/non-point-time monitors (:392-408), Ex-on-split x-planes (:423-439), SIBC (:440-444), material monitors (:445-450), non-point/non-uniform sources (:462-472). NCCL reserved-but-raises at :330-333. Hardware checks at run-prepare: `_validate_hardware` :474-498. Trainable scenes rejected earlier at `witwin/maxwell/simulation.py:779-784` (called from `prepare()` :458 and `_run_fdtd()` :656); `DistributedFDTD` itself has no trainable guard (defense-in-depth gap worth closing in Phase 7).
- **Monitor merge:** `fdtd/distributed/monitor_merge.py:364` `merge_sharded_monitor_payloads` — scene declaration order, single-owner point/x-normal payloads, y/z-normal planes stitched from owned x intervals (`_stitch_owned_component` :63, `_merge_tiled_plane` :300, ghost discard tested in `test_monitor_merge_ownership.py:153`), flux recomputed from stitched global components on result device.
- **Output/persistence:** gather `solver.py:850-935` (owned-slice copies into a global tensor on result_device); capacity preflights `fdtd/distributed/capacity.py` (invoked `solver.py:951-980`); sharded export/manifest `fdtd/distributed/persistence.py:195, 353` with `Result.save_sharded/load_sharded`; stats `solver.py:1043-1099` (topology, partitions, halo bytes, per-device peaks, truthful `None` phase timings).
- **Numerical core sharing:** every shard is a full `FDTD` instance over its padded local scene (`solver.py:536-543`); bounded launches pass `localXBegin/End` + `globalXOffset/Extent` (`solver.py:201-313`); source single-writer crop `fdtd/distributed/sources.py:64` plus interface Ex control-volume correction `fdtd/distributed/source_corrections.py`.

---

# PART B — Blueprint for Phase 7 (joint-solve adjoint, E2) and Phase 8 (single-node NCCL, E2)

## B.1 Mathematical frame — verified against the forward code, with one correction

The prompt's frame is correct with one ordering correction: in `_advance_one_step` the electric halo is exchanged **at the start of the step, before the H-update** (not after the E-update of the same step). One forward step is:

```
halo-E (ghost_Ey/Ez^L := P · Ey/Ez^R)  →  H-update  →  halo-H (ghost_Hy/Hz^R := P · Hy/Hz^L)  →  E-update
```

Its transpose, per reverse step:

```
E-update^T  →  halo-H^T (AdjHy/Hz^L[last owned cell] += P^T · AdjHy/Hz^R[ghost 0]; ghost := 0)
            →  H-update^T  →  halo-E^T (AdjEy/Ez^R[first owned node] += P^T · AdjEy/Ez^L[ghost node]; ghost := 0)
```

so the reverse-halo is **accumulation into the owner followed by ghost zeroing**, exactly as framed, and the halo-E^T accumulation is the *last* communication of a reverse step (it feeds AdjE consumed by the next-earlier step's reverse).

The decisive implementation finding: **the existing fused native reverse kernels are already interface-correct when launched per shard on the padded local tensors**, given a "ghost adjoint planes are zero at kernel launch" invariant:

- The single-GPU reverse step is three phases (`fdtd/adjoint/native.py:48-229`): Phase 1 `_reverse_electric_h{x,y,z}_standard` (AdjE_post → AdjH_mid), Phase 2 `_reverse_magnetic_e{x,y,z}_standard` (AdjH_mid + AdjE_post → AdjE_prev + grad_eps, x/y/z boundary modes), Phase 3 decay pullback.
- Phase-1 kernels gate every E contribution with `is_e*_active_index` (`cuda/kernels/adjoint.cu:66-80`) which excludes **tensor x-face nodes**. On a right shard this yields, at ghost cell 0, exactly the cross-interface term T(first owned node) — the plane to ship left and accumulate; on a left shard the ghost-node contribution is automatically excluded (arrives via communication instead).
- Phase-2 kernels (`adjoint.cu:382-458`) at a tensor face with mode NONE write `AdjE_prev[face] = AdjE_post[face] + Σ H-curl-transpose terms`; with the ghost `AdjE_post` zeroed, the left shard's ghost node receives exactly the cross-term ∝ `hz_curl·inv_dx_h·AdjHzMid[last owned cell]` — the plane to ship right. On the right shard, the `i>0` branch reads `AdjHzMid[ghost 0]`, which is zero after the halo-H^T send-and-zero, so no double counting. The grad_eps term at the interface node reads *forward* `hz_mid[ghost 0]` — a real forward value, available because replay refills forward ghosts.
- Therefore **no new CUDA kernels are required for the standard real path**. The work is orchestration: phase-split dispatch, two transposed transports per step, ghost-zero invariants, distributed checkpoints/replay/seeds, and the global parameter reduction. (CPML reverse kernels need the same face-semantics verification — listed as a risk below.)

Replay side: `_replay_segment_states` (`fdtd/adjoint/core.py:3642`) and `_step_state` (:2986) are Torch-only state-dict operations, already accepted as bit-compatible with the native forward for the standard/CPML real path (:3610-3639). Distributed replay = lockstep per-shard `_step_state` with two Torch plane copies per step (cross-device `copy_` in one process goes P2P); ghost writes by the full-tensor Torch update are garbage-but-overwritten, same as the forward serialized path.

## B.2 Phase 7 — file-by-file implementation map

Existing single-GPU adjoint entry points (anchors):
- `witwin/maxwell/simulation.py` — `_run_fdtd` :655-664 routes trainables to `_run_fdtd_with_gradient_bridge` :1060; blanket rejection `_reject_trainable_parallel_fdtd` :779-784, invoked at :458 (prepare) and :656 (run); prepare-time backend validation :460-463.
- `witwin/maxwell/fdtd/adjoint/bridge.py` — `run_fdtd_with_gradient_bridge` :925; `_FDTDGradientBridge` :222; `_validate_supported_configuration` :327; `_run_forward_with_checkpoints` :394; `forward` :498; `_backward_impl` :585 (segment replay :704-715, seed injection :722-723, `reverse_step` :748, grad accumulation :770-808, material pullback :813-849).
- `witwin/maxwell/fdtd/adjoint/dispatch.py` — `_select_reverse_backend` :484; `validate_native_adjoint_preparation` :597; `reverse_step` :630.
- `witwin/maxwell/fdtd/adjoint/native.py` — `_reverse_step_standard_native_core` :48; `_reverse_step_cpml_native_core` :278; kernel wrappers `fdtd/cuda/backend.py:2351-2560`.
- `witwin/maxwell/fdtd/checkpoint.py` — `checkpoint_schema` :124, `FDTDCheckpointState` :183, `capture_checkpoint_state` :209 (fields + expanded CPML psi + electric/magnetic ADE + TFSF aux + lumped scalar state).
- `witwin/maxwell/fdtd/adjoint/seeds.py` — `_build_output_seeds` / `_apply_seed_runtime` (imported at bridge.py:33-35).
- `witwin/maxwell/fdtd/material_pullback.py` — `pullback_material_input_gradients` :266; `pullback_density_gradients` :201; `_pullback_region_density` :83 (assumes the region slice covers the *full* region — the reason the distributed density guard exists).

Changes/creations:

1. **`fdtd/distributed/transport.py`** — add to `HaloTransport` + `CudaP2PHaloTransport`:
   - `exchange_magnetic_adjoint(shards)`: for each (L,R) in fixed rank order: copy R's `AdjHy/AdjHz[0]` ghost planes to a **preallocated staging plane on L** (comm stream, event-ordered), then `L.AdjHy/Hz[storage_cell_owned.stop-1].add_(staging)` on L's compute stream; zero R's ghost.
   - `exchange_electric_adjoint(shards)`: mirror for L's ghost node `AdjEy/AdjEz[storage_node_owned.stop]` → `R.AdjEy/Ez[storage_node_owned.start]`, zero L's ghost.
   - Staging buffers allocated in prepare (no per-step allocation); deterministic order = ascending rank.
2. **`fdtd/adjoint/native.py`** — factor `_reverse_step_standard_native_core` into `reverse_phase1_electric_to_h(...)`, `reverse_phase2_magnetic_to_e(...)`, `reverse_phase3_decay(...)` (pure refactor; the existing core calls the three in sequence — single-GPU tests must stay green bit-for-bit). Same factoring for `_reverse_step_cpml_native_core` (:278) once its face semantics are verified.
3. **New `fdtd/distributed/adjoint.py`** (or `fdtd/adjoint/distributed_bridge.py`) — `_DistributedFDTDGradientBridge`:
   - *Forward with checkpoints:* run `DistributedFDTD._advance_one_step` loop; at segment boundaries capture `capture_checkpoint_state(shard.solver, step)` per shard inside `torch.cuda.device(shard.device)`; store `(partition manifest, per-rank FDTDCheckpointState)` — never the transient receive halos. Disallow shutoff for trainable runs (single-GPU bridge has none either).
   - *Replay:* per segment, lockstep loop: `halo-E copy (torch) → per-shard _step_state magnetic half … halo-H copy → electric half`, i.e., a distributed `_replay_segment_states` that reuses `_step_state` unchanged per shard and inserts the two plane copies; capture mid-H per shard **after** the magnetic halo refresh so ghost forward values are valid for Phase-2 grad_eps reads.
   - *Reverse loop:* per step, per shard in rank order: seed injection (owned indices only) → Phase 1 all shards → `exchange_magnetic_adjoint` → Phase 2 all shards (produces local `grad_eps_*` incl. ghost columns) → `exchange_electric_adjoint` → Phase 3 → per-shard `_accumulate_source_term_gradients` (sources already single-writer via `crop_solver_source_terms_to_owned_x`). Maintain adjoint state dicts shaped like the padded local checkpoints, ghost planes zeroed after each accumulate.
   - *Seeds:* transpose of monitor merge. Baseline scope: point spectral/time monitors and full-field DFT seeds routed to the owner shard (ownership already computed by `_local_monitors`, solver.py:126-157); tiled plane/flux/mode seed scatter (slice cotangent by the same owned intervals `_merge_tiled_plane` records) is a follow-up gate — guard it initially.
   - *Global parameter reduction (the key simplification):* gather per-shard `grad_eps_ex/ey/ez` **owned slices** into global tensors on `result_device` via the existing `_gather_component` (deterministic stitch, solver.py:850-870), then run the **existing single-GPU** `pullback_material_input_gradients` / `pullback_density_gradients` once on the global logical scene. This sidesteps `_pullback_region_density`'s full-region assumption entirely (no per-shard density windowing math in v1), is deterministic by construction, and is numerically identical to single-GPU pullback. Cost: one eps-shaped global allocation at backward end — gate it with a `require_gather_capacity`-style preflight (capacity.py) that fails before the time loop starts.
   - *Forward density compile needs no new code:* `_sample_material_region_density` (`compiler/materials.py:1085-1112`) samples the density texture by **physical position** (`grid_sample` normalized against the region's global center/size), so each local scene automatically rasterizes its correct sub-window. Add a forward parity test to pin this before relying on it.
4. **`fdtd/distributed/solver.py`** — relax `_validate_static_capabilities` :385-389 (accept Box `MaterialRegion`); add a distributed adjoint capability validator (per-shard `_select_reverse_backend` must resolve to STANDARD or CPML and be identical across shards; anything else raises before allocation); add defense-in-depth trainable check inside `DistributedFDTD` itself.
5. **`witwin/maxwell/simulation.py`** — replace `_reject_trainable_parallel_fdtd` (:779-784) with capability-scoped validation; route trainable+parallel to the distributed bridge from `_run_fdtd` (:655-664) and validate at `prepare()` (:458-463); assemble `Result` with `parallel_stats` as today (:1056-1057).
6. **`fdtd/adjoint/capabilities.py`** — add distributed capability entries so `NATIVE_ADJOINT_CAPABILITIES` stays the single contract.

**Guard relaxation list (exact):** relax simulation.py:779-784 (blanket trainable×parallel) and solver.py:385-389 (MaterialRegion). Everything else stays, plus **new** prepare-time guards for: trainable geometry/perturbation tensors, trainable RF/excitation inputs, dispersive/conductive/nonlinear/aniso/modulated/complex trainable scenes (until their distributed reverse cores are verified), `normalize_source` with ≠1 source (inherited), `shutoff>0` on trainable runs, and objectives seeded from tiled plane/flux/mode monitors until seed scatter lands.

**Test plan (`tests/fdtd/multi_gpu/test_adjoint_*.py`):**
1. *Parity gates:* same scene run through the single-GPU bridge and the distributed bridge (2 GPUs): objective values within the plan's monitor gate (rtol 5e-5/atol 5e-6); parameter gradients within a declared gradient gate calibrated on the single-GPU FD error (propose rtol ≤ 1e-4 with an atol floor tied to grad magnitude; record the calibration — never loosen the forward gates).
2. *Interface finite difference:* central FD (3 step sizes, min-relative-error criterion) for (a) a density texel column exactly on the split, (b) a density texel on each shard interior, (c) source on the interface, (d) point-monitor objective on the interface node. Cases: vacuum standard, then CPML dielectric.
3. *Replay determinism:* distributed replayed owned states equal the forward run's owned states (assert exact equality for the pure-standard path, plan-gate tolerances otherwise).
4. *Repeat determinism:* two identical backward passes → `torch.equal` on every parameter gradient (single process, fixed rank order, assign-semantics kernels + ordered `add_` make bitwise repeatability achievable; if CPML breaks it, record the technical reason and a quantified tolerance).
5. *Guard regressions:* every unsupported trainable combination raises at prepare, before any distributed allocation.
6. *Checkpoint-stride variations* and *ghost-invariant asserts* (debug-mode check that ghost adjoint planes are zero at each phase boundary).

**Risks:**
- CPML reverse kernels (`native.py:278`, adjoint.cu CPML sections) may gate x-faces differently than the standard kernels (e.g. `cpml_correction_active`, `resolve_electric_cell_status`) — must be audited/tested at internal faces; fallback is bounded-x reverse CPML variants (new kernels, the only scenario requiring CUDA work).
- Torch-replay vs native-forward drift across shards compounding over segments — mitigated by the replay-parity test and by capturing mid-H from replay (bridge already does this single-GPU).
- Seed injection into non-owned indices silently corrupting the ghost-zero invariant — mitigated by debug asserts + ownership-derived seed routing.
- Global grad_eps gather erodes memory scaling for huge grids — acceptable and preflighted for E2; per-shard density windowing is the later optimization.
- eps leaves are per-shard local clones; `dynamic_electric_curls` (reverse_common.py:49) works unchanged, but the global pullback consumes only the gathered owned slices — ghost columns of local grad_eps must be excluded (use `_gather_component`, which already slices owned extents).

**Landable slices / estimate (total ≈ 4–6 engineering weeks):**
- S1 (3–5 d): transposed transport ops + staging buffers + synthetic-tensor unit tests; phase-split refactor of the standard native core (single-GPU suite bit-identical).
- S2 (1 wk): distributed checkpoint capture + distributed replay + replay-parity test.
- S3 (1–2 wk): distributed reverse for standard vacuum, point-monitor objectives; 1↔2-GPU objective/gradient parity + interface FD; guard set + prepare-time validator.
- S4 (1–2 wk): CPML + MaterialRegion density (forward compile parity test, global gather + single pullback, FD on split texel); repeat determinism; docs/FEATURE_LIST + reference-doc support-matrix updates.
- S5 (optional, post-gate): tiled plane/flux/mode seed scatter.

## B.3 Phase 8 — single-node NCCL transport

**What the plan demands (02 §Phase 8):** Linux **one-process-per-GPU** NCCL transport; timeout, rank-local logging, deterministic teardown; same conformance suite as P2P; transport capability interface preserved so multi-node later needs no public API change. Cross-node explicitly out of scope.

**What is reserved today:** `transport="nccl"` accepted structurally (`fdtd_parallel.py:103`) and rejected at `fdtd/distributed/solver.py:330-333`; `HaloTransport` ABC (`transport.py:34-52`) is the intended seam — but its methods receive the full shard tuple and dereference *other shards' tensors directly* (`destination.solver.Ey[...].copy_(source.solver.Ey[...])`), which only exists in one process. The abstraction reserves a name, not a shape.

**torch 2.13 facts (verified on the installed build, torch 2.13.0+cu130, NCCL 2.29.7):**
- `torch.distributed` NCCL backend available; the legacy same-process multi-device collectives (`all_reduce_multigpu` etc.) are **removed**; `init_process_group(..., device_id=...)` binds **one device per rank** — ProcessGroupNCCL's supported model is one process(-rank) per GPU; `batch_isend_irecv` available for neighbor point-to-point.
- The legacy in-process binding `torch.cuda.nccl` still exists (`init_rank`, `unique_id`, `all_reduce/broadcast/all_gather/reduce_scatter/reduce` over per-device tensor lists) — but it exposes **no send/recv**, is undocumented/legacy, has no timeout/watchdog/teardown machinery, and for two-plane copies offers nothing over the existing `cudaMemcpyPeerAsync` path already measured at 52+ GB/s on this NV4 topology.

**Honest conclusion:** same-process multi-device NCCL is technically constructible only via legacy collectives (halo = 2-rank broadcast, reverse-halo = 2-rank reduce) and is the wrong shape: it duplicates what CUDA P2P already does, on an unsupported API surface, with none of NCCL's operational benefits. **One process per device is required** for an NCCL transport that adds value (GIL-free per-rank stepping, ProcessGroupNCCL watchdog/timeout/abort, the only credible path to multi-node). This matches what the plan itself specifies.

**Concrete design:**
1. *Refactor (prerequisite):* split `DistributedFDTD` into a rank-local `ShardEngine` (local scene build from logical scene + `FDTDPartitionPlan` — pure deterministic metadata, so every rank derives its own layout identically; local solver; bounded launches; local monitors/DFT; owned-energy scalar) and a coordinator layer expressed **only through transport primitives**: `exchange_electric/magnetic`, `allreduce_scalar` (shutoff energy), `gather_component_slabs` (field gather), `gather_monitor_payloads` (post-loop, preallocated shapes derived from layout metadata — no object collectives in the hot loop), `gather_stats`. The in-process P2P transport implements these trivially; this refactor is what makes the reserved interface real.
2. *`NcclHaloTransport`:* `preflight()` = require `RANK/WORLD_SIZE/LOCAL_RANK` env (torchrun) else raise the current explicit error; `init_process_group("nccl", device_id=torch.device(local_rank), timeout=config.timeout)`; verify world_size == len(devices) and homogeneity via an all_gather of device properties. Halo exchange = `dist.batch_isend_irecv` on the neighbor pairs (Yee x-planes are contiguous views — sendable without packing); identical op ordering on both peers; boundary-strip kernels wait on `work.wait()` (overlap schedule preserved). Reverse halos (Phase 7 on NCCL): `irecv` into the preallocated staging plane + local `add_` — deterministic, no atomics.
3. *Result/persistence:* rank 0 assembles `Result` (monitor merge inputs gathered per rank in rank order — merge code unchanged); `save_sharded` becomes rank-local writes + rank-0 manifest (the manifest format at `persistence.py:353` already is per-rank). Nonzero ranks exit after barrier.
4. *Operational contract:* per-rank log files, `TORCH_NCCL_ASYNC_ERROR_HANDLING`/blocking-wait documented, `destroy_process_group()` in `finally` for deterministic teardown; failure matrix tests: peer rank death → timeout error propagation, mismatched world size, non-Linux → explicit error.
5. *Conformance:* parametrize the existing numerical matrix + monitor suites over transports; NCCL leg runs via a torchrun(2-rank) subprocess helper in pytest; identical tolerances (plan §7.2), public API byte-identical (`transport="nccl"` is the only difference).

**Estimate:** coordinator/ShardEngine refactor ~1 wk; NCCL transport + launcher + teardown ~1 wk; conformance + failure matrix ~1 wk → **2–3 weeks**, consistent with the precursor plan's estimate (`docs/plans/fdtd-multi-gpu-implementation-plan.md:485-486`).

**Sequencing note:** land Phase 7 on the P2P transport first (single-process, simplest determinism story), then the Phase 8 refactor — the phase-split reverse loop and transport-mediated design above were chosen so the same reverse schedule runs unchanged over `batch_isend_irecv`.
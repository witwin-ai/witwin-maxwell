# Distributed network embedding — implementation spec (from plan-03 dev agent, 2026-07-16)

> Extracted verbatim from the round-1 dev report's plan_gap. Basis for Round B1.

SCOPE NOTE: per instruction I did NOT start the distributed-network implementation; plan 03 Phase 4's multi-GPU work depends on plan 04's distributed port refactor, which is unmerged. Spec below.

=== GATE STATUS vs plan section 7 ===
Phase 0 (Touchstone I/O, E0) — MET, modulo CUDA note. Strict parser/writer, golden 1/2/4-port corpus (tests/rf/network/fixtures/{one_port_ri.s1p,two_port_ma.ts,four_port_db.s4p}), 21 import tests + 12 touchstone tests, line-numbered errors, explicit 2-port ordering assertion. Pure CPU/float64 — genuinely runs everywhere.
Phase 1 (physicality + rational realization, E1) — MET. tests/rational/test_core.py (9), test_fit.py (5), test_network_physicality.py (4). |z| < 1-1e-7 discrete-pole gate enforced in compile (compiler/networks.py:215-218 stability, :239-248 passivity certificate). Caveat already disclosed in docs: acceptance uses a pole-aware finite-band certificate, not an all-frequency Hamiltonian certificate.
Phase 2 (one-port embedding, E2) — MET but CUDA-gated. test_network_runtime.py::test_touchstone_fitted_series_rlc_matches_native_fdtd_termination covers the <1% mag / <2 deg gate; ::test_cuda_runtime_has_no_scalar_sync_or_host_device_copy + test_network_observer_hot_path.py cover "profiler 无逐步 CPU activity".
Phase 3 (multiport + delay, E2) — MET but CUDA-gated. <0.02 S-error: test_network_multiport_fdtd.py::test_touchstone_multiport_fit_matches_independent_network_reference. permutation-equivalence: test_network_multiport_runtime.py::test_port_permutation_preserves_physical_fields_currents_and_state. <3 deg long delay: test_network_delay_integration.py::test_long_fractional_delay_measured_phase_error_is_below_three_degrees. <2% steady-state: ::test_delayed_network_block_fdtd_runtime_matches_frequency_response.
Phase 4 (gradients, multi-GPU, production, E3) — 1 of 4 gates met.
  (a) residue/direct gradient vs FD < 2% — MET (test_fdtd_network_adjoint.py, 9 tests).
  (b) single/multi-GPU port V/I rtol <= 2e-5 — NOT MET, BLOCKED, never executed. No distributed network code exists.
  (c) no-feature FDTD regression < 1% — NOT MET. Was documented as passing on a -6.98% reading; corrected this pass (see below).
  (d) connected 8-port/order-32 step overhead < 10% — UNEVIDENCED. The cited artifact .cache/phase4-network-performance-true-grid272.json is absent from the repo AND from this host (find / turned up nothing), so 5.9403% cannot be re-verified. Recorded on different hardware (RTX 5080 / torch 2.10 / CUDA 12.8) than this host (A6000 / torch 2.13 / CUDA 13.0). Not re-run here per the no-timing rule.

=== SECTION 8.2 / SECTION 10 CI CORPUS GAP ===
None of the six plan-named E2E scenarios exist as named fixtures (grepped all six): one_port_rlc_touchstone, two_port_filter_embedding, coax_delay_line, four_port_coupler, antenna_matching_block, multi_gpu_split_ports. Five have equivalent physics under other test names (see gate list). TWO HAVE NO EQUIVALENT AT ALL:
  - antenna_matching_block: zero tests in tests/rf/network reference antenna(...) or realized_gain. Section 10 requires this closed loop (external matching network changes realized gain) for completion. This is single-device work and is NOT blocked by plan 04 — it is simply missing scope.
  - multi_gpu_split_ports: blocked on the below.

=== IMPLEMENTATION SPEC: witwin/maxwell/fdtd/distributed/networks.py ===
Read 04's witwin/maxwell/fdtd/distributed/circuits.py (652 lines, at .worktrees/spice-mna-cosimulation, read-only). Mirror it structurally; the shapes line up almost 1:1 because both reduce to "gather one voltage scalar per port -> one implicit solve on a single owner -> scatter one current scalar per port".

1. PREREQUISITE REFACTOR OF THE SINGLE-DEVICE RUNTIME (do this first; it is the real work).
   04 made fdtd/circuits.py parameterizable via `_apply_step(step_index, *, device_indexed_samples, free_voltages, scatter_field_currents)` with two entry points: `apply(free_voltages=None)` (single-device: samples fields itself, scatters itself) and `apply_external(free_voltages, *, apply_field_currents=False)` (distributed: takes gathered voltages, returns currents, never touches fields).
   fdtd/networks.py::apply_network_runtime currently FUSES five things: (1) terminal gather via terminal_groups -> free_voltage, (2) implicit LU solve -> branch_current, (3) field injection via index_add_, (4) state advance x[n+1]=Ax+Bv, (5) power/energy accumulation. Split into:
     - `advance_network_external(runtime, free_voltage) -> branch_current` = steps 2,4,5 only. Owner-side. Must keep the existing preallocated-buffer + no-sync discipline.
     - keep the gather (1) and scatter (3) as per-shard helpers, mirroring 04's static `_sample_voltage`/`_apply_current`.
   Keep apply_network_runtime as the single-device composition so the existing CUDA-Graph capture path and its regressions are unchanged.

2. PLAN OBJECTS (mirror DistributedBoundPortPlan / DistributedCircuitPlan):
   - `DistributedNetworkPortPlan`: port_name, owner_rank, voltage_component, minimum_global_index, local_voltage_indices, geometry.
   - `DistributedNetworkPlan`: network_name, owner_rank, owner_reference_port, ports; `port_owners` property.
   - `compile_distributed_network_plan(scene, partition_plan, *, geometries=None)`. Iterate `block.connected_port_names` where 04 iterates `circuit.bindings`. Reuse 04's `_owner_for_index(layouts, component, index)` verbatim (exactly-one-owner else RuntimeError).
   - CRITICAL CONSTRAINT to copy: each connected terminal must lie wholly within one x slab. 04 rejects a port spanning ranks with an actionable message ("Move the port off the split or change the partition; one port must have one scalar owner so circuit communication remains O(bound ports)"). This is what makes comms O(Nports) instead of O(edges) and satisfies plan 03 section 6 ("只有 O(Nports) 标量通信").
   - OWNER RULE — this is exactly plan 03 section 6's "拥有最小编号连接端口 reference point 的 shard": copy 04's deterministic `min(port_plans, key=lambda p: (*p.minimum_global_index, component_order[p.voltage_component]))` with component_order = {"Ex":0,"Ey":1,"Ez":2}. Do not invent a different tiebreak; 03 and 04 must agree or a scene with both will place them inconsistently.
   - Copy the overlapping-edge rejection (occupied_edges dict) across connected terminals.

3. OWNER PROXY (mirror `_owner_proxy_field`): the trick that lets the owner reuse the unmodified single-device prepare path. Build a zero-edge FieldPortCoupling on the owner device (empty linear_indices/voltage_weights/injection, scalar zeros for last_*), so the owner's PreparedNetworkRuntime has `terminal_groups=()` and takes free_voltage as an argument. For networks specifically: `feedback_impedance` (fdtd/networks.py:329) stacks each port's `lumped.discrete_port_impedance` — these now live on different shards; move them to the owner device at prepare time (cross-device copies at prepare are fine) before building loop_denominator = I + D*Z and its LU factorization on the owner.
   GOTCHA NOT PRESENT IN 04: `_prepare_terminal_groups` (fdtd/networks.py:126-168) batches terminal edges by field_name ACROSS ALL PORTS of a network. Distributed ports live on different shards with different field tensors, so the grouping key must become (shard/rank, field_name), not field_name alone. Grouping globally will silently mix tensors from different devices.

4. RUNTIME (mirror DistributedCircuitRuntime):
   - `_DistributedNetworkPortRuntime`: plan, shard, local_geometry, local_field, owner_field, owner_voltage, local_current, and the four cuda Events (voltage_ready, voltage_received, current_copied, current_received) allocated only when remote.
   - COPY `_OwnerCurrentReuseState` VERBATIM. It is not incidental: `runtime.branch_current` is a persistent preallocated buffer reused every step, so the owner must not overwrite step n+1's current before step n's async DtoD copy to the remote shard is acknowledged. Same hazard as 04, same fix.
   - `apply()` ordering, copy 04 exactly: (a) each shard's compute stream samples its local voltage, records voltage_ready if remote; (b) owner's COMMUNICATION stream waits voltage_ready, copies to owner_voltage non_blocking, records voltage_received; (c) owner's COMPUTE stream waits current_copied (if reuse pending) then voltage_received, assembles free_voltages (local ref if same-shard, owner_voltage if remote), calls advance_network_external, accumulates port observers, applies currents for same-shard ports, records solve_ready; (d) each remote shard's communication stream waits solve_ready, copies the current, records current_copied, applies it, records current_received, marks reuse; its compute stream then waits current_received.
   - Plan 03 section 6 "若网络所有端口均在同一 shard，运行时不得发起 collective" falls out for free: 04 allocates no Events and takes no comm-stream path when owner_rank matches. Assert it in a test via `remote_port_count == 0`.
   - `stats()`: mirror 04's dict; plan 03 section 8.3 requires communication bytes. 04's `communication_bytes_per_step = 2 * remote_port_count * element_size` is the exact contract to reuse.
   - CUDA GRAPHS: 04's distributed apply() is multi-stream + event-driven and is NOT graph-captured. 03's single-device full-step capture (_make_full_embedded_step_runner) MUST be disabled on the distributed path; do not try to capture across shards.

5. SOLVER.PY PORT-REJECTION REFACTOR (04 already did this; 03 must merge with it, NOT redo it):
   04 replaced the blanket `if self.logical_scene.ports: raise` with, in _validate_static_capabilities:
     - `unsupported_ports = tuple(p for p in ports if not isinstance(p, (LumpedPort, TerminalPort)))`; if any -> if any is ModePort, raise "Multi-GPU FDTD mode ports require distributed modal plane tiling; unsupported ports: {names}"; else raise "Multi-GPU FDTD does not support port types for {names}". i.e. reject only ModePort, allow bound LumpedPort/TerminalPort.
     - then `bound_port_names = {b.port_name for c in circuits for b in c.bindings}` and rejects any unbound Lumped/Terminal port.
     - `_build_local_scene` clones with `ports=(), circuits=()`; there is a second clone at solver.py:535 `self.logical_scene.clone(ports=(), circuits=())`.
   MERGE POINT / EXPECTED CONFLICT: after both land, `bound_port_names` must be the UNION of circuit bindings and network connected_port_names, otherwise 04's unbound-port guard will reject every network-connected terminal. Both worktrees also edit the same _validate_static_capabilities block and the same _build_local_scene clone kwargs. My interim guard (see below) is the thing to delete when this lands.

6. DEFERRED PER PLAN section 6: backward uses the reversed comm graph to reduce dV/dI; until the distributed adjoint contract exists, "trainable embedded model + spatial multi-GPU" must be rejected precisely (not silently detached). Result gather dedupes by network name, emitted only by the owner; sum power over ports before judging passivity.
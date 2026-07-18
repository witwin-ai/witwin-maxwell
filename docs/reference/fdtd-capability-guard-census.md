# FDTD Capability-Guard Census

This document is the maintained baseline for the AST guard gate in
`tests/api/public/test_guard_census.py`. The gate distinguishes capability gaps
from deliberate public-contract boundaries so new `NotImplementedError` paths
cannot enter unnoticed.

## Reconciled baseline

The 2026-07-17 integrated repository state (circuit co-simulation, Touchstone
network embedding, the thin-wire subgrid conductor series in plan 07 phases
0-3, the array basis / active-S feature series in plan 06 phases 0-1, and the
plan 07 Phase 4 multi-GPU wire forward slice, plus the plan 07 Phase 4
finite-conductor wire series-impedance slice, and the plan 05
nonlinear-circuit-device Phase 0 slice, all merged) contains 161 guards:

- 137 capability guards tracked by the non-increasing test budget;
- 24 contract guards excluded by exact file and message substring.

The single-GPU circuit adjoint provides the circuit replay and transpose
linear-solve VJP. Its remaining explicit limits are the omitted t=0
operating-point/initial-state pullback (therefore trainable DC source values and
circuit initial conditions), tensor seeds on the initial `CircuitData` sample,
and trainable port reference impedance. Multi-circuit execution, distributed
adjoints, and full-field/observer forward-resume accumulation remain separate
capability gaps rather than reclassified permanent contracts. The thin-wire
series adds ten capability guards on top of the 119-guard circuit/network
baseline (119 + 10 = 129); their disposition is detailed in the
[Thin-wire reconciliation](#thin-wire-reconciliation-2026-07-17) note below. The
array-basis series adds one further capability guard (129 + 1 = 130); its
disposition is detailed in the
[Array-basis reconciliation](#array-basis-reconciliation-2026-07-17) note below.
The plan 07 Phase 4 multi-GPU wire forward slice adds two further net capability
guards (130 + 2 = 132) by implementing the distributed wire forward and replacing
its single blanket guard with three narrower ones (CPML, Mur, and network/circuit
composition); its disposition is also detailed in the
[Thin-wire reconciliation](#thin-wire-reconciliation-2026-07-17) note.

The capability baseline is distributed as follows:

| Area | Capability guards |
| --- | ---: |
| External interoperability adapter | 18 |
| Material compilers | 12 |
| Frequency-domain runtime | 5 |
| Time-domain adjoint | 19 |
| Time-domain excitation | 12 |
| Time-domain ports and lumped elements | 16 |
| Time-domain runtime | 20 |
| Public simulation, result, and network workflows | 24 |
| Material models | 7 |
| Postprocessing | 4 |
| **Total** | **137** |

This integrated baseline is the 2026-07-16 circuit/network state (119 capability
guards) plus the ten thin-wire capability guards from plan 07 phases 0-3 (giving
129) plus the one array-basis capability guard from plan 06 phases 0-1 (giving
130) plus the two net capability guards from the plan 07 Phase 4 multi-GPU wire
forward slice (giving 132) plus the one array scene-gradient capability guard from
plan 06 Phase 4 (giving 133); its disposition is detailed in the
[Array scene-gradient reconciliation](#array-scene-gradient-reconciliation-2026-07-17)
note below. The plan 07 Phase 4 finite-conductor wire series-impedance slice adds
one further capability guard (giving 134); its disposition is detailed in the
[Finite-conductor wire reconciliation](#finite-conductor-wire-reconciliation-2026-07-17)
note below. The 119-guard baseline is itself the 2026-07-15 circuit co-simulation
state (113 capability guards) plus the six 2026-07-16 network-embedding
capability guards (Time-domain adjoint +3, Public simulation/result/network
workflows +3); that reconciliation is detailed in the
[Network-embedding reconciliation](#network-embedding-reconciliation-2026-07-16)
section below.

### Thin-wire reconciliation (2026-07-17)

The thin-wire subgrid conductor series (plan 07, phases 0-3) adds ten reviewed
capability guards rather than silently running unsupported compositions: two
compiler guards (a locally anisotropic self-term guard and a Bloch-boundary
phase-aware wire-topology guard); a set of single-device runtime host-composition
guards (off-diagonal, conductive, dispersive, and nonlinear/modulated host
material, plus a surface-impedance conductor-ownership guard); a distributed
wire ownership guard; and an adjoint checkpoint/reverse guard. They remain
capability debt, not contract exclusions.

The plan 07 Phase 4 multi-GPU wire forward slice (2026-07-17) implements the
distributed thin-wire forward and therefore removes the single blanket
distributed wire guard ("Multi-GPU ThinWire requires distributed fragment/state
ownership...") and replaces it with three narrower capability guards in
`fdtd/distributed/solver.py::_validate_distributed_wire_support`: a distributed
thin-wire + CPML boundary (no verified wire-edge/PML ownership across the split),
a distributed thin-wire + Mur absorbing boundary (wire-edge/boundary ownership on
the split is undocumented and unverified), and a thin-wire scene mixed with an
embedded network or lumped circuit (both claim owner-resident coordination state
with no distributed merge). The Mur guard is the fail-closed integration decision
folded into this merge: Mur + wire previously passed the PML-only gate but had no
verified split coupling. Net +2 in the Time-domain runtime area (18 -> 20).
Finite-conductor loss and distributed wire reverse/gradient are separate slices;
lower this budget when those new guards are implemented. Phase 4 also owns finite-conductor loss and SIBC ownership; the
remaining host-composition guards stay tracked capability debt for a later
compatibility plan. Each future phase must lower this budget as it removes its
guards. The thin-wire differentiable fixed-step
requirement is instead a differentiation contract (the automatic joint-CFL clamp
is a discrete preparation decision that is not differentiated with respect to
radius or material parameters) and is listed under CONTRACT_GUARDS, not counted
here.

### Array-basis reconciliation (2026-07-17)

The array basis / active-S feature series (plan 06, phases 0-1) adds one reviewed
capability guard. `witwin/maxwell/postprocess/array.py` raises
`NotImplementedError("Array basis extraction is FDTD-only.")` when a non-FDTD
`Result` is passed to `array_basis()`. It is a genuine capability gap rather
than a public contract: `array_basis()` consumes the retained in-memory
full-wave PortSweep field columns, which only the FDTD backend currently
produces. It stays counted in the budget (Postprocessing 3 -> 4) so extending
basis extraction to another backend later lowers the count again.

### Array scene-gradient reconciliation (2026-07-17)

The array codebook/MIMO/gradient series (plan 06, phases 2-4) adds one reviewed
capability guard. `witwin/maxwell/array.py::ArrayBasisData.scene_gradient_vjp`
raises `NotImplementedError("Scene-parameter gradients through the array basis
require the aggregated per-column adjoint envelope ...")`. Phases 2-4 land the
codebook/scan/max-hold workflows, the field- and S-parameter-based MIMO metrics,
and weight gradients through `combine()` (fully supported, no solver rerun). The
scene/material/geometry gradient through the basis is deliberately sliced out and
fails closed: the retained-column basis stores detached embedded-pattern tensors,
so it cannot back-propagate to scene parameters, and the aggregated per-column
adjoint (plan 06 Phase 4 exit gate, gated on the plan 02 Phase 7 distributed
result-aggregation contract) is not wired to this single-device basis. It is
counted under "Public simulation, result, and network workflows" (23 -> 24);
lower the budget when the aggregated adjoint lands.

### Finite-conductor wire reconciliation (2026-07-17)

The plan 07 Phase 4 finite-conductor wire series-impedance slice adds one
reviewed capability guard. `witwin/maxwell/compiler/thin_wire.py` raises
`NotImplementedError` ("ThinWire ... uses a finite conductor. The per-unit-length
series-impedance model is available via ...fit_series_impedance, but the lossy
current recurrence is not yet wired into the FDTD runtime; use a PEC conductor to
run a thin-wire FDTD simulation.") when a `ThinWire` declares a non-PEC
conductor. This slice lands the analytic round-wire skin-effect impedance and the
passive rational ADE fit (`witwin/maxwell/compiler/wire_impedance.py`, reusing
the shared network rational-fitting stack) plus their acceptance gates, but the
lossy current recurrence that would feed that impedance into the Yee update is
not yet wired into the FDTD runtime. The compiler therefore fails closed rather
than silently running a finite conductor as if it were lossless. It is a genuine
capability gap, not a public contract: the impedance model itself is implemented
and the guard disappears once the recurrence lands. It is counted under
"Material compilers" (11 -> 12); lower the budget when the lossy recurrence is
wired into the runtime.

### Nonlinear-circuit-device reconciliation (2026-07-17)

The plan 05 nonlinear-circuit-device Phase 0 slice adds the Device + Newton
contract (`witwin/maxwell/circuit_devices.py`,
`witwin/maxwell/compiler/nonlinear_devices.py`) and two reviewed contract guards,
with **no** change to the capability budget (it stays 134). The two guards are
the reserved transistor public surfaces `circuit_devices.py::BJT.__init__` and
`circuit_devices.py::MOSFET.__init__`, each raising `NotImplementedError`
("Transistor device BJT/MOSFET is gated behind the independent Phase 5 go/no-go
transistor evaluation ..."). They are contract guards, not capability debt: per
plan 05 §8 the diode/behavioral nonlinear device family (Diode, PiecewiseLinearIV,
PolynomialIV, VoltageDependentCapacitor) is delivered by Phases 0-4, while
transistor terminal physics, charge conservation, and gradients are a separate,
independent go/no-go phase whose non-completion does not block the Phase 0-4
release. Reserving the surface fail-closed makes "a parser or factory recognising
a model card is not the same thing as supported physics" machine-checkable. The
Device + Newton contract itself adds no `NotImplementedError`; that part of the
count is unchanged and the contract table above rises from 22 to 24.

### Nonlinear-device fail-closed hardening (2026-07-17)

Phase 0 admits the nonlinear device family through `Circuit.add` and validates
its DC topology in `compile_circuit_graph`, but the executable runtimes were not
yet extended: the linear MNA, coupled MNA, and FDTD Norton-companion paths build
a single constant-conductance stamp with no Newton iteration, and the standalone
Newton core consumes only the conduction law `i(v)`. Left unguarded, a diode
compiled through those paths ran with the device silently absent, and validated
`series_resistance` / `junction_capacitance` parameters were silently dropped.
This slice closes those fail-open gaps with three capability guards (capability
budget 134 -> 137, all under "Time-domain ports and lumped elements", 13 -> 16):

- `compiler/circuits.py::reject_nonlinear_devices` rejects any nonlinear device
  reaching a linear executable runtime, reached from `compile_mna_system`,
  `compile_coupled_mna_system` (both via `_compile_mna_system`), and
  `scene.compile_circuits()` (the FDTD circuit prepare path in
  `Simulation._validate_circuit_execution`).
- `compiler/nonlinear_devices.py::compile_nonlinear_devices` fails closed on a
  diode with nonzero `series_resistance`: the ohmic series branch (an internal
  node with the resistive drop) is not assembled into the ideal-Shockley `i(v)`,
  so honouring the parameter needs the extended device topology.
- `compiler/nonlinear_devices.py::newton_solve` fails closed on a diode with
  nonzero `junction_capacitance` entering the conduction-only DC solve, which
  never differentiates the stored charge `q(v)` into the system.

All three are genuine capability gaps, not permanent contracts: each guard is
removed when its runtime slice (nonlinear device-runtime integration, the series
branch, and the charge-aware transient solve) lands. The `q(v)` / `C(v)` charge
model is still fully exercised through `CompiledNonlinearDevice.charge`, so the
device math surface is unaffected.

## Contract exclusions

`CONTRACT_GUARDS` in the test is the canonical exact-match inventory. Its 24
entries cover these stable contract families:

| Contract family | Count |
| --- | ---: |
| Material frequency-evaluation domains | 9 |
| Module-style scene implementation requirement | 1 |
| Non-meshable complex polygon geometry | 1 |
| Time-domain adjoint input, complex-state, and fixed-step requirements | 3 |
| Frequency-domain adjoint input requirement | 1 |
| External adapter medium limitations | 2 |
| Bloch-boundary total-field/scattered-field slab requirement | 1 |
| Closed-surface exterior-sampling requirements | 3 |
| Time-domain-only embedded network feedback | 1 |
| Transistor Phase-5 go/no-go boundary | 2 |
| **Total** | **24** |

When an implementation removes a capability guard, lower
`CAPABILITY_GUARD_BUDGET` in the same commit. Reclassifying a guard as a public
contract requires adding an exact entry to `CONTRACT_GUARDS` and updating the
two tables above. Raising the budget requires an explicit capability review and
an updated baseline here.

## Network-embedding reconciliation (2026-07-16)

The Touchstone network-embedding series (`33c1ca5`, `f76a27a`, `2ff0003`) added
exactly seven `NotImplementedError` guards and removed none (verified by
`git log -S "raise NotImplementedError" master..HEAD -- witwin/`). Six are
capability gaps and raise `CAPABILITY_GUARD_BUDGET` by six; combined with the
five circuit co-simulation guards on the 2026-07-15 baseline of 108 this yields
the integrated budget of 119 (108 + 5 + 6). The seventh network-embedding guard
is a permanent architectural boundary reclassified into `CONTRACT_GUARDS`. Each
guard was reviewed individually below. All seven are correct fail-closed
rejections and are retained; this reconciliation records the baseline instead of
deleting them.

### Capability guards (+6: "Time-domain adjoint" +3, "Public simulation, result, and network workflows" +3)

| Guard | Message | Capability review |
| --- | --- | --- |
| `fdtd/adjoint/bridge.py` `_validate_supported_configuration` (delay branch) | "Differentiable embedded-network FDTD does not support explicit delay state." | Deferred capability. The forward explicit fractional-delay line is a fixed-length buffer recurrence; its reverse-mode adjoint is not yet derived, so a differentiable run with a delay network is rejected rather than returning a wrong gradient. Implementable later (delay-line adjoint), so it counts against the budget. |
| `fdtd/adjoint/bridge.py` `_validate_supported_configuration` (RationalModel poles/proportional) | "Differentiable embedded RationalModel supports residues and direct terms only; trainable {…} are not supported." | Deferred capability. Plan 03 §5.3 defers gradients through pole relocation / passivity enforcement to a later version; trainable `poles`/`proportional` are rejected instead of silently detached. Implementable later, counts against the budget. |
| `fdtd/adjoint/bridge.py` `_validate_supported_configuration` (state-space A/B/C/D) | "Differentiable embedded networks accept trainable residues/direct on a pre-fitted RationalModel; direct trainable state-space matrices are not supported." | Deferred capability. Direct differentiation of raw discrete `A/B/C/D` is not implemented; users differentiate a pre-fitted `RationalModel`'s residues/direct terms instead. Implementable later, counts against the budget. |
| `fdtd/networks.py` `finalize_embedded_networks` (replay delay branch) | "Differentiable embedded-network replay does not support explicit delay state." | Deferred capability. Mirrors the bridge delay guard on the replay path so the checkpoint/replay adjoint cannot silently run without delay-state gradients. Implementable later, counts against the budget. |
| `simulation.py` `_validate_trainable_rf_support` (RationalModel poles/proportional) | "Differentiable embedded RationalModel supports residues and direct terms only; trainable {…} are not supported." | Deferred capability. Public-entry mirror of the bridge poles/proportional guard so the rejection surfaces at `Simulation` construction, not deep in the adjoint. Implementable later, counts against the budget. |
| `simulation.py` `_validate_trainable_rf_support` (state-space A/B/C/D) | "Differentiable embedded networks accept trainable residues/direct on a pre-fitted RationalModel; direct trainable state-space matrices are not supported." | Deferred capability. Public-entry mirror of the bridge state-space guard. Implementable later, counts against the budget. |

### Contract guard (+1)

| Guard | Message | Contract review |
| --- | --- | --- |
| `simulation.py` `_validate_network_solver` | "Embedded network feedback is defined only for the time-domain FDTD update; frequency-domain solvers cannot ignore Scene.networks." | Permanent architectural boundary, not a capability gap. Per plan 03 (2026-07-14 decision, §2.1 goal 3 and §5.2) network embedding is unified as a **time-domain** passive state-space recurrence `x[n+1]=Ad x[n]+Bd v[n]` fed back into each FDTD step; there is no per-step update in a frequency-domain solve for a network to couple into, and the plan explicitly rejects a post-simulation S-matrix cascade. This is definitionally the same family as the material "…is defined only for / is not defined for…" contract guards, so it is excluded by exact `(file, substring)` match rather than counted against the capability budget. |

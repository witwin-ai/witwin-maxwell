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
finite-conductor wire series-impedance slice, the plan 05 nonlinear-device
Phase 0 slice, the plan 08 gyromagnetic-ferrite Phase 0 slices, and the plan 09
surface-impedance Phase 0 slices, and the plan 05 nonlinear-device N1 standalone
transient slice, all merged) contains 165 guards:

- 139 capability guards tracked by the non-increasing test budget;
- 26 contract guards excluded by exact file and message substring.

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
| External interoperability adapter | 19 |
| Material compilers | 14 |
| Frequency-domain runtime | 5 |
| Time-domain adjoint | 19 |
| Time-domain excitation | 12 |
| Time-domain ports and lumped elements | 16 |
| Time-domain runtime | 20 |
| Public simulation, result, and network workflows | 24 |
| Material models | 8 |
| Postprocessing | 4 |
| **Total** | **140** |

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

### Array scene-gradient implemented (2026-07-21, plan 06 Phase 4)

`ArrayBasisData.scene_gradient_vjp` is now implemented on the single-device
basis, so the fail-closed guard above is removed and `CAPABILITY_GUARD_BUDGET`
drops `176 -> 175` in the same change ("Public simulation, result, and network
workflows" `24 -> 23`). The method aggregates the per-column adjoints of the
linear beam combine `E = sum_n w_n e_n` onto the scene parameters: the caller
re-runs each column's forward under autograd and passes the resulting live
embedded-pattern columns; the method forms the combined field, takes the
combined-field cotangent `cot_E = autograd.grad(L, E)`, seeds each column with
`conj(w_n) * cot_E` (`= w_n^* . (dL/dE)^*`, the PyTorch complex-product backward
of the combine), and sums the per-column VJPs in a deterministic reduction
order. The seeded sum is bit-identical to end-to-end autograd of the combined
objective (verified in `tests/rf/array/test_array_scene_gradient.py`). Detached
columns, a no-contribution parameter set, wrong column counts, batched
`[B, F, N]` weights, and shape/dtype/device mismatches raise `ValueError` /
`TypeError` (usage errors, not capability gaps) and do not occupy budget. The
2-GPU ensemble aggregation of the same per-column VJPs is the plan 06 Phase 4b
follow-on (plan 02 Phase 7 distributed result-aggregation contract).

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

### Nonlinear-device N1 standalone-transient reconciliation (2026-07-18)

The plan 05 N1 slice implements the charge-aware transient solve, so the third
guard above is removed and the capability budget drops `140 -> 139`
("Time-domain ports and lumped elements"). `NonlinearMNASystem` now folds the
trapezoidal / backward-Euler stored-charge companion into the residual and
Jacobian (`_charge_companion`, `advance_charge_state`) and `run_nonlinear_transient`
drives Newton in the transient loop with gmin/source DC continuation
(`solve_dc_operating_point`) and local backward-Euler PWL-kink breakpoints. The
diode `junction_capacitance` and `VoltageDependentCapacitor` `q(v)` are therefore
consumed by the integrator, and the DC operating point correctly treats the
capacitance as an open circuit. The diode `series_resistance` guard stays: the
internal-node ohmic branch is still not assembled this round.

### Gyromagnetic ferrite reconciliation (2026-07-17)

The plan 08 Phase-0 gyromagnetic ferrite slices (0a physics contract, 0b torch
reference oracle, 1a `GyromagneticFerrite` material type) add four reviewed
`NotImplementedError` guards: two capability gaps and two permanent contracts.

Capability gaps (Phase 0 added +2, `134 -> 136`; slices 1b/1c then net +1 to the
integrated budget, see the 1b/1c reconciliation below):

- `witwin/maxwell/compiler/materials.py` `_static_structure_material` rejected a
  `GyromagneticFerrite` structure at Phase 0. **Removed in slice 1c**: a ferrite
  now compiles as its diagonal background and the gyrotropy is produced by the
  magnetization-ADE layout + FDTD forward hooks (see the slice-1b/1c
  reconciliation).
- `witwin/maxwell/media.py` `PerturbationMedium.__init__` rejects a
  `GyromagneticFerrite` base. The medium perturbs a scalar permittivity
  background, which would silently discard the ferrite's magnetization state
  (Material models `7 -> 8`); lower the budget when a ferrite perturbation branch
  is implemented.

Permanent contracts (+2, listed in `CONTRACT_GUARDS`, Material
frequency-evaluation domains `9 -> 11`):

- `GyromagneticFerrite.relative_permeability` and
  `GyromagneticFerrite.evaluate_at_frequency` both fail closed because a scalar
  (or diagonal) `FrequencyMaterialSample` cannot represent the off-diagonal
  gyromagnetic Polder tensor. Callers use `permeability_tensor_at_freq()` /
  `polder_tensor()` instead. These are definitionally the same family as the
  existing material "…is not defined for…" frequency-evaluation contracts, so they
  are excluded by exact `(file, substring)` match rather than counted against the
  capability budget.
### Gyromagnetic ferrite slices 1b/1c reconciliation (2026-07-18)

Slice 1b (compiler layout) added no guards: `compile_gyromagnetic_layout` and
`Scene.compile_gyromagnetic_materials` are pure lowering with no fail-closed path,
and the full-tensor `mu(f)` accessor is total.

Slice 1c (axis-aligned forward) is a net `+1` capability guard (`140 -> 141`):

- Removed (`-1`): the `compiler/materials.py` `_static_structure_material` ferrite
  reject. A `GyromagneticFerrite` now compiles as its diagonal background
  (`eps_r` / `mu_infinity` / `sigma_e`), and the non-reciprocal off-diagonal
  permeability is produced by `compile_gyromagnetic_layout` plus the FDTD
  gyromagnetic forward hooks (`fdtd/runtime/gyromagnetic.py`
  `advance_gyromagnetic_state` / `apply_gyromagnetic_correction`), so it no longer
  silently drops the gyrotropy.
- Added (`+2`), both in `witwin/maxwell/fdtd/runtime/gyromagnetic.py`
  `build_gyromagnetic`:
  1. A general (non-axis-aligned) bias, or a scene mixing bias axes, fails closed.
     Slice 1c advances only the axis-aligned z/x/y fast path where the two
     transverse H components co-locate on the shared Yee overlap (identity
     collocation); an arbitrary bias needs the local-frame rotation and 4-point
     collocation of the Phase-2 kernel. Genuine capability gap; lower the budget
     when the arbitrary-bias kernel lands.
  2. A Bloch-periodic ferrite run fails closed: the real-valued magnetization-ADE
     correction would break the complex Bloch phase relation (the magnetic mirror
     of the existing nonlinear / full-anisotropy / modulation + Bloch guards).
     Genuine capability gap; lower the budget when a complex-field ferrite
     correction lands.

The `PerturbationMedium`-wraps-a-ferrite reject and the two
`GyromagneticFerrite` scalar frequency-evaluation contracts are unchanged.

### Gyromagnetic ferrite consumer-side fail-close hardening (2026-07-18)

Slice 1c lifted the `compiler/materials.py` ferrite reject globally, but that
compiler guard had also protected every *non-FDTD-forward* consumer of a ferrite
scene, none of which received a replacement at the time. This hardening round adds
the missing consumer-side guards, a net `+3` (`141 -> 144`), so each consumer fails
closed instead of silently simulating a reciprocal medium:

- Added (`+1`): `witwin/maxwell/fdfd/solver.py` `_ensure_material_components`
  rejects a `GyromagneticFerrite`. The static material compile lowers a ferrite to
  its diagonal background only; the frequency-domain solver never runs the
  magnetization-ADE runtime, so it would ingest a reciprocal permeability and
  silently drop the off-diagonal Polder gyrotropy. Frequency-domain runtime gap;
  lower the budget when an FDFD gyromagnetic ingest lands.
- Added (`+1`): `witwin/maxwell/fdtd/distributed/shard_engine.py` `ShardEngine.build`
  rejects a `gyromagnetic_enabled` local solver. The shard phases never call the
  magnetization-ADE hooks, and each shard-local layout has no rank-seam handling
  (the overlap slice drops the top plane at rank seams), so a joint solve would
  silently simulate a reciprocal medium with uncorrected seams. This restores the
  frozen contract boundary 8 ("Multi-GPU ferrite -- rejected until Phase 4").
- Split (`+1`): the former single axis-aligned guard in
  `fdtd/runtime/gyromagnetic.py` `build_gyromagnetic` is now two guards -- a
  general (non-axis-aligned) bias reject and a mixed-bias-direction reject. The old
  `torch.unique(fast_axis)` check collapsed `+e_k` and `-e_k` to the same axis code,
  so a scene mixing `+z` and `-z` ferrites passed the axis-only guard and the single
  global transverse sign then silently inverted the non-reciprocity of the `-bias`
  region (the opposed-bias latching-circulator device class). The mixed guard now
  tests bias-vector uniformity, so both mixed axes and opposed signs on one axis
  fail closed.

Not a new guard node (no census delta):

- The adjoint reject (frozen contract boundary 9) is a new `return`-string branch in
  `fdtd/adjoint/bridge.py` `_unsupported_adjoint_medium`, reusing the existing generic
  `raise NotImplementedError(unsupported_medium_message)`; the non-reciprocal
  magnetization-ADE correction has no reverse (transpose) core, so a differentiable
  ferrite run would return gradients for a reciprocal medium.
- The checkpoint/resume schema (`fdtd/checkpoint.py`, `fdtd/resume.py`) gains the
  persistent gyromagnetic magnetization state names (`m_u` / `m_v` / `dm_u` / `dm_v`),
  so a mid-run save/resume round-trips the magnetization instead of silently
  re-zeroing it; `validate_checkpoint_state` now raises on a dropped gyromagnetic
  state name.

### Surface-impedance adapter reconciliation (2026-07-17)

The surface-impedance Phase 0 slice froze the contract, the rational model, the
shared fitter, and the compiler funnel with the guard census reconciled unchanged
at 134 (the single re-authored `_reject_surface_impedance` funnel replaced the
prior surface-impedance rejection, no net change). The follow-up fail-close in the
interoperability adapter adds one reviewed capability guard.
`witwin/maxwell/adapters/tidy3d.py::_convert_material` now raises
`NotImplementedError` for a generic `SurfaceImpedanceMedium` instead of letting it
fall through to the non-dispersive `td.Medium(permittivity=1.0)` path and silently
export as vacuum. A generic surface-impedance medium carries its physics in a
broadband causal rational tangential surface impedance
`E_t = Z_s(omega) (n x H)` with no finite bulk-permittivity equivalent, and the
reference backend has no surface-impedance construct mapped for a rational model
yet (the narrowband good-conductor `LossyMetalMedium` keeps its dedicated
lossy-metal surface path). It is a genuine capability gap, not a public contract:
the guard disappears once the surface-impedance adapter mapping is designed. It is
counted under "External interoperability adapter" (18 -> 19); lower the budget when
the mapping lands.

### Surface-impedance Phase 1 slices S1.1 + S1.2 reconciliation (2026-07-18)

The Phase 1 slices generalized the surface-impedance runtime (axis-aligned
exposed-face layout for finite blocks, mid-domain double-sided plates, multiple
metals, and multiple orientations; the generic per-edge Z-form ADE forward with the
narrowband good conductor as its order-0 case). The single re-authored
`witwin/maxwell/compiler/materials.py::_reject_surface_impedance` funnel now rejects
strictly fewer cases -- finite blocks, mid-domain plates, multiple metals, multiple
orientations, and the generic `SurfaceImpedanceMedium` all compile -- but it remains
one `raise NotImplementedError` site (the oblique/rotated/curved and Bloch cases
still funnel through it), so the capability count is unchanged. No guard was added or
removed: the budget stays at **140**. The adjoint bridge guard
(`fdtd/adjoint/bridge.py`) and the distributed-solver guard
(`fdtd/distributed/solver.py`, a `ValueError`, not census-counted) were widened to
also cover the generic `SurfaceImpedanceMedium` (previously only `LossyMetalMedium`),
closing a fail-open gap without changing the count. The interoperability-adapter
guard for a generic `SurfaceImpedanceMedium` remains until its export mapping lands.

### Surface-impedance staircase generalization G4a reconciliation (2026-07-21)

Stage G4a generalized the surface-impedance boundary from axis-aligned Box plates to
any voxelized (possibly curved) good conductor: a non-Box conductor is staircased from
its node occupancy and its axis-aligned voxel faces (all six orientations, mixed
orientations, cylinder/sphere) compile into masked exposed-face writes
(`witwin/maxwell/compiler/materials.py::_compile_voxel_surface_metal`,
`fdtd/runtime/materials.py::_voxel_surface_writes`). The single
`_reject_surface_impedance` funnel again rejects strictly fewer cases -- curved
conductors now compile instead of being turned away for being non-Box -- while
remaining one `raise NotImplementedError` site: it now funnels only the true
conformal/oblique cases (a rotated Box, whose grid-unaligned normal is not
staircased), a generic rational surface on a curved conductor (the per-edge ADE stays
Box-only), the tangential 2x2 model, the contradictory-owner cases, and the
Bloch-periodic run. No guard was added or removed and the budget is unchanged at
**176**. The remaining SIBC gap is the true oblique/conformal (non-staircase) surface,
not the staircased curved conductor, which is now supported.

## Contract exclusions

`CONTRACT_GUARDS` in the test is the canonical exact-match inventory. Its 24
entries cover these stable contract families:

| Contract family | Count |
| --- | ---: |
| Material frequency-evaluation domains | 11 |
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

### Gyromagnetic export reconciliation (2026-07-18)

The media-validation coverage fix (`is_gyromagnetic` registration) closed a
fail-open export path: a `GyromagneticFerrite` kept scalar `mu_r = mu_infinity`
and exposed no permeability pole, so it bypassed both the magnetic-dispersive and
`mu_r != 1` interoperability guards and silently exported as a plain reciprocal
isotropic medium, dropping the Polder tensor. `adapters/tidy3d.py` now raises
`NotImplementedError` for gyromagnetic media (the external format has no
non-reciprocal tensor model). One reviewed capability guard added, counted under
"External interoperability adapter" (19 -> 20); measured capability total 143 -> 144
and `CAPABILITY_GUARD_BUDGET` raised in the same change.

### Electrostatics reconciliation (2026-07-19, plan 12 Phase 0+1)

The new cell-centred finite-volume electrostatic (Laplace/Poisson) solver adds
eight reviewed capability guards; `CAPABILITY_GUARD_BUDGET` is raised `144 -> 152`
in the same change. All eight fail closed on features that are genuinely out of
this stage's scope and would otherwise be silently mishandled.

| Guard | Message substring | Capability review |
| --- | --- | --- |
| `compiler/electrostatic.py` `_require_electrostatic_boundary` | "requires Scene.boundary = BoundarySpec.none()" | Deferred capability. Electrostatics owns its boundary conditions through `ElectrostaticBoundarySpec`; a PML/periodic Scene boundary would extend the grid with absorber padding that has no electrostatic meaning. Periodic electrostatics is a later phase. |
| `compiler/electrostatic.py` `_static_epsilon_scalar` (PEC) | "represent conductors with ElectrostaticTerminal" | Deferred capability. A PEC-material structure has no dielectric permittivity; conductors are equipotential terminals, not zero-permittivity dielectrics. |
| `compiler/electrostatic.py` `_static_epsilon_scalar` (dispersive) | "do not define a zero-frequency permittivity" | Deferred capability. A dispersive material exposes no DC permittivity and the solver refuses to guess a zero-frequency limit; users must supply an explicit real static value (Phase 4 API). |
| `compiler/electrostatic.py` `_static_epsilon_scalar` (tensor eps) | "Anisotropic (tensor) permittivity is not supported" | Deferred capability. The scalar operator handles isotropic media only; SPD tensor-eps is Phase 4. |
| `compiler/electrostatic.py` `_static_epsilon_scalar` (per-cell tensor) | "Per-cell tensor permittivity is not supported" | Deferred capability, same Phase 4 tensor-eps family. |
| `compiler/electrostatic.py` `_static_epsilon_scalar` (complex) | "not a valid DC static permittivity" | Deferred capability. A complex permittivity is not a DC static value. |
| `electrostatic/runtime.py` `solve_electrostatics` (floating) | "requires the linear-superposition solve" | Deferred capability *at Phase 0+1*. A floating conductor with prescribed charge needs the Phase 2 linear-superposition solve; rejected rather than solved wrongly. Implemented and removed in Phase 2+3 (see below). |
| `electrostatic/runtime.py` `solve_electrostatics` (pure Neumann) | "gauge-singular" | Deferred capability. A pure-Neumann problem with no conductor and no Dirichlet boundary is defined only up to a constant and cannot be solved; the gauge-fixable cases (floating conductors, mean(phi)=0) are now handled in Phase 2. |

This is the historical Phase 0+1 record (eight guards, matching the `144 -> 152`
budget raise); the floating-conductor guard is removed by the Phase 2+3 slice
documented below. Lower this budget as tensor-eps / open boundary (Phase 4) land.

### Electrostatics reconciliation (2026-07-19, plan 12 Phase 2+3)

The Phase 2+3 slice (floating-conductor linear superposition, pure-Neumann gauge
handling, N-terminal capacitance matrix) **implements** the previously deferred
floating-conductor guard, so its `NotImplementedError` in
`electrostatic/runtime.py` (`"requires the linear-superposition solve"`) is
removed. Incompatible floating-charge constraints and missing capacitance return
paths now fail closed with `ValueError` diagnostics (not counted as capability
guards). Net measured capability total `152 -> 151`; `CAPABILITY_GUARD_BUDGET`
is a ceiling and stays `152` (`151 <= 152`).

### Electrostatics reconciliation (2026-07-19, plan 12 differentiability slice)

The reduced electrostatic solve now provides implicit-differentiation backward
(a `torch.autograd.Function`: forward Jacobi-PCG under `no_grad`, backward adjoint
solve on the SPD reduced operator plus a residual VJP), so `d(energy)/d(eps)`,
`dC_ij/d(eps)`, and `d(energy)/d(free_charge)` flow through the fixed-potential
path. Gradients through the **floating-conductor** superposition solve are not
implemented (the prescribed-charge `k x k` reduction is undifferentiated), so
`electrostatic/runtime.py` adds one reviewed capability guard,
`"Gradients through the floating-conductor superposition solve"`: a floating
terminal combined with a differentiable permittivity / free charge fails closed
rather than returning silently wrong gradients. Net measured capability total
`151 -> 152`, exactly at the `CAPABILITY_GUARD_BUDGET` ceiling of `152`. Lower the
budget when the floating-superposition gradient (or tensor-eps / open boundary,
Phase 4) lands.

### Electrostatics reconciliation (2026-07-19, plan 12 audit fix)

Audit finding: the electrostatic compiler silently ignored `Scene.material_regions`
(the RF density design-region object), solving with `eps_r = 1` inside a declared
region instead of failing closed. `compiler/electrostatic.py` `_reject_material_regions`
now raises `NotImplementedError` ("does not support Scene.material_regions") because
`MaterialRegion` design regions are an RF-path feature that is not rasterized into the
electrostatic permittivity; static dielectrics must be `Structure(material=Material(...))`
bulk media. One reviewed capability guard added; net measured capability total
`152 -> 153`; `CAPABILITY_GUARD_BUDGET` raised `152 -> 153` in the same change.

| Guard | Message substring | Capability review |
| --- | --- | --- |
| `compiler/electrostatic.py` `_reject_material_regions` | "does not support Scene.material_regions" | Deferred capability. `MaterialRegion` is the RF differentiable design-region object; the electrostatic compiler rasterizes permittivity only from `Structure` bulk media, so a region would otherwise solve as vacuum. Rejected rather than solved wrongly. |

Lower this budget when MaterialRegion electrostatics (or tensor-eps / open
boundary, Phase 4) lands.
### Deterministic dielectric breakdown reconciliation (2026-07-19)

Plan 13 Phase 4 lands the deterministic field-duration/latching dynamic
dielectric-breakdown feedback model (`DielectricBreakdown` composed into
`Material`, a compiled per-node state machine, a dynamic conductivity scatter, a
typed event log, and a dedicated breakdown-dissipation energy channel). It adds
fifteen fail-closed capability guards, all genuine gaps whose closure belongs to a
later breakdown phase; the measured capability total moves 144 -> 159 and
`CAPABILITY_GUARD_BUDGET` is raised in the same change.

- `media.py` (10): the `DielectricBreakdown` descriptor rejects the four
  unsupported descriptor settings whose state machines are a later phase
  (`model != "field_duration"`, `state != "latching"`, a reserved `recovery`
  model, a reserved `damage_parameters` model). A `Material` carrying a breakdown
  descriptor rejects the six coefficient-path combinations the scalar conductivity
  scatter cannot represent yet (PEC, anisotropic tensor, dispersive poles,
  instantaneous nonlinearity, time modulation, 2D sheet).
- `fdtd/runtime/breakdown.py` (3): `initialize_breakdown_runtime` rejects a complex
  (Bloch) field run, a scene mixing breakdown with dispersive / nonlinear /
  full-anisotropic / modulated media, and a gyromagnetic-ferrite scene — each
  shares or redefines the per-edge electric-update coefficient representation the
  breakdown scatter rewrites.
- `simulation.py` (2): `_validate_breakdown_support` rejects a frequency-domain
  (FDFD) breakdown scene (no dynamic conductivity update exists there) and a
  trainable breakdown scene (the hard field-duration/latching switch is
  non-differentiable at the trigger time; the smooth surrogate is deferred).

The multi-GPU breakdown reject (`fdtd/distributed/solver.py`
`_validate_static_capabilities`) and the event-buffer overflow guard
(`fdtd/runtime/breakdown.py` `finalize_breakdown_data`) raise `ValueError` /
`RuntimeError` respectively and are not counted against the capability budget.
Lower this budget as the later breakdown phases land (recovery/damage state
machines, the coefficient-path compositions, an FDFD or trainable smooth
surrogate, and distributed breakdown).

### SAR reconciliation (2026-07-19, plan 10 merge)

The SAR merge adds four reviewed capability guards; `CAPABILITY_GUARD_BUDGET`
rises `168 -> 172` in the merge-reconciliation change. `witwin/maxwell/sar.py`:
(1) `SARAveraging.connectivity` other than `"cube"` (tissue flood-fill
connectivity is a later averaging phase); (2) `SARAveraging.boundary_policy`
other than `"strict-interior"`; (3) `PowerNormalization.input_power`, because
this build exposes no total injected source-power diagnostic (use
`accepted_power` or `source`). `witwin/maxwell/result.py`: (4) `Result.sar`
consumes FDTD `PowerLossData` only. Four additional SAR raises that guarded pure
usage errors (asking for a mass-averaged peak / field / soft peak without
requesting averaging, and calling `accepted_power` on the bare reducer without a
`Result`) were converted to `ValueError` at merge so they do not occupy
capability budget.

### Electrostatic pre-bias reconciliation (2026-07-19, plan 13 Phase 3 slice)

The electrostatic pre-bias initial-condition slice adds four reviewed capability
guards; `CAPABILITY_GUARD_BUDGET` rises `172 -> 176` in this change. All four live
in `witwin/maxwell/simulation.py` `Simulation._validate_initial_condition_support`
and reject an unsupported combination of `Simulation.fdtd(initial_condition=...)`:

- a non-FDTD method (the pre-bias seeds the time-stepped E buffers; there is no
  time-domain state to seed in the frequency-domain solver);
- a distributed / multi-GPU run (seeding shard-local staggered E buffers with the
  correct halo/ownership is a later phase);
- a trainable / adjoint run (the seeded DC field would enter the taped forward
  state without a differentiated map back to the electrostatic solve);
- a Bloch-periodic run (the real-valued electrostatic seed does not carry the
  complex Bloch phase).

Lower this budget when any of those pre-bias combinations is implemented. The
grid-identity mismatch, the discrete-Gauss tolerance gate, the negative-tolerance
check, and the non-electrostatic-`Result` / wrong-type rejections in
`electrostatic/initial_condition.py` raise `ValueError` / `TypeError` and are not
counted against the capability budget.

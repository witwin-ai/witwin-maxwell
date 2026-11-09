# FDTD Capability-Guard Census

This document is the maintained baseline for the AST guard gate in
`tests/api/public/test_guard_census.py`. The gate distinguishes capability gaps
from deliberate public-contract boundaries so new `NotImplementedError` paths
cannot enter unnoticed.

## Current audited baseline — 2026-09-11

The current capability budget is **206**. The previous committed ceiling of
176 was already stale in the initial checkout
`2b995363649559316fccfc68f1ec0d7b3d079e27`: an AST census of that untouched
baseline finds **229** `NotImplementedError` raises, **29** matching the existing
contract exclusions and **200** capability guards. Thus **24** capability
guards above the old budget predate the SIE work. This reconciliation records
existing capability debt; it does not claim that those gaps were implemented.

The SIE/shared-query work adds **6** capability guards, for **235 total raises =
29 unchanged contract exclusions + 206 capability guards**. The budget update
is `176 + 24 pre-existing unrecorded guards + 6 newly reviewed guards = 206`.
No capability was reclassified as a contract, no exclusion or phrase allowlist
was expanded, and no runtime branch was removed to satisfy the census.

The historical reconciliation ledger below is retained for provenance. Its old
counts are snapshots, superseded by this measured file inventory. Guard counts
measure AST raise sites, not distinct physical features or a completeness score;
dynamic f-string messages contain only their literal fragments in this census.

### Six new capability guards

| Location | Guard and physical capability gap |
| --- | --- |
| `result.py`, physical query backend | `Physical E/H queries currently support SIE and FDTD Results`: the common physical-point query path has no FDFD/other-result interpolation and field-semantic adapter. |
| `result.py`, one-sided FDTD trace | `FDTD Yee interpolation does not define one-sided interface traces`: ordinary staggered-grid interpolation has no interface-side reconstruction. This can be implemented with geometry-aware reconstruction and remains capability debt. |
| `result.py`, multi-frequency FDTD query | `FDTD physical queries currently require one recorded frequency`: the query contract lacks a selected-frequency argument/reconstruction path for multiple retained spectra. |
| `result.py`, native RWG field access | `SIE Results contain RWG currents; use query_fields for E/H instead of Yee field access`: native surface currents cannot currently supply the existing dense Yee-array accessors. This counted compatibility gap does not authorize fabricating a Yee volume. |
| `result.py`, exterior TFSF reconstruction | `FDTD query stencils must lie strictly inside the TFSF total-field box; exterior reconstruction is not implemented`: the common query path does not reconstruct total/scattered semantics for exterior or crossing stencils. |
| `sie/runtime.py`, physical-parameter derivatives | `SIE geometry/material derivatives are not supported; fixed-q residual_loss supports coefficient gradients`: current-vector differentiation exists, while assembly derivatives with respect to geometry/material parameters are absent. |

All six remain capability guards in `test_guard_census.py`; none appears in
`CONTRACT_GUARDS`.

### Existing batch-message repairs

Two public-forward deferral phrases also predate this work, in
`execution/batch_runtime.py`. Their exception type, condition, and reachable
behavior are unchanged. Only the explanation was corrected:

- Material/full-field export states that material snapshots and magnetic
  full-field spectra have no output-buffer allocation in `PreparedBatch`.
- RF port excitation states that per-variant modal injection and port-state
  buffers are required, while the native batch runtime packs `PointDipole`
  drives.

These remain two of the **9** capability guards in that file. Rewording them
does not decrease the budget or pretend that either capability is available.

### Measured capability guards by file

The baseline column was measured directly from the initial checkout above with
the same unchanged contract-exclusion tuple. The current column is the tree
containing this reconciliation. Full current messages appear in the
[exact AST inventory](#exact-current-capability-guard-inventory).

| File | Initial checkout | Current | Net |
| --- | ---: | ---: | ---: |
| `witwin/maxwell/adapters/tidy3d.py` | 20 | 20 | +0 |
| `witwin/maxwell/compiler/circuits.py` | 1 | 1 | +0 |
| `witwin/maxwell/compiler/electrostatic.py` | 6 | 6 | +0 |
| `witwin/maxwell/compiler/materials.py` | 12 | 12 | +0 |
| `witwin/maxwell/compiler/nonlinear_devices.py` | 1 | 1 | +0 |
| `witwin/maxwell/compiler/power_loss.py` | 2 | 2 | +0 |
| `witwin/maxwell/compiler/thin_wire.py` | 2 | 2 | +0 |
| `witwin/maxwell/electrostatic/api.py` | 1 | 1 | +0 |
| `witwin/maxwell/electrostatic/runtime.py` | 3 | 3 | +0 |
| `witwin/maxwell/execution/adjoint.py` | 3 | 3 | +0 |
| `witwin/maxwell/execution/batch.py` | 1 | 1 | +0 |
| `witwin/maxwell/execution/batch_runtime.py` | 9 | 9 | +0 |
| `witwin/maxwell/execution/pool.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdfd/adjoint/bridge.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdfd/solver.py` | 5 | 5 | +0 |
| `witwin/maxwell/fdtd/adjoint/bridge.py` | 15 | 15 | +0 |
| `witwin/maxwell/fdtd/adjoint/core.py` | 4 | 4 | +0 |
| `witwin/maxwell/fdtd/checkpoint.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/distributed/shard_engine.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/distributed/solver.py` | 4 | 4 | +0 |
| `witwin/maxwell/fdtd/excitation/injection.py` | 3 | 3 | +0 |
| `witwin/maxwell/fdtd/excitation/modes/__init__.py` | 6 | 6 | +0 |
| `witwin/maxwell/fdtd/excitation/modes/tem.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/excitation/modes/vector.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/excitation/temporal.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/excitation/tfsf_state.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/lumped.py` | 2 | 2 | +0 |
| `witwin/maxwell/fdtd/networks.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/ports.py` | 10 | 10 | +0 |
| `witwin/maxwell/fdtd/runtime/batch.py` | 4 | 4 | +0 |
| `witwin/maxwell/fdtd/runtime/breakdown.py` | 3 | 3 | +0 |
| `witwin/maxwell/fdtd/runtime/gyromagnetic.py` | 1 | 1 | +0 |
| `witwin/maxwell/fdtd/runtime/materials.py` | 6 | 6 | +0 |
| `witwin/maxwell/fdtd/runtime/stepping.py` | 5 | 5 | +0 |
| `witwin/maxwell/fdtd/wire.py` | 7 | 7 | +0 |
| `witwin/maxwell/media.py` | 18 | 18 | +0 |
| `witwin/maxwell/network_sweep.py` | 3 | 3 | +0 |
| `witwin/maxwell/postprocess/antenna.py` | 1 | 1 | +0 |
| `witwin/maxwell/postprocess/array.py` | 1 | 1 | +0 |
| `witwin/maxwell/postprocess/stratton_chu.py` | 2 | 2 | +0 |
| `witwin/maxwell/result.py` | 2 | 7 | +5 |
| `witwin/maxwell/sar.py` | 3 | 3 | +0 |
| `witwin/maxwell/sie/runtime.py` | 0 | 1 | +1 |
| `witwin/maxwell/simulation.py` | 20 | 20 | +0 |
| `witwin/maxwell/waveport_sweep.py` | 5 | 5 | +0 |
| **Total** | **200** | **206** | **+6** |

### Reproduction and future changes

Run the unchanged three census gates with the `maxwell` environment:

```bash
python -m pytest tests/api/public/test_guard_census.py -q
```

`collect_guards(package_root=...)` also accepts a checkout's `witwin` directory
for a read-only comparison; the no-argument test path retains its existing
behavior. Apply the same `_is_contract` predicate to both inventories, and
compare `(relative_file, message)` multisets rather than line numbers, which
move when code is inserted. The two batch message replacements are wording
changes; excluding those replacements, the six new sites listed above are the
entire change from the initial checkout.

Lower the ceiling when a counted capability is implemented. A future increase
requires a contemporaneous, explicit census and capability explanation; a
deferred capability must not be disguised as a contract exclusion.

## Historical baseline and reconciliation ledger

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

**2026-07-21 update (plan 07 Phase 4, B2 lossy-wire recurrence):** the recurrence
is now wired. The passive series-impedance ADE is consumed by the wire current
update (`witwin/maxwell/fdtd/wire_lossy.py`), the `ohmic_loss` monitor emits real
dissipation, and a finite conductor compiles and runs. The finite-conductor
deferral guard above is REMOVED (-1, "Material compilers" 12 -> 11). The remaining
compiler kind check rejects only malformed (non-pec/finite) conductors and is a
defensive `ValueError`, not a counted capability guard. Two narrower capability
gaps replace it (+2): the finite-conductor reverse/adjoint replay
(`fdtd/wire.py::replay_wire_state`, gated until the B3 conductivity adjoint
transposes the ADE state) and the finite-conductor checkpoint/resume
(`fdtd/checkpoint.py`, the ADE loss state is not yet in the checkpoint schema).
Net budget `175 -> 176`; lower it when the lossy reverse and checkpoint schema
land (B3).

**2026-07-21 update (plan 07 Wave C, B3 conductivity adjoint + multi-GPU
disposition):** the deterministic conductivity adjoint of the dissipation channel
lands as the closed-form `d Re(Z')/d sigma` of the analytic scaled-Bessel internal
impedance (`witwin/maxwell/compiler/wire_impedance.py::internal_impedance_conductivity_gradient`)
plus a PyTorch-native autograd path
(`witwin/maxwell/fdtd/wire_lossy.py::analytic_ac_resistance`,
`LossySegmentModel.conductivity_ac_resistance_gradient`). No guard is removed: the
field-coupled current sensitivity `dI/dsigma` through the recurrence still fails
closed, because the ADE coefficients come from the nondeterministic shared rational
fit (B1) and are not a differentiable/reproducible map of `sigma`, so an exact
reverse replay cannot be certified. The `fdtd/wire.py::replay_wire_state` and
`fdtd/checkpoint.py` finite-conductor guards are only sharpened (the replay message
now names the conductivity-fit nondeterminism and points to the shipped analytic
adjoint), not lifted. One NEW capability guard is added (+1): the distributed
lossy-wire forward reject in
`fdtd/distributed/solver.py::_validate_distributed_wire_support`
("Multi-GPU ThinWire with a finite-conductivity (lossy) conductor is not
supported..."), because the distributed owner runtime builds only the lossless PEC
current update and never constructs the passive series-impedance ADE companion, so
a lossy wire would silently run as PEC across shards. Net budget `176 -> 177`;
lower it when the distributed lossy forward (and its owner-resident ADE state) is
implemented.

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

### Gyromagnetic ferrite general + mixed bias forward reconciliation (2026-07-21)

The plan 08 slice 2a/2b general-bias forward implements the arbitrary- and
mixed-bias magnetization ADE, so the two remaining gyromagnetic forward-runtime
capability guards in `witwin/maxwell/fdtd/runtime/gyromagnetic.py`
`build_gyromagnetic` are **removed**; `CAPABILITY_GUARD_BUDGET` drops `175 -> 173`
("Time-domain runtime" `20 -> 18`).

- Removed (`-1`): the general (non-axis-aligned) bias reject. The general-bias
  forward is a pure per-cell coordinate rotation of the SAME axis-aligned contracted
  implicit-midpoint (Cayley) update -- no new integrator, no new coefficients. The
  compiled layout already carries a per-cell bias, right-handed local frame
  `[u|v|w]` (`w = b`), and per-cell `Phi`/`Gamma`/`B`/`C`; the runtime gathers the
  RF drive `h_u = u . H`, `h_v = v . H` from all three lab `H` components and
  scatters the back-reaction `dM = dm_u u + dm_v v` onto all three, with identity
  collocation `C = I` reused from the axis-aligned slice (contract section 5). Because
  `[u|v]^T[u|v] = I`, the local reduction and the coupled 2x2 solve are algebraically
  identical to the fast path, so the general path reproduces it bit-for-bit for an
  axis-aligned bias (rotation-equivalence gate). Verified against the Phase-0 torch
  oracle for `b = (1,1,1)/sqrt(3)` at `reference_polder_rtol`.
- Removed (`-1`): the mixed-bias-direction reject (the former "single uniform bias
  direction" guard). A mixed-bias scene -- different bias axes, opposed signs on one
  axis (e.g. a `+z`/`-z` latching circulator), or differing magnitudes/materials --
  is supported through the SAME per-cell general path. The magnetization ADE is
  purely local (no spatial coupling in the magnetization update; fields couple only
  through the ordinary reciprocal Yee curl), so a mixed-bias scene is the direct sum
  of independent per-cell passive blocks: each cell precesses around its own `b` with
  the correct handedness (the right-handed local frame flips the lab-frame
  off-diagonal for an opposed bias), and per-cell passivity gives global passivity.
  Verified bit-for-bit against the two standalone single-material runs
  (`test_mixed_bias_per_cell_independence`) and by the opposed-region frame check in
  the forward suite. The old guard rejected mixed bias only because the axis-aligned
  fast path applied a single global transverse sign; the fast path is now used only
  for a uniform axis-aligned bias, and everything else routes to the per-cell path.

Only the Bloch-periodic ferrite reject remains in that file (the real
magnetization-ADE correction cannot carry the complex Bloch phase). The
identity-collocation general path is accurate for smooth/uniform fields and passive
(non-secularly-growing) in a closed lossless cavity, verified by
`test_cuda_oblique_driven_cavity_energy_non_growth_lossless` (energy-envelope
non-growth over 12k steps); the naive `eps0|E|^2 + mu0|H|^2` proxy oscillates within
a bounded envelope for an oblique mode because it is not the exact conserved
leapfrog invariant. The `PerturbationMedium`-wraps-a-ferrite reject, the FDFD /
adjoint / distributed-shard consumer rejects, and the two `GyromagneticFerrite`
scalar frequency-evaluation contract guards are all unchanged.

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

### Electrostatics reconciliation (2026-07-21, plan 12 Phase 4 SPD tensor eps, stage H2a)

The SPD tensor-permittivity finite-volume operator lands: a full symmetric
positive-definite 3x3 `epsilon(x)` is discretized as the existing conservative
two-point face flux for the diagonal entries plus a symmetric cross-derivative
coupling for the off-diagonal entries. The cross term is the gradient of a
discrete quadratic energy, so the operator is symmetric to floating-point
tolerance (verified: dense `<Ax,y>=<x,Ay>`), positive-definite (symmetric
eigenvalues), second-order convergent (rotated-frame MMS), and reduces exactly to
the isotropic/diagonal face-flux path when the off-diagonals vanish.

Net capability-guard change is **zero** (budget stays **175**): one guard is
removed and one reachable guard is added.

- Removed (`-1`): the `compiler/electrostatic.py` anisotropic-tensor reject
  (`"Anisotropic (tensor) permittivity is not supported..."`). A
  `DiagonalTensor3` / `Tensor3x3` material now compiles into a per-cell
  `TensorEpsilon` field and solves. (The former `_static_epsilon_scalar` extractor
  is now `_material_static_matrix`, returning a validated SPD 3x3; its PEC,
  dispersive, per-cell-tensor, and complex rejects are unchanged, and the
  symmetric/positive-definite validation of an invalid tensor is a `ValueError`,
  not a capability guard.)
- Added (`+1`): `electrostatic/runtime.py` `_reject_trainable_tensor`
  (`"Differentiable electrostatics through a full tensor (anisotropic)
  permittivity is not implemented..."`). The tensor operator's off-diagonal
  cross-flux has no reverse-mode (implicit-diff) VJP, so a trainable tensor
  permittivity or a trainable free charge alongside a tensor dielectric fails
  closed instead of silently detaching its gradient. Reachable (a trainable
  `ChargeDensity` in a tensor-dielectric scene). Lower this budget when the
  tensor-eps backward (or the open-boundary work, stage H2b) lands.

### Electrostatics reconciliation (2026-07-21, plan 12 Phase 4 open boundary / truncation, stage H2b)

The controlled open-boundary (domain-extension) piece lands: an opt-in
`TruncationEstimate` runs one additional enlarged grounded-box capacitance solve
(interior grid held byte-identical, the grounded box pushed out by whole cells) and
reports the finite-enclosure truncation error plus a 1/L Richardson extrapolation
to the infinite-domain limit (`CapacitanceData.truncation_estimate`). The second
solve is never run silently; it is triggered only by `Simulation.capacitance(...,
truncation_estimate=...)`.

Net capability-guard change is **+1** (budget `175 -> 176`):

- Added (`+1`): `electrostatic/api.py` `_normalize_bc_entry` rejects an `open`
  (infinite-domain) boundary kind (`"Open (infinite-domain) electrostatic
  boundaries are not implemented..."`). There is no exact radiation condition on
  the scalar potential at a finite Cartesian face, so an `open` face fails closed
  rather than silently solving a grounded/insulating truncation the user did not
  request. The supported path is the domain-extension `truncation_estimate`. A
  boundary-element exact open boundary is a later phase; lower this budget when it
  lands.

The `truncation_estimate` misuse cases (a non-Dirichlet enclosure, a non-uniform
grid) raise `ValueError` (usage errors, not capability guards). The tensor-eps
**differentiability disposition** is decided as **fail-closed**: Phase 4 owns the
forward SPD tensor-eps operator and the plan defers electrostatic gradients to
Phase 5, so the H2a `_reject_trainable_tensor` guard is retained unchanged and no
differentiability guard is added or removed here.

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

## 2026-09-08 normalization contract

The unified time planner adds three permanent normalization contracts: an incident
waveform must exist, every participating drive must share that waveform, and a
circuit-driven field has no implicit single incident waveform. These reject an
undefined quotient and do not change the capability budget. The matching entries
are recorded in `tests/api/public/test_guard_census.py`.

## Exact current capability-guard inventory

Snapshot: 2026-09-11, 206 AST raise sites. Line numbers identify this snapshot;
the complete literal message is the stable review key. Contract exclusions are
not included. HTML code quoting preserves punctuation inside table cells.

| # | File and line | Literal message fragments |
| ---: | --- | --- |
| 1 | `witwin/maxwell/adapters/tidy3d.py:286` | <code>Tidy3D&#x27;s FullyAnisotropicMedium is strictly non-dispersive, so a full off-diagonal permittivity tensor combined with dispersive poles has no Tidy3D equivalent; supply an axis-aligned DiagonalTensor3 (which exports as a per-axis dispersive AnisotropicMedium) or drop the dispersion.</code> |
| 2 | `witwin/maxwell/adapters/tidy3d.py:319` | <code>Tidy3D has no anisotropic magnetic medium: its AnisotropicMedium and FullyAnisotropicMedium describe the electric permittivity/conductivity tensor with mu_r = 1, so a mu_tensor (anisotropic permeability) has no Tidy3D equivalent.</code> |
| 3 | `witwin/maxwell/adapters/tidy3d.py:363` | <code>Tidy3D has no second-order (chi2 / second-harmonic-generation) nonlinear medium; its public nonlinear API is the chi3 / Kerr / two-photon-absorption family only. Validate chi2 against an FDTD analytic reference instead.</code> |
| 4 | `witwin/maxwell/adapters/tidy3d.py:433` | <code>Tidy3D&#x27;s ModulationSpec modulates only the non-dispersive permittivity (eps_inf) and conductivity, but maxwell folds the same per-step modulation factor through the dispersive polarization current and the instantaneous Kerr/chi2/TPA coefficient, so a time-modulated Material that also carries dispersion or a nonlinearity has no equivalent Tidy3D construct. Validate the modulated dispersive/nonlinear cell against an FDTD analytic reference instead.</code> |
| 5 | `witwin/maxwell/adapters/tidy3d.py:443` | <code>Tidy3D represents a spatially-varying modulation profile as a SpaceModulation carrying a SpatialDataArray on absolute simulation coordinates, but maxwell&#x27;s per-cell modulation amplitude/phase tensors are defined relative to the owning structure&#x27;s Box and are not available at material-conversion time. Export a scalar-amplitude, scalar-phase ModulationSpec instead.</code> |
| 6 | `witwin/maxwell/adapters/tidy3d.py:615` | <code>Tidy3D export of a spatially-varying custom dispersive pole would need a CustomPoleResidue on a SpatialDataArray in absolute simulation coordinates, but maxwell&#x27;s per-cell pole-strength grid is defined relative to the owning structure&#x27;s Box and cannot be resolved into Tidy3D coordinates at material-conversion time. Export a spatially-uniform custom pole (it lowers to the equivalent homogeneous pole) or validate the per-cell profile against an FDTD/analytic reference.</code> |
| 7 | `witwin/maxwell/adapters/tidy3d.py:660` | <code>Tidy3D export of a spatially-varying PerturbationMedium would need a CustomMedium on a SpatialDataArray in absolute simulation coordinates, but its perturbation field is defined relative to the owning structure&#x27;s Box and cannot be resolved into Tidy3D coordinates at material-conversion time. Export a spatially-uniform perturbation (it lowers to a homogeneous permittivity shift eps_base + eps_sensitivity*value) or validate the perturbed profile against an FDTD/analytic reference.</code> |
| 8 | `witwin/maxwell/adapters/tidy3d.py:721` | <code>Reference-backend export of a generic SurfaceImpedanceMedium is not yet designed: its physics is a broadband causal rational tangential surface impedance E_t = Z_s(omega) (n x H), which has no finite bulk-permittivity equivalent and no reference-backend surface-impedance construct mapped for it. The narrowband good-conductor LossyMetalMedium exports through the dedicated lossy-metal surface path; a rational model has no such mapping until the surface-impedance adapter is designed. Resolve the metal volumetrically with Material(sigma_e=...), use a LossyMetalMedium for a good conductor, or use Material.pec() for a lossless conductor.</code> |
| 9 | `witwin/maxwell/adapters/tidy3d.py:741` | <code>Tidy3D export for a gyromagnetic ferrite has no equivalent: its permeability is a non-reciprocal complex 3x3 Polder tensor, and Tidy3D has no gyromagnetic (magnetic, non-reciprocal) material model -- its FDTD solver fixes mu_r = 1 and exposes no permeability tensor. Validate the ferrite against the analytic LLG reference oracle (tests/materials/ferrite/test_ferrite_reference.py) instead.</code> |
| 10 | `witwin/maxwell/adapters/tidy3d.py:932` | <code>Geometry type &#x27;&#x27; exposes no surface mesh, so it has no Tidy3D geometry mapping. Analytic primitives (box, sphere, cylinder, cone) and any mesh-able primitive are supported.</code> |
| 11 | `witwin/maxwell/adapters/tidy3d.py:954` | <code>Tidy3D export supports MaterialRegion with Box geometry only.</code> |
| 12 | `witwin/maxwell/adapters/tidy3d.py:956` | <code>Tidy3D export cannot preserve a filtered MaterialRegion density field.</code> |
| 13 | `witwin/maxwell/adapters/tidy3d.py:964` | <code>Tidy3D export currently supports only spatially uniform MaterialRegion density.</code> |
| 14 | `witwin/maxwell/adapters/tidy3d.py:981` | <code>Tidy3D export does not support MaterialRegion with relative permeability other than 1.</code> |
| 15 | `witwin/maxwell/adapters/tidy3d.py:1275` | <code>Source type &#x27;&#x27; has no Tidy3D source mapping.</code> |
| 16 | `witwin/maxwell/adapters/tidy3d.py:1679` | <code>Monitor type &#x27;&#x27; has no Tidy3D monitor mapping.</code> |
| 17 | `witwin/maxwell/adapters/tidy3d.py:1857` | <code>Tidy3D symmetry is center-based; Maxwell face symmetry can only be exported when the scene is translation-invariant along the symmetry axis.</code> |
| 18 | `witwin/maxwell/adapters/tidy3d.py:1862` | <code>Tidy3D symmetry is center-based; Maxwell face symmetry can only be exported when every structure spans the symmetry axis.</code> |
| 19 | `witwin/maxwell/adapters/tidy3d.py:1869` | <code>Tidy3D symmetry is center-based; Maxwell MaterialRegion must span the symmetry axis for an equivalent export.</code> |
| 20 | `witwin/maxwell/adapters/tidy3d.py:1875` | <code>Tidy3D symmetry is center-based; Maxwell face symmetry with a localized or symmetry-axis-varying source has no equivalent direct export.</code> |
| 21 | `witwin/maxwell/compiler/circuits.py:275` | <code>Circuit  contains nonlinear devices  whose nonlinear terminal law cannot be assembled into the constant-conductance stamp of the linear MNA / FDTD Norton-companion runtime; that path has no Newton iteration and would solve the network with the device absent. This fails closed until the nonlinear device-runtime integration lands.</code> |
| 22 | `witwin/maxwell/compiler/electrostatic.py:99` | <code>The electrostatic solver uses ElectrostaticBoundarySpec for its boundary conditions and requires Scene.boundary = BoundarySpec.none() so the grid is not extended by PML/periodic padding (got Scene boundary kind ).</code> |
| 23 | `witwin/maxwell/compiler/electrostatic.py:116` | <code>The electrostatic solver does not support Scene.material_regions (found ); MaterialRegion design regions are an RF-path feature and are not rasterized into the electrostatic permittivity. Represent static dielectrics with Structure(material=Material(...)) bulk media instead.</code> |
| 24 | `witwin/maxwell/compiler/electrostatic.py:141` | <code>A PEC-material structure has no dielectric permittivity for the electrostatic solver; represent conductors with ElectrostaticTerminal (equipotential) instead.</code> |
| 25 | `witwin/maxwell/compiler/electrostatic.py:146` | <code>Dispersive materials do not define a zero-frequency permittivity; the electrostatic solver refuses to guess a DC limit. Supply a non-dispersive Material with an explicit real static permittivity.</code> |
| 26 | `witwin/maxwell/compiler/electrostatic.py:159` | <code>Per-cell tensor permittivity is not supported by the electrostatic compiler.</code> |
| 27 | `witwin/maxwell/compiler/electrostatic.py:167` | <code>Complex permittivity is not a valid DC static permittivity for electrostatics.</code> |
| 28 | `witwin/maxwell/compiler/materials.py:116` | <code>The Maxwell material compiler stores  on the diagonal Yee components; off-diagonal Tensor3x3 coupling is realized only for permittivity via the coupled-tensor eps kernel. A full off-diagonal  tensor has no corresponding coupled update kernel on the Yee grid.</code> |
| 29 | `witwin/maxwell/compiler/materials.py:176` | <code>The Maxwell material compiler currently supports scalar isotropic or axis-aligned DiagonalTensor3 material samples only.</code> |
| 30 | `witwin/maxwell/compiler/materials.py:319` | <code>Polarized (Kottke) subpixel averaging is not implemented for full (off-diagonal) anisotropic permittivity; use arithmetic subpixel averaging.</code> |
| 31 | `witwin/maxwell/compiler/materials.py:708` | <code>Polarized (Kottke) subpixel averaging is not implemented for full (off-diagonal) anisotropic permittivity; use arithmetic subpixel averaging.</code> |
| 32 | `witwin/maxwell/compiler/materials.py:1118` | <code>Partial sampling along periodic axes requires opposite-face halo routing.</code> |
| 33 | `witwin/maxwell/compiler/materials.py:1666` | <code> sheets currently support Box structure geometry only, got .</code> |
| 34 | `witwin/maxwell/compiler/materials.py:1674` | <code> sheets require an axis-aligned (unrotated) Box geometry.</code> |
| 35 | `witwin/maxwell/compiler/materials.py:1796` | <code>Surface-impedance boundary:  The runtime is generalized to this case in ; until then resolve the metal volumetrically with Material(sigma_e=...) or use Material.pec() for a lossless conductor.</code> |
| 36 | `witwin/maxwell/compiler/materials.py:2666` | <code>Local material reuse requires fixed static edge-native materials.</code> |
| 37 | `witwin/maxwell/compiler/materials.py:2668` | <code>Local node sampling for perturbation grids requires a physical-coordinate sampler.</code> |
| 38 | `witwin/maxwell/compiler/materials.py:2705` | <code>Local structure support is proven for Box and Sphere geometry.</code> |
| 39 | `witwin/maxwell/compiler/materials.py:2721` | <code>Local structure changes require static edge-native volume materials.</code> |
| 40 | `witwin/maxwell/compiler/nonlinear_devices.py:263` | <code>Diodes  declare a nonzero series_resistance, but the compiled conduction law is the ideal Shockley junction i(v); the series-resistance branch (an internal node with the ohmic drop) is not assembled, so honouring the parameter requires the extended device topology. This fails closed until that branch is implemented rather than silently solving the resistance-free junction.</code> |
| 41 | `witwin/maxwell/compiler/power_loss.py:141` | <code>Automatic conduction loss does not accept 2D sheet materials: the current material compiler lowers sheet conductance into volume conductivity and no longer retains the surface identity required for a surface-loss channel.</code> |
| 42 | `witwin/maxwell/compiler/power_loss.py:151` | <code>Automatic conduction loss currently supports static electric conductivity only; magnetic conductivity requires full-frequency H fields and a separate channel.</code> |
| 43 | `witwin/maxwell/compiler/thin_wire.py:875` | <code>ThinWire  with a Bloch boundary requires phase-aware wire topology, which is not yet supported.</code> |
| 44 | `witwin/maxwell/compiler/thin_wire.py:1547` | <code>ThinWire  requires a locally isotropic host material.</code> |
| 45 | `witwin/maxwell/electrostatic/api.py:124` | <code>Open (infinite-domain) electrostatic boundaries are not implemented: a grounded-box (Dirichlet) or insulating (Neumann) enclosure is a finite truncation of an open problem, and there is no exact radiation condition on the scalar potential at a finite Cartesian face. Model an isolated structure by enlarging the grounded-box domain until the capacitance converges, and quantify the residual truncation error with the opt-in truncation_estimate on Simulation.capacitance(...). A boundary-element exact open boundary is a later phase.</code> |
| 46 | `witwin/maxwell/electrostatic/runtime.py:581` | <code>Differentiable electrostatics through a full tensor (anisotropic) permittivity is not implemented: the tensor operator&#x27;s off-diagonal cross-flux terms have no reverse-mode (implicit-diff) VJP, so a trainable tensor permittivity or free charge would silently detach its gradient. Detach the permittivity and free charge, or use an isotropic Material for a differentiable electrostatic solve.</code> |
| 47 | `witwin/maxwell/electrostatic/runtime.py:751` | <code>Pure-Neumann electrostatic problems are gauge-singular (potential defined only up to a constant) and require charge-compatibility plus a gauge fix; supply at least one Dirichlet boundary face or a fixed-potential/grounded terminal.</code> |
| 48 | `witwin/maxwell/electrostatic/runtime.py:761` | <code>Gradients through the floating-conductor superposition solve are not implemented; the prescribed-charge k x k reduction is not yet differentiated. Detach the permittivity / free charge, or use only fixed-potential (grounded / potential=) terminals for a differentiable electrostatic solve.</code> |
| 49 | `witwin/maxwell/execution/adjoint.py:17` | <code>Batch adjoint does not register material-region geometry inputs; use density parameters.</code> |
| 50 | `witwin/maxwell/execution/adjoint.py:19` | <code>Batch adjoint has no material magnetic-permeability gradient channel.</code> |
| 51 | `witwin/maxwell/execution/adjoint.py:29` | <code>Batch geometry adjoint requires nonmagnetic, nonconductive dielectric structures; magnetic, conductivity and PEC occupancy gradient channels are not available.</code> |
| 52 | `witwin/maxwell/execution/batch.py:279` | <code>source_scales currently requires an explicit Scene source, not a port-generated source.</code> |
| 53 | `witwin/maxwell/execution/batch_runtime.py:75` | <code>PreparedBatch currently requires one GPU and one execution slot.</code> |
| 54 | `witwin/maxwell/execution/batch_runtime.py:79` | <code>Batch material snapshots and magnetic full-field spectra have no output-buffer allocation in PreparedBatch.</code> |
| 55 | `witwin/maxwell/execution/batch_runtime.py:84` | <code>Cross-device batch result export requires a destination allocation pool.</code> |
| 56 | `witwin/maxwell/execution/batch_runtime.py:223` | <code>Native batch execution does not yet support Scene..</code> |
| 57 | `witwin/maxwell/execution/batch_runtime.py:225` | <code>Batch RF port excitation requires per-variant modal injection and port-state buffers; the native batch runtime only packs PointDipole drives.</code> |
| 58 | `witwin/maxwell/execution/batch_runtime.py:230` | <code>The current native batch tier supports PointDipole sources.</code> |
| 59 | `witwin/maxwell/execution/batch_runtime.py:232` | <code>The current native batch tier supports point, plane and flux monitors.</code> |
| 60 | `witwin/maxwell/execution/batch_runtime.py:416` | <code>Batch full-field spectra require GPU-driven accumulation.</code> |
| 61 | `witwin/maxwell/execution/batch_runtime.py:812` | <code>Automatic slot selection requires measured throughput tuning.</code> |
| 62 | `witwin/maxwell/execution/pool.py:24` | <code>Batch hard budgets require the native PyTorch caching allocator.</code> |
| 63 | `witwin/maxwell/fdfd/adjoint/bridge.py:78` | <code></code> |
| 64 | `witwin/maxwell/fdfd/solver.py:156` | <code>FDFD static parity for Kerr nonlinear media is deferred (user decision 2026-07-11): a frequency-domain solve has no single linear operator for a field-amplitude-dependent permittivity. Use FDTD, which models the nonlinearity directly.</code> |
| 65 | `witwin/maxwell/fdfd/solver.py:162` | <code>FDFD static parity for full (off-diagonal Tensor3x3) anisotropic permittivity is deferred (user decision 2026-07-11): the curl-curl operator assembly here forms only per-axis diagonal material components. Use FDTD, which models the coupled tensor.</code> |
| 66 | `witwin/maxwell/fdfd/solver.py:168` | <code>FDFD static parity for magnetic media and magnetic dispersion is deferred (user decision 2026-07-11): this solver assembles an electric-only operator with mu_r = 1. Use FDTD, which models magnetic response.</code> |
| 67 | `witwin/maxwell/fdfd/solver.py:305` | <code>FDFD static parity for in-domain PEC materials is deferred (user decision 2026-07-11): the in-domain Dirichlet/mask handling for a perfect conductor is unimplemented here. Use FDTD for PEC-material scenes.</code> |
| 68 | `witwin/maxwell/fdfd/solver.py:316` | <code>FDFD does not support GyromagneticFerrite media: the frequency-domain solver would ingest only the diagonal background permeability and silently drop the non-reciprocal off-diagonal Polder gyrotropy. Use FDTD, whose magnetization-ADE runtime carries the gyromagnetic tensor.</code> |
| 69 | `witwin/maxwell/fdtd/adjoint/bridge.py:524` | <code>Differentiable embedded-network FDTD does not support explicit delay state: the bidirectional reference-plane ring couples steps up to max_delay_steps apart (possibly across checkpoint segments) and the fractional-delay filter is an IIR recurrence, neither of which the segment-local network pullback can reverse without a delay-aware reverse ring. Forward runs (including checkpoint/resume, whose schema now carries the delay ring, filter, and cursor) remain fully supported.</code> |
| 70 | `witwin/maxwell/fdtd/adjoint/bridge.py:542` | <code>Differentiable embedded RationalModel supports residues and direct terms only; trainable  are not supported.</code> |
| 71 | `witwin/maxwell/fdtd/adjoint/bridge.py:549` | <code>Differentiable embedded networks accept trainable residues/direct on a pre-fitted RationalModel; direct trainable state-space matrices are not supported.</code> |
| 72 | `witwin/maxwell/fdtd/adjoint/bridge.py:560` | <code></code> |
| 73 | `witwin/maxwell/fdtd/adjoint/bridge.py:573` | <code>Differentiable thin-wire FDTD requires a wire-aware reverse for TFSF, magnetic dispersion, and standalone lumped elements; these coupled state transposes are not part of the axis-aligned PEC wire reverse.</code> |
| 74 | `witwin/maxwell/fdtd/adjoint/bridge.py:586` | <code>FDTD adjoint does not support full (off-diagonal) anisotropic media overlapping the CPML absorber: the reverse off-diagonal correction uses the un-stretched collocated curl and does not replay the CPML coordinate stretch the forward applies.</code> |
| 75 | `witwin/maxwell/fdtd/adjoint/bridge.py:603` | <code>FDTD adjoint does not support electric conductivity on Bloch boundaries; the complex-field reverse replay applies the semi-implicit conduction-loss coefficient on the real electric update only.</code> |
| 76 | `witwin/maxwell/fdtd/adjoint/bridge.py:609` | <code>FDTD adjoint currently supports none, pml, pec, and Bloch faces only.</code> |
| 77 | `witwin/maxwell/fdtd/adjoint/bridge.py:619` | <code>Differentiable RF port  requires an active source or a series R/L/C termination; observer-only contour ports are not yet supported.</code> |
| 78 | `witwin/maxwell/fdtd/adjoint/bridge.py:624` | <code>Differentiable RF port  requires series R/L/C topology.</code> |
| 79 | `witwin/maxwell/fdtd/adjoint/bridge.py:628` | <code>Differentiable RF port  does not support an open internal series resistance.</code> |
| 80 | `witwin/maxwell/fdtd/adjoint/bridge.py:634` | <code>Differentiable lumped element  requires series topology.</code> |
| 81 | `witwin/maxwell/fdtd/adjoint/bridge.py:638` | <code>Differentiable lumped element  does not support an open internal series resistance.</code> |
| 82 | `witwin/maxwell/fdtd/adjoint/bridge.py:707` | <code></code> |
| 83 | `witwin/maxwell/fdtd/adjoint/bridge.py:712` | <code>normalize_source divides monitor spectra by a single incident source spectrum, which is ill-defined when the scene drives sources with different source-time spectra; give the sources a common source_time or disable normalize_source.</code> |
| 84 | `witwin/maxwell/fdtd/adjoint/core.py:394` | <code>Callable CustomSourceTime(fn) is not supported in the FDTD adjoint; provide a sampled (times, amplitudes) table instead.</code> |
| 85 | `witwin/maxwell/fdtd/adjoint/core.py:716` | <code>Bloch source-term reverse currently supports uniform point-source patches only.</code> |
| 86 | `witwin/maxwell/fdtd/adjoint/core.py:2925` | <code>FDTD adjoint replay does not support time-modulated media.</code> |
| 87 | `witwin/maxwell/fdtd/adjoint/core.py:3125` | <code>Complex-field CPML adjoint replay requires exactly one PML axis and two Bloch axes: the reverse split-field update carries the transverse wrap phase on the two periodic axes and the recursive-convolution stretch on the single absorbing axis, so a face layout mixing Bloch and PML on one axis or leaving more than one absorbing axis is not expressible on that split.</code> |
| 88 | `witwin/maxwell/fdtd/checkpoint.py:447` | <code>Checkpoint/resume of a finite-conductor thin wire is not yet implemented: the auxiliary ADE loss state is not part of the checkpoint schema (added with the lossy reverse pass, B3). Run a PEC wire to checkpoint a thin-wire simulation.</code> |
| 89 | `witwin/maxwell/fdtd/distributed/shard_engine.py:263` | <code>Multi-GPU distributed FDTD does not support GyromagneticFerrite media: the shard phases do not run the magnetization-ADE hooks and the shard-local gyromagnetic layout has no rank-seam handling, so a joint solve would silently drop the non-reciprocal gyrotropy. Run the ferrite scene on a single device.</code> |
| 90 | `witwin/maxwell/fdtd/distributed/solver.py:616` | <code>Multi-GPU ThinWire with a finite-conductivity (lossy) conductor is not supported: the distributed owner runtime builds only the lossless PEC current update and never constructs the passive series-impedance ADE companion, so a lossy wire would silently run as PEC across shards. The owner-reduction contract does not carry the per-segment ADE loss state. Run a lossy thin wire on the single-GPU FDTD path.</code> |
| 91 | `witwin/maxwell/fdtd/distributed/solver.py:637` | <code>Multi-GPU ThinWire with a distributed CPML boundary has no verified wire-edge/PML ownership across the x split; use a non-absorbing boundary or run single-device FDTD.</code> |
| 92 | `witwin/maxwell/fdtd/distributed/solver.py:643` | <code>Multi-GPU ThinWire with a distributed Mur absorbing boundary has no verified wire-edge/boundary ownership across the x split; use a non-absorbing boundary or run single-device FDTD.</code> |
| 93 | `witwin/maxwell/fdtd/distributed/solver.py:649` | <code>Multi-GPU ThinWire and an embedded network or lumped circuit each claim owner-resident coordination state on one shard with no distributed ownership merge between them; run them in separate simulations.</code> |
| 94 | `witwin/maxwell/fdtd/excitation/injection.py:649` | <code> currently supports only none, pml, pec, or pmc boundaries.</code> |
| 95 | `witwin/maxwell/fdtd/excitation/injection.py:790` | <code>PlaneWave soft injection does not support Bloch boundaries; use a TFSF source.</code> |
| 96 | `witwin/maxwell/fdtd/excitation/injection.py:801` | <code>PlaneWave soft injection supports periodic boundaries only on zero-phase transverse axes; incompatible axes: .</code> |
| 97 | `witwin/maxwell/fdtd/excitation/modes/__init__.py:560` | <code>ModeSource mode solving is undefined under a periodic/Bloch transverse boundary: the aperture wraps with the Bloch phase, so the guided profile is an eigenfunction of a complex-Hermitian Bloch-phase transverse operator, whereas this solver builds the real, zero-boundary transverse operator. Use none/pml/pec/pmc on the mode plane.</code> |
| 98 | `witwin/maxwell/fdtd/excitation/modes/__init__.py:596` | <code>Differentiable ModeSource rebuilding with an internal PEC conductor requires a PEC mode-shape eigen-adjoint.</code> |
| 99 | `witwin/maxwell/fdtd/excitation/modes/__init__.py:633` | <code>Lossy PEC-aware modes require a device-resident complex vector eigensolve; the scalar complex mode path cannot enforce internal conductors.</code> |
| 100 | `witwin/maxwell/fdtd/excitation/modes/__init__.py:641` | <code>Differentiable material parameters with an internal PEC conductor require a PEC mode-shape eigen-adjoint.</code> |
| 101 | `witwin/maxwell/fdtd/excitation/modes/__init__.py:720` | <code>ModeSource scalar eigenmode solve is undefined for a magnetic (mu_r != 1) plane: the scalar Helmholtz reduction folds mu into a single index and scales the magnetic profile by n_eff/eta0, both of which assume a non-magnetic medium. A magnetic plane must be solved with the full-vector operator, which carries mu per component.</code> |
| 102 | `witwin/maxwell/fdtd/excitation/modes/__init__.py:727` | <code>Differentiable ModeSource gradients through a lossy (complex) permittivity require a non-Hermitian eigen-adjoint; the differentiable path solves the Hermitian real-permittivity problem.</code> |
| 103 | `witwin/maxwell/fdtd/excitation/modes/tem.py:310` | <code>TEM WavePort electrostatic normalization requires a uniformly filled cross-section; use a hybrid mode for inhomogeneous transmission lines.</code> |
| 104 | `witwin/maxwell/fdtd/excitation/modes/vector.py:165` | <code>Differentiable PEC geometry requires a mode-shape eigen-adjoint; the PEC-aware forward mode solve uses a hard conductor boundary.</code> |
| 105 | `witwin/maxwell/fdtd/excitation/temporal.py:514` | <code>Batch source tables require undelayed point patches.</code> |
| 106 | `witwin/maxwell/fdtd/excitation/tfsf_state.py:769` | <code>TFSF injection currently supports only none, pml, pec, or pmc boundaries.</code> |
| 107 | `witwin/maxwell/fdtd/lumped.py:445` | <code>Differentiable lumped replay currently supports series R/L/C topology only.</code> |
| 108 | `witwin/maxwell/fdtd/lumped.py:531` | <code>Differentiable lumped pullback currently supports series R/L/C topology only.</code> |
| 109 | `witwin/maxwell/fdtd/networks.py:574` | <code>Differentiable embedded-network replay does not support explicit delay state: the bidirectional reference-plane ring couples a step to samples written up to max_delay_steps earlier (a coupling that can span checkpoint segments), and the Thiran fractional-delay filter is an IIR recurrence over consecutive steps. The segment-local network pullback reverses only a self-contained same-step recurrence, so a correct delay adjoint requires a delay-aware reverse ring carried across segments. The delay forward state is captured in the checkpoint schema for forward resume; the reverse pass is intentionally fail-closed rather than approximate.</code> |
| 110 | `witwin/maxwell/fdtd/ports.py:412` | <code>Port  trainable reference_impedance is not supported by the FDTD lumped adjoint.</code> |
| 111 | `witwin/maxwell/fdtd/ports.py:419` | <code>Port  trainable source_impedance is not supported by the FDTD lumped adjoint; train amplitude or series R/L/C values instead.</code> |
| 112 | `witwin/maxwell/fdtd/ports.py:429` | <code>Port  trainable ParallelRLC is not supported by the FDTD lumped adjoint; use a SeriesRLC termination.</code> |
| 113 | `witwin/maxwell/fdtd/ports.py:437` | <code>Lumped FDTD coupling currently requires real Yee fields.</code> |
| 114 | `witwin/maxwell/fdtd/ports.py:439` | <code>Lumped FDTD coupling in conductive media requires a conductance-aware port update coefficient.</code> |
| 115 | `witwin/maxwell/fdtd/ports.py:443` | <code>Lumped FDTD coupling in electrically dispersive media requires the ADE correction in the joint circuit equation.</code> |
| 116 | `witwin/maxwell/fdtd/ports.py:447` | <code>Lumped FDTD coupling in nonlinear media requires a field-dependent circuit update coefficient.</code> |
| 117 | `witwin/maxwell/fdtd/ports.py:451` | <code>Lumped FDTD coupling in time-modulated media requires a time-dependent circuit update coefficient.</code> |
| 118 | `witwin/maxwell/fdtd/ports.py:455` | <code>Lumped FDTD coupling in full anisotropy requires a tensor-valued circuit injection operator.</code> |
| 119 | `witwin/maxwell/fdtd/ports.py:598` | <code>PortExcitation requires a device-native CW, GaussianPulse, or RickerWavelet source_time.</code> |
| 120 | `witwin/maxwell/fdtd/runtime/batch.py:65` | <code>Native batch field execution does not yet support .</code> |
| 121 | `witwin/maxwell/fdtd/runtime/batch.py:69` | <code>Native batch field execution does not yet support .</code> |
| 122 | `witwin/maxwell/fdtd/runtime/batch.py:73` | <code>Native batch requires real float32 three-dimensional Yee fields.</code> |
| 123 | `witwin/maxwell/fdtd/runtime/batch.py:77` | <code>Native batch supports none, CPML, periodic, PEC and PMC boundaries.</code> |
| 124 | `witwin/maxwell/fdtd/runtime/breakdown.py:220` | <code>Dielectric breakdown requires the real-valued FDTD update; a Bloch-periodic run carries complex phase-shifted fields, for which the real conductivity scatter is undefined. Use a real-field boundary.</code> |
| 125 | `witwin/maxwell/fdtd/runtime/breakdown.py:232` | <code>Dielectric breakdown does not yet compose with  in the same scene: they share the per-edge E-update coefficient representation that the breakdown conductivity scatter rewrites.</code> |
| 126 | `witwin/maxwell/fdtd/runtime/breakdown.py:238` | <code>Dielectric breakdown does not compose with gyromagnetic ferrite media in the same scene: the ferrite forward advances a coupled magnetization ADE whose energy accounting the breakdown dissipation channel does not model.</code> |
| 127 | `witwin/maxwell/fdtd/runtime/gyromagnetic.py:122` | <code>FDTD gyromagnetic ferrite requires the real-valued magnetic update: a Bloch-periodic run carries complex phase-shifted fields, and the real magnetization-ADE correction would break the Bloch phase relation. Use a real-field boundary (PML/PEC/PMC/periodic) with the ferrite.</code> |
| 128 | `witwin/maxwell/fdtd/runtime/materials.py:28` | <code>FDTD nonlinear media require the real-valued update kernels: a Bloch-periodic run carries complex phase-shifted fields, for which the instantaneous &#124;E&#124;^2 Kerr/chi2/TPA coefficient is not defined. Use a real-field boundary (PML/PEC/PMC/periodic) with the nonlinearity.</code> |
| 129 | `witwin/maxwell/fdtd/runtime/materials.py:34` | <code>FDTD full (off-diagonal) anisotropic media use a real additive curl(H) correction: a Bloch-periodic run carries complex phase-shifted fields, and applying the real off-diagonal correction to a complex field would break the Bloch phase relation. Use a real-field boundary (PML/PEC/PMC/periodic) or an axis-aligned DiagonalTensor3.</code> |
| 130 | `witwin/maxwell/fdtd/runtime/materials.py:41` | <code>FDTD time-modulated media scale the real-valued update coefficients by a per-step modulation factor: a Bloch-periodic run carries complex phase-shifted fields, and the real scaling does not track the complex Bloch phase. Use a real-field boundary (PML/PEC/PMC/periodic) with the modulation.</code> |
| 131 | `witwin/maxwell/fdtd/runtime/materials.py:170` | <code>FDTD time-modulated media cannot be combined with full (off-diagonal) anisotropic media: modulating the crystal-frame permittivity tensor needs a per-step re-inversion of the coupled 3x3 tensor, which the diagonalized effective-permittivity update path does not provide. Use a diagonal (axis-aligned) tensor or drop the modulation.</code> |
| 132 | `witwin/maxwell/fdtd/runtime/materials.py:338` | <code>FDTD full (off-diagonal) anisotropic media cannot be combined with nonlinear media.</code> |
| 133 | `witwin/maxwell/fdtd/runtime/materials.py:1571` | <code>FDTD full (off-diagonal) anisotropic media overlapping a split-field PML/absorber layer are not supported: the graded-sigma absorber carries no per-direction auxiliary memory to coordinate-stretch the off-diagonal tensor coupling. Use the default CPML absorber for anisotropic structures that reach the boundary region.</code> |
| 134 | `witwin/maxwell/fdtd/runtime/stepping.py:1060` | <code>FDTD instantaneous nonlinear media cannot evolve on the complex Bloch fields: the &#124;E&#124;^2 Kerr/chi2/TPA polarization couples the single-wavevector Bloch field to the 2k/3k harmonic sectors a complex-Bloch run does not carry, so the nonlinear response is not closed on the real update kernels. Use a real-field boundary (PML/PEC/PMC/periodic).</code> |
| 135 | `witwin/maxwell/fdtd/runtime/stepping.py:1118` | <code>FDTD instantaneous nonlinear media cannot evolve on the complex Bloch fields: the &#124;E&#124;^2 Kerr/chi2/TPA polarization couples the single-wavevector Bloch field to the 2k/3k harmonic sectors a complex-Bloch run does not carry, so the nonlinear response is not closed on the real update kernels. Use a real-field boundary (PML/PEC/PMC/periodic).</code> |
| 136 | `witwin/maxwell/fdtd/runtime/stepping.py:1201` | <code>Mixed Bloch/CPML FDTD updates require exactly one PML axis (both faces PML) and two Bloch axes (both faces Bloch). The complex split-field Bloch update carries the transverse wrap phase on the two periodic axes while the single absorbing axis runs the recursive-convolution CPML stretch, so a face layout that mixes Bloch and PML on one axis or leaves more than one absorbing axis is not expressible on that split.</code> |
| 137 | `witwin/maxwell/fdtd/runtime/stepping.py:2144` | <code>normalize_source requires all sources to share one source-time spectrum.</code> |
| 138 | `witwin/maxwell/fdtd/runtime/stepping.py:2152` | <code>FDTD resume currently supports circuit/port workflows without full-field DFT or field/time/breakdown observers.</code> |
| 139 | `witwin/maxwell/fdtd/wire.py:173` | <code>Field-coupled reverse/adjoint replay of a finite-conductor thin wire is not implemented: transposing the lossy current recurrence would require the derivative of the passive series-impedance ADE coefficients with respect to conductivity, but those coefficients come from the shared rational vector fit, which is nondeterministic (B1) and not a differentiable map of sigma -- so an exact reverse replay of dI/dsigma through the recurrence cannot be certified. The deterministic conductivity adjoint of the dissipation channel (d Re(Z&#x27;)/d sigma via the analytic internal impedance) is available through witwin.maxwell.fdtd.wire_lossy.analytic_ac_resistance; run a PEC wire for field-coupled differentiable workflows.</code> |
| 140 | `witwin/maxwell/fdtd/wire.py:778` | <code>Thin-wire FDTD does not yet support Bloch-periodic phase topology.</code> |
| 141 | `witwin/maxwell/fdtd/wire.py:782` | <code>Thin-wire FDTD does not yet support full off-diagonal anisotropic electric updates.</code> |
| 142 | `witwin/maxwell/fdtd/wire.py:786` | <code>Thin-wire FDTD does not yet support a conductive background electric update.</code> |
| 143 | `witwin/maxwell/fdtd/wire.py:790` | <code>Thin-wire FDTD does not yet support dispersive background media.</code> |
| 144 | `witwin/maxwell/fdtd/wire.py:792` | <code>Thin-wire FDTD does not yet support nonlinear or time-modulated background media.</code> |
| 145 | `witwin/maxwell/fdtd/wire.py:799` | <code>Thin-wire FDTD cannot share conductor ownership with a surface-impedance boundary until wire loss and SIBC power accounting are coupled.</code> |
| 146 | `witwin/maxwell/media.py:577` | <code>DielectricBreakdown v1 supports model=&#x27;field_duration&#x27; only; got . Other criteria (instantaneous, damage) are a later phase.</code> |
| 147 | `witwin/maxwell/media.py:583` | <code>DielectricBreakdown v1 supports state=&#x27;latching&#x27; only; got . Recovering/failed state machines are a later phase.</code> |
| 148 | `witwin/maxwell/media.py:588` | <code>DielectricBreakdown.recovery is a reserved field with no v1 behavior; leave it unset until the recovery model lands.</code> |
| 149 | `witwin/maxwell/media.py:593` | <code>DielectricBreakdown.damage_parameters is a reserved field with no v1 behavior; leave it unset until the cumulative-damage model lands.</code> |
| 150 | `witwin/maxwell/media.py:699` | <code>Material.orientation (a separate crystal-frame rotation matrix) is not consumed: the tensor material path assembles per-edge Yee coefficients directly from the lab-frame permittivity/permeability/conductivity tensor, so a standalone rotation would need the tensor rotated into the lab frame before compilation. Pass the already-rotated tensor as Tensor3x3 rows (or an axis-aligned DiagonalTensor3) instead of an orientation matrix.</code> |
| 151 | `witwin/maxwell/media.py:712` | <code>Material. accepts a DiagonalTensor3 (axis-aligned) tensor only. A full off-diagonal Tensor3x3 would require a coupled per-edge 3x3 inverse on the  update, which only the electric permittivity update forms; supply an axis-aligned DiagonalTensor3, or move the off-diagonal coupling to epsilon_tensor.</code> |
| 152 | `witwin/maxwell/media.py:735` | <code>A nonlinear Material cannot carry an anisotropic permittivity tensor: the instantaneous Kerr/chi2/TPA channel updates the permittivity as a per-component field-dependent scalar, whereas an anisotropic tensor applies a coupled per-edge tensor inverse. Composing them would need a per-step re-inversion of a field-amplitude-dependent 3x3 tensor, which the scalar nonlinear coefficient update does not form. Use an isotropic (scalar) permittivity with the nonlinearity, or drop the tensor.</code> |
| 153 | `witwin/maxwell/media.py:752` | <code>A time-modulated Material cannot carry anisotropic tensors: modulating the crystal-frame permittivity tensor needs a per-step re-inversion of the coupled 3x3 tensor, which the diagonalized effective-permittivity update does not provide.</code> |
| 154 | `witwin/maxwell/media.py:758` | <code>A time-modulated Material cannot carry static electric conductivity: the semi-implicit loss fold 0.5*sigma*dt/eps must see the modulated eps_inf*m(t), which the static decay coefficient does not track. Model loss with a Drude/Debye pole (dispersive path) instead.</code> |
| 155 | `witwin/maxwell/media.py:789` | <code>A PEC Material cannot carry dielectric breakdown: a perfect conductor has no finite field-duration threshold.</code> |
| 156 | `witwin/maxwell/media.py:794` | <code>Dielectric breakdown requires an isotropic (scalar) Material: the anisotropic tensor update forms a per-edge 3x3 inverse the breakdown conductivity scatter does not model. Use scalar eps_r/sigma_e.</code> |
| 157 | `witwin/maxwell/media.py:800` | <code>Dielectric breakdown does not compose with dispersive poles yet: the ADE polarization current shares the same E-update coefficients the breakdown scatter rewrites. Use a non-dispersive scalar Material.</code> |
| 158 | `witwin/maxwell/media.py:806` | <code>Dielectric breakdown does not compose with the instantaneous nonlinear channels yet: both rewrite the per-step E-update coefficients. Use a linear scalar Material.</code> |
| 159 | `witwin/maxwell/media.py:812` | <code>Dielectric breakdown does not compose with time modulation yet: both scale the E-update coefficients per step. Drop the modulation.</code> |
| 160 | `witwin/maxwell/media.py:817` | <code>Dielectric breakdown requires a bulk (3D) Material, not a 2D sheet medium.</code> |
| 161 | `witwin/maxwell/media.py:977` | <code>Frequency evaluation of a dispersive Material is only defined for an axis-aligned DiagonalTensor3 permittivity with homogeneous poles: a fully anisotropic Tensor3x3 permittivity needs a coupled tensor ADE, and spatially-varying custom poles have no homogeneous sample. Evaluate the compiled scene material model (Scene.compile_relative_materials) instead.</code> |
| 162 | `witwin/maxwell/media.py:2326` | <code>PerturbationMedium cannot wrap a GyromagneticFerrite: it perturbs a scalar permittivity background, but a ferrite carries a non-reciprocal gyromagnetic permeability in a local magnetization state that this scalar-eps perturbation would silently discard. Perturb the ferrite parameters directly instead.</code> |
| 163 | `witwin/maxwell/media.py:2367` | <code>PerturbationMedium applies its scalar sensitivity field as an isotropic delta_eps to a scalar or axis-aligned (DiagonalTensor3) base permittivity; a full off-diagonal Tensor3x3 base has no single principal-axis eps to perturb, so the perturbation direction in the coupled tensor is ambiguous. Use a scalar or DiagonalTensor3 base permittivity.</code> |
| 164 | `witwin/maxwell/network_sweep.py:42` | <code>Port  has a trainable reference impedance, but PortSweep does not support RF-parameter gradients.</code> |
| 165 | `witwin/maxwell/network_sweep.py:75` | <code>PortSweep amplitude is trainable, but PortSweep does not support RF-parameter gradients.</code> |
| 166 | `witwin/maxwell/network_sweep.py:107` | <code>Port  has a trainable termination, but PortSweep does not support RF-parameter gradients.</code> |
| 167 | `witwin/maxwell/postprocess/antenna.py:598` | <code>Antenna gain currently requires a homogeneous lossless exterior.</code> |
| 168 | `witwin/maxwell/postprocess/array.py:276` | <code>Array basis extraction is FDTD-only.</code> |
| 169 | `witwin/maxwell/postprocess/stratton_chu.py:819` | <code>Closed-surface Huygens currently requires a homogeneous exterior medium immediately outside every face.</code> |
| 170 | `witwin/maxwell/postprocess/stratton_chu.py:833` | <code>Closed-surface Huygens currently requires a single homogeneous exterior medium across all faces; layered or piecewise exteriors are not supported.</code> |
| 171 | `witwin/maxwell/result.py:1276` | <code>Physical E/H queries currently support SIE and FDTD Results</code> |
| 172 | `witwin/maxwell/result.py:1280` | <code>FDTD Yee interpolation does not define one-sided interface traces</code> |
| 173 | `witwin/maxwell/result.py:1288` | <code>FDTD physical queries currently require one recorded frequency</code> |
| 174 | `witwin/maxwell/result.py:1317` | <code>SIE Results contain RWG currents; use query_fields for E/H instead of Yee field access</code> |
| 175 | `witwin/maxwell/result.py:1943` | <code>Automatic PowerLossMonitor field colocation currently supports FDTD results only.</code> |
| 176 | `witwin/maxwell/result.py:2067` | <code>Result.sar currently consumes FDTD PowerLossData only.</code> |
| 177 | `witwin/maxwell/result.py:2409` | <code>FDTD query stencils must lie strictly inside the TFSF total-field box; exterior reconstruction is not implemented</code> |
| 178 | `witwin/maxwell/sar.py:73` | <code>SAR averaging connectivity  is not implemented in ; only  (no tissue flood-fill) is supported.</code> |
| 179 | `witwin/maxwell/sar.py:78` | <code>SAR averaging boundary_policy  is not implemented; only  is supported.</code> |
| 180 | `witwin/maxwell/sar.py:165` | <code>PowerNormalization.input_power scaling requires a total injected source-power diagnostic, which this build does not expose; use accepted_power(port=...) or source(amplitude=...) instead.</code> |
| 181 | `witwin/maxwell/sie/runtime.py:172` | <code>SIE geometry/material derivatives are not supported; fixed-q residual_loss supports coefficient gradients</code> |
| 182 | `witwin/maxwell/simulation.py:495` | <code>Trainable RF port, source, and R/L/C parameters require Simulation.fdtd(...).</code> |
| 183 | `witwin/maxwell/simulation.py:730` | <code>Electrostatic pre-bias initial conditions seed the time-domain FDTD field buffers; the frequency-domain solver has no time-stepped state.</code> |
| 184 | `witwin/maxwell/simulation.py:735` | <code>Distributed/multi-GPU FDTD does not support an electrostatic pre-bias initial condition yet: seeding shard-local staggered E buffers with the correct halo/ownership is a later phase. Run the pre-bias on a single GPU.</code> |
| 185 | `witwin/maxwell/simulation.py:741` | <code>Trainable/adjoint FDTD does not support an electrostatic pre-bias initial condition yet: the seeded DC field would enter the taped forward state without a differentiated map back to the electrostatic solve. Detach the scene parameters to run a forward pre-bias.</code> |
| 186 | `witwin/maxwell/simulation.py:748` | <code>Bloch-periodic FDTD carries complex phase-shifted fields; the real-valued electrostatic pre-bias seed does not track the Bloch phase. Use a real-field boundary (PEC/PMC/periodic) for a pre-charged run.</code> |
| 187 | `witwin/maxwell/simulation.py:767` | <code>DielectricBreakdown is a time-domain (FDTD) feedback model; the frequency-domain solver has no dynamic conductivity update. Run the breakdown scene with the FDTD method.</code> |
| 188 | `witwin/maxwell/simulation.py:773` | <code>Trainable scenes cannot enable hard dielectric breakdown: the field-duration/latching switch is non-differentiable at the trigger time, so prepare() rejects backward rather than fabricating a gradient. For optimization use the separate non-feedback SmoothBreakdownRisk surrogate (mw.SmoothBreakdownRisk), which never drives the field solve.</code> |
| 189 | `witwin/maxwell/simulation.py:838` | <code>Circuit-bound ports do not support PortSweep execution.</code> |
| 190 | `witwin/maxwell/simulation.py:843` | <code>PortSweep amplitude is trainable, but WavePort sweeps do not support RF-parameter gradients.</code> |
| 191 | `witwin/maxwell/simulation.py:859` | <code>A direct multi-source FDTD run supports LumpedPort and TerminalPort only; WavePort channels require calibrated per-frequency execution.</code> |
| 192 | `witwin/maxwell/simulation.py:890` | <code>WavePort PortExcitation does not yet support trainable amplitude.</code> |
| 193 | `witwin/maxwell/simulation.py:919` | <code>Differentiable embedded RationalModel supports residues and direct terms only; trainable  are not supported.</code> |
| 194 | `witwin/maxwell/simulation.py:926` | <code>Differentiable embedded networks accept trainable residues/direct on a pre-fitted RationalModel; direct trainable state-space matrices are not supported.</code> |
| 195 | `witwin/maxwell/simulation.py:944` | <code>Differentiable WavePort runs cannot include lumped/terminal ports or standalone R/L/C elements because their auxiliary state is not replayed by the FDTD adjoint.</code> |
| 196 | `witwin/maxwell/simulation.py:951` | <code>Differentiable WavePort runs currently require exactly one fixed mode per aperture; multimode ports are .</code> |
| 197 | `witwin/maxwell/simulation.py:957` | <code>Differentiable WavePort declarations require a fixed single-mode direct excitation or WavePort sweep and cannot be mixed into a lumped workflow.</code> |
| 198 | `witwin/maxwell/simulation.py:962` | <code>Differentiable LumpedPort/TerminalPort PortSweep is not yet supported; use direct port excitations.</code> |
| 199 | `witwin/maxwell/simulation.py:1004` | <code>Circuit-coupled FDTD requires one circuit with at least one bound port; multi-circuit execution is not yet supported.</code> |
| 200 | `witwin/maxwell/simulation.py:1030` | <code>FDFD does not support nonuniform (GridSpec.custom / GridSpec.auto) grids yet; use Simulation.fdtd(...) or a uniform GridSpec.</code> |
| 201 | `witwin/maxwell/simulation.py:1894` | <code>FDTD resume currently requires a single-GPU detached forward run; distributed circuit-owner state and adjoint replay are separate contracts.</code> |
| 202 | `witwin/maxwell/waveport_sweep.py:220` | <code>WavePort   must be real for TE/TM impedance.</code> |
| 203 | `witwin/maxwell/waveport_sweep.py:362` | <code>Differentiable WavePort runs require trainable inputs that contribute to the prepared material tensors.</code> |
| 204 | `witwin/maxwell/waveport_sweep.py:387` | <code>Differentiable WavePort runs require fixed port cross-sections and launch planes; a trainable material or geometry input affects a selected WavePort aperture or its adjacent source plane. Move the design region away from the port launch.</code> |
| 205 | `witwin/maxwell/waveport_sweep.py:415` | <code>WavePort PortSweep does not yet support trainable amplitude.</code> |
| 206 | `witwin/maxwell/waveport_sweep.py:437` | <code>A WavePort run can retain lumped/terminal ports only as explicit passive terminations; unterminated entries are .</code> |

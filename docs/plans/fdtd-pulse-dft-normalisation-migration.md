# FDTD pulse spectral normalization migration

## R1 circuit identity fixture migration after rebase onto e6d2189a

The mixed `circuit` identity fixture retains its original undamped circuit sine,
transient port pulse, circuit topology and 301-step execution. It still compares
all time-monitor and per-step port histories, circuit and lumped state, raw
accumulator state, coordinate metadata and final Yee fields with `torch.equal`.
Port impedance, reflection, return loss and VSWR are now recorded explicitly;
the fixture retains network ratios if present. No normalized circuit spectrum
was previously recorded: circuit source normalization remains unsupported.
Only the eight absolute tensor keys below leave the mixed fixture.

Both new fixtures retain `R1`, `I1`, the shunt resistor, both LumpedPorts and
the `p2` circuit binding. `circuit_ongoing` replaces the `p1` Gaussian pulse
with CW and retains `I1` as a zero-current source without a waveform: the
field-driven circuit still solves and replays through `p2`. This uses the pure-CW
scale already supported by the frozen package; an active circuit sine has the
older integral scale there. `circuit_transient` retains the
`p1` pulse and changes only the sine damping to `1e9`, with zero offset.
They execute the same circuit-coupled step loop. The gate requires field and
circuit CUDA graph replay for all three fixtures; the separate port-step graph
remains disabled for circuit-bound ports, as in the original mixed fixture.

Moved-key list: each original `circuit/<suffix>` below moves to **both**
`circuit_ongoing/<suffix>` and `circuit_transient/<suffix>`. The gate asserts
that all eight keys are present in both destinations before comparing every
payload tensor with `torch.equal`.

| Original key suffix | Ongoing destination suffix | Transient destination suffix |
| --- | --- | --- |
| `monitors/probe/Ez` | `monitors/probe/Ez` | `monitors/probe/Ez` |
| `monitors/probe/components/Ez` | `monitors/probe/components/Ez` | `monitors/probe/components/Ez` |
| `monitors/probe/data` | `monitors/probe/data` | `monitors/probe/data` |
| `ports/p1/voltage` | `ports/p1/voltage` | `ports/p1/voltage` |
| `ports/p1/current` | `ports/p1/current` | `ports/p1/current` |
| `ports/p1/available_power` | `ports/p1/available_power` | `ports/p1/available_power` |
| `ports/p2/voltage` | `ports/p2/voltage` | `ports/p2/voltage` |
| `ports/p2/current` | `ports/p2/current` | `ports/p2/current` |

Pin decision: retain both `BASE_COMMIT` and `COPPER_BASE_COMMIT` at `95a3ec3f`.
The gate launches the working-tree scene script with the archived package on
`PYTHONPATH`; that package executes both new fixtures and all identity comparisons
pass. No new baseline needs staging for the cloud. The original mixed payload
was captured separately before migration: exactly the eight listed keys were
removed, and every retained original tensor remains bit-identical. Nonzero
circuit node-voltage entries number 301, 294 and 301 in the mixed, ongoing and
transient fixtures respectively, confirming that the ongoing fixture still
drives the coupled circuit despite its inactive independent current source.

The fail-closed negative assertion stays in
`test_pulse_dft_normalisation.py::test_mixed_sources_reject_absolute_spectra_but_keep_ratios`,
outside the identity gate. All branch-added mixed absolute-read boundaries
delegate to `require_absolute_spectra`, which raises `ValueError`, already
listed under `output_guards` in the capability manifest. No read-boundary
`NotImplementedError` was added, so no new capability-census entry is needed.

Validation used the prescribed GPU-1 lane-lock wrapper throughout:

| Scope | Final result |
| --- | --- |
| `tests/rf/lumped`, including the separately run identity gate | 337 passed; gate alone: 1 passed in 94.75 s |
| `tests/rf/circuits` | 127 passed |
| `tests/rf/wave_validation` | 598 passed |
| `tests/fdtd/test_pulse_dft_normalisation.py` | 4 passed |
| `tests/postprocess/scattering` | 53 passed |
| Two multi-frequency observer tests | 2 passed |
| `tests/fdtd/multi_gpu/test_parallel_public_api.py` | 10 passed |
| Mixed sharded-restoration negative test | 1 passed |
| `tests/api/public/test_guard_census.py` | 7 passed |

The combined suite initially reported 1,134 passed and four explicit-GPU-opt-in
skips in 1,259.92 s. All four skips then passed with `--gpu` in 171.10 s.
Including the separate gate, **1,139 unique tests passed, zero failed, and no
skips remain**. The four branch-owned mixed-read negative cases are included:
the executed mixed-source case, both nested-statistics port-view cases, and
sharded restoration. The complete NotImplementedError guard-identity sets also
match archived master, with no additions or removals.

An initial development-only gate attempt kept an active undamped circuit sine
in the ongoing fixture and failed on nine spectral/reference tensors. A fresh
`git archive e6d2189a` run of that exact fixture matched all 120 frozen-baseline
tensors and reproduced those nine differences against the initial branch run.
This classifies the mismatch as introduced by the branch's decided ongoing
scaling change, resolved by the CW-driven coupled-circuit fixture above.
There are no outstanding pre-existing or introduced test failures. Receipts:
`<scratch>/witwin_cov_dftbug/r1-gate-final.xml`, `r1-suites.xml`, `r1-optin.xml`, and
`r1-initial-ongoing-control.json` in the same directory.

## AMENDMENT — 2026-10-31, USER-DECIDED mixed-source absolute spectra

This decision supersedes the earlier pending mixed-source decision below.
Runs combining at least one ongoing source with at least one transient source,
with `normalize_source=False`, fail closed when an absolute spectral output is
read or exported. No horizon-independent absolute scale exists for that mix.
The time plan records `mixed_sources`, using the ongoing-source classification
from `f3090b97`: field/port CW, undamped or offset circuit sines, periodic or
nonzero-initial circuit pulses, nonzero DC, and nonzero-tail circuit PWL sources.
Preparation, execution, time histories and material reads remain available.
The common internal Fourier samples remain available for ratio reductions.

Every rejected read raises `ValueError` with this reason and migration advice:

> Absolute spectra of a run mixing ongoing (CW / undamped / periodic / DC) and transient sources have no horizon-independent scale; use normalize_source=True, ratios such as S/Z, or a run with only ongoing or only transient sources.

The single raising implementation is
`fdtd.runtime.representation.require_absolute_spectra`. Its read boundaries are:

| Boundary | Why it is needed |
| --- | --- |
| `Result._require_frequency_volume_fields` and `query_fields` | Cover E/H, field dictionaries, tensor/chart selections and physical field queries, including queries that consume stored fields directly. |
| `Result._require_monitor_spectra`, called by `monitor`, `_native_raw_monitor` and `monitors` | Cover point/plane/flux, wire, modal, emission, projection and closed-surface spectra. Time-domain and material payloads are exempt. |
| `Result.power_loss`, `conductor_losses`, `conductor_loss_columns` | Loss reductions may consume private field/side-head samples; foil reports bundle absolute spectral work with their time-domain diagnostics. |
| `Result.raw_output`, `embedded_network`, `embedded_networks`, `_snapshot_payload` | Prevent direct raw payload, embedded V/I/power and snapshot reads from bypassing spectral accessors. |
| `_RatioOnlyPortData.__getattr__` returned by `Result.port` and `Result.ports` | Reject V/I, available power, absolute waves, incident/reflected/accepted/delivered power and standalone port exports. Iteration over ports also yields views. Plain internal PortData has no attribute hook. |
| `_RatioOnlyScatteringData.__getitem__` | Flux S reductions return S entries without publishing absolute powers; indexing any `P_incident`, `P_reflected`, or `P_transmitted` entry raises the same error. |

The ratio-only view delegates impedance, reflection, return loss and VSWR to
the stored PortData and its private V/I samples, preserving the exact operation
order and live PyTorch graph. Frequencies, reference impedances, names, modal
descriptors and metadata remain available; terminal response V/I entries are
omitted from public view metadata. Non-mixed runs return the original PortData
object, with no attribute hook or per-attribute overhead. S/Z assembly uses private
column ports and private monitor containers. Modal extraction and flux S
postprocessing use an internal shallow Result view that admits intermediate
spectra for ratios; that view is never returned. Flux S results omit absolute
`P_*` values, including from iteration/serialization, and preserve S ratios for
dimensionless power closure calculations. Network S/Z conversions and passivity
checks remain available. No new source-normalization preparation restriction is
introduced: `normalize_source=True` retains its existing common-waveform and
circuit-drive validation, and the new absolute-read guard does not apply to it.

`Result.save` and `save_sharded` use the internal snapshot path to retain detached
common-scale samples for later ratio reductions. Their payload records an explicit
`mixed_absolute_spectra` flag. Both load paths restore the restriction independently
of diagnostic statistics: loaded ports remain ratio-only views, and absolute
field/monitor/raw/embedded reads remain guarded. Checkpoint samples are internal
storage, not admitted absolute physical spectra. Direct `_snapshot_payload` reads
on mixed Results still raise. Sweep aggregation reads real private PortData;
nested column statistics propagate the restriction to sweep and batch Results.

Pre-landing census inspected source constructors, implicit default pulses,
scene helpers and their readers under `tests/`, `benchmark/` (scenes and
runners), `scripts/dev/`, and `benchmark/datagen/`. The local `master` tree was
also inspected at `1f270da9`; it has no tracked `label_qualification` harness.
Existing census consumers were not edited.

| Path / scope | What it reads | Would now raise? | Suggested migration |
| --- | --- | --- | --- |
| `tests/support/port_step_graph_bit_identity_scene.py`, `circuit` fixture, invoked by `tests/rf/lumped/test_port_step_graph_bit_identity.py::test_port_step_graph_is_bitwise_identical_to_the_frozen_source` | Undamped circuit sine plus transient port pulse, default `normalize_source=False`; exports point spectra via `result.monitors`, port V/I/available power, and private stored field arrays. | **Yes**, first at the public monitor dictionary; port absolute reads also reject. | Keep the mixed fixture for time-history and S/Z/reflection assertions plus an explicit negative spectral read. Put absolute identity comparisons in separate all-ongoing and transient-only fixtures and re-freeze their references. |
| `tests/fdtd/test_time_planning.py`, `test_time_consumer_consistency.py`, `test_declared_drive_frequency.py`, `test_complex_source_representation.py` mixed cases | Plans, accumulator/window preparation, source clock or time histories; no absolute spectrum is read from a completed mixed run. | No new rejection. Complex analytic-CW/real-pulse mixtures retain their earlier preparation restriction. | None. |
| `benchmark/` scenes and runners | Absolute field/flux and port/antenna readouts, but CW and transient cases execute separately; no mixed-source consumer found. | No. | None. |
| `scripts/dev/` | CW/pulse controls and absolute spectrum audit scripts use separate runs; no mixed-source consumer found. | No. | None. |
| `benchmark/datagen/` recipes, profiler, scheduler and repro scripts on this branch and `master` | Field, flux, port and antenna labels; the periodic recipe chooses CW for its empty case and a pulse for its loaded case, rather than combining them. | No. | None. |
| `master:benchmark/label_qualification/` | No tracked harness at the inspected master tip. | Not applicable. | None. |

Validation used the prescribed GPU-1 lock wrapper and the `maxwell` interpreter.
No other physical device was used. The executed mixed-source negative test
covers the view, exact internal ratios and save/load. Additional CPU regressions
cover nested column statistics, terminal response metadata, independent persisted
flags and sharded restoration. Pre-existing census tests and fixtures remain unchanged.

| Validation | Result |
| --- | --- |
| `tests/rf/circuits` plus `tests/fdtd/test_pulse_dft_normalisation.py` | **131 passed**, 132.32 s. Includes the mixed-source negative test, ordinary/normalized spectra, ports and gradients. |
| Final mixed-source negative test; time planning; port/network contracts and persistence; modal PortData; waveport sweep; scattering parameters; modal FDTD sweep | **54 passed**, 50.65 s. Confirms ratio extraction and persistence of admitted outputs after the read-boundary changes. |
| Census hit: `test_port_step_graph_is_bitwise_identical_to_the_frozen_source` | **1 failed**, 53.41 s: the branch subprocess reaches the completed mixed `circuit` fixture, then `Result.monitors` raises the exact decided `ValueError`. The frozen reference completes. This is a known migration requirement, not an unexplained numerical regression. |
| Modified Python syntax, guard-census JSON, `git diff --check` | Passed. |

Coordinator redesign validation (2026-10-31): code and CPU checks completed
before GPU work; every GPU command used the prescribed lock wrapper with
`CUDA_VISIBLE_DEVICES=1` after 21:00 local time.

| Ratio-only view validation | Result |
| --- | --- |
| Plain pulse and CW ports against archived master `95219858` | **18/18 `torch.equal`** checks: V/I, available/incident/accepted power, impedance, reflection, sweep S and sweep impedance. Direct and sweep ports are plain PortData. |
| Original 131-case circuit/pulse suite | **131 passed**, 116.60 s; updated negative test checks exact internal ratios, all dimensional port reads and mixed sweep save/load. |
| Original 54-case suite plus two nested-statistics/view regressions | **56 passed**, 53.26 s. |
| CPU contracts, port/network persistence, modal PortData and guard census | **41 passed**, 29.92 s. |
| CPU port views and sharded persistence | **16 passed, 1 skipped**, 2.37 s; the skip requires CUDA. |
| Full `tests/rf/circuits`, `tests/rf/lumped`, `tests/rf/wave_validation` | **1,050 passed, 8 failed, 4 skipped**, 794.30 s. Every failure reproduces on unchanged `637a7dc3` (8/8, 72.61 s). |

The eight unchanged RF failures are:

- The frozen port-step identity fixture reads absolute monitor spectra from its
  mixed circuit/pulse run and raises the decided shared `ValueError`.
- Both `test_loss_dual_vi_execution.py` cases and all four
  `test_printed_line_loss_execution.py` cases fail during preparation because
  `PowerLossMonitor('line_roi_bulk')` requests automatic electric channels with
  multiple frequencies and `full_field_dft=False`.
- `test_lossy_tem_modes.py::test_complex_port_extraction_preserves_physical_vi_and_kurokawa_power`
  supplies a test double without `_spectral_ratio_view`, which the existing
  internal wave extraction helper requires. This also fails before the redesign.

No numeric thresholds, baseline pins or existing failing fixtures were changed.
The four named internal sweep/array modules already contain no
`object.__getattribute__` calls at `637a7dc3`; internal network assembly continues
to consume real private PortData. `network.py` now matches master `95219858`.
Receipts and comparison samples are under `<scratch>/witwin_cov_dftbug/`:
`portview-131.xml`, `portview-56.xml`, `portview-cpu.xml`,
`portview-cpu-final.xml`, `portview-rf.xml`, `portview-before-failures.xml`,
`portview-master.pt` and `portview-branch.pt`. The original-tip control asserts
that its imported package is the archived `637a7dc3` tree; only the pinned
baseline helper's Git lookup points back to the real repository.

## AMENDMENT — 2026-10-31, batch 2 readers and ongoing sources

Expected changes:

- On pulsed runs, `compute_s_parameters(incident_power="auto")` reconstructs
  `P_incident` with the observers' shared `spectral_scale`, including the
  per-bin frequency and explicit reference-frequency override. Auto-normalized
  S-parameters now use incident power consistent with measured flux. Relative
  to the old reader, power changes by `(dt*f / (2/sum_w))^2` under the default
  per-frequency reference; the analytic CW incident power is unchanged.
- Absolute spectra of all-ongoing circuit-driven runs return to averaged
  steady-state phasors. Every source must be CW or explicitly `ongoing=True`:
  undamped circuit sines, periodic pulse trains, nonzero DC drives, and PWL
  sources ending at a nonzero value qualify. Transient-only runs retain the
  Fourier integral. Mixed ongoing/transient runs retain their current Fourier
  integral pending a separate user decision. Measurement starts, windows,
  analytic CW handling, and complex-field checks retain their existing rules.

The distributed public-API test stand-in binds
`_spectral_reference_frequency=None`, matching the per-frequency solver default.
No public validation raises are added.

GPU factor checks against parent `1f270da9` (the two failing tests are unedited):

| Frequency (GHz) | Old auto incident power | New auto incident power | Measured flux, bitwise unchanged | New/old = `(dt*f / (2/sum_w))^2` |
| --- | --- | --- | --- | --- |
| 0.9 | 1.117735706806e-7 | 3.827627409385e-5 | 4.009571996285e-5 | 342.444764542936 |
| 1.0 | 1.249090960374e-7 | 5.280798269360e-5 | 5.508134255651e-5 | 422.771314250539 |
| 1.1 | 1.117735702036e-7 | 5.717813759988e-5 | 5.935081567150e-5 | 511.553290243152 |

The power-factor identity agrees to floating-point rounding (relative error
below `6e-16`); new auto power is 4.54%, 4.13%, 3.66% below measured flux.
A CW plane-wave run returns `[3.4791997151550786, 3.4791997151550786]`,
`torch.equal` to the master reader. Gaussian-pulse-port and pure-CW-port
V/I, waves, powers, full-field spectra and point spectra are `torch.equal`
to their saved parent runs.

The orientation test's voltage changes by `4.807923894211912`:
`0.022136266327 + 0.115463845713j` becomes
`0.106429483802 + 0.555141382722j`. For the same circuit-sine scene at
`dt=1/(64*3 GHz)`, the settled 4096/6144-step horizon comparison is:

| Output | Parent, 4096 / 6144 steps | Corrected, 4096 / 6144 steps | Corrected relative difference |
| --- | --- | --- | --- |
| V | `-0.265998975+15.890478798j` / `-0.399716485+23.836479567j` | `-0.008312468+0.496577462j` / `-0.008327427+0.496593324j` | 4.39001e-5 |
| I | `0.036694623+0.000649489j` / `0.055054442+0.000958469j` | `0.001146707+0.000020297j` / `0.001146968+0.000019968j` | 3.65539e-4 |

Parent long/short ratios are `1.500048656+0.000044371j` (V) and
`1.500333143-0.000435497j` (I), versus approximately one after correction.
The source-record audit checks both circuit current and voltage sources:
undamped/offset sines, periodic/nonzero-initial pulses, nonzero DC and PWL
nonzero tails all carry `ongoing=True`. Field/port CW uses `kind="cw"`;
sampled field tables are zero outside support, and callable support remains
unknown. A 16-case planning matrix also checks finite/damped sources,
zero DC, field/port pulses, all-ongoing combinations and mixed drives.
The second grep of `witwin/` found no further old pulse-scale implementations;
existing full-field, observer, port, wire, copper and adjoint readers already
call `spectral_scale`. No benchmark builder uses ongoing circuit sources.

Ongoing-source test census (paths relative to `tests/`; counts include all
parameters in the selected tests):

| Coverage | Parent → corrected status |
| --- | --- |
| `rf/circuits/` complete suite | 126 pass, 1 fail → 127 pass; only the orientation test moves. |
| Within that suite: driven sine in `test_fdtd_circuit_coupling.py`, `test_circuit_conservation.py`, `test_phase4_circuit_cuda_graph.py` | 10, 6, 6 pass → same. DC/periodic/sine standalone MNA, graph, contract and parser coverage also stays passing. |
| `gradients/test_fdtd_circuit_adjoint.py` | 25 pass → 25 pass. |
| `gradients/test_fdtd_conductive_port_adjoint.py` | 19 pass → 19 pass. |
| `fdtd/test_declared_drive_frequency.py` (sine, periodic pulses, DC/PWL scales) | 16 pass → 16 pass. |
| `rf/lumped/test_port_step_graph_bit_identity.py` (mixed sine circuit + transient port) | 1 pass → 1 pass, frozen-source tensors bitwise unchanged. |
| `materials/surface_impedance/test_copper_foil_material_batch.py::test_material_mna_pulse_switch_factor_classes_match_independent` | 2 pass → 2 pass, periodic circuit pulse. |
| `fdtd/multi_gpu/test_circuit_owner.py` | 6 pass + 1 two-GPU skip → same. |
| `fdtd/multi_gpu/test_guard_regressions.py::test_solver_trainable_guard_covers_circuit_parameter_channel` | 1 pass → 1 pass. |
| `fdtd/multi_gpu/test_copper_foil_owner.py::test_distributed_foil_lossy_ade_and_remote_mna_owner_match_single_domain` | 2 two-GPU skips → same. |
| `fdtd/multi_gpu/test_radiation_observation.py`: owner-history resume and finite-copper circuit-radiation cases | 5 two-GPU skips → same. |
| `core/scene/test_fixed_constructor_inputs.py::test_registered_circuit_waveform_keeps_tensor_ancestry` | 1 pass → 1 pass. |

The eight physical two-GPU census cases cannot run with the required single
visible GPU; no required regression is skipped. The requested regression run
passes all 162 cases (full scattering and stand-in files, circuits, pulse
normalization, both multi-frequency observer tests, and time planning).
The additional census passes 71 with the eight unchanged skips. The only
four fail-to-pass movers are broadband auto power, circuit orientation, and
the stand-in's two stats readers. No failing-test assertions were edited.

## AMENDMENT — 2026-10-31, user decision option (a)

This amendment supersedes the band-centre default described below. Batch 1
at `c971ac8a` found that adding another requested frequency changed absolute
pulsed outputs: at 0.9 GHz in a 0.9/1.1 GHz band, multi/single was `10/9`;
at 1.1 GHz it was `10/11`. The default must make each output independent
of the other requested frequencies.

`reference_frequency=None` now means a per-output-frequency reference period,
`T_ref(f) = 1/f`. The default absolute broadband spectrum is the Fourier
transform times `f`; an explicit reference frequency retains the Fourier
transform times one constant across bins. Relative to the band-centre
default `f_c`, amplitudes convert by `f/f_c` per bin and quadratic outputs
by `(f/f_c)^2`. Single-frequency numerical outputs and CW scaling are
unchanged. Source-normalized spectra and S-parameters retain their values
to rounding. Metadata and statistics report `"per-frequency"` for the
default, or the explicit float. Runtime state uses `None` for the default;
multi-device shards and adjoint seed solvers copy it unchanged.

The amendment's benchmark impact is narrower than the original decision-B
migration table below. These are source/configuration audits, not benchmark
reruns. The unified runner defaults to source normalization when all incident
waveforms match; the dedicated antenna and exposure runners use the sampler's
unnormalized default.

| Row / runner | Multi-frequency? | Amendment impact under current configuration |
| --- | --- | --- |
| `coverage/postprocess.py`: `directivity_two_dipoles` | No: 0.5 GHz | CW (`FAR_FIELD_PULSE` is a CW source), explicitly unnormalized; unchanged. |
| `coverage/postprocess.py`: `time_monitor_vacuum` | No: 0.5 GHz | Pulsed, explicitly unnormalized; unchanged, including time traces. |
| `sar/antenna_near_phantom`, dedicated `sar_validation.py` runner | No: 3 GHz | Pulsed, unnormalized; accepted/absorbed powers and SAR unchanged. |
| `antenna/half_wave_dipole`, `rf_validation.py` runner | Yes: 2.4, 2.7, 3.0, 3.21, 3.39, 3.6 GHz | Raw port/field/radiation amplitudes move by `f/3 GHz`, powers by its square. Impedance, S11, directivity and power closure retain their values to rounding. |
| `antenna/patch`, `rf_validation.py` runner | Yes: 2.9 through 3.5 GHz, step 0.1 GHz | Raw amplitudes move by `f/3.2 GHz`, powers by its square; ratios retain their values to rounding. The scene builder's separate default sweep also has eight frequencies. |
| `media/modulated_slab.py`: `modulated_slab`; `coverage/media.py`: `modulated_slab_phase` | Yes: carrier and sideband bins | CW, explicitly unnormalized; unchanged. |
| `batch.py`: `local_spectrum` | Yes: 31 bins, 0.7–1.3 GHz | Pulsed, unnormalized; absolute probe spectra move by `f/1 GHz`. Single-frequency `launch` and `bandwidth` stay unchanged. |
| `datagen/scenes.py`: printed and wire antenna samples | Yes: configured bin counts | Pulsed, unnormalized; absolute port/radiation spectra move by `f/f_c` and powers by its square. Ratios retain their values to rounding. |

`antenna/printed_dipole.py` and `antenna/printed_patch.py` are scene builders,
with three output frequencies by default (2.5/3/3.5 and 3/3.2/3.4 GHz).
Their raw pulsed spectra likewise move when run unnormalized; they do not
define additional registered validation rows. Other unified-runner pulsed
rows with matching incident waveforms use source normalization and retain
their outputs to rounding. The array runner is CW; the standalone
`fdfd_vs_fdtd.py` and `mom_fdtd.py` comparisons are single-frequency.

Reader audit (`rg 'spectral_scale|_spectral_reference_frequency' witwin tests
scripts benchmark`): all 12 direct `spectral_scale` calls now supply a
frequency; tensor scales broadcast on the leading frequency axis.

| Module | Audited readers / changes |
| --- | --- |
| `fdtd/runtime/representation.py` | `spectral_scale`: explicit `None` default and per-bin frequency argument. |
| `simulation.py`, `fdtd/runtime/initialization.py` | Configure/initialize `None`; Result metadata and solver statistics label it `"per-frequency"`. |
| `fdtd/observers.py` | `_spectral_scale` supplies the entry frequency; shared point/plane caches and both flux/surface finalization paths consume it. |
| `fdtd/postprocess.py` | `_build_frequency_solution_from_fields`, `_legacy_frequency_solution`, `_solver_source_spectrum`, every single/all-frequency solution branch. |
| `fdtd/ports.py` | `PortDFTAccumulator.phasors` broadcasts accumulator frequencies; its `reference_frequency` default changed from `1.0` to `None` (per-bin), so out-of-tree callers relying on the old default now get per-bin scaling; `finalize_port_data` passes the explicit/default reference to both field and drive accumulators and labels metadata/conventions. |
| `fdtd/wire.py` | `finalize_wire_data` scales every monitor entry at its frequency and labels its convention. |
| `fdtd/copper_foil.py` | `copper_loss_report` broadcasts side-spectrum frequencies; `copper_sampling_provenance` labels the reference. |
| `fdtd/adjoint/copper_spectra.py` | `prepare_spectral_outputs` retains `None` in internal descriptors; `rebuild_spectral_outputs` broadcasts descriptor frequencies. |
| `fdtd/runtime/spectral.py` | `source_normalization_spectrum` scales each source bin at its frequency. |
| `fdtd/adjoint/seeds.py` | Wire and port seeds supply bin frequencies; drive phasors use accumulator frequencies. The seed-solver assignment already copies `None` unchanged. |
| `fdtd/runtime/timeplan.py` | Existing shard assignment already copies `None` unchanged; no edit needed. |
| Tests | Five stand-in files use `None`: selected conductor queries, solver precision stats, mapped material adjoints, point-source normalization, RF lumped adjoints. Conductor power-loss and time-consumer spectrum reconstructions pass bin frequencies. The existing pulse explicit-reference test expects per-bin factors. |

No readers occur in `scripts/` or `benchmark/`; no capability guards or
raises changed. The two multi-frequency observer regressions are unedited.

Fresh-interpreter controls use `git archive master` at
`36f38e5772fe392ff360acb14881178534970eb7` and assert import provenance.
Every captured output tensor (Result snapshot, raw outputs and final Yee
fields) is checked with `torch.equal` for the single-frequency cases:

| Control | Captured tensors | Result |
| --- | ---: | --- |
| Pulsed dipole: full fields, point, plane and flux | 80 | Bit-identical. |
| Pulsed port/wire: fields, monitors, V/I, waves and powers | 110 | Bit-identical. |
| CW port/wire | 110 | Bit-identical. |
| Two-frequency dipole, source normalization enabled | 86 physical/raw-state tensors | Maximum relative difference `2.0269837e-7`. |
| Two-frequency port S11 and impedance | 2 tensors | Maximum relative difference `1.4099965e-16`. |

All deterministic statistics agree exactly. Wall-clock `elapsed_s`,
`steady_step_elapsed_s`, `steady_ms_per_step`, `ms_per_step` and
`steps_per_second` are excluded from equality checks. The only other
single-frequency changes are `reference_frequency` from a float to
`"per-frequency"` in Result/port metadata and statistics, and the pulsed
port/wire convention strings: `Fourier integral with per-frequency
reference period and exp(+i*omega*t) kernel` and `Kurokawa spectral waves
in sqrt(watt) per-frequency reference period`. CW convention strings agree.
The retained normalization denominators are not normalized physical outputs:
their twelve captured aliases change by `f/f_c` as expected, with conversion
error at most `7.1405594e-8`.

Probe/comparison scripts, archived tree and tensor captures are retained under
`<scratch>/witwin_cov_dftbug/perbin_*`, `perbin-master*` and `perbin-branch.pt`;
the machine-readable comparison is `perbin-comparison.json` in that directory.
All 162 unique fast-tier tests pass: the two unedited multi-frequency observer
tests and the 160 tests in the eight requested files. Receipts are
`perbin-observers.xml`, `perbin-fast.xml` and `perbin-pulse-final.xml`;
the final pulse receipt supersedes the initial explicit-reference expectation
failure (noninteger per-bin factors require two subnormal ULPs and factors on
the point tensor's device). `perbin-summary.json` consolidates the final case
outcomes; `perbin-reader-audit.txt` retains the complete grep output. No new
tests or baseline pins were added, and no broad suite or benchmark was run.

The follow-up [absolute-power pin and copper adjoint audit](../assessments/fdtd-pulse-per-bin-power-audit.md)
records the unchanged frozen-gate failures, per-bin power/amplitude conversion
checks and a new multi-frequency unnormalized pulse copper-loss finite difference.

### Pulsed TFSF physical entry-plane reference re-pin (`bacac15f`)

`coverage/tfsf-pulse-reference` anchors the pulsed TFSF incident reference at the physical upstream
Yee entry plane of the injection box (previously the PML-padded domain minimum, which made the
absolute incident phase depend on the PML layer count). User-approved re-pin. Proof
(`<evidence-root>/coverage-e-timing-proof/results/tfsf_ref_gpu/summary.json`, 16/16,
gate assertions unchanged, `torch.equal`, no tolerance): f52aa33f plus only the forward entry-plane
hunk reproduces bacac15f bitwise on both changed cases; the nine other field cases and the full
fingerprint scope are unchanged; master 3dfbda86 equals f52aa33f on both cases.

| Gate | Old pin | New pin | Reason |
| --- | --- | --- | --- |
| `tests/fdtd/test_bit_identity_baseline.py` (field) | `f52aa33f` | `bacac15f` (merge with `--no-ff`) | pulsed TFSF entry-plane reference (tfsf_plane_wave_gaussian, tfsf_plane_wave_ricker); the four soft-sheet sign reversals retire because bacac15f already contains the soft-sheet sign fix |

Unchanged: the fingerprint pins `739d1ca8` / `9237642a` and every other bit-identity gate.

### Item (e) re-pin: additive-J injection at (n+1/2)dt (`f52aa33f`)

`f52aa33f` injects additive electric currents (PointDipole, UniformCurrentSource,
electric CustomCurrentSource) at `(n+1/2)dt` and samples each source's
normalisation spectrum at its injection instant. The bit-identity gates below
changed ONLY in their additive-J cases. Proof
(`<evidence-root>/coverage-e-timing-proof/`, 44 GPU probes, 109
checks, gate assertions unchanged, `torch.equal`, no tolerance): each old
baseline tree plus only the additive-J timing hunk reproduces `f52aa33f`
bitwise on every changed case, and every non-J case is identical between the
original and patched baselines. Authorized by the coordinator; the port-step
gate re-pin is ACKed by its PCB owner.

| Gate | Old pin | New pin | Reason |
| --- | --- | --- | --- |
| `tests/fdtd/test_bit_identity_baseline.py` (field) | `42960ad2` | `f52aa33f` | (e) additive-J injection n -> n+1/2 (dipole_gaussian, dipole_cw) |
| `tests/boundaries/absorbers/test_boundary_bit_identity.py` | `74900e42` | `db3bd9b5` (= `f52aa33f` + checkpoint timing metadata; field-identical to `f52aa33f`) | (e) additive-J injection n -> n+1/2 (5 dipole configs); disk checkpoint must carry the n+1/2 metadata to resume |
| `tests/monitors/in_simulation/test_existing_monitor_bit_identity.py` | `95a3ec3f` | `f52aa33f` | (e) additive-J injection n -> n+1/2 (PointDipole in the scene) |
| `tests/rf/lumped/test_port_step_graph_bit_identity.py` | `95a3ec3f` | `f52aa33f` (copper override stays `95a3ec3f`) | (e) additive-J injection n -> n+1/2 (dipole, network_embedded field_0) |

| `tests/materials/surface_impedance/test_roughness_opt_in_bit_identity.py` | `075491bc` (copper `1b242d6d`) | `f52aa33f` (copper unchanged) | (e) additive-J injection n -> n+1/2 (box_lossy, box_rational, sphere_lossy E/H; PointDipole probes); missed by the first hand-picked gate list, proven separately: `<evidence-root>/coverage-e-timing-proof/results/added_gates_gpu/` (29/29) |

Unchanged: `tests/rf/lumped/test_rf_port_bit_identity.py` (`95a3ec3f`), `tests/rf/terminal_wave/test_terminal_wave_bit_identity.py` (`95a3ec3f`, proven unchanged), the
fingerprint pins `739d1ca8` and `9237642a`, and the port-step copper override.
All pure port-driven port-step subcases are bitwise identical across the re-pin.

### Batch-2 re-pin list after rebase onto `652833bc`

The final rebased implementation is `95a3ec3f`. These two authorized gate
re-pins supersede their original audit failures and the batch-1 pin list
below. Each gate regenerates reference payloads through its existing
`baseline_tree` git-archive helper and scene script in fresh interpreters;
no stored tensor numbers are edited.

| Gate | Old pin | Batch-2 pin |
| --- | --- | --- |
| `tests/rf/lumped/test_rf_port_bit_identity.py` | `c971ac8a` | `95a3ec3f` |
| `tests/rf/lumped/test_port_step_graph_bit_identity.py` | `c971ac8a`; copper override `1b242d6d` | `95a3ec3f` for both |

The [absolute-power audit](../assessments/fdtd-pulse-per-bin-power-audit.md)
checks every old-to-new absolute pulsed value against `(f/f_c)^k` per bin,
with `k=1` for amplitudes and `k=2` for powers, within `1e-6`. The two frozen
gates cover 92 quantities / 213 scalar values with maximum conversion error
`8.366747211083322e-16`. Exactly 31 graph metadata keys change from numeric
reference frequencies to `"per-frequency"`; its numeric walker omits the
new strings. Other baseline pins retain their existing values.

Fresh-interpreter single-frequency controls were repeated against a new
`git archive 652833bc`, superseding the first-pass base above. All 300 tensors
pass `torch.equal`: pulsed dipole 80, pulsed port/wire 110, CW 110. All 435
deterministic statistic entries (145 per case across both captured views)
agree exactly, with the same wall-clock exclusions and metadata-string
exceptions as above. The receipt is
`<scratch>/witwin_cov_dftbug/rebased-single-comparison.json`.

Master's explicit volume-DFT request handling also requires two fixture
corrections: the copper-gradient scene now requests `full_field_dft=True`
for its inherited bulk-loss monitor, and the graph probe reads stored
`Result._fields` directly. This retains its internal array capture when
public volume access is disabled for wire-gap and copper fixtures, without
changing their `False` requests. The original payload key set is preserved
apart from the 31 metadata keys. Physical scenes, objectives and numerical
assertions are unchanged.

Final rebased validation: **165 unique tests passed, 0 failed, 0 skipped**:
the two unedited multi-frequency observers, three pulse-normalization cases,
157 updated stand-in / spectrum-reconstruction cases, the new absolute
copper-gradient test and both re-pinned gates. The gates compare 30 and
2,179 non-flag tensors with `torch.equal`. Fresh old-pin comparisons confirm
the same `8.366747211083322e-16` worst conversion error and exactly 31
removed numeric metadata keys, with no added keys. Receipts are
`<scratch>/witwin_cov_dftbug/rebased-test-summary.json` and
`rebased-pin-summary.json` in that directory; the linked audit records the
underlying pytest and per-bin receipts. The copper finite-difference results
reproduce the original audit numbers exactly.

Decision B: 2026-10-30. Preserve the positive Fourier kernel `exp(+iωt)` and
all Yee time offsets. Normalize every driven non-CW spectrum per reference
period, with `T_ref = 1/f_ref`:

`X(f) = dt * sum_{n executed} x[n] exp(+i 2π f t_n) / T_ref`.

## Public contract

`Simulation.fdtd(..., reference_frequency=None)` (also available on the
`Simulation` constructor and `FDTDConfig`) resolves `f_ref` to the requested
band centre `(min(frequencies) + max(frequencies))/2`. An explicit value must
be finite and positive. The generic constructor rejects an explicit value for
other methods; the frequency-domain factory has no such argument.
The resolved value is stored as `reference_frequency` in Result metadata,
solver statistics, port metadata and retained conductor sampling provenance.
It survives Result save/load. The time-plan's existing `reference_frequency`
continues to mean the slow timing frequency; it does not set the spectral scale.

`runtime.representation.spectral_scale` multiplies pulse sums by `dt*f_ref`.
Full fields, point/plane/flux/closed-surface observations, port V/I and
available power, wire current/charge/loss, retained conductor side spectra
and adjoint seeds share this scale. Quadratic powers inherit `f_ref²`.
The result is independent of planned/executed horizon divisors and keeps
amplitude units (V, A, V/m; powers W). It is a Fourier integral per unit time,
not a CW peak amplitude or pulse energy integrated over a frequency bin.
The reference-period scale also avoids the tiny float32 powers/gradients of
the intermediate unscaled-integral implementation.

Classification is made once in `compile_fdtd_time_plan`: all declared source
times must be `cw` for CW averaging. Gaussian pulses, Ricker wavelets,
broadband pulses, custom tables/callables, circuit waveforms and real mixed
CW/non-CW runs use the full response from step zero without tapering.
Source-free averaging, source amplitude/phase, time histories and stepping
are unchanged. Raw accumulated sums and resume-record layout are unchanged.

**Recommended for labels and optimisation:** use
`SpectralSampler(normalize_source=True)`. Fields, ports, power waves, wire
spectra and dipole-emission power then divide by the same unit-waveform
source spectrum (its squared magnitude for powers), retaining physical
source amplitude and phase. The common `dt*f_ref` cancels. This requires the
existing unique incident-waveform contract.

**CW exception:** `normalize_source=False` retains bit-identical CW numerical
outputs. With `normalize_source=True`, previously unnormalized CW port/wire/
emission outputs intentionally receive the same per-source-spectrum divisor
as fields: `V_new=V_old/S_CW`, `I_new=I_old/S_CW`,
`a_new=a_old/S_CW`, `b_new=b_old/S_CW`, `P_new=P_old/abs(S_CW)^2`,
including available power. The existing CW window and analytic-complex
factor of one are unchanged. Explicit `f_ref` does not rescale CW outputs.

## Converting saved numbers

For a real pulse, old master used `A_old = 2 Σx/N_planned` and
`S_old = 2 Σu/N_planned`, where `u` is the unit-amplitude, zero-phase source.
The old-to-new amplitude factor is

`g = f_ref * dt * N_planned / 2 = (planned cycles at f_ref) / 2`.

Use the old **planned** count even if execution stopped early. More generally,
for old window-weight sum `W_plan` and phasor factor `c`,
`g = f_ref * dt * W_plan / c`.

| Output | Old master → decision B |
| --- | --- |
| Unnormalized pulse amplitudes (fields, V/I, a/b, far fields) | `A_new = g A_old` |
| Unnormalized pulse quadratic outputs (powers, flux, radiation, wire loss) | `P_new = abs(g)^2 P_old` |
| Already source-normalized fields/flux | Unchanged to rounding: the common scale cancels |
| Previously unnormalized ports/wires with source normalization enabled | `A_new = A_old/S_old`; `P_new = P_old/abs(S_old)^2` |

For example, a 1 GHz dipole over **34 planned cycles at 1 GHz** has
`g=34/2=17`: amplitudes increase ×17 and powers ×289. An explicit reference
frequency twice the band centre doubles unnormalized pulse amplitudes and
quadruples powers. Saved outputs from the intermediate pure-integral commits
`f915a989` through `e8bf29bf` instead need ×`f_ref` amplitudes and ×`f_ref²` powers.

## Fast-tier evidence

The fresh-interpreter comparison script
`scripts/dev/fdtd/pulse_dft_invariants.py` extracts both source trees with
`git archive` and asserts import provenance. Archived master is
`7ca086d178baa9d5228ab7856daa509cd6763bc7`. It compares dipole, wire-port and
TFSF scenes at three frequencies, with source normalization off/on; the
max-norm relative tolerance is `1e-5`. Port ratios include complex S11,
V/I, accepted/incident and incident/available power. Field ratios include
point/plane, flux/closed flux and directivity; normalized fields are also
compared directly. The CW case captures 30 outputs, including final Yee fields.

Decision-B implementation: `c971ac8a`.

| Case | normalize_source | Largest ratio / normalized-field relative error |
| --- | --- | ---: |
| Point dipole | false | `3.7252511e-7` |
| Point dipole | true | `3.3342999e-7` |
| Wire lumped port | false | `1.2789599e-7` |
| Wire lumped port | true | `2.6365342e-7` |
| TFSF dielectric sphere | false | `2.8712918e-7` |
| TFSF dielectric sphere | true | `2.2677015e-7` |
| CW wire | false | `0`; all 30 captured outputs bit-identical |
| CW wire | true | `4.7206355e-16`; normalized fields bit-identical |

All eight cases pass. The largest complex S11 relative error is
`4.7206355e-16`; the largest accepted/incident power ratio error is
`5.1920221e-16`. Machine-readable evidence:
`<scratch>/witwin_cov_dft/invariants-option-b/comparison.json`.

PEC dipole, `f_ref=2.4e9 Hz`, `dt=4.084967320261438e-12 s`:

| Planned / executed steps | Centre-frequency incident power |
| --- | ---: |
| 6,976 / 6,976 | `0.00030957539140457073` |
| 27,376 / 27,376 | `0.00030957539108083756` |
| 6,976 / 900 (shutoff 1e-5) | `0.0003094815009304288` |
| 27,376 / 900 (shutoff 1e-5) | `0.0003094815009304288` |

The largest fixed-horizon relative difference over 1.92, 2.40 and 2.88 GHz
is `1.5730621e-9` (acceptance `<0.005`). Shutoff spectra are bit-identical
across horizons. The source-normalization regression passes for both pulse
and CW ports, fields, wire spectra, powers and port gradients. Both geometry
gradient smoke tests pass with relative sensitivity checks.
Changing `f_ref` from `2.4e9` to `4.8e9 Hz` gives exact ×2 port V/I and
waves, exact ×2 point fields and exact ×4 incident power. Full-volume
float32 fields differ from exact ×2 by at most one subnormal ULP,
`1.401298464324817e-45`; the test permits only that rounding. Probe evidence:
`<scratch>/witwin_cov_dft/reference-scaling-option-b.json`.

The edited single-device threshold checks pass: 23 custom-source, tensor,
network, CPML, port and finite-difference cases, plus both geometry-gradient
checks (25 total). Receipts are `<scratch>/witwin_cov_dft/option-b-fast.xml` and
`<scratch>/witwin_cov_dft/option-b-pulse.xml`. The pulse module's final receipt,
including explicit-reference Result save/load, is
`<scratch>/witwin_cov_dft/option-b-pulse-final.xml` (3/3 pass).
All 28 unique requested single-device cases pass after the targeted test
corrections; `<scratch>/witwin_cov_dft/option-b-summary.json` combines the receipts.
No broad suite was run.

## Baseline pins and batch-1 follow-up

Keep only the three previously authorized absolute-spectrum re-pins, updated
to the decision-B implementation commit:

| Gate | Original pin | Intermediate integral pin | Decision-B pin |
| --- | --- | --- | --- |
| `tests/rf/lumped/test_rf_port_bit_identity.py` | `aa7e14fd` | `a6f3da59` | `c971ac8a` |
| `tests/rf/lumped/test_port_step_graph_bit_identity.py` | `283d4956` | `a6f3da59` | `c971ac8a` |
| `tests/monitors/in_simulation/test_existing_monitor_bit_identity.py` | `a5c7a318` | `a6f3da59` | `c971ac8a` |

No further re-pins in this fast tier. Batch-1 integration must review any
additional absolute pulse baseline failures using `g`/`g²`, and handle
benchmark caches/reference updates. Field/fingerprint pins `42960ad2`,
`739d1ca8`, `9237642a` and boundary pin `74900e42` remain unchanged.
Two known original-pin failures remain separate baseline upkeep:
terminal-wave pin `7944434f` predates the PEC node rule and already differs
on archived master; rough-surface pin `075491bc` already differs on the same
eight CPU fit tensors on master. The prior master controls and audit are
`<scratch>/witwin_cov_dft/preexisting-baselines.json`, `master_terminal.pt` and
`master_roughness.pt`. Neither failure justifies an additional re-pin here.
The threshold audit also replaces absolute pulsed gradient guards in the
two-device CPML parity tests and their process-group worker with relative
loss sensitivities. Their two-device validation belongs to batch-1: this
fast tier exposes only GPU 1. CW and instantaneous-field magnitude gates
are retained.

## Absolute-unit benchmark rows: batch-1 integration

These explicit `normalize_source=False` rows must have cached references
regenerated **or a documented scale/no-change justification** in batch-1:

| Definition / row | Required reference treatment |
| --- | --- |
| `benchmark/scenes/media/modulated_slab.py`: `modulated_slab` | CW: scale 1, carrier-relative sidebands unchanged; document this when retaining the cache. |
| `benchmark/scenes/coverage/media.py`: `modulated_slab_phase` | CW: scale 1, carrier-relative fields unchanged; document retained cache. |
| `benchmark/scenes/coverage/postprocess.py`: `directivity_two_dipoles` | Pulse raw fields ×`g`, radiation/power ×`g²`; regenerate absolute spectra or document conversion. Directivity and carrier-RMS-normalized metrics stay invariant. |
| `benchmark/scenes/coverage/postprocess.py`: `time_monitor_vacuum` | Pulse raw spectral fields ×`g`; regenerate or document conversion. Time traces and carrier-RMS-normalized metrics stay invariant. |

The default-unnormalized dedicated `sar/antenna_near_phantom` runner has
absolute `accepted_power` and `absorbed_power` metrics that move ×`g²`;
SAR at one watt accepted power stays invariant. `antenna/half_wave_dipole`
and `antenna/patch` raw pulse port/radiation spectra move, while impedance,
reflection, directivity, pattern correlation and power-closure ratios stay
invariant. CW RF sweeps and the CW array runner retain their outputs.
Batch-1 owns these cache conversions/regenerations, any further baseline
re-pins and rerunning metrics into `benchmark/RESULTS.md`. No broad suite or
full benchmark is part of this fast tier.

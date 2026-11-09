# Integration batch inventory and runner

Inventory at `f3cd5172`, 2026-10-30, with component-scoped planning. No GPU execution
was performed here. Frozen A6000 hash pins require sm_86 and the original toolchain;
live archived-source equality gates require matching GPU model/toolchain between
baseline and candidate. Counts below describe discovery, not qualification.

| Suite | Exact selector / runner | Classification | One A6000 estimate | Evidence / peak memory |
| --- | --- | --- | --- | --- |
| Full GPU pytest | Per-component `pytest-<name>` suites; without `--since`, equivalent to `python -m pytest tests --gpu` (one file per interpreter) | MIXED: exact equality/hash pins and tolerance/physics checks | **Guess:** 16.69 h of file weights, including baseline overlap | Owner reports clean CUDA collection: 9,989 tests. No complete current timing receipt. Peak unknown. |
| Resident equality gates | `resident-bitid` selectors in [the manifest](../../scripts/dev/integration_suites.toml): port graph files, five exact local-cache functions, batch planar-sheet equality function | BITID; local-cache file also executes its seven tolerance cases and one ordering guard because units are whole files | **Guess:** 10 min; included in full pytest | [Resident plan](fdtd_resident_design_updates.md): patch run 250 ms; 28 historical fixtures compared. Snapshot 41 MiB is **not** peak memory. Peak unknown. 13 nodes collected here, 19 additional exact local-cache cases inferred from source (32 selected in total, pending GPU collection). |
| Other FDTD frozen baselines | `fdtd-baselines`: six files listed below | BITID | **Guess:** 20 min including fresh native builds; included in full pytest | 30 collected nodes. Tests compare against archived source refs. No stored timing/peak receipt found. |
| PCB 44 non-thin-sheet | Six exact files in manifest `pcb-44`, preserved from the owner | MIXED: four live archived-source equality files; copper execution and planar terminal targets use tolerances | **Estimate:** 25 min uncontended | Owner: 44 passed in 33:44 contended; logs `<evidence-root>/thin-sheet-accept-20261030/{run.sh,regress.txt}`. Peak estimated <6 GiB. |
| Thin-sheet 112 | `tests/materials/thin_sheet`, manifest `thin-sheet-112` | MIXED: mostly tolerance physics; graph/eager equality and pinned PEC-operator hashes | **Estimate:** 75 min | Owner: 112 passed after `4dbf03c1` in 1:14:16 (earlier 1:07:48), contended. Peak about 3.2 GiB. |
| Unified benchmark | One `benchmark:<name>` unit per registered scenario, running `WITWIN_BENCHMARK_NO_CLOUD=1 python -m benchmark <name>` | TOLERANCE | Per-scenario original-run timing from RESULTS.md; otherwise **guess:** 40 min / scenario count | [Results](../../benchmark/RESULTS.md): 102 original-run rows sum to 655.22 s, excluding doubled-duration runs, plotting and four unmeasured scenarios. 106 registered scenarios. Largest recorded driver delta: 17,142 MiB (`anisotropic_slab`), not a current whole-run peak. |
| MoM public smoke | `tests/mom/test_public.py` | MIXED: persistence/mesh `torch.equal`; field queries use `rtol=5e-13, atol=1e-15`, geometry `1e-6/1e-8`, convergence/finite bounds | **Guess:** 30–40 s; included in full pytest | [Historical JUnit](../assessments/gsp-01-bound-lifted-public-009231c-v1/junit.xml): nine cases total 12.90 s. Peak unknown. |

Full-suite selection has no uniform GPU marker. `tests/conftest.py` uses `--gpu`
only for the cross-solver validation family; other modules use CUDA skip conditions,
fixtures or CUDA allocations during import. The four failures previously observed in
`tests/materials/compiler/test_local_material_cache.py`,
`tests/rf/network/test_network_cascade_crosscheck.py`,
`tests/rf/network/test_network_conservation.py` and
`tests/mom/test_spatial_validation.py` are module-level CUDA allocations, not broken
imports. With a GPU visible, the owner reports clean collection of **9,989 tests**.
**Planning and collection must run with `CUDA_VISIBLE_DEVICES` set to the lane GPU.**
Both `plan` and `run` fail immediately on any collection error; no partial assignment
is printed and no files are silently dropped. Those four test files are unchanged.

The resident design plan is stale about local recompile: the primary local-cache
test module already contains exact PlanarPEC controls. Its first two functions are
tolerance checks; the five selected functions call `assert_identical`. The resident
update API itself is absent from `simulation.py` at this commit.

Frozen refs are test-source pins, not stored field arrays:

| Test file | Default pin(s) |
| --- | --- |
| `tests/rf/lumped/test_port_step_graph_bit_identity.py` | `283d4956`; dump producer `tests/support/port_step_graph_bit_identity_scene.py` |
| `tests/fdtd/test_bit_identity_baseline.py` | fingerprints `739d1ca8`, subpixel `9237642a`, source fields `42960ad2`; `_bit_identity_probe.py` |
| `tests/boundaries/absorbers/test_boundary_bit_identity.py` | `74900e42`; `_boundary_state_probe.py` |
| `tests/monitors/in_simulation/test_existing_monitor_bit_identity.py` | `a5c7a318`; `_bit_identity_runner.py` |
| `tests/rf/lumped/test_rf_port_bit_identity.py` | `aa7e14fd`; support scene |
| `tests/rf/terminal_wave/test_terminal_wave_bit_identity.py` | `7944434f`; support scene |
| `tests/materials/surface_impedance/test_roughness_opt_in_bit_identity.py` | `075491bc`; `smooth_surface_probe.py` |

These tests archive their baseline source and write temporary tensor payloads.
Most failures expose only equality booleans or mismatched keys, not numerical old/new
values. The runner preserves pytest basetemp payloads and original assertion text.
Pytest JUnit logging captures failures' stdout/stderr. Reports read that XML and
include each failing BITID/MIXED assertion message plus the final 8,000 characters
of each captured stream in both Markdown and JSON; absent captures are explicit.
Value dumps can be added at the failing equality assertion in the named test
or its probe. Pin-definition changes are
reported separately, with old/new refs and the causing first-parent batch commit.
Environment baseline overrides remain in force and are recorded as effective test
pins alongside the committed definitions.

Pinned comparisons convert to numerical tiers when touched. The reference example
is `tests/rf/terminal_wave/test_terminal_wave_bit_identity.py`: it keeps the
`torch.equal` fast path and calls `record_pin_comparison(ref, new, nodeid=...,
name=..., level="run")` only on mismatch. The helper in
`tests/support/numerical_equivalence.py` passes T1 and raises on T2, even when logging
is disabled. The runner sets `WITWIN_PIN_TIER_LOG` to a fresh per-unit JSONL file in
each shard. Each receipt is a final comparator tensor row plus `nodeid`: `name`, `dtype`,
`shape`, `max_abs`, `max_ref`, `max_rel`, `bound`, `ratio_to_bound`, `tier` and `level`.
Unit JSON embeds these as `pin_tiers` and records `pin_tier_log`; batch JSON exposes
`bitid_tiers` keyed by unit ID, each with `counts`, `t1_movers` and all non-T0
`movers`. Report rows retain the comparator field names; declaration tables use
`max_rel_diff` for their disclosed maximum relative difference. Markdown includes
the same T0/T1/T2 table and each T1 tensor's maximum relative difference. Counts are
test nodes classified by their worst tensor tier: passing exact fast paths without
receipts count as T0, failures as T2, and skips remain unmeasured. T1 receipts add no
failure category and do not suppress any other failing assertion.

The shared comparator accepts nested tensor dicts/lists/tuples and returns
`{"tier": ..., "level": ..., "tensors": [...]}`. Each tensor row has a dotted or
indexed `name`, `dtype`, `shape`, `max_abs`, `max_ref`, `max_rel`, `bound`,
`ratio_to_bound`, `tier` and `level`, with ordinary Python values. Small leaves
are packed for on-device per-tensor reductions; metrics and exact-mismatch flags
transfer to the host once per device/dtype group, including discrete tensors.
The CPU and CUDA-marked performance tests cover 85,000 small continuous leaves
plus bool masks, check the transfer count and require comparison within five seconds.

`assert_numerically_equivalent(ref, new, *, level, discrete=False, name=None)`
applies setup/step bounds of `64*eps32*max_ref` in fp32 and `1e-12*max_ref` in
fp64; run bounds are `1e-4*max_ref` and `1e-10*max_ref`. Complex tensors use the
difference magnitude and their component precision. Zero references, integers,
bools, shapes, dtypes and keys require exact equality; `discrete=True` also forces
floating tensors to be exact. `classify_against_pin(pinned, current, *, level)`
returns the same report without raising on T2. Intended changes require T2 review
regardless of numerical size. Determinism and existing physics gates retain their rules.

`assert_run_level_equivalent(ref_half, new_half, ref_full, new_full, *,
discrete=False, name=None)` checks run bounds at both horizons and at most 4x
growth per tensor. Its rows add `half` and `full` metric rows plus the `growth`
ratio. A zero half-horizon difference allows a nonzero full difference only within
the full run bound (the ratio is then infinity). Histories may have different
half/full lengths, but must have the same leaf paths and dtypes.

Merge declarations disclose pytest movers as legacy node strings (treated as
T2-approved), or typed tables:

```toml
[[merge_declarations]]
branch = "infra/example"
pytest = [
  {node = "tests/example.py::test_rounding", tier = "T1", max_rel_diff = 1e-6},
  {node = "tests/example.py::test_behavior", tier = "T2", intended = true, approval = "user approval reference"},
]
benchmark = "none"
note = "Measured rounding mover and approved behavior change."
```

T0 needs no mover approval. T1 requires a finite nonnegative `max_rel_diff` and
needs no approval. T2 requires a non-empty `approval` reference. `intended = true`
or `discrete = true` forces T2, even if the supplied tier is T0 or T1; approval is
then mandatory. Existing `pytest = "none"` and benchmark lists retain their meaning.
Validation errors identify the declaration and mover index.

Planning classifies the manifest's `expected_movers` (pytest `selector` entries
may supply `tier`, `max_rel_diff`, `intended` and `discrete`). Unspecified tiers
remain T2. Declarations can also force the effective tier through their flags.
The plan refuses undeclared T2 movers and unlisted T1 movers; each T1 must be
listed with a `max_rel_diff` covering its measured value. T0 needs no listing.
The existing `require_merge_declarations = false` switch disables plan coverage
checks for synthetic histories; declaration syntax is always validated.

The report checks fresh pin receipts against the same declarations and records
`mover_declaration_errors` in JSON and Markdown, returning nonzero for missing
coverage. It applies intended/discrete flags to receipts and fails forced T2 test
nodes even if their numerical bound passed. T1 adds no new test failure and cannot
absorb an unrelated failure. Failure annotations still use `expected_movers` and
`known_failures`.

`scripts/dev/fdtd/subpixel_default_bitid.py` also supplies an eight-scene two-tree
dump/compare diagnostic. It has no frozen ref or stored baseline; choosing its
comparison trees remains an operator decision, so it is not a pinned tier suite.

The benchmark defaults to all registered field/reference scenarios, including four
FDFD cases with local FDTD references. It does not select the separate `mom`, `rf`,
`sar` or `batch` workflows. Scene definitions live in `benchmark/scenes`, reference
payloads/keys in `benchmark/cache`, comparison plots in `benchmark/plots`, aggregate
metrics in `benchmark/RESULTS.md`; key construction is in `benchmark/runner.py`.
`WITWIN_BENCHMARK_NO_CLOUD=1` makes missing/stale reference caches fail instead of
submitting remote work. It does not trust stale caches or skip convergence checks.
Worktree creation/reuse provisions `benchmark/cache` from the main checkout with
hardlinks (copies only across filesystems), preserving files tracked by the selected
commit. In-place runs read the cache at `cwd/benchmark/cache`. The cloud batch-gate task
mounts this cache as approved data under the cloud upload policy.
Plans and merged report headers record `benchmark_cache_sha256`, entry count and
bytes. The digest hashes compact JSON of the sorted `(relative path, size, file
SHA256)` list, including tracked cache files. Shards with different manifests cannot
be merged. Planning checks selected external-reference scenarios on CPU with
`CUDA_VISIBLE_DEVICES=` using the benchmark's key function, scene, frequencies,
runtime factor and resolved source normalization. `stale_caches` in the preview
and plan.json lists names, found keys and expected keys for missing files, missing
key attributes and mismatches. Affected units are INFRA (`stale reference cache`)
and skip execution and GPU-memory waits. Local-reference scenarios need no cache.
`WITWIN_BENCHMARK_TRUST_CACHE=1` skips key comparison and announces that choice;
missing files remain INFRA. Each scenario runs in a fresh interpreter and writes
its own shard-local RESULTS artifact via `WITWIN_BENCHMARK_RESULTS_PATH`, so
concurrent scenarios cannot overwrite aggregate results. Reports combine these
units into the benchmark suite row and list failing scenarios.
The runner applies the existing guide's field L2/Linf <0.1, correlation >0.99 and
flux error <0.05 targets from the manifest; unconverged/missing cases are incomplete.
Scalar errors are retained without inventing an additional scalar threshold.

Development scripts that keep run evidence outside the repository read its location from `WITWIN_EVIDENCE_ROOT`; without it they write to `evidence/` under the current directory, which is gitignored. `WITWIN_BENCHMARK_CACHE` overrides the reference-cache location that `benchmark/core_path/gates.py` otherwise resolves to the main checkout's `benchmark/cache`.

## Operation

Local commands use `python`. Examples below
abbreviate it as `$PY`; child processes use the runner's active interpreter.

```bash
PY=python
$PY scripts/dev/integration_batch.py plan --commit HEAD --since BASE --shards 4
$PY scripts/dev/integration_batch.py run --commit SHA --since BASE --shard 0/4 --out /absolute/results --lock /tmp/integration-gpu1.lock
$PY scripts/dev/integration_batch.py report /absolute/results
$PY scripts/dev/integration_batch.py bisect --good BASE --bad HEAD --units pytest:tests/rf/lumped/test_port_step_graph_bit_identity.py
$PY scripts/dev/integration_batch.py report /absolute/results /absolute/bisect-output
```

### Cloud in-place

Run `plan` or `run` with `--in-place` from the uploaded source directory. This uses
cwd directly, without worktrees or git calls. `--commit` is a report label and must
exactly match `.wcloud-source.json:git_head` when that file exists. Effective pins
are read from uploaded test files; pin-change history is unavailable without git.
`--since` and `--in-place` are mutually exclusive. Generate `changed-files.txt`
locally with `git diff --name-only BASE..HEAD > changed-files.txt` and upload it
with the code; `--changed-files changed-files.txt` scopes components identically
to that diff. Without the list, all components are selected. Use the same list
on every shard.

The per-shard command for `cloud/tasks/maxwell-batch.yaml` is:

```bash
CUDA_VISIBLE_DEVICES="$k" python scripts/dev/integration_batch.py run \
  --in-place --commit "$head" --changed-files changed-files.txt --shard "$k/$BATCH_SHARDS" \
  --out "$out" > "$out/shard-$k.log" 2>&1 &
```

The current cloud task still supplies `--since "$BATCH_SINCE"`; that argument must
be replaced with `--changed-files changed-files.txt` before launch. The task YAML
lives outside this repository. It exports `WITWIN_BASELINE_TREES="$HOME/wcloud/revs"`
and `WITWIN_REQUIRE_BASELINES=1`; both are inherited unchanged by children. With
required baselines, skipped tests in units belonging to a BITID suite are reported
as FAIL (missing baseline), including in merged reports.

Before each in-place pytest unit, the runner scans its test files for literal
`docs/...` paths and string-literal path joins. The best-effort regexes in
`docs_paths` and trailing slash/period stripping must stay in sync with the cloud
stager. Missing paths and references
containing files in `$WITWIN_TEST_DOCS_MANIFEST`'s `excluded` list produce an
unexecuted INFRA unit with reason `docs path not staged for cloud: <paths>`.
The manifest's top-level `docs_text_mentions` list exempts exact extracted paths
that tests mention only as text, such as warning-message regexes.
INFRA contributes no passes or failures and does not stop phase 1. Reports list
these units separately from NEW/KNOWN/EXPECTED; a phase-1 INFRA unit adds
`phase 1 incomplete (INFRA)` to the report header. Local worktree runs are unchanged.

The caller sets `CUDA_VISIBLE_DEVICES`; the runner never sets it. Shards are
zero-based: a single physical GPU-1 lane uses `CUDA_VISIBLE_DEVICES=1` with shard
`0/1`. Detached trees live under `.worktrees/ib-<sha8>`;
creation is locked, and reuse requires clean source at the requested commit (the
benchmark aggregate output is exempt while another shard is producing it).
Each unit is a fresh interpreter, optionally protected by a per-unit POSIX flock. Units
overlapping multiple suites run once. Each new `--out/shard-K` contains a frozen
plan, collection log, per-unit log, JUnit XML, JSON and retained temporary payloads.
Use a new output directory for reruns. Within each balanced shard, units execute
in this order: phase 1 `resident-bitid`, `fdtd-baselines`, `pcb-44`; phase 2
`mom-public-smoke`, `thin-sheet-112`, `benchmark`, then remaining component units
longest first. Overlapping units run once at their earliest stage; assignment to
shards is unchanged. The plan preview shows this execution order.

The default stops the shard on any non-pass in phase 1 and continues through phase
2. `--stop-on-fail` stops on a non-pass in either phase; `--keep-going` continues
through both. The complete plan is written before execution, so `report` retains
missing units as incomplete after an early stop and the command returns nonzero.
Before/after lines contain UTC time, phase, unit ID, cumulative per-shard test pass/fail
counts and unit elapsed seconds (including lock wait). They are flushed to stdout
and appended to `OUT/progress.log` for `tail -f`.

Locks remain per unit. Repeat `--lock-for PATTERN=PATH` to override `--lock` for
matching unit-ID globs; the last matching override wins. For example,
`--lock-for 'benchmark:*=<workspace>/wt/cuda1.lock'` uses that lock
for each benchmark scenario, releasing it before starting the next unit.

`--units-from FILE` replaces the balanced lane with newline-separated unit IDs.
`--label NAME` writes to `shard-NAME`; reports accept labels such as `1b` and `1c`
alongside `0` and `1`. `--claim-dir DIR` atomically claims each unit with O_EXCL
after the lane lock and before START; an existing claim skips that unit. Share the
directory across workers and retain claims for the batch. `--min-free-mib N`
waits via nvidia-smi for N MiB free on the first CUDA_VISIBLE_DEVICES entry before
START. Its default of zero performs no memory query.

`--no-start-after 2026-10-31T10:00/2026-10-31T12:15` reserves that local PDT
window. Repeat the option or list multiple `START/END` windows after it. Naive ISO
times use `America/Los_Angeles`; explicit offsets are also accepted. Admission
requires `now + 1.5 * estimated_seconds < next window start`, checked again after
acquiring the unit lock. If the margin is lost, the lock is released and a `DEFER`
line records the wait until the window end. The same unit is then retried. Running
units are never killed; the bound is based on estimates. Waiting does not hold a
lane lock, and elapsed progress includes lock/window waits.

Manifest `known_failures` entries retain selector, reason, breaking commit and
owner. A phase-1 failure whose failing nodes all match these entries does not stop
later units. Unknown failures still stop it. `expected_movers` entries annotate
only matching test selectors or benchmark row names, without checking the scale.
Reports split failures into NEW, KNOWN and EXPECTED sections in Markdown/JSON;
known/expected outcomes remain failures and do not qualify a batch. The five rows
for `f86d2183` are annotated; modulation rows and the protected field/fingerprint/
boundary pins remain unannotated.

Reports write `integration-report.md` and `.json` in the first input directory;
missing shards/units, skips and unresolved suites prevent a passing batch.

The data-only `scripts/dev/integration_suites.toml` supplies source globs and test
selectors under `[components]`. Test selectors partition every `test_*.py` and
`*_test.py` file exactly once; unowned or multiply owned files stop planning and
are listed. Source globs may overlap. `--since BASE` computes
`git diff --name-only BASE..COMMIT` and selects the touched components; changed test
paths select their owner. Shared API files and unmatched `witwin/maxwell/` sources
select all components, as do `witwin/core`, any `witwin-core` path and dependency
declaration changes in `pyproject.toml`. Without `--since`, all components run.
Repeatable `--exclude-component NAME` on `plan` and `run` removes only that
component's full-pytest suite, including selection through shared changes.
Unknown names fail. Always-on suites remain selected; `plan` and merged reports
label excluded components **EXCLUDED by operator**, never PASS. Use identical
exclusions on every shard. Batch 1 uses `--exclude-component mom`: MoM is frozen
at epoch v3 and the batch does not touch `witwin/maxwell/mom`.
The collection-only check for `--commit fe68136d --since f67019aa --shards 1
--exclude-component mom` with `CUDA_VISIBLE_DEVICES=1` exited 1 in this environment:
`test_local_material_cache.py`, `test_network_cascade_crosscheck.py` and
`test_network_conservation.py` raised `No CUDA GPUs are available` during
module-level allocation. No plan counts or time estimates were produced; the
same command still needs collection with lane GPU access.

The six single-device manifest suites remain always on: `resident-bitid`, `fdtd-baselines`,
`pcb-44`, `thin-sheet-112`, `benchmark`, `mom-public-smoke`. Full file units are
collected for their node selectors, and overlaps execute once. `plan` prints
component GPU-h, full-selection flags and the deduplicated always-on total.
Use the same `--since` on every shard. Update `base_commit` for each batch's pin
history. The controller supplies the manifest; the selected commit supplies the
solver, tests and benchmark source.

`two-device-slot` is READY/TOLERANCE and selects
`tests/fdtd/multi_gpu/test_adjoint_parity_cpml.py` and
`tests/fdtd/multi_gpu/test_monitor_numerical.py`. `--two-gpu` on `plan`/`run` admits
these files and asserts two distinct, nonempty `CUDA_VISIBLE_DEVICES` entries.
Without it the suite is NOT RUN; its two-device nodes remain excluded. With it,
only those files' two-device nodes are removed from NOT RUN; the other 173 stay
excluded. The slot comprises 12 two-device cases plus three CPU pinning guards;
its 10-minute wall estimate is a guess based on the CPML/monitor receipts below.

The `250ee2ba` expected-mover entry declares **no pin movers**: single-source
normalised runs remain `torch.equal` to the previous master. Differing source
waveforms with `normalize_source=True` now raise `ValueError` in place of
`NotImplementedError`, while amplitude-only multi-source normalisation runs.
Reports annotate EXPECTED only when the failure text contains both the old
`pytest.raises(NotImplementedError` expectation and the specific new incident-
spectrum `ValueError` message. Numerical failures or unrelated exceptions stay NEW.

## Milestone tier: full two-device pass

`two-device-full` is READY/MIXED, `requires_gpus=2`, `tier="milestone"`.
Its `tests` selector is restricted by `two_device_only=true` to the existing
fixture/marker/parameter declarations: all **185** recorded two-device cases,
including NCCL transport, distributed fields/adjoints, batch/array, copper and MoM
checks. `plan` and `run` ignore this suite unless `--tier milestone` is supplied.
Only `--tier milestone --two-gpu` admits all 185. Ordinary component selection and
the six mandatory suites still apply; overlaps execute once. The slot suite runs
first, followed by remaining full two-device units, before other component units.

**Agreed slot (2026-10-30):** the night of 2026-11-07, 22:00 to about 00:30 PDT, on local
GPU 0 + GPU 1, run on the batch-2 head.
- The PCB line (GPU 0 owner) keeps GPU 0 free from 21:45.
- maxwell-bench-dataset-reuse pauses LQ-3: no new LQ-3 tasks after 21:30, and running tasks
  (≤15 min) finish before 22:00. LQ-3 resumes after the pass.
- PCB's milestone reruns on the batch-2 head follow this pass.
- On a first failure that needs a bisect, the pass stops at the slot end and GPU 0 is handed
  back. The bisect runs on GPU 1 or on another night.
- Fallback: the night of 2026-11-08, same time.

**Estimate (guess): about 2 hours wall on two A6000s, or 4 reserved GPU-h.**
Successful historical JUnit cases cover 75/185 exact node IDs; summing each node's
slowest retained receipt gives 698.90 s (11.65 min). The remaining 110 cases have no
matching retained successful duration. Budgeting 30 s each for them gives another
55 min; round the 66.65-minute subtotal up to two hours for fresh interpreter,
worker startup, native builds and stress controls. Failures/timeouts can take longer.
This is not a new measurement or a qualification claim.

Evidence includes:

- [Two real CPML adjoint cases](../assessments/distributed-precision-stats-repair-2026-10-28/actual-two-CUDA-shards/actual.xml): 77.458 + 51.467 = 128.925 s; the [run log](../assessments/distributed-precision-stats-repair-2026-10-28/actual-two-CUDA-shards/pytest.log) records 129.12 s total on actual CUDA-P2P workers.
- [Distributed radiation](../assessments/fdtd-distributed-radiation-2026-10-22/pcb-distributed-radiation-final.xml): 27 matching successful nodes total 176.04 s; [patch evidence](../assessments/fdtd-distributed-radiation-2026-10-22/pcb-distributed-patch-evidence.xml) adds two slow-case receipts totaling 72.21 s. These scopes overlap; they are not added twice.
- [Owned-trace history](../assessments/distributed-copper-owned-trace-gather-2026-10-28/historical-original-full-failure/junit.xml): 26 matching successful nodes total 342.32 s inside a historically failing run; this timing evidence does not relabel that run as passing.
- [Public MoM receipt](../assessments/gsp-01-bound-lifted-public-009231c-v1/junit.xml): three MoM and two distributed persistence cases total 40.79 s. The [distributed regressions receipt](../assessments/fdtd-distributed-radiation-2026-10-22/pcb-distributed-radiation-regressions.xml) records five monitor cases totaling 1.441 s, with shared fixture setup. `tests/test_output` is absent in this checkout.

**Memory:** a measured whole-185-case peak is unknown. The CPML rerun's
[resource policy](../assessments/distributed-precision-stats-repair-2026-10-28/fresh-rerun-preparation/distributed2-rerun-GPU-config.json)
budgets 2,048 MiB per GPU; this is an upper *estimate* for two CPML cases, not a
measured peak or a bound on the other 183 cases.

**Recommendation (a): one local night slot on GPU 0 + GPU 1, agreed with the PCB
line (GPU 0 owner).** The retained [P2P execution evidence](../assessments/distributed-precision-stats-repair-2026-10-28/README.md)
and [NCCL acceptance](../assessments/h1-nccl-driver-acceptance-2026-07-21.md) already
cover this local pair. Reserve at least three hours for the two-device portion
and additional time for the mandatory single-device suites (about 2.814 h in the
earlier estimate), collection and any selected component work. No slot is booked
by this change. Alternative (b), a cloud 2×A6000 node, requires user approval per
launch; **RunPod 2×A6000 P2P/NCCL support is unverified** and must be established
before relying on it. No cloud command or GPU execution was performed here.

Bisection runs only the requested plan unit IDs, with a fresh detached tree per
probe and a separate bisection-control worktree. Skips, collection errors and
infrastructure failures cannot label a commit bad. Probe exceptions abort; incomplete
probes use git's skip status. Original logs and `bisect.json` retain the culprit;
including that directory in `report` attaches it to matching BITID failures.

## Capacity estimates and open items

These are **planning guesses**, not measured GPU utilization or current runtimes.
The historical full-inventory estimates below include two-device work. Single-lane
plans now exclude the explicit NOT RUN list and print the corresponding costs.
Component counts combine the saved 9,916-node collection, the runner's one CPU
case, and source enumeration of the four CUDA-dependent modules (72 cases), matching
the owner's 9,989 total. No new CUDA collection was run here. The base weight is
12 h / 9,989 per case; file weights take the maximum of that and each applicable
always-on suite's per-case budget. Consequently a file's cost stays fixed when the
component selection changes. Explicit file overrides still apply.

| Component | Cases | Full component GPU-h (includes always-on overlap) | Additional GPU-h beyond always-on |
| --- | ---: | ---: | ---: |
| fdtd | 1,450 | 1.871 | 1.718 |
| compiler | 574 | 0.756 | 0.657 |
| mom | 2,523 | 5.856 | 5.846 |
| fdfd | 103 | 0.124 | 0.124 |
| postprocess | 324 | 0.389 | 0.389 |
| rf | 1,741 | 2.437 | 2.035 |
| materials | 1,131 | 2.542 | 1.215 |
| monitors | 171 | 0.215 | 0.204 |
| sources | 276 | 0.332 | 0.332 |
| boundaries | 133 | 0.289 | 0.144 |
| shared | 1,563 | 1.878 | 1.878 |

Always-on suites total **2.814 GPU-h after file deduplication**, including the
benchmark's 0.667 h. Their raw, overlapping manifest budgets sum to 2.842 h.
Total time equals always-on plus the selected components' additional column.
Without `--since`, the full deduplicated batch estimate is **17.35 h**.

| Example `--since` diff | Full components | One A6000 wall time | Four independent A6000 shards, greedy file assignment |
| --- | --- | ---: | ---: |
| `fdtd/runtime/stepping.py` + `compiler/structures.py` under `witwin/maxwell/` | fdtd + compiler | 5.19 h | 1.30 h |
| `fdtd/ports.py` + `compiler/ports.py` under `witwin/maxwell/` | fdtd + compiler + rf | 7.22 h | 1.81 h |

Both examples include every always-on suite. Four shards sharing one locked GPU
serialize and retain the single-A6000 time; the parallel figures need four GPUs.
The heavy MoM file alone weighs 10,180.63 s, using the historical
4,269.11 + 5,911.52 s in the [resource review](../assessments/full-regression-operations-2026-10-28/heavy-sie-resource/README.md);
this constrains full-suite parallelism but is absent from both scoped examples.
Cold compilation, interpreter startup, contention and caches can dominate.

Open items: current per-file durations; baseline history/cache and external artifact availability;
scalar benchmark acceptance policy; peak memory for other suites. Two-device
discovery uses the existing `cuda_p2p_devices` fixture closure (144 cases), `nccl`
marker (22), two-device `skipif` reasons (17), and CUDA `devices` parameter lists
(2). Targeted collect-only discovery found **185 two-device cases among 494 collected**
with no collection errors. CPU-only controls in the same directories remain selected.
Plans and reports list every excluded node as **NOT RUN (needs 2 GPUs)**; a mixed
file runs only its remaining node IDs. These exclusions are never passes and leave
the associated suites incomplete. Separate two-device allocation is still needed
for complete qualification. No GPU,
cloud launch, long execution or solver modification was performed for this change.

Validation: the single CPU selection/balancing test covers
ownership, source/core/shared selection, collection-error refusal, phase ordering,
stop/continue policies, progress, lock overrides, window deferral with a simulated
clock/lock delay, known/expected matching and two-device exclusion. Synthetic JUnit/report probes
verify assertion messages, both 8,000-character tails and reporting an early stop
as failed/incomplete without running a solver. Estimates use stored discovery and
source counts only. GPU collection/execution and actual GPU bisection remain with
the operator.

# Source-bound pytest shard controller

Private regression tooling; it changes no solver or test assertions. Normal use
collects all `source_root/tests` with the real repository conftests. Each shard is
one module, in original collection order, at most128 nodes (configurable1..256),
and runs in a fresh Python process with its own session/process group. Collection
also uses a fresh process. Numerical execution is never moved to CPU by this tool.

```sh
PY=python
CONTROL=/tmp/pcb-sharded-regression-control-v3
SOURCE=/absolute/path/to/frozen/git-worktree
OUT=/absolute/path/outside/source-and-control
$PY "$CONTROL/run_shards.py" launch --source-root "$SOURCE" --output "$OUT" --gpu
$PY "$CONTROL/run_shards.py" status --output "$OUT"
# Resume with the exact original options and environment:
$PY "$CONTROL/run_shards.py" launch --source-root "$SOURCE" --output "$OUT" --gpu --resume
```

`collect` instead of `launch` freezes complete actual discovery and identities
without executing test bodies. That output is explicitly execution-pending;
`run --resume` with identical options starts its still-pending shards.

`launch` returns immediately with the detached controller PID/start ticks and a
fixed log path. Its pytest children have different new sessions. Controller
SIGTERM/SIGINT kills only the current verified worker group, seals the partial
attempt as interrupted, and exits. Resume reruns that interrupted node and pending
nodes, never completed nodes. A controller SIGKILL may leave its worker alive;
resume re-adopts and watches that exact PID/start-ticks worker before scheduling
anything else. Missing orphan waitpid status is explicitly reported as unknown;
a durable child exit marker is separately identified, never presented as waitpid.

Per-node timeout defaults to1800 seconds across setup/call/teardown; total shard
timeout defaults to7200 seconds, including imports/collection. Configure with
`--node-timeout`, `--shard-timeout`; `--termination-grace` defaults to2 seconds.
Timeout/crash records remain failures; later unstarted nodes are put in fresh
shards. A startup/collection failure with no started node retries that module as
singletons; a singleton still failing before node_start is explicitly
`not_executed_startup_failure`. It is not a test failure or pass. Complete initial
collection must succeed; an interrupted/failed discovery cannot be resumed as a
completed selection. Use a new output after investigating discovery failures.

Optional `--test-path` (repeatable) or `--nodes-file` enables targeted CPU/tool
probes and targeted qualification; default discovery remains every maintained
test. Every selection is frozen after actual collection. `--gpu` is forwarded to
the repository conftest, not emulated by the controller. `--pytest-arg=VALUE` adds
an explicit bound argument. Destructive pytest options (maxfail, xdist, lastfailed,
selection filtering) should not be used for a complete maintained run; actual
selection is checked again in every child before execution.

Source binding includes the package, tests, scripts/helpers, benchmark source,
tracked fixtures/configs, actual installed core source/native files, controller
files, interpreter, configured nvcc and GSP input paths. Child pytest imports prove
actual package/core/helper paths and versions before/after execution. The
controller independently rehashes all files before/after each worker, including
crashes with no child finalizer. Every finalized attempt's events, log, XML,
identity files, worker/exit records and request are hashed and rechecked on resume.
The complete environment is hash-bound (only shell `_`, `SHLVL`, `OLDPWD` and
controller-injected request/event locations are excluded; PWD/PYTHONPATH are
normalized). Environment values outside an explicit public subset are stored as
per-key hashes, not printed. Resume requires identical source, control, selection,
configuration and environment; a different source needs a new output. A lock
prevents two controllers adopting the same output.

Default input pins also cover every git-tracked text input across the source tree
(including reference/assessment JSON and XML), all reference-document text inputs,
and both guard-census baseline paths declared by the current manifest. Nonstandard
external inputs can be frozen with repeatable `--pin-path=FILE_OR_DIRECTORY`.
The output may not be inside an immutable input directory. Extra pins and their
contents participate in the same resume check; they are not test selectors.

`summary.json` and `node-outcomes.json` retain actual setup/call/teardown reports,
skip reasons, wasxfail, durations, failures, timeouts and crash histories. Original
JUnit is per shard; a crash can genuinely prevent that shard's XML from existing.
No synthetic passing JUnit is produced. An optional `--allowed-skips=FILE` tags
exact `(nodeid, original_detail)` matches under its `allowed_skips` array, without
turning any skip into PASS or requiring every allowed node to skip. Unlisted skips
remain explicit. Exit0 means every node passed and no infrastructure issue;
completed skips/xfails/failures/crashes/timeouts return1. Source/identity errors are
controller errors, not successful test completion. No count closes physical or
release gates automatically.

A process timeout after every test phase passed (for example, a hung interpreter
shutdown thread) is still an infrastructure failure and returns1. The phase PASS
records remain visible without making the overall regression successful. Event
readers accept only newline-terminated records from one immutable read snapshot;
partial final writes remain raw evidence until they complete or the writer dies.


## Module-scoped artifact environments

`--module-environment=FILE` enables the reviewed
`module-scoped-artifact-environment/1` manifest. Each exact relative Python test
module has its own `environment` map. `files` pins every declared input using an
absolute canonical path and SHA256; source records and transitive dependencies
can be pinned in the same map. Only artifact-valued `GSP_*` strings are accepted.
SHA pairs are checked, including explicitly named nonstandard pairs. Missing
files, changed hashes, escaped/missing modules and unsupported environment keys
refuse before collection. Availability is not numerical qualification.

The controller removes inherited GSP variables rather than globally merging
conflicting bundles. Complete pytest collection wraps each actual module
collector in its own input environment, including imports and parameter
generation, then restores the prior environment. An execution shard receives
only its one module's inputs before pytest imports. Actual GSP input values are
recorded at collection boundaries, in worker requests and in child identities.
A leak or altered input refuses qualification. Original node order, phase
outcomes, skips, failures, crashes and timeouts remain unchanged. Resume requires
the same source, manifest, inputs and base environment.

`build_module_inputs.py` copies reviewed existing bundle assignments and pins
their original records without editing a historical receipt. Its optional
current-source radial bridge affects only the consuming module and must retain
`production_qualification=false`; without that fresh consumer binding the policy
is preparation-only. A final launcher must refuse stale-source bridges.

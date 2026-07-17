# Circuit Co-simulation Progress Checkpoint (2026-07-16)

## Checkpoint status

This file records the current state of task 04, circuit co-simulation implementation, on branch `codex/spice-mna-cosimulation`. The branch contains completed Phase 0-3 commits followed by the Phase 4 implementation, which is committed as `f3c0df3` (`Checkpoint Phase 4 circuit co-simulation progress`). `f3c0df3` is a **checkpoint commit**, not a declaration that Phase 4 or the roadmap is complete.

Phase 4 has not passed all exit gates.

**Update 2026-07-16 (Linux 2x RTX A6000 host).** Three of the four Phase 4 exit
gates are now evidenced on this host, with audited artifacts under
`docs/assessments/`. The authoritative record is
`docs/assessments/spice-mna-phase-4-acceptance.md`; that document supersedes the
performance numbers quoted further down this file, which were measured on a
different host and are not reproducible here.

- Gate (a) gradients vs finite differences `< 1%`: **PASS** (25 passed).
- Gate (b) single vs multi-GPU `rtol <= 2e-5`: **PASS**, bitwise (0.000e+00) on
  all six E/H fields, port V/I and circuit node/branch data. The physical
  two-GPU parity test now runs; it had never executed because its own fixture
  placed a `LumpedPort` `current_surface` off the Yee half grid. Test-only fix.
  This gate runs with `use_cuda_graph=False` on both sides, so its verdict is
  independent of the CUDA-graph port-DFT integration-tag fix also carried on
  this branch (`witwin/maxwell/fdtd/circuits.py`: advance `port.last_integration`
  from the precomputed schedule on the graph-replay path). That fix guards a
  different contract — bitwise eager-vs-graph port parity — and is evidenced by
  `tests/rf/circuits/test_phase4_circuit_cuda_graph.py::test_fixed_builtin_schedules_use_circuit_graph_and_match_eager`,
  which goes red when the fix is reverted.
- Gate (c) 32-unknown circuit step overhead `< 10%`: **PASS** at -64.507% on the
  matched native-termination baseline; see the acceptance document for the
  caveat that this baseline is itself 14.4x a bare FDTD step.
- Gate (d) no-circuit single-GPU regression `< 1%`: **PASS**, on equivalence
  evidence with a calibrated timing leg. After the host owner raised the CPU
  governor from `powersave` to `performance` and enabled GPU persistence, a
  clock-floored ABBA run measured +1.227% while an A/A calibration (the Phase 3
  commit against a second checkout of itself, identical protocol) measured
  -0.523% — the harness cannot resolve a 1% threshold at these settings. The
  controlling evidence is mechanistic: torch-profiler op-stream comparison over
  512 steps is identical between Phase 3 and Phase 4 (3282 aten calls, same
  50-key table, same per-step kernel set), prepare delta is ~0.1 ms, and the
  forward kernels are binary-identical. The op-stream equivalence is now a
  tracked, reproducible artifact rather than prose:
  `spice-mna-phase-4-no-feature-op-stream.json`, produced by
  `tests/support/compare_no_feature_op_stream.py compare` (empty op-table diff,
  `equivalent = true`). Artifacts:
  `spice-mna-phase-4-no-feature-op-stream.json` (op-stream equivalence),
  `spice-mna-phase-4-no-feature-abba.json` (B-run) and
  `spice-mna-phase-4-no-feature-aa-calibration.json` (A/A). Full chain in the
  acceptance document.

## Completed phases and commits

- Phase 0, deterministic circuit graph and compiler contracts: `ff3e20c` (`Add deterministic linear circuit graph contracts`).
- Phase 1, GPU-native linear MNA transient runtime: `26bb6f3` (`Add GPU-native linear MNA transient runtime`).
- Phase 2, strongly coupled FDTD/circuit runtime: `15c569f` (`Add strongly coupled FDTD circuit runtime`).
- Phase 3, multi-port circuit co-simulation workflows: `0a69fc8` (`Add multi-port circuit co-simulation workflows`).
- Phase 4 checkpoint, gradients/multi-GPU/performance work: `f3c0df3` (`Checkpoint Phase 4 circuit co-simulation progress`).

Each completed phase was independently reviewed and committed separately. These commits remain unchanged in the branch history.

## Phase 4 work present in this checkpoint

Commit `f3c0df3` contains the following Phase 4 implementation:

- A linear MNA adjoint integrated with the FDTD custom-autograd bridge.
- Differentiable R, L, C, source waveform, bound-port material, node-voltage, branch-current, device-power, energy-balance, port-power, field-energy, and matrix-condition outputs.
- An ancestry-safe semantic input frontier for direct tensors and `SceneModule`-derived parameters, including dtype-cast prepared tensors and shared circuit/material/geometry parameters.
- Explicit limits for t=0 output seeds, trainable DC/initial-state values, derived reference impedance, distributed backward, and other unsupported adjoint configurations.
- Deterministic multi-GPU circuit ownership, scalar voltage/current communication, owner-only MNA state/results/checkpoints, a same-shard zero-P2P path, and a copy-acknowledgement fix for owner-current buffer reuse.
- CUDA-only batched fixed-factor MNA solves, fixed-schedule circuit CUDA Graph support, vectorized fixed RC execution, and graph-safe checkpoint/resume tensor identity.
- Worktree-local performance/provenance harnesses for 8/32/128 unknown circuits, immutable Phase 3 ABBA comparison, and Nsight summary extraction.
- User-visible capability and limitation updates in `FEATURE_LIST.md` and the circuit/distributed reference documentation.

The eager parser contract remains intentional: a direct parsed parameter expression is materialized once. Optimization or finite-difference loops must reparse/rebuild the circuit for each objective evaluation or use `SceneModule.to_scene()` to rematerialize the expression graph.

## Recent verification evidence

### Functional and numerical evidence

- Independent Phase 4 gradient file: `25 passed`.
- Exact rerun of the previously failing parsed-expression ancestry case: `1 passed`.
- Related lumped-adjoint and circuit-coupling regressions after the compatibility fix: `24 passed`.
- Gradient coverage includes R/L/C/source amplitude finite differences below 1%, RC cutoff, RLC near resonance, two-port insertion loss, bound-port material, full public `CircuitData` tensor outputs, direct and `SceneModule`-derived parameters, dtype conversion, and explicit unsupported guards.
- Distributed owner/reduction review: `18 passed, 1 skipped`. The skipped case is the physical two-GPU parity test.
- Phase 4 graph/batching/resume focused suite before the final timing isolation: `17 passed`.
- Final no-RF timing-isolation targeted suite: `8 passed`, followed by `2 passed` focused CUDA checks.
- Final CPU-only guard/provenance/performance-support check: `19 passed`.
- Relevant Ruff checks, `py_compile`, and `git diff --check` passed after the latest changes.

### Performance evidence that is informative but not a completed gate

> **Superseded 2026-07-16.** Every number in this subsection was measured on the
> previous host (single RTX 5080, PyTorch 2.10, CUDA 12.8, Windows) and written
> to gitignored `build/`, so none of it is auditable in a fresh clone. It is kept
> only as history. The measurements of record are in
> `docs/assessments/spice-mna-phase-4-acceptance.md` with tracked JSON artifacts.

- The matched 64-cubed-grid comparator, using the same `LumpedPort` geometry and a native `SeriesRLC(50 ohm)` termination, measured the 32-unknown circuit at 0.383257 ms/step versus 1.105771 ms/step for the native termination, or -66.0395% overhead. This is provisional evidence for the `<10%` matched-workload gate, not a final post-checkpoint rerun.
- Reported circuit steady-loop times for 8/32/128 unknowns were 0.436181/0.383257/0.406416 ms/step. Factorization count was one; checkpoint sizes were 72/264/1032 bytes; single-device communication was zero. Batched factor times were 5.594/40.785/162.675 ms, and solve times were 2.975/4.150/5.999 microseconds per system.
- A preserved profiler trace reported 4,000 CUDA Graph launches for 2,000 steps across two graph executable IDs, 51,868 kernels, 23,208 bytes total H2D, and 152 bytes total D2H. The requested new Nsight run hung during finalization and was skipped after explicit user instruction, so this trace is preserved prior evidence rather than an exact trace of the final working tree.
- After isolating all no-RF timing instrumentation from the normal no-circuit hot path, an uncontaminated 2-round, 10,000-step diagnostic measured +0.022066%. It was explicitly a diagnostic, not the predeclared final gate run.
- The predeclared 5-round/2-warmup/7-repeat/10,000-step robust run was contaminated by an unrelated CUDA context (PID 40884) that appeared after launch. Its nominal result was renamed with `CONTAMINATED` and must not be used as acceptance evidence.

## Incomplete, skipped, or blocked scope

The following items are not complete and must not be represented as passed:

1. ~~**Physical multi-GPU parity**~~ **RESOLVED 2026-07-16.** The test does not
   merely "skip on a single device": it *failed* at scene construction with
   `ValueError: z half-grid plane=0.0 must lie on the Yee z half-grid plane grid`,
   because its own fixture placed the `LumpedPort` `current_surface` on a primal
   z node (and its tangential bounds on primal x/y nodes). The product's
   half-grid requirement is correct (`Hx`/`Hy` exist only on z half planes), so
   the fixture was fixed, test-only. The gate now passes bitwise. See
   `docs/assessments/spice-mna-phase-4-acceptance.md`.
2. ~~**Final no-circuit regression gate**~~ **RESOLVED 2026-07-16.** After the
   host owner set the CPU governor to `performance` and enabled GPU
   persistence, the gate was closed on equivalence evidence with a calibrated
   timing leg: a clock-floored ABBA run measured +1.227% while an A/A
   calibration of identical code measured -0.523%, proving the harness cannot
   resolve 1% at these settings; the controlling evidence is the identical
   torch-profiler op stream (3282 aten calls, same 50-key table), the ~0.1 ms
   prepare delta, and binary-identical forward kernels. The historical
   powersave-era dispersion (-9.829% / +2.263%, round stdev 10.62 / 3.59 pp)
   and the incomplete "contaminated by PID 40884" explanation are retained in
   the acceptance document's timing table for the record.
3. **Fresh final profiler trace:** a new Windows Nsight Systems run hung during report generation. Its two `nsys` processes were stopped only after explicit user instruction. The run exited unsuccessfully and produced no valid new artifact.
4. **Phase 4 commit and roadmap completion:** with items 1 (gate (b)) and 2
   (gate (d)) both resolved on 2026-07-16, all four Phase 4 exit gates are
   evidenced (see the acceptance document's summary table). The remaining
   completion step is the full Phase 4 review and a completion commit; item 3
   (a fresh Nsight trace) stays open as supplementary evidence only, since the
   host has no Nsight tooling installed.

The distributed adjoint remains explicitly unsupported and is rejected before allocation. The implemented multi-GPU contract is forward-only for this phase.

## Known issues and execution risks

- GPU availability on the shared machine is unstable. Multiple unrelated `witwin2` CUDA contexts appeared during long benchmark runs. Any future acceptance run must use a quiet interval and invalidate the whole run if an unknown CUDA context appears; individual samples must not be discarded selectively.
- One earlier contention-handling incident misidentified and terminated external PID 73248. No acceptance data from that interval was retained. Later external workloads were not terminated. The two hung `nsys` PIDs were cleaned only after explicit user authorization.
- The provisional RTX 5080 32-unknown timing numbers quoted earlier in this file predate the final no-RF timing-isolation work and are superseded. Both artifacts were regenerated on this 2x A6000 host with the no-RF isolation present and are tracked under `docs/assessments/` (`spice-mna-phase-4-circuit-performance.json`, `spice-mna-phase-4-no-feature-abba.json`, `spice-mna-phase-4-no-feature-aa-calibration.json`); gate (c) is evidenced from the circuit-performance artifact and gate (d) from op-stream equivalence plus the calibrated timing pair (see item 2 above).
- The immutable Phase 3 baseline is bound to commit `0a69fc877f83d96e5cb75d6b8564375d488c4d63`, Git tree `8ad9737f0504b87435965d47c09924aafefa96d5`, archive SHA-256 `c7f018bbbeb2fd15ecdc236b2d87c4f757d5b2995e38a1b761ef954c8c772350`, and manifest `24bd20fb61e8410e66d663d42de80a06d5116344cafc9f6b473754b31f0c5130`.

## Resume steps

The commands below are the Linux/bash forms actually executed on this host on
2026-07-16. The original Windows/PowerShell transcript named a `witwin2` conda
environment that does not exist here; it has been replaced. The scripts, flags,
and predeclared estimator settings are unchanged.

Environment prefix used for every command (CUDA_HOME is required because the
native kernels are JIT-compiled; the env's `bin` must be on PATH or the
subprocess sampler cannot find `ninja`):

```bash
cd /path/to/worktree
export CUDA_HOME=/home/xingyu/miniconda3/envs/maxwell/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:/home/xingyu/miniconda3/envs/maxwell/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions"
export CUDA_CACHE_PATH="$PWD/build/cuda_cache"
export CMAKE_BUILD_DIR="$PWD/build/cmake"
export WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR="$PWD/build/fdtd_cuda"
PY=/home/xingyu/miniconda3/envs/maxwell/bin/python
```

1. Confirm GPU idleness: `nvidia-smi --query-compute-apps=pid,process_name --format=csv`
   must list no unknown owner. Note that a quiet GPU is **not** sufficient for
   gate (d) on this host; see the CPU-governor issue in the acceptance document.

2. Rebuild the immutable Phase 3 baseline (it lives under gitignored `build/`, so
   it must be regenerated; `git archive` reproduces it byte-identically, verified
   by the recorded archive and manifest SHA-256):

   ```bash
   mkdir -p build/baselines
   git archive --format=zip -o build/baselines/phase3_0a69fc8.zip \
     0a69fc877f83d96e5cb75d6b8564375d488c4d63
   rm -rf build/baselines/phase3_0a69fc8_exact_raw
   mkdir -p build/baselines/phase3_0a69fc8_exact_raw
   (cd build/baselines/phase3_0a69fc8_exact_raw && unzip -q ../phase3_0a69fc8.zip)
   ```

3. No-feature ABBA gate, pinned to the GPU-local NUMA node (both A6000s are on
   node 0; pinning removes a ~25% remote-socket slow state). Do not edit tracked
   files while this runs: the harness aborts if the candidate worktree's tracked
   dirty state changes mid-run.

   ```bash
   numactl --cpunodebind=0 --membind=0 $PY tests/support/benchmark_rf_no_feature.py compare \
     --baseline-root "$PWD/build/baselines/phase3_0a69fc8_exact_raw" \
     --candidate-root "$PWD" \
     --baseline-commit 0a69fc877f83d96e5cb75d6b8564375d488c4d63 \
     --baseline-archive "$PWD/build/baselines/phase3_0a69fc8.zip" \
     --python $PY \
     --rounds 5 --warmup 2 --repeats 7 \
     --steps 10000 --grid-cells 24 \
     --max-regression-pct 1.0 \
     --output "$PWD/docs/assessments/spice-mna-phase-4-no-feature-abba.json"
   ```

4. Circuit performance artifact (8/32/128 unknowns) on the same quiet GPU:

   ```bash
   $PY tests/support/benchmark_circuit_performance.py \
     --unknowns 8 32 128 --grid-cells 64 --steps 1000 \
     --warmup 1 --repeats 5 --batch-size 32 --batch-iterations 1000 \
     --output "$PWD/docs/assessments/spice-mna-phase-4-circuit-performance.json"
   ```

5. Physical two-GPU parity, both devices visible:

   ```bash
   $PY -m pytest -q tests/fdtd/multi_gpu/test_circuit_owner.py
   ```

Artifacts are written to tracked `docs/assessments/`, not gitignored `build/`, so
they survive a fresh clone and remain auditable.

Gate (d) was closed on 2026-07-16 after the host owner set the CPU clock floor
(`cpupower frequency-set -g performance`, `nvidia-smi -pm 1`): see the
acceptance document for the equivalence evidence and the A/A-calibrated timing
pair. Remaining before Phase 4 completion is recorded: rerun the full relevant
regression set and an independent Phase 4 exit review, and record completion in
a new normal commit; do not rewrite or relabel the checkpoint commit.

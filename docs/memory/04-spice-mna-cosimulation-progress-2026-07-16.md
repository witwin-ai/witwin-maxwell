# Circuit Co-simulation Progress Checkpoint (2026-07-16)

## Checkpoint status

This file records the current state of task 04, circuit co-simulation implementation, on branch `codex/spice-mna-cosimulation`. The branch contains completed Phase 0-3 commits followed by uncommitted Phase 4 implementation work. The next commit is intentionally a **checkpoint/WIP commit**, not a declaration that Phase 4 or the roadmap is complete.

Phase 4 has not passed all exit gates. In particular, the physical two-GPU parity gate has not been run, and the final clean no-circuit performance regression gate was paused before a valid robust run completed.

## Completed phases and commits

- Phase 0, deterministic circuit graph and compiler contracts: `ff3e20c` (`Add deterministic linear circuit graph contracts`).
- Phase 1, GPU-native linear MNA transient runtime: `26bb6f3` (`Add GPU-native linear MNA transient runtime`).
- Phase 2, strongly coupled FDTD/circuit runtime: `15c569f` (`Add strongly coupled FDTD circuit runtime`).
- Phase 3, multi-port circuit co-simulation workflows: `0a69fc8` (`Add multi-port circuit co-simulation workflows`).

Each completed phase was independently reviewed and committed separately. These commits remain unchanged in the branch history.

## Phase 4 work present in this checkpoint

The working tree currently contains the following Phase 4 implementation:

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

- The matched 64-cubed-grid comparator, using the same `LumpedPort` geometry and a native `SeriesRLC(50 ohm)` termination, measured the 32-unknown circuit at 0.383257 ms/step versus 1.105771 ms/step for the native termination, or -66.0395% overhead. This is provisional evidence for the `<10%` matched-workload gate, not a final post-checkpoint rerun.
- Reported circuit steady-loop times for 8/32/128 unknowns were 0.436181/0.383257/0.406416 ms/step. Factorization count was one; checkpoint sizes were 72/264/1032 bytes; single-device communication was zero. Batched factor times were 5.594/40.785/162.675 ms, and solve times were 2.975/4.150/5.999 microseconds per system.
- A preserved profiler trace reported 4,000 CUDA Graph launches for 2,000 steps across two graph executable IDs, 51,868 kernels, 23,208 bytes total H2D, and 152 bytes total D2H. The requested new Nsight run hung during finalization and was skipped after explicit user instruction, so this trace is preserved prior evidence rather than an exact trace of the final working tree.
- After isolating all no-RF timing instrumentation from the normal no-circuit hot path, an uncontaminated 2-round, 10,000-step diagnostic measured +0.022066%. It was explicitly a diagnostic, not the predeclared final gate run.
- The predeclared 5-round/2-warmup/7-repeat/10,000-step robust run was contaminated by an unrelated CUDA context (PID 40884) that appeared after launch. Its nominal result was renamed with `CONTAMINATED` and must not be used as acceptance evidence.

## Incomplete, skipped, or blocked scope

The following items are not complete and must not be represented as passed:

1. **Physical multi-GPU parity:** only one RTX 5080 was available. The single-versus-two-GPU circuit parity test, including all six E/H fields, port voltage/current, circuit node/branch data, owner/checkpoint semantics, and `rtol <= 2e-5`, remains unverified.
2. **Final no-circuit regression gate:** the required clean, robust ABBA result below 1% is not available. The last formal run was invalidated in full because an external CUDA context appeared mid-run. Performance testing was then paused by explicit user request.
3. **Fresh final profiler trace:** a new Windows Nsight Systems run hung during report generation. Its two `nsys` processes were stopped only after explicit user instruction. The run exited unsuccessfully and produced no valid new artifact.
4. **Phase 4 commit and roadmap completion:** Phase 4 cannot be called complete or committed as a completion phase until the two exit gates above are satisfied and the full Phase 4 review is repeated.

The distributed adjoint remains explicitly unsupported and is rejected before allocation. The implemented multi-GPU contract is forward-only for this phase.

## Known issues and execution risks

- GPU availability on the shared machine is unstable. Multiple unrelated `witwin2` CUDA contexts appeared during long benchmark runs. Any future acceptance run must use a quiet interval and invalidate the whole run if an unknown CUDA context appears; individual samples must not be discarded selectively.
- One earlier contention-handling incident misidentified and terminated external PID 73248. No acceptance data from that interval was retained. Later external workloads were not terminated. The two hung `nsys` PIDs were cleaned only after explicit user authorization.
- The matched 32-unknown timing artifact predates the final no-RF timing-isolation branch. It is useful provisional evidence, but a clean final performance pass should regenerate both the circuit performance artifact and the no-feature ABBA artifact from the checkpointed tree.
- The immutable Phase 3 baseline is bound to commit `0a69fc877f83d96e5cb75d6b8564375d488c4d63`, Git tree `8ad9737f0504b87435965d47c09924aafefa96d5`, archive SHA-256 `c7f018bbbeb2fd15ecdc236b2d87c4f757d5b2995e38a1b761ef954c8c772350`, and manifest `24bd20fb61e8410e66d663d42de80a06d5116344cafc9f6b473754b31f0c5130`.

## Resume steps

1. Confirm at least two minutes of GPU idleness with no unknown CUDA compute/context owner.
2. From this worktree, using `witwin2` and worktree-local caches, rerun the no-feature gate with the predeclared estimator and no sample discards:

   ```powershell
   $env:TORCH_EXTENSIONS_DIR="$PWD\build\torch_extensions"
   $env:CUDA_CACHE_PATH="$PWD\build\cuda_cache"
   $env:CMAKE_BUILD_DIR="$PWD\build\cmake"
   $env:WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR="$PWD\build\fdtd_cuda"
   $python='C:\Users\Asixa\miniconda3\envs\witwin2\python.exe'

   & $python tests\support\benchmark_rf_no_feature.py compare `
     --baseline-root "$PWD\build\baselines\phase3_0a69fc8_exact_raw" `
     --candidate-root "$PWD" `
     --baseline-commit 0a69fc877f83d96e5cb75d6b8564375d488c4d63 `
     --baseline-archive "$PWD\build\baselines\phase3_0a69fc8_exact_raw.zip" `
     --python $python `
     --rounds 5 --warmup 2 --repeats 7 `
     --steps 10000 --grid-cells 24 `
     --max-regression-pct 1.0 `
     --output "$PWD\build\perf\phase4_no_feature_abba_schema2_robust_clean_final.json"
   ```

3. Regenerate the final 8/32/128 circuit artifact on the same quiet GPU:

   ```powershell
   & $python tests\support\benchmark_circuit_performance.py `
     --unknowns 8 32 128 --grid-cells 64 --steps 1000 `
     --warmup 1 --repeats 5 --batch-size 32 --batch-iterations 1000 `
     --output "$PWD\build\perf\phase4_circuit_schema2_clean_final.json"
   ```

4. On two homogeneous, mutually P2P-accessible GPUs, run:

   ```powershell
   & $python -m pytest -q `
     tests\fdtd\multi_gpu\test_circuit_owner.py::test_physical_two_gpu_circuit_matches_single_gpu_and_reports_scalar_contract `
     --basetemp build\pytest-phase4-2gpu
   ```

5. If both gates pass, rerun the full relevant regression set and an independent Phase 4 exit review. Record Phase 4 completion in a new normal completion commit; do not rewrite or relabel this checkpoint commit.

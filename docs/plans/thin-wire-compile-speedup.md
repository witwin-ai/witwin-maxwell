# P-11: thin-wire compile speedup (torch level)

Status: spec accepted by the coordinator, 2026-10-31. Branch `perf/thin-wire-compile`, worktree
`<workspace>/wt/maxwell-p11-thinwire`, based on master d26351f3.

## Why

The LQ-3 smoke timing shows F4 (wire antenna) tasks taking 147 s each, against 59 s for F1. An uncached
thin-wire compile dominates the difference. The hot-path audit (item 7,
`docs/assessments/cuda-torch-hot-path-audit-2026-10-30.md`) finds Python loops, 44 host-sync sites
and no cache: 2–20 s per cold compile. On top of that, one run compiles the same network up to
three times: `compiler/ports.py:404`, `fdtd/wire.py:824` and `compile_wire_monitors` (thin_wire.py
around line 2159), plus the adjoint bridge.

The target is to cut the F4 LQ-3 task time and F4 data generation by more than 2×.

## Scope

Torch-level work in `witwin/maxwell/compiler/thin_wire.py`, plus the call sites that need the cache.
No CUDA migration and no change to the wire physics. The FDTD wire runtime (`fdtd/wire.py`,
`fdtd/cuda/kernels/wire.cu`) consumes the same compiled tensors.

1. **Vectorize.**
   - Replace the per-point and per-fragment Python loops with batched torch ops, over all fragments
     of a wire and then over all wires: `_expanded_path`, `_fragment_cell`,
     `_fragment_edge_coupling` (the component × transverse-bit loop) and `_point_edge_weights`.
   - Replace the numpy snap/argmin on grid nodes with a batched `torch.searchsorted` on the compile
     device.
   - Keep the float64 geometry arithmetic. Values must match bitwise (see gate a), so the op order
     of each reduction must reproduce the current one. Where a batched reduction cannot be ordered
     the same way, stop and report the max difference instead of accepting it.
2. **Remove host syncs.**
   - Remove every `.item()`, `float(tensor)`, `int(tensor)` and `bool(torch.all(...))` sync on the
     compile path, including the `abs(float(weight.detach().cpu()))` pruning near line 576. Pruning
     becomes a device mask.
   - Validation that must raise does one batched check per wire set (one sync), with the same error
     messages.
   - Measure the syncs with `torch.cuda.set_sync_debug_mode("warn")` around one F4 compile, and
     report the count before and after.
3. **Cache.**
   - Keep one compiled result (`CompiledWireNetwork`, and the monitor compile that depends on it) per
     PreparedScene and device. Every call site above reuses it.
   - The key is the existing `_grid_fingerprint` plus a fingerprint of every wire input: points,
     radius, conductor, ends, bindings and any parameter that enters the compile.
   - The cache is bypassed whenever any wire input requires grad, so the gradient path stays
     uncached and unchanged.
   - Frequency-dependent wire impedance (`compiler/wire_impedance.py`) stays per frequency. Only the
     geometry and coupling network is shared across a scene's frequencies.

## Gates (fast tier)

- **(a) Bitwise equality.**
  - Every tensor field of `CompiledWireNetwork` and `CompiledWireMonitor` is `torch.equal` to master
    (69725b22). Check this on the F4 recipe scenes and on every scene the thin-wire tests build.
  - F4 recipe scenes: the LQ-3 generator `benchmark/label_qualification/random_convergence.py`
    (`draw_scenes` / `build_sample`, family F4, kinds dipole, loop and helix, lossy and lossless),
    at least 6 scenes.
  - Implement this as a script that compiles each scene on this branch and compares against
    tensors dumped from a master checkout. Use a detached master worktree at
    `<workspace>/wt/maxwell-p11-master` (`git worktree add --detach`) to dump the
    master tensors. Never edit that worktree.
  - Cache invalidation (coordinator addition): on one PreparedScene, compile, then change a non-grad
    wire input (a radius, or one vertex), then compile again. Assert the second compile misses the
    cache, recompiles, and equals a fresh compile of the changed scene. A stale network must never
    be reused.
  - Also assert that a grad-requiring input bypasses the cache.
- **(b) Suites pass unchanged**, each file in its own GPU process:
  - `tests/fdtd/thin_wire/*` (8 files);
  - `tests/api/public/test_thin_wire_api.py`;
  - `tests/fdtd/cuda/test_cuda_wire_parity.py`;
  - `tests/gradients/test_fdtd_thin_wire_adjoint.py`, `test_fdtd_thin_wire_port_adjoint.py`,
    `test_fdtd_thin_wire_conductivity_adjoint.py`;
  - `tests/fdtd/multi_gpu/test_wire_owner.py`, only the parts that run on one GPU.
  - Run the same selections on master too, to record any pre-existing failure.
- **(c) Timing.**
  - Prepare and compile wall time, cold (first compile in the process) and warm (second prepare of
    the same scene), on three F4 scenes: one dipole, one loop and one helix. Report master vs
    branch.
  - Also report the number of compiles per `Simulation.run()`, before and after.
  - Report one full F4 LQ-3 task time if it fits in 10 minutes.

## Rules

- witwin-maxwell's `CLAUDE.md`/`AGENTS.md` apply. Write English comments. No compatibility shims.
- No defensive programming. Minimal tests: one cache-invalidation test in
  `tests/fdtd/thin_wire/test_thin_wire_compiler.py`. The bitwise and timing scripts go under
  `docs/assessments/thin-wire-compile-speedup-2026-11/` with their JSON output.
- One command per shell call, no `&&`.
- **GPU:** GPU 0 only.
  - Use exactly `env CUDA_VISIBLE_DEVICES=0 python <scratch> --log <file> <pytest args>`,
    or `... p11_gpu.py --log <file> --script <path> <args>` for scripts.
  - No other `VAR=`. The runner sets CUDA_HOME/PATH, puts this worktree first on sys.path, and
    stops any process above 6 GB.
  - Run `nvidia-smi` before each GPU command, and wait if less than 12 GB is free.
  - One GPU process at a time. FDTD jobs from other lines have priority.
  - The master-dump runs use the same runner, pointed at the master worktree: copy the runner and
    change WORKTREE in the copy.
- Never use GPU 1. No cloud.
- Commit as the repository owner. No AI
  attribution. Keep commercial product names out. No push or merge.
- Update `FEATURE_LIST.md` only if a user-visible behaviour changes. A cache is not user-visible
  unless it changes an API.
- Report:
  - files changed;
  - sync counts and compiles per run, before and after;
  - the bitwise table;
  - the timing table;
  - suite results (branch and master);
  - per-process GPU peaks;
  - anything that could not be made bitwise.

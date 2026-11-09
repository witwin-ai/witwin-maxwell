# Time-modulated media: accurate modulation clock and phase

Status: implemented and fast-tier validated (2026-11-01). Report:
`<scratch>/witwin-modclock/REPORT.md`. Branch `fix/modulation-clock`, worktree
`<workspace>/wt/maxwell-modclock`, base master 65de004b.
Evidence: `<scratch>/witwin-modgrowth/REPORT.md` ("Separate clock defect"), `clock_check.json` and
`clock_measured.json` in the same directory.

## Defect

The modulation device clock is a 2-element FP32 tensor, `solver._modulation_time = [t_prev, t_next]`,
initialized to `[-dt, -dt]` in `fdtd/runtime/materials.py::build_modulation`. It is advanced once
per step by `advance_modulation_time_kernel` in `fdtd/cuda/kernels/electric.cu` (L237–239):
`t_prev = t_next; t_next += dt`. This is a repeated FP32 addition. The phases are evaluated as
`sincosf(omega * t)` with an FP32 argument in `modulation_phase_from_omega` (L230–234).

Over the 7.39918 µs horizon the clock reaches 7.42756 µs, an error of +28.39 ns. At 0.35 GHz that is
+62.44 rad of modulation phase. At dt/2 and dt/4 the errors are −48.73 ns and −90.02 ns. The
saved device clock matches the predicted FP32 sequence exactly. Even with an exact clock, an FP32
`omega*t` argument of order 10³–10⁴ rad loses about 1e-4 to 1e-3 rad to rounding before
`sincosf`.

## Required behaviour

- **Exact clock.** The time used for step n is exactly the intended `t_n` of the current convention:
  the same first-step labelling, starting at −dt → 0 as today, to FP64 accuracy. Use a device
  integer step counter with `t = (n + offset) * dt` formed in FP64, or an equivalently exact FP64
  accumulator. Choose one and justify it. There must be no host sync or per-step host value:
  CUDA-graph capture must keep working with the clock as device state.
- **Accurate phase.** Form `omega*t` in FP64 and range-reduce it modulo 2π in FP64 before an FP32
  `sincosf`. Support per-edge omega maps (`mod_omega_*`), multiple modulation frequencies, and both
  scalar types the kernels are templated on.
- **All consumers.** Update every reader of `_modulation_time`:
  - `fdtd/runtime/stepping.py` (every ModulationTime argument, and the advance in
    `_field_update_block`);
  - `fdtd/runtime/materials.py` (around L891);
  - `fdtd/distributed/shard_engine.py` (around L329);
  - `fdtd/cuda/backend.py` `_advance_modulation_time` and the extension binding;
  - the modulated electric kernels, including `dispersive.cu::apply_polarization_modulated_kernel`;
  - the adjoint/replay path, if it consumes the clock.
  Grep the whole tree, list every reader, and say which were changed.
- **Checkpoint and resume.** `_modulation_time` is persisted (`fdtd/checkpoint.py` L601 graph state;
  `fdtd/resume.py` L539 and L770). If its dtype or shape changes, or a counter is added:
  - follow the resume-schema rule: an old checkpoint must load with an EXACT default, meaning the
    counter or the FP64 time reconstructed exactly from the old state where possible, otherwise a
    documented exact derivation from the saved step index;
  - bump `_CHECKPOINT_SCHEMA_VERSION` only if the repository's rule requires it;
  - test loading a checkpoint written by base master.
- **No changes to the physics.** Do not change the D/E update or the CPML. Unmodulated paths must stay
  bitwise unchanged: assert `torch.equal` of fields after N steps on an unmodulated scene, branch vs
  base.

## Gates (fast tier)

1. **Phase accuracy.** Over the full 7.39918 µs horizon at dt, dt/2 and dt/4, and for each modulation
   frequency in the benchmark scenes (0.2 and 0.35 GHz), plus one multi-frequency map: the device
   clock equals the exact `t_n` (FP64 comparison), and the evaluated phase error is < 1e-3 rad
   at every step sampled. Compare against an FP64 host reference computed independently.
2. **Graph, eager and resume consistency.** The same modulated scene run under CUDA graph and
   eager execution gives the same clock and fields, and resuming from a mid-run checkpoint gives the
   same final state as an uninterrupted run. Use torch.equal where the existing tests do.
3. **Old checkpoint.** A base-master checkpoint of a modulated run loads, and its continued run
   matches the new code's own continuation from the exact reconstructed clock, to the tolerance the
   resume tests already use.
4. **First-step convention.** For the first two steps, the m(t) values used match base master's
   intended `t_prev`/`t_next` labels exactly.
5. **Suites,** each file in its own GPU process, process exit code recorded, branch AND base:
   - every test that uses modulation (reader sweep with grep over `tests/`);
   - `tests/materials/combinations/test_modulated_slab_cpml.py`;
   - the checkpoint/resume and CUDA-graph coverage tests that include modulation;
   - `tests/api/public/test_guard_census.py` (CPU), reconciling any guard that moves in its own
     commit.
   - One new test file for gates 1 and 4 at small size. It needs an owner in
     `scripts/dev/integration_suites.toml` if its directory has none.
6. **Report:** the benchmark modulated_slab and modulated_slab_phase metrics before and after,
   against the REGENERATED caches in `benchmark/cache/`
   (read-only). Expect small changes at the bounded 85 ns horizon. It is information, not a gate.

## Rules

- witwin-maxwell `CLAUDE.md`/`AGENTS.md` apply. English comments. No shims. No commercial product
  names. FEATURE_LIST only if a user-visible behaviour is documented there.
- No defensive programming. Small commits. One command per shell call, no `&&`.
- **GPU:** only through
  `env CUDA_VISIBLE_DEVICES=1 python <scratch> --log <scratch>/witwin-modclock/<name>.log (--pytest <args> | --script <path> <args> | --module benchmark <args>)`.
  - The runner chooses the GPU: GPU 1 now, GPU 0 after the batch. It enforces the lane lock, the
    free-memory rule and the per-process memory cap.
  - Keep `CUDA_VISIBLE_DEVICES=1` on the command line regardless.
  - No other `VAR=`. Respect the STOP file.
  - Make no timing claims.
- No cloud. Never modify caches. Outputs go to `<scratch>/witwin-modclock/`.
- Commit as the repository owner. No AI
  attribution. No push or merge.
- **Report:** files changed; the design; the reader list; gate tables; checkpoint handling; suite exit
  codes for branch and base; guard census.

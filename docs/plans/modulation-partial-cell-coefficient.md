# Time-modulated media: absolute partial-cell modulation on Yee edges

Status: fix accepted by the coordinator (class-(c) triage, 2026-11-01). Branch `fix/modulation-partial-cell`,
worktree `<workspace>/wt/maxwell-mod-fix`, base master a3179770.
Diagnosis: `<scratch>` (copy; originals,
scripts and the CPU harmonic model are in `<scratch>/witwin-classc-modslab/`). Read it first.

## Defect

The public contract is `eps(t) = eps_static [1 + a cos(Omega t + phi)]` inside the modulated material.
At an E edge whose control volume is only partly inside that material, the physically correct
time-varying permittivity is the effective-medium average of the actual local `eps(x, t)`:
- tangential components use the arithmetic average;
- normal components use the harmonic average, or the existing per-edge tensor rule, consistent with
  how static eps is averaged on that edge.

Today `fdtd/runtime/materials.py::build_modulation` averages the RELATIVE quadratures
`a cos(phi)` and `a sin(phi)` onto the edges with `average_node_to_component`. The kernels then
multiply the separately averaged static edge eps by `m(t) = 1 + cos_field cos(Omega t) - sin_field
sin(Omega t)`. On a boundary edge this understates the absolute modulation Δε(t). The diagnosis
measured 35.16% / 31.25% on the two benchmark slabs, where the relative depth averages over the
vacuum part too. Sideband generation is very sensitive to this at coarse grids.

## Required behaviour

- Compile the modulation so that, on every E edge:
  - the runtime's `eps_edge * m_edge(t)` equals the edge's effective time-varying permittivity to
    first order in the modulation depth;
  - equivalently, `m_edge - 1` is the averaged ABSOLUTE modulation quadrature divided by that edge's
    averaged static eps.
- Use the same averaging rule (and subpixel/occupancy weights) as the static eps of that component.
  Do not invent a different interface rule.
- Interior edges (fully inside one material) must stay bitwise unchanged.
- Keep the existing old/new-E D update, the device clock, and the static-field-only kernel inputs:
  no per-step rebuild, no host sync.
- If the static edge averaging is harmonic (normal components), state exactly what "absolute
  modulation" means there (the first-order expansion of the harmonic mean of eps(x, t)), and
  implement that.
- Gradients: if modulation parameters or eps are differentiable today, they must stay
  differentiable. Run the existing adjoint/gradient tests that touch modulation.
- No fitted scalars. No new public API.

## Gates (fast tier)

1. **Direct constitutive check (CPU or GPU, small).** For a slab occupying fractions f ∈ {0.25, 0.5,
   0.75} of a boundary cell, the compiled edge `eps_edge * m_edge(t)` at four times equals the
   independently computed effective average of `eps(x, t)`. Check each component and both the
   tangential and normal orientations. Tolerance: first-order exact, residual O(a²).
2. **Thin, weakly modulated slab Born check.** Using the diagnosis's first-order Born / continuum
   arbiter (`<scratch>/witwin-classc-modslab/`): sideband amplitudes and phases at ω ± Ω from the runtime
   agree with the arbiter, and the agreement improves over the base code. Include a non-zero
   modulation phase so φ rotation is checked.
3. **Wave-action / power balance** across carrier and sidebands, with enough harmonics.
4. **Grid convergence:** at grids ×1/×2 (×4 if it fits), the sideband error against the continuum
   arbiter decreases.
5. **Benchmark modulated_slab,** against the read-only cache
   `benchmark/cache/media/modulated_slab.h5`. Report L2 / shape
   / Linf / corr per frequency, before and after. The diagnosis's CPU model predicts lower/upper L2
   0.333/0.483 → 0.230/0.333. This is not a pass gate, because the reference uses an unsmoothed
   active interface. modulated_slab_phase is quarantined: report it but ignore it.
6. **Suites,** each file in its own GPU process with the process exit code, on branch AND base master:
   - `tests/materials/combinations/test_modulated_slab_cpml.py`;
   - every test file that uses `ModulationSpec` / modulation (reader sweep over `tests/` with grep;
     list them; the ones found so far include test_combination_matrix, test_cuda_graph_capture_coverage,
     test_time_step_stability, test_fdtd_adjoint_bridge and test_tidy3d_adapter);
   - `tests/api/public/test_guard_census.py` (CPU).
   - Add one new regression test with gates 1–2 at small size. It needs an owner in
     `scripts/dev/integration_suites.toml` if its directory has none.
   - Bitwise check: non-modulated scenes and interior-only modulated edges are unchanged. Give
     evidence as a torch.equal of the compiled edge fields on a fully interior case.

## Rules

- witwin-maxwell `CLAUDE.md`/`AGENTS.md` apply.
  - English comments.
  - No shims.
  - No commercial product names in code, comments, commits or branch.
  - Update `FEATURE_LIST.md` if its modulation entry describes the interface behaviour.
- No defensive programming. Small commits. One command per shell call, no `&&`.
- **GPU:** GPU 1 only, through exactly
  `env CUDA_VISIBLE_DEVICES=1 python <scratch> --log <scratch>/witwin-mod-fix/<name>.log (--pytest <args> | --script <path> <args> | --module benchmark <args>)`
  - The runner serializes on one lane lock with the other fix task, waits for ≥26 GiB free, and
    kills a process above 2.6 GiB. The batch has priority.
  - No other `VAR=`.
  - If the STOP file `<scratch>` exists, start no GPU run.
- Never use GPU 0. No cloud. Never modify benchmark caches.
- Outputs go to `<scratch>/witwin-mod-fix/`.
- Commit as the repository owner. No AI
  attribution. No push or merge.
- **Report:**
  - files changed;
  - the exact averaging rule implemented per component;
  - gate tables;
  - the benchmark before/after;
  - suite exit codes on branch and master;
  - the reader sweep;
  - guard census;
  - GPU peaks.

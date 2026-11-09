# Current generalized integral solver checkpoint

Updated 2026-10-12. Recent code and documentation are consolidated from Maxwell
`1e6647859da71d30e4864be3417aa256da8567cf` and Bench
`54bb324bbb51fc60799f2c1501fb0d96bcd2f589`. The primary checkouts previously
stopped at `d054149` and `0284ea8`; 411 Maxwell and 75 Bench implementation
commits connect those baselines to the reviewed tips.

This is an implementation and integration checkpoint. Generalized physical,
whole-resource, asymptotic-complexity and final acceptance remain OPEN.

## Implemented paths

| Area | Current implementation | Acceptance boundary |
| --- | --- | --- |
| Public/lifted state | Scene -> Simulation -> Result, analytic carriers, scattered RWG J/M, immutable source identity, exact restore | Independent resonant/physical accuracy remains open |
| Operators and bodies | Shared forward/Hermitian/Riesz/Krylov protocol; coupled disjoint closed bodies and region-owned fields | No nested/open geometry or geometry gradients |
| Scalar magnetic media | Conductivity/dispersion sampling, passive outgoing branches, negative/near-zero guards | General physical sweeps and reference qualification remain open |
| Tensor and volume formulations | Configured dyadic kernels/carriers; linear surface-volume and DC/harmonic workflows | Explicit admission restrictions; full tensor/nonlinear physical qualification remains open |
| Periodic kernels | Versioned scalar/tensor 1D/2D/3D quasiperiodic compositions and dimensional outputs | Pole/sector/backend restrictions and independent periodic qualification remain open |
| Hierarchical/fast actions | H/H2, Cartesian/directional FMM, exact near actions, capacity/version guards | Whole resources, high frequency, crossover and complexity remain open |
| Recent FMM storage | Exact signed-axis/native-radius sharing, separable shifts, streamed/bounded cached reciprocal near coefficients | Preserves physical vectors and admitted degree; storage savings are not whole-memory qualification |
| Bench | Generalized candidate/holdout identities, exact versioned current/volume/harmonic archives and tamper checks | Candidates remain unqualified; qualified labels are withheld |

The [feature inventory](../../FEATURE_LIST.md#experimental-surface-integral-workflows)
and individual stage plans describe public configuration limits. The
[complete historical execution journal](../archive/memory/sie_gsp_execution_journal_20261004.md)
is retained byte for byte, including failed and superseded observations.

## Previously live executions

Read-only inspection outside the sandbox on 2026-10-12 found every recorded
runner/child PID absent. This observation establishes current absence, not an
unrecorded original process exit code.

| Original execution | Retained result | Current disposition |
| --- | --- | --- |
| Radial-series resources, handle 41039 | 20/24 measured cases; no collector receipt | Interrupted/unclosed; no complete-cohort scaling qualification |
| Signed-axis f30, handle 91003 | rc0 receipt and one passing preflight test; minimum capacity admitted; configured kernel/order admission failed | Terminal engineering preflight, physical/solve gate failed |
| Factored tensor AC, handle 50001 | Running/incomplete native JSON; no receipt/XML | Interrupted/unclosed; no terminal PASS |
| Automatic magnetic DAG, handle 5812 | Complete 792-column native JSON; no receipt/XML | Numerical output retained; execution closure unverified |

Snapshots and hashes are retained under
[the closeout evidence directory](../assessments/sie-integration-closeout-20261012/manifest.json).
The original collectors and worktrees were inspected without modification or
restart. Their successful output cannot manufacture missing terminal evidence.

## Integration verification and next work

The [integration assessment](../assessments/sie_integration_closeout_20261012.md)
records the new CUDA regression, test collection, Bench regression, skipped
historical-evidence controls and preservation checks. It also records the
remaining dirty historical worktrees, which were preserved without bulk staging.

Follow the [current acceptance plan](../plans/mom_generalized_production_development_plan.md)
for physical failures, missing collector closure, complete resources,
combination references and final four-repository acceptance. Existing
thresholds and original final scope remain unchanged.

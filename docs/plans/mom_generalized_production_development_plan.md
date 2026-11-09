# Generalized integral solver development and acceptance

Updated 2026-10-12. Implementation baseline: Maxwell `1e6647859da71d30e4864be3417aa256da8567cf`, Bench `54bb324bbb51fc60799f2c1501fb0d96bcd2f589`.
Recent implementation stages are integrated for review. Independent physical,
resource, complexity and final cross-repository acceptance remain OPEN.

The [current checkpoint](../memory/mom_gsp_execution_checkpoint.md) records
implemented paths and current job state. The [integration assessment](../assessments/sie_integration_closeout_20261012.md)
records this closeout's source boundary, tests and retained evidence.
The [original complete plan and journal](../archive/plans/sie_generalized_production_plan_20261004.md)
retains the original scope, numerical thresholds, failures and stage history.
Its historical LIVE/session statements are superseded by the current checkpoint.

## Formulations and scope

| Identity | Physical formulation |
| --- | --- |
| `pmchwt_scalar_sie` | Scalar isotropic regional Green functions, surface J/M |
| `anisotropic_dgf_sie` | Homogeneous tensor dyadic Green functions, surface J/M |
| `coupled_sie_vie_linear` | Surface J/M and linear volume polarization |
| `coupled_sie_vie_hb` | Surface/volume unknowns with DC and harmonic balance |

Kernel and operator backend remain separate configuration axes. Free-space,
1D/2D/3D quasiperiodic kernels, dense/H/H2/FMM and any periodic mesh algorithm
retain distinct versioned identities. A periodic mesh algorithm cannot qualify
as FMM. Unsupported combinations must reject explicitly.

Required final scope remains disjoint, nonnested homogeneous bodies; passive
scalar magnetic and reciprocal/nonreciprocal tensor materials; scalar nonlinear
chi2/chi3/Kerr/TPA steady-state harmonic balance; quasiperiodic kernels; CUDA
hierarchical/fast operators; public APIs; exact persistence and Bench evidence.
Open surfaces, nested/multilayer bodies, internal sources, geometry gradients
and general transient nonlinear MoM remain outside this scope.

## Next acceptance dependencies

1. Resolve the original resonant physical failures. The f30 minimum-order
   retained-capacity admission succeeded, but selected-order/kernel admission
   failed. No f30 solve or field qualification follows.
2. Close the original 24-case radial resource cohort with genuine terminal
   evidence. Twenty measured records exist; four records and the collector
   receipt are missing. All four sizes, both families and three repetitions
   remain required. Current accuracy remains null; historical errors cannot
   become current measurements. Partial/tail-only scaling fits cannot qualify.
3. Close tensor AC and automatic magnetic DAG executions. The latter retains
   a complete native comparison record, but neither collector has its final
   receipt/XML. Local numerical output alone cannot establish terminal PASS.
4. Complete independent converged references, quadrature/mesh/frequency sweeps,
   reciprocity/duality where applicable, energy/absorption and nonlinear/DC/
   harmonic/periodic/pole checks for each claimed combination.
5. Measure all-N setup/action/backward/solve and host/device resources, dense/H2
   crossover and high-frequency/changing-carrier behavior. Retained inventory
   and bounded numerical chunks cannot replace whole resource measurements.
6. Run current four-repository full acceptance and the original two independent
   final audits. The targeted integration regression is a separate engineering
   result and does not close these requirements.

The physical reference target remains 0.5%, and the modal cutoff/projection
budget remains 0.05%. Native algebra, original residual, kernel-error and
resource thresholds remain exactly those in the archived complete plan and
each versioned stage contract. Existing failures remain part of the evidence.

Continue through the stage-specific plans in this directory. Keep new runs in
new paths, preserve source/configuration hashes and actual failure states, and
commit each reviewed stage independently. Use consolidated regressions at
integration boundaries; retain successful historical evidence without replaying
its producers merely to recreate an archive.

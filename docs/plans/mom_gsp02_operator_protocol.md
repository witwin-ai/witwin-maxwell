# GSP-02 operator and kernel foundation

User execution order: GSP-01 stays open and retains its failed physical gates.
Subsequent dependency-ordered implementation continues. Full regression,
cross-repository acceptance and independent audit occur at stage merge points
and final acceptance, not after every edit.

The structural CUDA operator protocol owns shape/device/dtype, forward/adjoint,
Riesz inverse, preconditioner scales, a reentrant read lease, invalidatable
iterative state, direct capability, statistics and deterministic receipt.
GMRES and block GMRES use these actions through a solver view with no matrix
access. Direct/LU and direct fallback explicitly require DenseOperator.
Dense scaling computes balanced row norms in bounded row blocks. Existing
storage loans and cache invalidation retain their ownership semantics.

FreeSpaceHelmholtzKernel supplies the outgoing value and observation gradient.
Assembly, adaptive quadrature and field potentials receive the kernel;
restored DecodeHandle also owns its kernel. The public lifting path still
rejects non-free-space kernel contracts. This is a foundation for subsequent
periodic development, not periodic support.

Dense diagnostics retain exact small-matrix SVD. Matrix-free diagnostics use
a finite normal-operator Krylov subspace and projected singular values. They
may miss extremal/null modes and provide no certified stability bound.
Receipts hash matrix/Riesz bytes plus formulation/kernel/numerical identity;
execution timing is excluded. Statistics report unique owned CUDA storage and
action counters; per-action GPU time and native transient workspace remain
explicitly unmeasured, not zero and not performance qualification.

Targeted validation performed during this stage:
- Protocol/closed-form kernel/dense algebra/operator/public results initially
  passed48 with one restored-handle failure. The missing kernel was fixed;
  all23 affected protocol/public tests then passed.
- Additional kernel injection and direct-storage validation passed48 with one
  host lease mock incompatibility. The mock now implements the read-lease
  contract, and the production decorator shares that contract. All46 affected
  lease/protocol tests then passed in9.41 seconds; the three kernel tests passed
  in the preceding run.
- Tests explicitly use an operator whose matrix property raises, including
  six GMRES/block-GMRES/preconditioner combinations, Riesz gradients, adjoint
  dot identity, projected diagnostics and direct-capability rejection.

No full regression was repeated at this implementation checkpoint. The prior
GSP-01 576-node run was intentionally interrupted at the user's reprioritization;
it remains failed/incomplete, not accepted. Full source-bound and cross-repo
acceptance remain merge-point obligations. Proceed next to GSP-03 multi-body
operators, geometry validation, per-body state and physical queries.
### Bounded complete original weak residual diagnostic completed

The primary `operator_error.py` adds an opt-in scalar free-space diagnostic which streams every configured original J/M weak row with independent positive target/source batch limits. It calls the existing primary `ScalarSystemBlocks.sampled_action` and scaled norm primitives, constructs no global dense matrix and uses no compressed action. Every supplied RHS has its own native absolute/relative gate; noncontiguous multidimensional batches preserve their output shape. Zero RHS with zero error passes, while nonzero error at zero RHS has a JSON-null relative error and fails the unchanged zero absolute tolerance. The default relative gate is1e-10; the caller's finite nonnegative absolute tolerance is explicit. Invalid/empty/nonfinite/device/dtype inputs fail before original action.

The diagnostic binds input versions/modes, the original provider/configuration, all numerical scalar medium fields and body routing/identities, and rechecks them at target-batch boundaries and completion. Independent prelaunch v1 review found a P2 in identity-only configuration/material guards; the prepared v1 collector was never launched. The fix and new same-object configuration/material/body mutation controls are frozen separately before any GPU execution. The entire prior production module is unchanged when the appended function is removed;395 other production modules remain byteexact89621e7. No existing solver residual gate or default solve behavior changes.

This measures the complete supplied vector residual for the configured original quadrature. It does not measure error over all operator coefficients, certify quadrature convergence/continuum accuracy/rounding, qualify peak resources or grant production status. Selected scalar controls do not admit tensor or periodic providers. GSP-01 physical failures remain OPEN; the finer resonant geometry still requires compressed capacity admission and new source-bound physical acceptance rather than a dense allocation larger than the GPU memory. Full resource/scaling, tensor/nonlinear/periodic combinations, populations, current four-repository acceptance and at least two independent FINAL audits remain required. Goal ACTIVE; original scope and thresholds unchanged.

The one-shot new collector CLOSED13PASS/XML8.889s/runner10.790005689486861s/406 stable inputs. One module fixture constructs two actual disjoint lossy magnetic tetrahedra,12 edges/24 currents and one independent dense original reference. Native original controls cover15 noncontiguous RHS, exact target2/source3 physical-edge block enumeration,1e-12 native independent RHS norms, separated1e±120 currents and an error at original edge10 outside the old default8-edge sample, zero/nonzero action at zero RHS and no saved autograd tensors. Cached original controls test the scalar/explicit absolute gate, invalid arguments, input version/mode, provider/configuration/material/body mutations and nonfinite action without replaying original quadrature. Only new tests ran; no old PASS regression, physical producer solve or archive replay. Original session52414 drainedrc0 and observed runner3197581/pytest3197938 are absent outside the sandbox. Installed Core/nativeCUDA/GPU1 are recorded.

### Bounded original scalar RHS integration for compressed runtimes completed

The primary SystemOperator RHS dispatch now uses the existing selected original analytic lifting integrator for every local edge in free-space scalar hmatrix/H2/direct-H2/Cartesian-or-directional-FMM preparation. The integration primitive bounds classification by the declared pair/far batches and retains no global face-pair classification array. All local J/M rows, body order, analytic plane sources, integration/near-refinement policies and material differences remain. Dense, periodic and dyadic configurations keep their prior dispatch. The numerical lifting/kernel/quadrature primitives are reused, not copied or replaced.

The primary selected integrator now retains its existing near-refinement observations instead of discarding them: maximum order, pair evaluations and maximum accepted relative difference, together with the same near norm/estimator and independent touching/far convergence requirements. This is additive receipt metadata; no RHS global quadrature, continuum or production certificate is claimed. No solver residual threshold, public configuration default or constructor/solve-free archive contract changes.

Original v1 collector CLOSED3FAIL4PASS/XML129.946s/runner131.47759534232318s/406 stable inputs. All three failures were invalid test-helper stopping configurations before the new native RHS comparison/dispatch: direct-H2/FMM rejected the existing row-column-norm default. The whole original test and failure receipts remain preserved. Only the helper selects legal none preconditioning for those two backends; H/H2 helper defaults, all other test/fixture AST and all396 production bytes remain identical to the original numeric run. Independently reviewed failed-only v2 CLOSED3PASS/XML4.458s/runner6.0029570842161775s/420 stable inputs. The four previous successes are not replayed.

Seven DISTINCT targeted controls pass: sixNEW and one necessary affected selected-row ordering/validation control. Actual two disjoint lossy magnetic tetrahedra with two nonparallel analytic plane sources give complete J/M RHS relative difference3.66681773557313e-16 against the original integration, below unchanged2e-12. Per-body classification maximum11 and exact pair coverage are checked. Four cached-original route controls cover H/H2/direct-H2/FMM without replaying quadrature. A necessary actual public two-body shared-compact FMM prepare/run passes solve gates, bounded RHS and retained near-estimator checks; the new complete original weak residual diagnostic independently measures all120 rows with target19/source17 and gives1.3466771206667718e-13 against unchanged1e-10/atol0. No global original matrix or producer archive is constructed by this control. The public successful solve is not rerun in v2.

Original v1 session11831 drainedrc1 and observed runner3505483/pytest3505735 exited; v2 session26428 drainedrc0 and observed runner3602092 exited. Its short-lived pytest PID was not observed and remains explicit null; runner child-wait receipt/XML supply terminal evidence. Installed Core/nativeCUDA/GPU1 are recorded. No covered configuration suite, full regression, physical predecessor or archive is replayed.

Baseline92806ac and incoming54ecd1e remain explicit.394 other production modules are byteexact; runtime complete old-module inverse removes only the new scalar compressed dispatch; lifting entire prior prefix and every other numerical function remain unchanged. Primary reverse-patch validation binds the selected metadata-only near observation change. The stage does not qualify general tensor/periodic/nonlinear material/source gradients, full quadrature/physical convergence, RHS asymptotics or complete setup/solver GPU resources. GSP01/GSP09 remain OPEN. Original f30 capacity/growing3840/other collectors retain their frozen providers and original handles; no LIVE source change. Goal ACTIVE;final scope/gates unchanged;four-repository final acceptance and at least two independent FINAL audits remain required;production qualification false.

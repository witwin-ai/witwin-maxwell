# Surface solver evidence reproduction

Use the `maxwell` environment and a CUDA device. The independent analytic
sphere oracle uses scientific special functions on the host; the MoM solver
does not use that oracle as a numerical fallback.
Full-repository direct-solver tests also require the declared `direct-cu13`
extra on this host (`nvmath-python[cu13]` and the CUDA sparse-direct library).

```bash
export CUDA_HOME=<site-packages>/nvidia/cu13
export LD_LIBRARY_PATH="$CUDA_HOME/lib:<maxwell-env>/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WITWIN_RUN_CUDA_EXTENSION_BUILD=1
MAXWELL_PYTHON=python
$MAXWELL_PYTHON -m pytest tests --gpu -q
$MAXWELL_PYTHON -m benchmark.mom_convergence --levels 1 2 3 4 --order 4 --output /tmp/mom-sphere.json
$MAXWELL_PYTHON -m benchmark.mom_pilot --subdivisions 2 --output /tmp/mom-pilot.json
$MAXWELL_PYTHON -m benchmark.mom_fdtd --output /tmp/mom-cross.json
$MAXWELL_PYTHON -m benchmark.mom_fdtd --observables --output /tmp/mom-observables.json
$MAXWELL_PYTHON -m benchmark.mom_algebra_acceptance --output /tmp/mom-algebra.json
$MAXWELL_PYTHON -m benchmark.mom_archive create --report /tmp/mom-sphere.json --report /tmp/mom-cross.json --output /tmp/mom-evidence.zip
$MAXWELL_PYTHON -m benchmark.mom_archive verify /tmp/mom-evidence.zip
```

Inspect each command's `--help` for resolution, material and resource budgets.
The two CUDA library generations in this environment need the explicit library
search path above for sparse solver regressions. Other hosts should use their
own installed toolkit paths. Device visibility and shared GPU contention must
be recorded alongside timing. Dense double-current complex128 storage alone
is `64 * edges**2` bytes, before factorization, quadrature and decoder workspace.

The archive includes only tracked source plus explicitly selected JSON reports.
Commit new source before archiving; use `--allow-dirty` only when deliberately
recording a development snapshot. Missing or mismatched report source hashes
are retained in the manifest as provenance limitations. Matching a declared
subset of files is not proof that every dependency was fingerprinted. Integrity
verification checks bytes and paths, without extracting or executing source.

Physical acceptance requires reviewing independent reference errors, mesh and
query convergence, actual true residuals and the agreed budgets. A successful
archive or algebra solve cannot establish H2/G0, an external project handoff,
or the G2 prerequisite for choosing a physical extension.

The final qualified numerical campaigns and exact arguments are indexed in
[the local acceptance closure](../assessments/sie_local_acceptance_closure.md).
Use `python -m benchmark mom modes --help`,
`python -m benchmark.mom_ellipsoid_convergence --help`,
`python -m benchmark mom volume-reference --help` and
`python -m benchmark mom auxiliary --help` for the corresponding drivers.
The shared block campaign is `python -m benchmark mom block --rtol 1e-10
--output block.json`; the streamed experiment is
`python -m benchmark mom matrix-free --output streamed.json`.
The block report independently checks coefficient error against LU; older
1e-8 runs retain their failed coefficient-budget status despite residual passes.
Auxiliary comparisons require complete physical metadata or the archived
SHA-bound reviewed contract and source snapshot. Independent reference
qualification without MoM input still measures its own control-surface power
closure. None of these commands certifies an external gate.

The final outer bundle additionally includes raw logs, per-test inventories,
native current snapshots, original report inputs and every accompanying source
snapshot. These referenced files are not recursively included by the standalone
JSON-report archive command above. Verify the final outer ZIP with the same
`benchmark.mom_archive verify` command, and compare its separately published
SHA256. Historical absolute paths inside JSON remain original; locate relocated
payloads by their recorded hashes and the complete bundle member inventory.

For an exact Git checkout after unpacking the final outer ZIP, use the bundled
history instead of treating the plain source directory as a Git repository:

```bash
git clone delivery-20260911/final-closure/source-history.bundle maxwell-replay
cd maxwell-replay
```

The bundle includes the current source commit and its reachable history, so
individual historical report revisions can also be checked out. The standalone
source/environment ZIP and outer manifest record the same packaging revision.
Use each report's exact snapshot when a run records dirty source or several
provenance segments. A later digest-documentation commit does not change the
packaged numerical code. Dependencies and CUDA setup still follow the recorded
environment above; the Git bundle preserves source history, not installed binaries.

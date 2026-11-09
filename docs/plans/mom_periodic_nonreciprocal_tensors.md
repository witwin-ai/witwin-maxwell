# Nonreciprocal periodic tensor kernels

The internal periodic dyadic spectrum, outgoing plane/line sums and strictly
lossy image reference accept passive nonreciprocal TensorMedium samples. The
existing full Maxwell symbol, Schur elimination, independent modal forcing and
embedding, inverse eigenvector matrix and radiation-sheet selection do not
require transpose symmetry. Both proper coordinate rotations preserve the
reciprocal flag. No constitutive symmetrization or same-bias reciprocity is
introduced. Static subtraction still uses the symmetric part required by q.T T q.

Pole, singularity, passivity, strict image-loss and cutoff gates are unchanged.
The scalar analytic mode specialization remains restricted to exactly isotropic
constitutive matrices. Public solver and snapshot contracts are unchanged.

## Validation

/tmp/gsp-periodic-nonreciprocal-targeted-v1 terminates successfully: 49 passed,
zero failures/errors/skips in 389.54 seconds (runner 391.048 seconds). All 595
production/MoM test hashes match before, after and current source. Receipt SHA256:
f98d18c14d1a2fe3b2592ed27785caa35e0e90470e716bb993844c6d8bebdb52.
No full regression was repeated. Timing is not isolated performance evidence;
a reciprocal full-tensor control and an unrelated process also used GPU 1.

Eight new controls cover:
- 1D/2D/3D transposed-material and reversed-Bloch identities, same-bias
  nonreciprocity, Bloch translation and proper coordinate rotation.
- 3D Maxwell curl equations, modal adjoints and native bias derivatives.
- 1D/2D outgoing native bias derivatives against centered differences.
- Image material routing and strict positive-loss rejection, using a manufactured
  spatial value rather than claiming actual image-reference accuracy.
- Independent axial circular-channel responses with mu_s=mu_diagonal+s*bias
  and denominator q_z^2-k0^2*epsilon*mu_s. Both electric and magnetic source
  responses, bias reversal and absence of absorption from real gyrotropic bias
  agree with this analytic finite-mode reference.

Existing periodic spectral/line and dyadic-symbol controls also pass. Independent
static review found no hidden reciprocity prerequisite in these algorithms.

## Remaining scope

These results establish finite-mode/finite-sum implementation controls, not a
complete physical Green-function qualification. Actual nonreciprocal image and
source-plane controls, full material/loss regimes, public periodic tensor
assembly, nonlinear combinations, reference scattering, complexity and final
production acceptance remain open. GSP-01 and all previous unrelated failures
remain open. The overall goal and final thresholds are unchanged.

The reciprocal full-tensor 1D source-plane run is independent and still live in
<scratch>/witwin-maxwell-periodic-line-plane (session 36827). This stage does not edit
its frozen source or reuse its unfinished result as acceptance evidence.

## Actual nonreciprocal spatial representation control (in progress)

The appended parametrized test compares actual 1D/2D outgoing periodic spectra
against a separately evaluated free-space primary plus nonprimary real-space
image sum. The existing full non-diagonal gyrotropic fixture uses bias .2 and
strict electric/magnetic loss 2, two query points on opposite sides of the
source plane, and nonzero Bloch components. The final per-point balanced
Frobenius difference gate is 0.005 times the spectral norm plus 1e-5 ohm/m2.
Component budgets are separately fixed: spectral and image-tail absolute .002/8,
individual checked free-space integration .002/256, no relative tolerance.
Spectra use shells8 checked against16; 1D additionally uses sinh transverse
cutoff64/order64 with independent refinements. Images use shells1 checked
against2, excluding the primary. Free-space quadrature starts8x8 and has explicit
1024x128 limits. No failed budget is accepted or silently loosened.

Runner /tmp/gsp-periodic-nonreciprocal-spatial-v1.py, evidence directory with the
same stem, GPU1 session80848 is live. Source remains frozen until terminal.
Static independent review found no confirmed P1/P2. This is an independent
real-space image representation, not an entirely independent physical solver:
the underlying Maxwell/free-space modal implementation is shared. No unfinished
result is counted as passing; no source-plane, solved scattering, public API,
material-domain or production qualification is inferred.

### Terminal spatial evidence

The spatial runner is now TERMINAL PASSED:2passed,0failures/errors/skips,
981.466 runner seconds. All596production/MoM-test hashes are unchanged.
Receipt SHA2567e9ced0dca5ea20392967214fdffa11502a01607ce8a94cafcf586a9e15df260.
1D balanced differences [3.341853577127487e-5,2.4906636942193204e-5]
versus thresholds [.0953714004854048,.2564301511955555].
2D balanced differences [3.26439326453632e-5,2.672864112745712e-5]
versus thresholds [.0954253806590107,.256510115887747].
Both image-tail checks pass absolute.00025, and each primary/image integration
passed its explicit quadrature budget. The earlier in-progress paragraph is
superseded by this terminal record. This closes these selected strong-loss,
nonreciprocal off-plane representation controls only. Source-plane nonreciprocal
limits, material domains, scattering/power/reference qualification and public
periodic tensor assembly remain open; timing is not performance evidence.

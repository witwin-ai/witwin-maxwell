# One-dimensional tensor quasiperiodic Green kernel

This stage addresses the missing 1D spectral path of the actual tensor Maxwell
kernel required by GSP-07. It does not substitute scalar Ewald composition for
general tensor periodic Green functions. The 2D/3D building blocks remain in
the same primary module. Public tensor exterior assembly, source-plane primary
singular/contact subtraction and full production acceptance remain open.

For a line of length L, sum q_parallel=beta_parallel+2*pi*m/L. For each term,
integrate the second open wavenumber q over [-Q,Q] with dq/(2*pi*L), and integrate
the third wavenumber using existing outgoing Maxwell plane residues. A supplied
fixed unit normal perpendicular to the line defines the open direction; negative
height rotates to the opposite outgoing frame. Both constitutive tensors rotate
with the frame. The original pole guard is retained, with no implicit damping.
A finite quadrature result is not an unfiltered kernel or a primary regular part.

The estimator separately compares quadrature doubling, transverse range
doubling with increased quadrature, reciprocal cutoff doubling, quadrature at
the deepest cutoffs, and total original-to-final difference. Every gate uses
the existing impedance-balanced 6-by-6 Frobenius norm. Any failed gate raises a
receipt-bearing error; successful differences remain estimates, not bounds.

Targeted checks: scalar-limit comparison to independent scalar Ewald Hessians,
actual anisotropic source-free Maxwell curls, Bloch phase, reciprocity, rotation
covariance, native material gradients versus finite differences, source-plane
and input/ownership guards, and independent estimator-axis failures. No numerical
checks have run yet. GPU 1 is owned by frozen periodic-composition integration
regression session 84454; this stage must queue its validation without interrupting
or restarting that job. Development tree: <scratch>/witwin-maxwell-periodic-dyadic-line,
branch gsp-periodic-dyadic-line, base 547e1c4. Independent static audit requested.


Independent static audit found and then confirmed correction of a shared cutoff
P2: finite tolerances could overflow the acceptance threshold to infinity and
incorrectly pass. The helper now requires finite difference, reference norm and
threshold. A targeted synthetic overflow test preserves the reproducer. Explicit
real-dtype conversion of integer reciprocal orders avoids intermediate float32
phase loss. Real strongly dissipative full-tensor estimator success/failure
controls now supplement synthetic gate tests; their outcome remains unmeasured.
No unresolved confirmed P1/P2 remains in static review. Syntax and diff checks
pass; CUDA validation is queued behind merge-regression PID 1245808, whose live
process identity was independently observed. New numerical/test sources freeze
once the queued runner captures their hashes. Prior source-plane, lossless-limit,
public assembly and physical/complexity qualification gaps remain OPEN.


CUDA acceptance completed on physical GPU1 after the predecessor stage regression.
`/tmp/gsp-periodic-dyadic-line-targeted-v1` records 26 passed, zero failures,
errors or skips (pytest 46.04 seconds); all 590 numerical/test source hashes
are unchanged. Receipt SHA256:
e7696f8ae6114ed7708b076e758a678a3171836846f887bb4728229e4cffd7f7.
Coverage includes the 13 new line-kernel cases and 13 shared periodic-dyadic
cases. These are finite-budget physical identity and convergence controls, not
public tensor exterior assembly or production qualification. Earlier pending
entries above describe the preserved pre-run state.

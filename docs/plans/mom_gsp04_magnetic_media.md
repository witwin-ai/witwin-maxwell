# GSP-04 scalar magnetic implementation candidate

Based on multi-body candidate 34659b7; GSP-01 remains open. No physical failure
or final acceptance threshold is waived. Full regression, cross-repository
acceptance and independent audits remain combined merge-point obligations.

The scalar medium compiler now samples mu_r, sigma_m and magnetic Debye/Drude/
Lorentz poles together with electric responses, each conductivity once under
exp(-i omega t). Sampled and base passivity checks cover both constitutive
parameters. Snapshot medium metadata binds magnetic base values, conductivity,
all poles, frequency and branch identity. Backgrounds must be lossless and
support propagating waves with positive power impedance.

The passive refractive branch is sqrt(epsilon_r)*sqrt(mu_r), with each factor
on its passive branch. This gives negative real k for lossless double-negative
media and positive real power impedance. A principal root of the product alone
would lose this information. Single-negative interiors are allowed as passive
evanescent media. Finite nonzero near-zero samples are accepted; exact zero
constitutive values and exact divergent undamped poles are explicitly rejected.
Their limiting formulations are an open scope item, not silently approximated.

The lifting kernel's medium difference retains rationalization for weak
contrast, but switches to a direct difference when k_i+k_e becomes small.
The difference-kernel identity is versioned to branch_safe_v2. Existing matrix,
field, normal-jump and volume-loss formulas already use both epsilon and mu.
Regular modes and analytic carriers reject nonpropagating/lossy incident media.

Targeted CUDA/host validation:
- /tmp/gsp04-magnetic-v1.log: 7 passed in 11.94 s. Electric/magnetic loss and
  all magnetic poles; single/double-negative branches; near-zero/exact-zero
  distinction; 201-frequency branch-continuity sweep; public solved current and
  interior/exterior field electromagnetic duality; magnetic save/restore.
- /tmp/gsp04-targeted-v2.log: 16 passed in 6.30 s. Affected original physical
  contract tests, explicit exact-pole rejection, public lossy double-negative
  solve and independent direct evaluation of positive electric+magnetic volume
  dissipation. Old magnetic/negative-parameter rejection tests were updated to
  retain rejection of genuinely unsupported nonlinear/PEC materials.

These are implementation checks. Magnetic Mie TE/TM, independent volume
reference, near-zero conditioning, interface/energy/grid convergence and the
full precision/device matrix remain unaccepted until the combined campaign.
Impedance equality alone is not a zero-scattering condition for a curved finite
body with a different wavenumber: the exact zero-contrast control must match
both constitutive parameters. This physical distinction must be respected by
the independent oracle without loosening any numerical threshold.

### Linear complex scalar material admission correction completed

The original compact FAR stage exposed a linear Material(eps_r=2+0.05j,mu_r=1.3+0.02j) compiler TypeError: tpa_sigma_scale always attempted a real sqrt(eps_r), even with no TwoPhotonAbsorption declaration. The primary Material property now returns exact0 for no absorption declarations before any real-index conversion. All395 other primary modules remain exact; actual TPA formulas and their existing host-index restrictions remain. This corrects the existing finite-frequency scalar material contract without dropping imaginary constitutive values or loosening passivity/background gates. General complex DC/TPA/nonlinear qualification remains open.

Targeted GPU1/installed-Core collector CLOSED11PASS/XML8.597s/409 stable inputs: nine NEW host/public controls, one original failed maximum compact boundary under the exact original complex eps/mu context, and one necessary affected pre-existing real TPA formula control. New host controls forbid index sampling when no absorption exists, preserve passive outgoing branches and base metadata, count electric/magnetic conduction once, and retain gain/lossy-background rejection. Active complex TPA still refuses its unsupported real-index interpretation. Only original failed boundary context is retried; prior conductivity-boundary PASS and original120-row/public solve/archive/planner are not replayed. Original v3 TypeError and original producer aggregate failures remain archived.

New public Scene/Simulation preparation with complex eps2+0.05j/mu1.3+0.02j at90MHz/subdivision0/q4-singular3/native compact-shared reciprocal/separable p6/batch32 passes one complete original60-current matrix2.9926092838936833e-16, original H4.536429576002527e-16 and current-gradient3.852532514722163e-17 at unchanged1e-8 matrix/H and1e-11 gradient gates. Imaginary eps/mu are retained exactly. The original p12/32pair/4RHS boundary now forward1.405685772857564e-17/H2.2757372762507546e-17/native dot7.266535458506695e-16 below original1e-11. There is no solve, Result save, Bench archive or duplicate numerical producer. Host configuration/oracles introduce no CPU field/action fallback.

Original session43156 drainedrc0; runner604720 from execution receipt is absent outside the sandbox; short-lived pytest PID was not observed and remains null. Independent PHASE audit precedes human commit. This stage grants only implementation/selected discrete consistency evidence, no continuum physics/quadrature, complete allocator/resources/scaling/populations or production qualification. Original all-point24 series remains LIVE at frozen1204329 with no provider changes or relabeling; all other LIVE jobs and original GSP01/f30/open capability failures remain. Goal ACTIVE;GSP01/GSP09 OPEN; original complete scope/thresholds, current four-repository final acceptance and at least two independent FINAL audits remain required.

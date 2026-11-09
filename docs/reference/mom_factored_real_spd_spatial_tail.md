# Compiled factored spatial tails and cache ownership

### Factored real-SPD spatial tail integration completed

The primary dyadic Green module now exposes the explicit integration rule
factored_spd_tail_v1 for fixed exact-real symmetric SPD free-space media.
Distance-independent Maxwell/sign/projector state is compiled once for each
quadrature grid. All original lower-wavevector modal nodes and every original
infinite-tail node/weight remain; the strict evanescent bound selects the
factored polynomial remainder. EE/HH retain their explicit Coulomb differences;
EH/HE retain full physical curl terms. Existing native_v1 and closed-form rule
defaults and old default payload/key omission remain. No constitutive projection,
removed imaginary response, symmetrization, relaxed SPD gate, artificial physical
loss, finite transverse cutoff or CPU production fallback is introduced.

An exact incoming-sector generator reassignment avoids roundoff leakage from
an unused zero eigenvalue sector: exp(t*(i*A*P-alpha*(I-P)))*P=exp(i*t*A)*P
for commuting exact projectors. Its explicit difference includes +t*alpha*dP.
The physical outgoing eigenvalues are unchanged. This algebraic identity does
not certify rounded projectors or every spatial point. Native propagation uses
bounded256-node batches; construction workspace is not independently qualified.

Compiled grids/cache entries own and guard retained tensors, including identity,
version and requires_grad. The latter can change without a tensor version bump.
Unique GPU storage inventory includes all compiled tail state; cache reuse does
not reconstruct modal/sign matrices. Unsupported lossy/asymmetric/indefinite/
trainable samples, paired regional contrasts and periodic composition reject
before the new numerical strategy. Runtime, contrast and periodic primary
modules contain only the corresponding early guards;392 other primary modules
remain byteexact to baseline. General complex, contrast, periodic, material
gradients and public tensor/HB response remain required future Goal work.

Prelaunch review found one undefined-device P1 and five P2 gaps in gradient-mode
binding, per-query gates, failure evidence, unsupported-domain coverage and query
batch validation. Whole original primary/test/input and original audit remain
preserved. The exact correction inverse binds the prelaunch source; no first
candidate numerical run occurred. Clean incremental prelaunch review preceded
the original numerical launch. Historical audit input hashes refer to their
preserved preaudit snapshots, not to the later final417-input launch manifest.

Original session64606 CLOSED0, XML5.867s,3 targeted PASS,
all417 frozen SHA stable, execution-known runner1396371 absent outside sandbox.
The short-lived pytest PID was not observed and remains honestly null. GPU1,
installed Core, native CUDA production. Two genuinely new spatial points for
epsilon=mu=diag(2.4,1.7,1.2),30MHz have independent MP100 matched-SPD analytic
references; all eight individual query/block .005 gates pass, maximum relative
error6.287720281342211e-12. Adaptive independent
radial/azimuthal checks converge at orders[64, 32] with
unchanged rtol2e-4/atol1e-10. This is a successive-order estimator, not a formal
error bound. Actual unique retained cache35207488B
is below268435456B; it excludes caller/workspace/allocator/whole solver peaks.
One cache-hit control uses saved values with exact equality and forbids new
compilation. Gradient-flag mutation, in-place mutation and invalid query-batch
controls reject. New payload/unsupported runtime-periodic/contrast/periodic
kernel controls pass; the necessary affected native cache protocol uses synthetic
CUDA metadata without eigensolve/reference replay. Native/reference arrays are
saved for future reuse. No previously covered plane/spatial/HB/matrix/solve/
producer/archive/capacity/resource/full-regression PASS is replayed.

The original public tensor AC task also independently CLOSED1FAIL after
122779.59s: outgoing mode split unresolved at a grazing or defective spectral
point during cell-source integration. Its original session99995 is drained,
720 frozen SHA stable and both observed PIDs absent; it has no execution/Core
receipt and no identity is invented. Original f30 dyadic-center preflight closed
with its instrumented test PASS but actual capacity NOT admitted: planned
27026890736B, table23755200000B, near2974652336B exceed unchanged2GiB/1GiB gates.
Neither old collector used this new spatial rule or is relabelled current-source
evidence. Both failures remain preserved. Still-running providers are immutable.

This stage closes selected spatial integration and fixed-cache controls only.
Original covariance, public tensor AC/HB, full production physics/resources,
populations and current four-repository acceptance remain OPEN/unqualified.
GSP01 stays OPEN. Goal ACTIVE; original scope and every acceptance threshold
remain unchanged. Independent PHASE review precedes human-owned commit and one
root/checkpoint append after merge. At least two independent FINAL Goal audits
and full current acceptance remain required; no production qualification.

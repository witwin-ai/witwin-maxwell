# Unfiltered 3D outgoing layer representation

For lattice rows a1,a2,a3, form an oriented plane normal n with L=a3.n>0,
a tangent along a1, and transverse reciprocal vectors via QR/triangular solve.
Reduce each query by m*a3 into0<z<L and restore exp(i*m*beta.a3). All transverse
orders use beta projected into the plane plus the2D reciprocal lattice.

For positive outgoing modes, sum images indexed0,-1,-2,... through
1/(1-exp(i*(q.a3-beta.a3))). For negative outgoing modes, first evaluate r-a3
in the opposite proper frame, multiply exp(i*beta.a3), then sum1,2,3,... with
1/(1-exp(-i*(q.a3-beta.a3))). The negative mode's local longitudinal coordinate
remains positive, avoiding a growing exponential times a tiny geometric factor.
Both directions use the same checked Maxwell residues and per-plane area
normalization. Complex expm1 resolves the geometric denominator; unresolved
poles, nonfinite denominators or growing radiation sheets reject without adding
loss. Real propagating factors use the Abel value of their geometric series;
this formula alone is not a numerical limiting-absorption qualification.

The result is unfiltered3D Green data between planes, with finite transverse
reciprocal cutoffs. It is not a heat sum, source-plane singular/contact split,
certified full spatial kernel or globally checked resolvent. Source-plane queries
reject even away from the primary location. Lossless retained-mode poles reject;
omitted modes are not certified. Existing modal-cache ownership/gradient/budget
rules apply with a separate stacked key. No public solver or persistence route
selects this experimental representation yet.

## Validation

Initial /tmp/gsp-periodic-stacked-v1:4passed0skip/error/fail,12.387runnerseconds.
Independent audit identified a P2: forming the transverse Gram matrix squares
conditioning and rejects some lattices admitted by LatticeSpec. Replaced it with
QR plus solve_triangular, matching the lattice compiler, and added an admitted
thin-skew cell control against the same-area zero-order orthogonal cell.
Final /tmp/gsp-periodic-stacked-v2 is the targeted corrected-code run.

The old constant-modal line-mapping fixture lacked modal tensor state required
by the integrated cache. It now subclasses PlaneModes and supplies that state;
its constant-integral oracle is unchanged. Separate
/tmp/gsp-periodic-cache-mapping-regression-v1 passes3nodes,21.066runnerseconds,
including its measure and independent scalar-Ewald/material-gradient controls.
No compatibility fallback was added to production. This correction preceded
only the isolated3D QR change; line production is identical afterward.

3D controls compare actual lossy/lossless scalar-limit values against independent
3D Ewald Hessians on a skew left-handed lattice; actual nonreciprocal fulltensor
Bloch translations, transposed-bias identity and native bias finite differences;
source-plane/resonant denominator rejection; and thin-skew conditioning. They
do not constitute fulltensor spatial independent-reference, sourceplane,
scattering/power, all-material-domain, complexity or production qualification.
The independent fulltensor1D sourceplane job76454 remains frozen and live.

Corrected v2 TERMINAL5passed0skip/error/fail,13.199runnerseconds.
gsp-periodic-stacked-v1 receipt SHA256 945afb0a601f11ba1897ec4bc6e36d0079cd7133128de3852000418d6644e615; 599 source/test hashes unchanged during run.
gsp-periodic-stacked-v2 receipt SHA256 ce9a292b04b3fbe1759ab29b332fb7aa7e5be2bc1e2336803fcd71d6f5570c9e; 599 source/test hashes unchanged during run.
gsp-periodic-cache-mapping-regression-v1 receipt SHA256 859442b24d99b3fae8675bfa66398617b73626ed3859abc9ec7c78e7606f661d; 599 source/test hashes unchanged during run.

## Transverse cutoff gate and fulltensor reference control

estimate_stacked now compares the requested transverse shell depth with
max(1,2*shells), using a finite per-query balanced Frobenius gate. Failure retains
both budgets and the refined receipt with failed_axis=transverse_reciprocal_tail.
The successful return remains the original-budget value. Neither successive
agreement nor the analytic geometric sum certifies a global resolvent, omitted
poles, exact tails or a limiting-absorption procedure. One new actual control
rejects shells0→1 and accepts8→16 at absolute1e-5, verifies original value
ownership and invalid-tolerance rejection. /tmp/gsp-periodic-stacked-cutoff-v1
passes1node,7.782runnerseconds,599source/test hashes unchanged.

Prepared standalone /tmp/gsp-stacked-tensor-actual-v1.py compares the checked
stacked sum against a checked free primary plus independent real-space
nonprimary images for the same strongly lossy nonreciprocal fulltensor sample
used by prior1D/2D controls. Spectral/image-tail budgets remain.002/8; each free
integration is bounded by.002/256 with explicit8x8→1024x128 limits. Final error
must pass per-query0.5%+1e-5. Every completed free integration is saved with
query/material/frequency/reciprocity/config/value/receipt/hash, alongside source
and runner bindings, to preserve completed work if interrupted. No unfinished
reference run is counted as validation. The reference is an independent image
representation sharing underlying native Maxwell quadrature, not an entirely
independent physical solver. Full-domain/scattering/public qualification remains
open regardless of this selected control's outcome.

Cutoff receipt SHA256 f83d3dd35621136efad8573c0ffbd3496fe35754b9cd72b5e8d429bb7461a8ac

## Actual nonreciprocal full-tensor off-plane result

`/tmp/gsp-stacked-tensor-actual-v1` completed successfully in2884.499seconds.
The two balanced absolute differences are3.920049712137804e-5 and
4.0060817330820956e-5, below per-query0.5%+1e-5 thresholds
.09946641595274097 and.25668393046630056. The checked spectral8→16
component, checked free primary, nonprimaryimage1→2 tail, and all40 saved free
integration calls passed. All599 source/test hashes and saved artifact hashes
remained unchanged; imports and runner identity are bound in the receipt.
Receipt SHA256464de1f5a2ea6c5a995df737501ca758f3aea9aaa3dfc4254343f161fa8adc25;
periodic artifact184ac54406cdb1533045fb51a03744e0b689a0a126304372109d2ce36b529638;
reference artifact4b4d02bf4ae64806d18abdf43a2d027c85c5adb0861e4b00b24a0a79ce41f550.

This is a selected strongly lossy, full non-diagonal, nonreciprocal sample with
positive and negative off-plane queries. The reference uses independently
summed spatial images but shares the native free Maxwell integration kernel.
It does not qualify all material domains, source-plane tensors, scattering,
power, complexity, public backgrounds or production use.

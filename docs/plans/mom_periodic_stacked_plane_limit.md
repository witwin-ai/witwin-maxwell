# 3D source-plane primary-excluded remainder

estimate_stacked_plane_regular handles points on the plane spanned by the first
two lattice rows, including the primary origin. It extends the shared1D/2D
primary-subtraction limit engine with an explicit3D dispatch to evaluate_stacked.
The unit normal comes from the cross product of those rows; the initial probe
height must be strictly below half |a3.n|. All smaller probes then stay between
adjacent lattice layers. Other-layer queries and nonprimary lattice singularities
reject. The method does not wrap arbitrary queries or modify primary ownership.

All gates retain the same absolute budget A: checked free-primary A/8, every
point and side of the adaptive transverse reciprocal comparison A/8, roundoff
resolution below A/8, and two consecutive even-Richardson differences below A/2.
Only after individual spectral checks are the two sides averaged. No implicit
loss, projected source-plane coordinate, skipped singular image or larger budget
is introduced. The result excludes the primary, not a contact distribution;
finite agreement is not a certified global bound or limiting-absorption proof.

/tmp/gsp-stacked-plane-gates-v1 is TERMINAL PASSED11nodes,0failures/errors/skips,
9.123runnerseconds. Shared synthetic finite/quadratic/spectral/height/precision
controls run in2D and3D; additional3D guards check height, other layers and a
nonprimary singularity beyond the cutoff. Source hashes remained unchanged.
Independent static review found no confirmed P1/P2 and verified that the2D
computation sequence remains unchanged. An actual strong-loss scalar-limit
control with primary-origin and offset queries completed successfully in
/tmp/gsp-stacked-plane-scalar-v1, session77342: one passed in2513.831seconds,
600source/test hashes unchanged. It checks heights.12→.015,
reciprocal shells8 up to512, A=.02, against nonprimary3D images2→4 with1e-6
tail gate. Two consecutive Richardson maxima were .006193852609 and
.000412514601 below .01. The test asserts final agreement within .02 and the
image-tail gate; oracle values and final comparison errors were not separately
saved, so no more precise final-error claim is made. The3D tensor spatial
comparison53606 and1D fulltensor plane continuation76454 retain separate frozen
sources; no unfinished result from them is used by this implementation stage.

This is an internal estimator only. General fulltensor3D source-plane physics,
material/loss domains, gradients of this limit, contact/singular assembly,
public compilation/storage and production qualification remain open.

Gate receipt SHA256 4f6b8cd9380251cf127eb64c3018c443b07d5ff2aedde918138db69f333800a3; 600 source/test hashes.

Actual scalar receipt SHA256 d195d50caf43e31e45951750d6fc97549339ea72b46ff2861de2ed98834bd003. Saved limit artifact SHA256 af4bea854aeca1e8ea58e292de5d100ce564052d33511407db2360d1dace7ac6. The image representation is independent of the layer sum but uses the same underlying free Maxwell kernel. This selected strong-loss scalar control is not general full-tensor physical qualification.

## Selected full reciprocal tensor source-plane control

The actual CUDA control `/tmp/gsp-stacked-plane-tensor-v1.py` completed with
exit code 0 in 7002.782 seconds on source `5a18cf9`. At 300 MHz it uses
non-diagonal, distinct reciprocal tensors
`epsilon_r = [[1,.1,.05],[.1,1.3,.08],[.05,.08,1.6]] + 2j I` and
`mu_r = [[1.1,.04,0],[.04,1.2,.06],[0,.06,1.4]] + 2j I`, lattice rows
`(.7,0,0), (0,.8,0), (.05,.1,.9)`, Bloch vector `(.2,.1,0)`, and queries
at the primary origin and `(.1,.07,0)`.

All 22 stacked spectra were freshly evaluated. Heights `.12,.06,.03,.015`
required accepted reciprocal comparisons up to shells `64,128,256,512`.
Every point and side passed the `A/8 = .0025` spectral gate; the final maximum
spectral difference was `4.64504e-5`. Lower-cutoff failures remain in the saved
receipts. The maximum subtraction-resolution estimate was `6.23829e-8`.
Two consecutive Richardson maxima, `.0059105297633` and `.0003848799875`,
passed `A/2 = .01`. The four checked free-primary height artifacts were reused
from the already accepted 2D control after exact query/material/configuration
binding, pinned producer/consumer/runner hashes, and free-computation dependency
equivalence checks. The producer frequency is bound by its pinned runner because
the old height artifacts do not contain a frequency field. No periodic candidate
spectrum was reused.

A fresh nonprimary 3D image reference compared shells 1 and 2 with tail errors
`[2.8620153139542158e-5, 7.439075690168702e-5] < .0025`. Its 39 checked
free-space calls are saved individually; each uses absolute integration budget
`A/128 = .00015625`, initial orders 8 by 8 and maximum orders 1024 by 128.
The final balanced absolute differences are
`[2.6845974975251637e-5, 5.665441271624765e-5] < A = .02`.
All 600 source/test hashes, producer hashes and incremental artifact hashes
remained unchanged. Evidence lives in `/tmp/gsp-stacked-plane-tensor-v1`:

- Receipt SHA256: `241bcd499baef9a863f66ed5e7e5eee6882285519e6c345ccd1f4dceec22376d`.
- Candidate SHA256: `83b4d26b70d462324fc1313833290c688a4c8580f2fb3cbe2ee843badef34ef4`.
- Reference SHA256: `912372181b63e708e25ab7d402dc4fec96d3736c77bc07add67e624bcc38e9ee`.

This closes only the selected strongly lossy reciprocal full-tensor plane-limit
control. The image representation is independent of the stacked sum but shares
the underlying free Maxwell implementation. General material/loss-domain coverage,
nonreciprocal source-plane physics, gradients, actual assembly/scattering,
lossless/anomaly limits and public/production qualification remain open.

Independent terminal evidence audit found no confirmed P1/P2: it verified all
source/runner/producer/artifact hashes, exact query/material/configuration bindings,
finite complex128 shapes and each saved acceptance gate. No numerical rerun was
needed for this evidence audit.

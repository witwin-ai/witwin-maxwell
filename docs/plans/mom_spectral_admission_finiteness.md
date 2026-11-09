# Tensor spectral convergence admission finiteness

The shared free-space tensor spectral evaluator supplies regular kernels and
full-image terms for periodic tensor references. Finite spectral entries can
produce infinite balanced Frobenius norms; finite relative tolerances can also
overflow the acceptance threshold. Previously a zero directional difference
could satisfy an infinite threshold and return converged=True.

The evaluator now requires both directional errors, reference norm and threshold
to be finite before either directional gate passes. Invalid metrics fail
immediately with nonfinite_spectral_acceptance_metrics, retaining evaluated
orders and directional diagnostics. No solver or tolerance change is involved.

CUDA acceptance: /tmp/gsp-spectral-admission-v1, 12 passed, zero skips/errors/
failures in 3.62 seconds on physical GPU1. Five new controls cover reference,
threshold and separate directional norm overflow plus finite success. Existing
independent-axis cancellation, refinement reuse, actual budget exhaustion and
cache checks remain covered. All numerical/test hashes are unchanged during
execution. Receipt SHA256: e6df0db27aa2100df426efd69e1dea2138cc85d223400b6ba5ce0caaa0cc56a1.

This corrects an admission prerequisite; it does not establish tensor-periodic
source-plane spectral separation, public exterior assembly or physical
qualification. The existing LossyDyadicImageReference already handles strictly
lossy nonprimary images and must not be duplicated or relabeled as an Ewald
or lossless-capable kernel.

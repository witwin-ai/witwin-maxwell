# Explicit free-space FFT FMM policy

Use `mw.FMMConfig(table_storage="fft_v1")` with `operator_backend="fmm"`.
This remains experimental. It applies the original total-degree scalar Green
and three gradient translations via native CUDA circular convolution, composed
with existing PMCHWT J/M channels; exact near blocks and all original kernel,
order, residual and budget admission gates remain in effect.

Canonical `cartesian_fmm_fft_config_v1` binds every original construction/solve
parameter, actual kernel-bound rule, shift mode, and fixed implementation policy:
physical RHS chunks4, scalar RHS chunks4, visible tensor workspace budget128MiB,
float64/complex128 and order0..12. Future different workspace/chunk policies need
a new version. This policy takes precedence over each kernel-rule version.
Existing default/opt-in payloads are byte-exact, and old versions reject FFT tags.

Per-coupling storage is16*comb(2p+4,3)+32+56*comb(p+3,3) bytes: Taylor
coefficients, displacement, scale, integer moment indices, signed/unsigned
weights and two flattened FFT index vectors. p12 uses77928B. Every batch object
owns exact-size independent tensors; there is no persistent FFT grid/cache.
Coefficient construction batches32centers. The planner, retained capacity and
actual coupling inventory use the same storage formula; a compact admission
receipt cannot certify this representation's resource usage.

Scalar visible temporary allowance is L^3*(16*(32+20*4)+8*48) bytes. EM adds
4*comb(p+3,3)*16*128 bytes for bounded channels, fields, copy/finite checks,
curl/charge/output intermediates. The bound excludes backend FFT workspace,
allocator cache and caller input/output. Noncontiguous leading axes are gathered
and validated one chunk at a time. Full-process peaks must still be measured.

Periodicity is rejected at public configuration and all internal constructor,
order/shell/capacity entrypoints. Screened Ewald Taylor coefficients need separate
FFT composition and validation. The complete periodic/directional/tensor Goal
is not narrowed. Selected algebraic/public/archive checks do not qualify full
physical accuracy, system N scaling, high-frequency directionality or production.
See scalar/EM derivations and the phase assessment for measured evidence.

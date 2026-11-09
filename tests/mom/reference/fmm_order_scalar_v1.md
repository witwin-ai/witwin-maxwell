# Frozen scalar FMM order planner oracle

`fmm_order_scalar_v1.py` is a byte-exact test-only snapshot of
`witwin/maxwell/mom/fmm_order.py` at Maxwell revision08640e0 (unchanged by the
separable-shift phase). Source SHA256:
`51e3bb2aae8dd13475526b43c8a6a1cd4b4000d4feed28bd09b25cab78120009`.

The batched-order tests always verify this hash before evaluating the frozen
planner in the native MoM package context. Its kernel evaluation calls the
unchanged scalar Cauchy API, so planning decisions are compared against the
pre-batching loop. It is not imported by production or used as a fallback.
It requires native CUDA and is not an independent physical reference or a
floating-point enclosure. A changed source fixture must fail identity checks.

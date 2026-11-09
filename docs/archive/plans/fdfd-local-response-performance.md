# Native GPU local response optimization

Authorized implementation order (2026-09-16):

1. Remove snapshot copies from repeated context validation; retain mutation guards;
   add sparse row updates and reusable tile support.
2. Certify small updates using pivoted LU and an inverse bound, falling back to
   GPU SVD when uncertain; preserve near-singular rejection and residual checks.
3. Fuse fixed-rank GPU work and reuse scratch storage without exposing aliased
   trial results. Evaluate graph capture only where supported and worthwhile.
4. Reuse update factors, reduce acceptance writes, and qualify reconstruction
   frequency with long trajectories, zero RHS and near-singular cases.
5. Add same-generation candidate batching with sequential first-improvement
   semantics and multi-frequency support.
6. Run numerical regression and matched original optimizer timings at clean
   commits on physical GPU 1. Include p95/max, accepted updates and rebuilds.

Every implementation stage is committed after targeted validation. Final acceptance
uses complex128, unchanged 84/312/1200 physical problems and 90-proposal sequences,
original CPU 1/4-thread comparisons, fresh reduced LU and 1e-7 projection gate.
No CPU solver fallback, no weakened stale/owner/rollback semantics, and no changes
to differentiable material solve behavior. The performance target is total n1200
GPU sequence time below 0.340 seconds, not merely faster individual kernels.

Completed. All five stages have separate commits, followed by layout, automatic-grid,
precision and numerical-drift corrections. Final runtime is 478cce9. Full affected
regression: 220 passed. Original optimizer/fresh reduced-LU gates: all passed at
1e-7. Independent full-system/material-gradient qualification: passed. Independent
CSV readback: six comparisons passed. The n1200 sequence is 0.336692 s including
reconstruction, meeting the stated 0.340 s target and effectively tying the original
CPU 4-thread run. GPU batch throughput improves 5.85–5.95x against serial GPU calls.
Small CPU cases remain faster. CUDA Graph capture is not enabled for the dynamic
public exception/ownership contract; fixed-rank fusion and batching are delivered.
Evidence and limitations: docs/assessments/fdfd-pngf-optimized.md.

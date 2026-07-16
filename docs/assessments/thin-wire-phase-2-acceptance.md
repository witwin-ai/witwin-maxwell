# Thin-Wire Phase 2 Acceptance

Date: 2026-07-15
Status: accepted
Maturity: E2
Scope: axis-aligned PEC network topology, checkpoint/replay, and single-device adjoint

## Delivered

- Named endpoint nodes merge multiple wires into one deterministic physical
  charge node. The compiled graph carries sparse per-wire node membership,
  source-segment identity, junction IDs, minimum-wire node ownership, branch
  metadata, and cycle rank.
- Polyline branches and closed loops use the same compressed CUDA recurrence as
  straight wires. No cylinder voxelization or second field architecture was
  introduced.
- Checkpoint schema version 2 freezes wire current and charge alongside Yee,
  CPML, ADE, TFSF, and circuit state. Replay recomputes derived EMF on device.
- Standard and CPML native adjoint dispatch compose the Maxwell reverse with the
  exact transpose of current deposition, node continuity, current recurrence,
  and field sampling.
- Physical radius and local isotropic host permittivity/material density remain
  connected to the public `Scene -> Simulation -> WireData` autograd graph.

## Exit-Gate Evidence

- Real compiled degree-three branch and closed-loop networks ran 512 CUDA steps.
  The normalized continuity residuals were `6.18e-8` and `3.73e-8`; staggered
  energy drift was `7.87e-7` and `5.72e-7`, respectively. All are well below the
  registered 1% budget.
- The compiler rejects undeclared contact/crossing, mismatched or isolated named
  nodes, positive-length overlap, internal revisits, reserved internal-loop name
  collisions (including whitespace-normalized tampering), and invalid compiled
  membership/ownership. Junction mapping participates in the cache fingerprint.
- A float64 dense branch-network PyTorch oracle independently differentiated one
  full sample/update/continuity/deposit step. The fixture deliberately orders y
  segments before x segments so sampling and target order differ. Production
  CUDA-device VJPs for fields, current, charge, inductance, capacitance, and
  electric mass matched at `rtol=1e-10`, `atol=1e-12`.
- Public radius adjoint was `3.003022419082e4`; centered finite difference was
  `3.002967834473e4`, for relative error `1.81769e-5`.
- Public host-material adjoint was `-4.920937347412e1`; centered finite
  difference was `-4.922180175781e1`, for relative error `2.52496e-4`.
  Three successively halved steps produced radius errors
  `[2.29529e-4, 7.07909e-5, 1.81245e-5]` and material errors
  `[2.78017e-3, 7.17628e-4, 2.52961e-4]`. Both converge and remain below the
  registered 2% gradient budget.
- Checkpoint strides `1`, `7`, and the automatic square-root stride produced
  identical objectives and gradients. A nonzero `I/q` checkpoint survives an
  in-memory serialization round trip, is deep-cloned from live state, rejects a
  missing wire tensor, and replays four native steps within the float32 parity
  budget.
- Wire adjoint dispatch selected `WIRE_STANDARD` without absorbers and
  `WIRE_CPML` with CPML. Both produced finite, nonzero radius gradients. Runs
  whose preparation changes the Maxwell step through the joint wire CFL bound
  are rejected because that discrete clamp is not differentiated.
- The final focused Phase 2 matrix passed 163 tests with 8 hardware skips. It
  covered the reference/compiler/forward path, CUDA wire parity and ABI,
  native adjoint, the wire and general adjoint bridges, public API, and the
  capability-guard census. Ruff and `git diff --check` also passed.
- The broader CUDA/public/material-adjoint matrix passed 210 tests with 8
  hardware skips. Two no-wire material-effect precondition assertions failed;
  rerunning the same two tests from the Phase 1 commit reproduced the identical
  values (`1104.093` versus `1104.234`, and `3.740259e-7` versus
  `3.744493e-7`), establishing them as baseline failures rather than a Phase 2
  regression.

## State And Performance Contract

- Runtime wire recurrence storage remains one current and EMF value per segment
  plus one charge value per global node. Checkpoints append only current and
  charge, so storage is linear in graph size rather than Yee cell count.
- No-wire checkpoint schemas contain no wire names or allocation. The ordinary
  no-wire forward stepping path is unchanged from Phase 1, whose interleaved
  benchmark measured `-1.679%` relative to its branch-free baseline. A no-wire
  64-step CPML forward+adjoint benchmark used seven warmed samples per run: the
  Phase 1 median was `0.320995 s`, while two current medians were `0.309067 s`
  and `0.309517 s` (`-3.65%` at their mean), satisfying the `<1%` regression
  budget.
- Sampling and deposition retain pre-sorted CSR reductions and do not add an
  atomic or host fallback. The reverse uses the same segment/target CSR order
  for deposition-transpose and `G^T` segmented sums; repeated branch VJPs are
  bit exact. Reverse tensors remain on the CUDA device.
- The reproducible CUDA reverse microbenchmark reports `4133.499 ms` per
  100,000 segment-steps at 100 segments and `417.957 ms` at 1,000 segments.
  Peak incremental allocation was `22,528` and `140,288` bytes, respectively;
  raw seven-sample results are retained in
  `thin-wire-phase-2-reverse-performance.json`. The launch-dominated result is
  recorded for Phase 4 optimization rather than hidden by the no-wire gate.

## Independent Review

Independent topology review reached GO after reserved-namespace, fingerprint,
membership, owner, and permanent branch/loop gate findings were corrected.
Independent checkpoint/adjoint review reached GO after dispatch was made
fail-closed, reverse atomics were replaced by deterministic CSR reductions, and
a sampling-order/target-order `G^T` mismatch found with a y-to-x branch was
fixed and locked by the dense VJP. Final independent phase-level re-review
repeated the 12-test adjoint file and both reverse benchmark sizes, confirmed
the recorded peak allocations exactly and timing within 6.2%, and reached GO.

## Boundary Of This Acceptance

- Geometry remains axis-aligned on a uniform grid. Arbitrary-direction
  conservative stencils, custom/auto grids, periodic paths, and RF node/gap
  binding belong to Phase 3.
- Coordinate tensors, snap paths, topology, segment count, and owner decisions
  remain discrete compilation inputs. Phase 2 differentiates radius and local
  isotropic host permittivity/material density only.
- The PEC law has zero ohmic loss. Finite conductivity, rational skin effect,
  loss production, distributed fragment/state ownership, and single/multi-GPU
  reverse parity belong to Phase 4.

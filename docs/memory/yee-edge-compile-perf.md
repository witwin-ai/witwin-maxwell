# Shared Yee material compile reuse

2026-10-31. Base `cf9ab951` (Part A over `533b85b8`), branch
`coverage/fdfd-edge-materials`. This change accompanies Part A; it does not
change the material discretisation. GPU qualification uses the prescribed
GPU-1 Python wrapper; standalone shell device/driver probes do not reflect
its device access. All acceptance GPU runs use the wrapper, serially.

## Diagnosis and implementation

The six edge/face loops already call `_mesh_subsample_fill`, including the
`6180be02` band-only finite differences and bounded six-shift batching. They
were not missing that implementation. Two sources of repeated work remained:

- The converted p16s meshes disable optional topology diagnostics, so
  `is_watertight is None` prevented the existing world-AABB pruning even on
  closed, consistently oriented triangle surfaces.
- Every component independently recomputed coverage at sample points already
  evaluated on the nodes or earlier Yee components. At samples=4, p16s0 has
  260 distinct node sample coordinates per axis and 358 across nodes and edges;
  p16s1178 has 260 and 340. Algebraically equal lattice positions can have
  different float32 coordinates, so lattice-index reuse is insufficient.

The six shifted queries estimate coverage-ramp normals and Laplacians.
Multi-sample Kottke normals come later from accumulated material fields;
their reconstruction adds no SDF queries and remains unchanged.

The compile now certifies unknown fixed solid meshes by checking exact
cancellation of directed triangle boundary edges on the query device. It does
not change core topology metadata or invoke another SDF implementation. It
then applies the existing AABB/band packing and six-shift batching. Uniform,
nonperiodic compiles retain coverage in a compile-local point table indexed by
the integer bit patterns of the actual rounded node and Yee sample coordinates.
Reusing final coverage also reuses its center and shifted-normal SDF results;
neither needs recomputation at a coincident point. Every sample still executes
the original material blends and sums in the original order.

The tables share a 256 MiB budget per compile. Charts that do not fit retain
the packed-query computation, so many meshes cannot multiply cache storage
without bound. There is no persistent geometry/query
cache, host point arrays, CPU solver path or new distance approximation. The context
resets on successful or exceptional exit. Trainable meshes/coordinates and
surface-band meshes retain the existing path; surface-band width can depend
on the full query layout. Graded and periodic grids also retain the existing
path rather than inferring equivalence between different sampling windows or
periodic bridge coordinates.

## Exact pruning and reuse arguments

1. Let `h` be the vector of maximum sub-sample spacings and `R = ||h||_2`.
   The normalized finite-difference vector has norm at most one, so by
   Cauchy–Schwarz the projected width is at most `R`. Its minimum-spacing floor
   also cannot exceed `R`. The existing curvature correction satisfies
   `|c| <= w/4`. Therefore `d > 3R/4` implies `d+c > w/2` and the existing
   clamp returns exactly zero; `d < -3R/4` similarly returns exactly one.
   The existing floating-point margin retains the boundary of this band.
   Normals and Laplacians outside it cannot change the output.
2. A triangle chain whose directed edges cancel is a closed oriented cycle.
   Its winding number is zero outside the enclosing world AABB, including
   disconnected or self-intersecting cycles. Checking the sorted integer edge
   encodings against their reverses proves cancellation without assuming that
   an explicitly declared solid is closed. Unknown open meshes fail this
   check and retain all center queries. The original watertight case keeps
   its original admission and bound.
3. The existing bbox padding is `2R + delta + 2e-6 + margin`, with
   `delta = min(h)/4`. This conservatively contains the entire `3R/4` support,
   the shifted stencil, the core absolute distance smoothing
   `sqrt(r*r + 1e-12) - 1e-6`, and the existing coordinate-rounding margin.
   Actual rounded sub-sample coordinates are tested; coordinates are not
   replaced by an ideal lattice. Outside a certified closed mesh's padded
   box, the exterior coverage is saturated at zero.
4. Uniform coverage is a pointwise function of geometry, the exact query
   coordinates and the three scalar spacings. The core solid-mesh distance
   and winding reductions are per point, with the same triangle order when
   points are packed. The six shifted positions use the unchanged scalar
   delta and arithmetic. Thus equal coordinate bit patterns plus equal
   spacings permit reuse of the whole fill result. Signed zeros remain
   distinct. Separate meshes and spacing triples have separate fields.
   Graded coverage and surface fill are deliberately outside this argument.

## Verification

Fresh baseline interpreters import an unmodified `git archive cf9ab951` under
`<scratch>/witwin_cov_edgespeed/cf9ab951`; master invariants use a separate
`git archive 533b85b8`. Snapshots recursively include every compiled tensor,
including node and all edge components, and compare tensor keys and
`torch.equal` without tolerances. The harness and source hashes are recorded
alongside each snapshot in `<scratch>/witwin_cov_edgespeed/`.

The one added regression compares the entire cached/uncached compiled model,
including conductivity, on a non-binary uniform chart and a closed mesh with
topology diagnostics disabled. It requires less than half the SDF point work.
The original pruning regression remains unchanged.

All 20 scenes pass on GPU: **720/720 compiled tensors equal** to `cf9ab951`,
with identical keys, shapes and dtypes. Cases include p16s0/1178; eps_r 4 sphere
with samples=4 in polarized, polarized_tensor and arithmetic modes; a mesh in
both additional averaging modes; overlapping meshes; both mesh/primitive
priority orders; boundary contact and PML crossing; rotated anisotropic mesh
scaling; open solid and surface fill; millimetre scale; periodic and graded
controls; and samples=1. Two additional prepared FDTD models for p16s0/1178
match the full Part A models, adding **72/72 equal tensors**. Final records
check the current runtime source hashes.

Master `533b85b8` invariants also pass: **947/947 default FDTD tensors** and
**944/944 arithmetic FDTD tensors**, each across eight scenes including fields
after 80 steps, update/CPML coefficients, compiled materials and probe traces.
Vacuum/homogeneous FDFD adds **63/63 equal tensors** across single-sample,
arithmetic and polarized policies and vacuum, dielectric and magnetic/lossy
media: CSR data/indices/indptr, source vectors and converged fields. These
small solves use double-precision GMRES at tolerance 1e-9 with enough restart
cycles to converge. The combined final snapshot check covers **2,746 tensor
leaves**, all equal. The full requested fast tier passes **596 tests**, using
`tests/materials/compiler`, `tests/fdfd/test_magnetic_response.py` and
`tests/gradients/test_fdfd_adjoint.py` through the GPU-1 wrapper.

## Idle GPU timings and profile

Every row below is newly measured in a fresh interpreter, using the untouched
archives for master/Part A and the final worktree for reuse. Physical GPU 1's
compute-process list is empty immediately before and after each timed run,
excluding that run's own PID. FDFD includes public `prepare()` and lazy
`_ensure_material_components()`; prepare alone is not enough. FDTD includes
ordinary `prepare()` with the native extension warmed outside timing. Context
initialization and CUDA synchronization follow the supplied timing script.

| p16s position | Node-only master, idle GPU (s) | Part A, idle GPU (s) | Part A + reuse, idle GPU (s) | FDTD prepare, new code (s) |
|---|---:|---:|---:|---:|
| 0 | 19.808 | 133.088 | 8.521 | 8.417 |
| 1178 | 19.455 | 130.816 | 6.594 | 6.483 |

Speedups over Part A are **15.6x / 19.8x**. Both are below the <=40 s target
and below the old node-only compile time. Torch peak allocations for the new
FDFD compiles are 0.359 / 0.447 GB, versus 0.500 GB on both baselines.

The synchronized idle sample-0 profile measures 133.207 s before and 8.758 s
after the speedup. Before: mesh SDF is **127.783 s (95.9%)**, of which
**109.172 s** is in the six edge/face passes. This accounts for approximately
109 of the 113.280 s ordinary edge overhead above node-only compilation.
After: mesh SDF is **3.017 s**, with **2.525 s** in the six edge/face passes.
Each pass still has 128 mesh dispatches (center plus packed shifts for 64
samples); pruning and reuse reduce point/chunk work rather than merely the
dispatch count. Mesh point queries fall from **121,017,650 to 535,720** for
sample 0, and **121,258,746 to 793,908** for sample 1178.

Earlier shared-GPU profiles (186.940 / 174.998 s before and 23.830 / 11.963 s
after) were **contended** and are not acceptance timings. The idle profiles
and ordinary timings above replace them. CPU exploratory timings are likewise
not used for GPU acceptance.

## Part A consistency gates

Fresh same-branch FDTD runs supply `result.stats()['dt']`; FDFD then solves at
`sin(pi*f0*dt)/(pi*dt)`. One complex scale fits FDTD to FDFD over all three E
components on one common 45x45 plane. All residuals pass the <=0.5% gate.

| Case | Complex residual (%) | Fitted phase (rad) | Expected w0*dt/2 (rad) | Phase difference (rad) |
|---|---:|---:|---:|---:|
| p16s0 | 0.148830 | 0.102325938 | 0.102026212 | +0.000299726 |
| p16s168 | 0.006847 | 0.102026078 | 0.102026212 | -0.000000134 |
| LQ-1 eps_r 4, x=1 sphere | 0.103990 | 0.081561137 | 0.081620970 | -0.000059833 |

The sphere retains the LQ-1 level-20 custom grid, radius `1/(2*pi)` at
lambda0=1 m, seed-1729 offset, 12 PML cells and polarized samples=3. A common
CW point current at `(0.013, 0.007, -0.75)` drives both runtimes, isolating
material/operator consistency from plane-wave injection. Its comparison plane
is z=0.1 m and crosses the sphere; p16s uses the requested z=25% plane.
Every FDTD run uses 100 transient and 32 steady cycles with a Hanning window.
These gates establish shared discretisation, not physical accuracy at coarse
resolution. See [the comparison rule and unchanged-comparison audit](../reference/fdfd-fdtd-consistency.md).

Records and reproducible harnesses: `<scratch>/witwin_cov_edgespeed/`, including
`final_report.json`, `snapshot.py`, `time_prepare.py`, `verify_records.py`,
`gate_report.py`, `fast-tier-gpu1.xml`, `gpu_final_*`, `idle_*` and the two
fixed source archives. P16 gate fields are
`<scratch>/witwin_cov_sub3/diag/edgespeed_{fdtd,fdfd}_{0,168}*.pt`; sphere fields are
`sphere_{fdtd,fdfd}.pt` under the report directory. No comparison test or
benchmark implementation is modified.

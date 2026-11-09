# Mesh subpixel material compile diagnosis

Diagnosis date: 2026-10-30. Source baseline: Maxwell `042e0790`, branch
`coverage/mesh-subpixel-compile-perf`; installed core `0.4.5`. The original
diagnosis below contains measurements without implementation changes. The later
speed-fix and verification sections document the approved compiler changes.

## Findings

One small mesh triggers **3,136 full-grid signed-distance queries** for
`SubpixelSpec(samples=4, averaging="polarized")`. Each query runs two chunked
all-pairs Torch calculations: point-to-triangle distance and generalized winding
angle. The installed core has an accelerator registration hook, but importing
this Maxwell checkout leaves it **unregistered**. There is no active native
mesh SDF or BVH path here.

Measured full 32³ compilation is **286.6 s Mesh / 13.2 s Sphere**. At 64³,
Sphere measures **10.9 s**, while the Mesh measurements extrapolate to
**1,050–1,550 s (96–142×)**. Existing node-only compilation measures
**219.2 s at 64³**, with bit-identical node outputs demonstrated separately.

FDFD also pays for six unused Yee material grids: its material adapter consumes
the node components, whereas the shared compiler computes node components and
all six edge components. FDTD uses those edge components.

The historical native implementation is **not a numerically interchangeable
replacement**. Its direct winding kernel retains the dimensional epsilon that
core 0.4.5 removed; its unsigned-distance formulas differ too. Its static BVH
sign uses ray parity, rather than the same smooth generalized winding formula.

## Reproduction and measurement method

Use `scripts/dev/fdtd/mesh_subpixel_compile_perf.py` with the `maxwell`
interpreter. The supplied input archive has SHA-256
`34540b7e4e3783327498e50d18bef9c4393ee7ad47df6f7f52d7d86454d22fe0`.
It has 42 vertices and 80 faces, authored in metres. The script constructs one
solid `Mesh(recenter=False, scale=1.0)` with `eps_r=6.83`, or a sphere with the
same center and half the largest mesh extent as radius. The cubic physical
domain is approximately 2.5 times the largest mesh bounding-box extent, with
uniform 16/32/64 physical cells per axis, eight extra PML cells on each side,
CUDA material coordinates and four polarized sub-samples per axis.

All GPU runs use physical GPU 1 only, an RTX A6000, with Torch `2.13.0+cu130`.
The logical device inside the process is CUDA 0. The script warms the CUDA
context and geometry query payload before measuring. It synchronizes CUDA before
and after total timing; `--sdf-timers` additionally synchronizes each SDF call.
Those per-call timers perturb throughput and are reported separately from
ordinary total timings. Imports and native build time are outside the material
compile measurement. Each case uses a fresh scene, avoiding material cache hits.
An alarm bounds each script run to 570 seconds after imports.

Example command (one command per invocation; no extra environment assignments):

```bash
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_meshperf/run_gpu.py .worktrees/cov-meshperf scripts/dev/fdtd/mesh_subpixel_compile_perf.py compile --cells 16 --sdf-timers --output <scratch>/witwin_cov_meshperf/compile16.json
```

Modes: `compile` (fresh mesh/sphere cases), `micro` (distance/winding/dispatch
timings), `workload` (exact float32 sample reuse and coverage-band counts), and
`native-audit` (unchanged historical forward kernel, loaded in isolation).
`node-parity` compares the existing compiler's full and node-only outputs;
`compile --entry nodes` times the existing internal node-only mode without
changing either solver's consumer routing.
`--entry fdfd` measures `Simulation.fdfd(...).prepare()` followed by
`prepared.solver._ensure_material_components()`; **prepare alone is lazy and
does not compile these FDFD materials**. `--entry fdtd` measures ordinary
`Simulation.fdtd(...).prepare()`, which compiles during runtime initialization.
The direct mode measures `prepare_scene(scene).compile_materials()`; the current
public declarative `Scene` does not itself define `compile_materials`.

The converter log named in the request was zero bytes when inspected; the
1,400–2,150 second production measurements are user-supplied, not reconstructed
from that file. The repro contains only one structure, rather than the complete
converter scenes.

### Synchronized profile (corrected fixture, 16 physical cells + PML)

`--entry fdfd --cells 16 --sdf-timers`, 33³ nodes:

| Phase | Calls | Points/call | Mesh SDF seconds | Sphere SDF seconds |
|---|---:|---:|---:|---:|
| Nodes | 448 | 35,937 | 11.857 | 2.539 |
| Ex | 448 | 34,848 | 11.763 | 2.006 |
| Ey | 448 | 34,848 | 13.630 | 2.373 |
| Ez | 448 | 34,848 | 12.757 | 3.207 |
| Hx | 448 | 33,792 | 13.447 | 3.370 |
| Hy | 448 | 33,792 | 14.272 | 3.295 |
| Hz | 448 | 33,792 | 12.017 | 4.054 |

Total FDFD prepare plus material initialization: **101.955 s mesh** versus
**36.048 s sphere** with intrusive per-call timing. FDFD prepare itself took
0.0146 s / 0.1052 s, respectively; material initialization took 101.941 s /
35.943 s. Mesh SDF totals **89.744 s (88.0%)**. There are **108,351,936**
point queries at this resolution. The large fixed timing overhead on the
analytic sphere is why the uninstrumented comparison matters.

### Ordinary total timing and 64³ query microbenchmark

With synchronization only around the complete compile, the corrected **32³**
scene takes **286.587 s for Mesh versus 13.166 s for Sphere (21.77×)**.
Both have the 3,136 SDF calls and 359,332,288 points tabulated below. This is
the direct shared compile stage used by both solver families, without matrix
assembly or time stepping. Unlike the per-call profile, the SDF `seconds`
fields in this run are zero because individual calls were only counted.

At **64³ + PML**, five synchronized wall samples give these medians:

| Existing core operation | Points | Median seconds |
|---|---:|---:|
| Torch unsigned distance | 531,441 | 0.24904 |
| Torch generalized winding | 531,441 | 0.07981 |
| `Mesh.signed_distance` dispatch | 531,441 | 0.33072 |
| Same Torch SDF, packed padded AABB | 14,300 | 0.01387 |
| Same Torch SDF, packed surface band | 261 | 0.00770 |

The packed AABB and band SDF values each passed **`torch.equal`** against
the corresponding entries of the full query. This checks these point sets,
not an implemented sparse material compile. This microbenchmark uses node
coordinates without a sub-offset, hence 261 band points versus 274 in the
representative-offset workload count below. Padding was 39.41 µm, including
the conservative fill radius, quarter-sub-sample shift and 1 µm smoothing.

The full 64³ workload is 4.565× the 32³ point count. Scaling the observed
32³ total predicts about **1,308 s** for this single 80-face mesh; scaling
the 64³ per-query microbenchmark predicts about **1,021 s in SDF work** plus
material blending overhead. Scaling the separately measured 64³ node-only
compile (219.169 s) by its 6.889× point share gives another estimate of
1,510 s. A reasonable diagnostic estimate is therefore
**1,050–1,550 s per uncached 80-face repro scene**, with host scheduling and
GPU clocks contributing variation. Larger meshes and additional converter
objects readily explain the supplied 1,400–2,150 s range. A full 64³ mesh
compile was deliberately not run because it would exceed the ten-minute
per-run limit. The observed reduced-grid ratio is 21.77×, rather than a
directly measured 100× at 64³.
The full **64³ Sphere** control was measured directly at **10.898 s**, with
the expected 3,136 calls and 1,640,253,888 points. Relative to that control,
the full Mesh extrapolation is **96–142×**. Analytic timings do not scale
linearly with volume because fixed dispatch/host costs dominate; the 32³ and
64³ Sphere timings also show ordinary shared-workstation variation.

An actual native **FDTD `prepare()`** control at 8³ physical cells + PML
(25³ nodes) took **45.639 s Mesh versus 10.832 s Sphere**, again with **3,136
SDF calls**: 448 × 15,625 points for nodes; 448 × 15,000 for each electric
component; 448 × 14,400 for each magnetic component. The native FDTD extension
was built/loaded into `<scratch>/witwin_cov_meshperf/fdtd_cuda` first and took
30.274 s, excluded from these prepare timings. Loading that extension did not
install a mesh accelerator. This confirms the shared compile workload on the
FDTD preparation path, independently of the FDFD lazy-initialization check.

## Call graph and exact counts

1. `scene.py::PreparedScene.compile_materials` calls
   `compiler/materials.py::compile_material_model` and caches the whole model
   for fixed scene inputs.
2. `compile_material_model` iterates over `_sample_offsets`: `4^3 = 64`
   offsets on the node grid. `_compile_material_sample` requests geometry
   occupancy with sub-sample fill enabled.
3. `_geometry_occupancy` wraps the geometry in
   `compiler/subpixel.py::SubsampleFillGeometry`. Its `to_mask` calls
   `_subsample_fill_fraction`, which evaluates `d(p)` and `d(p +/- delta e_a)`
   for three axes: **seven SDF evaluations** per offset. Here
   `delta = min(dx,dy,dz)/(4*4)`.
4. `compile_edge_material_components` invokes `_compile_edge_component` for
   Ex, Ey, Ez, Hx, Hy and Hz. Each has another 64-offset loop and the same
   seven-query coverage calculation at that component's Yee positions.
5. Core `geometry/mesh.py::Mesh.signed_distance` stacks coordinates into
   `(N,3)`, obtains cached world vertices/triangles through `_sdf_query_data`,
   and calls `mesh_sdf.triangle_mesh_smooth_signed_distance`. With no
   registered accelerator this calls
   `_triangle_mesh_smooth_signed_distance_torch`, then
   `_triangle_mesh_unsigned_distance_torch_from_triangles` and
   `_triangle_mesh_winding_angle_torch_from_triangles`.

There are no periodic image copies for this PML-only scene. Geometry-state
caching in `Mesh._sdf_cache` saves transformed triangle data; it does **not**
cache distance values at query points. `_voxel_cache` is not used by this path.

| Phase | SDF calls | Points/call, 32 physical cells + PML | Points/call, 64 physical cells + PML |
|---|---:|---:|---:|
| Nodes | 448 | 49³ = 117,649 | 81³ = 531,441 |
| Ex, Ey, Ez, each | 448 | 48×49² = 115,248 | 80×81² = 524,880 |
| Hx, Hy, Hz, each | 448 | 48²×49 = 112,896 | 80²×81 = 518,400 |
| **Total** | **3,136** | **359,332,288 point queries** | **1,640,253,888 point queries** |

Thus an 80-face mesh causes 131.22 billion point-triangle pairs in the unsigned
distance pass and the same number in the winding pass at the requested 64³
physical resolution. A 300-face mesh additionally spans two triangle chunks.
Default chunk sizes are 8,192 points for distance, 4,096 for winding, and 256
triangles. At 32³ there are 45,696 distance blocks and 89,600 winding blocks;
at 64³, 202,496 and 402,304 respectively, before expanding each block into many
individual Torch CUDA operations.
The unsigned-distance helper and winding helper are each entered **3,136
times**. For 80 faces, `_point_triangle_squared_distance` runs once per
distance block (45,696 times at 32³); there is no single native distance-kernel
launch per SDF call. At 300 faces, each point block spans two triangle chunks.

**Normal distinction:** `_interface_normals` adds one SDF evaluation when the
single-sample polarized path is active, then uses `torch.gradient` on that
field. It is **not** used for the four-sample polarized edge loop because
`polarized_direct` is false. The multi-sample polarized reconstructions use
gradients of accumulated material fields and add **zero** SDF queries. The six
extra queries above are coverage ramp normal/Laplacian evaluations, even for
multi-sample arithmetic averaging. Changing averaging alone will not remove them.

## Native route audit

`FEATURE_LIST.md` advertises native mesh occupancy/BVH acceleration. In the
checked-out tree, searches for `mesh_sdf`, `bvh` and
`_register_mesh_sdf_accelerator` under `witwin/maxwell` find no implementation
or registration. Runtime inspection confirms
`witwin.core.geometry.mesh_sdf._MESH_SDF_ACCELERATOR is None`, no build error,
and no BVH. This is absence of a provider, rather than a failed extension build.

Historical source is recoverable from the **core** repository, not this
Maxwell tree. Core revision `6e75274` contains
`witwin/core/geometry/cuda/mesh_sdf_kernels.cu`; core commit `46c826d`
(0.4.0 preparation) removed that native backend while retaining the
solver-registration interface. The core 0.4.5 sign repair is `f31590a` and
changes the Torch winding implementation; it does not repair the removed CUDA
source.
The 0.4.5 sign repair does not install an accelerator or remove the all-pairs
work, so similar compile times on 0.4.4 and 0.4.5 are expected.

The relevant numerical contracts differ:

- Current core winding uses `2*atan2(N,D)` with **no dimensional denominator
  epsilon**, handles exact vertex queries, sums angles, and returns
  `unsigned * -tanh((abs(angle)-2*pi)/0.5)`. This is the scale-invariant
  generalized-winding sign fix.
- Historical direct CUDA `triangle_solid_angle` floors vertex-vector lengths
  at `1e-12` and applies `safe_signed_denom(D,1e-12)` to a denominator with
  units of length cubed. It is scale dependent and can reverse interior signs.
- Current Torch unsigned distance uses `torch.finfo(float32).eps` in several
  closest-point denominators, with masked overwrites; historical native distance
  uses early-return feature tests and different denominator floors. Both apply
  `sqrt(dist2 + 1e-12) - 1e-6`, but this shared smoothing does not make them equal.
  Preserving the current unsigned result was an explicit core 0.4.5 test contract.
- Historical static BVH signing uses jittered ray parity and a hard +/-1 sign.
  It is not the same computation as generalized winding with a smooth sign.
  Its ray-triangle routine has an absolute `1e-6` determinant epsilon as well.
  The historical BVH path was inspected, not numerically benchmarked. The core BVH
  construction threshold is **1,024 triangles**, so 80–300-face inputs would
  use direct native queries even with a provider installed.

An eventual solver provider can plug into core
`mesh_sdf._register_mesh_sdf_accelerator`: it must implement `supports`,
`is_available`, `smooth_signed_distance`, unsigned/winding queries and any
advertised static-BVH methods. Simply importing an FDTD CUDA extension does
not register this provider. A restored historical provider is not a safe
speed-only change.

### Historical native numerical measurements

The audit loads only the unchanged forward helpers and direct query kernel
from core `6e75274` through a scratch-only CUDA module (`--fmad=false`), then
uses the current core sign-composition function on its distance and winding
outputs. It **does not register a provider** or patch the material compiler.
Source SHA-256:
`3486431a4189492d421a65e4b82f911d030932c5d87298074672dc0e0d53b5f1`.

Queries are 8,192 seeded random points around the mesh bounding box, plus all
42 vertices and 80 face centroids. The random set has 1,156 core-classified
interior points at all three scales. Scale multiplies both authored coordinates
and query points, keeping their relative placement fixed.

| Scale | Max signed error (m) | Max unsigned error (m) | Max winding error (rad) | Random sign disagreements / 8,192 |
|---|---:|---:|---:|---:|
| 1 | 3.274e-5 | 3.274e-5 | 0.3293 | 0 |
| 1e-3 | 8.401e-7 | 1.598e-7 | 12.5663 | 898 |
| 1e3 | 1.073e-6 | 1.073e-6 | 7.629e-6 | 0 |

None is bit-identical. At the supplied scale the maximum error is **32.7 µm**,
larger than the 64³ repro's **24.8 µm sub-sample spacing**. That is not an
acceptable demonstrated tolerance for a speed-only material change; no
compiled-material tolerance is established by this SDF audit.
The historical direct query plus sign composition took approximately
14.08 ms for 117,649 grid points in the final script (three synchronized wall
samples: 14.011, 14.081, 14.080 ms). This includes the scratch loader's stream
and tensor interoperability overhead, rather than isolating kernel device
time. It is a performance reference for an incompatible algorithm, not an
approved replacement or a measured end-to-end compiler speed.

To reproduce the historical audit, export its source with:

```bash
git -C <workspace>/witwin-core show 6e75274:witwin/core/geometry/cuda/mesh_sdf_kernels.cu > <scratch>/witwin_cov_meshperf/historical_mesh_sdf_kernels.cu
```

Then use the GPU command prefix above with script arguments
`native-audit --cells 32 --native-source <scratch>/witwin_cov_meshperf/historical_mesh_sdf_kernels.cu`.

## Cheap wins and equivalence limits

`workload --cells 64` counts **234,321,984 requested center points** across
the seven 64-offset grid passes, but only **34,012,224 distinct float32 center
coordinates**: **6.889× reuse**. This is actual bit-pattern overlap, not an
idealized lattice estimate. It counts centers only, not the six shifted query
sets. Storing all 1.640 billion requested scalar SDF values would need 6.56 GB
even before coordinates, lookup structures or autograd; retain bounded active
regions instead. Cross-axis shifted points are generally different points:
caching cannot simply eliminate the six finite-difference evaluations.

For one representative node sub-offset at 64³, only **274 / 531,441 points
(0.05156%)** satisfy `abs(d) <= 0.75*sqrt(hx²+hy²+hz²)` with
`h = cell_spacing/4`. Here that threshold is **32.21 µm**. Outside this band,
the existing full coverage calculation agreed exactly with sign-based zero/one
fill at every checked point. The bound follows from `width <= ||h||₂` and
`abs(curvature_shift) <= width/4`: once `abs(d) > 0.75*||h||₂`, no possible
normal or clamped curvature shift can change the saturated fill. An
implementation should retain a rounding margin at the bound. The count is
one sub-offset, not a measured all-offset average or a mesh-independent ratio.

Evaluating center distance once over an active window and querying shifted
distances only in this band can reduce shifted work substantially without
changing the coverage formula. A conservative AABB exclusion is even cheaper
for this small isolated mesh. Preserve the original smooth-distance convention
(including the absolute 1 µm smoothing) when proving that points outside the
expanded AABB have exactly zero fill. A generic true-distance or Lipschitz
assumption is insufficient because the current core distance denominators
already contain scale-dependent floors.

Batching all offsets reduces launch/dispatch overhead but not the number of
point-triangle pairs by itself. Preserve float32 coordinate generation and
sub-sample reduction order; pack only the SDF queries, then scatter results
back into the original accumulation sequence. Do not materialize a full
`64 * grid_points * triangles * 3` tensor. Cache or batch reuse must account
for actual Yee offsets and geometry-state changes.

## Expected speed and numerical status

The strongest demonstrated zero-tolerance shortcut is the **existing
node-only compiler mode for FDFD**. `node-parity --cells 8` (25³ nodes with
PML) compared every tensor leaf outside `edge_components`: **24/24 tensors
were bit-identical**, with the same keys. Full compile took 47.631 s versus
6.794 s for node-only (7.01×). The experiment calls the existing compiler flag;
it does not change FDFD routing or caching. At 64³ the node-only point workload
is 238,085,568 versus 1,640,253,888, a 6.889× reduction.
**Measured node-only 64³ compile: 219.169 s**, with 448 calls, each containing
531,441 points. Thus this first shortcut alone brings the repro to about
**3.65 minutes**. Applying the point-share reduction to the user-supplied
1,400–2,150 s baseline suggests **200–315 s**, although wall-time gains also
depend on allocations, non-SDF work and GPU/host scheduling.
It does not benefit FDTD, which needs the Yee components.

The best proposed speed-only route retains the current core math and combines
node-only FDFD compilation, conservative fixed-mesh AABB exclusion, band-only
shifted queries, exact-coordinate reuse and bounded batching. The measured
packed-query medians give an illustrative cost model:

```text
one offset/grid: 13.87 ms center in AABB + 6 * 7.70 ms shifted band = 60.07 ms
all 448 offset/grid combinations without reuse: about 26.9 s in SDF queries
node-only 64 combinations: about 3.84 s in SDF queries
```

These are query estimates before material blending, indexing, cache/scatter
work and finite-difference halos. The point sets and sizes vary by sub-offset
and Yee component. Reusing common coordinates could reduce SDF work further;
larger batches can amortize the roughly 8 ms small-query dispatch cost. A useful
engineering target is **5–15 s for FDFD material compile and 20–40 s for the
full FDTD material model** for this isolated 80-face fixture. This corresponds
to roughly 70–310× / 25–80× versus the extrapolated current repro cost.
Those combined-path times are **predictions, not measured implemented fixes**.
The entire compiled model still needs a `torch.equal` qualification after any
such implementation; only node-only output parity and the packed SDF point
sets have been demonstrated here. No native algorithm currently provides a
measured acceptable replacement tolerance.

## Proposed fix boundaries (not implemented)

1. **Remove unused FDFD edge work first.** Update
   `witwin/maxwell/fdfd/materials.py::compile_fdfd_materials` and
   `witwin/maxwell/scene.py::PreparedScene.compile_materials` so the FDFD
   consumer can request a node-only model, using the existing
   `compiler/materials.py::compile_material_model(include_edges=False)`.
   Cache identity must distinguish node-only and full models: an FDFD compile
   must not leave a node-only cache entry that silently deprives a later FDTD
   consumer of Yee tensors. Node computations and accumulation order stay
   unchanged. Validate node tensors and resulting FDFD materials with
   `torch.equal`, plus FDFD-then-FDTD reuse. No distance approximation is needed.
2. **Bound fixed solid-mesh occupancy work.** In
   `compiler/materials.py::_geometry_occupancy` and
   `compiler/subpixel.py::_subsample_fill_fraction`, restrict static closed
   meshes to a conservative world-AABB window expanded by the fill ramp,
   finite-difference offsets and required reconstruction halo. Outside that
   support, exterior fill is exactly zero. Within it, evaluate center SDF first,
   then evaluate six shifted distances only in the conservative fill band.
   Keep the existing float32 point coordinates, coverage formulas, sample
   order, per-Yee offsets and gradient boundary stencils. An interior/exterior
   sign grid alone does not prove that a cell is outside the fill band.
3. **Reuse exact sample coordinates and bound batching.** Scope any query cache
   to one compile and fixed geometry state; key actual float32 coordinates,
   not mathematically equivalent lattice indices. Node and Yee offset formulas
   can round differently. Retain current triangle reductions and sample
   accumulation order. Batch query points in bounded GPU chunks; a single
   64-offset all-pairs tensor at 64³ would be impractically large. Cache a
   narrow active window, not billions of dense query results. Trainable
   geometry needs the original PyTorch graph and invalidation semantics.
4. **Only then restore native acceleration if needed.** Add a solver-owned
   provider in a new `witwin/maxwell/geometry/mesh_sdf.py`, register it from
   `witwin/maxwell/__init__.py` through core's
   `_register_mesh_sdf_accelerator`,
   and implement the current unsigned-distance contract and core 0.4.5
   generalized-winding sign. Do not reuse the historical BVH parity path as
   an equivalent sign implementation. Native reductions, FMA and feature
   selection require measured SDF **and compiled-material** tolerances at
   multiple mesh scales, near faces/edges/vertices, reversed orientation and
   trainable inputs. There is no established acceptable native tolerance yet.

Skipping magnetic occupancy is another scene-specific possibility: this
single nonmagnetic dielectric has `mu=1` and zero magnetic conductivity.
It cannot be generalized to layered scenes just by checking one structure;
structure occupancy can displace an earlier material. It is lower priority
than eliminating all unused FDFD edges and bounding mesh query support.

No changes under `witwin/`, feature inventory changes, merges or pushes are
part of this diagnosis.

## Retained run records and validation

Raw JSON records are in `<scratch>/witwin_cov_meshperf/`:

| Record | Script arguments after the required GPU prefix |
|---|---|
| `fdfd16_profile.json` | `compile --cells 16 --entry fdfd --sdf-timers` |
| `compile32_corrected.json` | `compile --cells 32` |
| `micro64_corrected.json` | `micro --cells 64 --repeats 5` |
| `workload64.json` | `workload --cells 64` |
| `node_parity8.json` | `node-parity --cells 8` |
| `fdtd8.json` | `compile --cells 8 --entry fdtd` |
| `nodes64.json` | `compile --cells 64 --entry nodes --kind mesh` |
| `sphere64.json` | `compile --cells 64 --kind sphere` |
| `native32_final.json` | `native-audit --cells 32 --native-source <scratch>/witwin_cov_meshperf/historical_mesh_sdf_kernels.cu` |

Each run prints environment/input metadata, call counts where relevant and
synchronized timings, and supports `--output` to save its record. The tables
above retain the substantive numerical evidence in the repository. GPU runs
were sequential, each below ten minutes, on GPU 1; no cloud resources were
used. The timing script passed a syntax parse and all exercised modes; the
diff passed whitespace validation. No solver regression suite was run because
the solver implementation is unchanged.

## Live historical native-SDF caller audit (2026-10-30)

Audited Maxwell master `7ca086d1` and installed core 0.4.5, including package
imports, native extension sources, examples, scripts and tests. **There are no
production callers of the historical CUDA mesh SDF/BVH in this Maxwell tree.**
The sole executable historical-kernel caller is the explicit scratch-source
`native_audit` diagnostic listed below.
`rg 'mesh_sdf|triangle_mesh_|_register_mesh_sdf|BVH' witwin/maxwell` finds only
a cache-description comment (and this change's Torch-query optimization).
Importing Maxwell and loading its FDTD extension leaves core's
`_MESH_SDF_ACCELERATOR` unset. The FDTD extension is a field-update extension,
not a mesh-SDF provider. MoM spatial trees are independent algorithms.

| Area and current path | Public reachability | Historical drift/sign exposure |
|---|---|---|
| FDTD and FDFD materials: `compiler/materials.py::_geometry_occupancy`, `compiler/subpixel.py::_subsample_fill_fraction`, core `geometry/mesh.py::Mesh.signed_distance` | Default when a scene contains a volumetric Mesh; core dispatches to Torch | Neither historical 32.7 µm native drift nor native dimensional-epsilon sign error; current core distance smoothing/floors remain unchanged |
| STL import: core `geometry/mesh.py::Mesh.from_stl`, `_load_stl`, `_read_stl_triangles` | Public `mw.Mesh.from_stl`; import only parses/welds/transforms vertices | Neither; import does not query any SDF |
| Material voxelization: `compiler/materials.py::_compile_voxel_surface_metal`, `_pec_occupancy`, `_pmc_occupancy` | Conditional on declared volume metal/PEC/PMC; calls geometry SDF/occupancy | No historical native exposure; Mesh queries use core Torch |
| Historical standalone mesh voxelizer | No current implementation or public entry; `_voxel_cache`/`clear_voxel_cache` in core Mesh are unused remnants | Dead cache fields do not invoke historical kernels |
| Autogrid: `fdtd/meshing/autogrid.py::_geometry_aabb` | Public `GridSpec.auto` when requested; calls `geometry.to_mesh()` and takes vertex bounds | Neither; no SDF/sign computation |
| Cut surface: `compiler/cut_surface.py`, `fdtd/cut_surface.py`, `geometry/surface_mesh.py` | Public surface-sheet/PEC preparation when declared | Neither; uses geometric triangle quadrature/cut traces. `SurfaceMesh.signed_distance` explicitly rejects occupancy queries |
| Examples: `examples/` | Explicit example execution | No historical native call; no `Mesh` SDF provider imports/registration |
| Ordinary development scripts constructing `mw.Mesh`: `scripts/dev/solve_mom_periodic_refined_control.py`, `solve_mom_tensor_public_control.py`, `solve_mom_dual_harmonic_spatial_control.py`, `measure_mom_leaf_preconditioner.py`, `solve_mom_dual_harmonic_power_control.py`; repro/compare scripts under `scripts/dev/fdtd/` | Explicit script execution; ordinary public geometry/runtime paths | No historical native exposure; MoM consumes triangles, material repros use core Torch |
| Historical numerical audit: `scripts/dev/fdtd/mesh_subpixel_compile_perf.py::native_audit` | Explicit `native-audit --native-source ...` only; never registered or called by public simulation | **Yes**: intentionally loads historical direct CUDA source supplied from scratch. Demonstrated 32.7 µm unsigned/signed drift and scale-dependent sign errors. Does not exercise historical static BVH |
| Mesh/material tests: `tests/core/geometry/test_mesh_geometry.py`, `tests/api/public/test_scene_dict.py`, `tests/materials/compiler/{test_material_compiler,test_curved_subpixel_convergence,test_fdtd_material_cache}.py`; MoM tests containing `mw.Mesh` | Explicit pytest execution, using public geometry/runtime or internal numerical checks | No historical native provider or kernel call |

No dead historical native implementation files remain in Maxwell to delete.
A Maxwell deletion commit would therefore remove **zero files**. The historical
core implementation was already removed by core `46c826d`; its deleted files
were `witwin/core/geometry/cuda/{__init__.py,backend.py,build.py,extension.cpp,mesh_sdf_kernels.cu}`,
`benchmark/benchmark_mesh_sdf.py`, `scripts/build_mesh_sdf_cuda_prebuilt.py` and
`scripts/verify_cuda_binary_arches.py`. Those paths are historical, not deletion
candidates in the current checkout. Keep the live core Torch mesh implementation,
its optional provider interface, independent MoM trees, and cut-surface runtime.
The feature inventory's old native-mesh acceleration claim is stale and is
corrected by this change.

## Bit-identical compile speed fix

The speed-only implementation keeps core's Torch SDF and the existing
coverage, curvature correction, material blending and accumulation order.
The node-only FDFD request this fix originally added (`include_edges=False`,
with a `(SubpixelSpec, include_edges)` cache key) was removed when FDFD began
consuming the compiled Yee-edge components (cf0dc081): `PreparedScene.compile_materials()`
again keys its cache by `SubpixelSpec` alone, and both runtimes read the same
model. An explicit material-model override retains its existing semantics.

For fixed solid meshes, the compiler packs center points inside the padded
world-space bounds returned by core's transformed triangle cache. It batches
the six finite-difference offsets for the potentially fractional band in one
SDF dispatch; core's per-point triangle reductions remain unchanged. There is
no persistent point-value cache or dense all-pairs allocation. Open or unknown
topology retains every center query. Surface-band meshes, trainable geometry
and trainable query coordinates retain the original seven-query path, because
surface thickness can depend on the original grid layout and geometry/query
gradients must retain their original graph.

The converted dataset meshes explicitly disable topology diagnostics, so their
center queries retain the full grid. Their improvements come from band-only
shifted queries and, for FDFD, omitting unused edges. A `fill_mode="solid"`
declaration alone is insufficient to prove saturation outside an open mesh's box.

Let `R = sqrt(hx_max**2 + hy_max**2 + hz_max**2)` and `delta` be one quarter
of the minimum sub-sample spacing. The existing width is at most `R` and the
clamped curvature shift has magnitude at most `width/4`. Thus, beyond
`abs(d) > 3*R/4`, the existing final clamp returns exactly zero or one,
independently of the normal or Laplacian. A floating-point margin retains
queries near the bound. Outside the retained band, sign-based saturated fill
therefore gives the same value and unused finite differences can be omitted.
For a watertight mesh, exterior winding is zero and points outside its AABB
have positive distance. The box padding is `2*R + delta + 2e-6 + margin`:
it includes the complete coverage/curvature support, normal stencil, core's
absolute distance smoothing and coordinate-rounding margin. Queries already
contain their actual rounded Yee, sub-sample and periodic-image offsets before
the bounds check. No idealized lattice coordinates replace those offsets.

Fresh-interpreter comparisons use
`scripts/dev/fdtd/mesh_subpixel_compile_compare.py` and an unmodified
`git archive master` snapshot (`f67019aa0e2868a9b304bc1d0d5f80aae29fa52b`).
All 24,487 tracked blobs in the archive were checked against that commit with
zero mismatches. Its complete `witwin/maxwell/` tree is identical to the branch
base `7ca086d1`. Master advanced independently later; this archive stays fixed.
Each timed run starts from a fresh scene. CUDA context, geometry data and the
FDTD extension are warmed outside timing; CUDA synchronizes at both timing
boundaries. FDFD timing includes `prepare()` and its lazy
`_ensure_material_components()` call. FDTD timing includes ordinary
`prepare()`. Tensor snapshots include every material-model tensor and every
prepared-solver tensor stored directly or in tensor containers, including
FDTD update and CPML coefficients. FDFD snapshots omit the intentionally
unused edge tensors; the corresponding FDTD comparison checks all six edge
components. Acceptance requires identical tensor keys and `torch.equal` for
every tensor, with no tolerance. The small `mesh` case also exercises both
cache keys on the same prepared scene and verifies full/node material parity.

### Final fresh verification

All 17 required/control scenes passed in both backends; the additional periodic
scene passed FDTD, while FDFD rejected periodic boundaries identically on master
and branch. These are 35 tensor comparisons and one matching rejection from 72
fresh interpreters. Every compared tensor passed `torch.equal`, with identical keys.
An independent reload of the saved snapshots rechecked all 3,300 tensor leaves
against the current source hashes; all passed. The fixed master archive was
also rechecked against all 24,487 tracked blobs with zero mismatches.
The table reports synchronized seconds and the number of material/prepared
tensor leaves checked per backend (shared aliases can appear under several keys).
Small controls use 8³ physical cells and two PML layers; `repro64` uses 64³
physical cells and eight PML layers. Dataset scenes retain their converted grids,
materials and four-sample polarized configuration. Timings are single fresh
preparations on the shared RTX A6000, so small-case differences include host
scheduling variation.

| Scene | FDFD master (s) | FDFD branch (s) | Speedup | Tensors | FDTD master (s) | FDTD branch (s) | Speedup | Tensors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mesh | 16.389 | 2.838 | 5.78× | 32 | 17.059 | 10.976 | 1.55× | 155 |
| graded | 17.657 | 1.396 | 12.65× | 32 | 16.676 | 14.893 | 1.12× | 155 |
| sphere | 6.851 | 0.603 | 11.37× | 32 | 5.334 | 5.511 | 0.97× | 155 |
| box | 8.703 | 1.303 | 6.68× | 32 | 7.524 | 6.516 | 1.15× | 155 |
| samples1 | 0.217 | 0.096 | 2.27× | 32 | 0.337 | 0.341 | 0.99× | 155 |
| overlap_mesh | 29.019 | 2.173 | 13.36× | 32 | 36.127 | 17.662 | 2.05× | 155 |
| overlap_primitive | 20.763 | 2.140 | 9.70× | 32 | 20.729 | 14.682 | 1.41× | 155 |
| primitive_first | 21.026 | 2.673 | 7.87× | 32 | 26.364 | 14.792 | 1.78× | 155 |
| boundary_touch | 14.466 | 0.826 | 17.51× | 32 | 13.822 | 8.038 | 1.72× | 155 |
| boundary_cross | 8.471 | 1.881 | 4.50× | 32 | 8.352 | 17.640 | 0.47× | 155 |
| transformed | 20.487 | 2.688 | 7.62× | 32 | 20.593 | 17.755 | 1.16× | 155 |
| open | 20.830 | 2.722 | 7.65× | 32 | 20.694 | 17.844 | 1.16× | 155 |
| surface | 27.530 | 4.253 | 6.47× | 32 | 27.724 | 27.670 | 1.00× | 155 |
| mm | 20.964 | 2.721 | 7.70× | 32 | 20.571 | 17.666 | 1.16× | 155 |
| periodic | unsupported | same rejection | — | 0 | 15.953 | 11.284 | 1.41× | 119 |
| repro64 | 973.273 | 3.144 | 309.57× | 32 | 962.358 | 20.013 | 48.09× | 155 |
| p16s0 | 1806.921 | 46.589 | 38.78× | 32 | 1407.155 | 136.371 | 10.32× | 156 |
| p16s1178 | 1435.847 | 22.511 | 63.78× | 32 | 1235.574 | 183.708 | 6.73× | 156 |

`overlap_mesh` checks two intersecting meshes; `overlap_primitive` and
`primitive_first` check both priority orders against a box. `boundary_touch`
and `boundary_cross` check contact with and crossing into the domain/PML.
`transformed` combines rotation and anisotropic scaling. The open solid and
surface-fill cases are accepted and match exactly; surface fill retains the
original query layout. `mm` sets the largest extent to 1 mm. `periodic` and
`graded` additionally cover image/wrap queries and per-axis spacing tensors.
The mesh control verifies full/node cache separation and node parity.
`samples1` uses arithmetic averaging; the public API rejects polarized averaging
with one sample.

Records: `<scratch>/witwin_cov_meshperf/verify_{master,branch}_{scene}_{backend}.{json,pt,log}`.
Each JSON records Torch/core versions, device, the harness SHA-256 hash and hashes
of all three changed runtime modules. The report generator verifies every record
against the final harness and runtime source, including full/node cache checks.
All GPU commands use the required `run_gpu_fix.py` prefix,
physical GPU 1 only, serially, with no cloud jobs.

The wrapper's worktree argument remains fixed for every run. Archived-master
runs use `<scratch>/witwin_cov_meshperf/compare_archived_master.py`, which selects
the untouched archive through `sys.path[0]` and its working directory before
executing the same repository comparison script. Metadata confirms the imported
runtime paths and hashes.

Fast-tier validation follows the user's per-change policy: shared `Mesh` class
tests (`tests/core/geometry/test_mesh_geometry.py`), the complete
`tests/materials/compiler` suite (including FDFD material support and FDTD
material-cache prepare/solve coverage), and
`tests/api/public/{test_simulation_smoke,test_fdtd_preparation_audit}.py`.
The unrelated 120-module name-match selection and MoM `VolumeMesh` physics
benchmarks are excluded from this completion run and belong to batch integration.
The fast-tier pytest command used the same GPU wrapper and GPU 1, serially.

`final_fast_tier_tests.xml`: {'tests': 585, 'failures': 0, 'errors': 0, 'skipped': 0}.

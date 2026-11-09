# Accelerated material setup

2026-11-03: CPU-authored primitive geometry must not pass a host descriptor
pointer to asynchronous CUDA distance or coverage kernels. `_DistanceGeometry`
and `_CoverageGeometry` now pack lazily and cache one contiguous float32
descriptor per query device. Allocation on the query stream lets the CUDA
caching allocator preserve the temporary descriptor's lifetime. Descriptor
values and kernel arithmetic are unchanged.

The binding-call audit found one additional upload requirement: CPU-authored
mesh BVH topology and triangle IDs now move to the triangle/query device as
contiguous int32 tensors. Polygon vertices already move to the query device
and now explicitly become contiguous; loop endpoints are concatenated there.
Rotation, size, position and other primitive parameters are copied by the
descriptor packer. PolySlab bounds and taper parameters already follow queries.

| Call site | Device status |
| --- | --- |
| `primitives.py`: distance, coverage | Fixed: lazy per-query-device descriptors; points and widths follow queries. |
| `primitives.py`: polygon_distance | Vertices explicitly contiguous on query device; points and loop endpoints built there. |
| `setup.py`: sample | Descriptors, values, fills, widths and outputs use scene device; axes derive from scene coordinates; bin offsets/IDs use descriptor device. |
| `subpixel.py`: distance, bulk, window, reconstruct | Descriptors supplied on scene device; query-derived distances, bands, windows, offsets and gradients remain there; material values use scene device. |
| `conductors.py`: occupancy, conductor_edges | Primitive queries use the fixed descriptor path; the contiguous union distance derives from scene coordinates. |
| `mesh.py`: mesh_keys, mesh_topology | Triangle/query tensors already share CUDA device; extent, keys and sorted IDs derive there. |
| `mesh.py`: mesh_bounds, mesh_distance | Fixed: CPU BVH topology/IDs uploaded as contiguous int32; bounds derive there; points/triangles explicitly contiguous. |
| `coefficients.py`: coefficients | Existing eligibility checks require all material, conductivity and profile tensors to be contiguous float32 on one CUDA device. |
| `tables.py`: hash, gather, remap | Existing eligibility checks require contiguous float32 CUDA inputs on one device; owners, slots and remap are contiguous int32 tensors derived there. |
| `poles.py` | No accel poles module or binding exists. |

All 17 entry points behind `cuda/binding.cpp` now check every pointer input
(and caller-supplied sample outputs) for CUDA placement, contiguity, same
device and kernel dtype before guarding the device or launching. Float inputs
require float32; bin/triangle/topology/table indices require int32; mesh Morton
keys require int64. The setup ABI/cache module version is `setup_abi_v6` (v5 added binding validation; v6 adds binary pole edges and their sample/pole bindings).

Regression coverage is in
`tests/materials/compiler/test_accel_descriptor_device.py`: CPU-authored
geometry/cache placement, 20 lossy-dielectric/PEC accelerated prepares with
immediate host-allocation churn and exact occupancy comparison, and direct
binding rejection of a CPU descriptor. GPU execution is pending; host checks
and an offline sm_86 extension build do not qualify GPU numerics.

Host validation passed for lazy descriptor cache/value equality, Python syntax
and the integration declaration. All three CUDA regressions collect and skip
on the host. The offline sm_86 ABI v5 extension compiled and loaded, then
rejected a CPU descriptor with the expected error; CUDA stayed uninitialized.

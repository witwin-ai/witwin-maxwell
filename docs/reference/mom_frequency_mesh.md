# Icosahedral edge-frequency surface meshes

The internal native CUDA surface constructors `sphere_mesh`, `ellipsoid_mesh`
and `deformed_sphere_mesh` accept an explicit integer `frequency=n`. This is a
mesh edge frequency, not electromagnetic frequency: each original icosahedron
edge has `n` segments. The resulting closed mesh has `20*n**2` triangles,
`30*n**2` edges and `10*n**2+2` vertices. For example, frequencies 3, 5 and 7
produce 270, 750 and 1,470 RWG edges without requiring a fourfold increase in
edge count between every resolution.

```python
from witwin.maxwell.mom.mesh import sphere_mesh, ellipsoid_mesh, deformed_sphere_mesh

sphere = sphere_mesh(radius=0.25, frequency=5, device="cuda")
ellipsoid = ellipsoid_mesh(axes=(0.20, 0.25, 0.30), frequency=5, device="cuda")
deformed = deformed_sphere_mesh(
    radius=0.25, frequency=5, harmonics={(2, 0): 0.15, (3, 2): -0.10},
    device="cuda",
)
```

These are internal discretization constructors, not a second public solver
entrypoint. Simulation continues through `Scene -> Simulation -> Result`.
Coordinates and the optional three-component `center` are in metres. Geometry
requires CUDA and float32 or float64 coordinates; no CPU numerical fallback is
provided.

Each original triangular face uses a barycentric integer lattice. Identical
integer edge/corner combinations share one vertex identifier across adjacent
faces, avoiding seams and T junctions. Only this connectivity and the existing
manifold checks use host work. Coordinate interpolation, radial projection,
ellipsoid scaling, harmonic deformation, areas, normals and intersection
predicates execute with native CUDA tensors. The existing seed and real
orthonormal harmonic evaluation are shared with the midpoint implementation.
The real harmonics retain the Condon--Shortley phase: positive orders use cosine
and negative orders use sine.

`frequency` must be a positive Python integer; booleans, strings and fractional
values are rejected. Supplying both `frequency` and an explicit `subdivisions`
value is an error, even if their triangle counts would agree. Omitting both
retains one midpoint subdivision. Explicit `subdivisions=0,1,2,...` retains the
existing deterministic recursive midpoint construction. The default signature
uses `subdivisions=None` solely to distinguish omission from a conflicting
explicit value. Radius, ellipsoid axes and center must have their documented
finite positive lengths or finite coordinate values.

Frequency construction performs one affine interpolation on the original
icosahedron face followed by radial projection. Recursive midpoint construction
projects after every recursion. Their samples generally differ even when
`frequency=2**subdivisions` gives the same triangle count. In particular,
frequency 4 and midpoint level 2 are not the same discrete mesh. No historical
midpoint solve, convergence ratio or physical acceptance transfers to this
frequency sequence by renaming a resolution field. A Bench consumer must
version its discretization declaration, preserve the continuous geometry and
source identity, and execute its required actual three-resolution sequence.

Vertices lie on the declared sphere, ellipsoid or radial harmonic surface.
Planar facets approximate that surface; this constructor does not apply a
volume correction. Any separately declared volume correction must retain its
own geometry metadata and query-separation checks. Positive harmonic radii are
checked at sampled vertices, followed by full native mesh validation; this is
not a global positivity certificate for arbitrary harmonic coefficients.
The caller's research-family constraints remain necessary. Geometry validity
and decreasing surface-area error do not certify electromagnetic accuracy,
external-reference precision, formal population acceptance or H2/G0.

`tests/mom/test_frequency_mesh.py` includes host incidence/orientation checks,
invalid control and CPU-device rejection, actual CUDA closure/intersection and
continuous-locus checks, translation/SI-scale checks, and invalid geometry
inputs. Midpoint regression tests load the separate immutable provider source
from commit `f112e8526511ce9d3cacf5df8d36bd8ba093dad6`, verify its mesh.py SHA256
`7b8ce062c65aec18b13184b5fd98bf848a6209f25c6461a85f79f854656e0456`, and compare actual
CUDA vertices and faces exactly. Test implementation and host-only checks do
not by themselves count as completed CUDA acceptance; actual results must be
recorded after the coordinated GPU run.

The coordinated GPU 1 run completed with 80 passing tests and no skips or
failures across the new frequency suite and existing `test_mesh.py`. Pytest
reported 5.53 seconds; the device was shared, so this is not a speed comparison.
The process asserted its mesh import originated in the isolated worktree, and
all four recorded source/document hashes remained unchanged during execution.
The actual original-source CUDA comparisons passed, including sphere midpoint
levels 0/1/2, default sphere, ellipsoid and harmonic constructors. No production
MoM solve or new formal numerical label was generated.

Evidence is retained separately in
`/tmp/frequency-mesh-gpu-acceptance-v1.json`, `.xml` and `.log`. The tested mesh
source SHA256 is
`235234eb759ef1eec662dcb5b4c49f09b0f07a02b501340ca7e1a07a8dc95261`, and the new test
source SHA256 is
`b54549a51951e824aacf3780000b60db49e5efe1f992dd8930a81243f62cd6af`.
This acceptance paragraph was added after that immutable test report; no tested
Python source changed. Independent integration review and any later actual
three-frequency scientific assessment are separate acceptance requirements.

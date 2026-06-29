# Grating TFSF Bloch/PML Notes

These notes summarize the supported public grating workflow implemented from
`docs/dev/grating-tfsf-bloch-pml-implementation-plan.md`.

## Public Workflow

The supported user path remains `Scene -> Simulation -> Result`:

- Build a `Scene` with `BoundarySpec.faces(x="bloch", y="bloch", z="pml", ...)`.
- Use either an explicit `bloch_wavevector=(kx, ky, 0.0)` or `bloch_wavevector="auto"`.
- Add a CW `PlaneWave` source with `injection=TFSF.slab(axis="z", bounds=(z0, z1))`.
- Run through `Simulation.fdtd(...)` and consume fields or monitors from `Result`.

When `bloch_wavevector="auto"` is used, solver preparation derives the transverse
Bloch phase from the CW TFSF plane-wave direction and source frequency. This is a
fixed-angle, single-frequency rule and is intentionally rejected for broadband
source-time objects.

## Supported Runtime Shape

The grating path is intentionally narrow:

- Bloch wrapping is supported on `x` and `y`.
- CPML absorption is supported on `z`.
- TFSF grating injection is a slab whose normal axis is `z`.
- The slab spans the transverse Bloch unit cell after solver preparation.
- The native CUDA FDTD runtime owns the production forward path.
- The FDTD adjoint bridge replays the mixed Bloch/CPML state, including complex
  field checkpoints and CPML auxiliary state for grating TFSF scenes.

## Explicit Rejections

The implementation fails early for unsupported variants rather than falling back
to a different physical model:

- TFSF slab boxes or partial-lateral grating slabs.
- Non-`PlaneWave` slab grating sources.
- Non-CW automatic Bloch phase requests.
- Mixed Bloch/CPML axes outside the x/y Bloch plus z PML grating layout.
- Unresolved automatic Bloch wavevectors in Tidy3D export.

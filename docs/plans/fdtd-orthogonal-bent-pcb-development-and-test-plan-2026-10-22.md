# Orthogonal bent-PCB route: coordinate contract, native tensors and independent gates

Status: proposed additional P12/R2 implementation route, with validation-only metric/oracle controls. Public APIs below are a concrete design, **not implemented support**. Production `SurfaceMesh` + finite-Cu preparation remains guarded. This route does not replace the required general arbitrary cut-surface split-cell work or the external bent-antenna reference.

## Geometry and public coordinate contract

The first chart has fixed signed constant curvature κ and a rigid right-handed frame. Its parameters are physical lengths `q=(s,u,z)`: s is neutral-surface arclength, u is signed normal distance, and z is the extrusion coordinate. In the local frame,

`γ(s)=(sin(κs)/κ, (1-cos(κs))/κ, 0)`,

`t(s)=(cos(κs),sin(κs),0)`, `n(s)=(-sin(κs),cos(κs),0)`,

`r(q)=origin+R0*(γ(s)+u*n(s)+z*ez)`.

Stable entire-function evaluations give the exact κ=0 rigid/flat limit. The scale is `h_s=1-κu`; the Jacobian has columns `J=R0*[h_s*t,n,ez]` and determinant `detJ=h_s`. Fixed bounds must satisfy a strict positive h_s margin and open angular span `|κ|Δs<2π`. These conditions make this constant-curvature tube globally injective; a merely positive local determinant is insufficient for a general varying-curvature chart. The first API rejects general centerlines rather than extrapolate this proof. A later variable-curvature chart must add an independent global tube/contact test.

Proposed public construction:

```python
chart = mw.BendCoordinates(curvature=-1/radius, origin=..., frame=...)
scene = mw.Scene(
    coordinates=chart,
    domain=mw.Domain(bounds=((s0,s1),(u0,u1),(z0,z1))),
    grid=mw.GridSpec.uniform(spacing),
    structures=(
        mw.Structure(geometry=mw.Box(position=(0,0,0),
            size=(board_length, substrate_thickness, board_width)),
            material=mw.Material(eps_r=eps_r,sigma_e=sigma)),
        mw.Structure(geometry=mw.Box(position=(0,foil_u,0),
            size=(copper_length,0,copper_width)),material=copper),
    ),
    ports=(physical_rf_feed_declared_as_a_chart_path,),
)
result = mw.Simulation.fdtd(scene, ...).run()
E_world,H_world = result.query_fields(world_points, coordinates="world")
```

`Scene.coordinates` explicitly changes all position, path, geometry-bound and domain-bound arguments to chart coordinates in SI meters. A Box therefore denotes its **true mapped curved body**, with u thickness and s neutral-surface arclength; it is not a world Cartesian box. Existing identity-coordinate scenes retain their current behavior. `scene.physical_positions(q)` exposes r(q) for rendering and validation. The public prepared-scene representation must retain both chart and physical geometry and preserve this interpretation on clone/save/load/resume.

Material scalar ε, μ and σ remain physical material properties, not pretransformed coefficients. Native stored fields are the covariant chart components `E'=J^T E_world`, `H'=J^T H_world`; each value is sampled at its actual native Yee position.

The proposed explicit field-frame contract is `result.chart_fields`: a typed `FramedFieldData` object carrying `FieldFrame(kind="covariant_chart", coordinates=chart)`, component labels `E_s/E_u/E_z/H_s/H_u/H_z`, native staggered chart sample coordinates and values. `result.chart_fields.tensor("E_s")` returns the native cochain component. A mapped `result.tensor("Ex")` raises a precise frame error; it must not expose E_s under a world Cartesian name. Existing Cartesian `Result.tensor("Ex")` remains unchanged. Existing internal reducers receive a private native-field accessor and are audited rather than route through the guarded display API. Serialization/export retains the frame and chart fingerprint; unframed mapped arrays are rejected on restore.

World queries retain the existing physical E/H tuple contract and reconstruct `E_world=J^-T E'`, `H_world=J^-T H'` after one-sided/staggered chart interpolation. An explicit `coordinates="chart"` input option changes point coordinates, not the default world Cartesian output frame. A future non-Cartesian vector output requires a typed frame argument. Power/loss/port code consumes native covariant fields, never a rotated public display array. Component-frame negative tests are a prerequisite for public export.

A κ=0 nonidentity rigid frame remains a useful real rotated-plane limit: physical sources, current direction, fields and monitor normals must rotate with the chart, and all physical observables must agree with the identity-frame control.

## Native constitutive tensors and continuous gradients

For each physical Cartesian constitutive tensor A,

`A'=detJ * J^-1 * A * J^-T`.

For isotropic physical ε, μ, electric σ and magnetic conductivity, this is the diagonal factor

`diag(1/h_s,h_s,h_s)`

times the corresponding physical scalar. ε, μ and σ must all transform; changing only ε changes both wave impedance and power. A later general physical tensor uses the full formula and requires the corresponding existing full-tensor solver/adjoint capabilities; initial support explicitly rejects unsupported physical tensors/nonlinear combinations.

The factor is evaluated on the **native component support**, not averaged afterward from a separate node grid. Static conductivity used for power must be exactly the conductivity consumed by the native update. Electric and magnetic ADE pole strengths, instantaneous background coefficients and all source/circuit electric-response factors transform consistently; dispersion cannot be inferred supported merely because the vacuum/static chart works. The initial static mapped baseline and the local ADE extension get separate commits and gates.

All numerical map evaluation, tensor multiplication, source/field interpolation and VJPs remain PyTorch on CUDA. Fixed geometry/curvature is sufficient for initial t/σ/substrate/feed gradients. Continuous material parameters must retain their tensors through metric compilation and native checkpoint reverse. Shape/curvature derivatives remain guarded until their own fixed-topology/mapping/monitor/source VJPs pass finite differences.

CFL is recomputed from the transformed native ε/μ and the smallest physical scale; the previous vacuum dt must not be reused. A conservative sufficient chart bound uses physical wave speed and `min(1,h_min)` with the chart mesh, followed by the existing material/ADE stability checks. No positive h_s floor, clipped material tensor or artificial damping may conceal an invalid chart.

## Copper: two actual traces and surface metric

The physical conductor is a native chart plane `u=u_f`. Grid anchors must represent that **same physical location** exactly; the material is not snapped or fitted to a reference. Existing native split left/right E traces and passive common/differential diffusion states remain independent. No unsplit multiplier or a reconstructed single E trace is substituted.

Let `h_f=1-κu_f`. The tangential physical power per chart area is `h_f*(E_s I_s+E_z I_z)`. Covariant components satisfy `E_s'=h_f E_s`, `E_z'=E_z`. Thus the same physical slab two-port becomes

`Y_s'=Y_physical/h_f`, `Y_z'=h_f*Y_physical`.

The modal rates stay unchanged, while each per-component positive residue/direct matrix scales by α (`1/h_f` or `h_f`): `B'=sqrt(α)B`, `D'=αD`. Chart control areas, transformed half-cell ε/σ and modal storage then give the invariant physical energy. The neutral surface has h_f=1, but **both layers of a real PCB generally do not**; treating the displaced layer isotropically is forbidden. Diagonal transformed bulk media are already consumed componentwise by the planar runtime, although its current broad error message says “isotropic”; full off-diagonal/nonlinear guards remain authoritative.

The slab approximation omits variation of curvature/metric across the physical copper thickness. Before enabling a radius/thickness band, independent full-Maxwell cylindrical-shell two-port controls bound this error using the original physical inner/outer radii and conductivity. The first tested radii are 30 and 300 mm, 0.01–30 skin depths at 1 GHz, harmonic orders 0/1/3 and TE/TM. No radius, thickness or reference plane is fitted. A configured κt bound alone is not a physical certificate; the actual error grid must be retained and the accepted band declared.

The substrate is an actual mapped u-interval, so its physical thickness is exact. Layer intersections, contact footprints and terminal apertures retain their true mapped physical areas and path lengths. PowerLossMonitor integrates `E'^* σ' E' dV_chart = E_world^* σ E_world dV_world`. Planar split-trace bulk-loss corrections require the metric on each physical side, with their own typed metadata and tests.

## Sources, ports and physical work

Voltage and current are coordinate invariants:

`V=∫ E_world·dr=∫ E'·dq`, `I=∮ H_world·dr=∮ H'·dq`.

Existing Terminal/Lumped path/contour geometry therefore maps to a real curved physical path. The RF branch V/I and SI reference impedance do not acquire a Jacobian factor. Finite aperture averaging and terminal footprint measures do use their physical line/area measures. Feed coupling still uses the exact transformed native electric-response coefficients, including σ/ADE. Direct RLC, MNA and embedded network sources sharing a physical support retain their existing joint elimination; they cannot be applied after a projection that erases source work.

Initial public mapped excitation may be RF feeds only. Unregistered field source types fail explicitly. A physical volume current transforms as `J'=detJ J^-1 J_world`; a physical Gaussian profile is evaluated at r(q), not at Euclidean chart distance. An ideal current moment includes the transformed physical delta measure. A physical plane wave has phase `k·r(q)` and covariant vector fields `J(q)^T E_world/H_world`, not a constant computational wavevector/polarization. Its Huygens injection needs the real mapped surface normals and both equivalent currents. Existing Cartesian TFSF must not be silently reused. The private cylinder test uses declared ring-mode drives and claims no public physical plane-wave mapping.

## Curved flux and radiation surfaces

For a chart face normal n_q,

`n_world dA_world = detJ J^-T n_q dA_q`.

Pointwise `(E'×H'^*)·n_q dA_q` equals physical `(E_world×H_world^*)·n_world dA_world`. A mapped closed chart box is a **curved physical closed surface**. Its probes interpolate native fields in q and reconstruct vectors with J^-T. A world Cartesian finite-plane monitor requires a separate inverse-map sampling implementation; it cannot be interpreted as an axis-aligned chart face without changing its declared geometry.

NF2FF requires physical positions r(q), cofactor normals/areas, physical E/H and physical outgoing phase `exp(-ik direction·r(q))`. The existing Cartesian phase at q is wrong. The shared physical-surface quadrature must be used for near flux, equivalent currents and radiation, with source enclosure/physical-vacuum checks. Add a typed mapped-surface sample record rather than convert each benchmark with an independent formula. The original bent-antenna gate still needs convergence of near flux/NF2FF, Cu/substrate heat and accepted RF power, then comparison with a separately converged external physical bent-antenna model.

Native CPML in spatially varying transformed ε/μ is **not assumed** an exact cylindrical PML. Its distance/thickness/refinement and possible transformed-metric history terms need independent vacuum radial/mixed-angle reflection and long-time checks. In particular an inward radial truncation must not cross the focal point or create unstable concave-PML modes. PML attenuation is never included as copper or substrate heat. Guard unsupported map/boundary combinations until these checks pass.

## Independent controls and ordered implementation

1. Metric identities: exact flat/rigid limit, inverse/Jacobian/determinant, no wrap/focal crossing, both Maxwell curls including σ, current-source work, bulk heat, port line/contour invariance, every face flux, and both surface Cu scales. Inverse branch unwrapping is anchored to the validated Scene chart bounds. Current validation-only suite has 20 passing CPU controls, including a nonidentity rigid rotation.
2. Independent annular full-Maxwell Hankel two-port: continuity of tangential E/H at both physical copper radii, no polar axis in the computational domain, TE/TM and angular harmonics. Vacuum gives the exact swap S matrix; finite Cu is passive/reciprocal and converges to the opaque PEC limit. Current independent oracle suite is separate from native numerical qualification.
3. Private native diagonal-metric controls, with no public map enabled: complete two-drive S extraction for vacuum and Cu, both polarizations/harmonics, matched source amplitudes, unchanged 3% transmission/absorption and absolute reflection gates, and spatial/time/PML convergence. Any failed grid remains evidence. Numerical Hankel de-embedding, if needed, must follow the exact native vacuum recurrence, retain a fixed physical reference plane and be validated on the independent vacuum before Cu; it may not fit material or expected scattering.
4. Freeze the public coordinate/array/source/monitor semantics; implement geometry/compiler/native component metric and timestep checks, two non-neutral copper layer scales, physical RF feeds and typed Result mapping. Public support requires the plane/cylinder gates first. Each addition gets focused native GPU tests and its own commit.
5. Shared mapped loss/closed flux/NF2FF, resume fingerprints/state, native Graph and million-step controls. Validate actual bent lossy substrate plus a physical RF feed and closed power.
6. Continuous t/σ/substrate/feed adjoints and finite-difference step scans; native batch activity/reset/graphs; distributed metric ownership, two-sided Cu support, halos, observations and resume.
7. Final external physical bent-antenna reference and performance/regression matrix. Report this mapped constant-curvature scope separately from arbitrary SurfaceMesh split-cell support, which remains required and unsupported until independently implemented and qualified.

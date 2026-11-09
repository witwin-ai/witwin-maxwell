# FDTD absorbing boundaries and CPML passivity

## Automatic absorber selection (2026-11-02, CPML 1+2)

`Simulation.fdtd(..., absorber="auto")` is now the default. Preparation resolves
it once, before building boundary coefficients:

| Fields and boundary topology | Resolved absorber |
|---|---|
| Real fields, closed-transverse PML | `"absorber"` (graded sigma) |
| Open topology, including an all-PML box | `"cpml"` |
| Complex fields, including nonzero fixed-k Bloch | `"cpml"` |
| No PML faces | `"none"` |

The shared `fdtd/boundary/runtime.py::closed_transverse_pml` predicate requires
at least one PML-bearing axis and at least one non-PML axis; every non-PML axis
must be closed on both faces (periodic, Bloch, PEC or PMC). Either PML face makes
an axis PML-bearing, irrespective of its opposite face. Mixed PEC/PMC faces are
closed. An axis with one open `none`, Mur or mode-ABC face is not closed.
Scene symmetry replaces the selected face by PEC/PMC before classification;
a symmetry plane alone does not close its opposite face. BoundarySpec-only
queries have no scene symmetry to apply. The non-PML-axis requirement prevents
vacuous classification of all-PML open boxes as closed-transverse.

Automatic graded sigma requires **>= 40 layers on every effective PML face**.
Fewer layers raise `ValueError`; increase the explicit layer count or select an
explicit absorber. Preparation never changes geometry or layer counts. The
single-device graded absorber is the permanent choice for this topology: the
energy-stable PML research track ended with NO-GO for every candidate (see
docs/archive/plans/fdtd-cpml-energy-stable-pml-2026-11-02.md). Its reflection
and phase accuracy still need validation for each application. The distributed
solver rejects automatic graded sigma with `NotImplementedError`: only its
CPML/stable-PML absorbing path is qualified. Automatic CPML selection in open
distributed scenes remains available. Explicit `"cpml"`, `"stablepml"`, `"pml"`
and `"absorber"` retain their existing coefficient laws and layer admission.

Result metadata records the resolved choice as `absorber` and its explanation
as `absorber_resolution_reason`. Selection is deterministic; there is no
runtime fallback or automatic absorber replacement.

## Growth guard (main single-device step loop)

The guard applies to active `"cpml"` or `"stablepml"` with either real-field
closed-transverse PML or complex fields with a nonzero resolved wavevector
component on a Bloch axis. Zero-k complex scenes are outside the latter case.
The opt-in core path refuses closed-transverse CPML with the blocker
`closed-transverse CPML (passivity guard)`; its stepping loop has no guard.
The distributed stepping loop and adjoint replay also have no guard. This is
an empirical detector of sustained energy growth, **not a passive CPML closure**.
Under user decision D2 (2026-11-06), real-field closed-transverse cells remain
fail-closed with `FDTDPassivityError`. Complex fixed-k cells emit exactly one
`RuntimeWarning` per run when the same criterion is met and continue running.
The warning gives the measured log-energy slope, standard error and window
start/end steps. No threshold, window, sampling interval or energy functional
changes with this decision.

Sampling runs at the same post-update location as energy shutoff, every **32
completed steps**, including both complex field banks. The existing shutoff
uses an electric-energy proxy; the guard extends this to Yee-volume-weighted
`(epsilon |E|^2 + mu |H|^2)/2`, plus resolved lumped LC, circuit and copper
storage. CopperFoil contributes
`sum(state^2 * area * admittance_scale)/2` for every real and imaginary
entry. Its native electric edge mass is replaced by the two half-cell trace
energies `area * (capacity_left * E_left^2 + capacity_right * E_right^2)/2`,
so the private right trace is counted without double-counting the left one.
CPML psi is excluded because it has no known positive storage functional.

Sheet scenes use the **modified Newmark functional** qualified by
`thin_sheet_energy(solver, modified=True)`, evaluated entirely with float64
device reductions. This replaces native patch masses by split-edge electric
and split-face magnetic masses, counts the independent chord electric masses
(removing promoted traces already represented by chords), and includes
`sum(copper.state^2 * copper.area)/2`. In patch notation, let `e` denote the
explicit electric unknowns, `c` the independent chord fields, `Kx` and `Kc`
their circulation operators, and `Lh = dual/(mu0 * area)`. The modified
explicit electric mass is `Me - dt^2 Kx.T Lh Kx/4`, and the magnetic field is
shifted by
`Hhat = H - dt Kx e/(2 mu0 area) - dt Kc (c + c_prev)/(4 mu0 area)`.
The on-sheet magnetic flux is recovered in its generated curl range, rather
than counted as a nonexistent runtime field: with
`B = Kc_sheet.T sqrt(Lh_sheet)`, solve for the minimum-norm `z` in
`B z = chord_f - Kc_bulk.T (dual H) + dt Kc_bulk.T Lh Kx_patch (e_new-e_old)/4`.
Its modified contribution is
`||z - dt sqrt(Lh_sheet) Kc_sheet (c+c_prev)/4||^2/2`.
The device implementation prepares a sparse range factorization once. It keeps
**all forcing rows** and evaluates the full-row minimum-norm least-squares map
`B.T (B B.T)^+ f`. Specifically, `J` selects independent face columns (with a
symmetric minimum-degree permutation), `A = B[:,J]`, and
`A.T A = L L.T`. Each sample computes `g = A.T f`, followed by the two
triangular solves `L y = g`, `L.T x = y`. Scatter `x` into face order, then
project out the normalized nullspace vectors of closed components. Thus the
column-range evaluation is algebraically identical to the full-row form;
it never restricts `f` to independent chord rows. The matvec precedes the
solves in this equivalent formulation. Sparse symmetric LU with numerical
pivoting disabled gives `LDL.T`; positive diagonal pivots are absorbed into
`L`, yielding fp64 Cholesky factors without a new sparse-factor dependency.

Rank revelation uses the sheet incidence structure (at most two nonzeros per
chord row). A one-face row anchors its connected face component. Otherwise,
a spanning tree constructs a candidate weighted constant null vector, all
rows verify it, and one face column is pinned when that null vector exists.
The final orthogonal projection selects the minimum-norm solution even when
the forcing is inconsistent. Both `L` and `L.T`, the column index, and `A.T`
are stored as device tensors, with CSR factors and fp64 values.

Each sample runs both triangular solves with `torch.triangular_solve` on the
sparse CSR factors (cuSPARSE on CUDA, MKL on CPU), on torch's current stream,
with no new dependency and no alternative route. On CUDA, the forcing matvec
and nullspace projection use explicit indexed gathers and fixed-width row
sums. Padding exists only in sample scratch; row widths are host metadata
computed at prepare. This avoids dependence on the library default sparse
matvec algorithm. There are no per-sample host reads (checked under
`torch.cuda.set_sync_debug_mode("error")`), convergence tests, warm starts, or
data-dependent loop counts, and repeated samples are `torch.equal`. Sampling
stays outside the captured field-update block.

At steps 20, 40 and 80 of source-free, seeded native-fp32 CPU controls:

| Case | `B` shape | Dependent rows / columns | Relative consistency residual range | `nnz(L) / nnz(B.T B)` (fill ratio) |
| --- | --- | --- | --- | ---: |
| small tilted patch | `(170,72)` | `98 / 0` | `7.2130e-4 .. 1.9671e-3` | `256 / 292` (0.877) |
| sheet cavity | `(674,310)` | `364 / 0` | `1.0530e-3 .. 2.8958e-3` | `1708 / 1406` (1.215) |
| mixed planar foil + sheet | `(18,6)` | `12 / 0` | `2.6912e-3 .. 1.9823e-2` | `10 / 14` (0.714) |
| closed prism loop | `(300,120)` | `180 / 0` | `4.6765e-4 .. 7.1414e-4` | `548 / 480` (1.142) |
| weighted synthetic closed loop | `(4,3)` | `2 / 1` | consistent and explicitly inconsistent forcings | `3 / 9` (0.333) |

The forcing is **inconsistent**, well above fp64 roundoff, so the full-row
least-squares case applies. The largest relative z error is `1.60e-14` against
host LSQR with `atol=btol=1e-15`; using `1e-13` in the raw-z oracle can stop
`3.25e-12` short on an inconsistent cavity state. The production host energy
oracle, including its existing `1e-13` LSQR tolerances, is unchanged. Plain
and modified energy errors are at most `6.53e-16` relative in these controls.
Three repeated samples are `torch.equal`, also under the existing host-read
prohibition. The physical closed-loop case tests dependent chord rows; the
synthetic weighted loop additionally tests a nontrivial column nullspace.
No iterative fallback or additional resume state is needed.

At the sampling point after completed step `n`, electric traces, chord fields
and copper modal states are at `n`; native/split magnetic fields are at
`n-1/2`, and `c_prev` is at `n-1`. Outside sheet patches the modified native
term is the Yee sum minus `dt * <H, curl E>/2`, using that same pairing.
Planar foil scenes without sheets retain the plain staggered sum: it can
oscillate within a bounded band during passive electric/magnetic exchange,
which is why the guard fits window maxima rather than individual increments.
Scenes containing both a planar CopperFoil and a separate thin sheet use the
modified functional throughout. In addition to the foil's electric mass swap
and modal storage, replace its native magnetic cross term by the split-trace
curl. The correction is
`sum(mu0 * V_H * H^(n-1/2) * delta_H(E_left^n, E_right^n))/2`,
where `delta_H` is the existing foil magnetic stencil, including its time-step
factor, evaluated at the current trace difference. It includes both the
tangential high-face correction and the adjacent normal-H corrections; the
begin-step `old_e` snapshot is not used. This equals
`-dt * sum(V_H * H^(n-1/2) * delta_curl(E^n))/2`.
Native Yee widths and sheet energy widths have separate caches. A mixed
scene uses the sheet's half-endpoint widths consistently for its base field
integral, foil electric replacement and foil magnetic cross correction.
An exact 300-step closed CPU cavity control with both conductor owners,
passive canonical copper modes and float64 prepared tensors checks lossless
and lossy `energy + cumulative copper heat` balance at `1e-13` relative.
CPU controls cover a lossless and loss-tracked sheet copper resonance
(`energy + cumulative heat` constant to accumulated float64 round-off), a
plain planar-foil resonance with bounded oscillations and decreasing window
maxima, and agreement of both sheet energy forms with the qualified host
functional. These controls do not qualify GPU execution or long-horizon CPML
growth detection with copper/sheets; those still require GPU verification,
including sparse flux-recovery cost and eager/graph sampling.

### Guard storage bytes and bounded capture check

Storage is reported as deterministic tensor bytes, not elapsed time. Let
`n_x,n_y,n_z` be cell counts **including exterior PML cells** (the solver's
`Nx,Ny,Nz` are `n_x+1,n_y+1,n_z+1`). Define
`W = 8 * (2*(n_x+n_y+n_z) + 3)` bytes. These are referenced storage bytes;
Python objects, allocator metadata and cached/free blocks are excluded.

| Guard owner | dtype and shape | Lifetime | Bytes |
| --- | --- | --- | --- |
| `_passivity_widths[(axis,"primal")]` | float64, `(n_axis,)` | persistent | `8*n_axis` per axis |
| `_passivity_widths[(axis,"dual")]` | float64, `(n_axis+1,)` | persistent | `8*(n_axis+1)` per axis |
| `_passivity_state["history"]` | float64, `(16,3)` | persistent | 384 |
| `peak`, `window_max` | float64, scalar each | persistent | 16 |

Thus planar copper guards own exactly **`W + 400` bytes**. Thin sheets and
mixed scenes additionally retain the sparse energy cache detailed below.
No dense native-domain volume array, previous native field bank, or additional copper
modal bank is retained. Widths scale with axis lengths, not volume.
For total persistent bytes `G`, the average bytes per computational cell
are `G/(n_x*n_y*n_z)`; for `F` sheet faces they are `G/F`, rather than a
fixed per-face charge. Guard off owns zero sampling bytes: sheet diagnostic
storage is initialized lazily by the guard at run start, or by an explicit
device-energy diagnostic request. Repeated sampling reuses those caches.
On CPU `as_tensor` aliases the scene's float64 NumPy widths; the inventory
counts their storage once, even though it does not newly allocate those host
arrays. On CUDA the six width arrays are device copies.

The energy sampler retains the following sheet diagnostic tensors and reads
existing stepping state. Rows marked **shared** are required for stepping
and add no guard-owned bytes; rows marked **sampling** are allocated lazily
and included in the guard inventory:

| Owner | dtype and shape | Referenced bytes |
| --- | --- | --- |
| shared planar foil `state` | float64, `(M_b,T_b)` in each real/imaginary entry `b` | `8*sum_b(M_b*T_b)` |
| shared foil `right_e`, native left E | field dtype, `(T_b,)`, left E is an indexed gather | `sizeof(field)*T_b` for existing right trace; no persistent left copy |
| shared foil `capacity`, `area`, `admittance_scale` | float64, `(2,T_b)`, `(T_b,)`, `(T_b,)` | `32*T_b` per distinct coefficient bank (complex entries share coefficients) |
| sampling sheet `_thin_sheet_energy_widths` | float64, six axis arrays as above | `W`, half-endpoint dual widths |
| sampling sheet `energy`: `e_mass`; `h_mass`, `dual`, `lh` | float64, `(E,)`; three `(H,)` | `8*(E+3*H)` per patch |
| sampling sheet `energy`: `K`, `K_patch` | CSR, `(H,E)`, `(H,P)` | `C(K)+C(K_patch)` |
| sampling copper-sheet `energy`: `Kc`, `KcT`, `Ks` | CSR, `(H,C)`, `(C,H)`, `(S,C)` | sum of `C` over these three operators |
| sampling flux `L`, `LT`, `BT` | CSR, `(R,R)`, `(R,R)`, `(R,C)` | `C(L)+C(L.T)+C(A.T)` |
| sampling flux column index | int64, `(R,)` | `8*R` |
| sampling flux nullspace and transpose, when present | CSR, `(S,S-R)` and `(S-R,S)` | `C(N)+C(N.T)` |
| sampling copper-sheet `energy`: `root_lh` | float64, `(S,)` | `8*S` |
| shared sheet `copper.state`, `copper.area`, `copper.mass` | float64, `(M,T)`, `(T,)`, `(Q,2)` | `8*(M*T+T+2*Q)` |
| shared sheet `chord_e`, `chord_e_prev`, `chord_f` | float64, each `(Q,2)` | `48*Q` |
| shared sheet `e_new`, `e_old` | float64, each `(P,)` | `16*P` |

Here `E,H` count all local explicit electric/magnetic unknowns, `P` counts
patch electric unknowns, `Q` counts chord pairs, `C=2*Q`, `S` counts sheet
magnetic face unknowns, `T` counts copper trace sites (including promoted
sites), `R` is the flux rank, and `M` counts modes. For a CSR operator with `r` rows and `k`
stored entries, **`C(A)=i*(r+1+k)+8*k`**, with `i=4` bytes if
`k < 2**31-1`, otherwise `i=8`. Index dtype is int32/int64; values are
float64. There is no universal per-sheet-face constant for these sparse
operators: connectivity and promoted chords determine `k`. Shared stepping
owners persist even with the guard disabled; their modal banks and histories
are never copied by sampling. Sampling caches persist after first use.
Let `D` be the sum of sampling rows in this table. On CUDA, full persistent
guard storage is **`G = W + 400 + D`**; without sheets `D=0`. Multiple sheets
sum their operator costs and share one final energy-width cache. On CPU the
sheet primal widths alias native primal widths, giving
**`G_cpu = W + 400 + D - 8*(n_x+n_y+n_z)`** when sheets are present.
Coefficient aliases and views are counted only once.

Per-sample scratch has the following deterministic element sizes/shapes,
and is released after reduction (the returned detached energy scalar is
float64, 8 bytes). It is not persistent guard state or a peak-memory claim:

| Scratch family | dtype, shape | Bytes per tensor |
| --- | --- | --- |
| native E/H double casts, squares, mass/width products | float64, corresponding Yee component shape | 8 per component location |
| modified native E casts and three circulations | float64, each E shape and corresponding H shape | 8 per location |
| foil left/right gathers, mass, trace difference, magnetic corrections | float64, `(T_b,)` or sparse magnetic support length | 8 per edge/support entry |
| foil modal square/product | float64, `(M_b,T_b)` | 8 per modal entry |
| mixed-foil decoded magnetic indices | int64, support length | 8 per index per axis |
| sheet `e,h,gx,shift,dx` and products | float64, `(E,)`, `(H,)`, `(H,)`, `(H,)`, `(P,)` | 8 per corresponding unknown |
| chord current | view `(C,)` | no new storage |
| chord pair, forcing | float64, `(C,)` | `8*C` each |
| triangular RHS and output | float64, `(R,)` | `8*R` each |
| CUDA fixed row sum offsets/column gathers, validity masks, value gathers/products | CSR index dtype, bool, float64, each `(r,w)` | `sizeof(index)*r*w`, `r*w`, `8*r*w` per corresponding tensor |
| recovered flux and shifted flux | float64, `(S,)` | `8*S` each |
| nullspace coefficients when present | float64, `(S-R,)` | `8*(S-R)` |
| sheet modal squares/products | float64, `(M,T)` | `8*M*T` each |
| reduction arithmetic scalars | float64 scalars | 8 each |

For `E_a`, the Yee component has length `n_a` along its own axis and
`n_b+1,n_c+1` along the transverse axes. For `H_a` those lengths are
`n_a+1,n_b,n_c`. Thus every native scratch tensor's exact cell-count shape
and byte total follow from these staggerings.
For CUDA row sums, `r` is the operator row count and `w`
is its maximum row width, fixed at prepare. Width metadata adds no device
tensor bytes.
An existing float64 input's `.double()` cast aliases its storage; only casts
from narrower fields allocate a new 8-byte-per-entry bank. Reshape and width
views also allocate no tensor storage.

Complex native squared magnitude additionally uses a second component-sized
cast/square and their sum. Sparse matvecs create output-length vectors;
expression intermediates have the same shapes as their operands. Window
roll scratch is `(16,3)` float64 (384 bytes), and a complete fitted window
copies those 384 bytes to CPU. No per-sample scratch survives in the guard's
owned inventory. Sparse backend workspaces, graph pools and allocator
rounding are separate from tensor storage bytes.

The exact marginal totals below use `thin_sheet_overhead.SCENARIOS` and the
planar `_simulation` grid in `test_copper_fused_kernel.py`. The CPU test
resolves their actual grid widths and inventories the resulting guard state;
it does not prepare the large board eigensystems or run those scenes.

| Scene / boundary | Computational cells `(n_x,n_y,n_z)` | Analytic = measured common bytes `W+400` |
| --- | --- | ---: |
| `copper_patch`, original all-PML grid | `(148,148,148)` | 7,528 conditional |
| `pec_board`, original all-PML grid | `(148,148,148)` | 7,528 conditional |
| `copper_board`, original all-PML grid | `(320,280,100)` | 11,624 conditional |
| `copper_board_60k`, original all-PML grid | `(240,220,100)` | 9,384 conditional |
| `copper_patch`, guarded z=PEC | `(148,148,128)` | 7,208 |
| `copper_board_60k`, guarded z=PEC | `(240,220,80)` | 9,064 |
| planar foil, x=PML, y=PMC, z=PEC | `(144,4,4)` | 2,856 |

The original all-PML topologies are **unguarded**, so their actual marginal
guard bytes are zero. Their conditional rows price the common part of enabled
sampling at that grid size; sheet sampling adds `D` from the sparse model.

The updated guarded-case inventories include the sparse factors. Measurements use the actual cut complexes and
host diagnostic tensors; CUDA values account for distinct width copies.
The 400 bytes of envelope state are included in every full total.

| Guarded case | CPU analytic = inventoried bytes | CUDA raw analytic bytes |
| --- | ---: | ---: |
| `copper_patch`, z=PEC | 33,473,664 | 33,477,056 |
| `copper_board_60k`, z=PEC | 219,257,496 | 219,261,816 |
| planar foil | 2,856 | 2,856 |

The 60k board has `B.shape=(121974,60638)`, `nnz(B)=242544`, rank 60,638,
61,336 dependent rows and no dependent columns. Its normal matrix has
301,778 entries, and `L` has 1,524,772 entries: fill ratio **5.05263** relative
to that normal matrix. The complete flux plan occupies **40,717,828 bytes**,
including `L`, `L.T`, `A.T` and the column index. Fill is practical, so no CG
fallback is enabled. The patch plan has rank 6,002, `nnz(L)=73,878` and
occupies 2,180,716 bytes.

Measured guard cost (GPU 1, exclusive, `guard_cost_ab.py --steps 2048`, five
A/B repeats after an excluded warm-up, CUDA graph capture active in both arms,
sampling every 32 steps):

| Case | Sample (ms) | Plain step (us) | Amortised fraction | Measured run overhead | Prepare on/off |
| --- | ---: | ---: | ---: | ---: | ---: |
| `copper_board_60k`, z=PEC | 67.31 | 1586.2 | 1.326 | 1.323 | 0.98x |
| `copper_patch`, z=PEC | 10.49 | 693.1 | 0.473 | 0.472 | 0.99x |
| planar CopperFoil, electric point source | 1.46 | 118.1 | 0.387 | 0.415 | n/a |

The sheet sample is dominated by the two sparse triangular solves, which are
latency-bound by the factor's dependency depth (two right-hand sides cost the
same as one). The planar foil has no sheet flux; its sample cost is the eager
launch sequence of the energy reduction on a 2,304-cell grid. The fixture's
closed-transverse plane wave is a magnetic surface source that keeps the field
update eager, so the cost script uses an electric point source with the same
foil, grid and boundaries.

#### Guard cost and the v0.6.0 copper/sheet skip

These costs fail the 2% amortised criterion (5% hard) by 19x to 66x. Sparser
sampling was tried and rejected. One sample per 2,048-step window would cost
2.07% / 0.74% / 0.60%, but point samples of an oscillating CW or LC energy
replace the 64-sample window maximum and inflate the fit standard error 4-60x.
The 3-sigma term then crossed `4e-7` on non-growing controls in
`test_passivity_gpu.py`:
- the CW modulated-slab no-trip control raised `FDTDPassivityError` at step 92,160 (slope 4.25e-7, stderr 2.18e-6);
- the 60-degree Bloch LC control warned at step 86,016 (slope 1.04e-5, stderr 3.26e-6);
- the below-resolution complex fixed-k control warned at step 98,304.

At 32 steps all three record no event. User decision (2026-11-06): in v0.6.0,
guarded scenes containing CopperFoil or thin sheets report
`passivity_guard = "skipped: energy storage not counted (CopperFoil, thin sheets)"`
with the `RuntimeWarning`, at zero sampling cost. All other guarded scenes keep
32-step sampling. The counted functional, byte model and flux solve stay
qualified by `test_guard_storage*.py` and `test_sheet_flux_solve.py`. Those
tests, and `guard_cost_ab.py`, empty `DEFERRED_STORAGE` to exercise it.

Follow-up (0.6.1 backlog): capture the energy sampler in a CUDA graph, which
removes the planar foil's launch overhead. Replace the latency-bound sparse
triangular solves with a deterministic, analysis-free, graph-capturable
sync-free triangular solve, or a lower-depth flux formulation. Then return
copper/sheet scenes to dense 32-step sampling at <= 2% and empty
`DEFERRED_STORAGE`.
The larger `copper_board` geometry has not been re-inventoried with this
factorization; apply the sparse table instead of its obsolete LSQR totals.
The PEC board has no recovered copper flux, so its unchanged original-grid
CPU/CUDA totals are 50,538,048 / 50,541,600 bytes when sampling is requested.
Original all-PML topologies remain unguarded and retain zero guard bytes.

For the guarded patch the raw CUDA bytes per computational cell are exactly
`33,477,056 / 2,803,712`, and for the guarded 60k board
`219,261,816 / 4,224,000`. Their sheet-face counts are 6,002 and 60,638.
These ratios include common state and all sampling operators.

The CPU fixture checks use `(18,18,22)` cells: copper measures
**1,352 = 1,352**, sheet and mixed each measure **208,504 = 208,504** full
guard bytes, versus zero off. The sheet cache alone is **207,616 bytes**,
with 464 bytes of primal-width aliases excluded from the combined inventory.
On CUDA its width copies are distinct: the expected raw totals are 1,352,
208,968 and 208,968 bytes. These match the independent geometric CSR model;
planar copper retains zero sheet diagnostic bytes.

`tests/boundaries/cpml/test_guard_storage_bytes.py` deduplicates underlying
storage pointers and counts whole storage `nbytes`, including CSR components.
Its reviewer-owned CUDA tests run only 64 steps per arm (two samples), check
capture remains active with and without the guard, and assert a 120 s budget
per case, with a traceback-and-exit watchdog at the same limit. The CUDA
allocation delta brackets state construction on prepared
solvers, before capture/sampling, so graph pools and scratch are excluded.
For raw tensor allocation sizes `b_j`, the rounding rule is 512 B blocks:
`sum(b_j) <= allocated_on-allocated_off <= sum(512*ceil(b_j/512))`.
The raw owned inventory must still equal `sum(b_j)` exactly after the run.

The previous sampler launched `4*S` LSQR iterations at every energy sample,
even after convergence. For `S=60,638`, this was 242,552 iterations per sample.
The prepared factorization removes that face-count-scaled iteration loop.
Guard sampling remains outside the captured field-update block.

CPU cProfile of the guarded z=PEC 60k board (45-minute traceback watchdog,
CUDA hidden) measured the following guard-specific costs. Full preparation
includes the CPU reference backend's geometry and spectral work. A separate
host-cache profile uses the same compiled geometry and vacuum material data,
without repeating the field allocation or eigensystems.

| Profile | Guard-specific seconds | Top cumulative entries (seconds) |
| --- | ---: | --- |
| original full CPU prepare + lazy guard setup | 1.597 | repeated side materials 0.836; energy setup 0.548; sparse reconstruction 0.243; native volumes 0.242 |
| new full CPU prepare + lazy guard setup | 3.156 | flux preparation 2.684; levels 0.915 (since removed); factor/rank preparation 0.771 |
| same host-cache data, original setup | 1.408 | repeated side-material validation and sparse reconstruction |
| same host-cache data, factored setup | 3.947 | includes factorization, both dependency schedules and the complete cache |

Original full guard-off preparation took 721.690 s (guard-on 723.287 s,
ratio 1.00221). Its common costs were spectral solves 471.857 s and sheet
compilation 211.340 s; those are outside guard-specific work. Reusing validated
host permittivity and `K` removes the repeated side-material validation and
bulk sparse reconstruction. The new factorization adds bounded prepare work
instead of repeating a solve iteration loop during sampling. The host-cache
profile shows this added factor setup explicitly; it is not a claim that
factor preparation is free or that the old GPU prepare slowdown reproduced
on CPU. Full after-change preparation measured **373.880 s off**, **377.036 s
on**, ratio **1.00844**, satisfying the 1.2x CPU prepare target. The original
full profile left BLAS threading at its environment default; the after profile
pinned BLAS and torch to one thread. Machine load also varied. Therefore the
common before/after prepare totals are not an optimization speedup claim;
the relevant gate is each profile's own on/off ratio. The same-host-cache
before/after pair used one thread throughout. cProfile artifacts are retained
at `/tmp/sheet_prepare_off.prof`, `/tmp/sheet_prepare_guard.prof`,
`/tmp/sheet_prepare_after_off.prof`, `/tmp/sheet_prepare_after_guard.prof` and
`/tmp/sf_copper_board_60k_{before,after}.prof`.

For the separate amortised-cost criterion, `scripts/dev/fdtd/guard_cost_ab.py`
uses 2048 steps by default, five AB repeats after excluded warm-up, CUDA
events, graph-active assertions in both arms, and one separately timed energy
sample per repeat. One-time guard/cache construction belongs to the prepare
phase, outside the timed run. It counts energy reductions before and after source
settling without altering the guard's start policy. Every prepare/run/sample
phase logs flushed progress. Prepare has a separate 1200 s traceback-and-exit
watchdog (`--phase-timeout-prepare`); run and sample default to 600 s;
JSON is replaced after each completed scene. The process allocator is capped
at 12 GiB by default. It reports sample/amortised/plain-step cost, fractions,
run overhead and median/min/max/absolute and relative spread; PASS is <=2%,
CHECK is >2% through 5%, FAIL is >5%. These are reviewer-run measurements on assigned GPU 1,
not timing evidence from CPU or shared GPU 0:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/dev/fdtd/guard_cost_ab.py --device-index 0 --steps 2048 --phase-timeout 600 --phase-timeout-prepare 1200 --memory-gib 12 --json /tmp/guard_cost_ab_gpu1.json
CUDA_VISIBLE_DEVICES=1 python -m pytest tests/boundaries/cpml/test_guard_storage_bytes.py tests/boundaries/cpml/test_sheet_flux_solve.py --gpu -v -s --junitxml=/tmp/guard_storage_bytes_gpu1.xml
```

Scenes whose runtime state stores energy outside that functional are **not
guarded**: thin wires, embedded networks, coaxial
elements, SIBC, dispersive/ADE and gyromagnetic media. Energy exchange between
that state and the fields could look like field-energy growth in a passive run.
Result metadata `passivity_guard` reports `active`, `inactive: ...` or
`skipped: energy storage not counted (...)`, or `warned: complex fixed-k growth`.
The adjacent `passivity_guard_warning` field is `None` before a growth warning,
then a structured record with `slope` (log energy per step), `stderr`, `window`
(start/end steps of the fit) and `step` (the window end when first warned).
Both fields are also in `Result.solver_stats`; solver attributes are
`passivity_guard_status` and `passivity_guard_warning`. A skipped run has no growth
detection and also emits a `RuntimeWarning` ("passivity guard skipped in a
topology with known CPML growth: ...") so it is never silent.

Scope decision (2026-11-03): embedded networks and SIBC stay **permanently**
skipped with the warning, because a passive storage functional for general
networks or the recursive-convolution SIBC is not available. CopperFoil and thin
sheet storage are counted by the functional (v0.6.0), but guarded scenes that
contain them stay skipped with the warning in v0.6.0 (`DEFERRED_STORAGE` in
`fdtd/runtime/passivity.py`; see "Guard cost and the v0.6.0 copper/sheet skip"
below). For
v0.6.0, dispersive/ADE and gyromagnetic media, thin wires and coaxial elements
stay skipped with the warning; counting their storage is a post-release item
(user decision 2026-11-04).
No full field is copied to the host. The peak includes source excitation;
pre-check samples and partial-window maxima stay on device.

For pulses, window accumulation starts at the latest compiled source settling
step across all sources, port excitations and circuit drives. Gaussian settling
is `delay + 6 sigma_t` (envelope <= exp(-18) of peak); Ricker settling is
`delay + 4/f`; custom tables are exactly zero beyond their last tabulated time.
For ongoing/CW sources and callables with unknown support, accumulation starts
after both the latest known settling time and the configured transient:
`max(transient_cycles / reference_frequency, 10 * domain_extent / c)`.
A callable's end cannot be inferred; supply a finite table for a guaranteed
source-off criterion. This transient applies even with spectral window `"none"`.

Let each **64-sample window (2,048 steps)** contribute its energy maximum
`M_i`. On every complete window, fit the latest **N = 16 complete windows** by
ordinary least squares: `log(M_i) = intercept + slope * step_i`, using the
window-end steps. The slope standard error is
`sqrt(sum(residual^2) / (N - 2) / sum((step_i - mean(step))^2))`.
Raise `FDTDPassivityError` for real fields, or warn once for complex fixed-k,
when all three conditions hold:

- `slope + 3 * standard_error > 4e-7` per step;
- the latest maximum is strictly greater than the earliest maximum in the fit;
- the latest maximum is strictly above `1e-10 * sampled_peak_energy`.

Non-finite energy in the fitted windows is a blow-up, not measured slow growth: it raises
`FDTDPassivityError` in every topology, including complex fixed-k cells that already warned.

**Growth slower than 4e-7 per step (about 1.5x energy over 1e6 steps) is not detected.**
This is the guard's finite growth-rate resolution, not a certification of
long-record stability below that rate. The three-standard-error term remains
part of the decision and accounts for uncertainty in a finite noisy fit.

Nonfinite window histories also raise for real fields and warn once for complex
fixed-k (recording infinite slope and stderr). The first fit needs 32,768 eligible
steps; shorter records cannot establish this bound. Device samples are reduced
to maxima before transfer, with one host synchronization per complete-window
check. This ports the LC acceptance fit in
`scripts/dev/fdtd/bloch_lumped_acceptance/common.py::assert_passivity` to a
bounded rolling history. It admits decay and beating whose window maxima are
bounded at this resolution; it cannot distinguish every arbitrarily slow beat
from growth in a finite record. Small-signal tails below the floor are ignored.
Checkpoint format v6 preserves the peak, rolling history, partial maximum,
sample/window counts, topology and checking start. D2 adds the `warning` state
field (`None` or the structured record above) without bumping schema version 6.
Checkpoint/resume preserves a warned state and its original measurements, so
the warning is never re-emitted after resume. Missing eligible guard history,
including the new state field, is rejected by the existing exact-key check.

The exception or warning names the topology, measured slope and uncertainty, and gives
the applicable workaround:

- Real-field closed-transverse: `absorber="auto"` or graded `"absorber"` with
  >= 40 layers. Explicit CPML remains available for short qualified records.
- Complex fixed-k: CW per frequency or short pulsed records. The graded
  absorber is **not** an accuracy-qualified replacement (measured r/t errors
  of 0.2–0.5). The current shifted CPML can still grow for guided TM content
  below the vacuum light line, measured at +2.5e-6/step log-energy slope.

Additional measured fixed-k log-energy rates motivating warn-only behavior:

- **2.1e-7/step**: point-dipole eps4 slab (below the guard's growth resolution).
- **8.2e-7/step**: eps4 TM slab CW, `k_parallel = 0.25 k0`, `0.9 f0`, resolution 160.
- **3.5–5.0e-7/step**: eps4 TM CW parity at `1.1 f0`, continuity at 20 degrees.

These are recorded numerical observations, not a passivity certificate. The
finite-window uncertainty term still participates in the warning decision.

GPU qualification cases are in
`tests/boundaries/cpml/test_passivity_gpu.py`: real eps3 TM growth, complex eps4
fixed-k TM with a cutoff pulse, open-PML dipole, automatic real periodic slab,
60-degree Bloch LC with the current default profile, CW-driven
`modulated_slab` with explicit CPML, and graph/checkpoint identity.
The CUDA-marked eps4 TM CW test uses the measured `0.25 k0`, `0.9 f0`,
resolution-160 cell and requires completion with one warning and matching
solver/result metadata; it does not replace the below-resolution dipole control.
The CW must-trip case keeps the benchmark grid and material/source definition,
uses a 1 us configured transient, and runs up to 7.399178367515743 us. At the
reported 16.667 ps step, a 16-window fit spans 0.546 us: the first fit occurs
near 1.546 us, and a full fit after the reported growth onset near 2.3 us
fits within the horizon. No guard threshold or window constant is overridden.
It writes `cw_modulated_slab_passivity.json` under the pytest temporary
directory and records its path, trip step/time, and onset estimate as JUnit
properties. The onset estimate is the first complete-window maximum more
than 0.1% above the maximum of the first 16 post-transient windows.
The artifact also contains the reported 2.3 us onset for comparison and all
observed envelope windows; this finite-window onset definition is separate
from the guard's trip rule. These fixtures require GPU review; CPU tests certify the resolver
and detector decisions, not numerical passivity or reflection accuracy.

## CPML profiles


Nonzero fixed-k Bloch runs use complex-frequency-shifted CPML by default. The
maximum electric alpha is `2*pi*epsilon_0*f_min`, where `f_min` is the lowest
requested simulation frequency. Thus alpha/epsilon_0 is an angular-frequency
relaxation rate on the simulated band scale. The magnetic alpha is scaled by
mu_0/epsilon_0 so both field updates have the same physical shift. An explicit
`cpml_config['alpha_max']` overrides this choice, including zero.

Sigma, kappa and alpha use the same physical layer depth, evaluated from the
float64 node coordinates for E and actual Yee half-point coordinates for H on
both sides. Sigma and kappa follow the configured polynomial grading; alpha
complements the same polynomial depth. Its outer-boundary fraction is
`min(1, 2*c*|k_parallel|/(2*pi*f_min))`, retaining memory relaxation on twice
the fixed-k light-line scale while capping alpha at its band-scale maximum.
Conductivity is multiplied by
`1 + (alpha_max/(epsilon_0*2*pi*f_min))^2`
to bound the shift-induced loss of attenuation throughout the band. Automatic shifted
profiles use quadratic conductivity grading. A propagating ray's layer return
scales as `R**cos(theta)`, so the automatic nominal reflection is
`10**(-4.2/cos(theta_edge))`, capped at eight decades of grading near grazing.
Here `theta_edge` is the propagation angle at the lowest requested frequency.
The 0.2-decade margin leaves room under the `1e-4` return budget; limiting excess
stretch also avoids unnecessarily trapping evanescent content. Explicit grading
and reflection settings are retained.
The high-side magnetic path retains the physical-half-point correction from
`e2a10746`; it does not grade at the adjacent outer node.

Shifted profiles match the static stretch and first time moment of the continuous
convolution. The memory pole uses the positive `[2/2]` Pade approximation of
`exp(-u)`, with `u=(sigma/kappa+alpha)*dt/vacuum_constant`; the curl feedthrough
and convolution coefficient are adjusted to preserve both moments. This avoids
the held-current exponential's time-moment error in stiff layers. The existing
one-memory CUDA update is unchanged; its kappa buffer stores the reciprocal of
the adjusted feedthrough, rather than the physical polynomial kappa. Explicit
zero alpha and all real-field paths retain their original coefficients.

Lossless LC acceptance checks the one-step radius and bounded total field-plus-LC
energy, as described in [the bound-mode note](../memory/bloch-bound-mode-passivity-2026-10-31.md).
R, RLC and static conductivity retain the strict 120,000-step source-off decay
threshold of 1e-6 for both field-energy and current-squared tail/peak ratios.
Both are energy-like quantities; using 1e-12 for current squared instead would
demand three orders stricter amplitude decay than the field-energy gate.
All 20 small-cell spectra, four LC energy records, all lossy decay records,
15 reflection points, 12 carrier-CW power balances and the regression checks
pass. The 60-degree RLC field/current-squared ratios are 3.71679e-9/4.23959e-9
at 120,000 steps, as recorded in the bound-mode note.

## Legacy `alpha_max = 0` is not passive for complex fixed-k runs (2026-11-01)

The legacy unshifted profile (`cpml_config={"alpha_max": 0.0}`) remains an explicit
opt-in. For complex fixed-k runs it is **not passive**, even without lumped
elements. Measured one-step amplification radii `rho - 1` of the small loaded
Bloch cell (`scripts/dev/fdtd/bloch_lumped_acceptance/one_step_spectrum.py --alpha 0`;
the float32 passivity floor is 2.38e-7):

| load | 0 deg | 10 deg | 30 deg | 60 deg |
|---|---|---|---|---|
| open | 4.3e-8 | 4.4e-8 | 1.31e-6 | 6.9e-8 |
| static conductivity | 4.2e-8 | 5.2e-7 | 4.3e-6 | 1.29e-5 |
| R / LC / RLC | 4.3e-8 | 2.9e-6 | 2.4e-5 | 7.4e-5 |

At 1.3e-6 per step, field energy grows about 1.4x over 120,000 steps, so keep the
legacy pin to short frozen-baseline comparisons. Smaller default shifts were also
measured. The default shift (`2*pi*epsilon_0*f_min`) is the only one tested that
passes the 120,000-step LC energy gate at 60 degrees: half of it gives a late
log-energy slope of +7.7e-7 per step against the 4e-7 limit. The default costs
oblique TE transmission accuracy (4.4% at 35 degrees on the 12-layer copper-foil
test, which is a strict xfail). A single profile law for real-periodic,
PEC/PMC-closed and complex fixed-k cells is under study.

Evidence (scan logs, probes and gate logs) is archived in
`<evidence-root>/cfs-cpml-c2-2026-11-01/`. The related real-field
periodic CPML growth diagnosis is archived in
`<evidence-root>/instamax-line-modulated-growth-20261101/`.

## Hazard: slab-guided TM modes under complex fixed-k (2026-11-02)

Complex fixed-k Bloch runs with dielectric layers are **not passive** under the current CPML for
slab-guided TM modes below the light line. Until the passive closure (CPML-B) lands:
- keep pulsed records short, or use CW per frequency;
- do not use the graded absorber as a workaround here. It gives r/t errors of 0.2–0.5 in complex
  fixed-k runs.

Evidence:
- A TM εr 4 slab (k∥ 0.55 k0) driven by a pulse with below-cutoff content keeps growing after
  source-off: late log-energy slope +2.5e-6 per step, and +8e-7 with a 2x thicker CPML.
- The guided mode leaks about 0.013 into r/t at 0.6 f0. The leak does not change with resolution
  or record length, and a CW run at 0.6 f0 does not excite the mode.
- The CPML-B CPU model shows the same defect: the current closure fails 98 of 288 fixed-k slab
  cases.

The hazard predates the broadband TFSF-slab injection. Pulsed point and current sources were
already admitted under fixed k.

## Changed defaults for complex fixed-k Bloch runs

**Behaviour change (337cee2f, the Bloch A instability fix):** existing complex fixed-k Bloch runs with CPML change their prepared CPML coefficients and fields. They are not bit-identical to earlier trees; only real-field, k = 0 and non-Bloch paths are. Pin the legacy coefficients with an explicit `cpml_config={"alpha_max": 0.0}` (short runs only; see above). In fixed-k Bloch runs, beams, mode and custom-field sources and soft plane waves must be CW. TFSF-slab plane waves (complex fixed-k incident line), point and current sources may be pulsed.

Compared with runs made before this change, a nonzero fixed-k Bloch run with CPML
now differs in three automatic settings:
- nonzero `alpha_max`;
- conductivity grading of order 2 instead of 3;
- the angle-scaled nominal `reflection` above instead of `1e-6`.

Each explicit `cpml_config` key overrides its own default. `f_min` is the minimum of
all requested output frequencies: `Simulation` assigns the full requested list to
the solver before CPML initialization, in both the single-device and the
distributed paths.

## Known limitation

**v0.6.0:** passivity growth detection is not applied to closed-transverse CPML
scenes containing CopperFoil or thin sheets. Such runs report
`skipped: energy storage not counted (CopperFoil, thin sheets)` and warn; see
"Guard cost and the v0.6.0 copper/sheet skip" above.

Explicit real-field CPML and zero-k CPML retain their existing non-CFS profiles.
Automatic real-field closed-transverse selection now uses graded sigma as above. Strong evanescent content within a few cells of a
non-CFS boundary remains a limitation: an absorbing boundary is not a replacement
for enough separation from the near field. Making CFS a general default is a
separate user decision. The broadband fixed-k TFSF injection defect is tracked
separately in [the injection note](../memory/bloch-broadband-tfsf-injection-2026-10-31.md).

# Finite ThinWire instability diagnosis

Resolved on `coverage/thinwire-passive-fit` by a positive-real RL realization
and final-dt full-digital-band/current-charge-ADE certification. The diagnosis
below records the original failure; implementation and acceptance measurements
are in [thinwire-passive-fit.md](../assessments/thinwire-passive-fit.md).

2026-10-30; branch `coverage/thinwire-finite-stability`, runtime examined at
`671ce32217e04a17e4c84cc1b42703f423b27de8`. Diagnosis only: no implementation
changes under `witwin/`, no merge, no push.

## Verdict

The finite-conductor impedance fit can be **active outside its fitting band**.
The build accepts it because its certificate covers only an isolated segment's
current and ADE state, **omitting even the wire's own node-charge feedback**.
Adding the actual capacitive termination produces eigenvalues outside the unit
circle at the actual solver timestep. Neither a lumped port nor Maxwell-field
feedback is necessary. Float64 also exhibits the instability. The material-aware
timestep binding is correct for this reproducer.

Thus suspect (b) is confirmed, more specifically than field coupling alone:
the certificate misses current/charge/ADE instability before the Yee grid is
included. Suspects (a), (c), and (d) are not necessary causes. Float32 roundoff
can seed the unstable mode and affect its onset; changing fit order between
preparations changes the growth rate too.

## Reproduction on physical GPU 1

The committed script is self-contained: `_cube`, `_kwargs`, `_surface`, and `C0`
are inlined from the supplied scene reference. It does not import the absent
`benchmark.datagen.scenes`. Logical `cuda:0` inside the process is physical GPU 1
under the required mask. The initial availability check returned `True`.

Run each command separately from the worktree:

```bash
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_wire/run_gpu.py .worktrees/cov-wire scripts/dev/fdtd/thinwire_finite_dipole_repro.py 19 40 64 cu
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_wire/run_gpu.py .worktrees/cov-wire scripts/dev/fdtd/thinwire_finite_dipole_repro.py 19 40 64
```

The supplied wrapper sets CUDA build, CUDA toolkit, and import paths inside
Python; there are no additional assignments following `env`. The isolated
extension build directory is `<scratch>/witwin_cov_wire_cuda`. No prebuilt override,
shared extension directory, cloud resources, or GPU 0 were used.

Both runs used `dt = 3.5260295084320742e-12 s` and ran 8082 steps. The auto time
planner's complete horizon spans about 68 carrier cycles despite the requested
40 steady cycles; the CSV reports actual elapsed cycles at 2.4 GHz. Step numbers
are one-based. Current entries below are maxima during the preceding carrier
cycle; energy is instantaneous electric plus magnetic energy at the listed step.

| Step | Cycle | Finite current (A) | PEC current (A) | Finite field energy (J) | PEC field energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 237 | 2.006 | 3.08708e-3 | 3.08921e-3 | 1.72370e-13 | 1.72614e-13 |
| 1419 | 12.008 | 2.79964e-6 | 2.03622e-6 | 5.70627e-20 | 5.88362e-20 |
| 2009 | 17.001 | 2.86523e-5 | 4.24056e-8 | 1.90223e-20 | 4.79223e-22 |
| 2955 | 25.007 | 3.59826e-3 | 3.49408e-9 | 5.63448e-16 | 3.69115e-22 |
| 4018 | 34.002 | 8.03242e-1 | 3.53476e-9 | 1.33062e-11 | 1.29970e-22 |
| 4727 | 40.002 | 3.02311e1 | 3.51652e-9 | 3.87435e-8 | 1.55731e-22 |
| 8036 | 68.004 | 6.36723e8 | 3.54639e-9 | 9.63342e6 | 7.60519e-23 |

The finite run's current minimum occurs around cycle 12, after which it grows
exponentially. The explicit threshold `max(abs(I)) > 1 A` is first crossed at
**step 4062** (about 34.38 cycles). PEC never crosses it. The finite run selected
order 9 with isolated certificate `0.9999682797403308`. Model dt, prepared solver
dt, and solver dt after execution were identical.

PEC gave `max|S11| = 0.975`, maximum incident power `3.178e-7 W`. This particular
finite fit gave `max|S11| = 0.863` and maximum incident power `0.1719 W`, despite
catastrophic late-time current and field-energy growth. The originally reported
`|S11| > 1`/NaN is not required to detect the failure: a finite-band DFT ratio
can hide an instability predominantly outside its observation band. Do not use
the reflection coefficient alone as a boundedness test.

## Code evidence and timestep audit

- `witwin/maxwell/compiler/wire_impedance.py:342` configures an unconstrained
  rational excess-impedance fit with `proportional=False`, checked over a finite
  band. `:357` converts it to the shared bilinear state-space realization. A
  stable ADE pole set and in-band positive resistance do not imply broadband
  positive-real impedance.
- `witwin/maxwell/fdtd/wire_lossy.py:423` checks resistance at 24 in-band points;
  `:430` checks `_companion_spectral_radius`; `:440` accepts the order.
- `witwin/maxwell/fdtd/wire_lossy.py:335` constructs only a `(k+1) x (k+1)`
  transition for `[I; x]`. No node capacitance, incidence matrix, field, or port
  enters that certificate. The claim at `:313` conflates this isolated stability
  with passivity.
- `witwin/maxwell/fdtd/wire_lossy.py:214` implements the accepted companion.
  `witwin/maxwell/fdtd/wire.py:1011` feeds `q/C` back into its drive and `:1026`
  advances charge using the new current. That mandatory feedback is missing
  from the acceptance matrix.
- `witwin/maxwell/fdtd/wire_lossy.py:267` clamps reported AC resistance to
  nonnegative values. This is a monitor calculation; it cannot stabilize the
  recurrence or certify its out-of-band power balance.

The material-aware change `2e239cbc` is not a dt-ordering regression here.
`witwin/maxwell/fdtd/runtime/materials.py:148` binds the material-aware step during
material construction. `runtime/stepping.py:1734` builds materials before wire
initialization, `wire.py:887` tightens dt to the joint limit, and `wire.py:922`
builds the lossy model using that final dt. `runtime/stepping.py:1741` then builds
the material and boundary/update coefficients. Numerical equality before and
after execution confirms this ordering in the failing scene.

History agrees with the observed gap: `4313b128` introduced runtime ADE
consumption and its isolated certificate; `b6c187fe` explicitly documented the
out-of-band active branch and lack of a coupled passivity proof. The ThinWire
inventory in `FEATURE_LIST.md:314` describes the combined current/ADE
spectral-radius gate, not a current/charge/field stability certificate. No feature
inventory change is made in this diagnosis-only commit.

## Causal experiments

Scratch scripts and the captured coefficient tensors are under
`<scratch>/witwin_cov_wire/` (`analyze.py`, `spectrum.py`, `ablate.py`, `cpu_fit.py`,
`model.pt`); these are diagnostic artifacts, not runtime modifications.

**No port, soft excitation.** The identical finite dipole geometry was built with
`build_scene(19, 64, finite=True, port=False)` and a central `PointDipole` with
`polarization="Ez"`, width equal to the cell size, and a 2.4 GHz Gaussian pulse
of width 1.92 GHz. There were no port excitations. The run selected order 13,
band `[1.2e9, 4.8e9] Hz`, certificate `0.9994379522440712`, and the same dt.
Maximum current in successive 500-step blocks was:

| End step | Peak current (A) |
| ---: | ---: |
| 500 | 2.39594e1 |
| 1000 | 2.01186e1 |
| 1500 | 2.02189e4 |
| 2000 | 1.74779e7 |
| 3000 | 1.01473e13 |
| 4000 | 7.63755e18 |
| 5000 | 5.17544e24 |

The soft-source amplitude/coupling differs from the port drive, so compare
boundedness rather than amplitudes or onset times between these runs.

**Remove Maxwell feedback too.** Copying that prepared runtime's coefficients to
CPU and analyzing its homogeneous recurrence in float64 gives:

| Recurrence | Spectral radius |
| --- | ---: |
| Current/ADE only, node potentials suppressed | 0.99943919 |
| Full 38-segment current/charge/ADE graph, no field or port | 1.01241074 |
| Same charge graph, PEC current update | 1.0000000000000042 |

The small difference between the first entry and the reported build certificate
comes from assembling the installed length-scaled float32 coefficient tensors,
then computing eigenvalues in float64. Both are below 1. The coupled violation
is orders of magnitude larger than this rounding difference.

**A single segment suffices.** Use one segment with
`L = 3.9341541135939906e-10 H`, length `0.0014956749510020018 m`, radius
`0.12 * (1.6*C0/2.4e9/64)`, conductivity `5.8e7 S/m`, and endpoint capacitance
`C = 3.163380744419103e-14 F` on each end. This is the same segment inductance
and endpoint half-segment capacitance as the compiled dipole. Set external EMF
to zero and define `V = (q_tail - q_head)/C`. The exact update is

```text
I_next = ((L/dt - G/2)*I + V - Cs*x) / (L/dt + G/2)
x_next = Ad*x + Bd*(I_next + I)/2
V_next = V - 2*dt*I_next/C
```

Including `V` extends the gate's matrix by just one state. With the captured
order-13 fit, its largest eigenvalue is
`-0.0211129869955 + 1.0121905818731j`, magnitude **1.01241075273451**,
oscillation frequency **71.84264466 GHz**, versus Nyquist **141.802 GHz**.
The eigenvector residual after state scaling is `7.6e-15`. Similarity scaling
`x = Bd*y` and `V = (L/dt)*v` avoids misleading conditioning from SI units.

The actual branch transfer from the average current to loss voltage is
`Z_branch(z) = G + Cs @ solve(z*I - Ad, Bd)`. On the unit circle at that mode's
frequency its real part is **-2.75633 ohm**, versus **+0.00749209 ohm** at
2.4 GHz. The direct companion coefficient is `G = -2.8120811 ohm`, despite
physical segment DC resistance `R0 = 5.84518e-5 ohm`.

For the same single-segment termination and timestep:

| Loss branch | Spectral radius | Norm after 2000 steps from a 1e-12 eigenmode seed |
| --- | ---: | ---: |
| Installed ADE | 1.012410752735 | 3.97625e-2 |
| DC resistance only | 0.999999738060 | 6.78535e-13 |
| PEC | 1.000000000000 | 6.78891e-13 |

These are scratch ablations, not proposed production substitutions. The lossless
normalized frequency squared is `2*dt**2/(L*C) = 1.99801638`, safely below the
leapfrog limit 4. The instability is not merely an excessive lossless CFL step.

**Precision and fit variation.** Fresh all-float64 builds also accept unstable
charge-coupled models: selected orders 12 and 10 gave isolated radii 0.99826646
and 0.99959324, but coupled radii 1.00935320 and 1.00157137 respectively. A fresh
float32 order-11 build gave coupled radius 1.00583022. Float32 accumulation is
therefore not required. Fits varied between preparations; the shared fitter uses
`torch.linalg.lstsq` at `witwin/maxwell/rational.py:1036`. These experiments do
not establish the precise origin of that fit variation, only demonstrate it.

## Minimal strict-xfail test and validation

`tests/fdtd/thin_wire/test_wire_lossy_recurrence.py::test_finite_wire_charge_coupled_spectral_radius`
builds one float64 segment through the production builder, verifies that its
isolated certificate passes, and forms the small charge-coupled transition using
the production `advance_current`. It requires no grid, port, CUDA compilation,
time marching, or external reference solver. Only the physically necessary
charge difference is added; the constant total-charge mode is eliminated.
The test uses similarity-scaled states and checks `rho <= 1 + 1e-10`.

It is marked `xfail(strict=True, raises=AssertionError)` with the reason that the
isolated certificate omits unstable charge coupling. There is exactly one new
test. A normal run reported **1 xfailed in 2.83 s**. Disabling xfail proved the
failure is the final stability assertion, not construction or an unrelated
exception: **rho = 1.00330257171**, **1 failed in 3.02 s**. Exact values vary with
the fresh fit; there are no retries or fixed expected unstable eigenvalues.

```bash
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_wire/run_gpu.py .worktrees/cov-wire -m pytest -q -rx tests/fdtd/thin_wire/test_wire_lossy_recurrence.py::test_finite_wire_charge_coupled_spectral_radius
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_wire/run_gpu.py .worktrees/cov-wire -m pytest -q --runxfail --tb=short tests/fdtd/thin_wire/test_wire_lossy_recurrence.py::test_finite_wire_charge_coupled_spectral_radius
```

## Proposed fix (not implemented)

Replace acceptance of the unconstrained finite-band fit with a genuinely
positive-real conductor realization and a discrete energy/passivity argument
for its connection to the leapfrog wire/grid. One suitable direction is a
constrained passive RL/Foster approximation of the excess impedance,
`sum(r_k*s/(s+p_k))` with `r_k >= 0`, `p_k > 0`, plus exact nonnegative DC
resistance, using the shared fitting infrastructure. Preserve the native
PyTorch runtime and energy-paired sampling/deposition.

Certify the **realized discrete branch over the whole accessible digital
frequency range**, with its actual current time centering, rather than only
checking monitor-band resistance or homogeneous `[I; x]` poles. Preserve final-dt
binding. Reject an uncertified realization at prepare time. A charge-coupled
spectral gate is a useful immediate falsifier, but a per-segment isolated gate
alone cannot prove stability for every Yee/charge/port termination. Qualification
must include the joint energy balance and long-horizon pulse decay of the coupled
system.

Reducing dt, using float64, lowering fit order until only the isolated gate
passes, clamping the loss monitor, or changing the port does not address the
demonstrated active impedance. The work here intentionally implements none of
these changes.

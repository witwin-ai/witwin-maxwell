# Public port V/I hooks from caller-supplied fields (W3)

Status: spec accepted in principle by the coordinator, 2026-10-31 (conditions folded into the tasks). Owner: PCB line. Consumer: InstaMax NeuralSimulation (infra v2, W3).
Target: batch 3 or later.

## Goal

A caller holding frequency-domain Yee fields (from a neural surrogate or from a native
full-field DFT) computes port V and I with the same definitions that `Result.port(name)`
uses, without reimplementing the port runtime's time conventions. InstaMax deletes its
private `port_signals` / `fdtd_port_context` / `port_terms` once this lands.

## What native FDTD reports today (read from fdtd/ports.py, master 1f270da9)

| Port state | Runtime branch | V observable | I observable |
|---|---|---|---|
| Excited, terminated, circuit-bound or network-bound | `lumped` runtime | midpoint average of the edge line integral, (V[n-1] + V[n]) / 2, phase-referenced at (n - 1/2) dt | the branch current (negated: into the field), at (n - 1/2) dt |
| Open observer (no excitation, termination, circuit or network) | no `lumped` | instantaneous edge line integral at n dt | **the Ampère H-loop integral** (`integrate_current`) at (n - 1/2) dt |
| Wire-bound | wire provider | wire terminal coordinate, which includes the wire charge state | branch current, or 0 when open |

Note on the open case: the InstaMax note says native reports I = 0 there. That is not what
the code does. It reports the H-loop current, which includes the gap edge's own
displacement current. Also, `_open_declared_pec_terminal_edges` runs only for coupled
ports. An open observer port over a PEC-covered gap leaves the edge suppressed, so it reads
V = 0 (a short), while the same port with a load reads the gap voltage.

## Semantic ruling: open observer ports (revised after the task-1 census)

The first draft ruled that open observers should report midpoint V and I = 0. The census
**overturned it**. An unbound port with no excitation, termination, circuit or network is
used in-repo as a **non-invasive field probe**:
- 8 coaxial physics tests (tests/rf/lumped/test_coaxial_lumped_physics.py) extract line
  Zc, beta, end admittances and S predictions from open-probe V and Ampere-loop I, with
  1e-3 / 1e-4 gates against the discrete lattice reference;
- the PCB external-reference contour control (scripts/dev/fdtd/pcb_external_reference.py
  and 8 archived drivers) reads probe Z and the current balance.

I = 0 would destroy those measurements. Midpoint V would shift the probe Zc by
cos(omega dt / 2), about 4e-3 at ppw 20, against a lattice reference built from the exact
discrete functionals.

Revised ruling, with **no change to native outputs**:
1. **Two documented port roles.**
   - *Coupled port* (excited, terminated, circuit-bound or network-bound): V and I are
     the branch quantities, midpoint V and branch current on the half-step clock.
   - *Probe* (open, unbound): V is the exact edge line integral at n dt and I is the exact
     Ampere-loop integral at (n - 1/2) dt. Both are non-invasive discrete field
     functionals; the probe does not alter the scene.
   The two roles differ by design: a probe has no branch, so the branch-current
   definition does not exist for it. Mixing probe V with coupled-port V in one extraction
   carries the cos(omega dt / 2) difference. This is documented, never silent.
2. **A probe does not open PEC.** Being non-invasive, a probe over a PEC-covered gap reads
   V = 0 identically, which is a silent wrong answer. The fix is a **guard**, not a
   geometry change: preparation raises `ValueError` when every voltage edge of a probe is
   PEC-suppressed ("add a termination to declare a feed gap, or move the probe"), with a
   census entry. The census found no consumer that hits this.
3. PortData metadata records the role (`metadata["port_role"] = "coupled" | "probe"`).
   FEATURE_LIST and the port docs state both definitions.

## Public API

All new names live in `witwin/maxwell/ports_postprocess.py` (new module) and are exported
from `witwin.maxwell`. Signatures are to be coordinated with InstaMax before code.

```python
mw.port_voltage(prepared, port_name, fields) -> torch.Tensor
```
The exact field-side line integral of the runtime-resolved geometry, with no time factor.
`fields` maps Yee component names to complex tensors with a trailing frequency axis (or
none). For coaxial ports this uses the material-resolved discrete-harmonic row
(`harmonic_coaxial_geometry`), built from the prepared materials.

```python
mw.port_phasors(prepared, port_name, fields, *, frequencies, dt,
                drive=None, termination=None) -> tuple[torch.Tensor, torch.Tensor]
```
V and I under the native `Result.port` semantics:
- coupled ports: V = cos(omega dt / 2) * port_voltage(...);
  - excited: I = (drive.source_spectrum - V) / drive.source_impedance;
  - terminated: I = -V / conj(trapezoidal_impedance(termination, omega, dt));
- probes (open, unbound): V = port_voltage(...) and I = the Ampere-loop integral of
  `fields` (needs the H components), exactly as native.

`termination` overrides the scene port's termination (matched load = `SeriesRLC(r=z0)`).

Time labelling of `fields`: each component is a phasor labelled at its own true sample
time, exactly as `Result` field monitors give them (fdtd/runtime/spectral.py
`observer_stagger_weights`: E at n dt, H retarded to its (n +/- 1/2) dt sample). The
native port observer labels probe I at the true H sample time, (n - 1/2) dt, on the
same public origin. `port_phasors` therefore applies **no** E/H half-step shift: Result
fields in gives native `Result.port` out. `dt` enters only the coupled-port cos factor and
the trapezoidal impedance. If the proving case shows a residual exp(+/- i omega dt / 2),
that is an origin mismatch in the native observer: report it, do not compensate in the
hook.

Consumer requirements (InstaMax, 2026-10-31):
- `dt=None` means continuous time (FDFD and neural kernels): V = port_voltage with no cos
  factor, and the termination impedance is `conj(termination.impedance(omega))` instead
  of the trapezoidal form.
- Differentiable w.r.t. `fields` (no `.item()`, no detach, no in-place ops on inputs).
  With a `drive`, gradients only need to flow through V.
- An unbound `TerminalPort` is resolved to a lumped gap (`_resolve_terminal_port`) and
  takes the lumped path; state this in the docstring. Wire-bound `TerminalPort` and
  `WirePortBinding` ports raise.

```python
Result.port_drive(name) -> PortDrive(source_spectrum, source_impedance)
```
- `source_spectrum`: the prescribed drive only (drive accumulator), with the run's
  normalization: per-bin reference period (16139614), divided by the source spectrum when
  `normalize_source=True`.
- `source_impedance`: the runtime-resolved real source resistance.
- Same frequency axis order as `Result.port(name)`. Raises if the run has no excited port of
  that name (never returns zeros).

This replaces requests 2 and 3 with one object.

`mw.trapezoidal_impedance` is exported at top level and documented. It already exists in
`witwin.maxwell.lumped.__all__`.

Out of scope, failing closed with `NotImplementedError` plus a census entry each:
wire-bound ports (V needs the wire charge state, not fields), sheet-gap feeds (D6, added
when D6 merges), embedded-network and circuit-bound ports (I needs the network state), and
mode/wave ports.

## Proving gate

A native FDTD run with a full-field DFT monitor. Use the port frequencies and the **same
window**; the cos(theta/2) identity is exact only up to window-edge terms, so use a tapered
window or a fully decayed run. `port_phasors` applied to those fields must match
`Result.port(name)` V and I to <= 1e-5 relative, for each of:
- excited R-source;
- matched passive R;
- SeriesRLC termination;
- probe (open observer) V and loop I, fed from `Result` field monitors (H included):
  `port_phasors(probe)` V and I equal `Result.port(probe)` voltage and current to 1e-5;
- one coaxial port;
- one unbound `TerminalPort`.

Plus a `dt=None` check against an FDFD run's port quantities, and one autograd check:
d|V|^2 / d(fields) from `torch.autograd.grad` equals the analytic linear-functional
gradient.

Run each at a single frequency (CW) and multi-frequency pulsed, with
`normalize_source=False` and `True`. Plus each out-of-scope port kind raising.

## Tasks (Codex, worktree pcb/port-vi-hooks)

1. **Census, no code change.** DONE 2026-10-31: 23 scene hits on V and 22 on I (open
   coax probes, frozen and graph port gates, the clock consistency test, the PCB
   contour control), 0 on PEC gaps. The data plan and InstaMax read no open-port V/I.
   This led to the revised ruling above (no native output change).
2. **Probe guard plus role metadata.** The PEC-suppressed-probe `ValueError` with a census
   entry, `metadata["port_role"]`, and the docs. Gate: every port output torch.equal to
   master (only metadata added); the frozen and graph port gates unchanged.
3. **Hooks.** `port_voltage`, `port_phasors`, `Result.port_drive`, the top-level
   `trapezoidal_impedance` export, the proving gate, census entries for the new raises,
   `FEATURE_LIST.md`, and an integration_suites owner for the new test file.

Merge checklist applies (suite owner, guard census, reader grep, downstream notice to
InstaMax, attribution rules).

# Multi-conductor terminal wave port and custom / composite V-I integrals (design proposal)

Status: implemented on branch `coverage/terminal-wave-port` (2026-10-29) following the §4 decisions and
the §5 lesson; see §7 for what was built, the validation record and the deferred items.
Scope: FDTD RF ports. This closes two gaps against the external reference solver:
a terminal wave port with more than two conductors, and the general custom-2D and
composite voltage/current integrals.

## 1. Current state

**Two-conductor `TerminalPort` (`witwin/maxwell/ports.py`).**
- The port names one positive and one negative conductor structure (`TerminalRef`) and an `AxisPath`.
- `geometry/terminal_support.py::resolve_named_terminals` resolves the axis-aligned voltage segment through the overlap of the two footprints. It also resolves the default `current_surface` rectangle and the contact and audit metadata.
- After resolution the port is lowered exactly like a `LumpedPort`:
  - `compiler/ports.py::_compile_voltage_path` produces a single-component E row of signed primal widths;
  - an optional `voltage_aperture` gives a normalized parallel-path quadrature;
  - `_compile_current` produces four H legs of a planar, unrotated, half-grid `Box`.
- Driven or terminated ports couple through the single-component `LumpedRuntime` / `FieldPortCoupling` (`fdtd/lumped.py`, `fdtd/ports.py`). The power-dual row is used for both measurement and deposition, and I is the branch current. Passive ports integrate the same row and loop.

**Wave-port V/I.**
- `WaveModeSpec` carries one `voltage_path` and one rectangular `current_contour`.
  - The path is a single axis-aligned segment, enforced by `_normalize_wave_voltage_path`.
  - The contour must be a `Box` on the magnetic half-plane adjacent in the propagation direction.
- `compiler/waveports.py::_compile_mode` lowers them with the same two helpers into `CompiledWaveModeSpec`: one `voltage_component` and four current legs.
- `waveport_sweep._integrate_mode_geometry` evaluates them on mode-solver fields. `_characteristic_impedance` gives a scalar Z per mode.
- The quasi-static line solver `_solve_quasistatic_line_modes` already accepts any list of `signal_potentials`.
- `fdtd/excitation/modes/tem.py::_tem_signal_potentials` only constructs even/odd patterns for one or two isolated conductors, and raises for three or more.

**What the PCB line does today (dev-only).** The CPW and coupled-pair work in `scripts/dev/fdtd/printed_line_*` and `docs/assessments/printed-lines-2026-10-22/*` builds multi-conductor V/I by hand:
- **Probe rows.**
  - Throwaway `+`/`-` probe `WavePort`s are compiled per station.
  - Their `(component, indices, weights)` rows are expanded into `PointMonitor` samples (`physical_contours.py::native_probe_monitors`, `coupled_contours.py::probes`).
  - Current is the average of the two half-plane loops.
- **Pair basis.**
  - The pair uses two `WaveModeSpec`s: "common" (Vp to ground, loop around both strips) and "differential" (Vp−Vn, loop around one strip).
  - These are mapped to the power-dual set Vc, Vd, Ic, Id by `printed_line_physical_modes.coupled_physical_state` and `printed_line_two_length.mixed_transform`, where A=[[.5,.5],[1,−1]] and I uses A⁻ᵀ.
- **Drives.**
  - Drives are one `LumpedPort` per conductor per side with a `voltage_aperture`. CPW is fed into both ground gaps.
  - The ports are mixed by fixed drive bases: `DRIVE_BASIS`, and `real_basis` = 0.75·I+0.25·ones.
  - Spectra are combined in `printed_line_loss_basis.coherent_fields`.
- **Matrices.**
  - `coupled_transfer` and `two_length_modes` take V, I of shape `[station, n_conductor, 2n drives]`.
  - They solve the station transfer matrix, keep the forward eigenmodes by signed power, and report Z = V_fwd·I_fwd⁻¹ (n×n).
  - The argument checks cap n at 2.
- **Public network side.** The only public route is four single-ended `WavePort`s plus an inline mixed-mode conversion `M S Mᵀ` in `tests/rf/wave_validation/test_microstrip_diffpair_wave_level.py`.

So the reusable pieces are:
- the sparse row format, which already behaves like a composite sum;
- the snapping helpers;
- the quasi-static solver for N potentials;
- the dev transfer/eigen algebra.

What is missing:
- a public N-conductor port;
- V and I matrices;
- a terminal↔modal transform;
- non-axis-aligned paths and nonrectangular loops.

## 2. Proposed API

### 2.1 Integral specifications (new module `witwin/maxwell/path_integrals.py`, re-exported)

```python
mw.AxisSegment(start, end)                     # one axis-aligned segment (today's voltage_path)
mw.PolylinePath(points, *, plane_axis, plane)  # in-plane polyline; each leg is staircased to Yee edges
mw.CustomVoltageIntegral(path)                 # V = integral of E . dl along a path
mw.CustomCurrentIntegral(loop)                 # I = circulation of H on a closed in-plane polyline
mw.CompositeVoltageIntegral(terms=((coef, VoltageIntegral), ...))
mw.CompositeCurrentIntegral(terms=((coef, CurrentIntegral), ...))
mw.RectangleLoop(box) / mw.CircleLoop(center, radius, axis)  # constructors for the common loops
```

Lowering:
- A polyline is lowered to Yee edges with a deterministic staircase rule. Each leg is split at the node lines it crosses, and every edge piece takes the signed length of its projection.
- An off-grid point is snapped with `_snap_index` semantics: it is rejected, never moved silently.
- A closed current loop is lowered as the boundary of the set of enclosed dual cells, with the same orientation rule as the coaxial loop (`compiler/coaxial.py::_current_loop`). Discrete Ampère's law is then exact for any loop shape.
- A composite is a linear combination of compiled rows. The result is a list of `(component, indices, weights)` terms instead of one `voltage_component`.
- The existing `AxisPath` and `Box` declarations keep their current compiled form exactly. They are not re-expressed through the new lowering.

### 2.2 Multi-conductor terminal wave port

```python
mw.TerminalWavePort(
    name, position, size, direction, reference_plane,
    terminals=(mw.TerminalRef("sig1"), mw.TerminalRef("sig2"), ...),
    reference=mw.TerminalRef("ground"),
    voltage_integrals=None,     # optional per-terminal CustomVoltageIntegral / Composite
    current_integrals=None,     # optional per-terminal CustomCurrentIntegral / Composite
    basis="terminal" | "modal",
    reference_impedance=50.0,   # scalar or per-terminal / per-mode vector
)
```

- **Terminal basis (default).**
  - For each signal conductor k, the default V_k runs from the reference to conductor k. It is the shortest in-aperture path found by the terminal resolver; for non-straight geometry the user supplies a polyline.
  - The default I_k is a loop that encloses only conductor k.
  - The quasi-static solver runs once per terminal with potential 1 on k and 0 on the others. This gives the N×N capacitance matrix C and a field basis F_k.
  - The modal fields are the eigenvectors of the per-unit-length (L, C) problem. L comes from a second magnetostatic solve or from the vacuum C₀ (L = μ₀ε₀ C₀⁻¹, quasi-TEM).
  - The port stores the terminal-to-mode matrices T_V and T_I = T_V⁻ᵀ, so that power is preserved.
- **Modal basis.** `PortData` channels are the modes `"<port>::M<i>"`. The terminal basis exposes `"<port>::T<k>"`. `NetworkData` gains `metadata["terminal_transform"]`, so terminal-to-modal conversion is one call: `NetworkData.to_basis(...)`.
- **Excitation.**
  - Each column of a `PortSweep` drives one terminal or mode as a launched modal field: the superposition of modal fields for a terminal column.
  - It uses the existing calibrated per-frequency CW `WavePort` machinery (`waveport_sweep.py`). That machinery already handles multimode `[M, F]` data, tracking and B = S·A extraction.
  - No new FDTD kernel is needed. A lumped (circuit-coupled) multi-terminal feed is out of scope and would be a separate item.
- **Extraction.**
  - The projected modal amplitudes come from the existing overlap.
  - Terminal V/I then follow as V = T_V·(a+b)-type combinations.
  - The forward characteristic matrix is Z_c = V_fwd·I_fwd⁻¹. This is the public counterpart of `two_length_modes`.

## 3. Implementation plan and files

| Change | Files | Kind |
|---|---|---|
| Integral specs, polyline staircase and loop lowering | new `path_integrals.py`, new `compiler/path_integrals.py` | new |
| Multi-term compiled row type and its integrator | new `compiler/path_integrals.py`; `waveport_sweep._integrate_mode_geometry` gains a branch for the new row type | new + additive branch |
| `WaveModeSpec` accepts `voltage_integral=` / `current_integral=` as alternatives to `voltage_path` / `current_contour` | `ports.py` (new keyword-only parameters, default `None`) | additive |
| `TerminalWavePort` declaration and resolution | `ports.py` (new class); `geometry/terminal_support.py` (new N-terminal resolver function next to `resolve_named_terminals`) | new |
| N-potential quasi-static basis, C matrix, (L, C) modes | `fdtd/excitation/modes/tem.py`: new `_tem_terminal_potentials(n)`; `_solve_quasistatic_line_modes` reused unchanged | new function |
| Compile and sweep dispatch | `compiler/waveports.py` and `waveport_sweep.py`: new `isinstance(port, TerminalWavePort)` branches | additive branch |
| Terminal/modal transforms on network data | `network.py` (new `NetworkData.to_basis`), `network_math.py` (public transform helpers) | additive |
| Reference adapter (terminal wave port, custom / composite integrals) | reference adapter module under `adapters/` | additive |

**Keeping existing paths bitwise identical.**
- `AxisPath`, `LumpedPort`, `TerminalPort`, rectangular `current_surface` / `current_contour` and the `WavePort` V/I paths keep their present compile functions. The new lowering is used only when a new spec type is present.
- No change to `fdtd/ports.py`, `fdtd/lumped.py` or any CUDA kernel. The terminal port runs on the existing wave-port CW column runner.
- The gate is the fresh-interpreter test added on the coaxial branch (`tests/rf/lumped/test_rf_port_bit_identity.py`). It runs `git archive` of the pre-change commit and the branch, and compares port V/I, S and fields with `torch.equal`. It is extended with a two-mode coupled-pair `WavePort` scene, then rerun.
- The dev scripts keep working. Porting them to the public API is a separate, optional change owned by the PCB line.

**Validation (planned).**
- Analytic stripline or coupled-coax even/odd impedances.
- Agreement with the existing dev `two_length_modes` Z matrix on the pair scenes at n=2.
- A three-conductor CPW-with-two-signals case against an independent 2D electrostatic C/L.
- Reciprocity and passivity of the terminal-basis S matrix.
- Polyline and loop lowering checked by discrete Stokes / Ampère identities on analytic fields, as done for the coaxial loop.

## 4. Decisions (PCB owner review, 2026-10-29)

These answers override the corresponding parts of sections 2 and 3.

1. **Terminal voltage path.**
   - CPW and every multi-ground case require an explicit polyline per terminal. Any ambiguity is rejected, never guessed.
   - A default path is allowed only for an unambiguous two-conductor case: one signal over one ground, as in microstrip. Its construction (straight segment through the footprint overlap) must be documented.
2. **Inductance and impedance.**
   - Quasi-TEM L = μ0ε0·C0⁻¹, with C0 taken from the air-filled solve, is accepted for constructing the port mode.
   - Z0 must be computed from the actual mode field by power–voltage (|V|²/2P) or power–current (2P/|I|²). The quasi-TEM V/I impedance is kept only as a cross-check.
   - There is no internal-inductance or finite-foil magnetostatic solve in this version.
3. **Drives.** Modal-field drives only, on the wave-port CW column machinery. A lumped multi-terminal circuit feed (FieldPortCoupling generalized to N rows) is a later step coordinated with the PCB line.
4. **Reference impedance.** Support a full reference-impedance matrix; per-terminal diagonal Z0 is a special case. Mixed-mode S of coupled lines requires the full matrix.
5. **Dev algebra.** Do not move the PCB dev transfer/eigen algebra (`printed_line_two_length`, `printed_line_physical_modes`) into the library. The library implements terminal and modal extraction independently, and the two are cross-checked later. The n = 2 dev results are an external comparison, not the backend.
6. **CopperFoil terminals.**
   - Split-trace CopperFoil counts as a terminal conductor.
   - Its terminal current is the sum of the two-sided foil current, read through the Cu bank / selected-Cu query in `result.py` (read-only, no change to `result.py`). The − side alone is never used.
7. **Where custom integrals live.** `voltage_integral=` / `current_integral=` exist only on the new `TerminalWavePort`. `WaveModeSpec` keeps its present semantics and signature; the `ports.py` row in the section 3 table is reduced to the new class only.

## 5. Lesson from the PCB line: current loops near conductor edges

The PCB line's fixed 2.5 mm V/I functional reads Z 23–26 % low on a converged field. It does not converge with refinement: 62.4 / 69.8 / 70.8 / 66.6 Ω at 2.5 / 1.25 / 0.625 / 0.3125 mm. The cause is that its current loop runs 0.625 mm from edge-peaked fields (`pcb-line-accuracy` worktree, `docs/assessments/printed-lines-2026-10-29-accuracy/README.md`).

The terminal port must therefore:
- place default and validated current loops well away from conductor edges, at several cells and a fixed fraction of the gap, with H interpolated to the loop geometry rather than snapped nodes;
- or use the power-based Z of decision 2 as the primary impedance.

Every custom loop gets a refinement-convergence test of Z, not only a single-grid comparison.

## 6. Original open questions (answered above)

1. Default terminal voltage path, or an explicit polyline for CPW?
2. Is quasi-TEM L = μ₀ε₀C₀⁻¹ acceptable, or is a magnetostatic or good-conductor solve needed?
3. Modal-field drives only, or also a lumped multi-terminal circuit feed?
4. Diagonal per-terminal Z₀ or a full reference-impedance matrix?
5. Should the dev `two_length_modes` / `coupled_transfer` algebra move into the library?
6. Should split-trace CopperFoil count as a terminal conductor?
7. Add `voltage_integral=` / `current_integral=` to `WaveModeSpec`, or keep them on the new class only?

## 7. Implementation record (2026-10-29)

What was built (§4 decisions override §2-3):

- `mw.TerminalWavePort` (new class only; `WaveModeSpec` unchanged) with per-terminal
  `voltage_integrals=` / `current_integrals=`, `basis="terminal" | "modal"`, a scalar,
  per-terminal or full real symmetric positive-definite `reference_impedance` matrix (or
  `"characteristic"`), `impedance_definition` and `conductor_model`.
- `path_integrals.py` / `compiler/path_integrals.py`: `AxisSegment`, `PolylinePath`,
  `RectangleLoop` / `PolygonLoop` / `CircleLoop`, custom and composite integrals. Paths are
  node-to-node primal-edge staircases (off-grid vertices rejected); loops are the boundary of
  the enclosed dual cells (loops through a node rejected). Every integral is audited against
  the resolved conductor labels (a terminal path must run terminal -> reference; a loop must
  enclose exactly its terminal and only whole conductors).
- Terminal basis in `fdtd/excitation/modes/tem.py`: one potential solve per terminal for `C`
  and `C0`, modes `C v = eps_eff C0 v`, degenerate clusters rotated onto `C0` eigenvectors.
  The mode solver receives explicit modal potentials through a private `terminal_mode`
  payload and an early dispatch in `_assemble_vector_mode_data`; the shared post-processing
  moved unchanged into `_finalize_vector_mode_data`.
- `terminal_waveport.py`: Zc from the actual mode field (power-voltage `Vm Vm^H / 2`,
  power-current, or V/I), quasi-TEM C/C0 impedance and loop V/I as cross-checks; modal-field
  drives on the CW column machinery (decision 3); generalized terminal power waves
  `a, b = (8 Zr)^-1/2 (V +- Zr I)`.
- `NetworkData(reference_matrix=...)`, `NetworkData.to_basis(...)` (terminal -> modal via
  `metadata["terminal_transform"]`), reference-adapter export of the terminal wave port and
  integrals.
- Split-trace CopperFoil terminals use `conductor_model="good_conductor"`; current loops must
  enclose the whole foil (both trace sides), a one-sided loop is rejected by the audit.
- Deviation from decision 6: terminal currents (and voltages) are recovered from the measured
  modal amplitudes, `V = Vm (a + b)` and `I = Im (a - b)`, not from an FDTD-field integral. The
  Cu-bank / selected-Cu field-current cross-check of `result.py` is deferred (see below).
- Aperture truncation: the quasi-static basis grounds the whole aperture boundary. Each mode
  reports the share of its terminal charge that ends on the artificial (non-conductor)
  aperture walls (`aperture_wall_charge_fraction`, node-wise magnitudes over `sum_k |Q_k|`),
  and `TerminalWavePortApertureWarning` fires above `APERTURE_WALL_CHARGE_LIMIT = 0.05`.
  Aperture rule for microstrip-like lines: top at least 7 h above the ground plane and lateral
  margin at least 6.5 h beyond the outermost strip edge (h = substrate height; 6.5 h = 5.2 W
  for this pair). Measured on the coupled pair (W = 1.25 h, s = 0.75 h, eps 4.4, dx = h/4):

  | Aperture (half width x height) | Margin / top | Even wall charge | Odd wall charge | abs(Scc21) 0.35 GHz | abs(Scc11) | abs(Sdd21) | abs(Scc21) 0.7 GHz |
  |---|---|---|---|---|---|---|---|
  | 0.10 x 0.07 | 3.4 h / 3.5 h | 10.7 % (warns) | 5.1 % | 0.952 | 0.183 | 0.995 | 0.898 |
  | 0.13 x 0.105 | 4.9 h / 5.3 h | 6.4 % (warns) | 2.1 % | - | - | - | - |
  | 0.16 x 0.14 | 6.4 h / 7 h | 4.6 % | 1.2 % | 0.983 | 0.055 | 0.999 | 0.930 |
  | 0.20 x 0.175 | 8.4 h / 8.8 h | 3.5 % | 0.7 % | - | - | - | - |
  | 0.24 x 0.21 | 10.4 h / 10.5 h | 2.9 % | 0.5 % | 0.987 | 0.045 | 0.999 | 0.941 |

  At the largest aperture abs(Scc21) is 0.987 / 0.941 / 0.876 at 0.35 / 0.7 / 1.0 GHz: the
  residual at higher frequency is quasi-TEM dispersion of this thick substrate (h = 0.02 m), not
  truncation. The earlier 0.78 figure was a 1 GHz run on a 0.10 x 0.07 aperture.

Validation (GPU 1):

| Case | Result | Gate |
|---|---|---|
| Coupled stripline W=b/2, s=b/4, eps 4.4, h/16 and h/32 | Ze 53.29 / 53.99, Zo 37.07 / 38.45 vs Cohn 54.75 / 39.85; Richardson 54.68 / 39.82 | monotone; finest < 2 % / 4 %; Richardson < 0.5 % |
| Coupled microstrip W/h=1.25, s/h=0.75, eps 4.4, h/8 and h/16 | eps_e 3.237 -> 3.299 (KJ 3.444), eps_o 2.805 -> 2.841 (2.896); Ze 72.1 (73.9), Zo 51.1 (52.0) | eps_eff monotone from below, < 6 %; Z0 < 4 % |
| CPW (two coplanar grounds), explicit polylines, G = 2/4/8 cells | Z 63.36 -> 69.19 -> 71.96 ohm (first order); loop V/I = power-voltage Z to 1e-6; composite (two paths, near + far loops) equals the plain integral to 1e-8 | step ratio < 0.6, last step < 5 % |
| Coupled-microstrip thru, Zr = coupled characteristic matrix, dx = h/4, 0.7 GHz | mixed mode abs(Sdd21) 0.990, abs(Scc21) 0.930, abs(Sdd11) 0.050, abs(Scc11) 0.074, abs(Sdc) < 1e-4 | > 0.97, > 0.88, < 0.10, < 0.12, < 0.02 |
| Split-trace CopperFoil pair (good conductor) | Zc within 0.02 % of PEC, beta imag > 0 | 0.2 % |

The microstrip under-loads eps_eff by first-order substrate-interface discretization; the coupled-microstrip
common mode in the 0.7 GHz thru is limited by quasi-TEM dispersion (see the aperture table
above). Both are stated resolution/budget limits.

Deferred:

- The Cu-bank / selected-Cu field-current cross-check of `result.py` (decision 6 read-only
  path). Terminal currents are recovered from the measured modal amplitudes; on the mode field
  the loop audit enforces two-sided foil enclosure, so a one-sided reading cannot occur.
- Lossy / magnetic cross-sections, direct `PortExcitation`, adjoint, mixing with single-mode
  `WavePort` and ensemble execution (explicit rejections, census-registered).
- Cross-check against the PCB dev two-length extraction (decision 5: owned by the PCB line).

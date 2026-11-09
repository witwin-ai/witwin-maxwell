# Coaxial-port gates on the node-consistent staircase line (2026-10-29)

Context: the staircase PEC fix (`docs/memory/rf-coax-curved-pec-slow-wave.md`) makes a
round vacuum coax an exact discrete TEM line. The round-coax gates in
`tests/rf/lumped/test_coaxial_lumped_physics.py` had been calibrated on the slow line
(effective-line self-consistency) and failed under the fix. They are now physical
gates with no fitted line parameter.

## Reference impedance: lattice capacitance of the conductor-node sets

For a Yee line whose PEC is a node set, the TEM mode is `E_t = -grad_h(phi)` of the
transverse lattice potential (1 on the inner, 0 on the outer conductor-node set) and
`V / I = eta0 / C~` exactly, where `C~ = C' / eps0` is the sum of `(d phi)^2` over the
free transverse edges (unit conductance per edge on a uniform grid). The 1D Yee
dispersion only affects beta. The test computes `C~` from the prepared solver
(conductor nodes are those whose Ez edge is shorted; the Laplacian uses the free
Ex/Ey edges). This defines capacitance-equivalent radii,
`(eta0 / 2 pi) ln(b_eff / a_eff) = eta0 / C~`.

| h (mm) | Z_ref (ohm) | deficit vs 49.940 | ratio | FDTD probes (ohm) |
|---|---|---|---|---|
| 0.25 | 56.910 | 6.970 | | 56.9106 (two-state), 56.911 (single-state) |
| 0.125 | 54.236 | 4.296 | 0.62 | 54.237 (single-state) |
| 0.0625 | 52.068 | 2.128 | 0.50 | not run (16x cost) |

First-order extrapolation from the two finest levels gives 49.90 ohm (nominal
`60 ln(2.3)` = 49.940). Z_ref exceeds nominal at every level because conductor nodes
are inside or on the bodies: the staircased rod is inscribed in the nominal rod and
the shield node set lies outside the nominal shield.

Rejected alternatives at 0.25 mm: equal-area radii of the closed conductor cells,
67.1 ohm (ignores the field concentration at stair corners); a log fit of the
mid-annulus potential, `a_eff` 0.910 mm and `b_eff` 2.367 mm, 57.28 ohm (agrees within
0.7 %, but is a fit). The square coax (Box inner 2 mm, square hole 3.5 mm) has
Z_ref = 28.448 ohm; the former single-impedance fit gave 26.9 ohm because it ignored
the end capacitances below.

## Conductor-end capacitances

The inner conductor ends at the feed (z = 1 mm, 1 mm from the PEC wall) and at the
load plane (z = 16 mm, 2 mm from the wall). An open standard (no resistor) with the
probe waves gives lossless constant shunt capacitances: round 61.5 fF (feed) and
50.5 fF (load), square 87.0 / 76.6 fF, frequency-independent within 0.2 %, |G/B| <=
1e-6. These are the reason the 50 ohm termination is not "matched": bare line theory
with Z_ref would give |S11| 0.040 / 0.076 / 0.104, and with the end capacitances it
gives 0.0235 / 0.0483 / 0.0753, which is what the port measures.

## Gates and measurements (round coax, 0.25 mm, 1 / 2 / 3 GHz)

| Gate | Reference | Measured | Tolerance |
|---|---|---|---|
| Z_ref convergence | nominal 49.940 ohm | deficit ratios 0.62 / 0.50, extrapolation 49.90 | ratio <= 0.7, extrapolation within 1 % |
| Velocity, 11 plane monitors 6-11 mm | c | 0.99989 / 0.99996 / 1.00000 | 1e-3 |
| Two-state probe Zc | Z_ref 56.9103 | rel. error 6e-6 | 1e-3 |
| Two-state probe beta | k0 | 5e-6 / 3e-6 / 1e-6 | 1e-4; det T = 1 within 1e-4 |
| Refinement, single-state probe Zc | Z_ref(h) | 2e-5 at 0.25 and 0.125 mm | 1e-3, and closer to nominal at 0.125 mm |
| End capacitances | lossless lumped C | spread 0.2 %, |G/B| 1e-6 | spread 1 %, |G/B| 1e-3; C_feed > C_load |
| Resistor `(Y_L - Y_open) R` | 1 | 1 + 3e-5 | 1e-3 |
| Fed loads 50 / 100 / 25 ohm | line(Z_ref, k0) + end C | |dS11| <= 3.4e-5 | 1e-3 |
| Two-port sweep (50 ohm ports) | cascade | max |dS| 2e-5, reciprocity 5e-9 | 1e-3 |
| Square loads 50 / 25 / 100 ohm | line(28.448, k0) + end C | |dS11| <= 2e-5; probe Zc 1e-7 | 1e-3 |

Plane placement matters: the conductor-end near fields fall about 3x per 0.5 mm (1e-2
of the TEM functional at 0.5 mm from an end, 1e-5 at 3 mm). The former 4-12 mm planes
include that tail and bias a 1 GHz two-wave fit (0.2 % over a 0.17 rad span); the
round gate uses 6-11 mm. The former two-node-average rule gave 0.877 / 0.872 / 0.868
on this rig.

`test_port_tabulated_drive.py` compares drives with each other and is unaffected; its
near-DC check reads 0.0023 at 0.1 GHz, as the line-plus-end-capacitance model predicts
(bound 5e-3).

## Bit-identity baseline

`tests/rf/lumped/test_rf_port_bit_identity.py` now freezes `be24cc99` (the PEC fix on
`pcb/pec-slowwave`). After the fix merges, switch `BASE_COMMIT` to the master merge
commit. Re-freeze evidence (CUDA, fresh interpreters):
- 442491e7 vs the fixed tree: 29 of 30 payload keys differ (only
  `direct/terminal/available_power` matches); for example the terminal/lumped sweep |S|
  moves from 0.885 / 0.055 / 0.218 to 0.908 / 0.043 / 0.208 at 0.4 GHz;
- 442491e7 plus only the `fdtd/runtime/materials.py` change of `be24cc99` vs the fixed
  tree: 0 of 30 keys differ. The coaxial-port branch therefore changes no existing
  port numerics under the new rule either.

## Other coax-dependent tests (former rule -> node rule)

- `coax_thru_scene(dx=0.01)` sweep at 1 GHz: |S11| 0.0159 -> 0.0041, |S21| 1.00049 ->
  0.99999, arg S21 0.8819 -> 0.8395 rad (k0 L = 0.8383; the former line was 5 % slow).
  `test_pec_frame` (frame changes S21 by < 1e-3; measured 0 under both rules) holds.
- WavePort TEM mode solve (`test_te10_mode_selection`): unchanged, beta = k0, Z0 89.52 /
  86.72 ohm at dx 0.01 / 0.005 against the nominal 83.12 ohm (5 % gate); the mode solver
  already used the node rule.
- `coax_sol_scene` (`test_open_short_match_wave_level`): matched |Gamma| 0.0118 ->
  0.0006, short 1.0001 -> 1.0000, open/short separation 121 / 112 -> 121 / 111 degrees.
  Thresholds and docstring hold.
- `test_planemonitor_waveport_passthrough` mode-profile correlation 0.851 -> 0.840
  (floor 0.75).
All of these pass unchanged.

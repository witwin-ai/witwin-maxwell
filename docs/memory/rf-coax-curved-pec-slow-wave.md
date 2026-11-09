# Resolved: TEM below c on curved/multi-box staircased PEC coax (2026-10-29)

Status: staircase policy fixed on branch `pcb/pec-slowwave`; `conformal` policy still
open (see below). The original open-item record follows the resolution.

## Root cause

`_pec_edge_open_fractions` (`witwin/maxwell/fdtd/runtime/materials.py`), staircase
branch, shorted an E edge when the **two-node average** of the smoothed PEC union
occupancy was `>= 0.5`. With `occupancy = (1 - tanh(sd / (h/2))) / 2` that is roughly
"edge midpoint inside", so an edge straddling a curved or off-grid surface was
shorted even when one endpoint lay outside the conductor. That exterior node then
carried a zero-E edge into the conductor while its tangential and longitudinal (Ez)
edges stayed free: the shorted set was not the edge set of any node set.

Evidence (enumerated on the compiled masks, CPU, 0.25 mm, plane `k` mid-line):
- no edge with both endpoints inside or on the PEC union was ever free (0 in every
  case), so the "free edges between conductor nodes" counted above were edges
  between exterior tooth-tip nodes;
- shorted edges with an exterior endpoint, Ex/Ey: Box/square 0/0, Box/round shield
  14/14, Cylinder/square 8/8, Cylinder/round 22/22, node-centred box union 18/18.
  The doc-style counts for the three curved cases (16/20, 8/12, 24/32) reproduce
  exactly; the node-centred union gives 8/24 (the record below says 8/18; its exact
  box layout was not recorded);
- a union of abutting **node-aligned** boxes (slab stair or per-cell boxes) was
  already consistent and ran at the square-coax speed. The 0.874c "stair" was a
  union of **node-centred** boxes (faces on half-cell planes): every crossing edge
  then sits at the tanh symmetry point, average exactly 0.5, and is shorted.

Not involved: PEC compile (`_pec_occupancy` union and half-weight faces are
correct), seams between abutting bodies (seam nodes get 0.5, seam edges are shorted),
float tolerance on aligned faces, the mode solver (it already used the node rule).

## Fix

Conductor nodes are `occupancy >= 0.5` (inside or on any PEC body); an edge is
shorted iff both endpoints are conductor nodes. A union of abutting bodies is one
conductor. On grid-aligned faces the edge set is unchanged bit for bit; the new rule
only reopens straddling edges, so it can never short a gap the former rule left
open.

FDFD (`FDFDSolver._ensure_pec_mask` and the local-response PEC candidate mask)
used the same midpoint rule (edge coverage >= 0.5); it now eliminates exactly the FDTD
staircase set through `_staircase_pec_shorted_edges`
(`tests/fdfd/test_pec_staircase_mask.py`).

## Measurements (vacuum coax, 4 radial dipoles, z-PML, 13 planes, two-wave fit, 5/8/11 GHz)

| Conductors, h | before | after |
|---|---|---|
| Box / square, 0.25 mm | 0.99998 / 0.99995 / 0.99990 | identical |
| Cylinder / square, 0.25 mm | 0.928 / 0.928 / 0.928 | 0.99998 / 0.99995 / 0.99990 |
| Box / 96-gon, 0.25 mm | 0.978 | 0.99998 / 0.99995 / 0.99990 |
| Cylinder / 96-gon, 0.25 mm | 0.9265 / 0.9264 / 0.9262 | 0.99998 / 0.99995 / 0.99990 |
| Cylinder / 256-gon, 0.125 mm | 0.9543 / 0.9543 / 0.9542 | 0.99999 / 0.99999 / 0.99997 |
| Node-centred boxes / square, 0.25 mm | 0.833 | 0.99998 / 0.99995 / 0.99990 |
| Node-centred boxes / square, 0.125 mm | 0.913 | 0.99999 / 0.99999 / 0.99997 |

Every consistent cross-section reproduces the aligned square line to fit precision:
only the 1D Yee dispersion along the line remains, and it falls at second order
(deficit ratio 0.25 at 11 GHz for h/2). Before, the deficit fell at about first order.

Gates: `tests/materials/compiler/test_pec_staircase_node_consistency.py` (CPU, edge
sets) and `tests/validation/physics/test_pec_curved_coax_tem.py` (CUDA, velocity
within 0.5 % of c and 2e-5 of the square line, second-order convergence).

## Compiled-coefficient changes

- Printed lines (`benchmark/scenes/rf/printed_lines.py`: microstrip, stripline, CPW,
  differential pair; PEC and copper; 2.5 and 1.25 mm; paddle perturbation): no
  volume PEC, edge sets identical.
- Round-coax RF fixtures (`coax_thru_scene`, `coax_sol_scene`, Cylinder plus hollow
  cylinder): Ey/Ez edge sets change by design (2.5k to 3.3k reopened edges).
- Rectangular waveguide short (`rectangular_waveguide_scene(short_far_port=True)`): the
  one-cell short plate has faces on half-cell planes; 406 Ex edges normal to it reopen.
  The shorted TE10 |S11| is identical to all printed digits (1.0000360).
- Rectangular PEC Boxes whose faces sit on half-cell planes (terminal/lumped test
  scenes on a 0.1 grid, e.g. `Box(position=(0.5, 0.5, 0.8), size=(0.5, 0.5, 0.2))`):
  the former rule shorted every edge crossing such a face (two-node average exactly
  0.5), a comb of teeth one node outside; the new rule keeps the floor node set
  (x 0.3-0.7 instead of teeth reaching 0.2/0.8). `test_rf_port_bit_identity` (frozen
  442491e7 source) therefore no longer matches bitwise, and the round-coax
  `test_coaxial_lumped_physics` gates calibrated on the slow line fail (matched
  |S11| 0.023 / 0.048 / 0.075 vs <= 0.02; probe Z 50.35-2.57j vs 49.94 within 2 %).
  A nearest-rounding consistent variant (conductor node iff sd <= h/2) also gives c
  but is further from 50 ohm on that rig (|S11| 0.050 / 0.090 / 0.113). Resolved:
  the port gates now use the lattice impedance of the staircase line and the
  bit-identity gate freezes the PEC fix; see `rf-coax-port-gates-staircase-line.md`.

## Still open: `conformal` policy (known defect, not fixed)

`SubpixelSpec(pec="conformal")` multiplies cut edges by `1 - fill` (a soft short) and
is unchanged here. Its cut edges also hang partial shorts off exterior nodes, and
the round coax still runs slow: Cylinder/96-gon 0.855c at 0.25 mm, 0.917c at
0.125 mm (256-gon); node-centred boxes 0.833c. A node-consistent conformal scheme
needs an area-weighted (Dey-Mittra-type) magnetic update, not an electric-edge
factor; until then prefer `staircase` for TEM lines with curved conductors.

---

# Original record (before root cause)

## Observation

In vacuum, an air coax built from a `Cylinder` inner conductor (r = 1 mm) and a
`ComplexPolySlab` shield (a square with a 96-gon hole, r = 2.3 mm) on a uniform
0.25 mm grid propagates its TEM-like wave at **v/c = 0.877 / 0.872 / 0.868** at
1 / 2 / 3 GHz.

Refining the grid to 0.125 mm (polygon 256) gives 0.919 / 0.910 / 0.900. The deficit
shrinks at about first order. Changing the shield polygon from 96 to 512 vertices
does not change the result.

## Ruled out

| Hypothesis | Check | Result |
|---|---|---|
| Port or phase-reference placement, or the measurement | Five `PlaneMonitor`s (z = 4 to 12 mm) record native Ex. The phase of a fixed transverse functional is fitted along z, independently of any port row. | Same 0.877 / 0.872 / 0.868 |
| Dielectric or magnetic fill | `eps_*`, `mu_*`, `sigma_e` and every positive `c*_curl` coefficient in the line interior | All exactly vacuum and uniform (`dt/eps0`, `dt/mu0`) |
| Higher-order / non-TEM content | max\|Ez\| / max\|Ex\| on the planes | 0.5 % (TE11 cutoff is about 29 GHz) |
| The new coaxial port or resistor | The same rig (same port, resistor and monitors) with grid-aligned square conductors: Box inner 2 mm, square hole 3.5 mm | **v/c = 1.0000** at all three frequencies |

## Correlation: PEC edge sets that are not node-consistent

I classified a node as a conductor node if it touches any suppressed edge, then
counted, in plane k = 30 and away from the domain frame:
- free in-plane edges between two conductor nodes;
- conductor nodes whose Ez edge is free.

| Conductors | Free in-plane edges / conductor nodes with free Ez | v/c (two-wave fit) |
|---|---|---|
| Box inner, square shield | 0 / 0 | 1.0000 |
| Box inner, circular shield | 16 / 20 | 0.98 |
| Cylinder inner, square shield | 8 / 12 | 0.93 |
| Union of node-aligned Boxes (staircase rod), square shield | 8 / 18 | 0.874 |
| Cylinder inner, circular shield | 24 / 32 | 0.93 (fit) / 0.877 (radial functional) |

Grid-aligned single Boxes produce a closed, node-consistent zero-E edge set and
propagate at c exactly. A Yee line whose PEC is a consistent node set has an exact
discrete TEM null space, so that result is expected. Curved primitives, and unions
of abutting Boxes, leave unsuppressed edges between nodes that are electrically on
the conductor. These act like one-cell slots or grooves, which plausibly add series
inductance.

The estimated propagation constant depends on the transverse functional (0.877
radial, 0.93 sign mask). The wave is therefore not one clean mode. The rectangular
PlanarPEC / CopperFoil printed lines have no such edges, which is consistent with
their beta matching the independent reference.

Not yet determined:
- where the edge classification is decided (material/PEC compile, edge-native
  sampling or conformal fill);
- whether the slots fully explain the deficit.

This involves the shared material compiler. The coaxial-port branch did not touch it.

## Reproduction (GPU, under 10 s per run)

Scene (`tests/rf/lumped/test_coaxial_lumped_physics.py::_scene`):
- domain x, y ±3 mm, z 0 to 18 mm, uniform 0.25 mm grid, `BoundarySpec.pec()`;
- shield: `ComplexPolySlab([square ±3.25 mm, 96-gon r=2.3 mm], bounds=(0, 18 mm))`;
- inner: `Cylinder(r=1 mm, z 1 to 16 mm)`;
- `CoaxialLumpedPort("feed", z=1 mm)` and `CoaxialLumpedResistor(50 ohm, z=16 mm)`;
- drive: `GaussianPulse(2 GHz, fwidth 1.5 GHz)`, 6000 steps, window `none`.

Add `PlaneMonitor(axis="z", position=z, fields=("Ex",))` for z = 4, 6, 8, 10 and
12 mm. Fit the unwrapped phase of `sum(Ex * x / r^2)` over a < r < b against z.
For the square comparison, swap in the Box inner and the square hole as in
`_square_scene`.

## Effect on the coaxial-port tests (superseded by `rf-coax-port-gates-staircase-line.md`)

The round-coax load test compares against the effective line measured by passive
probes with the same annular functional. It is a self-consistency check, not a
check of the line itself; tolerance 0.03, measured 0.025 or less.

The physical gates now use the square coax:
- c within 0.2 % (measured 1.0000);
- three loads against line theory with exact k0 and one fitted Z_c (26.9 ohm),
  within 0.03.

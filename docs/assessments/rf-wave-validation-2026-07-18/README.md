# RF wave-level validation artifacts (S1.3)

Machine-readable per-scene artifacts from `python -m benchmark rf`
(audit step S1, 2026-07-18; round-4, regenerated after two independent auditors
proved the round-2/3 root causes false by execution). One JSON per scene, each
carrying:

- `gate_class` -- verbatim S0.3 taxonomy class of the binding gate
  (`docs/reference/gate-classification.md`); `modal-eigensolve` for supporting rows;
- `status` -- separate axis: `pass` / `gap` / `fail` / `blocked` / `pending`;
- `reference` / `tidy3d_reference` -- analytic first-line reference and external
  cross-reference status (`pending-generation`: adapter generation not yet wired);
- `tolerance_basis` -- how the binding tolerance was derived (Yee numerical
  dispersion, not tuned-to-pass);
- `falsification` -- the perturbation that turns the gate red;
- `metrics` / `supporting` -- FDTD-measured quantities vs analytic; supporting
  modal-eigensolve rows never gate;
- `convergence` -- grid-tier study (beta rel error, Yee floor, mid-band and
  band-max passivity/reciprocity, |S11| range);
- `conservation` -- per-tier passivity / reciprocity diagnostics;
- `notes` -- the gap / root cause stated honestly.

The binding metric is measured from a real FDTD `Scene -> Simulation -> Result`
run wherever the two-port bench yields a usable S-matrix; it is NEVER taken from
the 2D mode eigensolve. This is **not** a set of passing wave-level scenes:

| Artifact | Scene | Status | Note |
|---|---|---|---|
| `rf__coax_thru.json` | coax TEM two-port | **pass** | terminated through the padded PML; S via `B=S*A`. a_passive/a_driven 0.17 (diagnostic), \|S11\|<0.02, \|S21\|~1, max sv ~1.0, cond(A) ~1.2, beta from arg(S21)/L within 0.83% of k0 (finest tier) |
| `rf__rectangular_waveguide.json` | TE10 two-port | **blocked** | selector now refuses the checkerboard-aliased TE10; the transverse vector operator cannot yet produce a clean full-grid TE10 (sublattice decoupling). Operator redesign filed as an open item |
| `rf__microstrip_two_port.json` | microstrip quasi-TEM | blocked | WaveModeSpec('tem') categorically inapplicable to substrate+air (hybrid solve required) |
| `rf__differential_pair.json` | coupled-line 4-port | blocked | same TEM inapplicability on the coupled inhomogeneous cross-section |
| `rf__series_parallel_rlc.json` | series RLC resonance | gap | parasitic-dominated bench; peak does not track C |
| `rf__lumped_open_short_match.json` | lumped 1-port cal | fail | feed decoupled from load: matched/short/open all read \|Gamma\| ~ 0.997 at the same phase |

See `docs/reference/rf-wave-validation-2026-07-18.md` for the full narrative, the
gate taxonomy re-labelling, and the honest-gap record.

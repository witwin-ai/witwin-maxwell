# RF wave-level validation artifacts (S1.3)

Machine-readable per-scene artifacts from `python -m benchmark rf`
(audit step S1, 2026-07-18). One JSON per scene, each carrying:

- `gate_class` -- S0.3 taxonomy class of the binding gate;
- `reference` / `tidy3d_reference` -- analytic first-line reference and external
  cross-reference status (`pending-generation` when offline);
- `metrics` -- measured vs analytic quantities with relative error;
- `convergence` -- grid study tiers (with fallback tiers flagged, not counted);
- `conservation` -- reciprocity / passivity / power-balance diagnostics;
- `status` -- `pass` / `gap` / `pending`, with `notes` stating the gap honestly.

| Artifact | Scene | Status |
|---|---|---|
| `rf__coax_thru.json` | coax TEM two-port | pass (Z0 ~1.1%, beta exact) |
| `rf__rectangular_waveguide.json` | TE10 two-port | pass (beta/Z ~0.08%) |
| `rf__microstrip_two_port.json` | microstrip quasi-TEM | gap |
| `rf__series_parallel_rlc.json` | series RLC resonance | open gap (parasitic-dominated bench) |
| `rf__lumped_open_short_match.json` | lumped 1-port cal | gap (near-field coupling) |
| `rf__differential_pair.json` | coupled-line 4-port | pending-generation |

See `docs/reference/rf-wave-validation-2026-07-18.md` for the full narrative,
the gate taxonomy re-labelling, and the honest-gap record.

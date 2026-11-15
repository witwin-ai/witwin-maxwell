# Thin-wire port diagnostic ladder

Run from the repository root in the `maxwell` environment:

```bash
python scripts/dev/fdtd/thin_wire_ladder/run.py --dry-run
python scripts/dev/fdtd/thin_wire_ladder/run.py --dry-run --only 'D1_*_r3_*'
python scripts/dev/fdtd/thin_wire_ladder/run.py --output /tmp/thin-wire-ladder
python scripts/dev/fdtd/thin_wire_ladder/analyze.py /tmp/thin-wire-ladder --output /tmp/thin-wire-ladder-summary.csv
```

The default includes shutoff and fixed-horizon arms: 72 configurations, 144
independent serial executions. `--horizons shutoff` or `--horizons fixed` selects
one arm. `--horizon-ns` defaults to 80; `--only` accepts comma-separated globs.
A filtered r3 request includes a D1 memory probe, also listed by the dry-run.

The CPU MoM straight-dipole reference is qualified against published thin-dipole
tables (King 1956, Tables 30.1, 30.8-30.10, 30.15; see `tests/support/test_thin_wire_mom.py`).
The loop reference has no verified table value and stays unqualified; CSV rows carry the
flag per reference. See the [local CPU qualification and sizing record](../../../../docs/assessments/thin-wire-port-ladder-2026-11-03.md)
for the original sizing and assumptions.

The scripts run public `Scene -> Simulation -> Result` workflows and inspect
prepared state read-only. They make no solver or port metric changes. Execution
requires CUDA; `--dry-run`, the oracle, and the analysis are CPU-only.

# Selected native FDTD field queries

Use the existing `Result.query_fields` API with explicit `components` to request only retained fields. The default call still returns the existing single-frequency `(E,H)` tuple. Explicit selection returns a dict; each component value has shape `[selected_frequency, point]`, including singleton axes. Frequencies follow the unique exact requested recorded values, or recorded order when omitted.

```python
samples = result.query_fields(
    points_on_signal,  # CUDA float64 points on the actual owned foil plane
    region_id="interior",  # use the existing geometry classification
    field_kind="total",
    components=("Ey",),
    owner="signal_foil",
    trace_side="negative",
    representation="published_native",
)
ey = samples["Ey"]  # [frequency, point]
```

The owner must name the actual finite copper structure; holes and active edge ownership remain strict. Negative/positive refer to increasing native normal coordinate. `constitutive` selects the actual saved finite complex128 Et head on the declared side. `published_native` selects only the original negative native primitive, currently complex64 in the qualified ordinary-field fixture. Float64 geometry weights can legitimately promote output arithmetic to complex128; the API never downcasts a constitutive head to fabricate a primitive. No abs current, head swap, phasor phase/scale fit, or missing-field zeros are introduced.

Only requested native fields are gathered. A Cu Et-only query does not invent H/normal-E or require their unrelated stencils. Retained full spectral fields, actual PointMonitors and uncollocated single-normal PlaneMonitor components can supply samples. A real dense last-step snapshot is ignored as a phasor source. Derived collocated or normal-interpolated plane presentations do not invalidate otherwise complete Point/native stencils. Missing corners, inconsistent exact native overlaps, malformed coordinates or mismatched frequency/count data fail rather than interpolate across missing support.

Strict Cu queries require actual saved per-face dt, frequency order, start/end/counter/window/norm, phase factor, physical time-bank representation and owner/component/normal. The physical convention is `E(t)=Re(a exp(-i omega t))`; actual DFT accumulation uses `exp(+i omega n dt)`. Saved source/observer/head schedules are checked deterministically. Old snapshots lacking this provenance reject the new strict operation; do not add guessed metadata or recover an arbitrary multiplier from native/left data. Existing legacy query behavior and normalization contract remain separate.

Initial selected support is a `real_projected` physical time bank, Cartesian nonperiodic geometry, total fields, raw source normalization and the none window. Fixed float64 query positions support live field/head and qualified continuous Cu material VJPs. `points.requires_grad=True` explicitly rejects; exact-node zero-weight neighbours do not establish coordinate or shape gradients. Mapped, periodic/Bloch, analytic banks, weighted windows, undoing source normalization and selected matched-reference incident/scattered queries remain explicit capability boundaries. Epsilon and electric conductivity need not equal1/0; the owned-Cu context uses the existing unit scalar permeability contract.

The genuine XYZ/eager/Graph, single-frequency mixed storage, buried epsilon4/sigma0.01, actual checkpoint/save/load, independent native clock and continuous-parameter FD controls passed in the [actual11GPU stage](../assessments/printed-lines-2026-10-28/public-selected-query-v3/actual/README.md), with108CPU and reviewed [capability evidence](../assessments/printed-lines-2026-10-28/public-selected-query-v3/preparation/README.md). This bounded API qualification does not establish mode, tail, continuum impedance/attenuation, external-reference or full maintained-suite acceptance.


The integrated workspace subsequently passes115 CPU tests and the10 maintained GPU cases on source59d9d, preserving the concurrent MoM feature. Independent raw replay verifies all41 retained files. Two earlier qualification-harness startup failures execute zero testcases and remain archived. See the [actual integrated-workspace evidence](../assessments/printed-lines-2026-10-28/public-selected-query-v3/MAIN-integration/README.md). The complete frozen939 regression is a separate source epoch.

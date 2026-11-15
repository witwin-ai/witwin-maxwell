# FDTD typed results

`Result.monitor(name)`, `Result.monitors` and `Result.fields` of an FDTD run return the physics-typed
data of `witwin.maxwell.results` (`FieldData`, `FluxData`, `ModeAmplitudeData`, `TensorData`). The
values are the tensors the solver recorded (views, or a `torch.stack` along `component`), so numbers
and gradients are those of the solver payloads; `Result.raw_monitor(name)` returns the payload
dictionaries themselves (quadrature, face and provenance metadata included).

## Conventions

- Spectral data are FDTD DFTs of an `exp(+i omega t)` sum: phasors with `exp(-i omega t)` time
  dependence (`attrs["phasor_convention"] = "exp(-i omega t), peak"`), peak phasors for CW drives and
  the Fourier transform times the reference frequency for pulses. `attrs["normalization"]` is `"raw"`,
  or `"per_unit_source_spectrum"` with `SpectralSampler(normalize_source=True)`.
- Frequency-domain data keep a leading `f` dimension, of length 1 for a single-frequency monitor.
  `Result.monitor(name, frequency=...)`, `freq_index=...` and `Result.at(...).monitor(name)` select one
  frequency and drop the dimension. FDFD results follow the same rule.
- Time traces have a `t` dimension (seconds); `attrs["field_time_offsets_steps"]` gives the E/H sample
  offsets in steps relative to `t`. Traces are detached (`attrs["differentiable"] = False`).
- Components that share sample points are stacked along `component` (labels `Ex` ... `Hz`, or ADE
  state labels). Components on different Yee grids come back as a dict of per-component `FieldData`.
  `attrs["component_units"]` maps each component to its unit; `units` is that unit or `"mixed"`.
  Selecting components recomputes `units`; selecting one component (dropping the dimension) names it
  in `attrs["component"]`, the same form as an FDFD `Result.fields[...]` entry.
- Component labels (`"Ex"` ... `"Hz"`) and the mode propagation labels (`"+"`, `"-"`) are plain
  strings, the values of `mw.monitors.FieldComponent` and `mw.monitors.Direction`; both Enums are
  `StrEnum`s, so `sel(component=mw.monitors.FieldComponent.EZ)` and
  `sel(propagation=mw.monitors.Direction.PLUS)` select them. FDFD results use the same strings.
- `attrs["yee_location"]`: `edge_<axis>` (E) or `face_<axis>` (H) for native samples, `native` for
  point samples (each component at its nearest own Yee node, `attrs["sample_indices"]`), `cell_center`
  for plane components averaged onto cell centres, `collocated` for `sample_spacing` quadrature points.

## Monitors

| Monitor | Result | Dims |
|---|---|---|
| `PointMonitor` | `FieldData` | `(component, f)` |
| `PlaneMonitor`, `FinitePlaneMonitor` | `FieldData`: one component on its native Yee samples, several on the aligned cell-centre (or `sample_spacing`) grid; a dict of native per-component `FieldData` when the alignment does not cover a component (the plane-normal H) | `(component, f, a, b)` |
| `compute_flux=True`, `FluxMonitor`, `IncidentPowerDensityMonitor`, `ClosedSurfaceMonitor` | `FluxData` (W) | `(f,)` |
| `ModeMonitor` | `ModeAmplitudeData` (relative to the solved mode; `attrs["power"]` in W and `attrs["power_fraction"]` follow the values) | `(mode, propagation, f)` |
| `GaussianOverlapMonitor`, `AstigmaticGaussianOverlapMonitor` | `ModeAmplitudeData` (sqrt(W) against the unit-power beam) | `(mode, propagation, f)` |
| `DipoleEmissionMonitor` | `FluxData` of the delivered power, `attrs["current_spectrum"]` | `(f,)` |
| `DiffractionMonitor` | `FluxData` of the power per Floquet order; per-order `m`, `n`, `kx`, `ky`, `kz`, `theta`, `phi`, `propagating`, `efficiency`, `amplitude_E`, `amplitude_H` selected with `order` | `(order, f)` |
| `FieldTimeMonitor`, `AuxFieldTimeMonitor` | `FieldData` (point traces, or one component); a dict of per-component `FieldData` for plane or volume traces of several components | `(component, t, ...)` |
| `FluxTimeMonitor` | `FluxData` | `(t,)` |
| `SurfaceFieldTimeMonitor` | `{face: {component: FieldData}}` | `(t, a, b)` |
| `PermittivityMonitor`, `MediumMonitor` | `{"eps", "mu", "sigma_e": TensorData}` | `(component, f, x, y, z)` |
| projection monitors | `ProjectedFieldData` | |
| `WireMonitor`, `PowerLossMonitor` | `WireData`, `PowerLossData` | |

```python
probe = result.monitor("probe")                       # FieldData (component, f)
ez = probe.sel(component="Ez").values                 # [f] tensor, differentiable
ez0 = result.monitor("probe", freq_index=0).sel(component="Ez").values   # 0-d
flux = result.monitor("flux").values                  # [f] W
mode = result.monitor("port_out")                     # ModeAmplitudeData
forward = mode.isel(mode=0).sel(propagation="+")
fraction = forward.attrs["power_fraction"]            # [f]
```

## Volume fields

`Result.fields` maps `EX`, `EY`, `EZ` to `FieldData` volume-DFT phasors with dims `(f, x, y, z)` on
each component's Yee grid. With `FDTDConfig(full_field_dft=True, volume_dft_components=("Ex", "Ey", "Ez",
"Hx", "Hy", "Hz"))` it also maps `HX`, `HY`, `HZ` (any subset of the H components can be added; E is always
recorded). Multi-GPU and differentiable runs refuse H components. A single-frequency run without the volume DFT (`full_field_dft=False`,
the single-frequency default) records no phasors: `Result.fields` raises and the real E fields after
the last step are `Result.time_snapshot` (dims `(x, y, z)`, `attrs["kind"] = "time_snapshot"`,
`attrs["step"]`, `attrs["time"]`). `Result.field("Ex")`, `Result.E.x` and `Result.at(...).H.y` return
the same `FieldData` (a frequency selection drops `f`; `expand_symmetry=True` mirrors values and
coordinates) and raise where `Result.fields` does.

### What the FDTD volume fields are (guarantee)

Dataset generation may rely on the following; tests in `tests/fdtd/test_result_field_record.py` pin it.

- **Raw Yee phasors.** The values are the solver's volume DFT of `Ex`, `Ey`, `Ez` (and `Hx`, `Hy`,
  `Hz` when recorded, with the same weights and scale), finalised as
  `(sum_m w_m E^m exp(+i omega m dt)) * scale` (`scale = dt * f` for pulses, or `dt` times the
  reference frequency; `2 / sum_m w_m` for CW), divided by the source spectrum with
  `normalize_source=True`. Nothing is interpolated or averaged.
- **Full computational grid.** Absorber layers are included: `EX` is `(F, Nx - 1, Ny, Nz)`, `EY`
  `(F, Nx, Ny - 1, Nz)`, `EZ` `(F, Nx, Ny, Nz - 1)` for `N` primal nodes per axis; `HX` is
  `(F, Nx, Ny - 1, Nz - 1)`, `HY` `(F, Nx - 1, Ny, Nz - 1)`, `HZ` `(F, Nx - 1, Ny - 1, Nz)`. The physical window
  is the nodes inside `scene.domain.bounds`; non-PML faces and symmetry faces add no layer.
- **Coordinates.** `coords["x"|"y"|"z"]` are float64: the primal nodes on an E component's two
  transverse axes and the cell midpoints (`nodes[:-1] + 0.5 * diff(nodes)`) on its own axis (an H
  component the reverse: primal nodes on its own axis, midpoints on the two others), from the
  prepared grid (a `GridSpec.custom` window is bit-equal to the input nodes).
- **No copy.** `FieldData.values` (from `Result.fields`, `Result.field(...)` or `Result.E`) is a view of
  the stored phasor tensor.
- **Periodic and Bloch axes.** Node arrays are endpoint-inclusive and the stored E includes the wrap
  plane: the last node plane is the periodic image of the first,
  `E[..., last, ...] = stats()["bloch_phase"][axis] * E[..., 0, ...]` for the components with a node
  along that axis (`1` on a real-periodic axis). H follows the same rule on its staggering (`HX` has
  its node along x, `HY` along y, `HZ` along z).
- **Sample time label.** The field after step `m` (`m = 1 ... steps_run`) is weighted at `t = m*dt`,
  the physical Yee time label of `FieldTimeMonitor` E traces: `attrs["field_time_offset_steps"] = 0.0`.
  Step `m` first updates H from `E^(m-1)` to `H^(m-1/2)` and then E to `E^m`, so the H held after step
  `m`, weighted at the same `m*dt`, is the field at `(m - 1/2)*dt`: H `FieldData` carry
  `attrs["field_time_offset_steps"] = -0.5`, the H offset of `FieldTimeMonitor` traces. The phasor at
  the component's own time is `values * exp(+i omega offset dt)`. With that offset the discrete
  Faraday relation of the stored phasors is the phase-free `i omega~ mu H = curl E` (`omega~ =
  (2/dt) sin(omega dt / 2)`) up to the record-end term (`tests/fdtd/test_result_h_field_record.py`).
- **Stored H and H derived from E.** The stored H is the field the update produced. It differs from
  the plain discrete Faraday step of the stored E wherever the H update adds more than `curl E`:
  next to a `CopperFoil` (the split-trace magnetic correction), at the TFSF-corrected H shell
  (`mw.fdtd.tfsf_h_mask`), in absorber layers and with magnetic media corrections.
- `stats()["dt"]` is the time step and `Result.frequencies` the DFT frequencies of the leading `f` axis.

### Run records for consumers of the volume fields

| `stats()` key | Value |
|---|---|
| `dft_window` | the window the spectra were actually recorded with: `"none"` for every pulse (non-CW) run, whatever was requested |
| `dft_window_requested` | `SpectralSampler.window` as configured |
| `last_step_field_ratio` | per DFT frequency, `||E^N|| / (2 sin(omega dt / 2) ||sum_m w_m E^m exp(+i omega m dt)||)` over the whole grid, with `E^N` the field after the last step run: the size of the end-of-record (truncation) term of the volume phasor relative to the phasor. Measured: 3.7e-6 for a decayed TFSF box (datagen `f2_box_32`), 2.1e-5 for a 64^3 vacuum dipole after 4800 steps, 1.0e-3 for the undecayed `f5_loaded_36` record. For a Hann-windowed CW record it is an upper bound (the window brings the last sample to zero weight). `None` without the volume DFT |
| `bloch_wavevector` | the resolved Bloch wavevector in rad/m (`bloch_wavevector="auto"` resolved to the numerical wavenumber of the source), zeros on non-Bloch axes |
| `bloch_phase` | per axis, the complex wrap phase `exp(i k L)` of the domain period `L`; `1` on non-Bloch axes |
| `tfsf` | `None` without a TFSF source, else `{"mode", "axis", "node_indices", "h_corrections"}`: `node_indices` are the snapped `(first, last)` primal node indices of the total-field region per axis on the computational grid, `h_corrections` every H block the update adds the incident correction to (`component`, `start` index, `shape` on that component's Yee grid) |

`mw.fdtd.tfsf_h_mask(result)` turns `h_corrections` into `{"HX", "HY", "HZ"}` boolean masks on the
full H grids (`HX` is `(Nx, Ny - 1, Nz - 1)`), `False` at the tangential H half a cell outside each TFSF
face, where the update also used the incident field and H therefore does not follow the plain discrete
Faraday step of the stored E (which is total inside the TFSF surface and scattered outside it).

### Memory and the dataset-generation idiom

A finished run keeps its solver (fields, absorber state, coefficients, DFT accumulators) alive in
`Result.solver`. For dataset generation call `result.release_solver()` right after `run()`: it drops
the solver and the prepared scene's compiled material cache, so the Result then holds its phasors and
small run records only (the 64^3-grid single-frequency case: 86 MB after `run()`, 7.0 MB (the 6.5 MB of phasors) after
`release_solver()`). Keep no other reference to the solver, such as a `PreparedSimulation`. The phasors
of a sweep are finalised into one preallocated `(F, ...)` tensor per component, without stacking
per-frequency copies. Recording H doubles the volume-DFT accumulators and phasors (same 64^3 scene, E vs E+H: one
frequency 106 -> 119 MB peak, 86 -> 99 MB after `run()`, 7.0 -> 12.9 MB after `release_solver()`;
eight frequencies 198 -> 302 MB peak, 52 -> 103 MB after `release_solver()`).

`ProjectedFieldData` (projection monitors) stays its own type. It holds both `E` and `H`
(`[f, *grid, 3]`, Cartesian, exact Stratton-Chu or far-field approximation) on one observation grid
(`theta x phi`, Cartesian `x x y` at finite distance, or `ux x uy` k-space with a propagation mask),
with the grid geometry that its consumers (antenna metrics, RCS, `FieldProjector`) need per sample:
points, unit directions, radii, angles, background wavenumbers and impedances. `FarFieldData` is one
far-field quantity over angles or directions (`r*exp(-ikr)*E` style values); folding two fields,
finite-distance Cartesian grids and the masked k-space grid into it would move the geometry into
untyped attrs and change the value layout every consumer reads, for no new capability.

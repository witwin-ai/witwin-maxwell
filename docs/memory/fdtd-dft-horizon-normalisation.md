# FDTD pulse DFT horizon normalisation diagnosis

Date: 2026-10-30. Solver inspected at `c201fae4` on
`coverage/pinc-horizon-diagnosis`. Diagnosis only; no changes under `witwin/`.
Data-generation builders inspected at both `f7914baf` and the local
`datagen/p0-profiler` tip `e4cddaba`; `benchmark/datagen/scenes.py` is identical
between those revisions.

## Verdict

The absolute-power contract for pulses has a normalisation bug: it exports a
CW-style finite-record average as if it were a horizon-independent pulse
spectrum/engineering power. The DFT accumulation itself is correct. The observed
factor is expected from the current `normalize_source=False` implementation,
not from a changing pulse, a Hann taper, or an incorrect port power-wave formula.

Specifically, the returned amplitude is `2 * sum(x[n] exp(+i*w*n*dt)) / N_plan`.
For a completed pulse the numerator converges, but the denominator grows when
zero trailing steps are added. Fields/voltage/current scale as `1/N_plan`, and
quadratic powers as `1/N_plan**2`. This convention is a legitimate finite-record
Fourier-series coefficient; it is unsuitable as the assumed absolute pulse
spectrum or a source-calibrated watt label. A raw Fourier integral and a
source-normalised peak phasor also have different units; neither should silently
be identified with this record average.

Shutoff deliberately preserves the **planned** denominator. At one fixed plan,
early stop and a forced full run agree up to the discarded physical tail. Changing
the cap changes the normalisation even when both executions stop at the same
sample. Replacing the planned denominator with `steps_run` would merely move the
dependence to the shutoff threshold; it would not fix the pulse convention.

## Root cause and code locations

All paths below are relative to the repository root at the inspected solver
revision; data-generation paths refer to the revisions above.

| Stage | Location | Behaviour |
| --- | --- | --- |
| Default | `witwin/maxwell/simulation.py:184` | `SpectralSampler.normalize_source=False`; requested window defaults to Hann. |
| Plan | `witwin/maxwell/fdtd/runtime/timeplan.py:88` | Automatic duration uses the slowest source/requested frequency, propagation/source support, transient cycles and steady cycles. `total_steps` is the cap. |
| Pulse window | `witwin/maxwell/fdtd/runtime/spectral.py:240` | Non-CW response forces `window="none"` and start step zero. Per-step weights are one; changing a Hann envelope is not the cause here. |
| Port DFT | `witwin/maxwell/fdtd/ports.py:73` | `PortDFTAccumulator.phasors("peak")` divides raw voltage/current sums by the accumulated window-weight sum and multiplies by two. |
| Port finalisation | `witwin/maxwell/fdtd/ports.py:2273` | Always uses peak-average phasors; drive available power is `abs(source_voltage)**2/(8R)`. This path does **not** apply `normalize_source`, even when enabled for fields. |
| Power waves | `witwin/maxwell/network_math.py:60`, `witwin/maxwell/network.py:293` | Linear V/I-to-a/b conversion; `incident_power=abs(a)**2`, reflected power `abs(b)**2`, accepted power their difference, delivered power `0.5*Re(V*I.conj())`. The common scale is squared. |
| Point/plane DFT | `witwin/maxwell/fdtd/observers.py:248`, `:1405`, `:1430` | Shared `phasor_factor/window_normalization` scale (two for this real pulse). |
| Full-field DFT | `witwin/maxwell/fdtd/postprocess.py:9` | Same scale. GPU DFT tables precompute full-plan normalisers/counts in `runtime/spectral.py:562`; those counts should not be mistaken for actual executed steps. |
| Flux | `witwin/maxwell/fdtd/observers.py:1552` | Forms flux from the already scaled E/H phasors, making the dependence quadratic. |
| Shutoff restore | `witwin/maxwell/fdtd/runtime/stepping.py:1833`, `:2602` | Restores full-plan window sums for DFT and observers; `ports.py:1236` restores the port/drive sums of weights. Actual port/observer sample counts remain shortened. |
| Source normalisation | `witwin/maxwell/fdtd/observers.py:1598`, `:1339`; `fdtd/postprocess.py:72` | Field/source spectra have the same `2/N` factor, so division cancels it; flux divides by the squared source-spectrum magnitude. `sources.py:826` removes unit-waveform shape, retaining physical amplitude/phase. |

Write `X(f)=sum_n x[n] exp(+i*w*n*dt)` and `U(f)` for the unit-waveform sum.
Unnormalised output is `2X/N`, while source-normalised field output is `X/U`.
Once the pulse and response have decayed, extending the horizon cannot change
`X/U` except for physical truncation and numerical accumulation error. Currently
ports still return `2V/N`, `2I/N` in either sampler setting.

## Reproduction and measured evidence

`scripts/dev/fdtd/dft_horizon_normalisation_repro.py` is self-contained. It copies
the PEC two-arm wire/helix builder and 50-ohm pulse drive, and adds point, plane,
plane-flux and closed-surface observations. It reports every frequency, raw
port/drive DFT sums, planned/actual sample counts, window sums, published port
powers, far-field amplitude norms, radiated power and peak directivity. Angular
integration uses 19 theta by 37 phi points; these are horizon comparisons, not
spatial-convergence or antenna-accuracy qualification.

All experiments use physical GPU 1, the `maxwell` interpreter and the supplied
wrapper. Reproduce the cheap PEC dipole matrix with this single command:

```bash
env CUDA_VISIBLE_DEVICES=1 python <scratch>/witwin_cov_pinc/run_gpu.py .worktrees/cov-pinc scripts/dev/fdtd/dft_horizon_normalisation_repro.py --scene dipole --cycles 40 80 200 --mode both --output <scratch>/witwin_cov_pinc/dipole-matrix.jsonl
```

The dipole has 24 physical cells per axis, eight PML layers, radius `0.12*dx`,
arm tips at `+/-0.24*lambda`, Gaussian `f_c=2.4 GHz`, `fwidth=0.8*f_c`, and
three frequencies 1.92, 2.40, 2.88 GHz. Runtime `dt=4.084967320261438e-12 s`;
source settling step 244. Values below are at 2.40 GHz.

| Cap, steady cycles | Execution | Planned steps / denominator | Actual samples | P_inc | abs(S11) | P_acc/P_inc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 40 | fixed | 6,976 | 6,976 | 2.6473649927e-7 | 0.5920459882 | 0.6494815479 |
| 80 | fixed | 12,076 | 12,076 | 8.8344711413e-8 | 0.5920459888 | 0.6494815472 |
| 200 | fixed | 27,376 | 27,376 | 1.7190431602e-8 | 0.5920459888 | 0.6494815472 |
| 40 | shutoff 1e-5 | 6,976 | 900 | 2.6465620789e-7 | 0.5925706741 | 0.6488599963 |
| 80 | shutoff 1e-5 | 12,076 | 900 | 8.8317917616e-8 | 0.5925706741 | 0.6488599963 |
| 200 | shutoff 1e-5 | 27,376 | 900 | 1.7185217969e-8 | 0.5925706741 | 0.6488599963 |

Fixed-run raw voltage-sum magnitudes are 24.1360907488, 24.1360907539,
24.1360907542; raw drive-sum magnitude is 26.5499996108 at all three horizons.
The 40-to-200 power ratio follows `(27376/6976)**2` (about 15.40), while
`P_inc*(N/2)**2` is 3.2208223722, 3.2208223690, 3.2208223688. This isolates the
denominator, without assuming the antenna has converged merely because the
source is off. All three shutoff runs have identical raw port sums and 900 actual
samples, yet different denominators and powers.

Other absolute outputs at 2.40 GHz, fixed runs:

| Cap | abs(point Ez) | Plane Ez norm | Plane flux | Closed flux | Far E norm at 1 m | P_rad | D_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.0772838220 | 0.9717498422 | 6.3041049531e-8 | 1.7194163341e-7 | 0.0703021586 | 1.6917417156e-7 | 1.70053649 |
| 80 | 0.0446448140 | 0.5613558888 | 2.1037322014e-8 | 5.7378516706e-8 | 0.0406117998 | 5.6454844355e-8 | 1.70054555 |
| 200 | 0.0196935833 | 0.2476230562 | 4.0935161072e-9 | 1.1164881697e-8 | 0.0179145243 | 1.0985183430e-8 | 1.70054102 |

At the same 200-cycle plan, fixed versus shutoff changes P_inc by about 0.0303%,
point amplitude by 0.0225%, plane flux by 0.00830%, and P_rad by 0.0768% at the
centre frequency. Such small residual-tail errors are distinct from the 15.40x
cap dependence; a 1e-5 instantaneous field-energy threshold is not itself a bound
of 1e-5 on coherent spectral error.

A second run used `--scene dipole --cycles 40 200 --mode decay
--normalize-source --full-field-dft` (all other defaults unchanged), writing
`<scratch>/witwin_cov_pinc/dipole-normalized.jsonl`. Both cases stopped at 900 steps.
At 2.40 GHz, point amplitude was 10.15085411 versus 10.15085316, plane norm
127.6980972 versus 127.6980820, flux 0.001088136965 versus 0.001088136729, far-field
E norm 9.239440918 versus 9.239439964, and P_rad 0.002922076732 versus
0.002922076266. The full-field Ez norms were exactly equal at 737.7693481.
The raw unit-source DFT magnitude was 26.54999953 for both. Nevertheless port
P_inc remained 2.6465620789e-7 versus 1.7185217969e-8, identical to the
unnormalised-port results above. This experimentally confirms cancellation for
field paths and its absence for port finalisation.

The original helix was also run with `--scene helix --n 48 --count 41
--cycles 40 200 --mode decay --output <scratch>/witwin_cov_pinc/helix.jsonl` using the
same wrapper prefix. This reproduces all 41 frequencies over 1.92–2.88 GHz and
the observed stopping steps exactly:

| Helix cap | Planned denominator | Actual steps | P_inc at 2.40 GHz | abs(S11) | P_acc/P_inc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 | 6,976 | 6,976; cap reached | 1.8237279961e-7 | 0.9755516579 | 0.0482989627 |
| 200 | 27,376 | 10,800; shutoff | 1.1794744908e-8 | 0.9845993710 | 0.0305640786 |

The observed centre-frequency power ratio is 15.4622080, versus the pure
normaliser prediction 15.4002241. Across all frequencies the ratio ranges from
15.2560280 to 15.5695788. The 40-cycle helix response is not fully converged:
raw voltage-sum magnitude at the centre changes from 25.76365374 to 25.83070479,
and the maximum bandwise difference in abs(S11) is 0.01207535. Accepted fraction
is particularly sensitive because it subtracts nearly equal incident/reflected
powers; at 1.92 GHz it changes from +0.01775769 to -0.00092941. Thus the ratios
are not exactly horizon-independent in this truncated helix. This residual
ringing is separate from the dominant `N_plan**-2` scaling isolated by the
converged dipole and identical-step shutoff controls. The negative accepted-power
channels are recorded, not repaired or independently qualified here.

## Affected outputs

| Output | Pulse, source normalisation off | Source normalisation on |
| --- | --- | --- |
| `Result.port`: V, I, a, b | `1/N_plan` after decay | Still `1/N_plan`; flag is not applied here. |
| Incident/reflected/accepted/delivered/available port power | `1/N_plan**2` | Still `1/N_plan**2`. |
| Point/plane E/H; full-field `Result.fields` | `1/N_plan` | Common source factor cancels, provided the common source spectrum is resolved and nonzero. |
| `FluxMonitor`, plane `power`, closed-surface flux | `1/N_plan**2` | Common squared source factor cancels. |
| Far-field/projected E/H and equivalent surface currents | `1/N_plan` from the surface fields | Inherit the normalised surface fields. |
| Far-field power density, radiation intensity U, P_rad, antenna EIRP | `1/N_plan**2` | Inherit the normalised surface fields. |
| `Result.antenna` p_incident/p_accepted | Port powers above | Port powers above; inconsistent with normalised radiation fields. |
| S-parameters, impedance, directivity | Common scale cancels after convergence | Same within each consistent input convention. |
| Efficiencies/gain/realized gain computed from radiation and port power | Common scale cancels when both are unnormalised consistently | **Can become horizon-dependent** because fields are normalised but ports are not. Turning on this flag alone is not an antenna-label fix. |

Far-field scaling follows the linear equivalent-current projection;
`postprocess/field_projection.py:161` forms quadratic power density, and
`postprocess/directivity.py:99` integrates U into P_rad before dividing it out of
directivity. `postprocess/antenna.py:346` takes incident/accepted powers directly
from `PortData`.

Additional code-traced exposures, not separately run: wire current/charge and
ohmic-loss spectra use `2/window_sum` in `fdtd/wire.py:1263`; dipole-emission
delivered power uses scaled current and field spectra in `fdtd/observers.py:1567`
and is excluded from the source-normalisation pass. Broadband plane-wave
`compute_s_parameters(incident_power="auto")` reconstructs the same `2/norm`
factor at `postprocess/scattering_parameters.py:225`, so its returned absolute
incident-power scale can carry this dependence while its ratios cancel.
Time-domain field samples and instantaneous/time-domain flux do not receive this
spectral denominator; longer execution can add time samples without rescaling
the already recorded ones (`fdtd/observers.py:1153`).

## Labels and benchmark implications

On `datagen/p0-profiler`, `benchmark/datagen/scenes.py:60` leaves source
normalisation at its default false. F4 wire scenes use the 40-cycle cap and
1e-5 shutoff; F1/F2/F3 and pulsed F5 use a fixed 12-cycle cap. Thus the affected
readout definitions are:

- F1 `fields` (16,384 sparse points): absolute complex field amplitudes.
- F2 `far_field_E`, `far_field_H`, `closed_surface_flux`; direction vectors are
  geometric and unaffected (`scenes.py:201`).
- F3 printed and F4 wire antennas: `port_incident_power`,
  `port_reflected_power`, `port_accepted_power`, `port_delivered_power`,
  `antenna_e_theta`, `antenna_e_phi`, `antenna_p_rad`, `antenna_p_accepted`,
  `antenna_p_incident`, `closed_surface_flux` (`scenes.py:119`). Impedance,
  reflection coefficient, directivity and consistently formed efficiencies are
  not affected by the common scale alone.
- F5 loaded pulsed scenes: `reflected_flux`, `transmitted_flux`. F5 ongoing
  Bloch CW scenes use a CW measurement window; their converged CW phasors do not
  have the completed-pulse `1/N` decay mechanism (`scenes.py:311`).

Different grid timesteps and caps can therefore embed record-duration factors
in these would-be absolute labels. Merely enforcing field decay does not define
their absolute scale. This is an audit of label/readout definitions, not a claim
that an existing training dataset has been regenerated or inspected.
`benchmark/datagen/profile.py:91` evaluates the readouts and `:50` serialises them
into an in-memory buffer to measure size, then discards them; the profiler itself
does not establish a persistent labelled corpus.

The separate tracked `experiments/fdtd_data_generator/generate.py:545` explicitly
uses `normalize_source=True` and full-field DFT, also recorded in its metadata at
`:896`. Its common-source field labels therefore cancel this particular horizon
factor; this is not evidence that every generator shares the F4 port-label issue.

The unified reference benchmark is a different workflow:

- `benchmark/runner.py:1285` normally enables source normalisation when all field
  sources share one waveform; `:1321` runs Maxwell with a fixed step count from
  `run_time_factor`. FDTD shutoff defaults to zero (`simulation.py:234`). Reference
  export uses the same physical duration and source-normalisation selection
  (`runner.py:1627`). Cache keys include `run_time_factor` and the false
  normalisation override (`runner.py:340`). No reference jobs were submitted.
- Default field comparisons are amplitude-sensitive complex fields in SI spatial
  units **after unit-source spectral normalisation**, not arbitrary peak-rescaled
  images. Scalar fields use phase-only alignment for the main field L2; a
  separate `field_shape_l2` fits amplitude too (`runner.py:2465`,
  `metrics.py:36`). Vector metrics retain amplitude/energy ratio in addition to
  shape (`metrics.py:141`). Reference unit conversion preserves amplitude
  (`runner.py:1368`).
- Flux comparisons retain the numerical power values. Their error is divided by
  one common incident/reference-power scale, not by separate maxima of each
  result (`runner.py:1757`). Thus a multiplicative power mismatch remains visible.
  Maxwell's usual source-normalised flux cancels this horizon factor; reference
  flux is retained as exported. These are source-calibrated comparisons, not raw
  pulse Fourier-integral watt labels.
- Explicit raw-spectrum exceptions such as `directivity_two_dipoles`,
  `modulated_slab`, `modulated_slab_phase`, and `time_monitor_vacuum` compare
  fields relative to a carrier/display RMS and/or scale-invariant scalars
  (`benchmark/scenes/coverage/postprocess.py:391`, `:474`,
  `benchmark/scenes/media/modulated_slab.py:26`,
  `benchmark/scenes/coverage/media.py:303`). Their shape metrics can hide the
  absolute factor. The doubled-duration convergence check (`runner.py:2500`)
  uses the original monitor arrays, so a raw completed pulse can falsely appear
  unconverged solely because of `1/N`, even if display comparison rescales it.
- The previous statement that planned-horizon DFT normalisation changes
  unnormalised amplitudes is correct. Shutoff does not introduce another scale at
  a fixed cap: it restores that cap's scale. The existing
  `tests/fdtd/test_fdtd_autoshutoff.py:88` verifies early-stop/full-run parity at
  **one** plan, not pulse-spectrum independence across different plans.

## Proposed fix, not implemented

Define pulse spectral semantics before changing a numerical factor. Keep CW
peak-phasor averaging (`2/sum(weights)`, or the analytic-CW representation factor)
for converged CW runs. For raw full-response pulses, return the Fourier integral
`dt*sum(x*exp(+iwt))`, with explicit spectral units and metadata; do not present
quadratic Fourier-integral values as ordinary CW watts. A unit-source transfer
phasor is a separate engineering convention: divide the response by the common
unit-waveform spectrum, retaining declared amplitude/phase.

For watt-valued port/antenna labels, apply that same engineering convention to
port V/I, drive available power, fields, flux and surface currents together.
Extend the existing `normalize_source` path to port finalisation (and related
wire/emission outputs) instead of enabling it only for field monitors. Preserve
the port drive's physical amplitude and reference impedance. Resolve multiple
distinct waveforms through the existing common-spectrum restriction or separate
excitation runs; there is no unique single source reference for arbitrary
simultaneous waveforms.

Preserve same-plan shutoff/full-run parity without making the pulse scale depend
on either planned or actual trailing sample count. Record the applied convention,
source denominator and actual sample support. Recompute/version any absolute
data labels whose convention changes; update raw-spectrum benchmark expectations
and retain amplitude-sensitive checks alongside shape/ratio checks. Do not alter
existing cached external references merely to absorb a scale mismatch.

No solver fix or test-expectation change is made in this diagnosis. The reproduction
is the numerical evidence; no additional xfail test was added.

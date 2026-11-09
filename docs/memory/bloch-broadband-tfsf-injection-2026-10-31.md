# Broadband fixed-k Bloch TFSF injection

The 30-degree vacuum reference of `scripts/dev/fdtd/bloch_lumped_acceptance/common.py`
violates the prescribed transverse Bloch phase away from its 3 GHz carrier.
For the saved upstream plane at z = -30 mm, the relative endpoint defect is
`norm(F(y_max) - exp(i*k_y*L_y)*F(y_min)) / norm(F(y_min))`, using native Ex/Hy
spectra, L_y = 20 mm and k_y = (2*pi*3 GHz/c)*sin(30 degrees).

| Frequency (GHz) | Ex defect | Hy defect |
|---|---:|---:|
| 2.70 | 5.9451088e-2 | 6.8099797e-2 |
| 2.85 | 2.8602472e-2 | 3.5080649e-2 |
| 3.00 | 1.0534540e-5 | 7.7427421e-6 |
| 3.15 | 2.7419021e-2 | 3.5827212e-2 |
| 3.30 | 5.5054024e-2 | 7.0734292e-2 |

Thus the carrier is phase-compatible to about 1e-5 relative, while the band edges
miss by 6–7%. These are GPU float32 measurements from the 120,000-step empty
reference with the CFS profile screened on 2026-10-31; they are not lumped-load
errors. Carrier residuals include the discrete/continuum wavenumber difference
and floating-point error.

`_initialize_grating_slab_cw_state` builds delayed-pulse patches using the carrier
direction, and `_launch_bloch_time_shifted_patch` deposits a real delayed
waveform. Across frequency this describes a fixed-angle pulse, whose transverse
wavenumber varies with frequency, rather than the boundary's single fixed k_y.
The carrier agrees; the broadband profile does not. This also affects Bloch B.

A fix needs a complex transverse profile with the same Bloch phase at every
frequency, together with the frequency-dependent normal Yee dispersion and E/H
impedance. For example, a dispersive fixed-k auxiliary incident line could supply
consistent fields at both slab faces and leapfrog clocks. Qualification must
check the endpoint phase across the band, independent absolute fields and
vacuum/loaded flux balance, including source normalization and Bloch B. This
branch keeps injection unchanged: pulsed runs gate source-off decay, their
broadband power residuals are diagnostics, and power conservation gates use CW
at the exactly compatible carrier.

## Full Airy acceptance confirms the defect (2026-11-01)

Branch bloch/fixed-k-floquet 4b6a7f6f (Bloch A merged). Run: `oblique_slab_airy.py`, εr 4 / 12 slab, k∥ = 0.25 / 0.55 k0, TE / TM, 9 frequencies over the band, resolutions 20 / 30 / 40 / 160 cells per vacuum λ. Log: `runs/airy_full.log`. Result: **FAIL**.

Maximum relative error of r or t against oblique Airy, per frequency from the band low edge to the high edge (carrier f0 in the middle):

| εr, k∥/k0, pol | ppw40 | ppw160 |
|---|---|---|
| 4, 0.25, TE | 3.6 … 1.7 … 2.6% | 2.9 … **0.1** … 3.5% |
| 4, 0.25, TM | 5.8 … 1.8 … 7.7% | 5.1 … **0.1** … 2.5% |
| 4, 0.55, TE | 26.9 … 1.3 … 10.9% | 28.5 … **0.1** … 12.4% |
| 4, 0.55, TM | 80.7 … 1.6 … 15.2% | 79.2 … **0.1** … 16.3% |
| 12, 0.25, TE/TM | up to 81% (resonance) | 8.3 / 4.6% at the edges, 0.3% at f0 |
| 12, 0.55, TE/TM | up to 200% | 91 / 243% at the low edge, 0.3% at f0 |

Reading:
- At the carrier the error is 0.1–0.4% at ppw160, so the slab physics, the Bloch boundary and the CFS-CPML are right.
- Away from the carrier the error grows with |f − f0| and with k∥. It is essentially unchanged between ppw40 and ppw160, so it is not discretization.
- The refinement orders collapse (≤ 0.5) for k∥ = 0.55 k0 at εr 4.
- This matches the measured Bloch-phase violation of the injected field off-carrier. The pulsed TFSF injects one real waveform travelling in the carrier direction, while fixed-k physics needs θ(ω) with sin θ = c k∥/ω at every frequency.
- The earlier fast subset looked fine only because it sat near the carrier.

**Fix needed:** fixed-k broadband injection. The incident field must satisfy the fixed-k dispersion k_z(ω) = sqrt(ω²/c² − k∥²) at every frequency, for example a 1D auxiliary line with the k∥² term (∂²/∂t² − c²∂z² + c²k∥²) per polarization. A carrier-only (CW or narrowband) guard is the alternative.

## CW-only guard and its merge gate (2026-11-01)

- **Guard:** in fixed-k Bloch runs (nonzero k∥ on a Bloch axis), directional sources must be CW: plane waves, Gaussian and astigmatic beams, mode sources and custom-field sources. Any other source time fails closed at prepare (`compiler/sources.py`). The census entry is `af76d41c…`, with negative and positive tests in `tests/sources/tfsf/test_fixed_k_source_guard.py`.
- **Broadband slab line (batch 4):** the complex fixed-k incident line (`fdtd/excitation/fixed_k.py`) makes broadband TFSF-slab plane waves exact. They are admitted, and the guard is now `84b8fe15…`. Beams, mode and custom-field sources and soft plane waves stay CW-only.
- **Narrowed 2026-11-01:** the first version (`b635c70c…`, retired) refused every non-CW source and broke 46 existing Bloch tests. Point and current sources are exact at every frequency, because the Bloch boundary supplies the per-frequency fixed-k phase. Only sources injected along one carrier direction are inexact.
  - `tests/sources/point/test_fixed_k_broadband_point_source.py` checks this: one pulsed point dipole (k∥ 0.4 k0, Bloch xy and CPML z) matches separate CW runs at 0.85 and 1.15 f0 to 3e-5 and 9e-5.
  - The pulse spectrum must stay clear of the fixed-k cutoff c·k∥/2π. A pulse with fwidth 0.3 f0 reaches it, and the slow decay near the cutoff then smears the finite-record DFT (1.8% error).
- **Merge gate** (amended by the coordinator, 2026-11-01): `oblique_slab_airy_cw.py` covers εr 4 / 12 × k∥ 0.25 / 0.55 k0 × TE / TM × f/f0 0.85 / 1.0 / 1.15, at ppw 20 / 30 / 40 / 160. It requires every relative error of r and t to be below 1% at ppw160, and the aggregate order 30→40 in (1.5, 2.5). The 20→30 order is reported, not gated.
- **εr 4 result**, all four (k∥, polarization) cases:
  - ppw160 maximum error 0.15–0.16%;
  - order 30→40 of 1.88–2.05 (r) and 2.04–2.05 (t);
  - order 20→30 for r of 1.43 at k∥ 0.25. That is pre-asymptotic: ppw20 is about 10 cells per material λ, and the third frequency sits near a reflection zero of r, with a per-frequency order of 0.86.
- Merged into batch 3 on the εr 4 evidence; εr 12 ran as a follow-up gate.
- **εr 12 result and amended criterion** (coordinator, 2026-11-01). The CW fixed-k path is accepted at εr 12 with no narrowing of the CW scope. Data: `docs/assessments/bloch-oblique-airy-cw-2026-11-01.json`, 100 rows.
  - **The absolute 1% bound fails near reflection minima.** It was calibrated on εr 4. At εr 12 and 1.15 f0, the slab (0.37 λ0, about 1.3–1.5 material wavelengths) sits near a Fabry-Perot minimum with |r| 0.27–0.42, so r is phase sensitive. At ppw160 (about 46 cells per material wavelength) r is off by 3.0–4.6% and t by 1.4–1.7%. Every other εr 12 frequency is below 0.9%.
  - **The error is discretization.** Between ppw 40 and 160 the per-frequency orders are 1.71–2.26. A two-level Richardson extrapolation (40, 160, p = 2) still leaves 0.75–1.07% in r at 1.15 f0, and a reference-free three-level fit gives 0.92–1.23%. Both are inconclusive because ppw 30 and 40 are pre-asymptotic at εr 12.
  - **Decisive check: ppw320 at εr 12, 1.15 f0,** in all four (k∥, polarization) cases. Each run is about 13 min and 0.62 GiB.
    - From 160 to 320, r falls from 4.62/4.63/3.05/3.12% to 1.31/1.31/0.90/0.94%, an order of 1.82/1.82/1.76/1.73. The Richardson residual is 0.37/0.39/0.26/0.31%.
    - t has an order of 1.91–1.94 and a residual of at most 0.03%.
    - This meets the fixed decision rule (order in (1.5, 2.5) and extrapolated r residual below 1%). The miss was extrapolation imprecision; there is no ~1% floor, so no face-snapping investigation is needed.
  - **εr 4 is gated on the 30→40 aggregate order as merged.** Its 40→160 order dips to 1.49–1.53 for r at 0.85 f0. That is a floor effect: the total ppw160 error is only 0.10–0.14%, and a CPML/DFT floor of about 0.05% flattens the order.
  - **Data-plan note.** Near reflection minima of high-index resonant films, relative r is not a usable label metric. Power and absolute metrics are in instamax `docs/plans/data_pretrain_posttrain.md` (§6 P-B, F5 range εr 2.2–4.6).
- **Pulsed (broadband) Airy acceptance, amended 2026-11-02** (coordinator), consistent with the CW criterion above:
  - **εr 4:** absolute error below 1% at ppw160 for every coefficient. All rows pass.
  - **Both permittivities:** the aggregate 30→40 order lies in (1.5, 2.5). Measured: 1.91–2.17.
  - **εr 12:** the absolute ppw160 error is reported, not gated.
    - Near reflection minima it misses 1%: 1.15 f0, plus 0.8 and 1.2 f0 with |r| between 0.03 and 0.34.
    - At 1.15 f0 the pulsed values are **identical to the CW run** (r 4.63/4.63/3.05/3.13%), which is the accepted second-order discretization error.
    - The pulse is qualified per frequency by `broadband_checks.py --check parity` (pulse = CW within 0.2%; all 21 rows pass).
- **Cutoff gate (record length).**
  - The cutoff pulse puts strong content at the fixed-k cutoff, which lingers.
  - With the default 64.7k-step record, the truncated tail moved the fitted below-cutoff decay rate by 23% at 0.45 f0. The pre-fix line was worse: 12% and 36%.
  - A 260k-step record fits the rate to 0.45% at 0.45 f0 and 0.04% at 0.5 f0. The line is then at 6.5e-7 of its peak energy, and the gate asserts below 1e-6.
  - **Incident-reference sub-check (coordinator, 2026-11-02).** The gate's matched-run comparison of the line and the 3D incident reference uses only the bins the narrow cutoff pulse (0.825 f0, fwidth 0.18 f0) drives at ≥0.5 of its peak spectral amplitude, 0.7–1.0 f0. The limit stays 1e-4.
    - Weaker bins amplify residual tail error and are excluded as weak-bin:
      - 1.05 and 1.10 f0, at 0.3–0.5 amplitude, measured 0.95e-4 and 1.4e-4 at 80 cells per wavelength;
      - 0.6 and 1.15–1.2 f0, at about 0.1 amplitude.
    - With the long record, the gap at 160 cells per wavelength was 2.5e-4. At 80 cells per wavelength it was:

| f/f0 | angle | gap in r, standard CPML | gap in r, 2× layers |
|---|---|---|---|
| 0.60 | 66° | 9.5e-5 | 3.5e-4 |
| 0.70 | 52° | 7.1e-5 | 1.6e-4 |
| 0.80–1.00 | 33–43° | 2.4–7.2e-5 | 0.7–2.1e-4 |
| 1.05 | 32° | 9.5e-5 | 2.8e-4 |
| 1.10 | 30° | 1.4e-4 | 4.2e-4 |
| 1.15 | 29° | 2.1e-4 | 6.4e-4 |
| 1.20 | 27° | 3.2e-4 | 9.9e-4 |

    - A thicker CPML makes the gap larger, not smaller, so it is not a grazing absorber return.
    - The full band is qualified by the standard broadband pulse in the parity and leakage gates, with an incident-reference gap of 1.6–5.6e-6.
  - **Masked Airy sub-check (coordinator, 2026-11-02).** Absolute |Δr| and |Δt| bounds at f ≥ 0.8 f0: 0.005 for εr 4 and 0.025 for εr 12. The εr 12 bound was revised after the t data: the accepted class reaches |Δt| = 0.0197 at 1.2 f0 (per-frequency 30→40 orders 1.76–1.90), plus a 0.005 pulse/record allowance. Relative error is ill-posed at the εr 12 Fabry-Perot minima and near the TM Brewster angle. The bounds were set after seeing the data.
    - Measured: εr 4 at most 0.0014 (TE) and 0.0033 (TM); εr 12 |Δr| at most 0.018 and |Δt| at most 0.0214, the accepted second-order class.
    - Regressions of the injection are caught by the pulse = CW parity gate.
  - **The 0.6 and 0.7 f0 bins are excluded.** The TM εr 4 error at 0.6 f0 is about 0.013 and stays flat across resolutions 80, 120 and 160 and across 1x, 2x and 4x records.
    - A 2x thicker CPML halves it, and a CW run is clean (0.001).
    - The slab run's energy grows after source-off (+2.5e-6 per step): a slab-guided TM mode below the light line, under the current complex fixed-k CPML.
    - This belongs to CPML-B (`docs/plans/fdtd-cpml-passive-closure-2026-11-02.md`, gate G2b), not to the injection. See the hazard section in `docs/reference/fdtd-cpml.md`.

## Incident-line slow modes (2026-11-04, CPU analysis)

Evidence: `<evidence-root>/fixed-k-line-band-edge-20261104/`. A numpy reimplementation is
bitwise identical to `FixedKIncidentLine`. Dense eigen analysis of its one-step operator (fields + CPML
memory) and 320k-step late tails, in complex128 and the production complex64. No eigenvalue exceeds 1
beyond 2e-14; the |lambda| = 1 modes are curl-free statics.

- **k_z = 0 cutoff slow waves (physical; not a defect).** The line's lower band edge is the physical cutoff
  f_lo = c k_t / 2pi, where the group velocity also vanishes. Extended standing waves near k_z ~ 0 decay in
  about 115-740 source periods, depending on the line's fixed cell counts; the line CPML stretches only
  d/dz. Their frequency does not move with the grid. The standard acceptance pulse holds this content at
  <= 1e-8. The cutoff-gate pulse excites it strongly (about 3e-2 of peak at +115 ns), and the cutoff gate
  handles it with 320k-step records and a late/peak limit of 1e-6. Do not mistake this tail for the
  band-edge defect below.
- **Upper (Yee) band-edge pole.** A 2-cell mode at 0.99996-0.999995 of
  f_up = asin(S sqrt(1 + (k_t ds/2)^2)) / (pi dt), with vg/c 0.003-0.01, moves with the grid. Its e-fold is
  5.6 us at 40 cells/lambda and 37 us at 160. The line CPML has |G| < 1, which lowers the local band top,
  so near-edge waves turn around 1-2 cells into the absorber: the CPML acts as a mirror. In the acceptance
  regime the source spectrum at f_up is <= 1e-15, the exact tail is <= 1e-9 and the complex64 roundoff floor
  is 2e-8 to 8e-7 of peak. It is unexcited unless a pulse reaches f_up (very coarse grids or very broad
  pulses); the fixed-k follow-up adds the shared prepare-time band-edge check.
- **TM static offset.** A drive with DC content leaves a static curl-free incident field, because the
  point Ex drive at k_t != 0 deposits charge. The cutoff pulse's 4.3e-5 DC gives 3.6e-6 to 1.2e-5 of peak;
  without DC it is 1.2e-9. The fixed-k follow-up adds a shared DC warning, and the line and the reference
  treat DC identically.

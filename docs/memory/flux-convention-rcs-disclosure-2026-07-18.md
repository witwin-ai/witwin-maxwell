# Flux-convention ruling — per-metric RCS disclosure + broadside_H attribution (2026-07-18)

Branch `codex/flux-convention`, worktree `.worktrees/flux-fix`, code HEAD `0c2aa48`
(`fix(fdtd): restore Yee time-stagger observers, full-primal NF2FF, and honest
S-parameter gate`). This note discloses the per-metric benchmark deltas the ruling
produces and closes the one open cell the diagnosis matrix never measured
(`rcs_broadside_H`), so the RCS worsening is attributed with evidence rather than left
implicit.

## Ruling recap (what landed in code)

- **Observers** (`fdtd/observers.py`): electric spectral observers stay on the plain
  running-DFT step phase (offset 0); magnetic observers are retarded by `-0.5*omega*dt`
  (offset -1/2 step). The E-H *relative* label offset is exactly +1/2 step — the physical
  Yee stagger `S = 1/2 Re(E x H*)` needs. The common step phase cancels in `S`.
- **NF2FF** (`postprocess/stratton_chu.py`): the boundary-cell half-weight clip is
  removed; equivalent currents use the full-primal cell-centred midpoint quadrature
  (surface-equivalence invariant, box-size independent).
- **Adjoint seed** (`fdtd/adjoint/seeds.py`): the exact transpose — electric fields pass
  through unshifted, magnetic fields have their `(cos, sin)` schedule rotated by the same
  `-0.5*omega*dt` per entry.
- **S-parameters** (`postprocess/scattering_parameters.py`): the Fresnel S11 gate is
  re-anchored about the analytic `-9.542 dB` with a grid-dispersion tolerance; the prior
  `atol=3` pass under the plain-plain observer is documented as coincidental compensation.

## PART A — `rcs_broadside_H` mechanism (the open matrix cell)

The diagnosis explored a 2x3 matrix of {observer convention} x {clip}. `rcs_broadside_H`
was never measured. Measured now on the diagnosis variant trees (RCS complex-err, 5e8 Hz):

| scenario | a = master (plain+clip) | e (rel-stagger+clip) | d (abs-stagger+no-clip) | f (rel-stagger+no-clip) |
| --- | ---: | ---: | ---: | ---: |
| rcs_dielectric_box | 1.2867e-2 | 6.4868e-2 | **1.3387e-1** | **1.3387e-1** |
| sphere_rcs         | 3.0225e-2 | 8.0018e-2 | **1.1734e-1** | **1.1734e-1** |
| rcs_pec_sphere     | 1.4725e-1 | 1.0078e-1 | 2.6060e-1 | 2.6060e-1 |

Variant `f` is the landed ruling; variant `d` is the "before" (absolute-stagger) state.

**Finding: `f` ≡ `d` to 5-6 significant figures on every RCS cut** (Maxwell magnitudes
1.431071e-1 vs 1.431073e-1 for the box, etc.). This is expected: `d` (E offset +1.0,
H offset +0.5) and `f` (E offset 0, H offset -0.5) differ only by a *common* phase
`exp(i*omega*dt)` applied equally to E and H, which cancels identically in both the
Poynting cross term and the normalized far-field ratio. **The `f` observer choice
introduces no forward change — it is not the mechanism of the RCS worsening.**

Decomposition of the worsening vs master (variant `a`):
- observer stagger alone (`a`->`e`, clip held): box 1.2867e-2 -> 6.4868e-2,
  sphere 3.0225e-2 -> 8.0018e-2 (the Yee-correct relative stagger);
- clip removal alone (`e`->`f`, stagger held): box 6.4868e-2 -> 1.3387e-1,
  sphere 8.0018e-2 -> 1.1734e-1 (the surface-equivalence full-primal quadrature).

Both contributing changes are physically-justified corrections established by the
diagnosis (box-size-independence of the full-primal quadrature; Yee time stagger of the
Poynting cross term). The master `1.2867e-2` / `3.0225e-2` were the *clip's coincidental
benefit* on these specific broadside-H cuts — the lowest-signal cuts of a flat-facet /
one-cell-staircased sphere pattern, where the normalized complex error is inherently
ill-conditioned (`rcs_broadside_E` is already 6.2e-1 even under master). **Verdict:
ACCEPTED-with-disclosure.**

## PART B — full three-way per-metric inventory (this tree vs 7932af3 vs ad3427f)

Full `python -m benchmark --solver fdtd` refreshed `benchmark/RESULTS.md` on this tree
(102 scenarios). Compared against (i) the regressed-master table `7932af3` and (ii) the
`ad3427f` table. Coverage: 84 scalar observables + 172 per-frequency field-metric cells,
full key overlap in all three tables. Jitter band: 5% relative or 5e-4 absolute.

- **Field-metric cells worsened vs 7932af3: 0.** No field L2 / Shape / Linf / Corr cell
  regressed. All hard gate columns (L2 <1e-1, Linf <1e-1, Corr >0.99, Flux <5e-2) are
  unaffected.
- **Scalar cells changed vs 7932af3: 21, and every one matches `ad3427f` exactly**
  (return-to-before). 14 improved, 7 worsened.
- **Genuinely-new regressions (worse than BOTH baselines beyond jitter): 0.**

Net improvements (better vs 7932af3, all == ad3427f), notably:
`rcs_dielectric_box::rcs_back` 1.6868e-1 -> 1.0059e-2, `::rcs_forward` 1.9667e-1 ->
3.9532e-2, `::rcs_broadside_E` 6.1854e-1 -> 4.6286e-1; `sphere_rcs::rcs_forward`
2.3904e-1 -> 9.1217e-2, `::rcs_broadside_E` 6.8158e-1 -> 5.5904e-1;
`rcs_pec_sphere::rcs_broadside_E` 5.2366e-1 -> 4.2281e-1; `antenna_directivity::D_max`
7.0819e-2 -> 6.4039e-2 and both beam widths 9.1744e-2 -> 6.2247e-2;
`directivity_two_dipoles::beam_width_E` 1.7976e-2 -> 3.9315e-3;
`grating_diffraction::eta_+/-3_0` 4.4028e-2 -> 4.1676e-2;
`mode_monitor_two_planes::forward_amplitude_ratio` 6.0542e-2 -> 5.7482e-2.

The 7 worsened cells (classified `return-to-before(=ad3427f)`; NF2FF-derived scalar cuts):

| cell | 7932af3 (i) | this tree | ad3427f (ii) | class |
| --- | ---: | ---: | ---: | --- |
| rcs_dielectric_box::rcs_broadside_H | 1.2867e-2 | 1.3387e-1 | 1.3387e-1 | return-to-before |
| sphere_rcs::rcs_broadside_H | 3.0225e-2 | 1.1734e-1 | 1.1734e-1 | return-to-before |
| rcs_pec_sphere::rcs_broadside_H | 1.4725e-1 | 2.6060e-1 | 2.6060e-1 | return-to-before |
| rcs_pec_sphere::rcs_back | 3.6084e-2 | 1.5669e-1 | 1.5669e-1 | return-to-before |
| rcs_pec_sphere::rcs_forward | 2.7685e-1 | 3.5857e-1 | 3.5857e-1 | return-to-before |
| directivity_two_dipoles::D_max | 3.2885e-2 | 3.8672e-2 | 3.8672e-2 | return-to-before |
| directivity_two_dipoles::beam_width_H | 8.8107e-3 | 1.9244e-2 | 1.9244e-2 | return-to-before |

Interpretation: `7932af3` ("re-anchored host" refresh) carried the clip (variant `a`
behaviour); `ad3427f` was already on the no-clip lineage. This tree returns the whole
scalar set to the `ad3427f` values. The clip in `7932af3` locally helped these 7 cuts at
the cost of the 14 it hurt. No scalar RCS complex-err column has a hard benchmark
acceptance threshold; nothing crosses a gate; the full battery is green. No STOP condition.

## PART C — accurate relationship to the diagnosis variant `vtrees/f`

For the record (correcting any overstated "byte-verified identical to vtrees/f" phrasing —
note: no such claim appears in any tracked doc on this branch; the HEAD commit message is
accurate, so this correction is preventive and lives only here):

- `postprocess/stratton_chu.py` — **byte-identical** to `vtrees/f`.
- `fdtd/observers.py` — **same observer math**, differing comments and local variable
  names only (`offset_cos`/`group_cos` vs `oc`/`gwc`; "Yee time-stagger convention" vs
  "VARIANT E"). Numerically identical, **not** byte-identical.
- `fdtd/adjoint/seeds.py` — **NOT identical**. `vtrees/f`'s
  `_shift_observer_schedule_for_field` was a **no-op** (`del ...; return cos_pack,
  sin_pack`). This tree's is the **real adjoint transpose** (magnetic-field schedule
  rotated by `-0.5*omega*dt`), pinned by
  `tests/fdtd/test_observer_time_stagger.py::test_observer_adjoint_schedule_is_exact_transpose_of_forward_yee_dft`.
  `vtrees/f`'s no-op would fail that test (consistent with its recorded 3 failures).

seeds.py affects only the adjoint/gradient path, not the forward benchmark, which is why
`vtrees/f` and this tree produce identical forward RCS despite the seeds.py difference.

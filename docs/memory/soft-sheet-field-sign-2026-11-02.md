# Soft equivalent-sheet incident-field sign

The power-normalized soft source compiler launched the negative of the prescribed
incident electric and magnetic fields. Reversing both fields preserves power and
Poynting direction, so power-only validation did not expose the defect.

Read-only diagnosis and absolute cached-reference receipts are in
`<evidence-root>/instamax-line-classc-triage-20261101/timing/origin-design/alpha0/`
(`SOFT_SIGN.md`, `ALPHA0_REPORT.md`, `empirical_receipts.json`). The four measured
soft rows, `planewave_vacuum`, `planewave_cw`, `planewave_periodic_vacuum`, and
`pml_only`, had full-monitor absolute field L2 near 2 and weighted complex ratios
near -1. The CW source-phase bridge was exactly 1. No fitted phase or scale was
removed.

Commit `ab944886` (2026-10-23) documented the same orientation error for TFSF
enclosures in
[the analytic CW source assessment](../assessments/fdtd-analytic-cw-source-contract-2026-10-23.md).
It reversed both sheets in `_set_tfsf_state` while deliberately leaving
"modal/open-plane specifications independent". This fix applies the corresponding
orientation correction locally to the power-normalized open-sheet compiler.

The only production change is a negation of the shared `scale` in
`_prepare_power_normalized_surface_source.build_profile_term`, reversing both
electric and magnetic deposits. `initialize_source_terms` is its sole caller,
dispatching soft `PlaneWave`, `GaussianBeam`, and `AstigmaticGaussianBeam` sources.
The shared templates and magnetic-vector helpers, TFSF, modal/custom-field
compilers, source timing, observers, and normalization retain their code.

The absolute field regression uses `Scene -> Simulation -> Result` with the
existing `cpu_reference` backend. Sixteen vacuum plane-wave cases cover x/z,
both directions, both transverse polarizations, and CW/pulse. Their source-normalized
E and H ratios use an independent Yee dispersion relation, distinct component
coordinates, unit-power field amplitude, and the current full-step observer
offset `exp(-i omega dt)`. Both complex ratios require positive real parts; the
relative E/H sign also checks forward Poynting direction. A seventeenth case
compares a propagated soft Gaussian beam near its axis with an independent
analytic beam whose waist lies on the launch plane. No measured phase is fitted.

CPU gates: all 17 new cases fail on `414eb543` (exit 1) and pass with the fix
(exit 0). The guard census passes all seven checks on both trees. Existing
power-scale controls pass 2/2, astigmatic definition controls pass 3/3, and
incident controls pass 4/4 on each tree; their CUDA cases remain skipped.
All 82 captured preparation tensors are `torch.equal` between base and branch:
axis pulse TFSF 44, oblique CW TFSF 24, ModeSource 8, CustomFieldSource 6.

GPU gates (GPU 0, shared runner; receipts in
`<evidence-root>/soft-sheet-sign-20261102/`, `REVIEWER_GPU_RC.txt`):
every source test file and the guard census exit 0 on both trees, except the new
regression (17 fail on base, 17 pass on the fix) and the BITID baseline below.

Absolute benchmark A/B against the cached external reference, with the current
source-phase bridge and no phase or amplitude alignment (benchmark-support L2,
full-monitor weighted ratio phase):

| Row | Base L2 | Fixed L2 | Base arg R | Fixed arg R |
|---|---:|---:|---:|---:|
| planewave_vacuum | 1.9968 | 0.1079 | 3.0353 | -0.1063 |
| planewave_cw | 1.9903 | 0.1951 | 2.9466 | -0.1950 |
| planewave_periodic_vacuum | 1.9971 | 0.1067 | 3.0351 | -0.1065 |
| pml_only | 1.9969 | 0.1070 | 3.0348 | -0.1068 |
| tfsf_vacuum | 1.5917 | 1.5917 | 1.9461 | 1.9461 |

|R| is unchanged (about 1.000). The remaining phase is the full-step observer clock
of the soft family plus the reference's own grid convention; it is the subject of
the separate observer-origin change. `tfsf_vacuum` keeps its separate
incident-reference mismatch.

The soft-beam benchmark rows use the same compiler (default soft injection) and
move the same way (`beam_absolute.py`, same metric, Ex on the display monitor):

| Row | Base L2 | Fixed L2 | Base arg R | Fixed arg R |
|---|---:|---:|---:|---:|
| gaussian_beam_normal | 1.9951 | 0.1086 | 3.0337 | -0.1079 |
| gaussian_beam_defocused | 2.0006 | 0.1356 | 3.0092 | -0.1324 |
| astigmatic_beam | 1.9963 | 0.1042 | 3.0384 | -0.1032 |

|R| is unchanged (0.998-1.005).

BITID baseline (`tests/fdtd/test_bit_identity_baseline.py`): 9 cases stay bitwise,
including TFSF plane-wave, Bloch TFSF, ModeSource and dipole configurations. Four
cases move: `soft_plane_wave_gaussian`, `soft_plane_wave_ricker`,
`gaussian_beam_gaussian`, `gaussian_beam_ricker`. For each, the moved fields are
`torch.equal` to the exact negation of the base fields, magnitudes are bitwise
unchanged, and an archived `414eb543` tree with only the single scale negation
reproduces the branch fields exactly. Re-pinning these four cases requires
explicit approval and is not part of this change.

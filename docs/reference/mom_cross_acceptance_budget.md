# Prospective local MoM/FDTD acceptance budget

Frozen on 2026-09-11 before executing the new FDTD spacings 0.03, 0.02,
and 0.01 m. This budget applies to the local shared-physics comparison subset.
It does not certify external H2/G0, a neural model's error budget, or independent
MoM reference accuracy. The independent core MoM reference target remains 0.5%.

The pilot used FDTD spacings 0.06, 0.05, and 0.04 m and MoM subdivisions 1,
2, and 3. Its finest ellipsoid complex far-field discrepancy was about 5.49%
under the historical unweighted sample norm, so the pilot does not pass the
prospective 5% far-field limit. Historical pilot norms and missing source
provenance remain explicitly preliminary. New acceptance uses the physical
solid-angle weighted norm below; a metric change does not retrospectively
certify the pilot.

## Frozen limits

| Observation or independent control | Maximum discrepancy |
| --- | ---: |
| Empty-reference electric and magnetic field shape | 0.5% each |
| Exterior total E and H, and interior total E and H | 2% each |
| Nonzero exterior scattered E and H | 5% each |
| Nonzero complex electric far-field amplitude | 5% |
| Nonzero RCS | 10% |
| Nonzero scattered power, extinction power, and independent volume loss | 5% each |
| Control-surface angular quadrature and radius sensitivity | 1% each |
| Independent FDTD window, time-step, and PML sensitivity | 1% each |
| Analytically zero quantities, using the absolute normalizations below | 0.5% |

The limits are fixed before the new fine-grid sequence. All failures must be
reported. A successful linear solve, small residual, or three executed meshes
does not establish these limits or demonstrate convergence.

## Physical norms and zero references

All complex fields use SI and exp(-i omega t). One complex gain estimated from
the empty-reference E field is applied to both E and H; there is no separately
fitted H gain. Native observer time staggering is applied exactly once.

Point comparisons use the Euclidean norm over vector components and the fixed
physical query points in each region. Exterior and interior total-field errors
are evaluated separately from saved raw arrays. Nonzero relative error is the
norm of the difference divided by the qualified MoM reference norm. E errors
retain V/m units, H errors A/m. Empty-reference shape uses the analytic plane
wave; zero scattered fields use the analytic incident norm as their denominator.

The far-field norm is sqrt(integral |F_E|^2 dOmega), evaluated using the stored
Gauss-polar and periodic-azimuth weights and directions. RCS uses the same
solid-angle measure. F_E has units V for an E field in V/m; when divided by the
incident electric amplitude, the scattering amplitude has units m. RCS has
units m^2. Nonzero far and RCS errors divide by their corresponding reference
norms. The raw amplitudes and weights must remain available for recomputation.

Analytically zero references have relative error null. Their absolute errors
must be stored with units and normalized as follows:

- Zero E/H: norm of the field error divided by the analytic incident E/H norm
  at the same physical points.
- Zero far amplitude: weighted absolute F_E error divided by
  sqrt(4*pi) * |E_inc| * a, where a is the largest physical semiaxis.
- Zero RCS: weighted absolute RCS error divided by sqrt(4*pi) * pi*a^2.
- Zero power: absolute error in W divided by incident intensity times the
  geometric projected area normal to incidence. For the declared +z cases,
  this area is pi*r_x*r_y; the sphere is the equal-semiaxis case.

No near-zero clamped denominator substitutes for these analytic-zero rules.

## Prospective sequence and independent qualifications

Use the same material, physical geometry, frequency, source direction,
polarization, and physical amplitude/phase for both backends. FDTD interpolation
stencil monitors are numerical sampling aids; they do not introduce extra MoM
query points. The new FDTD spacing sequence is 0.03, 0.02, and 0.01 m with a fixed
MoM subdivision-3 surface (1920 RWG edges). The fixed MoM surface isolates FDTD
refinement. Separate MoM spatial refinement and independent reference evidence
remain mandatory; agreement with an under-resolved MoM surface is insufficient.

MoM integration is fixed at regular order 4 and singular order 4, with adaptive
near-disjoint integration and explicit failure on its budget. Independent
1920-edge lossy-sphere calibration at orders 4/4 versus 6/6 measured a maximum
reference-normalized complex E/H change of 2.47847e-6 on matched interior and
exterior shells. This is calibration evidence for that case, not a universal
material or shape certificate. Query convergence and the true-residual stopping
contract remain enforced independently.

Retain the three spatial levels, the independent PML/time-step/window sweeps,
control-surface angular/radius sweeps, and independent conductivity-weighted Yee
and polyhedral-volume dissipation. Sensitivities compare the same physical
quantity against the finest corresponding control; analytic-zero quantities
retain their stricter 0.5% absolute normalization. Each acceptance report must
record which controls and reference qualifications passed, failed, or remain
unmeasured, together with the source snapshot and numerical configuration.

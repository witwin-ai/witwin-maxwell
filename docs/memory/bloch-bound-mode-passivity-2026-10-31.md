# Fixed-k bound modes and passivity gates

The split-patch fixture with L=5 nH and C=0.5 pF supports a non-radiating mode
below the vacuum fixed-k light line. At a 60-degree carrier angle, an autonomous
native propagation fit gives 2.563905002 GHz, below the 2.598076211 GHz light
line. The fitted amplitude radius is 0.999997300 per step; weak decay reflects
the finite absorbing boundary. A physical lossless bound mode need not decay.
Requiring its energy to vanish by 120,000 steps would favor an unphysical
absorber over a transparent boundary.

Lossless LC acceptance therefore requires the complete small-cell one-step
radius to be at most 1+2 float32 ULPs (1+2.3841858e-7) at 0/10/30/60 degrees,
and bounded field-plus-LC
stored energy over 120,000 steps. Field energy uses actual Yee control volumes;
inductor and capacitor energies include both complex-field banks. Fit the late
2048-step envelope maxima: the three-standard-error upper bound on log-energy
growth must be at most 4e-7 per step, conservatively corresponding to an
amplitude radius floor of 2e-7. This is a one-sided non-growth gate; decay also passes.
The late energy maximum must remain bounded by the same accumulated roundoff
budget. A negative control verifies that growing circuit energy fails even
when field energy decreases.

Lossy R, RLC and static-conductivity cases retain the strict 120,000-step decay
gate: both field-energy and circuit-current-squared tail/peak <1e-6 where
a circuit branch exists. Current squared is energy-like: inductor/branch
energy scales as L*I^2, so its threshold must match the field-energy threshold.
The former 1e-12 current-squared limit demanded current amplitude 1e-6,
three orders stricter than the field's amplitude 1e-3 at energy 1e-6.
With the revised profile, all four LC cases pass the energy gate. Their late
log-energy slopes plus three standard errors at
0/10/30/60 degrees are -3.81558e-7, -1.84383e-6, -1.70972e-4 and -5.38533e-6
per step; the maximum allowed value is 4e-7. All 20 small-cell spectra pass the
two-ULP radius floor, including open, R, LC, RLC and conductivity controls.

All lossy cases pass the matched energy-like decay limits at 120,000 steps.
At 60 degrees, RLC has a 2.567279899 GHz tail with fitted amplitude radius
0.999933367, below the same light line. Its field ratio is 3.71679e-9 at
120,000 steps and
its current-squared ratio is 4.23959e-9; both pass 1e-6.
A 192,000-step diagnostic reaches 7.01371e-13 and 2.68959e-13 respectively;
this longer diagnostic is not needed for the 120,000-step gate.
Broadband source injection remains unchanged and is tracked separately in
[the injection note](bloch-broadband-tfsf-injection-2026-10-31.md).

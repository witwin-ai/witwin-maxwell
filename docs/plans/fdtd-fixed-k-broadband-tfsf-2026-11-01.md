# Broadband fixed-k TFSF injection for oblique Bloch cells (plan, 2026-11-01)

## Problem

Measured in `docs/memory/bloch-broadband-tfsf-injection-2026-10-31.md` (branch `bloch/fixed-k-floquet`).
- A pulsed TFSF plane wave under fixed-k Bloch boundaries injects one real waveform travelling in the carrier direction.
- Fixed-k physics needs, at every frequency, sin θ(ω) = c·k∥/ω and k_z(ω) = sqrt(ω²/c² − k∥²).
- Oblique-slab Airy errors are 0.1–0.4% at the carrier but up to 80% (εr 4, k∥ = 0.55 k0) and 243% (εr 12) at the band edges, independent of resolution.
- Until this lands, fixed-k Bloch runs are restricted to CW sources (capability guard, Bloch B).

## Method

Under fixed k∥ (take k∥ along x; the y case is symmetric), the incident field is exp(i k∥ x) · u(z, t) for all frequencies at once. The fields u satisfy a 1D Maxwell system in which ∂/∂x is replaced by the 3D grid's discrete Bloch derivative.

Let K = (2/Δx)·sin(k∥Δx/2) be the discrete wavenumber of the 3D Yee grid.

TE (E along y; fields E_y, H_x, H_z):
- μ ∂H_x/∂t = ∂E_y/∂z
- μ ∂H_z/∂t = −i K E_y
- ε ∂E_y/∂t = ∂H_x/∂z − i K H_z

TM (H along y; fields H_y, E_x, E_z): the dual system with i K acting on E_z / H_y in the same way.

Discretise on a 1D complex auxiliary line that uses the 3D grid's z spacing and dt and the same Yee staggering in z. Then every frequency propagates with the 3D grid's own discrete fixed-k dispersion, so there is no dispersion mismatch at the TFSF planes. Below the light line the content is evanescent and decays on the line, which is physical.

**Source.** A soft current on the line at z_s, with the user's broadband source time.

**Line ends.** A 1D CPML. CFS is not needed on the line, because its evanescent content does not reach the ends.

**TFSF corrections.** On each z-normal TFSF plane, the correction is the line value at that z times exp(i k∥ x_j) at each transverse Yee position x_j, applied to the real and imaginary banks. This is the existing slab TFSF term layout with a complex, transversely phased profile.

**Incident normalisation.** The line's spectrum at the lower TFSF plane is the incident spectrum (amplitude and phase) per frequency. It is exposed as the run's incident reference, so Floquet r/t need no separate incident run, although the incident run stays available for comparison.

## Scope and files

- `witwin/maxwell/fdtd/excitation/`: a new auxiliary-line mode for fixed-k slab TFSF; complex profile construction in `tfsf_common.py` / `tfsf_specs.py`; line advance alongside `_advance_auxiliary_grid` in `tfsf_apply.py`.
- The line is advanced from the existing TFSF hook. **If the hook in `fdtd/runtime/stepping.py` or anything in `fdtd/ports.py` needs to change, that hunk goes to PCB for ACK before merge.**
- Checkpoint/resume must save and restore the line state, using the existing `tfsf_aux_*` pattern in checkpoint.py.
- Real-field, k = 0 and non-Bloch TFSF paths stay bit-identical.
- The CW-only capability guard is lifted when the gate below passes, and its census entry is retired in the same change.

## Gate (all on GPU, A6000)

1. **Pulsed oblique Airy:** εr 4 and 12; k∥ = 0.25 and 0.55 k0; TE and TM; f/f0 = 0.80 … 1.20 (9 points, band edges included); 20 / 30 / 40 / 160 cells per vacuum λ. Every relative error of r and t below 1% at ppw160, and aggregate refinement order in (1.5, 2.5).
2. **Empty-cell leakage:** with no structure, scattered-field-region energy and Floquet r below 1e-4 of the incident, across the band, for all angles above.
3. **Pulsed vs CW parity:** at f0, 0.9 f0 and 1.1 f0, pulsed r/t matches separate CW runs to < 0.2%.
4. **Graph and resume:** torch.equal between graph and eager runs, and on disk resume mid-pulse.
5. **Bit-identity:** existing TFSF, CPML and port bit-identity tests unchanged; k = 0 pulsed TFSF bit-identical.
6. **Evanescent cutoff** (coordinator, 2026-11-01): k∥ = 0.55 k0 with a band reaching below the fixed-k cutoff ω = c·k∥, e.g. f/f0 from 0.45 to 1.2. The line must show no growth, and its below-cutoff content must decay along z at the discrete evanescent rate; check it against the analytic κ(ω) = sqrt(K² − ω²/c²), adjusted for the grid. Above-cutoff frequencies must still meet gate 1. Below-cutoff r/t are masked in labels, per data plan P-B.
7. **Census:** the CW-only guard entry retired; any new raise gets a reviewed entry and a negative test.

## Estimate

About 4–6 h: Codex implementation 2–3 h, the gate run about 1–2 GPU·h on the local A6000, then review. Batch 4.

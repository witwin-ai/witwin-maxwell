# Array workflow Phase 3 acceptance (MIMO metrics)

Status: functional and analytic-reference gates accepted on branch
`codex/array-phases-2-4`.
Date: 2026-07-17
Scope: one device.

## Delivered contract

- `MultipathEnvironment` carries the angular power spectrum (`power_density`),
  cross-polar ratio (`cross_polar_ratio_db`), and polarization correlation, and
  validates its angular grid against the full-sphere quadrature contract.
- `ArrayBasisData.mimo(environment)` integrates the dual-polarized embedded
  patterns (Taga/Vaughan convention) into:
  - a Hermitian, positive-semidefinite complex correlation matrix `[F, N, N]`,
  - the envelope correlation coefficient (ECC) `[F, N, N]`,
  - apparent diversity gain `10*sqrt(1 - ECC)`,
  - mean effective gain `[F, N]` against the environment's XPR-weighted spectra.
- `ArrayBasisData.ecc_from_scattering()` is the distinct, explicitly named
  lossless S-parameter ECC approximation (Blanch form) and fails closed when
  `1 - sum_n |S_ni|^2 <= 0`.
- All metrics remain in the torch autograd graph (no NumPy detach).

## Independent analytic reference gate

Reference: Clarke's spatial correlation for isotropic 3-D scattering
(R. H. Clarke, Bell Syst. Tech. J., 1968; Vaughan & Andersen, *Antenna Diversity
in Mobile Communications*). For two co-polarized isotropic sensors separated by
`d` along the array axis in a uniform (isotropic) multipath field, the complex
correlation is the closed form `sin(k d)/(k d)`, so `ECC = (sin(k d)/(k d))^2`.
This reference is not produced by the code under test — it is a hand-derived
closed form. The environment's cross-polar ratio cancels in the normalized
correlation, so the gate holds for any XPR.

Frozen budget (`AcceptanceBudget.reference_ecc_error = 0.02`).

| Spacing d (m) | k d | Measured ECC | Analytic (sin kd/kd)^2 | Abs. error |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 7.335 | 0.014017 | 0.014020 | 2.27e-6 |

The parametrized gate runs d in {0.05, 0.15, 0.35, 0.6}; every spacing satisfies
`|ECC - analytic| <= 0.02`, and the measured ECC tracks the spacing-dependent
closed form (a constant or mis-specified formula would diverge across spacings).

Falsification (2026-07-17): substituting a wrong reference `(sin(2kd)/(2kd))^2`
diverges from the measured value (error 1.06e-2 at d=0.35 and larger at smaller
spacings), so the gate discriminates the correct physics.

## Mean-effective-gain analytic gate

Taga's identity: an ideal lossless single-polarization isotropic antenna has
`MEG = XPR/(1+XPR)`.

| Environment XPR | Expected MEG | Measured MEG | Tolerance |
| ---: | ---: | ---: | ---: |
| 0 dB | 0.5 | 0.5 | atol 1e-6 |
| 6 dB | 0.79924 | 0.79924 | atol 1e-6 |

## Correlation-matrix physicality

- Hermitian to a 512*eps float64 band (symmetrized on construction).
- Smallest eigenvalue `>= -1e-9 * largest` (positive semidefinite): measured
  eigenvalue spread on the d=0.25 m, 3 dB XPR case is strictly positive.
- Orthogonal-polarization patterns reach `ECC <= 0.02`; identical co-polarized
  patterns reach `ECC = 1` to 1e-9 and diagonal ECC exactly 1.

## Command

```bash
conda run -n maxwell --no-capture-output python -m pytest tests/rf/array/test_array_mimo.py -q
# 12 passed
```

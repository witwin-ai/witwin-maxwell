# FDTD Grid Convergence

- **Updated:** 2026-07-14 02:03 UTC
- **Scene:** dielectric sphere scattering at 1.500000e+09 Hz
- **Method:** three geometrically refined grids; complex Ex planes are coordinate-aligned and best-fit complex-scaled before shape differencing.

| Resolution (m) | Grid shape | Maxwell time (s) | ms/step | steps/s | DFT samples | Peak GPU (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 6.00000000e-02 | 33x33x33 | 1.032 | 0.1853 | 5397.11 | 2702 | 70.00 |
| 4.00000000e-02 | 41x41x41 | 0.520 | 0.1862 | 5370.02 | 2702 | 24.00 |
| 2.66666667e-02 | 54x54x54 | 0.462 | 0.1632 | 6126.84 | 2702 | 52.00 |

| Comparison | Best-fit-scale shape L2 |
| --- | ---: |
| coarse vs medium | 9.956274e-01 |
| medium vs fine | 6.723757e-01 |

**Observed order:** 0.9682

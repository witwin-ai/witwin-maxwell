# FDFD (iterative) vs FDTD

- **Updated:** 2026-07-08 01:40 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **FDFD config:** sqmr+ssor+double (tol 1e-06)
- **Command:** `python -m benchmark.fdfd_vs_fdtd`

Identical scene through both runtimes. Peak GPU is the driver-level high-water device-usage delta above the pre-run baseline (uniform across the CuPy and Torch allocators). FDFD time is assembly + preconditioner + iterative solve for one source; FDTD time is the stepping-loop wall time for one frequency (extra frequencies are free via the running DFT). The direct (cuDSS) FDFD path is excluded by design.

| Grid | Unknowns | FDFD time (s) | FDFD peak (GB) | FDFD matvecs | FDFD resid | FDTD steps | FDTD time (s) | FDTD peak (GB) | Mem FDFD/FDTD | Time FDFD/FDTD |
|------|----------|---------------|----------------|--------------|------------|------------|---------------|----------------|---------------|----------------|
| 32^3 | 95,232 | 1.9 | 0.22 | 677 | 5.66e-06 | 3101 | 14.6 | 0.01 | 23.0x | 0.1x |
| 48^3 | 324,864 | 19.2 | 0.54 | 4662 | 6.66e-05 | 3901 | 15.4 | 0.06 | 9.6x | 1.2x |
| 64^3 | 774,144 | 90.9 | 1.24 | 13199 | 2.98e-05 | 4702 | 22.8 | 0.10 | 12.9x | 4.0x |
| 96^3 | 2,626,560 | 352.4 | 4.16 | 16000* | 1.23e-02 | 6303 | 31.2 | 0.29 | 14.3x | 11.3x |
| 128^3 | 6,242,304 | 823.8 | 8.01 | 16000* | 2.39e-02 | 7904 | 50.8 | 0.67 | 12.0x | 16.2x |

`*` = FDFD did not reach the tolerance within the matvec budget (time is budget-bound, not solution-bound).

# FDFD Performance Benchmark

- **Updated:** 2026-07-07 02:25 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **Command:** `python -m benchmark.fdfd_performance`

Peak GPU memory is the CuPy memory-pool high-water mark per case (torch-side scene tensors excluded). Matvecs count operator applications, not preconditioner applications.

| Grid | Unknowns | Solver | Precond | Precision | Assembly (s) | Precond setup (s) | Solve (s) | Reuse solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |
|------|----------|--------|---------|-----------|--------------|-------------------|-----------|-----------------|---------|-----------|----------|---------------|--------|
| 32^3 | 95,232 | sqmr(mi=16000,tol=1e-06) | ssor | double | 0.29 | 0.11 | 2.41 | 0.000 | 677 | yes | 5.66e-06 | 0.14 | ok |
| 32^3 | 95,232 | gmres(mi=16000,tol=1e-06,r=200) | ssor | double | 0.03 | 0.04 | 6.43 | 0.000 | 2011 | yes | 5.80e-06 | 0.43 | ok |
| 48^3 | 324,864 | sqmr(mi=16000,tol=1e-06) | ssor | double | 0.03 | 0.05 | 22.37 | 0.000 | 4662 | yes | 6.66e-05 | 0.50 | ok |
| 48^3 | 324,864 | gmres(mi=16000,tol=1e-06,r=200) | ssor | double | 0.03 | 0.04 | 89.51 | 0.000 | 16081 | no | 1.17e-02 | 1.46 | ok |
| 64^3 | 774,144 | sqmr(mi=16000,tol=1e-06) | ssor | double | 0.08 | 0.05 | 111.44 | 0.000 | 13199 | yes | 2.98e-05 | 1.19 | ok |
| 64^3 | 774,144 | gmres(mi=16000,tol=1e-06,r=200) | ssor | double | 0.07 | 0.06 | 174.87 | 0.000 | 16081 | no | 3.65e-02 | 3.49 | ok |

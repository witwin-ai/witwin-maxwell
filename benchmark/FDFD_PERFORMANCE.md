# FDFD Performance Benchmark

- **Updated:** 2026-07-07 00:04 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **Command:** `python -m benchmark.fdfd_performance`

Peak GPU memory is the CuPy memory-pool high-water mark per case (torch-side scene tensors excluded). Matvecs count operator applications, not preconditioner applications.

| Grid | Unknowns | Solver | Precond | Assembly (s) | Precond setup (s) | Solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |
|------|----------|--------|---------|--------------|-------------------|-----------|---------|-----------|----------|---------------|--------|
| 32^3 | 95,232 | gmres(mi=2000,tol=1e-06,r=200) | none | 0.35 | 0.00 | 1.61 | 2011 | no | 1.39e+00 | 0.24 | ok |
| 32^3 | 95,232 | gmres(mi=2000,tol=1e-06,r=200) | jacobi | 0.02 | 0.00 | 0.56 | 2011 | no | 1.21e-01 | 0.24 | ok |
| 32^3 | 95,232 | gmres(mi=2000,tol=1e-06,r=200) | ssor | 0.02 | 0.03 | 6.30 | 2011 | no | 3.44e-02 | 0.26 | ok |
| 32^3 | 95,232 | gmres(mi=2000,tol=1e-06,r=200) | ilu | 0.02 | 0.02 | 7.34 | 2011 | no | 1.22e+01 | 0.26 | ok |
| 48^3 | 324,864 | gmres(mi=2000,tol=1e-06,r=200) | none | 0.04 | 0.00 | 2.89 | 2011 | no | 4.83e-02 | 0.81 | ok |
| 48^3 | 324,864 | gmres(mi=2000,tol=1e-06,r=200) | jacobi | 0.03 | 0.00 | 3.99 | 2011 | no | 6.61e-02 | 0.81 | ok |
| 48^3 | 324,864 | gmres(mi=2000,tol=1e-06,r=200) | ssor | 0.03 | 0.02 | 11.58 | 2011 | no | 2.95e-02 | 0.89 | ok |
| 48^3 | 324,864 | gmres(mi=2000,tol=1e-06,r=200) | ilu | 1.13 | 0.03 | 11.84 | 2011 | no | 9.50e+00 | 0.89 | ok |
| 64^3 | 774,144 | gmres(mi=2000,tol=1e-06,r=200) | none | 0.07 | 0.00 | 9.65 | 2011 | no | 9.39e-02 | 1.94 | ok |
| 64^3 | 774,144 | gmres(mi=2000,tol=1e-06,r=200) | jacobi | 0.07 | 0.00 | 8.08 | 2011 | no | 1.05e-01 | 1.94 | ok |
| 64^3 | 774,144 | gmres(mi=2000,tol=1e-06,r=200) | ssor | 0.13 | 0.04 | 25.76 | 2011 | no | 4.02e-02 | 2.13 | ok |
| 64^3 | 774,144 | gmres(mi=2000,tol=1e-06,r=200) | ilu | 0.06 | 0.04 | 27.33 | 2011 | no | 8.63e+00 | 2.13 | ok |

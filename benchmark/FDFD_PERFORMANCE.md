# FDFD Performance Benchmark

- **Updated:** 2026-07-07 01:55 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **Command:** `python -m benchmark.fdfd_performance`

Peak GPU memory is the CuPy memory-pool high-water mark per case (torch-side scene tensors excluded). Matvecs count operator applications, not preconditioner applications.

| Grid | Unknowns | Solver | Precond | Assembly (s) | Precond setup (s) | Solve (s) | Reuse solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |
|------|----------|--------|---------|--------------|-------------------|-----------|-----------------|---------|-----------|----------|---------------|--------|
| 32^3 | 95,232 | gmres(mi=2000,tol=1e-06,r=200) | jacobi | 0.30 | 0.00 | 0.83 | 0.000 | 2011 | no | 4.93e-02 | 0.24 | ok |
| 32^3 | 95,232 | bicgstab(mi=2000,tol=1e-06) | jacobi | 0.04 | 0.00 | 0.39 | 0.000 | 438 | no | 1.38e+00 | 0.10 | ok |
| 32^3 | 95,232 | tfqmr(mi=2000,tol=1e-06) | jacobi | 0.04 | 0.00 | 2.15 | 0.000 | 2002 | no | nan | 0.10 | ok |
| 32^3 | 95,232 | idr(mi=2000,tol=1e-06) | jacobi | 0.04 | 0.00 | 3.48 | 0.000 | 2000 | no | 3.40e-01 | 0.10 | ok |
| 32^3 | 95,232 | sqmr(mi=2000,tol=1e-06) | jacobi | 0.04 | 0.00 | 1.68 | 0.000 | 2000 | no | 4.46e-04 | 0.10 | ok |
| 48^3 | 324,864 | gmres(mi=2000,tol=1e-06,r=200) | jacobi | 0.04 | 0.00 | 2.10 | 0.000 | 2011 | no | 7.55e-02 | 0.82 | ok |
| 48^3 | 324,864 | bicgstab(mi=2000,tol=1e-06) | jacobi | 0.03 | 0.00 | 3.56 | 0.000 | 4000 | no | 1.73e+02 | 0.34 | ok |
| 48^3 | 324,864 | tfqmr(mi=2000,tol=1e-06) | jacobi | 0.04 | 0.00 | 1.95 | 0.000 | 2002 | no | nan | 0.34 | ok |
| 48^3 | 324,864 | idr(mi=2000,tol=1e-06) | jacobi | 0.04 | 0.00 | 3.43 | 0.000 | 2000 | no | 3.12e-01 | 0.34 | ok |
| 48^3 | 324,864 | sqmr(mi=2000,tol=1e-06) | jacobi | 0.03 | 0.00 | 1.73 | 0.000 | 2000 | no | 7.15e-02 | 0.34 | ok |

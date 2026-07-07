# FDFD Performance Benchmark

- **Updated:** 2026-07-07 00:24 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **Command:** `python -m benchmark.fdfd_performance`

Peak GPU memory is the CuPy memory-pool high-water mark per case (torch-side scene tensors excluded). Matvecs count operator applications, not preconditioner applications.

| Grid | Unknowns | Solver | Precond | Assembly (s) | Precond setup (s) | Solve (s) | Reuse solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |
|------|----------|--------|---------|--------------|-------------------|-----------|-----------------|---------|-----------|----------|---------------|--------|
| 32^3 | 95,232 | direct(mi=2000,tol=1e-06) | jacobi | 0.38 | 0.01 | 1.60 | 0.071 | 0 | yes | 2.56e-05 | 0.10 | ok |
| 48^3 | 324,864 | direct(mi=2000,tol=1e-06) | jacobi | 0.03 | 0.00 | 7.26 | 0.146 | 0 | yes | 1.61e-05 | 0.33 | ok |
| 64^3 | 774,144 | direct(mi=2000,tol=1e-06) | jacobi | 0.06 | 0.00 | 71.66 | 3.608 | 0 | yes | 4.25e-05 | 0.79 | ok |
| 96^3 | 0 | direct(mi=2000,tol=1e-06) | jacobi | 0.00 | 0.00 | 0.00 | 0.000 | 0 | no | nan | 2.69 | failed |

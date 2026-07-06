# FDFD Performance Benchmark

- **Updated:** 2026-07-06 23:41 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **Command:** `python -m benchmark.fdfd_performance`

Peak GPU memory is the CuPy memory-pool high-water mark per case (torch-side scene tensors excluded). Matvecs count operator applications, not preconditioner applications.

| Grid | Unknowns | Solver | Assembly (s) | Solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |
|------|----------|--------|--------------|-----------|---------|-----------|----------|---------------|--------|
| 32^3 | 95,232 | gmres(mi=2000,tol=1e-06,r=200) | 0.69 | 1.15 | 2011 | no | 1.29e-01 | 0.24 | ok |
| 48^3 | 324,864 | gmres(mi=2000,tol=1e-06,r=200) | 0.35 | 4.78 | 2011 | no | 6.61e-02 | 0.81 | ok |
| 64^3 | 774,144 | gmres(mi=2000,tol=1e-06,r=200) | 1.21 | 15.82 | 2011 | no | 1.05e-01 | 1.94 | ok |
| 96^3 | 2,626,560 | gmres(mi=2000,tol=1e-06,r=200) | 3.80 | 81.94 | 2011 | no | 1.08e-01 | 6.60 | ok |

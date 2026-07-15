# FDFD Performance Benchmark

- **Updated:** 2026-07-07 04:38 UTC
- **GPU:** NVIDIA GeForce RTX 5080 (15.9 GB)
- **CuPy:** 13.6.0, **PyTorch:** 2.10.0, **Platform:** Windows
- **Scene:** z-dipole + eps_r=4 cube, resolution 0.02 m (15 cells/wavelength at 1.0 GHz), PML 8 layers
- **Command:** `python -m benchmark.fdfd_performance`

Peak GPU memory is the CuPy memory-pool high-water mark per case (torch-side scene tensors excluded). Matvecs count operator applications, not preconditioner applications.

| Grid | Unknowns | Solver | Precond | Precision | Assembly (s) | Precond setup (s) | Solve (s) | Reuse solve (s) | Matvecs | Converged | Residual | Peak GPU (GB) | Status |
|------|----------|--------|---------|-----------|--------------|-------------------|-----------|-----------------|---------|-----------|----------|---------------|--------|
| 128^3 | 6,242,304 | sqmr(mi=16000,tol=1e-06) | ssor | double | 0.73 | 0.58 | 858.31 | 0.000 | 16000 | no | 2.39e-02 | 9.63 | ok |

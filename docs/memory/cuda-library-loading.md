# CUDA library loading

- On import, torch's legacy-folder dependency glob preloads
  `nvidia/cusolver/lib/libcusolver.so.11` from the cu12 wheel.
- cuda-pathfinder's already-loaded check accepts `libcusolver.so.11` for
  the cusolver descriptor, although the CuPy binding needs `libcusolver.so.12`
  from `nvidia/cu13/lib` and has no RUNPATH.
- CuPy cusolver import can therefore raise `ImportError: libcusolver.so.12`
  depending on execution order. Earlier torch dense algebra masks it by
  loading `.so.12` through torch's RPATH.
- Rule: FDFD dense linear algebra goes through `torch.linalg`, with CUDA
  arrays shared through DLPack; sparse algebra and norms remain in CuPy.
- Related NVRTC builtins issue: CuPy-only cold compiles fail when a
  hand-made unversioned `nvidia/cu13/lib/libnvrtc.so` link exists, because
  it defeats cuda-pathfinder's builtins workaround. Importing torch first
  masks this through torch's RPATH. Removing the link is an environment
  fix (approved 2026-10-31, applied after batch 1), not a code change here;
  `tests/test_cupy_fresh_compile.py` guards it.
- nvmath 1.0.0 has a `cudss_data_ifc` module-import thread-local bug:
  `_tls.size_written` exists only in the importing thread. The
  `_hybrid_min_device_memory` workaround in `fdfd/solver.py` initialises it;
  remove when nvmath initialises the thread-local per thread.

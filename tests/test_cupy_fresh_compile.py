"""A cold-cache CuPy NVRTC compile works without CUDA_HOME or LD_LIBRARY_PATH.

Importing torch first loads NVRTC through torch's RPATH and hides a broken loader
setup, so the child imports only CuPy and compiles into an empty kernel cache.
"""
import os
import subprocess
import sys

CODE = (
    "import cupy as cp\n"
    "k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x) { x[threadIdx.x] += 1.0f; }', 'f')\n"
    "x = cp.zeros(4, dtype=cp.float32)\n"
    "k((1,), (4,), (x,))\n"
    "assert x.get().tolist() == [1.0] * 4\n"
)


def test_cold_cache_cupy_compile_needs_no_cuda_environment(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ('CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH')}
    env['CUPY_CACHE_DIR'] = str(tmp_path / 'kernels')
    process = subprocess.run([sys.executable, '-I', '-c', CODE], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr

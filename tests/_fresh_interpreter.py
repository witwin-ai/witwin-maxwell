"""Run a host-only test check in a fresh interpreter, where CUDA starts uninitialized.

An in-process ``assert not torch.cuda.is_initialized()`` fails whenever an earlier GPU
test in the same pytest process has initialized CUDA, so no-CUDA contracts run here.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_without_cuda(path, function, *args):
    """Call module-level ``function(*args)`` of the file at ``path`` in a fresh interpreter.

    The call must succeed and must leave CUDA uninitialized. ``args`` must be literals
    that round-trip through ``repr``. The child's traceback is the assertion message.
    """
    code = (
        "import runpy,torch\n"
        f"runpy.run_path({str(path)!r})[{function!r}](*{args!r})\n"
        "assert not torch.cuda.is_initialized(), 'host-only check initialized CUDA'\n"
    )
    process = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr

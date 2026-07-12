"""Manual perf harness for the fully-native FDTD adjoint reverse hot path.

Measures per-step wall time of one FDTD adjoint reverse step under the fused
native CUDA reverse backend (``WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND=native``) vs
the analytic Torch reference backend (``torch_reference``), for every
differentiable scene class that has a native reverse runner. The reverse *math*
is the block P6 moved onto fused kernels; the once-per-step mid-H / ADE replay,
the ``dynamic_electric_curls`` coefficient cast, and the source-term VJP stay
Torch in both backends, so the native/reference ratio isolates the reverse-math
speedup.

Timing is deliberately kept OUT of the pytest suite: the shared dev GPU is
contended, so a wall-clock assertion would be flaky. Run this by hand on a quiet
GPU::

    conda activate witwin2
    python scripts/perf_native_adjoint_reverse.py
    python scripts/perf_native_adjoint_reverse.py --iters 400 --classes cpml kerr

Each reverse step is timed with CUDA events over ``--iters`` iterations after a
warmup; the printed ``ms/step`` is the mean device time. The harness reuses the
canonical per-class reverse setups from the native CUDA parity suite (the same
fake reverse solvers), plus a physical Bloch+dispersive checkpoint state.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Allow running as ``python scripts/perf_native_adjoint_reverse.py`` from the
# maxwell root: put that root (which holds both ``witwin`` and ``tests``) at the
# FRONT of the path so the shared per-class reverse case builders resolve to the
# maxwell subproject, ahead of any sibling ``tests`` package elsewhere in the
# witwin monorepo.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# The reverse cases reuse the parity-suite fake reverse solvers.
from tests.gradients.test_fdtd_adjoint_p6_acceptance import (  # noqa: E402
    _NATIVE_CLASSES,
    _build_native_case,
)
from witwin.maxwell.fdtd.adjoint.dispatch import reverse_step  # noqa: E402


def _time_reverse(case, mode, *, iters, warmup):
    solver, forward_state, adjoint_state, eps_ex, eps_ey, eps_ez, _label = case
    os.environ["WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND"] = mode

    def one_step():
        return reverse_step(
            solver,
            forward_state,
            adjoint_state,
            time_value=0.0,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
        )

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        one_step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200, help="timed iterations per backend")
    parser.add_argument("--warmup", type=int, default=30, help="warmup iterations per backend")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(_NATIVE_CLASSES),
        help="reverse classes to time (default: all native classes)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the native adjoint reverse perf harness.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"iters={args.iters} warmup={args.warmup}\n")
    header = f"{'class':<22}{'native ms/step':>16}{'torch ms/step':>16}{'speedup':>12}"
    print(header)
    print("-" * len(header))

    for class_name in args.classes:
        case = _build_native_case(class_name, seed=4242)
        native_ms = _time_reverse(case, "native", iters=args.iters, warmup=args.warmup)
        torch_ms = _time_reverse(case, "torch_reference", iters=args.iters, warmup=args.warmup)
        speedup = torch_ms / native_ms if native_ms > 0 else float("nan")
        print(f"{class_name:<22}{native_ms:>16.4f}{torch_ms:>16.4f}{speedup:>11.2f}x")


if __name__ == "__main__":
    main()

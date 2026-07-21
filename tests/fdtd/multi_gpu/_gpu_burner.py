"""Standalone GPU burner: saturate the single visible CUDA device until killed.

Launched as a co-tenant subprocess by the NCCL adjoint worker's stress mode
(``WITWIN_NCCL_ADJ_STRESS=1``) to reproduce the concurrent-GPU-load condition
under which the distributed reverse gradient was previously corrupted. The parent
pins one physical GPU per burner via ``CUDA_VISIBLE_DEVICES`` and terminates the
process when the stressed gate finishes.
"""

from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(0)
    n = 4096
    a = torch.randn(n, n, device="cuda:0")
    b = torch.randn(n, n, device="cuda:0")
    while True:
        for _ in range(50):
            a = (a @ b).sin_() * 1.0001
        torch.cuda.synchronize("cuda:0")


if __name__ == "__main__":
    main()

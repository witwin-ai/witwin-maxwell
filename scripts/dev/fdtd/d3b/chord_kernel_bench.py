"""CUDA-event timings for each chord CSR specialization and the tensor products.

Run in an assigned GPU window, for example with CUDA_VISIBLE_DEVICES=1. Each
sample replays a graph containing 200 calls, keeping Python dispatch gaps out
of the event interval. Preparation, warmup and capture are outside timing.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.dev.fdtd.d3b.step2_scenes import simulation


def _event_us(run):
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(200):
            run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("copper_patch", "copper_board_60k"), required=True)
    args = parser.parse_args()
    os.environ["WITWIN_MAXWELL_THIN_SHEET_KERNEL"] = "fused"
    prepared = simulation(args.scenario, 1, graph=False).prepare()
    cu = prepared.solver._thin_sheets[0].copper
    plan = cu["fused"]["chord_solve"]
    generator = torch.Generator(device="cuda").manual_seed(314)
    x = torch.randn(2 * cu["count"], dtype=torch.float64, device="cuda", generator=generator)
    y, residual, temporary, direction = (torch.zeros_like(x) for _ in range(4))
    alpha, beta = plan["coefficients"][0] if plan["coefficients"] else (0., plan["inverse_theta"])
    timings = []
    for lanes in (1, 2, 4, 8):
        runs = (
            ("residual", "S", lambda: plan["residual_kernel"](*plan["S"], lanes, x, y, residual)),
            ("matvec", "R", lambda: plan["matvec_kernel"](*plan["R"], lanes, x, temporary)),
            ("direction", "R", lambda: plan["direction_kernel"](
                *plan["R"], lanes, x, direction, alpha, beta, False)),
        )
        for name, matrix, run in runs:
            y.zero_()
            residual.copy_(x)
            direction.copy_(x)
            timings.append(dict(kernel=name, matrix=matrix, L=lanes,
                                selected=lanes == plan[matrix + "_lanes"], us=_event_us(run)))
    for name in ("S", "R"):
        matrix = torch.sparse_csr_tensor(*plan[name], size=cu[name].shape, device=x.device)
        timings.append(dict(kernel="torch.mv", matrix=name, L=None,
                            us=_event_us(lambda: torch.mv(matrix, x, out=temporary))))
    matrices = {}
    for name, key in (("Acx", "acx"), ("Acc", "acc"), ("R", "R"), ("S", "S")):
        native = plan if name in ("R", "S") else cu["fused"]
        matrix = cu[name]
        matrices[name] = dict(rows=matrix.shape[0], nnz=matrix.values().numel(),
                              mean_nnz=matrix.values().numel() / matrix.shape[0],
                              L=native[key + "_lanes"], index_dtype=str(native[key][0].dtype))
    print(json.dumps(dict(scenario=args.scenario, device=torch.cuda.get_device_name(),
                          warmup=20, reps=200, timing="CUDA events over 200-call graph",
                          matrices=matrices, timings=timings), indent=2), flush=True)


if __name__ == "__main__":
    main()

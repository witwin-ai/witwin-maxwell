"""D3 sweep cost table on the F-A spiral: adaptive sweep vs a uniform sweep, per operator backend.

Usage: python scripts/dev/rfic/bench_sweep_d3.py <out_dir> [backend ...]   (default: dense matrix_free hmatrix)

Per backend: one warm-up call, a warm single-frequency call (10 GHz), the adaptive sweep
(0.1-20 GHz, tolerance 1e-3, <= 25 solves), and one uniform 100-point call (warm-started
block GMRES on the iterative backends). Reports solves, block GMRES steps and operator
actions, wall seconds, the adaptive model error against the 100 independent solves, a
passive macromodel of the adaptive samples, and adaptive / single and uniform / single cost
ratios. Writes <out_dir>/sweep_d3.json and <out_dir>/sweep_d3_<backend>.npz.
"""

from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from tests.support.rfic_fixtures import fixture_fa

UNIFORM = torch.linspace(1e8, 20e9, 100, dtype=torch.float64)


def timed(simulation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = simulation.run()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def counts(metadata):
    return dict(steps=list(metadata.get("block_gmres_steps", ())), actions=list(metadata.get("operator_actions", ())))


def main(out, backends):
    out.mkdir(parents=True, exist_ok=True)
    print(mw.__file__, flush=True)
    table = {}
    for backend in backends:
        fixture = fixture_fa()
        layered = replace(fixture.config.layered, operator_backend=backend)
        config = replace(fixture.config, layered=layered)
        timed(mw.Simulation.mom(fixture.scene, frequencies=(5e9,), config=config))
        single, single_seconds = timed(mw.Simulation.mom(fixture.scene, frequencies=(1e10,), config=config))
        sweep_config = replace(config, layered=replace(layered, sweep=mw.mom.AdaptiveSweep(1e8, 20e9, 1e-3, 25)))
        adaptive, adaptive_seconds = timed(mw.Simulation.mom(fixture.scene, config=sweep_config))
        uniform, uniform_seconds = timed(mw.Simulation.mom(fixture.scene, frequencies=tuple(UNIFORM.tolist()),
                                                           config=config))
        sweep = mw.mom.surface_currents(adaptive).sweep
        surrogate = sweep.evaluate_network(UNIFORM).s
        model = mw.mom.passive_macromodel(adaptive.network, tolerance=1e-3)
        macro = model.system.evaluate(UNIFORM.to(model.system.A.device))
        reference = uniform.network.s
        entry = dict(
            single_seconds=single_seconds, single=counts(single.network.metadata),
            adaptive_solves=len(adaptive.network.frequencies), adaptive_seconds=adaptive_seconds,
            adaptive=counts(adaptive.network.metadata),
            uniform_points=len(UNIFORM), uniform_seconds=uniform_seconds, uniform=counts(uniform.network.metadata),
            adaptive_over_single=adaptive_seconds / single_seconds, uniform_over_single=uniform_seconds / single_seconds,
            surrogate_max_error=float((surrogate - reference).abs().max()),
            macromodel_order=model.report.order, macromodel_max_error=float((macro - reference).abs().max()),
            macromodel_passive=model.report.after.passive, macromodel_enforcement_steps=model.report.enforcement_steps)
        table[backend] = entry
        np.savez(out / f"sweep_d3_{backend}.npz", uniform=UNIFORM.numpy(), reference=reference.cpu().numpy(),
                 surrogate=surrogate.cpu().numpy(), samples=adaptive.network.frequencies.cpu().numpy(),
                 sample_s=adaptive.network.s.cpu().numpy())
        print(backend, json.dumps(entry), flush=True)
        del single, adaptive, uniform
        torch.cuda.empty_cache()
    if "dense" in table:
        dense = np.load(out / "sweep_d3_dense.npz")["reference"]
        for backend in table:
            other = np.load(out / f"sweep_d3_{backend}.npz")["reference"]
            table[backend]["uniform_vs_dense_max"] = float(np.abs(other - dense).max())
    (out / "sweep_d3.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:] or ["dense", "matrix_free", "hmatrix"])

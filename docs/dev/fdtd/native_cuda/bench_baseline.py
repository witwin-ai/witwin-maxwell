"""Reproducible performance baseline for the native CUDA FDTD backend.

Measures forward solve ms/step for benchmark scenarios and forward+adjoint
ms/step for a representative differentiable scene. Run with:

    WITWIN_MAXWELL_FDTD_BACKEND=cuda WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION=1 \
        python docs/dev/fdtd/native_cuda/bench_baseline.py [scenario ...]

Timing uses torch.cuda.synchronize around the solve loop, after one warm-up
solve in the same process (extension load + Tidy3D-free).
"""
from __future__ import annotations

import argparse
import os
import time

import torch

import witwin.maxwell as mw
from benchmark.runner import _clone_scene, _compute_num_steps
from benchmark.scenes import SCENARIOS
from witwin.maxwell.simulation import Simulation, TimeConfig


def time_forward(name: str, *, repeats: int = 2) -> dict:
    scenario = SCENARIOS[name]
    rows = []
    n_steps = None
    for _ in range(1 + repeats):  # first run is warm-up
        scene = _clone_scene(scenario.builder(), device="cuda")
        n_steps = _compute_num_steps(scene, scenario.run_time_factor)
        source = scene.sources[0] if scene.sources else None
        normalize = isinstance(source, mw.PlaneWave) and getattr(source, "injection", "soft") == "soft"
        sim = Simulation.fdtd(
            scene,
            frequencies=scenario.frequencies,
            run_time=TimeConfig(time_steps=n_steps),
            spectral_sampler=mw.SpectralSampler(normalize_source=normalize),
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = sim.run()
        torch.cuda.synchronize()
        rows.append(time.perf_counter() - start)
        del result
    timed = rows[1:]
    best = min(timed)
    return {
        "scenario": name,
        "steps": n_steps,
        "elapsed_s": best,
        "ms_per_step": best * 1e3 / n_steps,
        "steps_per_s": n_steps / best,
        "all_runs_s": rows,
    }


class _AdjointScene(mw.SceneModule):
    def __init__(self, shape=(8, 8, 8), init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
            grid=mw.GridSpec.uniform(0.03),
            boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.24, 0.24, 0.24)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.18),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
        return scene


def time_adjoint(*, time_steps: int = 300, repeats: int = 2) -> dict:
    fwd_times, bwd_times = [], []
    for run in range(1 + repeats):  # first run is warm-up
        model = _AdjointScene()
        sim = Simulation.fdtd(
            model,
            frequencies=[1e9],
            run_time=TimeConfig(time_steps=time_steps),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = sim.run()
        value = result.monitor("probe")["components"]["Ez"]
        loss = (value.real**2 + value.imag**2).sum()
        torch.cuda.synchronize()
        fwd = time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd = time.perf_counter() - start
        if run > 0:
            fwd_times.append(fwd)
            bwd_times.append(bwd)
        del model, sim, result, loss
    fwd, bwd = min(fwd_times), min(bwd_times)
    return {
        "scenario": "adjoint_multivoxel",
        "steps": time_steps,
        "forward_s": fwd,
        "forward_ms_per_step": fwd * 1e3 / time_steps,
        "backward_s": bwd,
        "backward_ms_per_step": bwd * 1e3 / time_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="*", default=["dipole_vacuum", "planewave_vacuum"])
    parser.add_argument("--skip-adjoint", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    backend = os.environ.get("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    use_ext = os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    print(f"backend={backend} extension={use_ext} gpu={torch.cuda.get_device_name(0)}")

    scenarios = args.scenarios or ["dipole_vacuum", "planewave_vacuum"]
    for name in scenarios:
        row = time_forward(name, repeats=args.repeats)
        print(
            f"FORWARD {row['scenario']}: steps={row['steps']} best={row['elapsed_s']:.3f}s "
            f"ms/step={row['ms_per_step']:.3f} steps/s={row['steps_per_s']:.1f} runs={['%.3f' % r for r in row['all_runs_s']]}"
        )

    if not args.skip_adjoint:
        row = time_adjoint(repeats=args.repeats)
        print(
            f"ADJOINT {row['scenario']}: steps={row['steps']} fwd={row['forward_s']:.3f}s "
            f"({row['forward_ms_per_step']:.3f} ms/step) bwd={row['backward_s']:.3f}s "
            f"({row['backward_ms_per_step']:.3f} ms/step)"
        )


if __name__ == "__main__":
    main()

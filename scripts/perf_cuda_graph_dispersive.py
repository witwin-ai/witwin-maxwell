"""Manual perf harness for the P5.8 field-update CUDA graph.

Measures per-step wall time with ``cuda_graph=False`` vs ``cuda_graph=True`` on a
launch-bound dispersive AutoGrid scene -- a small grid whose per-step cost is
dominated by kernel-launch overhead, and a Drude/Lorentz medium whose ADE advance
and polarization-current subtraction add several extra launches per step. This is
exactly the regime the CUDA graph targets: capture collapses the whole
field-update block (plus the GPU-driven DFT tail) into a single replay, removing
the per-launch CPU overhead the kernels themselves are too cheap to hide.

Timing is deliberately kept OUT of the pytest suite: the shared dev GPU is
contended, so a wall-clock assertion would be flaky. Run this by hand on a quiet
GPU to confirm the >=2x per-step speedup the phase targets::

    conda activate witwin2
    python scripts/perf_cuda_graph_dispersive.py

The script also re-checks bit-exactness (graph replay must reproduce the eager DFT
field exactly) so a perf run doubles as a correctness smoke test.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import witwin.maxwell as mw

_FREQ = 2.0e9


def _dispersive_autogrid_scene(*, half_extent: float) -> mw.Scene:
    """Small AutoGrid dispersive box + point dipole.

    AutoGrid resolves the Yee step from the material wavelength, so the grid is
    intentionally coarse (launch-bound) while the Drude medium keeps the per-step
    kernel count high (E/H curl + ADE advance + polarization subtraction + DFT).
    """
    bounds = ((-half_extent, half_extent),) * 3
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.auto(min_steps_per_wavelength=12, wavelength=3.0e8 / _FREQ),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(half_extent, half_extent, half_extent)),
            material=mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.5 * half_extent, 0.0, -0.5 * half_extent),
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=_FREQ, fwidth=0.7e9),
        )
    )
    return scene


def _run(scene: mw.Scene, *, cuda_graph: bool, time_steps: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQ,),
        full_field_dft=True,
        run_time=mw.TimeConfig(time_steps=time_steps),
        cuda_graph=cuda_graph,
    ).run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    view = result.at(frequency=_FREQ)
    field = np.asarray(view.E.z.detach().cpu())
    return result.stats(), elapsed, field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-steps", type=int, default=4000)
    parser.add_argument("--half-extent", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=2, help="Timed repeats; the fastest is reported.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This perf harness requires a CUDA device.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    grid_scene = _dispersive_autogrid_scene(half_extent=args.half_extent)
    solver = mw.Simulation.fdtd(
        grid_scene, frequencies=(_FREQ,), run_time=mw.TimeConfig(time_steps=1)
    ).prepare().solver
    print(f"AutoGrid resolved grid: {solver.Nx}x{solver.Ny}x{solver.Nz} cells, dt={solver.dt:.3e}s")
    print(f"Time steps: {args.time_steps}\n")

    def best(cuda_graph: bool):
        best_ms = float("inf")
        stats = None
        field = None
        for _ in range(max(1, args.repeats)):
            stats, elapsed, field = _run(
                _dispersive_autogrid_scene(half_extent=args.half_extent),
                cuda_graph=cuda_graph,
                time_steps=args.time_steps,
            )
            best_ms = min(best_ms, elapsed * 1e3 / args.time_steps)
        return best_ms, stats, field

    off_ms, off_stats, off_field = best(False)
    on_ms, on_stats, on_field = best(True)

    assert off_stats["cuda_graph_active"] is False
    if not on_stats["cuda_graph_active"]:
        raise SystemExit("cuda_graph=True did not engage capture; cannot report a speedup.")

    bit_exact = np.array_equal(off_field, on_field)
    speedup = off_ms / on_ms if on_ms > 0 else float("nan")

    print(f"graph OFF : {off_ms:8.4f} ms/step  (cuda_graph_active={off_stats['cuda_graph_active']})")
    print(
        f"graph ON  : {on_ms:8.4f} ms/step  (cuda_graph_active={on_stats['cuda_graph_active']}, "
        f"tail_graph_active={on_stats['tail_graph_active']})"
    )
    print(f"speedup   : {speedup:6.2f}x")
    print(f"bit-exact : {bit_exact}")
    if not bit_exact:
        raise SystemExit("Graph replay diverged from the eager path; perf number is meaningless.")


if __name__ == "__main__":
    main()

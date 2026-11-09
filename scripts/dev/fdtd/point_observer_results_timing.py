"""Time synchronized result assembly for 16,384 six-field point monitors.

Preparation, time stepping, and one warmup export are outside the measured region.
Acceptance requires a median get_observer_results wall time below 0.5 seconds.
"""

import sysconfig
import gc
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

import witwin.maxwell as mw


def main():
    # Forty physical nodes plus four PML nodes per face give 48**3 = 110,592.
    nodes = np.linspace(-0.195, 0.195, 40)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.195, 0.195),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_source(mw.PointDipole(
        name="source", position=(-0.08, 0.015, -0.04), polarization="Ez", width=0.01,
        source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.6e9),
    ))
    for index in range(16_384):
        position = (-0.155 + 0.01 * (index % 32),
                    -0.155 + 0.01 * ((index // 32) % 32),
                    -0.15 + 0.02 * (index // 1024))
        scene.add_monitor(mw.PointMonitor(f"point_{index:05d}", position,
                                          fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))
    prepared = mw.Simulation.fdtd(
        scene, frequencies=(1.0e9,), run_time=mw.TimeConfig(time_steps=128),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        full_field_dft=False,
    ).prepare()
    solver = prepared.solver
    result = prepared.run()
    assert len(result.monitors) == 16_384
    del result
    warmup = solver.get_observer_results()
    torch.cuda.synchronize()
    del warmup
    print(f"grid={solver.Nx}x{solver.Ny}x{solver.Nz}, "
          f"nodes={solver.Nx * solver.Ny * solver.Nz}, monitors=16384, fields=6, frequencies=1",
          flush=True)

    timings = []
    for repeat in range(5):
        gc.collect()
        torch.cuda.synchronize()
        start = time.perf_counter()
        payload = solver.get_observer_results()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        assert len(payload) == 16_384
        del payload
        print(f"repeat {repeat + 1}: get_observer_results={elapsed:.6f} s", flush=True)
    median = statistics.median(timings)
    print(f"median={median:.6f} s, min={min(timings):.6f} s, max={max(timings):.6f} s")
    assert median < 0.5, f"Point-result assembly took {median:.6f} s; required < 0.5 s"
    print("PASS: median get_observer_results < 0.5 s.")


if __name__ == "__main__":
    main()

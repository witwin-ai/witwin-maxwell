"""Eight sequential FDTD runs WITHOUT gc.collect(): allocated GPU memory must return to baseline after each.

Exercises the grouped single-observer engine (several plane and point observers at several frequencies),
which used to form the cycle FDTD -> _single_observer_engine -> slot.solvers[0] -> FDTD.
Run: env CUDA_VISIBLE_DEVICES=1 python scripts/dev/fdtd/observer_engine_cycle_memory.py
"""
import sysconfig
import os
import sys

os.environ.setdefault("CUDA_HOME", os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"))
os.environ.setdefault("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", os.path.expanduser("~/.cache/witwin-bc-fdtd-build"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import gc
import torch
import witwin.maxwell as mw

gc.disable()
frequencies = (0.9e9, 1.0e9, 1.1e9)


def run_once():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, 0.3),) * 3), grid=mw.GridSpec.uniform(0.012),
                     boundary=mw.BoundarySpec.pml(num_layers=8), device="cuda")
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.05), size=(0.12, 0.12, 0.12)),
                                     material=mw.Material(eps_r=4.0)))
    scene.add_source(mw.PointDipole(position=(0.0, 0.0, -0.1), polarization="Ez", profile="ideal",
                                    source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9)))
    scene.add_monitor(mw.PlaneMonitor(name="xz", axis="y", position=0.0, fields=("Ex", "Ez"), frequencies=frequencies))
    scene.add_monitor(mw.PlaneMonitor(name="xy", axis="z", position=0.15, fields=("Ex", "Ey"), frequencies=frequencies))
    for i in range(4):
        scene.add_monitor(mw.PointMonitor(f"p{i}", position=(0.05 * i, 0.02, 0.1), fields=("Ex", "Ez")))
    result = mw.Simulation.fdtd(scene, frequencies=frequencies).prepare().run()
    return float(result.monitor("p0")["Ez"].abs().max())


run_once()
torch.cuda.synchronize()
baseline = torch.cuda.memory_allocated()
print(f"baseline after warm-up run: {baseline / 2**20:.1f} MiB")
for i in range(8):
    value = run_once()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    print(f"run {i}: allocated {allocated / 2**20:.1f} MiB (|Ez| {value:.3e})")
    assert allocated <= baseline + 1024 ** 2, "allocated memory grew without gc.collect(): a finished solver survived"
print("PASS: allocated memory returns to baseline after every run without gc.collect()")

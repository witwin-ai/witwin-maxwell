"""Compare old/new antenna postprocessing over identical saved FDTD tensors.

Each scene is solved once. Fresh old/new worker interpreters reload its snapshot
on CUDA and time their first antenna call, without warming the surface cache.
The old worker loads the four changed modules from BASELINE using git show;
all other code and all solver samples are shared. No checkout is modified.
Use --baseline f18ead5c --profile to compare the first optimization with the
follow-up and print CPU-side cProfile statistics (no CUDA profiler).

The printed builder below vendors only the helpers needed by ``printed`` from
datagen/p0-profiler:benchmark/datagen/scenes.py at
e4cddaba4c7cf782b2a21b445e9c55fdfdbe68dd. Its geometry, grid, frequencies and
excitation/run configuration are the f3_medium_21f catalog entry.

Run under the coordinator's GPU lock:
    env CUDA_VISIBLE_DEVICES=1 python scripts/dev/fdtd/antenna_postprocess_bitid_timing.py
"""

from __future__ import annotations

import sysconfig
import argparse
import cProfile
from dataclasses import fields, is_dataclass
import gc
import math
import os
from pathlib import Path
import pstats
import subprocess
import sys
import tempfile
import time
import types

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.scenes.antenna.half_wave_dipole import (
    default_frequencies,
    half_wave_dipole_scene,
)

BASELINE = "042e0790c149301ba3b1a3a31a7d4472f5aa601a"
SCENES = ("f3_medium_21f", "half_wave_dipole")
PML = 8


def _grid_scene(axes, *, device):
    return mw.Scene(
        domain=mw.Domain(tuple((float(axis[0]), float(axis[-1])) for axis in axes)),
        grid=mw.GridSpec.custom(*axes),
        boundary=mw.BoundarySpec.pml(num_layers=PML),
        device=device,
    )


def _box(position, size):
    return mw.Box(position=torch.tensor(position, dtype=torch.float64),
                  size=torch.tensor(size, dtype=torch.float64))


def _partition(anchors, step, *, fine=(), half_planes=()):
    """Partition fixed physical anchors, preserving native H-contour half nodes."""
    contour_step = min(step, .0004)
    brackets = tuple((plane - contour_step / 2, plane + contour_step / 2)
                     for plane in half_planes)
    required = sorted(set((*anchors, *(x for lo, hi, _ in fine for x in (lo, hi)),
                           *(x for pair in brackets for x in pair))))
    nodes = [required[0]]
    for lo, hi in zip(required, required[1:]):
        target = step
        for first, last, local_step in fine:
            if first <= (lo + hi) / 2 <= last:
                target = min(target, local_step)
        count = (1 if (lo, hi) in brackets else
                 max(1, math.ceil((hi - lo) / target - 1e-12)))
        nodes.extend(np.linspace(lo, hi, count + 1)[1:])
    return np.asarray(nodes)


def _printed_axes(lateral, air, finest, height):
    x = _partition((-.065, -.045, -.040, -.001, 0., .001, .040, .045, .065), air,
                   fine=((-.045, .045, lateral), (-.002, .002, finest)))
    y = _partition((-.025, -.0055, -.001, 0., .001, .0055, .025), air,
                   fine=((-.0055, .0055, lateral),), half_planes=(-.0035, .0035))
    z = _partition((-.025, -height, 0., .025), air,
                   fine=((-height, 0., min(height / 7, finest)),),
                   half_planes=(-.0025, .0025))
    return x, y, z


def printed(*, lateral, air, finest, height, count, lossy, device):
    """90 x 11 mm copper printed dipole with an explicit finite-width LumpedPort."""
    frequencies = tuple(np.linspace(1.8e9, 3.0e9, count))
    axes = _printed_axes(lateral, air, finest, height)
    scene = _grid_scene(axes, device=device)
    scene.add_structure(mw.Structure(
        _box((0., 0., -height / 2), (.090, .011, height)),
        mw.Material(eps_r=3.5, sigma_e=.01 if lossy else 0.), name="substrate",
    ))
    for arm, lo, hi in (("negative", -.040, -.001), ("positive", .001, .040)):
        scene.add_structure(mw.Structure(
            _box(((lo + hi) / 2, 0., 0.), (hi - lo, .002, 0.)),
            mw.CopperFoil(thickness=35e-6, frequency_range=(.1e9, 10e9)), name=arm,
        ))
    scene.add_port(mw.LumpedPort(
        "feed", positive=(.001, 0., 0.), negative=(-.001, 0., 0.),
        voltage_path=mw.AxisPath("x"), reference_impedance=50.,
        current_surface=_box((axes[0][np.searchsorted(axes[0], 0.) + 1] / 2, 0., 0.),
                             (0., .007, .005)),
        voltage_aperture=_box((0., 0., 0.), (.002, .002, 0.)),
    ))
    scene.add_monitor(mw.ClosedSurfaceMonitor.box(
        "radiation", size=(.110, .035, .035), frequencies=frequencies,
    ))
    return scene, frequencies


def _scene(name):
    if name == "f3_medium_21f":
        return printed(lateral=.001, air=.002, finest=.0002, height=.0014,
                       count=21, lossy=False, device="cuda")
    frequencies = default_frequencies(3.0e9)
    return half_wave_dipole_scene(frequencies=frequencies), frequencies


def _solve_once(name, destination):
    scene, frequencies = _scene(name)
    if name == "f3_medium_21f":
        center = (frequencies[0] + frequencies[-1]) / 2
        result = mw.Simulation.fdtd(
            scene, frequencies=frequencies,
            run_time=mw.TimeConfig.auto(steady_cycles=12, transient_cycles=8),
            shutoff=0.0, full_field_dft=False,
            excitations=(mw.PortExcitation(
                "feed", source_time=mw.GaussianPulse(frequency=center, fwidth=.8 * center),
            ),),
        ).run()
    else:
        # Match the maintained half-wave antenna benchmark's 12 ns solve.
        simulation = mw.Simulation.fdtd(
            scene, frequencies=list(frequencies),
            excitations=mw.PortExcitation(
                "feed", source_time=mw.GaussianPulse(frequency=3.0e9, fwidth=1.5e9),
            ),
            run_time=mw.TimeConfig(time_steps=1),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        )
        prepared = simulation.prepare()
        simulation.config.run_time = mw.TimeConfig(
            time_steps=math.ceil(12.0e-9 / float(prepared.solver.dt))
        )
        result = prepared.run()
    result.save(destination)
    print(f"{name}: solved once; saved common observer tensors", flush=True)


def _load_old_modules(revision):
    for suffix in ("result", "postprocess.stratton_chu", "postprocess.nfft", "postprocess.antenna"):
        name = "witwin.maxwell." + suffix
        path = "witwin/maxwell/" + suffix.replace(".", "/") + ".py"
        source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / path)
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
        exec(compile(source, f"{revision}:{path}", "exec"), module.__dict__)
        setattr(sys.modules[module.__package__], name.rpartition(".")[2], module)
    mw.Result = sys.modules["witwin.maxwell.result"].Result


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if is_dataclass(value):
        return {field.name: _snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _worker(args):
    if args.worker == "old":
        _load_old_modules(args.baseline)
    scene, _ = _scene(args.scene)
    result = mw.Result.load(args.result, scene=scene, map_location="cuda")
    # Restore the prepared material state that the completed solve already had.
    # Do not retrieve a monitor or construct any surface currents before timing.
    result.prepared_scene.compile_materials()
    torch.cuda.synchronize()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
    start = time.perf_counter()
    data = result.antenna(surface="radiation", driven_port="feed",
                          theta_points=19, phi_points=37)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if args.profile:
        profiler.disable()
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)
    print(f"{args.scene}: {args.worker} first antenna(19 x 37) = {elapsed:.6f} s", flush=True)
    antenna = _snapshot(data)
    for name in ("directivity_db", "gain_db", "realized_gain_db", "axial_ratio_db",
                 "directivity_max", "gain_max", "realized_gain_max"):
        antenna[name] = _snapshot(getattr(data, name))
    monitor = _snapshot(result.monitor("radiation"))
    selected = [_snapshot(result.monitor("radiation", freq_index=index))
                for index in range(len(result.frequencies))]
    repeated = result.antenna(surface="radiation", driven_port="feed",
                              theta_points=19, phi_points=37)
    _assert_equal(_snapshot(data), _snapshot(repeated), "repeated antenna")
    _assert_equal(monitor, _snapshot(result.monitor("radiation")), "repeated monitor")
    torch.save(dict(antenna=antenna, monitor=monitor, selected=selected,
                    closed_surface_flux=monitor["flux"], seconds=elapsed), args.output)


def _assert_equal(old, new, path):
    if isinstance(old, torch.Tensor):
        assert old.shape == new.shape and old.dtype == new.dtype, path
        assert torch.equal(old, new), path
    elif isinstance(old, dict):
        assert old.keys() == new.keys(), path
        for key in old:
            _assert_equal(old[key], new[key], f"{path}.{key}")
    elif isinstance(old, (tuple, list)):
        assert type(old) is type(new) and len(old) == len(new), path
        for index, (first, second) in enumerate(zip(old, new)):
            _assert_equal(first, second, f"{path}[{index}]")
    else:
        assert old == new, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE, help="Git revision for the old postprocessing modules.")
    parser.add_argument("--profile", action="store_true", help="Print host-side cProfile data for the first antenna call.")
    parser.add_argument("--worker", choices=("old", "new"), help=argparse.SUPPRESS)
    parser.add_argument("--scene", choices=SCENES, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    with tempfile.TemporaryDirectory(prefix="antenna-postprocess-") as temporary:
        directory = Path(temporary)
        for name in SCENES:
            result_path = directory / f"{name}-result.pt"
            _solve_once(name, result_path)
            gc.collect()
            torch.cuda.empty_cache()
            outputs = {}
            for version in ("old", "new"):
                output = directory / f"{name}-{version}.pt"
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker", version,
                    "--scene", name, "--result", str(result_path), "--output", str(output),
                    "--baseline", args.baseline,
                ]
                if args.profile:
                    command.append("--profile")
                subprocess.run(command, cwd=ROOT, check=True)
                outputs[version] = torch.load(output, map_location="cpu", weights_only=False)
            for key in ("antenna", "monitor", "selected", "closed_surface_flux"):
                _assert_equal(outputs["old"][key], outputs["new"][key], f"{name}.{key}")
            old_time, new_time = (outputs[version]["seconds"] for version in ("old", "new"))
            print(f"{name}: ALL OUTPUTS torch.equal; old={old_time:.6f} s, "
                  f"new={new_time:.6f} s, speedup={old_time / new_time:.2f}x", flush=True)
            if name == "f3_medium_21f":
                print(f"f3_medium_21f target < 2 s: {new_time < 2.0}", flush=True)


if __name__ == "__main__":
    main()

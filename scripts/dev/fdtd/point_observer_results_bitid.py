"""Compare point-monitor payloads against the original assembly in fresh processes.

Run with the maxwell interpreter; the default command launches both workers and
asserts torch.equal for every returned tensor, including component aliases.
The baseline function is read from a git revision (--baseline-rev, default 31c1e750) without changing the tree;
use the pre-merge master when re-checking after a merge.
"""

import sysconfig
import argparse
import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["CUDA_HOME"] = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = os.path.expanduser("~/.cache/witwin-bc-fdtd-build")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: snapshot(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(child) for child in value)
    if isinstance(value, list):
        return [snapshot(child) for child in value]
    return value


def dump(mode, output, baseline_rev):
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd import observers

    if mode == "baseline":
        source = subprocess.check_output(
            ["git", "show", f"{baseline_rev}:witwin/maxwell/fdtd/observers.py"],
            cwd=ROOT, text=True,
        )
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "get_observer_results")
        namespace = dict(vars(observers))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<baseline observers>", "exec"), namespace)
        solver_module = importlib.import_module("witwin.maxwell.fdtd.solver")
        solver_module.get_observer_results_impl = namespace["get_observer_results"]

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.32, 0.32),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_structure(mw.Structure(
        name="dielectric", geometry=mw.Box(position=(0.04, 0.08, 0.0), size=(0.16, 0.16, 0.16)),
        material=mw.Material(eps_r=3.5),
    ))
    scene.add_source(mw.PointDipole(
        name="source", position=(-0.12, -0.04, 0.04), polarization="Ez", width=0.04,
        source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.6e9, amplitude=100.0),
    ))
    field_sets = (
        ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), ("Ez",), ("Hx",),
        ("Ex", "Hy"), ("Ey", "Hz"), ("Ez", "Hx", "Hy"),
    )
    for index in range(200):
        position = (-0.24 + 0.048 * (index % 10),
                    -0.20 + 0.10 * ((index // 10) % 5),
                    -0.18 + 0.12 * (index // 50))
        scene.add_monitor(mw.PointMonitor(f"point_{index:03d}", position,
                                          fields=field_sets[index % len(field_sets)]))
    result = mw.Simulation.fdtd(
        scene, frequencies=(0.9e9, 1.1e9), run_time=mw.TimeConfig(time_steps=160),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        full_field_dft=False,
    ).prepare().run()
    payload = snapshot(result.monitors)
    assert len(payload) == 200
    assert any(torch.count_nonzero(data).item() > 0
               for monitor in payload.values() for data in monitor["components"].values())
    torch.save(payload, output)
    print(f"{mode}: saved {len(payload)} monitor payloads", flush=True)


def compare(left, right, path="monitors"):
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("baseline", "vectorized"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-rev", default="31c1e750")
    args = parser.parse_args()
    if args.worker:
        dump(args.worker, args.output, args.baseline_rev)
        return
    with tempfile.TemporaryDirectory(prefix="point-observer-bitid-") as directory:
        paths = [Path(directory) / f"{mode}.pt" for mode in ("baseline", "vectorized")]
        for mode, output in zip(("baseline", "vectorized"), paths):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--worker", mode, "--output", str(output), "--baseline-rev", args.baseline_rev], cwd=ROOT, check=True)
        compare(*(torch.load(path, weights_only=True) for path in paths))
    print("PASS: all 200 monitor payloads are bit-identical (mixed fields, two frequencies).")


if __name__ == "__main__":
    main()

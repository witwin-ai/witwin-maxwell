"""Profile CPU sheet compilation and compare every output against archived master.

Run with CUDA_VISIBLE_DEVICES='' in the maxwell environment. Workers import
each tree through PYTHONPATH from a temporary working directory.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import fields, is_dataclass
import io
import json
import os
from pathlib import Path
import pickle
import pstats
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

NONPLANAR_SCENES = ("copper_bent", "copper_vee", "pec_bent", "pec_vee", "copper_curved_cpml")
SCENES = ("copper_patch", "copper_board_60k", "pec_feed", *NONPLANAR_SCENES)


def snapshot(value):
    import numpy as np
    import torch

    if isinstance(value, (np.ndarray, torch.Tensor)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    if is_dataclass(value):
        return {field.name: snapshot(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return snapshot(vars(value))
    return value


def compare(left, right, path="compiled"):
    import numpy as np
    import torch

    assert type(left) is type(right), (path, type(left), type(right))
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert np.array_equal(left, right), path
        assert left.tobytes() == right.tobytes(), (path, "bits")
    elif isinstance(left, dict):
        assert list(left) == list(right), (path, "key order")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]")
    else:
        a, b = np.asarray(left), np.asarray(right)
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), path
        if a.dtype.kind in "fciu":
            assert a.tobytes() == b.tobytes(), (path, "bits")


def worker(args):
    # The worker script lives in the worktree; scene helpers must come from
    # the selected PYTHONPATH tree, including when profiling archived master.
    sys.path.pop(0)
    import numpy as np
    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.compiler import thin_sheet as compiler
    from witwin.maxwell.fdtd import thin_sheet as runtime
    from witwin.maxwell.scene import prepare_scene

    torch.set_num_threads(1)
    stages = {}

    def timed(name, function):
        def call(*positional, **keywords):
            start = time.perf_counter()
            result = function(*positional, **keywords)
            stages[name] = stages.get(name, 0.0) + time.perf_counter() - start
            return result
        return call

    start = time.perf_counter()
    if args.scene == "pec_feed":
        import scipy.sparse.linalg as sla

        from tests.materials.thin_sheet.test_thin_sheet_feeds import _runtime_scene
        scene = prepare_scene(_runtime_scene("pec", tilted=True))
    elif args.scene == "copper_curved_cpml":
        from tests.materials.thin_sheet.test_thin_sheet_feed_antennas import WIDTH, cylindrical_strip

        # Extrude to mixed end walls so tangential edges are free on PMC and
        # held on PEC, with CPML on the other four faces.
        scene = prepare_scene(mw.Scene(
            domain=mw.Domain(bounds=((-0.12, 0.12), (-WIDTH / 2, WIDTH / 2), (-0.08, 0.06))),
            grid=mw.GridSpec.uniform(0.005), device="cpu",
            boundary=mw.BoundarySpec.faces(default="pml", num_layers=4, y=("pmc", "pec")),
            structures=[mw.Structure(geometry=cylindrical_strip(), material=mw.CopperFoil(
                thickness=35e-6, frequency_range=(0.5e9, 2e9)))]))
    elif args.scene in NONPLANAR_SCENES:
        from tests.materials.thin_sheet.test_thin_sheet_faceted import _structure
        from tests.materials.thin_sheet.test_thin_sheet_runtime import _cavity

        material, kind = args.scene.split("_")
        scene = prepare_scene(_cavity(_structure(kind, material), device="cpu"))
    else:
        from thin_sheet_overhead import SCENARIOS, _scene
        original = mw.Scene

        def cpu_scene(**keywords):
            return original(**(keywords | {"device": "cpu"}))

        with patch.object(mw, "Scene", cpu_scene):
            scene = prepare_scene(_scene(SCENARIOS[args.scene], True))
    stages["scene"] = time.perf_counter() - start
    profile = cProfile.Profile()
    with patch.object(compiler, "_compile_chords", timed("chords", compiler._compile_chords)), \
         patch.object(compiler, "_compile_sheet_feeds", timed("feeds", compiler._compile_sheet_feeds)):
        profile.enable()
        compiled = timed("compile", compiler.compile_thin_sheet)(scene, scene.structures[0])
        profile.disable()
    if args.scene in NONPLANAR_SCENES:
        assert not compiled.surface.planar
        assert compiled.statistics["fold_faces"] > 0
        coverage = {name: sum(entry.callcount for entry in profile.getstats()
                              if getattr(entry.code, "co_name", None) == name)
                    for name in ("pierce", "box_below_volume")}
        assert all(coverage.values()), coverage
        coverage.update(facets=len(compiled.surface.facets), fold_faces=compiled.statistics["fold_faces"])
        if args.scene == "copper_curved_cpml":
            tangential = compiled.e_component != 1
            pmc = tangential & (compiled.e_index[:, 1] == 0)
            pec = tangential & (compiled.e_index[:, 1] == len(scene.y_nodes64) - 1)
            assert pmc.any() and pec.any()
            assert not compiled.e_fixed[pmc].any() and compiled.e_fixed[pec].all()
            coverage.update(pmc_wall_edges=int(pmc.sum()), pec_wall_edges=int(pec.sum()))
        print(json.dumps(dict(scene=args.scene, coverage=coverage)), flush=True)
    result = {"direct": snapshot(compiled)}
    if args.scene == "pec_feed":
        # Uniform-vacuum inputs to the host-only dt preparation function; no
        # field stepping, material-grid compilation or CUDA initialization.
        solver = SimpleNamespace(scene=scene, device="cpu", cpu_reference=True,
                                 mu0=4e-7 * np.pi, dt=2e-12, source_frequency=1e9)
        shape = tuple(len(getattr(scene, f"{axis}_nodes64")) for axis in "xyz")
        for component in ("Ex", "Ey", "Ez"):
            setattr(solver, "eps_" + component, torch.full(shape, 8.8541878128e-12, dtype=torch.float64))
            setattr(solver, "sigma_e_" + component, torch.zeros(shape, dtype=torch.float64))
        dt_profile = cProfile.Profile()
        with patch.object(compiler, "_compile_chords", timed("pec_chords", compiler._compile_chords)), \
             patch.object(compiler, "_compile_sheet_feeds", timed("pec_feeds", compiler._compile_sheet_feeds)), \
             patch.object(sla, "eigsh", timed("pec_eigsh", sla.eigsh)):
            dt_profile.enable()
            timed("pec_time_step", runtime.prepare_thin_sheet_time_step)(solver)
            dt_profile.disable()
        dt_profile.dump_stats(str(args.output.with_suffix(".dt.prof")))
        stages["pec_chord_builds"] = sum(entry.callcount for entry in dt_profile.getstats()
                                         if getattr(entry.code, "co_name", None) == "_compile_chords")
        result["time_step"] = snapshot(solver._thin_sheet_complexes)
        result["dt"] = solver.dt
        result["cfl_limit"] = solver._thin_sheet_cfl_limit
    args.output.with_suffix(".pickle").write_bytes(pickle.dumps(result, protocol=5))
    profile.dump_stats(str(args.output.with_suffix(".prof")))
    report = io.StringIO()
    for order in ("cumulative", "tottime"):
        pstats.Stats(profile, stream=report).strip_dirs().sort_stats(order).print_stats(20)
    if args.scene == "pec_feed":
        pstats.Stats(dt_profile, stream=report).strip_dirs().sort_stats("cumulative").print_stats(20)
    args.output.with_suffix(".txt").write_text(report.getvalue())
    args.output.with_suffix(".json").write_text(json.dumps(stages, indent=2) + "\n")
    print(json.dumps(dict(scene=args.scene, stages=stages,
                         traces=compiled.statistics["traces"],
                         chords=0 if compiled.chord_face is None else len(compiled.chord_face))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[3]
    results = (args.results or Path(tempfile.mkdtemp(prefix="thin-sheet-cpu-results-"))).resolve()
    results.mkdir(parents=True, exist_ok=True)
    scenes = (args.scene,) if args.scene else SCENES
    with tempfile.TemporaryDirectory(prefix="thin-sheet-master-") as temporary:
        baseline = Path(temporary) / "master"
        baseline.mkdir()
        archive = subprocess.run(["git", "archive", "master"], cwd=root, check=True, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(baseline)], input=archive.stdout, check=True)
        for name in scenes:
            for label, tree in (("master", baseline), ("worktree", root)):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLCONFIGDIR="/tmp/thin-sheet-mpl",
                           PYTHONPATH=os.pathsep.join((str(tree), str(tree / "scripts/dev/fdtd"))))
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--scene", name,
                                "--output", str(results / f"{name}-{label}")],
                               cwd=temporary, env=env, check=True)
            a, b = (pickle.loads((results / f"{name}-{label}.pickle").read_bytes())
                    for label in ("master", "worktree"))
            compare(a, b, name)
            print(f"{name}: PASS (every array/tensor/scalar, dtype, shape, bits and container order)", flush=True)
    print(f"Profiles and per-stage CPU times: {results}", flush=True)


if __name__ == "__main__":
    main()

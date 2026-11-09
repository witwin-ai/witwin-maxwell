"""CPU CopperFoil setup profile and exact comparison against an archived master."""
from __future__ import annotations

import argparse
import cProfile
import io
import os
from pathlib import Path
import pickle
import pstats
import subprocess
import sys
import tarfile
import tempfile
import time


def snapshot(value):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return ("tensor", str(value.dtype), tuple(value.shape), value.detach().cpu().numpy().copy())
    if isinstance(value, np.ndarray):
        return ("array", str(value.dtype), value.shape, value.copy())
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [snapshot(item) for item in value])
    if hasattr(value, "__dict__"):
        return {key: snapshot(item) for key, item in vars(value).items()}
    if value is None or isinstance(value, (str, bool, int, float, complex, np.generic)):
        return (type(value).__name__, value)
    raise TypeError(f"Uncaptured compiled value: {type(value)}")


def worker(root, output):
    import torch
    import witwin.maxwell as mw
    from benchmark.datagen.scenes import scene_builders
    from tests.materials.surface_impedance.test_copper_foil_support_roundoff import _faces
    from witwin.maxwell.compiler.copper_foil import (
        compile_copper_faces, compile_copper_response, copper_edge_admittance_scale,
        copper_edge_feature_cells, copper_free_edge_sides,
    )
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.fdtd.runtime import stepping

    assert Path(mw.__file__).resolve().is_relative_to(root), mw.__file__
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(1)
    records = {}
    for name in ("f3_coarse_11f", "support_roundoff_60cubed"):
        times, compiled = {}, {}
        profile = cProfile.Profile()

        def stage(label, operation):
            started = time.perf_counter()
            profile.enable()
            value = operation()
            profile.disable()
            times[label] = time.perf_counter() - started
            print(f"  {name}/{label}: {times[label]:.6f}s", flush=True)
            return value

        if name.startswith("f3_"):
            sample = stage("build", lambda: next(b for b in scene_builders() if b.name == name).build(device="cpu"))
            scene, options = sample.scene, sample.fdtd_kwargs
        else:
            prepared, _ = stage("build", lambda: _faces(2))
            scene, options = prepared._public_scene, dict(frequency=1e9)
        prepared = stage("prepare_scene", lambda: prepare_scene(scene))
        faces = stage("faces", lambda: compile_copper_faces(prepared, device="cpu"))
        compiled["faces"] = snapshot(faces)
        materials = list({id(face.material): face.material for face in faces}.values())
        for label in ("modal_first", "modal_repeat"):
            compiled[label] = snapshot(stage(label, lambda: tuple(compile_copper_response(m) for m in materials)))

        def edges():
            result = []
            for face in faces:
                sides = copper_free_edge_sides(prepared, face, face.indices, faces)
                scale, geometry = copper_edge_admittance_scale(prepared, face, face.indices, faces)
                features = copper_edge_feature_cells(prepared, face, face.indices, faces, sides)
                result.append((sides, scale, geometry, features))
            return result

        compiled["edges"] = snapshot(stage("edge_tables", edges))
        options = dict(options, backend="cpu_reference", cuda_graph=False,
                       run_time=mw.TimeConfig(time_steps=1))
        initialize = stepping.initialize_copper_foils

        def timed_initialize(solver):
            started = time.perf_counter()
            initialize(solver)
            times["runtime_copper_init"] = time.perf_counter() - started

        stepping.initialize_copper_foils = timed_initialize
        runtime = stage("runtime_prepare", lambda: mw.Simulation.fdtd(scene, **options).prepare())
        stepping.initialize_copper_foils = initialize
        compiled["runtime_copper"] = snapshot(runtime.solver._copper_foils)
        compiled["runtime_faces"] = snapshot(runtime.solver._copper_edge_faces)
        assert not torch.cuda.is_initialized()
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(16)
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats("copper_foil.py", 12)
        print(f"{name}: {times}\n{stream.getvalue()}", flush=True)
        records[name] = dict(times=times, compiled=compiled)
    with open(output, "wb") as handle:
        pickle.dump(records, handle)


def compare(expected, actual, path=""):
    import numpy as np

    assert type(expected) is type(actual), path
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        return sum(compare(expected[key], actual[key], f"{path}/{key}") for key in expected)
    if isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual), path
        return sum(compare(a, b, f"{path}/{i}") for i, (a, b) in enumerate(zip(expected, actual)))
    if isinstance(expected, np.ndarray):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape, path
        assert np.array_equal(expected, actual), path
        assert expected.tobytes() == actual.tobytes(), path
    else:
        assert expected == actual, (path, expected, actual)
        if isinstance(expected, (float, complex)):
            assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes(), path
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return
    root = Path(__file__).resolve().parents[3]
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="copper-setup-") as temporary:
        temporary = Path(temporary)
        baseline = temporary / "master"
        baseline.mkdir()
        archive = temporary / "master.tar"
        with archive.open("wb") as handle:
            subprocess.run(["git", "archive", "master"], cwd=root, stdout=handle, check=True)
        with tarfile.open(archive) as handle:
            handle.extractall(baseline, filter="data")
        results = []
        for label, source in (("master", baseline), ("working", root)):
            output = temporary / f"{label}.pickle"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(source),
                       MPLCONFIGDIR=str(temporary / "matplotlib"), OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            print(f"=== {label} ===", flush=True)
            subprocess.run([sys.executable, str(script), "--worker", str(source), "--output", str(output)],
                           cwd=source, env=env, check=True)
            with output.open("rb") as handle:
                results.append(pickle.load(handle))
        for name in results[0]:
            before, after = (result[name] for result in results)
            count = compare(before["compiled"], after["compiled"], name)
            print(f"PASS {name}: {count} exact leaves (dtype, shape, values and bytes)")
            for stage in before["times"]:
                print(f"  {stage}: master={before['times'][stage]:.6f}s working={after['times'][stage]:.6f}s")


if __name__ == "__main__":
    main()

"""Compare sequential execution with one and two persistent GPU worker slots."""

from __future__ import annotations

import sysconfig
import os
import argparse
import gc
from pathlib import Path
import sys
import tempfile
import time


# Support the coordinator's direct script invocation from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCENES = (
    "f2_box_32", "f2_plate_40", "f2_holes_48", "f5_loaded_36", "f5_loaded_48",
    "f5_bloch_48_2.1ghz", "f5_bloch_48_2.4ghz", "f5_bloch_48_2.7ghz",
    "f4_dipole_32", "f4_loop_40", "f2_box_64", "f4_dipole_64",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Default: a new temporary directory")
    args = parser.parse_args()

    # Set these before importing benchmark, whose package initializer imports witwin.
    import os
    os.environ["CUDA_HOME"] = (
        os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
    )
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = (
        os.path.expanduser("~/.cache/witwin-bc-fdtd-build")
    )
    import torch
    from benchmark.datagen.scenes import scene_builders
    from benchmark.datagen.scheduler import Task, execute_task, git_commit, run_tasks

    output = args.output_dir.resolve() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="datagen-scheduler-acceptance-")
    )
    sequential = output / "sequential"
    sequential.mkdir(parents=True, exist_ok=True)
    tasks = [Task(name, name, index) for index, name in enumerate(SCENES)]
    builders = {builder.name: builder for builder in scene_builders()}
    commit = git_commit()
    started = time.perf_counter()
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    for task in tasks:
        execute_task(task, sequential, commit, builders, device_name)
        gc.collect()
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - started
    print(f"(a) sequential: {baseline_seconds:.3f}s, "
          f"{len(tasks) * 3600 / baseline_seconds:.1f} samples/hour", flush=True)
    # Release baseline allocations before either worker-pool measurement.
    gc.collect()
    torch.cuda.empty_cache()

    manifests = {}
    for slots, label in ((1, "b"), (2, "c")):
        manifest = run_tasks(tasks, output / f"slots_{slots}", ["cuda:0"], slots)
        manifests[slots] = manifest
        seconds = manifest["wall_seconds"]
        print(f"({label}) {slots} slot(s): {seconds:.3f}s, "
              f"{manifest['samples_per_hour']:.1f} samples/hour, "
              f"{baseline_seconds / seconds:.3f}x sequential", flush=True)

    for slots, manifest in manifests.items():
        assert len(manifest["tasks"]) == len(tasks)
        for task in tasks:
            record = manifest["tasks"][task.task_id]
            assert record["status"] == "ok", record
            expected = torch.load(sequential / f"{task.task_id}.pt", map_location="cpu")["readouts"]
            actual = torch.load(record["path"], map_location="cpu")["readouts"]
            assert actual.keys() == expected.keys(), (slots, task.task_id)
            for name in expected:
                assert torch.equal(torch.as_tensor(actual[name]), torch.as_tensor(expected[name])), (
                    slots, task.task_id, name,
                )
        print(f"PASS: {slots} slot(s), every saved readout equals (a) exactly", flush=True)

    one_ratio = baseline_seconds / manifests[1]["wall_seconds"]
    two_ratio = baseline_seconds / manifests[2]["wall_seconds"]
    print(f"(b) criterion: close to (a) allowing process overhead; observed {one_ratio:.3f}x")
    print(f"(c) criterion: >= 1.3x over (a) on an otherwise idle GPU; "
          f"observed {two_ratio:.3f}x ({'PASS' if two_ratio >= 1.3 else 'BELOW TARGET'})")
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()

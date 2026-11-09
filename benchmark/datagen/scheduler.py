"""Persistent, spawned FDTD workers sharing one task queue.

CLI task JSON is a list of {"task_id": "box-0", "builder": "f2_box_32",
"seed": 0} records. Each output contains ``readouts`` and ``provenance``;
manifest.json contains task statuses and aggregate wall-time measurements.
Device indices refer to the parent's CUDA_VISIBLE_DEVICES ordering.
"""

from __future__ import annotations

import sysconfig
import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback


@dataclass(frozen=True)
class Task:
    task_id: str
    builder: str
    seed: int = 0


def configure_environment():
    os.environ["CUDA_HOME"] = (
        os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
    )
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = (
        os.path.expanduser("~/.cache/witwin-bc-fdtd-build")
    )


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
    ).strip()


def execute_task(task, output_dir, commit, builders, device_name):
    """Run and save one task on local cuda:0; imports remain cached across tasks."""
    import numpy as np
    import torch
    import witwin.maxwell as mw

    random.seed(task.seed)
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)
    stages = {}

    def measure(name, operation):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        stages[name] = time.perf_counter() - started
        return value

    sample = measure("build", lambda: builders[task.builder].build(device="cuda:0"))
    prepared = measure(
        "prepare", lambda: mw.Simulation.fdtd(sample.scene, **sample.fdtd_kwargs).prepare(),
    )
    result = measure("run", prepared.run)
    readouts = measure(
        "readouts", lambda: {name: readout(result) for name, readout in sample.readouts},
    )
    provenance = dict(
        task_id=task.task_id, builder=task.builder, seed=task.seed,
        git_commit=commit, stage_seconds=stages,
        steps_run=result.solver_stats["steps_run"], device_name=device_name,
    )
    path = Path(output_dir) / f"{task.task_id}.pt"
    started = time.perf_counter()
    torch.save(dict(readouts=readouts, provenance=provenance), path)
    # Save duration is known after serialization, so it belongs in the manifest.
    stages["save"] = time.perf_counter() - started
    sample.readouts.clear()
    return dict(status="ok", path=str(path), stage_seconds=stages)


def _worker(visible_device, device, slot, tasks, reports, output_dir, commit):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    configure_environment()
    import torch
    from benchmark.datagen.scenes import scene_builders

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    builders = {builder.name: builder for builder in scene_builders()}
    while (task := tasks.get()) is not None:
        started = time.perf_counter()
        try:
            record = execute_task(task, output_dir, commit, builders, device_name)
        except Exception:
            path = Path(output_dir) / f"{task.task_id}.error.txt"
            path.write_text(traceback.format_exc())
            record = dict(status="error", path=str(path))
        gc.collect()
        ended = time.perf_counter()
        reports.put((task.task_id, dict(
            record, seconds=ended - started, started=started, ended=ended,
            device=device, device_name=device_name, slot=slot,
        )))


def run_tasks(tasks: list[Task], output_dir, devices=("cuda:0",), slots_per_device=2):
    """Run a finite task list, returning the manifest written to output_dir.

    Utilisation measures the union of task wall-time intervals per device, not
    hardware SM activity. Slot utilisation divides busy slot seconds by capacity.
    Total wall time includes worker startup and shutdown.
    """
    configure_environment()
    import torch.multiprocessing as mp

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    context = mp.get_context("spawn")
    queue = context.Queue()
    reports = context.Queue()
    for task in tasks:
        queue.put(task)
    for _ in range(len(devices) * slots_per_device):
        queue.put(None)

    original_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = original_visibility.split(",") if original_visibility is not None else None
    workers = []
    started = time.perf_counter()
    for device in devices:
        index = int(device.split(":")[1])
        physical_device = visible[index] if visible is not None else str(index)
        for slot in range(slots_per_device):
            # benchmark.__init__ imports torch before the child target is unpickled.
            # Pin the inherited environment before starting the interpreter as well.
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
            worker = context.Process(
                target=_worker,
                args=(physical_device, device, slot, queue, reports, str(output_dir), commit),
            )
            worker.start()
            workers.append(worker)
    if original_visibility is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_visibility

    records = {}
    for _ in tasks:
        task_id, record = reports.get()
        records[task_id] = record
        print(f"{task_id}: {record['status']} ({record['seconds']:.3f}s)", flush=True)
    for worker in workers:
        worker.join()
    seconds = time.perf_counter() - started
    queue.close()
    reports.close()
    queue.join_thread()
    reports.join_thread()

    utilisation = {}
    for device in devices:
        intervals = sorted((r["started"], r["ended"]) for r in records.values()
                           if r["device"] == device)
        busy = 0.0
        previous_end = started
        for first, last in intervals:
            busy += max(0.0, last - max(first, previous_end))
            previous_end = max(previous_end, last)
        slot_seconds = sum(r["seconds"] for r in records.values() if r["device"] == device)
        utilisation[device] = dict(
            wall_fraction=busy / seconds,
            slot_fraction=slot_seconds / (seconds * slots_per_device),
        )
        print(f"{device}: wall utilisation {busy / seconds:.1%}, "
              f"slot utilisation {utilisation[device]['slot_fraction']:.1%}", flush=True)
    completed = sum(r["status"] == "ok" for r in records.values())
    throughput = completed * 3600 / seconds
    manifest = dict(tasks=records, wall_seconds=seconds, samples_per_hour=throughput,
                    slots_per_device=slots_per_device, device_utilisation=utilisation)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{completed}/{len(tasks)} samples in {seconds:.3f}s: "
          f"{throughput:.1f} samples/hour", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="JSON list of Task records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--slots-per-device", type=int, default=2)
    args = parser.parse_args()
    tasks = [Task(**record) for record in json.loads(args.tasks.read_text())]
    manifest = run_tasks(tasks, args.output_dir, args.devices, args.slots_per_device)
    return int(any(record["status"] == "error" for record in manifest["tasks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())

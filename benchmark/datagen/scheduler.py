"""Persistent, spawned FDTD workers sharing one task queue.

CLI task JSON is a list of {"task_id": "box-0", "builder": "f2_box_32",
"seed": 0} records. Each output contains ``readouts`` and ``provenance``;
manifest.json contains task statuses and aggregate wall-time measurements.
Device indices refer to the parent's CUDA_VISIBLE_DEVICES ordering.

``--format pt`` (default) writes one ``<task_id>.pt`` per task. ``--format emfc``
(opt-in, needs emfc >= 0.2.0) instead appends one EMFC record per task, with the
volume E of every DFT frequency cropped to the domain, to ``<output-dir>/shards``
(sample id = task index, one shard writer per worker; see
``benchmark/datagen/emfc_records.py``); rerunning on the same output directory
runs only the tasks whose records are missing.
"""

from __future__ import annotations

import sysconfig
import argparse
from dataclasses import dataclass, replace
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
    """Fill in CUDA_HOME and the FDTD JIT build dir only where the caller left them unset.

    A caller's values win, so concurrent sessions can keep separate JIT build dirs.
    """
    os.environ.setdefault(
        "CUDA_HOME", os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"),
    )
    os.environ.setdefault(
        "WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", os.path.expanduser("~/.cache/witwin-bc-fdtd-build"),
    )


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
    ).strip()


def _simulate(task, builders, *, full_field_dft=False, admit=None):
    """Build, prepare, run and read out one task on local cuda:0; returns the stage timings too.

    ``full_field_dft=True`` records the volume E DFT the builders leave off by default (and
    ``sample.fdtd_kwargs`` then holds the config that ran); ``admit(sample)`` may refuse the
    built sample before anything runs.
    """
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

    sample = measure("build", lambda: builders[task.builder].build())
    if admit is not None:
        admit(sample)
    if full_field_dft:
        config = replace(sample.fdtd_kwargs["config"], full_field_dft=True)
        sample.fdtd_kwargs = dict(sample.fdtd_kwargs, config=config)
    prepared = measure(
        "prepare", lambda: mw.Simulation.fdtd(sample.scene, **sample.fdtd_kwargs).prepare(),
    )
    result = measure("run", prepared.run)
    readouts = measure(
        "readouts", lambda: {name: readout(result) for name, readout in sample.readouts},
    )
    return sample, result, readouts, stages


def execute_task(task, output_dir, commit, builders, device_name):
    """Run and save one task on local cuda:0; imports remain cached across tasks."""
    import torch

    sample, result, readouts, stages = _simulate(task, builders)
    provenance = dict(
        task_id=task.task_id, builder=task.builder, seed=task.seed,
        git_commit=commit, stage_seconds=stages,
        steps_run=result.stats()["steps_run"], device_name=device_name,
    )
    path = Path(output_dir) / f"{task.task_id}.pt"
    started = time.perf_counter()
    torch.save(dict(readouts=readouts, provenance=provenance), path)
    # Save duration is known after serialization, so it belongs in the manifest.
    stages["save"] = time.perf_counter() - started
    sample.readouts.clear()
    return dict(status="ok", path=str(path), stage_seconds=stages)


def execute_emfc_task(task, sample_id, writer, commit, builders, device_name, options, software):
    """Run one task with the volume E DFT and append its record; a failure appends nothing."""
    import torch
    from benchmark.datagen import emfc_records

    sample, result, readouts, stages = _simulate(
        task, builders, full_field_dft=True, admit=emfc_records.admit_sample)
    provenance = dict(
        task_id=task.task_id, builder=task.builder, family=sample.family, seed=task.seed,
        git_commit=commit, **software, stage_seconds=dict(stages),
        steps_run=result.stats()["steps_run"], device_name=device_name, sample_id=sample_id,
    )
    config = emfc_records.fdtd_config(builders[task.builder], sample.fdtd_kwargs["config"])
    sample.readouts.clear()
    # The record scene's material field, read while the compiled materials are still cached.
    eps_r = result.material("eps_r")
    # Only the phasors and run records are needed from here on.
    result.release_solver()
    torch.cuda.synchronize()
    started = time.perf_counter()
    container, audit = emfc_records.encode_sample(result, readouts, provenance, sample_id=sample_id,
                                                  options=options, fdtd_config=config,
                                                  scene=sample.scene, eps_r=eps_r)
    del result, readouts, eps_r
    torch.cuda.synchronize()
    stages["encode"] = time.perf_counter() - started
    started = time.perf_counter()
    writer.write_bytes(container, sample_id=sample_id,
                       labels=emfc_records.labels(task, sample.family, audit))
    # Encode and append durations are known after the record is sealed in metadata.
    stages["append"] = time.perf_counter() - started
    return dict(status="ok", path=str(writer.directory), sample_id=sample_id,
                audit_lossless=audit, record_bytes=len(container), stage_seconds=stages)


def _worker(visible_device, device, slot, tasks, reports, output_dir, commit, rank=0, emfc_options=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    configure_environment()
    import torch
    from benchmark.datagen.scenes import scene_builders

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    builders = {builder.name: builder for builder in scene_builders()}
    writer = writer_error = None
    if emfc_options is not None:
        from benchmark.datagen import emfc_records
        try:
            software = emfc_records.software_provenance()
            writer = emfc_records.open_writer(output_dir, rank, emfc_options)
        except Exception:
            # Report it per task, so the parent still receives one report per task.
            writer_error = traceback.format_exc()
    while (item := tasks.get()) is not None:
        task, sample_id = (item, None) if emfc_options is None else item
        started = time.perf_counter()
        try:
            if emfc_options is None:
                record = execute_task(task, output_dir, commit, builders, device_name)
            elif writer is None:
                raise RuntimeError(f"shard writer rank {rank} failed to open:\n{writer_error}")
            else:
                record = execute_emfc_task(task, sample_id, writer, commit, builders, device_name,
                                           emfc_options, software)
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
    if writer is not None:
        writer.close()


def run_tasks(tasks: list[Task], output_dir, devices=("cuda:0",), slots_per_device=2,
              emfc_options=None):
    """Run a finite task list, returning the manifest written to output_dir.

    Utilisation measures the union of task wall-time intervals per device, not
    hardware SM activity. Slot utilisation divides busy slot seconds by capacity.
    Total wall time includes worker startup and shutdown.

    With ``emfc_options`` (``emfc_records.EmfcOptions``) task ``i`` becomes the
    record with sample id ``i`` in ``output_dir/shards``; tasks whose records are
    already sealed there are skipped, and the manifest gains an ``emfc`` block.

    Workers are started with the ``spawn`` method, which re-imports the caller's main
    module: a script that calls this must do so under ``if __name__ == "__main__":``,
    or each worker re-runs the script and the parent waits for reports forever.
    """
    configure_environment()
    import torch.multiprocessing as mp

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    queued = tasks
    if emfc_options is not None:
        from benchmark.datagen import emfc_records
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("--format emfc needs unique task ids")
        emfc_options, pending, emfc_info = emfc_records.prepare_dataset(output_dir, tasks, emfc_options)
        queued = [tasks[sample_id] for sample_id in pending]
        sample_ids = {tasks[sample_id].task_id: sample_id for sample_id in pending}
    context = mp.get_context("spawn")
    queue = context.Queue()
    reports = context.Queue()
    for task in queued:
        queue.put(task if emfc_options is None else (task, sample_ids[task.task_id]))
    for _ in range(len(devices) * slots_per_device):
        queue.put(None)

    original_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = original_visibility.split(",") if original_visibility is not None else None
    workers = []
    started = time.perf_counter()
    for device_rank, device in enumerate(devices):
        index = int(device.split(":")[1])
        physical_device = visible[index] if visible is not None else str(index)
        for slot in range(slots_per_device):
            # benchmark.__init__ imports torch before the child target is unpickled.
            # Pin the inherited environment before starting the interpreter as well.
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
            worker = context.Process(
                target=_worker,
                args=(physical_device, device, slot, queue, reports, str(output_dir), commit,
                      device_rank * slots_per_device + slot, emfc_options),
            )
            worker.start()
            workers.append(worker)
    if original_visibility is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_visibility

    records = {}
    for _ in queued:
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
    if emfc_options is not None:
        emfc_block = emfc_records.finish_dataset(output_dir, emfc_options, records, sample_ids, emfc_info)
    completed = sum(r["status"] == "ok" for r in records.values())
    throughput = completed * 3600 / seconds
    manifest = dict(tasks=records, wall_seconds=seconds, samples_per_hour=throughput,
                    slots_per_device=slots_per_device, device_utilisation=utilisation)
    if emfc_options is not None:
        manifest.update(format="emfc", emfc=emfc_block)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{completed}/{len(queued)} samples in {seconds:.3f}s: "
          f"{throughput:.1f} samples/hour", flush=True)
    if emfc_options is not None:
        print(f"emfc dataset {emfc_block['dataset_uuid']}: {emfc_block['record_count']} records "
              f"({emfc_block['skipped_present']} already present)", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="JSON list of Task records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--slots-per-device", type=int, default=2)
    parser.add_argument("--format", choices=("pt", "emfc"), default="pt",
                        help="pt: one .pt per task; emfc: one EMFC record per task in output-dir/shards")
    emfc_group = parser.add_argument_group("--format emfc")
    emfc_group.add_argument("--dataset-seed", type=int,
                            help="dataset seed of the lossless audit rule (required with --format emfc)")
    emfc_group.add_argument("--field-tolerance", type=float,
                            help="relative L2 bound per E component (default 1e-3)")
    emfc_group.add_argument("--h-field-tolerance", type=float,
                            help="relative L2 bound per stored H component (default: --field-tolerance); "
                                 "H is stored only where H cannot be derived from E (amendment A5)")
    emfc_group.add_argument("--audit-fraction", type=float,
                            help="fraction of samples stored lossless (default 0.01)")
    emfc_group.add_argument("--shard-target-bytes", type=int,
                            help="shard roll threshold in bytes (default 1 GiB): records reach disk only "
                                 "when their shard seals, so a crashed worker loses at most one open "
                                 "shard of tasks per rank; they rerun on resume")
    args = parser.parse_args()
    emfc_flags = ("dataset_seed", "field_tolerance", "h_field_tolerance", "audit_fraction", "shard_target_bytes")
    emfc_options = None
    if args.format == "pt":
        given = [name for name in emfc_flags if getattr(args, name) is not None]
        if given:
            parser.error(f"{', '.join('--' + name.replace('_', '-') for name in given)} need --format emfc")
    else:
        if args.dataset_seed is None:
            parser.error("--format emfc requires --dataset-seed")
        from benchmark.datagen.emfc_records import EmfcOptions
        emfc_options = EmfcOptions(**{name: getattr(args, name) for name in emfc_flags
                                      if getattr(args, name) is not None})
    tasks = [Task(**record) for record in json.loads(args.tasks.read_text())]
    manifest = run_tasks(tasks, args.output_dir, args.devices, args.slots_per_device, emfc_options)
    return int(any(record["status"] == "error" for record in manifest["tasks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())

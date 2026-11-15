"""End to end: the datagen scheduler with ``--format emfc`` on four small catalog builders.

Writes shards from two spawned workers (two shard ranks), indexes them, reads every record back
with ``emfc.shard.loader`` and compares it with an in-process rerun of the same task (FDTD runs
are deterministic per device type): readouts bit-exact, E (and the F3 board's H) bit-exact for
the audit sample and within the certified per-component relative L2 otherwise, metadata as
maxwell reports it, and the record scene (the builder's Scene document and the cropped eps_r).
A second run on the same directory moves crash debris aside and runs only the missing tasks.
Needs CUDA and emfc >= 0.2.0.
"""
import json
import os

import numpy as np
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FDTD required")

TASKS = (("box", "f2_box_32"), ("bloch", "f5_bloch_48_2.4ghz"), ("dipole", "f4_dipole_32"),
         ("board", "f3_coarse_11f"))   # the smallest F3 builder: CopperFoil and a graded grid store H (A5)
# Spawned workers inherit PYTHONPATH; this hook caps their CUDA memory like the test process
# and records their peak allocation. Only multiprocessing children start CUDA here.
WORKER_HOOK = '''import atexit, os, sys
if "--multiprocessing-fork" in sys.argv and os.environ.get("DATAGEN_TEST_WORKER_CAP_GIB"):
    import torch
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ["DATAGEN_TEST_WORKER_CAP_GIB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    def _peak(path=os.environ["DATAGEN_TEST_WORKER_PEAK_DIR"]):
        if torch.cuda.is_initialized():
            with open(os.path.join(path, f"{os.getpid()}.txt"), "w") as handle:
                handle.write(str(torch.cuda.max_memory_allocated(0)))
    atexit.register(_peak)
'''


def _relative_l2(got, want):
    return float(torch.linalg.vector_norm(got - want) / torch.linalg.vector_norm(want))


@pytest.fixture
def worker_hook(tmp_path, monkeypatch):
    hook, peaks = tmp_path / "hook", tmp_path / "peaks"
    hook.mkdir()
    peaks.mkdir()
    (hook / "sitecustomize.py").write_text(WORKER_HOOK)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(hook), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("DATAGEN_TEST_WORKER_CAP_GIB", "8")
    monkeypatch.setenv("DATAGEN_TEST_WORKER_PEAK_DIR", str(peaks))
    return peaks


def test_emfc_format_shards_read_back_and_resume(tmp_path, worker_hook):
    emfc = pytest.importorskip("emfc")
    if not hasattr(emfc, "encode_record"):
        pytest.skip(f"emfc {getattr(emfc, '__version__', '?')} has no record containers (needs >= 0.2.0)")
    import emfc.shard
    import witwin.maxwell as mw
    from benchmark.datagen import emfc_records
    from benchmark.datagen.scenes import scene_builders
    from benchmark.datagen.scheduler import Task, _simulate, run_tasks

    tasks = [Task(task_id, builder, seed) for seed, (task_id, builder) in enumerate(TASKS)]
    # A dataset seed whose audit subset is exactly sample 0 (the rule is a pure function).
    seed = next(s for s in range(100_000)
                if [emfc_records.audit_selected(s, i, 0.01) for i in range(4)] == [True, False, False, False])
    options = emfc_records.EmfcOptions(dataset_seed=seed)
    output = tmp_path / "out"

    first = run_tasks(tasks[:2], output, devices=("cuda:0",), slots_per_device=2, emfc_options=options)
    assert {task_id: r["status"] for task_id, r in first["tasks"].items()} == {"box": "ok", "bloch": "ok"}, \
        {task_id: Path(r["path"]).read_text() for task_id, r in first["tasks"].items() if r["status"] != "ok"}
    assert first["format"] == "emfc" and first["emfc"]["record_count"] == 2
    assert first["emfc"]["audit_records"] == 1 and first["tasks"]["box"]["audit_lossless"] is True
    uuid = first["emfc"]["dataset_uuid"]

    # A crash leaves an unsealed shard: it is moved aside and only the missing task runs.
    debris = output / "shards" / "shard-r000007-000000.emfcs.tmp"
    debris.write_bytes(b"unsealed")
    second = run_tasks(tasks, output, devices=("cuda:0",), slots_per_device=1, emfc_options=options)
    assert {task_id: r["status"] for task_id, r in second["tasks"].items()} == {"dipole": "ok", "board": "ok"}, \
        {task_id: Path(r["path"]).read_text() for task_id, r in second["tasks"].items() if r["status"] != "ok"}
    assert second["emfc"]["skipped_present"] == 2 and len(second["emfc"]["moved_aside"]) == 1
    assert not debris.exists()
    assert second["emfc"]["dataset_uuid"] == uuid and second["emfc"]["record_count"] == 4
    on_disk = json.loads((output / "manifest.json").read_text())
    assert on_disk["emfc"]["record_count"] == 4

    index = emfc.shard.DatasetIndex.open(output / "shards")
    assert index.dataset_uuid == uuid and len(index) == 4
    assert index.where(audit_lossless=1) == [0] and index.where(family="F5") == [1]
    assert index.where(task_id="dipole") == [2] and index.where(family="F3") == [3]
    for entry in index:
        assert bool(entry.record.flags & emfc.shard.LOSSY) == (entry.sample_id != 0)
    record_bytes, scenes = {}, {}
    for position in range(len(index)):
        container = index.read(position)
        sample_id = index.entry(position).sample_id
        info = emfc.inspect_record(container)
        field_bytes = sum(field["bytes"] for field in info.fields.values())
        h_bytes = sum(field["bytes"] for name, field in info.fields.items() if name.startswith("H/"))
        _, scenes[sample_id] = emfc.decode_record(container, return_scene=True)
        eps_bytes = int(torch.as_tensor(scenes[sample_id].eps_r).numel() * torch.as_tensor(
            scenes[sample_id].eps_r).element_size())
        record_bytes[TASKS[sample_id][1]] = dict(total=len(container), fields=field_bytes, h_fields=h_bytes,
                                                 readouts=info.readout_bytes if hasattr(info, "readout_bytes") else None,
                                                 scene_eps_r_raw=eps_bytes)
        formats = {field["format"] for field in info.fields.values()}
        assert formats == {"lossless-staggered-v1" if sample_id == 0 else "lossy-staggered-v1"}
        assert any(name.startswith("H/") for name in info.fields) is (sample_id == 3)
    index.close()
    print("record bytes per sample:", record_bytes)
    peaks = [int(p.read_text()) for p in worker_hook.iterdir()]
    print("worker peak allocated GiB:", [round(p / 2**30, 3) for p in peaks])
    assert peaks and max(peaks) < 8 * 2**30

    decoded = {}
    for batch in emfc.shard.loader(output / "shards", batch_size=1, num_workers=0, shuffle=False,
                                   drop_last=False, record_fields=True):
        for sample_id, record in zip(batch.sample_ids, batch.fields):
            decoded[sample_id] = record
    assert sorted(decoded) == [0, 1, 2, 3]

    builders = {builder.name: builder for builder in scene_builders()}
    for sample_id, task in enumerate(tasks):
        record = decoded[sample_id]
        torch.cuda.reset_peak_memory_stats()
        sample, result, readouts, _ = _simulate(task, builders, full_field_dft=True,
                                                admit=emfc_records.admit_sample)
        print(f"{task.builder}: in-process rerun peak allocated "
              f"{torch.cuda.max_memory_allocated() / 2**30:.3f} GiB (H {'on' if 'HX' in result.fields else 'off'})")
        stats = result.stats()
        sample.readouts.clear()
        eps_r = result.material("eps_r").detach().cpu()
        result.release_solver()
        f3 = task.builder.startswith("f3_")
        meta = record.metadata
        assert {key: meta[key] for key in ("task_id", "builder", "family", "seed", "sample_id", "steps_run")} == dict(
            task_id=task.task_id, builder=task.builder, family=sample.family, seed=task.seed,
            sample_id=sample_id, steps_run=stats["steps_run"])
        maxwell = meta["maxwell_record"]
        assert maxwell["schema"] == "maxwell-datagen-record/1" and maxwell["dt_s"] == float(stats["dt"])
        assert maxwell["quantity"] == ("E+H" if f3 else "E") and maxwell["h_stored"] is f3
        assert meta["steps_run"] == maxwell["steps_run"]
        software = emfc_records.software_provenance()
        assert {key: meta[key] for key in software} == software
        if f3:
            assert maxwell["h_derivation"] is None
            assert maxwell["h_derivation_unavailable"] == ["copper_foil_h_correction"]
            assert maxwell["h_stored_source"] == "maxwell_volume_dft"
            assert maxwell["h_field_time_offset_steps"] == result.fields["HX"].attrs["field_time_offset_steps"]
            assert maxwell["field_fidelity"]["h_tolerance"] == 1e-3
        else:
            assert maxwell["h_derivation"] == {"method": "h_from_e", "mu_h_per_m": 1.25663706212e-6,
                                               "sigma_m": 0, "excluded": "tfsf_h_shell"}
        assert maxwell["fdtd_config"]["kwargs"] == emfc_records._json(builders[task.builder].build.keywords)
        assert maxwell["fdtd_config"]["config_snapshot"]["full_field_dft"] is True
        assert set(maxwell["boundary"]["z"]) <= {"cpml", "pml"}
        fidelity = maxwell["field_fidelity"]
        assert fidelity["audit"] is (sample_id == 0) and fidelity["dataset_seed"] == seed
        assert maxwell["tfsf"] == emfc_records._json(None if stats["tfsf"] is None else {
            key: stats["tfsf"][key] for key in ("mode", "axis", "node_indices", "h_corrections")})

        # Readouts: bit-exact against the rerun.
        grid_arrays = {f"_grid/nodes_m/{axis}" for axis in "xyz"}
        assert set(record.arrays) == set(readouts) | {"_field_frequencies_hz"} | grid_arrays
        for name, value in readouts.items():
            want = torch.as_tensor(value).detach().cpu()
            assert torch.equal(record.arrays[name].cpu(), want), name
        frequencies = result.fields["EX"].coords["f"].detach().cpu().to(torch.float64)
        assert torch.equal(record.arrays["_field_frequencies_hz"].cpu(), frequencies)
        assert maxwell["field_frequencies_hz"] == frequencies.tolist()

        # Crop: absorber layers removed, periodic/Bloch wrap plane kept.
        nodes, starts, stops = emfc_records.domain_window(result)
        grid = dict(maxwell["grid"])
        spacing, uniform, per_axis = grid.pop("reference_spacing_m"), grid.pop("uniform"), grid.pop("spacing_m")
        assert grid == {"full_shape_nodes": [len(n) for n in nodes], "crop_node_start": list(starts),
                        "crop_node_stop": list(stops)}
        if f3:   # the printed board's grid is graded on every axis
            assert uniform is False and per_axis == [None, None, None]
        else:    # cube/slab grids are uniform per axis; the Bloch slab's z width differs from x/y
            assert uniform is True and min(per_axis) == spacing
            assert (len(set(per_axis)) > 1) is (task.builder == "f5_bloch_48_2.4ghz")
        for axis, name in enumerate("xyz"):
            assert torch.equal(record.arrays[f"_grid/nodes_m/{name}"].cpu(), nodes[axis][starts[axis]:stops[axis]])
        for axis, faces in enumerate(result.scene.boundary.face_kinds):
            if str(faces[0]) == "pml":
                assert starts[axis] == 8 and stops[axis] == len(nodes[axis]) - 8
            else:
                assert (starts[axis], stops[axis]) == (0, len(nodes[axis]))
                assert "xyz"[axis] in maxwell["periodic_axes"]

        # Scene: the builder's Scene document and the compiled eps_r on the window, bit-exact
        # (Result.material returns it on the primal nodes here, so the window keeps its nodes).
        scene = scenes[sample_id]
        fresh = emfc.SceneData(eps_r=np.ones((2, 2, 2)), definition=sample.scene).definition
        # witwin.scene/2 numbers structure identities per scene, so a process that built other
        # scenes first writes the same document.
        assert scene.definition.document == fresh.document
        rebuilt = mw.Scene.from_dict(scene.definition.document, arrays=scene.definition.arrays)
        assert rebuilt.to_dict(arrays={}) == scene.definition.document
        stored_eps = torch.as_tensor(scene.eps_r)
        assert torch.equal(stored_eps, torch.as_tensor(emfc_records.scene_payload(sample.scene, eps_r,
                                                                                  maxwell["grid"]).eps_r))
        assert tuple(stored_eps.shape) in (tuple(e - s for s, e in zip(starts, stops)),
                                           tuple(e - s - 1 for s, e in zip(starts, stops)))

        # E (and H): lossless audit sample bit-exact, others within the certified per-component bound.
        assert maxwell["field_names"] == [f"E/f{i:03d}" for i in range(len(frequencies))]
        h_names = maxwell.get("h_field_names", [])
        assert h_names == ([f"H/f{i:03d}" for i in range(len(frequencies))] if f3 else [])
        assert sorted(record.field_names) == sorted(maxwell["field_names"] + h_names)
        for kind, names, components in (("electric", maxwell["field_names"], ("EX", "EY", "EZ")),
                                        ("magnetic", h_names, ("HX", "HY", "HZ"))):
            slices = emfc_records.component_slices(starts, stops, kind)
            for index_f, name in enumerate(names):
                field = record.fields[name]
                assert field.kind == kind and field.quantity == "field"
                for c, component in enumerate(components):
                    want = result.fields[component].values[(index_f, *slices[c])]
                    got = field.components[c]
                    assert got.shape == want.shape and got.dtype == torch.complex64
                    if sample_id == 0:
                        assert torch.equal(got, want), (name, component)
                    else:
                        assert _relative_l2(got, want) <= 1e-3 * (1 + 1e-6), (name, component)
                assert field.reference_spacing_m == spacing
                for axis in range(3):
                    assert torch.equal(field.nodes[axis].cpu(),
                                       (nodes[axis][starts[axis]:stops[axis]] / spacing).to(torch.float64))
        del result, readouts


class _Field:
    """A FieldData stand-in: values, float64 coordinates and attrs, as Result.fields carries them."""

    def __init__(self, values, coords, attrs):
        self.values, self.coords, self.attrs = values, coords, attrs


class _Result:
    def __init__(self, scene, fields, stats):
        self.scene, self.fields, self._stats = scene, fields, stats

    def stats(self):
        return self._stats


def _synthetic_result(with_h):
    """A graded PEC-box scene with random E (and H) phasors on its full Yee grids, on CUDA."""
    import witwin.maxwell as mw

    nodes = [torch.tensor([0.0, 0.1, 0.15, 0.2, 0.3, 0.4], dtype=torch.float64),
             torch.linspace(0.0, 0.4, 5, dtype=torch.float64), torch.linspace(0.0, 0.3, 4, dtype=torch.float64)]
    scene = mw.Scene(domain=mw.Domain(tuple((float(n[0]), float(n[-1])) for n in nodes)),
                     grid=mw.GridSpec.custom(*(n.numpy() for n in nodes)), boundary=mw.BoundarySpec.pec())
    frequencies = torch.tensor([1.0e9, 1.5e9], dtype=torch.float64)
    generator = torch.Generator(device="cuda").manual_seed(7)
    fields = {}
    for kind, names, offset in (("electric", ("EX", "EY", "EZ"), 0.0), ("magnetic", ("HX", "HY", "HZ"), -0.375)):
        if kind == "magnetic" and not with_h:
            continue
        for c, name in enumerate(names):
            own_is_cell = kind == "electric"
            axes = [n[:-1] + 0.5 * n.diff() if (a == c) == own_is_cell else n for a, n in enumerate(nodes)]
            shape = (2, *(len(a) for a in axes))
            values = torch.complex(torch.randn(shape, generator=generator, device="cuda"),
                                   torch.randn(shape, generator=generator, device="cuda"))
            fields[name] = _Field(values, {"f": frequencies, **dict(zip("xyz", axes))},
                                  {"field_time_offset_steps": offset, "phasor_convention": "exp(-i omega t), peak",
                                   "normalization": "raw"})
    stats = {"dt": 1e-12, "steps_run": 100, "dft_window": "none", "dft_window_requested": "none",
             "last_step_field_ratio": (1e-6, 2e-6), "bloch_wavevector": (0.0, 0.0, 0.0),
             "bloch_phase": (1 + 0j, 1 + 0j, 1 + 0j), "tfsf": None, "absorber": "cpml"}
    return _Result(scene, fields, stats)


def test_h_path_stores_magnetic_fields_when_the_result_carries_them(tmp_path):
    emfc = pytest.importorskip("emfc")
    if not hasattr(emfc, "encode_record"):
        pytest.skip(f"emfc {getattr(emfc, '__version__', '?')} has no record containers (needs >= 0.2.0)")
    from benchmark.datagen import emfc_records

    provenance = dict(task_id="synthetic", builder="synthetic", family="F0", seed=0, git_commit="none",
                      stage_seconds={}, steps_run=100, device_name="cuda", sample_id=0,
                      **emfc_records.software_provenance())
    config = {"source": "builder_kwargs", "builder": "synthetic", "kwargs": {}, "config_snapshot": {},
              "authoritative": "builder_kwargs"}
    # A graded axis demands H (A5); without H phasors the sample is refused, never stored E-only.
    with pytest.raises(ValueError, match="amendment A5: graded_grid"):
        emfc_records.encode_sample(_synthetic_result(False), {}, provenance, sample_id=0,
                                   options=emfc_records.EmfcOptions(dataset_seed=0), fdtd_config=config)

    result = _synthetic_result(True)
    nodes, starts, stops = emfc_records.domain_window(result)
    for fraction, label in ((1.0, "lossless"), (0.0, "tolerance")):
        options = emfc_records.EmfcOptions(dataset_seed=0, audit_fraction=fraction)
        # A node-sampled eps_r on the full grid, as Result.material returns it for the catalog builders.
        eps_r = torch.full(tuple(len(n) for n in nodes), 2.0)
        container, audit = emfc_records.encode_sample(result, {"r": torch.ones(2)}, provenance, sample_id=0,
                                                      options=options, fdtd_config=config,
                                                      scene=result.scene, eps_r=eps_r)
        (tmp_path / f"synthetic-{label}.emfc").write_bytes(container)
        record = emfc.decode_record(container, device="cuda", fields=True)
        maxwell = record.metadata["maxwell_record"]
        assert audit is (fraction == 1.0)
        assert maxwell["quantity"] == "E+H" and maxwell["h_stored"] is True
        assert maxwell["h_stored_reason"] == "graded_grid" and maxwell["h_derivation"] is not None
        assert maxwell["h_field_time_offset_steps"] == -0.375 and maxwell["field_time_offset_steps"] == 0.0
        assert maxwell["field_names"] == ["E/f000", "E/f001"] and maxwell["h_field_names"] == ["H/f000", "H/f001"]
        assert maxwell["h_stored_source"] == "maxwell_volume_dft"
        assert maxwell["field_fidelity"]["h_tolerance"] == (None if audit else 1e-3)
        assert maxwell["grid"]["uniform"] is False and maxwell["grid"]["spacing_m"][0] is None
        _, scene = emfc.decode_record(container, return_scene=True)
        assert torch.equal(torch.as_tensor(scene.eps_r), eps_r) and scene.definition.document["schema"] == "witwin.scene/2"
        for kind, names in (("electric", ("EX", "EY", "EZ")), ("magnetic", ("HX", "HY", "HZ"))):
            slices = emfc_records.component_slices(starts, stops, kind)
            for index in range(2):
                field = record.fields[f"{'E' if kind == 'electric' else 'H'}/f{index:03d}"]
                assert field.kind == kind
                for c, name in enumerate(names):
                    want = result.fields[name].values[(index, *slices[c])]
                    got = field.components[c]
                    assert got.shape == want.shape
                    if audit:
                        assert torch.equal(got, want), (kind, index, name)
                    else:
                        assert _relative_l2(got, want) <= 1e-3 * (1 + 1e-6), (kind, index, name)

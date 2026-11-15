"""benchmark/datagen ``--format emfc``: pure rules, metadata, the .pt path's independence of emfc, resume.

The rule, crop and metadata tests need neither CUDA nor emfc. Tests that touch emfc skip
unless emfc >= 0.2.0 (record containers and shards) is importable; the GPU end-to-end
test is tests/validation/benchmark/test_datagen_emfc_format_gpu.py.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmark.datagen import emfc_records
from benchmark.datagen.scheduler import Task

ROOT = Path(__file__).resolve().parents[3]
MAXWELL_RECORD_KEYS = {
    "schema", "quantity", "h_stored", "field_names", "field_frequencies_hz", "dt_s", "steps_run",
    "field_time_offset_steps", "phasor_convention", "normalization", "dft_window",
    "dft_window_requested", "last_step_field_ratio", "grid", "boundary", "periodic_axes",
    "bloch_wavevector", "bloch_phase", "tfsf", "h_derivation", "fdtd_config", "field_fidelity",
}
PROVENANCE_KEYS = {"task_id", "builder", "family", "seed", "git_commit", "stage_seconds", "steps_run",
                   "device_name", "sample_id", "maxwell_version", "witwin_core_version", "witwin_core_tree_sha256",
                   "witwin_core_commit"}


def _emfc_records_api():
    emfc = pytest.importorskip("emfc")
    if not hasattr(emfc, "encode_record"):
        pytest.skip(f"emfc {getattr(emfc, '__version__', '?')} has no record containers (needs >= 0.2.0)")
    return emfc


# --- audit rule -------------------------------------------------------------------------------
def test_audit_rule_is_the_published_formula():
    for seed, sample_id in ((0, 0), (7, 110), (2**40, 999_999)):
        digest = hashlib.sha256(f"emfc-audit-v1:{seed}:{sample_id}".encode()).digest()
        expected = int.from_bytes(digest[:8], "little") < round(0.01 * 2**64)
        assert emfc_records.audit_selected(seed, sample_id, 0.01) is expected
    # Pinned values: anyone recomputing the rule must get these.
    assert [i for i in range(300) if emfc_records.audit_selected(0, i, 0.01)] == [110]


def test_audit_rule_is_deterministic_and_has_the_requested_rate():
    ids = range(20_000)
    first = [emfc_records.audit_selected(3, i, 0.01) for i in ids]
    assert first == [emfc_records.audit_selected(3, i, 0.01) for i in ids]
    assert 0.008 < sum(first) / len(first) < 0.012
    assert first != [emfc_records.audit_selected(4, i, 0.01) for i in ids]
    assert not any(emfc_records.audit_selected(3, i, 0.0) for i in range(1000))
    assert all(emfc_records.audit_selected(3, i, 1.0) for i in range(1000))


# --- crop window --------------------------------------------------------------------------------
def _pml_axis(n, span, pml=8):
    h = span / n
    return np.concatenate([-span / 2 - h * np.arange(pml, 0, -1), np.linspace(-span / 2, span / 2, n + 1),
                           span / 2 + h * np.arange(1, pml + 1)])


def test_crop_window_drops_absorber_layers():
    nodes = _pml_axis(32, 1.0)
    start, stop = emfc_records.crop_window(nodes, -0.5, 0.5)
    assert (start, stop) == (8, len(nodes) - 8)
    assert stop - start == 33 and nodes[start] == -0.5 and nodes[stop - 1] == 0.5


def test_crop_window_keeps_the_periodic_wrap_plane():
    nodes = np.linspace(-0.4, 0.4, 49)   # periodic/Bloch axis: endpoint-inclusive, no absorber
    assert emfc_records.crop_window(nodes, -0.4, 0.4) == (0, 49)


def test_crop_window_graded_axis_and_refusals():
    graded = np.array([-3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    assert emfc_records.crop_window(graded, -1.0, 1.0) == (2, 9)
    with pytest.raises(ValueError, match="not primal nodes"):
        emfc_records.crop_window(graded, -0.9, 1.0)
    with pytest.raises(ValueError, match="fewer than two"):
        emfc_records.crop_window(graded, 0.1, 0.2)


def test_component_slices_follow_the_yee_staggering():
    full = (49, 47, 66)
    starts, stops = (8, 0, 8), (41, 47, 58)
    # Full-grid E shapes: component c has N - 1 samples along its own axis.
    shapes = [tuple(n - (axis == c) for axis, n in enumerate(full)) for c in range(3)]
    for c, slices in enumerate(emfc_records.component_slices(starts, stops)):
        cropped = np.empty(shapes[c])[slices].shape
        cells = [stop - start - 1 for start, stop in zip(starts, stops)]
        assert cropped == tuple(cells[a] if a == c else cells[a] + 1 for a in range(3))


# --- metadata -----------------------------------------------------------------------------------
def _metadata(audit, *, tfsf=True, bloch=True, h_reasons=(), ratio=(3.7e-6, 1.2e-4), h_attrs=None,
              h_field_tolerance=None):
    stats = {
        "dt": 1.25e-12, "steps_run": 3100, "dft_window": "none", "dft_window_requested": "hann",
        "last_step_field_ratio": ratio,
        "bloch_wavevector": (12.5, 0.0, 0.0) if bloch else None,
        "bloch_phase": (complex(0.6, 0.8), 1 + 0j, 1 + 0j) if bloch else None,
        "tfsf": ({"mode": "slab", "axis": "z", "node_indices": ((0, 48), (0, 48), (20, 60)),
                  "h_corrections": ({"component": "Hx", "start": (0, 0, 19), "shape": (49, 48, 1)},),
                  "extra": "dropped"} if tfsf else None),
    }
    attrs = {"field_time_offset_steps": 0.0, "phasor_convention": "exp(-i omega t), peak",
             "normalization": "raw", "yee_location": "edge_x"}
    provenance = dict(task_id="t", builder="f5_bloch_48_2.4ghz", family="F5", seed=0, git_commit="abc",
                      stage_seconds={"build": 0.1, "run": 2.0}, steps_run=3100, device_name="gpu", sample_id=4,
                      maxwell_version="0.6.0", witwin_core_version="0.4.6", witwin_core_tree_sha256="0" * 64,
                      witwin_core_commit="abc")
    boundary = emfc_records.boundary_kinds((("bloch", "bloch"), ("bloch", "bloch"), ("pml", "pml")), "cpml")
    grid, _ = emfc_records.grid_record((49, 49, 97), (0, 0, 8), (49, 49, 89),
                                       [np.linspace(-0.4, 0.4, 49)] * 2 + [np.linspace(-1.0, 1.0, 81)])
    options = emfc_records.EmfcOptions(dataset_seed=11, h_field_tolerance=h_field_tolerance)
    config = {"source": "builder_kwargs", "builder": "f5_bloch_48_2.4ghz", "kwargs": {"n": 48},
              "config_snapshot": {}, "authoritative": "builder_kwargs"}
    return emfc_records.record_metadata(
        provenance=provenance, stats=stats, field_attrs=attrs, frequencies=[2.4e9, 2.5e9],
        boundary=boundary, grid=grid, h_reasons=list(h_reasons), mu_0=1.25663706212e-6, fdtd_config=config,
        audit=audit, options=options, h_field_attrs=h_attrs)


def test_record_metadata_has_exactly_the_v1_keys_and_is_strict_json():
    meta = _metadata(False)
    assert set(meta) == PROVENANCE_KEYS | {"maxwell_record"}
    record = meta["maxwell_record"]
    assert set(record) == MAXWELL_RECORD_KEYS
    json.dumps(meta, allow_nan=False)
    assert record["schema"] == "maxwell-datagen-record/1" and record["quantity"] == "E"
    assert record["field_names"] == ["E/f000", "E/f001"]
    assert record["field_frequencies_hz"] == [2.4e9, 2.5e9]
    assert record["last_step_field_ratio"] == [3.7e-6, 1.2e-4]
    grid = dict(record["grid"])
    assert grid.pop("reference_spacing_m") == pytest.approx(0.8 / 48, rel=1e-12) and grid.pop("uniform") is True
    assert grid.pop("spacing_m") == pytest.approx([0.8 / 48, 0.8 / 48, 2.0 / 80], rel=1e-12)
    assert grid == {"full_shape_nodes": [49, 49, 97], "crop_node_start": [0, 0, 8], "crop_node_stop": [49, 49, 89]}
    assert record["boundary"] == {"x": ["bloch", "bloch"], "y": ["bloch", "bloch"], "z": ["cpml", "cpml"]}
    assert record["periodic_axes"] == ["x", "y"]
    assert record["bloch_wavevector"] == [12.5, 0.0, 0.0]
    assert record["bloch_phase"] == [[0.6, 0.8], [1.0, 0.0], [1.0, 0.0]]
    assert record["tfsf"] == {"mode": "slab", "axis": "z", "node_indices": [[0, 48], [0, 48], [20, 60]],
                              "h_corrections": [{"component": "Hx", "start": [0, 0, 19], "shape": [49, 48, 1]}]}
    assert record["h_derivation"] == {"method": "h_from_e", "mu_h_per_m": 1.25663706212e-6, "sigma_m": 0,
                                      "excluded": "tfsf_h_shell"}
    assert "h_derivation_unavailable" not in record
    assert record["fdtd_config"]["authoritative"] == "builder_kwargs"
    assert "h_field_names" not in record and "h_stored_source" not in record
    assert record["field_fidelity"] == {"mode": "tolerance", "tolerance": 1e-3, "h_tolerance": None, "audit": False,
                                        "audit_rule": "emfc-audit-v1", "audit_fraction": 0.01, "dataset_seed": 11}


def test_record_metadata_audit_and_absent_tfsf_bloch():
    record = _metadata(True, tfsf=False, bloch=False)["maxwell_record"]
    assert record["field_fidelity"]["mode"] == "lossless" and record["field_fidelity"]["tolerance"] is None
    assert record["field_fidelity"]["audit"] is True
    assert record["tfsf"] is None and record["bloch_wavevector"] is None and record["bloch_phase"] is None
    unavailable = _metadata(False, h_reasons=["structure s: mu_r != 1"])["maxwell_record"]
    assert unavailable["h_derivation"] is None
    assert unavailable["h_derivation_unavailable"] == ["structure s: mu_r != 1"]


def test_boundary_vocabulary_is_closed():
    kinds = emfc_records.boundary_kinds((("pml", "pml"), ("periodic", "periodic"), ("pec", "mur")), "absorber")
    assert kinds == {"x": ["pml", "pml"], "y": ["periodic", "periodic"], "z": ["pec", "mur"]}
    assert emfc_records.boundary_kinds((("pml", "pmc"), ("mode_abc", "bloch"), ("pml", "pml")),
                                       "cpml")["x"] == ["cpml", "pmc"]
    for unknown in ("none", "mixed", "symmetry"):
        with pytest.raises(ValueError, match="outside the record vocabulary"):
            emfc_records.boundary_kinds(((unknown, "pml"), ("pml", "pml"), ("pml", "pml")), "cpml")
    # Every maxwell face kind is either mapped or refused, never passed through.
    import witwin.maxwell as mw
    for kind in mw.BoundaryKind:
        faces = ((kind, kind), ("pml", "pml"), ("pml", "pml"))
        if str(kind) in ("none", "mixed"):
            with pytest.raises(ValueError):
                emfc_records.boundary_kinds(faces, "cpml")
        else:
            assert emfc_records.boundary_kinds(faces, "cpml")["x"][0] in emfc_records.BOUNDARY_KINDS


def test_grid_record_uniform_and_graded():
    uniform, h = emfc_records.grid_record((10, 10, 10), (0, 0, 0), (10, 10, 10), [np.linspace(0, 0.9, 10)] * 3)
    assert uniform["uniform"] is True and h == pytest.approx(0.1)
    assert uniform["spacing_m"] == pytest.approx([0.1] * 3)
    # Per-axis uniform with different widths is uniform; a graded axis is not.
    mixed, h = emfc_records.grid_record((10, 5, 5), (0, 0, 0), (10, 5, 5),
                                        [np.linspace(0, 0.9, 10), np.linspace(0, 0.8, 5), np.linspace(0, 1, 5)])
    assert mixed["uniform"] is True and h == pytest.approx(0.1) and mixed["spacing_m"] == pytest.approx([0.1, 0.2, 0.25])
    graded, h = emfc_records.grid_record((5, 5, 5), (0, 0, 0), (5, 5, 5),
                                         [np.array([0.0, 0.1, 0.15, 0.2, 0.3]), np.linspace(0, 1, 5),
                                          np.linspace(0, 1, 5)])
    assert graded["uniform"] is False and h == pytest.approx(0.05) and graded["reference_spacing_m"] == h
    assert graded["spacing_m"][0] is None and graded["spacing_m"][1:] == pytest.approx([0.25, 0.25])


def _scene(**kwargs):
    import witwin.maxwell as mw
    return mw.Scene(domain=mw.Domain(((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.01),
                    boundary=mw.BoundarySpec.pml(num_layers=8), **kwargs)


def test_h_derivation_reasons_and_symmetry_refusal():
    import witwin.maxwell as mw
    pulse = mw.GaussianPulse(frequency=1e9, fwidth=0.5e9)
    clean = _scene()
    clean = clean.updated(sources=(
        *clean.sources,
        mw.PointDipole(position=(0, 0, 0), polarization="Ez", source_time=pulse),
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=pulse,
                     injection=mw.TFSF(((-0.05, 0.05),) * 3)),
    ), structures=(
        *clean.structures,
        mw.Structure(mw.Box(position=(0, 0, 0), size=(0.04,) * 3), mw.Material(eps_r=4.0)),
    ))
    assert emfc_records.h_derivation_reasons(clean) == []
    emfc_records.refuse_unsupported(clean)

    magnetic = _scene()
    magnetic = magnetic.updated(structures=(
        *magnetic.structures,
        mw.Structure(mw.Box(position=(0, 0, 0), size=(0.04,) * 3),
                     mw.Material(mu_r=2.0, sigma_m=1.0), name="ferrite"),
    ), sources=(
        *magnetic.sources,
        mw.PlaneWave(direction=(0, 0, 1), polarization="Ex", source_time=pulse),
    ))
    reasons = emfc_records.h_derivation_reasons(magnetic)
    assert any("ferrite: mu_r != 1" in r for r in reasons)
    assert any("ferrite: sigma_m != 0" in r for r in reasons)
    assert any("PlaneWave, not TFSF" in r for r in reasons)

    with pytest.raises(ValueError, match="symmetry"):
        emfc_records.refuse_unsupported(_scene(symmetry=("PEC", None, None)))


def test_copper_foil_builders_have_no_h_derivation_and_are_refused():
    from benchmark.datagen.scenes import scene_builders
    for builder in scene_builders():
        scene = builder.build().scene
        reasons = emfc_records.h_derivation_reasons(scene)
        assert reasons == (["copper_foil_h_correction"] if builder.family == "F3" else []), builder.name
        emfc_records.refuse_unsupported(scene)
        # A5: F3 must store H (CopperFoil, and its printed-board grids are graded); the others need not.
        assert (emfc_records.scene_h_storage(scene) is not None) is (builder.family == "F3"), builder.name


def test_admit_requests_the_h_dft_or_refuses():
    from dataclasses import dataclass, replace
    from types import SimpleNamespace
    from benchmark.datagen.scenes import scene_builders
    import witwin.maxwell as mw

    builders = {b.name: b for b in scene_builders()}
    f3, f2 = builders["f3_coarse_11f"].build(), builders["f2_box_32"].build()
    has_switch = "volume_dft_components" in {f.name for f in __import__("dataclasses").fields(mw.FDTDConfig)}
    if not has_switch:
        # This maxwell records no volume H DFT: F3 is a task error before anything runs.
        with pytest.raises(ValueError, match="amendment A5: no H derivation"):
            emfc_records.admit_sample(f3)
    else:
        # F3 needs H (A5): the run records the volume H DFT too.
        emfc_records.admit_sample(f3)
        config = f3.fdtd_kwargs["config"]
        assert config.full_field_dft is True
        assert tuple(map(str, config.volume_dft_components)) == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    before = f2.fdtd_kwargs["config"]
    emfc_records.admit_sample(f2)   # uniform, H derivable: nothing requested
    assert f2.fdtd_kwargs["config"] is before

    @dataclass(frozen=True)
    class SwitchConfig:   # the announced opt-in: FDTDConfig(full_field_dft=..., volume_dft_components=...)
        full_field_dft: bool = False
        volume_dft_components: tuple = ("Ex", "Ey", "Ez")

    sample = SimpleNamespace(scene=f3.scene, fdtd_kwargs={"frequencies": (1e9,), "config": SwitchConfig()})
    emfc_records.admit_sample(sample)
    assert sample.fdtd_kwargs["config"] == replace(SwitchConfig(), full_field_dft=True,
                                                   volume_dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    assert sample.fdtd_kwargs["frequencies"] == (1e9,)


def test_scene_h_storage_from_the_grid_spec():
    import witwin.maxwell as mw
    graded = mw.Scene(domain=mw.Domain(((0.0, 0.4), (0.0, 0.4), (0.0, 0.3))),
                      grid=mw.GridSpec.custom([0.0, 0.1, 0.15, 0.2, 0.3, 0.4], np.linspace(0, 0.4, 5),
                                              np.linspace(0, 0.3, 4)),
                      boundary=mw.BoundarySpec.pec())
    assert emfc_records.scene_h_storage(graded) == "graded_grid"
    assert emfc_records.scene_h_storage(_scene()) is None
    auto = mw.Scene(domain=mw.Domain(((-0.1, 0.1),) * 3), grid=mw.GridSpec.auto(wavelength=0.3),
                    boundary=mw.BoundarySpec.pml(num_layers=8))
    assert emfc_records.scene_h_storage(auto).startswith("graded_grid")


def test_h_storage_rule():
    uniform = {"spacing_m": [0.1, 0.1, 0.2]}
    assert emfc_records.h_storage_required([], uniform) is None
    assert emfc_records.h_storage_required([], {"spacing_m": [0.1, None, 0.2]}) == "graded_grid"
    assert "copper_foil_h_correction" in emfc_records.h_storage_required(["copper_foil_h_correction"])
    with pytest.raises(ValueError, match="graded_grid"):
        emfc_records.refuse_h_storage(emfc_records.h_storage_required([], {"spacing_m": [None, 0.1, 0.1]}))
    emfc_records.refuse_h_storage(None)
    assert _metadata(False)["maxwell_record"]["h_stored"] is False


def test_magnetic_component_slices_follow_the_yee_staggering():
    full = (12, 10, 9)
    starts, stops = (2, 0, 1), (10, 10, 8)
    nodes = [stop - start for start, stop in zip(starts, stops)]
    for c, slices in enumerate(emfc_records.component_slices(starts, stops, "magnetic")):
        # Full-grid H shapes: component c is a node sample along its own axis, a cell sample elsewhere.
        shape = tuple(n if axis == c else n - 1 for axis, n in enumerate(full))
        assert np.empty(shape)[slices].shape == tuple(nodes[a] if a == c else nodes[a] - 1 for a in range(3))


def test_fdtd_config_from_a_catalog_builder_is_strict_json():
    from benchmark.datagen.scenes import scene_builders
    builder = next(b for b in scene_builders() if b.name == "f4_dipole_32")
    sample = builder.build()
    config = emfc_records.fdtd_config(builder, sample.fdtd_kwargs["config"])
    json.dumps(config, allow_nan=False)
    assert config["source"] == "builder_kwargs" and config["authoritative"] == "builder_kwargs"
    assert config["builder"] == "f4_dipole_32"
    assert config["kwargs"] == {"n": 32, "kind": "dipole", "count": 11, "finite": False}
    snapshot = config["config_snapshot"]
    assert snapshot["full_field_dft"] is False and snapshot["shutoff"] == 1e-5
    assert snapshot["absorber"] == "auto" and snapshot["field_dtype"] == "torch.float32"
    assert snapshot["run_time"]["steady_cycles"] == 160


def test_software_provenance_keys():
    record = emfc_records.software_provenance()
    import witwin.maxwell as mw
    assert record["maxwell_version"] == mw.__version__ and record["witwin_core_version"]
    assert (record["witwin_core_commit"] is None) == ("witwin_core_commit_reason" in record)


def test_record_metadata_is_accepted_by_encode_record():
    emfc = _emfc_records_api()
    meta = _metadata(False)
    container = emfc.encode_record({"_field_frequencies_hz": np.array([2.4e9, 2.5e9])}, metadata=meta)
    assert emfc.inspect_record(container).metadata == meta


def test_non_finite_last_step_field_ratio_is_a_task_error():
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="not finite .*no record is written"):
            _metadata(False, ratio=(3.7e-6, bad))


def test_options_validation():
    options = emfc_records.EmfcOptions(dataset_seed=0)
    assert options.shard_target_bytes == 1 << 30 and options.h_field_tolerance == options.field_tolerance == 1e-3
    assert emfc_records.EmfcOptions(dataset_seed=0, field_tolerance=2e-3).h_field_tolerance == 2e-3
    assert emfc_records.EmfcOptions(dataset_seed=0, h_field_tolerance=5e-3).settings()["h_field_tolerance"] == 5e-3
    with pytest.raises(ValueError, match="h_field_tolerance"):
        emfc_records.EmfcOptions(dataset_seed=0, h_field_tolerance=0.0)
    with pytest.raises(ValueError):
        emfc_records.EmfcOptions(dataset_seed=-1)
    with pytest.raises(ValueError):
        emfc_records.EmfcOptions(dataset_seed=0, field_tolerance=0.0)
    with pytest.raises(ValueError):
        emfc_records.EmfcOptions(dataset_seed=0, audit_fraction=1.5)


# --- the .pt path never imports emfc --------------------------------------------------------------
def _poisoned_python(tmp_path, code):
    poison = tmp_path / "poison" / "emfc"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text("raise ImportError('emfc must not be imported on this path')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(poison.parent), str(ROOT)]), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=300)


def test_pt_path_imports_no_emfc_and_cli_flags_are_checked(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]")
    code = f"""
import sys
from benchmark.datagen import scheduler, emfc_records
assert 'emfc' not in sys.modules
for argv, message in (
        (['--format', 'pt', '--dataset-seed', '1'], '--dataset-seed need --format emfc'),
        (['--format', 'emfc'], '--format emfc requires --dataset-seed')):
    sys.argv = ['scheduler', '--tasks', {str(tasks)!r}, '--output-dir', 'out', *argv]
    try:
        scheduler.main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError(argv)
assert 'emfc' not in sys.modules
try:
    emfc_records.require_emfc()
except RuntimeError as error:
    assert 'requires the emfc package' in str(error)
else:
    raise AssertionError('poisoned emfc imported')
print('OK')
"""
    run = _poisoned_python(tmp_path, code)
    assert run.returncode == 0, run.stderr
    assert "OK" in run.stdout
    assert "--dataset-seed need --format emfc" in run.stderr
    assert "--format emfc requires --dataset-seed" in run.stderr


# --- dataset identity, resume, finish (records without fields: no CUDA) -------------------------------
def _write(options, tmp_path, rank, sample_ids, tasks):
    emfc = _emfc_records_api()
    writer = emfc_records.open_writer(tmp_path, rank, options)
    for sample_id in sample_ids:
        task = tasks[sample_id]
        container = emfc.encode_record({"x": np.array([float(sample_id)])}, metadata={"sample_id": sample_id})
        writer.write_bytes(container, sample_id=sample_id, labels=emfc_records.labels(task, "F2", False))
    writer.close()


def test_resume_skips_sealed_records_moves_debris_and_checks_settings(tmp_path):
    _emfc_records_api()
    tasks = [Task(f"t{i}", "f2_box_32" if i % 2 else "f5_loaded_36", i) for i in range(5)]
    options = emfc_records.EmfcOptions(dataset_seed=5)
    resolved, pending, info = emfc_records.prepare_dataset(tmp_path, tasks, options)
    assert pending == [0, 1, 2, 3, 4] and info == {"present": 0, "moved_aside": []}
    stored = json.loads((tmp_path / "emfc_dataset.json").read_text())
    assert stored["dataset_uuid"] == resolved.dataset_uuid and stored["dataset_seed"] == 5

    _write(resolved, tmp_path, 0, (0, 3), tasks)
    _write(resolved, tmp_path, 1, (1,), tasks)
    debris = tmp_path / "shards" / "shard-r000002-000000.emfcs.tmp"
    debris.write_bytes(b"partial")
    again, pending, info = emfc_records.prepare_dataset(tmp_path, tasks, options)
    assert again.dataset_uuid == resolved.dataset_uuid
    assert pending == [2, 4] and info["present"] == 3
    assert not debris.exists() and len(info["moved_aside"]) == 1
    assert (tmp_path / "shards-aside" / info["moved_aside"][0]).read_bytes() == b"partial"

    with pytest.raises(ValueError, match="other settings"):
        emfc_records.prepare_dataset(tmp_path, tasks, emfc_records.EmfcOptions(dataset_seed=6))
    with pytest.raises(ValueError, match="past the task list"):
        emfc_records.prepare_dataset(tmp_path, tasks[:2], options)
    swapped = [Task(t.task_id, "f2_plate_40", t.seed) if i == 3 else t for i, t in enumerate(tasks)]
    with pytest.raises(ValueError, match="disagree with --tasks"):
        emfc_records.prepare_dataset(tmp_path, swapped, options)

    records = {"t2": {"status": "ok"}, "t4": {"status": "error", "path": "x"}}
    _write(again, tmp_path, 0, (4,), tasks)   # t2 reported ok but its record never sealed
    block = emfc_records.finish_dataset(tmp_path, again, records, {"t2": 2, "t4": 4}, info)
    assert records["t2"]["status"] == "error" and "missing from the sealed shards" in Path(
        records["t2"]["path"]).read_text()
    assert block["record_count"] == 4 and block["dataset_uuid"] == resolved.dataset_uuid
    assert block["skipped_present"] == 3 and block["audit_records"] == 0


def test_a_directory_without_the_dataset_file_is_refused(tmp_path):
    _emfc_records_api()
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "foreign.emfcs").write_bytes(b"")
    with pytest.raises(ValueError, match="not a datagen emfc dataset"):
        emfc_records.prepare_dataset(tmp_path, [Task("a", "f2_box_32")], emfc_records.EmfcOptions(dataset_seed=0))


def test_h_metadata_uses_the_h_tolerance_and_copies_the_h_time_label():
    attrs = {"field_time_offset_steps": -0.5}
    record = _metadata(False, h_reasons=["copper_foil_h_correction"], h_attrs=attrs,
                       h_field_tolerance=2e-3)["maxwell_record"]
    assert record["quantity"] == "E+H" and record["h_stored"] is True
    assert record["h_field_names"] == ["H/f000", "H/f001"] and record["field_names"] == ["E/f000", "E/f001"]
    assert record["h_stored_source"] == "maxwell_volume_dft" and record["h_field_time_offset_steps"] == -0.5
    assert record["h_derivation"] is None and "h_stored_reason" not in record
    assert record["field_fidelity"]["h_tolerance"] == 2e-3 and record["field_fidelity"]["tolerance"] == 1e-3
    audit = _metadata(True, h_reasons=["copper_foil_h_correction"], h_attrs=attrs)["maxwell_record"]
    assert audit["field_fidelity"]["h_tolerance"] is None


def test_witwin_core_tree_sha256_file_set_and_line_format(tmp_path):
    import hashlib
    import os
    root = tmp_path / "core"
    (root / "sub" / "__pycache__").mkdir(parents=True)
    files = {"__init__.py": b"a", "sub/b.py": b"bb", "sub/ext.so": b"\x00so", "win.pyd": b"pyd"}
    for name, data in files.items():
        (root / name).write_bytes(data)
    (root / "sub" / "__pycache__" / "c.py").write_bytes(b"ignored")
    (root / "x.pyc").write_bytes(b"ignored")
    (root / "notes.txt").write_bytes(b"ignored")
    (tmp_path / "outside.py").write_bytes(b"linked")
    os.symlink(tmp_path / "outside.py", root / "link.py")
    files["link.py"] = b"linked"
    lines = "".join(f"{name}\0{hashlib.sha256(files[name]).hexdigest()}\n"
                    for name in sorted(files, key=lambda n: n.encode()))
    assert emfc_records.witwin_core_tree_sha256(root) == hashlib.sha256(lines.encode()).hexdigest()
    import witwin.core
    provenance = emfc_records.software_provenance()
    assert provenance["witwin_core_tree_sha256"] == emfc_records.witwin_core_tree_sha256(
        Path(witwin.core.__file__).parent)


def test_config_snapshot_never_records_an_object_address():
    class Opaque:
        pass
    assert emfc_records._snapshot({"a": Opaque(), "b": "torch.float32", "c": 1.5}) == {
        "a": "Opaque", "b": "torch.float32", "c": 1.5}


def test_scene_payload_crops_eps_and_stores_the_scene_document():
    emfc = _emfc_records_api()
    import torch
    scene = _scene()
    grid = {"full_shape_nodes": [37, 37, 37], "crop_node_start": [8, 8, 8], "crop_node_stop": [29, 29, 29]}
    cells = torch.arange(36 ** 3, dtype=torch.float32).reshape(36, 36, 36)
    payload = emfc_records.scene_payload(scene, cells, grid)
    assert torch.equal(torch.as_tensor(payload.eps_r), cells[8:28, 8:28, 8:28])
    assert type(payload.definition).__name__ == "SceneDocument"
    nodes = torch.zeros(37, 37, 37)
    assert tuple(emfc_records.scene_payload(scene, nodes, grid).eps_r.shape) == (21, 21, 21)
    with pytest.raises(ValueError, match="neither the full cells nor the full nodes"):
        emfc_records.scene_payload(scene, torch.zeros(5, 5, 5), grid)
    # The record carries it as the envelope scene and gives the document back.
    container = emfc.encode_record({"x": np.zeros(1)}, scene=payload)
    _, decoded = emfc.decode_record(container, return_scene=True)
    assert decoded.definition.document == payload.definition.document

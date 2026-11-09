"""CPU synthetic receipts for exact determinism and shared Rule-2 bounds."""

import json
import os
from contextlib import contextmanager
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from benchmark.core_path import gates, setup_gates


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Rule-2 gate tests must never initialize CUDA")

    # Patching the lazy initializer catches any CUDA use by these tests, even when earlier
    # GPU tests in the same process have already initialized CUDA.
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)
    yield


def snapshot_pair(tmp_path, reference, candidate, *, gate="g4", half=None):
    for arm, payload in (("baseline", reference), ("candidate", candidate)):
        directory = tmp_path / arm / "vacuum"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(gate=gate, arm=arm, steps=8, bounded_items={},
                        core_runtime_active=arm == "candidate")
        (directory / "capture.json").write_text(json.dumps(metadata))
        torch.save(payload, directory / "tensors.pt")
        if half is not None:
            metadata["horizon_steps"] = 4
            (directory / "capture_half.json").write_text(json.dumps(metadata))
            torch.save(half[arm == "candidate"], directory / "tensors_half.pt")
    return SimpleNamespace(output=tmp_path, gate=gate, scenes=["vacuum"],
                           equivalence="rule2", bounded_items=None)


@pytest.mark.parametrize("level,delta,tier", [
    ("setup", 0., "T0"), ("setup", 1e-6, "T1"), ("setup", 1e-3, "T2"),
    ("run", 0., "T0"), ("run", 2e-5, "T1"), ("run", 1e-3, "T2"),
])
def test_tiers_through_gate_wrapper(level, delta, tier):
    reference = {"fields/Ex": torch.ones(3)}
    candidate = {"fields/Ex": reference["fields/Ex"] + delta}
    kwargs = dict(ref_half=reference, new_half=candidate) if level == "run" else {}
    row = gates.rule2_rows(reference, candidate, level=level, **kwargs)["fields/Ex"]
    assert row["tier"] == tier
    assert {"max_rel", "bound", "ratio_to_bound", "structural_zero"} <= row.keys()


@pytest.mark.parametrize("name,dtype", [
    ("setup/material_indices", torch.float32), ("setup/active_mask", torch.float32),
    ("setup/edge_owner", torch.float32), ("setup/sample_count", torch.float32),
    ("setup/csr_starts", torch.float64), ("setup/csr_targets", torch.float64),
    ("fields/Ex", torch.int64), ("fields/Ex", torch.bool),
])
@pytest.mark.parametrize("level", ["setup", "run"])
def test_discrete_leaves_stay_exact(name, dtype, level):
    reference = {name: torch.tensor([True] if dtype == torch.bool else [1], dtype=dtype)}
    new = reference[name].clone()
    if dtype == torch.bool:
        new.logical_not_()
    elif dtype.is_floating_point:
        new = torch.nextafter(new, torch.full_like(new, 2))
    else:
        new += 1
    candidate = {name: new}
    kwargs = dict(ref_half=reference, new_half=candidate) if level == "run" else {}
    row = gates.rule2_rows(reference, candidate, level=level, **kwargs)[name]
    assert row["tier"] == "T2" and row["bound"] == 0
    assert not row["structural_zero"]


@pytest.mark.parametrize("prefix", ["fields", "phasors", "dft_accumulators", "monitors/probe"])
def test_structural_zero_families(prefix):
    reference = {f"{prefix}/Ex": torch.ones(2), f"{prefix}/Ey": torch.full((2,), 1e-30)}
    candidate = reference | {f"{prefix}/Ey": torch.full((2,), 1e-12)}
    rows = gates.rule2_rows(reference, candidate, level="run", ref_half=reference, new_half=candidate)
    row = rows[f"{prefix}/Ey"]
    assert row["tier"] == "T1" and row["structural_zero"]
    assert row["family_max_ref"] == 1
    # Setup has no family floor.
    assert gates.rule2_rows(reference, candidate, level="setup")[f"{prefix}/Ey"]["tier"] == "T2"
    leaked = reference | {f"{prefix}/Ey": torch.full((2,), 1e-5)}
    assert gates.rule2_rows(reference, leaked, level="run", ref_half=reference,
                            new_half=leaked)[f"{prefix}/Ey"]["tier"] == "T2"


def test_families_do_not_mix_records_or_field_types():
    names = ["fields/Ex", "fields/Ey", "fields/Hx", "phasors/EX", "phasors/EY",
             "dft_accumulators/Ex/real", "dft_accumulators/Ey/imag",
             "monitors/one/Ex", "monitors/one/Ey", "monitors/two/Ey",
             "physics/ade_J", "physics/ade_P", "outputs/ports/p/voltage", "outputs/ports/p/current"]
    families = gates.run_families(names)
    assert families["fields/Ex"] == families["fields/Ey"]
    assert families["phasors/EX"] == families["phasors/EY"]
    assert families["dft_accumulators/Ex/real"] == families["dft_accumulators/Ey/imag"]
    assert families["monitors/one/Ex"] == families["monitors/one/Ey"]
    assert len(set(families.values())) == 10


def test_monitor_metadata_cannot_inflate_component_family():
    reference = {"monitors/probe/components/Ex/data": torch.ones(1),
                 "monitors/probe/components/Ey/data": torch.tensor([1e-30]),
                 "monitors/probe/components/Ex/coords/0": torch.tensor([1e12]),
                 "monitors/probe/frequency": torch.tensor([1e12])}
    candidate = reference | {"monitors/probe/components/Ey/data": torch.tensor([1e-5])}
    rows = gates.rule2_rows(reference, candidate, level="run", ref_half=reference, new_half=candidate)
    row = rows["monitors/probe/components/Ey/data"]
    assert row["tier"] == "T2" and row["family_max_ref"] == 1
    assert "monitors/probe/frequency" not in gates.run_families(reference)
    assert "monitors/probe/components/Ex/coords/0" not in gates.run_families(reference)


def test_run_growth_rejected_and_reported(tmp_path):
    reference = {"fields/Ex": torch.ones(2)}
    half = {"fields/Ex": reference["fields/Ex"] + 1e-6}
    full = {"fields/Ex": reference["fields/Ex"] + 1e-5}
    args = snapshot_pair(tmp_path, reference, full, gate="g1", half=(reference, half))
    with pytest.raises(SystemExit, match="1"):
        gates.compare(args)
    report = json.loads((tmp_path / "comparison.json").read_text())
    row = report["items"]["vacuum/fields/Ex"]
    assert row["half"]["tier"] == row["full"]["tier"] == "T1"
    assert row["tier"] == "T2" and row["growth"] > 4
    assert report["summary"] == {"T0": 0, "T1": 0, "T2": 1}
    assert not report["passed"]


def test_half_horizon_rejection_is_not_hidden_by_equal_full():
    reference = {"fields/Ex": torch.ones(1)}
    half = {"fields/Ex": torch.tensor([1.1])}
    row = gates.rule2_rows(reference, reference, level="run", ref_half=reference,
                            new_half=half)["fields/Ex"]
    assert row["full"]["tier"] == "T0" and row["tier"] == "T2"


@pytest.mark.parametrize("gate", ["g1", "g4"])
def test_rule2_comparison_receipt(tmp_path, gate):
    reference = {"fields/Ex": torch.ones(1), "state/sample_count": torch.tensor([4])}
    candidate = reference | {"fields/Ex": torch.tensor([1.000001])}
    args = snapshot_pair(tmp_path, reference, candidate, gate=gate,
                         half=(reference, candidate) if gate == "g1" else None)
    gates.compare(args)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["passed"]
    assert report["summary"] == {"T0": 1, "T1": 1, "T2": 0}


def test_rule2_g1_requires_both_horizons(tmp_path):
    reference = {"fields/Ex": torch.ones(1)}
    args = snapshot_pair(tmp_path, reference, reference, gate="g1")
    with pytest.raises(FileNotFoundError):
        gates.compare(args)


@pytest.mark.parametrize("delta", [0., 0.25])
def test_bitwise_output_is_byte_identical(tmp_path, delta):
    reference = {"fields/Ex": torch.tensor([1.0, 2.0])}
    candidate = {"fields/Ex": reference["fields/Ex"] + delta}
    args = snapshot_pair(tmp_path, reference, candidate)
    del args.equivalence  # Existing callers also retain the exact default.
    if delta:
        with pytest.raises(SystemExit, match="1"):
            gates.compare(args)
    else:
        gates.compare(args)
    # Frozen serialization of the original synthetic exact comparison.
    expected = '{\n  "passed": PASSED,\n  "items": {\n    "vacuum/fields/Ex": {\n' \
               '      "equal": PASSED,\n      "max_abs": DELTA,\n      "max_rel": DELTA,\n' \
               '      "passed": PASSED\n    }\n  }\n}\n'
    expected = expected.replace("PASSED", "false" if delta else "true").replace("DELTA", str(delta))
    assert (tmp_path / "comparison.json").read_bytes() == expected.encode()


def test_setup_wrapper_preserves_metadata_and_discrete_leaves():
    reference = {"_type": "Model", "table": torch.ones(1), "indices": torch.ones(1), "count": 7}
    candidate = reference | {"table": torch.tensor([1.000001])}
    rows = setup_gates.setup_rule2_rows(reference, candidate)
    assert rows["setup/table"]["tier"] == "T1"
    assert rows["setup/count"]["tier"] == "T0"
    candidate["indices"] = torch.tensor([1.000001])
    assert setup_gates.setup_rule2_rows(reference, candidate)["setup/indices"]["tier"] == "T2"
    with pytest.raises(AssertionError, match="metadata"):
        setup_gates.setup_rule2_rows(reference, candidate | {"_type": "OtherModel"})


@pytest.mark.parametrize("change", ["shape", "dtype", "nan"])
def test_structural_and_nonfinite_changes_fail(change):
    reference = {"fields/Ex": torch.ones(1)}
    candidate = {"fields/Ex": {"shape": torch.ones(2), "dtype": torch.ones(1, dtype=torch.float64),
                                "nan": torch.tensor([float("nan")])}[change]}
    assert gates.rule2_rows(reference, candidate, level="setup")["fields/Ex"]["tier"] == "T2"


def repeated_captures(tmp_path, *, arm="candidate"):
    directories = [tmp_path / f"repeat-{i}" for i in range(3)]
    for index, directory in enumerate(directories):
        scene = directory / arm / "vacuum"
        scene.mkdir(parents=True)
        torch.save({"fields/Ex": torch.ones(2)}, scene / "tensors.pt")
        torch.save({"fields/Ex": torch.zeros(2)}, scene / "tensors_half.pt")
        for filename in ("capture.json", "capture_half.json"):
            (scene / filename).write_text(json.dumps({"process_id": index + 1}))
    return directories


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_determinism_command(tmp_path, monkeypatch, arm):
    directories = repeated_captures(tmp_path, arm=arm)
    monkeypatch.setattr(sys, "argv", ["gates", "determinism", "--dirs", *map(str, directories),
                                     "--arm", arm, "--output", str(tmp_path)])
    gates.main()
    report = json.loads((tmp_path / "determinism.json").read_text())
    assert report["passed"] and len(report["items"]) == 2


@pytest.mark.parametrize("failure", ["mismatch", "missing", "missing_half", "metadata", "process",
                                     "tensor", "dtype", "two", "duplicate"])
def test_determinism_rejects_bad_captures(tmp_path, failure):
    directories = repeated_captures(tmp_path)
    target = directories[-1] / "candidate" / "vacuum"
    if failure == "mismatch":
        torch.save({"fields/Ex": torch.tensor([1., 1.000001])}, target / "tensors.pt")
    elif failure.startswith("missing"):
        (target / ("tensors_half.pt" if failure == "missing_half" else "tensors.pt")).unlink()
    elif failure == "metadata":
        (target / "capture.json").unlink()
    elif failure == "process":
        (target / "capture.json").write_text(json.dumps({"process_id": 1}))
    elif failure in ("tensor", "dtype"):
        torch.save({"other": torch.ones(2)} if failure == "tensor" else
                   {"fields/Ex": torch.ones(2, dtype=torch.float64)}, target / "tensors.pt")
    elif failure == "two":
        directories.pop()
    else:
        directories[-1] = directories[0]
    with pytest.raises((AssertionError, ValueError, FileNotFoundError, SystemExit)):
        gates.determinism(SimpleNamespace(dirs=directories, arm="candidate", output=tmp_path))


def test_cpu_cli_in_fresh_process(tmp_path):
    directories = repeated_captures(tmp_path)
    script = ("import runpy, torch; from unittest.mock import patch; "
              "guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')); "
              "guard.start(); runpy.run_module('benchmark.core_path.gates', run_name='__main__'); "
              "assert not torch.cuda.is_initialized()")
    result = subprocess.run([sys.executable, "-c", script, "determinism", "--dirs", *map(str, directories),
                             "--output", str(tmp_path)], capture_output=True, text=True,
                            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stdout + result.stderr


def test_capture_dispatches_fresh_half_and_full_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gates", "capture", "--scenes", "planewave_vacuum", "--half-horizon",
                                     "--output", str(tmp_path)])
    with patch.object(gates.subprocess, "run") as run:
        gates.main()
    assert run.call_count == 4
    commands = [call.args[0] for call in run.call_args_list]
    assert all("--half-horizon" in command for command in commands)
    assert ["--snapshot-half" in command for command in commands] == [False, True, False, True]


def test_half_capture_uses_full_configuration_and_stop(tmp_path, monkeypatch):
    fields = {name: torch.ones(1) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    solver = SimpleNamespace(**fields, _dft_batched_fields={}, _shutoff_triggered=False,
                             _shutoff_step=None, dft_enabled=False, _observer_spectral_entries=[],
                             tfsf_enabled=False)
    result = SimpleNamespace(fields={"EX": torch.ones(1)}, monitors={})
    simulation = SimpleNamespace(prepare=lambda: SimpleNamespace(solver=solver))
    stops = []
    simulation._execute_fdtd_solve = lambda target, **kwargs: stops.append((target, kwargs))
    case = SimpleNamespace(steps=8, name="vacuum", initial_fields=False, boundaries=False,
                           capture_failure=False, shutoff=0, plan=lambda: {})
    monkeypatch.setattr(gates, "build_core_case", lambda name: case)
    monkeypatch.setattr(gates, "make_simulation", lambda *args, **kwargs: simulation)
    monkeypatch.setattr(gates, "partial_result", lambda target: result)
    args = SimpleNamespace(output=tmp_path, gate="g1", scene="vacuum", arm="baseline", steps=256,
                           bounded_items=None, half_horizon=True, snapshot_half=True, chunk_size=32)
    gates.capture_case(args)
    assert stops == [(solver, {"stop_step": 4})]
    path = tmp_path / "baseline" / "vacuum"
    assert (path / "tensors_half.pt").is_file() and not (path / "tensors.pt").exists()
    metadata = json.loads((path / "capture_half.json").read_text())
    assert metadata["steps"] == 8 and metadata["horizon_steps"] == 4


def test_partial_export_matches_public_field_names(monkeypatch):
    from witwin.maxwell.fdtd.runtime import stepping
    from witwin.maxwell.fdtd import copper_foil

    field = torch.ones(2, dtype=torch.complex64)
    solver = SimpleNamespace(
        dft_enabled=True, observers_enabled=True, time_observers_enabled=True,
        _sync_dft_legacy_state=lambda: None, _sync_observer_legacy_state=lambda: None,
        get_frequency_solution=lambda **kwargs: {"Ex": field, "frequencies": (1., 2.)},
        get_observer_results=lambda: {"spectral": {"Ex": field}},
        get_time_observer_results=lambda: {"time": {"data": field}},
    )
    monkeypatch.setattr(stepping, "finalize_wire_data", lambda target: {})
    monkeypatch.setattr(stepping, "finalize_port_data", lambda target: {"port": field})
    monkeypatch.setattr(stepping, "finalize_circuit_data", lambda target: {})
    monkeypatch.setattr(stepping, "finalize_embedded_networks", lambda target, ports: {})
    monkeypatch.setattr(copper_foil, "copper_loss_report", lambda target: {})
    result = gates.partial_result(solver)
    assert result.fields.keys() == {"EX"}
    assert torch.equal(result.fields["EX"], field)
    assert result.monitors.keys() == {"spectral", "time"}
    assert result.ports.keys() == {"port"}


@pytest.mark.parametrize("equivalence,delta,passed", [
    ("bitwise", 0., True), ("bitwise", 1e-6, False),
    ("rule2", 1e-6, True), ("rule2", 1e-3, False),
])
def test_accelerated_g4_modes(tmp_path, monkeypatch, equivalence, delta, passed):
    scene = SimpleNamespace(subpixel=SimpleNamespace(pec=None))
    scene.clone = lambda **kwargs: scene
    monkeypatch.setattr(setup_gates, "scene_for", lambda name: scene)
    monkeypatch.setattr(setup_gates, "MODES", ((1, "arithmetic"),))
    monkeypatch.setattr(setup_gates.mw, "SubpixelSpec", lambda **kwargs: kwargs)

    def simulation(target, *, accelerated):
        solver = SimpleNamespace(Nx=1, Ny=1, Nz=1,
                                 _compiled_material_model={"table": torch.tensor([1. + delta * accelerated])})
        return SimpleNamespace(prepare=lambda: SimpleNamespace(solver=solver))

    @contextmanager
    def record():
        class Calls(list):
            pieces = {}
        yield Calls([True])

    monkeypatch.setattr(setup_gates, "setup_simulation", simulation)
    monkeypatch.setattr(setup_gates, "record_setup", record)
    monkeypatch.setattr(setup_gates, "coefficient_payload", lambda target: {})
    for method in ("empty_cache", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, method, lambda: 0)
    if passed:
        setup_gates.g4(["synthetic"], tmp_path, equivalence=equivalence)
    else:
        with pytest.raises(AssertionError):
            setup_gates.g4(["synthetic"], tmp_path, equivalence=equivalence)
    if equivalence == "rule2":
        row, = json.loads((tmp_path / "g4.json").read_text())
        assert row["passed"] == passed
        assert row["items"]["setup/model/table"]["tier"] == ("T1" if passed else "T2")
        # A rejected row is never treated as completed on the next invocation.
        if not passed:
            with pytest.raises(AssertionError):
                setup_gates.g4(["synthetic"], tmp_path, equivalence=equivalence)
            assert len(json.loads((tmp_path / "g4.json").read_text())) == 2

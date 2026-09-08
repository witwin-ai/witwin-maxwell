import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.batch import BATCH_WORKLOADS, audit, build_simulation, main, compare_probes
from witwin.maxwell.execution.capacity import tensor_memory_manifest


def test_manifest_counts_storage_once_and_records_views_and_cycles():
    state = torch.zeros(64)
    root = SimpleNamespace(field=state, tail=state[16:], broadcast=state[:1].expand(100))
    root.self = root
    manifest = tensor_memory_manifest({"slot": root, "output_alias": state})
    assert manifest["storage_bytes_by_device"] == {"cpu": 256}
    assert len(manifest["allocations"]) == 1
    assert len(manifest["allocations"][0]["owners"]) == 4
    assert manifest["container_aliases"]["slot.self"] == "slot"
    json.dumps(manifest)


def test_manifest_tracks_separate_storage_and_reports_unsupported_layouts():
    a = torch.ones(3, requires_grad=True)
    sparse = torch.sparse_coo_tensor([[0]], [1.0], (2,))
    manifest = tensor_memory_manifest({"a": a, "b": a.clone(), "sparse": sparse})
    assert manifest["storage_bytes_by_device"]["cpu"] == 24
    assert manifest["unsupported_tensors"] == ["sparse"]
    assert a.grad is None


@pytest.mark.parametrize("name", BATCH_WORKLOADS)
def test_fixed_workloads_have_local_variations_and_identical_controls(name):
    workload = BATCH_WORKLOADS[name]
    a = build_simulation(workload, index=0, device="cpu")
    b = build_simulation(workload, index=9, device="cpu")
    same = build_simulation(workload, index=9, identical=True, device="cpu")
    assert a.scene is not b.scene
    assert a.config.run_time.dt == b.config.run_time.dt
    assert a.frequencies == b.frequencies
    assert a.scene.structures[0].material.eps_r != b.scene.structures[0].material.eps_r
    assert a.scene.structures[0].material.eps_r == same.scene.structures[0].material.eps_r


def test_inventory_does_not_initialize_cuda(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inventory must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    main(["--inventory"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["workloads"]) == 3
    assert len(payload["workloads"][2]["frequencies"]) == 31
    assert payload["deduplication"] is False


def test_audit_has_no_cpu_solver_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA driver"):
        audit(BATCH_WORKLOADS["launch"], device="cuda:0")


def test_tuning_cli_routes_devices_and_preserves_measurements(monkeypatch, tmp_path, capsys):
    import benchmark.batch_profile as profile
    captured = {}
    def tune(workload, **kwargs):
        captured.update(kwargs)
        return dict(workload=workload.name, measurements=[.1, .2, .3])
    monkeypatch.setattr(profile, "audit_tuning", tune)
    target = tmp_path / "tune.json"
    main(["--tune-audit", "launch", "--devices", "cuda:0", "cuda:1", "--variants", "5",
          "--concurrency", "3", "--output", str(target)])
    assert captured["devices"] == ("cuda:0", "cuda:1")
    assert captured["variants"] == 5 and captured["capacity"] == 3
    assert json.loads(target.read_text()) == json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        main(["--inventory", "--devices", "cuda:0"])


def test_tuning_report_survives_regeneration(monkeypatch, tmp_path):
    from benchmark import report
    target = tmp_path / "RESULTS.md"
    monkeypatch.setattr(report, "RESULTS_MD", target)
    payload = dict(workload=dict(name="launch"), variants=3, devices=["cuda:0"],
        plan=dict(candidate_measurements=[dict(policy=dict(batch_size=3, slots_per_device=1, graph_steps=32),
            median_s=.1, peak_bytes={"cuda:0": 1024**2})], selected_policy=dict(batch_size=3),
            tuning=dict(elapsed_s=2., selected_relative_throughput=.97)),
        holdout_median_s=.11, max_probe_absolute_error=0.)
    report.record_batch_tuning_audit(payload, tmp_path / "raw.json")
    original = target.read_text()
    report.record_batch_tuning_audit(payload, tmp_path / "raw.json")
    assert target.read_text() == original
    assert "0.9700" in original and "| 3 | 1 | 32 |" in original
    report.write_results_markdown([])
    assert original.strip() in target.read_text()


def test_probe_comparison_detects_phase_and_lane_errors():
    probe = {"shape": [2], "real": [1.0, 2.0], "imag": [0.0, 0.5]}
    assert compare_probes([probe], [probe])[0]["rel_l2"] == 0
    shifted = dict(probe, imag=[0.0, 0.6])
    with pytest.raises(AssertionError, match="spectrum mismatch"):
        compare_probes([probe], [shifted])
    with pytest.raises(ValueError, match="counts differ"):
        compare_probes([probe], [probe, probe])
    with pytest.raises(ValueError, match="shape"):
        compare_probes([probe], [dict(probe, real=[1.0])])
    with pytest.raises(AssertionError):
        compare_probes([dict(probe, real=[float("nan"), 1.0])], [probe])


def test_comparison_cli_preserves_protocol_configuration(monkeypatch, tmp_path, capsys):
    import benchmark.batch as module
    captured = {}
    def compare(workload, **kwargs):
        captured.update(kwargs)
        captured["steps"] = workload.time_steps
        return {"schema_version": 1, "smoke_only": True}
    monkeypatch.setattr(module, "benchmark_baselines", compare)
    target = tmp_path / "report.json"
    main(["--compare", "launch", "--variants", "100", "--identical", "--steps", "32", "--output", str(target)])
    assert captured["variants"] == 100
    assert captured["identical"] is True
    assert captured["warmups"] == 2 and captured["repeats"] == 5
    assert captured["steps"] == 32
    assert json.loads(target.read_text()) == json.loads(capsys.readouterr().out)


def test_report_update_preserves_other_results_and_replaces_its_section(monkeypatch, tmp_path):
    from benchmark import report
    target = tmp_path / "RESULTS.md"
    target.write_text("Existing solver results\n", encoding="utf-8")
    monkeypatch.setattr(report, "RESULTS_MD", target)
    payload = {"workload": {"name": "launch", "time_steps": 32}, "variants": 3,
               "identical": False, "device": "cuda:0", "warmups": 0, "repeats": 1,
               "summary": {"sequential": {"median_wall_s": 1.0, "min_wall_s": 0.9, "max_wall_s": 1.1}}}
    report.record_batch_baseline(payload, tmp_path / "raw.json")
    report.record_batch_baseline(payload, tmp_path / "raw.json")
    text = target.read_text()
    assert text.startswith("Existing solver results")
    assert text.count("## Independent execution baseline") == 1
    assert "raw.json" in text
    report.write_results_markdown([])
    regenerated = target.read_text()
    assert regenerated.count("## Independent execution baseline") == 1
    assert "raw.json" in regenerated


def test_native_cli_and_report_keep_historical_baselines(monkeypatch, tmp_path, capsys):
    import benchmark.batch as module
    from benchmark import report
    captured = {}
    def compare(workload, **kwargs):
        captured.update(kwargs)
        return {"workload": {"name": workload.name, "time_steps": workload.time_steps},
                "variants": kwargs["variants"], "identical": kwargs["identical"],
                "device": kwargs["device"], "warmups": kwargs["warmups"], "repeats": kwargs["repeats"],
                "include_native": kwargs["include_native"],
                "summary": {"native_batch": {"median_wall_s": 1., "min_wall_s": .9, "max_wall_s": 1.1}},
                "isolated_memory": [{"mode": "native_batch", "driver_peak_increment_bytes": 1024**2}]}
    monkeypatch.setattr(module, "benchmark_baselines", compare)
    target = tmp_path / "native.json"
    module.main(["--compare", "launch", "--include-native", "--output", str(target)])
    assert captured["include_native"] is True
    payload = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(report, "RESULTS_MD", tmp_path / "RESULTS.md")
    report.record_batch_baseline(dict(payload, include_native=False), tmp_path / "old.json")
    report.record_batch_baseline(payload, target)
    report.write_results_markdown([])
    text = report.RESULTS_MD.read_text()
    assert "Independent execution baseline" in text
    assert "Execution comparison including native Batch" in text
    assert "| native_batch | 1.00 |" in text
    assert "old.json" in text and "native.json" in text

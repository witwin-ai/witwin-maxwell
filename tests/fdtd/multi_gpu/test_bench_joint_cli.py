from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def bench_joint():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "dev" / "fdtd" / "multi_gpu" / "bench_joint.py"
    module_name = "_witwin_test_bench_joint"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_existing_cli_defaults_remain_vacuum_compatible(bench_joint):
    args = bench_joint._parse_args([])

    assert args.workload == "vacuum"
    assert tuple(args.devices) == ("cuda:0", "cuda:1")
    assert (args.nodes_x, args.nodes_y, args.nodes_z) == (257, 129, 129)
    assert args.steps == 200
    assert args.warmups == 1
    assert args.repeats == 5
    assert args.gather_fields is False
    assert args.weak_scaling is False


def test_monitor_error_reports_maximum_numeric_payload_difference(bench_joint):
    reference = {
        "probe": {
            "data": torch.tensor((1.0, 2.0)),
            "coords": (np.asarray((0.0, 0.1)),),
        }
    }
    actual = {
        "probe": {
            "data": torch.tensor((1.0, 2.25)),
            "coords": (np.asarray((0.0, 0.1)),),
        }
    }

    error = bench_joint._monitor_error(actual, reference)

    assert error["max_abs"] == pytest.approx(0.25)
    assert error["max_rel"] == pytest.approx(0.125)
    assert error["finite"] is True
    assert error["passed"] is False


@pytest.mark.parametrize(
    ("name", "frequencies", "full_field_dft", "cpml_mode", "structure_count"),
    (
        ("vacuum", (1.0e9,), False, "disabled", 0),
        ("cpml_dielectric", (1.0e9,), False, "slab", 1),
        ("multifrequency_dft", (0.8e9, 1.2e9), True, "disabled", 0),
    ),
)
def test_workload_specs_build_expected_declarative_scene(
    bench_joint,
    name,
    frequencies,
    full_field_dft,
    cpml_mode,
    structure_count,
):
    spec = bench_joint._workload_spec(name)
    scene = bench_joint._workload_scene(
        name,
        (17, 9, 9),
        spacing=0.01,
        device=torch.device("cpu"),
    )

    assert spec.frequencies == frequencies
    assert spec.full_field_dft is full_field_dft
    assert spec.cpml_mode == cpml_mode
    assert len(scene.structures) == structure_count
    assert len(scene.sources) == 1
    assert tuple(monitor.name for monitor in scene.monitors) == ("probe",)
    assert scene.boundary.uses_kind("pml") is (cpml_mode != "disabled")


@pytest.mark.parametrize(
    ("workload", "expected_cpml_config"),
    (
        ("vacuum", None),
        ("cpml_dielectric", {"memory_mode": "slab"}),
        ("multifrequency_dft", None),
    ),
)
def test_simulation_routes_cpml_memory_mode_only_when_enabled(
    bench_joint,
    monkeypatch,
    workload,
    expected_cpml_config,
):
    calls = []
    monkeypatch.setattr(
        bench_joint.mw.Simulation,
        "fdtd",
        lambda scene, **kwargs: calls.append((scene, kwargs)) or object(),
    )

    bench_joint._simulation(
        workload,
        (17, 9, 9),
        spacing=0.01,
        steps=5,
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    kwargs = calls[0][1]
    if expected_cpml_config is None:
        assert "cpml_config" not in kwargs
    else:
        assert kwargs["cpml_config"] == expected_cpml_config


@pytest.mark.parametrize(
    ("workload", "expected_cpml", "expected_dft"),
    (
        ("vacuum", "disabled", False),
        ("cpml_dielectric", "slab", False),
        ("multifrequency_dft", "disabled", True),
    ),
)
def test_report_schema_records_workload_shapes_runtime_and_errors(
    bench_joint,
    monkeypatch,
    tmp_path,
    capsys,
    workload,
    expected_cpml,
    expected_dft,
):
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    calls = []
    topology = {
        "kind": "cuda_p2p",
        "neighbor_pairs": [{"left": "cuda:0", "right": "cuda:1"}],
    }
    local_shapes = {"cuda:0": (6, 7, 7), "cuda:1": (7, 7, 7)}

    monkeypatch.setattr(bench_joint, "_require_devices", lambda _values: devices)
    monkeypatch.setattr(
        bench_joint,
        "_p2p_bandwidth",
        lambda *_args, **_kwargs: {
            "cuda:0->cuda:1": {"bandwidth_gbps_median": 100.0},
            "cuda:1->cuda:0": {"bandwidth_gbps_median": 100.0},
        },
    )
    monkeypatch.setattr(
        bench_joint,
        "_hardware",
        lambda _devices: {
            "torch_version": "2.test",
            "torch_cuda_version": "12.test",
            "devices": ({"device": "cuda:0"}, {"device": "cuda:1"}),
        },
    )

    def fake_diagnostics(shape, *, workload, spacing, steps, devices):
        calls.append(("diagnostics", workload, shape, steps))
        return {
            "passed": True,
            "field_max_abs": 1.0e-7,
            "field_max_rel": 2.0e-7,
            "monitor_max_abs": 3.0e-7,
            "monitor_max_rel": 4.0e-7,
        }

    def fake_single(shape, *, workload, spacing, steps, device):
        calls.append(("single", workload, shape, steps))
        return {
            "elapsed_s": 2.0,
            "ms_per_step": 200.0,
            "peak_memory_bytes": 2000,
            "local_node_shapes": {str(device): shape},
            "cpml_memory_mode": expected_cpml,
        }

    def fake_multi(
        shape,
        *,
        workload,
        spacing,
        steps,
        devices,
        overlap,
        gather_fields,
    ):
        calls.append(("multi", workload, shape, steps))
        return {
            "elapsed_s": 1.0,
            "ms_per_step": 100.0,
            "peak_memory_bytes": {"cuda:0": 800, "cuda:1": 850},
            "halo_bytes_per_step": 128,
            "halo_bytes_total": 1280,
            "overlap_active": overlap,
            "transport": "cuda_p2p",
            "partitions": (
                {"rank": 0, "device": "cuda:0"},
                {"rank": 1, "device": "cuda:1"},
            ),
            "topology": topology,
            "local_node_shapes": local_shapes,
            "cpml_memory_mode": expected_cpml,
        }

    monkeypatch.setattr(bench_joint, "_parity_diagnostics", fake_diagnostics)
    monkeypatch.setattr(bench_joint, "_single_sample", fake_single)
    monkeypatch.setattr(bench_joint, "_multi_sample", fake_multi)

    output_path = tmp_path / f"{workload}.json"
    returned = bench_joint.main(
        [
            "--workload",
            workload,
            "--nodes-x",
            "11",
            "--nodes-y",
            "7",
            "--nodes-z",
            "7",
            "--steps",
            "10",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--json",
            str(output_path),
        ]
    )
    rendered = json.loads(output_path.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)

    assert returned["workload"] == workload
    assert stdout == rendered
    assert rendered["workload"] == workload
    assert rendered["shapes"] == {
        "global_node_shape": [11, 7, 7],
        "local_node_shapes": {
            "cuda:0": [6, 7, 7],
            "cuda:1": [7, 7, 7],
        },
    }
    assert rendered["execution"]["steps"] == 10
    assert rendered["execution"]["full_field_dft"] is expected_dft
    assert rendered["execution"]["graph_mode"] == "disabled"
    assert rendered["execution"]["cpml_mode_actual"] == expected_cpml
    assert rendered["communication"]["halo_bytes_per_step"] == 128
    assert rendered["communication"]["halo_bytes_total"] == 1280
    assert rendered["communication"]["topology"] == topology
    assert rendered["hardware"]["torch_version"] == "2.test"
    assert rendered["hardware"]["torch_cuda_version"] == "12.test"
    assert rendered["numerical_error"] == {
        "diagnostics_run": True,
        "field_max_abs": 1.0e-7,
        "field_max_rel": 2.0e-7,
        "monitor_max_abs": 3.0e-7,
        "monitor_max_rel": 4.0e-7,
    }
    assert rendered["memory"]["two_gpu_peak_bytes"] == {
        "cuda:0": 800,
        "cuda:1": 850,
    }
    assert {row[0] for row in calls} == {"diagnostics", "single", "multi"}
    assert {row[1] for row in calls} == {workload}

from collections.abc import Mapping

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.result import RESULT_SNAPSHOT_SCHEMA_VERSION


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _data(name="network", *, requires_grad=False):
    return mw.CircuitData(
        circuit_name=name,
        times=torch.tensor([0.0, 1.0], dtype=torch.float64),
        node_names=("0", "signal"),
        node_voltages=torch.tensor(
            [[0.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
        branch_names=("R1",),
        branch_currents=torch.tensor(
            [[0.0], [0.02]],
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
        device_powers={
            "R1": torch.tensor(
                [0.0, 0.02],
                dtype=torch.float64,
                requires_grad=requires_grad,
            )
        },
        energy_balance=torch.tensor(
            [0.0, 1.0e-12],
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
        diagnostics={
            "factorization_count": 1,
            "condition": torch.tensor(
                [1.0, 1.5],
                dtype=torch.float64,
                requires_grad=requires_grad,
            ),
            "method": "backward_euler",
        },
    )


def _result(*, circuits=None):
    return mw.Result(
        method="fdtd",
        scene=mw.Scene(device="cpu"),
        frequency=1.0e9,
        circuits=circuits,
    )


def _assert_circuit_data_equal(actual, expected):
    assert actual.circuit_name == expected.circuit_name
    assert actual.node_names == expected.node_names
    assert actual.branch_names == expected.branch_names
    assert actual.diagnostics["factorization_count"] == 1
    assert actual.diagnostics["method"] == "backward_euler"
    torch.testing.assert_close(actual.times, expected.times.detach())
    torch.testing.assert_close(actual.node_voltages, expected.node_voltages.detach())
    torch.testing.assert_close(actual.branch_currents, expected.branch_currents.detach())
    torch.testing.assert_close(actual.device_powers["R1"], expected.device_powers["R1"].detach())
    torch.testing.assert_close(actual.energy_balance, expected.energy_balance.detach())
    torch.testing.assert_close(
        actual.diagnostics["condition"],
        expected.diagnostics["condition"].detach(),
    )


def test_result_circuit_accessor_preserves_identity_and_reports_stats():
    data = _data()
    result = _result(circuits={"network": data})

    assert result.circuit("network") is data
    assert result.circuits == {"network": data}
    assert result.stats()["num_circuits"] == 1
    with pytest.raises(KeyError, match="Choices:.*network"):
        result.circuit("missing")


def test_result_rejects_invalid_circuit_mappings():
    data = _data()
    with pytest.raises(TypeError, match="circuits must be a mapping"):
        _result(circuits=(data,))
    with pytest.raises(TypeError, match="CircuitData instance"):
        _result(circuits={"network": object()})
    with pytest.raises(ValueError, match="does not match"):
        _result(circuits={"other": data})
    with pytest.raises(ValueError, match="Duplicate normalized circuit key"):
        _result(circuits={1: _data("1"), "1": _data("1")})


def test_circuit_data_v1_round_trip_is_detached_and_cpu_resident(tmp_path):
    data = _data(requires_grad=True)
    path = tmp_path / "nested" / "circuit.pt"

    data.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema_version"] == 1
    assert payload["data_type"] == "CircuitData"
    assert set(payload) == {
        "schema_version",
        "data_type",
        "circuit_name",
        "times",
        "node_names",
        "node_voltages",
        "branch_names",
        "branch_currents",
        "device_powers",
        "energy_balance",
        "diagnostics",
    }
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(payload))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(payload))

    loaded = mw.CircuitData.load(path, map_location="cpu")
    _assert_circuit_data_equal(loaded, data)
    assert all(not tensor.requires_grad for tensor in _iter_tensors(loaded.diagnostics))
    assert data.node_voltages.requires_grad


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(schema_version=2), "Unsupported CircuitData schema_version"),
        (lambda payload: payload.update(data_type="Other"), "invalid data_type"),
        (lambda payload: payload.pop("diagnostics"), "missing required keys: diagnostics"),
    ),
)
def test_circuit_data_load_rejects_malformed_v1_payloads(tmp_path, mutation, message):
    path = tmp_path / "circuit.pt"
    _data().save(path)
    payload = torch.load(path, weights_only=True)
    mutation(payload)
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        mw.CircuitData.load(path)


def test_circuit_data_save_prevalidates_diagnostics_before_filesystem_side_effects(
    tmp_path,
):
    data = _data()
    data.diagnostics["unsafe"] = object()
    path = tmp_path / "nested" / "circuit.pt"

    with pytest.raises(TypeError, match="CircuitData.diagnostics.*object"):
        data.save(path)

    assert path.parent.exists() is False


def test_result_v2_round_trip_embeds_circuit_data(tmp_path):
    data = _data(requires_grad=True)
    result = _result(circuits={"network": data})
    path = tmp_path / "result.pt"

    result.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema_version"] == RESULT_SNAPSHOT_SCHEMA_VERSION == 2
    assert tuple(payload["circuits"]) == ("network",)
    assert payload["circuits"]["network"]["schema_version"] == 1
    assert payload["circuits"]["network"]["data_type"] == "CircuitData"
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(payload["circuits"]))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(payload["circuits"]))

    loaded = mw.Result.load(path, scene=mw.Scene(device="cpu"))
    _assert_circuit_data_equal(loaded.circuit("network"), data)
    assert data.node_voltages.requires_grad


def test_result_rejects_superseded_v1_and_v2_without_circuits(tmp_path):
    path = tmp_path / "result.pt"
    _result(circuits={"network": _data()}).save(path)
    payload = torch.load(path, weights_only=True)

    # v1 is superseded, not supported alongside v2: there is no backward-support path.
    payload["schema_version"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unsupported result checkpoint schema_version=1"):
        mw.Result.load(path, scene=mw.Scene(device="cpu"))

    payload["schema_version"] = 2
    payload.pop("circuits")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema v2 is missing required key: circuits"):
        mw.Result.load(path, scene=mw.Scene(device="cpu"))


def test_result_save_prevalidates_circuits_before_filesystem_side_effects(tmp_path):
    data = _data()
    data.diagnostics["unsafe"] = object()
    path = tmp_path / "nested" / "result.pt"

    with pytest.raises(TypeError, match="CircuitData.diagnostics.*object"):
        _result(circuits={"network": data}).save(path)

    assert path.parent.exists() is False

import pytest
import torch

import witwin.maxwell as mw


def _data(name="network"):
    return mw.CircuitData(
        circuit_name=name,
        times=torch.tensor([0.0, 1.0]),
        node_names=("0", "signal"),
        node_voltages=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        branch_names=("R1",),
        branch_currents=torch.tensor([[0.0], [0.02]]),
    )


def _result(*, circuits=None):
    return mw.Result(
        method="fdtd",
        scene=mw.Scene(device="cpu"),
        frequency=1.0e9,
        circuits=circuits,
    )


def test_result_circuit_accessor_preserves_identity_and_reports_stats():
    data = _data()
    result = _result(circuits={"network": data})

    assert result.circuit("network") is data
    assert result.circuits == {"network": data}
    assert result.stats()["num_circuits"] == 1
    with pytest.raises(KeyError, match="Choices:.*network"):
        result.circuit("missing")


def test_result_rejects_invalid_circuit_mappings_and_prevents_silent_save(tmp_path):
    data = _data()
    with pytest.raises(TypeError, match="circuits must be a mapping"):
        _result(circuits=(data,))
    with pytest.raises(TypeError, match="CircuitData instance"):
        _result(circuits={"network": object()})
    with pytest.raises(ValueError, match="does not match"):
        _result(circuits={"other": data})
    with pytest.raises(ValueError, match="Duplicate normalized circuit key"):
        _result(circuits={1: _data("1"), "1": _data("1")})

    result = _result(circuits={"network": data})
    with pytest.raises(NotImplementedError, match="silently omitting circuit data"):
        result.save(tmp_path / "result.pt")
    assert not (tmp_path / "result.pt").exists()
    with pytest.raises(NotImplementedError, match="silently omitting circuit data"):
        result.save_sharded(tmp_path / "sharded")
    assert not (tmp_path / "sharded").exists()

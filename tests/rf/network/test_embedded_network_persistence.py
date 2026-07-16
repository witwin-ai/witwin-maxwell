from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed.persistence import (
    FieldComponentArtifact,
    FieldShardArtifact,
)
from witwin.maxwell.rational import FitReport, NetworkFitReport
from witwin.maxwell.result import Result


_FREQUENCIES = (1.0e9, 2.0e9)


def _scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )


def _fit_report() -> FitReport:
    return FitReport(
        rms_error=1.0e-4,
        max_error=4.0e-4,
        relative_rms_error=2.0e-3,
        relative_max_error=5.0e-3,
        frequency_band=_FREQUENCIES,
        order=4,
        iterations=3,
        condition_numbers=(2.0, 3.0),
        warnings=("fit warning",),
    )


def _network_fit_report() -> NetworkFitReport:
    return NetworkFitReport(
        rms_error=2.0e-4,
        max_error=5.0e-4,
        relative_rms_error=3.0e-3,
        relative_max_error=6.0e-3,
        frequency_band=_FREQUENCIES,
        order=6,
        iterations=4,
        passivity_margin=0.02,
        port_count=2,
        delay_seconds=(0.0, 0.5e-9),
        delay_estimation_rank=2,
        delay_equation_count=3,
        delay_residual_seconds=1.0e-13,
        delay_phase_error_degrees=0.25,
        delay_reembedding_max_error=1.0e-3,
        warnings=("network fit warning",),
    )


def _embedded(
    name: str,
    fit_report: FitReport,
    *,
    metadata=None,
) -> mw.EmbeddedNetworkData:
    frequencies = torch.tensor(_FREQUENCIES, dtype=torch.float64)
    voltage = torch.tensor(
        [[1.0 + 0.1j, 0.8 - 0.2j], [0.4 + 0.3j, 0.2 + 0.5j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    current = torch.tensor(
        [[0.02 + 0.001j, 0.016 - 0.002j], [0.008 + 0.003j, 0.004 + 0.005j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    port_power = torch.tensor(
        [[0.02, 0.013], [0.004, -0.001]],
        dtype=torch.float64,
        requires_grad=True,
    )
    return mw.EmbeddedNetworkData(
        name=name,
        frequencies=frequencies,
        port_names=("left", "right"),
        voltage=voltage,
        current=current,
        port_power=port_power,
        absorbed_power=torch.tensor(
            [0.024, 0.012], dtype=torch.float64, requires_grad=True
        ),
        generated_power=torch.tensor(
            [0.0, 0.001], dtype=torch.float64, requires_grad=True
        ),
        state_norm=torch.tensor(0.75, dtype=torch.float64, requires_grad=True),
        model_id=f"{name}-model",
        fit_report=fit_report,
        runtime_warnings=("runtime warning",),
        metadata={
            "calibration": torch.tensor(
                [1.0, 2.0], dtype=torch.float64, requires_grad=True
            ),
            "labels": ("left", "right"),
        }
        if metadata is None
        else metadata,
    )


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif hasattr(value, "__dict__"):
        yield from _iter_tensors(vars(value))


def _result(*networks: mw.EmbeddedNetworkData, sharded: bool = False) -> Result:
    solver = None
    if sharded:
        artifact = FieldShardArtifact(
            rank=0,
            device="cpu",
            frequencies=_FREQUENCIES,
            components=(
                FieldComponentArtifact(
                    name="Ex",
                    tensor=torch.zeros((2, 2, 2, 2), dtype=torch.float32),
                    global_x_slice=(0, 2),
                ),
            ),
        )
        solver = SimpleNamespace(export_field_shards=lambda: (artifact,))
    return Result(
        method="fdtd",
        scene=_scene(),
        frequencies=_FREQUENCIES,
        solver=solver,
        embedded_networks={network.name: network for network in networks},
    )


def _assert_restored(
    restored: mw.EmbeddedNetworkData,
    reference: mw.EmbeddedNetworkData,
) -> None:
    assert restored.name == reference.name
    assert restored.port_names == reference.port_names
    assert restored.model_id == reference.model_id
    assert restored.runtime_warnings == reference.runtime_warnings
    assert restored.metadata["labels"] == reference.metadata["labels"]
    assert type(restored.fit_report) is type(reference.fit_report)
    assert restored.fit_report == reference.fit_report
    for field_name in (
        "frequencies",
        "voltage",
        "current",
        "port_power",
        "absorbed_power",
        "generated_power",
        "state_norm",
    ):
        torch.testing.assert_close(
            getattr(restored, field_name),
            getattr(reference, field_name),
        )
    torch.testing.assert_close(
        restored.metadata["calibration"], reference.metadata["calibration"]
    )
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(restored))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(restored))


def test_result_round_trip_preserves_detached_typed_embedded_networks(tmp_path):
    plain = _embedded("plain", _fit_report())
    multiport = _embedded("multiport", _network_fit_report())
    path = tmp_path / "embedded-networks.pt"

    _result(plain, multiport).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    loaded = Result.load(path, scene=_scene(), map_location=torch.device("cpu"))

    assert tuple(payload["embedded_networks"]) == ("plain", "multiport")
    assert payload["embedded_networks"]["plain"]["data_type"] == (
        "EmbeddedNetworkData"
    )
    assert payload["embedded_networks"]["plain"]["fit_report"]["data_type"] == (
        "FitReport"
    )
    assert payload["embedded_networks"]["multiport"]["fit_report"]["data_type"] == (
        "NetworkFitReport"
    )
    assert tuple(loaded.embedded_networks) == ("plain", "multiport")
    _assert_restored(loaded.embedded_network("plain"), plain)
    _assert_restored(loaded.embedded_network("multiport"), multiport)


def test_sharded_result_round_trip_preserves_embedded_network_data(tmp_path):
    embedded = _embedded("two-port", _network_fit_report())
    directory = tmp_path / "sharded"

    _result(embedded, sharded=True).save_sharded(directory)
    payload = torch.load(directory / "result.pt", weights_only=True)
    loaded = Result.load_sharded(
        directory,
        scene=_scene(),
        gather_fields=False,
        map_location="cpu",
    )

    assert payload["embedded_networks"]["two-port"]["voltage"].device.type == "cpu"
    assert not payload["embedded_networks"]["two-port"]["voltage"].requires_grad
    assert loaded.fields == {}
    _assert_restored(loaded.embedded_network("two-port"), embedded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(schema_version=99), "schema_version"),
        (lambda payload: payload.update(data_type="OtherData"), "data_type"),
        (
            lambda payload: payload["fit_report"].update(schema_version=99),
            "fit report schema_version",
        ),
        (
            lambda payload: payload["fit_report"].update(data_type="OtherReport"),
            "fit report has an invalid data_type",
        ),
        (
            lambda payload: payload["fit_report"].update(values=[]),
            "fit report values must be a mapping",
        ),
    ),
)
def test_result_load_rejects_malformed_embedded_network_payload(
    tmp_path,
    mutation,
    message,
):
    path = tmp_path / "malformed.pt"
    _result(_embedded("network", _network_fit_report())).save(path)
    payload = torch.load(path, weights_only=True)
    mutation(payload["embedded_networks"]["network"])
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        Result.load(path, scene=_scene())


def test_result_load_accepts_version_one_snapshot_without_embedded_networks(tmp_path):
    path = tmp_path / "existing-result.pt"
    _result().save(path)
    payload = torch.load(path, weights_only=True)
    del payload["embedded_networks"]
    torch.save(payload, path)

    loaded = Result.load(path, scene=_scene())

    assert loaded.embedded_networks == {}


@pytest.mark.parametrize("sharded", (False, True))
def test_result_save_rejects_unsafe_embedded_network_metadata(tmp_path, sharded):
    embedded = replace(
        _embedded("unsafe", _fit_report()),
        metadata={"unsafe": object()},
    )
    result = _result(embedded, sharded=sharded)

    with pytest.raises(TypeError, match=r"embedded_networks\['unsafe'\]\.metadata"):
        if sharded:
            result.save_sharded(tmp_path / "sharded-unsafe")
        else:
            result.save(tmp_path / "unsafe.pt")

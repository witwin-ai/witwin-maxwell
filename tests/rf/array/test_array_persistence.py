import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.array import ARRAY_PERSISTENCE_SCHEMA_VERSION


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif is_dataclass(value):
        for item in fields(value):
            yield from _iter_tensors(getattr(value, item.name))


def _basis():
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    frequencies = torch.tensor([1.0e9, 1.5e9], dtype=real_dtype)
    scattering = torch.tensor(
        [
            [[0.10 + 0.02j, 0.03 - 0.01j], [0.03 - 0.01j, -0.08 + 0.01j]],
            [[0.12 - 0.01j, 0.02 + 0.02j], [0.02 + 0.02j, -0.06 - 0.02j]],
        ],
        dtype=complex_dtype,
        requires_grad=True,
    )
    z0 = torch.tensor(
        [[50.0 + 4.0j, 60.0 - 3.0j], [51.0 + 2.0j, 61.0 - 1.0j]],
        dtype=complex_dtype,
        requires_grad=True,
    )
    calibration = torch.tensor([0.25, 0.75], dtype=real_dtype, requires_grad=True)
    network = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=z0,
        port_names=("left", "right"),
        valid_columns=torch.tensor([True, True]),
        metadata={
            "calibration": calibration,
            "provenance": {"reference": "network/reference", "revision": 7},
        },
    )

    theta_vector = torch.linspace(0.0, math.pi, 7, dtype=real_dtype)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, 13, dtype=real_dtype)
    theta, phi = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    element_1 = torch.sin(theta).to(complex_dtype) * torch.exp(0.2j * torch.cos(phi))
    element_2 = torch.sin(theta).to(complex_dtype) * torch.exp(-0.3j * torch.sin(phi))
    e_theta = (
        torch.stack((element_1, element_2), dim=0)[None]
        .expand(2, -1, -1, -1)
        .clone()
        .detach()
        .requires_grad_(True)
    )
    e_phi = (0.25j * torch.flip(e_theta.detach(), dims=(-1,))).requires_grad_(True)
    observation_radius = torch.tensor([1.0, 1.25], dtype=real_dtype, requires_grad=True)
    wave_impedance = torch.full(
        theta.shape,
        376.730313668,
        dtype=real_dtype,
        requires_grad=True,
    )
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies,
        port_names=("left", "right"),
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        phase_center=torch.tensor([0.1, -0.2, 0.3], dtype=real_dtype),
        frame=torch.eye(3, dtype=real_dtype),
        observation_radius=observation_radius,
        wave_impedance=wave_impedance,
        polarization_basis=mw.Ludwig3(reference_angle=0.37),
        phase_center_source="array_aabb",
    )
    quality = torch.tensor(0.995, dtype=real_dtype, requires_grad=True)
    radiated_power_matrix = torch.tensor(
        [
            [[0.8 + 0.0j, 0.1 + 0.2j], [0.1 - 0.2j, 0.6 + 0.0j]],
            [[0.7 + 0.0j, -0.05 + 0.1j], [-0.05 - 0.1j, 0.5 + 0.0j]],
        ],
        dtype=complex_dtype,
        requires_grad=True,
    )
    basis = mw.ArrayBasisData(
        network=network,
        embedded_patterns=patterns,
        fingerprint="persistence-two-port",
        metadata={
            "quality": quality,
            "provenance": {
                "source": "basis/source",
                "tags": ("fdtd", "embedded-elements"),
            },
        },
        radiated_power_matrix=radiated_power_matrix,
    )
    live_grad_tensors = (
        scattering,
        z0,
        calibration,
        e_theta,
        e_phi,
        observation_radius,
        wave_impedance,
        quality,
        radiated_power_matrix,
    )
    return basis, live_grad_tensors


def _assert_pattern_equal(actual, expected):
    for name in (
        "frequencies",
        "theta",
        "phi",
        "e_theta",
        "e_phi",
        "phase_center",
        "frame",
        "observation_radius",
        "wave_impedance",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    assert actual.port_names == expected.port_names
    assert type(actual.polarization_basis) is type(expected.polarization_basis)
    assert actual.polarization_basis.reference_angle == expected.polarization_basis.reference_angle
    for name in (
        "phase_center_source",
        "field_basis",
        "power_normalization",
        "phasor_convention",
        "power_wave_convention",
        "field_units",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def test_embedded_pattern_round_trip_preserves_contract_and_detaches(tmp_path):
    basis, live_grad_tensors = _basis()
    path = tmp_path / "nested" / "embedded-pattern.pt"

    basis.embedded_patterns.save(path)
    loaded = mw.EmbeddedElementPatternData.load(path, map_location=torch.device("cpu"))

    _assert_pattern_equal(loaded, basis.embedded_patterns)
    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == ARRAY_PERSISTENCE_SCHEMA_VERSION
    assert payload["data_type"] == "EmbeddedElementPatternData"
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(payload))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(payload))
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(loaded))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(loaded))
    assert all(tensor.requires_grad for tensor in live_grad_tensors)


def test_array_basis_round_trip_preserves_full_network_metadata_and_behavior(tmp_path):
    basis, live_grad_tensors = _basis()
    path = tmp_path / "nested" / "array-basis.pt"

    basis.save(path)
    loaded = mw.ArrayBasisData.load(path, map_location="cpu")

    assert loaded.fingerprint == basis.fingerprint
    assert loaded.metadata["provenance"] == basis.metadata["provenance"]
    torch.testing.assert_close(loaded.metadata["quality"], basis.metadata["quality"])
    for name in ("frequencies", "s", "z0", "valid_columns"):
        torch.testing.assert_close(getattr(loaded.network, name), getattr(basis.network, name))
    assert loaded.network.port_names == basis.network.port_names
    assert loaded.network.phasor_convention == basis.network.phasor_convention
    assert loaded.network.power_wave_convention == basis.network.power_wave_convention
    assert loaded.network.metadata["provenance"] == basis.network.metadata["provenance"]
    torch.testing.assert_close(
        loaded.network.metadata["calibration"], basis.network.metadata["calibration"]
    )
    _assert_pattern_equal(loaded.embedded_patterns, basis.embedded_patterns)
    torch.testing.assert_close(
        loaded.radiated_power_matrix, basis.radiated_power_matrix
    )

    weights = torch.tensor([0.6 + 0.1j, -0.2 + 0.7j], dtype=torch.complex128)
    expected_beam = basis.combine(weights)
    actual_beam = loaded.combine(weights)
    torch.testing.assert_close(actual_beam.network.b, expected_beam.network.b)
    torch.testing.assert_close(actual_beam.far_field.e_theta, expected_beam.far_field.e_theta)
    torch.testing.assert_close(actual_beam.antenna.p_rad, expected_beam.antenna.p_rad)

    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == ARRAY_PERSISTENCE_SCHEMA_VERSION
    assert payload["data_type"] == "ArrayBasisData"
    assert payload["network"]["data_type"] == "NetworkData"
    assert payload["embedded_patterns"]["data_type"] == "EmbeddedElementPatternData"
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(payload))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(payload))
    assert all(tensor.device.type == "cpu" for tensor in _iter_tensors(loaded))
    assert all(not tensor.requires_grad for tensor in _iter_tensors(loaded))
    assert all(tensor.requires_grad for tensor in live_grad_tensors)


@pytest.mark.parametrize(
    ("data_type", "loader"),
    [
        ("EmbeddedElementPatternData", mw.EmbeddedElementPatternData.load),
        ("ArrayBasisData", mw.ArrayBasisData.load),
    ],
)
def test_array_persistence_rejects_unknown_top_level_schema(tmp_path, data_type, loader):
    path = tmp_path / "future.pt"
    torch.save(
        {"schema_version": ARRAY_PERSISTENCE_SCHEMA_VERSION + 1, "data_type": data_type},
        path,
    )

    with pytest.raises(ValueError, match="Unsupported .* schema_version"):
        loader(path)


@pytest.mark.parametrize("nested_key", ["network", "embedded_patterns"])
def test_array_basis_load_validates_nested_schema_versions(tmp_path, nested_key):
    basis, _ = _basis()
    path = tmp_path / "basis.pt"
    basis.save(path)
    payload = torch.load(path, weights_only=True)
    payload[nested_key]["schema_version"] += 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="Unsupported .* schema_version"):
        mw.ArrayBasisData.load(path)


def test_array_basis_save_reuses_safe_network_metadata_contract(tmp_path):
    basis, _ = _basis()
    basis.metadata["unsafe"] = tmp_path

    with pytest.raises(TypeError, match="unsupported persistence type"):
        basis.save(tmp_path / "unsafe.pt")


def test_array_basis_load_rejects_missing_power_operator_key(tmp_path):
    basis, _ = _basis()
    path = tmp_path / "basis.pt"
    basis.save(path)
    payload = torch.load(path, weights_only=True)
    del payload["radiated_power_matrix"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="missing radiated_power_matrix"):
        mw.ArrayBasisData.load(path)

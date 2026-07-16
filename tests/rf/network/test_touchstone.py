import math

import pytest
import torch

from witwin.maxwell.network import NetworkData
from witwin.maxwell.touchstone import write_touchstone


def _network(
    s,
    *,
    frequencies=None,
    z0=50.0,
    valid_columns=None,
) -> NetworkData:
    scattering = torch.as_tensor(s, dtype=torch.complex128)
    frequency_count, port_count, _ = scattering.shape
    if frequencies is None:
        frequencies = torch.arange(1, frequency_count + 1, dtype=torch.float64) * 1.0e9
    return NetworkData(
        frequencies=torch.as_tensor(frequencies, dtype=torch.float64),
        s=scattering,
        z0=z0,
        port_names=tuple(f"p{index + 1}" for index in range(port_count)),
        valid_columns=valid_columns,
    )


def _touchstone_lines(path):
    return [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if not line.startswith("! Port[")
    ]


@pytest.mark.parametrize(
    "frequency_unit, expected_frequency",
    (("hz", 1.0e9), ("khz", 1.0e6), ("mhz", 1.0e3), ("ghz", 1.0)),
)
def test_write_touchstone_one_port_ri_and_frequency_units(
    tmp_path,
    frequency_unit,
    expected_frequency,
):
    network = _network([[[1.25 - 0.5j]]])
    path = tmp_path / "one.s1p"

    write_touchstone(network, path, frequency_unit=frequency_unit)

    lines = _touchstone_lines(path)
    assert lines[0] == f"# {frequency_unit.upper()} S RI R 50"
    assert "[Version]" not in lines[0]
    values = [float(value) for value in lines[1].split()]
    assert values == [expected_frequency, 1.25, -0.5]


def test_network_data_exposes_touchstone_export_method(tmp_path):
    network = _network([[[0.25 + 0.5j]]])
    path = tmp_path / "method.s1p"

    returned = network.to_touchstone(path)

    assert returned == path
    assert path.is_file()


def test_write_touchstone_two_port_uses_historical_column_order(tmp_path):
    network = _network(
        [[[11.0 + 0.1j, 12.0 + 0.2j], [21.0 + 0.3j, 22.0 + 0.4j]]],
        z0=(50.0, 75.0),
    )
    path = tmp_path / "two.ts"

    write_touchstone(network, path)

    lines = _touchstone_lines(path)
    assert lines[:8] == [
        "[Version] 2.0",
        "# HZ S RI R 50",
        "[Number of Ports] 2",
        "[Two-Port Data Order] 21_12",
        "[Number of Frequencies] 1",
        "[Reference] 50 75",
        "[Matrix Format] Full",
        "[Network Data]",
    ]
    values = [float(value) for value in lines[8].split()]
    assert values == [
        1.0e9,
        11.0,
        0.1,
        21.0,
        0.3,
        12.0,
        0.2,
        22.0,
        0.4,
    ]
    assert lines[-1] == "[End]"


def test_write_touchstone_three_port_is_row_major_with_one_frequency_token(tmp_path):
    matrix = torch.tensor(
        [
            [11.0 + 1.0j, 12.0 + 2.0j, 13.0 + 3.0j],
            [21.0 + 4.0j, 22.0 + 5.0j, 23.0 + 6.0j],
            [31.0 + 7.0j, 32.0 + 8.0j, 33.0 + 9.0j],
        ],
        dtype=torch.complex128,
    )
    path = tmp_path / "three.s3p"

    write_touchstone(_network(matrix.unsqueeze(0)), path)

    lines = _touchstone_lines(path)
    assert len(lines) == 4
    assert [float(value) for value in lines[1].split()] == [
        1.0e9,
        11.0,
        1.0,
        12.0,
        2.0,
        13.0,
        3.0,
    ]
    assert [float(value) for value in lines[2].split()] == [21.0, 4.0, 22.0, 5.0, 23.0, 6.0]
    assert [float(value) for value in lines[3].split()] == [31.0, 7.0, 32.0, 8.0, 33.0, 9.0]


@pytest.mark.parametrize(
    "data_format, expected_first, expected_second",
    (
        ("ri", 3.0, 4.0),
        ("ma", 5.0, math.degrees(math.atan2(4.0, 3.0))),
        ("db", 20.0 * math.log10(5.0), math.degrees(math.atan2(4.0, 3.0))),
    ),
)
def test_write_touchstone_formats_complex_pairs(
    tmp_path,
    data_format,
    expected_first,
    expected_second,
):
    path = tmp_path / "format.s1p"

    write_touchstone(_network([[[3.0 + 4.0j]]]), path, format=data_format)

    lines = _touchstone_lines(path)
    assert lines[0] == f"# HZ S {data_format.upper()} R 50"
    _, first, second = (float(value) for value in lines[1].split())
    assert first == pytest.approx(expected_first)
    assert second == pytest.approx(expected_second)


def test_write_touchstone_explicit_version_two_supports_per_port_real_z0(tmp_path):
    network = _network(
        torch.eye(3, dtype=torch.complex128).unsqueeze(0),
        z0=(25.0, 50.0, 75.0),
    )
    path = tmp_path / "explicit.s3p"

    write_touchstone(network, path, version="2.0")

    text = "\n".join(_touchstone_lines(path)) + "\n"
    assert text.startswith("[Version] 2.0\n")
    assert "[Reference] 25 50 75\n" in text
    assert text.endswith("[End]\n")


def test_write_touchstone_version_one_rejects_per_port_z0(tmp_path):
    network = _network(
        torch.eye(2, dtype=torch.complex128).unsqueeze(0),
        z0=(50.0, 75.0),
    )

    with pytest.raises(ValueError, match="1.0 requires one uniform"):
        write_touchstone(network, tmp_path / "network.s2p", version="1.0")


@pytest.mark.parametrize(
    "z0, match",
    (
        (50.0 + 1.0j, "complex reference impedances"),
        (torch.tensor([[50.0], [51.0]]), "frequency-invariant"),
    ),
)
def test_write_touchstone_rejects_unsupported_z0(tmp_path, z0, match):
    network = _network(
        [[[0.1 + 0.2j]], [[0.2 + 0.3j]]],
        z0=z0,
    )

    with pytest.raises(ValueError, match=match):
        write_touchstone(network, tmp_path / "network.s1p")


def test_write_touchstone_rejects_incomplete_or_unsorted_network(tmp_path):
    incomplete = _network(
        torch.eye(2, dtype=torch.complex128).unsqueeze(0),
        valid_columns=(True, False),
    )
    unsorted = _network(
        [[[0.1 + 0.0j]], [[0.2 + 0.0j]]],
        frequencies=(2.0e9, 1.0e9),
    )

    with pytest.raises(ValueError, match="complete excitation columns"):
        write_touchstone(incomplete, tmp_path / "incomplete.s2p")
    with pytest.raises(ValueError, match="strictly increasing"):
        write_touchstone(unsorted, tmp_path / "unsorted.s1p")


@pytest.mark.parametrize("suffix", (".s1p", ".txt"))
def test_write_touchstone_rejects_mismatched_or_unknown_suffix(tmp_path, suffix):
    network = _network(torch.eye(2, dtype=torch.complex128).unsqueeze(0))

    with pytest.raises(ValueError, match="suffix|path must end"):
        write_touchstone(network, tmp_path / f"network{suffix}")


def test_write_touchstone_supports_four_ports(tmp_path):
    network = _network(torch.eye(4, dtype=torch.complex128).unsqueeze(0))
    path = tmp_path / "network.s4p"

    write_touchstone(network, path)

    assert path.is_file()
    assert len(_touchstone_lines(path)) == 5


def test_write_touchstone_detaches_without_mutating_source_graph(tmp_path):
    frequencies = torch.tensor([1.0e9], dtype=torch.float64, requires_grad=True)
    scattering = torch.tensor([[[0.25 + 0.5j]]], dtype=torch.complex128, requires_grad=True)
    network = NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("p1",),
    )

    write_touchstone(network, tmp_path / "detached.s1p")

    assert frequencies.requires_grad
    assert scattering.requires_grad

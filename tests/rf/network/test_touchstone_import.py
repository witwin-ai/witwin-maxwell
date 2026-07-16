import cmath
from pathlib import Path

import pytest
import torch

import witwin.maxwell as mw


FIXTURES = Path(__file__).parent / "fixtures"


def _network(port_count: int) -> mw.NetworkData:
    generator = torch.Generator().manual_seed(100 + port_count)
    real = torch.rand((3, port_count, port_count), generator=generator, dtype=torch.float64)
    imag = torch.rand((3, port_count, port_count), generator=generator, dtype=torch.float64)
    return mw.NetworkData(
        frequencies=torch.tensor((1.0e9, 1.5e9, 2.0e9), dtype=torch.float64),
        s=0.2 * torch.complex(real, imag),
        z0=torch.linspace(45.0, 75.0, port_count, dtype=torch.float64),
        port_names=tuple(str(index + 1) for index in range(port_count)),
    )


def test_read_one_port_v1_ri_golden():
    network = mw.NetworkData.from_touchstone(FIXTURES / "one_port_ri.s1p")

    torch.testing.assert_close(
        network.frequencies,
        torch.tensor((1.0e9, 2.0e9), dtype=torch.float64),
    )
    torch.testing.assert_close(
        network.s[:, 0, 0],
        torch.tensor((0.25 - 0.5j, -0.1 + 0.2j), dtype=torch.complex128),
    )
    torch.testing.assert_close(
        network.z0.real,
        torch.full((2, 1), 75.0, dtype=torch.float64),
    )
    assert network.port_names == ("1",)
    assert network.metadata["touchstone"]["comments"] == ("! one-port RI fixture",)


def test_read_two_port_v2_alternate_order_and_per_port_z0():
    network = mw.NetworkData.from_touchstone(FIXTURES / "two_port_ma.ts")

    expected = torch.tensor(
        [
            [0.1, cmath.rect(0.8, -torch.pi / 6)],
            [cmath.rect(0.7, torch.pi / 9), -0.2],
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(network.s[0], expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        network.z0,
        torch.tensor([[50.0, 75.0]], dtype=torch.complex128),
    )
    assert network.port_names == ("input", "output")
    assert network.metadata["touchstone"]["two_port_data_order"] == "12_21"


def test_read_four_port_v1_row_major_golden():
    network = mw.NetworkData.from_touchstone(FIXTURES / "four_port_db.s4p")

    assert network.s.shape == (1, 4, 4)
    torch.testing.assert_close(
        network.s[0, 0, 0],
        torch.tensor(0.1 + 0j, dtype=torch.complex128),
    )
    torch.testing.assert_close(
        network.s[0, 0, 1],
        torch.tensor(cmath.rect(0.01, torch.pi / 18), dtype=torch.complex128),
    )
    torch.testing.assert_close(
        network.s[0, 3, 3],
        torch.tensor(10 ** (-18 / 20), dtype=torch.complex128),
    )


@pytest.mark.parametrize("port_count", (1, 2, 4))
@pytest.mark.parametrize("data_format", ("ri", "ma", "db"))
def test_touchstone_round_trip_complex128_below_1e_10(tmp_path, port_count, data_format):
    source = _network(port_count)
    path = tmp_path / f"roundtrip.s{port_count}p"

    source.to_touchstone(path, format=data_format, version="2.0")
    loaded = mw.NetworkData.from_touchstone(path)

    torch.testing.assert_close(loaded.frequencies, source.frequencies, rtol=0.0, atol=0.0)
    torch.testing.assert_close(loaded.s, source.s, rtol=0.0, atol=1.0e-10)
    torch.testing.assert_close(loaded.z0, source.z0.to(torch.complex128), rtol=0.0, atol=0.0)
    assert loaded.port_names == source.port_names


@pytest.mark.parametrize(
    "parameter, version, value, expected_s",
    (
        ("z", "1.0", 2.0, 1.0 / 3.0),
        ("z", "2.0", 100.0, 1.0 / 3.0),
        ("y", "1.0", 1.0, 0.0),
        ("y", "2.0", 0.02, 0.0),
    ),
)
def test_read_z_y_respects_version_one_normalization(
    tmp_path,
    parameter,
    version,
    value,
    expected_s,
):
    path = tmp_path / ("network.s1p" if version == "1.0" else "network.ts")
    if version == "1.0":
        text = f"# Hz {parameter.upper()} RI R 50\n1 {value} 0\n"
    else:
        text = (
            "[Version] 2.0\n"
            f"# Hz {parameter.upper()} RI R 50\n"
            "[Number of Ports] 1\n"
            "[Number of Frequencies] 1\n"
            "[Network Data]\n"
            f"1 {value} 0\n"
            "[End]\n"
        )
    path.write_text(text, encoding="ascii")

    network = mw.NetworkData.from_touchstone(path)

    torch.testing.assert_close(
        network.s[0, 0, 0],
        torch.tensor(expected_s, dtype=torch.complex128),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert network.metadata["touchstone"]["parameter"] == parameter


def test_write_and_read_z_y_round_trip(tmp_path):
    source = _network(2)
    for parameter in ("z", "y"):
        path = tmp_path / f"network_{parameter}.ts"
        source.to_touchstone(path, version="2.0", parameter=parameter)
        loaded = mw.NetworkData.from_touchstone(path)
        torch.testing.assert_close(loaded.s, source.s, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize(
    "body, match, line",
    (
        ("# Hz S RI R 50\n1 nope 0\n", "invalid numeric token", 2),
        ("# Hz S RI R 50\n1 0.1\n", "incomplete network data", 2),
        ("# Hz S RI R 50\n1 nan 0\n", "NaN", 2),
        ("# Hz S RI R 50\n2 0 0\n1 0 0\n", "strictly increasing", 3),
        ("# Hz S RI R 0\n1 0 0\n", "reference resistance", 1),
    ),
)
def test_parse_errors_include_source_line(tmp_path, body, match, line):
    path = tmp_path / "bad.s1p"
    path.write_text(body, encoding="ascii")

    with pytest.raises(mw.TouchstoneParseError, match=match) as caught:
        mw.NetworkData.from_touchstone(path)

    assert caught.value.line_number == line
    assert f"line {line}" in str(caught.value)


def test_parser_rejects_suffix_port_mismatch(tmp_path):
    path = tmp_path / "bad.s2p"
    path.write_text(
        "[Version] 2.0\n"
        "# Hz S RI R 50\n"
        "[Number of Ports] 1\n"
        "[Number of Frequencies] 1\n"
        "[Network Data]\n"
        "1 0 0\n"
        "[End]\n",
        encoding="ascii",
    )

    with pytest.raises(mw.TouchstoneParseError, match="suffix declares 2 ports"):
        mw.NetworkData.from_touchstone(path)


def test_parser_expands_lower_matrix_symmetrically(tmp_path):
    path = tmp_path / "lower.ts"
    path.write_text(
        "[Version] 2.0\n"
        "# Hz S RI R 50\n"
        "[Number of Ports] 3\n"
        "[Number of Frequencies] 1\n"
        "[Matrix Format] Lower\n"
        "[Network Data]\n"
        "1 0.1 0 0.2 0 0.3 0 0.4 0 0.5 0 0.6 0\n"
        "[End]\n",
        encoding="ascii",
    )

    network = mw.NetworkData.from_touchstone(path)

    torch.testing.assert_close(network.s, network.s.transpose(-2, -1))
    torch.testing.assert_close(
        network.s[0, 2, 1],
        torch.tensor(0.5 + 0j, dtype=torch.complex128),
    )


def test_parser_expands_upper_matrix_symmetrically(tmp_path):
    path = tmp_path / "upper.ts"
    path.write_text(
        "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 3\n"
        "[Number of Frequencies] 1\n[Matrix Format] Upper\n[Network Data]\n"
        "1 0.1 0 0.2 0 0.3 0 0.4 0 0.5 0 0.6 0\n[End]\n",
        encoding="ascii",
    )

    network = mw.NetworkData.from_touchstone(path)

    torch.testing.assert_close(network.s, network.s.transpose(-2, -1))
    torch.testing.assert_close(
        network.s[0, 1, 2],
        torch.tensor(0.5 + 0j, dtype=torch.complex128),
    )


def test_parser_uses_historical_two_port_order_for_version_one(tmp_path):
    path = tmp_path / "historical.s2p"
    path.write_text(
        "# Hz S RI R 50\n1 11 0 21 0 12 0 22 0\n",
        encoding="ascii",
    )

    network = mw.NetworkData.from_touchstone(path)

    torch.testing.assert_close(
        network.s[0],
        torch.tensor([[11.0, 12.0], [21.0, 22.0]], dtype=torch.complex128),
    )


def test_from_touchstone_places_tensors_on_requested_device():
    network = mw.NetworkData.from_touchstone(FIXTURES / "one_port_ri.s1p", device="cpu")

    assert network.frequencies.device.type == "cpu"
    assert network.s.device.type == "cpu"
    assert network.s.dtype == torch.complex128


def test_parser_preserves_warning_metadata_for_ignored_option_line(tmp_path):
    path = tmp_path / "warning.s1p"
    path.write_text(
        "# Hz S RI R 50\n# GHz S MA R 75\n1 0.25 0\n",
        encoding="ascii",
    )

    network = mw.NetworkData.from_touchstone(path)

    assert network.metadata["touchstone"]["parser_warnings"] == (
        "line 2: ignored additional option line",
    )


def test_named_ports_survive_export_import_round_trip(tmp_path):
    source = mw.NetworkData(
        frequencies=torch.tensor([1.0e9], dtype=torch.float64),
        s=torch.zeros((1, 2, 2), dtype=torch.complex128),
        z0=(50.0, 75.0),
        port_names=("input", "output"),
    )
    path = tmp_path / "named.ts"

    source.to_touchstone(path, version="2.0")
    loaded = mw.NetworkData.from_touchstone(path)

    assert loaded.port_names == source.port_names


def test_network_data_rejects_port_name_boundary_whitespace():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        mw.NetworkData(
            frequencies=torch.tensor([1.0], dtype=torch.float64),
            s=torch.zeros((1, 1, 1), dtype=torch.complex128),
            z0=50.0,
            port_names=(" input ",),
        )


def test_version_one_five_port_rows_wrap_at_four_pairs(tmp_path):
    source = _network(5)
    uniform = mw.NetworkData(
        frequencies=source.frequencies,
        s=source.s,
        z0=50.0,
        port_names=source.port_names,
    )
    path = tmp_path / "wrapped.s5p"

    uniform.to_touchstone(path, version="1.0")
    loaded = mw.NetworkData.from_touchstone(path)

    data_lines = [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith(("!", "#"))
    ]
    assert all(len(line.split()) <= 9 for line in data_lines)
    torch.testing.assert_close(loaded.s, uniform.s, rtol=0.0, atol=1.0e-12)


def test_parser_accepts_zero_frequency(tmp_path):
    path = tmp_path / "dc.s1p"
    path.write_text("# Hz S RI R 50\n0 0.25 0\n1 0.5 0\n", encoding="ascii")

    network = mw.NetworkData.from_touchstone(path)

    torch.testing.assert_close(network.frequencies, torch.tensor([0.0, 1.0], dtype=torch.float64))


@pytest.mark.parametrize(
    "body,line",
    (
        ("# Hz S RI R 50\n1 0 0 0 0\n0 0 0 0 0\n", 2),
        ("# Hz S RI R 50\n1 0 0 0 0\n0 0 0 0 0 0 0\n", 2),
        ("# Hz S RI R 50\n1 0 0 0 0 0 0 0 0 0 0\n", 2),
    ),
)
def test_version_one_rejects_malformed_line_layout(tmp_path, body, line):
    path = tmp_path / "bad.s2p"
    path.write_text(body, encoding="ascii")

    with pytest.raises(mw.TouchstoneParseError, match="require exactly") as caught:
        mw.NetworkData.from_touchstone(path)

    assert caught.value.line_number == line


def test_parser_rejects_non_ascii_and_noncanonical_keyword_spacing(tmp_path):
    non_ascii = tmp_path / "non_ascii.s1p"
    non_ascii.write_bytes("! café\n# Hz S RI R 50\n1 0 0\n".encode("utf-8"))
    bad_keyword = tmp_path / "bad_keyword.ts"
    bad_keyword.write_text(
        "[Version] 2.0\n# Hz S RI R 50\n[ Number of Ports ] 1\n",
        encoding="ascii",
    )

    with pytest.raises(mw.TouchstoneParseError, match="ASCII") as caught:
        mw.NetworkData.from_touchstone(non_ascii)
    assert caught.value.line_number == 1
    with pytest.raises(mw.TouchstoneParseError, match="expected.*keyword"):
        mw.NetworkData.from_touchstone(bad_keyword)


def test_db_zero_export_uses_finite_numeric_token(tmp_path):
    source = mw.NetworkData(
        frequencies=torch.tensor([1.0], dtype=torch.float64),
        s=torch.zeros((1, 1, 1), dtype=torch.complex128),
        z0=50.0,
        port_names=("load",),
    )
    path = tmp_path / "zero.s1p"

    source.to_touchstone(path, format="db")
    text = path.read_text(encoding="ascii").lower()
    loaded = mw.NetworkData.from_touchstone(path)

    assert "inf" not in text
    assert torch.abs(loaded.s[0, 0, 0]) < 1.0e-10


def test_duplicate_resolved_port_name_reports_second_comment_line(tmp_path):
    path = tmp_path / "duplicate.s2p"
    path.write_text(
        "! Port[1] = same\n"
        "! Port[2] = same\n"
        "# Hz S RI R 50\n"
        "1 0 0 0 0 0 0 0 0\n",
        encoding="ascii",
    )

    with pytest.raises(mw.TouchstoneParseError, match="duplicate resolved") as caught:
        mw.NetworkData.from_touchstone(path)

    assert caught.value.line_number == 2

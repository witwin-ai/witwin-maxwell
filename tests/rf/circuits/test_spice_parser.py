from pathlib import Path

import pytest
import torch

import witwin.maxwell as mw


SUPPORTED_NETLIST = """
* supported linear corpus
.param rval=2k cval=3p lval=4n
Vdrive in 0 PULSE(0 1 1n 2n 3n 4n 10n)
Isense aux 0 SIN(0 1m 2meg)
Rload in out {rval}
Cload out 0 {cval}
Lprimary in aux {lval}
Lsecondary coupled 0 5n
Kcore Lprimary Lsecondary 0.8
Ebuffer controlled 0 out 0 2
Gfeedback out 0 controlled 0 1m
Fmirror mirror 0 Vdrive 3
Htrans trans 0 Vdrive 4
Ipwl coupled 0 PWL(0 0 1n 1m 2n 0)
.ic V(out)=0.25
.end
"""


def test_supported_corpus_parses_with_fixed_suffix_semantics_and_order():
    circuit = mw.parse_spice(SUPPORTED_NETLIST, name="supported")

    assert tuple(node.name for node in circuit.nodes) == (
        "0",
        "in",
        "aux",
        "out",
        "coupled",
        "controlled",
        "mirror",
        "trans",
    )
    assert tuple(device.name for device in circuit.devices) == (
        "Vdrive",
        "Isense",
        "Rload",
        "Cload",
        "Lprimary",
        "Lsecondary",
        "Kcore",
        "Ebuffer",
        "Gfeedback",
        "Fmirror",
        "Htrans",
        "Ipwl",
    )
    torch.testing.assert_close(circuit.devices[1].waveform.frequency, torch.tensor(2.0e6))
    torch.testing.assert_close(circuit.devices[2].resistance, torch.tensor(2.0e3))
    torch.testing.assert_close(circuit.devices[3].capacitance, torch.tensor(3.0e-12))
    torch.testing.assert_close(circuit.devices[4].inductance, torch.tensor(4.0e-9))
    assert circuit.initial_conditions["out"][1]


def test_canonical_serialization_round_trip_is_stable():
    first = mw.parse_spice(SUPPORTED_NETLIST, name="supported")
    serialized = first.to_spice()
    second = mw.parse_spice(serialized, name="supported")

    assert second.to_spice() == serialized
    assert tuple(node.name for node in second.nodes) == tuple(node.name for node in first.nodes)
    assert tuple(type(device) for device in second.devices) == tuple(type(device) for device in first.devices)
    assert tuple(device.name for device in second.devices) == tuple(device.name for device in first.devices)


def test_parameter_override_preserves_tensor_identity_and_gradient():
    resistance = torch.tensor(75.0, dtype=torch.float64, requires_grad=True)
    circuit = mw.parse_spice(
        ".param rval=50\nR1 in 0 {rval}\n.end",
        parameters={"rval": resistance},
    )

    assert circuit.parameters["rval"] is resistance
    assert circuit.devices[0].resistance is resistance


def test_subcircuits_flatten_with_hierarchical_names_and_local_nodes():
    circuit = mw.parse_spice(
        """
        .subckt pair p n params: rv=10
        R1 p internal {rv}
        C1 internal n 1p
        .ends pair
        Xleft in 0 pair rv=20
        Xright out 0 pair rv=30
        Rlink in out 5
        .end
        """,
        name="flattened",
    )

    assert tuple(device.name for device in circuit.devices) == (
        "RXleft.R1",
        "CXleft.C1",
        "RXright.R1",
        "CXright.C1",
        "Rlink",
    )
    assert "Xleft:internal" in tuple(node.name for node in circuit.nodes)
    assert "Xright:internal" in tuple(node.name for node in circuit.nodes)
    torch.testing.assert_close(circuit.devices[0].resistance, torch.tensor(20.0))
    torch.testing.assert_close(circuit.devices[2].resistance, torch.tensor(30.0))
    flattened = circuit.to_spice()
    assert mw.parse_spice(flattened, name="flattened").to_spice() == flattened


def test_file_include_is_sandboxed_and_resolves_inside_root(tmp_path: Path):
    root = tmp_path / "netlists"
    root.mkdir()
    (root / "load.inc").write_text("Rload out 0 50\n", encoding="utf-8")
    main = root / "main.cir"
    main.write_text("V1 out 0 1\n.include load.inc\n.end\n", encoding="utf-8")

    circuit = mw.Circuit.from_spice(main, name="included", include_root=root)

    assert tuple(device.name for device in circuit.devices) == ("V1", "Rload")


def test_file_include_rejects_parent_escape(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.inc").write_text("Rbad in 0 1\n", encoding="utf-8")
    main = root / "main.cir"
    main.write_text(".include ../outside.inc\n.end\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes.*sandbox"):
        mw.Circuit.from_spice(main, include_root=root)


def test_nested_include_cycle_is_a_hard_error(tmp_path: Path):
    first = tmp_path / "first.cir"
    second = tmp_path / "second.inc"
    first.write_text(".include second.inc\n.end\n", encoding="utf-8")
    second.write_text(".include first.cir\n", encoding="utf-8")

    with pytest.raises(ValueError, match="include cycle"):
        mw.Circuit.from_spice(first, include_root=tmp_path)


@pytest.mark.parametrize(
    "statement",
    (
        ".tran 1n 10n",
        ".ac dec 10 1 1meg",
        ".model switch sw",
        ".control",
        ".foo bar",
    ),
)
def test_unsupported_directives_are_hard_errors(statement):
    with pytest.raises(ValueError, match="Unsupported SPICE directive"):
        mw.parse_spice(f"{statement}\n.end")


@pytest.mark.parametrize("statement", ("D1 a 0 model", "Q1 a b 0 model", "M1 d g s b model", "S1 a 0 c 0 model"))
def test_unsupported_devices_are_hard_errors(statement):
    with pytest.raises(ValueError, match="Unsupported SPICE device"):
        mw.parse_spice(f"{statement}\n.end")


@pytest.mark.parametrize(
    "expression",
    (
        "__import__('os').system('echo unsafe')",
        "(1).__class__",
        "values[0]",
        "(lambda: 1)()",
        "[x for x in (1,)]",
    ),
)
def test_malicious_expressions_cannot_execute(expression, tmp_path: Path):
    marker = tmp_path / "owned"
    payload = expression.replace("unsafe", str(marker))
    with pytest.raises(ValueError):
        mw.parse_spice(f".param x={{{payload}}}\nR1 in 0 {{x}}\n.end")
    assert not marker.exists()


def test_plain_text_include_requires_a_file_sandbox():
    with pytest.raises(ValueError, match="file-backed.*include root"):
        mw.parse_spice(".include load.inc\n.end")


def test_statements_after_end_and_duplicate_casefolded_names_are_rejected():
    with pytest.raises(ValueError, match="after .end"):
        mw.parse_spice("R1 in 0 1\n.end\nR2 in 0 2")
    with pytest.raises(ValueError, match="already present"):
        mw.parse_spice("R1 in 0 1\nr1 in 0 2\n.end")


def test_unsupported_directive_in_unused_subcircuit_and_ends_prefix_are_rejected():
    with pytest.raises(ValueError, match="Unsupported directive.*model"):
        mw.parse_spice(".subckt s a b\n.model bad sw\n.ends s\n.end")
    with pytest.raises(ValueError, match="Unsupported directive.*endsbogus"):
        mw.parse_spice(".subckt s a b\nR1 a b 1\n.endsbogus\n.end")
    with pytest.raises(ValueError, match="Unsupported SPICE device.*D1"):
        mw.parse_spice(".subckt s a b\nD1 a b model\n.ends s\n.end")
    with pytest.raises(ValueError, match="Unsupported syntax"):
        mw.parse_spice(
            ".subckt s a b\n.param x={__import__('os')}\nR1 a b {x}\n.ends s\n.end"
        )


def test_subcircuit_after_end_is_rejected_and_prior_parameter_is_visible():
    with pytest.raises(ValueError):
        mw.parse_spice(".end\n.subckt s a b\nR1 a b 1\n.ends s")

    circuit = mw.parse_spice(
        ".param base=2\n.subckt s a b params: rv={base}\nR1 a b {rv}\n.ends s\nX1 in 0 s\n.end"
    )
    torch.testing.assert_close(circuit.devices[0].resistance, torch.tensor(2.0))


def test_invalid_parameter_override_and_hierarchy_collision_are_rejected():
    with pytest.raises(ValueError, match="override name"):
        mw.parse_spice("R1 in 0 1\n.end", parameters={"bad;name": 2.0})
    with pytest.raises(ValueError, match="collides with an explicit node"):
        mw.parse_spice(
            "Rtop X1:internal 0 1\n"
            ".subckt s a b\nR1 a internal 1\nR2 internal b 1\n.ends s\n"
            "X1 in 0 s\n.end"
        )


def test_serializer_refuses_a_device_name_with_the_wrong_designator():
    circuit = mw.Circuit("invalid")
    node = circuit.node("node")
    circuit.add(mw.Resistor("load", node, circuit.ground, 50.0))
    with pytest.raises(ValueError, match="designator.*'R'"):
        circuit.to_spice()

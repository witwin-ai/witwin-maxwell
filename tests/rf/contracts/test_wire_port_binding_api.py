import pytest

import witwin.maxwell as mw


def _nodes():
    return mw.WireNodeRef("dipole", 0), mw.WireNodeRef("dipole", -1)


def test_wire_port_binding_types_are_top_level_public_api():
    assert mw.WireNodeRef.__module__ == "witwin.maxwell.ports"
    assert mw.WirePortBinding.__module__ == "witwin.maxwell.ports"


def test_wire_node_refs_and_binding_polarity_are_immutable_and_explicit():
    negative, positive = _nodes()
    nodes = mw.WirePortBinding.nodes(negative=negative, positive=positive)
    gap = mw.WirePortBinding.gap(negative=negative, positive=positive)

    assert nodes.kind == "nodes"
    assert gap.kind == "gap"
    assert nodes.negative == gap.negative == negative
    assert nodes.positive == gap.positive == positive


@pytest.mark.parametrize("point", (True, 1.5, "1"))
def test_wire_node_ref_requires_an_integer_source_point(point):
    with pytest.raises(TypeError, match="integer source-point"):
        mw.WireNodeRef("wire", point)


def test_wire_binding_rejects_same_terminal_and_cross_wire_gap():
    node = mw.WireNodeRef("wire", 0)
    with pytest.raises(ValueError, match="distinct"):
        mw.WirePortBinding.nodes(negative=node, positive=node)
    with pytest.raises(ValueError, match="one wire"):
        mw.WirePortBinding.gap(
            negative=node,
            positive=mw.WireNodeRef("other", 1),
        )


def test_wire_bound_lumped_port_uses_binding_instead_of_axis_geometry():
    negative, positive = _nodes()
    binding = mw.WirePortBinding.gap(negative=negative, positive=positive)
    port = mw.LumpedPort("feed", wire_binding=binding)

    assert port.wire_binding is binding
    assert port.negative is port.positive is None
    assert port.voltage_path is port.current_surface is None
    scene = mw.Scene(device="cpu", ports=(port,))
    assert scene.ports == [port]


def test_wire_bound_lumped_port_rejects_ambiguous_field_geometry():
    negative, positive = _nodes()
    binding = mw.WirePortBinding.nodes(negative=negative, positive=positive)
    with pytest.raises(ValueError, match="must not also specify field geometry"):
        mw.LumpedPort(
            "feed",
            wire_binding=binding,
            negative=(0.0, 0.0, -0.1),
            positive=(0.0, 0.0, 0.1),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(position=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.0)),
        )


def test_unbound_lumped_port_keeps_the_existing_geometry_requirement():
    with pytest.raises(TypeError, match="voltage_path must be an AxisPath"):
        mw.LumpedPort("feed")


def test_wire_bound_terminal_port_does_not_require_structure_geometry():
    negative, positive = _nodes()
    binding = mw.WirePortBinding.nodes(negative=negative, positive=positive)
    port = mw.TerminalPort("feed", wire_binding=binding)

    assert port.wire_binding is binding
    assert port.positive_terminal is port.negative_terminal is None
    assert port.integration_path is port.voltage_path is None
    assert port.reference_plane is None

    with pytest.raises(ValueError, match="must not also specify structure terminal geometry"):
        mw.TerminalPort(
            "ambiguous",
            positive_terminal=mw.TerminalRef("positive"),
            negative_terminal=mw.TerminalRef("negative"),
            integration_path=mw.AxisPath("z"),
            reference_plane=0.0,
            wire_binding=binding,
        )

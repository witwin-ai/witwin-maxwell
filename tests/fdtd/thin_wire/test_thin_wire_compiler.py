from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.thin_wire import (
    CompiledWireNetwork,
    compile_thin_wires,
    compile_wire_monitors,
)
from witwin.maxwell.scene import prepare_scene


@dataclass(frozen=True)
class _Kind:
    kind: str


class _Wire(mw.ThinWire):
    def __init__(
        self,
        name,
        points,
        radius=1.0e-3,
        conductor=None,
        endpoints=None,
        snap="strict",
    ):
        super().__init__(
            name=name,
            points=points,
            radius=radius,
            conductor=mw.WireConductor.pec() if conductor is None else conductor,
            endpoints=endpoints,
            snap=snap,
        )


@dataclass(frozen=True)
class _Monitor:
    name: str
    wire: str
    frequencies: tuple[float, ...] = ()
    quantities: tuple[str, ...] = ("current", "charge")


def _prepared(*, wires=(), nodes=None, boundary=None, structures=(), monitors=(), ports=()):
    if nodes is None:
        grid = mw.GridSpec.uniform(0.25)
    else:
        grid = mw.GridSpec.custom(nodes, nodes, nodes)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=grid,
        boundary=boundary or mw.BoundarySpec.none(),
        structures=structures,
        thin_wires=tuple(wires),
        monitors=monitors,
        ports=ports,
        device="cpu",
    )
    return prepare_scene(scene)


def _straight(**changes):
    values = {
        "name": "straight",
        "points": ((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)),
    }
    values.update(changes)
    return _Wire(**values)


def _unsafe_straight(**changes):
    wire = _straight()
    for name, value in changes.items():
        object.__setattr__(wire, name, value)
    return wire


def _ground_structure(*, name="ground", pec=True, position=(0.25, 0.5, 0.5)):
    return mw.Structure(
        name=name,
        geometry=mw.Box(position=position, size=(0.1, 0.1, 0.1)),
        material=mw.Material.pec() if pec else mw.Material(eps_r=2.0),
    )


def test_straight_wire_splits_into_cell_fragments_and_builds_node_csr():
    network = compile_thin_wires(_prepared(wires=(_straight(),)))

    assert isinstance(network, CompiledWireNetwork)
    assert network.field_shapes == ((4, 5, 5), (5, 4, 5), (5, 5, 4))
    assert network.wire_names == ("straight",)
    assert network.node_count == 3
    assert network.segment_count == 2
    torch.testing.assert_close(network.length, torch.tensor([0.25, 0.25], dtype=torch.float64))
    assert network.length.sum() == pytest.approx(0.5)
    torch.testing.assert_close(network.tail, torch.tensor([0, 1]))
    torch.testing.assert_close(network.head, torch.tensor([1, 2]))
    torch.testing.assert_close(network.node_offsets, torch.tensor([0, 1, 3, 4]))
    torch.testing.assert_close(network.node_segments, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(network.node_signs, torch.tensor([1, -1, 1, -1], dtype=torch.int32))
    assert network.node_signs.dtype == torch.int32
    torch.testing.assert_close(network.open_endpoints, torch.tensor([True, False, True]))
    assert not torch.any(network.grounded)

    # Ex is flattened in C order: (i * Ny + j) * Nz + k.
    torch.testing.assert_close(network.edge_components, torch.tensor([0, 0], dtype=torch.int32))
    torch.testing.assert_close(network.edge_offsets, torch.tensor([37, 62]))
    torch.testing.assert_close(network.weights, torch.tensor([0.25, 0.25], dtype=torch.float64))
    torch.testing.assert_close(network.segment_offsets, torch.tensor([0, 1, 2]))

    expected_distance = 0.25 * math.exp(
        math.log(2.0) / 3.0 + math.pi / 3.0 - 25.0 / 12.0
    )
    expected_l_prime = 1.25663706212e-6 * math.log(expected_distance / 1.0e-3) / (
        2.0 * math.pi
    )
    expected_c_prime = 1.25663706212e-6 * 8.8541878128e-12 / expected_l_prime
    torch.testing.assert_close(
        network.coupling_distance,
        torch.full((2,), expected_distance, dtype=torch.float64),
    )
    torch.testing.assert_close(
        network.inductance,
        torch.full((2,), expected_l_prime * 0.25, dtype=torch.float64),
    )
    torch.testing.assert_close(
        network.capacitance_per_length,
        torch.full((2,), expected_c_prime, dtype=torch.float64),
    )

    contributions = 0.5 * network.capacitance_per_length * network.length
    expected_capacitance = torch.stack(
        (contributions[0], contributions[0] + contributions[1], contributions[1])
    )
    torch.testing.assert_close(network.node_capacitance, expected_capacitance)


def test_l_shape_preserves_length_orientation_and_corner_connectivity():
    wire = _Wire(
        name="elbow",
        points=((0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.75, 0.75, 0.5)),
        endpoints=(mw.WireEnd.grounded(structure="ground"), mw.WireEnd.open()),
    )
    ground = _ground_structure(position=(0.25, 0.25, 0.5))
    network = compile_thin_wires(_prepared(wires=(wire,), structures=(ground,)))

    assert network.node_count == 5
    assert network.segment_count == 4
    assert network.length.sum() == pytest.approx(1.0)
    torch.testing.assert_close(network.segment_axes, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(network.segment_directions, torch.ones(4, dtype=torch.int8))
    torch.testing.assert_close(network.grounded, torch.tensor([True, False, False, False, False]))
    torch.testing.assert_close(network.open_endpoints, torch.tensor([False, False, False, False, True]))
    corner = 2
    start = int(network.node_offsets[corner])
    end = int(network.node_offsets[corner + 1])
    torch.testing.assert_close(network.node_segments[start:end], torch.tensor([1, 2]))
    torch.testing.assert_close(network.node_signs[start:end], torch.tensor([-1, 1], dtype=torch.int32))


def test_sampling_and_sorted_deposition_are_exact_power_transposes():
    wires = (
        _Wire("z_wire", ((0.25, 0.25, 0.25), (0.25, 0.25, 0.75))),
        _Wire("x_wire", ((0.25, 0.75, 0.75), (0.75, 0.75, 0.75))),
    )
    network = compile_thin_wires(_prepared(wires=wires))
    generator = torch.Generator().manual_seed(71)
    fields = [torch.randn(shape, dtype=torch.float64, generator=generator) for shape in network.field_shapes]
    current = torch.randn((network.segment_count,), dtype=torch.float64, generator=generator)

    sampled = []
    for segment in range(network.segment_count):
        begin = int(network.segment_offsets[segment])
        finish = int(network.segment_offsets[segment + 1])
        value = torch.zeros((), dtype=torch.float64)
        for entry in range(begin, finish):
            component = int(network.edge_components[entry])
            offset = int(network.edge_offsets[entry])
            value = value + network.weights[entry] * fields[component].reshape(-1)[offset]
        sampled.append(value)
    wire_power = torch.dot(torch.stack(sampled), current)

    field_power = torch.zeros((), dtype=torch.float64)
    keys = list(zip(network.target_components.tolist(), network.target_offsets.tolist()))
    assert keys == sorted(keys)
    for group, (component, offset) in enumerate(keys):
        begin = int(network.edge_group_offsets[group])
        finish = int(network.edge_group_offsets[group + 1])
        deposited = torch.zeros((), dtype=torch.float64)
        for entry in range(begin, finish):
            segment = int(network.contribution_segments[entry])
            deposited = deposited + network.contribution_weights[entry] * current[segment]
        field_power = field_power + fields[component].reshape(-1)[offset] * deposited
    torch.testing.assert_close(field_power, wire_power)


def test_custom_grid_uses_physical_cell_lengths_and_local_transverse_spacing():
    nodes = np.array([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    wire = _Wire("graded", ((0.1, 0.3, 0.6), (0.6, 0.3, 0.6)), radius=1.0e-4)
    network = compile_thin_wires(_prepared(wires=(wire,), nodes=nodes))

    torch.testing.assert_close(network.length, torch.tensor([0.2, 0.3], dtype=torch.float64))
    assert network.length.sum() == pytest.approx(0.5)
    assert torch.all(network.coupling_distance > network.radius)
    assert torch.all(network.radius_to_spacing <= 0.2)
    assert network.metadata["grid_fingerprint"]


def test_local_host_permittivity_and_permeability_enter_line_coefficients():
    vacuum = compile_thin_wires(_prepared(wires=(_straight(),)))
    host = mw.Structure(
        name="host",
        geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(2.0, 2.0, 2.0)),
        material=mw.Material(eps_r=4.0, mu_r=2.0),
    )
    embedded = compile_thin_wires(_prepared(wires=(_straight(),), structures=(host,)))

    torch.testing.assert_close(embedded.inductance, 2.0 * vacuum.inductance)
    torch.testing.assert_close(
        embedded.capacitance_per_length,
        4.0 * vacuum.capacitance_per_length,
    )
    assert (
        embedded.metadata["compile_fingerprint"]
        != vacuum.metadata["compile_fingerprint"]
    )


def test_nearest_snap_is_explicit_and_strict_snap_rejects_off_grid_points():
    nearest = _straight(
        points=((0.21, 0.49, 0.49), (0.76, 0.49, 0.49)),
        snap="nearest",
    )
    network = compile_thin_wires(_prepared(wires=(nearest,)))
    torch.testing.assert_close(
        network.node_positions,
        torch.tensor(
            ((0.25, 0.5, 0.5), (0.5, 0.5, 0.5), (0.75, 0.5, 0.5)),
            dtype=torch.float64,
        ),
    )

    strict = _straight(points=nearest.points, snap="strict")
    with pytest.raises(ValueError, match="snap='strict'"):
        compile_thin_wires(_prepared(wires=(strict,)))


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        (
            _unsafe_straight(points=((0.25, 0.5, 0.5), (0.25, 0.5, 0.5))),
            "zero-length",
        ),
        (_straight(points=((0.25, 0.5, 0.5), (0.5, 0.75, 0.5))), "axis-aligned"),
        (_unsafe_straight(radius=0.0), "positive"),
        (_straight(radius=0.1), "validity band"),
        (_unsafe_straight(conductor=_Kind("finite")), "PEC"),
        (
            _unsafe_straight(endpoints=(_Kind("node"), _Kind("open"))),
            "open or grounded",
        ),
    ],
)
def test_invalid_phase1_wire_contracts_are_rejected(wire, message):
    with pytest.raises(ValueError, match=message):
        compile_thin_wires(_prepared(wires=(wire,)))


def test_trainable_points_are_rejected_but_radius_keeps_autograd():
    points = torch.tensor(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), requires_grad=True)
    with pytest.raises(ValueError, match="trainable points"):
        compile_thin_wires(_prepared(wires=(_straight(points=points),)))

    radius = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    network = compile_thin_wires(_prepared(wires=(_straight(radius=radius),)), device="cpu")
    assert all(tensor.device.type == "cpu" for tensor in network.tensor_fields())
    network.inductance.sum().backward()
    assert radius.grad is not None
    assert radius.grad < 0.0

    subnormal = torch.tensor(1.0e-45, dtype=torch.float32)
    with pytest.raises(ValueError, match="normal floating-point range"):
        compile_thin_wires(_prepared(wires=(_straight(radius=subnormal),)))


@pytest.mark.parametrize(
    ("structures", "structure_name", "message"),
    [
        ((), "missing", "does not exist"),
        ((_ground_structure(pec=False),), "ground", "must be PEC"),
        (
            (_ground_structure(position=(0.75, 0.75, 0.75)),),
            "ground",
            "on or inside PEC structure",
        ),
    ],
)
def test_grounded_endpoints_require_a_named_contacting_pec_structure(
    structures, structure_name, message
):
    wire = _straight(
        endpoints=(
            mw.WireEnd.grounded(structure=structure_name),
            mw.WireEnd.open(),
        )
    )
    with pytest.raises(ValueError, match=message):
        compile_thin_wires(_prepared(wires=(wire,), structures=structures))


def test_undeclared_pec_contact_and_unbound_port_edge_ownership_are_rejected():
    embedded_pec = mw.Structure(
        name="embedded",
        geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(0.75, 0.2, 0.2)),
        material=mw.Material.pec(),
    )
    with pytest.raises(ValueError, match="overlaps PEC"):
        compile_thin_wires(_prepared(wires=(_straight(),), structures=(embedded_pec,)))

    port = mw.LumpedPort(
        "shared_edge",
        positive=(0.75, 0.5, 0.5),
        negative=(0.25, 0.5, 0.5),
        voltage_path=mw.AxisPath("x"),
        current_surface=mw.Box(
            position=(0.375, 0.5, 0.5),
            size=(0.0, 0.75, 0.75),
        ),
    )
    with pytest.raises(ValueError, match="overlaps a ThinWire coupling edge"):
        compile_thin_wires(_prepared(wires=(_straight(),), ports=(port,)))


def test_neighboring_conductor_distance_and_sparse_ownership_are_reported():
    adjacent = (
        _Wire("a", ((0.25, 0.25, 0.5), (0.75, 0.25, 0.5))),
        _Wire("b", ((0.25, 0.5, 0.5), (0.75, 0.5, 0.5))),
    )
    network = compile_thin_wires(_prepared(wires=adjacent))
    assert network.metadata["validity"]["minimum_neighbor_distance"] == pytest.approx(0.25)
    assert (
        network.metadata["validity"]["proximity_criterion"]
        == "physical_radius_and_sparse_target_ownership"
    )

    nodes = np.array([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    close_but_distinct = (
        _Wire("a", ((0.1, 0.1, 0.6), (0.6, 0.1, 0.6)), radius=1.0e-4),
        _Wire("b", ((0.1, 0.3, 0.6), (0.6, 0.3, 0.6)), radius=1.0e-4),
    )
    graded = compile_thin_wires(_prepared(wires=close_but_distinct, nodes=nodes))
    assert graded.metadata["validity"]["minimum_neighbor_distance"] == pytest.approx(0.2)

    self_colliding = _Wire(
        "u_bend",
        ((0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.5, 0.01, 0.5), (0.0, 0.01, 0.5)),
        radius=(1.0e-3, 1.0e-3, 2.0e-2),
    )
    collision_nodes = np.array([0.0, 0.01, 0.5, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="overlap or touch"):
        compile_thin_wires(
            _prepared(wires=(self_colliding,), nodes=collision_nodes)
        )

    axial_nodes = np.array([0.0, 0.25, 0.26, 0.75, 1.0], dtype=np.float64)
    subdivided_straight = _Wire(
        "subdivided",
        ((0.0, 0.75, 0.75), (0.75, 0.75, 0.75)),
        radius=2.0e-2,
    )
    subdivided = compile_thin_wires(
        _prepared(wires=(subdivided_straight,), nodes=axial_nodes)
    )
    assert subdivided.segment_count == 3


def test_narrow_pec_intersection_between_grid_samples_is_rejected():
    sliver = mw.Structure(
        name="sliver",
        geometry=mw.Box(position=(0.3125, 0.5, 0.5), size=(0.02, 0.1, 0.1)),
        material=mw.Material.pec(),
    )
    with pytest.raises(ValueError, match="overlaps PEC"):
        compile_thin_wires(_prepared(wires=(_straight(),), structures=(sliver,)))


def test_phase1_rejects_anisotropic_host_self_coefficients():
    host = mw.Structure(
        name="anisotropic_host",
        geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(2.0, 2.0, 2.0)),
        material=mw.Material(
            epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0),
        ),
    )
    with pytest.raises(NotImplementedError, match="locally isotropic"):
        compile_thin_wires(_prepared(wires=(_straight(),), structures=(host,)))


def test_pml_domain_shared_node_and_loop_contracts_are_rejected():
    pml_wire = _Wire("pml", ((-0.25, 0.5, 0.5), (0.25, 0.5, 0.5)))
    with pytest.raises(ValueError, match="enters the PML"):
        compile_thin_wires(
            _prepared(wires=(pml_wire,), boundary=mw.BoundarySpec.pml(num_layers=2))
        )

    with pytest.raises(ValueError, match="outside the physical domain"):
        compile_thin_wires(
            _prepared(wires=(_Wire("outside", ((0.25, 0.5, 0.5), (1.25, 0.5, 0.5))),))
        )

    boundary_wire = _Wire(
        "boundary_contact",
        ((0.25, 0.0, 0.5), (0.75, 0.0, 0.5)),
    )
    with pytest.raises(ValueError, match="touches the PEC y-low boundary"):
        compile_thin_wires(
            _prepared(wires=(boundary_wire,), boundary=mw.BoundarySpec.pec())
        )

    shared = (
        _Wire("a", ((0.25, 0.5, 0.5), (0.5, 0.5, 0.5))),
        _Wire("b", ((0.5, 0.5, 0.5), (0.5, 0.75, 0.5))),
    )
    with pytest.raises(ValueError, match="junctions are Phase 2"):
        compile_thin_wires(_prepared(wires=shared))

    loop = _unsafe_straight(
        name="loop",
        points=(
            (0.25, 0.25, 0.5),
            (0.75, 0.25, 0.5),
            (0.75, 0.75, 0.5),
            (0.25, 0.75, 0.5),
            (0.25, 0.25, 0.5),
        ),
    )
    with pytest.raises(ValueError, match="loops, branches"):
        compile_thin_wires(_prepared(wires=(loop,)))


def test_wire_order_and_fingerprints_are_deterministic_and_sensitive():
    a = _Wire("a", ((0.25, 0.25, 0.25), (0.75, 0.25, 0.25)), radius=1.0e-3)
    b = _Wire("b", ((0.25, 0.75, 0.75), (0.75, 0.75, 0.75)), radius=2.0e-3)
    forward = compile_thin_wires(_prepared(wires=(a, b)))
    reverse = compile_thin_wires(_prepared(wires=(b, a)))

    assert forward.wire_names == reverse.wire_names == ("a", "b")
    assert forward.metadata["compile_fingerprint"] == reverse.metadata["compile_fingerprint"]
    assert forward.metadata["validity"]["coupling_kernel"] == "BS1xBS1"
    for name in (
        "node_grid_indices",
        "tail",
        "head",
        "edge_components",
        "edge_offsets",
        "target_components",
        "target_offsets",
        "contribution_segments",
    ):
        torch.testing.assert_close(getattr(forward, name), getattr(reverse, name))

    changed_radius = compile_thin_wires(
        _prepared(wires=(_Wire(**(a.__dict__ | {"radius": 1.5e-3})), b))
    )
    assert changed_radius.metadata["grid_fingerprint"] == forward.metadata["grid_fingerprint"]
    assert changed_radius.metadata["compile_fingerprint"] != forward.metadata["compile_fingerprint"]

    grounded = _Wire(
        **(
            a.__dict__
            | {
                "endpoints": (
                    mw.WireEnd.grounded(structure="ground"),
                    mw.WireEnd.open(),
                )
            }
        )
    )
    changed_endpoint = compile_thin_wires(
        _prepared(
            wires=(grounded, b),
            structures=(_ground_structure(position=(0.25, 0.25, 0.25)),),
        )
    )
    assert (
        changed_endpoint.metadata["compile_fingerprint"]
        != forward.metadata["compile_fingerprint"]
    )

    custom_nodes = np.array([0.0, 0.25, 0.5, 0.8, 1.0])
    custom_a = _Wire("a", ((0.25, 0.25, 0.25), (0.8, 0.25, 0.25)), radius=1.0e-3)
    changed_grid = compile_thin_wires(_prepared(wires=(custom_a,), nodes=custom_nodes))
    assert changed_grid.metadata["grid_fingerprint"] != forward.metadata["grid_fingerprint"]


def test_static_network_cache_hits_and_trainable_coefficients_bypass_it():
    prepared = _prepared(wires=(_straight(),))
    first = compile_thin_wires(prepared)
    second = compile_thin_wires(prepared)
    assert first is second
    assert first.metadata["cache_enabled"] is True

    radius = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    trainable = _prepared(wires=(_straight(radius=radius),))
    first_trainable = compile_thin_wires(trainable)
    second_trainable = compile_thin_wires(trainable)
    assert first_trainable is not second_trainable
    assert first_trainable.metadata["cache_enabled"] is False


def test_empty_network_and_wire_monitor_resolution():
    empty = compile_thin_wires(_prepared())
    assert empty.node_count == 0
    assert empty.segment_count == 0
    assert empty.edge_group_offsets.tolist() == [0]

    monitors = (
        SimpleNamespace(
            name="state",
            wire="straight",
            frequencies=(1.0e9, 2.0e9),
            quantities=("current",),
        ),
    )
    prepared = _prepared(wires=(_straight(),), monitors=monitors)
    network = compile_thin_wires(prepared)
    compiled = compile_wire_monitors(prepared, device="cpu")
    assert len(compiled) == 1
    assert compiled[0].frequencies == (1.0e9, 2.0e9)
    assert compiled[0].segment_indices.tolist() == [0, 1]
    assert compiled[0].node_indices.tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="unknown wire"):
        compile_wire_monitors(
            prepared,
            network,
            (_Monitor("bad", "missing"),),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        compile_wire_monitors(
            prepared,
            network,
            (_Monitor("bad_frequency", "straight", (0.0,)),),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_requested_cuda_device_contains_every_compiled_tensor():
    radius = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
    network = compile_thin_wires(_prepared(wires=(_straight(radius=radius),)), device="cuda")
    assert all(tensor.is_cuda for tensor in network.tensor_fields())
    network.inductance.sum().backward()
    assert radius.grad is not None

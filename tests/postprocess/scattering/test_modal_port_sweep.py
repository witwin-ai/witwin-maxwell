"""CPU coverage for ModalPortSweep planning, column scenes and S assembly."""

from __future__ import annotations

import cmath

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.modal_sweep import (
    ModalColumnWaves,
    assemble_modal_network,
    build_modal_column_scene,
    plan_modal_port_sweep,
)


_FREQUENCIES = (2.9e8, 3.0e8, 3.1e8)


def _scene(*, left_offset=0.1, right_offset=-0.1, extra_port=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    )
    scene.add_port(
        mw.ModePort(
            "left",
            position=(-0.5, 0.0, 0.0),
            size=(0.0, 0.6, 0.6),
            direction="+",
            polarization="Ez",
            monitor_offset=left_offset,
        )
    )
    scene.add_port(
        mw.ModePort(
            "right",
            position=(0.5, 0.0, 0.0),
            size=(0.0, 0.6, 0.6),
            direction="-",
            polarization="Ez",
            monitor_offset=right_offset,
            source_time=mw.CW(frequency=3.0e8),
        )
    )
    if extra_port is not None:
        scene.add_port(extra_port)
    return scene


def test_plan_orders_channels_by_port_then_mode_with_labels():
    plan = plan_modal_port_sweep(
        _scene(),
        mw.ModalPortSweep(modes={"left": (0, 1, ("Ey", 0))}),
        _FREQUENCIES,
    )

    assert plan.channel_labels == ("left::Ez0", "left::Ez1", "left::Ey0", "right::Ez0")
    assert plan.port_names == ("left", "right")
    assert plan.run_columns == (0, 1, 2, 3)
    assert plan.valid_columns == (True,) * 4
    # Inward-positive monitor distances: both monitors sit 0.1 inside their port.
    assert [channel.reference_offset for channel in plan.channels] == pytest.approx([0.1] * 4)
    assert [channel.reference_position for channel in plan.channels] == pytest.approx([-0.5] * 3 + [0.5])
    # The sweep source overrides nothing here: port-declared CW is kept, the
    # source-less port receives the band-covering default pulse.
    assert plan.source_times[3].kind == "cw"
    assert plan.source_times[0].kind == "gaussian_pulse"
    metadata = plan.metadata()
    assert metadata["run_channels"] == plan.channel_labels
    assert metadata["valid_columns"] == (True,) * 4


def test_plan_resolves_run_only_in_channel_order_and_element_mappings():
    sweep = mw.ModalPortSweep(
        run_only=("right::Ez0", ("left", 0)),
        source_time=mw.CW(frequency=3.0e8),
    )
    plan = plan_modal_port_sweep(_scene(), sweep, _FREQUENCIES)
    assert plan.run_columns == (0, 1)
    assert all(source.kind == "cw" for source in plan.source_times)

    one_column = mw.ModalPortSweep(
        run_only=("left::Ez0",),
        element_mappings=(
            (("right::Ez0", "left::Ez0"), ("left::Ez0", "right::Ez0"), 1.0),
            (("left::Ez0", "left::Ez0"), (("right", 0), ("right", 0)), 1.0),
        ),
    )
    plan = plan_modal_port_sweep(_scene(), one_column, _FREQUENCIES)
    assert plan.run_columns == (0,)
    assert plan.element_mappings == ((1, 0, 0, 1, 1.0 + 0.0j), (0, 0, 1, 1, 1.0 + 0.0j))
    assert plan.valid_columns == (True, True)

    partial = mw.ModalPortSweep(run_only="left::Ez0")
    assert plan_modal_port_sweep(_scene(), partial, _FREQUENCIES).valid_columns == (True, False)


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        (mw.ModalPortSweep(ports=("left", "missing")), "missing ModePorts"),
        (mw.ModalPortSweep(modes={"right": (("Ex", 0),)}), "Ex"),
        (mw.ModalPortSweep(modes={"left": (0, 0)}), "repeats mode"),
        (mw.ModalPortSweep(run_only=("left::Ez3",)), "not a sweep channel"),
        (
            mw.ModalPortSweep(
                run_only=("left::Ez0",),
                element_mappings=((("left::Ez0", "right::Ez0"), ("right::Ez0", "left::Ez0"), 1.0),),
            ),
            "neither computed",
        ),
        (
            mw.ModalPortSweep(
                element_mappings=((("left::Ez0", "left::Ez0"), ("right::Ez0", "right::Ez0"), 1.0),),
            ),
            "already computed",
        ),
    ],
)
def test_plan_rejects_inconsistent_requests(sweep, message):
    with pytest.raises(ValueError, match=message):
        plan_modal_port_sweep(_scene(), sweep, _FREQUENCIES)


def test_plan_rejects_driven_monitor_behind_launch_plane():
    with pytest.raises(ValueError, match="monitor_offset"):
        plan_modal_port_sweep(_scene(left_offset=-0.1), mw.ModalPortSweep(), _FREQUENCIES)
    with pytest.raises(ValueError, match="monitor_offset"):
        plan_modal_port_sweep(_scene(right_offset=0.0), mw.ModalPortSweep(), _FREQUENCIES)
    # A monitor-only port may sample anywhere, so run_only can skip it.
    plan = plan_modal_port_sweep(
        _scene(right_offset=0.0),
        mw.ModalPortSweep(run_only="left::Ez0"),
        _FREQUENCIES,
    )
    assert plan.channels[1].reference_offset == 0.0


def test_plan_rejects_foreign_sources_ports_and_bad_frequencies():
    scene = _scene()
    scene.add_source(
        mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", source_time=mw.CW(frequency=3.0e8))
    )
    with pytest.raises(ValueError, match="Scene.sources"):
        plan_modal_port_sweep(scene, mw.ModalPortSweep(), _FREQUENCIES)
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_modal_port_sweep(_scene(), mw.ModalPortSweep(), (3.0e8, 2.0e8))
    with pytest.raises(TypeError):
        mw.ModalPortSweep(modes={"left": (("Ez", 0, 1),)})
    with pytest.raises(TypeError):
        plan_modal_port_sweep(_scene(), mw.PortSweep(), _FREQUENCIES)


def test_column_scene_excites_only_the_driven_channel():
    scene = _scene()
    plan = plan_modal_port_sweep(
        scene,
        mw.ModalPortSweep(modes={"left": (0, 1)}, run_only=("left::Ez1", "right::Ez0")),
        _FREQUENCIES,
    )
    first = build_modal_column_scene(scene, plan, 0)
    ports = {port.name: port for port in first.ports}
    assert ports["left"].source_time is not None
    assert ports["left"].mode_index == 1
    assert ports["right"].source_time is None
    assert ports["left"].frequencies == _FREQUENCIES
    assert ports["right"].frequencies == _FREQUENCIES
    assert ports["left"].monitor_offset == pytest.approx(0.1)
    sources = first.resolved_sources()
    assert [source.name for source in sources] == ["left::source"]

    second = build_modal_column_scene(scene, plan, 1)
    ports = {port.name: port for port in second.ports}
    assert ports["left"].source_time is None
    assert ports["right"].source_time.kind == "cw"
    assert [source.name for source in second.resolved_sources()] == ["right::source"]
    # The user scene is untouched.
    assert scene.ports[0].source_time is None


def _synthetic_columns(plan, s_ref, beta, *, passive_leak=0.0, seed=0):
    """Monitor-plane waves that realize ``s_ref`` at the ModePort reference planes."""

    generator = torch.Generator().manual_seed(seed)
    frequency_count, count, _ = s_ref.shape
    offsets = torch.tensor([channel.reference_offset for channel in plan.channels], dtype=torch.float64)
    columns = []
    for column in plan.run_columns:
        drive = torch.complex(
            torch.rand(frequency_count, generator=generator, dtype=torch.float64) + 0.5,
            torch.rand(frequency_count, generator=generator, dtype=torch.float64) - 0.5,
        )
        a_ref = torch.zeros((frequency_count, count), dtype=torch.complex128)
        a_ref[:, column] = drive
        for other in range(count):
            if other != column:
                a_ref[:, other] = passive_leak * drive
        b_ref = s_ref[:, :, column] * drive.unsqueeze(-1)
        columns.append(
            ModalColumnWaves(
                incident=a_ref * torch.exp(1j * beta * offsets),
                outgoing=b_ref * torch.exp(-1j * beta * offsets),
                beta=beta,
            )
        )
    return tuple(columns)


def _reference_s(frequency_count, count, seed=1):
    generator = torch.Generator().manual_seed(seed)
    real = torch.rand((frequency_count, count, count), generator=generator, dtype=torch.float64)
    imag = torch.rand((frequency_count, count, count), generator=generator, dtype=torch.float64)
    return torch.complex(real - 0.5, imag - 0.5)


def test_assembly_normalizes_by_driven_incident_and_deembeds_monitor_offsets():
    scene = _scene(left_offset=0.15, right_offset=-0.05)
    plan = plan_modal_port_sweep(scene, mw.ModalPortSweep(modes={"left": (0, 1)}), _FREQUENCIES)
    count = len(plan.channels)
    beta = torch.tensor(
        [[9.0 + 0.0j, 7.5 + 0.0j, 9.5 + 0.1j]] * len(_FREQUENCIES), dtype=torch.complex128
    ) * torch.tensor(_FREQUENCIES, dtype=torch.float64).unsqueeze(-1) / 3.0e8
    s_ref = _reference_s(len(_FREQUENCIES), count)

    ports, network = assemble_modal_network(plan, _synthetic_columns(plan, s_ref, beta, passive_leak=1e-3))

    assert network.port_names == ("left::Ez0", "left::Ez1", "right::Ez0")
    assert tuple(network.s.shape) == (len(_FREQUENCIES), count, count)
    assert network.is_complete
    torch.testing.assert_close(network.s, s_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(network.metadata["propagation_constants"], beta)
    torch.testing.assert_close(network.z0, torch.ones_like(network.z0))
    ratio = network.metadata["passive_incident_ratio"]
    assert float(ratio.max()) == pytest.approx(1e-3, rel=1e-9)
    assert float(torch.diagonal(ratio, dim1=-2, dim2=-1).abs().max()) == 0.0

    # Moving the reference planes back onto the monitor planes reproduces the raw
    # monitor-plane ratios, i.e. the de-embedding matches shift_reference_planes.
    offsets = torch.tensor([channel.reference_offset for channel in plan.channels], dtype=torch.float64)
    at_monitors = network.shift_reference_planes(-offsets)
    raw = torch.exp(-1j * beta * offsets).unsqueeze(-1) * s_ref * torch.exp(-1j * beta * offsets).unsqueeze(-2)
    torch.testing.assert_close(at_monitors.s, raw, rtol=1e-12, atol=1e-12)

    left = ports["left"]
    assert left.mode_names == ("Ez0", "Ez1")
    assert tuple(left.voltage.shape) == (count, 2, len(_FREQUENCIES))
    assert left.direction == "+"
    assert ports["right"].direction == "-"
    assert left.reference_plane == pytest.approx(-0.5)
    assert ports["right"].reference_plane == pytest.approx(0.5)
    torch.testing.assert_close(left.b[0], s_ref[:, 0:2, 0].transpose(0, 1) * left.a[0, 0].unsqueeze(0))


def test_assembly_phase_matches_beta_length_for_matched_line():
    plan = plan_modal_port_sweep(_scene(), mw.ModalPortSweep(), _FREQUENCIES)
    beta = (2.0 * cmath.pi * torch.tensor(_FREQUENCIES, dtype=torch.float64) * 1.6 / 3.0e8)
    beta = beta.to(torch.complex128).unsqueeze(-1).expand(-1, 2).contiguous()
    length = 1.0  # distance between the two ModePort reference planes
    transmission = torch.exp(1j * beta[:, 0] * length)
    s_ref = torch.zeros((len(_FREQUENCIES), 2, 2), dtype=torch.complex128)
    s_ref[:, 1, 0] = transmission
    s_ref[:, 0, 1] = transmission

    _, network = assemble_modal_network(plan, _synthetic_columns(plan, s_ref, beta))

    torch.testing.assert_close(network.s, s_ref, rtol=1e-12, atol=1e-12)
    phase = torch.angle(network.s[:, 1, 0])
    expected = torch.remainder(beta[:, 0].real * length + cmath.pi, 2.0 * cmath.pi) - cmath.pi
    torch.testing.assert_close(phase, expected, rtol=0.0, atol=1e-12)


def test_assembly_fills_symmetric_entries_and_flags_incomplete_columns():
    scene = _scene()
    beta = torch.full((len(_FREQUENCIES), 2), 8.0 + 0.0j, dtype=torch.complex128)
    s_ref = _reference_s(len(_FREQUENCIES), 2, seed=3)
    s_ref[:, 0, 1] = s_ref[:, 1, 0]
    s_ref[:, 1, 1] = s_ref[:, 0, 0]

    mapped = plan_modal_port_sweep(
        scene,
        mw.ModalPortSweep(
            run_only="left::Ez0",
            element_mappings=(
                (("right::Ez0", "left::Ez0"), ("left::Ez0", "right::Ez0"), 1.0),
                (("left::Ez0", "left::Ez0"), ("right::Ez0", "right::Ez0"), 1.0),
            ),
        ),
        _FREQUENCIES,
    )
    _, network = assemble_modal_network(mapped, _synthetic_columns(mapped, s_ref, beta))
    assert network.is_complete
    torch.testing.assert_close(network.s, s_ref, rtol=1e-12, atol=1e-12)

    partial = plan_modal_port_sweep(scene, mw.ModalPortSweep(run_only="left::Ez0"), _FREQUENCIES)
    ports, network = assemble_modal_network(partial, _synthetic_columns(partial, s_ref, beta))
    assert network.valid_columns.tolist() == [True, False]
    assert not network.is_complete
    assert network.metadata["known_elements"].tolist() == [[True, False], [True, False]]
    torch.testing.assert_close(network.s[:, :, 0], s_ref[:, :, 0], rtol=1e-12, atol=1e-12)
    assert bool(torch.all(network.s[:, :, 1] == 0))
    with pytest.raises(RuntimeError, match="complete excitation columns"):
        network.to_z()
    assert tuple(ports["left"].voltage.shape) == (1, 1, len(_FREQUENCIES))


def test_assembly_rejects_weak_drive_and_shape_errors():
    plan = plan_modal_port_sweep(_scene(), mw.ModalPortSweep(), _FREQUENCIES)
    beta = torch.full((len(_FREQUENCIES), 2), 8.0 + 0.0j, dtype=torch.complex128)
    columns = list(_synthetic_columns(plan, _reference_s(len(_FREQUENCIES), 2), beta))
    weak = columns[0].incident.clone()
    weak[1, 0] = 0.0
    columns[0] = ModalColumnWaves(incident=weak, outgoing=columns[0].outgoing, beta=beta)
    with pytest.raises(RuntimeError, match="below threshold at frequency indices \\[1\\]"):
        assemble_modal_network(plan, tuple(columns))
    with pytest.raises(ValueError, match="column count"):
        assemble_modal_network(plan, tuple(columns[:1]))
    bad = ModalColumnWaves(incident=weak[:, :1], outgoing=columns[0].outgoing, beta=beta)
    with pytest.raises(ValueError, match="shape"):
        assemble_modal_network(plan, (bad, columns[1]))

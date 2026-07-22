import pytest
import torch

import witwin.maxwell as mw


def _state_space(*, conductance: float = 0.02) -> mw.StateSpaceNetwork:
    return mw.StateSpaceNetwork(
        A=torch.tensor([[-2.0e9]], dtype=torch.float64),
        B=torch.tensor([[0.0]], dtype=torch.float64),
        C=torch.tensor([[0.0]], dtype=torch.float64),
        D=torch.tensor([[conductance]], dtype=torch.float64),
        representation="Y",
        port_order=("load",),
        passivity_margin=conductance,
    )


def _network_data() -> mw.NetworkData:
    frequencies = torch.tensor([0.0, 1.0e9], dtype=torch.float64)
    admittance = torch.full((2, 1, 1), 0.02 + 0.0j, dtype=torch.complex128)
    return mw.NetworkData.from_y(
        frequencies=frequencies,
        y=admittance,
        z0=50.0,
        port_names=("load",),
    )


def _port(name: str = "feed") -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=50.0,
    )


def _scene(*, ports=None, networks=()) -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(),) if ports is None else ports,
        networks=networks,
        device="cpu",
    )


def _block(**overrides) -> mw.NetworkBlock:
    values = {
        "name": "matching",
        "network": _network_data(),
        "connections": {1: "feed"},
        "fit": False,
        "model": _state_space(),
    }
    values.update(overrides)
    return mw.NetworkBlock(**values)


def test_network_block_normalizes_one_based_connections_and_scene_clone() -> None:
    block = _block()
    scene = _scene(networks=(block,))

    assert block.connections == {"load": "feed"}
    assert block.port_order == ("load",)
    assert block.connected_port_names == ("feed",)
    assert scene.clone().networks == [block]


def test_scene_rejects_unknown_or_reused_network_connections() -> None:
    with pytest.raises(ValueError, match="unknown Scene port"):
        _scene(networks=(_block(connections={"load": "missing"}),))

    scene = _scene(networks=(_block(name="first"),))
    with pytest.raises(ValueError, match="already connected"):
        scene.add_network(_block(name="second"))


def test_network_block_requires_explicit_prefit_contract() -> None:
    with pytest.raises(ValueError, match="fit=False requires"):
        mw.NetworkBlock(
            name="missing_model",
            network=_network_data(),
            connections={"load": "feed"},
            fit=False,
        )
    with pytest.raises(ValueError, match="requires fit=False"):
        mw.NetworkBlock(
            name="ambiguous",
            network=_network_data(),
            connections={"load": "feed"},
            model=_state_space(),
        )


def test_network_compiler_builds_fixed_single_port_discrete_descriptor() -> None:
    scene = _scene(networks=(_block(),))

    (compiled,) = scene.compile_networks(dt=1.0e-12, device="cpu")

    assert compiled.name == "matching"
    assert compiled.port_order == ("load",)
    assert compiled.connection_names == ("feed",)
    assert compiled.port_count == 1
    assert compiled.state_count == 1
    assert compiled.discrete.A.device.type == "cpu"
    assert compiled.discrete.pole_radius < 1.0 - 1.0e-7


def test_compiler_builds_fixed_multiport_discrete_descriptor() -> None:
    frequencies = torch.tensor([0.0, 1.0e9], dtype=torch.float64)
    scattering = torch.zeros((2, 2, 2), dtype=torch.complex128)
    data = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("a", "b"),
    )
    model = mw.StateSpaceNetwork(
        A=-2.0e9 * torch.eye(2, dtype=torch.float64),
        B=torch.zeros((2, 2), dtype=torch.float64),
        C=torch.zeros((2, 2), dtype=torch.float64),
        D=0.02 * torch.eye(2, dtype=torch.float64),
        representation="Y",
        port_order=("a", "b"),
    )
    block = mw.NetworkBlock(
        "two_port",
        data,
        {"a": "p1", "b": "p2"},
        fit=False,
        model=model,
    )
    scene = _scene(ports=(_port("p1"), _port("p2")), networks=(block,))

    (compiled,) = scene.compile_networks(dt=1.0e-12, device="cpu")

    assert compiled.port_order == ("a", "b")
    assert compiled.connection_names == ("p1", "p2")
    assert compiled.port_count == 2
    assert compiled.discrete.B.shape == (2, 2)
    assert compiled.discrete.C.shape == (2, 2)
    assert compiled.discrete.D.shape == (2, 2)


def test_state_space_certificate_rejects_active_response_between_samples() -> None:
    poles = torch.tensor(
        [
            complex(-1.0e7, 2.0 * torch.pi * 0.5e9),
            complex(-1.0e7, -2.0 * torch.pi * 0.5e9),
        ],
        dtype=torch.complex128,
    )
    active = mw.RationalModel(
        poles=poles,
        residues=torch.tensor([-1.0e6, -1.0e6], dtype=torch.complex128),
        direct=0.02,
        representation="Y",
    ).to_state_space(port_order=("load",))
    endpoints = torch.tensor([0.0, 1.0e9], dtype=torch.float64)
    endpoint_response = active.evaluate(endpoints)[:, 0, 0].real
    assert bool(torch.all(endpoint_response > 0.0))
    assert active.evaluate(torch.tensor([0.5e9], dtype=torch.float64))[0, 0, 0].real < 0.0

    data = mw.NetworkData.from_y(
        frequencies=endpoints,
        y=active.evaluate(endpoints),
        z0=50.0,
        port_names=("load",),
    )
    block = mw.NetworkBlock(
        "hidden_active_band",
        data,
        {"load": "feed"},
        fit=False,
        model=active,
    )
    scene = _scene(networks=(block,))

    with pytest.raises(ValueError, match="certified passivity"):
        scene.compile_networks(dt=1.0e-12, device="cpu")


def test_network_embedding_rejects_waveport_terminal_with_accurate_reason() -> None:
    # E4b disposition: WavePort embedding stays fail-closed. A modal WavePort has
    # no scalar time-domain (V, I) terminal contract for a lumped state-space
    # network to couple to; the rejection must name that design gap, not read as
    # a transient bug.
    waveport = mw.WavePort(
        name="wp",
        position=(-0.01, 0.0, 0.0),
        size=(0.0, 0.03, 0.03),
        direction="+",
        reference_plane=-0.01,
        modes=(
            mw.WaveModeSpec(
                family="tem",
                voltage_path=((-0.01, -0.01, 0.0), (-0.01, 0.01, 0.0)),
                current_contour=mw.Box(
                    position=(-0.01, 0.0, 0.0), size=(0.0, 0.02, 0.02)
                ),
            ),
        ),
    )
    block = mw.NetworkBlock(
        name="modal_load",
        network=_network_data(),
        connections={"load": "wp"},
        fit=False,
        model=_state_space(),
    )
    with pytest.raises(ValueError, match=r"WavePort.*modal port.*no scalar time-domain"):
        _scene(ports=(waveport,), networks=(block,))


def test_zero_state_direct_network_discretizes_and_certifies_band() -> None:
    model = mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64),
        B=torch.zeros((0, 1), dtype=torch.float64),
        C=torch.zeros((1, 0), dtype=torch.float64),
        D=torch.tensor([[0.02]], dtype=torch.float64),
        representation="Y",
        port_order=("load",),
    )

    report = model.check_passivity([0.0, 1.0e9])
    discrete = model.discretize(1.0e-12)

    assert report.passive and report.certified
    assert report.margin == pytest.approx(0.02)
    assert discrete.state_count == 0
    assert discrete.pole_radius == 0.0
    torch.testing.assert_close(discrete.D, model.D)


def test_fdfd_rejects_scene_network_instead_of_ignoring_it() -> None:
    with pytest.raises(NotImplementedError, match="time-domain FDTD"):
        mw.Simulation.fdfd(_scene(networks=(_block(),)), frequency=5.0e8)

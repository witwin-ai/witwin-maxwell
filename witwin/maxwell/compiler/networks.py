from __future__ import annotations

from dataclasses import dataclass

import torch

from ..network import NetworkBlock
from ..ports import LumpedPort, TerminalPort, WavePort
from ..rational import (
    DiscreteStateSpaceNetwork,
    FitReport,
    RationalModel,
    StateSpaceNetwork,
)
from .ports import CompiledPortGeometry, compile_port_geometry


@dataclass(frozen=True)
class CompiledNetworkBlock:
    """Solver-ready, fixed-shape descriptor for one embedded network."""

    name: str
    port_order: tuple[str, ...]
    connection_names: tuple[str, ...]
    ports: tuple[CompiledPortGeometry, ...]
    continuous: StateSpaceNetwork
    discrete: DiscreteStateSpaceNetwork
    fit_report: FitReport | None
    model_id: str
    frequency_band: tuple[float, float]
    extrapolation: str = "reject"

    @property
    def port_count(self) -> int:
        return len(self.port_order)

    @property
    def state_count(self) -> int:
        return self.discrete.state_count


def _move_rational(model: RationalModel, device: torch.device) -> RationalModel:
    return RationalModel(
        poles=model.poles.to(device=device),
        residues=model.residues.to(device=device),
        direct=model.direct.to(device=device),
        proportional=model.proportional.to(device=device),
        representation=model.representation,
        report=model.report,
    )


def _move_state_space(model: StateSpaceNetwork, device: torch.device) -> StateSpaceNetwork:
    return StateSpaceNetwork(
        A=model.A.to(device=device),
        B=model.B.to(device=device),
        C=model.C.to(device=device),
        D=model.D.to(device=device),
        representation=model.representation,
        port_order=model.port_order,
        passivity_margin=model.passivity_margin,
        report=model.report,
    )


def _resolve_continuous_model(
    block: NetworkBlock,
    *,
    device: torch.device,
) -> tuple[StateSpaceNetwork, RationalModel | StateSpaceNetwork]:
    if block.model is None:
        model = block.network.fit_rational(block.fit, representation="Y")
        model = _move_rational(model, device)
        return model.to_state_space(port_order=block.port_order), model
    if isinstance(block.model, RationalModel):
        model = _move_rational(block.model, device)
        return model.to_state_space(port_order=block.port_order), model
    model = _move_state_space(block.model, device)
    return model, model


def _effective_frequency_band(
    block: NetworkBlock,
    continuous: StateSpaceNetwork,
) -> tuple[float, float]:
    if continuous.report is not None:
        return continuous.report.frequency_band
    if block.fit is not False and block.fit.band is not None:
        return block.fit.band
    return (
        float(block.network.frequencies[0].item()),
        float(block.network.frequencies[-1].item()),
    )


def compile_network_block(
    scene,
    block: NetworkBlock,
    *,
    dt: float,
    device: str | torch.device | None = None,
) -> CompiledNetworkBlock:
    """Compile one single-port block for FDTD state-space feedback."""

    from ..scene import prepare_scene

    if not isinstance(block, NetworkBlock):
        raise TypeError("compile_network_block expects a NetworkBlock.")
    if len(block.port_order) != 1:
        raise NotImplementedError(
            "Phase 2 network compilation supports one port; multiport implicit coupling is Phase 3."
        )
    resolved_scene = prepare_scene(scene)
    target_device = torch.device(resolved_scene.device if device is None else device)
    ports_by_name = {port.name: port for port in resolved_scene.ports}
    compiled_ports: list[CompiledPortGeometry] = []
    for connection_name in block.connected_port_names:
        try:
            port = ports_by_name[connection_name]
        except KeyError as exc:
            raise ValueError(
                f"Network {block.name!r} references unknown Scene port {connection_name!r}."
            ) from exc
        if isinstance(port, WavePort):
            raise ValueError(
                f"Network {block.name!r} cannot connect to WavePort {connection_name!r}; "
                "time-domain terminal injection is unavailable."
            )
        if not isinstance(port, (LumpedPort, TerminalPort)):
            raise TypeError(
                f"Network {block.name!r} requires LumpedPort or resolved TerminalPort targets."
            )
        compiled_ports.append(compile_port_geometry(resolved_scene, port, device=target_device))

    continuous, certificate_model = _resolve_continuous_model(
        block,
        device=target_device,
    )
    if continuous.representation != "Y":
        raise ValueError("FDTD embedded networks require an admittance (Y) realization.")
    if continuous.input_count != 1 or continuous.output_count != 1:
        raise ValueError("The compiled state-space dimensions must match the one-port block.")
    if continuous.state_count and bool(
        torch.any(torch.linalg.eigvals(continuous.A.clone()).real >= 0.0)
    ):
        raise ValueError("FDTD embedded networks require a strictly stable continuous model.")

    frequency_band = _effective_frequency_band(block, continuous)
    source_frequencies = block.network.frequencies.to(
        device=target_device,
        dtype=continuous.A.dtype,
    )
    in_band = source_frequencies[
        (source_frequencies >= frequency_band[0])
        & (source_frequencies <= frequency_band[1])
    ]
    endpoints = torch.tensor(
        frequency_band,
        dtype=continuous.A.dtype,
        device=target_device,
    )
    certificate_frequencies = torch.unique(
        torch.cat((in_band, endpoints)),
        sorted=True,
    )
    tolerance = block.fit.passivity_tolerance if block.fit is not False else 1.0e-9
    passivity = certificate_model.check_passivity(
        certificate_frequencies,
        tolerance=tolerance,
    )
    if not passivity.passive or not passivity.certified:
        raise ValueError(
            "FDTD embedded networks require certified passivity over the full band; "
            f"maximum violation is {passivity.max_violation:.6g}, "
            f"certificate={passivity.certified}."
        )

    discrete = continuous.discretize(float(dt))
    model_id = str(block.network.metadata.get("model_id", ""))
    if not model_id:
        model_id = f"{block.name}:Y:{continuous.state_count}"
    return CompiledNetworkBlock(
        name=block.name,
        port_order=block.port_order,
        connection_names=block.connected_port_names,
        ports=tuple(compiled_ports),
        continuous=continuous,
        discrete=discrete,
        fit_report=continuous.report,
        model_id=model_id,
        frequency_band=frequency_band,
        extrapolation=block.extrapolation,
    )


def compile_networks(
    scene,
    networks=None,
    *,
    dt: float,
    device: str | torch.device | None = None,
) -> tuple[CompiledNetworkBlock, ...]:
    """Compile Scene network declarations in stable scene order."""

    selected = tuple(scene.networks if networks is None else networks)
    return tuple(
        compile_network_block(scene, block, dt=dt, device=device)
        for block in selected
    )


__all__ = ["CompiledNetworkBlock", "compile_network_block", "compile_networks"]

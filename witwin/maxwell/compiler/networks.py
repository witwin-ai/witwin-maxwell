from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..network import NetworkBlock, NetworkData
from ..ports import LumpedPort, TerminalPort, WavePort
from ..rational import (
    DiscreteStateSpaceNetwork,
    FitReport,
    RationalModel,
    StateSpaceNetwork,
)
from .ports import CompiledPortGeometry, compile_port_geometry
from .delay import (
    CompiledNetworkDelay,
    compile_network_delay,
    reembed_scattering,
)


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
    delay: CompiledNetworkDelay | None = None
    reference_impedance: torch.Tensor | None = None

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
    dt: float,
    device: torch.device,
) -> tuple[
    StateSpaceNetwork,
    RationalModel | StateSpaceNetwork,
    CompiledNetworkDelay | None,
    torch.Tensor | None,
]:
    representation = "Y"
    delay: CompiledNetworkDelay | None = None
    reference_impedance: torch.Tensor | None = None
    fit_network = block.network
    if block.delay_seconds is not None:
        representation = "S"
        band = None if block.fit is False else block.fit.band
        delay, core_scattering = compile_network_delay(
            block.network,
            block.delay_seconds,
            dt=dt,
            max_delay_steps=block.max_delay_steps,
            band=band,
        )
        fit_network = NetworkData(
            frequencies=block.network.frequencies,
            s=core_scattering,
            z0=block.network.z0,
            port_names=block.network.port_names,
            valid_columns=block.network.valid_columns,
            metadata=block.network.metadata,
            phasor_convention=block.network.phasor_convention,
            power_wave_convention=block.network.power_wave_convention,
        )
        z0 = block.network.z0
        tolerance = 256.0 * torch.finfo(z0.real.dtype).eps
        scale = max(float(torch.max(torch.abs(z0)).item()), 1.0)
        if float(torch.max(torch.abs(z0.imag)).item()) > tolerance * scale:
            raise ValueError("Explicit time-domain delay requires real reference impedances.")
        real_z0 = z0.real
        if not torch.allclose(
            real_z0,
            real_z0[0].expand_as(real_z0),
            rtol=tolerance,
            atol=tolerance * scale,
        ):
            raise ValueError(
                "Explicit time-domain delay requires frequency-independent reference impedances."
            )
        if not bool(torch.all(real_z0[0] > 0.0)):
            raise ValueError("Explicit time-domain delay requires positive reference impedances.")
        reference_impedance = real_z0[0].to(device=device)
    if block.model is None:
        model = fit_network.fit_rational(block.fit, representation=representation)
        model = _move_rational(model, device)
        return (
            model.to_state_space(port_order=block.port_order),
            model,
            delay,
            reference_impedance,
        )
    if isinstance(block.model, RationalModel):
        model = _move_rational(block.model, device)
        return (
            model.to_state_space(port_order=block.port_order),
            model,
            delay,
            reference_impedance,
        )
    model = _move_state_space(block.model, device)
    return model, model, delay, reference_impedance


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
    """Compile one fixed-size N-port block for FDTD state-space feedback."""

    from ..scene import prepare_scene

    if not isinstance(block, NetworkBlock):
        raise TypeError("compile_network_block expects a NetworkBlock.")
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
                f"Network {block.name!r} cannot connect to WavePort {connection_name!r}: "
                "an embedded state-space network couples through a scalar "
                "voltage/current terminal on a single lumped Yee edge "
                "(LumpedPort or resolved TerminalPort), but a WavePort is a modal "
                "port defined by a cross-sectional mode-overlap field pattern with "
                "no scalar time-domain terminal (V, I) contract. This is a missing "
                "design contract, not a bug; use a LumpedPort or TerminalPort "
                "terminal for embedded-network connections."
            )
        if not isinstance(port, (LumpedPort, TerminalPort)):
            raise TypeError(
                f"Network {block.name!r} requires LumpedPort or resolved TerminalPort targets."
            )
        compiled_ports.append(compile_port_geometry(resolved_scene, port, device=target_device))

    continuous, certificate_model, delay, reference_impedance = _resolve_continuous_model(
        block,
        dt=float(dt),
        device=target_device,
    )
    expected_representation = "S" if delay is not None else "Y"
    if continuous.representation != expected_representation:
        raise ValueError(
            f"FDTD embedded networks require a {expected_representation} realization "
            "for the selected delay mode."
        )
    port_count = len(block.port_order)
    if (
        continuous.input_count != port_count
        or continuous.output_count != port_count
    ):
        raise ValueError(
            "The compiled state-space input/output dimensions must match the network port count."
        )
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

    if delay is not None:
        selected = (
            (source_frequencies >= frequency_band[0])
            & (source_frequencies <= frequency_band[1])
        )
        fitted_core = continuous.evaluate(source_frequencies[selected])
        fitted = reembed_scattering(
            fitted_core,
            source_frequencies[selected],
            delay.delay_seconds,
        )
        target = block.network.s.to(device=target_device)[selected]
        reembedding_error = float(torch.max(torch.abs(fitted - target)).item())
        if reembedding_error > 0.02:
            raise ValueError(
                f"Delayed network re-embedding error {reembedding_error:.6g} exceeds 0.02."
            )
        delay = replace(delay, reembedding_max_error=reembedding_error)
        if continuous.report is not None:
            updated_report = delay.update_report(
                continuous.report,
                port_count=port_count,
            )
            continuous = replace(continuous, report=updated_report)
            certificate_model = replace(certificate_model, report=updated_report)

    discrete = continuous.discretize(float(dt))
    model_id = str(block.network.metadata.get("model_id", ""))
    if not model_id:
        model_id = f"{block.name}:{expected_representation}:{continuous.state_count}"
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
        delay=delay,
        reference_impedance=reference_impedance,
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

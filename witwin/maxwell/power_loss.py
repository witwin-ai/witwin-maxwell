from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch


_ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")


def _frozen_mapping(values: Mapping) -> Mapping:
    return MappingProxyType(dict(values))


def _validate_frequencies(frequencies: torch.Tensor) -> torch.Tensor:
    if not isinstance(frequencies, torch.Tensor):
        raise TypeError("frequencies must be a torch.Tensor.")
    if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
        raise TypeError("frequencies must be a real floating-point tensor.")
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("frequencies must contain only finite values.")
    if not bool(torch.all(frequencies > 0.0)):
        raise ValueError("frequencies must be strictly positive.")
    return frequencies


def _validate_optional_component_metadata(
    values,
    *,
    name: str,
    component_volumes: Mapping[str, torch.Tensor],
    device: torch.device,
    is_occupancy: bool,
) -> Mapping[str, torch.Tensor] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a component mapping or None.")
    resolved = {}
    for component, value in values.items():
        if component not in component_volumes:
            raise ValueError(f"{name} contains unknown component {component!r}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}[{component!r}] must be a torch.Tensor.")
        if value.device != device:
            raise ValueError(f"{name} tensors must be on device {device}.")
        if value.shape != component_volumes[component].shape:
            raise ValueError(
                f"{name}[{component!r}] must match the selected component shape."
            )
        if is_occupancy:
            if value.is_complex() or not value.dtype.is_floating_point:
                raise TypeError(
                    f"{name}[{component!r}] must be real floating-point data."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"{name}[{component!r}] must contain only finite values."
                )
            if not bool(torch.all((value >= 0.0) & (value <= 1.0))):
                raise ValueError(f"{name}[{component!r}] must lie in [0, 1].")
        elif (
            value.dtype == torch.bool
            or value.dtype.is_floating_point
            or value.is_complex()
        ):
            raise TypeError(f"{name}[{component!r}] must contain integer identifiers.")
        resolved[str(component)] = value
    return _frozen_mapping(resolved)


def _validate_measure_density(
    densities,
    measures,
    *,
    density_name: str,
    measure_name: str,
    frequency_count: int,
    channel_power: Mapping[str, torch.Tensor],
    device: torch.device,
):
    if not isinstance(densities, Mapping) or not isinstance(measures, Mapping):
        raise TypeError(f"{density_name} and {measure_name} must be mappings.")
    resolved_measures = {}
    for entity, measure in measures.items():
        entity_name = str(entity)
        if not entity_name:
            raise ValueError(f"{measure_name} keys must not be empty.")
        if not isinstance(measure, torch.Tensor):
            raise TypeError(f"{measure_name}[{entity_name!r}] must be a torch.Tensor.")
        if measure.device != device:
            raise ValueError("All PowerLossData tensors must be on one device.")
        if measure.ndim != 1 or measure.numel() == 0:
            raise ValueError(f"{measure_name}[{entity_name!r}] must have shape [N].")
        if measure.is_complex() or not measure.dtype.is_floating_point:
            raise TypeError(f"{measure_name} values must be real floating-point data.")
        if not bool(torch.all(torch.isfinite(measure))) or not bool(torch.all(measure > 0.0)):
            raise ValueError(f"{measure_name} values must be finite and positive.")
        resolved_measures[entity_name] = measure

    resolved_densities = {}
    referenced_entities = set()
    for channel, entity_map in densities.items():
        channel_name = str(channel)
        if channel_name not in channel_power:
            raise ValueError(f"{density_name} contains unavailable channel {channel_name!r}.")
        if not isinstance(entity_map, Mapping) or not entity_map:
            raise ValueError(f"{density_name}[{channel_name!r}] must be a non-empty mapping.")
        resolved_entities = {}
        for entity, density in entity_map.items():
            entity_name = str(entity)
            if entity_name not in resolved_measures:
                raise ValueError(
                    f"{density_name}[{channel_name!r}] references unknown entity {entity_name!r}."
                )
            if not isinstance(density, torch.Tensor):
                raise TypeError(
                    f"{density_name}[{channel_name!r}][{entity_name!r}] must be a torch.Tensor."
                )
            expected_shape = (frequency_count, resolved_measures[entity_name].numel())
            if density.shape != expected_shape:
                raise ValueError(
                    f"{density_name}[{channel_name!r}][{entity_name!r}] must have shape [F, N]."
                )
            if density.device != device:
                raise ValueError("All PowerLossData tensors must be on one device.")
            if density.is_complex() or not density.dtype.is_floating_point:
                raise TypeError(f"{density_name} values must be real floating-point data.")
            if not bool(torch.all(torch.isfinite(density))) or not bool(torch.all(density >= 0.0)):
                raise ValueError(f"{density_name} values must be finite and non-negative.")
            referenced_entities.add(entity_name)
            resolved_entities[entity_name] = density
        resolved_densities[channel_name] = _frozen_mapping(resolved_entities)
    unused = set(resolved_measures) - referenced_entities
    if unused:
        raise ValueError(f"{measure_name} contains unreferenced entities {tuple(sorted(unused))}.")
    return _frozen_mapping(resolved_densities), _frozen_mapping(resolved_measures)


@dataclass(frozen=True)
class PowerLossData:
    """Device-resident loss channels on selected Yee electric edges.

    Integrated channel powers and ``total`` have strict ``[F]`` shape. Volume
    densities use ``[F, N_component]`` after sparse monitor selection and retain
    the matching edge control volumes and global Yee identifiers.
    """

    frequencies: torch.Tensor
    channel_power: Mapping[str, torch.Tensor]
    volume_density: Mapping[str, Mapping[str, torch.Tensor]]
    total: torch.Tensor
    component_volumes: Mapping[str, torch.Tensor]
    global_ids: Mapping[str, torch.Tensor]
    monitor_name: str
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    occupancy: Mapping[str, torch.Tensor] | None = None
    material_ids: Mapping[str, torch.Tensor] | None = None
    geometry_ids: Mapping[str, torch.Tensor] | None = None
    surface_density: Mapping[str, Mapping[str, torch.Tensor]] = field(default_factory=dict)
    face_areas: Mapping[str, torch.Tensor] = field(default_factory=dict)
    line_density: Mapping[str, Mapping[str, torch.Tensor]] = field(default_factory=dict)
    line_lengths: Mapping[str, torch.Tensor] = field(default_factory=dict)
    source_result_fingerprint: str | None = None
    autograd_provenance: str = "live graph is preserved from supplied torch tensors; persisted data is detached"
    power_unit: str = "W"
    volume_density_unit: str = "W/m^3"
    surface_density_unit: str = "W/m^2"
    line_density_unit: str = "W/m"
    volume_unit: str = "m^3"
    area_unit: str = "m^2"
    length_unit: str = "m"
    collocation: str = "Yee electric edges with component dual control volumes"
    normalization: str = (
        "peak phasor; time-average static electric conduction density "
        "q=0.5*sigma_e*|E|^2; total is the sum of explicitly available channels"
    )
    phasor_convention: str = "peak phasor with exp(-i*omega*t) time dependence"

    def __post_init__(self):
        frequencies = _validate_frequencies(self.frequencies)
        device = frequencies.device
        frequency_count = frequencies.numel()

        if (
            not isinstance(self.component_volumes, Mapping)
            or not self.component_volumes
        ):
            raise ValueError("component_volumes must be a non-empty component mapping.")
        component_volumes = {}
        for component, volume in self.component_volumes.items():
            if component not in _ELECTRIC_COMPONENTS:
                raise ValueError(f"Unsupported loss-density component {component!r}.")
            if not isinstance(volume, torch.Tensor):
                raise TypeError(
                    f"component_volumes[{component!r}] must be a torch.Tensor."
                )
            if volume.device != device:
                raise ValueError("All PowerLossData tensors must be on one device.")
            if volume.ndim != 1 or volume.numel() == 0:
                raise ValueError(
                    f"component_volumes[{component!r}] must have non-empty shape [N]."
                )
            if volume.is_complex() or not volume.dtype.is_floating_point:
                raise TypeError(
                    f"component_volumes[{component!r}] must be real floating-point data."
                )
            if not bool(torch.all(torch.isfinite(volume))) or not bool(
                torch.all(volume > 0.0)
            ):
                raise ValueError(
                    f"component_volumes[{component!r}] must be finite and positive."
                )
            component_volumes[str(component)] = volume

        if set(self.global_ids) != set(component_volumes):
            raise ValueError(
                "global_ids and component_volumes must contain identical components."
            )
        global_ids = {}
        all_global_ids = []
        for component, volume in component_volumes.items():
            identifiers = self.global_ids[component]
            if not isinstance(identifiers, torch.Tensor):
                raise TypeError(f"global_ids[{component!r}] must be a torch.Tensor.")
            if identifiers.device != device:
                raise ValueError("All PowerLossData tensors must be on one device.")
            if identifiers.shape != volume.shape:
                raise ValueError(
                    f"global_ids[{component!r}] must match its component volume shape."
                )
            if (
                identifiers.dtype == torch.bool
                or identifiers.dtype.is_floating_point
                or identifiers.is_complex()
            ):
                raise TypeError(
                    f"global_ids[{component!r}] must contain integer identifiers."
                )
            global_ids[component] = identifiers
            all_global_ids.append(identifiers)
        concatenated_ids = torch.cat(all_global_ids)
        if torch.unique(concatenated_ids).numel() != concatenated_ids.numel():
            raise ValueError(
                "global_ids must be unique across all selected components."
            )

        if not isinstance(self.channel_power, Mapping) or not self.channel_power:
            raise ValueError(
                "channel_power must contain at least one available loss channel."
            )
        channel_power = {}
        for channel, power in self.channel_power.items():
            channel_name = str(channel)
            if not channel_name:
                raise ValueError("Loss channel names must not be empty.")
            if not isinstance(power, torch.Tensor):
                raise TypeError(
                    f"channel_power[{channel_name!r}] must be a torch.Tensor."
                )
            if power.device != device:
                raise ValueError("All PowerLossData tensors must be on one device.")
            if power.shape != (frequency_count,):
                raise ValueError(
                    f"channel_power[{channel_name!r}] must have shape [F]."
                )
            if power.is_complex() or not power.dtype.is_floating_point:
                raise TypeError(
                    f"channel_power[{channel_name!r}] must be real floating-point data."
                )
            if not bool(torch.all(torch.isfinite(power))) or not bool(
                torch.all(power >= 0.0)
            ):
                raise ValueError(
                    f"channel_power[{channel_name!r}] must be finite and non-negative."
                )
            channel_power[channel_name] = power

        if not isinstance(self.volume_density, Mapping):
            raise TypeError("volume_density must be a channel mapping.")
        volume_density = {}
        for channel, component_map in self.volume_density.items():
            channel_name = str(channel)
            if channel_name not in channel_power:
                raise ValueError(
                    f"volume_density contains unavailable channel {channel_name!r}."
                )
            if not isinstance(component_map, Mapping) or not component_map:
                raise ValueError(
                    f"volume_density[{channel_name!r}] must be a non-empty component mapping."
                )
            resolved_components = {}
            for component, density in component_map.items():
                if component not in component_volumes:
                    raise ValueError(
                        f"volume_density[{channel_name!r}] contains unknown component {component!r}."
                    )
                if not isinstance(density, torch.Tensor):
                    raise TypeError(
                        f"volume_density[{channel_name!r}][{component!r}] must be a torch.Tensor."
                    )
                expected_shape = (frequency_count, component_volumes[component].numel())
                if density.shape != expected_shape:
                    raise ValueError(
                        f"volume_density[{channel_name!r}][{component!r}] must have shape [F, N]."
                    )
                if density.device != device:
                    raise ValueError("All PowerLossData tensors must be on one device.")
                if density.is_complex() or not density.dtype.is_floating_point:
                    raise TypeError(
                        "Volume loss density must be real floating-point data."
                    )
                if not bool(torch.all(torch.isfinite(density))) or not bool(
                    torch.all(density >= 0.0)
                ):
                    raise ValueError(
                        "Volume loss density must be finite and non-negative."
                    )
                resolved_components[str(component)] = density
            volume_density[channel_name] = _frozen_mapping(resolved_components)

        if not isinstance(self.total, torch.Tensor) or self.total.shape != (
            frequency_count,
        ):
            raise ValueError("total must be a torch.Tensor with shape [F].")
        if self.total.device != device:
            raise ValueError("All PowerLossData tensors must be on one device.")
        expected_total = torch.stack(tuple(channel_power.values()), dim=0).sum(dim=0)
        if not torch.allclose(self.total, expected_total):
            raise ValueError(
                "total must equal the sum of all available channel powers."
            )

        monitor_name = str(self.monitor_name)
        if not monitor_name:
            raise ValueError("monitor_name must not be empty.")
        if len(self.bounds) != 3 or any(
            len(axis_bounds) != 2 for axis_bounds in self.bounds
        ):
            raise ValueError("bounds must contain three (lower, upper) pairs.")

        occupancy = _validate_optional_component_metadata(
            self.occupancy,
            name="occupancy",
            component_volumes=component_volumes,
            device=device,
            is_occupancy=True,
        )
        material_ids = _validate_optional_component_metadata(
            self.material_ids,
            name="material_ids",
            component_volumes=component_volumes,
            device=device,
            is_occupancy=False,
        )
        geometry_ids = _validate_optional_component_metadata(
            self.geometry_ids,
            name="geometry_ids",
            component_volumes=component_volumes,
            device=device,
            is_occupancy=False,
        )
        surface_density, face_areas = _validate_measure_density(
            self.surface_density,
            self.face_areas,
            density_name="surface_density",
            measure_name="face_areas",
            frequency_count=frequency_count,
            channel_power=channel_power,
            device=device,
        )
        line_density, line_lengths = _validate_measure_density(
            self.line_density,
            self.line_lengths,
            density_name="line_density",
            measure_name="line_lengths",
            frequency_count=frequency_count,
            channel_power=channel_power,
            device=device,
        )
        fingerprint = self.source_result_fingerprint
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("source_result_fingerprint must be a non-empty string or None.")
        if not isinstance(self.autograd_provenance, str) or not self.autograd_provenance:
            raise ValueError("autograd_provenance must be a non-empty string.")

        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "channel_power", _frozen_mapping(channel_power))
        object.__setattr__(self, "volume_density", _frozen_mapping(volume_density))
        object.__setattr__(
            self, "component_volumes", _frozen_mapping(component_volumes)
        )
        object.__setattr__(self, "global_ids", _frozen_mapping(global_ids))
        object.__setattr__(self, "monitor_name", monitor_name)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "material_ids", material_ids)
        object.__setattr__(self, "geometry_ids", geometry_ids)
        object.__setattr__(self, "surface_density", surface_density)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "line_density", line_density)
        object.__setattr__(self, "line_lengths", line_lengths)

    @property
    def device(self) -> torch.device:
        return self.frequencies.device

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.channel_power)

    @property
    def conduction(self) -> torch.Tensor | None:
        return self.channel_power.get("conduction")

    @property
    def surface(self) -> torch.Tensor | None:
        return self.channel_power.get("surface")

    @property
    def wire(self) -> torch.Tensor | None:
        return self.channel_power.get("wire")

    def channel(self, name: str) -> torch.Tensor:
        channel_name = str(name)
        if channel_name not in self.channel_power:
            raise KeyError(
                f"Power-loss channel {channel_name!r} is unavailable; choices are {self.channels}."
            )
        return self.channel_power[channel_name]

    def density(self, channel: str, component: str) -> torch.Tensor:
        channel_name = str(channel)
        component_name = str(component)
        if channel_name not in self.volume_density:
            raise KeyError(
                f"Power-loss channel {channel_name!r} has no volume-density data."
            )
        if component_name not in self.volume_density[channel_name]:
            choices = tuple(self.volume_density[channel_name])
            raise KeyError(
                f"Power-loss channel {channel_name!r} has no {component_name!r} density; "
                f"choices are {choices}."
            )
        return self.volume_density[channel_name][component_name]


__all__ = ["PowerLossData"]

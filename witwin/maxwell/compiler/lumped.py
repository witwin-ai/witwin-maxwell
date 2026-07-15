from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from ..lumped import Capacitor, Inductor, Resistor
from ..scene import prepare_scene
from .ports import _compile_voltage_path, _integrate_sparse_term


_LUMPED_ELEMENT_TYPES = (Resistor, Capacitor, Inductor)


@dataclass(frozen=True)
class CompiledLumpedElement:
    """Sparse Yee-grid voltage geometry for one passive two-terminal element."""

    element_name: str
    kind: str
    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    axis: str
    direction: int
    voltage_component: str
    voltage_indices: torch.Tensor
    voltage_weights: torch.Tensor
    value: torch.Tensor

    @property
    def port_name(self) -> str:
        """Compatibility name used by the shared lumped field runtime."""

        return self.element_name

    def integrate_voltage(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return _integrate_sparse_term(
            fields,
            self.voltage_component,
            self.voltage_indices,
            self.voltage_weights,
        )


def _element_axis(element: Resistor | Capacitor | Inductor) -> str:
    differing_axes = tuple(
        axis
        for axis, positive, negative in zip("xyz", element.positive, element.negative)
        if not math.isclose(positive, negative, rel_tol=0.0, abs_tol=1.0e-12)
    )
    if len(differing_axes) != 1:
        raise ValueError(
            f"Lumped element {element.name!r} terminals must define an axis-aligned path."
        )
    return differing_axes[0]


def compile_lumped_elements(
    scene,
    *,
    device: str | torch.device | None = None,
) -> tuple[CompiledLumpedElement, ...]:
    """Compile all passive two-terminal elements into sparse voltage weights."""

    resolved_scene = prepare_scene(scene)
    target_device = torch.device(resolved_scene.device if device is None else device)
    compiled = []
    for element in resolved_scene.lumped_elements:
        if not isinstance(element, _LUMPED_ELEMENT_TYPES):
            raise TypeError(
                "Lumped elements must be Resistor, Capacitor, or Inductor instances."
            )
        axis = _element_axis(element)
        direction, component, indices, weights = _compile_voltage_path(
            resolved_scene,
            positive=element.positive,
            negative=element.negative,
            axis=axis,
            device=target_device,
        )
        compiled.append(
            CompiledLumpedElement(
                element_name=element.name,
                kind=element.kind,
                positive=element.positive,
                negative=element.negative,
                axis=axis,
                direction=direction,
                voltage_component=component,
                voltage_indices=indices,
                voltage_weights=weights,
                value=element.value.to(device=target_device),
            )
        )
    return tuple(compiled)


__all__ = ["CompiledLumpedElement", "compile_lumped_elements"]

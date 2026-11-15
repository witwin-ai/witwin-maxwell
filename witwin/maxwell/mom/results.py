"""Typed MoM result data: PointMonitor samples as ``FieldData``, far fields as ``FarFieldData``.

MoM values are peak phasors with ``exp(-i*omega*t)`` time dependence for the
declared source amplitudes (``normalization="raw"``). A PointMonitor samples
the total E/H at one physical position and becomes one ``FieldData`` with dims
``("component", "f")``: the components are the monitor's requested fields in
declaration order, and ``f`` has length one for a linear run or lists the
retained positive harmonics ``n*f0`` of a coupled harmonic run. Far fields over
a list of unit directions become ``FarFieldData`` with dims
``("f", "direction", "component")``. Values are the computed tensors; autograd
flows.
"""

from __future__ import annotations

import torch

from ..results import FarFieldData, FieldData, field_component_units

PHASOR_ATTRS = {"phasor_convention": "exp(-i omega t), peak", "normalization": "raw"}
POINT_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _frequency_coordinate(frequencies) -> torch.Tensor:
    return torch.tensor(tuple(frequencies), dtype=torch.float64)


def point_monitor_data(monitor, samples: torch.Tensor, frequencies, **attrs) -> FieldData:
    """``FieldData`` of one PointMonitor from ``samples[f, 6]`` ordered ``Ex..Hz``."""

    components = tuple(name.capitalize() for name in monitor.fields)
    units, component_units = field_component_units(components)
    indices = [POINT_COMPONENTS.index(name) for name in components]
    return FieldData(
        values=samples[:, indices].transpose(0, 1),
        dims=("component", "f"),
        coords={"component": components, "f": _frequency_coordinate(frequencies)},
        units=units,
        attrs={**PHASOR_ATTRS, "component_units": component_units, "position": monitor.position,
               "field_kind": "total", "sampling": "physical_position", **attrs},
    )


def far_field_data(electric: torch.Tensor, magnetic: torch.Tensor, directions: torch.Tensor,
                   frequency: float) -> tuple[FarFieldData, FarFieldData]:
    """Outgoing amplitudes ``F_E [V]``, ``F_H [A]`` (``[D, 3]`` each, ``E ~ exp(ikr) F_E / r``) as ``FarFieldData``.

    Each is dims ``("f", "direction", "component")`` with ``f`` of length one;
    the values are views of the computed tensors and the unit directions live in
    ``attrs["ux"]``, ``attrs["uy"]``, ``attrs["uz"]``. The phase is referenced to
    the global coordinate origin.
    """

    f = _frequency_coordinate((frequency,))
    return tuple(
        FarFieldData.from_directions(values[None], directions, f, components, units,
                                     attrs={**PHASOR_ATTRS, "quantity": quantity, "phase_origin": (0.0, 0.0, 0.0)})
        for values, components, units, quantity in (
            (electric, ("Ex", "Ey", "Ez"), "V", "r*exp(-ikr)*E"),
            (magnetic, ("Hx", "Hy", "Hz"), "A", "r*exp(-ikr)*H"),
        )
    )

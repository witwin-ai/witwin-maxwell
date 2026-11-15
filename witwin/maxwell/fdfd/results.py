"""Typed FDFD result data: Yee electric fields and linear observations.

FDFD fields and observations are peak phasors with ``exp(-i*omega*t)`` time
dependence, solved for the declared source amplitudes (``normalization="raw"``).
Every value keeps a leading ``f`` dimension, of length one for a single-frequency
run, so single runs and sweeps share one layout.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..fdfd_response import FarFieldProjection
from ..results import FarFieldData, FieldData, field_component_units

PHASOR_ATTRS = {"phasor_convention": "exp(-i omega t), peak", "normalization": "raw"}

# Each electric component lives on the edge centres along its own axis: half-cell
# coordinates on that axis, primal nodes on the other two.
_YEE_COORDINATES = {
    "EX": ("x_half64", "y_nodes64", "z_nodes64"),
    "EY": ("x_nodes64", "y_half64", "z_nodes64"),
    "EZ": ("x_nodes64", "y_nodes64", "z_half64"),
}


def _frequency_coordinate(frequencies) -> torch.Tensor:
    return torch.tensor(tuple(frequencies), dtype=torch.float64)


def field_data(prepared_scene, frequencies, fields) -> dict[str, FieldData]:
    """``{"EX", "EY", "EZ"}`` Yee components as ``FieldData`` with dims ``(f, x, y, z)``.

    The components sit on different Yee grids, so each is its own ``FieldData``
    with its native coordinates. Values are views of the stored tensors.
    """

    f = _frequency_coordinate(frequencies)
    typed = {}
    for name, tensor in fields.items():
        coordinates = (torch.from_numpy(getattr(prepared_scene, attribute)) for attribute in _YEE_COORDINATES[name])
        component = name[0] + name[1].lower()
        units, component_units = field_component_units((component,))
        typed[name] = FieldData(
            values=tensor.reshape(f.numel(), *tensor.shape[-3:]),
            dims=("f", "x", "y", "z"),
            coords=dict(zip(("f", "x", "y", "z"), (f, *coordinates))),
            units=units,
            attrs={**PHASOR_ATTRS, "component": component, "component_units": component_units,
                   "yee_location": f"edge_{name[1].lower()}"},
        )
    return typed


def observation_data(observations, values: torch.Tensor, frequency: float) -> dict:
    """One ``FieldData``/``FarFieldData`` per observation, dims ``("f",)``, from ``[observation]`` values."""

    f = _frequency_coordinate((frequency,))
    return {spec.name: _observation(spec, values[index:index + 1], f) for index, spec in enumerate(observations)}


def _observation(spec, values, f):
    if isinstance(spec, FarFieldProjection):
        # One direction (D=1): the unit vector is attrs ux/uy/uz; the value projects onto `polarization`.
        return FarFieldData.from_directions(
            values.reshape(-1, 1), torch.tensor((tuple(spec.direction),), dtype=torch.float64), f, None, "V",
            attrs={**PHASOR_ATTRS, "quantity": "r*exp(-ikr)*E", "polarization": tuple(spec.polarization),
                   "bounds": tuple(map(tuple, spec.bounds))},
        )
    if spec.path is not None:
        units, attrs = "V", {"projection": "voltage", "path": tuple(map(tuple, spec.path))}
    elif spec.position is not None:
        units, component_units = field_component_units((spec.component,))
        attrs = {"projection": "point", "component": spec.component, "component_units": component_units,
                 "position": tuple(spec.position)}
    else:
        # Weighted indices: the weights define the quantity and its units.
        units, attrs = "1", {"projection": "weighted"}
    return FieldData(values=values, dims=("f",), coords={"f": f}, units=units, attrs={**PHASOR_ATTRS, **attrs})


def concat_frequencies(items, frequencies):
    """Join single-frequency typed data along ``f``; autograd flows through ``torch.cat``."""

    return replace(items[0], values=torch.cat([item.values for item in items]),
                   coords={**items[0].coords, "f": _frequency_coordinate(frequencies)})

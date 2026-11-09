"""Passive electric/magnetic channels of the finite-copper two-port.

This module supplies constitutive coefficients only. Native curved field
coupling retains its explicit guard until the joint mortar solve is qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..copper_foil import CopperFoil, MU_0
from .copper_foil import _truncated_copper_response


@dataclass(frozen=True)
class CompiledCopperChannels:
    rates: torch.Tensor
    electric_residue: torch.Tensor
    magnetic_residue: torch.Tensor
    electric_direct: torch.Tensor
    magnetic_inductance: torch.Tensor
    frequency_range: tuple[float, float]
    relative_error_bound: float

    def evaluate(self, frequencies):
        frequency = torch.as_tensor(frequencies, dtype=torch.float64, device=self.rates.device)
        if frequency.ndim == 0:
            frequency = frequency.reshape(1)
        if frequency.ndim != 1 or not frequency.numel() or not bool(torch.isfinite(frequency).all()):
            raise ValueError("Copper channel frequencies require a finite nonempty vector.")
        if not bool(((frequency >= self.frequency_range[0]) & (frequency <= self.frequency_range[1])).all()):
            raise ValueError("Copper channel frequencies are outside the certified band.")
        s = torch.complex(torch.zeros_like(frequency), -2*math.pi*frequency)
        denominator = s[:, None]+self.rates[None]
        electric = self.electric_direct+(self.electric_residue/denominator).sum(-1)
        magnetic = s*self.magnetic_inductance+(self.magnetic_residue*s[:, None]/denominator).sum(-1)
        return electric, magnetic

    def midpoint_coefficients(self, dt):
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("Copper channel dt must be finite and positive.")
        denominator = 1+self.rates*dt/2
        return dict(decay=(1-self.rates*dt/2)/denominator,
            electric_drive=dt*self.electric_residue.sqrt()/denominator,
            magnetic_drive=dt*(self.magnetic_residue*self.rates).sqrt()/denominator,
            electric_history=self.electric_residue.sqrt()/denominator,
            magnetic_history=-(self.magnetic_residue*self.rates).sqrt()/denominator,
            electric_gain=self.electric_direct+(dt*self.electric_residue/(2*denominator)).sum(),
            magnetic_gain=(self.magnetic_residue/denominator).sum())


def compile_copper_channels(material: CopperFoil, *, tolerance=2e-4, max_modes=256):
    """Transform the certified slab model into two separate positive-real ports.

    The electric Stieltjes tail becomes a positive magnetic inductance after
    multiplication by s*mu/sigma. It must be included in the field mass, not
    discarded or fitted into a dissipative coefficient.
    """
    # The channel residues and tail require the exact diffusion lattice.
    parameter_tensors = [value for value in (material.thickness, material.conductivity) if torch.is_tensor(value)]
    device = parameter_tensors[0].device if parameter_tensors else torch.device("cpu")
    response = _truncated_copper_response(material, tolerance, max_modes, device)
    count = (len(response.rates)-1)//2
    rates = response.rates[:count]
    thickness = torch.as_tensor(material.thickness, dtype=torch.float64, device=rates.device)
    conductivity = torch.as_tensor(material.conductivity, dtype=torch.float64, device=rates.device)
    electric_residue = torch.ones_like(rates)*(8/(MU_0*thickness))
    electric_direct = conductivity*thickness-(electric_residue/rates).sum()
    if not bool(torch.isfinite(electric_direct)) or bool(electric_direct < 0):
        raise RuntimeError("The analytic copper-channel tail must be nonnegative without clipping.")
    scale = MU_0/conductivity
    return CompiledCopperChannels(rates=rates, electric_residue=electric_residue,
        magnetic_residue=scale*electric_residue, electric_direct=electric_direct,
        magnetic_inductance=scale*electric_direct, frequency_range=material.frequency_range,
        relative_error_bound=response.relative_error_bound)


def compile_copper_yee_response(integrated, nodes):
    """Expand electric support for the exact native curl of magnetic reaction.

    Returns the physical volume-weighted curl transpose, not an interpolated
    surface derivative. The operator satisfies native Yee summation by parts.
    Its sparse magnetic columns contain all electric neighbors, including the
    halo needed by the magnetic-to-electric current response.
    """
    shapes_e, shapes_h = integrated["shapes"], integrated["magnetic_shapes"]
    magnetic_support = integrated["magnetic_support"]
    nodes = tuple(torch.as_tensor(axis, dtype=torch.float64, device=magnetic_support.device) for axis in nodes)
    primal = tuple(axis[1:]-axis[:-1] for axis in nodes)
    dual = tuple(torch.cat((axis[:1]/2, (axis[:-1]+axis[1:])/2, axis[-1:]/2)) for axis in primal)
    offsets_e = [0]
    for shape in shapes_e:
        offsets_e.append(offsets_e[-1]+math.prod(shape))
    rows, columns, values, offset_h = [], [], [], 0
    for component, shape in enumerate(shapes_h):
        positions = torch.nonzero((magnetic_support >= offset_h) &
                                  (magnetic_support < offset_h+math.prod(shape))).flatten()
        local = magnetic_support[positions]-offset_h
        index = torch.stack((local//(shape[1]*shape[2]), local//shape[2]%shape[1], local%shape[2]), -1)
        for electric, derivative, sign in (((component+1)%3, (component+2)%3, 1.),
                                            ((component+2)%3, (component+1)%3, -1.)):
            eshape = shapes_e[electric]
            for endpoint in (0, 1):
                target = index.clone()
                target[:, derivative] += endpoint
                flat = (target[:, 0]*eshape[1]+target[:, 1])*eshape[2]+target[:, 2]+offsets_e[electric]
                volume = torch.ones(len(target), device=local.device, dtype=torch.float64)
                for axis in range(3):
                    if axis != derivative:
                        widths = primal[axis] if axis == electric else dual[axis]
                        volume *= widths[target[:, axis]]
                rows.append(flat)
                columns.append(positions)
                values.append(volume*(sign if endpoint == 0 else -sign))
        offset_h += math.prod(shape)
    rows, columns, values = torch.cat(rows), torch.cat(columns), torch.cat(values)
    support = torch.unique(torch.cat((integrated["support"], rows)), sorted=True)
    row_local = torch.searchsorted(support, rows)
    curl = torch.sparse_coo_tensor(torch.stack((row_local, columns)), values,
                                  (len(support), len(magnetic_support))).coalesce()
    ge = integrated["matrix"]
    ge_columns = torch.searchsorted(support, integrated["support"][ge.indices()[1]])
    ge = torch.sparse_coo_tensor(torch.stack((ge.indices()[0], ge_columns)), ge.values(),
                                (ge.shape[0], len(support))).coalesce()
    if len(ge)*len(support)*8 > 512*1024**2:
        raise NotImplementedError("Expanded copper electric response exceeds the explicit 512 MiB cap.")
    return dict(electric_support=support, magnetic_support=magnetic_support,
                electric_trace=ge, magnetic_trace=integrated["magnetic_matrix"], curl_transpose=curl)

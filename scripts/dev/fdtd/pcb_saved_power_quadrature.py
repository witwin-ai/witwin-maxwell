"""Diagnose radiation quadrature using saved, unmodified spectral Yee fields."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.adapters.pcb import import_pcb_scene
from witwin.maxwell.constants import C_0, EPSILON_0, MU_0
from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
from witwin.maxwell.postprocess.stratton_chu import (
    EquivalentCurrentsSurface, equivalent_surface_currents_from_fields,
)
from scripts.dev.fdtd.pcb_application_campaign import power_observations
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.pcb_recompute_bulk_loss import digest


def interpolate_retained_plane(component, axes):
    """Interpolate retained native tangential samples, before Yee alignment."""
    values = component['data']
    if values.ndim == 3:
        values = values[0]
    brackets = []
    for native, points in zip(component['coords'], axes):
        native = torch.as_tensor(native, device=points.device, dtype=points.dtype)
        if bool((points[0] < native[0]) | (points[-1] > native[-1])):
            raise ValueError('Saved plane does not retain the requested interpolation neighbors.')
        high = torch.searchsorted(native.contiguous(), points.contiguous()).clamp(1, len(native)-1)
        low = high-1
        fraction = (points-native[low])/(native[high]-native[low])
        brackets.append(((low, 1-fraction), (high, fraction)))
    return sum(values[i[:, None], j[None, :]]*wi[:, None]*wj[None, :]
               for i, wi in brackets[0] for j, wj in brackets[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=(.0024, .0012, .0006))
    parser.add_argument('--expansions', type=float, nargs='+', default=(0.,))
    args = parser.parse_args()
    frequency = 2.45e9
    source = package_identity(mw, 'witwin-maxwell')
    input_sha256, result_sha256 = digest(args.input), digest(args.result)
    driver_sha256 = digest(Path(__file__))
    definition = json.loads(args.input.read_text())
    scene = import_pcb_scene(definition, frequency=frequency)
    observations = power_observations(definition, scene, frequency)
    scene = scene.clone(monitors=observations)
    result = mw.Result.load(args.result, scene=scene, map_location='cuda')
    if tuple(result.frequencies) != (frequency,):
        raise ValueError('This fixed-frequency application diagnostic requires only the declared 2.45 GHz spectrum.')
    original = np.asarray(observations[0].bounds)
    theta = torch.linspace(0, math.pi, 91, device='cuda', dtype=torch.float64)
    phi = torch.linspace(0, 2*math.pi, 73, device='cuda', dtype=torch.float64)
    t, p = torch.meshgrid(theta, phi, indexing='ij')
    directions = torch.stack((t.sin()*p.cos(), t.sin()*p.sin(), t.cos()), -1)
    records = []
    for expansion in args.expansions:
        bounds = original + np.array([-expansion, expansion])
        for spacing in args.spacings:
            surfaces, flux = [], 0.
            for normal in range(3):
                tangent = [i for i in range(3) if i != normal]
                axes, widths = [], []
                for dim in tangent:
                    low, high = bounds[dim]
                    count = max(2, math.ceil((high-low)/spacing))
                    width = (high-low)/count
                    axes.append(torch.tensor(low+(np.arange(count)+.5)*width, device='cuda'))
                    widths.append(width)
                a, b = torch.meshgrid(*axes, indexing='ij')
                area = widths[0]*widths[1]
                weights = torch.full_like(a, area)
                for side in range(2):
                    xyz = torch.empty((*a.shape, 3), device='cuda', dtype=torch.float64)
                    xyz[..., normal] = bounds[normal, side]
                    xyz[..., tangent[0]], xyz[..., tangent[1]] = a, b
                    if expansion == 0:
                        # Full magnetic volumes are optional; the original
                        # monitor retains the complete tangential Yee planes.
                        face = observations[0].faces[2*normal+side]
                        payload = result._monitors[face.name]
                        electric = torch.zeros((*a.shape, 3), device='cuda', dtype=torch.complex128)
                        magnetic = torch.zeros_like(electric)
                        for dim in tangent:
                            for prefix, vector in (('E', electric), ('H', magnetic)):
                                vector[..., dim] = interpolate_retained_plane(payload['components'][prefix+'xyz'[dim]], axes)
                    else:
                        electric, magnetic = result.query_fields(xyz.reshape(-1, 3), region_id='exterior', field_kind='total')
                        electric, magnetic = electric.reshape(*a.shape, 3), magnetic.reshape(*a.shape, 3)
                    sign = 2*side-1
                    flux += float(.5*(torch.cross(electric, magnetic.conj(), dim=-1)[..., normal].real*sign*area).sum())
                    fields = {prefix+axis: value[..., i] for prefix, value in (('E', electric), ('H', magnetic))
                              for i, axis in enumerate('xyz')}
                    surfaces.append(equivalent_surface_currents_from_fields(
                        axis='xyz'[normal], position=float(bounds[normal, side]), frequency=frequency,
                        u=axes[0], v=axes[1], fields=fields, normal_direction=sign,
                        quadrature_rule='cell_centered', area_weights=weights))
            transformer = NearFieldFarFieldTransformer(EquivalentCurrentsSurface(tuple(surfaces)),
                c=C_0, eps0=EPSILON_0, mu0=MU_0, device='cuda')
            far = transformer.transform_directions(directions.reshape(-1, 3), radius=10., batch_size=256)
            intensity = far['E'].abs().square().sum(-1).reshape(t.shape)*100/(2*float(transformer.eta))
            radiated = float(torch.trapezoid(torch.trapezoid(intensity*t.sin(), phi, dim=1), theta, dim=0))
            record = {'expansion_m': expansion, 'spacing_m': spacing, 'surface_flux_w': flux,
                      'radiated_w': radiated, 'radiation_vs_flux_relative': radiated/flux-1}
            records.append(record)
            print(json.dumps(record), flush=True)
    document = {'schema': 'saved-field-power-quadrature/1', 'source': source,
                'source_unchanged': source['source_sha256'] == package_identity(mw, 'witwin-maxwell')['source_sha256'],
                'driver_sha256': driver_sha256, 'driver_unchanged': driver_sha256 == digest(Path(__file__)),
                'input_sha256': input_sha256, 'input_unchanged': input_sha256 == digest(args.input),
                'result_sha256': result_sha256, 'result_unchanged': result_sha256 == digest(args.result),
                'input': str(args.input.resolve()), 'result': str(args.result.resolve()), 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2)+'\n')
    if not all(document[key] for key in ('source_unchanged', 'driver_unchanged', 'input_unchanged', 'result_unchanged')):
        raise RuntimeError('Diagnostic numerical or retained-field identity changed; evidence cannot qualify.')


if __name__ == '__main__':
    main()

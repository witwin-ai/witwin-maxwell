"""Coherent physical-drive spectra for detached line-loss diagnostics.

This benchmark sidecar binds the executed foil realization to saved complex
spectra. It changes neither the physical run nor the public Result schema.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch

from witwin.maxwell.fdtd.copper_foil import copper_spectral_power


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
            and left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return left == right


@dataclass(frozen=True)
class SpectralContract:
    source_sha256: str
    grid_sha256: str
    material_sha256: str
    observation_sha256: str
    planned_steps: int
    window: str
    normalize_source: bool = False
    phasor_convention: str = 'exp(-iwt)'

    def __post_init__(self):
        for value in (self.source_sha256, self.grid_sha256, self.material_sha256, self.observation_sha256):
            if len(value) != 64 or any(letter not in '0123456789abcdef' for letter in value):
                raise ValueError('Coherent execution signatures must be complete SHA256 digests.')
        if (type(self.planned_steps) is not int or self.planned_steps <= 0
            or self.window not in ('none', 'hanning', 'ramp')
            or self.normalize_source or self.phasor_convention != 'exp(-iwt)'):
            raise ValueError('Coherent loss requires a positive fixed sampling horizon and raw native phasors.')


@dataclass(frozen=True)
class FoilRealization:
    face: str
    component: str
    normal_axis: str
    normal_index: int
    banks: tuple[str, ...]
    indices: torch.Tensor
    area: torch.Tensor
    B: torch.Tensor
    rates: torch.Tensor
    D: torch.Tensor
    spectral_norm: torch.Tensor


@dataclass(frozen=True)
class LossBasis:
    """One physical drive; CPU diagnostics only, with explicit executed pins."""
    frequencies: tuple[float, ...]
    dt: float
    execution_contract: SpectralContract
    realizations: tuple[FoilRealization, ...]
    conductor_records: dict


def film_area(trace, realization):
    """Native film area times the executed free-edge admittance scale 1/kappa.

    Results recorded before the CopperFoil edge model carry no per-trace
    factor; their rims executed with the bare unit scale.
    """
    factor = trace.get('edge_impedance_factor')
    return realization.area if factor is None else realization.area / factor


def capture_loss_basis(result, execution_contract):
    """Read actual runtime coefficients after a run; never reconstruct a model.

    The contract must include source/grid/observer/time-window signatures common
    to all drives. Drive index and source amplitudes belong in separate metadata.
    """
    if not isinstance(execution_contract, SpectralContract) or result.stats().get('normalize_source', False):
        raise ValueError('Loss basis requires an executed contract and raw common spectral normalization.')
    solver = result.solver
    if solver is None:
        raise ValueError('Capture executed realization coefficients before releasing the runtime solver.')
    dt = float(solver.dt)
    frequencies = tuple(float(f) for f in result.frequencies)
    if not frequencies or any(not math.isfinite(f) or f <= 0 for f in frequencies) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Loss basis requires finite positive frequencies and actual timestep.')
    records = _cpu(result.conductor_losses)
    entries = tuple(getattr(solver, '_copper_foils', ()))
    realizations = []
    claimed = set()
    for entry in entries:
        if entry['imaginary']:
            continue
        face = entry['face']
        component = entry['electric_name']
        key = (face.name, component)
        if key in claimed:
            raise ValueError('Loss basis requires unique face/component runtime ownership.')
        claimed.add(key)
        partners = [candidate for candidate in entries if candidate['imaginary']
            and candidate['face'] is face and candidate['electric_name'] == component + '_imag']
        if len(partners) > 1:
            raise ValueError('Loss basis has duplicate imaginary banks.')
        for partner in partners:
            if any(not _equal(_cpu(entry[name]), _cpu(partner[name]))
                   for name in ('indices', 'area', 'B', 'rates', 'D', 'spectral_frequencies',
                       'spectral_norm', 'spectral_end', 'spectral_step', 'spectral_window')):
                raise ValueError('Loss basis complex banks disagree on realization or normalization.')
        if (entry['spectral_end'] != execution_contract.planned_steps
            or float(entry['spectral_step']) != execution_contract.planned_steps
            or entry['spectral_window'] != execution_contract.window):
            raise ValueError('Loss basis must complete its declared fixed sampling horizon and window.')
        if tuple(float(f) for f in entry['spectral_frequencies']) != frequencies or not bool((entry['spectral_norm'] > 0).all()):
            raise ValueError('Loss basis runtime frequency order or sampling support changed.')
        realization = FoilRealization(face.name, component, 'xyz'[face.axis], int(face.normal_index),
            ('real', 'imaginary') if partners else ('real',),
            *(_cpu(entry[name]) for name in ('indices', 'area', 'B', 'rates', 'D', 'spectral_norm')))
        realizations.append(realization)
    # Every imaginary bank must have exactly one real owner, including zero-edge
    # faces. An orphan cannot silently disappear from the diagnostic identity.
    if sum(1 for entry in entries if entry['imaginary']) != sum(len(r.banks) - 1 for r in realizations):
        raise ValueError('Loss basis has an orphan imaginary bank.')
    basis = LossBasis(frequencies, dt, _cpu(execution_contract), tuple(realizations), records)
    _validate_records(basis)
    return basis


def _validate_records(basis):
    if not isinstance(basis.execution_contract, SpectralContract):
        raise ValueError('Loss basis must retain its typed executed sampling contract.')
    for realization in basis.realizations:
        count = len(realization.indices)
        if (realization.indices.shape != (count, 3) or realization.indices.dtype != torch.int64
            or realization.area.shape != (count,) or not bool((realization.area > 0).all())
            or realization.B.ndim != 2 or realization.B.shape[1] != 2
            or realization.rates.shape != (len(realization.B),) or realization.D.shape != (2, 2)
            or realization.spectral_norm.shape != (len(basis.frequencies),)
            or not bool((realization.spectral_norm > 0).all())
            or any(not bool(torch.isfinite(value).all()) for value in (realization.area, realization.B,
                realization.rates, realization.D, realization.spectral_norm))):
            raise ValueError('Loss basis has an invalid native support, realization or sampling normalization.')
    expected = {(r.face, r.component) for r in basis.realizations}
    if len(expected) != len(basis.realizations):
        raise ValueError('Loss basis has duplicate face/component realization ownership.')
    for name, record in basis.conductor_records.items():
        if (tuple(float(f) for f in record['frequencies']) != basis.frequencies
            or not bool(record['spectral_valid'].all())
            or record['coordinate_frame'] != 'cartesian' or record['periodic_axes']):
            raise ValueError('Coherent line loss requires valid Cartesian nonperiodic physical spectra.')
        surface = record['surface_power_traces']
        bulk = record['bulk_electric_traces']
        for traces in (surface, bulk):
            keys = [(name, trace['component']) for trace in traces]
            if len(set(keys)) != len(keys) or set(keys) != {key for key in expected if key[0] == name}:
                raise ValueError('Loss basis trace ownership is incomplete or duplicated.')
        for realization in (r for r in basis.realizations if r.face == name):
            trace = next(t for t in surface if t['component'] == realization.component)
            electric = next(t for t in bulk if t['component'] == realization.component)
            if (not _equal(trace['indices'], realization.indices)
                or not _equal(electric['indices'], realization.indices)
                or not _equal(trace['area'], realization.area)
                or trace['normal_axis'] != realization.normal_axis
                or electric['normal_axis'] != realization.normal_axis
                or trace['normal_coordinate'] != electric['normal_coordinate']):
                raise ValueError('Loss basis runtime and saved native support disagree.')
            fields = electric['electric_fields']
            if fields.shape != (len(basis.frequencies), 2, len(realization.indices)) or not fields.is_complex() or not bool(torch.isfinite(fields).all()):
                raise ValueError('Loss basis requires finite complete complex two-face fields.')
    if set(basis.conductor_records) != {r.face for r in basis.realizations}:
        raise ValueError('Loss basis is missing a runtime foil face.')


def coherent_fields(fields, coefficients):
    """Combine [frequency,...] complex spectra, never their individual powers."""
    if not fields or not fields[0]:
        raise ValueError('Coherent loss requires at least one drive.')
    coefficient = torch.as_tensor(coefficients, dtype=torch.complex128, device='cpu')
    initial = next(iter(fields[0].values()))
    if initial.ndim < 1:
        raise ValueError('Coherent spectra require an explicit frequency axis.')
    count = initial.shape[0]
    if coefficient.shape != (count, len(fields)) or not bool(torch.isfinite(coefficient).all()):
        raise ValueError('Coherent coefficients must have finite shape [frequency, drive].')
    keys = fields[0].keys()
    if any(drive.keys() != keys for drive in fields):
        raise ValueError('Coherent electric component keys differ between drives.')
    output = {}
    for name, first in fields[0].items():
        if first.ndim < 1 or first.shape[0] != count or any(value[name].shape != first.shape or not value[name].is_complex()
               or not bool(torch.isfinite(value[name]).all()) for value in fields):
            raise ValueError('Coherent spectra must have matching finite complex shapes.')
        weights = coefficient.reshape(count, len(fields), *([1] * (first.ndim - 1)))
        output[name] = (torch.stack([value[name].detach().cpu().to(torch.complex128) for value in fields], 1) * weights).sum(1)
    return output


def coherent_conductor_records(bases, coefficients):
    """Return diagnostic records accepted by existing shared ROI/bulk loss APIs.

    Transient heat/storage from individual drives are deliberately absent: they
    are not the transient heat/storage of this frequency-dependent combination.
    """
    if not bases:
        raise ValueError('Coherent loss requires at least one drive.')
    first = bases[0]
    for basis in bases:
        _validate_records(basis)
        if (basis.frequencies != first.frequencies or basis.dt != first.dt
            or basis.execution_contract != first.execution_contract
            or len(basis.realizations) != len(first.realizations)
            or any(not _equal(vars(a), vars(b)) for a, b in zip(basis.realizations, first.realizations))):
            raise ValueError('Coherent loss bases differ in executed realization or common spectral contract.')
    # Validate coefficients even for a PEC basis with no foil records.
    coherent_fields([{'dummy': torch.zeros(len(first.frequencies), dtype=torch.complex128)} for _ in bases], coefficients)
    output = _cpu(first.conductor_records)
    for name, record in output.items():
        for key in ('dissipated_energy', 'stored_energy'):
            record.pop(key, None)
        for realization in (r for r in first.realizations if r.face == name):
            traces = [next(t for t in basis.conductor_records[name]['bulk_electric_traces']
                           if t['component'] == realization.component) for basis in bases]
            # Static material/geometry/quadrature must be identical. Only the
            # physical complex endpoint fields may change with the drive.
            for trace in traces[1:]:
                if not _equal({k: v for k, v in traces[0].items() if k != 'electric_fields'},
                              {k: v for k, v in trace.items() if k != 'electric_fields'}):
                    raise ValueError('Coherent split-trace material or geometry coefficients differ.')
            surface = [next(t for t in basis.conductor_records[name]['surface_power_traces']
                            if t['component'] == realization.component) for basis in bases]
            for trace in surface[1:]:
                if not _equal({k: v for k, v in surface[0].items() if k != 'power'},
                              {k: v for k, v in trace.items() if k != 'power'}):
                    raise ValueError('Coherent film geometry or quadrature differs.')
            phasor = coherent_fields([{'E': t['electric_fields']} for t in traces], coefficients)['E']
            total, edge = copper_spectral_power(phasor, torch.tensor(first.frequencies, dtype=torch.float64),
                first.dt, realization.B, realization.rates, realization.D, film_area(surface[0], realization),
                return_edge_power=True)
            next(t for t in record['bulk_electric_traces'] if t['component'] == realization.component)['electric_fields'] = phasor
            next(t for t in record['surface_power_traces'] if t['component'] == realization.component)['power'] = edge
            record.setdefault('_coherent_totals', []).append(total)
        record['surface_power'] = sum(record.pop('_coherent_totals'), torch.zeros(len(first.frequencies), dtype=torch.float64))
        record['autograd_provenance'] = 'Detached coherent benchmark spectra; no transient or gradient claim.'
    return output

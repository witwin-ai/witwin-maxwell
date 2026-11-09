"""Owned zero-induced volume state for the linear SIE-VIE degeneration.

The regional Green functions already contain the homogeneous linear material.
These arrays describe the additional induced source, not total polarization.
Nonzero constitutive closure belongs to the harmonic-balance path.
"""
from dataclasses import dataclass
import torch
from .volume_mesh import TetrahedralVolumeMesh, bind_volume_region

PARTITION = 'regional_linear_green_zero_additional_polarization_v1'


@dataclass(frozen=True)
class LinearCoupledConfig:
    """Explicit conforming volumes for the zero-induced-source linear limit."""
    volumes: tuple[TetrahedralVolumeMesh, ...]

    def __post_init__(self):
        if (type(self.volumes) is not tuple or not self.volumes
                or any(type(v) is not TetrahedralVolumeMesh for v in self.volumes)):
            raise TypeError('Linear coupled volumes must be a nonempty tuple of tetrahedral meshes')
        if len({v.region_id for v in self.volumes}) != len(self.volumes):
            raise ValueError('Linear coupled volume regions must be unique')
        for volume in self.volumes:
            volume.assert_current()


class LinearVolumeState:
    """Bind every body to exact volume geometry and owned zero P/J arrays.

    This is a state contract, not an independent solve or a prescribed-source
    API. Restoring it binds to an already restored surface system without
    assembling an operator, solving, or applying a constitutive contrast.
    """
    def __init__(self, system, config, *, dtype):
        if type(config) is not LinearCoupledConfig:
            raise TypeError('Linear state requires LinearCoupledConfig')
        system.assert_current()
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError('Linear state requires complex field precision')
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        if system.dtype != real_dtype:
            raise ValueError('Linear state field and geometry precision must match')
        volumes = tuple(sorted(config.volumes, key=lambda v: v.region_id))
        if tuple(v.region_id for v in volumes) != tuple(range(1, len(system.bodies)+1)):
            raise ValueError('Linear state requires exactly one conforming volume per body')
        self.system, self.dtype = system, dtype
        self.bindings = tuple(bind_volume_region(v, system) for v in volumes)
        self.polarization = tuple(torch.zeros(len(v.cells), 3, device=system.device, dtype=dtype)
                                  for v in volumes)
        self.currents = tuple(torch.zeros_like(p) for p in self.polarization)
        self._binding = self._identity()

    def _identity(self):
        return (id(self.system), self.dtype, tuple(map(id, self.bindings)),
                tuple((id(v), v._version) for v in (*self.polarization, *self.currents)))

    def assert_current(self):
        if self._identity() != self._binding:
            raise ValueError('Linear coupled state or region binding changed')
        self.system.assert_current()
        for binding in self.bindings:
            binding.assert_current()

    def snapshot(self):
        self.assert_current()
        return dict(version='linear-zero-induced-volume-state-v1',
            formulation_id='coupled_sie_vie_linear', material_partition=PARTITION,
            source_role='additional_induced_volume_current',
            dtype=str(self.dtype), physical_qualification=False,
            regions=[dict(region_id=b.volume.region_id,
                vertices=b.volume.vertices.detach().cpu().clone(),
                cells=b.volume.cells.detach().cpu().clone(),
                additional_polarization=p.detach().cpu().clone(),
                additional_current=j.detach().cpu().clone())
                for b, p, j in zip(self.bindings, self.polarization, self.currents)])

    @classmethod
    def restore(cls, payload, system, *, dtype):
        expected = dict(version='linear-zero-induced-volume-state-v1',
            formulation_id='coupled_sie_vie_linear', material_partition=PARTITION,
            source_role='additional_induced_volume_current', dtype=str(dtype),
            physical_qualification=False)
        if (type(payload) is not dict or set(payload) != {*expected, 'regions'}
                or any(payload[k] != v for k, v in expected.items())
                or payload['physical_qualification'] is not False
                or type(payload['regions']) is not list
                or len(payload['regions']) != len(system.bodies)):
            raise ValueError('Invalid linear coupled state contract')
        volumes = []
        for region_id, row in enumerate(payload['regions'], 1):
            if (type(row) is not dict or set(row) != {'region_id', 'vertices', 'cells',
                    'additional_polarization', 'additional_current'}
                    or type(row['region_id']) is not int or row['region_id'] != region_id):
                raise ValueError('Linear state requires canonical body order')
            vertices, cells = row['vertices'], row['cells']
            if (not isinstance(vertices, torch.Tensor) or vertices.dtype != system.dtype
                    or not isinstance(cells, torch.Tensor) or cells.dtype != torch.int64
                    or cells.ndim != 2 or cells.shape[1] != 4):
                raise ValueError('Stored linear volume precision or connectivity is invalid')
            for name in ('additional_polarization', 'additional_current'):
                value = row[name]
                if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                        or value.shape != (len(cells), 3) or value.requires_grad
                        or not bool(torch.isfinite(value).all()) or bool((value != 0).any())):
                    raise ValueError('Linear degeneration requires exact zero additional P/J')
            volumes.append(TetrahedralVolumeMesh(vertices.to(system.device),
                cells.to(system.device), region_id=region_id))
        return cls(system, LinearCoupledConfig(tuple(volumes)), dtype=dtype)

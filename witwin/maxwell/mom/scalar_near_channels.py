"""Owned original single-medium PMCHWT near blocks with two scalar channels."""
from dataclasses import asdict
import json
import math
import torch
from .nearfield import scalar_regional_channels, _selected_edges, _validate_scalar_channel_medium
from .operator import IntegrationConfig, PreparedRWGGeometry
from .direct_system import DirectSystemActions
from .fmm_fft import _rhs_chunks


def _basis_inputs(basis):
    return (basis.mesh.vertices, basis.mesh.faces, basis.edges, basis.face_edges,
            basis.face_signs, basis.edge_lengths, basis.plus_faces, basis.minus_faces)


class ScalarNearChannels:
    """A=[K,aD;bD,K], storing independently integrated D/K rather than four blocks."""
    apply = DirectSystemActions.apply
    adjoint = DirectSystemActions.adjoint

    def __init__(self, test_basis, source_basis, test_edges, source_edges, medium, *,
                 integration=IntegrationConfig(), geometry=None, rhs_batch=4,
                 retained_budget_bytes=1024**3):
        _validate_scalar_channel_medium(medium)
        if not isinstance(integration, IntegrationConfig):
            raise TypeError('Near channels require IntegrationConfig')
        if (test_basis.mesh.device.type != 'cuda' or source_basis.mesh.device != test_basis.mesh.device
                or test_basis.mesh.dtype != torch.float64 or source_basis.mesh.dtype != torch.float64
                or test_basis.mesh.vertices.requires_grad or source_basis.mesh.vertices.requires_grad):
            raise ValueError('Near channels require matching fixed CUDA float64 RWG spaces')
        if any(type(v) is not int or v < 1 for v in (rhs_batch, retained_budget_bytes)):
            raise ValueError('Near channel RHS batch and retained budget must be positive integers')
        for basis in (test_basis, source_basis):
            mesh = basis.mesh
            if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
                raise ValueError('Near channel RWG geometry changed before integration')
        _selected_edges(test_basis, test_edges); _selected_edges(source_basis, source_edges)
        nr, nc = len(test_edges), len(source_edges)
        expected = 2*nr*nc*16+8*(nr+nc)
        if expected > retained_budget_bytes:
            raise MemoryError('Near channel retained budget exceeded before integration')
        if geometry is not None:
            if (not isinstance(geometry, (tuple, list)) or len(geometry) != 2
                    or any(not isinstance(g, PreparedRWGGeometry) for g in geometry)):
                raise ValueError('Near channel geometry must be an explicit matching shared pair')
            geometry = tuple(geometry)
            geometry[0].assert_current(test_basis); geometry[1].assert_current(source_basis)
        self.test_basis, self.source_basis = test_basis, source_basis
        self.test_edges, self.source_edges = test_edges.clone(), source_edges.clone()
        self.medium, self.integration, self.geometry = medium, integration, geometry
        self.rhs_batch, self.retained_budget_bytes = rhs_batch, retained_budget_bytes
        self.planned_retained_bytes = expected
        self.shape = (2*nr, 2*nc)
        self.device, self.dtype = test_basis.mesh.device, torch.complex128
        self.electric, self.magnetic = 1j*medium.omega*medium.epsilon, -1j*medium.omega*medium.mu
        if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in (complex(self.electric), complex(self.magnetic))):
            raise ValueError('Near channel material factors exceed native precision')
        self.electric_kernel, self.magnetic_kernel, metadata = scalar_regional_channels(
            test_basis, source_basis, self.test_edges, self.source_edges, medium, integration, geometry=geometry)
        for field in (self.electric_kernel, self.magnetic_kernel):
            if (field.shape != (nr, nc) or field.device != self.device or field.dtype != self.dtype
                    or field.requires_grad or not bool(torch.isfinite(field).all())):
                raise ValueError('Near channels require fixed finite native original kernel arrays')
        self._quadrature_text = json.dumps(metadata, sort_keys=True, allow_nan=False)
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != expected:
            raise RuntimeError('Near channel retained tensors differ from structural plan')
        self._binding = self._state()
        self._input_versions = tuple((id(t), t._version) for t in self.input_tensors())

    def tensors(self):
        return (self.test_edges, self.source_edges, self.electric_kernel, self.magnetic_kernel)

    def input_tensors(self):
        return (*_basis_inputs(self.test_basis), *_basis_inputs(self.source_basis))

    def _state(self):
        m = self.medium
        return (id(self.test_basis), id(self.source_basis), id(m), m.frequency,
                m.epsilon_r, m.mu_r, m.k, m.eta, self.electric, self.magnetic,
                id(self.integration), tuple(asdict(self.integration).items()), id(self.geometry),
                None if self.geometry is None else tuple(map(id, self.geometry)),
                self.shape, self.device, self.dtype, self.rhs_batch, self.retained_budget_bytes,
                self.planned_retained_bytes, self.retained_bytes, self._quadrature_text,
                tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        if (self._state() != self._binding
                or tuple((id(t), t._version) for t in self.input_tensors()) != self._input_versions):
            raise ValueError('Scalar near channels or owning physical data changed')
        if self.geometry is not None:
            self.geometry[0].assert_current(self.test_basis)
            self.geometry[1].assert_current(self.source_basis)

    def _action(self, value, *, adjoint):
        self.assert_current()
        input_size = self.shape[0] if adjoint else self.shape[1]
        output_size = self.shape[1] if adjoint else self.shape[0]
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != input_size
                or value.device != self.device or value.dtype != self.dtype):
            raise ValueError('Near channel action requires matching complex128 J/M coefficients')
        result = value.new_empty((math.prod(value.shape[:-1]), output_size))
        width = input_size//2
        d, k = self.electric_kernel, self.magnetic_kernel
        for start, chunk in _rhs_chunks(value, 1, self.rhs_batch):
            j, m = chunk[:, :width], chunk[:, width:]
            if adjoint:
                first = j@k.conj()+self.magnetic.conjugate()*(m@d.conj())
                second = self.electric.conjugate()*(j@d.conj())+m@k.conj()
            else:
                first = j@k.mT+self.electric*(m@d.mT)
                second = self.magnetic*(j@d.mT)+m@k.mT
            block = torch.cat((first, second), dim=-1)
            if not bool(torch.isfinite(block).all()):
                raise ValueError('Near channel action exceeds native precision')
            result[start:start+len(block)] = block
        self.assert_current()
        return result.reshape(*value.shape[:-1], output_size)

    def receipt(self):
        self.assert_current()
        return dict(version='experimental_scalar_single_medium_near_channels_v1',
            kernel_id='free_space_helmholtz', shape=self.shape,
            ordering='J then M, selected edge order', kernel_channels='original independently integrated D/K',
            retained_bytes=self.retained_bytes, rhs_batch=self.rhs_batch,
            quadrature=json.loads(self._quadrature_text), shared_basis_and_geometry_excluded=True,
            old_full_matrix_bit_equivalence_claimed=False, assembly_workspace_qualified=False,
            packed_storage_integrated=False, public_integration_qualified=False,
            material_gradient_qualified=False, physical_qualification=False,
            complexity_qualification=False, peak_memory_qualified=False)

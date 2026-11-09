"""Scalar regional volume-current potentials with distributional self terms.

These fixed-order primitives do not constitute a coupled SIE-VIE solver or an
adaptive error certificate. They support cell-interior and off-cell queries;
cell-face queries require explicit principal-value semantics; edges fail.
"""
import math
import torch
from .fields import _static_panel_gradient, _triangle_distance, _pair_triangle_distance, _static_panel_potential
from .lifting_fields import _smooth_background_gradient
from .quadrature import triangle_rule, duffy_fan
from .physics import Medium
from .volume_mesh import TetrahedralVolumeMesh


_MAX_PAIR_SAMPLES = 262144

def _radial_first_moment_remainder(z):
    """Integral of t*(exp(z*t)-1), t in [0,1], without static cancellation."""
    small = z.abs() < .5
    safe = torch.where(small, torch.ones_like(z), z)
    direct = (torch.exp(safe)*(safe-1)+1)/safe.square()-.5
    small_z = torch.where(small, z, torch.zeros_like(z))
    term, series = torch.ones_like(z), torch.zeros_like(z)
    for n in range(1, 18):
        term = term*small_z/n
        series = series+term/(n+2)
    return torch.where(small, series, direct)


class PreparedVolumeGreenGeometry:
    """Guard the complete-mesh extent reused by selected native integrations."""
    def __init__(self,mesh):
        if type(mesh) is not TetrahedralVolumeMesh:raise TypeError('Prepared volume geometry requires a native mesh')
        mesh.assert_current()
        self.mesh=mesh
        self.extent=torch.linalg.vector_norm(mesh.vertices.amax(0)-mesh.vertices.amin(0))
        self._binding=(id(mesh),id(self.extent),self.extent._version)

    def assert_current(self,mesh):
        if mesh is not self.mesh or self._binding!=(id(self.mesh),id(self.extent),self.extent._version):
            raise ValueError('Prepared volume Green geometry changed')
        self.mesh.assert_current()


def cell_green_potentials(mesh, points, k, *, order=6, principal_value=False, include_hessian=True, cell_indices=None, geometry=None):
    """Return integral(g), integral(grad g), integral(distributional Hess g).

    The Hessian equals minus the face integral of grad(g) outer source normal,
    so its interior contact term is included without sampling a singular delta.
    The volume scalar integral uses signed radial fans anchored at the query;
    its weak 1/r singularity is integrated analytically along each radial ray.
    Results have leading dimensions [query,cell]. Face-interior PV is opt-in.
    With include_hessian=False, the third result is None and its work is skipped.
    Optional CUDA int64 cell_indices selects columns before integration, preserving
    order and repetitions. The original owning mesh and trace guards remain.
    Optional prepared geometry reuses the guarded complete-mesh extent; selected
    columns never change the full-mesh trace tolerance.
    """
    if not isinstance(mesh, TetrahedralVolumeMesh):
        raise TypeError('Volume Green integration requires a tetrahedral mesh')
    mesh.assert_current()
    if geometry is not None:
        if type(geometry) is not PreparedVolumeGreenGeometry:raise TypeError('Invalid prepared volume Green geometry')
        geometry.assert_current(mesh)
    if (not isinstance(points, torch.Tensor) or points.device != mesh.vertices.device
            or points.dtype != mesh.vertices.dtype or points.ndim != 2 or points.shape[1] != 3
            or not bool(torch.isfinite(points).all())):
        raise ValueError('Volume queries require finite matching CUDA real [Q,3] points')
    if type(include_hessian) is not bool:
        raise TypeError('include_hessian must be an explicit boolean')
    if type(principal_value) is not bool:
        raise TypeError('principal_value must be an explicit boolean')
    if type(order) is not int or order < 2:
        raise ValueError('Volume Green quadrature requires order >= 2')
    k = complex(k)
    if not math.isfinite(k.real) or not math.isfinite(k.imag):
        raise ValueError('Volume Green wavenumber must be finite')
    cells = mesh.cells
    if cell_indices is not None:
        if (not isinstance(cell_indices,torch.Tensor) or cell_indices.device != mesh.vertices.device
                or cell_indices.dtype != torch.int64 or cell_indices.ndim != 1
                or bool(((cell_indices < 0) | (cell_indices >= len(cells))).any())):
            raise ValueError('Selected volume cells require matching CUDA int64 indices in range')
        cells = cells[cell_indices]
    count = len(cells)
    dtype = torch.complex128 if points.dtype == torch.float64 else torch.complex64
    if not len(points) or not count:
        return (torch.empty((len(points),count),device=points.device,dtype=dtype),
                torch.empty((len(points),count,3),device=points.device,dtype=dtype),
                torch.empty((len(points),count,3,3),device=points.device,dtype=dtype) if include_hessian else None)
    tetrahedra = mesh.vertices[cells]
    face_ids = torch.tensor([[1,2,3], [0,3,2], [0,1,3], [0,2,1]], device=points.device)
    triangles = tetrahedra[:, face_ids].reshape(-1, 3, 3)
    normal = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    extent = (torch.linalg.vector_norm(mesh.vertices.amax(0)-mesh.vertices.amin(0))
              if geometry is None else geometry.extent)
    tolerance = 128*torch.finfo(points.dtype).eps*extent
    values, gradients, hessians = [], [], []
    face_count = len(triangles)
    diameter = torch.linalg.vector_norm(triangles-triangles.roll(1, dims=1), dim=-1).amax(-1)
    # Bound dynamic quadrature by query-face pairs, including all three fans.
    # A single high-order pair is indivisible; this is not a peak-memory bound.
    pair_batch = min(16*face_count, max(1, _MAX_PAIR_SAMPLES//(3*order**2))) if k != 0 else 16*face_count
    if principal_value and len(points):
        for batch in points.split(16):
            distances = _triangle_distance(batch, triangles)
            if bool(((distances <= tolerance).reshape(len(batch), -1, 4).sum(-1) > 1).any()):
                raise ValueError('Volume cell-edge and vertex traces are undefined')
    for start in range(0, len(points)*face_count, pair_batch):
        indices = torch.arange(start, min(start+pair_batch, len(points)*face_count), device=points.device)
        query_ids, face_ids = indices//face_count, indices%face_count
        observers, panels, normals = points[query_ids], triangles[face_ids], normal[face_ids]
        # Pairwise distances without constructing a Cartesian query-face array.
        distances = _pair_triangle_distance(observers, panels)
        on_face = distances <= tolerance
        if bool(on_face.any()):
            if not principal_value:
                raise ValueError('Volume cell-face queries require explicit trace treatment')
        static_gradient = _static_panel_gradient(observers, panels, principal_value=on_face if principal_value else None)
        static_potential = _static_panel_potential(observers, panels, normals, static_gradient)
        height = ((panels[:, 0]-observers)*normals).sum(-1)
        panel_value, panel_gradient = static_potential, static_gradient
        volume_terms = .5*height*static_potential
        if k != 0:
            nodes, weights = triangle_rule(panels, order)
            use_fan = distances < .5*diameter[face_ids]
            if bool(use_fan.any()):
                nodes = nodes.repeat(1,3,1)
                weights = weights.repeat(1,3)/3
                fan_nodes, fan_weights = duffy_fan(observers[use_fan], panels[use_fan], order)
                nodes[use_fan], weights[use_fan] = fan_nodes, fan_weights
            delta = observers[:,None,:]-nodes
            distance = torch.linalg.vector_norm(delta, dim=-1)
            z = 1j*k*distance
            panel_value = static_potential+(torch.expm1(z)/(4*math.pi*distance)*weights).sum(-1)
            if include_hessian:
                panel_gradient = static_gradient+(_smooth_background_gradient(delta,k)*weights[...,None]).sum(1)
            volume_terms = volume_terms+height*(_radial_first_moment_remainder(z)/distance*weights).sum(-1)/(4*math.pi)
        values.append(volume_terms)
        gradients.append(-panel_value[:, None]*normals)
        if include_hessian:
            hessians.append(-panel_gradient[:, :, None]*normals[:, None, :])
    def combine(parts, shape):
        return torch.cat(parts).reshape(len(points),count,4,*shape).sum(2).to(dtype)
    return combine(values,()),combine(gradients,(3,)),combine(hessians,(3,3)) if include_hessian else None


def cell_current_field_blocks(mesh, points, medium, *, cell_indices=None, order=6, principal_value=False):
    """Selected electric-current E/H blocks, ordered [query,xyz,cell,xyz].

    Selection is applied before any query/cell quadrature. Distributional self
    terms and PV rejection match full integration. This is a fixed-order block
    provider, not an adaptive error certificate or a compressed operator.
    """
    if not isinstance(medium, Medium):
        raise TypeError('Volume field blocks require a scalar regional Medium')
    value,gradient,hessian = cell_green_potentials(mesh,points,medium.k,order=order,
        principal_value=principal_value,cell_indices=cell_indices)
    identity = torch.eye(3,device=value.device,dtype=value.dtype)
    electric = (1j*medium.omega*medium.mu*value[:,:,None,None]*identity
                +1j/(medium.omega*medium.epsilon)*hessian)
    magnetic = torch.linalg.cross(gradient[:,:,None,:],identity[None,None,:,:],dim=-1)
    return electric.permute(0,2,1,3),magnetic.permute(0,3,1,2)


def cell_current_fields(mesh, currents, points, medium, *, order=6, principal_value=False):
    """Electric volume currents [A/m^2] to regional E[V/m],H[A/m]."""
    if not isinstance(medium, Medium):
        raise TypeError('This volume-current primitive requires a scalar regional Medium')
    dtype = torch.complex128 if mesh.vertices.dtype == torch.float64 else torch.complex64
    if (not isinstance(currents, torch.Tensor) or currents.device != mesh.vertices.device
            or currents.dtype != dtype or currents.shape != (len(mesh.cells),3)
            or not bool(torch.isfinite(currents).all())):
        raise ValueError('Volume currents require finite matching CUDA complex [cells,3]')
    value, gradient, hessian = cell_green_potentials(mesh, points, medium.k, order=order, principal_value=principal_value)
    electric = 1j*medium.omega*medium.mu*torch.einsum('qt,ti->qi', value, currents)
    electric += 1j/(medium.omega*medium.epsilon)*torch.einsum('qtij,tj->qi', hessian, currents)
    magnetic = torch.linalg.cross(gradient, currents[None].expand_as(gradient), dim=-1).sum(1)
    return electric, magnetic


class VolumeFieldConvergenceError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def evaluate_cell_current_fields(mesh, currents, points, medium, *, order=3, config=None, principal_value=False):
    """Refine complete E/H fields with separate per-query error requirements.

    Absolute tolerance is in V/m for E and divided by |eta| for H. Accepted
    values retain native gradients; previous refinement graphs are discarded.
    Successive-order differences are empirical estimates, not certified bounds.
    """
    from dataclasses import asdict
    from .fields import FieldQueryConfig
    if config is None:
        config = FieldQueryConfig()
    if not isinstance(config, FieldQueryConfig):
        raise TypeError('Volume query settings require FieldQueryConfig')
    if type(order) is not int or order < 2 or order > config.max_order:
        raise ValueError('Initial volume order must be >= 2 and within the configured budget')
    receipt = {'kernel_version': 'scalar_tetrahedral_distributional_volume_v1', 'config': asdict(config),
        'converged': None, 'certified_error_bound': False, 'checks': [], 'evaluated_orders': [],
        'region_id': mesh.region_id, 'distributional_contact_included': True,
        'linear_material_contrast_added': False, 'surface_trace': 'principal_value_per_cell' if principal_value else False, 'physical_qualification': False}
    def evaluate(current_order):
        values = cell_current_fields(mesh, currents, points, medium, order=current_order, principal_value=principal_value)
        if any(not bool(torch.isfinite(value).all()) for value in values):
            receipt.update(converged=False, failure_reason='nonfinite_volume_field')
            raise VolumeFieldConvergenceError('Nonfinite volume field', receipt)
        receipt['evaluated_orders'].append(current_order)
        return values
    previous = evaluate(order)
    if not config.adaptive:
        receipt.update(final_order=order, failure_reason=None, estimator='not_requested')
        return previous, receipt
    receipt.update(converged=False, estimator='complete_E_and_H_successive_order_differences')
    previous = tuple(value.detach() for value in previous)
    def maximum(value):
        return float(value.max().detach()) if value.numel() else 0.
    while order < config.max_order:
        order = min(2*order, config.max_order)
        current = evaluate(order)
        checks = {'order': order}
        passed = True
        for name, value, old, absolute in zip(('E','H'), current, previous,
                                             (config.absolute_tolerance, config.absolute_tolerance/abs(medium.eta))):
            error = torch.linalg.vector_norm(value-old, dim=-1)
            magnitude = torch.linalg.vector_norm(value, dim=-1)
            threshold = absolute+config.relative_tolerance*magnitude
            success = bool((error <= threshold).all())
            passed = passed and success
            checks[name] = {'maximum_absolute_difference': maximum(error),
                'maximum_relative_difference': maximum(error/magnitude.clamp_min(torch.finfo(magnitude.dtype).tiny)),
                'passed': success}
        receipt['checks'].append(checks)
        if passed:
            receipt.update(converged=True, final_order=order, failure_reason=None)
            return current, receipt
        previous = tuple(value.detach() for value in current)
    receipt['failure_reason'] = 'volume_order_budget_exhausted'
    raise VolumeFieldConvergenceError('Volume current fields did not converge within the order budget', receipt)


def cell_static_polarization_fields(mesh, polarization, points, epsilon, *, order=6):
    """Regional DC potential/E from cellwise polarization, including contact.

    polarization is a real DC density [C/m^2], epsilon the finite positive
    regional absolute permittivity [F/m]. The dielectric boundary response must
    be solved separately; this primitive does not impose interface conditions.
    No division by frequency or fictitious low-frequency limit is used.
    """
    mesh.assert_current()
    if (not isinstance(polarization, torch.Tensor) or polarization.device != mesh.vertices.device
            or polarization.shape != (len(mesh.cells),3)
            or polarization.dtype != mesh.vertices.dtype
            or not bool(torch.isfinite(polarization).all())):
        raise ValueError('DC polarization requires finite matching real CUDA cell vectors')
    if not isinstance(epsilon, torch.Tensor):
        if type(epsilon) not in (float,int):
            raise ValueError('DC permittivity requires a finite positive real scalar')
        epsilon = mesh.vertices.new_tensor(epsilon)
    if (epsilon.device != mesh.vertices.device or epsilon.dtype != mesh.vertices.dtype
            or epsilon.ndim != 0 or not bool(torch.isfinite(epsilon)) or not bool(epsilon > 0)):
        raise ValueError('DC permittivity requires a matching finite positive real scalar')
    _, gradient, hessian = cell_green_potentials(mesh,points,0.,order=order)
    potential = -torch.einsum('qti,ti->q',gradient.real,polarization)/epsilon
    electric = torch.einsum('qtij,tj->qi',hessian.real,polarization)/epsilon
    return (potential,electric), {'kernel_version': 'static_tetrahedral_polarization_contact_v1',
        'frequency_hz': 0., 'distributional_contact_included': True,
        'linear_material_contrast_added': False, 'interface_conditions_solved': False,
        'physical_qualification': False}

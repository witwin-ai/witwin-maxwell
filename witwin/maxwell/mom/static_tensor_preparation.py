"""Fixed-discretization tensor DC cell response for harmonic preparation."""
import torch
from .static_tensor_interface import StaticTensorInterfaceOperator


def prepare_static_tensor_cell_response(operator, bindings, *, max_response_bytes=256*1024**2):
    """Differentiate the induced-polarization map on an existing SPD interface.

    Columns follow the supplied binding order, then cell and Cartesian axis.
    Surface rows retain the operator's body/face order. These are fixed linear
    response matrices; a zero source does not certify quadrature or physics.
    Material and geometry gradients through this preparation are not retained.
    The budget covers returned response matrices, not peak autograd workspace.
    """
    if not isinstance(operator, StaticTensorInterfaceOperator):
        raise TypeError('Tensor DC preparation requires a static tensor interface')
    operator.assert_current()
    if operator.potential_basis!='p0':
        raise ValueError('Tensor DC cell preparation currently requires P0 potential')
    bindings = tuple(bindings)
    if not bindings:
        raise ValueError('Tensor DC preparation requires volume bindings')
    if type(max_response_bytes) is not int or max_response_bytes < 1:
        raise ValueError('Tensor DC response budget must be a positive integer')
    regions = []
    for binding in bindings:
        if getattr(binding, 'surface_system', None) is not operator.system:
            raise ValueError('Tensor DC volumes must bind the exact surface system')
        binding.assert_current()
        regions.append(binding.volume.region_id)
    if len(set(regions)) != len(regions):
        raise ValueError('Tensor DC preparation requires unique volume regions')
    counts = tuple(len(binding.volume.cells) for binding in bindings)
    points = tuple(binding.volume.vertices[binding.volume.cells].mean(1) for binding in bindings)
    cells, faces = sum(counts), len(operator.centers)
    required = (3*cells+3*faces)*(3*cells)*operator.matrix.element_size()
    if required > max_response_bytes:
        raise ValueError('Tensor DC retained response exceeds its explicit budget')
    last = {}

    def evaluate(polarization):
        response = operator.solve_uniform(operator.matrix.new_zeros(3),
            bindings=bindings, polarizations=polarization.split(counts))
        fields = torch.cat([response.fields(query, region)[1]
                            for query, region in zip(points, regions)]).flatten()
        last['receipt'] = response.receipt
        return torch.cat((fields, response.potential, response.conormal, response.normal_p))

    zero = operator.matrix.new_zeros(cells, 3)
    jacobian = torch.autograd.functional.jacobian(evaluate, zero, vectorize=False).reshape(-1, 3*cells)
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError('Tensor DC response is not finite')
    operator.assert_current()
    for binding in bindings:
        binding.assert_current()
    field_end = 3*cells
    return dict(electric_response=jacobian[:field_end],
        potential_response=jacobian[field_end:field_end+faces],
        conormal_response=jacobian[field_end+faces:field_end+2*faces],
        normal_p_response=jacobian[field_end+2*faces:], points=torch.cat(points),
        regions=tuple(regions), cell_counts=counts,
        receipt=dict(last['receipt'],
            preparation='native_jacobian_of_fixed_regional_tensor_dc_operator_v1',
            retained_response_bytes=required, material_geometry_gradients_retained=False,
            quadrature_convergence_certified=False, physical_qualification=False))

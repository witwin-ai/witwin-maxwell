"""Absolute absorbing lattice-image tail bounds over complete support spheres."""
import math
import torch
from .periodic import LatticeSpec
from .fmm_expansion import _points
from .interpolation_error import KernelBoundUnavailable, KernelErrorBudget


class PeriodicImageTailError(ValueError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Periodic outer-image tail exceeds the declared kernel budget')


def absorbing_image_tail_bound(lattice, displacement, test_radius, source_radius, wave_number, shells):
    """Bound all images with max(abs(n)) > shells, independent of real Bloch phase.

    If sigma is the smallest lattice singular value and B=|R|+a+b, distances
    in shell m are >= sigma*m-B. Shell counts are 2, 8m, or 24m^2+2.
    Sum their exponentially damped polynomial tails exactly, bounding every
    denominator by the first omitted shell's positive separation.
    """
    if not isinstance(lattice, LatticeSpec):
        raise TypeError('Image tail requires LatticeSpec')
    lattice.assert_current()
    _points(displacement)
    if (displacement.shape != (3,) or displacement.device != lattice.vectors.device
            or displacement.dtype != lattice.vectors.dtype):
        raise ValueError('Image tail displacement must match lattice coordinates')
    if type(shells) is not int or shells < 0:
        raise ValueError('Image tail shells must be a nonnegative integer')
    for radius in (test_radius, source_radius):
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
            raise ValueError('Image tail support radii must be finite nonnegative scalars')
    if type(wave_number) not in (int, float, complex):
        raise ValueError('Image tail wave number must be a fixed scalar')
    wave = complex(wave_number)
    if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag <= 0:
        raise ValueError('Image tail requires positive absorption')
    sigma = torch.linalg.svdvals(lattice.vectors.double())[-1]
    extent = torch.linalg.vector_norm(displacement.double())+test_radius+source_radius
    first = shells+1
    distance = sigma*first-extent
    if not bool(torch.isfinite(distance) & (distance > 0)):
        raise KernelBoundUnavailable('First omitted image shell does not separate complete supports')
    logq = -wave.imag*sigma
    logdenom = (-torch.expm1(logq)).log()
    logm = math.log(first)
    zero = -logdenom
    one = torch.logsumexp(torch.stack((logm-logdenom, logq-2*logdenom)), 0)
    two = torch.logsumexp(torch.stack((2*logm-logdenom,
        math.log(2)+logm+logq-2*logdenom,
        logq+torch.log1p(torch.exp(logq))-3*logdenom)), 0)
    if lattice.dimension == 1:
        logcount = math.log(2)+zero
    elif lattice.dimension == 2:
        logcount = math.log(8)+one
    else:
        logcount = torch.logaddexp(math.log(24)+two, math.log(2)+zero)
    loggreen = -wave.imag*distance+logcount-math.log(4*math.pi)-distance.log()
    loggradient = loggreen+torch.log(distance.new_tensor(abs(wave))+1/distance)
    logs = torch.stack((loggreen, loggradient))
    if bool(torch.isnan(logs).any() | (logs > math.log(torch.finfo(torch.float64).max)).any()):
        raise KernelBoundUnavailable('Image tail bound exceeds finite representation')
    bounds = logs.clamp_min(math.log(torch.finfo(torch.float64).tiny)).exp()
    lattice.assert_current()
    return dict(version='absorbing_lattice_shell_tail_v1', dimension=lattice.dimension, shells=shells,
        first_omitted_distance_lower=float(distance), green_absolute=float(bounds[0]),
        gradient_component_absolute=float(bounds[1]),
        scope='ideal_infinite_outer_image_tail_over_complete_supports',
        floating_point_certified=False, expansion_error_certified=False, physical_qualification=False)


def require_image_tail_budget(receipt, budget):
    if not isinstance(budget, KernelErrorBudget):
        raise TypeError('Image tail budget must be KernelErrorBudget')
    if (receipt['green_absolute'] > budget.green_absolute or
            receipt['gradient_component_absolute'] > budget.gradient_component_absolute):
        raise PeriodicImageTailError(receipt)


def periodic_image_partition(lattice, displacement, radius_sum, shells, separation_ratio, image_batch):
    """Stream the shared finite-image near/far partition used by planning and setup."""
    width, dimension = 2*shells+1, lattice.dimension
    for start in range(0, width**dimension, image_batch):
        lattice.assert_current()
        flat = torch.arange(start, min(start+image_batch, width**dimension), device=displacement.device)
        indices = torch.stack([(flat//width**power)%width for power in reversed(range(dimension))], -1)-shells
        delta = displacement-lattice.translations(indices)
        distance = torch.linalg.vector_norm(delta, dim=-1)
        far = (distance > 0) & (radius_sum <= separation_ratio*distance)
        yield indices, delta, far
    lattice.assert_current()


def periodic_multipole_error_bound(lattice, displacement, test_radius, source_radius, wave_number, *,
                                   order, shells, separation_ratio=.5, image_batch=16):
    """Triangle sum of retained-image Cauchy tails plus all omitted images."""
    from .fmm_error import multipole_truncation_bound
    from .fmm_expansion import multi_indices
    multi_indices(order)
    if type(image_batch) is not int or image_batch < 1:
        raise ValueError('Image error batch must be a positive integer')
    if type(separation_ratio) not in (int, float) or not math.isfinite(separation_ratio) or not 0 < separation_ratio < 1:
        raise ValueError('Image error separation ratio must lie strictly between zero and one')
    tail = absorbing_image_tail_bound(lattice, displacement, test_radius, source_radius, wave_number, shells)
    accumulated = torch.zeros(2, device=displacement.device, dtype=torch.float64)
    radii = [displacement.new_tensor(radius) for radius in (test_radius, source_radius)]
    near_count, far_count = 0, 0
    for indices, delta, far in periodic_image_partition(lattice, displacement, test_radius+source_radius,
                                                      shells, separation_ratio, image_batch):
        near_count += int((~far).sum())
        for center in delta[far]:
            bound = multipole_truncation_bound(center, *radii, wave_number, order)
            accumulated += accumulated.new_tensor([bound['green_absolute'], bound['gradient_component_absolute']])
            far_count += 1
    total = accumulated+accumulated.new_tensor([tail['green_absolute'], tail['gradient_component_absolute']])
    if not bool(torch.isfinite(total).all()):
        raise KernelBoundUnavailable('Combined periodic M2L bound exceeds finite representation')
    return dict(version='periodic_image_multipole_kernel_bound_v1', order=order, shells=shells,
        far_images=far_count, exact_near_images=near_count,
        green_absolute=float(total[0]), gradient_component_absolute=float(total[1]),
        retained_green_absolute=float(accumulated[0]), retained_gradient_component_absolute=float(accumulated[1]),
        image_tail=tail, scope='ideal_retained_multipole_plus_infinite_outer_image_tail',
        near_quadrature_certified=False, floating_point_certified=False, physical_qualification=False)


class PeriodicMultipoleError(ValueError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Combined periodic M2L and image tail exceed the declared kernel budget')


def require_periodic_multipole_budget(receipt, budget):
    if not isinstance(budget, KernelErrorBudget):
        raise TypeError('Periodic combined kernel budget must be KernelErrorBudget')
    if (receipt['green_absolute'] > budget.green_absolute or
            receipt['gradient_component_absolute'] > budget.gradient_component_absolute):
        raise PeriodicMultipoleError(receipt)


def periodic_hierarchy_error_bound(test_tree, source_tree, lattice, wave_number, *,
                                   order, shells, separation_ratio=.5, image_batch=16):
    """Bound kernel truncation across root images and every near-image hierarchy.

    Within one image, the tree partition is disjoint in test/source support
    ownership: use the maximum block bound. Across images, use a triangle sum.
    Original near quadrature, RWG projection and floating-point errors remain
    separate requirements; this is not an implemented operator certificate.
    """
    from dataclasses import replace
    from .cluster import sphere_block_partition
    from .fmm_error import multipole_truncation_bound
    test_tree.assert_current()
    source_tree.assert_current()
    if test_tree.expansion_centers is not None or source_tree.expansion_centers is not None:
        raise NotImplementedError('Dyadic expansion centers currently require free space')
    tc = test_tree.lower*.5+test_tree.upper*.5
    sc = source_tree.lower*.5+source_tree.upper*.5
    tr = torch.linalg.vector_norm((test_tree.upper-test_tree.lower)*.5, dim=-1)
    sr = torch.linalg.vector_norm((source_tree.upper-source_tree.lower)*.5, dim=-1)
    displacement = tc[0]-sc[0]
    remote = periodic_multipole_error_bound(lattice, displacement, float(tr[0]), float(sr[0]),
        wave_number, order=order, shells=shells, separation_ratio=separation_ratio, image_batch=image_batch)
    accumulated = tc.new_zeros(2, dtype=torch.float64)
    images, far_blocks, exact_blocks = 0, 0, 0
    for indices, _, far in periodic_image_partition(lattice, displacement, float(tr[0])+float(sr[0]),
                                                   shells, separation_ratio, image_batch):
        for index in indices[~far]:
            translation = lattice.translations(index)
            shifted = replace(source_tree, lower=source_tree.lower+translation, upper=source_tree.upper+translation)
            # Match construction's translated support arithmetic exactly.
            centers = shifted.lower*.5+shifted.upper*.5
            radii = torch.linalg.vector_norm((shifted.upper-shifted.lower)*.5, dim=-1)
            maximum = accumulated.new_zeros(2)
            for aa, bb, flags in sphere_block_partition(test_tree, shifted, separation_ratio=separation_ratio):
                exact_blocks += int((~flags).sum())
                for a, b in zip(aa[flags].tolist(), bb[flags].tolist()):
                    bound = multipole_truncation_bound(tc[a]-centers[b], tr[a], radii[b], wave_number, order)
                    maximum = torch.maximum(maximum, maximum.new_tensor(
                        [bound['green_absolute'], bound['gradient_component_absolute']]))
                    far_blocks += 1
            accumulated += maximum
            images += 1
    total = accumulated+accumulated.new_tensor([remote['green_absolute'], remote['gradient_component_absolute']])
    if not bool(torch.isfinite(total).all()):
        raise KernelBoundUnavailable('Periodic hierarchy bound exceeds finite representation')
    test_tree.assert_current()
    source_tree.assert_current()
    lattice.assert_current()
    return dict(version='periodic_hierarchy_kernel_bound_v1', order=order, shells=shells,
        green_absolute=float(total[0]), gradient_component_absolute=float(total[1]),
        near_hierarchy_green_absolute=float(accumulated[0]),
        near_hierarchy_gradient_component_absolute=float(accumulated[1]),
        hierarchical_images=images, hierarchical_far_blocks=far_blocks, exact_leaf_blocks=exact_blocks,
        remote=remote, scope='ideal_exterior_kernel_all_hierarchies_and_infinite_image_tail',
        near_quadrature_certified=False, rwg_projection_certified=False,
        floating_point_certified=False, implemented_operator_certified=False, physical_qualification=False)

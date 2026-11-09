"""Weak tensor Green testing with analytic integration of static derivatives.

Each body retains its own RWG basis. The leading diagonal Hessian is moved
onto surface divergences over complete RWG supports; weak pair contributions
must therefore be assembled over those supports before comparison with a
strong pointwise trace. Cross kernels retain their principal-value quadrature.
"""
from __future__ import annotations

import torch
from ..constants import EPSILON_0, MU_0
from .operator import local_geometry
from .quadrature import touching_pair_rule, triangle_rule
from .dyadic_green import electrostatic_kernel


def weak_dyadic_pairs(test_basis, source_basis, rows, cols, kernel, *, category, order):
    """Return [pair,row-kind,column-kind,local-test,local-source] weak blocks.

    kernel.evaluate_regular returns the physical 6x6 current Green tensor with
    only its leading diagonal static Hessians removed, plus its error receipt.
    Testing rows are (H,-E); source columns are (J,M). The removed derivatives
    are restored weakly as anisotropic Coulomb potentials times RWG divergences.
    """
    medium = kernel.medium
    medium.assert_current()
    if (test_basis.mesh.device != medium.epsilon_r.device or source_basis.mesh.device != medium.epsilon_r.device
            or test_basis.mesh.dtype != medium.epsilon_r.real.dtype or source_basis.mesh.dtype != test_basis.mesh.dtype):
        raise ValueError('Tensor Green medium and both bases must share CUDA device and precision')
    if category not in (0, 1, 2, 3, 4) or isinstance(category, bool):
        raise ValueError('Invalid dyadic pair category')
    if type(order) is not int or order < 1:
        raise ValueError('Panel order must be a positive integer')
    for indices, basis in ((rows, test_basis), (cols, source_basis)):
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
                or indices.device != medium.epsilon_r.device or indices.ndim != 1
                or bool(((indices < 0) | (indices >= len(basis.mesh.faces))).any())):
            raise ValueError('Panel indices must be valid CUDA int64 vectors')
    if rows.shape != cols.shape:
        raise ValueError('Test/source panel index vectors must have equal shape')
    if not len(rows):
        return medium.epsilon_r.new_empty((0, 2, 2, 3, 3)), {'query_count': 0, 'converged': True}
    _, free_test, test_slopes = local_geometry(test_basis)
    _, free_source, source_slopes = local_geometry(source_basis)
    left, right = test_basis.mesh.triangles[rows], source_basis.mesh.triangles[cols]
    if category < 3:
        if test_basis is not source_basis:
            raise ValueError('Cross-body panels cannot use local vertex or edge contact identities')
        obs, src, weights = touching_pair_rule(left, right, test_basis.mesh.faces[rows],
                                               source_basis.mesh.faces[cols], category, order)
    else:
        x, wx = triangle_rule(left, order)
        y, wy = triangle_rule(right, order)
        obs = x[:, :, None].expand(-1, -1, y.shape[1], -1).flatten(1, 2)
        src = y[:, None].expand(-1, x.shape[1], -1, -1).flatten(1, 2)
        weights = (wx[:, :, None]*wy[:, None]).flatten(1, 2)
    displacement = obs-src
    regular, receipt = kernel.evaluate_regular(displacement.reshape(-1, 3))
    if (regular.shape != (*displacement.shape[:2], 6, 6) and regular.shape != (displacement.numel()//3, 6, 6)):
        raise ValueError('Regular dyadic kernel returned an incompatible shape')
    regular = regular.reshape(*displacement.shape[:2], 6, 6)
    if regular.dtype != medium.epsilon_r.dtype or regular.device != medium.epsilon_r.device or not bool(torch.isfinite(regular).all()):
        raise ValueError('Regular dyadic kernel returned invalid device, precision or values')
    if (not isinstance(receipt, dict) or receipt.get('static_diagonal_restored') is not False
            or receipt.get('static_diagonal_singularity_subtracted') is not True):
        raise ValueError('Weak tensor assembly requires an explicit regular-only kernel; static terms must not be counted twice')
    if receipt.get('converged') is not True:
        raise ValueError('Weak tensor assembly requires a converged regular-kernel evaluation')
    test = (test_slopes[rows, None, :, None]*(obs[:, :, None]-free_test[rows, None])).to(regular.dtype)
    source = (source_slopes[cols, None, :, None]*(src[:, :, None]-free_source[cols, None])).to(regular.dtype)
    result = regular.new_empty((len(rows), 2, 2, 3, 3))
    for row, field_offset, sign in ((0, 3, 1), (1, 0, -1)):
        for column, source_offset in ((0, 0), (1, 3)):
            block = regular[:, :, field_offset:field_offset+3, source_offset:source_offset+3]
            result[:, row, column] = sign*torch.einsum('bqli,bqij,bqmj,bq->blm', test, block, source, weights.to(regular.dtype))
    divergence = 4*test_slopes[rows, :, None]*source_slopes[cols, None, :]
    for row, column, kind, coefficient in ((0, 1, 'magnetic', -1j/(medium.omega*MU_0)),
                                         (1, 0, 'electric', 1j/(medium.omega*EPSILON_0))):
        potential, _, _ = electrostatic_kernel(displacement.reshape(-1, 3), medium, constitutive=kind)
        integral = (potential.reshape(weights.shape)*weights).sum(-1)
        result[:, row, column] += coefficient*divergence*integral[:, None, None]
    return result, {'kernel': receipt, 'panel_order': order, 'category': category,
        'test_source_bases_distinct': test_basis is not source_basis,
        'static_derivatives_integrated_by_parts': True, 'panel_error_certified': False}


def _integrated_dyadic_pairs(basis, source_basis, rows, cols, kernel, category, config):
    """Shared fixed/contact and adaptive near quadrature for dense/selected blocks."""
    import math
    order = config.order if category == 4 else config.singular_order
    blocks, receipt = weak_dyadic_pairs(basis, source_basis, rows, cols, kernel, category=category, order=order)
    if category == 3:
        previous = blocks
        checks = []
        accepted = False
        eta0 = math.sqrt(MU_0/EPSILON_0)
        scale = blocks.new_tensor([[eta0, eta0**2], [1., eta0]])[None, :, :, None, None]
        while order < config.near_max_order:
            order = min(2*order, config.near_max_order)
            blocks, receipt = weak_dyadic_pairs(basis, source_basis, rows, cols, kernel, category=category, order=order)
            difference = torch.linalg.vector_norm(((blocks-previous)*scale).flatten(1), dim=-1)
            norm = torch.linalg.vector_norm((blocks*scale).flatten(1), dim=-1)
            relative = difference/norm.clamp_min(torch.finfo(norm.dtype).tiny)
            checks.append({'order': order, 'maximum_relative_difference': float(relative.max().detach())})
            if bool((relative <= config.near_tolerance).all()):
                accepted = True
                break
            previous = blocks
        if not accepted:
            raise RuntimeError(f'Tensor near-panel quadrature did not converge: {checks}')
        receipt['near_checks'] = checks
    return blocks, receipt


def assemble_tensor_region_weak(basis, medium, config, *, kernel=None, source_basis=None):
    """Assemble one regional tensor weak matrix with explicit accuracy receipts.

    Isotropic samples use the identical scalar weak formulation, including its
    full divergence integration. General tensors use regular dyadic potentials
    and static divergence terms. Fixed surface orders remain unqualified until
    a separate surface-order convergence campaign passes.
    """
    import time
    from .dyadic_green import DyadicGreenKernel
    from .tensor_medium import scalar_medium_if_isotropic
    from .operator import assemble_region_weak, assemble_cross, classify_cross_pairs, IntegrationConfig
    from .quadrature import classify_all_pairs
    if not isinstance(config, IntegrationConfig):
        raise TypeError('Tensor regional assembly requires IntegrationConfig')
    source_basis = basis if source_basis is None else source_basis
    medium.assert_current()
    if any(b.mesh.device != medium.epsilon_r.device or b.mesh.dtype != medium.epsilon_r.real.dtype
           for b in (basis, source_basis)):
        raise ValueError('Tensor regional assembly requires matching CUDA mesh and medium precision')
    if medium.epsilon_r.requires_grad or medium.mu_r.requires_grad:
        raise NotImplementedError('Tensor weak assembly currently requires fixed material samples; coefficient gradients remain supported')
    kernel = DyadicGreenKernel(medium) if kernel is None else kernel
    if kernel.medium is not medium:
        raise ValueError('Regional kernel must bind the exact sampled medium')
    scalar = scalar_medium_if_isotropic(medium)
    if scalar is not None:
        if source_basis is basis:
            matrix, receipt = assemble_region_weak(basis, scalar, config)
        else:
            matrix, receipt = assemble_cross(basis, source_basis, scalar, config)
        receipt.update(formulation_id='anisotropic_dgf_region_v1', kernel_specialization='exact_isotropic_scalar_weak',
                       physical_qualification=False)
        return matrix, receipt
    started = time.perf_counter()
    ne, ns = basis.n_edges, source_basis.n_edges
    nf = len(source_basis.mesh.faces)
    if source_basis is basis:
        classes = classify_all_pairs(basis.mesh)
    else:
        count = len(basis.mesh.faces)*nf
        classes = torch.empty(count, dtype=torch.uint8, device=basis.mesh.device)
        for start in range(0, count, 4096):
            ids = torch.arange(start, min(start+4096, count), device=basis.mesh.device)
            category, distance = classify_cross_pairs(basis, source_basis, ids//nf, ids % nf)
            if not bool((distance > 0).all()):
                raise ValueError('Tensor cross assembly requires strictly separated surfaces')
            classes[start:start+len(ids)] = category
    matrix = medium.epsilon_r.new_zeros((2*ne, 2*ns))
    counts, records = [], []
    for category in range(5):
        selected = torch.nonzero(classes == category).flatten()
        counts.append(len(selected))
        if not len(selected):
            continue
        batch_size = config.far_pair_batch if category == 4 else config.pair_batch
        for ids in selected.split(batch_size):
            rows, cols = ids//nf, ids % nf
            blocks, receipt = _integrated_dyadic_pairs(
                basis, source_basis, rows, cols, kernel, category, config)
            records.append({'pair_count': len(ids), **receipt})
            ri = basis.face_edges[rows, :, None].expand(-1, -1, 3).flatten()
            ci = source_basis.face_edges[cols, None, :].expand(-1, 3, -1).flatten()
            for row in range(2):
                for column in range(2):
                    matrix.index_put_((ri+row*ne, ci+column*ns), blocks[:, row, column].flatten(), accumulate=True)
    torch.cuda.synchronize(matrix.device)
    return matrix, {'formulation_id': 'anisotropic_dgf_region_v1', 'kernel_specialization': 'general_tensor_weyl',
        'contribution': 'exterior_only' if source_basis is not basis else 'single_region',
        'triangle_pair_counts': counts, 'quadrature': vars(config), 'batch_receipts': records,
        'assembly_seconds': time.perf_counter()-started, 'physical_qualification': False,
        'fixed_surface_orders_certified': False, 'spectral_cache': kernel.mode_cache.stats()}


def assemble_generalized_system(system, exterior, config, *, interior_kernels=None, exterior_kernel=None):
    """Compose a shared free-space exterior and body-local scalar/tensor regions.

    Only the exterior contributes cross-body blocks. This weak matrix does not
    by itself supply a lifted RHS, a field decoder or physical qualification.
    """
    from .tensor_medium import TensorMedium
    from .operator import assemble_weak, assemble_region_weak, assemble_cross
    system.assert_current()
    tensor_exterior = isinstance(exterior, TensorMedium)
    if exterior_kernel is not None and not tensor_exterior:
        raise ValueError('An exterior dyadic kernel requires a tensor exterior')
    if tensor_exterior:
        from .dyadic_green import DyadicGreenKernel
        exterior_kernel = DyadicGreenKernel(exterior) if exterior_kernel is None else exterior_kernel
        if exterior_kernel.medium is not exterior:
            raise ValueError('Exterior kernel must bind the exact sampled medium')
    kernels = {} if interior_kernels is None else dict(interior_kernels)
    tensor_regions = {body.region_id for body in system.bodies if isinstance(body.interior, TensorMedium)}
    if set(kernels)-tensor_regions:
        raise ValueError('Interior kernels must identify existing tensor body regions')
    if any(body.interior.frequency != exterior.frequency for body in system.bodies):
        raise ValueError('Every regional contribution must use the common frequency')
    dtype = torch.complex128 if system.dtype == torch.float64 else torch.complex64
    matrix = None if len(system.bodies) == 1 else torch.empty((2*system.total_edges, 2*system.total_edges),
                                                            device=system.device, dtype=dtype)
    records = []
    for i, test in enumerate(system.bodies):
        for j, source in enumerate(system.bodies):
            if tensor_exterior:
                block, outside_receipt = assemble_tensor_region_weak(test.basis, exterior, config,
                    kernel=exterior_kernel, source_basis=source.basis)
                receipt = {'exterior': outside_receipt, 'contribution': 'exterior_only'}
                if i == j:
                    if isinstance(test.interior, TensorMedium):
                        inside, inside_receipt = assemble_tensor_region_weak(test.basis, test.interior,
                            config, kernel=kernels.get(test.region_id))
                    else:
                        inside, inside_receipt = assemble_region_weak(test.basis, test.interior, config)
                    block = block+inside
                    receipt = {'exterior': outside_receipt, 'interior': inside_receipt}
            elif i != j:
                block, receipt = assemble_cross(test.basis, source.basis, exterior, config)
            elif isinstance(test.interior, TensorMedium):
                block, inside = assemble_tensor_region_weak(test.basis, test.interior, config,
                                                            kernel=kernels.get(test.region_id))
                outside, outside_receipt = assemble_region_weak(test.basis, exterior, config)
                block = block+outside
                receipt = {'interior': inside, 'exterior': outside_receipt}
            else:
                block, receipt = assemble_weak(test.basis, exterior, test.interior, config)
            records.append({'test_region': test.region_id, 'source_region': source.region_id, **receipt})
            if len(system.bodies) == 1:
                matrix = block
                continue
            n, m = test.basis.n_edges, source.basis.n_edges
            for row, rs in enumerate((system.J_slices[i], system.M_slices[i])):
                for column, cs in enumerate((system.J_slices[j], system.M_slices[j])):
                    matrix[rs, cs] = block[row*n:(row+1)*n, column*m:(column+1)*m]
    return matrix, {'formulation_id': 'anisotropic_dgf_sie', 'formulation_version': 'passive_tensor_pmchwt_v1',
        'operator_backend': 'dense', 'kernel_id': 'free_space_regional_dyadic_v1',
        'unknown_order': ['all_body_J_then_all_body_M'], 'blocks': records,
        'body_count': len(system.bodies), 'physical_qualification': False}


def tensor_regional_block(test_basis, source_basis, test_edges, source_edges, medium,
                          config, *, kernel=None):
    """Integrate selected tensor RWG rows/columns without a full regional matrix.

    Only complete supports of the requested edges are visited. Classification
    and panel workspaces are batched; the returned block, geometry and the
    spectral kernel's own cache are separate retained costs. This is a fixed
    material primitive, not a public tensor hierarchical solver.
    """
    from .tensor_medium import TensorMedium, scalar_medium_if_isotropic
    from .dyadic_green import DyadicGreenKernel
    from .operator import IntegrationConfig, classify_cross_pairs
    from .quadrature import classify_pairs
    from .nearfield import _selected_edges, _local_indices, scalar_regional_block
    if not isinstance(medium, TensorMedium) or not isinstance(config, IntegrationConfig):
        raise TypeError('Selected tensor blocks require TensorMedium and IntegrationConfig')
    medium.assert_current()
    if medium.epsilon_r.requires_grad or medium.mu_r.requires_grad:
        raise NotImplementedError('Selected tensor blocks require fixed material samples')
    for basis in (test_basis, source_basis):
        mesh = basis.mesh
        if mesh.device != medium.epsilon_r.device or mesh.dtype != medium.epsilon_r.real.dtype:
            raise ValueError('Selected tensor geometry must match medium device and precision')
        if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
            raise ValueError('Selected tensor basis geometry changed')
    if kernel is not None and kernel.medium is not medium:
        raise ValueError('Selected tensor kernel must bind the exact sampled medium')
    scalar = scalar_medium_if_isotropic(medium)
    if scalar is not None:
        block, receipt = scalar_regional_block(test_basis, source_basis, test_edges, source_edges,
                                               (scalar,), config)
        return block, dict(receipt, version='tensor_selected_weak_block_v1',
                           kernel_specialization='exact_isotropic_scalar_weak')
    re, rp, rf = _selected_edges(test_basis, test_edges)
    ce, cp, cf = _selected_edges(source_basis, source_edges)
    kernel = DyadicGreenKernel(medium) if kernel is None else kernel
    nr, nc = len(re), len(ce)
    result = medium.epsilon_r.new_zeros((2*nr, 2*nc))
    counts = [0]*5
    maximum_pairs = maximum_order = evaluations = 0
    classification_batch = max(config.pair_batch, config.far_pair_batch)
    for start in range(0, len(rf)*len(cf), classification_batch):
        ids = torch.arange(start, min(start+classification_batch, len(rf)*len(cf)), device=result.device)
        rows, cols = rf[ids//len(cf)], cf[ids % len(cf)]
        maximum_pairs = max(maximum_pairs, len(ids))
        if test_basis is source_basis:
            classes = classify_pairs(test_basis.mesh, rows, cols)
        else:
            classes, distances = classify_cross_pairs(test_basis, source_basis, rows, cols)
            if not bool((distances > 0).all()):
                raise ValueError('Selected tensor cross blocks require strictly separated surfaces')
        for category in range(5):
            selected = (classes == category).nonzero().flatten()
            counts[category] += len(selected)
            batch = config.far_pair_batch if category == 4 else config.pair_batch
            for subset in selected.split(batch):
                if not len(subset):
                    continue
                r, c = rows[subset], cols[subset]
                blocks, receipt = _integrated_dyadic_pairs(
                    test_basis, source_basis, r, c, kernel, category, config)
                maximum_order = max(maximum_order, receipt['panel_order'])
                evaluations += 1+len(receipt.get('near_checks', ()))
                ri, rv = _local_indices(test_basis.face_edges[r], re, rp)
                ci, cv = _local_indices(source_basis.face_edges[c], ce, cp)
                keep = rv[:, :, None] & cv[:, None, :]
                rr = ri[:, :, None].expand(-1, -1, 3)[keep]
                cc = ci[:, None, :].expand(-1, 3, -1)[keep]
                for row in range(2):
                    for column in range(2):
                        result.index_put_((rr+row*nr, cc+column*nc),
                                          blocks[:, row, column][keep], accumulate=True)
    return result, {'version': 'tensor_selected_weak_block_v1',
        'kernel_specialization': 'general_tensor_weyl', 'triangle_pair_counts': counts,
        'maximum_classification_pairs': maximum_pairs, 'maximum_panel_order': maximum_order,
        'panel_batch_evaluations': evaluations, 'matrix_bytes': result.numel()*result.element_size(),
        'fixed_surface_orders_certified': False, 'physical_qualification': False}

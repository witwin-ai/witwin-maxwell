"""Selected RWG weak blocks with bounded triangle-pair workspaces.

These blocks reuse the dense reference integration rules, including all five
pair classes. They never allocate a global matrix or global triangle-pair list.
"""
from dataclasses import asdict
import math
import torch
from .kernels import FREE_SPACE
from .operator import (NEAR_FALLBACK_BATCH, IntegrationConfig, PreparedRWGGeometry, _anisotropic_touching_integrals,
                       _pair_integrals, _adaptive_near_integrals, _merge_rule_stats, classify_cross_pairs)
from .quadrature import anisotropic_touching_pairs, classify_pairs, _classify_pairs
from .physics import Medium
from ..constants import EPSILON_0, MU_0


def _touching_values(basis, rows, cols, medium, config, category, *, geometry, statistics=None):
    """Self/edge/vertex D/K blocks with the dense assembly's rule choice per pair.

    Pairs with a high-aspect triangle take the anisotropy-aware rule, the rest
    the joint Duffy rules, exactly as ``_assemble_scalar_regions`` routes them.
    """
    order = config.singular_order
    mask = anisotropic_touching_pairs(basis.mesh.triangles[rows], basis.mesh.triangles[cols])
    if not bool(mask.any()):
        return _pair_integrals(basis, rows, cols, medium, order, order, True, category, kernel=FREE_SPACE,
                               geometry=geometry)
    dtype = torch.complex128 if basis.mesh.dtype == torch.float64 else torch.complex64
    d = torch.empty((len(rows), 3, 3), device=rows.device, dtype=dtype)
    k = torch.empty_like(d)
    regular = (~mask).nonzero().flatten()
    if len(regular):
        dr, kr = _pair_integrals(basis, rows[regular], cols[regular], medium, order, order, True, category,
                                 kernel=FREE_SPACE, geometry=geometry)
        d[regular], k[regular] = dr.to(dtype), kr.to(dtype)
    for part in mask.nonzero().flatten().split(NEAR_FALLBACK_BATCH):
        da, ka, record = _anisotropic_touching_integrals(basis, rows[part], cols[part], medium, config, category,
                                                          kernel=FREE_SPACE)
        d[part], k[part] = da.to(dtype), ka.to(dtype)
        if statistics is not None:
            _merge_rule_stats(statistics, 'touching_anisotropic_'+('self', 'edge', 'vertex')[category], record)
    return d, k


def _selected_edges(basis, edges):
    if (not isinstance(edges, torch.Tensor) or edges.device != basis.mesh.device
            or edges.dtype != torch.int64 or edges.ndim != 1 or not len(edges)):
        raise ValueError('Selected edges must be a nonempty CUDA int64 vector on the basis device')
    if not bool(((edges >= 0) & (edges < basis.n_edges)).all()):
        raise ValueError('Selected edge index is outside its basis')
    sorted_edges, positions = edges.sort()
    if bool((sorted_edges[1:] == sorted_edges[:-1]).any()):
        raise ValueError('Selected edges must be unique')
    faces = torch.unique(torch.cat((basis.plus_faces[edges], basis.minus_faces[edges])), sorted=True)
    return sorted_edges, positions, faces


def _local_indices(face_edges, selected, positions):
    indices = torch.searchsorted(selected, face_edges)
    safe = indices.clamp_max(len(selected)-1)
    valid = (indices < len(selected)) & (selected[safe] == face_edges)
    return positions[safe], valid


def scalar_regional_block(test_basis, source_basis, test_edges, source_edges, media,
                          config=IntegrationConfig(), *, geometry=None):
    """Return the selected [J,M] block for free-space scalar regions.

    Each medium contributes once. A self-body PMCHWT block supplies exterior
    and interior media; a cross-body block supplies only the exterior. Edge
    selection order is preserved. Periodic corrections and tensor regions
    require their own explicit composition and cannot enter this entrypoint.
    """
    return _scalar_regional_accumulate(test_basis, source_basis, test_edges, source_edges,
        media, config, geometry=geometry, components=False)


def _validate_scalar_channel_medium(medium):
    if not isinstance(medium, Medium):
        raise TypeError('Selected D/K channels require exactly one sampled scalar medium')
    if (type(medium.frequency) not in (int, float) or not math.isfinite(medium.frequency)
            or medium.frequency <= 0):
        raise ValueError('Selected D/K channels require a finite positive scalar frequency')
    for value in (medium.epsilon_r, medium.mu_r, medium.k, medium.eta):
        if (type(value) not in (int, float, complex)
                or not math.isfinite(complex(value).real) or not math.isfinite(complex(value).imag)):
            raise ValueError('Selected D/K channels require fixed finite scalar medium coefficients')
    if medium.k == 0 or complex(medium.k).imag < 0:
        raise ValueError('Selected D/K channels require a finite-frequency passive scalar wave number')


def scalar_regional_channels(test_basis, source_basis, test_edges, source_edges, medium,
                             config=IntegrationConfig(), *, geometry=None):
    """Independent original D/K integrals for one fixed scalar medium.

    Multiplying material factors after accumulation may change floating-point
    order relative to the unchanged default four-block path.
    """
    _validate_scalar_channel_medium(medium)
    if not isinstance(config, IntegrationConfig):
        raise TypeError('Selected D/K channels require IntegrationConfig')
    values, metadata = _scalar_regional_accumulate(test_basis, source_basis,
        test_edges, source_edges, (medium,), config, geometry=geometry, components=True)
    metadata['version'] = 'scalar_selected_weak_channels_v1'
    metadata['channel_bytes'] = metadata.pop('matrix_bytes')
    # No shared doubled storage hidden behind component views.
    return values[0].clone(), values[1].clone(), metadata


def _scalar_regional_accumulate(test_basis, source_basis, test_edges, source_edges, media,
                                config, *, geometry=None, components=False):
    if (test_basis.mesh.device.type != 'cuda' or source_basis.mesh.device != test_basis.mesh.device
            or source_basis.mesh.dtype != test_basis.mesh.dtype):
        raise ValueError('Selected blocks require matching CUDA geometry spaces')
    for basis in (test_basis, source_basis):
        mesh = basis.mesh
        if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
            raise ValueError('Basis geometry changed; rebuild before selected-block integration')
    if geometry is not None:
        geometry[0].assert_current(test_basis)
        geometry[1].assert_current(source_basis)
    media = tuple(media)
    if not media or any(not isinstance(m, Medium) for m in media):
        raise TypeError('Selected scalar blocks require sampled scalar media')
    if any(m.frequency != media[0].frequency for m in media):
        raise ValueError('Regional contributions must share one frequency')
    if source_basis is not test_basis and len(media) != 1:
        raise ValueError('Cross-body blocks contain only the common exterior medium')
    re, rp, rf = _selected_edges(test_basis, test_edges)
    ce, cp, cf = _selected_edges(source_basis, source_edges)
    nr, nc = len(re), len(ce)
    dtype = torch.complex128 if test_basis.mesh.dtype == torch.float64 else torch.complex64
    shape = (2, nr, nc) if components else (2*nr, 2*nc)
    result = torch.zeros(shape, device=test_basis.mesh.device, dtype=dtype)
    counts = [0]*5
    near = {'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
    maximum_pairs = 0
    # Classification itself is bounded, independently of the requested block.
    classification_batch = max(config.pair_batch, config.far_pair_batch)
    for start in range(0, len(rf)*len(cf), classification_batch):
        ids = torch.arange(start, min(start+classification_batch, len(rf)*len(cf)), device=result.device)
        rows, cols = rf[ids//len(cf)], cf[ids % len(cf)]
        maximum_pairs = max(maximum_pairs, len(ids))
        if source_basis is test_basis:
            classes = (classify_pairs(test_basis.mesh, rows, cols) if geometry is None else
                _classify_pairs(test_basis.mesh, rows, cols, geometry[0].centres, geometry[0].radii, 2.))
        else:
            classes, distances = classify_cross_pairs(test_basis, source_basis, rows, cols, geometry=geometry)
            if not bool((distances > 0).all()):
                raise ValueError('Cross-body selected blocks require strictly separated surfaces')
        for category in range(5):
            selected = (classes == category).nonzero().flatten()
            counts[category] += len(selected)
            batch = config.far_pair_batch if category == 4 else config.pair_batch
            for subset in selected.split(batch):
                if not len(subset):
                    continue
                r, c = rows[subset], cols[subset]
                ri, rv = _local_indices(test_basis.face_edges[r], re, rp)
                ci, cv = _local_indices(source_basis.face_edges[c], ce, cp)
                keep = rv[:, :, None] & cv[:, None, :]
                rr = ri[:, :, None].expand(-1, -1, 3)[keep]
                cc = ci[:, None, :].expand(-1, 3, -1)[keep]
                order = config.order if category == 4 else config.singular_order
                for medium in media:
                    if category == 3:
                        d, k, stats = _adaptive_near_integrals(test_basis, r, c, medium, config,
                                                             kernel=FREE_SPACE, source_basis=source_basis, geometry=geometry)
                        for name in ('maximum_order', 'maximum_accepted_relative_difference'):
                            near[name] = max(near[name], stats[name])
                        near['pair_evaluations'] += stats['pair_evaluations']
                    elif category < 3:
                        d, k = _touching_values(test_basis, r, c, medium, config, category, geometry=geometry)
                    else:
                        d, k = _pair_integrals(test_basis, r, c, medium, order, order,
                            category != 4, category, kernel=FREE_SPACE, source_basis=source_basis, geometry=geometry)
                    if components:
                        result[0].index_put_((rr, cc), d[keep].to(dtype), accumulate=True)
                        result[1].index_put_((rr, cc), k[keep].to(dtype), accumulate=True)
                    else:
                        omega = 2*math.pi*medium.frequency
                        for i, j, values in ((0, 0, k), (1, 1, k),
                                (0, 1, 1j*omega*EPSILON_0*medium.epsilon_r*d),
                                (1, 0, -1j*omega*MU_0*medium.mu_r*d)):
                            result.index_put_((rr+i*nr, cc+j*nc), values[keep].to(dtype), accumulate=True)
    return result, {'version': 'scalar_selected_weak_block_v1', 'kernel_id': FREE_SPACE.kernel_id,
        'quadrature': asdict(config), 'triangle_pair_counts': counts,
        'near_quadrature': near, 'maximum_classification_pairs': maximum_pairs,
        'matrix_bytes': result.numel()*result.element_size(), 'physical_qualification': False}


class ScalarSystemBlocks:
    """On-demand free-space multi-body scalar blocks in global J/M ordering.

    The provider owns no global matrix. Its edge selections return all selected
    J coefficients followed by all selected M coefficients. Full-support boxes
    can seed a shared geometry tree; material contributions remain body local.
    """
    def __init__(self, system, exterior, config=IntegrationConfig()):
        system.assert_current()
        if not isinstance(config, IntegrationConfig):
            raise TypeError('Selected system blocks require IntegrationConfig')
        if not isinstance(exterior, Medium) or any(not isinstance(b.interior, Medium) for b in system.bodies):
            raise TypeError('Selected system blocks require scalar media')
        if any(b.interior.frequency != exterior.frequency for b in system.bodies):
            raise ValueError('All system regions must share one frequency')
        self.system, self.exterior, self.config = system, exterior, config
        self._bound = (system, exterior, config)
        self.geometry = tuple(PreparedRWGGeometry(body.basis) for body in system.bodies)
        self._geometry = self.geometry
        self.edge_stops = torch.tensor([b.edge_offset+b.basis.n_edges for b in system.bodies],
                                       device=system.device, dtype=torch.int64)
        self._edge_stops = (id(self.edge_stops), self.edge_stops._version)
        self.metadata = {'version': 'scalar_system_blocks_v1', 'kernel_id': FREE_SPACE.kernel_id,
            'block_requests': 0, 'triangle_pair_counts': [0]*5, 'maximum_classification_pairs': 0,
            'maximum_block_bytes': 0, 'physical_qualification': False}

    def assert_current(self):
        if any(a is not b for a, b in zip((self.system, self.exterior, self.config), self._bound)):
            raise ValueError('Selected-block provider binding changed')
        self.system.assert_current()
        if self.geometry is not self._geometry:
            raise ValueError('Prepared system geometry binding changed')
        if (id(self.edge_stops), self.edge_stops._version) != self._edge_stops:
            raise ValueError('Prepared system edge routing changed')
        for body, geometry in zip(self.system.bodies, self.geometry):
            geometry.assert_current(body.basis)

    def geometry_tensors(self):
        return (*tuple(t for geometry in self.geometry for t in geometry.tensors()), self.edge_stops)

    def input_tensors(self):
        """Distinct CUDA storages of the shared input system/bases/meshes."""
        objects = [self.system]
        for body in self.system.bodies:
            objects.extend((body.basis, body.mesh))
        unique = {}
        for obj in objects:
            for value in vars(obj).values():
                if isinstance(value, torch.Tensor):
                    key = (value.device, value.untyped_storage().data_ptr())
                    unique[key] = value
        return tuple(unique.values())

    def support_boxes(self):
        self.assert_current()
        lower, upper = [], []
        for body in self.system.bodies:
            basis = body.basis
            triangles = self.geometry[body.region_id-1].triangles
            plus, minus = triangles[basis.plus_faces], triangles[basis.minus_faces]
            lower.append(torch.minimum(plus.amin(dim=1), minus.amin(dim=1)))
            upper.append(torch.maximum(plus.amax(dim=1), minus.amax(dim=1)))
        return torch.cat(lower), torch.cat(upper)

    def _validate_selection(self, edges):
        if (not isinstance(edges, torch.Tensor) or edges.device != self.system.device
                or edges.dtype != torch.int64 or edges.ndim != 1 or not len(edges)):
            raise ValueError('Global edge selections require nonempty CUDA int64 vectors')
        if (not bool(((edges >= 0) & (edges < self.system.total_edges)).all())
                or len(edges.unique()) != len(edges)):
            raise ValueError('Global edge selections must be unique and in range')

    def __call__(self, test_edges, source_edges):
        return self._regional_selection(test_edges, source_edges, include_interiors=True)

    def _body_selection(self, edges):
        body_ids = torch.bucketize(edges, self.edge_stops, right=True)
        return tuple((index, (body_ids == index).nonzero().flatten())
                     for index in body_ids.unique().cpu().tolist())

    def exterior_block(self, test_edges, source_edges):
        """Common exterior only, including same-body pairs on a global tree."""
        return self._regional_selection(test_edges, source_edges, include_interiors=False)

    def exterior_cross_block(self, source_provider, test_edges, source_edges):
        """Exterior-only block between separately routed global coefficient spaces."""
        if (not isinstance(source_provider, ScalarSystemBlocks)
                or source_provider.exterior is not self.exterior or source_provider.config is not self.config
                or source_provider.system.device != self.system.device or source_provider.system.dtype != self.system.dtype):
            raise ValueError('Global exterior blocks require shared medium, integration, device and precision')
        return self._regional_selection(test_edges, source_edges, include_interiors=False, source_provider=source_provider)

    def exterior_channels(self, test_edges, source_edges, *, source_provider=None):
        """Independent original D/K on globally routed exterior selections."""
        source_provider = self if source_provider is None else source_provider
        if (not isinstance(source_provider, ScalarSystemBlocks)
                or source_provider.exterior is not self.exterior or source_provider.config is not self.config
                or source_provider.system.device != self.system.device or source_provider.system.dtype != self.system.dtype):
            raise ValueError('Global exterior channels require shared medium, integration and native spaces')
        _validate_scalar_channel_medium(self.exterior)
        if self.system.dtype != torch.float64:
            raise ValueError('Global exterior channels require native float64 geometry')
        return self._regional_selection(test_edges, source_edges, include_interiors=False,
            source_provider=source_provider, channels=True)

    def _regional_selection(self, test_edges, source_edges, *, include_interiors, source_provider=None, channels=False):
        self.assert_current()
        source_provider = self if source_provider is None else source_provider
        source_provider.assert_current()
        self._validate_selection(test_edges)
        source_provider._validate_selection(source_edges)
        nr, nc = len(test_edges), len(source_edges)
        dtype = torch.complex128 if self.system.dtype == torch.float64 else torch.complex64
        result = torch.zeros((2, nr, nc) if channels else (2*nr, 2*nc), device=self.system.device, dtype=dtype)
        source_selection = source_provider._body_selection(source_edges)
        for test_index, selected_r in self._body_selection(test_edges):
            test = self.system.bodies[test_index]
            for source_index, selected_c in source_selection:
                source = source_provider.system.bodies[source_index]
                if channels:
                    d, k, receipt = scalar_regional_channels(test.basis, source.basis,
                        test_edges[selected_r]-test.edge_offset, source_edges[selected_c]-source.edge_offset,
                        self.exterior, self.config,
                        geometry=(self.geometry[test.region_id-1], source_provider.geometry[source.region_id-1]))
                    result[0, selected_r[:, None], selected_c] = d
                    result[1, selected_r[:, None], selected_c] = k
                else:
                    block, receipt = self._body_block(test, source,
                        test_edges[selected_r]-test.edge_offset, source_edges[selected_c]-source.edge_offset,
                        include_interiors=include_interiors, source_provider=source_provider)
                    rr, cc = torch.cat((selected_r, selected_r+nr)), torch.cat((selected_c, selected_c+nc))
                    result[rr[:, None], cc] = block
                for i, count in enumerate(receipt['triangle_pair_counts']):
                    self.metadata['triangle_pair_counts'][i] += count
                self.metadata['maximum_classification_pairs'] = max(
                    self.metadata['maximum_classification_pairs'], receipt['maximum_classification_pairs'])
        self.metadata['block_requests'] += 1
        self.metadata['maximum_block_bytes'] = max(self.metadata['maximum_block_bytes'], result.numel()*result.element_size())
        return (result[0].clone(), result[1].clone()) if channels else result

    def _body_block(self, test, source, rows, cols, *, include_interiors, source_provider):
        media = (self.exterior, test.interior) if include_interiors and test is source else (self.exterior,)
        return scalar_regional_block(test.basis, source.basis, rows, cols, media, self.config,
            geometry=(self.geometry[test.region_id-1], source_provider.geometry[source.region_id-1]))

    def _principal_blocks_many(self, edges, *, entry_budget=1 << 16):
        """Equal-size pFFT leaf blocks with bounded paired-edge quadrature.

        Only the reduction of the four support contributions changes relative
        to selected blocks; quadrature, material factors and J/M layout agree.
        """
        self.assert_current()
        count, width = edges.shape
        result = torch.zeros((count, 2*width, 2*width), device=edges.device, dtype=torch.complex128)
        body_ids = torch.bucketize(edges, self.edge_stops, right=True)
        counts = torch.zeros(5, device=edges.device, dtype=torch.int64)
        maximum_pairs = torch.zeros((), device=edges.device, dtype=torch.int64)
        step = max(1, entry_budget//(width*width))
        for start in range(0, count, step):
            group = edges[start:start+step]
            ids = body_ids[start:start+step]
            row_edges = group[:, :, None].expand(-1, width, width).reshape(-1)
            col_edges = group[:, None, :].expand(-1, width, width).reshape(-1)
            row_bodies = ids[:, :, None].expand(-1, width, width).reshape(-1)
            col_bodies = ids[:, None, :].expand(-1, width, width).reshape(-1)
            blocks = result[start:start+step]
            for ti, test in enumerate(self.system.bodies):
                for si, source in enumerate(self.system.bodies):
                    selected = ((row_bodies == ti) & (col_bodies == si)).nonzero().flatten()
                    if not len(selected):
                        continue
                    b, r, c = selected//(width*width), selected//width % width, selected % width
                    re, ce = row_edges[selected]-test.edge_offset, col_edges[selected]-source.edge_offset
                    # Preserve selected-block accounting, counting a face pair once
                    # per logical block even when quadrature shares it across leaves.
                    rf = _edge_face_slots(test.basis)[0][re]
                    cf = _edge_face_slots(source.basis)[0][ce]
                    nt, ns = len(test.mesh.faces), len(source.mesh.faces)
                    pairs = torch.unique(((b[:, None, None]*nt+rf[:, :, None])*ns+cf[:, None, :]).flatten())
                    tr, tc = pairs//ns % nt, pairs % ns
                    _, per_block = torch.unique_consecutive(pairs//(nt*ns), return_counts=True)
                    maximum_pairs = torch.maximum(maximum_pairs, per_block.max().clamp_max(
                        max(self.config.pair_batch, self.config.far_pair_batch)))
                    geometry = (self.geometry[ti], self.geometry[si])
                    classes = (_classify_pairs(test.mesh, tr, tc, geometry[0].centres, geometry[0].radii, 2.)
                               if ti == si else classify_cross_pairs(test.basis, source.basis, tr, tc,
                                                                    geometry=geometry)[0])
                    counts += (classes[:, None] == torch.arange(5, device=edges.device)).sum(0)
                    media = (self.exterior, test.interior) if ti == si else (self.exterior,)
                    for medium in media:
                        d, k = scalar_edge_pair_channels(test.basis, source.basis,
                            re, ce, medium, self.config, geometry=geometry)
                        blocks[b, r, c] += k
                        blocks[b, r+width, c+width] += k
                        blocks[b, r, c+width] += 1j*medium.omega*EPSILON_0*medium.epsilon_r*d
                        blocks[b, r+width, c] += -1j*medium.omega*MU_0*medium.mu_r*d
        self.metadata['block_requests'] += count
        for category, value in enumerate(counts.tolist()):
            self.metadata['triangle_pair_counts'][category] += value
        self.metadata['maximum_classification_pairs'] = max(self.metadata['maximum_classification_pairs'],
                                                            int(maximum_pairs))
        self.metadata['maximum_block_bytes'] = max(self.metadata['maximum_block_bytes'], (2*width)**2*16)
        return result


    def sampled_action(self, test_edges, value, *, source_batch=64):
        """Original weak action on selected test edges with bounded source blocks."""
        return self._sampled_action(test_edges, value, source_batch, self)

    def _sampled_action(self, test_edges, value, source_batch, block_action):
        self.assert_current()
        self._validate_selection(test_edges)
        dtype = torch.complex128 if self.system.dtype == torch.float64 else torch.complex64
        total = self.system.total_edges
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != 2*total
                or value.device != self.system.device or value.dtype != dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Original action requires finite global complex CUDA coefficients')
        if type(source_batch) is not int or source_batch < 1:
            raise ValueError('source_batch must be a positive integer')
        result = torch.zeros((*value.shape[:-1], 2*len(test_edges)), device=value.device, dtype=value.dtype)
        for start in range(0, total, source_batch):
            columns = torch.arange(start, min(start+source_batch, total), device=value.device)
            block = block_action(test_edges, columns)
            coefficients = torch.cat((value[..., columns], value[..., columns+total]), dim=-1)
            result = result+coefficients@block.mT
        return result

    def exterior_channels_many(self, row_edges, col_edges, *, source_provider=None, entry_budget=1 << 20):
        """Global exterior D/K for B equal-shape selections, routed by body pair."""
        source_provider = self if source_provider is None else source_provider
        if (not isinstance(source_provider, ScalarSystemBlocks) or source_provider.exterior is not self.exterior
                or source_provider.config is not self.config or self.system.dtype != torch.float64):
            raise ValueError('Global exterior channels require shared medium, integration and native spaces')
        _validate_scalar_channel_medium(self.exterior)
        self.assert_current()
        source_provider.assert_current()
        count, nr = row_edges.shape
        nc = col_edges.shape[1]
        d = torch.zeros((count, nr, nc), device=row_edges.device, dtype=torch.complex128)
        k = torch.zeros_like(d)
        row_body = torch.bucketize(row_edges, self.edge_stops, right=True)
        col_body = torch.bucketize(col_edges, source_provider.edge_stops, right=True)
        step = max(1, entry_budget//(nr*nc))
        for start in range(0, count, step):
            rb, cb = row_body[start:start+step], col_body[start:start+step]
            re, ce = row_edges[start:start+step], col_edges[start:start+step]
            pair_r = rb[:, :, None].expand(-1, nr, nc)
            pair_c = cb[:, None, :].expand(-1, nr, nc)
            edge_r = re[:, :, None].expand(-1, nr, nc)
            edge_c = ce[:, None, :].expand(-1, nr, nc)
            for ti, si in torch.stack((pair_r.reshape(-1), pair_c.reshape(-1)), 1).unique(dim=0).tolist():
                mask = (pair_r == ti) & (pair_c == si)
                test, source = self.system.bodies[ti], source_provider.system.bodies[si]
                dv, kv = scalar_edge_pair_channels(test.basis, source.basis, edge_r[mask]-test.edge_offset,
                    edge_c[mask]-source.edge_offset, self.exterior, self.config,
                    geometry=(self.geometry[test.region_id-1], source_provider.geometry[source.region_id-1]),
                    statistics=self.metadata)
                d[start:start+step][mask], k[start:start+step][mask] = dv, kv
        self.metadata['block_requests'] += count
        return d, k


class ScalarEdgeBlockSampler:
    """Coefficient-row/column access to a selected global physical RWG block."""
    def __init__(self, provider, test_edges, source_edges):
        provider.assert_current()
        provider._validate_selection(test_edges)
        provider._validate_selection(source_edges)
        self.provider = provider
        self._bound_provider = provider
        self.test_edges, self.source_edges = test_edges.clone(), source_edges.clone()
        self._versions = tuple((id(t), t._version) for t in (self.test_edges, self.source_edges))
        self.shape = (2*len(test_edges), 2*len(source_edges))
        self.device = provider.system.device
        self.dtype = torch.complex128 if provider.system.dtype == torch.float64 else torch.complex64

    def assert_current(self):
        if self.provider is not self._bound_provider:
            raise ValueError('Selected block sampler provider changed')
        self.provider.assert_current()
        if tuple((id(t), t._version) for t in (self.test_edges, self.source_edges)) != self._versions:
            raise ValueError('Selected block sampler edges changed')

    def sample_rows(self, rows):
        self.assert_current()
        if (not isinstance(rows, torch.Tensor) or rows.device != self.device or rows.dtype != torch.int64
                or rows.ndim != 1 or not len(rows) or not bool(((rows >= 0) & (rows < self.shape[0])).all())):
            raise ValueError('Sample row indices must be a nonempty in-range CUDA int64 vector')
        n = len(self.test_edges)
        positions, inverse = torch.unique(rows % n, sorted=True, return_inverse=True)
        values = self.provider(self.test_edges[positions], self.source_edges)
        return values[inverse+(rows >= n).to(torch.int64)*len(positions)]

    def sample_column(self, column):
        self.assert_current()
        if type(column) is not int or not 0 <= column < self.shape[1]:
            raise ValueError('Sample column must be an in-range integer')
        n = len(self.source_edges)
        values = self.provider(self.test_edges, self.source_edges[column % n:column % n+1])
        return values[:, int(column >= n)]


def _edge_face_slots(basis):
    """Per edge: (plus, minus) faces and the edge's local slot in each face."""
    cached = getattr(basis, '_edge_face_slots', None)
    if cached is not None and cached[0] == basis._mesh_version:
        return cached[1]
    edges = torch.arange(basis.n_edges, device=basis.mesh.device)
    faces = torch.stack((basis.plus_faces, basis.minus_faces), 1)
    slots = (basis.face_edges[faces] == edges[:, None, None]).to(torch.int8).argmax(-1)
    value = (faces, slots)
    basis._edge_face_slots = (basis._mesh_version, value)
    return value


def scalar_edge_pair_channels(test_basis, source_basis, test_edges, source_edges, medium,
                              config=IntegrationConfig(), *, geometry, statistics=None):
    """Original D/K values for many paired RWG edges, sharing each face pair once.

    Entry m is the same Galerkin integral that ``scalar_regional_channels``
    stores at (test_edges[m], source_edges[m]): the sum over the two supports
    of each edge. Unique triangle pairs across the whole request are classified
    and integrated once with the unchanged per-class rules and batch bounds.
    Only the summation order of the four support contributions can differ.
    """
    _validate_scalar_channel_medium(medium)
    if test_basis.mesh.device.type != 'cuda' or source_basis.mesh.device != test_basis.mesh.device:
        raise ValueError('Paired channels require matching CUDA geometry spaces')
    geometry[0].assert_current(test_basis)
    geometry[1].assert_current(source_basis)
    if (test_edges.shape != source_edges.shape or test_edges.ndim != 1 or test_edges.dtype != torch.int64
            or source_edges.dtype != torch.int64):
        raise ValueError('Paired channels require matching int64 edge vectors')
    dtype = torch.complex128 if test_basis.mesh.dtype == torch.float64 else torch.complex64
    device = test_basis.mesh.device
    if not len(test_edges):
        empty = torch.zeros(0, device=device, dtype=dtype)
        return empty, empty.clone()
    row_faces, row_slots = (t[test_edges] for t in _edge_face_slots(test_basis))
    col_faces, col_slots = (t[source_edges] for t in _edge_face_slots(source_basis))
    width = len(source_basis.mesh.faces)
    keys = (row_faces[:, :, None]*width+col_faces[:, None, :]).reshape(-1)
    unique, inverse = torch.unique(keys, return_inverse=True)
    rows, cols = unique//width, unique % width
    classes = torch.empty(len(unique), device=device, dtype=torch.uint8)
    step = max(config.pair_batch, config.far_pair_batch)
    for start in range(0, len(unique), step):
        r, c = rows[start:start+step], cols[start:start+step]
        if source_basis is test_basis:
            classes[start:start+step] = _classify_pairs(test_basis.mesh, r, c, geometry[0].centres,
                                                        geometry[0].radii, 2.)
        else:
            value, distances = classify_cross_pairs(test_basis, source_basis, r, c, geometry=geometry)
            if not bool((distances > 0).all()):
                raise ValueError('Cross-body selected blocks require strictly separated surfaces')
            classes[start:start+step] = value
    d_values = torch.empty((len(unique), 3, 3), device=device, dtype=dtype)
    k_values = torch.empty_like(d_values)
    for category in range(5):
        selected = (classes == category).nonzero().flatten()
        if statistics is not None:
            statistics['triangle_pair_counts'][category] += len(selected)
        batch = config.far_pair_batch if category == 4 else config.pair_batch
        order = config.order if category == 4 else config.singular_order
        for subset in selected.split(batch):
            if not len(subset):
                continue
            r, c = rows[subset], cols[subset]
            if category == 3:
                d, k, near = _adaptive_near_integrals(test_basis, r, c, medium, config, kernel=FREE_SPACE,
                                                      source_basis=source_basis, geometry=geometry)
                if statistics is not None and 'fallback' in near:
                    _merge_rule_stats(statistics, 'near_fallback', near['fallback'])
            elif category < 3:
                d, k = _touching_values(test_basis, r, c, medium, config, category, geometry=geometry,
                                        statistics=statistics)
            else:
                d, k = _pair_integrals(test_basis, r, c, medium, order, order, category != 4, category,
                                       kernel=FREE_SPACE, source_basis=source_basis, geometry=geometry)
            d_values[subset], k_values[subset] = d.to(dtype), k.to(dtype)
    inverse = inverse.reshape(-1, 2, 2)
    slot_r = row_slots[:, :, None].expand(-1, 2, 2)
    slot_c = col_slots[:, None, :].expand(-1, 2, 2)
    d = d_values[inverse, slot_r, slot_c].sum((1, 2))
    k = k_values[inverse, slot_r, slot_c].sum((1, 2))
    return d, k


def scalar_block_channels(test_basis, source_basis, row_edges, col_edges, medium, config=IntegrationConfig(),
                          *, geometry, statistics=None, entry_budget=1 << 20):
    """D/K blocks [B, nr, nc] for B equal-shape selections in one body pair."""
    count, nr = row_edges.shape
    nc = col_edges.shape[1]
    dtype = torch.complex128 if test_basis.mesh.dtype == torch.float64 else torch.complex64
    d = torch.empty((count, nr, nc), device=row_edges.device, dtype=dtype)
    k = torch.empty_like(d)
    step = max(1, entry_budget//(nr*nc))
    for start in range(0, count, step):
        r = row_edges[start:start+step]
        c = col_edges[start:start+step]
        er = r[:, :, None].expand(-1, nr, nc).reshape(-1)
        ec = c[:, None, :].expand(-1, nr, nc).reshape(-1)
        dv, kv = scalar_edge_pair_channels(test_basis, source_basis, er, ec, medium, config,
                                           geometry=geometry, statistics=statistics)
        d[start:start+step], k[start:start+step] = dv.reshape(-1, nr, nc), kv.reshape(-1, nr, nc)
    return d, k

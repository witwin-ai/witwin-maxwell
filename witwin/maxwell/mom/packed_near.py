"""Owned variable-size block CSR with bounded grouped CUDA actions.

CSR entries identify geometric leaf nodes. Equal-shaped exact blocks share
contiguous tensor storage; no padding or global electromagnetic matrix is used.
This is a storage/action primitive, not a complete setup-memory admission rule.
"""
from collections import defaultdict
from dataclasses import asdict
import math
import torch


def _check_node_maps(records, size, device, range_message, consistency_message):
    """Validate (node, indices) maps with one CUDA reduction per index length.

    Equivalent to checking every record individually: each map is a nonempty
    CUDA int64 vector with unique entries (in [0, size) when ``size`` is given),
    and every occurrence of a node carries an identical map.
    """
    by_length, lengths = defaultdict(list), {}
    for node, indices in records:
        if (not isinstance(indices, torch.Tensor) or indices.device != device or indices.dtype != torch.int64
                or indices.ndim != 1 or not len(indices)):
            raise ValueError(range_message)
        if lengths.setdefault(node, len(indices)) != len(indices):
            raise ValueError(consistency_message)
        by_length[len(indices)].append((node, indices))
    for length, group in by_length.items():
        stacked = torch.stack([indices for _, indices in group])
        if size is not None and not bool(((stacked >= 0) & (stacked < size)).all()):
            raise ValueError(range_message)
        ordered = stacked.sort(1).values
        if length > 1 and bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError(range_message)
        nodes = torch.tensor([node for node, _ in group], device=device)
        _, inverse = torch.unique(nodes, return_inverse=True)
        positions = torch.arange(len(group), device=device)
        first = torch.full((int(inverse.max())+1,), len(group), device=device).scatter_reduce(
            0, inverse, positions, reduce='amin')
        if not torch.equal(stacked, stacked[first[inverse]]):
            raise ValueError(consistency_message)


class PackedNearBlocks:
    """Pack (test node, source node, rows, columns, matrix) records.

    Node IDs are nonnegative integers and each node has one fixed coefficient
    map. The input matrices are fixed; gradients are supported for action inputs.
    Each forward gather/product uses at most action_batch block entries per
    RHS chunk. Autograd may retain intermediates across chunks for backward.
    """
    def __init__(self, records, size, *, action_batch=32, retained_budget_bytes=None):
        records = tuple(records)
        if not records:
            raise ValueError('Packing requires at least one exact block')
        first = records[0][4]
        if not isinstance(first, torch.Tensor):
            raise ValueError('Exact blocks require native complex CUDA tensors')
        values = {(a, b): value for a, b, _, _, value in records}
        self._initialize(tuple(record[:4] for record in records), size,
            lambda a, b, rows, cols: values[a, b], device=first.device, dtype=first.dtype,
            action_batch=action_batch, retained_budget_bytes=retained_budget_bytes)

    @classmethod
    def from_layout(cls, layout, size, block_factory, *, device, dtype,
                    action_batch=32, retained_budget_bytes=None):
        """Allocate final groups first, then produce and release one block at a time.

        Layout entries are (test node, source node, rows, columns). The factory
        receives those four arguments and must return one finite fixed matrix.
        A failed build never returns a partially initialized packed object.
        """
        instance = cls.__new__(cls)
        instance._initialize(tuple(layout), size, block_factory, device=device, dtype=dtype,
            action_batch=action_batch, retained_budget_bytes=retained_budget_bytes)
        return instance

    def _initialize(self, layout, size, block_factory, *, device, dtype,
                    action_batch, retained_budget_bytes):
        if type(size) is not int or size < 1 or type(action_batch) is not int or action_batch < 1:
            raise ValueError('Positive integer size and action_batch are required')
        if not layout:
            raise ValueError('Packing requires at least one exact block')
        if retained_budget_bytes is not None and (type(retained_budget_bytes) is not int or retained_budget_bytes < 1):
            raise ValueError('Retained byte budget must be a positive integer')
        device = torch.device(device)
        if device.type != 'cuda' or dtype not in (torch.complex64, torch.complex128):
            raise ValueError('Exact blocks require native complex CUDA tensors')
        if not callable(block_factory):
            raise TypeError('Exact block factory must be callable')
        if not isinstance(layout[0][2], torch.Tensor):
            raise ValueError('Exact coefficient maps must be CUDA tensors')
        self.device, self.dtype = layout[0][2].device, dtype
        if self.device.type != 'cuda' or (device.index is not None and device != self.device):
            raise ValueError('Exact layout must match its CUDA device')
        self.size, self.action_batch = size, action_batch
        pairs = set()
        grouped = defaultdict(list)
        for a, b, rows, cols in layout:
            if type(a) is not int or type(b) is not int or min(a, b) < 0 or (a, b) in pairs:
                raise ValueError('Exact block node pairs must be unique nonnegative integers')
            pairs.add((a, b))
            grouped[len(rows), len(cols)].append((a, b, rows, cols))
        _check_node_maps(((node, indices) for a, b, rows, cols in layout for node, indices in ((a, rows), (b, cols))),
            size, self.device, 'Exact coefficient maps must contain unique in-range CUDA indices',
            'One leaf node cannot have different coefficient maps')
        ordered_nodes = sorted({a for a, _, _, _ in layout})
        locations = {}
        byte_count = 8*(len(ordered_nodes)*2+1+3*len(layout))
        itemsize = 16 if dtype == torch.complex128 else 8
        for gi, (shape, group) in enumerate(grouped.items()):
            for bi, (a, b, rows, cols) in enumerate(group):
                locations[a, b] = (gi, bi)
                byte_count += itemsize*math.prod(self._block_shape(shape))+8*(len(rows)+len(cols))
        if retained_budget_bytes is not None and byte_count > retained_budget_bytes:
            raise MemoryError('Packed exact retained budget exceeded before allocation')
        entries = sorted(pairs)
        self.entries = tuple((a, b, *locations[a, b]) for a, b in entries)
        counts = {node: 0 for node in ordered_nodes}
        for a, _ in entries:
            counts[a] += 1
        rowptr = [0]
        for node in ordered_nodes:
            rowptr.append(rowptr[-1]+counts[node])
        self.row_nodes = torch.tensor(ordered_nodes, device=self.device, dtype=torch.int64)
        self.rowptr = torch.tensor(rowptr, device=self.device, dtype=torch.int64)
        self.column_nodes = torch.tensor([b for _, b in entries], device=self.device, dtype=torch.int64)
        self.locations = torch.tensor([locations[pair] for pair in entries], device=self.device, dtype=torch.int64)
        groups = []
        for shape, group in grouped.items():
            rows = torch.stack([record[2] for record in group])
            cols = torch.stack([record[3] for record in group])
            blocks = torch.empty((len(group), *self._block_shape(shape)), device=self.device, dtype=self.dtype)
            many = getattr(block_factory, 'many', None)
            if many is None:
                for bi, (a, b, rr, cc) in enumerate(group):
                    value = block_factory(a, b, rr, cc)
                    self._store_block(blocks[bi], value, shape)
                    del value
            else:
                # An optional batched factory integrates equal-shape blocks together
                # (sharing triangle pairs); every block passes the same storage checks.
                step = max(1, getattr(block_factory, 'many_batch', 256))
                for first in range(0, len(group), step):
                    chunk = group[first:first+step]
                    self._store_stacked(blocks[first:first+len(chunk)], many(chunk), shape)
            groups.append((rows, cols, blocks))
        self.groups = tuple(groups)
        self._binding = (self.size, self.action_batch, self.device, self.dtype, self.entries)
        self._versions = tuple((id(t), t._version) for t in self.tensors())
        self._retained_bytes = byte_count

    def _block_shape(self, shape):
        return shape

    def _store_stacked(self, destination, values, shape):
        if len(values) != len(destination):
            raise ValueError('Batched exact factory must return one value per block')
        for target, value in zip(destination, values):
            self._store_block(target, value, shape)

    def _store_block(self, destination, value, shape):
        if (not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != self.dtype
                or value.shape != shape or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError('Exact values must be finite fixed matrices matching their maps')
        destination.copy_(value)

    def tensors(self):
        return (self.row_nodes, self.rowptr, self.column_nodes, self.locations,
                *(t for group in self.groups for t in group))

    def assert_current(self):
        if ((self.size, self.action_batch, self.device, self.dtype, self.entries) != self._binding
                or tuple((id(t), t._version) for t in self.tensors()) != self._versions):
            raise ValueError('Packed exact layout or values changed')

    def action(self, value, *, adjoint=False):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.size
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Exact actions require finite matching complex CUDA vectors')
        flat = value.reshape(-1, self.size)
        output = torch.zeros_like(flat)
        for rows, cols, blocks in self.groups:
            target, source = (cols, rows) if adjoint else (rows, cols)
            for start in range(0, len(blocks), self.action_batch):
                stop = start+self.action_batch
                matrix = blocks[start:stop].mH if adjoint else blocks[start:stop]
                source_ids = source[start:stop]
                target_ids = target[start:stop].flatten()
                # A fixed RHS chunk prevents an arbitrary leading batch shape
                # from multiplying the entire near-field workspace at once.
                for rhs_start in range(0, len(flat), self.action_batch):
                    rhs_stop = rhs_start+self.action_batch
                    gathered = flat[rhs_start:rhs_stop, source_ids].permute(1, 2, 0)
                    product = torch.bmm(matrix, gathered).permute(2, 0, 1).flatten(1)
                    output[rhs_start:rhs_stop].index_add_(1, target_ids, product)
        self.assert_current()
        return output.reshape(value.shape)

    def apply(self, value):
        return self.action(value)

    def adjoint(self, value):
        return self.action(value, adjoint=True)

    def receipt(self):
        self.assert_current()
        return {'version': 'grouped_variable_block_csr_v1', 'blocks': len(self.column_nodes),
                'block_rows': len(self.row_nodes), 'shape_groups': len(self.groups),
                'retained_bytes': self._retained_bytes, 'action_block_batch': self.action_batch,
                'action_rhs_batch': self.action_batch, 'padding_bytes': 0,
                'peak_memory_qualified': False}


class PackedScalarNearChannels(PackedNearBlocks):
    """Original single-medium D/K grouped storage in rectangular J/M embedding."""
    def __init__(self, *args, **kwargs):
        raise TypeError("Packed scalar channels require explicit from_layout physical bindings")

    @classmethod
    def from_layout(cls, layout, test_size, source_size, block_factory, *, medium,
                    integration, device, dtype, action_batch=32, retained_budget_bytes=None,
                    _source_node_offset=None):
        from .nearfield import _validate_scalar_channel_medium
        from .operator import IntegrationConfig
        _validate_scalar_channel_medium(medium)
        if not isinstance(integration, IntegrationConfig):
            raise TypeError('Packed scalar channels require IntegrationConfig')
        if (any(type(n) is not int or n < 2 or n % 2 for n in (test_size, source_size))
                or dtype != torch.complex128):
            raise ValueError('Packed scalar channels require native complex128 J/M spaces')
        layout = tuple(layout)
        grouped = defaultdict(list)
        for _, _, rows, cols in layout:
            for side, (indices, offset, width) in enumerate(((rows, 0, test_size//2), (cols, test_size, source_size//2))):
                if (not isinstance(indices, torch.Tensor) or indices.ndim != 1 or len(indices) % 2
                        or not len(indices) or indices.dtype != torch.int64 or indices.device.type != 'cuda'):
                    raise ValueError('Packed scalar maps require complete CUDA J/M ordering')
                grouped[side, offset, width, len(indices), indices.device].append(indices)
        # One reduction per side and map length; identical to the per-map checks.
        for (_, offset, width, length, _), maps in grouped.items():
            stacked, n = torch.stack(maps), length//2
            if (not bool(((stacked[:, :n] >= offset) & (stacked[:, :n] < offset+width)).all())
                    or not torch.equal(stacked[:, n:], stacked[:, :n]+width)):
                raise ValueError('Packed scalar maps require complete J then M ordering')
        instance = cls.__new__(cls)
        if _source_node_offset is not None:
            instance._reciprocal_node_offset = _source_node_offset
        instance.medium, instance.integration = medium, integration
        instance.test_size, instance.source_size = test_size, source_size
        instance.electric, instance.magnetic = 1j*medium.omega*medium.epsilon, -1j*medium.omega*medium.mu
        if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in (complex(instance.electric), complex(instance.magnetic))):
            raise ValueError('Packed scalar material factors exceed native precision')
        instance._initialize(layout, test_size+source_size, block_factory, device=device, dtype=dtype,
            action_batch=action_batch, retained_budget_bytes=retained_budget_bytes)
        instance._channel_binding = instance._channel_state()
        return instance

    def _block_shape(self, shape):
        return (2, shape[0]//2, shape[1]//2)

    def _store_stacked(self, destination, values, shape):
        """Stacked (D [m,r,c], K [m,r,c]) from a batched factory, one finiteness check."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError('Packed scalar factory must supply original D/K kernels')
        for i, field in enumerate(values):
            if (not isinstance(field, torch.Tensor) or field.device != self.device or field.dtype != self.dtype
                    or field.shape != (len(destination), shape[0]//2, shape[1]//2)
                    or field.requires_grad or not bool(torch.isfinite(field).all())):
                raise ValueError('Packed scalar kernels must be finite fixed native matrices')
            destination[:, i].copy_(field)

    def _store_block(self, destination, value, shape):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('Packed scalar factory must supply original D/K kernels')
        for i, field in enumerate(value):
            if (not isinstance(field, torch.Tensor) or field.device != self.device or field.dtype != self.dtype
                    or field.shape != (shape[0]//2, shape[1]//2)
                    or field.requires_grad or not bool(torch.isfinite(field).all())):
                raise ValueError('Packed scalar kernels must be finite fixed native matrices')
            destination[i].copy_(field)

    def _channel_state(self):
        m = self.medium
        return (id(m), m.frequency, m.epsilon_r, m.mu_r, m.k, m.eta,
                id(self.integration), tuple(asdict(self.integration).items()),
                self.test_size, self.source_size, self.electric, self.magnetic, self._retained_bytes)

    def assert_current(self):
        super().assert_current()
        if self._channel_state() != self._channel_binding:
            raise ValueError('Packed scalar channels or integration changed')

    def _action(self, value, *, adjoint):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.size
                or value.device != self.device or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Packed scalar actions require finite matching complex128 currents')
        # Preserve noncontiguous leading batches without flattening the entire input.
        from .fmm_fft import _rhs_chunks
        output = value.new_zeros((math.prod(value.shape[:-1]), self.size))
        for rows, cols, fields, transposed in self._action_groups():
            target, source = (cols, rows) if adjoint else (rows, cols)
            for start in range(0, len(fields), self.action_batch):
                stop = start+self.action_batch
                d, k = fields[start:stop, 0], fields[start:stop, 1]
                if transposed:d, k = d.mT, k.mT
                if adjoint:d, k = d.mH, k.mH
                source_ids = source[start:stop]
                n = source_ids.shape[-1]//2
                target_ids = target[start:stop].flatten()
                for rhs_start, chunk in _rhs_chunks(value, 1, 4):
                    gathered = chunk[:, source_ids].permute(1, 2, 0)
                    j, m = gathered[:, :n], gathered[:, n:]
                    kj, km = torch.bmm(k, j), torch.bmm(k, m)
                    dm, dj = torch.bmm(d, m), torch.bmm(d, j)
                    a = self.magnetic.conjugate() if adjoint else self.electric
                    b = self.electric.conjugate() if adjoint else self.magnetic
                    product = torch.cat((kj+a*dm, b*dj+km), dim=1).permute(2, 0, 1).flatten(1)
                    if not bool(torch.isfinite(product).all()):
                        raise ValueError('Packed scalar action exceeds native precision')
                    output[rhs_start:rhs_start+len(chunk)].index_add_(1, target_ids, product)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Packed scalar accumulated action exceeds native precision')
        self.assert_current()
        return output.reshape(value.shape)

    def _action_groups(self):
        for rows, cols, fields in self.groups:
            yield rows, cols, fields, False

    def apply(self, value):
        # Import lazily because regional direct actions use this shared storage module.
        from .direct_system import DirectSystemActions
        return DirectSystemActions.apply(self, value)

    def adjoint(self, value):
        from .direct_system import DirectSystemActions
        return DirectSystemActions.adjoint(self, value)

    def action(self, value, *, adjoint=False):
        return self.adjoint(value) if adjoint else self.apply(value)

    def receipt(self):
        return {**super().receipt(), 'version': 'grouped_scalar_original_dk_csr_v1',
                'kernel_channels': 'original independently integrated D/K',
                'action_rhs_batch': 4, 'single_medium': True,
                'current_backward': 'fixed_operator_adjoint_no_saved_action_intermediates',
                'physical_qualification': False, 'complexity_qualification': False}


def _reciprocal_scalar_layout(layout, test_size, source_size, node_offset):
    """Validate a complete reciprocal layout and select one ordered direction.

    The caller supplies one common scalar kernel and aligned physical spaces.
    This does not infer reciprocity for arbitrary tensor or periodic kernels.
    """
    if type(node_offset) is not int or node_offset < 1 or test_size != source_size:
        raise ValueError('Reciprocal scalar storage requires aligned square spaces and a node offset')
    pairs = {}
    for a, b, rows, cols in layout:
        if (type(a) is not int or type(b) is not int or not 0 <= a < node_offset
                or not node_offset <= b < 2*node_offset or (a,b-node_offset) in pairs):
            raise ValueError('Reciprocal scalar node pairs must be unique aligned physical IDs')
        pairs[a,b-node_offset] = (a,b,rows,cols)
    message = 'Reciprocal scalar physical node maps must remain identical across pairs'
    device = layout[0][2].device if layout else None
    if any(not isinstance(t, torch.Tensor) or t.device != device for _, _, rows, cols in layout for t in (rows, cols)):
        raise ValueError(message)
    _check_node_maps(((node, indices) for a, b, rows, cols in layout
                      for node, indices in ((a, rows), (b-node_offset, cols-test_size))),
                     None, device, message, message)
    for (a,b) in pairs:
        if (b,a) not in pairs:
            raise ValueError('Reciprocal scalar layout requires every reverse pair')
    # With one map per physical node, rows(a,b) = map(a) = cols(b,a)-test_size and
    # cols(a,b)-test_size = map(b) = rows(b,a): the reverse J/M maps agree exactly.
    # Off-diagonal blocks precede diagonal blocks within every shape group.
    return tuple(pairs[pair] for pair in sorted(pairs,key=lambda pair:(pair[0]==pair[1],*pair))
        if pair[0]<=pair[1])


class PackedReciprocalScalarNearChannels(PackedScalarNearChannels):
    """One native D/K allocation for each reciprocal off-diagonal pair.

    Only fixed free-space scalar kernels on identical physical spaces qualify
    for this explicit factory contract. Reverse values are transposes, not
    conjugates. Diagonal blocks remain independently integrated in full.
    """
    @classmethod
    def from_layout(cls, layout, test_size, source_size, block_factory, *, source_node_offset,
                    medium, integration, device, dtype, action_batch=32, retained_budget_bytes=None):
        return super().from_layout(layout,test_size,source_size,block_factory,medium=medium,
            integration=integration,device=device,dtype=dtype,action_batch=action_batch,
            retained_budget_bytes=retained_budget_bytes,_source_node_offset=source_node_offset)

    def _initialize(self, layout, size, block_factory, **kwargs):
        canonical = _reciprocal_scalar_layout(layout,self.test_size,self.source_size,
            self._reciprocal_node_offset)
        counts = {}
        for a,b,rows,cols in canonical:
            shape = len(rows),len(cols)
            counts.setdefault(shape,0)
            counts[shape] += int(a!=b-self._reciprocal_node_offset)
        super()._initialize(canonical,size,block_factory,**kwargs)
        self._reverse_group_counts = tuple(counts.values())
        self._logical_blocks = len(layout)
        self._reciprocal_binding = self._reciprocal_state()

    def _reciprocal_state(self):
        return self._reciprocal_node_offset,self._reverse_group_counts,self._logical_blocks

    def assert_current(self):
        super().assert_current()
        if self._reciprocal_state()!=self._reciprocal_binding or any(t.requires_grad for t in self.tensors()):
            raise ValueError('Packed reciprocal scalar layout, modes or policy changed')

    def _action_groups(self):
        for (rows,cols,fields),count in zip(self.groups,self._reverse_group_counts,strict=True):
            yield rows,cols,fields,False
            for start in range(0,count,self.action_batch):
                stop = min(start+self.action_batch,count)
                yield (cols[start:stop]-self.test_size,rows[start:stop]+self.test_size,
                    fields[start:stop],True)

    def receipt(self):
        return {**super().receipt(),'version':'grouped_reciprocal_scalar_original_dk_csr_v1',
            'logical_blocks':self._logical_blocks,'stored_blocks':len(self.column_nodes),
            'reverse_kernel_rule':'D_transpose_K_transpose_without_conjugation',
            'diagonal_storage':'full_independent_original_integrals',
            'reverse_index_block_batch':self.action_batch,'production_qualification':False}


class StreamedReciprocalScalarNearChannels(PackedReciprocalScalarNearChannels):
    """Recompute original D/K blocks on bounded canonical map chunks.

    The caller supplies an original integration factory and a physical owner
    guard. Complete reciprocal J/M maps, scalar material validation, transpose
    reuse and fixed-current autograd use the existing packed implementation.
    No numerical kernel is retained between actions or substituted by a fit.
    """

    @classmethod
    def from_layout(cls, layout, test_size, source_size, block_factory, *,
                    source_node_offset, medium, integration, device, dtype,
                    physical_guard, physical_inputs, action_batch=32,
                    coefficient_workspace_budget_bytes=128*1024**2,
                    retained_budget_bytes=None):
        if not callable(block_factory) or not callable(physical_guard):
            raise TypeError('Streamed original channels require a factory and physical guard')
        if (type(coefficient_workspace_budget_bytes) is not int
                or coefficient_workspace_budget_bytes < 1):
            raise ValueError('Streamed coefficient workspace budget must be a positive integer')
        inputs = tuple(physical_inputs)
        if (not inputs or any(not isinstance(t, torch.Tensor) or t.device.type != 'cuda'
                              or t.requires_grad for t in inputs)):
            raise ValueError('Streamed physical inputs must be explicit fixed CUDA tensors')
        physical_guard()
        instance = super().from_layout(layout, test_size, source_size,
            lambda a, b, rows, cols: None, source_node_offset=source_node_offset,
            medium=medium, integration=integration, device=device, dtype=dtype,
            action_batch=action_batch, retained_budget_bytes=retained_budget_bytes)
        if any(t.device != instance.device for t in inputs):
            raise ValueError('Streamed physical inputs must match the mapped CUDA device')
        instance._channel_factory = block_factory
        instance._physical_guard = physical_guard
        instance._physical_inputs = inputs
        instance.coefficient_workspace_budget_bytes = coefficient_workspace_budget_bytes
        pairs = [[] for _ in instance.groups]
        for a, b, gi, bi in instance.entries:
            pairs[gi].append((bi, a, b))
        instance._group_pairs = tuple(tuple((a,b) for _,a,b in sorted(group)) for group in pairs)
        # Three channel arrays cover the final stacked chunk, factory results,
        # and the transient list while stacking. Original quadrature workspaces
        # and electromagnetic gather/product workspaces have separate scopes.
        for rows, cols, _ in instance.groups:
            single = 3*2*(rows.shape[-1]//2)*(cols.shape[-1]//2)*16
            if single > coefficient_workspace_budget_bytes:
                raise MemoryError('One original near block exceeds coefficient workspace before integration')
        instance._stream_binding = instance._stream_state()
        instance.assert_current()
        return instance

    def _block_shape(self, shape):
        # Keep the existing packed map/CSR owner with zero coefficient storage.
        return (0, shape[0]//2, shape[1]//2)

    def _store_block(self, destination, value, shape):
        if value is not None or destination.numel():
            raise ValueError('Streamed setup must not construct or retain near coefficients')

    def _stream_state(self):
        return (id(self._channel_factory), id(self._physical_guard), self._group_pairs,
                self.coefficient_workspace_budget_bytes,
                tuple((id(t), t._version, t.requires_grad, t.device, t.dtype, tuple(t.shape))
                      for t in self._physical_inputs))

    def assert_current(self):
        super().assert_current()
        if self._stream_state() != self._stream_binding:
            raise ValueError('Streamed factory, physical inputs or workspace policy changed')
        self._physical_guard()

    def _action_groups(self):
        for gi, ((rows, cols, _), reverse_count) in enumerate(
                zip(self.groups, self._reverse_group_counts, strict=True)):
            nr, nc = rows.shape[-1]//2, cols.shape[-1]//2
            step = min(self.action_batch, self.coefficient_workspace_budget_bytes//(3*2*nr*nc*16))
            for start in range(0, len(rows), step):
                stop = min(start+step, len(rows))
                fields = []
                for bi in range(start, stop):
                    self.assert_current()
                    a, b = self._group_pairs[gi][bi]
                    values = self._channel_factory(a, b, rows[bi], cols[bi])
                    if not isinstance(values, (tuple, list)) or len(values) != 2:
                        raise ValueError('Streamed factory must supply original D/K kernels')
                    for t in values:
                        if (not isinstance(t, torch.Tensor) or t.shape != (nr,nc)
                                or t.device != self.device or t.dtype != self.dtype
                                or t.requires_grad or not bool(torch.isfinite(t).all())):
                            raise ValueError('Streamed kernels must be finite fixed native matrices')
                    backing = {t.untyped_storage().data_ptr():t.untyped_storage().nbytes()
                               for t in values}
                    if sum(backing.values()) > 2*nr*nc*16:
                        raise ValueError('Streamed kernels exceed their native channel backing storage')
                    fields.append(torch.stack(values))
                    del t, values
                chunk = torch.stack(fields)
                del fields
                yield rows[start:stop], cols[start:stop], chunk, False
                reverse_stop = min(stop, reverse_count)
                if start < reverse_stop:
                    count = reverse_stop-start
                    yield (cols[start:reverse_stop]-self.test_size,
                           rows[start:reverse_stop]+self.test_size, chunk[:count], True)
                del chunk

    def receipt(self):
        return {**super().receipt(), 'version':'streamed_reciprocal_scalar_original_dk_csr_v1',
                'retained_kernel_bytes':0, 'coefficient_storage':'recompute_original_per_action',
                'diagonal_storage':'recompute_full_independent_original_integrals',
                'coefficient_workspace_budget_bytes':self.coefficient_workspace_budget_bytes,
                'coefficient_workspace_scope':'three bounded D/K channel arrays only',
                'physical_inputs_borrowed_not_copied':True,
                'near_quadrature_workspace_qualified':False,
                'public_integration_qualified':False, 'production_qualification':False}


from threading import RLock
from types import MappingProxyType


def bounded_original_near_cache_plan(group_shapes, reverse_counts, budget_bytes):
    """Select native diagonal prefixes, then canonical off-diagonal prefixes.

    This host topology plan contains no numerical approximation or coefficient
    evaluation. It is shared with future structural admission callers.
    Each returned segment is (group, first block, exclusive end, native bytes).
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError('Original coefficient cache budget must be a nonnegative integer')
    shapes, reverse = tuple(group_shapes), tuple(reverse_counts)
    if not shapes or len(shapes) != len(reverse):
        raise ValueError('Cache planning requires matching original grouped topology')
    for shape, count in zip(shapes, reverse, strict=True):
        if (not isinstance(shape, tuple) or len(shape) != 3
                or any(type(v) is not int or v < 1 for v in shape)
                or type(count) is not int or not 0 <= count <= shape[0]):
            raise ValueError('Cache groups require positive counts and original channel widths')
    remaining, segments = budget_bytes, []
    for diagonal in (True, False):
        for gi, ((count, nr, nc), mirrored) in enumerate(zip(shapes, reverse, strict=True)):
            first, available = (mirrored, count-mirrored) if diagonal else (0, mirrored)
            native_bytes = 2*nr*nc*16
            selected = min(available, remaining//native_bytes)
            if selected:
                segments.append((gi, first, first+selected, selected*native_bytes))
                remaining -= selected*native_bytes
    return tuple(segments), budget_bytes-remaining


class BoundedCachedReciprocalScalarNearChannels:
    """Bounded first-use cache around the original streamed scalar primitive.

    All coefficient construction, fixed-input validation, reciprocal transpose,
    CSR actions and current derivatives use the original implementation. Cache
    entries copy only whole validated native D/K blocks. Uncached blocks retain
    the original per-action integration. Nothing is fitted or substituted.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError('Cached original channels require explicit from_layout physical bindings')

    @classmethod
    def from_layout(cls, layout, test_size, source_size, block_factory, *,
                    source_node_offset, medium, integration, device, dtype,
                    physical_guard, physical_inputs, action_batch=32,
                    coefficient_workspace_budget_bytes=128*1024**2,
                    coefficient_cache_budget_bytes=128*1024**2,
                    retained_budget_bytes=None):
        if not callable(block_factory):
            raise TypeError('Original coefficient cache requires the original integration factory')
        if type(coefficient_cache_budget_bytes) is not int or coefficient_cache_budget_bytes < 0:
            raise ValueError('Original coefficient cache budget must be a nonnegative integer')
        holder = {}
        def original_or_cached(a, b, rows, cols):
            owner = holder['owner']
            owner.assert_current()
            if not owner._in_action:
                raise RuntimeError('Cached factory requires its guarded original action')
            location = owner._cache_lookup.get((a,b))
            if location is not None:
                si, bi = location
                if owner._ready[si]:
                    if si not in owner._waited:
                        stream = torch.cuda.current_stream(owner.device)
                        stream.wait_event(owner._events[si])
                        owner._waited.add(si)
                    native = owner._cache_arrays[si]
                    native.record_stream(torch.cuda.current_stream(owner.device))
                    # The original streamed contract bounds factory backing
                    # storage per block. Native clones preserve that contract
                    # when the retained cache uses a larger grouped allocation.
                    return native[bi,0].clone(), native[bi,1].clone()
            return owner._original_factory(a,b,rows,cols)
        streamed = StreamedReciprocalScalarNearChannels.from_layout(layout,test_size,source_size,
            original_or_cached,source_node_offset=source_node_offset,medium=medium,
            integration=integration,device=device,dtype=dtype,physical_guard=physical_guard,
            physical_inputs=physical_inputs,action_batch=action_batch,
            coefficient_workspace_budget_bytes=coefficient_workspace_budget_bytes,
            retained_budget_bytes=retained_budget_bytes)
        shapes = tuple((len(rows),rows.shape[-1]//2,cols.shape[-1]//2)
                       for rows,cols,_ in streamed.groups)
        plan, native_bytes = bounded_original_near_cache_plan(shapes,
            streamed._reverse_group_counts,coefficient_cache_budget_bytes)
        total = streamed.receipt()['retained_bytes']+native_bytes
        if retained_budget_bytes is not None and total > retained_budget_bytes:
            raise MemoryError('Cached original retained budget exceeded before coefficient cache allocation')
        instance = cls.__new__(cls)
        instance._streamed = streamed
        instance._original_factory = block_factory
        instance.coefficient_cache_budget_bytes = coefficient_cache_budget_bytes
        instance._cache_plan = plan
        instance._cache_bytes = native_bytes
        # Pending entries are never read. Avoid an asynchronous initialization
        # write that could race a first fill on another CUDA stream.
        instance._cache_arrays = tuple(torch.empty((stop-first,2,*shapes[gi][1:]),
            device=streamed.device,dtype=streamed.dtype) for gi,first,stop,_ in plan)
        lookup = {}
        for si,(gi,first,stop,_) in enumerate(plan):
            for bi in range(first,stop):
                key = streamed._group_pairs[gi][bi]
                lookup[key] = (si,bi-first)
        instance._cache_lookup = MappingProxyType(lookup)
        instance._ready = (False,)*len(plan)
        instance._events = (None,)*len(plan)
        instance._action_lock = RLock()
        instance._in_action = False
        instance._waited = set()
        instance._cache_binding = instance._cache_state()
        holder['owner'] = instance
        instance.assert_current()
        return instance

    @property
    def size(self): return self._streamed.size

    @property
    def test_size(self): return self._streamed.test_size

    @property
    def source_size(self): return self._streamed.source_size

    @property
    def device(self): return self._streamed.device

    @property
    def dtype(self): return self._streamed.dtype

    @property
    def action_batch(self): return self._streamed.action_batch

    @property
    def electric(self): return self._streamed.electric

    @property
    def magnetic(self): return self._streamed.magnetic

    def tensors(self):
        return (*self._streamed.tensors(),*self._cache_arrays)

    def _cache_state(self):
        return (id(self._streamed),id(self._original_factory),id(self._cache_lookup),
                id(self._action_lock),self.coefficient_cache_budget_bytes,
                self._cache_plan,self._cache_bytes,self._ready,
                tuple(id(event) if event is not None else None for event in self._events),
                tuple((id(t),t._version,t.requires_grad,t.device,t.dtype,tuple(t.shape))
                      for t in self._cache_arrays))

    def assert_current(self):
        with self._action_lock:
            self._streamed.assert_current()
            if self._cache_state() != self._cache_binding:
                raise ValueError('Cached original owner, controls or native coefficients changed')

    def _action_groups(self):
        gi, start = 0, 0
        for rows,cols,fields,transposed in self._streamed._action_groups():
            self.assert_current()
            if not transposed:
                stop = start+len(fields)
                for si,(group,first,end,_) in enumerate(self._cache_plan):
                    if group != gi or self._ready[si]:
                        continue
                    lo,hi = max(first,start),min(end,stop)
                    if lo < hi:
                        native = self._cache_arrays[si]
                        stream = torch.cuda.current_stream(self.device)
                        # A failed action may have copied an incomplete prefix
                        # on another stream. Order every retry after that copy,
                        # even though the pending prefix cannot serve reads.
                        if self._events[si] is not None:
                            stream.wait_event(self._events[si])
                        native.record_stream(stream)
                        native[lo-first:hi-first].copy_(fields[lo-start:hi-start])
                        event = torch.cuda.Event()
                        event.record(stream)
                        self._events = self._events[:si]+(event,)+self._events[si+1:]
                        if hi == end:
                            self._ready = self._ready[:si]+(True,)+self._ready[si+1:]
                        self._cache_binding = self._cache_state()
                start = stop
                if start == len(self._streamed.groups[gi][0]):
                    gi,start = gi+1,0
            yield rows,cols,fields,transposed
        self.assert_current()

    def _action(self, value, *, adjoint):
        with self._action_lock:
            if self._in_action:
                raise RuntimeError('Cached original near actions cannot reenter their factory')
            self._waited = set()
            self._in_action = True
            try:
                return PackedScalarNearChannels._action(self,value,adjoint=adjoint)
            finally:
                self._in_action = False
                self._waited = set()

    apply = PackedScalarNearChannels.apply
    adjoint = PackedScalarNearChannels.adjoint
    action = PackedScalarNearChannels.action

    def receipt(self):
        with self._action_lock:
            self.assert_current()
            base = self._streamed.receipt()
            return {**base,'version':'bounded_cached_original_reciprocal_scalar_near_v1',
                'retained_bytes':base['retained_bytes']+self._cache_bytes,
                'retained_kernel_bytes':self._cache_bytes,
                'coefficient_cache_budget_bytes':self.coefficient_cache_budget_bytes,
                'coefficient_cache_scope':'one original regional owner, native payload only',
                'cache_priority':'independent diagonal prefixes then canonical off_diagonal prefixes',
                'cache_population':'copy complete validated original first_use chunks',
                'cache_planned_blocks':sum(stop-first for _,first,stop,_ in self._cache_plan),
                'cache_ready_blocks':sum(stop-first for ready,(_,first,stop,_) in
                                        zip(self._ready,self._cache_plan,strict=True) if ready),
                'cache_segments':len(self._cache_plan),
                'coefficient_storage':'bounded original cache plus uncached original per_action integration',
                'cross_stream_population':'native CUDA event wait for each completed segment',
                'diagonal_storage':'full independent original integrals',
                'cache_host_metadata_and_events_peak_qualified':False,
                'public_integration_qualified':False,'production_qualification':False}

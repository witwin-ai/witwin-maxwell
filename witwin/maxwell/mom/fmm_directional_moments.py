"""All-node fixed-carrier RWG source/receiver maps on complete support trees."""
import math
import torch
from .cluster import SupportOctree
from .fmm_expansion import _points, multi_indices
from .fmm_shift import SeparableMultipoleShift
from .fmm_directional_shift import DirectionalFixedCarrierShift, DirectionalChangingCarrierShift, ChangingCarrierPhaseLayout
from .fmm_directional_rwg import DirectionalRWGProjection
from .fmm_fft import _rhs_chunks
from .nearfield import ScalarSystemBlocks
from .operator import PreparedRWGGeometry


class _UpwardCurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents, owner, kind):
        ctx.owner, ctx.kind, ctx.leading = owner, kind, currents.shape[:-1]
        ctx.set_materialize_grads(False)
        return owner._upward(currents, kind)

    @staticmethod
    def backward(ctx, *coefficients):
        ctx.owner.assert_current()
        result = ctx.owner._downward_dispatch(coefficients, ctx.kind, ctx.leading, allow_missing=True)
        return result, None, None


class _DownwardCurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, kind, leading, *coefficients):
        ctx.owner, ctx.kind = owner, kind
        ctx.present = tuple(v is not None for v in coefficients)
        ctx.set_materialize_grads(False)
        return owner._downward(coefficients, kind, leading)

    @staticmethod
    def backward(ctx, gradient):
        ctx.owner.assert_current()
        if gradient is None:
            return (None, None, None, *(None for _ in ctx.present))
        result = ctx.owner._upward_dispatch(gradient, ctx.kind)
        return (None, None, None, *(v if present else None for present, v in zip(ctx.present, result)))


class NestedDirectionalRWGProjection:
    """Leaf integration plus four distinct all-node maps for one fixed carrier.

    source_upward returns U_s currents; source_downward is U_s^H.
    receiver_downward is physical R_r locals; receiver_upward is R_r^H.
    Every node participates, including arbitrary inputs at internal nodes.
    """
    def __init__(self, basis, tree, direction, wave_number, order, *, quadrature_order,
                 geometry=None, layout=None, face_batch=32, rhs_batch=4,
                 retained_budget_bytes=1024**3, tensor_workspace_budget_bytes=256*1024**2,
                 node_directions=None, node_orders=None, phase_orders=None, phase_relative_budget=1e-12,
                 phase_storage='dense_v1'):
        _points(direction)
        if not isinstance(tree, SupportOctree):
            raise TypeError('Directional nesting requires a support octree')
        tree.assert_current()
        if tree.expansion_centers is not None:
            raise NotImplementedError('Directional projections require original support-box centers')
        if type(order) is not int or not 0 <= order <= 12:
            raise ValueError('Directional nested order must lie between zero and twelve')
        if any(type(v) is not int or v < 1 for v in
               (quadrature_order, face_batch, rhs_batch, retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Directional nested quadrature, batches and budgets must be positive integers')
        if quadrature_order < (order+4)//2:
            raise ValueError('Directional nested quadrature cannot underresolve its polynomial factor')
        if type(wave_number) not in (int, float, complex):
            raise ValueError('Directional nested wave number must be a fixed passive scalar')
        wave = complex(wave_number)
        if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
            raise ValueError('Directional nested wave number must be finite and passive')
        global_space = isinstance(basis, ScalarSystemBlocks)
        if global_space:
            basis.assert_current()
            if geometry is not None:
                raise ValueError('Global directional nesting shares provider geometry')
            mesh, size = basis.system, basis.system.total_edges
        else:
            mesh, size = basis.mesh, basis.n_edges
            if basis._mesh_version != (id(mesh.vertices), mesh.vertices._version, id(mesh.faces), mesh.faces._version):
                raise ValueError('RWG geometry changed before directional nesting')
            if mesh.vertices.requires_grad:
                raise ValueError('Directional nesting requires fixed RWG geometry')
        if (tree.size != size or tree.device != mesh.device or tree.lower.dtype != torch.float64
                or mesh.dtype != torch.float64 or direction.shape != (3,)
                or direction.device != tree.device or direction.dtype != torch.float64
                or not bool((direction.norm()-1).abs() <= 512*torch.finfo(direction.dtype).eps)
                or not torch.equal(tree.permutation.sort().values, torch.arange(size, device=tree.device))):
            raise ValueError('Directional nesting requires complete matching fixed CUDA float64 spaces and unit carrier')
        _points(tree.lower); _points(tree.upper)
        if (tree.lower.shape != (tree.node_count, 3) or tree.upper.shape != tree.lower.shape
                or tree.upper.device != tree.device or tree.upper.dtype != torch.float64
                or not bool((tree.lower <= tree.upper).all())
                or tree.children.shape != (tree.node_count, 8)
                or any(t.dtype != torch.int64 or t.device != tree.device for t in
                       (tree.permutation, tree.starts, tree.stops, tree.children, tree.depth))
                or any(t.shape != (tree.node_count,) for t in (tree.starts, tree.stops, tree.depth))):
            raise ValueError('Directional tree topology and support tensors must match their native node layout')
        children = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
        starts, stops = tuple(tree.starts.cpu().tolist()), tuple(tree.stops.cpu().tolist())
        parents = {}
        for node, child_ids in enumerate(children):
            if not 0 <= starts[node] < stops[node] <= size:
                raise ValueError('Directional tree ranges must be nonempty and complete')
            next_start = starts[node]
            for child in child_ids:
                if (child <= node or child >= tree.node_count or child in parents
                        or starts[child] != next_start or stops[child] > stops[node]):
                    raise ValueError('Directional tree requires disjoint complete topological child ranges')
                parents[child] = node; next_start = stops[child]
            if child_ids and next_start != stops[node]:
                raise ValueError('Directional tree children must cover their parent range')
        if starts[0] != 0 or stops[0] != size or set(parents) != set(range(1, tree.node_count)):
            raise ValueError('Directional tree requires one complete connected root')
        changing = any(v is not None for v in (node_directions,node_orders,phase_orders))
        if phase_storage not in ('dense_v1','compact_phase_v1'):
            raise ValueError('Unsupported directional tree phase storage')
        compact = phase_storage == 'compact_phase_v1'
        if compact and not changing:
            raise ValueError('Compact phase trees require explicit changing carrier node configuration')
        if changing:
            if any(v is None for v in (node_directions,node_orders,phase_orders)):
                raise ValueError('Changing directional trees require all node carriers, orders and phase orders')
            _points(node_directions)
            if (node_directions.shape != (tree.node_count,3) or node_directions.dtype != torch.float64
                    or node_directions.device != tree.device
                    or not bool(((node_directions.norm(dim=-1)-1).abs() <= 512*torch.finfo(torch.float64).eps).all())
                    or not torch.equal(node_directions[0],direction)):
                raise ValueError('Changing directional tree carriers must match complete fixed native unit node layout and root')
            if (not isinstance(node_orders,(tuple,list)) or not isinstance(phase_orders,(tuple,list))
                    or len(node_orders) != tree.node_count or len(phase_orders) != tree.node_count
                    or any(type(p) is not int or not 0 <= p <= 12 for p in (*node_orders,*phase_orders))
                    or node_orders[0] != order or phase_orders[0] != 0
                    or any(node_orders[c] < node_orders[p]+phase_orders[c] for c,p in parents.items())):
                raise ValueError('Changing directional tree orders must cover every parent plus phase order within zero to twelve')
            if (type(phase_relative_budget) not in (int,float) or not math.isfinite(phase_relative_budget)
                    or not 0 < phase_relative_budget < 1):
                raise ValueError('Changing directional tree phase budget must be finite and between zero and one')
            if layout is not None:
                raise ValueError('Changing directional trees use owned phase tables without a shared fixed layout')
            node_orders, phase_orders = tuple(node_orders),tuple(phase_orders)
        else:
            node_orders, phase_orders = (order,)*tree.node_count,(0,)*tree.node_count
        node_counts = tuple(len(multi_indices(p)) for p in node_orders)
        phase_keys = tuple(sorted({(node_orders[c],node_orders[p],phase_orders[c])
                                  for c,p in parents.items()})) if compact else ()
        center_orders = tuple(sorted({key[1] for key in phase_keys}))
        phase_layout_bytes = sum(8*len(multi_indices(pp))*len(multi_indices(L))+32*len(multi_indices(L))
                                 for ps,pp,L in phase_keys)
        center_layout_bytes = sum(144*len(multi_indices(pp))*(pp+1) for pp in center_orders)
        if quadrature_order < (max(node_orders)+4)//2:
            raise ValueError('Directional nested quadrature cannot underresolve any actual leaf polynomial factor')
        if layout is not None:
            if (tree.node_count == 1 or not isinstance(layout, SeparableMultipoleShift)
                    or layout.order != order or layout.device != tree.device or layout.real_dtype != torch.float64):
                raise ValueError('Shared directional tree layout must match a nontrivial native tree')
            layout.assert_current()
        if geometry is not None:
            if not isinstance(geometry, PreparedRWGGeometry):
                raise TypeError('Shared directional tree geometry must be prepared RWG geometry')
            geometry.assert_current(basis)
        count = len(multi_indices(order))
        routes = {}
        for node, child_ids in enumerate(children):
            if not child_ids:
                edges = tree.permutation[starts[node]:stops[node]]
                routes[node] = basis._body_selection(edges) if global_space else ((None, None),)
        leaf_groups = sum(len(groups) for groups in routes.values())
        owns_geometry = not global_space and geometry is None
        geometry_bytes = len(mesh.faces)*16*8 if owns_geometry else 0
        layout_bytes = 6*count*(order+1)*24 if tree.node_count > 1 and not changing else 0
        owns_layout = tree.node_count > 1 and layout is None and not changing
        leaf_bytes = 128*sum((stops[n]-starts[n])*node_counts[n] for n in routes)
        transfer_bytes = (sum(128+32*len(multi_indices(phase_orders[c])) if compact
                             else 128+32*node_counts[c]*node_counts[p] for c,p in parents.items())
                          if changing else 104*(tree.node_count-1))
        expected = (32*tree.node_count+24+8*size+leaf_bytes+48*leaf_groups+transfer_bytes
                    +24*tree.node_count*changing+geometry_bytes+(layout_bytes if owns_layout else 0)
                    +phase_layout_bytes+center_layout_bytes)
        maximum_faces = max(len(body.mesh.faces) for body in basis.system.bodies) if global_space else len(mesh.faces)
        setup_workspace = (2*expected+face_batch*quadrature_order**2*(128*max(node_counts)+1024)
                           +size*64+maximum_faces*16*8*3)
        shift_workspace = (2*layout_bytes+rhs_batch*count*(order+1)*4*16*8
                           +count*(order+1)*8*16+512) if tree.node_count > 1 else 0
        if changing:
            if compact:
                shift_workspace = max((3*(8*node_counts[p]*len(multi_indices(phase_orders[c]))
                    +32*len(multi_indices(phase_orders[c]))+144*node_counts[p]*(node_orders[p]+1)
                    +128+32*len(multi_indices(phase_orders[c])))
                    +rhs_batch*node_counts[p]*len(multi_indices(phase_orders[c]))*4*16*8
                    +rhs_batch*node_counts[p]*(node_orders[p]+1)*4*16*8
                    +node_counts[p]*len(multi_indices(phase_orders[c]))*128+4096
                    for c,p in parents.items()),default=0)
            else:
                shift_workspace = max((3*(128+32*node_counts[c]*node_counts[p])+4096*1024
                    +rhs_batch*(node_counts[c]+node_counts[p])*4*16*8 for c,p in parents.items()),default=0)
        action_workspace = 2*sum(node_counts)*rhs_batch*4*16+4*rhs_batch*size*16+shift_workspace
        workspace = max(setup_workspace, action_workspace, shift_workspace)
        if expected > retained_budget_bytes:
            raise MemoryError('Directional nested retained budget exceeded before geometry, moments and shifts')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Directional nested tensor workspace exceeded before geometry, moments and shifts')
        self.basis, self.tree = basis, tree
        self.geometry = basis if global_space else (PreparedRWGGeometry(basis) if owns_geometry else geometry)
        self.owns_geometry, self.owned_geometry_bytes = owns_geometry, geometry_bytes
        self.owns_layout, self.owned_layout_bytes = owns_layout, layout_bytes if owns_layout else 0
        self.direction = direction.clone()
        self.node_directions = node_directions.clone() if changing else None
        self.changing_carriers, self.node_orders, self.phase_orders = changing,node_orders,phase_orders
        self.phase_storage = phase_storage
        self.owned_phase_layout_bytes, self.owned_center_layout_bytes = phase_layout_bytes,center_layout_bytes
        self.node_counts, self.phase_relative_budget = node_counts,phase_relative_budget
        self.wave_number, self.order, self.count = wave, order, count
        self.quadrature_order, self.face_batch, self.rhs_batch = quadrature_order, face_batch, rhs_batch
        self.device, self.dtype = tree.device, torch.complex128
        self.children, self.starts, self.stops = children, starts, stops
        self.retained_budget_bytes, self.tensor_workspace_budget_bytes = retained_budget_bytes, tensor_workspace_budget_bytes
        self.planned_retained_bytes, self.planned_tensor_workspace_bytes = expected, workspace
        if global_space:
            lower, upper = basis.support_boxes()
        else:
            supports = torch.cat((self.geometry.triangles[basis.plus_faces], self.geometry.triangles[basis.minus_faces]), 1)
            lower, upper = supports.amin(1), supports.amax(1)
        for node in range(tree.node_count):
            edges = tree.permutation[starts[node]:stops[node]]
            if not bool((lower[edges] >= tree.lower[node]).all() & (upper[edges] <= tree.upper[node]).all()):
                raise ValueError('Directional tree must enclose complete RWG supports at every node')
        self.centers = tree.lower*.5+tree.upper*.5
        self.radii = torch.linalg.vector_norm((tree.upper-tree.lower)*.5, dim=-1)
        if changing:
            # Bound the actual full-support axis corners about the computed
            # centers, including center rounding, before any leaf quadrature.
            radii=[]
            for node in range(tree.node_count):
                edges=tree.permutation[starts[node]:stops[node]]
                corner=torch.maximum((lower[edges]-self.centers[node]).abs(),(upper[edges]-self.centers[node]).abs())
                bound=corner.norm(dim=-1).amax()
                radius=bound*(1+256*torch.finfo(torch.float64).eps)
                if not bool(torch.isfinite(radius) & (radius > 0) & (corner.norm(dim=-1) <= radius).all()):
                    raise ValueError('Changing directional tree radius must enclose all complete RWG support corners')
                radii.append(radius)
            self.radii=torch.stack(radii)
        if not bool(torch.isfinite(self.centers).all() & torch.isfinite(self.radii).all()):
            raise ValueError('Directional tree centers and radii exceed native precision')
        self.layout = SeparableMultipoleShift(order, device=self.device, dtype=torch.float64) if owns_layout else layout
        self.phase_layouts = {key: ChangingCarrierPhaseLayout(*key,device=self.device) for key in phase_keys}
        self.center_layouts = {pp: SeparableMultipoleShift(pp,device=self.device,dtype=torch.float64) for pp in center_orders}
        if changing:
            self.transfers = {child: DirectionalChangingCarrierShift(self.centers[child]-self.centers[parent],
                self.node_directions[child],self.node_directions[parent],wave,node_orders[child],node_orders[parent],
                phase_orders[child],child_radius=self.radii[child],phase_relative_budget=phase_relative_budget,
                rhs_batch=rhs_batch,retained_budget_bytes=(128+32*len(multi_indices(phase_orders[child]))
                    if compact else 128+32*node_counts[child]*node_counts[parent]),
                tensor_workspace_budget_bytes=tensor_workspace_budget_bytes,phase_storage=phase_storage,
                phase_layout=self.phase_layouts.get((node_orders[child],node_orders[parent],phase_orders[child])),
                center_layout=self.center_layouts.get(node_orders[parent])) for child,parent in parents.items()}
        else:
            self.transfers = {child: DirectionalFixedCarrierShift(self.centers[child]-self.centers[parent],
                self.direction, wave, order, layout=self.layout, rhs_batch=rhs_batch,
                retained_budget_bytes=104, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                for child, parent in parents.items()}
        leaves = {}
        for node, groups in routes.items():
            edges = tree.permutation[starts[node]:stops[node]]
            records = []
            for bi, selected in groups:
                if global_space:
                    body = basis.system.bodies[bi]
                    local_basis, offset, geom = body.basis, body.edge_offset, basis.geometry[bi]
                    local_edges = edges[selected]-offset
                else:
                    local_basis, offset, geom, local_edges = basis, 0, self.geometry, edges
                projection = DirectionalRWGProjection(local_basis, local_edges, self.centers[node],
                    self.direction if not changing else self.node_directions[node],wave,node_orders[node],
                    quadrature_order=quadrature_order, face_batch=face_batch,
                    geometry=geom, retained_budget_bytes=expected,
                    tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                records.append((offset, projection))
            leaves[node] = tuple(records)
        self.leaves = leaves
        self.retained_bytes = sum(t.numel()*t.element_size() for t in self.tensors())
        if self.retained_bytes != expected:
            raise RuntimeError('Directional nested owned storage differs from preallocation plan')
        self._binding = self._state()

    def tensors(self):
        return (self.centers, self.radii, self.direction,
                *((self.node_directions,) if self.changing_carriers else ()),
                *(self.geometry.tensors() if self.owns_geometry else ()),
                *(self.layout.tensors() if self.owns_layout else ()),
                *(t for shared in (*self.phase_layouts.values(),*self.center_layouts.values()) for t in shared.tensors()),
                *(t for transfer in self.transfers.values() for t in transfer.tensors()),
                *(t for records in self.leaves.values() for _, projection in records for t in projection.tensors()))

    def _state(self):
        return (id(self.basis), id(self.tree), id(self.geometry), id(self.layout),
                self.tree.version, self.tree.leaf_size, self.tree.morton_bits,
                self.owns_geometry, self.owned_geometry_bytes, self.owns_layout, self.owned_layout_bytes,
                self.wave_number, self.order, self.count, self.quadrature_order, self.face_batch, self.rhs_batch,
                self.changing_carriers,self.node_orders,self.phase_orders,self.node_counts,self.phase_relative_budget,
                self.phase_storage,self.owned_phase_layout_bytes,self.owned_center_layout_bytes,
                tuple((key,id(shared)) for key,shared in self.phase_layouts.items()),
                tuple((key,id(shared)) for key,shared in self.center_layouts.items()),
                self.device, self.dtype, self.children, self.starts, self.stops,
                self.retained_budget_bytes, self.tensor_workspace_budget_bytes,
                self.planned_retained_bytes, self.planned_tensor_workspace_bytes, self.retained_bytes,
                tuple((node, tuple((offset, id(projection)) for offset, projection in records))
                      for node, records in self.leaves.items()),
                tuple((node, id(transfer)) for node, transfer in self.transfers.items()),
                tuple((id(t), t._version) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding:
            raise ValueError('Directional nested RWG maps or tree configuration changed')
        self.tree.assert_current()
        if isinstance(self.basis, ScalarSystemBlocks):self.basis.assert_current()
        else:self.geometry.assert_current(self.basis)
        if self.layout is not None:self.layout.assert_current()
        for shared in (*self.phase_layouts.values(),*self.center_layouts.values()):shared.assert_current()
        for transfer in self.transfers.values():transfer.assert_current()
        for records in self.leaves.values():
            for _, projection in records:projection.assert_current()

    def _validate(self, value, tail):
        if (not isinstance(value, torch.Tensor) or value.ndim < len(tail)
                or value.shape[-len(tail):] != tail or value.device != self.device
                or value.dtype != self.dtype):
            raise ValueError('Directional nested action requires matching complex128 values')

    @staticmethod
    def _finite(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Directional nested accumulation exceeds native precision')
        return value

    def _transfer(self, parent, child):
        return self.transfers[child]

    def _upward_dispatch(self, currents, kind):
        self.assert_current();self._validate(currents, (self.tree.size,))
        if torch.is_grad_enabled() and currents.requires_grad:
            return _UpwardCurrent.apply(currents, self, kind)
        return self._upward(currents, kind)

    def _upward(self, currents, kind):
        self.assert_current();self._validate(currents, (self.tree.size,))
        rows = math.prod(currents.shape[:-1])
        output = tuple(currents.new_empty((rows, count, 4)) for count in self.node_counts)
        for start, chunk in _rhs_chunks(currents, 1, self.rhs_batch):
            values = [None]*len(self.node_counts)
            for node in reversed(range(len(self.node_counts))):
                if node in self.leaves:
                    value = chunk.new_zeros((len(chunk), self.node_counts[node], 4))
                    for offset, projection in self.leaves[node]:
                        local = chunk[..., projection.edges+offset]
                        value = value+(projection.source_moments(local) if kind == 'source'
                                       else projection.receiver_adjoint(local))
                else:
                    value = chunk.new_zeros((len(chunk), self.node_counts[node], 4))
                    for child in self.children[node]:
                        shift = self._transfer(node, child)
                        value = value+(shift.source(values[child]) if kind == 'source'
                                       else shift.receiver(values[child], adjoint=True))
                values[node] = self._finite(value)
                output[node][start:start+len(chunk)] = value
        self.assert_current()
        return tuple(v.reshape(*currents.shape[:-1], count, 4) for v,count in zip(output,self.node_counts))

    def _downward_dispatch(self, coefficients, kind, leading=None, *, allow_missing=False):
        self.assert_current()
        if not isinstance(coefficients, (tuple, list)) or len(coefficients) != len(self.node_counts):
            raise ValueError('Directional downward requires all node coefficient slots')
        for node,value in enumerate(coefficients):
            if value is None and allow_missing:continue
            self._validate(value, (self.node_counts[node], 4))
            if leading is None:leading = value.shape[:-2]
            if value.shape[:-2] != leading:
                raise ValueError('Directional node coefficient batches must agree')
        if leading is None:
            raise ValueError('Missing directional node batch metadata')
        if torch.is_grad_enabled() and any(v is not None and v.requires_grad for v in coefficients):
            return _DownwardCurrent.apply(self, kind, leading, *coefficients)
        return self._downward(coefficients, kind, leading)

    def _downward(self, coefficients, kind, leading):
        self.assert_current()
        rows = math.prod(leading)
        output = torch.zeros((rows, self.tree.size), device=self.device, dtype=self.dtype)
        iterators = [None if v is None else iter(_rhs_chunks(v, 2, self.rhs_batch)) for v in coefficients]
        for start in range(0, rows, self.rhs_batch):
            batch = min(self.rhs_batch, rows-start)
            values = [output.new_zeros((batch,count,4)) if it is None else next(it)[1]
                      for count,it in zip(self.node_counts,iterators)]
            result = output.new_zeros((batch, self.tree.size))
            for node, child_ids in enumerate(self.children):
                self._finite(values[node])
                if child_ids:
                    for child in child_ids:
                        shift = self._transfer(node, child)
                        inherited = (shift.source(values[node], adjoint=True) if kind == 'source'
                                     else shift.receiver(values[node]))
                        values[child] = self._finite(values[child]+inherited)
                else:
                    for offset, projection in self.leaves[node]:
                        local = (projection.source_adjoint(values[node]) if kind == 'source'
                                 else projection.receiver_evaluate(values[node]))
                        result = result.index_add(-1, projection.edges+offset, local)
            output[start:start+batch] = self._finite(result)
        self.assert_current()
        return output.reshape(*leading, self.tree.size)

    def source_upward(self, currents):return self._upward_dispatch(currents, 'source')
    def source_downward(self, coefficients):return self._downward_dispatch(coefficients, 'source')
    def receiver_upward(self, currents):return self._upward_dispatch(currents, 'receiver')
    def receiver_downward(self, coefficients):return self._downward_dispatch(coefficients, 'receiver')

    def receipt(self):
        self.assert_current()
        record = dict(version='experimental_fixed_carrier_nested_rwg_v1', order=self.order,
            coefficients_per_channel=self.count, nodes=self.tree.node_count,
            integrated_leaves=len(self.leaves), integrated_leaf_body_groups=sum(map(len, self.leaves.values())),
            internal_rwg_matrices_retained=False, full_node_input_and_output=True,
            source_receiver_independently_integrated=True, direction_changes_admitted=False,
            quadrature_order=self.quadrature_order, quadrature_convergence_certified=False,
            polynomial_exact=False, rhs_batch=self.rhs_batch,
            owned_retained_bytes=self.retained_bytes, owned_geometry_included=self.owns_geometry,
            owned_geometry_bytes=self.owned_geometry_bytes, owned_layout_included=self.owns_layout,
            owned_layout_bytes=self.owned_layout_bytes, shared_tree_excluded=True,
            planned_tensor_workspace_bytes=self.planned_tensor_workspace_bytes,
            tensor_workspace_excludes=('caller all-node input/output tensors', 'allocator cache', 'whole system'),
            current_backward='distinct_guarded_full_Hermitian_maps_no_saved_node_graph',
            fixed_carrier_tree_integrated=True, cone_tree_integrated=False,
            truncation_certified=False, physical_qualification=False,
            complexity_qualification=False, peak_memory_qualified=False)
        if self.changing_carriers:
            record.update(version='experimental_changing_carrier_nested_rwg_v1',
                node_orders=self.node_orders,coefficients_per_node=self.node_counts,phase_orders=self.phase_orders,
                direction_changes_admitted=True,per_edge_phase_relative_budget=self.phase_relative_budget,
                admission_scope='per-edge ideal relative phase only; not whole-tree current/kernel/derivative/FP error',
                actual_complete_support_radius_checked=True,owned_node_carrier_bytes=24*self.tree.node_count,
                fixed_carrier_tree_integrated=False,changing_carrier_tree_integrated=True,
                floating_point_certified=False,phase_derivative_error_certified=False,
                relative_current_error_certified=False,kernel_error_certified=False)
            if self.phase_storage == 'compact_phase_v1':
                record.update(version='experimental_compact_changing_carrier_nested_rwg_v1',
                    phase_storage=self.phase_storage,shared_phase_layout_count=len(self.phase_layouts),
                    shared_center_layout_count=len(self.center_layouts),
                    tree_owned_phase_layout_bytes=self.owned_phase_layout_bytes,
                    tree_owned_center_layout_bytes=self.owned_center_layout_bytes,
                    edge_owned_layout_bytes=0,shared_layouts_counted_once_by_tree=True,
                    native_order_or_admission_range_identical_to_dense=False,
                    bit_exact_action_replay=False,production_qualification=False)
        return record


    def _ideal_phase_path_log_products(self, tensor_workspace_budget_bytes):
        """CUDA maximum log product over all descendant leaf-to-node paths."""
        self.assert_current()
        if (type(tensor_workspace_budget_bytes) is not int
                or tensor_workspace_budget_bytes < 1):
            raise ValueError('Directional phase-path workspace must be a positive integer')
        planned = directional_phase_path_workspace_bytes(self.tree.node_count)
        if planned > tensor_workspace_budget_bytes:
            raise MemoryError('Directional phase-path workspace exceeded before accumulation')
        increments = self.centers.new_zeros(self.tree.node_count)
        if self.changing_carriers:
            for child, transfer in self.transfers.items():
                # Exact edges store a zero log sentinel, not log(delta)=0.
                if not transfer.exact_phase:
                    delta = transfer.log_phase_bound.clamp_min(
                        math.log(torch.finfo(torch.float64).tiny)).exp()
                    increments[child] = torch.log1p(delta)
        products = torch.zeros_like(increments)
        for node in reversed(range(self.tree.node_count)):
            children = self.children[node]
            if children:
                products[node] = (increments[list(children)]+products[list(children)]).amax()
        if not bool(torch.isfinite(products).all() & (products >= 0).all()):
            raise ValueError('Directional phase-path log products exceed native range')
        self.assert_current()
        return products

    def ideal_phase_path_bound(self, node, *, tensor_workspace_budget_bytes=128*1024**2):
        """Ideal multiplicative phase defect; excludes native/quadrature errors."""
        if type(node) is not int or not 0 <= node < self.tree.node_count:
            raise ValueError('Directional phase-path node is outside the prescribed tree')
        products = self._ideal_phase_path_log_products(tensor_workspace_budget_bytes)
        logged = products[node]
        value = torch.expm1(logged)
        if not bool(torch.isfinite(value) & ((logged == 0) | (value > 0))):
            raise ValueError('Directional positive phase-path defect exceeds native range')
        return dict(version='directional_nested_ideal_phase_path_v1', node=node,
            edge_range_start=self.starts[node], edge_range_stop=self.stops[node],
            max_log_one_plus_relative_bound=float(logged), relative_bound=float(value),
            exact_phase_path=bool(logged == 0),
            scope='ideal_leaf_to_prescribed_node_phase_product_only',
            complete_support_and_padding_checked=True, source_receiver_signs_covered=True,
            phase_derivative_error_certified=False, quadrature_certified=False,
            floating_point_certified=False, global_operator_certified=False,
            production_qualification=False)


def directional_phase_path_workspace_bytes(nodes):
    """Conservative visible full-node accumulation arrays and index gathers."""
    return nodes*8*16+4096


class SparseDirectionalRWGProjection:
    """Only requested node/carrier/order descendant contexts on a physical tree.

    Shared upward/downward and current backward code handles a forest of active
    contexts. Equal carrier/order requests share descendants and shift layouts;
    absent physical nodes have no map, transfer or coefficient cache slots.
    This is prescribed sparse context construction, not automatic global FMM.
    """
    _transfer = NestedDirectionalRWGProjection._transfer
    _validate = NestedDirectionalRWGProjection._validate
    _finite = staticmethod(NestedDirectionalRWGProjection._finite)
    _upward_dispatch = NestedDirectionalRWGProjection._upward_dispatch
    _upward = NestedDirectionalRWGProjection._upward
    _downward_dispatch = NestedDirectionalRWGProjection._downward_dispatch
    _downward = NestedDirectionalRWGProjection._downward
    source_upward = NestedDirectionalRWGProjection.source_upward
    source_downward = NestedDirectionalRWGProjection.source_downward
    receiver_upward = NestedDirectionalRWGProjection.receiver_upward
    receiver_downward = NestedDirectionalRWGProjection.receiver_downward

    def __init__(self, basis, tree, carriers, orders, requests, wave_number, *,
                 quadrature_order, geometry=None, face_batch=32, rhs_batch=4,
                 retained_budget_bytes=1024**3, tensor_workspace_budget_bytes=256*1024**2):
        self._initialize(basis, tree, carriers, orders, requests, wave_number,
            quadrature_order=quadrature_order, geometry=geometry, face_batch=face_batch, rhs_batch=rhs_batch,
            retained_budget_bytes=retained_budget_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)

    def _initialize(self, basis, tree, carriers, orders, requests, wave_number, *,
                    quadrature_order, geometry=None, face_batch=32, rhs_batch=4,
                    retained_budget_bytes=1024**3, tensor_workspace_budget_bytes=256*1024**2,
                    _dag_configuration=None):
        from .fmm_directional_rwg import directional_rwg_projection_capacity
        from .direct_capacity import storage_bytes
        dag = _dag_configuration is not None
        if dag and not isinstance(self, DirectionalRWGDAGProjection):
            raise TypeError('Changing DAG initialization requires its distinct context protocol')
        _points(carriers)
        if not isinstance(tree, SupportOctree):
            raise TypeError('Sparse directional contexts require a physical support octree')
        tree.assert_current()
        if tree.expansion_centers is not None:
            raise NotImplementedError('Directional projections require original support-box centers')
        if (carriers.ndim != 2 or carriers.shape[1] != 3
                or carriers.device != tree.device or carriers.dtype != torch.float64
                or not bool(((carriers.norm(dim=-1)-1).abs() <= 512*torch.finfo(torch.float64).eps).all())):
            raise ValueError('Sparse directional contexts require bounded fixed native unit carriers')
        orders = tuple(orders); requests = tuple(tuple(v) for v in requests)
        if (len(orders) != len(carriers) or any(type(p) is not int or not 0 <= p <= 12 for p in orders)
                or any(len(v) != 2 or any(type(i) is not int for i in v)
                    or not 0 <= v[0] < tree.node_count or not 0 <= v[1] < len(carriers) for v in requests)
                or len(set(requests)) != len(requests)
                or (not dag and {v[1] for v in requests} != set(range(len(carriers))))):
            raise ValueError('Sparse contexts require unique in-range requests and used carrier/order slots')
        if any(type(v) is not int or v < 1 for v in
               (quadrature_order, face_batch, rhs_batch, retained_budget_bytes, tensor_workspace_budget_bytes)):
            raise ValueError('Sparse quadrature, batches and capacities must be positive integers')
        catalog_workspace = _directional_carrier_catalog_workspace_bytes(len(carriers))
        if catalog_workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Directional catalog workspace exceeded before canonical validation')
        if len(carriers):
            # Preserve exact float64 bit identity, including signed zeros, and
            # order. Only the unique row count leaves CUDA as integer metadata.
            bits = carriers.contiguous().view(torch.int64)
            degrees = torch.tensor(orders, device=carriers.device, dtype=torch.int64)
            keys = torch.cat((bits, degrees[:,None]), dim=1)
            if torch.unique(keys, dim=0).shape[0] != len(carriers):
                raise ValueError('Sparse carrier/order slots must be canonical and distinct')
            del bits, degrees, keys
        if orders and quadrature_order < (max(orders)+4)//2:
            raise ValueError('Sparse quadrature underresolves an active polynomial order')
        if type(wave_number) not in (int, float, complex):
            raise ValueError('Sparse wave number must be fixed and passive')
        wave = complex(wave_number)
        if not math.isfinite(wave.real) or not math.isfinite(wave.imag) or wave.imag < 0:
            raise ValueError('Sparse wave number must be finite and passive')
        global_space = isinstance(basis, ScalarSystemBlocks)
        if global_space:
            basis.assert_current()
            if geometry is not None:
                raise ValueError('Global sparse contexts share actual provider geometry')
            size, device, dtype = basis.system.total_edges, basis.system.device, basis.system.dtype
            geometry = basis
            faces = sum(len(body.basis.mesh.faces) for body in basis.system.bodies)
        else:
            if not isinstance(geometry, PreparedRWGGeometry):
                raise TypeError('Sparse single-body contexts require caller prepared geometry')
            geometry.assert_current(basis)
            size, device, dtype = basis.n_edges, basis.mesh.device, basis.mesh.dtype
            faces = len(basis.mesh.faces)
        _require_sparse_fixed_geometry(basis, geometry)
        if (tree.size != size or tree.device != device or dtype != torch.float64 or tree.lower.dtype != torch.float64
                or not torch.equal(tree.permutation.sort().values, torch.arange(size, device=device))):
            raise ValueError('Sparse contexts require a complete matching physical coefficient space')
        _points(tree.lower); _points(tree.upper)
        if (tree.lower.shape != (tree.node_count, 3) or tree.upper.shape != tree.lower.shape
                or tree.upper.dtype != torch.float64 or tree.upper.device != device
                or tree.children.shape != (tree.node_count, 8) or not bool((tree.lower <= tree.upper).all())
                or any(t.dtype != torch.int64 or t.device != device for t in
                    (tree.permutation, tree.starts, tree.stops, tree.children, tree.depth))
                or any(t.shape != (tree.node_count,) for t in (tree.starts, tree.stops, tree.depth))):
            raise ValueError('Sparse contexts require matching physical tree topology')
        original_children = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
        starts, stops = tuple(tree.starts.cpu().tolist()), tuple(tree.stops.cpu().tolist())
        parents = {}
        for node, child_ids in enumerate(original_children):
            if not 0 <= starts[node] < stops[node] <= size:
                raise ValueError('Sparse physical tree ranges must be nonempty and in range')
            next_start = starts[node]
            for child in child_ids:
                if child <= node or child >= tree.node_count or child in parents or starts[child] != next_start or stops[child] > stops[node]:
                    raise ValueError('Sparse physical children must form disjoint topological ranges')
                parents[child] = node; next_start = stops[child]
            if child_ids and next_start != stops[node]:
                raise ValueError('Sparse physical children must cover their parent range')
        if starts[0] != 0 or stops[0] != size or set(parents) != set(range(1, tree.node_count)):
            raise ValueError('Sparse physical tree must have one complete connected root')
        if dag:
            keys, children, phases = _directional_dag_topology(original_children, requests, orders, _dag_configuration[0])
            index = {key:i for i,key in enumerate(keys)}
            if {family for _,family in keys} != set(range(len(carriers))):
                raise ValueError('DAG catalog requires every carrier/order family to be used')
        else:
            # Host integer topology only; each family shares its closures.
            active = [set() for _ in carriers]
            for node, carrier in requests:
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current in active[carrier]:continue
                    active[carrier].add(current); stack.extend(original_children[current])
            keys = tuple((node, carrier) for carrier, nodes in enumerate(active) for node in sorted(nodes))
            index = {key: i for i, key in enumerate(keys)}
            children = tuple(tuple(index[(child, carrier)] for child in original_children[node]) for node, carrier in keys)
        routes, context_orders, counts, shift_orders, phase_keys, expected, workspace = _directional_context_capacity(
            basis, tree, keys, children, phases if dag else {}, orders, len(carriers),
            quadrature_order, face_batch, rhs_batch, dag=dag)
        if expected > retained_budget_bytes:
            raise MemoryError('Sparse retained capacity exceeded before geometry, maps and transfers')
        if workspace > tensor_workspace_budget_bytes:
            raise MemoryError('Sparse component workspace exceeded before geometry, maps and transfers')
        if global_space:
            lower, upper = basis.support_boxes()
        else:
            supports = torch.cat((geometry.triangles[basis.plus_faces], geometry.triangles[basis.minus_faces]), 1)
            lower, upper = supports.amin(1), supports.amax(1)
            del supports
        node_ids = torch.tensor([n for n, _ in keys], device=device, dtype=torch.int64)
        centers = tree.lower[node_ids]*.5+tree.upper[node_ids]*.5
        radii = []
        for context, (node, _) in enumerate(keys):
            edges = tree.permutation[starts[node]:stops[node]]
            if not bool((lower[edges] >= tree.lower[node]).all() & (upper[edges] <= tree.upper[node]).all()):
                raise ValueError('Sparse physical node does not enclose complete actual RWG supports')
            corners = torch.maximum((lower[edges]-centers[context]).abs(), (upper[edges]-centers[context]).abs())
            radius = corners.norm(dim=-1).amax()*(1+512*torch.finfo(torch.float64).eps)
            if not bool(torch.isfinite(radius) & (radius > 0)):
                raise ValueError('Sparse complete support radius is not natively representable')
            radii.append(radius)
        self.basis, self.tree, self.geometry = basis, tree, geometry
        self.context_keys, self.request_indices = keys, tuple(index[v] for v in requests)
        self.context_node_ids, self.context_carrier_ids = node_ids, torch.tensor([c for _, c in keys], device=device, dtype=torch.int64)
        self.carriers, self.carrier_orders = carriers.clone(), orders
        self.centers, self.radii = centers, torch.stack(radii) if radii else centers.new_empty((0,))
        self.children, self.node_orders, self.node_counts = children, context_orders, counts
        self.wave_number, self.quadrature_order, self.face_batch, self.rhs_batch = wave, quadrature_order, face_batch, rhs_batch
        self.device, self.dtype = device, torch.complex128
        self.retained_budget_bytes, self.tensor_workspace_budget_bytes = retained_budget_bytes, tensor_workspace_budget_bytes
        self.retained_bytes, self.planned_tensor_workspace_bytes = expected, workspace
        self._caller_carriers, self._caller_carrier_state = carriers, (id(carriers), carriers._version)
        if dag:
            self.phase_relative_budget = _dag_configuration[1]
            self.dag_links = tuple((a,c,phases[a,c]) for a,row in enumerate(children) for c in row)
            self.changing_carriers = True
            if self.plan_guard is not None:
                plan = self.plan_guard
                plan.assert_current()
                if (self.context_keys != plan.context_keys
                        or not torch.equal(self.centers,plan.region_plan.centers[self.context_node_ids])
                        or not torch.equal(self.radii,plan.region_plan.radii[self.context_node_ids])):
                    raise ValueError('Automatic DAG path plan differs from actual context support before maps')
            # Admit every physical phase defect before layout or leaf factories.
            for a,c,L in self.dag_links:
                delta = self.carriers[keys[a][1]]-self.carriers[keys[c][1]]
                x = delta.norm()*abs(wave)*self.radii[c]
                exact = wave == 0 or torch.equal(self.carriers[keys[a][1]],self.carriers[keys[c][1]])
                if not bool(torch.isfinite(x)) or (not exact and not bool(x>0)):
                    raise ValueError('DAG phase argument exceeds native precision')
                if not exact and bool(2*x+(L+1)*x.log()-math.lgamma(L+2)>math.log(self.phase_relative_budget)):
                    raise ValueError('DAG ideal relative phase bound exceeds configured budget before maps')
        self.layouts = {p:SeparableMultipoleShift(p, device=device, dtype=torch.float64) for p in shift_orders}
        if dag:
            self.phase_layouts = {key:ChangingCarrierPhaseLayout(*key,device=device) for key in phase_keys}
            self.transfers = {(a,c):DirectionalChangingCarrierShift(centers[c]-centers[a],
                self.carriers[keys[c][1]],self.carriers[keys[a][1]],wave,context_orders[c],context_orders[a],L,
                child_radius=self.radii[c],phase_relative_budget=self.phase_relative_budget,phase_storage='compact_phase_v1',
                phase_layout=self.phase_layouts[context_orders[c],context_orders[a],L],center_layout=self.layouts[context_orders[a]],
                rhs_batch=rhs_batch,retained_budget_bytes=128+32*len(multi_indices(L)),tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                for a,c,L in self.dag_links}
        else:
            self.transfers = {child:DirectionalFixedCarrierShift(centers[child]-centers[parent], self.carriers[keys[child][1]], wave,
                context_orders[child], layout=self.layouts[context_orders[child]], rhs_batch=rhs_batch,
                retained_budget_bytes=104, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                for parent, row in enumerate(children) for child in row}
        self.leaves = {}
        for context, groups in routes.items():
            node, carrier = keys[context]; edges = tree.permutation[starts[node]:stops[node]]
            records = []
            for bi, selected in groups:
                if global_space:
                    body = basis.system.bodies[bi]
                    local_basis, offset, geom = body.basis, body.edge_offset, basis.geometry[bi]
                    local_edges = edges[selected]-offset
                else:
                    local_basis, offset, geom, local_edges = basis, 0, geometry, edges
                projection = DirectionalRWGProjection(local_basis, local_edges, centers[context], self.carriers[carrier], wave,
                    context_orders[context], quadrature_order=quadrature_order, face_batch=face_batch, geometry=geom,
                    retained_budget_bytes=retained_budget_bytes, tensor_workspace_budget_bytes=tensor_workspace_budget_bytes)
                records.append((offset, projection))
            self.leaves[context] = tuple(records)
        self._binding = self._state()
        if storage_bytes(self.tensors()) != expected:
            raise RuntimeError('Sparse owned inventory differs from structural plan')
        self.assert_current()

    def tensors(self):
        return (self.context_node_ids, self.context_carrier_ids, self.centers, self.radii, self.carriers,
            *(t for layout in self.layouts.values() for t in layout.tensors()),
            *(t for transfer in self.transfers.values() for t in transfer.tensors()),
            *(t for records in self.leaves.values() for _, projection in records for t in projection.tensors()))

    def _state(self):
        return (id(self.basis), id(self.tree), id(self.geometry), self.tree.version, self.tree.leaf_size, self.tree.morton_bits,
            tuple(t.requires_grad for t in self.tree.tensors()),
            self.context_keys, self.request_indices, self.carrier_orders, self.children, self.node_orders, self.node_counts,
            self.wave_number, self.quadrature_order, self.face_batch, self.rhs_batch, self.device, self.dtype,
            self.retained_budget_bytes, self.tensor_workspace_budget_bytes, self.retained_bytes, self.planned_tensor_workspace_bytes,
            id(self._caller_carriers), self._caller_carriers._version, self._caller_carriers.requires_grad, self._caller_carrier_state,
            tuple((p, id(layout)) for p, layout in self.layouts.items()),
            tuple((c, id(shift)) for c, shift in self.transfers.items()),
            tuple((c, tuple((offset,id(p)) for offset,p in records)) for c,records in self.leaves.items()),
            tuple((id(t), t._version, t.requires_grad) for t in self.tensors()))

    def assert_current(self):
        if self._state() != self._binding or (id(self._caller_carriers), self._caller_carriers._version) != self._caller_carrier_state:
            raise ValueError('Sparse directional context or caller carrier configuration changed')
        self.tree.assert_current()
        if isinstance(self.basis, ScalarSystemBlocks):self.basis.assert_current()
        else:self.geometry.assert_current(self.basis)
        _require_sparse_fixed_geometry(self.basis, self.geometry)
        for layout in self.layouts.values():layout.assert_current()
        for shift in self.transfers.values():shift.assert_current()
        for records in self.leaves.values():
            for _, projection in records:projection.assert_current()

    def receipt(self):
        self.assert_current()
        return dict(version='prescribed_sparse_directional_rwg_contexts_v1', physical_nodes=self.tree.node_count,
            active_contexts=len(self.context_keys), carrier_order_families=len(self.carriers),
            requested_context_indices=list(self.request_indices), integrated_leaf_contexts=len(self.leaves),
            integrated_leaf_body_groups=sum(map(len, self.leaves.values())), transfer_contexts=len(self.transfers),
            shared_order_layouts=len(self.layouts), owned_retained_bytes=self.retained_bytes,
            planned_component_workspace_bytes=self.planned_tensor_workspace_bytes,
            caller_geometry_and_tree_excluded=True, absent_contexts_not_allocated=True,
            source_receiver_independently_integrated=True, same_carrier_descendant_phase_truncation_exact_zero=True,
            finite_phase_native_certified=False, quadrature_convergence_certified=False,
            automatic_global_cones=False, complete_system_certified=False, complexity_qualification=False,
            physical_qualification=False, production_qualification=False)


def _require_sparse_fixed_geometry(basis, geometry):
    geometries = basis.geometry if isinstance(basis, ScalarSystemBlocks) else (geometry,)
    for prepared in geometries:
        if prepared.basis.mesh.vertices.requires_grad or any(t.requires_grad for t in prepared.tensors()):
            raise ValueError('Sparse directional contexts require fixed actual geometry')


def _directional_dag_topology(physical_children, requests, orders, links):
    """Canonical integer DAG; every parent covers each physical child once."""
    if not isinstance(links,(tuple,list)):
        raise ValueError('DAG links require explicit parent/child keys and phase orders')
    keys=set(requests);edges={};adjacency={}
    for link in links:
        if not isinstance(link,(tuple,list)) or len(link)!=3:
            raise ValueError('DAG link requires parent key, child key and phase order')
        parent,child,L=link
        if any(not isinstance(v,(tuple,list)) or len(v)!=2 or any(type(i) is not int for i in v) for v in (parent,child)) or type(L) is not int or not 0<=L<=12:
            raise ValueError('DAG link metadata must be bounded integer context/phase keys')
        parent,child=tuple(parent),tuple(child)
        for node,carrier in (parent,child):
            if not 0<=node<len(physical_children) or not 0<=carrier<len(orders):
                raise ValueError('DAG context key is outside its physical tree/carriers')
        if child[0] not in physical_children[parent[0]] or orders[child[1]]<orders[parent[1]]+L or (parent,child) in edges:
            raise ValueError('DAG links require unique direct physical children and source order covering parent+phase')
        keys.update((parent,child));edges[parent,child]=L
        adjacency.setdefault(parent,[]).append(child)
    keys=tuple(sorted(keys));index={key:i for i,key in enumerate(keys)}
    children=[];phases={}
    for parent in keys:
        selected=sorted(adjacency.get(parent,()),key=lambda c:c[0])
        if tuple(c[0] for c in selected)!=physical_children[parent[0]]:
            raise ValueError('DAG parent contexts must cover every physical child exactly once')
        children.append(tuple(index[c] for c in selected))
        for c in selected:phases[index[parent],index[c]]=edges[parent,c]
    seen=set();pending=[index[key] for key in requests]
    while pending:
        node=pending.pop()
        if node in seen:continue
        seen.add(node);pending.extend(children[node])
    if seen!=set(range(len(keys))):
        raise ValueError('DAG contexts must be reachable from the explicit requests')
    return keys,tuple(children),phases


class DirectionalRWGDAGProjection:
    """Prescribed changing-carrier contexts sharing physical descendant maps.

    A shared child may have several parents; source reuse is one computation,
    while its Hermitian downward action sums every incoming parent contribution.
    Contexts are topological by physical node then canonical carrier/order.
    This distinct protocol is not accepted by fixed-phase sparse M2L admission.
    """
    _initialize = SparseDirectionalRWGProjection._initialize
    _validate = SparseDirectionalRWGProjection._validate
    _finite = staticmethod(SparseDirectionalRWGProjection._finite)
    _upward_dispatch = SparseDirectionalRWGProjection._upward_dispatch
    _upward = SparseDirectionalRWGProjection._upward
    _downward_dispatch = SparseDirectionalRWGProjection._downward_dispatch
    _downward = SparseDirectionalRWGProjection._downward
    source_upward = SparseDirectionalRWGProjection.source_upward
    source_downward = SparseDirectionalRWGProjection.source_downward
    receiver_upward = SparseDirectionalRWGProjection.receiver_upward
    receiver_downward = SparseDirectionalRWGProjection.receiver_downward

    def __init__(self,basis,tree,carriers,orders,requests,links,wave_number,*,quadrature_order,
                 phase_relative_budget=1e-12,geometry=None,face_batch=32,rhs_batch=4,
                 retained_budget_bytes=1024**3,tensor_workspace_budget_bytes=256*1024**2,
                 plan_guard=None):
        if plan_guard is not None:
            from .fmm_order import AutomaticDirectionalDAGPlan
            if not isinstance(plan_guard,AutomaticDirectionalDAGPlan):
                raise TypeError('DAG planning guard requires its admitted automatic plan')
            plan_guard.assert_current()
            if (plan_guard.region_plan.basis is not basis or plan_guard.region_plan.tree is not tree
                    or plan_guard.region_plan.medium.k!=wave_number or not torch.equal(plan_guard.carriers,carriers)
                    or plan_guard.orders!=tuple(orders) or plan_guard.requests!=tuple(requests) or plan_guard.links!=tuple(links)
                    or any(getattr(plan_guard,name)!=value for name,value in (
                        ('quadrature_order',quadrature_order),('face_batch',face_batch),('rhs_batch',rhs_batch),
                        ('phase_relative_budget',phase_relative_budget),('retained_budget_bytes',retained_budget_bytes),
                        ('tensor_workspace_budget_bytes',tensor_workspace_budget_bytes)))):
                raise ValueError('DAG configuration differs from its bound automatic plan')
        self.plan_guard=plan_guard
        if type(phase_relative_budget) not in (int,float) or not math.isfinite(phase_relative_budget) or not 0<phase_relative_budget<1:
            raise ValueError('DAG phase budget must be positive finite and smaller than one')
        self._initialize(basis,tree,carriers,orders,requests,wave_number,quadrature_order=quadrature_order,
            geometry=geometry,face_batch=face_batch,rhs_batch=rhs_batch,
            retained_budget_bytes=retained_budget_bytes,tensor_workspace_budget_bytes=tensor_workspace_budget_bytes,
            _dag_configuration=(links,phase_relative_budget))

    def _transfer(self,parent,child):return self.transfers[parent,child]

    def tensors(self):
        return (*SparseDirectionalRWGProjection.tensors(self),
            *(t for layout in self.phase_layouts.values() for t in layout.tensors()))

    def _state(self):
        return (SparseDirectionalRWGProjection._state(self),self.dag_links,self.phase_relative_budget,
            self.changing_carriers,tuple((key,id(layout)) for key,layout in self.phase_layouts.items()),id(self.plan_guard))

    def assert_current(self):
        SparseDirectionalRWGProjection.assert_current(self)
        for layout in self.phase_layouts.values():layout.assert_current()
        if self.plan_guard is not None:self.plan_guard.assert_current()

    def _ideal_phase_path_log_products(self,tensor_workspace_budget_bytes):
        self.assert_current()
        if type(tensor_workspace_budget_bytes) is not int or tensor_workspace_budget_bytes<directional_phase_path_workspace_bytes(len(self.node_counts)):
            raise MemoryError('DAG phase-path workspace exceeded before accumulation')
        if self.plan_guard is not None:
            # Fixed planner versions/configuration are in the owner validation
            # chain. Canonical keys and complete supports were checked before
            # maps; borrowing this array retains no second path cache.
            return self.plan_guard.path_products
        products=self.centers.new_zeros(len(self.node_counts))
        for parent in reversed(range(len(self.node_counts))):
            values=[]
            for child in self.children[parent]:
                transfer=self._transfer(parent,child)
                delta=products.new_zeros(()) if transfer.exact_phase else transfer.log_phase_bound.clamp_min(math.log(torch.finfo(torch.float64).tiny)).exp()
                values.append(torch.log1p(delta)+products[child])
            if values:products[parent]=torch.stack(values).amax()
        if not bool(torch.isfinite(products).all() & (products>=0).all()):
            raise ValueError('DAG phase products exceed native representation')
        self.assert_current()
        return products

    def ideal_phase_path_bound(self,context,*,tensor_workspace_budget_bytes=128*1024**2):
        if type(context) is not int or not 0<=context<len(self.context_keys):
            raise ValueError('DAG phase context is outside the active graph')
        logged=self._ideal_phase_path_log_products(tensor_workspace_budget_bytes)[context]
        value=torch.expm1(logged)
        if not bool(torch.isfinite(value) & ((logged==0)|(value>0))):
            raise ValueError('DAG phase defect exceeds native representation')
        return dict(version='directional_dag_ideal_phase_path_v1',context=context,
            physical_node=self.context_keys[context][0],max_log_one_plus_relative_bound=float(logged),relative_bound=float(value),
            exact_phase_path=bool(logged==0),scope='ideal_prescribed_context_to_leaf_phase_product_only',
            shared_parents_do_not_multiply_independent_paths=True,native_error_certified=False,
            phase_derivative_error_certified=False,quadrature_certified=False,production_qualification=False)

    def receipt(self):
        self.assert_current()
        indegree=[0]*len(self.context_keys)
        for row in self.children:
            for child in row:indegree[child]+=1
        return dict(version='prescribed_shared_changing_directional_rwg_dag_v1',physical_nodes=self.tree.node_count,
            active_contexts=len(self.context_keys),requested_contexts=len(self.request_indices),
            shared_child_contexts=sum(i>1 for i in indegree),transfer_edges=len(self.transfers),
            integrated_leaf_contexts=len(self.leaves),shared_phase_layouts=len(self.phase_layouts),
            shared_center_layouts=len(self.layouts),owned_retained_bytes=self.retained_bytes,
            planned_component_workspace_bytes=self.planned_tensor_workspace_bytes,
            caller_geometry_tree_and_python_metadata_excluded=True,automatic_cones=False,
            complete_system_certified=False,complexity_qualification=False,production_qualification=False)


def _directional_carrier_catalog_workspace_bytes(carrier_count):
    """Native key/sort tensor allowance; allocator/library workspace excluded."""
    return 4096+128*carrier_count


def _directional_context_capacity(basis, tree, keys, children, phases, orders, carrier_count,
                                  quadrature_order, face_batch, rhs_batch, *, dag):
    """Shared exact structural inventory before map/layout/transfer allocation."""
    from .fmm_directional_rwg import directional_rwg_projection_capacity
    global_space = isinstance(basis, ScalarSystemBlocks)
    starts, stops = tree.starts.tolist(), tree.stops.tolist()
    size = tree.size
    bodies = tuple(b.basis for b in basis.system.bodies) if global_space else (basis,)
    faces = sum(len(b.mesh.faces) for b in bodies)
    routes = {}
    for context, (node, _) in enumerate(keys):
        if not children[context]:
            edges = tree.permutation[starts[node]:stops[node]]
            routes[context] = basis._body_selection(edges) if global_space else ((None, None),)
    context_orders = tuple(orders[c] for _, c in keys)
    counts = tuple(len(multi_indices(p)) for p in context_orders)
    shift_orders = tuple(sorted({context_orders[parent] for parent,row in enumerate(children) if row})) if dag else tuple(sorted({context_orders[c] for row in children for c in row}))
    layout_bytes = sum(144*len(multi_indices(p))*(p+1) for p in shift_orders)
    phase_keys = tuple(sorted({(context_orders[c],context_orders[a],phases[a,c]) for a,row in enumerate(children) for c in row})) if dag else ()
    phase_layout_bytes = sum(8*len(multi_indices(pp))*len(multi_indices(L))+32*len(multi_indices(L)) for ps,pp,L in phase_keys)
    leaf_bytes, projection_workspace = 0, 0
    for context, groups in routes.items():
        node, _ = keys[context]
        edges = tree.permutation[starts[node]:stops[node]]
        for bi, selected in groups:
            local_basis = basis.system.bodies[bi].basis if global_space else basis
            n = len(selected) if global_space else len(edges)
            _, amount, workspace = directional_rwg_projection_capacity(local_basis, n, context_orders[context], quadrature_order, face_batch, False)
            leaf_bytes += amount; projection_workspace = max(projection_workspace, workspace)
    transfer_count = sum(map(len, children))
    transfer_bytes = sum(128+32*len(multi_indices(phases[a,c])) for a,row in enumerate(children) for c in row) if dag else 104*transfer_count
    expected = 48*len(keys)+24*carrier_count+layout_bytes+phase_layout_bytes+transfer_bytes+leaf_bytes
    shift_workspace = max((2*144*len(multi_indices(p))*(p+1)
        +rhs_batch*len(multi_indices(p))*(p+1)*4*16*8
        +len(multi_indices(p))*(p+1)*8*16+512 for p in shift_orders), default=0)
    if dag:
        shift_workspace = max((3*(8*len(multi_indices(pp))*len(multi_indices(L))+32*len(multi_indices(L))
            +144*len(multi_indices(pp))*(pp+1)+128+32*len(multi_indices(L)))
            +rhs_batch*len(multi_indices(pp))*len(multi_indices(L))*4*16*8
            +rhs_batch*len(multi_indices(pp))*(pp+1)*4*16*8
            +len(multi_indices(pp))*len(multi_indices(L))*128+4096 for ps,pp,L in phase_keys), default=0)
    workspace = (2*expected+projection_workspace+faces*4096+size*512
        +2*sum(counts)*rhs_batch*4*16+4*rhs_batch*size*16+shift_workspace)
    workspace = max(workspace, _directional_carrier_catalog_workspace_bytes(carrier_count))
    return routes, context_orders, counts, shift_orders, phase_keys, expected, workspace

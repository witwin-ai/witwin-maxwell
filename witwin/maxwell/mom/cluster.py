"""Owned CUDA support-box octrees and bounded-workspace block traversal.

This geometry layer does not assemble or approximate an electromagnetic
operator. Admissibility uses complete basis support boxes, never centroids
alone. Numerical data stay on the CUDA device; traversal control is Python.
"""
from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class SupportOctree:
    permutation: torch.Tensor
    starts: torch.Tensor
    stops: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    children: torch.Tensor
    depth: torch.Tensor
    leaf_size: int
    morton_bits: int
    expansion_centers: torch.Tensor | None = None
    expansion_center_policy: str = 'support_box_midpoint_v1'
    binary_lattice_exponent: int | None = None
    version: str = field(default='cuda_support_octree_v1', init=False)
    _versions: tuple = field(default=(), init=False, repr=False)
    _center_policy_binding: str = field(default='', init=False, repr=False)
    _binary_exponent_binding: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.expansion_center_policy not in ('support_box_midpoint_v1','dyadic_cell_v1','binary_lattice_v1'):
            raise ValueError('Unsupported expansion center policy')
        if self.expansion_center_policy == 'support_box_midpoint_v1':
            if self.expansion_centers is not None:
                raise ValueError('Support-box centers require no additional center tensor')
        else:
            center = self.expansion_centers
            if (not isinstance(center,torch.Tensor) or center.shape != self.lower.shape
                    or center.device != self.lower.device or center.device.type != 'cuda' or center.dtype != torch.float64
                    or self.lower.dtype != torch.float64 or center.requires_grad
                    or not bool(torch.isfinite(center).all())):
                raise ValueError('Dyadic expansion centers require fixed matching CUDA float64 nodes')
            object.__setattr__(self,'version','cuda_support_octree_dyadic_centers_v1')
        if self.expansion_center_policy == 'binary_lattice_v1':
            exponent = self.binary_lattice_exponent
            if type(exponent) is not int or not -1022 <= exponent <= 1023:
                raise ValueError('Binary lattice centers require an explicit normal float64 spacing')
            power = torch.tensor(-exponent,device=self.expansion_centers.device,dtype=torch.int64)
            indices = torch.ldexp(self.expansion_centers,power)
            if (not bool(torch.isfinite(indices).all()) or bool((indices.abs() > 2**51).any())
                    or not torch.equal(indices,indices.round())):
                raise ValueError('Binary lattice centers must fit the exact native integer grid')
            object.__setattr__(self,'version','cuda_support_octree_binary_lattice_v1')
        elif self.binary_lattice_exponent is not None:
            raise ValueError('Explicit lattice spacing requires binary lattice centers')
        object.__setattr__(self,'_binary_exponent_binding',self.binary_lattice_exponent)
        object.__setattr__(self,'_center_policy_binding',self.expansion_center_policy)
        object.__setattr__(self, '_versions', tuple((id(t), t._version) for t in self.tensors()))

    def tensors(self):
        return (self.permutation, self.starts, self.stops, self.lower,
                self.upper, self.children, self.depth) + (
                    () if self.expansion_centers is None else (self.expansion_centers,))

    def assert_current(self):
        if (tuple((id(t), t._version) for t in self.tensors()) != self._versions
                or self.expansion_center_policy != self._center_policy_binding
                or self.binary_lattice_exponent != self._binary_exponent_binding
                or self.expansion_center_policy == 'binary_lattice_v1' and type(self.binary_lattice_exponent) is not int
                or self.expansion_centers is not None and (
                    self.expansion_centers.requires_grad or self.expansion_centers.dtype != torch.float64
                    or self.expansion_centers.device != self.lower.device
                    or self.expansion_centers.shape != self.lower.shape)):
            raise ValueError('Support octree changed after construction')

    @property
    def device(self):
        return self.lower.device

    @property
    def node_count(self):
        return self.starts.numel()

    @property
    def size(self):
        return self.permutation.numel()

    def receipt(self):
        self.assert_current()
        leaves = (self.children < 0).all(dim=1)
        return {'version': self.version, 'points': self.size, 'nodes': self.node_count,
                **({} if self.expansion_centers is None else dict(
                    expansion_center_policy=self.expansion_center_policy,
                    support_bounds='original_complete_support_boxes',
                    radius_rule='complete_box_corners_about_explicit_center')),
                **({} if self.expansion_center_policy != 'binary_lattice_v1' else dict(
                    binary_lattice_exponent=self.binary_lattice_exponent,
                    coefficient_displacements='actual_native_center_differences',
                    center_integer_magnitude_bound=2**51)),
                'leaves': int(leaves.sum()), 'leaf_size': self.leaf_size,
                'morton_bits': self.morton_bits,
                'oversized_leaves': int((leaves & (self.stops-self.starts > self.leaf_size)).sum()),
                'retained_bytes': sum(t.numel()*t.element_size() for t in self.tensors()),
                'physical_qualification': False}


def _binary_lattice_centers(origin, extent, coordinates, depths):
    """Select actual centers on native global power-of-two grids.

    All centers fit within 51 integer bits on one finest grid, so their
    ordered differences are exactly representable native float64 values.
    This changes the expansion centers themselves, never coefficient keys
    or physical supports. Downstream spheres enclose the original boxes.
    """
    if not bool(extent > 0):
        raise ValueError('Binary lattice centers require a nonzero partition span')
    mantissa, exponent = torch.frexp(extent)
    root_exponent = exponent.to(torch.int64)-(mantissa == .5).to(torch.int64)
    pitch_exponents = root_exponent-depths-1
    finest = int(pitch_exponents.min())
    if not -1022 <= finest <= 1023:
        raise ValueError('Binary lattice spacing exceeds normal float64 precision')
    # Form fractions before multiplying by extent, avoiding the overflow
    # of an intermediate extent times a large Morton cell index.
    desired = origin+extent*torch.ldexp(coordinates.to(origin.dtype)+.5,-depths[:,None])
    normalized = torch.ldexp(desired,-pitch_exponents[:,None])
    finest_coordinates = torch.ldexp(desired,-pitch_exponents.min())
    if (not bool(torch.isfinite(normalized).all() & torch.isfinite(finest_coordinates).all())
            or bool((finest_coordinates.abs() > 2**51).any())):
        raise ValueError('Binary lattice centers exceed the exact native integer range')
    centers = torch.ldexp(torch.round(normalized),pitch_exponents[:,None])
    return centers, finest


def build_support_octree(lower, upper, *, leaf_size=64, morton_bits=21, partition_points=None,
                         expansion_center_policy='support_box_midpoint_v1'):
    """Build a Morton octree over full supports using linear device workspaces.

    Equal quantized centers terminate together and may exceed ``leaf_size``;
    this is recorded explicitly. No point-pair list is constructed. Node boxes
    enclose every member's entire support, including overlapping supports.
    Geometry gradients are outside the fixed-geometry MoM contract.
    Optional interior partition_points choose Morton keys only; complete
    support boxes still define every node and admissibility test.
    """
    if type(leaf_size) is not int or leaf_size < 1:
        raise ValueError('leaf_size must be a positive integer')
    if type(morton_bits) is not int or not 1 <= morton_bits <= 21:
        raise ValueError('morton_bits must be an integer in [1, 21]')
    if expansion_center_policy not in ('support_box_midpoint_v1','dyadic_cell_v1','binary_lattice_v1'):
        raise ValueError('Unsupported expansion center policy')
    if (not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor)
            or lower.device.type != 'cuda' or upper.device != lower.device
            or lower.dtype not in (torch.float32, torch.float64) or upper.dtype != lower.dtype
            or lower.ndim != 2 or lower.shape[1] != 3 or lower.shape[0] < 1
            or upper.shape != lower.shape):
        raise ValueError('Support boxes require matching nonempty CUDA float32/64 [N,3] tensors')
    if expansion_center_policy in ('dyadic_cell_v1','binary_lattice_v1') and lower.dtype != torch.float64:
        raise ValueError('Dyadic expansion centers require native float64 geometry')
    if lower.requires_grad or upper.requires_grad:
        raise ValueError('Support octree requires fixed geometry')
    if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all() & (lower <= upper).all()):
        raise ValueError('Support boxes must be finite and ordered')
    lower, upper = lower.clone(), upper.clone()
    if partition_points is None:
        centers = lower*.5+upper*.5
    else:
        if (not isinstance(partition_points,torch.Tensor) or partition_points.device!=lower.device
                or partition_points.dtype!=lower.dtype or partition_points.shape!=lower.shape
                or partition_points.requires_grad or not bool(torch.isfinite(partition_points).all())
                or not bool(((partition_points>=lower)&(partition_points<=upper)).all())):
            raise ValueError('Morton partition points must be fixed matching CUDA points inside support boxes')
        centers = partition_points.clone()
    origin = centers.amin(dim=0)
    span = centers.amax(dim=0)-origin
    if not bool(torch.isfinite(span).all()):
        raise ValueError('Support box coordinate range exceeds native precision')
    # A common scale gives cubic Morton cells in physical coordinates.
    # Per-axis normalization would distort slender/disconnected geometries.
    extent = span.max()
    scale = torch.where(extent > 0, extent, torch.ones_like(extent))
    quantized = (((centers-origin)/scale).clamp(0, 1)*((1 << morton_bits)-1)).to(torch.int64)
    keys = torch.zeros(len(lower), device=lower.device, dtype=torch.int64)
    for bit in range(morton_bits):
        for axis in range(3):
            keys |= ((quantized[:, axis] >> bit) & 1) << (3*bit+axis)
    permutation = torch.argsort(keys, stable=True)
    keys, lower, upper = keys[permutation], lower[permutation], upper[permutation]
    device = lower.device
    n = len(lower)
    starts = torch.zeros(1, device=device, dtype=torch.int64)
    stops = torch.full_like(starts, n)
    lo, hi = lower.amin(dim=0, keepdim=True), upper.amax(dim=0, keepdim=True)
    prefixes = torch.zeros_like(starts)
    branch = (stops-starts > leaf_size) & (keys[starts] != keys[stops-1])
    all_starts, all_stops, all_lo, all_hi = [starts], [stops], [lo], [hi]
    all_depth = [torch.zeros_like(starts)]
    all_prefixes = [prefixes] if expansion_center_policy in ('dyadic_cell_v1','binary_lattice_v1') else None
    all_children = [torch.full((1, 8), -1, device=device, dtype=torch.int64)]
    count = 1
    for depth in range(1, morton_bits+1):
        if not bool(branch.any()):
            break
        point_prefix = keys >> (3*(morton_bits-depth))
        unique, inverse, counts = torch.unique_consecutive(point_prefix, return_inverse=True, return_counts=True)
        group_stops = counts.cumsum(0)
        group_starts = group_stops-counts
        parents = torch.searchsorted(prefixes, unique >> 3)
        valid = parents < len(prefixes)
        safe = parents.clamp_max(len(prefixes)-1)
        valid &= (prefixes[safe] == (unique >> 3)) & branch[safe]
        selected = valid.nonzero().flatten()
        if not len(selected):
            break
        bounds_lo = torch.full((len(unique), 3), torch.inf, device=device, dtype=lower.dtype)
        bounds_hi = torch.full_like(bounds_lo, -torch.inf)
        indices = inverse[:, None].expand(-1, 3)
        bounds_lo.scatter_reduce_(0, indices, lower, reduce='amin', include_self=True)
        bounds_hi.scatter_reduce_(0, indices, upper, reduce='amax', include_self=True)
        starts, stops = group_starts[selected], group_stops[selected]
        lo, hi = bounds_lo[selected], bounds_hi[selected]
        ids = torch.arange(count, count+len(selected), device=device)
        all_children[-1][safe[selected], unique[selected] & 7] = ids
        prefixes = unique[selected]
        branch = (stops-starts > leaf_size) & (keys[starts] != keys[stops-1])
        all_starts.append(starts)
        all_stops.append(stops)
        all_lo.append(lo)
        all_hi.append(hi)
        all_depth.append(torch.full_like(starts, depth))
        if all_prefixes is not None:
            all_prefixes.append(prefixes)
        all_children.append(torch.full((len(selected), 8), -1, device=device, dtype=torch.int64))
        count += len(selected)
    depths = torch.cat(all_depth)
    expansion_centers = None
    binary_lattice_exponent = None
    if all_prefixes is not None:
        codes = torch.cat(all_prefixes)
        coordinates = torch.zeros((count,3),device=device,dtype=torch.int64)
        for bit in range(morton_bits):
            for axis in range(3):
                coordinates[:,axis] |= ((codes >> (3*bit+axis)) & 1) << bit
        # Native world centers are explicitly retained. Reconstructing them
        # from rounded symmetric support bounds would destroy exact sharing.
        if expansion_center_policy == 'binary_lattice_v1':
            if bool(extent > 0):
                expansion_centers, binary_lattice_exponent = _binary_lattice_centers(origin,extent,coordinates,depths)
            else:
                # Equal partition points still have complete finite supports.
                # Preserve their single physical leaf and choose its binary
                # center from the original support enclosure, not key jitter.
                assert count == 1
                support_extent = (hi[0]-lo[0]).amax()
                expansion_centers, binary_lattice_exponent = _binary_lattice_centers(lo[0],support_extent,coordinates,depths)
        else:
            expansion_centers = origin+torch.ldexp(extent*(coordinates.to(lower.dtype)+.5),-depths[:,None])
    return SupportOctree(permutation, torch.cat(all_starts), torch.cat(all_stops),
        torch.cat(all_lo), torch.cat(all_hi), torch.cat(all_children), depths,
        leaf_size, morton_bits, expansion_centers, expansion_center_policy, binary_lattice_exponent)


def support_spheres(tree, nodes=None):
    """Complete support enclosure about the selected fixed expansion centers."""
    tree.assert_current()
    lower,upper = (tree.lower,tree.upper) if nodes is None else (tree.lower[nodes],tree.upper[nodes])
    if tree.expansion_centers is None:
        return lower*.5+upper*.5, torch.linalg.vector_norm((upper-lower)*.5,dim=-1)
    centers = tree.expansion_centers.clone() if nodes is None else tree.expansion_centers[nodes]
    corners = torch.maximum((lower-centers).abs(),(upper-centers).abs())
    radii = torch.linalg.vector_norm(corners,dim=-1)
    # One outward ULP guards the native enclosing norm without dropping any
    # physical support or weakening the same separation/truncation gate.
    radii = torch.nextafter(radii,torch.full_like(radii,torch.inf))
    if not bool(torch.isfinite(radii).all()):
        raise ValueError('Expansion center support radius exceeds native precision')
    return centers,radii


def support_block_partition(test, source=None, *, admissibility=1., pair_batch=1024):
    """Yield (test-node IDs, source-node IDs, admissible flags) on CUDA.

    Strong admissibility is max(diameter) <= eta * support-box distance.
    Touching boxes always descend to exact leaf blocks. Depth-first batches
    bound pending pair workspace by tree depth and ``pair_batch``, even when
    every pair is near; the caller must consume blocks instead of retaining a
    global pair list. Each ordered point pair is covered exactly once.
    """
    import math
    if type(admissibility) not in (float, int) or not math.isfinite(admissibility) or admissibility <= 0:
        raise ValueError('admissibility must be positive and finite')
    if type(pair_batch) is not int or pair_batch < 1:
        raise ValueError('pair_batch must be a positive integer')
    yield from _block_partition(test, source, admissibility, pair_batch, sphere=False)


def sphere_block_partition(test, source=None, *, separation_ratio=.5, pair_batch=1024):
    """Partition complete support spheres for Cartesian multipole/local expansion.

    Far pairs require (r_test+r_source)/center_distance <= separation_ratio < 1.
    Failed separation descends; terminal leaf pairs use original exact near
    integration. This geometric criterion does not certify truncation error.
    """
    import math
    if (type(separation_ratio) not in (int, float) or not math.isfinite(separation_ratio)
            or not 0 < separation_ratio < 1):
        raise ValueError('separation_ratio must lie strictly between zero and one')
    if type(pair_batch) is not int or pair_batch < 1:
        raise ValueError('pair_batch must be a positive integer')
    yield from _block_partition(test, source, separation_ratio, pair_batch, sphere=True)


def _block_partition(test, source, threshold, pair_batch, *, sphere):
    source = test if source is None else source
    test.assert_current()
    source.assert_current()
    if test.device != source.device or test.lower.dtype != source.lower.dtype:
        raise ValueError('Block trees must share CUDA device and precision')
    root = torch.zeros((1, 2), device=test.device, dtype=torch.int64)
    pending = [root]
    while pending:
        test.assert_current()
        source.assert_current()
        pairs = pending.pop()
        a, b = pairs.unbind(dim=1)
        if sphere:
            ca,da = support_spheres(test,a)
            cb,db = support_spheres(source,b)
            distance = torch.linalg.vector_norm(ca-cb, dim=1)
            finite = torch.isfinite(distance) & torch.isfinite(da) & torch.isfinite(db)
            if not bool(finite.all()):
                raise ValueError('Multipole support sphere separation exceeds native precision')
            admissible = (distance > 0) & (da+db <= threshold*distance)
        else:
            gap = torch.maximum(test.lower[a]-source.upper[b], source.lower[b]-test.upper[a]).clamp_min(0)
            distance = torch.linalg.vector_norm(gap, dim=1)
            da = torch.linalg.vector_norm(test.upper[a]-test.lower[a], dim=1)
            db = torch.linalg.vector_norm(source.upper[b]-source.lower[b], dim=1)
            admissible = (distance > 0) & (torch.maximum(da, db) <= threshold*distance)
        ac, bc = test.children[a], source.children[b]
        al, bl = (ac < 0).all(dim=1), (bc < 0).all(dim=1)
        terminal = admissible | (al & bl)
        if bool(terminal.any()):
            yield a[terminal], b[terminal], admissible[terminal]
        active = ~terminal
        if not bool(active.any()):
            continue
        split_a = ~al[active] & (bl[active] | (da[active] >= db[active]))
        children = torch.where(split_a[:, None], ac[active], bc[active])
        aa = torch.where(split_a[:, None], children, a[active, None].expand(-1, 8))
        bb = torch.where(split_a[:, None], b[active, None].expand(-1, 8), children)
        next_pairs = torch.stack((aa, bb), dim=-1)[children >= 0]
        # Reverse push preserves deterministic depth-first order.
        pending.extend(reversed(next_pairs.split(pair_batch)))

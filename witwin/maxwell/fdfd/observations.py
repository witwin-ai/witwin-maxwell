"""CUDA Yee field, voltage and vacuum Huygens observation compilation."""
from __future__ import annotations
import itertools
import math
import torch
from ..fdfd_response import FarFieldProjection
from .materials import compile_fdfd_materials


def _flat(scene, component, indices):
    axis = 'xyz'.index(component[1].lower())
    shape = tuple(getattr(scene, f'N{a}_{component.lower()}') for a in 'xyz')
    offset = (0, scene.N_ex, scene.N_ex + scene.N_ey)[axis]
    return offset + (indices[...,0]*shape[1]+indices[...,1])*shape[2]+indices[...,2]


def gap_layout(scene, start, end):
    """Return electric indices, signed lengths, and transverse dual area."""
    axis = next(i for i in range(3) if start[i] != end[i])
    nodes = [getattr(scene, a) for a in 'xyz']
    if any(min(s,e) < float(c[0]) or max(s,e) > float(c[-1]) for s,e,c in zip(start,end,nodes)):
        raise ValueError("Gap path lies outside the Yee grid.")
    first = [int((c-s).abs().argmin()) for c,s in zip(nodes,start)]
    last = int((nodes[axis]-end[axis]).abs().argmin())
    lo, hi = sorted((first[axis], last))
    if lo == hi:
        raise ValueError("Gap endpoints snap to the same Yee node.")
    coordinates = torch.tensor(first, device=scene.device).repeat(hi-lo,1)
    coordinates[:,axis] = torch.arange(lo,hi, device=scene.device)
    length = torch.as_tensor(getattr(scene, f'd{"xyz"[axis]}_primal64'), device=scene.device)
    lengths = length[lo:hi] * (1 if last > first[axis] else -1)
    area = 1.
    for transverse in range(3):
        if transverse != axis:
            area *= float(getattr(scene, f'd{"xyz"[transverse]}_dual64')[first[transverse]])
    return _flat(scene, 'E'+'xyz'[axis], coordinates), lengths, area


def field_stencil(scene, component, points, frequency, mu_samples=None):
    """Sparse rows sampling E or curl-derived H by native-grid interpolation."""
    own = 'xyz'.index(component[1].lower())
    magnetic = component.startswith('H')
    if magnetic and mu_samples is None:
        _, permeability = compile_fdfd_materials(scene, frequency, edges=True)
        mu_samples = {'H' + axis: value for axis, value in permeability.items()}
    coordinates = [getattr(scene, a+('_half' if (i != own if magnetic else i == own) else ''))
                   for i,a in enumerate('xyz')]
    lower, fractions = [], []
    for axis, c in enumerate(coordinates):
        if bool(((points[:,axis] < c[0]) | (points[:,axis] > c[-1])).any()):
            raise ValueError(f"{component} observation lies outside its native grid.")
        if c.numel() == 1:
            lower.append(torch.zeros(points.shape[0],device=points.device,dtype=torch.int64))
            fractions.append(torch.zeros_like(points[:,axis]))
        else:
            lo = (torch.searchsorted(c.contiguous(), points[:,axis].contiguous())-1).clamp(0,c.numel()-2)
            lower.append(lo)
            fractions.append((points[:,axis]-c[lo])/(c[lo+1]-c[lo]))
    indices, weights = [], []
    for corner in itertools.product((0,1), repeat=3):
        ijk = torch.stack([lower[a]+(corner[a] if coordinates[a].numel()>1 else 0) for a in range(3)],dim=1)
        w = torch.ones(points.shape[0],device=points.device,dtype=points.dtype)
        for a in range(3):
            w = w * (fractions[a] if corner[a] else 1-fractions[a])
        if not magnetic:
            indices.append(_flat(scene, component, ijk)); weights.append(w.to(torch.complex128))
            continue
        # H = curl(E)/(i omega mu0), for exp(-i omega t).
        for derivative, electric, sign in (((own+1)%3,(own+2)%3,1),((own+2)%3,(own+1)%3,-1)):
            advanced = ijk.clone(); advanced[:,derivative] += 1
            spacing = torch.as_tensor(getattr(scene, f'd{"xyz"[derivative]}_primal64'),device=points.device)
            # Observations are physical: refuse PML samples rather than silently
            # exporting stretched-coordinate magnetic fields as SI fields.
            axis = 'xyz'[derivative]
            nlow = scene.pml_thickness_for_face(axis,'low')
            nhigh = scene.pml_thickness_for_face(axis,'high')
            active = w != 0
            if bool((active & ((ijk[:,derivative] < nlow) | (ijk[:,derivative] >= spacing.numel()-nhigh))).any()):
                raise ValueError("Magnetic observations must lie outside PML cells.")
            mu = mu_samples[component][ijk[:,0],ijk[:,1],ijk[:,2]]
            if bool((mu == 0).any()):
                raise ValueError("Magnetic observation requires nonzero permeability.")
            factor = w * sign / (1j * 2*math.pi*frequency * (4*math.pi*1e-7) * mu * spacing[ijk[:,derivative]])
            indices.extend((_flat(scene,'E'+'xyz'[electric],advanced),_flat(scene,'E'+'xyz'[electric],ijk)))
            weights.extend((factor,-factor))
    return torch.stack(indices,dim=1), torch.stack(weights,dim=1)


def _farfield_terms(spec, scene, frequency, mu_samples=None):
    nodes = [getattr(scene,a) for a in 'xyz']
    limits = []
    for axis,(c,pair) in enumerate(zip(nodes,spec.bounds)):
        bounds = scene.domain.bounds[axis]
        if pair[0] <= bounds[0] or pair[1] >= bounds[1]:
            raise ValueError("Huygens box must lie strictly inside the physical domain.")
        lo,hi = (int((c-v).abs().argmin()) for v in pair)
        if hi <= lo:
            raise ValueError("Huygens box must span at least one cell per axis.")
        limits.append((lo,hi))
    direction = torch.tensor(spec.direction, dtype=torch.float64,device=scene.device)
    polarization = torch.tensor(spec.polarization,dtype=torch.float64,device=scene.device)
    k = 2*math.pi*frequency/299792458.
    eta = 4*math.pi*1e-7*299792458.
    for normal in range(3):
        tangents = [a for a in range(3) if a != normal]
        lengths = [torch.as_tensor(getattr(scene,f'd{"xyz"[a]}_primal64'),device=scene.device)
                   [slice(*limits[a])] for a in tangents]
        halves = [getattr(scene,'xyz'[a]+'_half')[slice(*limits[a])].double() for a in tangents]
        aa,bb = torch.meshgrid(*halves,indexing='ij')
        area = (lengths[0][:,None]*lengths[1][None,:]).flatten()
        for side in (0,1):
            points = torch.empty((area.numel(),3),dtype=torch.float64,device=scene.device)
            points[:,normal] = nodes[normal][limits[normal][side]]
            points[:,tangents[0]] = aa.flatten(); points[:,tangents[1]] = bb.flatten()
            outward = torch.zeros(3,dtype=torch.float64,device=scene.device); outward[normal] = 2*side-1
            phase = 1j*k/(4*math.pi)*torch.exp(-1j*k*(points@direction))*area
            # E_inf = ik/(4pi) integral [eta J_transverse - rhat x M]
            # with J=n x H and M=-n x E.
            for component_axis in range(3):
                basis = torch.eye(3,dtype=torch.float64,device=scene.device)[component_axis]
                eweight = torch.dot(polarization, torch.linalg.cross(direction,torch.linalg.cross(outward,basis)))
                hweight = eta*torch.dot(polarization,torch.linalg.cross(outward,basis))
                for kind, factor in (('E',eweight),('H',hweight)):
                    if float(factor) != 0:
                        indices,weights = field_stencil(scene,kind+'xyz'[component_axis],points,frequency,mu_samples)
                        yield indices.flatten(), (weights * (phase*factor)[:,None]).flatten()


def compile_projection(projections, scene, dtype, frequency):
    mu_samples = None
    if any(isinstance(p,FarFieldProjection) or p.component.startswith('H') for p in projections):
        _, permeability = compile_fdfd_materials(scene, frequency, edges=True)
        mu_samples = {'H' + axis: value for axis, value in permeability.items()}
    rows, columns, values = [], [], []
    for row, projection in enumerate(projections):
        if isinstance(projection, FarFieldProjection):
            terms = list(_farfield_terms(projection,scene,frequency,mu_samples))
        elif projection.path is not None:
            indices, lengths, _ = gap_layout(scene,*projection.path)
            terms = [(indices,-lengths.to(dtype))]
        elif projection.position is not None:
            points = torch.tensor([projection.position],dtype=torch.float64,device=scene.device)
            indices,weights = field_stencil(scene,projection.component,points,frequency,mu_samples)
            terms = [(indices.flatten(),weights.flatten())]
        else:
            indices,weights = projection.indices,projection.weights
            if not isinstance(indices,torch.Tensor) or indices.device != scene.x.device or indices.dtype != torch.int64 or indices.ndim != 1 or not indices.numel():
                raise ValueError("Projection indices must be a nonempty int64 CUDA vector.")
            if bool(((indices<0)|(indices>=scene.N_vector_total)).any()):
                raise ValueError("Projection index is outside the electric field.")
            if not isinstance(weights,torch.Tensor) or weights.device != scene.x.device or weights.shape != indices.shape or weights.requires_grad or not bool(torch.isfinite(weights).all()):
                raise ValueError("Projection weights must be fixed finite CUDA values matching indices.")
            terms = [(indices,weights)]
        for indices,weights in terms:
            rows.append(torch.full_like(indices,row));columns.append(indices);values.append(weights.to(dtype))
    return torch.sparse_coo_tensor(torch.stack((torch.cat(rows),torch.cat(columns))),torch.cat(values).detach().clone(),
                                  (len(projections),scene.N_vector_total),device=scene.device,check_invariants=True).coalesce()


@torch.no_grad()
def validate_farfield_support(projections, solver, *, design=None, epsilon=None, pec=None):
    """Check the vacuum exterior required by the closed-surface functional."""
    from .response import edge_materials
    scene = solver.scene
    if not any(isinstance(p, FarFieldProjection) for p in projections):
        return
    epsilon = edge_materials(scene, solver.frequency) if epsilon is None else epsilon
    _, permeability = compile_fdfd_materials(scene, solver.frequency)
    pec = torch.from_dlpack(solver._ensure_pec_mask()) if pec is None else pec
    for projection in projections:
        if not isinstance(projection, FarFieldProjection):
            continue
        # The exterior must be vacuum and all possible material edits must
        # be enclosed. Validate on the actual staggered discretization.
        interior = []
        for component in ('Ex', 'Ey', 'Ez'):
            shape = tuple(getattr(scene, f'N{a}_{component.lower()}') for a in 'xyz')
            mask = torch.ones(shape, device=scene.x.device, dtype=torch.bool)
            for axis, (name, pair) in enumerate(zip('xyz', projection.bounds)):
                nodes = getattr(scene, name)
                lo, hi = (nodes[(nodes-v).abs().argmin()] for v in pair)
                coordinates = getattr(scene, name+('_half' if name == component[1].lower() else ''))
                view = [1,1,1]; view[axis] = -1
                mask &= ((coordinates > lo) & (coordinates < hi)).reshape(view)
            interior.append(mask.flatten())
        interior = torch.cat(interior)
        if design is not None and bool((~interior[design]).any()):
            raise ValueError("The entire design support must be strictly inside the Huygens box.")
        if bool((pec[~interior] | (epsilon[~interior] != 1)).any()):
            raise ValueError("Huygens surface and exterior must be vacuum and enclose all scatterers.")
        node_interior = torch.ones((scene.Nx,scene.Ny,scene.Nz),device=scene.x.device,dtype=torch.bool)
        for axis,(name,pair) in enumerate(zip('xyz',projection.bounds)):
            nodes = getattr(scene,name)
            lo,hi = (nodes[(nodes-v).abs().argmin()] for v in pair)
            view = [1,1,1];view[axis] = -1
            node_interior &= ((nodes>lo)&(nodes<hi)).reshape(view)
        if any(bool((mu[~node_interior] != 1).any()) for mu in permeability.values()):
            raise ValueError("Huygens surface and exterior must have vacuum permeability.")
        scale = torch.from_dlpack(solver._ensure_symmetrization_scale())
        rhs = torch.from_dlpack(solver._build_source_vector_yee()) / scale
        if bool((rhs[~interior] != 0).any()):
            raise ValueError("Huygens box must enclose the complete discretized source support.")

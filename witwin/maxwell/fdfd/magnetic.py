"""Diagonal-permeability Yee curl assembly on CUDA.

The fast nonmagnetic stencil and this factorized curl form share the same
metric, outer ghost closure, PEC elimination and reciprocal similarity.
"""
from __future__ import annotations
import math
import cupy as cp
from cupyx.scipy import sparse
import torch


def magnetic_samples(components):
    result = {}
    for own,name in enumerate('xyz'):
        value = components[name]
        for axis in range(3):
            if axis != own:
                low,high = [slice(None)]*3,[slice(None)]*3
                low[axis],high[axis] = slice(None,-1),slice(1,None)
                value = (value[tuple(low)]+value[tuple(high)])*.5
        result['H'+name] = value
    return result


def flatten_permeability(components):
    return torch.cat([components[name].flatten() for name in 'xyz'])


def require_fixed_permeability(components, inputs):
    """Reject material parameters requiring an unimplemented magnetic pullback."""
    values = flatten_permeability(components)
    if not inputs or not values.requires_grad:
        return
    real = values.real
    imag = values.imag if values.is_complex() else torch.zeros_like(real)
    weights = torch.sin(torch.arange(values.numel(),device=values.device,dtype=real.dtype)+.19)
    gradients = torch.autograd.grad((weights*(real+.83*imag)).sum(),inputs,
                                    allow_unused=True,retain_graph=True)
    if any(g is not None and bool((g!=0).any()) for g in gradients):
        raise NotImplementedError('FDFD material gradients require fixed permeability; magnetic parameter/geometry gradients are unsupported.')


def assemble_magnetic_operator(solver):
    scene = solver.scene
    main,half = solver._stretched_metrics()
    padded = [cp.concatenate((v[:1],v,v[-1:])) for v in half]
    reference = float(scene.dx_primal64[0] if scene.grid.is_custom else scene.grid.dx)
    counts = (scene.Nx,scene.Ny,scene.Nz)
    hshapes = [tuple(n if axis==own else n+1 for axis,n in enumerate(counts)) for own in range(3)]
    hoffsets = (0, math.prod(hshapes[0]),
                math.prod(hshapes[0])+math.prod(hshapes[1]))
    hcount = sum(math.prod(s) for s in hshapes)
    rows,columns,entries = [],[],[]
    for own,component in enumerate(('Ex','Ey','Ez')):
        shape = tuple(getattr(scene,f'N{axis}_{component.lower()}') for axis in 'xyz')
        position = cp.indices(shape,dtype=cp.int32).reshape(3,-1)
        offset = (0,scene.N_ex,scene.N_ex+scene.N_ey)[own]
        column = cp.arange(math.prod(shape),dtype=cp.int32)+offset
        for derivative,output,sign in (((own+1)%3,(own+2)%3,-1),((own+2)%3,(own+1)%3,1)):
            for side in (0,1):
                target = position.copy()
                target[own] += 1
                target[derivative] += side
                hs = hshapes[output]
                row = hoffsets[output]+(target[0]*hs[1]+target[1])*hs[2]+target[2]
                value = sign*(1 if side==0 else -1)/(reference*padded[derivative][target[derivative]])
                rows.append(row);columns.append(column);entries.append(value)
    curl = sparse.coo_matrix((cp.concatenate(entries),(cp.concatenate(rows),cp.concatenate(columns))),
                              shape=(hcount,scene.N_vector_total)).tocsr().astype(solver._iteration_dtype())
    from ..compiler.materials import _pmc_occupancy
    pmc = _pmc_occupancy(scene)
    weights = []
    for own,name in enumerate('xyz'):
        permeability = cp.asarray(solver.material_mu_edges[name].detach()).astype(solver._iteration_dtype())
        permeability = cp.pad(permeability,tuple((0,0) if axis==own else (1,1) for axis in range(3)),mode='edge')
        if not bool(cp.isfinite(permeability).all()) or bool((permeability==0).any()):
            raise ValueError('Diagonal permeability must be finite and nonzero on the magnetic Yee grid.')
        volume = 1
        for axis in range(3):
            view = [1,1,1];view[axis] = -1
            volume = volume*(main[axis] if axis==own else padded[axis]).reshape(view)
        weight = volume/permeability
        if pmc is not None:
            # PMC: a covered H dual edge carries no magnetic flux, the dual of the
            # PEC E-edge elimination. Same four-node staircase test as FDTD.
            covered = magnetic_samples({axis: pmc for axis in 'xyz'})['H'+name] >= .5
            open_fraction = cp.asarray((~covered).to(torch.float32)).astype(solver._iteration_dtype())
            weight = weight*cp.pad(open_fraction,tuple((0,0) if axis==own else (1,1) for axis in range(3)),mode='edge')
        weights.append(weight.flatten())
    scale = solver._ensure_symmetrization_scale()
    scaled = curl@sparse.diags(1/scale)
    from .response import edge_materials
    epsilon = cp.asarray(edge_materials(scene,solver.frequency).detach()).astype(solver._iteration_dtype())
    matrix = sparse.diags(solver.k0**2*epsilon)-scaled.T@sparse.diags(cp.concatenate(weights))@scaled
    matrix = matrix.tocoo()
    mask = solver._ensure_pec_mask()
    keep = ~(mask[matrix.row]|mask[matrix.col])
    constrained = cp.flatnonzero(mask).astype(cp.int32)
    return sparse.coo_matrix((cp.concatenate((matrix.data[keep],cp.ones(constrained.size,dtype=matrix.dtype))),
                              (cp.concatenate((matrix.row[keep],constrained)),cp.concatenate((matrix.col[keep],constrained)))),
                             shape=matrix.shape).tocsr()

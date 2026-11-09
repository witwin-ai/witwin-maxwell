"""Native GPU selected background inverse and local design solve.

This is an independent implementation of the local resolvent equations; no
third-party optimizer code is included. All numerical arrays stay on CUDA.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import time
import numpy as np
import torch

from ..fdfd_response import LocalResponseConfig
from ..result import Result
from ..scene import SceneModule, prepare_scene
from .observations import compile_projection
from .materials import compile_fdfd_materials


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value):
        return (type(value), {f.name: _snapshot(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(v) for v in value)
    if hasattr(value, '__dict__'):
        return (type(value), _snapshot(vars(value)))
    return value


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.device == b.device and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _fixed_context(scene):
    return _snapshot((scene.domain, scene.grid, scene.boundary, scene.sources, scene.ports,
                      scene.symmetry, scene.subpixel, scene.device))


def _matches(snapshot, value):
    """Compare a frozen snapshot without copying live GPU tensors or arrays."""
    if isinstance(snapshot, torch.Tensor):
        return isinstance(value, torch.Tensor) and snapshot.device == value.device and snapshot.shape == value.shape and torch.equal(snapshot, value)
    if isinstance(snapshot, np.ndarray):
        return isinstance(value, np.ndarray) and np.array_equal(snapshot, value)
    if is_dataclass(value):
        return isinstance(snapshot, tuple) and len(snapshot) == 2 and snapshot[0] is type(value) and all(_matches(snapshot[1][f.name], getattr(value, f.name)) for f in fields(value))
    if isinstance(value, dict):
        return isinstance(snapshot, dict) and snapshot.keys() == value.keys() and all(_matches(snapshot[k], v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return isinstance(snapshot, tuple) and len(snapshot) == len(value) and all(_matches(a, b) for a, b in zip(snapshot, value))
    if hasattr(value, '__dict__'):
        return isinstance(snapshot, tuple) and len(snapshot) == 2 and snapshot[0] is type(value) and _matches(snapshot[1], vars(value))
    return snapshot == value


def _trainable(value):
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, dict):
        return any(_trainable(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_trainable(v) for v in value)
    if is_dataclass(value):
        return any(_trainable(getattr(value, f.name)) for f in fields(value))
    if hasattr(value, '__dict__') and not isinstance(value, type):
        return _trainable(vars(value))
    return False


def edge_materials(scene, frequency):
    components, _ = compile_fdfd_materials(scene, frequency, edges=True)
    return torch.cat([components[axis].reshape(-1) for axis in 'xyz'])


def _indices(value, count, device):
    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != torch.int64:
        raise ValueError("Indices must be an int64 tensor on the simulation GPU.")
    if value.ndim != 1 or value.numel() == 0 or bool(((value < 0) | (value >= count)).any()):
        raise ValueError("Indices must be a nonempty in-range vector.")
    return value.detach().clone()


def design_indices(region, scene):
    if region.indices is not None:
        indices = _indices(region.indices, scene.N_vector_total, scene.x.device)
        if indices.unique().numel() != indices.numel():
            raise ValueError("Design indices must be unique.")
        return indices.sort().values
    node = torch.ones((scene.Nx, scene.Ny, scene.Nz), dtype=torch.bool, device=scene.device)
    for dimension, (axis, pair) in enumerate(zip('xyz', region.bounds)):
        coordinate = getattr(scene, axis)
        shape = [1, 1, 1]
        shape[dimension] = -1
        node &= ((coordinate >= pair[0]) & (coordinate <= pair[1])).reshape(shape)
    masks = ((node[:-1] | node[1:]), (node[:, :-1] | node[:, 1:]), (node[:, :, :-1] | node[:, :, 1:]))
    mask = torch.cat([(m if c in region.components else torch.zeros_like(m)).flatten()
                      for c, m in zip(('Ex', 'Ey', 'Ez'), masks)])
    indices = mask.nonzero().flatten()
    if not indices.numel():
        raise ValueError("Design envelope contains no Yee degrees of freedom.")
    return indices


def _scalar(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _checked_solve(matrix, rhs, tolerance):
    solution, info = torch.linalg.solve_ex(matrix, rhs)
    if bool((info != 0).any()) or not bool(torch.isfinite(solution).all()):
        raise RuntimeError("Local response matrix is singular or nonfinite.")
    residual = torch.linalg.vector_norm(matrix @ solution - rhs) / torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.real.dtype).tiny)
    if not bool(torch.isfinite(residual)) or _scalar(residual) > tolerance:
        raise RuntimeError(f"Local response residual {_scalar(residual)} exceeds {tolerance}.")
    return solution, _scalar(residual)


class _BackgroundSolve(torch.autograd.Function):
    """Implicit PyTorch adjoint for a frozen physical background operator."""
    @staticmethod
    def forward(ctx, response, rhs):
        ctx.response = response
        return response.solver.solve_physical_rhs(rhs, tol=response.config.tolerance)

    @staticmethod
    def backward(ctx, gradient):
        response = ctx.response
        # A^-H g = conjugate(D S^-1 D^-1 conjugate(g)). Express the
        # transpose action through the checked physical solve A^-1.
        gradient = gradient.masked_fill(response._base_pec, 0)
        d = torch.from_dlpack(response.solver._ensure_symmetrization_scale())
        try:
            solved = response.solver.solve_physical_rhs(
                (gradient.conj() / d.square()).resolve_conj(), tol=response.config.tolerance)
        finally:
            if response._closed:
                response.solver._release_direct_solver()
        return None, (d.square() * solved).conj()


class PreparedLocalResponse:
    """Reusable frozen-background response, obtained from Simulation.prepare()."""
    @torch.no_grad()
    def __init__(self, simulation, solver, *, response_cache=None):
        self.simulation = simulation
        self.solver = solver
        self.config = simulation.config.local_response
        if not isinstance(self.config, LocalResponseConfig):
            raise TypeError("Local response configuration is required.")
        if simulation.config.solver.device_memory_limit_bytes is not None:
            raise ValueError("local responses require GPU-resident direct factors; "
                             "device_memory_limit_bytes is not supported")
        solver.gpu_only = True
        self.scene = solver.scene
        self.device = self.scene.x.device
        self.dtype = torch.complex128 if solver.precision == 'double' else torch.complex64
        self.frequency = solver.frequency
        self._context = _fixed_context(self.scene)
        self._public_context = _fixed_context(simulation.scene)
        self._config_context = _snapshot(simulation.config)
        if _trainable((self.scene.sources, self.scene.ports)):
            raise ValueError("Local responses require fixed sources and ports; only material inputs may be trainable.")
        self._indices = design_indices(self.config.design, self.scene)
        self._projection = compile_projection(self.config.observations, self.scene, self.dtype, self.frequency)
        self._names = tuple(p.name for p in self.config.observations)
        self._base_eps = edge_materials(self.scene, self.frequency).detach().to(self.dtype).clone()
        from .magnetic import flatten_permeability
        _, permeability = compile_fdfd_materials(self.scene, self.frequency, edges=True)
        self._base_mu = flatten_permeability(permeability).detach().clone()
        solver._ensure_system_matrix()
        self._base_pec = torch.from_dlpack(solver._ensure_pec_mask()).clone()
        if bool(self._base_pec[self._indices].any()):
            raise ValueError("The reference design support must be free of fixed PEC constraints.")
        n, N, q = self._indices.numel(), self.scene.N_vector_total, len(self._names)
        batch = min(n, self.config.rhs_batch_size)
        byte_size = torch.empty((), dtype=self.dtype).element_size()
        estimated = byte_size * (8*n*n + 2*q*n + 4*N*batch + 4*N + 6*n)
        if estimated > self.config.max_response_bytes:
            raise MemoryError(f"Local response arrays need approximately {estimated} bytes, budget is {self.config.max_response_bytes}; background factors are additional.")
        self._outside = torch.ones(N, device=self.device, dtype=torch.bool)
        self._outside[self._indices] = False
        self._validate_farfield()
        self._identity = torch.eye(n, dtype=self.dtype, device=self.device)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.cache_hit = response_cache is not None
        if self.cache_hit:
            from .response_cache import load_cache
            load_cache(self, response_cache)
        else:
            self._green = torch.empty((n,n),dtype=self.dtype,device=self.device)
            self._observation_green = torch.empty((q,n),dtype=self.dtype,device=self.device)
            max_residual = 0.
            with torch.no_grad():
                for begin in range(0, n, batch):
                    stop = min(begin + batch, n)
                    rhs = torch.zeros((N, stop-begin), dtype=self.dtype, device=self.device)
                    rhs[self._indices[begin:stop], torch.arange(stop-begin, device=self.device)] = 1
                    response = solver.solve_physical_rhs(rhs, tol=self.config.tolerance)
                    self._green[:, begin:stop] = response[self._indices]
                    self._observation_green[:, begin:stop] = torch.sparse.mm(self._projection, response)
                    max_residual = max(max_residual, float(solver.rhs_residuals.max()))
                scale = torch.from_dlpack(solver._ensure_symmetrization_scale())
                self._rhs = (torch.from_dlpack(solver._build_source_vector_yee()) / scale).clone()
                self._background = solver.solve_physical_rhs(self._rhs, tol=self.config.tolerance)
                self._u0 = self._background[self._indices].clone()
                self._y0 = torch.sparse.mm(self._projection, self._background[:, None])[:, 0]
            self.precompute_residual = max(max_residual, float(solver.rhs_residuals.max()))
        torch.cuda.synchronize(self.device)
        self.precompute_seconds = time.perf_counter()-start
        if not self.cache_hit:
            self.original_precompute_seconds = self.precompute_seconds
        self.response_bytes_estimate = estimated
        self._closed = False
        self._generation = 0
        self._accepted_delta = torch.zeros(n, dtype=self.dtype, device=self.device)
        self._accepted_pec = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._inverse = self._identity.clone()
        self._accepted_matrix = self._identity.clone()
        self._accepted_current = torch.zeros_like(self._u0)
        self._accepted_projection = self._y0.clone()
        self._projected_inverse = self._observation_green.clone()
        self._accepted_rhs = torch.zeros_like(self._u0)
        self._update_workspace = {}
        self._accepted_residual = 0.
        self._rebuild_count = 0
        coordinate=torch.linspace(.5,1.5,n,device=self.device,dtype=self._u0.real.dtype)
        self._inverse_probes=torch.stack((coordinate.to(self.dtype),torch.exp(7j*coordinate)),dim=1)
        self._probe_norms=torch.linalg.vector_norm(self._inverse_probes,dim=0)

    @property
    def design_indices(self):
        return self._indices.clone()

    def _validate(self):
        if self._closed:
            raise RuntimeError("Local response context is closed.")
        self.simulation._refresh_scene()
        scene = self.simulation.scene
        context = (scene.domain, scene.grid, scene.boundary, scene.sources, scene.ports,
                   scene.symmetry, scene.subpixel, scene.device)
        if self.frequency != self.simulation.frequency or not _matches(self._public_context, context):
            raise ValueError("Background grid, boundary, source or frequency changed; prepare a new response.")
        if scene.grid.is_auto and not _same(self._context,_fixed_context(prepare_scene(scene))):
            raise ValueError("Background grid, boundary, source or frequency changed; prepare a new response.")
        if not _matches(self._config_context, self.simulation.config):
            raise ValueError("FDFD configuration changed; prepare a new response.")

    def _contrast(self, scene=None, delta=None):
        self._validate()
        if scene is not None and delta is not None:
            raise ValueError("Specify a candidate scene or a contrast, not both.")
        if delta is not None:
            if not isinstance(delta, torch.Tensor) or delta.device != self.device or delta.shape != self._u0.shape:
                raise ValueError("delta must be a CUDA vector matching design_indices.")
            result = delta.to(self.dtype)
        else:
            candidate = self.simulation.scene if scene is None else (scene.to_scene() if isinstance(scene, SceneModule) else scene)
            prepared = prepare_scene(candidate)
            if _trainable((prepared.sources, prepared.ports)):
                raise ValueError("Local response sources and ports must be fixed.")
            for structure in prepared.structures:
                if getattr(structure.material, 'is_pec', False) and _trainable(structure.geometry):
                    raise ValueError("Binary PEC geometry is not differentiable; use finite dielectric materials for gradients.")
            if not _same(self._context, _fixed_context(prepared)):
                raise ValueError("Candidate must preserve grid, boundary, sources and fixed solver context.")
            from .solver import _validate_supported_fdfd_materials
            _validate_supported_fdfd_materials(prepared)
            from .magnetic import flatten_permeability, require_fixed_permeability
            from ..adjoint_inputs import scene_trainable_material_tensors
            _, permeability = compile_fdfd_materials(prepared, self.frequency, edges=True)
            if not torch.equal(flatten_permeability(permeability).detach(),self._base_mu):
                raise ValueError("Local dielectric responses require unchanged background permeability.")
            require_fixed_permeability(permeability,scene_trainable_material_tensors(candidate))
            eps = edge_materials(prepared, self.frequency).to(self.dtype)
            if not torch.equal(eps[self._outside].detach(), self._base_eps[self._outside]):
                raise ValueError("Material changed outside the declared design support.")
            if eps.requires_grad:
                from ..adjoint_inputs import scene_trainable_material_tensors
                inputs = scene_trainable_material_tensors(candidate)
                if inputs and bool(self._outside.any()):
                    exterior = eps[self._outside]
                    weight = torch.sin(torch.arange(exterior.numel(), device=self.device, dtype=exterior.real.dtype) + .37)
                    dependencies = torch.autograd.grad((weight * (exterior.real + .71*exterior.imag)).sum(),
                                                       inputs, allow_unused=True, retain_graph=True)
                    if any(g is not None and bool((g != 0).any()) for g in dependencies):
                        raise ValueError("Trainable material influences the background outside design support.")
            result = self.solver.k0**2 * (eps[self._indices] - self._base_eps[self._indices])
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Material contrast must be finite.")
        return result

    def _validate_farfield(self):
        from .observations import validate_farfield_support
        validate_farfield_support(self.config.observations, self.solver, design=self._indices,
                                  epsilon=self._base_eps, pec=self._base_pec)

    def _design(self, scene=None, delta=None, pec=None):
        contrast = self._contrast(scene, delta)
        if scene is not None or delta is None:
            if pec is not None:
                raise ValueError("Explicit PEC mask requires an explicit delta.")
            candidate = self.simulation.scene if scene is None else (scene.to_scene() if isinstance(scene, SceneModule) else scene)
            prepared = prepare_scene(candidate)
            from ..fdtd.runtime.materials import _staircase_pec_shorted_edges
            shorted = _staircase_pec_shorted_edges(prepared)
            candidate_mask = torch.zeros_like(self._base_pec)
            if shorted is not None:
                candidate_mask = torch.cat([shorted[c].flatten() for c in ('Ex', 'Ey', 'Ez')])
            # Domain-face constraints are fixed by the context contract.
            domain_mask = self._domain_pec_mask()
            candidate_mask |= domain_mask
            if not torch.equal(candidate_mask[self._outside], self._base_pec[self._outside]):
                raise ValueError("PEC changed outside the declared design support.")
            pec = candidate_mask[self._indices]
        elif pec is None:
            pec = torch.zeros_like(contrast, dtype=torch.bool)
        if not isinstance(pec, torch.Tensor) or pec.device != self.device or pec.dtype != torch.bool or pec.shape != contrast.shape:
            raise ValueError("PEC must be a boolean CUDA vector matching the design support.")
        return contrast, pec

    def _domain_pec_mask(self):
        masks = []
        for component in ('Ex', 'Ey', 'Ez'):
            shape = tuple(getattr(self.scene, f'N{axis}_{component.lower()}') for axis in 'xyz')
            mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
            for normal, axis in enumerate('xyz'):
                if component[1].lower() == axis:
                    continue
                for side, index in (('low', 0), ('high', -1)):
                    if self.scene.boundary.face_kind(axis, side) == 'pec':
                        selection = [slice(None)] * 3; selection[normal] = index
                        mask[tuple(selection)] = True
            masks.append(mask.flatten())
        return torch.cat(masks)

    def _rows(self, contrast, pec):
        # Normalized constitutive rows: a*j + b*G*j = b*u0.
        # PEC is exactly a=0,b=1, with no artificial large conductivity.
        # Finite contrasts use a=1,b=delta, including delta=0.
        a = (~pec).to(self.dtype)
        b = torch.where(pec, torch.ones_like(contrast), contrast)
        return a, b

    def run(self, *, scene=None, delta=None, pec=None, full_fields=False):
        contrast, mask = self._design(scene, delta, pec)
        a, b = self._rows(contrast, mask)
        matrix = self._identity * a[:, None] + self._green * b[:, None]
        currents, residual = _checked_solve(matrix, b * self._u0, self.config.tolerance)
        observed = self._y0 - self._observation_green @ currents
        return self._result(currents, observed, contrast, mask, residual, scene, full_fields)

    def _result(self, currents, observed, contrast, pec, residual, scene=None, full_fields=False, _prepared_scene=None, **stats):
        result_scene = self.simulation.scene if scene is None else (scene.to_scene() if isinstance(scene, SceneModule) else scene)
        u = self._u0 - self._green @ currents
        fields = {}
        if full_fields:
            rhs = self._rhs.clone()
            rhs[self._indices] -= currents
            electric = _BackgroundSolve.apply(self, rhs)
            offset = 0
            for component in ('Ex', 'Ey', 'Ez'):
                shape = tuple(getattr(self.scene, f'N{axis}_{component.lower()}') for axis in 'xyz')
                count = int(np.prod(shape))
                fields[component.upper()] = electric[offset:offset+count].reshape(shape)
                offset += count
        return Result(method='fdfd', scene=result_scene,
                      prepared_scene=prepare_scene(result_scene) if _prepared_scene is None else _prepared_scene, frequency=self.frequency, fields=fields,
                      monitors={name: {'value': observed[i], 'frequency': self.frequency, 'kind': 'linear_projection'}
                                for i, name in enumerate(self._names)},
                      solver_stats={'converged': True, 'final_residual': residual,
                                    'local_response': {'design_dofs': self._indices.numel(),
                                     'observation_count': len(self._names), 'precompute_seconds': self.precompute_seconds,
                                     'cache_hit': self.cache_hit, 'original_precompute_seconds': self.original_precompute_seconds,
                                     'precompute_residual': self.precompute_residual,
                                     'response_bytes_estimate': self.response_bytes_estimate, 'device': str(self.device), **stats}},
                      raw_output={'local_field': u, 'projections': observed.clone(), 'contrast': contrast.clone(),
                                  'design_indices': self._indices.clone(), 'pec': pec.clone()})

    @torch.no_grad()
    def trial(self, *, scene=None, delta=None, pec=None):
        """Evaluate a design without changing accepted state or forming an n*n update."""
        contrast, mask = self._design(scene, delta, pec)
        old_a, old_b = self._rows(self._accepted_delta, self._accepted_pec)
        a, b = self._rows(contrast, mask)
        changed = ((a != old_a) | (b != old_b)).nonzero().flatten()
        return self._trial_rows(contrast, mask, changed, a, b, old_a, old_b)

    @torch.no_grad()
    def trial_updates(self, indices, *, delta=None, pec=None):
        """Propose absolute material/PEC values on explicitly selected local rows."""
        self._validate()
        indices = _indices(indices, self._indices.numel(), self.device)
        if indices.unique().numel() != indices.numel():
            raise ValueError("Update indices must be unique.")
        contrast, mask = self._accepted_delta.clone(), self._accepted_pec.clone()
        for value, dtype, target, name in ((delta, self.dtype, contrast, 'delta'), (pec, torch.bool, mask, 'pec')):
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.device != self.device or value.shape != indices.shape or (name == 'pec' and value.dtype != dtype):
                    raise ValueError(f"{name} must be a CUDA vector matching the update indices.")
                if name == 'delta' and not bool(torch.isfinite(value).all()):
                    raise ValueError("Material contrast must be finite.")
                target[indices] = value.to(dtype)
        a, b = self._rows(contrast, mask)
        old_a, old_b = self._rows(self._accepted_delta, self._accepted_pec)
        # Unchanged selected rows are harmless zero updates and avoid compaction.
        return self._trial_rows(contrast, mask, indices, a, b, old_a, old_b)

    def _trial_rows(self, contrast, mask, changed, a, b, old_a, old_b):
        if changed.numel() == 0:
            return LocalResponseTrial(self, self._generation, contrast.clone(), mask.clone(), changed,
                                      None, None, self._accepted_current.clone(), self._accepted_projection.clone(), 0.)
        if changed.numel()<=32:
            from .update_cuda import assemble_update,finish_update
            from .update_solve import solve_update
            v,selected,product,small,forcing,rhs=assemble_update(self,contrast.contiguous(),mask.contiguous(),changed.contiguous())
            correction,residual,inverse=solve_update(small,product,rhs,self.config.tolerance)
            current,projected=finish_update(self,changed,selected,forcing,correction)
            return LocalResponseTrial(self,self._generation,contrast.clone(),mask.clone(),changed,
                                      v,small,current,projected,residual,inverse)
        da, db = (a-old_a)[changed], (b-old_b)[changed]
        # U consists of selected identity columns. Only V and selected inverse
        # columns are materialized; rejected trials never write inverse state.
        v = self._green[changed] * db[:, None]
        v[torch.arange(changed.numel(), device=self.device), changed] += da
        selected = self._inverse[:, changed]
        product = v @ selected
        small = torch.eye(changed.numel(), dtype=self.dtype, device=self.device) + product
        forcing = db * self._u0[changed]
        from .update_solve import solve_update
        correction, residual, small_inverse = solve_update(small, product,
            v @ self._accepted_current + product @ forcing, self.config.tolerance)
        step = forcing - correction
        current = self._accepted_current + selected @ step
        projected = self._accepted_projection - self._projected_inverse[:, changed] @ step
        return LocalResponseTrial(self, self._generation, contrast.clone(), mask.clone(), changed,
                                  v, small, current, projected, residual, small_inverse)

    @torch.no_grad()
    def trial_updates_batch(self, indices, *, delta=None, pec=None):
        """Return same-generation trials for a CUDA (batch, rank) index array."""
        self._validate()
        if not isinstance(indices,torch.Tensor) or indices.device!=self.device or indices.dtype!=torch.int64 or indices.ndim!=2 or not indices.numel():
            raise ValueError('Batch indices must be a nonempty CUDA int64 matrix.')
        count,rank=indices.shape
        ordered=indices.sort(dim=1).values
        valid=((indices>=0)&(indices<self._indices.numel())).all()&(ordered[:,1:]!=ordered[:,:-1]).all()
        if not bool(valid):
            raise ValueError('Batch indices must be unique and in range within each proposal.')
        needed=count*(5*self._indices.numel()*rank+8*rank*rank+6*self._indices.numel())*self._u0.element_size()
        if needed>self.config.max_response_bytes:
            raise MemoryError('Candidate batch exceeds the configured response memory budget.')
        contrast=self._accepted_delta.expand(count,-1).clone()
        masks=self._accepted_pec.expand(count,-1).clone()
        for value,target,name in ((delta,contrast,'delta'),(pec,masks,'pec')):
            if value is not None:
                if not isinstance(value,torch.Tensor) or value.device!=self.device or value.shape!=indices.shape or (name=='pec' and value.dtype!=torch.bool):
                    raise ValueError(f'{name} must match the batch index shape on CUDA.')
                if name=='delta' and not bool(torch.isfinite(value).all()):
                    raise ValueError('Material contrast must be finite.')
                target.scatter_(1,indices,value.to(target.dtype))
        if rank>32:
            return tuple(self.trial_updates(row,delta=None if delta is None else delta[i],
                         pec=None if pec is None else pec[i]) for i,row in enumerate(indices))
        from .update_cuda import batch_update
        return batch_update(self,contrast,masks,indices.contiguous())

    @torch.no_grad()
    def accept(self, trial):
        """Commit one current-generation trial after independent residual checks."""
        return self._accept(trial, materialize=True)

    def _rebuild_state(self, matrix, rhs, b):
        """Eliminate exact identity rows before reconstructing inverse actions."""
        active=(b!=0).nonzero().flatten()
        inverse=self._identity.clone()
        current=torch.zeros_like(rhs)
        if active.numel():
            block=matrix[active][:,active]
            right=-matrix[active].clone()
            right[:,active]=torch.eye(active.numel(),device=self.device,dtype=self.dtype)
            solved,_=_checked_solve(block,torch.cat((right,rhs[active,None]),dim=1),self.config.tolerance)
            inverse[active]=solved[:,:-1]
            current[active]=solved[:,-1]
        residual=torch.linalg.vector_norm(matrix@current-rhs)/torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.real.dtype).tiny)
        value=float(residual)
        if not np.isfinite(value) or value>self.config.tolerance:
            raise RuntimeError(f'Rebuilt local current residual {value} exceeds {self.config.tolerance}.')
        return inverse,current,value

    @torch.no_grad()
    def _accept(self, trial, *, materialize=False):
        self._validate()
        if not isinstance(trial, LocalResponseTrial) or trial._owner is not self or trial._generation != self._generation:
            raise ValueError("Trial belongs to a different context or stale accepted generation.")
        a, b = self._rows(trial._delta, trial._pec)
        changed=trial._changed
        matrix=self._accepted_matrix
        if changed.numel():
            matrix=matrix.clone()
            matrix[changed]=self._identity[changed]*a[changed,None]+self._green[changed]*b[changed,None]
        rhs=b*self._u0
        rebuild=(self._generation+1)%self.config.rebuild_interval==0
        inverse=self._inverse
        projected_inverse=self._projected_inverse
        if changed.numel() and not rebuild:
            solved=trial._small_inverse@(trial._v@inverse)
            inverse=inverse-inverse[:,changed]@solved
            projected_inverse=projected_inverse-projected_inverse[:,changed]@solved
        if rebuild:
            inverse,rebuilt_current,_=self._rebuild_state(matrix,rhs,b)
        # Exact zero forcing has an exact zero solution in a nonsingular system.
        # Avoid dividing cancellation remnants by the smallest normal float.
        current=rebuilt_current if rebuild else torch.where(rhs.abs().amax()==0,torch.zeros_like(rhs),trial._current)
        probes=self._inverse_probes
        defect=(torch.linalg.vector_norm(matrix@(inverse@probes)-probes,dim=0)/self._probe_norms).amax()
        residual=torch.linalg.vector_norm(matrix@current-rhs)/torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.real.dtype).tiny)
        finite=torch.isfinite(current).all()&torch.isfinite(defect)&torch.isfinite(residual)
        # Keep headroom for subsequent rejected trials, whose inverse actions
        # cannot benefit from the acceptance-time correction.
        guard_tolerance=self.config.tolerance/16
        valid=finite&(defect<=guard_tolerance)&(residual<=guard_tolerance)
        valid_host,residual_host=torch.stack((valid,residual)).cpu().tolist()
        if not valid_host:
            inverse,current,residual_host=self._rebuild_state(matrix,rhs,b)
            rebuild=True
        if rebuild or (self._generation+1)%8==0:
            projected_inverse=self._observation_green@inverse
        projected=self._y0-self._observation_green@current
        result=None
        if materialize:
            result=self._result(current,projected,trial._delta,trial._pec,residual_host,
                                generation=self._generation+1,rebuilt=rebuild,_prepared_scene=self.scene)
        self._inverse=inverse.contiguous()
        self._accepted_matrix,self._accepted_rhs=matrix,rhs
        self._accepted_current=current.clone()
        self._accepted_projection=projected
        self._projected_inverse=projected_inverse
        self._accepted_delta,self._accepted_pec=trial._delta.clone(),trial._pec.clone()
        self._generation+=1
        self._accepted_residual=residual_host
        self._rebuild_count+=int(rebuild)
        return result

    def _accepted_result(self):
        self._validate()
        return self._result(self._accepted_current,self._accepted_projection,
            self._accepted_delta,self._accepted_pec,self._accepted_residual,
            generation=self._generation,rebuild_count=self._rebuild_count,_prepared_scene=self.scene)

    def close(self):
        if not self._closed:
            self.solver._release_direct_solver()
            self._closed = True
            self._update_workspace.clear()
            # Autograd retains precisely the tensors needed by live results.
            # Release the large reusable caches when the context is closed.
            for name in ('_green', '_observation_green', '_identity', '_inverse',
                         '_accepted_matrix', '_projected_inverse'):
                setattr(self, name, None)

    def save_cache(self, path):
        """Save immutable background responses; accepted design state is excluded."""
        from .response_cache import save_cache
        return save_cache(self, path)

    def optimize_binary(self, groups, objective, **kwargs):
        from .optimization import optimize_binary
        return optimize_binary(self, groups, objective, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LocalResponseTrial:
    """Candidate with owner/generation checks; public output owns its tensors."""
    def __init__(self, owner, generation, delta, pec, changed, v, small, current, projections, residual, small_inverse=None):
        self._owner, self._generation = owner, generation
        self._delta, self._pec, self._changed = delta, pec, changed
        self._v, self._small = v, small
        self._small_inverse = small_inverse
        self._current, self._projections, self._residual = current, projections, residual

    @property
    def projections(self):
        return self._projections.clone()

    @property
    def rank(self):
        return self._changed.numel()

    def result(self):
        # Full local field recovery is explicit: projecting a trial alone does
        # not pay the O(n^2) Green matvec required for all local field values.
        self._owner._validate()
        return self._owner._result(self._current, self._projections, self._delta, self._pec,
                                   self._residual, residual_scope='small_update_system',
                                   generation=self._generation, trial_rank=self.rank)

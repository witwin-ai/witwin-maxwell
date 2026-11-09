"""Linear-storage exact RWG Gram actions and residual-checked CUDA inverses."""
import math
import torch


class _MetricInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, metric):
        ctx.metric = metric
        return metric._solve(value)

    @staticmethod
    def backward(ctx, gradient):
        # The fixed real symmetric metric has the same inverse adjoint.
        return _MetricInverse.apply(gradient, ctx.metric), None


class LocalRWGRiesz:
    """Impedance-balanced L2 proxy from exact local 3x3 face Gram blocks.

    Global coefficient order is all-body J followed by all-body M. The metric
    is diag(G/|eta|^2,G). No global dense Gram or factor is formed. Its inverse
    uses Jacobi-preconditioned CG and checks the actual residual each iteration;
    failures cannot be treated as valid residual norms by the outer solver.
    This remains the existing L2 proxy, not a certified trace-space norm.
    """
    def __init__(self, system, impedance, *, rtol=None, atol=0., maxiter=256):
        system.assert_current()
        if isinstance(impedance, torch.Tensor) and impedance.requires_grad:
            raise ValueError('Local Riesz impedance must be fixed')
        eta = abs(complex(impedance))
        if not math.isfinite(eta) or eta <= 0 or not math.isfinite(eta**2):
            raise ValueError('Riesz impedance must have positive finite squared magnitude')
        if type(maxiter) is not int or maxiter < 1:
            raise ValueError('Riesz maxiter must be a positive integer')
        rtol = (1e-12 if system.dtype == torch.float64 else 1e-5) if rtol is None else rtol
        for name, value in (('rtol', rtol), ('atol', atol)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(name+' must be finite and nonnegative')
        if rtol == 0 and atol == 0:
            raise ValueError('Riesz inverse requires a positive stopping tolerance')
        self.system = system
        self._system = system
        self._basis_bindings = tuple(body.basis for body in system.bodies)
        self._source_versions = tuple((id(t), t._version) for t in self.source_tensors())
        self.device = system.device
        self.dtype = torch.complex128 if system.dtype == torch.float64 else torch.complex64
        self.shape = (2*system.total_edges,)*2
        self.rtol, self.atol, self.maxiter = rtol, atol, maxiter
        self.impedance_squared = eta**2
        self._configuration = (rtol, atol, maxiter, eta**2)
        records, diagonals = [], []
        for body in system.bodies:
            blocks = body.basis.mass_blocks()
            if blocks.requires_grad:
                raise ValueError('Local Riesz metric requires fixed geometry')
            indices = body.basis.face_edges.clone()
            diagonal = torch.zeros(body.basis.n_edges, device=self.device, dtype=system.dtype)
            diagonal.scatter_add_(0, indices.flatten(), blocks.diagonal(dim1=1, dim2=2).flatten())
            if not bool(torch.isfinite(blocks).all() & torch.isfinite(diagonal).all() & (diagonal > 0).all()):
                raise ValueError('RWG metric has invalid local blocks or diagonal')
            records.append((body.edge_offset, body.basis.n_edges, indices, blocks.to(self.dtype)))
            diagonals.append(diagonal)
        gram_diagonal = torch.cat(diagonals)
        self.diagonal = torch.cat((gram_diagonal/self.impedance_squared, gram_diagonal))
        self.records = tuple(records)
        self._layout = tuple(record[:2] for record in self.records)
        self._versions = tuple((id(t), t._version) for t in self.tensors())
        self.last_solve = None

    def source_tensors(self):
        return tuple(t for body in self.system.bodies for t in
            (body.mesh.vertices, body.mesh.faces, body.basis.edges, body.basis.face_edges,
             body.basis.face_signs, body.basis.edge_lengths))

    def tensors(self):
        return tuple(t for record in self.records for t in record[2:])+(self.diagonal,)

    def assert_current(self):
        if self.system is not self._system:
            raise ValueError('Riesz system binding changed')
        self.system.assert_current()
        if (len(self._basis_bindings) != len(self.system.bodies) or any(
                body.basis is not basis for body, basis in zip(self.system.bodies, self._basis_bindings))):
            raise ValueError('Riesz source basis binding changed')
        if tuple((id(t), t._version) for t in self.source_tensors()) != self._source_versions:
            raise ValueError('Riesz source basis changed after construction')
        if (self.rtol, self.atol, self.maxiter, self.impedance_squared) != self._configuration:
            raise ValueError('Riesz configuration changed after construction')
        if tuple(record[:2] for record in self.records) != self._layout:
            raise ValueError('Riesz block layout changed')
        if tuple((id(t), t._version) for t in self.tensors()) != self._versions:
            raise ValueError('Riesz tensors changed after construction')

    def _validate(self, value):
        self.assert_current()
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.dtype != self.dtype or value.device != self.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Riesz vectors require finite matching complex CUDA coefficient space')

    def apply(self, value):
        self._validate(value)
        result = torch.zeros_like(value)
        total = self.shape[0]//2
        for start, count, indices, blocks in self.records:
            for offset, scale in ((start, 1/self.impedance_squared), (start+total, 1.)):
                local = torch.einsum('tij,...tj->...ti', blocks, value[..., offset:offset+count][..., indices])
                scatter_indices = indices.flatten().expand(*value.shape[:-1], -1)
                result[..., offset:offset+count].scatter_add_(-1, scatter_indices, local.flatten(-2)*scale)
        return result

    def inverse(self, value):
        self._validate(value)
        return _MetricInverse.apply(value, self)

    def _solve(self, value):
        self.last_solve = None
        self._validate(value)
        rhs = value.reshape(-1, self.shape[0])
        x = torch.zeros_like(rhs)
        residual = rhs.clone()
        norm_b = torch.linalg.vector_norm(rhs, dim=-1)
        threshold = torch.maximum(norm_b*self.rtol, torch.full_like(norm_b, self.atol))
        active = norm_b > threshold
        z = residual/self.diagonal
        direction = torch.where(active[:, None], z, 0)
        rho = (residual.conj()*z).sum(-1).real
        norm = norm_b
        for iteration in range(self.maxiter+1):
            if not bool(active.any()):
                self.last_solve = {'version': 'local_rwg_pcg_v1', 'converged': True,
                    'iterations': iteration, 'maximum_absolute_residual': float(norm.max()) if norm.numel() else 0.,
                    'maximum_relative_residual': float((norm/norm_b.clamp_min(torch.finfo(norm.dtype).tiny)).max()) if norm.numel() else 0.,
                    'rtol': self.rtol, 'atol': self.atol, 'actual_residual_checked': True}
                return x.reshape_as(value)
            if iteration == self.maxiter:
                break
            image = self.apply(direction)
            denominator = (direction.conj()*image).sum(-1).real
            if not bool(torch.isfinite(denominator[active]).all() & (denominator[active] > 0).all()):
                raise RuntimeError('Riesz PCG encountered nonpositive or nonfinite curvature')
            safe = torch.where(active, denominator, 1.)
            alpha = torch.where(active, rho/safe, 0.)
            x = x+alpha[:, None]*direction
            residual = rhs-self.apply(x)
            norm = torch.linalg.vector_norm(residual, dim=-1)
            if not bool(torch.isfinite(norm).all()):
                raise RuntimeError('Riesz PCG residual became nonfinite')
            active = norm > threshold
            z = residual/self.diagonal
            new_rho = (residual.conj()*z).sum(-1).real
            beta = new_rho/torch.where(rho > 0, rho, 1.)
            direction = torch.where(active[:, None], z+beta[:, None]*direction, 0.)
            rho = new_rho
        self.last_solve = {'version': 'local_rwg_pcg_v1', 'converged': False,
            'iterations': self.maxiter, 'maximum_absolute_residual': float(norm.max()),
            'maximum_relative_residual': float((norm/norm_b.clamp_min(torch.finfo(norm.dtype).tiny)).max()),
            'rtol': self.rtol, 'atol': self.atol, 'actual_residual_checked': True}
        raise RuntimeError(f'Riesz inverse did not converge in {self.maxiter} iterations: relative residual {self.last_solve["maximum_relative_residual"]:.6g}')

    def receipt(self):
        self.assert_current()
        return {'version': 'local_rwg_riesz_v1', 'shape': list(self.shape),
            'retained_bytes': sum(t.numel()*t.element_size() for t in self.tensors()),
            'impedance_squared': self.impedance_squared, 'rtol': self.rtol, 'atol': self.atol,
            'maxiter': self.maxiter, 'metric': 'impedance_balanced_L2_dual_proxy',
            'trace_norm_certified': False}

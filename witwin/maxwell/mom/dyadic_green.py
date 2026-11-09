"""Fourier Maxwell resolvent for homogeneous electric and magnetic tensors.

The symbol uses exp(i q.x - i omega t), electric fields E, and impedance-scaled
magnetic fields eta0 H. This module is a spectral building block, not a spatial
Green quadrature or a qualified anisotropic surface discretization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections import OrderedDict
from threading import RLock
import torch
from ..constants import EPSILON_0, MU_0


def cross_matrix(vector):
    """Return the bilinear cross-product matrix, with no complex conjugation."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def maxwell_symbol(wavevectors, epsilon_r, mu_r, frequency):
    """Dimensionless 6x6 symbol acting on [E, eta0 H].

    [eps, C(q/k0); C(q/k0), -mu] [E; eta0 H]
      = [-i eta0 J/k0; i M/k0].
    Complex wavevectors are allowed for explicit contour evaluations. Radiation
    prescriptions belong to the spatial quadrature, never to a hidden damping.
    """
    if (not isinstance(wavevectors, torch.Tensor) or wavevectors.device.type != 'cuda'
            or wavevectors.ndim < 1 or wavevectors.shape[-1] != 3
            or wavevectors.dtype not in (torch.complex64, torch.complex128)):
        raise ValueError('Spectral wavevectors must be complex CUDA [...,3] tensors')
    for name, tensor in (('epsilon_r', epsilon_r), ('mu_r', mu_r)):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (3, 3)
                or tensor.device != wavevectors.device or tensor.dtype != wavevectors.dtype
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f'{name} must be a finite matching CUDA complex 3x3 tensor')
    if not bool(torch.isfinite(wavevectors).all()):
        raise ValueError('Spectral wavevectors must be finite')
    if isinstance(frequency, bool) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('frequency must be finite and positive in Hz')
    k0 = 2*math.pi*frequency*math.sqrt(EPSILON_0*MU_0)
    curl = cross_matrix(wavevectors/k0)
    eps = epsilon_r.expand_as(curl)
    mu = mu_r.expand_as(curl)
    return torch.cat((torch.cat((eps, curl), -1), torch.cat((curl, -mu), -1)), -2)


def maxwell_resolvent(wavevectors, epsilon_r, mu_r, frequency):
    """Invert an explicitly specified spectral contour, without regularization."""
    symbol = maxwell_symbol(wavevectors, epsilon_r, mu_r, frequency)
    identity = torch.eye(6, device=symbol.device, dtype=symbol.dtype).expand_as(symbol)
    inverse, info = torch.linalg.solve_ex(symbol, identity)
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse).all()):
        raise ValueError('Singular Maxwell symbol: specify a valid radiation contour or physical loss')
    return inverse


def current_green_spectrum(wavevectors, medium):
    """Physical Fourier mapping [J, M] to [E, H] for an owned tensor sample.

    Maxwell source signs are curl E = i omega mu H - M and
    curl H = -i omega epsilon E + J. Fourier normalization (2*pi)^-3 and
    the outgoing contour are the spatial integrator's explicit obligations.
    """
    medium.assert_current()
    inverse = maxwell_resolvent(wavevectors, medium.epsilon_r, medium.mu_r, medium.frequency)
    eta0 = math.sqrt(MU_0/EPSILON_0)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    source_scale = inverse.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
    field_scale = inverse.new_tensor([1.]*3+[1/eta0]*3)
    return field_scale[:, None]*inverse*source_scale[None, :]


def current_green_spectrum_difference(wavevectors, exterior, interior):
    """Return G_interior-G_exterior without subtracting two full resolvents.

    The resolvent identity retains the constitutive contrast as an explicit
    factor, including at exactly zero contrast with differentiable inputs.
    Both symbols use the same explicitly supplied contour and frequency.
    This spectral primitive is not yet a spatial lifted-carrier quadrature.
    """
    exterior.assert_current()
    interior.assert_current()
    if (exterior.frequency != interior.frequency
            or exterior.epsilon_r.device != interior.epsilon_r.device
            or exterior.epsilon_r.dtype != interior.epsilon_r.dtype):
        raise ValueError('Spectral contrast requires common frequency, device and precision')
    outside = maxwell_resolvent(wavevectors, exterior.epsilon_r, exterior.mu_r, exterior.frequency)
    inside = maxwell_resolvent(wavevectors, interior.epsilon_r, interior.mu_r, interior.frequency)
    contrast = torch.block_diag(interior.epsilon_r-exterior.epsilon_r,
                                exterior.mu_r-interior.mu_r)
    difference = -(inside@contrast@outside)
    eta0 = math.sqrt(MU_0/EPSILON_0)
    k0 = exterior.omega*math.sqrt(EPSILON_0*MU_0)
    source_scale = difference.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
    field_scale = difference.new_tensor([1.]*3+[1/eta0]*3)
    return field_scale[:, None]*difference*source_scale[None, :]


def _plane_eigensystem(generator, transverse, medium):
    """Use checked analytic TE/TM columns for fixed exact scalar media."""
    from .spectral_linalg import batched_eig4
    eye = torch.eye(3, device=medium.epsilon_r.device, dtype=medium.epsilon_r.dtype)
    epsilon, mu = medium.epsilon_r[0, 0], medium.mu_r[0, 0]
    if (medium.epsilon_r.requires_grad or medium.mu_r.requires_grad
            or not torch.equal(medium.epsilon_r, epsilon*eye)
            or not torch.equal(medium.mu_r, mu*eye) or bool(mu == 0) or len(transverse) == 0):
        from .matched_spd_green import matched_spd_plane_modes,proportional_spd_plane_modes
        matched=matched_spd_plane_modes(generator,transverse,medium)
        if matched is None:matched=proportional_spd_plane_modes(generator,transverse,medium)
        return batched_eig4(generator) if matched is None else matched
    transverse_complex = transverse.to(generator.dtype)
    length = torch.linalg.vector_norm(transverse, dim=-1)
    safe_length = length.clamp_min(torch.finfo(length.dtype).tiny)
    te = torch.stack((-transverse[:, 1]/safe_length, transverse[:, 0]/safe_length,
                      torch.zeros_like(length)), -1).to(generator.dtype)
    te = torch.where((length == 0)[:, None], te.new_tensor([1., 0., 0.]), te)
    pole = torch.sqrt(epsilon*mu-transverse.square().sum(-1))
    roots = torch.stack((pole, pole, -pole, -pole), -1)
    vectors = []
    for sign in (1, -1):
        q = torch.cat((transverse_complex, (sign*pole)[:, None]), -1)
        tm = torch.linalg.cross(q, te)
        # q x (q x TE) = -epsilon*mu*TE. Evaluating the second
        # cross product subtracts O(|q_transverse|^2) terms in the
        # evanescent tail and needlessly destroys the TM magnetic field.
        for electric, magnetic in ((te, tm/mu), (tm, -epsilon*te)):
            vectors.append(torch.cat((electric[:, :2], magnetic[:, :2]), -1))
    vectors = torch.stack(vectors, -1)
    norms = torch.linalg.vector_norm(vectors, dim=-2)
    vectors = vectors/norms.clamp_min(torch.finfo(length.dtype).tiny)[:, None, :]
    residual = torch.linalg.vector_norm(generator@vectors-vectors*roots[:, None, :], dim=-2)
    scale = torch.linalg.matrix_norm(generator, ord='fro')[:, None]+roots.abs()
    failed = ((residual > 512*torch.finfo(length.dtype).eps*scale).any(-1)
              | ~torch.isfinite(vectors).all(dim=(-2, -1)) | (norms == 0).any(-1) | (pole == 0))
    analytic_fallback_count = int(failed.sum())
    fallback_count = 0
    if analytic_fallback_count:
        r, v, receipt = batched_eig4(generator[failed])
        fallback_count = receipt['fallback_count']
        roots = roots.clone()
        vectors = vectors.clone()
        roots[failed], vectors[failed] = r, v
    return roots, vectors, {'version': 'analytic_scalar_te_tm_checked_v1', 'matrix_count': len(generator),
        'fallback_count': fallback_count, 'analytic_fallback_count': analytic_fallback_count,
        'residual_certifies_eigenvalue_error': False}


def compile_plane_modes(transverse, medium, *, pole_relative_tolerance=None):
    """Compile the outgoing q_z modal residues independently of observation points.

    transverse contains dimensionless [q_x/k0,q_y/k0]. The returned physical
    Green tensor still requires the transverse integral d q_x d q_y/(2*pi)^2.
    Longitudinal contact distributions at z=0 are excluded, not approximated.
    No radiation damping is introduced: decaying modes use Im(q_z)>0, and real
    propagating modes use positive longitudinal Poynting flux.
    """
    if pole_relative_tolerance is not None and (type(pole_relative_tolerance) not in (int,float)
            or not math.isfinite(pole_relative_tolerance) or pole_relative_tolerance<=0):
        raise ValueError('Plane pole tolerance must be finite and positive')
    medium.assert_current()
    if (not isinstance(transverse, torch.Tensor) or transverse.device != medium.epsilon_r.device
            or transverse.dtype != medium.epsilon_r.real.dtype or transverse.ndim != 2
            or transverse.shape[1] != 2 or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Transverse coordinates must be finite matching real CUDA [N,2] tensors')
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    q = torch.cat((transverse, torch.zeros_like(transverse[:, :1])), -1).to(medium.epsilon_r.dtype)
    symbol = maxwell_symbol(k0*q, medium.epsilon_r, medium.mu_r, medium.frequency)
    # Tangential [Ex,Ey,eta0 Hx,eta0 Hy], longitudinal [Ez,eta0 Hz].
    tangential = torch.tensor([0, 1, 3, 4], device=q.device)
    longitudinal = torch.tensor([2, 5], device=q.device)
    tt = symbol[:, tangential[:, None], tangential]
    tl = symbol[:, tangential[:, None], longitudinal]
    lt = symbol[:, longitudinal[:, None], tangential]
    ll = symbol[:, longitudinal[:, None], longitudinal]
    inverse_ll, info = torch.linalg.inv_ex(ll)
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse_ll).all()):
        raise ValueError('Longitudinal constitutive block is singular for this integration axis')
    reduced = tt-tl@inverse_ll@lt
    cz = cross_matrix(q.new_tensor([0., 0., 1.]))
    zero = torch.zeros_like(cz)
    derivative = torch.cat((torch.cat((zero, cz), -1), torch.cat((cz, zero), -1)), -2)
    derivative = derivative[tangential[:, None], tangential]
    generator = -torch.linalg.solve(derivative, reduced)
    roots, vectors, eigensolver = _plane_eigensystem(generator, transverse, medium)
    tolerance = 128*torch.finfo(transverse.dtype).eps*torch.linalg.matrix_norm(generator, ord=2)
    flux = (vectors[:, 0]*vectors[:, 3].conj()-vectors[:, 1]*vectors[:, 2].conj()).real
    near_real = roots.imag.abs() <= tolerance[:, None]
    outgoing = (roots.imag > tolerance[:, None]) | (near_real & (flux > 128*torch.finfo(transverse.dtype).eps))
    if bool((outgoing.sum(-1) != 2).any()):
        raise ValueError('Outgoing mode split is unresolved at a grazing or defective spectral point')
    if pole_relative_tolerance is not None:
        # Opposite radiation sheets must remain spectrally distinguishable.
        # Same-sheet polarization degeneracy is allowed. Roots are q_z/k0,
        # so the unit floor makes this a relative physical-wavenumber test.
        separation=(roots[:,:,None]-roots[:,None,:]).abs()
        scale=torch.maximum(roots.abs()[:,:,None],roots.abs()[:,None,:]).clamp_min(1.)
        opposite=outgoing[:,:,None]!=outgoing[:,None,:]
        relative=torch.where(opposite,separation/scale,torch.full_like(separation,float('inf')))
        if bool((relative.amin(dim=(-2,-1))<=pole_relative_tolerance).any()):
            raise ValueError('Outgoing/incoming Maxwell sheets are unresolved at the configured pole tolerance')
    identity4 = torch.eye(4, device=q.device, dtype=q.dtype).expand(len(q), 4, 4)
    embedding = torch.cat((identity4, -inverse_ll@lt), -2)
    forcing = torch.cat((identity4, -tl@inverse_ll), -1)
    coefficients, info = torch.linalg.solve_ex(vectors, torch.linalg.solve(derivative, forcing))
    if bool((info != 0).any()) or not bool(torch.isfinite(coefficients).all()):
        raise ValueError('Defective Maxwell eigenspace requires a contour-projector integration')
    selected = outgoing.to(torch.int64).argsort(dim=-1, descending=True)[:, :2]
    poles = roots.gather(1, selected)
    right = (embedding@vectors).gather(2, selected[:, None, :].expand(-1, 6, -1)).transpose(-1, -2)
    left = coefficients.gather(1, selected[:, :, None].expand(-1, -1, 6))
    residues = 1j*k0*right[..., :, None]*left[..., None, :]
    reorder = torch.tensor([0, 1, 4, 2, 3, 5], device=q.device)
    residues = residues[:, :, reorder[:, None], reorder]
    eta0 = math.sqrt(MU_0/EPSILON_0)
    source_scale = q.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
    field_scale = q.new_tensor([1.]*3+[1/eta0]*3)
    residues = field_scale[:, None]*residues*source_scale[None, :]
    return PlaneModes(k0*transverse, k0*poles, residues, medium, eigensolver=eigensolver)


@dataclass(frozen=True)
class PlaneModes:
    """Reusable outgoing modal residues for one fixed transverse spectral grid."""
    transverse: torch.Tensor
    longitudinal: torch.Tensor
    residues: torch.Tensor
    medium: object
    eigensolver: dict | None = field(default=None, repr=False)
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_versions', tuple((id(v), v._version) for v in
                           (self.transverse, self.longitudinal, self.residues)))

    def _assert_current(self):
        self.medium.assert_current()
        if tuple((id(v), v._version) for v in (self.transverse, self.longitudinal, self.residues)) != self._versions:
            raise ValueError('Compiled plane modes changed; rebuild the spectral grid')

    def spectrum(self, distance):
        self._assert_current()
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError('Outgoing plane residues require a strictly positive separation')
        phase = torch.exp(1j*distance*self.longitudinal)
        return (phase[..., None, None]*self.residues).sum(1)

    def integrate(self, displacements, weights, *, query_batch=128):
        """Integrate many local z>0 queries with no repeated eigenproblem.

        weights include the physical transverse measure /(2*pi)^2. Only the
        phase array is batched; a [queries,nodes,6,6] Green array is never built.
        """
        self._assert_current()
        if (not isinstance(displacements, torch.Tensor) or displacements.device != self.transverse.device
                or displacements.dtype != self.transverse.dtype or displacements.ndim != 2
                or displacements.shape[1] != 3 or not bool(torch.isfinite(displacements).all())
                or bool((displacements[:, 2] <= 0).any())):
            raise ValueError('Modal queries require finite matching CUDA real [N,3] with positive z')
        if (not isinstance(weights, torch.Tensor) or weights.shape != (len(self.transverse),)
                or weights.device != self.transverse.device or weights.dtype != self.transverse.dtype
                or not bool(torch.isfinite(weights).all())):
            raise ValueError('Spectral weights must match the real CUDA transverse grid')
        if type(query_batch) is not int or query_batch < 1:
            raise ValueError('query_batch must be a positive integer')
        coefficients = (self.residues*weights[:, None, None, None]).reshape(-1, 36)
        output = []
        for points in displacements.split(query_batch):
            phase_argument = ((points[:, :2]@self.transverse.T)[..., None]
                              +points[:, 2, None, None]*self.longitudinal[None])
            phase = torch.exp(1j*phase_argument).reshape(len(points), -1)
            output.append((phase@coefficients).reshape(-1, 6, 6))
        return torch.cat(output) if output else self.residues.new_empty((0, 6, 6))


def outgoing_plane_spectrum(transverse, distance, medium):
    """Evaluate positive-z residues; repeated queries should reuse PlaneModes."""
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Outgoing plane residues require a strictly positive separation')
    return compile_plane_modes(transverse, medium).spectrum(distance)


def _validate_factored_plane_medium(medium):
    medium.assert_current()
    for tensor in (medium.epsilon_r, medium.mu_r):
        if (tensor.dtype != torch.complex128 or tensor.requires_grad
                or bool((tensor.imag != 0).any()) or not torch.equal(tensor, tensor.mT)):
            raise ValueError('Factored planes require fixed exact real symmetric tensors')
        _, info = torch.linalg.cholesky_ex(tensor.real)
        if bool(info != 0):
            raise ValueError('Factored planes require strictly positive definite tensors')
    return 2*torch.sqrt(torch.linalg.matrix_norm(medium.epsilon_r, ord=2)
                        *torch.linalg.matrix_norm(medium.mu_r, ord=2))


def _factored_plane_polynomial_components(transverse, medium, length, x):
    """Compile exact inverse-length coefficients for admitted real-SPD media.

    The caller validates the fixed constitutive sample and evanescent domain.
    Returned intermediates belong to compilation workspace; compiled grids
    separately own and inventory their retained state. Both single and paired
    constructions share these original coefficient operations.
    """
    u = transverse/length[:, None]
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    rotation = torch.stack((u[:, 0], -u[:, 1], zero, u[:, 1], u[:, 0], zero,
                            zero, zero, one), -1).reshape(-1, 3, 3).to(medium.epsilon_r.dtype)
    eps = rotation.mT@medium.epsilon_r@rotation
    mu = rotation.mT@medium.mu_r@rotation
    material = torch.cat((torch.cat((eps, torch.zeros_like(eps)), -1),
                          torch.cat((torch.zeros_like(mu), -mu), -1)), -2)
    curl = cross_matrix(eps.new_tensor([1., 0., 0.]))
    empty = torch.zeros_like(curl)
    wave = torch.cat((torch.cat((empty, curl), -1), torch.cat((curl, empty), -1)), -2)
    t = torch.tensor([0, 1, 3, 4], device=eps.device)
    l = torch.tensor([2, 5], device=eps.device)
    tt = material[:, t[:, None], t]
    b0, c0 = material[:, t[:, None], l], material[:, l[:, None], t]
    b1, c1 = wave[t[:, None], l].expand_as(b0), wave[l[:, None], t].expand_as(c0)
    inverse, info = torch.linalg.inv_ex(material[:, l[:, None], l])
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse).all()):
        raise ValueError('Singular longitudinal block in factored plane')
    cz = cross_matrix(eps.new_tensor([0., 0., 1.]))
    derivative = torch.cat((torch.cat((empty, cz), -1), torch.cat((cz, empty), -1)), -2)[t[:, None], t]
    coefficients = [
        -torch.linalg.solve(derivative, tt-b0@inverse@c0),
        torch.linalg.solve(derivative, b1@inverse@c0+b0@inverse@c1),
        torch.linalg.solve(derivative, b1@inverse@c1)]
    powers = torch.tensor([1, 0, 1, 0], device=eps.device)
    g = [torch.zeros_like(coefficients[0]) for _ in range(3)]
    for degree, coefficient in enumerate(coefficients):
        exponent = degree-1+powers[None, :]-powers[:, None]
        if bool((coefficient[:, exponent > 0] != 0).any()):
            raise ValueError('Unexpected positive power in balanced plane generator')
        for order in range(3):
            g[order] = g[order]+torch.where(exponent == -order, coefficient, 0.)
    return eps, rotation, material, inverse, b0, c0, b1, c1, derivative, powers, coefficients, g


def _factored_plane_state_factors(components, x, identity):
    """Build original polynomial field/source factors from shared components."""
    (eps, rotation, material, inverse, b0, c0, b1, c1, derivative, powers,
     coefficients, g) = components
    embedding0 = eps.new_zeros((len(x), 6, 4))
    embedding0[:, :4] = identity
    ec = -inverse@c0
    ew = -inverse@c1
    embedding0[:, 4:] = torch.where(powers[None, None, :] == 1, ec, ew)
    embedding_delta = eps.new_zeros((len(x), 6, 4))
    embedding_delta[:, 4:] = torch.where(powers[None, None, :] == 0, x[:, None, None]*ec, 0.)
    embedding = embedding0+embedding_delta
    full_powers = torch.tensor([1, 0, 1, 0, 1, 1], device=eps.device)
    forcing_coefficients = [
        torch.linalg.solve(derivative, torch.cat((identity, -b0@inverse), -1)),
        torch.linalg.solve(derivative, torch.cat((torch.zeros_like(identity), -b1@inverse), -1))]
    f = [eps.new_zeros((len(x), 4, 6)) for _ in range(3)]
    for degree, coefficient in enumerate(forcing_coefficients):
        exponent = degree-1+full_powers[None, :]-powers[:, None]
        if bool((coefficient[:, exponent > 0] != 0).any()):
            raise ValueError('Unexpected positive power in balanced plane forcing')
        for order in range(3):
            f[order] = f[order]+torch.where(exponent == -order, coefficient, 0.)
    forcing0 = f[0]
    forcing_delta = x[:, None, None]*f[1]+x.square()[:, None, None]*f[2]
    forcing = forcing0+forcing_delta
    return embedding, embedding0, embedding_delta, forcing, forcing0, forcing_delta, full_powers


def _compile_factored_evanescent_plane(transverse, medium):
    """Own a fixed real-SPD grid and compile its distance-independent matrices."""
    medium.assert_current()
    if (not isinstance(transverse, torch.Tensor) or transverse.device != medium.epsilon_r.device
            or transverse.dtype != torch.float64 or transverse.ndim != 2
            or transverse.shape[1] != 2 or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Factored planes require matching finite CUDA float64 [N,2] coordinates')
    bound = _validate_factored_plane_medium(medium)
    transverse = transverse.clone()
    length = torch.linalg.vector_norm(transverse, dim=-1)
    if not bool((length > bound).all()):
        raise ValueError('Factored planes require transverse coordinates beyond the evanescent bound')
    x = length.reciprocal()
    (eps, rotation, material, inverse, b0, c0, b1, c1, derivative, powers,
     coefficients, g) = _factored_plane_polynomial_components(transverse, medium, length, x)
    delta = x[:, None, None]*g[1]+x.square()[:, None, None]*g[2]
    a0, a = g[0], g[0]+delta
    identity = torch.eye(4, device=eps.device, dtype=eps.dtype).expand_as(a)
    # Coupled Newton iterates retain sign(A)-sign(A0) as an explicit factor.
    sx, s0, ds = 1j*a, 1j*a0, 1j*delta
    tolerance = 256*torch.finfo(x.dtype).eps
    iterations = 0
    for iterations in range(1, 33):
        ix, info_x = torch.linalg.inv_ex(sx)
        i0, info_0 = torch.linalg.inv_ex(s0)
        if bool((info_x != 0).any() | (info_0 != 0).any()):
            raise ValueError('Singular iteration in evanescent plane projector')
        ds = .5*(ds-ix@ds@i0)
        sx, s0 = .5*(sx+ix), .5*(s0+i0)
        residual = torch.maximum((sx@sx-identity).norm(dim=(-2, -1)),
                                 (s0@s0-identity).norm(dim=(-2, -1)))
        if bool((residual <= tolerance).all()):
            break
    else:
        raise ValueError('Evanescent plane projector did not converge')
    p, p0, dp = .5*(identity-sx), .5*(identity-s0), -.5*ds
    if (not bool(torch.isfinite(p).all() & torch.isfinite(dp).all())
            or not bool(((p.diagonal(dim1=-2, dim2=-1).sum(-1)-2).abs() <= tolerance).all())):
        raise ValueError('Evanescent plane projector is unresolved')
    (embedding, embedding0, embedding_delta, forcing, forcing0, forcing_delta,
     full_powers) = _factored_plane_state_factors(
         (eps, rotation, material, inverse, b0, c0, b1, c1, derivative, powers, coefficients, g), x, identity)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    factors = torch.where(full_powers[None, :] == 1, length[:, None], 1.)
    physical_rotation = eps.new_zeros((len(x), 6, 6))
    physical_rotation[:, :3, :3] = rotation
    physical_rotation[:, 3:, 3:] = rotation
    state = dict(a=a, a0=a0, delta=delta, p=p, p0=p0, dp=dp,
                 embedding=embedding, embedding0=embedding0, embedding_delta=embedding_delta,
                 forcing=forcing, forcing_delta=forcing_delta, length=length,
                 factors=factors, physical_rotation=physical_rotation)
    if not all(bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('Nonfinite compiled evanescent plane matrices')
    return _FactoredPlaneGrid(k0*transverse, medium, state, iterations)


def _evaluate_factored_plane_state(state, distance, k0):
    """Propagate compiled matrices with the original factored channel algebra."""
    a, a0, delta = state['a'], state['a0'], state['delta']
    p, p0, dp = state['p'], state['p0'], state['dp']
    embedding, embedding0, embedding_delta = state['embedding'], state['embedding0'], state['embedding_delta']
    forcing, forcing_delta = state['forcing'], state['forcing_delta']
    length, factors, physical_rotation = state['length'], state['factors'], state['physical_rotation']
    phase = 1j*k0*distance*length[:, None, None]
    identity = torch.eye(4, device=a.device, dtype=a.dtype)
    unused_scale = torch.maximum(a.norm(dim=(-2, -1)), a0.norm(dim=(-2, -1)))[:, None, None]
    unused_exponent = k0*distance*length[:, None, None]*unused_scale
    # The incoming sector is annihilated by P. Choosing a negative exponent
    # there preserves exp(i*t*A)P exactly and suppresses roundoff leakage at
    # large t; outgoing eigenvalues and constitutive data remain unchanged.
    k = phase*a@p-unused_exponent*(identity-p)
    k_base = phase*a0@p0-unused_exponent*(identity-p0)
    dk = phase*(delta@p+a0@dp)+unused_exponent*dp
    block = a.new_zeros((len(length), 8, 8))
    block[:, :4, :4], block[:, :4, 4:], block[:, 4:, 4:] = k, dk, k_base
    exponential = torch.linalg.matrix_exp(block)
    ex, ex0, dex = exponential[:, :4, :4], exponential[:, 4:, 4:], exponential[:, :4, 4:]
    propagator, propagator0 = ex@p, ex0@p0
    propagator_delta = dex@p+ex0@dp
    difference = (embedding_delta@propagator@forcing
                  +embedding0@propagator_delta@forcing
                  +embedding0@propagator0@forcing_delta)
    # The leading cross blocks are physical curl terms, not static Coulomb
    # singularities. Only EE and HH subtract the x=0 longitudinal baseline.
    complete = embedding@propagator@forcing
    electric = torch.tensor([True, True, False, False, True, False], device=a.device)
    diagonal_block = electric[:, None] == electric[None, :]
    difference = torch.where(diagonal_block, difference, complete)
    difference = 1j*k0*length[:, None, None]*factors[:, :, None]*difference/factors[:, None, :]
    reorder = torch.tensor([0, 1, 4, 2, 3, 5], device=a.device)
    difference = difference[:, reorder[:, None], reorder]
    eta0 = math.sqrt(MU_0/EPSILON_0)
    source = a.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
    field_scale = a.new_tensor([1.]*3+[1/eta0]*3)
    difference = field_scale[:, None]*difference*source[None, :]
    difference = physical_rotation@difference@physical_rotation.mT
    if not bool(torch.isfinite(difference).all()):
        raise ValueError('Nonfinite factored evanescent plane remainder')
    return difference


@dataclass(frozen=True)
class _FactoredPlaneGrid:
    """Immutable compiled tail with bounded per-query propagation workspace."""
    transverse: torch.Tensor
    medium: object
    state: dict = field(repr=False)
    sign_iterations: int
    _versions: tuple = field(init=False, repr=False)
    _keys: tuple = field(init=False, repr=False)

    def tensors(self):
        return (self.transverse, self.medium.epsilon_r, self.medium.mu_r,
                *(self.state[k] for k in sorted(self.state)))

    def __post_init__(self):
        object.__setattr__(self, '_keys', tuple(sorted(self.state)))
        object.__setattr__(self, '_versions', tuple((id(t), t._version, t.requires_grad) for t in self.tensors()))

    def _assert_current(self):
        self.medium.assert_current()
        if (tuple(sorted(self.state)) != self._keys
                or tuple((id(t), t._version, t.requires_grad) for t in self.tensors()) != self._versions):
            raise ValueError('Compiled factored plane matrices changed; rebuild the grid')

    def _validate_distance(self, distance):
        if isinstance(distance, torch.Tensor):
            if (distance.shape != () or distance.device != self.transverse.device
                    or distance.dtype != self.transverse.dtype or not bool(torch.isfinite(distance))
                    or bool(distance <= 0)):
                raise ValueError('Factored planes require a matching positive finite CUDA distance')
        elif isinstance(distance, bool) or not math.isfinite(distance) or distance <= 0:
            raise ValueError('Factored planes require a positive finite distance')

    def _sections(self, distance):
        k0 = self.medium.omega*math.sqrt(EPSILON_0*MU_0)
        for start in range(0, len(self.transverse), 256):
            stop = min(start+256, len(self.transverse))
            state = {k: t[start:stop] for k, t in self.state.items()}
            yield start, stop, _evaluate_factored_plane_state(state, distance, k0)

    def spectrum(self, distance):
        self._assert_current()
        self._validate_distance(distance)
        values = [value for _, _, value in self._sections(distance)]
        return torch.cat(values) if values else self.medium.epsilon_r.new_empty((0, 6, 6))

    def integrate(self, displacements, weights, *, query_batch=128):
        self._assert_current()
        _validate_regular_plane_queries(displacements, weights, self.transverse, query_batch)
        values = []
        for point in displacements.unbind(0):
            value = self.medium.epsilon_r.new_zeros((6, 6))
            for start, stop, spectrum in self._sections(point[2]):
                phase = torch.exp(1j*(self.transverse[start:stop]@point[:2]))*weights[start:stop]
                value = value+torch.einsum('n,nij->ij', phase, spectrum)
            values.append(value)
        return torch.stack(values) if values else self.medium.epsilon_r.new_empty((0, 6, 6))


def _validate_regular_plane_queries(displacements, weights, transverse, query_batch):
    if type(query_batch) is not int or query_batch < 1:
        raise ValueError('query_batch must be a positive integer')
    if (not isinstance(displacements, torch.Tensor) or displacements.device != transverse.device
            or displacements.dtype != transverse.dtype or displacements.ndim != 2
            or displacements.shape[1] != 3 or not bool(torch.isfinite(displacements).all())
            or bool((displacements[:, 2] <= 0).any())):
        raise ValueError('Regular plane queries require matching finite real CUDA [N,3] and positive z')
    if (not isinstance(weights, torch.Tensor) or weights.shape != (len(transverse),)
            or weights.device != transverse.device or weights.dtype != transverse.dtype
            or not bool(torch.isfinite(weights).all())):
        raise ValueError('Regular plane weights must match the real CUDA transverse grid')


def _factored_evanescent_plane_spectrum(transverse, distance, medium):
    """Evaluate a plane; spatial callers reuse the compiled tail instead."""
    grid = _compile_factored_evanescent_plane(transverse, medium)
    return grid.spectrum(distance), dict(version='experimental_factored_evanescent_plane_v2',
        sign_iterations=grid.sign_iterations, constitutive_projection=False,
        static_subtraction_factored=True, spatial_qualification=False)


@dataclass(frozen=True)
class _RegularPlaneGrid:
    """Keep all original nodes, with modal low nodes and a compiled SPD tail."""
    transverse: torch.Tensor
    medium: object
    low_indices: torch.Tensor
    tail_indices: torch.Tensor
    low: object
    static: object
    tail: object
    _versions: tuple = field(init=False, repr=False)

    @property
    def eigensolver(self):
        return None if self.low is None else self.low.eigensolver

    def tensors(self):
        values = [self.transverse, self.medium.epsilon_r, self.medium.mu_r,
                  self.low_indices, self.tail_indices]
        for modes in (self.low, self.static):
            if modes is not None:
                values.extend((modes.transverse, modes.longitudinal, modes.residues,
                               modes.medium.epsilon_r, modes.medium.mu_r))
        if self.tail is not None:
            values.extend(self.tail.tensors())
        return tuple(values)

    def __post_init__(self):
        object.__setattr__(self, '_versions', tuple((id(t), t._version, t.requires_grad) for t in self.tensors()))

    def integrate(self, displacements, weights, *, query_batch=128):
        self.medium.assert_current()
        if tuple((id(t), t._version, t.requires_grad) for t in self.tensors()) != self._versions:
            raise ValueError('Compiled mixed regular plane changed; rebuild the grid')
        _validate_regular_plane_queries(displacements, weights, self.transverse, query_batch)
        value = self.medium.epsilon_r.new_zeros((len(displacements), 6, 6))
        if self.low is not None:
            value = (self.low.integrate(displacements, weights[self.low_indices], query_batch=query_batch)
                     -self.static.integrate(displacements, weights[self.low_indices], query_batch=query_batch))
        if self.tail is not None:
            value = value+self.tail.integrate(displacements, weights[self.tail_indices], query_batch=query_batch)
        return value


def _compile_regular_plane_grid(transverse, medium):
    bound = _validate_factored_plane_medium(medium)
    if (not isinstance(transverse, torch.Tensor) or transverse.device != medium.epsilon_r.device
            or transverse.dtype != torch.float64 or transverse.ndim != 2
            or transverse.shape[1] != 2 or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Regular planes require matching finite CUDA float64 [N,2] coordinates')
    transverse = transverse.clone()
    tail_mask = torch.linalg.vector_norm(transverse, dim=-1) > bound
    tail_indices = torch.nonzero(tail_mask).flatten()
    low_indices = torch.nonzero(~tail_mask).flatten()
    low = compile_plane_modes(transverse[low_indices], medium) if len(low_indices) else None
    static = compile_electrostatic_plane_modes(transverse[low_indices], medium) if len(low_indices) else None
    tail = _compile_factored_evanescent_plane(transverse[tail_indices], medium) if len(tail_indices) else None
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    return _RegularPlaneGrid(k0*transverse, medium, low_indices, tail_indices, low, static, tail)


class SpectralModeCache:
    """LRU of retained CUDA spectral grids, bound to one immutable medium.

    The byte limit covers uniquely owned retained tensor storages, not temporary
    construction workspaces or grids still referenced by an active caller.
    Differentiable material graphs are never retained across evaluations.
    """
    def __init__(self, medium, max_bytes=256*1024**2):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Spectral cache max_bytes must be a nonnegative integer')
        medium.assert_current()
        self.medium, self.max_bytes = medium, max_bytes
        self._lock = RLock()
        self._entries = OrderedDict()
        self._storage_refs = {}
        self.retained_bytes = self.hits = self.misses = self.evictions = self.bypasses = 0
        self.modal_matrices_built = self.native_eigen_fallbacks = 0

    def validate(self, medium):
        if medium is not self.medium:
            raise ValueError('Spectral cache belongs to a different material sample')
        try:
            medium.assert_current()
        except ValueError:
            self.clear()
            raise

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._storage_refs.clear()
            self.retained_bytes = 0

    @staticmethod
    def _tensors(entry):
        modes, static_modes, weights = entry
        tensors = [weights]
        for item in (modes, static_modes):
            if item is not None:
                if isinstance(item, _RegularPlaneGrid):
                    tensors.extend(item.tensors())
                else:
                    tensors.extend((item.transverse, item.longitudinal, item.residues,
                                    item.medium.epsilon_r, item.medium.mu_r))
        return tensors

    def _release(self, storages):
        for key, size in storages.items():
            count = self._storage_refs[key][1]-1
            if count:
                self._storage_refs[key] = (size, count)
            else:
                del self._storage_refs[key]
                self.retained_bytes -= size

    def get(self, key, build, *, retain=True):
        with self._lock:
            return self._get(key, build, retain=retain)

    def _get(self, key, build, *, retain=True):
        self.validate(self.medium)
        if retain and key in self._entries:
            entry, storages, versions = self._entries.pop(key)
            if tuple((id(t), t._version, t.requires_grad) for t in self._tensors(entry)) != versions:
                self._release(storages)
                raise ValueError('Cached spectral tensors changed; rebuild the entry')
            self._entries[key] = (entry, storages, versions)
            self.hits += 1
            return entry
        self.misses += 1
        entry = build()
        eigen_receipt = getattr(entry[0], 'eigensolver', None)
        if eigen_receipt is not None:
            self.modal_matrices_built += eigen_receipt['matrix_count']
            self.native_eigen_fallbacks += eigen_receipt['fallback_count']
        tensors = self._tensors(entry)
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        size = sum(storages.values())
        if not retain or size > self.max_bytes or any(t.requires_grad for t in tensors):
            self.bypasses += 1
            return entry
        def additional_bytes():
            return sum(size for key, size in storages.items() if key not in self._storage_refs)
        while self.retained_bytes+additional_bytes() > self.max_bytes:
            _, (_, old_storages, _) = self._entries.popitem(last=False)
            self._release(old_storages)
            self.evictions += 1
        versions = tuple((id(t), t._version, t.requires_grad) for t in tensors)
        self._entries[key] = (entry, storages, versions)
        for key, size in storages.items():
            if key in self._storage_refs:
                self._storage_refs[key] = (size, self._storage_refs[key][1]+1)
            else:
                self._storage_refs[key] = (size, 1)
                self.retained_bytes += size
        return entry

    def stats(self):
        with self._lock:
            return self._stats()

    def _stats(self):
        return {'entries': len(self._entries), 'retained_cuda_bytes': self.retained_bytes,
                'max_retained_cuda_bytes': self.max_bytes, 'hits': self.hits, 'misses': self.misses,
                'evictions': self.evictions, 'bypasses': self.bypasses,
                'modal_matrices_built': self.modal_matrices_built, 'native_eigen_fallbacks': self.native_eigen_fallbacks,
                'modal_counter_scope': 'cumulative completed cache-miss builds, including bypassed entries',
                'scope': 'unique entry tensor storages; workspace and tensors held only by callers or the bound input excluded'}


def _validate_spatial_cache(cache, medium, exterior):
    if cache is None:
        return
    from .dyadic_contrast import SpectralContrastCache
    if exterior is None:
        if isinstance(cache, SpectralContrastCache):
            raise ValueError('A paired contrast cache cannot serve a single-medium kernel')
        cache.validate(medium)
    else:
        if not isinstance(cache, SpectralContrastCache):
            raise ValueError('Spatial contrast requires a cache bound to both samples')
        cache.validate_pair(exterior, medium)


def _weyl_radial_rule(nodes, weights, k0, reference, exterior=None, *, material_split=None):
    """Map the infinite radial axis, resolving a lossless scalar branch ring."""
    if material_split is not None:
        if exterior is not None:
            raise ValueError('Material-split radial mapping requires a single regional kernel')
        inner=material_split*nodes
        tail=material_split+nodes/(1-nodes)/(k0*reference)
        return (torch.cat((inner,tail)),torch.cat((weights*material_split**2*nodes,
                weights*tail/(1-nodes).square()/(k0*reference))))
    branch = None
    if exterior is not None:
        eye = torch.eye(3, device=exterior.epsilon_r.device, dtype=exterior.epsilon_r.dtype)
        epsilon, mu = exterior.epsilon_r[0, 0], exterior.mu_r[0, 0]
        if (torch.equal(exterior.epsilon_r, epsilon*eye) and torch.equal(exterior.mu_r, mu*eye)
                and bool(epsilon.imag == 0) and bool(mu.imag == 0) and bool((epsilon*mu).real > 0)):
            branch = torch.sqrt((epsilon*mu).real)
    if branch is None:
        rho = nodes/(1-nodes)/(k0*reference)
        return rho, weights*rho/(1-nodes).square()/(k0*reference)
    # Below the branch: rho=kappa*sin(theta); above: rho=kappa*sqrt(1+t^2).
    # The Jacobians cancel the outgoing 1/sqrt(kappa^2-rho^2) singularity.
    theta = nodes*(math.pi/2)
    propagating = branch*torch.sin(theta)
    propagating_weights = weights*branch**2*torch.sin(theta)*torch.cos(theta)*(math.pi/2)
    t = nodes/(1-nodes)/(k0*branch*reference)
    evanescent = branch*torch.sqrt(1+t**2)
    evanescent_weights = weights*branch*t/(k0*reference*(1-nodes).square())
    return torch.cat((propagating, evanescent)), torch.cat((propagating_weights, evanescent_weights))


def integrate_spatial_green(displacements, medium, *, radial_order, azimuthal_order, subtract_static=True, restore_static=True, mode_cache=None, _exterior=None, radial_map="infinite_rational", spatial_rule='native_v1'):
    """Fixed-order off-source Weyl integral on the complete transverse plane.

    This primitive does not certify quadrature convergence. Callers must refine
    both orders and handle panel singularities separately before MoM assembly.
    The radial map rho=s/(1-s)/(k0*z_ref) includes the infinite evanescent tail;
    there is no finite wavevector cutoff and no artificial constitutive loss.
    The static diagonal is subtracted spectrally and restored analytically.
    """
    from .quadrature import gauss_rule
    from .tensor_medium import TensorMedium
    _validate_spatial_cache(mode_cache, medium, _exterior)
    medium.assert_current()
    if spatial_rule not in ('native_v1', 'factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
        raise ValueError('Unsupported fixed-order dyadic spatial rule')
    if spatial_rule == 'factored_spd_tail_v1':
        if _exterior is not None or not subtract_static:
            raise ValueError('Factored SPD tails require single-medium static-subtracted integration')
        _validate_factored_plane_medium(medium)
    if spatial_rule == 'factored_spd_pair_tail_v1':
        if not subtract_static:
            raise ValueError('Factored SPD pairs require static-subtracted integration')
        _validate_factored_plane_medium(medium)
        if _exterior is not None:
            _validate_factored_plane_medium(_exterior)
    if _exterior is not None:
        _exterior.assert_current()
        if (medium.frequency != _exterior.frequency or medium.epsilon_r.device != _exterior.epsilon_r.device
                or medium.epsilon_r.dtype != _exterior.epsilon_r.dtype):
            raise ValueError('Spatial contrast requires common frequency, device and precision')

    if (not isinstance(displacements, torch.Tensor) or displacements.device != medium.epsilon_r.device
            or displacements.dtype != medium.epsilon_r.real.dtype or displacements.ndim != 2
            or displacements.shape[1] != 3 or not bool(torch.isfinite(displacements).all())):
        raise ValueError('Displacements must be finite matching real CUDA [N,3] tensors')
    for order in (radial_order, azimuthal_order):
        if type(order) is not int or order < 2:
            raise ValueError('Both spectral integration orders must be integers >= 2')
    if type(subtract_static) is not bool or type(restore_static) is not bool:
        raise ValueError('Static subtraction/restoration flags must be boolean')
    if not restore_static and not subtract_static:
        raise ValueError('A regular kernel requires static spectral subtraction')
    if radial_map not in ('infinite_rational','material_split_v1','matched_spd_branch_v1','proportional_spd_branch_v1','spd_reference_branch_v1','paired_spd_reference_branch_v1'):
        raise ValueError('Unsupported dyadic radial mapping')
    if radial_map == 'paired_spd_reference_branch_v1' and spatial_rule != 'factored_spd_pair_tail_v1':
        raise ValueError('Paired SPD reference coordinates require the explicit paired tail rule')
    material_split=None
    if radial_map=='material_split_v1':
        if _exterior is not None:
            raise ValueError('Material-split radial mapping requires a single regional kernel')
        # This allocates nodes, not a claimed bound on anisotropic modal poles.
        material_split=2*torch.sqrt(torch.linalg.matrix_norm(medium.epsilon_r,ord=2))*torch.sqrt(
            torch.linalg.matrix_norm(medium.mu_r,ord=2))
        if not bool(torch.isfinite(material_split)) or not bool(material_split>0):
            raise ValueError('Material radial split must be positive and representable')
    radii = torch.linalg.vector_norm(displacements, dim=-1)
    if bool((radii == 0).any()):
        raise ValueError('Coincident dyadic nodes require explicit singular integration')
    nodes, weights = gauss_rule(radial_order, device=displacements.device, dtype=displacements.dtype)
    phi = torch.arange(azimuthal_order, device=displacements.device, dtype=displacements.dtype)*(2*math.pi/azimuthal_order)
    circle = torch.stack((torch.cos(phi), torch.sin(phi)), -1)
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    identity = torch.eye(3, device=displacements.device, dtype=displacements.dtype)
    output = medium.epsilon_r.new_empty((len(displacements), 6, 6))
    dominant = displacements.abs().argmax(-1)
    for axis_index in range(3):
        for sign in (-1, 1):
            group = (dominant == axis_index) & (sign*displacements[:, axis_index] > 0)
            indices = torch.nonzero(group).flatten()
            if not len(indices):
                continue
            z = sign*identity[axis_index]
            x = identity[(axis_index+1) % 3]
            y = torch.linalg.cross(z, x)
            rotation_real = torch.stack((x, y, z), -1)
            rotation = rotation_real.to(medium.epsilon_r.dtype)
            local_points = displacements[indices]@rotation_real
            # Rotated materials are needed only when a spectral grid is built.
            # Reuse within this axis group, without retaining material graphs
            # across evaluations or allocating outside the cache byte budget.
            local = local_exterior = None
            # A shared grid spans at most a factor of two in positive z. Only
            # bin IDs cross to host for scheduling; all geometry stays on CUDA.
            bins = torch.floor(torch.log2(local_points[:, 2])).to(torch.int64)
            transform = torch.block_diag(rotation, rotation)
            for bin_id in torch.unique(bins).tolist():
                selected = bins == bin_id
                points = local_points[selected]
                reference = math.ldexp(1., bin_id)
                key = (axis_index, sign, bin_id, radial_order, azimuthal_order, subtract_static, radial_map)
                if spatial_rule != 'native_v1':
                    key = key+(spatial_rule,)
                def build():
                    nonlocal local, local_exterior
                    if local is None:
                        local = TensorMedium(medium.frequency, rotation.mT@medium.epsilon_r@rotation,
                                             rotation.mT@medium.mu_r@rotation, reciprocal=medium.reciprocal)
                        if _exterior is not None:
                            local_exterior = TensorMedium(_exterior.frequency,
                                rotation.mT@_exterior.epsilon_r@rotation, rotation.mT@_exterior.mu_r@rotation,
                                reciprocal=_exterior.reciprocal)
                    if radial_map == 'paired_spd_reference_branch_v1':
                        from .matched_spd_green import spd_reference_transverse_rule, paired_spd_reference_transverse_rule
                        if local_exterior is None:
                            transverse, spectral_weights = spd_reference_transverse_rule(nodes, weights, k0, reference, local, circle)
                        else:
                            transverse, spectral_weights = paired_spd_reference_transverse_rule(nodes, weights, k0, reference, local_exterior, local, circle)
                    elif radial_map in ('matched_spd_branch_v1','proportional_spd_branch_v1','spd_reference_branch_v1'):
                        from .matched_spd_green import matched_spd_transverse_rule,proportional_spd_transverse_rule,spd_reference_transverse_rule
                        rule={'matched_spd_branch_v1':matched_spd_transverse_rule,
                              'proportional_spd_branch_v1':proportional_spd_transverse_rule,
                              'spd_reference_branch_v1':spd_reference_transverse_rule}[radial_map]
                        if local_exterior is not None:
                            raise ValueError('Matched-SPD branch mapping requires a single regional kernel')
                        transverse,spectral_weights=rule(nodes,weights,k0,reference,local,circle)
                    else:
                        rho, radial_weights = _weyl_radial_rule(nodes, weights, k0, reference, local_exterior, material_split=material_split)
                        transverse = (rho[:, None, None]*circle[None]).reshape(-1, 2)
                        spectral_weights = radial_weights.repeat_interleave(azimuthal_order)*k0**2/(2*math.pi*azimuthal_order)
                    if local_exterior is None:
                        if spatial_rule in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
                            modes, static_modes = _compile_regular_plane_grid(transverse, local), None
                        else:
                            modes = compile_plane_modes(transverse, local)
                            static_modes = compile_electrostatic_plane_modes(transverse, local) if subtract_static else None
                    else:
                        if spatial_rule == 'factored_spd_pair_tail_v1':
                            from .dyadic_contrast import _compile_regular_plane_contrast_grid
                            modes, static_modes = _compile_regular_plane_contrast_grid(transverse, local_exterior, local), None
                        else:
                            from .dyadic_contrast import compile_plane_contrast, compile_electrostatic_plane_contrast
                            modes = compile_plane_contrast(transverse, local_exterior, local)
                            static_modes = compile_electrostatic_plane_contrast(transverse, local_exterior, local) if subtract_static else None
                    return modes, static_modes, spectral_weights
                modes, static_modes, spectral_weights = build() if mode_cache is None else mode_cache.get(key, build)
                value = modes.integrate(points, spectral_weights)
                if static_modes is not None:
                    value = value-static_modes.integrate(points, spectral_weights)
                output[indices[selected]] = transform@value@transform.mT
    if subtract_static and restore_static:
        for offset, kind, vacuum in ((0, 'electric', EPSILON_0), (3, 'magnetic', MU_0)):
            if _exterior is None:
                _, _, hessian = electrostatic_kernel(displacements, medium, constitutive=kind)
            else:
                from .dyadic_contrast import electrostatic_kernel_difference
                _, _, hessian = electrostatic_kernel_difference(displacements, _exterior, medium, constitutive=kind)
            output[:, offset:offset+3, offset:offset+3] = output[:, offset:offset+3, offset:offset+3]+1j*hessian/(medium.omega*vacuum)
    return output


@dataclass(frozen=True)
class DyadicIntegrationConfig:
    radial_order: int = 16
    azimuthal_order: int = 16
    max_radial_order: int = 256
    max_azimuthal_order: int = 128
    relative_tolerance: float = 1e-5
    absolute_tolerance: float = 1e-10
    radial_map: str = "infinite_rational"
    spatial_rule: str = 'native_v1'

    def __post_init__(self):
        if self.spatial_rule not in ('native_v1','proportional_spd_closed_form_v1','factored_spd_tail_v1','factored_spd_pair_tail_v1'):
            raise ValueError('Unsupported dyadic spatial rule')
        if self.radial_map not in ('infinite_rational','material_split_v1','matched_spd_branch_v1','proportional_spd_branch_v1','spd_reference_branch_v1','paired_spd_reference_branch_v1'):
            raise ValueError('Unsupported dyadic radial mapping')
        if self.radial_map == 'paired_spd_reference_branch_v1' and self.spatial_rule != 'factored_spd_pair_tail_v1':
            raise ValueError('Paired SPD reference coordinates require the explicit paired tail rule')
        for name in ('radial_order', 'azimuthal_order', 'max_radial_order', 'max_azimuthal_order'):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f'{name} must be an integer >= 2')
        if (self.max_radial_order < 2*self.radial_order
                or self.max_azimuthal_order < 2*self.azimuthal_order):
            raise ValueError('Each spectral axis needs room for an independent order doubling')
        for value in (self.relative_tolerance, self.absolute_tolerance):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Spectral tolerances must be finite and nonnegative')
        if self.relative_tolerance == self.absolute_tolerance == 0:
            raise ValueError('At least one spectral tolerance must be positive')


    def payload(self):
        """Canonical numerical identity: unchanged default, explicit new mapping."""
        from dataclasses import asdict
        values=asdict(self)
        if self.radial_map=='infinite_rational':values.pop('radial_map')
        if self.spatial_rule=='native_v1':values.pop('spatial_rule')
        return values


class DyadicConvergenceError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def evaluate_spatial_green(displacements, medium, config=DyadicIntegrationConfig(), *, regular_part=False, mode_cache=None, _exterior=None):
    """Refine radial and angular quadrature independently before returning.

    The error norm balances both electric/magnetic source and field units with
    eta0. Two directional successive-order differences must each fit half the
    requested tolerance. This is a numerical estimator, not a certified bound.
    Failure retains its settings, measured errors and evaluated order pairs.
    """
    if not isinstance(config, DyadicIntegrationConfig):
        raise TypeError('config must be DyadicIntegrationConfig')
    if type(regular_part) is not bool:
        raise ValueError('regular_part must be boolean')
    _validate_spatial_cache(mode_cache, medium, _exterior)
    medium.assert_current()
    if config.spatial_rule == 'factored_spd_tail_v1':
        if _exterior is not None:
            raise ValueError('Factored SPD spatial integration does not support regional contrasts')
        _validate_factored_plane_medium(medium)
    if config.spatial_rule == 'factored_spd_pair_tail_v1':
        _validate_factored_plane_medium(medium)
        if _exterior is not None:
            _validate_factored_plane_medium(_exterior)
    receipt = {'kernel_version': 'dyadic_weyl_residue_v1' if _exterior is None else 'dyadic_contrast_factored_plane_v1', 'config': config.payload(),
        'converged': False, 'certified_error_bound': False, 'checks': [], 'evaluated_orders': [],
        'estimator': 'independent_radial_and_azimuthal_successive_order_differences',
        'norm': 'Frobenius diag(I,eta0 I) G diag(I,eta0 I) per displacement',
        'absolute_tolerance_units': 'ohm/m^2', 'contact_distribution_included': False,
        'static_diagonal_singularity_subtracted': True, 'static_diagonal_restored': not regular_part,
        'artificial_loss': 0., 'finite_transverse_cutoff': None,
        'radial_rule': 'lossless_scalar_exterior_branch_split_otherwise_infinite_rational' if _exterior is not None else 'infinite_rational',
        'radial_order_scope': 'per radial interval'}
    if config.radial_map=='material_split_v1':
        receipt.update(kernel_version='dyadic_weyl_residue_material_split_v1',
            radial_rule='material_scale_finite_interval_plus_infinite_rational_tail_v1',
            radial_intervals=2,radial_nodes_per_order=2)
    if config.radial_map=='matched_spd_branch_v1':
        receipt.update(kernel_version='dyadic_weyl_matched_spd_branch_v1',
            radial_rule='angle_dependent_matched_spd_propagating_evanescent_split_v1',
            radial_intervals=2,radial_nodes_per_order=2,
            branch_scope='fixed_exact_matched_real_SPD_relative_epsilon_and_mu')
    if config.radial_map=='proportional_spd_branch_v1':
        receipt.update(kernel_version='dyadic_weyl_proportional_spd_branch_v1',
            radial_rule='angle_dependent_proportional_spd_propagating_evanescent_split_v1',
            radial_intervals=2,radial_nodes_per_order=2,
            branch_scope='fixed_exact_positive_proportional_real_SPD_relative_epsilon_and_mu')
    if config.radial_map=='spd_reference_branch_v1':
        receipt.update(kernel_version='dyadic_weyl_spd_reference_branch_v1',
            radial_rule='reference_metric_propagating_evanescent_coordinates_v1',
            radial_intervals=2,radial_nodes_per_order=2,
            branch_scope='reference_node_placement_only_actual_fixed_real_SPD_material_unchanged',
            exact_modal_branch_claimed=False)
    if config.spatial_rule == 'factored_spd_tail_v1':
        receipt.update(kernel_version='dyadic_weyl_factored_spd_tail_v1',
                       static_subtraction='factored_real_SPD_evanescent_modal_low_v1',
                       tail_propagation_node_batch=256, constitutive_projection=False)
    if config.radial_map == 'paired_spd_reference_branch_v1':
        receipt.update(radial_rule='common_ordered_reference_ellipses_complete_radial_coordinates_v1',
            radial_intervals=2 if _exterior is None else 3,
            radial_nodes_per_order=2 if _exterior is None else 3,
            branch_scope='reference_node_placement_only_actual_fixed_real_SPD_pair_unchanged',
            exact_modal_branch_claimed=False)
    if config.spatial_rule == 'factored_spd_pair_tail_v1':
        receipt.update(kernel_version='dyadic_weyl_factored_spd_pair_capable_tail_v1' if _exterior is None
                       else 'dyadic_contrast_two_factor_spd_tail_v1',
            static_subtraction='factored_real_SPD_evanescent_modal_low_v1' if _exterior is None
                       else 'two_factor_paired_real_SPD_evanescent_modal_low_v1',
            tail_propagation_node_batch=256 if _exterior is None else 64,
            constitutive_projection=False, physical_qualification=False)
    cache = {}
    eta0 = math.sqrt(MU_0/EPSILON_0)

    def evaluate(radial, angular):
        key = (radial, angular)
        if key not in cache:
            try:
                contrast_options = {} if _exterior is None else {'_exterior': _exterior}
                if config.radial_map!='infinite_rational':contrast_options['radial_map']=config.radial_map
                if config.spatial_rule in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
                    contrast_options['spatial_rule'] = config.spatial_rule
                value = integrate_spatial_green(displacements, medium, radial_order=radial, azimuthal_order=angular,
                    restore_static=not regular_part, mode_cache=mode_cache, **contrast_options)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('Nonfinite dyadic spectral integral')
                medium.assert_current()
            except (ValueError, RuntimeError) as error:
                receipt.update(failure_reason='spectral_evaluation_failed', failed_orders=list(key), detail=str(error))
                raise DyadicConvergenceError(str(error), receipt) from error
            cache[key] = value
            receipt['evaluated_orders'].append(list(key))
        return cache[key]

    def norm(value):
        scale = value.new_tensor([1.]*3+[eta0]*3)
        return torch.linalg.matrix_norm(scale[:, None]*value*scale[None, :], ord='fro')

    radial, angular = config.radial_order, config.azimuthal_order
    while True:
        high_r, high_a = 2*radial, 2*angular
        if high_r > config.max_radial_order or high_a > config.max_azimuthal_order:
            receipt['failure_reason'] = 'spectral_order_budget_exhausted'
            raise DyadicConvergenceError('Dyadic quadrature did not converge within both order budgets', receipt)
        radial_only = evaluate(high_r, angular)
        angular_only = evaluate(radial, high_a)
        high = evaluate(high_r, high_a)
        radial_error, angular_error = norm(high-angular_only), norm(high-radial_only)
        magnitude = norm(high)
        threshold = config.absolute_tolerance+config.relative_tolerance*magnitude
        finite_metrics = bool((torch.isfinite(radial_error)&torch.isfinite(angular_error)
                               &torch.isfinite(magnitude)&torch.isfinite(threshold)).all())
        radial_pass = finite_metrics and bool((radial_error <= .5*threshold).all())
        angular_pass = finite_metrics and bool((angular_error <= .5*threshold).all())
        def maximum(value): return float(value.max().detach()) if value.numel() else 0.
        tiny = torch.finfo(magnitude.dtype).tiny
        receipt['checks'].append({'orders': [high_r, high_a],
            'radial_max_absolute_error': maximum(radial_error),
            'azimuthal_max_absolute_error': maximum(angular_error),
            'radial_max_relative_error': maximum(radial_error/magnitude.clamp_min(tiny)),
            'azimuthal_max_relative_error': maximum(angular_error/magnitude.clamp_min(tiny)),
            'radial_pass': radial_pass, 'azimuthal_pass': angular_pass,
            'finite_acceptance_metrics': finite_metrics})
        if not finite_metrics:
            receipt['failure_reason'] = 'nonfinite_spectral_acceptance_metrics'
            raise DyadicConvergenceError('Nonfinite dyadic spectral convergence metrics', receipt)
        if radial_pass and angular_pass:
            receipt.update(converged=True, final_orders=[high_r, high_a], failure_reason=None)
            if mode_cache is not None: receipt['spectral_cache'] = mode_cache.stats()
            return high, receipt
        # Keep an already adequate direction fixed and reuse evaluated corners.
        if not radial_pass: radial = high_r
        if not angular_pass: angular = high_a


def electrostatic_kernel(displacements, medium, *, constitutive='electric'):
    """Anisotropic Coulomb potential, gradient and Hessian off the source.

    The inverse symbol is 1/(q.T T q), hence only the transpose-symmetric part
    of T enters. Square roots follow passive limiting absorption, including
    all-negative scalar media. This is the leading electric/magnetic diagonal
    singularity; magnetic cross kernels and contact distributions are separate.
    """
    potential, v, squared, inverse = _electrostatic_components(displacements, medium, constitutive=constitutive)
    if not bool(torch.isfinite(squared).all() & torch.isfinite(v).all() & (potential.abs()>0).all()):
        raise ValueError('Static point kernel geometry is not representable')
    metric_gradient = v/squared[:, None]
    gradient = -potential[:, None]*metric_gradient
    # Squaring the quadratic distance can overflow while the Hessian remains
    # representable. Factor first, retaining native constitutive derivatives.
    hessian = potential[:, None, None]*(3*metric_gradient[..., :, None]*metric_gradient[..., None, :]
                                        -inverse/squared[:, None, None])
    if not bool(torch.isfinite(potential).all() & torch.isfinite(gradient).all() & torch.isfinite(hessian).all()):
        raise ValueError('Static point kernel derivatives are not representable')
    return potential, gradient, hessian


def _electrostatic_components(displacements, medium, *, constitutive):
    """Shared branch-selected components for kernels and factored contrasts."""
    medium.assert_current()
    if constitutive not in ('electric', 'magnetic'):
        raise ValueError('constitutive must be electric or magnetic')
    tensor = medium.epsilon_r if constitutive == 'electric' else medium.mu_r
    if (not isinstance(displacements, torch.Tensor) or displacements.device != tensor.device
            or displacements.dtype != tensor.real.dtype or displacements.ndim != 2
            or displacements.shape[1] != 3 or not bool(torch.isfinite(displacements).all())):
        raise ValueError('Static kernel needs finite matching CUDA real [N,3] displacements')
    symmetric = .5*(tensor+tensor.mT)
    inverse, info = torch.linalg.inv_ex(symmetric)
    if bool(info != 0) or not bool(torch.isfinite(inverse).all()):
        raise ValueError('Singular static constitutive tensor requires a separate limiting formulation')
    eigenvalues = torch.linalg.eigvals(symmetric)
    tolerance = 64*torch.finfo(displacements.dtype).eps*torch.linalg.matrix_norm(symmetric)
    if bool((eigenvalues.imag < -tolerance).any()):
        raise ValueError('Static tensor lies outside the passive limiting-absorption branch')
    # Suppress only roundoff-negative imaginary eigenvalues. A principal square
    # root of det(T) would select the wrong sign after multiple eigenvalue turns.
    passive_eigenvalues = torch.complex(eigenvalues.real, eigenvalues.imag.clamp_min(0))
    determinant_root = torch.sqrt(passive_eigenvalues).prod()
    points = displacements.to(tensor.dtype)
    v = points@inverse.mT
    squared = (points*v).sum(-1)
    root = torch.sqrt(squared)
    root = torch.where(root.imag > 0, -root, root)
    if bool((root == 0).any()):
        raise ValueError('Static Green singularity on the source or an anisotropic characteristic cone')
    prefactor = 1/(4*math.pi*determinant_root)
    potential = prefactor/root
    return potential, v, squared, inverse


def compile_electrostatic_plane_modes(transverse, medium):
    """Longitudinal static diagonal residues, excluding contact distributions.

    These terms can be subtracted before transverse integration and restored
    analytically with electrostatic_kernel, avoiding numerical integration of
    the leading 1/r^3 dyadic singularity. No transverse finite cutoff is used.
    """
    medium.assert_current()
    if (not isinstance(transverse, torch.Tensor) or transverse.device != medium.epsilon_r.device
            or transverse.dtype != medium.epsilon_r.real.dtype or transverse.ndim != 2
            or transverse.shape[1] != 2 or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Static spectrum needs finite matching real CUDA [N,2] coordinates')
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    result = medium.epsilon_r.new_zeros((len(transverse), 2, 6, 6))
    longitudinal = medium.epsilon_r.new_zeros((len(transverse), 2))
    selected = transverse.square().sum(-1) > 0
    u = transverse[selected].to(medium.epsilon_r.dtype)
    for offset, tensor, vacuum in ((0, medium.epsilon_r, EPSILON_0), (3, medium.mu_r, MU_0)):
        symmetric = .5*(tensor+tensor.mT)
        a = symmetric[2, 2]
        if bool(a == 0):
            raise ValueError('Singular longitudinal static coefficient requires a different integration axis')
        b = u@symmetric[2, :2]
        c = torch.einsum('ni,ij,nj->n', u, symmetric[:2, :2], u)
        discriminant = torch.sqrt(b*b-a*c)
        roots = torch.stack(((-b+discriminant)/a, (-b-discriminant)/a), -1)
        tolerance = 128*torch.finfo(transverse.dtype).eps*roots.abs().amax(-1)
        derivative = 2*(a*roots+b[:, None])
        outgoing = ((roots.imag > tolerance[:, None])
            | ((roots.imag.abs() <= tolerance[:, None]) & (derivative.real < -tolerance[:, None]*a.abs())))
        if bool((outgoing.sum(-1) != 1).any()):
            raise ValueError('Static outgoing root is unresolved at a characteristic spectral point')
        root = torch.where(outgoing, roots, torch.zeros_like(roots)).sum(-1)
        denominator = 2*k0*(a*root+b)
        if bool((denominator == 0).any()):
            raise ValueError('Double static spectral root requires singular contour integration')
        scalar = 1j/denominator
        q = k0*torch.cat((u, root[:, None]), -1)
        hessian = -q[..., :, None]*q[..., None, :]*scalar[:, None, None]
        mode = offset//3
        result[selected, mode, offset:offset+3, offset:offset+3] = 1j*hessian/(medium.omega*vacuum)
        longitudinal[selected, mode] = k0*root
    return PlaneModes(k0*transverse, longitudinal, result, medium)


def electrostatic_plane_spectrum(transverse, distance, medium):
    """Evaluate static diagonal residues at a positive separation."""
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Static plane spectrum requires positive distance')
    return compile_electrostatic_plane_modes(transverse, medium).spectrum(distance)


@dataclass(frozen=True)
class DyadicGreenKernel:
    medium: object
    integration: DyadicIntegrationConfig = field(default_factory=DyadicIntegrationConfig)
    cache_max_bytes: int = 256*1024**2
    mode_cache: SpectralModeCache = field(init=False, repr=False)
    _isotropic: bool = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.integration, DyadicIntegrationConfig):
            raise TypeError('integration must be DyadicIntegrationConfig')
        if self.integration.spatial_rule in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
            _validate_factored_plane_medium(self.medium)
        object.__setattr__(self, 'mode_cache', SpectralModeCache(self.medium, self.cache_max_bytes))
        eye = torch.eye(3, device=self.medium.epsilon_r.device, dtype=self.medium.epsilon_r.dtype)
        object.__setattr__(self, '_isotropic', torch.equal(self.medium.epsilon_r, self.medium.epsilon_r[0, 0]*eye)
                           and torch.equal(self.medium.mu_r, self.medium.mu_r[0, 0]*eye))

    def evaluate(self, displacements):
        self.mode_cache.validate(self.medium)
        if self._isotropic and self.integration.spatial_rule not in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
            return isotropic_dyadic_green(displacements, self.medium)
        if self.integration.spatial_rule=='proportional_spd_closed_form_v1':
            from .matched_spd_green import proportional_spd_spatial_green
            result=proportional_spd_spatial_green(displacements,self.medium)
            if result is not None:return result
        return evaluate_spatial_green(displacements, self.medium, self.integration, mode_cache=self.mode_cache)

    def evaluate_regular(self, displacements):
        self.mode_cache.validate(self.medium)
        if self._isotropic and self.integration.spatial_rule not in ('factored_spd_tail_v1', 'factored_spd_pair_tail_v1'):
            return isotropic_dyadic_green(displacements, self.medium, regular_part=True)
        if self.integration.spatial_rule=='proportional_spd_closed_form_v1':
            from .matched_spd_green import proportional_spd_spatial_green
            result=proportional_spd_spatial_green(displacements,self.medium,regular_part=True)
            if result is not None:return result
        return evaluate_spatial_green(displacements, self.medium, self.integration, regular_part=True,
                                      mode_cache=self.mode_cache)


def isotropic_dyadic_green(displacements, medium, *, regular_part=False):
    """Exact scalar specialization, with cancellation-free static subtraction.

    The regular diagonal uses derivatives of (exp(i*k*r)-1)/(4*pi*r).
    Its second radial derivative starts at cubic order in z=i*k*r; a series
    avoids subtracting two quadratic terms near the source.
    """
    from .lifting import _gradient_remainder
    medium.assert_current()
    eye = torch.eye(3, device=medium.epsilon_r.device, dtype=medium.epsilon_r.dtype)
    eps, mu = medium.epsilon_r[0, 0], medium.mu_r[0, 0]
    if not torch.equal(medium.epsilon_r, eps*eye) or not torch.equal(medium.mu_r, mu*eye):
        raise ValueError('Analytic scalar dyadic specialization requires exactly isotropic tensors')
    if bool((eps == 0) | (mu == 0)):
        raise ValueError('Zero constitutive parameter requires a separate limiting formulation')
    if (not isinstance(displacements, torch.Tensor) or displacements.device != eye.device
            or displacements.dtype != eye.real.dtype or displacements.ndim != 2 or displacements.shape[1] != 3
            or not bool(torch.isfinite(displacements).all())):
        raise ValueError('Scalar dyadic queries must be finite matching CUDA real [N,3]')
    radius = torch.linalg.vector_norm(displacements, dim=-1)
    if bool((radius == 0).any()):
        raise ValueError('Coincident dyadic nodes require singular integration')
    roots = torch.sqrt(torch.stack((eps, mu)))
    roots = torch.where(roots.imag < 0, -roots, roots)
    k = medium.omega*math.sqrt(EPSILON_0*MU_0)*roots.prod()
    z = 1j*k*radius
    green = torch.exp(z)/(4*math.pi*radius)
    direction = (displacements/radius[:, None]).to(eye.dtype)
    nn = direction[..., :, None]*direction[..., None, :]
    first_difference = _gradient_remainder(z)/(4*math.pi*radius**2)
    polynomial = torch.zeros_like(z)
    for degree in range(14, 2, -1):
        polynomial = (degree-1)*(degree-2)/math.factorial(degree)+z*polynomial
    second_numerator = torch.where(z.abs() < .25, z**3*polynomial,
                                    z*z*torch.exp(z)-2*_gradient_remainder(z))
    second_difference = second_numerator/(4*math.pi*radius**3)
    hessian_difference = (second_difference[:, None, None]*nn
                          +(first_difference/radius)[:, None, None]*(eye-nn))
    ee = 1j*medium.omega*MU_0*mu*green[:, None, None]*eye+1j*hessian_difference/(medium.omega*EPSILON_0*eps)
    hh = 1j*medium.omega*EPSILON_0*eps*green[:, None, None]*eye+1j*hessian_difference/(medium.omega*MU_0*mu)
    if not regular_part:
        static = (3*nn-eye)/(4*math.pi*radius[:, None, None]**3)
        ee = ee+1j*static/(medium.omega*EPSILON_0*eps)
        hh = hh+1j*static/(medium.omega*MU_0*mu)
    curl = cross_matrix((green*(z-1)/radius)[:, None]*direction)
    value = torch.cat((torch.cat((ee, -curl), -1), torch.cat((curl, hh), -1)), -2)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite scalar dyadic Green value')
    return value, {'kernel_version': 'analytic_isotropic_dyadic_v1', 'converged': True,
        'query_count': len(displacements), 'certified_error_bound': False,
        'static_diagonal_singularity_subtracted': True, 'static_diagonal_restored': not regular_part,
        'quadrature': 'closed_form_no_spectral_quadrature', 'contact_distribution_included': False}

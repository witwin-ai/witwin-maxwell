"""Contrast-factored outgoing plane Green differences.

This is a spectral plane primitive. Spatial integration, static subtraction and
surface lifting must separately establish convergence before using this path.
"""
import math
from dataclasses import dataclass, field
import torch

from ..constants import EPSILON_0, MU_0
from .dyadic_green import maxwell_symbol, cross_matrix, SpectralModeCache, _plane_eigensystem


def _reduce(transverse, medium):
    medium.assert_current()
    k0 = medium.omega*math.sqrt(EPSILON_0*MU_0)
    q = torch.cat((transverse, torch.zeros_like(transverse[:, :1])), -1).to(medium.epsilon_r.dtype)
    symbol = maxwell_symbol(k0*q, medium.epsilon_r, medium.mu_r, medium.frequency)
    t = torch.tensor([0, 1, 3, 4], device=q.device)
    l = torch.tensor([2, 5], device=q.device)
    tt, tl = symbol[:, t[:, None], t], symbol[:, t[:, None], l]
    lt, ll = symbol[:, l[:, None], t], symbol[:, l[:, None], l]
    inverse, info = torch.linalg.inv_ex(ll)
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse).all()):
        raise ValueError('Contrast plane has a singular longitudinal constitutive block')
    cz = cross_matrix(q.new_tensor([0., 0., 1.]))
    zero = torch.zeros_like(cz)
    derivative = torch.cat((torch.cat((zero, cz), -1), torch.cat((cz, zero), -1)), -2)[t[:, None], t]
    generator = -torch.linalg.solve(derivative, tt-tl@inverse@lt)
    identity = torch.eye(4, device=q.device, dtype=q.dtype).expand(len(q), 4, 4)
    embedding = torch.cat((identity, -inverse@lt), -2)
    forcing = torch.linalg.solve(derivative, torch.cat((identity, -tl@inverse), -1))
    roots, vectors, receipt = _plane_eigensystem(generator, transverse, medium)
    inverse_vectors, info = torch.linalg.inv_ex(vectors)
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse_vectors).all()):
        raise ValueError('Contrast plane requires a resolved Maxwell eigenspace')
    tolerance = 128*torch.finfo(transverse.dtype).eps*torch.linalg.matrix_norm(generator, ord=2)
    flux = (vectors[:, 0]*vectors[:, 3].conj()-vectors[:, 1]*vectors[:, 2].conj()).real
    outgoing = ((roots.imag > tolerance[:, None])
                | ((roots.imag.abs() <= tolerance[:, None]) & (flux > 128*torch.finfo(transverse.dtype).eps)))
    if bool((outgoing.sum(-1) != 2).any()):
        raise ValueError('Contrast plane outgoing split is unresolved at a grazing or defective point')
    return {'roots': roots, 'vectors': vectors, 'inverse_vectors': inverse_vectors,
            'outgoing': outgoing, 'embedding': embedding, 'forcing': forcing,
            'inverse_ll': inverse, 'tl': tl, 'lt': lt, 'derivative': derivative,
            't': t, 'l': l, 'eigensolver': receipt}


def _outgoing_divided_difference(inside, outside, phase_distance):
    """Loewner matrix of exp(i*z*A) on the outgoing spectral subspace.

    Equal-branch nearly coincident poles use expm1, including the analytic limit
    at equality. Mixed branches use their non-canceling numerator. No pole
    perturbation, added damping or scalar eigenvector matching is introduced.
    """
    a, b = inside['roots'][:, :, None], outside['roots'][:, None, :]
    pi, pe = inside['outgoing'][:, :, None], outside['outgoing'][:, None, :]
    difference = a-b
    argument = 1j*phase_distance*difference
    # Evaluate only the selected branches: decaying exponentials on the other
    # side must not produce unused overflowing intermediates.
    result = torch.zeros_like(difference)
    both = (pi & pe).expand_as(result)
    if bool(both.any()):
        x = argument[both]
        # Choose the endpoint with larger exponential magnitude, so expm1's
        # argument has nonpositive real part even for widely separated poles.
        use_b = x.real <= 0
        endpoint = torch.where(use_b, b.expand_as(result)[both], a.expand_as(result)[both])
        signed_x = torch.where(use_b, x, -x)
        nonzero = signed_x != 0
        ratio = torch.ones_like(signed_x)
        ratio[nonzero] = torch.expm1(signed_x[nonzero])/signed_x[nonzero]
        result[both] = (1j*phase_distance)*torch.exp(1j*phase_distance*endpoint)*ratio
    mixed = (pi ^ pe).expand_as(result)
    if bool(mixed.any()):
        delta = difference[mixed]
        if bool((delta == 0).any()):
            raise ValueError('Coincident poles have inconsistent radiation branches')
        roots = torch.where(pi.expand_as(result)[mixed], a.expand_as(result)[mixed], b.expand_as(result)[mixed])
        signs = torch.where(pi.expand_as(result)[mixed], 1., -1.)
        result[mixed] = signs*torch.exp(1j*phase_distance*roots)/delta
    return result


def compile_plane_contrast(transverse, exterior, interior):
    """Compile reusable G_i-G_e with explicit constitutive contrast in every term.

    The exact Schur-complement and matrix-function divided-difference identities
    avoid subtracting full Green tensors. Contact distributions at z=0 are
    excluded. Inputs use the same dimensionless transverse wavevector grid as
    compile_plane_modes. This primitive does not assert spatial convergence.
    """
    exterior.assert_current()
    interior.assert_current()
    if (exterior.frequency != interior.frequency or exterior.epsilon_r.device != interior.epsilon_r.device
            or exterior.epsilon_r.dtype != interior.epsilon_r.dtype):
        raise ValueError('Contrast planes require common frequency, device and precision')
    if (not isinstance(transverse, torch.Tensor) or transverse.device != exterior.epsilon_r.device
            or transverse.dtype != exterior.epsilon_r.real.dtype or transverse.ndim != 2
            or transverse.shape[1] != 2 or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Contrast transverse coordinates require finite matching real CUDA [N,2] tensors')
    transverse = transverse.clone()
    i, e = _reduce(transverse, interior), _reduce(transverse, exterior)
    delta = torch.block_diag(interior.epsilon_r-exterior.epsilon_r, exterior.mu_r-interior.mu_r)
    t, l = i['t'], i['l']
    dtt, dtl, dlt, dll = delta[t[:, None], t], delta[t[:, None], l], delta[l[:, None], t], delta[l[:, None], l]
    li, le = i['inverse_ll'], e['inverse_ll']
    dl = -li@dll@le
    reduced_difference = dtt-dtl@li@i['lt']-e['tl']@dl@i['lt']-e['tl']@le@dlt
    da = -torch.linalg.solve(i['derivative'], reduced_difference)
    dembed = torch.cat((torch.zeros_like(i['embedding'][:, :4]), -li@dlt-dl@e['lt']), -2)
    dforce = torch.linalg.solve(i['derivative'], torch.cat((torch.zeros_like(i['forcing'][:, :, :4]),
                                                         -dtl@li-e['tl']@dl), -1))
    return PlaneContrast(transverse, exterior, interior, i, e, da, dembed, dforce)


class _ContrastQueries:
    """Shared guarded positive-z query operations for compiled contrasts."""

    def __post_init__(self):
        object.__setattr__(self, '_versions', tuple((id(v), v._version) for v in self.tensors()))

    def _assert_current(self):
        self.exterior.assert_current()
        self.interior.assert_current()
        if tuple((id(v), v._version) for v in self.tensors()) != self._versions:
            raise ValueError('Compiled contrast plane tensors changed; rebuild the spectral grid')

    def spectrum(self, distance):
        self._assert_current()
        if isinstance(distance, torch.Tensor):
            if (distance.shape != () or distance.device != self.transverse.device
                    or distance.dtype != self.transverse.dtype or not bool(torch.isfinite(distance))
                    or bool(distance <= 0)):
                raise ValueError('Contrast distance requires a matching positive finite CUDA scalar')
        elif isinstance(distance, bool) or not math.isfinite(distance) or distance <= 0:
            raise ValueError('Contrast planes require strictly positive finite separation')
        return self._spectrum(distance)

    def integrate(self, displacements, weights):
        """Apply a supplied physical transverse quadrature, without certifying it."""
        self._assert_current()
        if (not isinstance(displacements, torch.Tensor) or displacements.device != self.transverse.device
                or displacements.dtype != self.transverse.dtype or displacements.ndim != 2
                or displacements.shape[1] != 3 or not bool(torch.isfinite(displacements).all())
                or bool((displacements[:, 2] <= 0).any())):
            raise ValueError('Contrast queries require finite matching CUDA [N,3] and positive z')
        if (not isinstance(weights, torch.Tensor) or weights.device != self.transverse.device
                or weights.dtype != self.transverse.dtype or weights.shape != (len(self.transverse),)
                or not bool(torch.isfinite(weights).all())):
            raise ValueError('Contrast weights must match the real CUDA transverse grid')
        k0 = self.interior.omega*math.sqrt(EPSILON_0*MU_0)
        values = []
        for point in displacements.unbind(0):
            phase = torch.exp(1j*k0*(self.transverse@point[:2]))*weights
            values.append(torch.einsum('n,nij->ij', phase, self._spectrum(point[2])))
        return torch.stack(values) if values else self.interior.epsilon_r.new_empty((0, 6, 6))



@dataclass(frozen=True)
class PlaneContrast(_ContrastQueries):
    """A fixed spectral grid reusable across positive-z queries.

    Integration stores only one [nodes,6,6] evaluation at a time. It does not
    construct a queries-by-nodes Green array or repeat material eigenproblems.
    Retained tensors and both constitutive samples are guarded against mutation.
    """
    transverse: torch.Tensor
    exterior: object
    interior: object
    _inside: dict = field(repr=False)
    _outside: dict = field(repr=False)
    _generator_difference: torch.Tensor = field(repr=False)
    _embedding_difference: torch.Tensor = field(repr=False)
    _forcing_difference: torch.Tensor = field(repr=False)
    _versions: tuple = field(init=False, repr=False)

    @property
    def eigensolver(self):
        receipts = [self._inside['eigensolver'], self._outside['eigensolver']]
        return {key: sum(receipt[key] for receipt in receipts) for key in ('matrix_count', 'fallback_count')}

    def tensors(self):
        values = [self.transverse, self._generator_difference, self._embedding_difference, self._forcing_difference]
        for data in (self._inside, self._outside):
            values.extend(value for _, value in sorted(data.items()) if isinstance(value, torch.Tensor))
        return tuple(values)




    def _spectrum(self, distance):
        i, e = self._inside, self._outside
        da, dembed, dforce = self._generator_difference, self._embedding_difference, self._forcing_difference
        k0 = self.interior.omega*math.sqrt(EPSILON_0*MU_0)
        phase_distance = k0*distance
        def propagator(data):
            values = torch.zeros_like(data['roots'])
            selected = data['outgoing']
            values[selected] = torch.exp(1j*phase_distance*data['roots'][selected])
            return (data['vectors']*values[:, None, :])@data['inverse_vectors']
        fi, fe = propagator(i), propagator(e)
        loewner = _outgoing_divided_difference(i, e, phase_distance)
        df = i['vectors']@(loewner*(i['inverse_vectors']@da@e['vectors']))@e['inverse_vectors']
        difference = 1j*k0*(dembed@fi@i['forcing'] + e['embedding']@df@i['forcing']
                            + e['embedding']@fe@dforce)
        reorder = i['t'].new_tensor([0, 1, 4, 2, 3, 5])
        difference = difference[:, reorder[:, None], reorder]
        eta0 = math.sqrt(MU_0/EPSILON_0)
        source = difference.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
        field_scale = difference.new_tensor([1.]*3+[1/eta0]*3)
        value = field_scale[:, None]*difference*source[None, :]
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite outgoing contrast plane spectrum')
        return value



def outgoing_plane_spectrum_difference(transverse, distance, exterior, interior):
    """Evaluate one plane; repeated distances should reuse compile_plane_contrast."""
    return compile_plane_contrast(transverse, exterior, interior).spectrum(distance)


def _four_state_blocks(diagonal, inside_dynamic, outside_dynamic,
                       material_full, material_static, mixed):
    """Encode two finite differences as one upper triangular block matrix.

    State order is inside/full, inside/static, outside/full, outside/static.
    The (0,3) block carries the material difference of dynamic differences.
    Rectangular field and source factors use the same block algebra.
    """
    rows, columns = diagonal[0].shape[-2:]
    result = diagonal[0].new_zeros((*diagonal[0].shape[:-2], 4*rows, 4*columns))
    for index, value in enumerate(diagonal):
        result[..., index*rows:(index+1)*rows, index*columns:(index+1)*columns] = value
    for i, j, value in ((0, 1, inside_dynamic), (2, 3, outside_dynamic),
                        (0, 2, material_full), (1, 3, material_static), (0, 3, mixed)):
        result[..., i*rows:(i+1)*rows, j*columns:(j+1)*columns] = value
    return result


def _factored_constitutive_differences(components_e, components_i, exterior, interior, x):
    """Retain actual constitutive differences before congruences and inverses."""
    eps, rotation, _, li, bi, ci, b1, c1, derivative, powers, _, _ = components_i
    _, _, _, le, be, ce, _, _, _, _, _, _ = components_e
    de = rotation.mT@(interior.epsilon_r-exterior.epsilon_r)@rotation
    dm = rotation.mT@(interior.mu_r-exterior.mu_r)@rotation
    zero = torch.zeros_like(de)
    material = torch.cat((torch.cat((de, zero), -1), torch.cat((zero, -dm), -1)), -2)
    t = torch.tensor([0, 1, 3, 4], device=eps.device)
    l = torch.tensor([2, 5], device=eps.device)
    dtt = material[:, t[:, None], t]
    db, dc, dll = material[:, t[:, None], l], material[:, l[:, None], t], material[:, l[:, None], l]
    dl = -li@dll@le
    coefficients = [
        -torch.linalg.solve(derivative, dtt-db@li@ci-be@dl@ci-be@le@dc),
        torch.linalg.solve(derivative, b1@dl@ci+b1@le@dc+db@li@c1+be@dl@c1),
        torch.linalg.solve(derivative, b1@dl@c1)]
    g = [torch.zeros_like(coefficients[0]) for _ in range(3)]
    for degree, coefficient in enumerate(coefficients):
        exponent = degree-1+powers[None, :]-powers[:, None]
        if bool((coefficient[:, exponent > 0] != 0).any()):
            raise ValueError('Unexpected positive power in factored contrast generator')
        for order in range(3):
            g[order] = g[order]+torch.where(exponent == -order, coefficient, 0.)
    dynamic_generator = x[:, None, None]*g[1]+x.square()[:, None, None]*g[2]
    dec, dew = -li@dc-dl@ce, -dl@c1
    embedding_static = eps.new_zeros((len(x), 6, 4))
    embedding_static[:, 4:] = torch.where(powers[None, None, :] == 1, dec, dew)
    embedding_dynamic = torch.zeros_like(embedding_static)
    embedding_dynamic[:, 4:] = torch.where(powers[None, None, :] == 0, x[:, None, None]*dec, 0.)
    identity = torch.eye(4, device=eps.device, dtype=eps.dtype).expand(len(x), 4, 4)
    forcing_coefficients = [
        torch.linalg.solve(derivative, torch.cat((torch.zeros_like(identity), -db@li-be@dl), -1)),
        torch.linalg.solve(derivative, torch.cat((torch.zeros_like(identity), -b1@dl), -1))]
    full_powers = t.new_tensor([1, 0, 1, 0, 1, 1])
    f = [eps.new_zeros((len(x), 4, 6)) for _ in range(3)]
    for degree, coefficient in enumerate(forcing_coefficients):
        exponent = degree-1+full_powers[None, :]-powers[:, None]
        if bool((coefficient[:, exponent > 0] != 0).any()):
            raise ValueError('Unexpected positive power in factored contrast forcing')
        for order in range(3):
            f[order] = f[order]+torch.where(exponent == -order, coefficient, 0.)
    forcing_dynamic = x[:, None, None]*f[1]+x.square()[:, None, None]*f[2]
    return g[0], dynamic_generator, embedding_static, embedding_dynamic, f[0], forcing_dynamic


def _compile_factored_plane_remainder_contrast(transverse, exterior, interior):
    """Compile two explicit small differences for fixed real-SPD evanescence.

    This internal positive-z plane primitive is not wired into spatial/public
    quadrature. Material and inverse-length differences remain off-diagonal
    factors through Newton sign iterations and the propagation exponential.
    """
    from .dyadic_green import (_validate_factored_plane_medium,
        _factored_plane_polynomial_components, _factored_plane_state_factors)
    if (exterior.frequency != interior.frequency or exterior.epsilon_r.device != interior.epsilon_r.device
            or exterior.epsilon_r.dtype != interior.epsilon_r.dtype):
        raise ValueError('Factored contrast requires common frequency, device and precision')
    bound = torch.maximum(_validate_factored_plane_medium(exterior), _validate_factored_plane_medium(interior))
    if (not isinstance(transverse, torch.Tensor) or transverse.device != interior.epsilon_r.device
            or transverse.dtype != torch.float64 or transverse.requires_grad
            or transverse.ndim != 2 or transverse.shape[1] != 2
            or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Factored contrast requires matching finite CUDA float64 [N,2] coordinates')
    transverse = transverse.clone()
    length = torch.linalg.vector_norm(transverse, dim=-1)
    if not bool((length > bound).all()):
        raise ValueError('Factored contrast requires both media beyond their evanescent bound')
    x = length.reciprocal()
    ce = _factored_plane_polynomial_components(transverse, exterior, length, x)
    ci = _factored_plane_polynomial_components(transverse, interior, length, x)
    ge, gi = ce[-1], ci[-1]
    ae0, ai0 = ge[0], gi[0]
    de = x[:, None, None]*ge[1]+x.square()[:, None, None]*ge[2]
    di = x[:, None, None]*gi[1]+x.square()[:, None, None]*gi[2]
    ae, ai = ae0+de, ai0+di
    da0, dd, dembed0, ddembed, dforce0, ddforce = _factored_constitutive_differences(ce, ci, exterior, interior, x)
    generator = _four_state_blocks((ai, ai0, ae, ae0), di, de, da0+dd, da0, dd)
    identity = torch.eye(16, device=generator.device, dtype=generator.dtype)
    sign = 1j*generator
    tolerance = 256*torch.finfo(x.dtype).eps
    for iterations in range(1, 33):
        inverse, info = torch.linalg.inv_ex(sign)
        if bool((info != 0).any()) or not bool(torch.isfinite(inverse).all()):
            raise ValueError('Singular iteration in paired evanescent projector')
        sign = .5*(sign+inverse)
        residual = sign@sign-identity
        diagonal_residual = torch.stack([residual[:, j:j+4, j:j+4].norm(dim=(-2, -1))
                                         for j in (0, 4, 8, 12)], -1)
        if bool((diagonal_residual <= tolerance).all()):
            break
    else:
        raise ValueError('Paired evanescent projector did not converge')
    projector = .5*(identity-sign)
    if (not bool(torch.isfinite(projector).all())
            or any(not bool(((projector[:, j:j+4, j:j+4].diagonal(dim1=-2, dim2=-1).sum(-1)-2).abs() <= tolerance).all())
                   for j in (0, 4, 8, 12))):
        raise ValueError('Paired evanescent projector is unresolved')
    eye4 = identity[:4, :4].expand(len(x), 4, 4)
    ei, ei0, dei, fi, fi0, dfi, full_powers = _factored_plane_state_factors(ci, x, eye4)
    ee, ee0, dee, fe, fe0, dfe, _ = _factored_plane_state_factors(ce, x, eye4)
    embedding = _four_state_blocks((ei, ei0, ee, ee0), dei, dee, dembed0+ddembed, dembed0, ddembed)
    forcing = _four_state_blocks((fi, fi0, fe, fe0), dfi, dfe, dforce0+ddforce, dforce0, ddforce)
    rotation = ci[1]
    physical_rotation = generator.new_zeros((len(x), 6, 6))
    physical_rotation[:, :3, :3] = physical_rotation[:, 3:, 3:] = rotation
    factors = torch.where(full_powers[None, :] == 1, length[:, None], 1.)
    state = dict(generator=generator, projector=projector, embedding=embedding,
                 forcing=forcing, length=length, factors=factors, physical_rotation=physical_rotation)
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Nonfinite compiled paired evanescent state')
    return _FactoredPlaneRemainderContrast(transverse, exterior, interior, state, iterations)


@dataclass(frozen=True)
class _FactoredPlaneRemainderContrast(_ContrastQueries):
    """Owned paired plane state;16x16 propagation is bounded to64 nodes."""
    transverse: torch.Tensor
    exterior: object
    interior: object
    state: dict = field(repr=False)
    sign_iterations: int
    _versions: tuple = field(init=False, repr=False)
    _state_keys: tuple = field(init=False, repr=False)

    def _guard_tensors(self):
        return self.tensors()+(self.exterior.epsilon_r, self.exterior.mu_r,
                               self.interior.epsilon_r, self.interior.mu_r)

    def __post_init__(self):
        object.__setattr__(self, '_state_keys', tuple(sorted(self.state)))
        object.__setattr__(self, '_versions', tuple((id(v), v._version, v.requires_grad) for v in self._guard_tensors()))

    def tensors(self):
        return (self.transverse,)+tuple(v for _, v in sorted(self.state.items()))

    def _assert_current(self):
        self.exterior.assert_current()
        self.interior.assert_current()
        if (tuple(sorted(self.state)) != self._state_keys
                or any(not isinstance(v, torch.Tensor) for v in self.state.values())
                or tuple((id(v), v._version, v.requires_grad) for v in self._guard_tensors()) != self._versions):
            raise ValueError('Compiled paired remainder tensors changed; rebuild the spectral grid')

    def _spectrum(self, distance):
        values = []
        k0 = self.interior.omega*math.sqrt(EPSILON_0*MU_0)
        for start in range(0, len(self.transverse), 64):
            state = {key: value[start:start+64] for key, value in self.state.items()}
            a, p = state['generator'], state['projector']
            identity = torch.eye(16, device=a.device, dtype=a.dtype)
            scale = torch.stack([a[:, j:j+4, j:j+4].norm(dim=(-2, -1)) for j in (0, 4, 8, 12)], -1).amax(-1)
            time = k0*distance*state['length'][:, None, None]
            exponent = 1j*time*(a@p)-time*scale[:, None, None]*(identity-p)
            propagator = torch.linalg.matrix_exp(exponent)@p
            green = state['embedding']@propagator@state['forcing']
            mixed, material = green[:, :6, 18:24], green[:, :6, 12:18]
            electric = torch.tensor([True, True, False, False, True, False], device=a.device)
            difference = torch.where(electric[:, None] == electric[None, :], mixed, material)
            factors = state['factors']
            difference = 1j*k0*state['length'][:, None, None]*factors[:, :, None]*difference/factors[:, None, :]
            reorder = torch.tensor([0, 1, 4, 2, 3, 5], device=a.device)
            difference = difference[:, reorder[:, None], reorder]
            eta0 = math.sqrt(MU_0/EPSILON_0)
            source = a.new_tensor([-1j*eta0/k0]*3+[1j/k0]*3)
            field_scale = a.new_tensor([1.]*3+[1/eta0]*3)
            difference = field_scale[:, None]*difference*source[None, :]
            rotation = state['physical_rotation']
            difference = rotation@difference@rotation.mT
            if not bool(torch.isfinite(difference).all()):
                raise ValueError('Nonfinite paired evanescent plane remainder')
            values.append(difference)
        return torch.cat(values) if values else self.interior.epsilon_r.new_empty((0, 6, 6))


@dataclass(frozen=True)
class _RegularPlaneContrastGrid(_ContrastQueries):
    """One shared node set with paired modal low nodes and factored tails."""
    transverse: torch.Tensor
    exterior: object
    interior: object
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
        values = [self.transverse, self.low_indices, self.tail_indices]
        for item in (self.low, self.static, self.tail):
            if item is not None:
                values.extend(item.tensors())
        return tuple(values)

    def _guard_tensors(self):
        return self.tensors()+(self.exterior.epsilon_r, self.exterior.mu_r,
                               self.interior.epsilon_r, self.interior.mu_r)

    def __post_init__(self):
        object.__setattr__(self, '_versions', tuple((id(t), t._version, t.requires_grad) for t in self._guard_tensors()))

    def _assert_current(self):
        self.exterior.assert_current()
        self.interior.assert_current()
        for item in (self.low, self.static, self.tail):
            if item is not None:
                item._assert_current()
        if tuple((id(t), t._version, t.requires_grad) for t in self._guard_tensors()) != self._versions:
            raise ValueError('Compiled paired regular grid changed; rebuild the nodes')

    def _spectrum(self, distance):
        value = self.interior.epsilon_r.new_zeros((len(self.transverse), 6, 6))
        if self.low is not None:
            value[self.low_indices] = self.low.spectrum(distance)-self.static.spectrum(distance)
        if self.tail is not None:
            value[self.tail_indices] = self.tail.spectrum(distance)
        return value


def _compile_regular_plane_contrast_grid(transverse, exterior, interior):
    from .dyadic_green import _validate_factored_plane_medium
    if (exterior.frequency != interior.frequency or exterior.epsilon_r.device != interior.epsilon_r.device
            or exterior.epsilon_r.dtype != interior.epsilon_r.dtype):
        raise ValueError('Regular contrast requires common frequency, device and precision')
    bound = torch.maximum(_validate_factored_plane_medium(exterior), _validate_factored_plane_medium(interior))
    if (not isinstance(transverse, torch.Tensor) or transverse.device != interior.epsilon_r.device
            or transverse.dtype != torch.float64 or transverse.requires_grad or transverse.ndim != 2
            or transverse.shape[1] != 2 or not bool(torch.isfinite(transverse).all())):
        raise ValueError('Regular contrasts require fixed matching CUDA float64 [N,2] coordinates')
    transverse = transverse.clone()
    mask = torch.linalg.vector_norm(transverse, dim=-1) > bound
    low_indices, tail_indices = torch.nonzero(~mask).flatten(), torch.nonzero(mask).flatten()
    if (torch.equal(exterior.epsilon_r, interior.epsilon_r)
            and torch.equal(exterior.mu_r, interior.mu_r)):
        return _RegularPlaneContrastGrid(transverse, exterior, interior, low_indices, tail_indices, None, None, None)
    low = compile_plane_contrast(transverse[low_indices], exterior, interior) if len(low_indices) else None
    static = compile_electrostatic_plane_contrast(transverse[low_indices], exterior, interior) if len(low_indices) else None
    tail = _compile_factored_plane_remainder_contrast(transverse[tail_indices], exterior, interior) if len(tail_indices) else None
    return _RegularPlaneContrastGrid(transverse, exterior, interior, low_indices, tail_indices, low, static, tail)


def electrostatic_kernel_difference(displacements, exterior, interior, *, constitutive='electric'):
    """Contrast-factored Coulomb potential, gradient and Hessian differences.

    The 3x3 determinant ratio is evaluated as 1 plus an explicit contrast
    polynomial. log1p/expm1 retain small differences; the actual passive static
    kernels select the root sign for finite contrasts across branch cuts.
    Contact distributions and anisotropic characteristic cones are excluded.
    """
    from .dyadic_green import _electrostatic_components
    if (exterior.frequency != interior.frequency or exterior.epsilon_r.device != interior.epsilon_r.device
            or exterior.epsilon_r.dtype != interior.epsilon_r.dtype):
        raise ValueError('Static contrast requires common frequency, device and precision')
    pe, ve, se, inverse_e = _electrostatic_components(displacements, exterior, constitutive=constitutive)
    pi, vi, si, inverse_i = _electrostatic_components(displacements, interior, constitutive=constitutive)
    te = exterior.epsilon_r if constitutive == 'electric' else exterior.mu_r
    ti = interior.epsilon_r if constitutive == 'electric' else interior.mu_r
    delta = .5*((ti-te)+(ti-te).mT)
    delta_inverse = -inverse_i@delta@inverse_e
    points = displacements.to(ti.dtype)
    dv = points@delta_inverse.mT
    ds = (points*dv).sum(-1)
    relative_tensor = inverse_e@delta
    trace = relative_tensor.diagonal().sum()
    determinant = (relative_tensor[0]*torch.linalg.cross(relative_tensor[1], relative_tensor[2])).sum()
    determinant_delta = trace+.5*(trace**2-(relative_tensor@relative_tensor).diagonal().sum())+determinant
    log_determinant = torch.log1p(determinant_delta)
    log_squared = torch.log1p(ds/se)
    log_potential_ratio = -.5*(log_determinant+log_squared)
    unsigned_ratio = torch.exp(log_potential_ratio)
    actual_ratio = pi/pe
    negative = (actual_ratio+unsigned_ratio).abs() < (actual_ratio-unsigned_ratio).abs()
    def relative_difference(log_ratio):
        positive_difference = torch.expm1(log_ratio)
        return torch.where(negative, -positive_difference-2, positive_difference)
    dp = pe*relative_difference(log_potential_ratio)
    be, ce = pe/se, pe/se**2
    db = be*relative_difference(log_potential_ratio-log_squared)
    dc = ce*relative_difference(log_potential_ratio-2*log_squared)
    dg = -db[:, None]*vi-be[:, None]*dv
    dh = (3*dc[:, None, None]*vi[:, :, None]*vi[:, None, :]
          +3*ce[:, None, None]*(dv[:, :, None]*vi[:, None, :]+ve[:, :, None]*dv[:, None, :])
          -db[:, None, None]*inverse_i-be[:, None, None]*delta_inverse)
    if any(not bool(torch.isfinite(value).all()) for value in (dp, dg, dh)):
        raise ValueError('Nonfinite static contrast kernel or unresolved determinant branch')
    return dp, dg, dh


def compile_electrostatic_plane_contrast(transverse, exterior, interior):
    """Compile contrast-factored static poles and residues on a shared grid."""
    from .dyadic_green import compile_electrostatic_plane_modes
    if (exterior.frequency != interior.frequency or exterior.epsilon_r.device != interior.epsilon_r.device
            or exterior.epsilon_r.dtype != interior.epsilon_r.dtype):
        raise ValueError('Static contrast planes require common frequency, device and precision')
    outside = compile_electrostatic_plane_modes(transverse, exterior)
    inside = compile_electrostatic_plane_modes(transverse, interior)
    transverse = transverse.clone()
    k0 = interior.omega*math.sqrt(EPSILON_0*MU_0)
    selected = transverse.square().sum(-1) > 0
    u = transverse[selected].to(interior.epsilon_r.dtype)
    delta_roots = torch.zeros_like(outside.longitudinal)
    delta_residues = torch.zeros_like(outside.residues)
    for mode, (te, ti, vacuum) in enumerate(((exterior.epsilon_r, interior.epsilon_r, EPSILON_0),
                                           (exterior.mu_r, interior.mu_r, MU_0))):
        se, si = .5*(te+te.mT), .5*(ti+ti.mT)
        delta = .5*((ti-te)+(ti-te).mT)
        ae, ai, da = se[2, 2], si[2, 2], delta[2, 2]
        be, bi, db = u@se[2, :2], u@si[2, :2], u@delta[2, :2]
        dc = torch.einsum('ni,ij,nj->n', u, delta[:2, :2], u)
        re, ri = outside.longitudinal[selected, mode]/k0, inside.longitudinal[selected, mode]/k0
        root_denominator = ai*(ri+re)+2*bi
        if bool((root_denominator == 0).any()):
            raise ValueError('Static contrast pole difference requires a resolved common radiation branch')
        dr = -(da*re**2+2*db*re+dc)/root_denominator
        de, di = 2*k0*(ae*re+be), 2*k0*(ai*ri+bi)
        dd = 2*k0*(da*ri+ae*dr+db)
        qe = k0*torch.cat((u, re[:, None]), -1)
        qi = k0*torch.cat((u, ri[:, None]), -1)
        dq = torch.zeros_like(qe)
        dq[:, 2] = k0*dr
        outer_delta = dq[:, :, None]*qi[:, None, :]+qe[:, :, None]*dq[:, None, :]
        residue_delta = (outer_delta/di[:, None, None]
                         -qe[:, :, None]*qe[:, None, :]*(dd/(di*de))[:, None, None])/(interior.omega*vacuum)
        offset = 3*mode
        delta_roots[selected, mode] = k0*dr
        delta_residues[selected, mode, offset:offset+3, offset:offset+3] = residue_delta
    return StaticPlaneContrast(transverse, exterior, interior, outside.longitudinal,
                               inside.longitudinal, delta_roots, outside.residues, delta_residues)


@dataclass(frozen=True)
class StaticPlaneContrast(_ContrastQueries):
    """Two static poles with contrast-factored phase and residue changes."""
    transverse: torch.Tensor
    exterior: object
    interior: object
    exterior_roots: torch.Tensor
    interior_roots: torch.Tensor
    root_difference: torch.Tensor
    exterior_residues: torch.Tensor
    residue_difference: torch.Tensor
    _versions: tuple = field(init=False, repr=False)

    def tensors(self):
        return (self.transverse, self.exterior_roots, self.interior_roots, self.root_difference,
                self.exterior_residues, self.residue_difference)

    def _spectrum(self, distance):
        pi = torch.exp(1j*distance*self.interior_roots)
        pe = torch.exp(1j*distance*self.exterior_roots)
        argument = 1j*distance*self.root_difference
        use_e = argument.real <= 0
        signed_argument = torch.where(use_e, argument, -argument)
        phase_difference = torch.where(use_e, pe, -pi)*torch.expm1(signed_argument)
        result = (pi[..., None, None]*self.residue_difference
                  +phase_difference[..., None, None]*self.exterior_residues).sum(1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite static contrast plane spectrum')
        return result


def evaluate_spatial_green_difference(displacements, exterior, interior, config=None, *, regular_part=False, mode_cache=None):
    """Adapt the factored difference using the shared radial/azimuthal estimator.

    Both materials share every quadrature node. Static contrast is subtracted
    before transverse integration and restored analytically unless regular_part
    is requested. No single-medium cache may be reused for this paired problem.
    """
    from .dyadic_green import evaluate_spatial_green, DyadicIntegrationConfig
    if config is None:
        config = DyadicIntegrationConfig()
    return evaluate_spatial_green(displacements, interior, config, regular_part=regular_part, mode_cache=mode_cache, _exterior=exterior)


class SpectralContrastCache(SpectralModeCache):
    """Bounded spectral LRU bound to an ordered pair of immutable samples."""
    def __init__(self, exterior, interior, max_bytes=256*1024**2):
        exterior.assert_current()
        self.exterior = exterior
        super().__init__(interior, max_bytes)

    def validate(self, medium):
        super().validate(medium)
        try:
            self.exterior.assert_current()
        except ValueError:
            self.clear()
            raise

    def validate_pair(self, exterior, interior):
        if exterior is not self.exterior:
            raise ValueError('Spectral contrast cache belongs to a different exterior sample')
        self.validate(interior)

    @staticmethod
    def _tensors(entry):
        modes, static_modes, weights = entry
        tensors = [weights]
        for item in (modes, static_modes):
            if item is not None:
                tensors.extend(item.tensors())
                for medium in (item.exterior, item.interior):
                    tensors.extend((medium.epsilon_r, medium.mu_r))
        return tensors


@dataclass(frozen=True)
class DyadicContrastKernel:
    """Paired regular Green and static contrasts for analytic carrier testing."""
    version = 'factored_dynamic_static_plane_and_analytic_static_contrast'
    exterior: object
    interior: object
    integration: object
    cache_max_bytes: int = 256*1024**2
    mode_cache: SpectralContrastCache = field(init=False, repr=False)

    def __post_init__(self):
        from .dyadic_green import DyadicIntegrationConfig
        if not isinstance(self.integration, DyadicIntegrationConfig):
            raise TypeError('Contrast integration must be DyadicIntegrationConfig')
        if self.integration.spatial_rule == 'factored_spd_tail_v1':
            raise NotImplementedError('Factored SPD tails do not support paired regional contrasts')
        if self.integration.spatial_rule == 'factored_spd_pair_tail_v1':
            from .dyadic_green import _validate_factored_plane_medium
            if (self.exterior.frequency != self.interior.frequency
                    or self.exterior.epsilon_r.device != self.interior.epsilon_r.device
                    or self.exterior.epsilon_r.dtype != self.interior.epsilon_r.dtype):
                raise ValueError('Factored spatial pairs require common frequency, device and precision')
            _validate_factored_plane_medium(self.exterior)
            _validate_factored_plane_medium(self.interior)
        object.__setattr__(self, 'mode_cache', SpectralContrastCache(self.exterior, self.interior, self.cache_max_bytes))

    def evaluate_regular(self, displacements):
        return evaluate_spatial_green_difference(displacements, self.exterior, self.interior,
            self.integration, regular_part=True, mode_cache=self.mode_cache)

    def evaluate_static(self, displacements, *, constitutive):
        return electrostatic_kernel_difference(displacements, self.exterior, self.interior, constitutive=constitutive)

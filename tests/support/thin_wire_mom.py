"""CPU-only thin-wire integral-equation diagnostic, never a runtime backend.

Straight wires: Hallen, piecewise-sinusoidal basis and point matching, with the
circumferentially averaged *exact cylindrical* kernel

    K(z) = (1/2pi) integral exp(-jk R)/(4pi R) dphi,
    R**2 = z**2 + 4*a**2*sin(phi/2)**2.

The static singular part is integrated analytically along each linear element;
only the smooth dynamic remainder uses Gaussian quadrature. Open ends carry zero
current. Finite gaps impose E_gap=1/g over (-g/2,g/2) on the integration
contour, including the equivalent gap displacement current. Input current is
the reciprocal gap-average current. Delta feeds impose unit integrated field
and sample the central current. This is an impressed-field thin-wire feed,
not a resolved cylindrical end-face electrostatic model.

Circular loops: closed-path Pocklington EFIE, periodic rooftop Galerkin basis,
with reduced kernel exp(-jk*sqrt(|r-r'|**2+a**2))/(4pi*sqrt(...)). A uniform gap
field is tested against the rooftops with its exact physical width; the gap is
an impressed-field interval on the closed path (not a missing conductor arc).
This feed idealization is explicitly different from a physical cut loop.

Assumptions: PEC, vacuum, circular cross section, a/lambda <= 0.002 and
(for loops) a/R_loop <= 0.02. This includes lambda/800 and lambda/2000 over
1.6--3.2 GHz. These are thin-wire modeling bounds, NOT an accuracy certificate.
The delta feed is mesh regularized: exact-kernel delta feed current has a
logarithmic singularity; reduced-kernel equations have small-mesh pathologies.
Never interpret a passing adjacent-mesh check as continuum delta-gap convergence.
Qualification (tests/support/test_thin_wire_mom.py): the straight-dipole oracle
is checked against published thin-dipole tables (R. W. P. King, The Theory of
Linear Antennas, Harvard University Press, 1956: Tables 30.1, 30.8-30.10 and
30.15) at Omega = 12.5, 15, 20, half-wave and full-wave lengths, so ``dipole``
results are marked qualified. The loop oracles have no verified table value yet
and stay unqualified. No fitting, empirical correction, or external solver data
is used.

Background: E. Hallen, Nova Acta Regiae Soc. Sci. Upsaliensis (4) 11 (1938);
G. Fikioris and T. T. Wu, 'On the application of numerical methods to Hallen's
equation', IEEE TAP 49 (2001), 383--392, doi:10.1109/8.918612.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import c as C0, epsilon_0 as EPS0, mu_0 as MU0
from scipy.linalg import solve

ETA0 = np.sqrt(MU0 / EPS0)
# Per-geometry qualification against published tables (see the module docstring).
DIPOLE_QUALIFIED = True
LOOP_QUALIFIED = False


@dataclass(frozen=True)
class MoMResult:
    impedance: complex
    segments: int
    gap: float
    formulation: str
    feed_model: str
    qualified: bool


def _validate(length, radius, frequency, gap, segments):
    if not all(np.isfinite(x) for x in (length, radius, frequency, gap)):
        raise ValueError("Parameters must be finite")
    if min(length, radius, frequency) <= 0 or not 0 <= gap < length:
        raise ValueError("Require positive geometry/frequency and 0 <= gap < length")
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 16 or segments % 2:
        raise ValueError("segments must be an even integer >= 16")
    if radius * frequency / C0 > 0.002:
        raise ValueError("Outside the stated thin-wire a/lambda <= 0.002 range")


@lru_cache(maxsize=16)
def _gauss(n):
    t, w = leggauss(n)
    return (t + 1) / 2, w / 2


def _hat_kernel(obs, edges, radius, k, angular_order=64, order=12):
    """Integrate exact cylindrical Green function times the two element hats."""
    u = edges[:-1][None, :] - obs[:, None]
    v = edges[1:][None, :] - obs[:, None]
    width = v - u
    left = np.zeros(u.shape, dtype=complex)
    right = np.zeros_like(left)
    angles, aw = _gauss(angular_order)
    for angle, weight in zip(np.pi * angles, aw):
        b = 2 * radius * np.sin(angle / 2)
        primitive = np.arcsinh(v / b) - np.arcsinh(u / b)
        moment = np.sqrt(v * v + b * b) - np.sqrt(u * u + b * b)
        left += weight * (v * primitive - moment) / width
        right += weight * (moment - u * primitive) / width
    # (exp(-jkR)-1)/R has a finite limit at the coincident point.
    angles, aw = _gauss(8)
    nodes, weights = _gauss(order)
    for angle, angle_weight in zip(np.pi * angles, aw):
        b = 2 * radius * np.sin(angle / 2)
        for t, weight in zip(nodes, weights):
            distance = np.sqrt((u + t * width)**2 + b*b)
            pleft = np.sin(k*width*(1-t))/np.sin(k*width)
            pright = np.sin(k*width*t)/np.sin(k*width)
            factor = width*weight*angle_weight/distance
            dynamic = np.expm1(-1j*k*distance)
            left += ((pleft-(1-t))+dynamic*pleft)*factor
            right += ((pright-t)+dynamic*pright)*factor
    return left / (4*np.pi), right / (4*np.pi)


def dipole(*, length: float, radius: float, frequency: float,
           gap: float = 0.0, segments: int = 320) -> MoMResult:
    """Unit-voltage Hallen PWS solution; length is the outer tip-to-tip distance."""
    _validate(length, radius, frequency, gap, segments)
    k = 2*np.pi*frequency/C0
    edges = np.linspace(0, length/2, segments//2 + 1)
    obs = edges.copy()
    left, right = _hat_kernel(obs, edges, radius, k)
    # Mirror positive-arm basis onto the negative arm; even current symmetry.
    ml, mr = _hat_kernel(-obs, edges, radius, k)
    left += ml
    right += mr
    matrix = left.copy()
    matrix[:, 1:] += right[:, :-1]
    # Last (outer) current coefficient is zero; homogeneous constant is unknown.
    matrix = np.column_stack((matrix, -np.cos(k*obs)))
    if gap == 0:
        source = -1j / (2*ETA0) * np.sin(k*obs)
    else:
        convolution = np.where(obs < gap/2,
            (1-np.cos(k*gap/2)*np.cos(k*obs))/(k*k*gap),
            np.sin(k*obs)*np.sin(k*gap/2)/(k*k*gap))
        source = -1j*k/ETA0*convolution
    coefficients = solve(matrix, source)
    current = coefficients[0]
    if gap > 0:
        # Integrate the PWS current over the driven interval, split at knots.
        amplitudes = np.append(coefficients[:-1], 0)
        current = 0j
        nodes, weights = _gauss(8)
        for j in range(len(edges)-1):
            lo, hi = edges[j], min(edges[j+1], gap/2)
            if hi <= lo:
                break
            position = lo+(hi-lo)*nodes
            width = edges[j+1]-edges[j]
            values = (amplitudes[j]*np.sin(k*(edges[j+1]-position))
                      +amplitudes[j+1]*np.sin(k*(position-edges[j])))/np.sin(k*width)
            current += 2*(hi-lo)*np.dot(weights, values)/gap
    return MoMResult(complex(1/current), segments, gap,
                     "Hallen/exact-cylinder/PWS-point-match",
                     "delta (mesh regularized)" if gap == 0 else "uniform impressed interval, equivalent gap current",
                     DIPOLE_QUALIFIED)


def loop(*, circumference: float, radius: float, frequency: float,
         gap: float = 0.0, segments: int = 160, quadrature: int = 16) -> MoMResult:
    """Closed circular path EFIE with a uniform-field impressed gap interval.

    Periodic rooftop Galerkin matrix is circulant. Its first row is assembled
    from vector-potential and charge terms, then diagonalized by an FFT. The
    source and reciprocal current readout use the same normalized gap weighting.
    """
    _validate(circumference, radius, frequency, gap, segments)
    b = circumference / (2*np.pi)
    if radius/b > 0.02:
        raise ValueError("Outside the stated loop a/R <= 0.02 range")
    ds = circumference/segments
    k = 2*np.pi*frequency/C0
    omega = 2*np.pi*frequency
    # Two panels support each rooftop; derivatives are with respect to arc length.
    t, w = _gauss(quadrature)
    x = np.concatenate((-ds+ds*t, ds*t))
    weight = np.tile(w*ds, 2)
    shape = 1-np.abs(x)/ds
    derivative = np.repeat([1/ds, -1/ds], quadrature)
    separation = x[:, None, None] - x[None, :, None] - np.arange(segments)[None, None, :]*ds
    angle = separation/b
    distance = np.sqrt((2*b*np.sin(angle/2))**2 + radius**2)
    green = np.exp(-1j*k*distance)/(4*np.pi*distance)
    weights = weight[:, None]*weight[None, :]
    vector = np.einsum('ij,ijk->k', weights*shape[:, None]*shape[None, :], green*np.cos(angle))
    scalar = np.einsum('ij,ijk->k', weights*derivative[:, None]*derivative[None, :], green)
    row = 1j*omega*MU0*vector + scalar/(1j*omega*EPS0)
    if gap == 0:
        rhs = np.zeros(segments)
        rhs[0] = 1
    else:
        # Analytic integral of each rooftop over [-g/2,g/2], including wraps.
        rhs = np.zeros(segments)
        def primitive(y):
            return np.where(y < -ds, 0, np.where(y < 0, (y+ds)**2/(2*ds),
                   np.where(y < ds, ds/2+y-y*y/(2*ds), ds)))
        for shift in (-circumference, 0, circumference):
            centers = np.arange(segments)*ds + shift
            rhs += (primitive(gap/2-centers)-primitive(-gap/2-centers))/gap
    current = np.fft.ifft(np.fft.fft(rhs)/np.fft.fft(row))
    zin = 1/np.dot(rhs, current)
    return MoMResult(complex(zin), segments, gap, "Pocklington/closed-circle/reduced/rooftop-Galerkin",
                     "delta (mesh regularized)" if gap == 0 else "uniform impressed interval on closed path",
                     LOOP_QUALIFIED)


def loop_path(*, points, feed_segment: int, radius: float, frequency: float,
              finite_gap: bool = True, segments: int = 320, quadrature: int = 12) -> MoMResult:
    """Pocklington on the exact closed FDTD polygon, including its feed chord.

    ``points`` repeats the first point at the end; ``feed_segment`` is the
    straight source chord, whose physical length sets E=1/g. That source chord
    represents equivalent displacement current. The other segments represent
    PEC conduction current. Each source span is subdivided without changing the
    contour. Finite and delta excitation share this exact contour, and use
    reciprocal source/current testing. Delta is at the chord midpoint.

    Reduced cylindrical kernel; same thin-wire bounds as ``loop``. This general
    dense Galerkin assembly is slower than the ideal-circle circulant assembly.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4 or not np.array_equal(p[0], p[-1]):
        raise ValueError("points must be a closed [N+1,3] polygon")
    if not 0 <= feed_segment < len(p)-1:
        raise ValueError("feed_segment outside polygon")
    source_length = np.linalg.norm(np.diff(p, axis=0), axis=1)
    circumference = source_length.sum()
    gap = float(source_length[feed_segment])
    _validate(circumference, radius, frequency, gap, segments)
    if (source_length <= 0).any() or radius/(circumference/(2*np.pi)) > .02:
        raise ValueError("Invalid polygon or outside thin-loop curvature range")
    count = np.maximum(1, np.ceil(source_length/(circumference/segments)).astype(int))
    nodes = np.concatenate([p[j]+np.arange(n)[:, None]/n*(p[j+1]-p[j]) for j, n in enumerate(count)])
    vectors = np.roll(nodes, -1, axis=0)-nodes
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors/lengths[:, None]
    t, w = _gauss(quadrature)
    samples = nodes[:, None, :]+t[None, :, None]*vectors[:, None, :]
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=complex)
    omega = 2*np.pi*frequency
    k = omega/C0
    left_w, right_w = w*(1-t), w*t
    all_ids = np.arange(n)
    for begin in range(0, n, 4):
        ids = all_ids[begin:begin+4]
        separation = samples[ids, None, :, None, :]-samples[None, :, None, :, :]
        distance = np.sqrt(np.sum(separation**2, axis=-1)+radius**2)
        green = np.exp(-1j*k*distance)/(4*np.pi*distance)
        # ds ds' cancels the derivative factors in the scalar-potential term.
        scalar = np.einsum('mnij,i,j->mn', green, w, w)/(1j*omega*EPS0)
        multiplier = 1j*omega*MU0*(tangents[ids]@tangents.T)*lengths[ids, None]*lengths[None, :]
        for a, wa in enumerate((left_w, right_w)):
            for b, wb in enumerate((left_w, right_w)):
                potential = multiplier*np.einsum('mnij,i,j->mn', green, wa, wb)
                potential += scalar*(1 if a == b else -1)
                matrix[np.ix_((ids+a) % n, (all_ids+b) % n)] += potential
    rhs = np.zeros(n)
    begin = int(count[:feed_segment].sum())
    number = int(count[feed_segment])
    if finite_gap:
        for j in range(begin, begin+number):
            rhs[j] += lengths[j]/(2*gap)
            rhs[(j+1) % n] += lengths[j]/(2*gap)
    else:
        midpoint = begin+number//2
        if number % 2:
            rhs[midpoint] = .5
            rhs[(midpoint+1) % n] = .5
        else:
            rhs[midpoint % n] = 1.
    current = solve(matrix, rhs)
    return MoMResult(complex(1/np.dot(rhs, current)), n, gap if finite_gap else 0.,
                     "Pocklington/exact-FDTD-polygon/reduced/rooftop-Galerkin",
                     "uniform straight gap E=1/g" if finite_gap else "delta at same chord midpoint (mesh regularized)",
                     LOOP_QUALIFIED)

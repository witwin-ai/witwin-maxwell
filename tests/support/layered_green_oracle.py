"""Independent CPU fp64 references; never imported by the layered runtime.

O1 uses input-impedance transfer, not the CUDA reflection recursion. O2 integrates
the unextracted spectrum adaptively on the real axis, then on decaying Hankel tails.
O3 gives the outgoing homogeneous and grounded-image limits.
"""

from bisect import bisect_right

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import hankel1, hankel2, jv


EPS0 = 8.8541878128e-12
C0 = 299792458.0


def homogeneous(epsilon, frequency, rho, zs, zo, *, ground=None):
    k = 2 * np.pi * frequency / C0 * np.sqrt(complex(epsilon))
    r = np.hypot(rho, zo - zs)
    g = np.exp(1j * k * r) / (4 * np.pi * r)
    if ground is not None:
        r = np.hypot(rho, zs + zo - 2 * ground)
        g -= np.exp(1j * k * r) / (4 * np.pi * r)
    return np.array((g, g / epsilon))


def spectral(stack, frequency, q, zs, zo):
    """O1: unit shunt-current source feeding the two input impedances."""
    if zo < zs:
        zs, zo = zo, zs
    edges = stack.interfaces()
    media = (stack.bottom, *stack.layers, stack.top)
    w = 2 * np.pi * frequency
    k0 = w / C0
    epsilon = np.array([complex(m.eps_r) + 1j * m.sigma / (w * EPS0)
                        if m != "pec" else 1.0 for m in media], dtype=np.complex128)
    kz = np.sqrt(k0**2 * epsilon - complex(q)**2)
    kz = np.where(kz.imag < 0, -kz, kz)
    source, observer = bisect_right(edges, zs), bisect_right(edges, zo)

    def transfer(zc, load, phase):
        t = np.tan(phase)
        return zc * (load - 1j * zc * t) / (zc - 1j * load * t)

    values = []
    for tm in (False, True):
        zc = kz / (k0**2 * epsilon) if tm else 1 / kz

        def input_impedance(layer, z, direction):
            exterior = len(media) - 1 if direction > 0 else 0
            load = 0j if media[exterior] == "pec" else zc[exterior]
            # Move a load impedance through every intervening finite section.
            for section in range(exterior - direction, layer, -direction):
                load = transfer(zc[section], load, kz[section] * (edges[section] - edges[section - 1]))
            if layer != exterior:
                boundary = edges[layer] if direction > 0 else edges[layer - 1]
                load = transfer(zc[layer], load, kz[layer] * abs(boundary - z))
            return load

        upper = input_impedance(source, zs, 1)
        lower = input_impedance(source, zs, -1)
        voltage = 1j * upper * lower / (upper + lower)
        position = zs
        for section in range(source, observer + 1):
            target = zo if section == observer else edges[section]
            if target != position:
                load = input_impedance(section, target, 1)
                decay = np.exp(1j * kz[section] * (target - position))
                # Stable transfer-matrix voltage ratio; no growing exponential.
                ratio = zc[section] / load
                voltage *= 2 * decay / ((1 + ratio) + (1 - ratio) * decay**2)
            position = target
        values.append(voltage)
    return np.array((values[0], k0**2 * (values[0] - values[1]) / complex(q)**2))


def spatial(stack, frequency, rho, zs, zo, *, tolerance=2e-10):
    """O2: adaptive, unextracted integration using the impedance-transfer spectrum.

    The head stays on the real axis in lossy media. Beyond Q=max(4*k_max,48/rho),
    split J0 into two Hankel functions and close upward/downward. Integrate the
    paired tails adaptively in u=rho*Im(q), up to 56. This independent path uses
    neither CUDA nodes, static images, C3 extraction, nor the CUDA TL recursion.
    Vector quadrature computes both components together. Returned errors include
    its estimates and the exponentially small omitted endpoint envelope.
    """
    separation = abs(zo - zs)
    if separation == 0:
        raise ValueError("The unextracted oracle requires a vertical offset")
    media = (stack.bottom, *stack.layers, stack.top)
    w = 2 * np.pi * frequency
    wave_scales = [w / C0 * abs(np.sqrt(complex(m.eps_r) + 1j * m.sigma / (w * EPS0)))
                   for m in media if m != "pec"]
    end = max(48 / rho, 4 * max(wave_scales))
    # Geometric partition plus material branch-point neighborhoods. These are
    # real-axis integration subintervals, unrelated to the CUDA contour panels.
    points = np.unique(np.clip(np.r_[0, np.geomspace(min(wave_scales) / 16, end, 24),
                                       np.array(wave_scales) * 0.99,
                                       np.array(wave_scales), np.array(wave_scales) * 1.01, end], 0, end))
    result, errors = np.zeros(2, complex), np.zeros(2)
    def integrand(q):
        return spectral(stack, frequency, q, zs, zo) * jv(0, q * rho) * q / (2 * np.pi)

    for a, b in zip(points[:-1], points[1:]):
        value, error = quad_vec(integrand, a, b, epsabs=1e-10, epsrel=tolerance)
        result += value
        errors += error

    def tails(u):
        upper, lower = end + 1j * u / rho, end - 1j * u / rho
        return 1j / (4 * np.pi * rho) * (
            spectral(stack, frequency, upper, zs, zo) * upper * hankel1(0, upper * rho) -
            spectral(stack, frequency, lower, zs, zo) * lower * hankel2(0, lower * rho))

    value, error = quad_vec(tails, 0, 56, epsabs=1e-10, epsrel=tolerance)
    result += value
    errors += error + abs(tails(56))
    return result, errors


def vertical_spectral(stack, frequency, q, zs, zo):
    """Independent impedance transfer plus analytic TL endpoint derivatives.

    Components (Gzz, Gzrho, Grhoz, Kphi) use J0, J1, J1, J0 respectively;
    each spectrum includes the q factor of a radial cross derivative.
    """
    edges = stack.interfaces()
    media = (stack.bottom, *stack.layers, stack.top)
    w = 2 * np.pi * frequency
    k0 = w / C0
    epsilon = np.array([complex(m.eps_r) + 1j * m.sigma / (w * EPS0)
                        if m != "pec" else 1 for m in media])
    k = np.sqrt(k0**2 * epsilon - complex(q)**2)
    k = np.where(k.imag < 0, -k, k)
    a, b = bisect_right(edges, zs), bisect_right(edges, zo)
    v = spectral(stack, frequency, q, zs, zo)
    voltages = (v[0], v[0] - q**2 / k0**2 * v[1])
    derivatives = []
    for tm, voltage in enumerate(voltages):
        zc = k / (k0**2 * epsilon) if tm else 1 / k

        def input_impedance(layer, z, upward):
            direction = 1 if upward else -1
            exterior = len(media) - 1 if upward else 0
            load = 0j if media[exterior] == "pec" else zc[exterior]
            for section in range(exterior - direction, layer, -direction):
                phase = k[section] * (edges[section] - edges[section - 1])
                tangent = np.tan(phase)
                load = zc[section] * (load - 1j * zc[section] * tangent) / (
                    zc[section] - 1j * load * tangent)
            if layer != exterior:
                boundary = edges[layer] if upward else edges[layer - 1]
                tangent = np.tan(k[layer] * abs(boundary - z))
                load = zc[layer] * (load - 1j * zc[layer] * tangent) / (
                    zc[layer] - 1j * load * tangent)
            return load

        ds = (1j if zs > zo else -1j) * k[a] * zc[a] / input_impedance(a, zs, zs > zo)
        dz = (1j if zo >= zs else -1j) * k[b] * zc[b] / input_impedance(b, zo, zo >= zs)
        derivatives.append((voltage * ds, voltage * dz, voltage * ds * dz))
    hs, hz, hsz = derivatives[0]
    es, ez, esz = derivatives[1]
    scalar_s = k0**2 / q**2 * (hs - es)
    scalar_z = k0**2 / q**2 * (hz - ez)
    scalar_sz = k0**2 / q**2 * (hsz - esz)
    # A real-axis oracle must also avoid subtracting two O(q/k0^2)
    # contributions to the vertical static limit. Expand the coefficient
    # algebraically after all material wave scales are far behind the contour.
    if abs(q) > 8 * max(abs(k0 * np.sqrt(epsilon))):
        coefficient = (k0**2 * (epsilon[a] + epsilon[b]) -
                       k0**4 * epsilon[a] * epsilon[b] / q**2) / (k[a]**2 * k[b]**2)
        zz = hsz / q**2 + esz * coefficient
        zr = (hz - k0**2 * epsilon[b] * ez / k[b]**2) / q
        rz = -(hs - k0**2 * epsilon[a] * es / k[a]**2) / q
    else:
        zz = scalar_sz / k0**2 + q**2 * esz / (k[a]**2 * k[b]**2)
        zr = q * (scalar_z / k0**2 - ez / k[b]**2)
        rz = -q * (scalar_s / k0**2 - es / k[a]**2)
    return np.array((zz, zr, rz, v[1]))


def vertical_spatial(stack, frequency, rho, zs, zo, *, tolerance=2e-9):
    """Unextracted lossy real-axis quadrature with independent Hankel tails."""
    if zs == zo:
        raise ValueError("The unextracted oracle requires a vertical offset")
    media = (stack.bottom, *stack.layers, stack.top)
    w = 2 * np.pi * frequency
    scales = [w / C0 * abs(np.sqrt(complex(m.eps_r) + 1j * m.sigma / (w * EPS0)))
              for m in media if m != "pec"]
    end = max(48 / rho, 4 * max(scales))
    points = np.unique(np.clip(np.r_[0, np.geomspace(min(scales) / 16, end, 24),
                                    np.array(scales) * .99, scales,
                                    np.array(scales) * 1.01, end], 0, end))
    orders = np.array((0, 1, 1, 0))
    def integrand(q):
        return vertical_spectral(stack, frequency, q, zs, zo) * jv(orders, q * rho) * q / (2 * np.pi)
    result, error = np.zeros(4, complex), np.zeros(4)
    for left, right in zip(points[:-1], points[1:]):
        value, estimate = quad_vec(integrand, left, right, epsabs=1e-9, epsrel=tolerance)
        result += value
        error += estimate

    def tails(u):
        upper, lower = end + 1j * u / rho, end - 1j * u / rho
        return 1j / (4 * np.pi * rho) * (
            vertical_spectral(stack, frequency, upper, zs, zo) * upper * hankel1(orders, upper * rho) -
            vertical_spectral(stack, frequency, lower, zs, zo) * lower * hankel2(orders, lower * rho))
    value, estimate = quad_vec(tails, 0, 56, epsabs=1e-9, epsrel=tolerance)
    return result + value, error + estimate + abs(tails(56))

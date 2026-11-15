"""Transmission-line characterization and port de-embedding for layered networks.

Conventions: ``exp(-i*omega*t)``; a line wave travels as ``exp(-gamma z)`` with
``gamma = alpha - i*beta`` (``alpha, beta >= 0``), i.e. as ``exp(i k z)`` with
the complex wavenumber ``k = i*gamma = beta + i*alpha`` that
``NetworkData.shift_reference_planes(propagation_constants=k)`` uses. ABCD
matrices use ``V1 = A V2 + B I2, I1 = C V2 + D I2`` with ``I2`` leaving port 2.

Two-line method: two lines that differ only in length by ``delta`` and share
identical reciprocal port error boxes E give
``M = A_long A_short^-1 = E L(delta) E^-1``, so ``tr(M)/2 = cosh(gamma*delta)``
exactly, whatever the port discontinuity. ``sqrt(M12/M21)`` equals the line
impedance only up to the error box (exact when E is the identity); it is
reported as ``impedance_two_line`` with that caveat.
"""

import math

import torch

from .extraction import port_matrix

C0 = 299792458.0


def abcd(z):
    """``[..., 2, 2]`` ABCD matrix of a two-port impedance matrix."""
    z11, z12, z21, z22 = z[..., 0, 0], z[..., 0, 1], z[..., 1, 0], z[..., 1, 1]
    return torch.stack((torch.stack((z11, z11 * z22 - z12 * z21), -1),
                        torch.stack((torch.ones_like(z11), z22), -1)), -2) / z21[..., None, None]


def impedance_from_abcd(t):
    """Inverse of ``abcd``."""
    a, b, c, d = t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1]
    return torch.stack((torch.stack((a, a * d - b * c), -1), torch.stack((torch.ones_like(a), d), -1)), -2) / c[..., None, None]


def line_abcd(gamma, impedance, length):
    """ABCD of a uniform line ``[F, 2, 2]``."""
    ch, sh = torch.cosh(gamma * length), torch.sinh(gamma * length)
    return torch.stack((torch.stack((ch, impedance * sh), -1), torch.stack((sh / impedance, ch), -1)), -2)


def _branch(values, frequencies):
    """Choose gamma*delta among +-acosh + 2 pi i n: Im <= 0 and continuity in frequency.

    The lowest frequency needs beta*delta < pi; later samples follow the
    prediction beta proportional to frequency from the previous sample.
    Decisions are detached; the chosen branch stays differentiable.
    """
    chosen = []
    previous = None
    for value, frequency in zip(values, frequencies.tolist()):
        x = complex(value.detach())
        candidates = [(s, n) for s in (1, -1) for n in range(-64, 65)]
        if previous is None:
            admissible = [(s, n) for s, n in candidates if (s * x + 2j * math.pi * n).imag <= 1e-12]
            s, n = min(admissible, key=lambda c: abs((c[0] * x + 2j * math.pi * c[1]).imag))
        else:
            target = previous[1] * frequency / previous[0]
            s, n = min(candidates, key=lambda c: abs((c[0] * x + 2j * math.pi * c[1]).imag - target))
        chosen.append(s * value + 2j * math.pi * n)
        previous = (frequency, (s * x + 2j * math.pi * n).imag)
    return torch.stack(chosen)


def line_parameters(short, long, length_difference, *, ports=(0, 1), others="open"):
    """Propagation constant and effective permittivity from two lines (``long`` - ``short`` = delta).

    ``ports`` selects the two line ports of each network (other ports open or
    shorted per ``others``). Returns a dict of ``[F]`` tensors: ``gamma``,
    ``wavenumber`` (``k = i gamma``), ``eps_eff = (beta/k0)^2``,
    ``attenuation_np_per_m`` (alpha), ``impedance_two_line`` (see module
    docstring) and ``reciprocity`` (``|det M - 1|``).
    """
    if not torch.equal(short.frequencies, long.frequencies):
        raise ValueError("line_parameters needs both lines at identical frequencies")
    if not length_difference > 0:
        raise ValueError("length_difference must be positive (long minus short line length)")
    m = (abcd(port_matrix(long, ports, others=others)) @
         torch.linalg.inv(abcd(port_matrix(short, ports, others=others))))
    half_trace = (m[:, 0, 0] + m[:, 1, 1]) / 2
    gamma = _branch(torch.acosh(half_trace), short.frequencies) / length_difference
    k0 = 2 * math.pi * short.frequencies.to(torch.float64) / C0
    impedance = torch.sqrt(m[:, 0, 1] / m[:, 1, 0])
    impedance = torch.where(impedance.real < 0, -impedance, impedance)
    return dict(gamma=gamma, wavenumber=1j * gamma, eps_eff=(-gamma.imag / k0) ** 2,
                attenuation_np_per_m=gamma.real, impedance_two_line=impedance,
                reciprocity=(torch.linalg.det(m) - 1).abs())


def open_short(measured, open_, short):
    """Open-short de-embedding of pad (shunt) and feed (series) parasitics, all ports at once.

    ``Y_dut = ((Y_m - Y_o)^-1 - (Y_s - Y_o)^-1)^-1``: the open standard (DUT
    removed) carries the shunt pad/feed admittance; the short standard (DUT
    replaced by a short to the reference at the DUT reference planes) adds the
    series feed impedance. Exact for a lumped shunt-then-series fixture;
    distributed feeds leave an O((beta l)^2) residue. Returns NetworkData with
    the measured ports and references. Differentiable.
    """
    for standard in (open_, short):
        if standard.port_names != measured.port_names or not torch.equal(standard.frequencies, measured.frequencies):
            raise ValueError("open_short standards need the measured ports and frequencies")
    y_open = open_.to_y()
    shunt_removed = torch.linalg.inv(measured.to_y() - y_open)
    series = torch.linalg.inv(short.to_y() - y_open)
    from ...network import NetworkData
    metadata = dict(measured.metadata, deembedding="open_short")
    return NetworkData.from_z(frequencies=measured.frequencies, z=shunt_removed - series, z0=measured.z0,
                              port_names=measured.port_names, metadata=metadata)


def open_short_standards(scene, *, dut, short_region):
    """Auto-generated (open, short) standard Scenes of a device scene.

    ``dut`` names the device Structures; the open standard drops them and
    keeps pads, feeds, grounds and ports. The short standard also drops them
    and fills ``short_region`` = (x0, x1, y0, y1, z) with PEC metal, which must
    join every feed end at its DUT reference plane to the port reference
    conductor (for example the opening of a ground ring). Ports are shared
    unchanged.
    """
    from ...geometry import PlanarPEC
    from ...media import Material
    from ...scene import Structure
    names = set(dut)
    present = {s.name for s in scene.structures}
    if not names or not names <= present:
        raise ValueError(f"open_short_standards: unknown device structures {sorted(names - present)}")
    x0, x1, y0, y1, z = map(float, short_region)
    if not (x0 < x1 and y0 < y1):
        raise ValueError("short_region needs x0 < x1 and y0 < y1")
    standards = []
    for shorted in (False, True):
        structures = tuple(s for s in scene.structures if s.name not in names)
        if shorted:
            structures += (Structure(PlanarPEC([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], normal_axis="z",
                                               coordinate=z), Material.pec(), name="deembed_short"),)
        standards.append(scene.updated(structures=structures))
    return tuple(standards)


def _cross(a, b):
    """2D cross product of ``[..., 2]`` projective vectors (zero when parallel)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _eigenvector(m, value):
    """Eigenvector of ``[F, 2, 2]`` m for eigenvalue ``value``, from the better-conditioned row."""
    first = torch.stack((m[:, 0, 1], value - m[:, 0, 0]), -1)
    second = torch.stack((value - m[:, 1, 1], m[:, 1, 0]), -1)
    better = first.abs().sum(-1) >= second.abs().sum(-1)
    return torch.where(better[:, None], first, second)


def _port_vector(s, z0):
    """Projective [V, I] of a one-port reflection s against reference impedance z0: [z0 (1+s), 1-s]."""
    return torch.stack((z0 * (1 + s), 1 - s), -1)


def trl(thru, reflect, line, line_length, *, reflect_kind="short", feed_lengths=None, line_impedance=None,
        ports=(0, 1), others="open"):
    """Thru-reflect-line calibration of the two port error boxes of a layered fixture.

    Standards (same frequencies, port 1 left, port 2 right): ``thru`` joins the
    two feeds at the reference plane (zero length), ``line`` inserts
    ``line_length`` of the feed line there, and ``reflect`` terminates both
    feeds at the reference planes with the same unknown high reflection
    (``reflect_kind`` "short" or "open" selects the root, Gamma near -1 or +1).
    The measured device is then ``M = X D Y`` (ABCD) with the left box X (port
    1 to the left plane) and the right box Y (right plane to port 2). The line
    eigen-decomposition gives gamma and X up to two column scales; the reflect
    fixes their ratio and reciprocity (det X = 1) their product.

    TRL alone references the boxes to the unknown line impedance. The absolute
    level comes from one of:

    - ``feed_lengths=(l1, l2)``: physical port-plane to reference-plane feed
      lengths. Each box is a lumped port discontinuity P (series then shunt,
      D(P) = 1: the port voltage is the line voltage at the port plane up to
      the port's series drop) followed by the uniform feed line; the D = 1
      condition on both sides fixes the impedance level (``port_model_residual``
      = the left/right disagreement, zero when the model holds);
    - ``line_impedance``: a given characteristic impedance ([F] or scalar).

    Returns a dict of ``[F]`` / ``[F, 2, 2]`` tensors: ``left``, ``right`` (ABCD
    error boxes), ``gamma``, ``eps_eff``, ``line_impedance``,
    ``port_model_residual`` (or None), ``reciprocity`` (``|det Y - 1|``) and
    ``reflect_gamma`` (the reflect against the line impedance). Root and branch
    decisions are detached; the values are differentiable.
    """
    networks = (thru, reflect, line)
    if any(not torch.equal(n.frequencies, thru.frequencies) for n in networks):
        raise ValueError("trl needs all standards at identical frequencies")
    if not line_length > 0:
        raise ValueError("line_length must be positive")
    if reflect_kind not in ("short", "open"):
        raise ValueError("reflect_kind must be 'short' or 'open'")
    if (feed_lengths is None) == (line_impedance is None):
        raise ValueError("trl needs exactly one of feed_lengths and line_impedance for the impedance level")
    frequencies = thru.frequencies
    reference = 50.0
    mt = abcd(port_matrix(thru, ports, others=others))
    ml = abcd(port_matrix(line, ports, others=others))
    m = ml @ torch.linalg.inv(mt)
    gamma_length = _branch(torch.acosh((m[:, 0, 0] + m[:, 1, 1]) / 2), frequencies)
    gamma = gamma_length / line_length
    v1, v2 = _eigenvector(m, torch.exp(gamma_length)), _eigenvector(m, torch.exp(-gamma_length))
    s = reflect.s[:, list(ports)][:, :, list(ports)]
    z0 = reflect.z0[:, list(ports)].to(s.dtype)
    left = _port_vector(s[:, 0, 0], z0[:, 0])
    right = _port_vector(s[:, 1, 1], z0[:, 1]) * torch.tensor([1, -1], dtype=s.dtype, device=s.device)
    inverse_thru = torch.linalg.inv(mt)
    u1, u2 = ((inverse_thru @ v[..., None])[..., 0] for v in (v1, v2))
    p1 = -_cross(left, v1) / _cross(left, v2)
    p2 = -_cross(right, u2) / _cross(right, u1)
    rho = torch.sqrt(p1 * p2)
    flip = (rho.detach().real < 0) if reflect_kind == "open" else (rho.detach().real > 0)
    rho = torch.where(flip, -rho, rho)
    ratio = p1 / rho
    vectors = torch.stack((v1, v2), -1)
    alpha = torch.sqrt(-2 / (reference * ratio * torch.linalg.det(vectors)))
    w_inverse = torch.tensor([[1, reference], [1, -reference]], dtype=s.dtype, device=s.device) / 2
    x = vectors * torch.stack((alpha, alpha * ratio), -1)[:, None, :] @ w_inverse
    y = torch.linalg.inv(x) @ mt
    if feed_lengths is None:
        impedance = torch.as_tensor(line_impedance, dtype=s.dtype, device=s.device).expand(len(frequencies))
        scale = torch.sqrt(impedance / reference)
        residual = None
    else:
        l1, l2 = feed_lengths
        base = torch.full_like(gamma, reference)
        q = x @ torch.linalg.inv(line_abcd(gamma, base, l1))
        r = torch.linalg.inv(line_abcd(gamma, base, l2)) @ y
        n_left, n_right = 1 / q[:, 1, 1], 1 / r[:, 0, 0]
        scale = n_left * torch.sqrt(n_right / n_left)
        impedance = scale ** 2 * reference
        residual = (n_left / n_right - 1).abs()
    zeros = torch.zeros_like(scale)
    n = torch.stack((torch.stack((scale, zeros), -1), torch.stack((zeros, 1 / scale), -1)), -2)
    left_box, right_box = x @ torch.linalg.inv(n), n @ y
    k0 = 2 * math.pi * frequencies.to(torch.float64) / C0
    return dict(left=left_box, right=right_box, gamma=gamma, wavenumber=1j * gamma, eps_eff=(-gamma.imag / k0) ** 2,
                line_impedance=impedance, port_model_residual=residual,
                reciprocity=(torch.linalg.det(y) - 1).abs(), reflect_gamma=rho)


def deembed(measured, left, right, *, ports=(0, 1), others="open"):
    """Remove ABCD error boxes from a measured two-port: ``D = left^-1 M right^-1`` (NetworkData, differentiable).

    ``left``/``right`` are ``[F, 2, 2]`` boxes (``trl``, or any cascade such as
    ``line_abcd`` feeds). Per port, ``[V_m, I_m] = E [V_d, I_d]`` with
    ``E_1 = left`` and ``E_2 = J right^-1 J`` (``J = diag(1, -1)``), so
    ``Z_D = (A - Z_M C)^-1 (Z_M D - B)`` with diagonal per-port A, B, C, D: no
    division by ``z21``, hence uncoupled standards (open, short, reflect) also
    de-embed. The result keeps the measured port names, ``z0`` and
    frequencies; ``metadata["deembedding"] = "error_boxes"``.
    """
    from ...network import NetworkData
    z = port_matrix(measured, ports, others=others)
    j = torch.diag(torch.tensor([1, -1], dtype=z.dtype, device=z.device))
    boxes = torch.stack((left, j @ torch.linalg.inv(right) @ j), 1)
    a, b, c, d = (torch.diag_embed(boxes[:, :, r, q]) for r, q in ((0, 0), (0, 1), (1, 0), (1, 1)))
    device = torch.linalg.solve(a - z @ c, z @ d - b)
    names = [measured.port_names[p] for p in ports]
    return NetworkData.from_z(frequencies=measured.frequencies, z=device,
                              z0=measured.z0[:, list(ports)], port_names=names,
                              metadata=dict(measured.metadata, deembedding="error_boxes"))


def trl_standards(scene, *, dut, planes, line_length, axis="x"):
    """Auto-generated TRL standard Scenes (thru, reflect, line) of a two-port fixture along ``axis``.

    ``dut`` names the device Structures between the reference planes
    ``planes = (a, b)`` (a < b along ``axis``). Every other Structure must be an
    axis-aligned rectangular PlanarPEC lying wholly on one side of the planes
    (a feed, a pad) or spanning both (a ground); ports must lie outside (a, b).

    - thru: the device is removed and everything beyond ``b`` moves by
      ``a - b``, so the two feeds join at the reference plane (spanning
      structures shrink by the same amount);
    - reflect: the device is removed and nothing moves: both feeds end open at
      their reference planes (``trl(..., reflect_kind="open")``);
    - line: as thru, but moved by ``line_length + a - b``, and each feed ending at
      ``a`` is continued by a copy of its cross-section over the inserted
      ``line_length``, which must meet a feed starting at ``b`` with the same
      cross-section.

    Returns ``dict(thru, reflect, line, line_length, feed_lengths)`` where
    ``feed_lengths`` are the distances from port 1 / port 2 to their planes (the
    ``trl(feed_lengths=...)`` input).
    """
    from witwin.core import Box
    from ...geometry import PlanarPEC
    from ...ports import LumpedPort
    from ...scene import Structure
    if axis not in ("x", "y"):
        raise ValueError("trl_standards axis must be 'x' or 'y'")
    k = "xy".index(axis)
    a, b = map(float, planes)
    if not (a < b and line_length > 0):
        raise ValueError("trl_standards needs planes a < b and a positive line_length")
    names = set(dut)
    if not names or not names <= {s.name for s in scene.structures}:
        raise ValueError(f"trl_standards: unknown device structures {sorted(names - {s.name for s in scene.structures})}")
    if len(scene.ports) != 2:
        raise ValueError("trl_standards needs a two-port scene")

    def bounds(structure):
        g = structure.geometry
        if not isinstance(g, PlanarPEC) or g.normal_axis != "z" or g.holes or len(g.vertices) != 4:
            raise ValueError(f"trl_standards: structure {structure.name!r} must be a rectangular z-normal PlanarPEC")
        v = g.vertices.detach().to(torch.float64).cpu()
        lo, hi = v.min(0).values.tolist(), v.max(0).values.tolist()
        if sorted(map(tuple, v.tolist())) != sorted({(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])}):
            raise ValueError(f"trl_standards: structure {structure.name!r} must be an axis-aligned rectangle")
        return lo, hi

    def rebuilt(structure, lo, hi, name=None):
        g = structure.geometry
        vertices = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
        return Structure(PlanarPEC(vertices, normal_axis="z", coordinate=g.coordinate,
                                   physical_thickness=g.physical_thickness), structure.material,
                         name=structure.name if name is None else name)

    kept = [(s, *bounds(s)) for s in scene.structures if s.name not in names]
    for s, lo, hi in kept:
        if not (hi[k] <= a or lo[k] >= b or (lo[k] <= a and hi[k] >= b)):
            raise ValueError(f"trl_standards: structure {s.name!r} crosses a reference plane without spanning both")
    positions = []
    for port in scene.ports:
        points = [float(port.positive[k]), float(port.negative[k])]
        if any(a < p < b for p in points) or (min(points) < a < max(points)) or (min(points) < b < max(points)):
            raise ValueError(f"trl_standards: port {port.name!r} must lie outside the reference planes")
        positions.append(points[0])
    if not (positions[0] <= a and positions[1] >= b):
        raise ValueError("trl_standards: port 1 must lie before plane a and port 2 beyond plane b")

    def moved(target, fill):
        # Coordinates beyond b map to target + (x - b): plane b lands exactly on target.
        def place(x):
            return target + (x - b) if x >= b else x

        structures = []
        for s, lo, hi in kept:
            lo, hi = list(lo), list(hi)
            lo[k], hi[k] = place(lo[k]), place(hi[k])
            structures.append(rebuilt(s, lo, hi))
        if fill:
            for s, lo, hi in kept:
                if hi[k] != a:
                    continue
                partner = [t for t, l, h in kept if l[k] == b and l[1-k] == lo[1-k] and h[1-k] == hi[1-k]
                           and t.geometry.coordinate == s.geometry.coordinate]
                if not partner:
                    raise ValueError(f"trl_standards: feed {s.name!r} ending at plane a has no matching feed at plane b")
                start, stop = list(lo), list(hi)
                start[k], stop[k] = a, target
                structures.append(rebuilt(s, start, stop, name=f"{s.name}_trl_line"))
        ports = []
        for port in scene.ports:
            if float(port.positive[k]) < b:
                ports.append(port)
                continue
            def point(values):
                values = [float(v) for v in values]
                values[k] = place(values[k])
                return values

            aperture = port.voltage_aperture
            if aperture is not None:
                aperture = Box(position=torch.tensor(point(aperture.position.detach().to(torch.float64).cpu()),
                                                     dtype=torch.float64),
                               size=aperture.size.detach().to(torch.float64).cpu())
            ports.append(LumpedPort(port.name, positive=point(port.positive), negative=point(port.negative),
                                    voltage_path=port.voltage_path, reference_impedance=port.reference_impedance,
                                    voltage_aperture=aperture))
        return scene.updated(structures=tuple(structures), ports=tuple(ports))

    return dict(thru=moved(a, False), reflect=moved(b, False), line=moved(a + line_length, True),
                line_length=float(line_length), feed_lengths=(a - positions[0], positions[1] - b))

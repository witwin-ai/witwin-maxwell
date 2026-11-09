"""Native CUDA Sommerfeld tables for horizontal and declared vertical currents.

Only stack geometry, quadrature rules and static bounce-path topology are host
metadata. Material loss, path weights, spectral recursions, Bessel functions,
table derivatives, interpolation and admission are evaluated in CUDA kernels.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math

import torch

from ...layers import HalfSpace, LayerStack
from .build import build_extension


_EPS0 = 8.8541878128e-12
_C0 = 299792458.0

# Fixed Gauss-Legendre rule: host constants only; contour nodes are CUDA output.
_GAUSS_NODES = (
    -.9972638618494816, -.9856115115452684, -.9647622555875064, -.9349060759377397,
    -.8963211557660521, -.8493676137325700, -.7944837959679424, -.7321821187402897,
    -.6630442669302152, -.5877157572407623, -.5068999089322294, -.42135127613063533,
    -.33186860228212767, -.23928736225213706, -.1444719615827965, -.048307665687738324,
    .048307665687738324, .1444719615827965, .23928736225213706, .33186860228212767,
    .42135127613063533, .5068999089322294, .5877157572407623, .6630442669302152,
    .7321821187402897, .7944837959679424, .8493676137325700, .8963211557660521,
    .9349060759377397, .9647622555875064, .9856115115452684, .9972638618494816,
)
_GAUSS_WEIGHTS = (
    .007018610009470506, .016274394730905743, .025392065309262024, .034273862913021765,
    .042835898022226836, .05099805926237609, .058684093478535565, .06582222277636168,
    .07234579410884834, .07819389578707023, .08331192422694671, .08765209300440378,
    .09117387869576378, .09384439908080451, .09563872007927471, .09654008851472766,
    .09654008851472766, .09563872007927471, .09384439908080451, .09117387869576378,
    .08765209300440378, .08331192422694671, .07819389578707023, .07234579410884834,
    .06582222277636168, .058684093478535565, .05099805926237609, .042835898022226836,
    .034273862913021765, .025392065309262024, .016274394730905743, .007018610009470506,
)


def _image_paths(stack, planes, pairs, fmin, fmax, limit, *, vertical=False):
    """Enumerate static rays; CUDA evaluates their frequency-dependent weights.

    Only rays shorter than ``limit`` are extracted; longer bounce terms stay in
    the spectral remainder, where they decay like exp(-q d).

    An event is (incident region, next region, transmission). PEC uses next=-1.
    Bounds used for pruning are host metadata, never spatial Green values.
    """
    media = (stack.bottom, *stack.layers, stack.top)
    edges = stack.interfaces()

    def dielectric(index, frequency):
        m = media[index]
        return complex(m.eps_r) + 1j * m.sigma / (2 * math.pi * frequency * _EPS0)

    def coefficient(a, b, transmission):
        if media[b] == "pec":
            return 1.0
        ea, eb = dielectric(a, fmin), dielectric(b, fmin)
        rlow = (ea - eb) / (ea + eb)
        ea, eb = dielectric(a, fmax), dielectric(b, fmax)
        rhigh = (ea - eb) / (ea + eb)
        return max(abs(1 + rlow), abs(1 + rhigh)) if transmission else max(abs(rlow), abs(rhigh))

    distances, sources, offsets, events, multiplicities = [], [], [0], [], []
    pair_offsets = [0]
    directions = []
    # Plane-pair iteration here constructs small host image-path metadata only.
    for i, j in pairs:
        zs, zo = planes[i], planes[j]
        source = bisect_right(edges, zs)
        if stack.top == "pec" and zs == edges[-1]:
            source -= 1
        # Events commute in a scalar static network. Merge permutations at each
        # event depth, retaining an integer multiplicity instead of a ray tree.
        rays = {(source, zs, 1, (), 0, False, 1 if vertical else 0): (0.0, 1.0, 1),
                (source, zs, -1, (), 0, False, -1 if vertical else 0): (0.0, 1.0, 1)}
        arrivals = {}
        while rays:
            next_rays = {}

            def enqueue(key, distance, bound, count):
                old = next_rays.get(key)
                next_rays[key] = (distance, bound, count) if old is None else (
                    old[0], max(bound, old[1]), old[2] + count)

            for (region, z, direction, history, bounces, reflected_start, initial), (distance, bound, count) in rays.items():
                # With mu_r=1, TE transmits every dielectric interface unchanged.
                # A small TM transmission must not prune a surviving TE image.
                te_alive = all(event[1] < 0 or event[2] for event in history)
                boundary = (edges[region] if region < len(media) - 1 else math.inf) if direction > 0 else (
                    edges[region - 1] if region > 0 else -math.inf)
                hits = direction * (zo - z) >= 0 and direction * (boundary - zo) >= 0
                # At an interface include the incoming and reflected voltages;
                # the transmitted voltage at that same point must not be added.
                if hits and distance + abs(zo - z) < limit and (
                        zo != z or reflected_start or (not history and direction > 0)):
                    arrival = (round(distance + abs(zo - z), 16), history, -initial, direction) if vertical else (round(distance + abs(zo - z), 16), history)
                    arrivals[arrival] = arrivals.get(arrival, 0) + count
                if not math.isfinite(boundary):
                    continue
                nxt = region + direction
                traveled = distance + abs(boundary - z)
                if traveled >= limit:
                    continue
                pec = media[nxt] == "pec"
                if bounces < 64:
                    reflected = bound * coefficient(region, nxt, False)
                    if reflected >= 1e-6 or (pec and te_alive) or traveled == 0:
                        event = (region, -1 if pec else nxt, 0)
                        updated = tuple(sorted((*history, event)))
                        enqueue((region, boundary, -direction, updated, bounces + 1, True, initial),
                                traveled, reflected, count)
                if not pec:
                    transmitted = bound * coefficient(region, nxt, True)
                    if transmitted >= 1e-6 or te_alive:
                        updated = tuple(sorted((*history, (region, nxt, 1))))
                        enqueue((nxt, boundary, direction, updated, bounces, False, initial),
                                traveled, transmitted, count)
            rays = next_rays
        for arrival, count in sorted(arrivals.items()):
            distance, history = arrival[:2]
            if vertical:
                directions.append(arrival[2:])
            distances.append(distance)
            sources.append(source)
            events.extend(history)
            offsets.append(len(events))
            multiplicities.append(float(count))
        pair_offsets.append(len(distances))
    result = distances, sources, offsets, events, pair_offsets, multiplicities
    return (*result, directions) if vertical else result


@dataclass(frozen=True)
class _SommerfeldBin:
    begin: int
    end: int
    rho_min: float
    rho_max: float
    linear_panels: int
    nodes: int
    chunk_nodes: int
    bessel_bytes: int


def _quadrature_bins(rho_nodes, near, rho_max, wave_bound, truncation, frequencies):
    """Host geometry/phase metadata; all actual nodes are generated by CUDA."""
    split = rho_nodes // 4
    log_step = math.log(rho_max / near) / (rho_nodes - 1 - split)

    def coarse(index):
        return near * index / split if index <= split else near * math.exp(log_step * (index - split))

    def coordinate(index):
        left = coarse(index // 2)
        return (left + coarse(index // 2 + 1)) * .5 if index % 2 else left

    rho_nodes = 2 * rho_nodes - 1

    bins = []
    for index in range(4):
        begin = index * rho_nodes // 4
        end = (index + 1) * rho_nodes // 4 if index < 3 else rho_nodes + 4
        rmin, rmax = coordinate(begin), coordinate(min(end, rho_nodes + 3))
        real_end = truncation if index == 0 else max(2 * wave_bound, 32 / rmin)
        # 32 GL nodes per <=4*pi of phase: eight nodes per J0 half-period.
        linear = max(1, math.ceil(real_end * rmax / (4 * math.pi)))
        nodes = 32 * (32 + linear + (16 if index else 0))
        # One complex128 B matrix <=2 GiB, including all frequency batches.
        capacity = ((2 * 1024**3) // (16 * frequencies * (end - begin))) // 32 * 32
        chunk = min(nodes, capacity)
        bins.append(_SommerfeldBin(begin, end, rmin, rmax, linear, nodes, chunk,
                                   16 * frequencies * (end - begin) * chunk))
    return tuple(bins)


@dataclass
class LayeredGreenTable:
    """Contiguous [frequency, reciprocal pair, component, rho] remainder table.

    Components are (G^A_xx, K^Phi). Float plane arguments to evaluate are z
    coordinates; Python integers and int64 CUDA arguments index sorted ``planes``. Pair
    arguments are scalar or have the same number of entries as ``rho``.
    Output shape is [frequency, component, *rho.shape]. At a coincident source
    the analytic 1/R singularity is retained; assembly must integrate it.
    Four unqueried guard nodes beyond rho_max keep physical slopes centered.
    Admission uses every other node as a coarse table and the interleaved nodes
    as directly transformed midpoints; the retained table has half the spacing.
    Admission is the maximum over physical midpoints and rows (frequency, pair,
    component) of |interp-direct| / max(|G_total(mid)|, 1e-6*max_row|G_total|),
    where G_total=images+remainder and max_row uses physical midpoints only.
    """

    planes: torch.Tensor
    frequencies: torch.Tensor
    rho: torch.Tensor
    remainder: torch.Tensor
    derivatives: torch.Tensor
    image_distances: torch.Tensor
    image_coefficients: torch.Tensor
    image_offsets: torch.Tensor
    interpolation_error: torch.Tensor
    tail_error_estimate: torch.Tensor
    media: torch.Tensor
    sigma: torch.Tensor
    pairs: torch.Tensor
    edges: torch.Tensor
    tails: torch.Tensor
    rho_max: float
    regularizer: float
    truncation: float
    exact_mode: int
    ground: float
    pec_bottom: bool
    pec_top: bool
    physical_nodes: int
    quadrature_bins: tuple[_SommerfeldBin, ...]
    vertical: object = None

    @classmethod
    def build(cls, stack: LayerStack, planes, frequencies, rho_max, *, device, extract_distance=math.inf,
              via_spans=(), z_nodes=8):
        """Build from host stack/plane/frequency metadata onto the CUDA device.

        Declared via_spans add (zz,zrho,rhoz,phi) tables on per-layer Chebyshev
        segments. z_nodes defaults to eight; separate z/rho admission is fail-closed.
        Existing horizontal storage is independent of these additional tables.

        Static images with vertical distance >= ``extract_distance`` stay in the
        tabulated remainder instead of the analytic image list. Assembly passes
        its near-interaction distance, so only images that can be near-singular on
        the mesh are extracted. It must be at least the thinnest finite layer,
        which keeps every unextracted exp(-q d) term below exp(-24) at the real-axis
        truncation. Closed-form (exact-mode) stacks always keep their images.
        """
        # Cubic phase interpolation needs this density for the 1e-9, five-wavelength gate.
        result = cls._build(stack, planes, frequencies, rho_max, device=device,
                            rho_nodes=8192 if math.isinf(extract_distance) else 5121,
                            extract_distance=extract_distance)
        if via_spans:
            from .vertical import VerticalGreenTable
            result.vertical = VerticalGreenTable.build(result, stack, via_spans, z_nodes=z_nodes,
                                                        extract_distance=extract_distance)
        return result

    @classmethod
    def _build(cls, stack, planes, frequencies, rho_max, *, device, rho_nodes, extract_distance=math.inf):
        ext = build_extension()
        heights = tuple(sorted(set(planes)))
        frequency_values = tuple(frequencies)
        fmin, fmax = min(frequency_values), max(frequency_values)
        regions = (stack.bottom, *stack.layers, stack.top)
        dielectrics = tuple(m for m in regions if m != "pec")
        reference = dielectrics[0]
        uniform = all((m.eps_r, m.sigma) == (reference.eps_r, reference.sigma) for m in dielectrics)
        bottom, top = stack.bottom == "pec", stack.top == "pec"
        exact_mode = (2 if bottom else 3 if top else 1) if uniform and not (bottom and top) else 0
        ground = 0.0 if bottom else stack.interfaces()[-1]
        # Reciprocal pair numbering is independent of the request order.
        pair_values = tuple((i, j) for i in range(len(heights)) for j in range(i, len(heights)))
        dmin = min(layer.thickness for layer in stack.layers) if stack.layers else rho_max
        if extract_distance < dmin:
            raise ValueError(f"extract_distance {extract_distance:g} is below the thinnest layer {dmin:g}")
        distances, sources, path_offsets, events, pair_offsets, multiplicities = _image_paths(
            stack, heights, pair_values, fmin, fmax, math.inf if exact_mode else extract_distance)

        def tensor(data, dtype):
            return torch.tensor(data, dtype=dtype, device=device)

        media = tensor(tuple(complex(m.eps_r) if m != "pec" else 1.0 for m in regions), torch.complex128)
        sigma = tensor(tuple(m.sigma if m != "pec" else 0.0 for m in regions), torch.float64)
        edges = tensor(stack.interfaces(), torch.float64)
        plane_tensor = tensor(heights, torch.float64)
        freq = tensor(frequency_values, torch.float64)
        pairs = tensor(pair_values, torch.int64)
        # Group equal-distance images before spectral/evaluation kernels. The
        # fixed-order CUDA gather sums their signed coefficients with compensation.
        grouped_distances, grouped_offsets, group_ranges = [], [0], []
        for p in range(len(pair_values)):
            previous = None
            for path in range(pair_offsets[p], pair_offsets[p + 1]):
                if distances[path] != previous:
                    group_ranges.append(path)
                    grouped_distances.append(distances[path])
                    previous = distances[path]
            grouped_offsets.append(len(grouped_distances))
        group_ranges.append(len(distances))
        image_distances = tensor(grouped_distances, torch.float64)
        image_offsets = tensor(grouped_offsets, torch.int64)
        images = ext.images(media, sigma, freq, tensor(sources, torch.int64),
                            tensor(path_offsets, torch.int64), tensor(events, torch.int64).reshape(-1, 3),
                            tensor(distances, torch.float64), tensor(multiplicities, torch.float64))
        images = ext.group_images(images, tensor(group_ranges, torch.int64))
        # These conservative scales are quadrature metadata, not evaluated fields.
        km = 2 * math.pi * fmax / _C0 * math.sqrt(max(
            abs(complex(m.eps_r)) + m.sigma / (2 * math.pi * fmin * _EPS0) for m in dielectrics))
        # Unextracted static rays have d >= extract_distance. The extracted
        # q^-3 term leaves the algebraic O(T^-3) tail covered by 512*km.
        decay_distance = dmin if math.isinf(extract_distance) else extract_distance
        truncation = max(512 * km, 24 / decay_distance)
        near = min(rho_max / 100, 1 / km, dmin)
        coarse_rho = ext.grid(freq, rho_nodes, near, rho_max)
        rho = torch.empty((2 * coarse_rho.numel() - 1,), dtype=torch.float64, device=device)
        rho[::2] = coarse_rho
        rho[1::2] = (coarse_rho[:-1] + coarse_rho[1:]) * .5
        tails = ext.tail(media, sigma, edges, plane_tensor, pairs, freq, bottom, top)
        total_nodes = rho.numel()
        values = torch.empty((len(frequency_values), len(pair_values), 2, total_nodes),
                             dtype=torch.complex128, device=device)
        gl, gw = tensor(_GAUSS_NODES, torch.float64), tensor(_GAUSS_WEIGHTS, torch.float64)
        empty = torch.empty(0, dtype=torch.complex128, device=device)
        bins = _quadrature_bins(rho_nodes, near, rho_max, km, truncation, len(frequency_values))
        bounds = torch.zeros((4, *values.shape[:-1]), dtype=torch.float64, device=device)
        # Chunk iteration schedules vendor GEMMs, never individual nodes/pairs/frequencies.
        # Far tails split into H1 upward and H2 downward; exp(-48*rho/rho_min)
        # replaces the slowly converging, oscillatory real-axis truncation.
        for bin_index, plan in enumerate(bins):
            if not exact_mode:
                total = torch.zeros((len(frequency_values), plan.end - plan.begin, len(pair_values) * 2),
                                    dtype=values.dtype, device=device)
                correction = torch.zeros_like(total)
                for first in range(0, plan.nodes, plan.chunk_nodes):
                    count = min(plan.chunk_nodes, plan.nodes - first)
                    q, weights, kinds = ext.nodes(gl, gw, freq, media, sigma, truncation,
                                                  plan.rho_min, plan.rho_max, plan.linear_panels, first, count)
                    spectrum = ext.spectral(media, sigma, edges, plane_tensor, pairs, freq, q,
                                            image_offsets, image_distances, images, tails, km, bottom, top, True)
                    bessel = ext.bessel(q, weights, kinds, rho, plan.begin, plan.end, False)
                    product = torch.matmul(bessel, spectrum)
                    ext.accumulate(total, correction, product)
                    del bessel, product
                    endpoints = ((plan.nodes - 32,) if bin_index == 0 else
                                 ((32 + plan.linear_panels + 8) * 32 - 32, plan.nodes - 32))
                    if any(first <= endpoint < first + count for endpoint in endpoints):
                        ext.tail_bound(spectrum, q, truncation, plan.rho_min, bounds,
                                       bin_index, first, plan.linear_panels)
                    del q, weights, kinds, spectrum
            ext.pack(empty if exact_mode else total, rho, plane_tensor, pairs, media, sigma, freq,
                     tails, plan.begin, plan.end, False, km, exact_mode, ground, values)
            if not exact_mode:
                del total, correction
        coarse_values = values[..., ::2].contiguous()
        coarse_derivatives = ext.slopes(coarse_values, coarse_rho)
        error = ext.admission(coarse_values, coarse_derivatives, values[..., 1::2].contiguous(),
                              coarse_rho, image_offsets, image_distances, images)
        derivatives = ext.slopes(values, rho)
        tail_error = ext.tail_max(bounds)
        if error.item() > 1e-6:
            raise RuntimeError(f"Green remainder interpolation admission failed: {error.item():.6g} > 1e-6")
        return cls(plane_tensor, freq, rho, values, derivatives, image_distances, images, image_offsets,
                   error, tail_error, media, sigma, pairs, edges, tails, rho_max, km, truncation,
                   exact_mode, ground, bottom, top, 2 * rho_nodes - 1, bins)

    def evaluate(self, pair_i, pair_j, rho):
        """Evaluate CUDA queries; float coordinates must exactly name metal planes."""
        def plane_tensor(value):
            return value.contiguous() if isinstance(value, torch.Tensor) else torch.tensor(
                value, dtype=torch.int64 if isinstance(value, int) else torch.float64, device=self.rho.device)

        shape = rho.shape
        out = build_extension().evaluate(
            self.remainder, self.derivatives, self.rho, self.planes, self.image_offsets,
            self.image_distances, self.image_coefficients, self.media, self.sigma, self.frequencies,
            plane_tensor(pair_i), plane_tensor(pair_j), rho.contiguous(), self.exact_mode, self.ground, self.rho_max)
        return out.reshape((self.frequencies.numel(), 2, *shape))

    def spectral_samples(self, q, *, subtract=False):
        """Internal diagnostic K1 entry, [frequency,q,pair*2] complex128 CUDA."""
        return build_extension().spectral(
            self.media, self.sigma, self.edges, self.planes, self.pairs, self.frequencies, q,
            self.image_offsets, self.image_distances, self.image_coefficients, self.tails,
            self.regularizer, self.pec_bottom, self.pec_top, subtract)

    def evaluate_vertical(self, rho, zs, zo):
        """Evaluate (Gzz, Gzrho, Grhoz, Kphi) in declared spans; shape [F,4,*rho.shape]."""
        if self.vertical is None:
            raise RuntimeError("No via spans were declared when building this Green table")
        return self.vertical.evaluate(zs, zo, rho)

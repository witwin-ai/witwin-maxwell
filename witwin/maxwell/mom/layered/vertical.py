"""Layer-segment Chebyshev tables for vertical formulation-C potentials."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from weakref import proxy

import torch

from .build import build_extension
from .green import _GAUSS_NODES, _GAUSS_WEIGHTS, _SommerfeldBin, _image_paths, _quadrature_bins


@dataclass
class VerticalGreenTable:
    """image_kinds: 0 static, 1 outgoing direct, 2 outgoing PEC-only ray."""

    parent: object
    spans: tuple
    segments: torch.Tensor
    z: torch.Tensor
    image_geometry: torch.Tensor
    image_kinds: torch.Tensor
    image_offsets: torch.Tensor
    image_coefficients: torch.Tensor
    remainder: torch.Tensor
    derivatives: torch.Tensor
    z_interpolation_error: torch.Tensor
    rho_interpolation_error: torch.Tensor
    tail_error_estimate: torch.Tensor
    truncation: float
    z_nodes: int
    boundaries: torch.Tensor | None = None
    horizontal_remainder: torch.Tensor | None = None
    horizontal_derivatives: torch.Tensor | None = None
    horizontal_coefficients: torch.Tensor | None = None

    def tensors(self):
        p = self.parent
        return [p.rho, self.z, self.segments, self.remainder, self.derivatives,
                self.image_coefficients, self.image_geometry, self.image_kinds,
                self.image_offsets, p.media, p.sigma, p.frequencies, p.edges, self.boundaries]

    @classmethod
    def build(cls, parent, stack, via_spans, *, z_nodes, extract_distance):
        spans = tuple(sorted(tuple(map(float, span)) for span in via_spans))
        if z_nodes < 4:
            raise ValueError("vertical z_nodes must be at least four")
        if any(not (math.isfinite(a) and math.isfinite(b) and a < b) for a, b in spans):
            raise ValueError("via spans must have finite increasing endpoints")
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        spans = tuple(merged)
        edges = stack.interfaces()
        if ((stack.bottom == "pec" and spans[0][0] < edges[0]) or
                (stack.top == "pec" and spans[-1][1] > edges[-1])):
            raise ValueError("via spans cannot enter a PEC half space")
        segments = []
        for a, b in spans:
            cuts = (a, *(z for z in edges if a < z < b), b)
            segments.extend(zip(cuts[:-1], cuts[1:]))
        centers = tuple((a + b) / 2 for a, b in segments)
        heights = tuple((a + b) / 2 - (b - a) / 2 * math.cos((2 * j + 1) * math.pi / (2 * z_nodes))
                        for a, b in segments for j in range(z_nodes))
        segment_pairs = tuple((i, j) for i in range(len(segments)) for j in range(i, len(segments)))
        frequencies = parent.frequencies.cpu().tolist()
        # Include every ray that can enter the extraction budget anywhere in a
        # segment pair. A fixed ray set keeps z interpolation continuous.
        margin = 2 * max(b - a for a, b in segments)
        paths = _image_paths(stack, centers, segment_pairs, min(frequencies), max(frequencies),
                             extract_distance + margin, vertical=True)
        distances, sources, offsets, events, pair_offsets, multiplicities, directions = paths
        kept_distances, kept_sources, kept_offsets, kept_events = [], [], [0], []
        kept_pairs, kept_multiplicities, kept_directions = [0], [], []
        for p, (a, b) in enumerate(segment_pairs):
            for i in range(pair_offsets[p], pair_offsets[p + 1]):
                ds, dz = directions[i]
                offset = distances[i] - ds * centers[a] - dz * centers[b]
                minimum = min(ds * x + dz * y + offset for x in segments[a] for y in segments[b])
                is_direct = offsets[i] == offsets[i + 1]
                if is_direct or minimum < extract_distance:
                    kept_distances.append(distances[i])
                    kept_sources.append(sources[i])
                    kept_events.extend(events[offsets[i]:offsets[i + 1]])
                    kept_offsets.append(len(kept_events))
                    kept_multiplicities.append(multiplicities[i])
                    kept_directions.append(directions[i])
            kept_pairs.append(len(kept_distances))
        distances, sources, offsets, events = kept_distances, kept_sources, kept_offsets, kept_events
        pair_offsets, multiplicities, directions = kept_pairs, kept_multiplicities, kept_directions
        geometry, direct, observers = [], [], []
        for p, (a, b) in enumerate(segment_pairs):
            for image in range(pair_offsets[p], pair_offsets[p + 1]):
                ds, dz = directions[image]
                geometry.append((ds, dz, distances[image] - ds * centers[a] - dz * centers[b]))
                observers.append(bisect_right(edges, centers[b]))
                history = events[offsets[image]:offsets[image + 1]]
                # A PEC-only ray stays in one homogeneous region. Its outgoing
                # wave is analytic, including a grounded span's corner.
                direct.append(1 if not history else 2 if all(event[1] < 0 for event in history) else 0)
        device = parent.rho.device
        def tensor(data, dtype=torch.float64):
            return torch.tensor(data, dtype=dtype, device=device)
        ext = build_extension()
        geometry = tensor(geometry).reshape(-1, 3)
        raw = ext.images(parent.media, parent.sigma, parent.frequencies,
                         tensor(sources, torch.int64), tensor(offsets, torch.int64),
                         tensor(events, torch.int64).reshape(-1, 3),
                         tensor(distances), tensor(multiplicities))
        coefficients = ext.vertical_coeff(raw, parent.media, parent.sigma, parent.frequencies,
                                          tensor(sources, torch.int64), tensor(observers, torch.int64), geometry)
        # No direct singularity is interpolated: the local outgoing homogeneous
        # kernel is extracted exactly, in both the spectrum and space.
        min_distance = min((abs(z - edge) for z in heights for edge in edges), default=parent.rho_max)
        truncation = max(parent.truncation, 32 / min_distance)
        empty = torch.empty(0, dtype=torch.complex128, device=device)
        result = cls(proxy(parent), spans, tensor(segments), tensor(heights), geometry,
                     tensor(direct, torch.int64), tensor(pair_offsets, torch.int64), coefficients,
                     empty, empty, tensor(0.), tensor(0.), tensor(0.), truncation, z_nodes)
        pairs = tuple((i, j) for i in range(len(heights)) for j in range(i, len(heights)))
        sources = tensor([heights[i] for i, _ in pairs])
        observers = tensor([heights[j] for _, j in pairs])
        segment_index = tensor([i // z_nodes * len(segments) -
                                (i // z_nodes) * (i // z_nodes - 1) // 2 +
                                j // z_nodes - i // z_nodes for i, j in pairs], torch.int64)
        result.boundaries = tensor((edges[0] if parent.pec_bottom else math.inf,
                                    edges[-1] if parent.pec_top else math.inf))
        result.horizontal_coefficients = raw[:, :1].contiguous()
        if parent.exact_mode:
            # All surviving direct/PEC rays are extracted as exact waves.
            values = torch.zeros((len(frequencies), len(pairs), 5, len(parent.rho)),
                                 dtype=torch.complex128, device=device)
        else:
            values = result.transform(sources, observers, segment_index, components=5)
        result.remainder = values[:, :, :4].contiguous()
        result.horizontal_remainder = values[:, :, 4].contiguous()
        result.derivatives = ext.slopes(result.remainder, parent.rho)
        result.horizontal_derivatives = ext.slopes(result.horizontal_remainder, parent.rho)
        result.admit(sources, observers)
        return result

    def transform(self, sources, observers, segment_index, *, rho=None, components=4):
        p, ext = self.parent, build_extension()
        radial = p.rho if rho is None else rho
        F, P, N = p.frequencies.numel(), sources.numel(), radial.numel()
        out = torch.empty((F, P, components, N), dtype=torch.complex128, device=p.rho.device)
        coarse_nodes = (p.physical_nodes + 1) // 2
        # The radial grid is inherited unchanged, including its near/log split.
        near = float(p.rho[2 * (coarse_nodes // 4)].item())
        bins = _quadrature_bins(coarse_nodes, near, p.rho_max, p.regularizer,
                                self.truncation, F)
        if rho is not None:
            coordinates = radial.cpu().tolist()
            plans = []
            for b in range(4):
                begin, end = b * N // 4, (b + 1) * N // 4
                rmin, rmax = (0.0 if b == 0 else coordinates[begin]), coordinates[end - 1]
                real_end = self.truncation if rmin == 0 else max(2 * p.regularizer, 32 / rmin)
                linear = max(1, math.ceil(real_end * rmax / (4 * math.pi)))
                count = 32 * (32 + linear + (16 if rmin else 0))
                plans.append(_SommerfeldBin(begin, end, rmin, rmax, linear, count, count, 0))
            bins = tuple(plans)
        gl = torch.tensor(_GAUSS_NODES, dtype=torch.float64, device=p.rho.device)
        gw = torch.tensor(_GAUSS_WEIGHTS, dtype=torch.float64, device=p.rho.device)
        bounds = torch.zeros((4, F, P, components), dtype=torch.float64, device=p.rho.device)
        for bin_index, plan in enumerate(bins):
            wave_bound = 2 * p.regularizer * (1 + 64 * torch.finfo(torch.float64).eps)
            shared = (wave_bound < 16 * math.pi / plan.rho_max and
                      (plan.rho_min == 0 or wave_bound < 32 / plan.rho_min))
            totals = [torch.zeros((F, plan.end - plan.begin, P * components),
                                   dtype=torch.complex128, device=p.rho.device) for _ in range(2)]
            corrections = [torch.zeros_like(total) for total in totals]
            # Bound the combined spectrum and TL cache as well as the Bessel
            # matrix; z-node count can exceed the horizontal plane-pair count.
            capacity = (512 * 1024**2 // (16 * F * (P * components + p.media.numel() * 10))) // 32 * 32
            b_capacity = (1024**3 // (16 * F * (plan.end - plan.begin))) // 32 * 32
            chunk_nodes = min(plan.chunk_nodes, capacity, b_capacity)
            for first in range(0, plan.nodes, chunk_nodes):
                count = min(chunk_nodes, plan.nodes - first)
                q, weights, kinds = ext.nodes(gl, gw, p.frequencies, p.media, p.sigma,
                                              self.truncation, plan.rho_min, plan.rho_max,
                                              plan.linear_panels, first, count)
                spectrum = ext.vertical_spectral(p.media, p.sigma, p.edges, p.frequencies,
                                                  q, sources, observers, p.pec_bottom, p.pec_top, components == 5)
                ext.vertical_subtract(spectrum, q, p.frequencies, p.media, p.sigma, p.edges,
                                      sources, observers, segment_index, self.image_offsets,
                                      self.image_geometry, self.image_kinds,
                                      self.image_coefficients if components == 4 else
                                      torch.cat((self.image_coefficients, self.horizontal_coefficients), 1))
                spectrum = spectrum.reshape(F, count, P * components)
                endpoints = ((plan.nodes - 32,) if plan.rho_min == 0 else
                             ((32 + plan.linear_panels + 8) * 32 - 32, plan.nodes - 32))
                if any(first <= endpoint < first + count for endpoint in endpoints):
                    ext.vertical_tail_bound(spectrum, q, self.truncation, plan.rho_min, bounds,
                                            bin_index, first, plan.linear_panels)
                # Beyond the 32 deformed/log panels, linear and tail nodes
                # are frequency-independent when both endpoints exceed the
                # conservative wave bound. Preserve the complete B matrix and
                # GEMM shape so spectral reduction/accumulation stays bitwise.
                dynamic = min(count, max(0, 32 * 32 - first)) if shared else count
                for order, operation in enumerate((ext.bessel, ext.bessel_one)):
                    if dynamic == count:
                        bessel = operation(q, weights, kinds, radial, plan.begin, plan.end, False)
                    else:
                        common = operation(q[:1, dynamic:].contiguous(),
                                           weights[:1, dynamic:].contiguous(),
                                           kinds[:1, dynamic:].contiguous(),
                                           radial, plan.begin, plan.end, False).expand(F, -1, -1)
                        if dynamic:
                            prefix = operation(q[:, :dynamic].contiguous(),
                                               weights[:, :dynamic].contiguous(),
                                               kinds[:, :dynamic].contiguous(),
                                               radial, plan.begin, plan.end, False)
                            bessel = torch.cat((prefix, common), -1)
                            del prefix
                        else:
                            bessel = common.contiguous()
                        del common
                    product = torch.matmul(bessel, spectrum)
                    ext.accumulate(totals[order], corrections[order], product)
                    del bessel, product
                del q, weights, kinds, spectrum
            ext.vertical_pack(*totals, out, plan.begin)
        if rho is None:
            self.tail_error_estimate = ext.tail_max(bounds)
        return out

    def assembly_tensors(self):
        return self.tensors() + [self.horizontal_remainder, self.horizontal_derivatives,
                                 self.horizontal_coefficients]

    def evaluate(self, zs, zo, rho):
        def tensor(value):
            return value.contiguous() if isinstance(value, torch.Tensor) else torch.tensor(
                value, dtype=torch.float64, device=self.parent.rho.device)
        out = build_extension().vertical_evaluate(self.tensors(), tensor(zs), tensor(zo),
                                                   rho.contiguous(), self.parent.rho_max)
        return out.reshape((self.parent.frequencies.numel(), 4, *rho.shape))

    def spectral_samples(self, q, zs, zo):
        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float64, device=self.parent.rho.device).reshape(-1).contiguous()
        return build_extension().vertical_spectral(
            self.parent.media, self.parent.sigma, self.parent.edges, self.parent.frequencies,
            q, tensor(zs), tensor(zo), self.parent.pec_bottom, self.parent.pec_top)

    def admit(self, sources, observers):
        """Coarse rho midpoints and independent z midpoint/endpoint transforms.

        Errors are normalized per component by the maximum total kernel on the
        checked row. The rho gate floors cross rows at 1e-6 of Gzz; the z
        gate also floors vanishing scalar rows at a PEC endpoint.
        z admission checks the tensor product of every inter-node midpoint and
        both one-sided segment endpoints at 17 positive radial nodes.
        """
        p, ext = self.parent, build_extension()
        if p.exact_mode:
            zero = self.remainder.new_zeros((1, 4, 1))
            self.z_interpolation_error = ext.vertical_error(zero, zero)
            if self.z_interpolation_error.item() > 1e-6:
                raise RuntimeError("Vertical z interpolation admission failed: exact-wave remainder")
            return
        coarse = self.remainder[..., ::2].contiguous()
        grid = p.rho[::2].contiguous()
        tensors = self.tensors()
        tensors[0], tensors[3], tensors[4] = grid, coarse, ext.slopes(coarse, grid)
        self.rho_interpolation_error = ext.vertical_rho_error(
            tensors, self.remainder[..., 1::2].contiguous(), sources, observers, p.rho_max)
        horizontal = self.assembly_tensors()
        horizontal[0] = grid
        horizontal[14] = self.horizontal_remainder[..., ::2].contiguous()
        horizontal[15] = ext.slopes(horizontal[14], grid)
        h_error = ext.horizontal_rho_error(horizontal, self.horizontal_remainder[..., 1::2].contiguous(),
                                            sources, observers)
        self.rho_interpolation_error = torch.maximum(self.rho_interpolation_error, h_error)
        if self.rho_interpolation_error.item() > 1e-6:
            raise RuntimeError(f"Vertical rho interpolation admission failed: {self.rho_interpolation_error.item():.6g} > 1e-6")
        heights = self.z.cpu().tolist()
        segments = self.segments.cpu().tolist()
        checks = []
        for i, (a, b) in enumerate(segments):
            nodes = heights[i * self.z_nodes:(i + 1) * self.z_nodes]
            checks.extend((i, z) for z in (math.nextafter(a, b),
                          *((x + y) / 2 for x, y in zip(nodes[:-1], nodes[1:])),
                          math.nextafter(b, a)))
        pairs = [(a, za, b, zb) for i, (a, za) in enumerate(checks) for b, zb in checks[i:]]
        device = p.rho.device
        def tensor(values, dtype=torch.float64):
            return torch.tensor(values, dtype=dtype, device=device)
        zs, zo = tensor([za for _, za, _, _ in pairs]), tensor([zb for _, _, _, zb in pairs])
        segment_indices = tensor([a * len(segments) - a * (a - 1) // 2 + b - a
                                  for a, _, b, _ in pairs], torch.int64)
        indices = [1 + i * (p.physical_nodes - 2) // 16 for i in range(17)]
        radial = p.rho[indices].contiguous()
        full_reference = self.transform(zs, zo, segment_indices, rho=radial, components=5)
        horizontal_reference = full_reference[:, :, 4].contiguous()
        reference = full_reference[:, :, :4].contiguous()
        reference = reference.permute(0, 2, 1, 3).reshape(p.frequencies.numel(), 4, -1).contiguous()
        radial_values = radial.cpu().tolist()
        query_zs = tensor([za for _, za, _, _ in pairs for _ in radial_values])
        query_zo = tensor([zb for _, _, _, zb in pairs for _ in radial_values])
        query_rho = tensor(radial_values * len(pairs))
        expected = ext.vertical_reference(self.tensors(), reference, query_zs, query_zo, query_rho, p.rho_max)
        actual = self.evaluate(query_zs, query_zo, query_rho)
        def pair_rows(values):
            return values.reshape(p.frequencies.numel(), 4, len(pairs), len(indices)).permute(
                0, 2, 1, 3).reshape(-1, 4, len(indices)).contiguous()
        self.z_interpolation_error = ext.vertical_error(pair_rows(actual), pair_rows(expected))
        h_error = ext.horizontal_z_error(self.assembly_tensors(), horizontal_reference,
                                          query_zs, query_zo, query_rho)
        self.z_interpolation_error = torch.maximum(self.z_interpolation_error, h_error)
        if self.z_interpolation_error.item() > 1e-6:
            raise RuntimeError(f"Vertical z interpolation admission failed: {self.z_interpolation_error.item():.6g} > 1e-6")

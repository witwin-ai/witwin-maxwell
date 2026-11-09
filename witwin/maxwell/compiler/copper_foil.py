"""Passive diffusion-mode realization and planar support for finite copper foil."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from copy import copy, deepcopy
import math

import numpy as np
import torch

from ..copper_foil import CopperFoil, MU_0
from ..ports import _coordinate_roundoff
from ..rational import RationalModel, StateSpaceNetwork
from .ports import _axis_values, _box_values, _snap_index
from .passive_fit import fit_passive_foster


_COPPER_FIT_CACHE_CAPACITY = 64
_COPPER_FIT_CACHE = OrderedDict()


def _fit_parameter_key(value):
    if torch.is_tensor(value):
        return (value.dtype, value.device, float(value).hex())
    return (type(value), float(value).hex())


@dataclass(frozen=True)
class CompiledCopperResponse:
    rational: RationalModel
    state_space: StateSpaceNetwork
    rates: torch.Tensor
    max_relative_error: float
    relative_error_bound: float
    frequency_range: tuple[float, float]


@dataclass(frozen=True)
class CompiledCopperFace:
    name: str
    axis: int
    normal_index: int
    component: int
    indices: torch.Tensor
    material: CopperFoil


def compile_copper_response(material, *, tolerance=2e-4, max_modes=256):
    """Compile a passive diffusion realization with exact DC moments.

    Automatic order uses shared nonnegative residues on paired logarithmic diffusion grids;
    explicit order keeps the differentiable exact-pole expansion and DC tail.
    Both retain the exact odd integrator and the symmetric B/C realization.
    Targets are 5e-5 in band and twice the reference-series extended-band error; in-band DFT observables use in-band admittance, while broadband transients depend on both bands.
    """
    if not isinstance(material, CopperFoil):
        raise TypeError("compile_copper_response requires CopperFoil.")
    if not 0 < tolerance < 1 or max_modes < 2:
        raise ValueError("Copper response tolerance/mode limit are invalid.")
    band = material.frequency_range
    parameter_tensors = [value for value in (material.thickness, material.conductivity) if torch.is_tensor(value)]
    device = parameter_tensors[0].device if parameter_tensors else torch.device("cpu")
    if material.mode_count is None:
        key = (type(material), _fit_parameter_key(material.thickness),
               _fit_parameter_key(material.conductivity), material.frequency_range,
               _fit_parameter_key(tolerance), max_modes, device, torch.get_default_dtype())
        if key in _COPPER_FIT_CACHE:
            _COPPER_FIT_CACHE.move_to_end(key)
            return deepcopy(_COPPER_FIT_CACHE[key])
        response = _fit_copper_response(material, min(tolerance, 5e-5), min(max_modes, 128), device)
        # Public response tensors remain private to their caller, including on a miss.
        _COPPER_FIT_CACHE[key] = deepcopy(response)
        if len(_COPPER_FIT_CACHE) > _COPPER_FIT_CACHE_CAPACITY:
            _COPPER_FIT_CACHE.popitem(last=False)
        return response
    return _truncated_copper_response(material, tolerance, max_modes, device)


def _truncated_copper_response(material, tolerance, max_modes, device):
    band = material.frequency_range
    frequencies = torch.logspace(math.log10(band[0]), math.log10(band[1]), 257, dtype=torch.float64, device=device)
    frequencies[0], frequencies[-1] = band
    exact = material.modal_admittance(frequencies)
    s = -2j * math.pi * frequencies
    thickness, sigma = material.thickness, material.conductivity
    residue = 4 / (MU_0 * thickness)
    integrator = 2 / (MU_0 * thickness)
    vectors = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64, device=device) / math.sqrt(2)
    count = material.mode_count or 2
    while True:
        mode_index = torch.arange(count, dtype=torch.float64, device=device)
        even_rates = ((2 * mode_index + 1) * math.pi / thickness).square() / (MU_0 * sigma)
        odd_rates = ((2 * (mode_index + 1)) * math.pi / thickness).square() / (MU_0 * sigma)
        even_tail = sigma * thickness / 2 - torch.sum(residue / even_rates)
        odd_tail = sigma * thickness / 6 - torch.sum(residue / odd_rates)
        even = torch.sum(residue / (s[:, None] + even_rates), -1) + even_tail
        odd = integrator / s + torch.sum(residue / (s[:, None] + odd_rates), -1) + odd_tail
        relative = torch.abs(torch.stack((even, odd), -1) - exact) / torch.abs(exact)
        error = float(relative.max().detach())
        # For every omitted Stieltjes term, replacing w/(s+r) by w/r
        # incurs at most omega*w/r^2. Its exact tail moment bounds the whole
        # frequency interval, using the monotone real part at the upper edge.
        full_moments = torch.stack((torch.as_tensor(MU_0 * sigma**2 * thickness**3 / 24, device=device, dtype=torch.float64),
                                    torch.as_tensor(MU_0 * sigma**2 * thickness**3 / 360, device=device, dtype=torch.float64)))
        retained_moments = torch.stack((torch.sum(residue / even_rates.square()),
                                       torch.sum(residue / odd_rates.square())))
        tail_moments = (full_moments - retained_moments).clamp_min(0)
        tail_moments += 32 * torch.finfo(torch.float64).eps * full_moments
        bound = float((2 * math.pi * band[1] * tail_moments / exact[-1].real).max().detach())
        if error <= tolerance and bound <= tolerance:
            break
        if material.mode_count is not None or count >= max_modes:
            raise ValueError(f"CopperFoil diffusion reduction exceeds {max_modes} modes per parity; residual={error}.")
        count = min(2 * count, max_modes)
    rates = torch.cat((even_rates, odd_rates, torch.zeros(1, dtype=torch.float64, device=device)))
    mode_vectors = torch.cat((vectors[0].repeat(count, 1), vectors[1].repeat(count + 1, 1)))
    weights = torch.ones_like(rates) * residue
    weights[-1] = integrator
    return _copper_response(rates, weights, mode_vectors, even_tail, odd_tail, error, bound, band)


def _copper_response(rates, weights, mode_vectors, even_tail, odd_tail, error, bound, band):
    vectors = mode_vectors.new_tensor([[1., 1.], [1., -1.]]) / math.sqrt(2)
    B = weights.sqrt()[:, None] * mode_vectors
    D = even_tail * torch.outer(vectors[0], vectors[0]) + odd_tail * torch.outer(vectors[1], vectors[1])
    if min(float(even_tail.detach()), float(odd_tail.detach())) < 0:
        raise RuntimeError("CopperFoil analytic omitted-mode tail must be nonnegative.")
    residues = torch.einsum("ni,nj,n->ijn", mode_vectors, mode_vectors, weights)
    rational = RationalModel(poles=-rates.to(torch.complex128), residues=residues.to(torch.complex128),
                             direct=D.to(torch.complex128), representation="Y")
    state = StateSpaceNetwork(A=-torch.diag(rates), B=B, C=B.T, D=D,
                              representation="Y", port_order=("low", "high"), passivity_margin=0.0)
    return CompiledCopperResponse(rational, state, rates, error, bound, band)


def _fit_copper_response(material, tolerance, max_modes, device):
    """Paired positive diffusion poles with shared nonnegative residues.

    The even/odd exact series has equal residues and interlaced poles. Sampling
    that lattice on a logarithmic modal-index grid preserves their cancellation
    in the two-port mutual response. Separate nonnegative direct terms impose
    both DC equalities; the odd integrator remains exact. Both frequency bands
    have hard interpolation-certified error gates.
    """
    band = material.frequency_range
    extended_band = (band[0] / 10, band[1] * 10)
    reference = copy(material)
    object.__setattr__(reference, "frequency_range", extended_band)
    frequencies = np.geomspace(*band, 4097)
    frequencies[0], frequencies[-1] = band
    extended_frequencies = np.geomspace(*extended_band, 4097)
    fit_frequencies = np.unique(np.r_[frequencies[::4], extended_frequencies[::4]])
    def exact_at(samples):
        return reference.modal_admittance(torch.as_tensor(samples, device=device)).detach().cpu().numpy()
    exact, extended_exact, fit_exact = map(exact_at, (frequencies, extended_frequencies, fit_frequencies))
    previous = _truncated_copper_response(material, 2e-4, 256, device)
    previous_y = previous.rational.evaluate(torch.as_tensor(extended_frequencies, device=device)).detach().cpu().numpy()
    previous_modal = np.stack((previous_y[:, 0, 0] + previous_y[:, 0, 1],
                               previous_y[:, 0, 0] - previous_y[:, 0, 1]), -1)
    # An absolute floor keeps near-constant thin-film admittances from being held to the reference series' round-off-level accuracy.
    extended_tolerance = max(2 * float(np.max(np.abs(previous_modal - extended_exact) / np.abs(extended_exact))), 1e-6)
    sample_weights = np.where((fit_frequencies >= band[0]) & (fit_frequencies <= band[1]),
                              1., tolerance / extended_tolerance)
    thickness, sigma = float(material.thickness), float(material.conductivity)
    moments = np.array([sigma * thickness / 2, sigma * thickness / 6])
    integrator = 2 / (MU_0 * thickness)
    target = fit_exact.copy()
    target[:, 1] -= integrator / (-2j * math.pi * fit_frequencies)
    # The odd poles are exactly logarithmic; their paired even poles keep the
    # half-mode shift of the physical diffusion lattice, including its first atom.
    maximum_index = max(1., math.sqrt(100 * 2 * math.pi * band[1] * MU_0 * sigma) * thickness / (2 * math.pi))

    def columns(samples, even_rates, odd_rates):
        s = -2j * math.pi * samples
        even = np.column_stack((moments[0] * even_rates / (s[:, None] + even_rates),
                                np.full(len(s), moments[0]), np.zeros(len(s))))
        odd = np.column_stack((moments[0] * even_rates / (s[:, None] + odd_rates),
                               np.zeros(len(s)), np.full(len(s), moments[1])))
        return np.stack((even, odd), 1)

    def evaluate(samples, basis, fractions):
        values = basis @ fractions
        values[:, 1] += integrator / (-2j * math.pi * samples)
        return values

    def error_bound(samples, actual, expected):
        residual = np.abs(actual - expected)
        # For a Stieltjes atom in log(omega), |Y''| <= r/(2p).
        # Exact/fitted regular parts each have the same DC moment; the exact
        # integrator cancels. Linear interpolation certifies between samples,
        # using the decreasing |Y| at the right endpoint of each interval.
        curvature = np.diff(np.log(samples))[:, None]**2 * moments[None] / 8
        return float(np.max((np.maximum(residual[:-1], residual[1:]) + curvature) / np.abs(expected[1:])))

    for count in range(max_modes + 1):
        index = np.geomspace(1., maximum_index + 1, count) - 1
        even_rates = ((2 * index + 1) * math.pi / thickness)**2 / (MU_0 * sigma)
        odd_rates = ((2 * index + 2) * math.pi / thickness)**2 / (MU_0 * sigma)
        fit_columns = columns(fit_frequencies, even_rates, odd_rates)
        relative_columns = fit_columns / np.abs(fit_exact[:, :, None]) * sample_weights[:, None, None]
        relative_target = target / np.abs(fit_exact) * sample_weights[:, None]
        # x_k = r_k/(p_even,k * even_DC). Two nonnegative direct fractions
        # complete the exact even and odd DC moment equalities.
        equality = np.zeros((2, count + 2))
        equality[0, :count], equality[0, -2] = 1., 1.
        equality[1, :count] = moments[0] * even_rates / (moments[1] * odd_rates)
        equality[1, -1] = 1.
        fractions = fit_passive_foster(
            fit_frequencies, relative_target, relative_columns,
            equalities=(equality, np.ones(2)),
        )
        # Return discarded active-set dust to its two DC direct terms.
        discarded = fractions[:count] < 1e-12
        fractions[-2] += fractions[:count][discarded].sum()
        fractions[-1] += np.sum(equality[1, :count][discarded] * fractions[:count][discarded])
        fractions[:count][discarded] = 0
        actual = evaluate(frequencies, columns(frequencies, even_rates, odd_rates), fractions)
        error = float(np.max(np.abs(actual - exact) / np.abs(exact)))
        bound = error_bound(frequencies, actual, exact)
        extended_actual = evaluate(extended_frequencies,
                                  columns(extended_frequencies, even_rates, odd_rates), fractions)
        extended_bound = error_bound(extended_frequencies, extended_actual, extended_exact)
        if bound <= tolerance and extended_bound <= extended_tolerance:
            break
    else:
        raise ValueError(f"CopperFoil diffusion reduction exceeds {max_modes} modes per parity; "
                         f"residual={error}, extended_bound={extended_bound}, extended_limit={extended_tolerance}.")
    active = fractions[:count] > 0
    residues = moments[0] * even_rates[active] * fractions[:count][active]
    rates = torch.as_tensor(np.r_[even_rates[active], odd_rates[active], 0.], dtype=torch.float64, device=device)
    weights = torch.as_tensor(np.r_[residues, residues, integrator], dtype=torch.float64, device=device)
    vectors = rates.new_tensor([[1., 1.], [1., -1.]]) / math.sqrt(2)
    mode_vectors = torch.cat((vectors[0].repeat(len(residues), 1), vectors[1].repeat(len(residues) + 1, 1)))
    return _copper_response(rates, weights, mode_vectors, rates.new_tensor(moments[0] * fractions[-2]),
                            rates.new_tensor(moments[1] * fractions[-1]), error, bound, band)


def _support_bound_tolerance(geometry, axis, coordinates):
    """Retain source ULPs when center/size arithmetic reconstructs a boundary.

    This budget cannot move a real footprint by a useful fraction of a cell.
    In particular it does not extend a foil across a resolved physical gap.
    """
    position, size = _box_values(geometry)
    input_budget = (_coordinate_roundoff(geometry.position)[axis]
                    + .5 * _coordinate_roundoff(geometry.size)[axis])
    arithmetic_scale = abs(position[axis]) + .5 * abs(size[axis]) + max(abs(coordinates[0]),abs(coordinates[-1]))
    arithmetic_budget = 32 * np.finfo(np.float64).eps * arithmetic_scale
    return min(input_budget + arithmetic_budget, float(np.min(np.diff(coordinates))) / 32)


def compile_copper_faces(scene, *, device, include_halos=False):
    """Compile full Yee tangential-edge supports without volumetric copper cells."""
    faces = []
    layout = getattr(scene, "_fdtd_shard_layout", None)
    for structure in scene.structures:
        material = structure.material
        if not isinstance(material, CopperFoil) or not bool(getattr(structure, "enabled", True)):
            continue
        if getattr(structure.geometry, "kind", None) == "surface_mesh":
            # Planar SurfaceMesh copper is owned by the conformal two-sided
            # thin-sheet runtime (fdtd/thin_sheet.py), which validates scope.
            continue
        position, size = _box_values(structure.geometry)
        zero = np.flatnonzero(size == 0)
        if len(zero) != 1 or np.any(size < 0):
            raise ValueError(f"CopperFoil {structure.name!r} requires an axis-aligned planar Box.")
        axis = int(zero[0])
        if layout is not None and axis == 0:
            global_nodes = scene._fdtd_global_x_nodes64
            global_index = _snap_index(global_nodes, float(position[0]), label="foil plane",
                                       roundoff=_coordinate_roundoff(structure.geometry.position)[0])
            owned = layout.component("Ey").owned_global_slice[0]
            if not owned.start <= global_index < owned.stop:
                continue
        node_coords = _axis_values(scene, "xyz"[axis], half=False)
        normal_index = _snap_index(node_coords, float(position[axis]), label="foil plane",
                                   roundoff=_coordinate_roundoff(structure.geometry.position)[axis])
        low_pml = scene.pml_thickness_for_face("xyz"[axis], "low")
        high_pml = scene.pml_thickness_for_face("xyz"[axis], "high")
        if normal_index <= max(0, low_pml) or normal_index >= len(node_coords) - max(1, high_pml + 1):
            raise ValueError(f"CopperFoil {structure.name!r} needs exterior cells on both sides outside PML.")
        for component in range(3):
            if component == axis:
                continue
            coordinate_arrays = []
            for index, coordinate_axis in enumerate("xyz"):
                coords = _axis_values(scene, coordinate_axis, half=index == component)
                if index == axis:
                    coordinate_arrays.append(np.array([normal_index]))
                elif index == component:
                    nodes = _axis_values(scene, coordinate_axis, half=False)
                    lower, upper = position[index] - size[index] / 2, position[index] + size[index] / 2
                    tolerance = _support_bound_tolerance(structure.geometry,index,nodes)
                    coordinate_arrays.append(np.flatnonzero((nodes[:-1] >= lower-tolerance) & (nodes[1:] <= upper+tolerance)))
                else:
                    lower, upper = position[index] - size[index] / 2, position[index] + size[index] / 2
                    tolerance = _support_bound_tolerance(structure.geometry,index,coords)
                    coordinate_arrays.append(np.flatnonzero((coords >= lower-tolerance) & (coords <= upper+tolerance)))
            indices = np.stack(np.meshgrid(*coordinate_arrays, indexing="ij"), -1).reshape(-1, 3)
            if layout is not None and not include_halos:
                owned = layout.component("E" + "xyz"[component]).owned_local_slice[0]
                indices = indices[(indices[:, 0] >= owned.start) & (indices[:, 0] < owned.stop)]
            if not len(indices):
                if layout is not None:
                    continue
                raise ValueError(f"CopperFoil {structure.name!r} has no resolved E{'xyz'[component]} support.")
            faces.append(CompiledCopperFace(str(structure.name), axis, normal_index, component,
                                            torch.as_tensor(indices, dtype=torch.int64, device=device), material))
    return tuple(faces)


# Discrete Yee half-plane edge constants of a zero-thickness split-trace sheet,
# keyed by the cell aspect ratio r = (normal cell) / (tangential dual width).
# For a uniform grid of tangential step h, the edge current of the continuum
# sheet K/sqrt(s) is carried by the rim trace as c0*K*sqrt(h), and the native
# two-face loss within a >> h of the edge is Rs*K^2/2*(ln(a/h) + g). Both come
# from the 5-point potential problem of a conducting half-line on that grid
# (Dirichlet far field Re sqrt(z), N = 200 and 400 node boxes, Richardson in
# 1/N; benchmark/copper_foil_edge_constants.py regenerates the table).
_EDGE_LOG2_ASPECT = tuple(0.5 * index for index in range(-6, 7))
_EDGE_C0 = (1.77587, 1.77930, 1.78603, 1.79913, 1.82400, 1.86939, 1.94737,
            2.07159, 2.25459, 2.50662, 2.83678, 3.25487, 3.77287)
_EDGE_G = (3.36283, 3.36913, 3.38348, 3.41308, 3.47158, 3.58352, 3.78977,
           4.15276, 4.76091, 5.73488, 7.23916, 9.50114, 12.83880)


def copper_edge_stopping_distance(thickness):
    """Lewin stopping distance of a thin rectangular good-conductor edge.

    The exterior of a semi-infinite slab of thickness t is the image of the
    upper half plane under z = -(t/pi)[w sqrt(w^2-1) - acosh(w)]. Its surface
    current, integrated over the two faces and the side wall within a >> t of
    the edge, equals the zero-thickness 1/sqrt(s) integral cut off at
    Delta = t / (4 pi e^pi). This is the length in the Owyang-Wu and Pucel
    "pi + ln(4 pi W / t)" thick-edge loss terms.
    """
    return thickness / (4 * math.pi * math.exp(math.pi))


def copper_edge_half_plane_constants(aspect):
    """Interpolated (c0, g) of the discrete half-plane edge, clamped to the table."""
    values = torch.as_tensor(aspect, dtype=torch.float64)
    log_aspect = torch.log2(values).clamp(_EDGE_LOG2_ASPECT[0], _EDGE_LOG2_ASPECT[-1])
    nodes = torch.tensor(_EDGE_LOG2_ASPECT, dtype=torch.float64, device=values.device)
    position = (log_aspect - nodes[0]) / .5
    lower = position.floor().clamp(0, len(nodes) - 2).long()
    fraction = position - lower
    tables = []
    for table in (_EDGE_C0, _EDGE_G):
        table = torch.tensor(table, dtype=torch.float64, device=values.device)
        tables.append(table[lower] * (1 - fraction) + table[lower + 1] * fraction)
    return tuple(tables)


def copper_edge_impedance_factor(thickness, tangent_width, aspect, sides, weight=1.):
    """Rim-trace two-face impedance multiplier kappa >= 1 of a free foil edge.

    The physical loss of one edge region exceeds the native rim-trace loss by
    Rs*K^2/2*(ln(h/Delta) - g), while the rim itself dissipates Rs*K^2*c0^2/2.
    Scaling the rim impedance (resistance and internal reactance alike) by
    kappa = 1 + sides*(ln(h/Delta) - g)/c0^2 restores the edge loss to first
    order in Zs/eta. ``sides`` counts free edges of the same trace (a strip one
    node wide has two). ``weight`` (CopperFoil.edge_weight) blends the
    correction in over the thin-to-thick transition. The full factor is
    independent of frequency and conductivity; the correction is clamped at zero
    for h < ~0.15 t, where ln(h/Delta) < g (kappa(h = t) = 1.50). The model is
    asymptotic in h >> t; it is not a thickness-resolving edge model.
    """
    width = torch.as_tensor(tangent_width, dtype=torch.float64)
    c0, g = copper_edge_half_plane_constants(torch.as_tensor(aspect, dtype=torch.float64, device=width.device))
    thickness = torch.as_tensor(thickness, dtype=torch.float64, device=width.device)
    correction = (torch.log(width / copper_edge_stopping_distance(thickness)) - g) / c0.square()
    sides = torch.as_tensor(sides, dtype=torch.float64, device=width.device)
    weight = torch.as_tensor(weight, dtype=torch.float64, device=width.device)
    return 1 + weight * sides * correction.clamp_min(0)


def _component_shape(scene, component):
    shape = [len(getattr(scene, f"{axis}_nodes64")) for axis in "xyz"]
    shape[component] -= 1
    return shape


def _flat(points, shape):
    return (points[:, 0] * shape[1] + points[:, 1]) * shape[2] + points[:, 2]


def _periodic(scene, axis):
    return scene.boundary.axis_kind("xyz"[axis]) in ("periodic", "bloch")


def copper_free_edge_sides(scene, face, indices, all_faces, *, contact=None, per_side=False):
    """Count, per foil trace, the in-plane sides across which the conductor ends.

    The side lies across the trace's tangent axis (the in-plane axis normal to
    its current). The conductor continues there when a connecting tangent-axis
    edge from either node of the trace carries a foil trace on the same plane,
    is a conductor or attachment edge of ``contact`` (a mapping from
    ``"Ex"``/``"Ey"``/``"Ez"`` to boolean masks: PEC, conformal PEC, good
    conductors, wires, ports, lumped elements), or leaves a non-periodic domain
    (the sheet runs into the boundary). Periodic and Bloch axes wrap through
    their seam. A trace whose node touches a contact edge along the foil normal
    (a via or vertical feed at the rim) has no free side. Separate supports never
    share an edge, so a one-cell gap between two foils leaves both edges free.
    ``per_side`` returns the low/high free-side booleans instead of their count.
    """
    component, normal = face.component, face.axis
    tangent = 3 - component - normal
    shape = _component_shape(scene, tangent)
    members = [candidate.indices for candidate in all_faces
               if candidate.axis == normal and candidate.component == tangent
               and candidate.normal_index == face.normal_index]
    occupied = _flat(torch.cat(members).to(indices.device), shape) if members else indices.new_zeros(0)
    mask = None if contact is None else contact.get("E" + "xyz"[tangent])
    wrap = _periodic(scene, tangent)
    free = []
    for offset in (-1, 0):
        continued = torch.zeros(len(indices), dtype=torch.bool, device=indices.device)
        for node in (0, 1):
            edge = indices.clone()
            edge[:, tangent] += offset
            edge[:, component] += node
            inside = (edge[:, tangent] >= 0) & (edge[:, tangent] < shape[tangent])
            if wrap:
                edge[:, tangent] %= shape[tangent]
                inside = torch.ones_like(inside)
            else:
                edge[:, tangent].clamp_(0, shape[tangent] - 1)
            # A trace on the far end of the component axis has no edge there.
            exists = edge[:, component] < shape[component]
            edge[:, component].clamp_(max=shape[component] - 1)
            hit = torch.isin(_flat(edge, shape), occupied)
            if mask is not None:
                hit |= mask.to(indices.device)[tuple(edge.T)]
            continued |= (hit & exists) | ~inside
        free.append(~continued)
    sides = free[0].double() + free[1].double()
    normal_mask = None if contact is None else contact.get("E" + "xyz"[normal])
    if normal_mask is not None and bool((sides > 0).any()):
        normal_shape = _component_shape(scene, normal)
        touched = torch.zeros(len(indices), dtype=torch.bool, device=indices.device)
        for node in (0, 1):
            for offset in (-1, 0):
                edge = indices.clone()
                edge[:, component] += node
                edge[:, normal] += offset
                valid = (edge[:, normal] >= 0) & (edge[:, normal] < normal_shape[normal]) & (
                    edge[:, component] < normal_shape[component])
                edge[:, normal].clamp_(0, normal_shape[normal] - 1)
                edge[:, component].clamp_(max=normal_shape[component] - 1)
                touched |= valid & normal_mask.to(indices.device)[tuple(edge.T)]
        sides = torch.where(touched, torch.zeros_like(sides), sides)
        free = [side & ~touched for side in free]
    return tuple(free) if per_side else sides


def copper_edge_feature_cells(scene, face, indices, all_faces, sides, *, contact=None, limit=8):
    """Narrowest free-edge strip and slot, in tangential cells, for one face.

    Along the tangent axis a strip spans from a free edge to the next free edge
    of the same foil row; a slot spans from a free edge across open nodes to the
    next foil trace of the same component and plane. Returns (strip, slot), each
    None when absent or wider than ``limit`` cells. Used for applicability
    warnings only.
    """
    component, normal, tangent = face.component, face.axis, 3 - face.component - face.axis
    low, high = copper_free_edge_sides(scene, face, indices, all_faces, contact=contact, per_side=True)
    selected = (sides > 0) & (low | high)
    if not bool(selected.any()):
        return None, None
    members = [candidate.indices for candidate in all_faces
               if candidate.axis == normal and candidate.component == component
               and candidate.normal_index == face.normal_index]
    shape = _component_shape(scene, component)
    occupied = _flat(torch.cat(members), shape).cpu().numpy()
    count = len(getattr(scene, "xyz"[tangent] + "_nodes64"))
    rows = indices[selected].cpu().numpy()
    steps = np.arange(1, limit + 2)
    stride = math.prod(shape[tangent + 1:])
    offsets = _flat(rows, shape)
    strip = 0 if bool((low[selected] & high[selected]).any()) else None
    slot = None
    for direction, free_side in ((-1, low[selected]), (1, high[selected])):
        free_side = free_side.cpu().numpy()
        position = rows[:, tangent, None] + direction * steps
        inside = (position >= 0) & (position < count)
        # Free sides stop at the next foil node; connected sides at the first gap.
        hit = np.isin(offsets[:, None] + direction * steps * stride, occupied)
        stopped = inside & (hit == free_side[:, None])
        first = stopped.argmax(axis=1) + 1
        found = stopped.any(axis=1) & (first <= limit)
        widths = first[found & ~free_side] - 1
        gaps = first[found & free_side]
        if widths.size:
            strip = int(widths.min()) if strip is None else min(strip, int(widths.min()))
        if gaps.size:
            slot = int(gaps.min()) if slot is None else min(slot, int(gaps.min()))
    return strip, slot


def copper_edge_geometry(scene, face, indices, all_faces, *, contact=None):
    """Free-edge trace positions, edge counts, tangential widths and aspect ratios.

    Returns None when the material keeps the bare rim or the face has no free
    edge. The geometry is fixed; only the foil thickness (and, through the
    edge weight, its skin-depth ratio) enters kappa.
    """
    if not face.material.edge_correction_active or not len(indices):
        return None
    sides = copper_free_edge_sides(scene, face, indices, all_faces, contact=contact)
    positions = torch.nonzero(sides > 0).reshape(-1)
    if not len(positions):
        return None
    tangent = 3 - face.component - face.axis
    tangent_dual = torch.as_tensor(getattr(scene, f"d{'xyz'[tangent]}_dual64"), device=indices.device)
    normal_primal = getattr(scene, f"d{'xyz'[face.axis]}_primal64")
    normal_cell = (float(normal_primal[face.normal_index - 1]) + float(normal_primal[face.normal_index])) / 2
    width = tangent_dual[indices[positions, tangent]].double()
    return dict(positions=positions, sides=sides[positions], width=width, aspect=normal_cell / width,
                count=len(indices))


def copper_edge_scale_from_geometry(geometry, material, *, thickness=None, conductivity=None, count=None, device=None):
    """Per-trace admittance scale 1/kappa; exactly one away from free edges.

    ``thickness``/``conductivity`` default to the material's (detached) values;
    pass differentiable tensors to obtain the scale's parameter derivative.
    """
    if geometry is None:
        return torch.ones(count, dtype=torch.float64, device=device)
    def detached(value):
        return value.detach() if torch.is_tensor(value) else value
    thickness = detached(material.thickness) if thickness is None else thickness
    conductivity = detached(material.conductivity) if conductivity is None else conductivity
    weight = material.edge_weight(thickness=thickness, conductivity=conductivity)
    width = geometry["width"]
    kappa = copper_edge_impedance_factor(torch.as_tensor(thickness, dtype=torch.float64).to(width.device), width,
                                         geometry["aspect"], geometry["sides"],
                                         torch.as_tensor(weight, dtype=torch.float64).to(width.device))
    scale = torch.ones(geometry["count"], dtype=kappa.dtype, device=kappa.device)
    return scale.index_put((geometry["positions"].to(kappa.device),), 1 / kappa)


def copper_edge_admittance_scale(scene, face, indices, all_faces, *, contact=None):
    """Per-trace 1/kappa admittance scale and its edge geometry (or None)."""
    geometry = copper_edge_geometry(scene, face, indices, all_faces, contact=contact)
    scale = copper_edge_scale_from_geometry(geometry, face.material, count=len(indices), device=indices.device)
    return scale.to(indices.device), geometry


# Feature widths below which the edge model leaves a known residual against the
# finite-thickness reference: a strip between two free edges under-predicts
# (W = 1/2/4/8 cells: -35/-21/-12/-6%), a slot between facing free edges
# over-predicts (gap 1/2/4/8 cells: +30..40/+15/+5/+2..3%).
EDGE_MIN_STRIP_CELLS = 4
EDGE_MIN_SLOT_CELLS = 3


def warn_copper_edge_features(scene, face, indices, all_faces, geometry, *, contact=None):
    """Warn once per face when free-edge strips or slots are below their cell limits."""
    if geometry is None:
        return
    sides = torch.zeros(geometry["count"], dtype=torch.float64, device=geometry["sides"].device)
    sides[geometry["positions"]] = geometry["sides"]
    strip, slot = copper_edge_feature_cells(scene, face, indices, all_faces, sides, contact=contact)
    import warnings
    if strip is not None and strip < EDGE_MIN_STRIP_CELLS:
        warnings.warn(f"CopperFoil {face.name!r}: a free-edge strip is {strip} tangential cells wide "
                      f"(< {EDGE_MIN_STRIP_CELLS}); the edge model under-predicts its loss "
                      "(about -12% at 4, -21% at 2, -35% at 1 cells).", stacklevel=3)
    if slot is not None and slot < EDGE_MIN_SLOT_CELLS:
        warnings.warn(f"CopperFoil {face.name!r}: a slot between free edges is {slot} tangential cells wide "
                      f"(< {EDGE_MIN_SLOT_CELLS}); the edge model over-predicts its loss "
                      "(about +15% at 2, +30-40% at 1 cells).", stacklevel=3)

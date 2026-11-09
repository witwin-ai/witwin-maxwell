"""Fixed primitive binary pole ownership at the exact Yee centres."""
import torch
from witwin.core import Box, Sphere, Cylinder

from ...media import PerturbationMedium
from .build import build_extension
from .dispatch import UNCHANGED, requires_grad, setup_enabled, _fp32_geometry
from .geometry import descriptors


POLE_KEYS = tuple(prefix + kind + "_poles" for prefix in ("", "mu_")
                  for kind in ("debye", "drude", "lorentz"))


def scope_reason(scene, model, layout, *, eps_background, mu_background,
                 region_densities, geometry_periodic_shifts):
    """Return a numerical-scope exclusion, without allocating or initializing CUDA."""
    from .. import materials as m

    if m._resolve_subpixel(scene.subpixel)[1] != "kottke_edge":
        return "averaging"
    if any(entry.get("uniform_amplitude") is not None
           for key in POLE_KEYS for entry in layout[key]):
        return "uniform_amplitude"
    if any(entry.get("graded", False) or entry.get("amplitude") is not None
           for key in POLE_KEYS for entry in layout[key]):
        return "graded"
    if scene.material_regions or region_densities:
        return "material_regions"
    if any(scene.boundary.axis_kind(a) in ("periodic", "bloch") for a in "xyz"):
        return "periodic"
    if any(options is None or options != ((0.,), (0.,), (0.,))
           for options in geometry_periodic_shifts.values()):
        return "periodic_shifts"
    if scene.coordinates is not None or m._material_dtype(scene) != torch.float32:
        return "coordinates_or_dtype"
    if (not all(isinstance(v, (int, float)) for v in (eps_background, mu_background))
            or requires_grad((model["eps_components"], model["mu_components"]))):
        return "trainable_background"
    if m.material_model_has_full_anisotropy(model):
        return "tensor_background"
    if m.material_model_has_modulation(model):
        return "modulation"
    for structure in scene.structures:
        if getattr(structure.material, "modulation", None) is not None:
            return "modulation"
        if isinstance(structure.material, PerturbationMedium):
            return "perturbation"
        if requires_grad(structure.geometry):
            return "trainable_geometry"
        # These are the primitive SDF/fill implementations already qualified by
        # accelerated setup. Composite trees and mesh/slab distances stay torch.
        if type(structure.geometry) not in (Box, Sphere, Cylinder):
            return "geometry"
        # Float64 declarations have different closed-box bound semantics and
        # need their own ownership qualification; bulk setup admission is wider.
        if not _fp32_geometry(structure.geometry):
            return "geometry_dtype"
    return None


def _geometry_data(structures, device):
    """Pack small descriptors, preserving the source dtype of every operation."""
    geometry = descriptors(structures, device=device, native_coverage=True)
    geometry64 = geometry.to(torch.float64)
    boxes = torch.zeros((len(structures), 7), device=device, dtype=torch.float32)
    for i, structure in enumerate(structures):
        g = structure.geometry
        if type(g) is Box:
            # The float64 SDF converts size BEFORE dividing, unlike fp32 fill.
            geometry64[i, 0, 16:19] = g.size.to(device=device, dtype=torch.float64) / 2
            boxes[i, 0] = float(torch.equal(g.rotation.abs(), g.rotation.new_tensor((1., 0., 0., 0.))))
            boxes[i, 1:4] = g.position
            boxes[i, 4:7] = g.size
        elif type(g) is Cylinder:
            geometry64[i, 0, 17] = g.height.to(device=device, dtype=torch.float64) / 2
    return geometry, geometry64, boxes


def try_compile_binary_pole_edges(scene, model, layout, *, samples, eps_background,
                                  mu_background, region_densities, geometry_periodic_shifts):
    from .. import materials as m

    if not setup_enabled.get() or torch.device(scene.device).type != "cuda":
        return UNCHANGED
    if not any(layout[key] for key in POLE_KEYS):
        return None
    if scope_reason(scene, model, layout, eps_background=eps_background,
                    mu_background=mu_background, region_densities=region_densities,
                    geometry_periodic_shifts=geometry_periodic_shifts) is not None:
        return UNCHANGED
    structures = m._bulk_structures(scene)
    parts = [m._static_structure_material(s) for s in structures]
    active = []
    for prefix, components, channel, background in (
        ("", m._ELECTRIC_EDGE_COMPONENTS, "eps", eps_background),
        ("mu_", m._MAGNETIC_EDGE_COMPONENTS, "mu", mu_background),
    ):
        keys = tuple(prefix + kind + "_poles" for kind in ("debye", "drude", "lorentz"))
        carriers = [any(getattr(part[0], key, ()) for key in keys) for part in parts]
        if any(layout[key] for key in keys) and any(carriers):
            active.append((prefix, components, channel, background, keys, carriers))
    if not active:
        return None
    geometry, geometry64, boxes = _geometry_data(structures, scene.device)
    nodes = [torch.tensor(getattr(scene, a + "_nodes64"), device=scene.device, dtype=torch.float64)
             for a in "xyz"]
    # Sorting and first-index nearest-plane ties agree with the reference argmin.
    planes = [torch.cat((n, .5 * (n[:-1] + n[1:]))).sort().values for n in nodes]
    extension = build_extension()
    for prefix, components, channel, background, keys, carriers in active:
        entries = [entry for key in keys for entry in model[key][:len(layout[key])]]
        slots = [(index, model[key][slot])
                 for index, structure_slots in enumerate(layout["structure_slots"])
                 for kind, key in zip(("debye", "drude", "lorentz"), keys)
                 for slot in structure_slots[prefix + kind]]
        for entry in entries:
            entry["edge_weights"] = {}
        slot_owners = torch.tensor([index for index, _ in slots], device=scene.device, dtype=torch.int64)
        carrier_tensor = torch.tensor(carriers, device=scene.device, dtype=torch.bool)
        edge = model["edge_components"]
        seed = (torch.stack([entry["weight"] for entry in entries]).contiguous() if edge is None
                else torch.empty(0, device=scene.device, dtype=torch.float32))
        for component in components:
            stagger = m._EDGE_STAGGER_AXES[component]
            axis = component[-1].lower()
            bases = [m._edge_axis_base_1d(scene, a, i in stagger) for i, a in enumerate("xyz")]
            bases64 = [m._edge_axis_base_1d(scene, a, i in stagger, dtype=torch.float64)
                       for i, a in enumerate("xyz")]
            centres = [(b + 0.).contiguous() for b in bases64]
            offsets = [m._edge_axis_offsets(scene, a, n, i in stagger)
                       for i, (a, n) in enumerate(zip("xyz", samples))]
            spacing = m._subsample_spacing(scene, samples, stagger)
            query_bases = bases if spacing is not None else bases64
            # Upload rounded 1D queries rather than reconstructing sums in CUDA.
            queries = [torch.stack([b + off for off in offs]).contiguous()
                       for b, offs in zip(query_bases, offsets)]
            shape = tuple(len(b) for b in bases)
            widths = [torch.as_tensor(h, device=scene.device, dtype=torch.float32)
                      .expand(shape[i]).contiguous() for i, h in enumerate(spacing or (0., 0., 0.))]
            delta = (.25 * min(float(h.min()) if torch.is_tensor(h) else h for h in spacing)
                     if spacing is not None else 1.)
            values = [part[1 if prefix == "" else 3][axis] * float(background) for part in parts]
            # Fixed tensor-valued backgrounds remain on device; do not extract
            # their scalar values through torch.tensor(list_of_cuda_tensors).
            values = (torch.stack([torch.as_tensor(v, device=scene.device, dtype=torch.float32) for v in values])
                      if any(torch.is_tensor(v) for v in values) else
                      torch.tensor(values, device=scene.device, dtype=torch.float32))
            old = (model[channel + "_components"][axis] if edge is None
                   else edge[channel][component]).contiguous()
            weights, selected = extension.poles(
                geometry, geometry64, boxes, planes, centres, queries, widths,
                values, carrier_tensor, seed, old, slot_owners,
                [*shape, *[len(n) for n in nodes], *[len(o) for o in offsets],
                 "xyz".index(axis), int(prefix != ""), int(edge is None)],
                [float(background), delta, m._NORMAL_EPS * delta * delta,
                 1. / (delta * delta)])
            for row, (_, entry) in enumerate(slots):
                entry["edge_weights"][component] = weights[row].view(shape)
            if edge is None:
                model.setdefault("pole_edge_background", {})[component] = selected.view(shape)
            else:
                edge[channel][component] = selected.view(shape)
    return None

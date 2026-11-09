"""Native chronological sample loops and exact interface reconstruction."""
from math import prod

import torch
from .build import build_extension
from .geometry import coarse_bins, descriptors, native_geometry, torch_fills


def sample_fields(scene, structures, backgrounds, bases, offsets, spacing, shifts, *, edge=False, stagger=(), poles=()):
    from .. import materials as m

    shape = tuple(len(base) for base in bases)
    counts = tuple(len(values) for values in offsets)
    # Upload the exact rounded 1D queries, never reconstruct coordinates in CUDA.
    axes = [torch.stack([base + offset for offset in values]) for base, values in zip(bases, offsets)]
    widths = [torch.as_tensor(value, device=scene.device, dtype=torch.float32).expand(shape[a]).contiguous()
              for a, value in enumerate(spacing or (0., 0., 0.))]
    delta = .25 * min(float(value.min()) if torch.is_tensor(value) else value for value in spacing) if spacing else 1.
    periodic = any(scene.boundary.axis_kind(a) in ("periodic", "bloch") for a in "xyz")
    native = [spacing is not None and not periodic and native_geometry(s.geometry) for s in structures]
    geometry = descriptors(structures, device=scene.device, native_coverage=spacing is not None and not periodic)
    values = []
    for structure in structures:
        part = m._static_structure_material(structure)
        eps, mu = part[1]["x"] * backgrounds[0], part[3]["x"] * backgrounds[1]
        values.append((eps, mu, part[4]["x"], part[5]["x"], 1./(eps + m._HARMONIC_EPS), 1./(mu + m._HARMONIC_EPS)))
    values = torch.tensor(values, device=scene.device, dtype=torch.float32).reshape(-1, 6)
    background = [*backgrounds, 0., 0., 1./(backgrounds[0]+m._HARMONIC_EPS), 1./(backgrounds[1]+m._HARMONIC_EPS)]
    bins, ids = coarse_bins(axes, widths, geometry, native)
    outputs = [torch.empty(shape, device=scene.device, dtype=torch.float32) for _ in range(6)]
    retained = any(not value for value in native)
    # Limit packed torch coverage to about 32 MiB, retaining complete y/z axes.
    slab = max(1, (8*1024*1024)//max(1, prod(counts)*len(structures)*shape[1]*shape[2])) if retained else shape[0]
    extension = build_extension()
    constants = [delta, m._NORMAL_EPS*delta*delta, 1./(delta*delta), 1./prod(counts), *background]
    pole_outputs = [entry["weight"] for _, entry in poles]
    amplitudes = [entry["amplitude"].contiguous() if entry["amplitude"] is not None
                  else torch.empty(0, device=scene.device, dtype=torch.float32) for _, entry in poles]
    owners = [owner for owner, _ in poles]
    for low in range(0, shape[0], slab):
        high = min(shape[0], low+slab)
        count = (high-low)*shape[1]*shape[2]
        fills = (torch_fills(scene, structures, native, axes, spacing, delta, shifts, slice(low, high), stagger)
                 if retained else torch.empty(0, device=scene.device, dtype=torch.float32))
        extension.sample(outputs, axes+widths, geometry, values, fills, bins, ids,
                         [*shape, *counts, low*shape[1]*shape[2], count, int(edge)], constants,
                         pole_outputs, amplitudes, owners)
    return outputs


def compile_setup(scene, eps_background, mu_background, subpixel, include_edges):
    from .. import materials as m
    from .subpixel import reconstruct
    from .conductors import try_occupancy, try_pec_edge_fill
    from .dispatch import UNCHANGED

    samples, averaging, pec_mode = m._resolve_subpixel(subpixel)
    m.validate_legacy_thin_pec(scene)
    structures = m._bulk_structures(scene)
    shifts = {id(s.geometry): m._static_periodic_shift_options(scene, s.geometry) for s in structures}
    layout = m._build_dispersive_layout(scene)
    model = m._new_material_model(scene, layout, eps_fill=0., mu_fill=0.)
    poles = [(index, model[kind + "_poles"][slot])
             for index, slots in enumerate(layout["structure_slots"])
             for kind, indices in slots.items() for slot in indices]
    offsets = [m._axis_sample_offsets(scene, a, n) for a, n in zip("xyz", samples)]
    fields = sample_fields(scene, structures, (eps_background, mu_background),
                           [getattr(scene, a) for a in "xyz"], offsets,
                           m._subsample_spacing(scene, samples), shifts, poles=poles)
    for key, field in zip(("eps_components", "mu_components", "sigma_e_components", "sigma_m_components"), fields):
        model[key] = {a: field.clone() for a in "xyz"}
    if samples != (1, 1, 1):
        for key, field in zip(("eps_components", "mu_components"), fields[4:]):
            model[key] = reconstruct(scene, model[key], {a: field for a in "xyz"})
    del fields
    model = m._apply_sheet_structures(scene, m._refresh_model_summary_aliases(model))
    pec, pmc = try_occupancy(scene), try_occupancy(scene, magnetic=True)
    pec_edges = try_pec_edge_fill(scene) if pec_mode == "conformal" else None
    model.update(pec_occupancy=m._pec_occupancy(scene) if pec is UNCHANGED else pec, pec_mode=pec_mode,
                 pec_edge_fill=m._pec_edge_fill(scene) if pec_edges is UNCHANGED else pec_edges,
                 pmc_occupancy=m._pmc_occupancy(scene) if pmc is UNCHANGED else pmc,
                 surface_impedance=m.compile_surface_impedance_layout(scene),
                 edge_components=None)
    m.attach_planar_pec(scene, model)
    if include_edges and m._edge_native_eligible(scene, model, model["surface_impedance"]):
        from .subpixel import kottke_component
        bulk_geometry = descriptors(structures, device=scene.device, native_coverage=True)
        edges = {key: {} for key in ("eps", "sigma_e", "mu", "sigma_m")}
        for component in (*m._ELECTRIC_EDGE_COMPONENTS, *m._MAGNETIC_EDGE_COMPONENTS):
            electric = component[0] == "E"
            stagger = m._EDGE_STAGGER_AXES[component]
            bases = [m._edge_axis_base_1d(scene, a, i in stagger) for i, a in enumerate("xyz")]
            if averaging in ("kottke_edge", "kottke_smooth"):
                # Preserve bulk shortcuts and ascending 4096-cell windows.
                material = kottke_component(scene, component, dict(zip("xyz", bases)), samples,
                                            eps_background if electric else mu_background,
                                            structures, bulk_geometry, shifts, smooth=averaging == "kottke_smooth")
                if all(m._static_structure_material(s)[4 if electric else 5][component[1].lower()] == 0 for s in structures):
                    sigma = torch.zeros_like(material)
                else:
                    offsets = [m._edge_axis_offsets(scene, a, n, i in stagger)
                               for i, (a, n) in enumerate(zip("xyz", samples))]
                    # The scalar torch path discards its reconstructed mean in
                    # favor of the Kottke window. Keep its chronological six sums,
                    # including sigma and inverse, without the unused gradients.
                    fields = sample_fields(scene, structures, (eps_background, mu_background), bases, offsets,
                                           m._subsample_spacing(scene, samples, stagger), shifts,
                                           edge=True, stagger=stagger)
                    sigma = fields[2 if electric else 3]
                    del fields
            else:
                offsets = [m._edge_axis_offsets(scene, a, n, i in stagger) for i, (a, n) in enumerate(zip("xyz", samples))]
                fields = sample_fields(scene, structures, (eps_background, mu_background), bases, offsets,
                                       m._subsample_spacing(scene, samples, stagger), shifts, edge=True, stagger=stagger)
                material, sigma = fields[0 if electric else 1], fields[2 if electric else 3]
                if averaging == "polarized":
                    coords = m._edge_component_coords(dict(zip("xyz", bases)), (0., 0., 0.))
                    material = reconstruct(scene, material, fields[4 if electric else 5], coords, component[1].lower())
                del fields
            edges["eps" if electric else "mu"][component] = material
            edges["sigma_e" if electric else "sigma_m"][component] = sigma
        model["edge_components"] = edges
    if include_edges and averaging == "kottke_edge":
        m._compile_binary_pole_edges(scene, model, layout, samples=samples,
            eps_background=eps_background, mu_background=mu_background,
            region_densities=(), geometry_periodic_shifts=shifts)
    return model

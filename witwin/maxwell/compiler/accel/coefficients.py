"""Scoped exact staggered coefficient composition."""
import torch

from .build import build_extension
from .dispatch import UNCHANGED, eligible


def try_build_update_coefficients(solver):
    scene = solver.scene
    if not eligible(scene, 1., 1., scene.subpixel, coefficients=True):
        return UNCHANGED
    if any(getattr(solver, name, False) for name in (
            'full_aniso_enabled', 'dispersive_enabled', 'nonlinear_enabled',
            'modulation_enabled', 'breakdown_enabled', 'gyromagnetic_enabled')):
        return UNCHANGED
    from ...fdtd.runtime import materials as m

    components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
    material = [getattr(solver, ('eps_' if c[0] == 'E' else 'mu_') + c) for c in components]
    sigma = [getattr(solver, ('sigma_e_' if c[0] == 'E' else 'sigma_m_') + c) for c in components]
    profiles = ([getattr(solver, 'sigma_' + a) for a in 'xyz']
                if solver.active_absorber_type in ('pml', 'absorber') else [])
    if any(t.ndim != 3 or t.dtype != torch.float32 or t.device.type != 'cuda' or t.device != material[0].device
           or not t.is_contiguous() or t.requires_grad for t in (*material, *sigma, *profiles)):
        return UNCHANGED
    outputs = build_extension().coefficients(
        material, sigma, profiles, [solver.Nx, solver.Ny, solver.Nz], solver.dt,
        solver.conductive_enabled, solver.magnetically_conductive_enabled)
    for index, component in enumerate(components):
        prefix = 'c' + component.lower()
        setattr(solver, prefix + '_decay', outputs[2*index])
        setattr(solver, prefix + '_curl', outputs[2*index+1])
    m._store_nonlinear_external_decay(solver, None, None, None)
    m._build_full_aniso_curl_coefficients(solver)
    m._apply_pec_edge_suppression(solver)
    m._apply_pmc_face_suppression(solver)
    m._configure_surface_impedance(solver)
    m._store_coefficient_uniformity(solver)
    return True

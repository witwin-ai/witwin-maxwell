"""Admission of a bounded primary-native FP64 detached forward path."""
import torch


def validate_fp64_forward(simulation, scene):
    """Require the exact supported field/source/material execution contract."""
    from witwin.core import Box
    from ...ports import LumpedPort
    from ...media import Material
    from ...copper_foil import CopperFoil
    from ...monitors import PointMonitor, FinitePlaneMonitor, ClosedSurfaceMonitor, PowerLossMonitor
    config=simulation.config
    def has_gradient(value, seen):
        if id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            return value.requires_grad
        if isinstance(value, dict):
            return any(has_gradient(child, seen) for child in value.values())
        if isinstance(value, (tuple, list)):
            return any(has_gradient(child, seen) for child in value)
        if type(value).__module__.startswith("witwin.") and hasattr(value, "__dict__"):
            return any(has_gradient(child, seen) for child in vars(value).values())
        return False
    if has_gradient((scene, simulation.excitations, config), set()):
        raise ValueError('FP64 native forward requires detached declarative inputs; FP64 adjoint kernels are not qualified.')
    if simulation.has_trainable_parameters:
        raise ValueError('FP64 native forward currently requires detached parameters; FP64 adjoint kernels are not qualified.')
    if scene.grid.is_auto:
        raise ValueError('FP64 native forward currently requires a fixed declared grid; auto-grid precision is unqualified.')
    if any(type(monitor) not in (PointMonitor, FinitePlaneMonitor, ClosedSurfaceMonitor, PowerLossMonitor) for monitor in scene.monitors):
        raise ValueError('FP64 native forward currently supports point, finite-plane, closed-surface and bulk-loss declarations only.')
    if config.parallel is not None or simulation.material_cache is not None:
        raise ValueError('FP64 native forward requires one uncached device; distributed and material-cache ownership are not qualified.')
    if config.cuda_graph or config.full_field_dft or len(simulation.frequencies)!=1:
        raise ValueError('FP64 native forward requires eager execution, no full-field DFT, and one frequency.')
    if config.spectral_sampler.observer_dtype!=torch.float64 or config.spectral_sampler.normalize_source:
        raise ValueError('FP64 native forward requires unnormalized FP64 spectral observers.')
    if config.initial_condition is not None or config.shutoff:
        raise ValueError('FP64 native forward requires fixed-duration execution without initial-condition injection.')
    if any(str(kind) != 'pml' for pair in scene.boundary.face_kinds for kind in pair):
        raise ValueError('FP64 native forward currently requires all six faces to use PML; mixed boundary kernels are not qualified.')
    from ..boundary.runtime import resolve_absorber
    # Resolve 'auto' as preparation will; symmetry can still select the graded absorber.
    absorber, _ = resolve_absorber(scene, config.absorber, complex_fields=scene.boundary.uses_kind("bloch"))
    if absorber!='cpml':
        raise ValueError('FP64 native forward requires the CPML absorber.')
    if config.cpml_config.get('memory_mode','auto')not in('auto','dense'):
        raise ValueError('FP64 native forward currently supports dense CPML memory only.')
    if scene.sources or scene.material_regions or scene.thin_wires or scene.circuits or scene.networks or scene.lumped_elements:
        raise ValueError('FP64 native forward currently requires fixed Box materials and direct Norton ports; other source/device runtimes are unqualified.')
    for port in scene.ports:
        if type(port)is not LumpedPort or port.termination is not None or port.wire_binding is not None:
            raise ValueError('FP64 native forward currently supports direct LumpedPort Norton drives without reactive terminations.')
    for structure in scene.structures:
        material=structure.material
        if structure.phase_screen is not None or structure.uv is not None:
            raise ValueError('FP64 native forward does not qualify optical phase screens or UV material assignments.')
        if type(material) not in (Material, CopperFoil):
            raise ValueError('FP64 native forward currently supports Material and CopperFoil declarations only.')
        if getattr(material, 'modulation', None) is not None or getattr(material, 'breakdown', None) is not None:
            raise ValueError('FP64 native forward requires static materials; modulation and breakdown are not qualified.')
        if not isinstance(structure.geometry,Box):
            raise ValueError('FP64 native forward currently requires Box geometry.')
        if any(bool(getattr(material,name,False))for name in('is_anisotropic','is_dispersive','is_nonlinear','is_pec','is_pmc','is_surface_impedance','is_lossy_metal')):
            raise ValueError('FP64 native forward requires scalar nondispersive bulk epsilon/conductivity or two-sided CopperFoil.')
        for name in('mu_r','sigma_m'):
            value=getattr(material,name,1. if name=='mu_r'else 0.)
            expected=1. if name=='mu_r'else 0.
            if isinstance(value,torch.Tensor):
                valid=value.ndim==0 and not value.requires_grad and bool(value==expected)
            else:valid=float(value)==expected
            if not valid:raise ValueError('FP64 native forward currently requires physical mu_r=1 and sigma_m=0.')

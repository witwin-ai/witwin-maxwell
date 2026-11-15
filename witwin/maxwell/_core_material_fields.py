"""Core ``PhysicalMaterial`` surface-model fields that the Maxwell volume and surface solvers do not read."""
import inspect

from witwin.core import PhysicalMaterial

# The solver-neutral material carries radio-propagation surface descriptors
# (wall thickness, layering, roughness, scattering, a frequency power law and a
# perfect-conductor switch). MoM and electrostatics read only eps_r, mu_r,
# sigma_e (and the Maxwell-specific fields), so any non-default value here
# would otherwise be dropped without notice.
_SURFACE_FIELDS = ("thickness_m", "layers", "geometry_mode", "roughness_front", "roughness_back",
                   "dispersion", "gain", "scattering_coefficient", "xpd_coefficient", "conductor_model")
_DEFAULTS = {name: inspect.signature(PhysicalMaterial.__init__).parameters[name].default
             for name in _SURFACE_FIELDS}


def _is_default(value, default) -> bool:
    if default is None:
        return value is None
    try:
        return bool(value == default)
    except (TypeError, RuntimeError, ValueError):
        return False


def declared_core_surface_fields(material) -> tuple[str, ...]:
    """Names of core surface-model fields set to a non-default value."""
    if not isinstance(material, PhysicalMaterial):
        return ()
    return tuple(name for name in _SURFACE_FIELDS if not _is_default(getattr(material, name), _DEFAULTS[name]))
